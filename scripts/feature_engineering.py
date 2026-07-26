"""
feature_engineering.py
=======================
Transforms raw access logs into a model-ready feature matrix.

Core idea: Stage 1 (Isolation Forest / LSTM-Autoencoder) and Stage 2
(XGBoost classifier) don't need raw categorical fields like "geo_city" or
"device_id" directly -- they need DEVIATION features that quantify how far
a given event is from that specific user's own established baseline.
That's what actually makes credential misuse, impossible travel, device
spoofing, brute force, and lateral movement separable.

Feature groups produced:
    1. Geo-velocity features       -> Impossible Travel signal
    2. Device/OS/Browser novelty   -> Device Spoofing / Credential Misuse signal
    3. Time-of-day deviation       -> Credential Misuse signal
    4. Failed-login burst features -> Brute Force signal
    5. Resource/department mismatch-> Lateral Movement signal
    6. Rolling per-user history stats (with COLD-START handling via
       Bayesian shrinkage toward department-level priors)

Output: dataset/processed/features.csv
"""

import os
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
COLD_START_MIN_EVENTS = 10   # below this many prior events, shrink toward dept baseline
SHRINKAGE_STRENGTH = 8        # pseudo-count controlling how fast shrinkage fades

RAW_LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw", "access_logs.csv")
USERS_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw", "users.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "features.csv")


# ----------------------------------------------------------------------------
# Geo utility: haversine distance in km between two lat/lon points
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# Max plausible human travel speed (commercial flight + margin), km/h.
# Any implied speed above this between two consecutive logins for the same
# user is a strong "impossible travel" signal.
MAX_PLAUSIBLE_SPEED_KMH = 950.0


def load_data():
    logs = pd.read_csv(RAW_LOGS_PATH, parse_dates=["timestamp"])
    users = pd.read_csv(USERS_PATH)
    return logs, users


def add_department_priors(logs: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Merge department-level baselines (used for cold-start shrinkage) --
    e.g. department's typical login-hour center, typical failed-login rate.
    """
    dept_login_center = users.groupby("department")[["login_start_hour", "login_end_hour"]].mean()
    dept_login_center["dept_login_mid_hour"] = (
        dept_login_center["login_start_hour"] + dept_login_center["login_end_hour"]
    ) / 2
    logs = logs.merge(
        dept_login_center[["dept_login_mid_hour"]],
        left_on="department", right_index=True, how="left"
    )
    return logs


def compute_geo_velocity_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    For each user, sort events chronologically and compute the implied
    travel speed between consecutive logins. This is the core Impossible
    Travel signal: if a user "moves" faster than any real transport allows,
    it's a strong indicator of credential compromise / session hijack.
    """
    logs = logs.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    logs["prev_lat"] = logs.groupby("user_id")["geo_lat"].shift(1)
    logs["prev_lon"] = logs.groupby("user_id")["geo_lon"].shift(1)
    logs["prev_timestamp"] = logs.groupby("user_id")["timestamp"].shift(1)

    # Floor on elapsed time: two logins minutes apart shouldn't be treated as
    # "instantaneous teleportation" math -- below this floor we can't say
    # anything meaningful about travel speed, so we floor the denominator
    # rather than let it blow up into absurd implied speeds.
    MIN_ELAPSED_HOURS = 0.25  # 15 minutes

    def row_velocity(row):
        if pd.isna(row["prev_lat"]):
            return 0.0, 0.0  # first event ever for this user -> no velocity signal (cold start)
        dist_km = haversine_km(row["prev_lat"], row["prev_lon"], row["geo_lat"], row["geo_lon"])
        raw_hours_elapsed = (row["timestamp"] - row["prev_timestamp"]).total_seconds() / 3600.0
        hours_elapsed = max(raw_hours_elapsed, MIN_ELAPSED_HOURS)
        speed_kmh = dist_km / hours_elapsed
        return dist_km, speed_kmh

    results = logs.apply(row_velocity, axis=1, result_type="expand")
    logs["distance_from_prev_km"] = results[0]
    logs["implied_speed_kmh"] = results[1]
    logs["impossible_travel_flag"] = (logs["implied_speed_kmh"] > MAX_PLAUSIBLE_SPEED_KMH).astype(int)

    logs.drop(columns=["prev_lat", "prev_lon", "prev_timestamp"], inplace=True)
    return logs


def compute_device_novelty_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Tracks, per user, the set of devices/OS/browsers seen SO FAR (expanding
    window, respecting time order) so a "new device" flag reflects only
    information available up to that point in time -- avoiding lookahead
    bias and naturally supporting cold start (a user's very first event has
    no history, so novelty is intentionally left neutral rather than
    penalized).
    """
    logs = logs.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    def novelty_flags(group):
        seen_devices, seen_os, seen_browsers = set(), set(), set()
        device_new, os_new, browser_new, prior_event_count = [], [], [], []
        events_seen_so_far = 0
        for _, row in group.iterrows():
            prior_event_count.append(events_seen_so_far)  # actual count of prior events for this user
            device_new.append(0 if row["device_id"] in seen_devices or not seen_devices else 1)
            os_new.append(0 if row["os"] in seen_os or not seen_os else 1)
            browser_new.append(0 if row["browser"] in seen_browsers or not seen_browsers else 1)
            seen_devices.add(row["device_id"])
            seen_os.add(row["os"])
            seen_browsers.add(row["browser"])
            events_seen_so_far += 1
        group = group.copy()
        group["is_new_device"] = device_new
        group["is_new_os"] = os_new
        group["is_new_browser"] = browser_new
        group["prior_event_count"] = prior_event_count
        return group

    # NOTE: pandas >=2.2 excludes the groupby key column from the sub-frame
    # passed into apply() by default, so we preserve original index to
    # re-attach user_id afterward rather than relying on it surviving inside.
    original_user_ids = logs["user_id"]
    result = logs.groupby("user_id", group_keys=False).apply(novelty_flags, include_groups=False)
    result["user_id"] = original_user_ids.loc[result.index]
    return result


def compute_time_deviation_features(logs: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Login-hour deviation from the user's own usual window, with COLD-START
    shrinkage: if a user doesn't have enough history yet, we blend their
    (thin) personal baseline with the department-level baseline using a
    Bayesian-style weighted average, so a brand-new employee isn't flagged
    for every login just because we don't know their habits yet.
    """
    user_baseline = users.set_index("user_id")[["login_start_hour", "login_end_hour"]]
    logs = logs.merge(user_baseline, on="user_id", how="left", suffixes=("", "_baseline"))

    logs["login_hour"] = logs["timestamp"].dt.hour
    user_mid_hour = (logs["login_start_hour"] + logs["login_end_hour"]) / 2

    # Cold-start shrinkage: weight = n / (n + k). More history -> trust
    # personal baseline more; little history -> lean on department prior.
    n = logs["prior_event_count"].clip(lower=0)
    weight_personal = n / (n + SHRINKAGE_STRENGTH)
    blended_mid_hour = weight_personal * user_mid_hour + (1 - weight_personal) * logs["dept_login_mid_hour"]

    logs["login_hour_deviation"] = (logs["login_hour"] - blended_mid_hour).abs()
    logs["is_odd_hour_login"] = ((logs["login_hour"] < 6) | (logs["login_hour"] > 22)).astype(int)
    logs["cold_start_flag"] = (n < COLD_START_MIN_EVENTS).astype(int)

    return logs


def compute_resource_mismatch_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Lateral Movement signal: does this event access a resource that
    belongs to a DIFFERENT department than the user's own? Combined with
    session_duration and is_sensitive_resource, this becomes a strong
    lateral-movement indicator.
    """
    # Static resource->department map built directly from generator config
    resource_dept_map = {
        "source_code_repo": "Engineering", "ci_cd_pipeline": "Engineering",
        "internal_wiki": "Engineering", "jira": "Engineering",
        "erp_system": "Finance", "payroll_db": "Finance",
        "financial_reports": "Finance", "banking_portal": "Finance",
        "hr_management_system": "HR", "employee_records": "HR",
        "recruitment_portal": "HR",
        "crm_system": "Sales", "sales_dashboard": "Sales", "contracts_repo": "Sales",
        "cms": "Marketing", "analytics_dashboard": "Marketing",
        "social_media_tools": "Marketing",
        "admin_console": "IT_Ops", "network_config": "IT_Ops",
        "server_room_access": "IT_Ops", "domain_controller": "IT_Ops",
        "compliance_db": "Legal", "legal_case_management": "Legal",
        "research_data_lake": "R&D", "patent_db": "R&D",
        "lab_equipment_control": "R&D",
        "ticketing_system": "Customer_Support", "customer_db": "Customer_Support",
        "knowledge_base": "Customer_Support",
        "board_reports": "Executive", "strategic_plans_drive": "Executive",
    }
    logs["resource_owner_dept"] = logs["resource_accessed"].map(resource_dept_map)
    logs["is_cross_department_access"] = (
        logs["resource_owner_dept"] != logs["department"]
    ).astype(int)
    return logs


def compute_bruteforce_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Brute Force signal: raw failed_login_count plus a rolling per-user
    failed-login rate over their recent history, which captures a sudden
    spike relative to that user's own norm (some users are just naturally
    fumble-fingered typers -- we care about deviation, not the raw count
    alone).
    """
    logs = logs.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    logs["rolling_avg_failed_logins"] = (
        logs.groupby("user_id")["failed_login_count"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0)
    )
    logs["failed_login_spike"] = (
        logs["failed_login_count"] - logs["rolling_avg_failed_logins"]
    ).clip(lower=0)
    return logs


def build_feature_matrix():
    print("[1/6] Loading raw logs and user profiles...")
    logs, users = load_data()

    print(f"[2/6] Merging department-level priors ({len(users)} users)...")
    logs = add_department_priors(logs, users)

    print("[3/6] Computing geo-velocity / impossible-travel features...")
    logs = compute_geo_velocity_features(logs)

    print("[4/6] Computing device/OS/browser novelty features (cold-start safe)...")
    logs = compute_device_novelty_features(logs)

    print("[5/6] Computing time-of-day deviation (with Bayesian cold-start shrinkage)...")
    logs = compute_time_deviation_features(logs, users)

    print("[6/6] Computing resource-mismatch and brute-force features...")
    logs = compute_resource_mismatch_features(logs)
    logs = compute_bruteforce_features(logs)

    # Clean up: normal rows have attack_type == NaN when read back by pandas
    # (a cosmetic artifact of writing the literal string "None" to CSV).
    logs["attack_type"] = logs["attack_type"].fillna("None")

    return logs


FEATURE_COLUMNS = [
    "user_id", "department", "timestamp",
    # Geo-velocity / impossible travel
    "distance_from_prev_km", "implied_speed_kmh", "impossible_travel_flag",
    # Device novelty
    "is_new_device", "is_new_os", "is_new_browser", "prior_event_count",
    # Time deviation
    "login_hour", "login_hour_deviation", "is_odd_hour_login", "cold_start_flag",
    # Resource mismatch / lateral movement
    "is_cross_department_access", "is_sensitive_resource",
    # Brute force
    "failed_login_count", "rolling_avg_failed_logins", "failed_login_spike",
    # Other raw signals kept as-is (already meaningful without transformation)
    "vpn_used", "mfa_used", "file_download_size_mb", "session_duration_min",
    # Labels (kept for supervised Stage 2 + evaluation; NOT fed as features to Stage 1)
    "label_is_attack", "attack_type",
]


if __name__ == "__main__":
    full_df = build_feature_matrix()
    feature_df = full_df[FEATURE_COLUMNS].copy()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Feature Engineering Summary ===")
    print(f"Rows: {len(feature_df)}   Columns: {len(feature_df.columns)}")
    print(f"\nCold-start rows (users with <{COLD_START_MIN_EVENTS} prior events): "
          f"{feature_df['cold_start_flag'].sum()} "
          f"({feature_df['cold_start_flag'].mean()*100:.1f}%)")
    print(f"Impossible-travel flagged rows: {feature_df['impossible_travel_flag'].sum()}")
    print(f"Cross-department access rows:   {feature_df['is_cross_department_access'].sum()}")
    print("\nMean feature values by attack_type:")
    print(feature_df.groupby("attack_type")[[
        "implied_speed_kmh", "is_new_device", "login_hour_deviation",
        "failed_login_spike", "is_cross_department_access"
    ]].mean().round(2))

    print(f"\nSaved feature matrix to: {OUTPUT_PATH}")
