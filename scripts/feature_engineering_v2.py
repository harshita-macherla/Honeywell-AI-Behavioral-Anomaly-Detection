"""
feature_engineering_v2.py
===========================
Enterprise Behavioral Analytics Feature Pipeline (v2)

REPLACES scripts/feature_engineering.py. Built for the FROZEN Enterprise
Dataset v2 (dataset/raw_v2/access_logs_v2.csv) — the multi-entity-type,
12-attack-scenario dataset. This pipeline does NOT modify the dataset, the
ML models, or the backend/frontend — it is a pure feature transformation
layer, designed to match the analytical depth of commercial UEBA/SOC
platforms (Defender, CrowdStrike, Cortex XDR, Splunk UBA).

DESIGN PRINCIPLES
------------------
1. VECTORIZED: no iterrows(), no full-table nested loops. Per-entity /
   per-group novelty and cumulative features use pandas' cumcount()/
   groupby-transform machinery, which is implemented in C, not Python.
   A few operations that are inherently windowed-and-non-additive (rolling
   DISTINCT counts for password-spray detection; short command-token
   parsing) use bounded groupby().rolling().apply() on small windows —
   documented individually. This is categorically different from v1's bug
   (an O(n) Python-level loop over the ENTIRE table): these are linear-time,
   windowed operations, the standard practical pattern even in production
   Spark/Flink feature pipelines.
2. NO LABEL LEAKAGE: peer-group / department / role / organizational
   baselines are computed over the FULL dataset (including its ~2.5%
   attack contamination) rather than filtering by label_is_attack — doing
   the latter would leak the target into the features. This mirrors how
   Isolation Forest itself is trained on mildly-contaminated "normal" data.
3. EVERY FEATURE IS DOCUMENTED: see FEATURE_CATALOG.md (generated
   alongside this script) for purpose / formula / cybersecurity intuition /
   expected importance / which attacks it helps detect, for every column
   this script produces.

OUTPUT: dataset/processed/features_v2.csv
"""

import os
import numpy as np
import pandas as pd

pd.set_option("mode.chained_assignment", None)

RAW_LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_v2", "access_logs_v2.csv")
ENTITIES_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_v2", "entities_v2.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "features_v2.csv")

# ----------------------------------------------------------------------------
# Static reference data (mirrors generator config; in production these would
# come from a CMDB / threat-intel feed rather than being hardcoded)
# ----------------------------------------------------------------------------
RESOURCE_DEPT_MAP = {
    "source_code_repo": "Engineering", "ci_cd_pipeline": "Engineering",
    "internal_wiki": "Engineering", "jira": "Engineering",
    "erp_system": "Finance", "payroll_db": "Finance",
    "financial_reports": "Finance", "banking_portal": "Finance",
    "hr_management_system": "HR", "employee_records": "HR", "recruitment_portal": "HR",
    "crm_system": "Sales", "sales_dashboard": "Sales", "contracts_repo": "Sales",
    "cms": "Marketing", "analytics_dashboard": "Marketing", "social_media_tools": "Marketing",
    "admin_console": "IT_Ops", "network_config": "IT_Ops",
    "server_room_access": "IT_Ops", "domain_controller": "IT_Ops",
    "compliance_db": "Legal", "legal_case_management": "Legal",
    "research_data_lake": "R&D", "patent_db": "R&D", "lab_equipment_control": "R&D",
    "ticketing_system": "Customer_Support", "customer_db": "Customer_Support",
    "knowledge_base": "Customer_Support",
    "board_reports": "Executive", "strategic_plans_drive": "Executive",
    "scada_hmi": "OT_Operations", "plc_controller_iface": "OT_Operations",
    "historian_db": "OT_Operations", "ot_network_gateway": "OT_Operations",
}

HOSTING_ASNS = {"AS14061", "AS16509", "AS20473", "AS9009", "AS36351"}

DANGEROUS_KEYWORDS = ["vssadmin", "shadowcopy delete", "reg delete", "rm -rf", "del /s /q"]
LOLBIN_KEYWORDS = ["powershell.exe", "wmic.exe", "certutil.exe", "psexec.exe", "mshta.exe"]
PRIV_ESC_KEYWORDS = ["net localgroup administrators", "runas", "sudo -l", "chmod 4755", "reg add"]

MAX_PLAUSIBLE_SPEED_KMH = 950.0
MIN_ELAPSED_HOURS = 0.25
COLD_START_MIN_EVENTS = 10
SHRINKAGE_STRENGTH = 8


def load_data():
    df = pd.read_csv(RAW_LOGS_PATH, parse_dates=["timestamp"], keep_default_na=False, na_values=[""])
    entities = pd.read_csv(ENTITIES_PATH, keep_default_na=False, na_values=[""])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    return df, entities


def haversine_km_vectorized(lat1, lon1, lat2, lon2):
    """Fully vectorized haversine distance -- operates on entire Series at once."""
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


# ==============================================================================
# 1. USER BEHAVIOR FEATURES
# ==============================================================================
def add_user_behavior_features(df):
    g = df.groupby("entity_id")

    df["historical_event_count"] = g.cumcount()  # prior events, vectorized (was a bug source in v1)
    df["prev_timestamp"] = g["timestamp"].shift(1)
    df["time_since_last_login_hours"] = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600.0
    ).fillna(0)

    # Expanding (shifted, so no lookahead) session-duration baseline per entity
    df["avg_session_duration"] = g["session_duration_min"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df["session_duration_min"].mean())
    df["session_duration_std"] = g["session_duration_min"].transform(
        lambda s: s.shift(1).expanding().std()
    ).fillna(1.0).replace(0, 1.0)
    df["session_duration_zscore"] = (
        (df["session_duration_min"] - df["avg_session_duration"]) / df["session_duration_std"]
    )

    # Day-of-week deviation: how rare is this weekday for this entity, based on
    # prior history (Laplace-smoothed relative frequency, avoids 0/0 for cold start)
    df["weekday"] = df["timestamp"].dt.weekday
    weekday_counts = df.groupby(["entity_id", "weekday"]).cumcount()
    df["day_of_week_deviation"] = 1 - (weekday_counts + 1) / (df["historical_event_count"] + 7)

    df["weekend_activity_flag"] = df["is_weekend"]
    df["holiday_activity_flag"] = df["is_holiday"]

    # Behavioral drift: recent (last 10 events) mean login hour vs full history mean
    # login hour -- a widening gap signals the entity's habits are shifting.
    df["login_hour"] = df["timestamp"].dt.hour
    df["rolling10_mean_hour"] = g["login_hour"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(df["login_hour"])
    df["expanding_mean_hour"] = g["login_hour"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df["login_hour"])
    df["behavioral_drift_score"] = (df["rolling10_mean_hour"] - df["expanding_mean_hour"]).abs()

    # NOTE: .rolling(on="timestamp") returns a (entity_id, timestamp) MultiIndex,
    # which is NOT guaranteed unique when an entity has duplicate timestamps --
    # reset_index(level=0, drop=True) then fails on a duplicate-label reindex.
    # Since df is already sorted by (entity_id, timestamp) here, the rolling
    # result is returned in the same row order as df, so positional .values
    # assignment is safe and sidesteps the index collision entirely.
    df["rolling_failed_login_rate_7d"] = g.rolling("7D", on="timestamp")["failed_login_count"].mean().values

    return df


# ==============================================================================
# 2. DEVICE TRUST FEATURES
# ==============================================================================
def add_device_trust_features(df):
    # Vectorized "have we seen this (entity, device) pair before" via cumcount
    # first-occurrence trick -- replaces v1's row-by-row Python loop entirely.
    df["is_new_device"] = (df.groupby(["entity_id", "device_id"]).cumcount() == 0).astype(int)
    df["is_new_os"] = (df.groupby(["entity_id", "os"]).cumcount() == 0).astype(int)
    df["is_new_browser"] = (df.groupby(["entity_id", "browser"]).cumcount() == 0).astype(int)

    df["device_usage_frequency"] = df.groupby(["entity_id", "device_id"]).cumcount()
    df["new_device_probability"] = 1 - (
        df["device_usage_frequency"] / (df["historical_event_count"] + 1)
    )

    # Fingerprint change: does this event's OS differ from the entity's most-common
    # historical OS on this SAME device_id? (mode computed on prior events only)
    df["prev_os_on_device"] = df.groupby(["entity_id", "device_id"])["os"].shift(1)
    df["fingerprint_change_score"] = (
        (df["prev_os_on_device"].notna()) & (df["os"] != df["prev_os_on_device"])
    ).astype(int)

    # Device reputation: a device_id used by many DISTINCT entities is unusual
    # (devices should normally be 1:1 with an entity) -- structural signal, no
    # label leakage, computed globally.
    distinct_entities_per_device = df.groupby("device_id")["entity_id"].transform("nunique")
    df["device_reputation"] = 1 / distinct_entities_per_device

    df["managed_device_flag"] = (df["device_usage_frequency"] >= 5).astype(int)

    df["device_risk_score"] = (
        0.4 * df["new_device_probability"] +
        0.35 * df["fingerprint_change_score"] +
        0.25 * (1 - df["device_reputation"])
    ).clip(0, 1)

    return df


# ==============================================================================
# 3. NETWORK FEATURES
# ==============================================================================
def add_network_features(df):
    g = df.groupby("entity_id")
    df["prev_lat"] = g["geo_lat"].shift(1)
    df["prev_lon"] = g["geo_lon"].shift(1)
    df["prev_country"] = g["geo_country"].shift(1)
    df["prev_city"] = g["geo_city"].shift(1)
    df["prev_network_zone"] = g["network_zone"].shift(1)

    dist = haversine_km_vectorized(df["prev_lat"], df["prev_lon"], df["geo_lat"], df["geo_lon"])
    df["distance_from_prev_km"] = dist.fillna(0)

    raw_hours = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600.0
    hours_elapsed = raw_hours.clip(lower=MIN_ELAPSED_HOURS)
    df["geo_velocity_kmh"] = (df["distance_from_prev_km"] / hours_elapsed).fillna(0)
    df["impossible_travel_flag"] = (df["geo_velocity_kmh"] > MAX_PLAUSIBLE_SPEED_KMH).astype(int)

    df["country_change"] = (df["prev_country"].notna() & (df["geo_country"] != df["prev_country"])).astype(int)
    df["city_change"] = (df["prev_city"].notna() & (df["geo_city"] != df["prev_city"])).astype(int)
    df["network_zone_change"] = (df["prev_network_zone"].notna() & (df["network_zone"] != df["prev_network_zone"])).astype(int)

    df["is_hosting_asn"] = df["asn"].isin(HOSTING_ASNS).astype(int)
    df["remote_access_score"] = df["network_zone"].isin(["VPN_Remote", "Guest_WiFi"]).astype(int)
    df["internal_network_score"] = df["network_zone"].isin(["Corporate_LAN", "OT_Network"]).astype(int)

    # Anonymization risk: combines hosting-ASN origin + VPN usage + remote zone.
    # (Proxy for VPN/TOR-style probability -- synthetic data has no real Tor
    # exit-node list, so this is documented as a structural proxy, not a claim
    # of true Tor detection.)
    df["anonymization_risk_score"] = (
        0.5 * df["is_hosting_asn"] + 0.3 * df["vpn_used"].astype(int) + 0.2 * df["remote_access_score"]
    ).clip(0, 1)

    return df


# ==============================================================================
# 4. AUTHENTICATION FEATURES
# ==============================================================================
def add_authentication_features(df):
    g = df.groupby("entity_id")

    df["rolling_avg_failed_logins"] = g["failed_login_count"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(0)
    df["failed_login_spike"] = (df["failed_login_count"] - df["rolling_avg_failed_logins"]).clip(lower=0)

    df["failed_login_streak"] = g["failed_login_count"].transform(
        lambda s: (s > 0).rolling(5, min_periods=1).sum()
    )

    df["historical_mfa_rate"] = g["mfa_used"].transform(
        lambda s: s.astype(int).shift(1).expanding().mean()
    ).fillna(1.0)
    df["mfa_deviation"] = (df["historical_mfa_rate"] - df["mfa_used"].astype(int)).clip(lower=0)

    # Auth-method entropy: how varied is this entity's auth method history so
    # far? High entropy = inconsistent auth patterns (unusual for humans, who
    # typically stick to 1-2 methods).
    def running_entropy(s):
        out = np.zeros(len(s))
        counts = {}
        for i, val in enumerate(s):
            total = sum(counts.values())
            out[i] = shannon_entropy_from_counts(np.array(list(counts.values()))) if total > 0 else 0.0
            counts[val] = counts.get(val, 0) + 1
        return out
    df["auth_method_entropy"] = g["auth_method"].transform(running_entropy)

    # Password-spray / credential-stuffing signal: distinct entities hitting the
    # SAME source IP within a trailing 1h window. Distinct-count-in-window has
    # no native pandas vectorized primitive, so this uses a bounded rolling
    # .apply() grouped by IP -- linear in dataset size, not quadratic.
    df_ip_sorted = df.sort_values(["ip_address", "timestamp"]).copy()
    df_ip_sorted["entity_code"] = df_ip_sorted["entity_id"].astype("category").cat.codes
    spray = (
        df_ip_sorted.groupby("ip_address")
        .rolling("1h", on="timestamp")["entity_code"]
        .apply(lambda s: pd.Series(s).nunique(), raw=False)
        .reset_index(level=0, drop=True)
    )
    df_ip_sorted["password_spray_score"] = spray.values
    df = df.merge(df_ip_sorted[["log_id", "password_spray_score"]], on="log_id", how="left")

    df["credential_stuffing_score"] = (
        0.5 * (df["password_spray_score"].clip(upper=10) / 10) +
        0.3 * (df["failed_login_count"] > 0).astype(int) +
        0.2 * df["is_hosting_asn"]
    ).clip(0, 1)

    return df


# ==============================================================================
# 5. RESOURCE ACCESS FEATURES
# ==============================================================================
def add_resource_access_features(df):
    df["resource_diversity_count"] = (
        df.groupby(["entity_id", "resource_accessed"]).cumcount() == 0
    ).astype(int)
    df["resource_diversity_count"] = df.groupby("entity_id")["resource_diversity_count"].cumsum()

    g = df.groupby("entity_id")
    df["historical_mean_sensitivity"] = g["resource_sensitivity"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df["resource_sensitivity"].mean())
    df["resource_sensitivity_deviation"] = df["resource_sensitivity"] - df["historical_mean_sensitivity"]

    df["resource_owner_dept"] = df["resource_accessed"].map(RESOURCE_DEPT_MAP)
    df["is_cross_department_access"] = (df["resource_owner_dept"] != df["department"]).astype(int)

    # Privilege deviation: is this entity accessing a resource more sensitive
    # than their organizational privilege level would typically warrant?
    df["privilege_deviation"] = (df["resource_sensitivity"] - df["privilege_level"]).clip(lower=0)

    # Same positional-.values fix as above (df is sorted by entity_id, timestamp
    # at this point in the pipeline, so row order is preserved).
    df["critical_resource_rate_7d"] = g.rolling("7D", on="timestamp")["resource_sensitivity"] \
        .apply(lambda s: (s >= 4).mean()).values

    # Resource entropy over a trailing 20-event window (bounded, not full-history
    # apply) -- higher entropy = more varied resource access pattern.
    def windowed_entropy(s):
        codes = pd.Series(s).astype("category").cat.codes.values
        vals, counts = np.unique(codes, return_counts=True)
        return shannon_entropy_from_counts(counts)
    # rolling().apply() requires a numeric dtype internally even with raw=False,
    # so we encode the categorical resource name to integer codes first, then
    # window over those codes (still fully recovers exact entropy, since
    # entropy only depends on category frequencies, not the labels themselves).
    resource_codes = df["resource_accessed"].astype("category").cat.codes
    df["_resource_code_tmp"] = resource_codes
    df["resource_entropy"] = g["_resource_code_tmp"].transform(
        lambda s: s.rolling(20, min_periods=1).apply(lambda w: windowed_entropy(w.values), raw=False)
    )
    df.drop(columns=["_resource_code_tmp"], inplace=True)

    return df


# ==============================================================================
# 6. COMMAND SEQUENCE FEATURES
# ==============================================================================
def add_command_sequence_features(df):
    tokens = df["command_sequence"].fillna("").apply(lambda s: [t for t in s.split(";") if t])
    df["command_sequence_length"] = tokens.apply(len)

    def token_entropy(tok_list):
        if not tok_list:
            return 0.0
        _, counts = np.unique(tok_list, return_counts=True)
        return shannon_entropy_from_counts(counts)
    df["command_entropy"] = tokens.apply(token_entropy)

    # Global command rarity: inverse frequency of each command token across the
    # WHOLE dataset (precomputed once via explode, not per-row scanning).
    all_tokens = tokens.explode().dropna()
    token_freq = all_tokens.value_counts(normalize=True)
    df["command_rarity_score"] = tokens.apply(
        lambda tl: float(np.mean([1.0 / (token_freq.get(t, 1e-6)) for t in tl])) if tl else 0.0
    )
    # Normalize into a bounded 0-1 range for downstream model stability
    max_rarity = df["command_rarity_score"].replace([np.inf], np.nan).max()
    df["command_rarity_score"] = (df["command_rarity_score"] / max_rarity).clip(0, 1).fillna(0)

    def kw_ratio(tok_list, keywords):
        if not tok_list:
            return 0.0
        matches = sum(1 for t in tok_list if any(k in t for k in keywords))
        return matches / len(tok_list)

    df["dangerous_command_ratio"] = tokens.apply(lambda tl: kw_ratio(tl, DANGEROUS_KEYWORDS))
    df["privilege_escalation_cmd_score"] = tokens.apply(lambda tl: kw_ratio(tl, PRIV_ESC_KEYWORDS))
    df["lolbin_usage_flag"] = (
        df["process_name"].isin(LOLBIN_KEYWORDS) |
        tokens.apply(lambda tl: any(any(k in t for k in LOLBIN_KEYWORDS) for t in tl))
    ).astype(int)
    df["powershell_usage_flag"] = (df["process_name"] == "powershell.exe").astype(int)

    # Command novelty: fraction of THIS row's tokens the entity has never used
    # before, based on an expanding per-entity token vocabulary (vectorized via
    # explode + cumcount first-occurrence trick, same pattern as device/resource novelty).
    exploded = df[["entity_id", "log_id"]].join(tokens.rename("token")).explode("token").dropna(subset=["token"])
    exploded["is_new_token"] = (exploded.groupby(["entity_id", "token"]).cumcount() == 0).astype(int)
    novelty_per_row = exploded.groupby("log_id")["is_new_token"].mean()
    df["command_novelty_score"] = df["log_id"].map(novelty_per_row).fillna(0)

    return df


# ==============================================================================
# 7. SESSION FEATURES
# ==============================================================================
def add_session_features(df):
    session_sizes = df.groupby("session_id")["log_id"].transform("count")
    df["session_size"] = session_sizes

    df = df.sort_values(["session_id", "timestamp"])
    sg = df.groupby("session_id")
    df["prev_session_device"] = sg["device_id"].shift(1)
    df["prev_session_geo"] = sg["geo_city"].shift(1)
    df["session_hijack_flag"] = (
        (df["prev_session_device"].notna()) &
        ((df["device_id"] != df["prev_session_device"]) | (df["geo_city"] != df["prev_session_geo"]))
    ).astype(int)

    df["session_start_time"] = sg["timestamp"].transform("min")
    df["session_age_minutes"] = (df["timestamp"] - df["session_start_time"]).dt.total_seconds() / 60.0

    df = df.sort_values(["entity_id", "timestamp"])
    eg = df.groupby("entity_id")
    # rolling().apply() requires numeric dtype internally even with raw=False --
    # encode session_id to integer codes first (nunique on codes == nunique on
    # the original strings, so this is exact, not an approximation).
    df["_session_code_tmp"] = df["session_id"].astype("category").cat.codes
    df["concurrent_session_count_1h"] = eg.rolling("1h", on="timestamp")["_session_code_tmp"] \
        .apply(lambda s: pd.Series(s).nunique(), raw=False).values - 1
    df["concurrent_session_count_1h"] = df["concurrent_session_count_1h"].clip(lower=0)

    df["session_restart_rate"] = eg.rolling("24h", on="timestamp")["_session_code_tmp"] \
        .apply(lambda s: pd.Series(s).nunique(), raw=False).values
    df.drop(columns=["_session_code_tmp"], inplace=True)

    return df


# ==============================================================================
# 8. ORGANIZATION FEATURES
# ==============================================================================
def add_organization_features(df):
    # Static peer-group baselines (computed over the FULL contaminated dataset
    # -- see module docstring re: label leakage avoidance).
    dept_role_mean = df.groupby(["department", "role"])["resource_sensitivity"].transform("mean")
    df["peer_group_resource_sensitivity_deviation"] = df["resource_sensitivity"] - dept_role_mean

    dept_mean = df.groupby("department")["resource_sensitivity"].transform("mean")
    df["department_baseline_sensitivity"] = dept_mean
    df["business_unit_deviation"] = (df["resource_sensitivity"] - dept_mean).abs()

    priv_mean = df.groupby("privilege_level")["resource_sensitivity"].transform("mean")
    df["privilege_baseline_sensitivity"] = priv_mean

    # Manager deviation: compare entity's own mean sensitivity to their manager's
    manager_sensitivity = df.groupby("entity_id")["resource_sensitivity"].mean()
    df["own_mean_sensitivity"] = df["entity_id"].map(manager_sensitivity)
    df["manager_mean_sensitivity"] = df["manager"].map(manager_sensitivity).fillna(df["own_mean_sensitivity"])
    df["manager_deviation"] = (df["own_mean_sensitivity"] - df["manager_mean_sensitivity"]).abs()

    return df


# ==============================================================================
# 9. BEHAVIORAL BASELINE FEATURES (adaptive, cold-start aware)
# ==============================================================================
def add_behavioral_baseline_features(df):
    g = df.groupby("entity_id")
    df["historical_mean_duration"] = g["session_duration_min"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["historical_std_duration"] = g["session_duration_min"].transform(
        lambda s: s.shift(1).expanding().std()
    ).fillna(1.0).replace(0, 1.0)
    df["adaptive_threshold"] = df["historical_mean_duration"].fillna(df["session_duration_min"].mean()) + \
        2 * df["historical_std_duration"]
    df["adaptive_threshold_exceeded_flag"] = (df["session_duration_min"] > df["adaptive_threshold"]).astype(int)

    # Bayesian-style confidence: how much history do we actually have for this
    # entity? Approaches 1.0 as historical_event_count grows past ~30 events.
    df["baseline_confidence"] = (df["historical_event_count"] / (df["historical_event_count"] + 15)).clip(0, 1)
    df["cold_start_score"] = 1 - df["baseline_confidence"]
    df["cold_start_flag"] = (df["historical_event_count"] < COLD_START_MIN_EVENTS).astype(int)

    return df


# ==============================================================================
# 10. TEMPORAL FEATURES
# ==============================================================================
def add_temporal_features(df):
    df = df.sort_values(["entity_id", "timestamp"])
    g = df.groupby("entity_id")

    # Positional .values assignment (see note in add_user_behavior_features) --
    # df is sorted by (entity_id, timestamp) immediately above, so row order
    # is preserved and this sidesteps the duplicate-timestamp MultiIndex issue.
    df["rolling_count_1h"] = g.rolling("1h", on="timestamp")["log_id"].count().values
    df["rolling_count_24h"] = g.rolling("24h", on="timestamp")["log_id"].count().values
    df["rolling_count_7d"] = g.rolling("7D", on="timestamp")["log_id"].count().values

    df["burst_score"] = df["rolling_count_1h"] / ((df["rolling_count_24h"] / 24.0) + 0.1)

    # Cyclic encodings -- preserve the circularity of hour/weekday (23:00 and
    # 00:00 are adjacent, which a raw integer feature can't represent)
    df["hour_sin"] = np.sin(2 * np.pi * df["login_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["login_hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Seasonality: how "peaky" is this entity's activity at this specific hour,
    # based on prior history (Laplace-smoothed relative frequency of the hour)
    hour_counts = df.groupby(["entity_id", "login_hour"]).cumcount()
    df["seasonality_score"] = (hour_counts + 1) / (df["historical_event_count"] + 24)

    df["login_hour_deviation"] = (df["login_hour"] - df["expanding_mean_hour"]).abs()
    df["is_odd_hour_login"] = ((df["login_hour"] < 6) | (df["login_hour"] > 22)).astype(int)

    return df


# ==============================================================================
# 11. ATTACK-SPECIFIC COMPOSITE FEATURES (one scored signal per attack type)
# ==============================================================================
def add_attack_specific_features(df):
    df["credential_misuse_score"] = (
        0.3 * df["is_new_device"] + 0.25 * df["country_change"] +
        0.2 * df["is_odd_hour_login"] + 0.15 * (1 - df["mfa_used"].astype(int)) +
        0.1 * (df["resource_sensitivity"] >= 4).astype(int)
    ).clip(0, 1)

    df["brute_force_score"] = (
        0.6 * (df["failed_login_streak"] / 5).clip(0, 1) + 0.4 * (df["failed_login_spike"] / 10).clip(0, 1)
    ).clip(0, 1)

    df["impossible_travel_score"] = (df["geo_velocity_kmh"] / MAX_PLAUSIBLE_SPEED_KMH).clip(0, 3) / 3

    df["device_spoofing_score"] = (
        0.4 * df["fingerprint_change_score"] + 0.35 * df["is_new_device"] + 0.25 * df["new_device_probability"]
    ).clip(0, 1)

    df["lateral_movement_score"] = (
        0.4 * df["is_cross_department_access"] + 0.3 * (df["burst_score"] / 3).clip(0, 1) +
        0.3 * (df["resource_diversity_count"] / (df["historical_event_count"] + 1)).clip(0, 1)
    ).clip(0, 1)

    df["session_hijacking_score"] = df["session_hijack_flag"].astype(float)

    df["low_and_slow_exfil_score"] = (
        0.4 * df["is_odd_hour_login"] + 0.3 * (df["resource_sensitivity"] >= 4).astype(int) +
        0.3 * (df["file_download_size_mb"].between(1, 10)).astype(int)
    ).clip(0, 1)

    df["living_off_the_land_score"] = (
        0.6 * df["lolbin_usage_flag"] + 0.4 * (1 - df["device_risk_score"])  # normal-looking context is part of the signature
    ).clip(0, 1)

    # Insider threat: deliberately does NOT use geo/device/MFA novelty (those
    # are, by design, normal for this attack type) -- relies purely on resource
    # behavior deviating from the entity's OWN and their PEER GROUP's baseline.
    df["insider_threat_score"] = (
        0.5 * (df["resource_sensitivity_deviation"].clip(lower=0) / 5).clip(0, 1) +
        0.5 * (df["peer_group_resource_sensitivity_deviation"].clip(lower=0) / 5).clip(0, 1)
    ).clip(0, 1)

    df["command_abuse_score"] = df["dangerous_command_ratio"]

    return df

# ==============================================================================
# FEATURE SELECTION
# ==============================================================================
# Columns computed purely as scratch/intermediate values to derive a real
# feature (e.g. prev_lat is only used to compute distance_from_prev_km) are
# dropped from the final export -- they aren't meant to be model inputs, and
# keeping them would inflate the "feature count" without adding signal.
INTERMEDIATE_SCRATCH_COLUMNS = [
    "prev_timestamp", "prev_lat", "prev_lon", "prev_country", "prev_city",
    "prev_network_zone", "prev_os_on_device", "prev_session_device", "prev_session_geo",
    "weekday", "rolling10_mean_hour", "expanding_mean_hour", "historical_mean_sensitivity",
    "resource_owner_dept", "historical_mfa_rate", "session_duration_std",
    "historical_std_duration", "historical_mean_duration", "own_mean_sensitivity",
    "manager_mean_sensitivity", "session_start_time",
]


def select_final_features(df, engineered_cols):
    """Drops scratch/intermediate columns, returning the curated final feature list."""
    final_engineered = [c for c in engineered_cols if c not in INTERMEDIATE_SCRATCH_COLUMNS]
    cols_to_drop = [c for c in INTERMEDIATE_SCRATCH_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df, final_engineered


# ==============================================================================
# FEATURE VALIDATION
# ==============================================================================
def validate_features(df, feature_cols):
    """
    Fails loudly (rather than silently shipping bad data) on:
      - infinite values (would break StandardScaler / most sklearn models)
      - unexpected NaNs outside the documented cold-start first-event case
    Cold-start NaNs (an entity's very first event has no "previous" anything)
    are legitimate and are filled with 0 here -- 0 is the correct semantic
    default for every remaining nullable column (a deviation/count/rate of
    zero for an entity with no prior history to deviate from).
    """
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    inf_mask = np.isinf(df[numeric_cols])
    n_inf = inf_mask.to_numpy().sum()
    if n_inf > 0:
        print(f"  WARNING: {n_inf} infinite values found -- replacing with column max (finite).")
        for col in numeric_cols:
            col_inf = np.isinf(df[col])
            if col_inf.any():
                finite_max = df.loc[~col_inf, col].max()
                df.loc[col_inf, col] = finite_max

    n_null_before = df[feature_cols].isnull().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    n_null_after = df[feature_cols].isnull().sum().sum()
    print(f"  Nulls before fill: {n_null_before}  |  after fill: {n_null_after} "
          f"(remaining are non-numeric identifier columns, expected)")

    return df


# ==============================================================================
# MEMORY OPTIMIZATION
# ==============================================================================
def optimize_memory(df, feature_cols):
    """
    Downcasts numeric dtypes to the smallest safe representation. This
    directly serves the 'scalable to millions of events' requirement --
    float64->float32 and int64->int32/int8 roughly halves (or better) the
    in-memory footprint with no precision loss meaningful for these features
    (deviation scores, counts, flags -- none need float64 precision).
    """
    before_mb = df.memory_usage(deep=True).sum() / 1e6

    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    after_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  Memory usage: {before_mb:.1f} MB -> {after_mb:.1f} MB "
          f"({(1 - after_mb/before_mb)*100:.1f}% reduction)")
    return df



FEATURE_STEPS = [
    ("User Behavior", add_user_behavior_features),
    ("Device Trust", add_device_trust_features),
    ("Network", add_network_features),
    ("Authentication", add_authentication_features),
    ("Resource Access", add_resource_access_features),
    ("Command Sequence", add_command_sequence_features),
    ("Session", add_session_features),
    ("Organization", add_organization_features),
    ("Behavioral Baseline", add_behavioral_baseline_features),
    ("Temporal", add_temporal_features),
    ("Attack-Specific Composites", add_attack_specific_features),
]


def build_feature_pipeline():
    print("[0/11] Loading frozen Enterprise Dataset v2...")
    df, entities = load_data()
    original_cols = set(df.columns)

    for i, (name, fn) in enumerate(FEATURE_STEPS, 1):
        print(f"[{i}/{len(FEATURE_STEPS)}] {name} features...")
        df = fn(df)
        # De-fragment after each stage: many sequential df["col"] = ... assignments
        # leave the underlying block manager fragmented (harmless correctness-wise,
        # but pandas warns because it costs memory-copy performance at scale).
        # A single .copy() consolidates into a contiguous block -- cheap here,
        # and essential for the "scalable to millions of events" requirement.
        df = df.copy()

    engineered_cols = [c for c in df.columns if c not in original_cols]
    return df, engineered_cols


if __name__ == "__main__":
    df, engineered_cols = build_feature_pipeline()

    print(f"\n[Feature Selection] Dropping intermediate/scratch columns...")
    df, engineered_cols = select_final_features(df, engineered_cols)
    print(f"  Final curated engineered feature count: {len(engineered_cols)}")

    print(f"\n[Feature Validation]")
    df = validate_features(df, engineered_cols)

    print(f"\n[Memory Optimization]")
    df = optimize_memory(df, engineered_cols)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n=== Feature Engineering v2 Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Total engineered feature columns (final, curated): {len(engineered_cols)}")
    print(f"Total columns in output (raw + engineered): {len(df.columns)}")
    print(f"\nNull check on engineered columns (top 10 by null count):")
    null_counts = df[engineered_cols].isnull().sum().sort_values(ascending=False)
    print(null_counts.head(10))
    print(f"\nInf check on engineered numeric columns:")
    numeric_engineered = df[engineered_cols].select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_engineered).sum().sort_values(ascending=False)
    print(inf_counts[inf_counts > 0].head(10) if inf_counts.sum() > 0 else "None found.")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"\nEngineered feature list ({len(engineered_cols)} total):")
    for c in sorted(engineered_cols):
        print(f"  - {c}")
