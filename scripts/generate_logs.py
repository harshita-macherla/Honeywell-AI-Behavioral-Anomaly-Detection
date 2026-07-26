"""
generate_logs.py
=================
Synthetic Enterprise Access Log Generator
------------------------------------------
Generates realistic access logs for a simulated enterprise of 500 employees,
establishes a per-user behavioral baseline (home location, usual devices,
usual login hours, department, role), then simulates normal daily activity
around that baseline PLUS a small percentage (~2%) of injected attack
sessions representing:

    - Credential Misuse
    - Brute Force
    - Impossible Travel
    - Device Spoofing
    - Lateral Movement

Design principle: attacks are generated as *deviations from a user's own
baseline*, not just "weird random rows". This matters because Stage 1
(Isolation Forest / LSTM-AE) learns what's normal per-user/per-department,
so the synthetic attacks need to actually violate that learned normalcy to
be a meaningful benchmark, and cold-start / concept-drift logic downstream
needs a believable baseline to shrink toward.

Output: dataset/raw/access_logs.csv  (~50,000 rows)
        dataset/raw/users.csv        (500 user baseline profiles, for reference)

Author: Generated for Honeywell AI Hackathon prototype
"""

import numpy as np
import pandas as pd
import uuid
import random
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------------
NUM_USERS = 500
TOTAL_RECORDS = 50_000
ATTACK_RATIO = 0.02  # ~2% of records are attacks
SIMULATION_DAYS = 45
START_DATE = datetime(2026, 6, 1)

DEPARTMENTS = [
    "Engineering", "Finance", "HR", "Sales", "Marketing",
    "IT_Ops", "Legal", "R&D", "Customer_Support", "Executive"
]

# Department -> typical office locations (city, country, lat, lon)
OFFICE_LOCATIONS = {
    "HQ_NewYork":     (40.7128, -74.0060, "USA"),
    "HQ_London":      (51.5074, -0.1278, "UK"),
    "Branch_Bengaluru": (12.9716, 77.5946, "India"),
    "Branch_Singapore": (1.3521, 103.8198, "Singapore"),
    "Branch_Berlin":  (52.5200, 13.4050, "Germany"),
    "Branch_Toronto": (43.6532, -79.3832, "Canada"),
    "Branch_Sydney":  (-33.8688, 151.2093, "Australia"),
}

# Plausible "attacker" / anomalous locations, deliberately far from all
# offices, used for impossible-travel and credential-misuse simulation.
FAR_LOCATIONS = [
    ("Lagos", 6.5244, 3.3792, "Nigeria"),
    ("Moscow", 55.7558, 37.6173, "Russia"),
    ("Bucharest", 44.4268, 26.1025, "Romania"),
    ("Jakarta", -6.2088, 106.8456, "Indonesia"),
    ("Sao_Paulo", -23.5505, -46.6333, "Brazil"),
    ("Manila", 14.5995, 120.9842, "Philippines"),
]

BROWSERS = ["Chrome", "Firefox", "Edge", "Safari"]
OS_LIST = ["Windows_11", "Windows_10", "macOS", "Ubuntu_22"]

RESOURCE_TYPES = {
    "Engineering": ["source_code_repo", "ci_cd_pipeline", "internal_wiki", "jira"],
    "Finance": ["erp_system", "payroll_db", "financial_reports", "banking_portal"],
    "HR": ["hr_management_system", "employee_records", "recruitment_portal"],
    "Sales": ["crm_system", "sales_dashboard", "contracts_repo"],
    "Marketing": ["cms", "analytics_dashboard", "social_media_tools"],
    "IT_Ops": ["admin_console", "network_config", "server_room_access", "domain_controller"],
    "Legal": ["contracts_repo", "compliance_db", "legal_case_management"],
    "R&D": ["research_data_lake", "patent_db", "lab_equipment_control"],
    "Customer_Support": ["ticketing_system", "customer_db", "knowledge_base"],
    "Executive": ["board_reports", "financial_reports", "strategic_plans_drive"],
}

SENSITIVE_RESOURCES = {
    "payroll_db", "banking_portal", "employee_records", "domain_controller",
    "server_room_access", "financial_reports", "board_reports",
    "strategic_plans_drive", "customer_db", "patent_db"
}

ATTACK_TYPES = [
    "Credential_Misuse", "Brute_Force", "Impossible_Travel",
    "Device_Spoofing", "Lateral_Movement"
]


# ----------------------------------------------------------------------------
# Step 1: Build per-user baseline profiles
# ----------------------------------------------------------------------------
def generate_user_profiles(num_users: int) -> pd.DataFrame:
    """
    Creates a stable behavioral baseline for each user: their department,
    home office location, usual devices/browsers/OS, and usual login-hour
    window. Downstream feature engineering and Stage-1 models rely on
    deviations FROM this baseline to define "anomalous".
    """
    profiles = []
    office_names = list(OFFICE_LOCATIONS.keys())

    for i in range(num_users):
        user_id = f"U{i+1:04d}"
        dept = random.choice(DEPARTMENTS)
        home_office = random.choice(office_names)
        lat, lon, country = OFFICE_LOCATIONS[home_office]

        # Each user typically uses 1-2 devices habitually
        num_devices = random.choice([1, 1, 1, 2])  # weighted towards 1
        devices = [f"DEV-{uuid.uuid4().hex[:8]}" for _ in range(num_devices)]

        usual_os = random.choice(OS_LIST)
        usual_browser = random.choice(BROWSERS)

        # Usual login window varies by role a bit (most 8am-7pm local)
        login_start_hour = random.choice([7, 8, 9])
        login_end_hour = random.choice([17, 18, 19, 20])

        # MFA enrollment: 85% of employees have MFA enabled (realistic gap)
        mfa_enabled = random.random() < 0.85

        # VPN usage habit: some users routinely VPN in (remote workers)
        vpn_habit = random.random() < 0.35

        profiles.append({
            "user_id": user_id,
            "department": dept,
            "home_office": home_office,
            "home_lat": lat,
            "home_lon": lon,
            "home_country": country,
            "devices": devices,
            "usual_os": usual_os,
            "usual_browser": usual_browser,
            "login_start_hour": login_start_hour,
            "login_end_hour": login_end_hour,
            "mfa_enabled": mfa_enabled,
            "vpn_habit": vpn_habit,
        })

    return pd.DataFrame(profiles)


# ----------------------------------------------------------------------------
# Step 2: Generate a single NORMAL log record for a given user
# ----------------------------------------------------------------------------
def generate_normal_record(user: dict, timestamp: datetime) -> dict:
    dept = user["department"]
    device = random.choice(user["devices"])
    resource = random.choice(RESOURCE_TYPES[dept])

    # Slight jitter around home location (simulates GPS/IP noise, still same city)
    lat = user["home_lat"] + np.random.normal(0, 0.02)
    lon = user["home_lon"] + np.random.normal(0, 0.02)

    # Login hour drawn from the user's usual window, with occasional overtime
    hour = random.randint(user["login_start_hour"], user["login_end_hour"])
    ts = timestamp.replace(hour=hour, minute=random.randint(0, 59))

    failed_logins = np.random.choice([0, 0, 0, 1], p=[0.85, 0.1, 0.03, 0.02])

    return {
        "log_id": str(uuid.uuid4()),
        "user_id": user["user_id"],
        "department": dept,
        "timestamp": ts,
        "ip_address": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                       if user["vpn_habit"] and random.random() < 0.5
                       else f"{random.randint(20,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "geo_city": user["home_office"],
        "geo_lat": round(lat, 4),
        "geo_lon": round(lon, 4),
        "geo_country": user["home_country"],
        "device_id": device,
        "os": user["usual_os"],
        "browser": user["usual_browser"],
        "vpn_used": user["vpn_habit"] and random.random() < 0.6,
        "mfa_used": user["mfa_enabled"] and random.random() < 0.95,
        "failed_login_count": int(failed_logins),
        "resource_accessed": resource,
        "is_sensitive_resource": resource in SENSITIVE_RESOURCES,
        "file_download_size_mb": round(np.random.exponential(5), 2),
        "session_duration_min": max(1, int(np.random.normal(45, 20))),
        "label_is_attack": 0,
        "attack_type": "None",
    }


# ----------------------------------------------------------------------------
# Step 3: Generate an ATTACK log record for a given user
# ----------------------------------------------------------------------------
def generate_attack_record(user: dict, timestamp: datetime, attack_type: str) -> dict:
    """
    Each attack type is crafted to violate a *specific* dimension of the
    user's baseline, since that's exactly what the Stage-1 anomaly detector
    and Stage-2 classifier need to learn to distinguish.
    """
    base = generate_normal_record(user, timestamp)
    base["label_is_attack"] = 1
    base["attack_type"] = attack_type

    if attack_type == "Credential_Misuse":
        # Correct credentials, but wildly atypical context: odd hour,
        # unfamiliar device, sensitive resource access, no MFA.
        far = random.choice(FAR_LOCATIONS)
        base["geo_city"], base["geo_lat"], base["geo_lon"], base["geo_country"] = far
        base["device_id"] = f"DEV-{uuid.uuid4().hex[:8]}"  # unseen device
        base["os"] = random.choice(OS_LIST)
        base["browser"] = random.choice(BROWSERS)
        base["timestamp"] = timestamp.replace(hour=random.choice([1, 2, 3, 4]), minute=random.randint(0, 59))
        base["mfa_used"] = False
        base["resource_accessed"] = random.choice(list(SENSITIVE_RESOURCES))
        base["is_sensitive_resource"] = True
        base["failed_login_count"] = random.choice([0, 1])

    elif attack_type == "Brute_Force":
        # Many failed logins in rapid succession before an eventual login,
        # often from an unfamiliar IP.
        base["failed_login_count"] = random.randint(6, 25)
        base["mfa_used"] = False
        base["ip_address"] = f"{random.randint(20,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        base["session_duration_min"] = random.randint(1, 5)

    elif attack_type == "Impossible_Travel":
        # Same user, a location that is geographically impossible to reach
        # from their last known location within the elapsed time window.
        far = random.choice(FAR_LOCATIONS)
        base["geo_city"], base["geo_lat"], base["geo_lon"], base["geo_country"] = far
        # Timestamp very close to a "previous" plausible login elsewhere
        # (the feature-engineering stage will compute velocity from this).
        base["failed_login_count"] = 0
        base["mfa_used"] = True  # travel doesn't necessarily fail MFA — that's what makes it subtle

    elif attack_type == "Device_Spoofing":
        # Same-ish location/hours, but device fingerprint doesn't match
        # anything in the user's device history, and OS/browser combo is
        # inconsistent (e.g. claims same OS but header signature mismatches
        # — represented here simply as a completely new, unlinked device).
        base["device_id"] = f"DEV-{uuid.uuid4().hex[:8]}"
        base["os"] = random.choice([o for o in OS_LIST if o != user["usual_os"]])
        base["browser"] = random.choice(BROWSERS)
        base["mfa_used"] = random.random() < 0.3

    elif attack_type == "Lateral_Movement":
        # Access to resources far outside the user's own department,
        # especially sensitive/admin resources, often in a burst.
        other_depts = [d for d in DEPARTMENTS if d != user["department"]]
        target_dept = random.choice(other_depts)
        base["resource_accessed"] = random.choice(RESOURCE_TYPES[target_dept])
        base["is_sensitive_resource"] = base["resource_accessed"] in SENSITIVE_RESOURCES
        base["session_duration_min"] = random.randint(1, 10)
        base["failed_login_count"] = random.choice([0, 1, 2])

    return base


# ----------------------------------------------------------------------------
# Step 4: Main generation loop
# ----------------------------------------------------------------------------
def generate_dataset(num_users=NUM_USERS, total_records=TOTAL_RECORDS,
                      attack_ratio=ATTACK_RATIO, sim_days=SIMULATION_DAYS):

    print(f"[1/4] Generating {num_users} user baseline profiles...")
    users_df = generate_user_profiles(num_users)
    users_list = users_df.to_dict("records")

    num_attack_records = int(total_records * attack_ratio)
    num_normal_records = total_records - num_attack_records

    print(f"[2/4] Generating {num_normal_records} normal records...")
    normal_records = []
    for _ in range(num_normal_records):
        user = random.choice(users_list)
        day_offset = random.randint(0, sim_days - 1)
        ts = START_DATE + timedelta(days=day_offset)
        normal_records.append(generate_normal_record(user, ts))

    print(f"[3/4] Generating {num_attack_records} attack records "
          f"(~{attack_ratio*100:.1f}% of total, distributed across 5 attack types)...")
    attack_records = []
    for _ in range(num_attack_records):
        user = random.choice(users_list)
        day_offset = random.randint(0, sim_days - 1)
        ts = START_DATE + timedelta(days=day_offset)
        attack_type = random.choice(ATTACK_TYPES)
        attack_records.append(generate_attack_record(user, ts, attack_type))

    print("[4/4] Merging, shuffling, and saving to disk...")
    all_records = normal_records + attack_records
    random.shuffle(all_records)
    logs_df = pd.DataFrame(all_records)
    logs_df.sort_values("timestamp", inplace=True)
    logs_df.reset_index(drop=True, inplace=True)

    return logs_df, users_df


if __name__ == "__main__":
    import os

    logs_df, users_df = generate_dataset()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")
    os.makedirs(output_dir, exist_ok=True)

    logs_path = os.path.join(output_dir, "access_logs.csv")
    users_path = os.path.join(output_dir, "users.csv")

    logs_df.to_csv(logs_path, index=False)
    users_df.drop(columns=["devices"]).to_csv(users_path, index=False)  # devices list saved separately if needed

    print("\n=== Dataset Generation Summary ===")
    print(f"Total records:        {len(logs_df)}")
    print(f"Attack records:       {logs_df['label_is_attack'].sum()} "
          f"({logs_df['label_is_attack'].mean()*100:.2f}%)")
    print("\nAttack type breakdown:")
    print(logs_df[logs_df.label_is_attack == 1]["attack_type"].value_counts())
    print(f"\nSaved logs to:  {logs_path}")
    print(f"Saved users to: {users_path}")
