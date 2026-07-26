"""
generate_logs_v2.py
====================
Enterprise Behavioral Anomaly Detection — Synthetic Access Log Generator v2.

REPLACES scripts/generate_logs.py per the architecture review findings:
v1 only modeled human employees and 5 attack types. v2 models the full
entity taxonomy and attack taxonomy required by the official Honeywell
schema (domain-agnostic: users, service accounts, and edge/OT devices, not
just IT/HR employees), plus adds fields needed for enterprise realism and
for the explainability layer to eventually produce MITRE ATT&CK-mapped
alerts.

WHAT'S NEW VS v1
----------------
- 6 entity types instead of 1: user, service_account, edge_device,
  iot_device, industrial_controller, server
- Organizational hierarchy: users have a manager, a role, and a numeric
  privilege_level (1-5) -- needed for Privilege Escalation and Insider
  Threat detection, which both require knowing what a user's *expected*
  privilege footprint looks like.
- Richer per-event schema: session_id, command_sequence, process_name,
  application, network_zone, ASN/ISP (residential vs. hosting-provider
  signal), resource_type + resource_sensitivity (graded, not binary).
- Realistic work schedules per entity type: office-hours humans,
  scheduled-job service accounts, always-on OT devices, weekend/holiday
  flags -- foundational for concept-drift and insider-drift work later.
- 12 attack scenarios (vs. 5 in v1), matching the official Honeywell
  taxonomy plus the additional scenarios from the enterprise brief:
  Credential Misuse, Credential Stuffing, Brute Force, Impossible Travel,
  Device Spoofing, Lateral Movement, Insider Threat, Privilege Escalation,
  Low-and-Slow Exfiltration, Command Abuse, Living-off-the-Land, Session
  Hijacking.
- Each attack row carries a `mitre_technique` field -- lays the groundwork
  for the MITRE ATT&CK-mapped explainability the enterprise brief asks for,
  without pretending that layer is fully built yet (it isn't; this is only
  the data-level hook for it).

WHAT'S DELIBERATELY NOT CHANGED
--------------------------------
Attacks are still built as deviations from each entity's own baseline
(the core methodology validated in v1) -- that part worked, so it's kept,
just extended to more entity types and more attack mechanics.

Output: dataset/raw_v2/access_logs_v2.csv
        dataset/raw_v2/entities_v2.csv
"""

import numpy as np
import pandas as pd
import uuid
import random
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# Simulation window
# ----------------------------------------------------------------------------
SIMULATION_DAYS = 45
START_DATE = datetime(2026, 6, 1)
HOLIDAYS = {START_DATE + timedelta(days=d) for d in [10, 24, 38]}  # 3 sim holidays

# ----------------------------------------------------------------------------
# Organization
# ----------------------------------------------------------------------------
DEPARTMENTS = [
    "Engineering", "Finance", "HR", "Sales", "Marketing",
    "IT_Ops", "Legal", "R&D", "Customer_Support", "Executive", "OT_Operations"
]

ROLES_BY_LEVEL = {
    1: ["Associate", "Analyst", "Support_Agent"],
    2: ["Senior_Analyst", "Engineer", "Specialist"],
    3: ["Senior_Engineer", "Team_Lead", "Senior_Specialist"],
    4: ["Manager", "Principal_Engineer", "Department_Head"],
    5: ["Director", "VP", "CxO"],
}

NUM_HUMAN_USERS = 500
NUM_SERVICE_ACCOUNTS = 40
NUM_EDGE_DEVICES = 60
NUM_IOT_DEVICES = 40
NUM_INDUSTRIAL_CONTROLLERS = 20
NUM_SERVERS = 25

OFFICE_LOCATIONS = {
    "HQ_NewYork":     (40.7128, -74.0060, "USA", "AS7018", "AT&T_Corporate"),
    "HQ_London":      (51.5074, -0.1278, "UK", "AS5089", "Virgin_Media_Business"),
    "Branch_Bengaluru": (12.9716, 77.5946, "India", "AS55836", "Reliance_Jio_Business"),
    "Branch_Singapore": (1.3521, 103.8198, "Singapore", "AS9506", "SingTel_Business"),
    "Branch_Berlin":  (52.5200, 13.4050, "Germany", "AS3320", "Deutsche_Telekom_Business"),
    "Branch_Toronto": (43.6532, -79.3832, "Canada", "AS812", "Rogers_Business"),
    "Branch_Sydney":  (-33.8688, 151.2093, "Australia", "AS1221", "Telstra_Business"),
}

# Hosting/VPN-provider ASNs -- attackers and compromised infra overwhelmingly
# originate from datacenter/hosting ASNs rather than residential ISPs. This
# is itself a real-world detection signal we now expose as a feature.
HOSTING_ASNS = [
    ("AS14061", "DigitalOcean"), ("AS16509", "Amazon_AWS"),
    ("AS20473", "Choopa_VPN_Hosting"), ("AS9009", "M247_VPN_Hosting"),
    ("AS36351", "SoftLayer_Hosting"),
]

FAR_LOCATIONS = [
    ("Lagos", 6.5244, 3.3792, "Nigeria"),
    ("Moscow", 55.7558, 37.6173, "Russia"),
    ("Bucharest", 44.4268, 26.1025, "Romania"),
    ("Jakarta", -6.2088, 106.8456, "Indonesia"),
    ("Sao_Paulo", -23.5505, -46.6333, "Brazil"),
    ("Manila", 14.5995, 120.9842, "Philippines"),
]

NETWORK_ZONES = ["Corporate_LAN", "DMZ", "VPN_Remote", "Cloud_VPC", "Guest_WiFi", "OT_Network"]

BROWSERS = ["Chrome", "Firefox", "Edge", "Safari", "N/A"]
OS_LIST = ["Windows_11", "Windows_10", "macOS", "Ubuntu_22", "RHEL_9", "Embedded_RTOS", "IoT_Linux"]
AUTH_METHODS = ["password", "token", "certificate", "biometric", "sso"]

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
    "OT_Operations": ["scada_hmi", "plc_controller_iface", "historian_db", "ot_network_gateway"],
}
RESOURCE_TYPE_TAGS = {  # resource -> resource_type category
    "source_code_repo": "code", "ci_cd_pipeline": "infra", "internal_wiki": "docs", "jira": "app",
    "erp_system": "app", "payroll_db": "data", "financial_reports": "data", "banking_portal": "app",
    "hr_management_system": "app", "employee_records": "data", "recruitment_portal": "app",
    "crm_system": "app", "sales_dashboard": "app", "contracts_repo": "docs",
    "cms": "app", "analytics_dashboard": "app", "social_media_tools": "app",
    "admin_console": "infra", "network_config": "infra", "server_room_access": "physical",
    "domain_controller": "infra", "compliance_db": "data", "legal_case_management": "app",
    "research_data_lake": "data", "patent_db": "data", "lab_equipment_control": "ot",
    "ticketing_system": "app", "customer_db": "data", "knowledge_base": "docs",
    "board_reports": "data", "strategic_plans_drive": "docs",
    "scada_hmi": "ot", "plc_controller_iface": "ot", "historian_db": "ot", "ot_network_gateway": "ot",
}
# Graded sensitivity 1 (low) - 5 (critical), replacing v1's binary flag
RESOURCE_SENSITIVITY = {
    "source_code_repo": 3, "ci_cd_pipeline": 4, "internal_wiki": 1, "jira": 1,
    "erp_system": 3, "payroll_db": 5, "financial_reports": 4, "banking_portal": 5,
    "hr_management_system": 3, "employee_records": 5, "recruitment_portal": 2,
    "crm_system": 2, "sales_dashboard": 1, "contracts_repo": 3,
    "cms": 1, "analytics_dashboard": 1, "social_media_tools": 1,
    "admin_console": 5, "network_config": 5, "server_room_access": 5, "domain_controller": 5,
    "compliance_db": 4, "legal_case_management": 3,
    "research_data_lake": 4, "patent_db": 5, "lab_equipment_control": 5,
    "ticketing_system": 1, "customer_db": 4, "knowledge_base": 1,
    "board_reports": 5, "strategic_plans_drive": 5,
    "scada_hmi": 5, "plc_controller_iface": 5, "historian_db": 4, "ot_network_gateway": 5,
}

PROCESS_BY_RESOURCE_TYPE = {
    "code": ["vscode.exe", "git.exe", "pycharm.exe"],
    "infra": ["kubectl.exe", "docker.exe", "terraform.exe", "jenkins-agent.exe"],
    "docs": ["chrome.exe", "acrobat.exe", "word.exe"],
    "app": ["chrome.exe", "outlook.exe", "teams.exe"],
    "data": ["excel.exe", "sql_client.exe", "tableau.exe"],
    "physical": ["badge_reader.svc"],
    "ot": ["scada_client.exe", "modbus_poll.exe", "historian_agent.exe"],
}

# Command vocabularies -- used to populate command_sequence for
# privileged/technical resource access, and to construct Command Abuse /
# Living-off-the-Land / Privilege Escalation attack signatures.
NORMAL_COMMANDS = ["ls", "cd", "cat", "git pull", "git status", "npm install",
                    "docker ps", "kubectl get pods", "select_report", "open_ticket", "update_record"]
ADMIN_COMMANDS = ["sudo su", "whoami", "net user", "icacls /grant", "reg query"]
PRIV_ESC_COMMANDS = ["net localgroup administrators /add", "runas /user:admin",
                      "sudo -l", "chmod 4755", "reg add HKLM\\...\\Debug"]
LOTL_COMMANDS = ["powershell.exe -enc SGVsbG8=", "wmic.exe process call create",
                  "certutil.exe -urlcache -f", "psexec.exe \\\\target", "mshta.exe http://"]
DESTRUCTIVE_COMMANDS = ["vssadmin delete shadows /all", "wmic shadowcopy delete",
                         "reg delete HKLM /f", "rm -rf /data/backups", "del /s /q C:\\backups"]
SCADA_COMMANDS = ["read_register", "poll_status", "modbus_read", "write_setpoint", "update_firmware"]

MITRE_MAP = {
    "Credential_Misuse": "T1078 - Valid Accounts",
    "Credential_Stuffing": "T1110.004 - Credential Stuffing",
    "Brute_Force": "T1110 - Brute Force",
    "Impossible_Travel": "T1078 - Valid Accounts (Anomalous Login)",
    "Device_Spoofing": "T1036 - Masquerading",
    "Lateral_Movement": "T1021 - Remote Services",
    "Insider_Threat": "T1078.002 - Valid Accounts (Insider)",
    "Privilege_Escalation": "T1068 - Exploitation for Privilege Escalation",
    "Low_and_Slow_Exfiltration": "T1030 - Data Transfer Size Limits",
    "Command_Abuse": "T1059 - Command and Scripting Interpreter",
    "Living_off_the_Land": "T1218 - System Binary Proxy Execution",
    "Session_Hijacking": "T1563 - Remote Service Session Hijacking",
    "None": "N/A",
}

ATTACK_TYPES = list(MITRE_MAP.keys())[:-1]  # exclude "None"


# ----------------------------------------------------------------------------
# Step 1: Build organizational hierarchy + all entity baselines
# ----------------------------------------------------------------------------
def build_organization():
    entities = []
    office_names = list(OFFICE_LOCATIONS.keys())

    # --- Human users, with manager chain ---
    dept_managers = {}
    user_counter = 0
    for dept in DEPARTMENTS:
        if dept == "Executive":
            continue
        mgr_id = f"U{user_counter+1:04d}"
        dept_managers[dept] = mgr_id
        user_counter += 1

    for i in range(NUM_HUMAN_USERS):
        user_id = f"U{i+1:04d}"
        is_manager_slot = user_id in dept_managers.values()
        dept = random.choice(DEPARTMENTS) if not is_manager_slot else \
            [d for d, m in dept_managers.items() if m == user_id][0]

        privilege_level = 4 if is_manager_slot else random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        if dept == "Executive":
            privilege_level = 5
        role = random.choice(ROLES_BY_LEVEL[privilege_level])
        manager = dept_managers.get(dept) if dept in dept_managers and dept_managers[dept] != user_id else (
            None if dept == "Executive" else dept_managers.get(dept))

        home_office = random.choice(office_names)
        lat, lon, country, asn, isp = OFFICE_LOCATIONS[home_office]
        num_devices = random.choice([1, 1, 1, 2])
        devices = [f"DEV-{uuid.uuid4().hex[:8]}" for _ in range(num_devices)]

        entities.append({
            "entity_id": user_id, "entity_type": "user", "department": dept,
            "role": role, "privilege_level": privilege_level, "manager": manager,
            "home_office": home_office, "home_lat": lat, "home_lon": lon, "home_country": country,
            "home_asn": asn, "home_isp": isp,
            "devices": devices, "usual_os": random.choice(OS_LIST[:5]),
            "usual_browser": random.choice(BROWSERS[:4]),
            "usual_auth_method": random.choices(AUTH_METHODS, weights=[0.3, 0.25, 0.15, 0.1, 0.2])[0],
            "login_start_hour": random.choice([7, 8, 9]),
            "login_end_hour": random.choice([17, 18, 19, 20]),
            "mfa_enabled": random.random() < 0.85,
            "vpn_habit": random.random() < 0.35,
            "works_weekends": random.random() < 0.08,
            "schedule_type": "office_hours",
        })

    # --- Service accounts: scheduled jobs, fixed network context, no "geo travel" ---
    for i in range(NUM_SERVICE_ACCOUNTS):
        sid = f"SVC{i+1:03d}"
        home_office = random.choice(office_names)
        lat, lon, country, asn, isp = OFFICE_LOCATIONS[home_office]
        entities.append({
            "entity_id": sid, "entity_type": "service_account", "department": "IT_Ops",
            "role": "Automation", "privilege_level": random.choice([2, 3, 4]), "manager": None,
            "home_office": home_office, "home_lat": lat, "home_lon": lon, "home_country": country,
            "home_asn": asn, "home_isp": isp,
            "devices": [f"SVR-{uuid.uuid4().hex[:8]}"], "usual_os": "RHEL_9", "usual_browser": "N/A",
            "usual_auth_method": "certificate",
            "login_start_hour": 0, "login_end_hour": 23,  # scheduled jobs run any hour
            "mfa_enabled": False, "vpn_habit": False, "works_weekends": True,
            "schedule_type": "scheduled_job",
        })

    # --- Edge/IoT/Industrial/Server devices: always-on, OT network, minimal drift ---
    device_specs = [
        (NUM_EDGE_DEVICES, "edge_device", "OT_Operations", "Embedded_RTOS"),
        (NUM_IOT_DEVICES, "iot_device", "OT_Operations", "IoT_Linux"),
        (NUM_INDUSTRIAL_CONTROLLERS, "industrial_controller", "OT_Operations", "Embedded_RTOS"),
        (NUM_SERVERS, "server", "IT_Ops", "RHEL_9"),
    ]
    counters = {}
    for count, etype, dept, os_ in device_specs:
        counters[etype] = 0
        for i in range(count):
            counters[etype] += 1
            did = f"{etype.upper()[:3]}{counters[etype]:03d}"
            home_office = random.choice(office_names)
            lat, lon, country, asn, isp = OFFICE_LOCATIONS[home_office]
            entities.append({
                "entity_id": did, "entity_type": etype, "department": dept,
                "role": "N/A", "privilege_level": 2, "manager": None,
                "home_office": home_office, "home_lat": lat, "home_lon": lon, "home_country": country,
                "home_asn": asn, "home_isp": isp,
                "devices": [f"HW-{uuid.uuid4().hex[:8]}"], "usual_os": os_, "usual_browser": "N/A",
                "usual_auth_method": "certificate",
                "login_start_hour": 0, "login_end_hour": 23,
                "mfa_enabled": False, "vpn_habit": False, "works_weekends": True,
                "schedule_type": "always_on",
            })

    return pd.DataFrame(entities)


# ----------------------------------------------------------------------------
# Step 2: Normal event generation
# ----------------------------------------------------------------------------
def pick_command_sequence(resource: str, entity_type: str) -> str:
    tag = RESOURCE_TYPE_TAGS.get(resource, "app")
    if tag not in ("infra", "physical", "ot") and entity_type == "user" and random.random() > 0.35:
        return ""  # most everyday app/doc access has no meaningful command sequence
    if tag == "ot":
        pool = SCADA_COMMANDS
    elif tag == "infra":
        pool = NORMAL_COMMANDS + ADMIN_COMMANDS
    else:
        pool = NORMAL_COMMANDS
    length = random.randint(2, 5)
    return ";".join(random.choices(pool, k=length))


def generate_normal_record(entity: dict, day: datetime) -> dict:
    dept = entity["department"] if entity["department"] in RESOURCE_TYPES else "IT_Ops"
    device = random.choice(entity["devices"])
    resource = random.choice(RESOURCE_TYPES[dept])
    resource_type = RESOURCE_TYPE_TAGS[resource]
    sensitivity = RESOURCE_SENSITIVITY[resource]

    is_weekend = day.weekday() >= 5
    is_holiday = day.date() in {h.date() for h in HOLIDAYS}

    if entity["schedule_type"] == "office_hours":
        if (is_weekend and not entity["works_weekends"]) or is_holiday:
            if random.random() > 0.05:  # 95% chance no login on off day
                return None
        hour = random.randint(entity["login_start_hour"], entity["login_end_hour"])
    else:
        hour = random.randint(0, 23)  # scheduled jobs / always-on devices run any hour

    ts = day.replace(hour=hour, minute=random.randint(0, 59))
    lat = entity["home_lat"] + np.random.normal(0, 0.02)
    lon = entity["home_lon"] + np.random.normal(0, 0.02)

    failed_logins = np.random.choice([0, 0, 0, 1], p=[0.85, 0.1, 0.03, 0.02])
    process = random.choice(PROCESS_BY_RESOURCE_TYPE[resource_type])
    network_zone = "OT_Network" if entity["department"] == "OT_Operations" else \
        ("VPN_Remote" if entity["vpn_habit"] and random.random() < 0.5 else "Corporate_LAN")

    return {
        "log_id": str(uuid.uuid4()), "session_id": str(uuid.uuid4()),
        "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
        "department": entity["department"], "role": entity["role"],
        "privilege_level": entity["privilege_level"], "manager": entity["manager"],
        "timestamp": ts, "is_weekend": int(is_weekend), "is_holiday": int(is_holiday),
        "geo_city": entity["home_office"], "geo_lat": round(lat, 4), "geo_lon": round(lon, 4),
        "geo_country": entity["home_country"], "asn": entity["home_asn"], "isp": entity["home_isp"],
        "ip_address": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "device_id": device, "device_fingerprint": f"{entity['usual_os']}|{device}",
        "os": entity["usual_os"], "browser": entity["usual_browser"],
        "auth_method": entity["usual_auth_method"],
        "network_zone": network_zone,
        "vpn_used": entity["vpn_habit"] and random.random() < 0.6,
        "mfa_used": entity["mfa_enabled"] and random.random() < 0.95,
        "failed_login_count": int(failed_logins),
        "resource_accessed": resource, "resource_type": resource_type, "resource_sensitivity": sensitivity,
        "process_name": process, "application": process,
        "command_sequence": pick_command_sequence(resource, entity["entity_type"]),
        "file_download_size_mb": round(np.random.exponential(5), 2),
        "session_duration_min": max(1, int(np.random.normal(45, 20))),
        "label_is_attack": 0, "attack_type": "None", "mitre_technique": "N/A",
    }


# ----------------------------------------------------------------------------
# Step 3: Attack generation (12 scenarios)
# ----------------------------------------------------------------------------
def _base_attack_row(entity, day, attack_type):
    row = generate_normal_record(entity, day)
    if row is None:  # off-day skip doesn't apply to attacks -- attacker doesn't respect your PTO
        row = generate_normal_record(entity, day.replace(hour=12))
        if row is None:
            row = {**generate_normal_record({**entity, "schedule_type": "always_on"}, day)}
    row["label_is_attack"] = 1
    row["attack_type"] = attack_type
    row["mitre_technique"] = MITRE_MAP[attack_type]
    return row


def attack_credential_misuse(entity, day):
    row = _base_attack_row(entity, day, "Credential_Misuse")
    far = random.choice(FAR_LOCATIONS)
    row["geo_city"], row["geo_lat"], row["geo_lon"], row["geo_country"] = far
    asn, isp = random.choice(HOSTING_ASNS)
    row["asn"], row["isp"] = asn, isp
    row["device_id"] = f"DEV-{uuid.uuid4().hex[:8]}"
    row["device_fingerprint"] = f"{random.choice(OS_LIST)}|{row['device_id']}"
    row["timestamp"] = row["timestamp"].replace(hour=random.choice([1, 2, 3, 4]))
    row["mfa_used"] = False
    sensitive = [r for r, s in RESOURCE_SENSITIVITY.items() if s >= 4]
    row["resource_accessed"] = random.choice(sensitive)
    row["resource_sensitivity"] = RESOURCE_SENSITIVITY[row["resource_accessed"]]
    return [row]


def attack_credential_stuffing(entity, day, all_entities):
    """Correlated multi-row attack: one attacker IP hits MANY entity_ids, mostly failing."""
    asn, isp = random.choice(HOSTING_ASNS)
    attacker_ip = f"{random.randint(20,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    targets = random.sample(all_entities, k=min(8, len(all_entities)))
    rows = []
    base_hour = random.randint(1, 5)
    for j, tgt in enumerate(targets):
        row = _base_attack_row(tgt, day, "Credential_Stuffing")
        row["ip_address"] = attacker_ip
        row["asn"], row["isp"] = asn, isp
        row["timestamp"] = day.replace(hour=base_hour, minute=(j * 3) % 60)
        row["failed_login_count"] = random.randint(3, 10) if random.random() < 0.85 else 0
        row["mfa_used"] = False
        rows.append(row)
    return rows


def attack_brute_force(entity, day):
    row = _base_attack_row(entity, day, "Brute_Force")
    row["failed_login_count"] = random.randint(6, 25)
    row["mfa_used"] = False
    row["ip_address"] = f"{random.randint(20,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    row["session_duration_min"] = random.randint(1, 5)
    return [row]


def attack_impossible_travel(entity, day):
    row = _base_attack_row(entity, day, "Impossible_Travel")
    far = random.choice(FAR_LOCATIONS)
    row["geo_city"], row["geo_lat"], row["geo_lon"], row["geo_country"] = far
    row["failed_login_count"] = 0
    row["mfa_used"] = True
    return [row]


def attack_device_spoofing(entity, day):
    row = _base_attack_row(entity, day, "Device_Spoofing")
    row["device_id"] = f"DEV-{uuid.uuid4().hex[:8]}"
    spoofed_os = random.choice([o for o in OS_LIST if o != entity["usual_os"]])
    row["device_fingerprint"] = f"{spoofed_os}|{row['device_id']}"
    row["os"] = spoofed_os
    row["mfa_used"] = random.random() < 0.3
    return [row]


def attack_lateral_movement(entity, day):
    """Burst of accesses across MULTIPLE unfamiliar resources/departments in one short window."""
    rows = []
    other_depts = [d for d in DEPARTMENTS if d != entity["department"] and d in RESOURCE_TYPES]
    burst_size = random.randint(3, 6)
    base_hour = random.randint(0, 23)
    for j in range(burst_size):
        row = _base_attack_row(entity, day, "Lateral_Movement")
        target_dept = random.choice(other_depts)
        row["resource_accessed"] = random.choice(RESOURCE_TYPES[target_dept])
        row["resource_type"] = RESOURCE_TYPE_TAGS[row["resource_accessed"]]
        row["resource_sensitivity"] = RESOURCE_SENSITIVITY[row["resource_accessed"]]
        row["session_duration_min"] = random.randint(1, 8)
        row["timestamp"] = day.replace(hour=base_hour, minute=(j * 7) % 60)
        rows.append(row)
    return rows


def attack_insider_threat(entity, day):
    """
    Deliberately SUBTLE: correct device, correct geo, correct hours, MFA fine.
    The only signal is gradually accessing more sensitive resources than the
    entity's role/peer group would normally touch. This is the official
    'ambiguous, used for false-positive tuning' edge case -- it should NOT
    be trivially catchable by the same rules that catch loud attacks.
    """
    row = _base_attack_row(entity, day, "Insider_Threat")
    high_sensitivity = [r for r, s in RESOURCE_SENSITIVITY.items() if s == 5]
    row["resource_accessed"] = random.choice(high_sensitivity)
    row["resource_sensitivity"] = 5
    row["resource_type"] = RESOURCE_TYPE_TAGS[row["resource_accessed"]]
    row["file_download_size_mb"] = round(np.random.exponential(15), 2)  # somewhat elevated, not extreme
    # deliberately: geo/device/mfa/hours all stay NORMAL (inherited from base row)
    return [row]


def attack_privilege_escalation(entity, day):
    row = _base_attack_row(entity, day, "Privilege_Escalation")
    row["resource_accessed"] = "admin_console" if entity["privilege_level"] < 4 else "domain_controller"
    row["resource_type"] = RESOURCE_TYPE_TAGS[row["resource_accessed"]]
    row["resource_sensitivity"] = RESOURCE_SENSITIVITY[row["resource_accessed"]]
    row["command_sequence"] = ";".join(random.sample(PRIV_ESC_COMMANDS, k=min(3, len(PRIV_ESC_COMMANDS))))
    row["process_name"] = "cmd.exe"
    return [row]


def attack_low_and_slow_exfil(entity, day):
    """Small, off-hours download -- attack signal only visible in AGGREGATE over many days."""
    row = _base_attack_row(entity, day, "Low_and_Slow_Exfiltration")
    row["timestamp"] = row["timestamp"].replace(hour=random.choice([23, 0, 1, 2, 3]))
    sensitive = [r for r, s in RESOURCE_SENSITIVITY.items() if s >= 4]
    row["resource_accessed"] = random.choice(sensitive)
    row["resource_sensitivity"] = RESOURCE_SENSITIVITY[row["resource_accessed"]]
    row["file_download_size_mb"] = round(random.uniform(1, 8), 2)  # deliberately SMALL, not a huge dump
    row["session_duration_min"] = random.randint(3, 10)
    return [row]


def attack_command_abuse(entity, day):
    row = _base_attack_row(entity, day, "Command_Abuse")
    row["command_sequence"] = ";".join(random.sample(DESTRUCTIVE_COMMANDS, k=min(3, len(DESTRUCTIVE_COMMANDS))))
    row["process_name"] = "powershell.exe"
    row["resource_accessed"] = "domain_controller" if entity["department"] != "OT_Operations" else "scada_hmi"
    row["resource_type"] = RESOURCE_TYPE_TAGS[row["resource_accessed"]]
    row["resource_sensitivity"] = RESOURCE_SENSITIVITY[row["resource_accessed"]]
    return [row]


def attack_living_off_the_land(entity, day):
    row = _base_attack_row(entity, day, "Living_off_the_Land")
    row["command_sequence"] = ";".join(random.sample(LOTL_COMMANDS, k=min(3, len(LOTL_COMMANDS))))
    row["process_name"] = random.choice(["powershell.exe", "wmic.exe", "certutil.exe"])
    # deliberately keeps device/geo NORMAL -- the whole point of LotL is blending in
    return [row]


def attack_session_hijacking(entity, day):
    """Same session_id continues, but IP/device changes abruptly mid-session -- no new auth event."""
    row1 = _base_attack_row(entity, day, "Session_Hijacking")
    shared_session = row1["session_id"]
    row2 = dict(row1)
    row2["log_id"] = str(uuid.uuid4())
    row2["session_id"] = shared_session  # SAME session continues
    far = random.choice(FAR_LOCATIONS)
    row2["geo_city"], row2["geo_lat"], row2["geo_lon"], row2["geo_country"] = far
    asn, isp = random.choice(HOSTING_ASNS)
    row2["asn"], row2["isp"] = asn, isp
    row2["device_id"] = f"DEV-{uuid.uuid4().hex[:8]}"
    row2["timestamp"] = row1["timestamp"] + timedelta(minutes=random.randint(2, 15))
    row2["failed_login_count"] = 0  # no new auth -- that's the point, it's a hijacked session
    return [row1, row2]


ATTACK_FUNCS = {
    "Credential_Misuse": attack_credential_misuse,
    "Brute_Force": attack_brute_force,
    "Impossible_Travel": attack_impossible_travel,
    "Device_Spoofing": attack_device_spoofing,
    "Lateral_Movement": attack_lateral_movement,
    "Insider_Threat": attack_insider_threat,
    "Privilege_Escalation": attack_privilege_escalation,
    "Low_and_Slow_Exfiltration": attack_low_and_slow_exfil,
    "Command_Abuse": attack_command_abuse,
    "Living_off_the_Land": attack_living_off_the_land,
    "Session_Hijacking": attack_session_hijacking,
    # Credential_Stuffing handled separately -- needs the full entity pool
}


# ----------------------------------------------------------------------------
# Step 4: Main generation loop
# ----------------------------------------------------------------------------
def generate_dataset(total_records=60000, attack_ratio=0.025, sim_days=SIMULATION_DAYS):
    print(f"[1/4] Building organization ({NUM_HUMAN_USERS} users, {NUM_SERVICE_ACCOUNTS} service accounts, "
          f"{NUM_EDGE_DEVICES+NUM_IOT_DEVICES+NUM_INDUSTRIAL_CONTROLLERS+NUM_SERVERS} devices)...")
    entities_df = build_organization()
    entities_list = entities_df.to_dict("records")
    user_entities = [e for e in entities_list if e["entity_type"] == "user"]

    num_attack_records_target = int(total_records * attack_ratio)
    num_normal_records = total_records - num_attack_records_target

    print(f"[2/4] Generating ~{num_normal_records} normal records...")
    normal_records = []
    attempts = 0
    while len(normal_records) < num_normal_records and attempts < num_normal_records * 3:
        entity = random.choice(entities_list)
        day = START_DATE + timedelta(days=random.randint(0, sim_days - 1))
        row = generate_normal_record(entity, day)
        attempts += 1
        if row is not None:
            normal_records.append(row)

    print(f"[3/4] Generating attack records across 12 scenarios (~{attack_ratio*100:.1f}% of total)...")
    attack_records = []
    scenarios = list(ATTACK_FUNCS.keys()) + ["Credential_Stuffing"]
    per_scenario_target = max(1, num_attack_records_target // len(scenarios))
    for scenario in scenarios:
        generated = 0
        tries = 0
        while generated < per_scenario_target and tries < per_scenario_target * 5:
            tries += 1
            entity = random.choice(user_entities if scenario in (
                "Insider_Threat", "Privilege_Escalation", "Credential_Stuffing") else entities_list)
            day = START_DATE + timedelta(days=random.randint(0, sim_days - 1))
            if scenario == "Credential_Stuffing":
                rows = attack_credential_stuffing(entity, day, user_entities)
            else:
                rows = ATTACK_FUNCS[scenario](entity, day)
            attack_records.extend(rows)
            generated += len(rows)

    print("[4/4] Merging, shuffling, saving...")
    all_records = normal_records + attack_records
    random.shuffle(all_records)
    logs_df = pd.DataFrame(all_records)
    logs_df.sort_values("timestamp", inplace=True)
    logs_df.reset_index(drop=True, inplace=True)

    return logs_df, entities_df


if __name__ == "__main__":
    import os

    logs_df, entities_df = generate_dataset()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_v2")
    os.makedirs(output_dir, exist_ok=True)
    logs_path = os.path.join(output_dir, "access_logs_v2.csv")
    entities_path = os.path.join(output_dir, "entities_v2.csv")

    logs_df.to_csv(logs_path, index=False)
    entities_df.drop(columns=["devices"]).to_csv(entities_path, index=False)

    print("\n=== v2 Dataset Generation Summary ===")
    print(f"Total records:   {len(logs_df)}")
    print(f"Attack records:  {logs_df['label_is_attack'].sum()} "
          f"({logs_df['label_is_attack'].mean()*100:.2f}%)")
    print(f"\nEntity type breakdown (all events):")
    print(logs_df["entity_type"].value_counts())
    print(f"\nAttack type breakdown:")
    print(logs_df[logs_df.label_is_attack == 1]["attack_type"].value_counts())
    print(f"\nSaved logs to:     {logs_path}")
    print(f"Saved entities to: {entities_path}")
