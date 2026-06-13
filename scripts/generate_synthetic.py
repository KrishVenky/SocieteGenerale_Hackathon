"""
generate_synthetic.py — Rule-based synthetic data generator

Produces realistic enterprise access log data with labeled anomalies.
Kept clearly separate from the real PS4 dataset — different user ID range,
different output directory, explicit data_source column.

Outputs (gitignored):
    data/synthetic/synthetic_logs.csv      ~150k events, 500 users
    data/synthetic/synthetic_profiles.csv  500 user profiles
    data/synthetic/synthetic_labels.csv    per-event anomaly labels

Run:
    python scripts/generate_synthetic.py
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "synthetic"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Scale parameters ──────────────────────────────────────────────────────────
# 150 users × ~300 events each = ~45k events, tractable in <5 min for features
# 20% anomaly rate = 30 users, each with 2 anomaly injections = ~80-120 anomaly events
N_USERS       = 150          # synthetic users (IDs USR00100–USR00249)
DAYS          = 365          # one calendar year, same as real data
START_DATE    = datetime(2025, 4, 21)
ANOMALY_RATE  = 0.20         # 20% of users get anomaly injection(s)

# ── Resources and sensitivity ─────────────────────────────────────────────────
RESOURCES = [
    "Customer_Vault", "GL_System", "HRIS", "PROD_DB", "Admin_Console",
    "SIEM", "Data_Lake", "BI_Tool", "File_Share", "Email_Archive",
]
SENSITIVITY = {
    "Customer_Vault": "high",  "GL_System": "high",   "HRIS": "high",
    "PROD_DB": "high",         "Admin_Console": "medium", "SIEM": "medium",
    "Data_Lake": "medium",     "BI_Tool": "low",       "File_Share": "low",
    "Email_Archive": "medium",
}

# Approved systems tokens that map to resources (mirrors real profiles)
RESOURCE_TO_TOKEN = {
    "PROD_DB": "PROD_DB",
    "SIEM": "SIEM",
    "Admin_Console": "ADMIN_SYS",
}

# ── Department archetypes ─────────────────────────────────────────────────────
# Each defines typical resource access, action mix, time pattern, volume

ARCHETYPES = {
    "Finance": {
        "resources":       ["GL_System", "Customer_Vault", "BI_Tool", "Data_Lake", "Email_Archive"],
        "res_weights":     [0.35, 0.28, 0.20, 0.10, 0.07],
        "actions":         ["sql_query", "file_access", "export_data", "login"],
        "act_weights":     [0.42, 0.30, 0.18, 0.10],
        "time_dist":       {"business_hours": 0.80, "unusual_hours": 0.12, "weekend": 0.08},
        "events_mean":     3.5, "events_std": 1.4,
        "approved_tokens": ["AD", "Azure_AD", "EMAIL", "Salesforce"],
        "privilege_dist":  {"user": 0.50, "power-user": 0.35, "admin": 0.10, "service-account": 0.05},
    },
    "Security": {
        "resources":       ["SIEM", "Admin_Console", "Data_Lake", "Email_Archive", "File_Share"],
        "res_weights":     [0.40, 0.30, 0.15, 0.10, 0.05],
        "actions":         ["api_call", "file_access", "sql_query", "admin_operation", "login"],
        "act_weights":     [0.35, 0.25, 0.20, 0.12, 0.08],
        "time_dist":       {"business_hours": 0.70, "unusual_hours": 0.20, "weekend": 0.10},
        "events_mean":     4.0, "events_std": 1.8,
        "approved_tokens": ["SIEM", "ADMIN_SYS", "Okta", "VPN", "Azure_AD"],
        "privilege_dist":  {"user": 0.35, "power-user": 0.40, "admin": 0.20, "service-account": 0.05},
    },
    "Engineering": {
        "resources":       ["PROD_DB", "Admin_Console", "Data_Lake", "File_Share", "SIEM"],
        "res_weights":     [0.32, 0.28, 0.20, 0.12, 0.08],
        "actions":         ["admin_operation", "api_call", "sql_query", "file_access", "login"],
        "act_weights":     [0.30, 0.30, 0.20, 0.12, 0.08],
        "time_dist":       {"business_hours": 0.65, "unusual_hours": 0.20, "weekend": 0.15},
        "events_mean":     4.5, "events_std": 2.0,
        "approved_tokens": ["GCP", "PROD_DB", "VPN", "AWS_IAM", "ADMIN_SYS"],
        "privilege_dist":  {"user": 0.40, "power-user": 0.30, "admin": 0.20, "service-account": 0.10},
    },
    "HR": {
        "resources":       ["HRIS", "File_Share", "Email_Archive", "BI_Tool"],
        "res_weights":     [0.50, 0.25, 0.15, 0.10],
        "actions":         ["sql_query", "file_access", "export_data", "login"],
        "act_weights":     [0.40, 0.35, 0.15, 0.10],
        "time_dist":       {"business_hours": 0.90, "unusual_hours": 0.07, "weekend": 0.03},
        "events_mean":     2.5, "events_std": 1.0,
        "approved_tokens": ["EMAIL", "ServiceNow", "Okta", "VPN"],
        "privilege_dist":  {"user": 0.60, "power-user": 0.30, "admin": 0.08, "service-account": 0.02},
    },
    "IT": {
        "resources":       ["Admin_Console", "SIEM", "Data_Lake", "File_Share", "PROD_DB"],
        "res_weights":     [0.35, 0.25, 0.20, 0.12, 0.08],
        "actions":         ["admin_operation", "api_call", "file_access", "sql_query", "login"],
        "act_weights":     [0.35, 0.28, 0.18, 0.12, 0.07],
        "time_dist":       {"business_hours": 0.70, "unusual_hours": 0.18, "weekend": 0.12},
        "events_mean":     5.0, "events_std": 2.2,
        "approved_tokens": ["ADMIN_SYS", "Azure_AD", "Okta", "ServiceNow", "VPN"],
        "privilege_dist":  {"user": 0.30, "power-user": 0.35, "admin": 0.25, "service-account": 0.10},
    },
    "Sales": {
        "resources":       ["Customer_Vault", "BI_Tool", "Email_Archive", "File_Share", "Data_Lake"],
        "res_weights":     [0.35, 0.28, 0.20, 0.12, 0.05],
        "actions":         ["file_access", "sql_query", "api_call", "login", "export_data"],
        "act_weights":     [0.35, 0.28, 0.20, 0.10, 0.07],
        "time_dist":       {"business_hours": 0.82, "unusual_hours": 0.13, "weekend": 0.05},
        "events_mean":     3.0, "events_std": 1.3,
        "approved_tokens": ["Salesforce", "EMAIL", "Azure_AD", "VPN"],
        "privilege_dist":  {"user": 0.55, "power-user": 0.30, "admin": 0.08, "service-account": 0.07},
    },
    "Legal": {
        "resources":       ["Email_Archive", "File_Share", "HRIS", "BI_Tool"],
        "res_weights":     [0.40, 0.30, 0.20, 0.10],
        "actions":         ["file_access", "sql_query", "login", "export_data"],
        "act_weights":     [0.45, 0.30, 0.15, 0.10],
        "time_dist":       {"business_hours": 0.85, "unusual_hours": 0.12, "weekend": 0.03},
        "events_mean":     2.8, "events_std": 1.1,
        "approved_tokens": ["EMAIL", "AD", "Azure_AD", "Salesforce"],
        "privilege_dist":  {"user": 0.55, "power-user": 0.35, "admin": 0.08, "service-account": 0.02},
    },
    "Operations": {
        "resources":       ["PROD_DB", "Data_Lake", "BI_Tool", "File_Share", "Email_Archive"],
        "res_weights":     [0.32, 0.28, 0.20, 0.12, 0.08],
        "actions":         ["api_call", "sql_query", "file_access", "login", "export_data"],
        "act_weights":     [0.35, 0.30, 0.18, 0.10, 0.07],
        "time_dist":       {"business_hours": 0.72, "unusual_hours": 0.15, "weekend": 0.13},
        "events_mean":     4.0, "events_std": 1.8,
        "approved_tokens": ["PROD_DB", "EMAIL", "ServiceNow", "VPN", "AD"],
        "privilege_dist":  {"user": 0.50, "power-user": 0.30, "admin": 0.12, "service-account": 0.08},
    },
    "Compliance": {
        "resources":       ["Email_Archive", "File_Share", "HRIS", "Data_Lake", "SIEM"],
        "res_weights":     [0.30, 0.25, 0.22, 0.13, 0.10],
        "actions":         ["file_access", "sql_query", "api_call", "login"],
        "act_weights":     [0.40, 0.35, 0.15, 0.10],
        "time_dist":       {"business_hours": 0.88, "unusual_hours": 0.09, "weekend": 0.03},
        "events_mean":     3.2, "events_std": 1.2,
        "approved_tokens": ["Azure_AD", "Okta", "EMAIL", "AD"],
        "privilege_dist":  {"user": 0.55, "power-user": 0.35, "admin": 0.08, "service-account": 0.02},
    },
    "Marketing": {
        "resources":       ["BI_Tool", "Data_Lake", "Email_Archive", "File_Share", "Customer_Vault"],
        "res_weights":     [0.35, 0.28, 0.18, 0.12, 0.07],
        "actions":         ["file_access", "api_call", "sql_query", "login", "export_data"],
        "act_weights":     [0.38, 0.28, 0.18, 0.10, 0.06],
        "time_dist":       {"business_hours": 0.83, "unusual_hours": 0.12, "weekend": 0.05},
        "events_mean":     3.0, "events_std": 1.2,
        "approved_tokens": ["Salesforce", "EMAIL", "Azure_AD", "AWS_IAM"],
        "privilege_dist":  {"user": 0.60, "power-user": 0.30, "admin": 0.05, "service-account": 0.05},
    },
    "Executive": {
        "resources":       ["BI_Tool", "Customer_Vault", "Email_Archive", "GL_System"],
        "res_weights":     [0.40, 0.28, 0.20, 0.12],
        "actions":         ["file_access", "login", "sql_query", "api_call"],
        "act_weights":     [0.40, 0.30, 0.20, 0.10],
        "time_dist":       {"business_hours": 0.78, "unusual_hours": 0.15, "weekend": 0.07},
        "events_mean":     2.0, "events_std": 0.9,
        "approved_tokens": ["AD", "EMAIL", "Salesforce", "Azure_AD"],
        "privilege_dist":  {"user": 0.60, "power-user": 0.30, "admin": 0.08, "service-account": 0.02},
    },
    "Support": {
        "resources":       ["File_Share", "Email_Archive", "Admin_Console", "BI_Tool"],
        "res_weights":     [0.38, 0.28, 0.22, 0.12],
        "actions":         ["file_access", "api_call", "login", "sql_query"],
        "act_weights":     [0.40, 0.28, 0.20, 0.12],
        "time_dist":       {"business_hours": 0.80, "unusual_hours": 0.13, "weekend": 0.07},
        "events_mean":     3.0, "events_std": 1.2,
        "approved_tokens": ["ServiceNow", "EMAIL", "ADMIN_SYS", "Okta"],
        "privilege_dist":  {"user": 0.60, "power-user": 0.25, "admin": 0.10, "service-account": 0.05},
    },
}

DEPARTMENTS = list(ARCHETYPES.keys())

FIRST_NAMES = ["aditya","alice","amitabh","arjun","charles","daniel","david","deepika",
               "diya","donald","edward","elena","george","harsh","isha","jacob","jason",
               "jeffrey","jonathan","joshua","karan","kenneth","kevin","leila","mark",
               "maria","matthew","meera","michael","nadia","neha","nicholas","nikhil",
               "nisha","paul","pooja","priya","richard","robert","ronald","ryan","sanjana",
               "sophia","stephen","steven","thomas","timothy","varun","vikram","william",
               "xiulan","yuki","zainab"]
LAST_NAMES  = ["anderson","becker","bhat","burke","chen","clark","colombo","dubois",
               "ghosh","gonzalez","gupta","harris","he","hernandez","huang","hwang",
               "iyer","jackson","jang","jo","johnson","jones","kang","kim","kumar",
               "lewis","li","lim","lopez","martin","martinez","menon","meyer","moore",
               "muller","murphy","o'brien","park","patel","petit","pillai","quinn",
               "ramirez","rao","rodriguez","romano","schulz","sharma","singh","smith",
               "sullivan","sun","taylor","thomas","thompson","wagner","wang","weber",
               "white","wilson","xu"]

JOB_TITLES = {
    "Finance":    ["Analyst","Senior Analyst","Director","Lead","Coordinator"],
    "Security":   ["Engineer","Developer","Analyst","Lead","Administrator"],
    "Engineering":["Engineer","Architect","Lead","Developer","Administrator"],
    "HR":         ["Officer","Coordinator","Analyst","Manager","Director"],
    "IT":         ["Specialist","Administrator","Manager","Executive","Engineer"],
    "Sales":      ["Coordinator","Architect","Officer","Manager","Director"],
    "Legal":      ["Officer","Analyst","Manager","Director","Lead"],
    "Operations": ["Coordinator","Lead","Analyst","Manager","Officer"],
    "Compliance": ["Engineer","Coordinator","Analyst","Manager","Lead"],
    "Marketing":  ["Coordinator","Manager","Executive","Analyst","Director"],
    "Executive":  ["Officer","Manager","Director","Executive","Developer"],
    "Support":    ["Engineer","Lead","Coordinator","Specialist","Analyst"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick(items, weights=None):
    return random.choices(items, weights=weights, k=1)[0]

def _time_class_to_hour(tc: str) -> int:
    if tc == "business_hours":
        return random.randint(9, 17)
    elif tc == "unusual_hours":
        return random.choices([7, 8, 18, 19, 20, 21], weights=[1,1,2,2,2,1])[0]
    elif tc == "night":
        return random.choices([0,1,2,3,4,22,23], weights=[2,2,2,1,1,1,1])[0]
    else:  # weekend — any hour
        return random.randint(8, 20)

def _ip_for_user(user_idx: int, dept: str) -> str:
    """Give each department a subnet so peer group IPs cluster naturally."""
    dept_octet = (list(ARCHETYPES.keys()).index(dept) + 10) % 250
    user_octet = (user_idx % 200) + 1
    return f"192.168.{dept_octet}.{user_octet}"

def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


# ── Profile generation ────────────────────────────────────────────────────────

def generate_profiles() -> pd.DataFrame:
    rows = []
    used_names = set()

    # Distribute users evenly across departments
    dept_cycle = (DEPARTMENTS * ((N_USERS // len(DEPARTMENTS)) + 1))[:N_USERS]
    random.shuffle(dept_cycle)

    for i in range(N_USERS):
        uid = f"USR{(i + 100):05d}"   # USR00100 – USR00599
        dept = dept_cycle[i]
        arch = ARCHETYPES[dept]

        # Unique username
        for _ in range(20):
            fn = random.choice(FIRST_NAMES)
            ln = random.choice(LAST_NAMES)
            uname = f"{fn}.{ln}"
            if uname not in used_names:
                used_names.add(uname)
                break

        priv = _pick(list(arch["privilege_dist"].keys()),
                     list(arch["privilege_dist"].values()))

        # Approved systems: archetype defaults + random extras
        base_tokens = arch["approved_tokens"][:]
        extras = random.sample(["AD","Azure_AD","Okta","VPN","GCP","AWS_IAM","EMAIL","ServiceNow"],
                               k=random.randint(0, 2))
        systems = list(dict.fromkeys(base_tokens + extras))
        random.shuffle(systems)

        hire_date = START_DATE - timedelta(days=random.randint(180, 1500))
        last_login = START_DATE - timedelta(days=random.randint(0, 90))
        days_inactive = (START_DATE - last_login).days

        rows.append({
            "user_id": uid,
            "username": uname,
            "email": f"{uname}@company.com",
            "department": dept,
            "job_title": random.choice(JOB_TITLES[dept]),
            "privilege_level": priv,
            "systems_access": "|".join(systems),
            "last_login": last_login.strftime("%Y-%m-%d"),
            "days_inactive": days_inactive,
            "is_active": True,
            "hire_date": hire_date.strftime("%Y-%m-%d"),
            "data_source": "synthetic",
        })

    return pd.DataFrame(rows)


# ── Event generation ──────────────────────────────────────────────────────────

def generate_normal_events(profile: pd.Series, arch: dict) -> list[dict]:
    """Generate ~1 year of normal events for a single user."""
    events = []
    current = START_DATE

    while current < START_DATE + timedelta(days=DAYS):
        # Skip some days (holidays, sick days, vacations)
        # 40% base skip + weekend skip keeps events sparse like real PS4 data
        if random.random() < 0.40 or (current.weekday() >= 5 and random.random() < 0.80):
            current += timedelta(days=1)
            continue

        tc_candidates = list(arch["time_dist"].keys())
        tc_weights = list(arch["time_dist"].values())
        # Force weekends to "weekend" classification
        if current.weekday() >= 5:
            tc = "weekend"
        else:
            tc = _pick(tc_candidates, tc_weights)

        n_events = max(1, int(np.random.normal(arch["events_mean"], arch["events_std"])))

        for _ in range(n_events):
            hour = _time_class_to_hour(tc)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = current.replace(hour=hour, minute=minute, second=second)

            resource = _pick(arch["resources"], arch["res_weights"])
            action = _pick(arch["actions"], arch["act_weights"])

            # Occasional failures (5%)
            status = "failure" if random.random() < 0.05 else "success"

            events.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": profile["user_id"],
                "username": profile["username"],
                "action": action,
                "resource": resource,
                "resource_sensitivity": SENSITIVITY[resource],
                "status": status,
                "source_ip": _ip_for_user(
                    int(profile["user_id"].replace("USR","")) % 200,
                    profile["department"]
                ),
                "time_classification": tc,
                "is_anomaly": False,
                "anomaly_type": "",
                "data_source": "synthetic",
            })

        current += timedelta(days=1)

    return events


# ── Anomaly injection ─────────────────────────────────────────────────────────

def inject_off_hours_export(profile: pd.Series) -> list[dict]:
    """T1048 — export high-sensitivity data at night from a new IP."""
    ts = START_DATE + timedelta(days=random.randint(200, 340),
                                hours=random.randint(1, 4),
                                minutes=random.randint(0, 59))
    # New IP: different subnet
    anomaly_ip = f"185.{random.randint(100,200)}.{random.randint(1,254)}.{random.randint(1,254)}"
    resource = random.choice(["Customer_Vault", "GL_System", "PROD_DB"])
    return [{
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": profile["user_id"],
        "username": profile["username"],
        "action": "export_data",
        "resource": resource,
        "resource_sensitivity": "high",
        "status": "success",
        "source_ip": anomaly_ip,
        "time_classification": "night",
        "is_anomaly": True,
        "anomaly_type": "off_hours_export",
        "data_source": "synthetic",
    }]


def inject_brute_force(profile: pd.Series) -> list[dict]:
    """T1110 — multiple failures then success."""
    ts = START_DATE + timedelta(days=random.randint(100, 300), hours=random.randint(2, 5))
    resource = random.choice(["Admin_Console", "PROD_DB", "SIEM"])
    events = []
    for i in range(random.randint(3, 5)):
        t = ts + timedelta(minutes=i * 3)
        events.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": profile["user_id"], "username": profile["username"],
            "action": "login", "resource": resource,
            "resource_sensitivity": SENSITIVITY[resource],
            "status": "failure",
            "source_ip": _ip_for_user(int(profile["user_id"].replace("USR","")) % 200, profile["department"]),
            "time_classification": "night",
            "is_anomaly": True, "anomaly_type": "brute_force",
            "data_source": "synthetic",
        })
    # Final success
    t = ts + timedelta(minutes=len(events) * 3 + 1)
    events.append({
        "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": profile["user_id"], "username": profile["username"],
        "action": "sql_query", "resource": resource,
        "resource_sensitivity": SENSITIVITY[resource],
        "status": "success",
        "source_ip": _ip_for_user(int(profile["user_id"].replace("USR","")) % 200, profile["department"]),
        "time_classification": "night",
        "is_anomaly": True, "anomaly_type": "brute_force",
        "data_source": "synthetic",
    })
    return events


def inject_scope_violation(profile: pd.Series) -> list[dict]:
    """T1530 — access resource that requires a token not in systems_access."""
    approved = set(profile["systems_access"].split("|"))
    # Find a resource whose required token they don't have
    out_of_scope = [r for r, tok in RESOURCE_TO_TOKEN.items() if tok not in approved]
    if not out_of_scope:
        out_of_scope = ["HRIS"]   # fallback: HRIS not usually approved
    resource = random.choice(out_of_scope)
    ts = START_DATE + timedelta(days=random.randint(150, 330),
                                hours=random.randint(1, 5), minutes=random.randint(0, 59))
    return [{
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": profile["user_id"], "username": profile["username"],
        "action": "sql_query", "resource": resource,
        "resource_sensitivity": SENSITIVITY[resource],
        "status": "success",
        "source_ip": _ip_for_user(int(profile["user_id"].replace("USR","")) % 200, profile["department"]),
        "time_classification": "night",
        "is_anomaly": True, "anomaly_type": "scope_violation",
        "data_source": "synthetic",
    }]


def inject_dormant_export(profile: pd.Series, all_events: list[dict]) -> list[dict]:
    """T1078 — user who was inactive suddenly exports high-sensitivity data."""
    # Find last event timestamp, then inject after a 60+ day gap
    user_events = [e for e in all_events if e["user_id"] == profile["user_id"]]
    if not user_events:
        return []
    last_ts = max(datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S") for e in user_events)
    gap_days = random.randint(60, 90)
    ts = last_ts + timedelta(days=gap_days, hours=random.randint(1, 4))
    if ts > START_DATE + timedelta(days=DAYS):
        return []
    resource = random.choice(["Customer_Vault", "GL_System", "PROD_DB"])
    return [{
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": profile["user_id"], "username": profile["username"],
        "action": "export_data", "resource": resource,
        "resource_sensitivity": "high", "status": "success",
        "source_ip": _ip_for_user(int(profile["user_id"].replace("USR","")) % 200, profile["department"]),
        "time_classification": "night",
        "is_anomaly": True, "anomaly_type": "dormant_export",
        "data_source": "synthetic",
    }]


def inject_service_acct_night(profile: pd.Series) -> list[dict]:
    """T1078.004 — service account accessing unexpected resource off-hours."""
    ts = START_DATE + timedelta(days=random.randint(100, 300), hours=random.randint(0, 5))
    # Access something clearly outside their archetype
    dept = profile["department"]
    arch = ARCHETYPES[dept]
    all_res = set(RESOURCES)
    typical = set(arch["resources"])
    unusual = list(all_res - typical)
    resource = random.choice(unusual) if unusual else "HRIS"
    return [{
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": profile["user_id"], "username": profile["username"],
        "action": "sql_query", "resource": resource,
        "resource_sensitivity": SENSITIVITY[resource],
        "status": "success",
        "source_ip": _ip_for_user(int(profile["user_id"].replace("USR","")) % 200, profile["department"]),
        "time_classification": "night",
        "is_anomaly": True, "anomaly_type": "service_acct_night",
        "data_source": "synthetic",
    }]


ANOMALY_INJECTORS = {
    "off_hours_export":  inject_off_hours_export,
    "brute_force":       inject_brute_force,
    "scope_violation":   inject_scope_violation,
    "service_acct_night": inject_service_acct_night,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gen] generating {N_USERS} synthetic users over {DAYS} days...")

    profiles_df = generate_profiles()

    all_events: list[dict] = []
    anomaly_counts: dict[str, int] = {k: 0 for k in ANOMALY_INJECTORS}
    n_anomaly_users = max(1, int(N_USERS * ANOMALY_RATE))
    anomaly_user_ids = set(
        random.sample(profiles_df["user_id"].tolist(), n_anomaly_users)
    )

    for i, (_, profile) in enumerate(profiles_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"[gen]   {i+1}/{N_USERS} users processed  ({len(all_events):,} events so far)")

        # Normal events
        normal = generate_normal_events(profile, ARCHETYPES[profile["department"]])
        all_events.extend(normal)

        # Inject anomaly for selected users — each gets 2 different types
        if profile["user_id"] in anomaly_user_ids:
            atypes_available = list(ANOMALY_INJECTORS.keys())
            # Ensure no service-account-only types for non-service accounts
            if profile["privilege_level"] != "service-account":
                atypes_available = [a for a in atypes_available if a != "service_acct_night"]
            if len(atypes_available) < 2:
                atypes_available = ["off_hours_export", "brute_force"]
            chosen = random.sample(atypes_available, k=min(2, len(atypes_available)))
            for atype in chosen:
                if atype == "dormant_export":
                    injected = inject_dormant_export(profile, all_events)
                else:
                    injected = ANOMALY_INJECTORS[atype](profile)
                if injected:
                    all_events.extend(injected)
                    anomaly_counts[atype] += len(injected)

    logs_df = pd.DataFrame(all_events).sort_values("timestamp").reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    logs_out     = OUT_DIR / "synthetic_logs.csv"
    profiles_out = OUT_DIR / "synthetic_profiles.csv"
    labels_out   = OUT_DIR / "synthetic_labels.csv"

    logs_df.to_csv(logs_out, index=False)
    profiles_df.to_csv(profiles_out, index=False)

    # Labels file: one row per event, mirrors the PS4 label format
    labels_df = logs_df[["timestamp", "user_id", "is_anomaly", "anomaly_type", "data_source"]].copy()
    labels_df.to_csv(labels_out, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_anomaly = int(logs_df["is_anomaly"].sum())
    n_total   = len(logs_df)
    print(f"\n{'-'*52}")
    print(f"  Users generated   : {N_USERS}")
    print(f"  Total events      : {n_total:,}")
    print(f"  Anomalous events  : {n_anomaly:,}  ({n_anomaly/n_total:.1%})")
    print(f"  Anomaly users     : {n_anomaly_users}")
    print(f"  By type           :")
    for atype, cnt in anomaly_counts.items():
        if cnt:
            print(f"    {atype:<25} {cnt}")
    print(f"{'-'*52}")
    print(f"  Saved:")
    print(f"    {logs_out}")
    print(f"    {profiles_out}")
    print(f"    {labels_out}")
    print(f"\n  Next: python scripts/prepare_data.py --synthetic")


if __name__ == "__main__":
    main()
