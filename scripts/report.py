"""
WatchDog: Terminal Anomaly Report
----------------------------------
Usage:
  python scripts/report.py                   # top 20 alerts, threshold 65
  python scripts/report.py --top 10
  python scripts/report.py --threshold 70
  python scripts/report.py --severity CRITICAL
  python scripts/report.py --no-color
  python scripts/report.py --demo            # 4 scripted demo personas
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd

from src.narrator import SG_RESOURCE_MAP, SG_DEPT_MAP

# ── ANSI colours ──────────────────────────────────────────────────────────────

USE_COLOR = True

def _c(code: str) -> str:
    return code if USE_COLOR else ""

RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SEV_COLOR = {"CRITICAL": RED, "HIGH": YELLOW, "MEDIUM": CYAN, "LOW": GREEN}

def sc(sev: str) -> str:
    return _c(SEV_COLOR.get(sev, WHITE))

# ── Data helpers ──────────────────────────────────────────────────────────────

def _safe_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            pass
    return []

def _tenure(hire_date: str) -> str:
    try:
        hd = pd.to_datetime(hire_date)
        months = max(0, int((datetime.now() - hd.to_pydatetime()).days / 30))
        if months < 12:
            return f"{months} months tenure"
        return f"{months // 12} yr tenure"
    except Exception:
        return ""

def _anomaly_label(signals: list[str]) -> str:
    if "off_hours_export" in signals or "high_sens_export" in signals:
        return "BULK EXPORT OF RESTRICTED DATA"
    if "dormant_activation" in signals:
        return "DORMANT ACCOUNT REACTIVATION"
    if "failure_burst" in signals:
        return "BRUTE FORCE / CREDENTIAL ATTACK"
    if "new_resource_off_hours_admin" in signals or "off_hours_admin" in signals:
        return "OFF-HOURS ADMIN OPERATION"
    if "service_acct_anomaly" in signals or "service_acct_new_resource" in signals:
        return "SERVICE ACCOUNT SCOPE VIOLATION"
    if "resource_scope_violation" in signals:
        return "UNAUTHORIZED RESOURCE ACCESS"
    if "is_new_ip" in signals:
        return "UNUSUAL SOURCE IP / LOCATION"
    return "SUSPICIOUS ACCESS PATTERN"

_SIGNAL_LABELS = {
    "off_hours_export":             "Export action during off-hours (exfiltration risk)",
    "high_sens_export":             "Export of high-sensitivity / Restricted data",
    "dormant_activation":           "Account was dormant before this event",
    "is_new_ip":                    "Source IP not seen in 90-day baseline history",
    "resource_scope_violation":     "System not in user's approved access list",
    "service_acct_anomaly":         "Service account active outside business hours",
    "service_acct_new_resource":    "Service account accessed resource type never seen before",
    "new_resource_off_hours_admin": "Admin operation on a new resource after hours",
    "off_hours_admin":              "Admin operation outside business hours",
    "failure_burst":                "Multiple auth failures followed by success",
    "is_export":                    "Export action on a Restricted-class system",
    "cross_dept_access":            "Accessed resources outside department scope",
    "is_admin":                     "Admin-level operation (elevated blast radius)",
    "is_new_resource":              "First-ever access to this resource type for this user",
    "volume_spike":                 "Unusually high event volume vs 30-day rolling baseline",
    "sensitivity_escalation":       "Accessed higher sensitivity level than historical max",
}

def _recommendation(score: int) -> str:
    if score >= 85:
        return "BLOCK + INVESTIGATE IMMEDIATELY + preserve audit trail"
    if score >= 70:
        return "ESCALATE to CISO + investigate within 1 hour"
    if score >= 50:
        return "FLAG for review + monitor user activity 24h"
    return "MONITOR"

# ── Print helpers ─────────────────────────────────────────────────────────────

W = 62

def hr(char: str = "=") -> None:
    print(_c(DIM) + char * W + _c(RESET))

def blank() -> None:
    print()

def print_header() -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    blank()
    hr("=")
    print(_c(BOLD) + _c(WHITE) + f"WATCHDOG: DATA ACCESS ANOMALY REPORT  {date_str}" + _c(RESET))
    print(_c(DIM) + "GCN + Bi-LSTM + Peer Group Analysis  |  PS4  |  Societe Generale" + _c(RESET))
    hr("=")
    blank()

def print_summary(df: pd.DataFrame, threshold: int) -> None:
    n_alerts = int((df["risk_score"] >= threshold).sum())
    n_crit   = int((df["risk_score"] >= 85).sum())
    n_high   = int(((df["risk_score"] >= 70) & (df["risk_score"] < 85)).sum())
    n_med    = int(((df["risk_score"] >= 50) & (df["risk_score"] < 70)).sum())
    print(
        f"Events processed: {len(df):,}  |  "
        f"Alerts (>={threshold}): {n_alerts}  |  "
        f"{_c(RED)}Critical: {n_crit}{_c(RESET)}  |  "
        f"{_c(YELLOW)}High: {n_high}{_c(RESET)}  |  "
        f"{_c(CYAN)}Medium: {n_med}{_c(RESET)}"
    )
    blank()

# ── Alert block ───────────────────────────────────────────────────────────────

def print_alert(rank: int, row: pd.Series | dict, persona: str = "") -> None:
    get = (lambda k, d=None: row.get(k, d)) if isinstance(row, dict) else \
          (lambda k, d=None: row.get(k, d))

    sev       = str(get("severity", "LOW"))
    score     = int(get("risk_score", 0))
    color     = sc(sev)
    username  = str(get("username", ""))
    action    = str(get("action", ""))
    resource  = str(get("resource", ""))
    sg_name, sg_class = SG_RESOURCE_MAP.get(resource, (resource, "Unknown"))
    dept      = str(get("department", ""))
    sg_dept   = SG_DEPT_MAP.get(dept, dept)
    job_title = str(get("job_title", ""))
    priv      = str(get("privilege_level", ""))
    ts        = str(get("timestamp", ""))
    time_cls  = str(get("time_classification", ""))
    status    = str(get("status", ""))
    source_ip = str(get("source_ip", ""))
    inactive  = int(get("days_inactive", 0) or 0)
    hire_date = str(get("hire_date", ""))
    peer_dev  = float(get("peer_deviation_pct", 0) or 0)
    signals   = _safe_list(get("triggered_signals"))
    mitre     = _safe_list(get("mitre_techniques"))

    label = _anomaly_label(signals)
    tenure = _tenure(hire_date)

    if persona:
        print(_c(BLUE) + _c(BOLD) + f"[ {persona} ]" + _c(RESET))

    print(color + _c(BOLD) + f"Alert {rank}: {label}" + _c(RESET))
    blank()

    user_meta = f"{sg_dept}"
    if job_title:
        user_meta += f", {job_title}"
    if tenure:
        user_meta += f", {tenure}"
    if inactive > 0:
        user_meta += f", {inactive}d inactive"

    print(f"User:      {_c(BOLD)}{username}{_c(RESET)} ({user_meta})")
    print(f"Privilege: {priv}")
    print(f"Action:    {action}")
    print(f"System:    {sg_name}  {color}[{sg_class}]{_c(RESET)}")
    print(f"Time:      {ts}  {color}({time_cls}){_c(RESET)}")
    print(f"Source IP: {source_ip}")
    status_col = _c(RED) if status == "failure" else _c(GREEN)
    print(f"Status:    {status_col}{status}{_c(RESET)}")
    print(f"Risk Score:{color} {_c(BOLD)}{score}/100  {sev}{_c(RESET)}")

    blank()

    # Context block
    ctx: list[str] = []
    for sig in signals:
        lbl = _SIGNAL_LABELS.get(sig, sig.replace("_", " "))
        ctx.append(lbl)
    if peer_dev > 0:
        ctx.append(f"Peer deviation +{peer_dev:.0f}% vs {sg_dept} baseline")
    if mitre:
        m_str = ", ".join(
            f"{m['technique_id']} ({m.get('tactic', '?')})" for m in mitre[:3]
        )
        ctx.append(f"MITRE ATT&CK: {m_str}")

    if ctx:
        print("Context:")
        for line in ctx:
            print(_c(DIM) + f"  - {line}" + _c(RESET))
        blank()

    rec = _recommendation(score)
    print(f"Recommendation: {color}{_c(BOLD)}{rec}{_c(RESET)}")
    hr("-")
    blank()


# ── Demo mode ─────────────────────────────────────────────────────────────────

def run_demo() -> None:
    demo_path = ROOT / os.getenv(
        "DEMO_EVENTS_PATH", "data/processed/real/demo_replay/scripted_events.jsonl"
    )
    if not demo_path.exists():
        print(f"ERROR: demo events not found at {demo_path}", file=sys.stderr)
        sys.exit(1)

    events: list[dict] = []
    with open(demo_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events = sorted(events, key=lambda e: e.get("scored", {}).get("risk_score", 0), reverse=True)

    print_header()
    print(_c(BOLD) + "Demo Replay  --  4 Pre-Scripted Anomaly Personas" + _c(RESET))
    print(_c(DIM) + "Ground-truth labelled events, sorted by risk score" + _c(RESET))
    blank()

    for rank, ev in enumerate(events, 1):
        event   = ev.get("event", {})
        profile = ev.get("profile", {})
        scored  = ev.get("scored", {})
        flat = {
            **event, **profile,
            "risk_score":         scored.get("risk_score", 0),
            "severity":           scored.get("severity", "LOW"),
            "triggered_signals":  scored.get("triggered_signals", []),
            "mitre_techniques":   scored.get("mitre_techniques", []),
            "peer_deviation_pct": ev.get("peer_deviation_pct", 0),
        }
        print_alert(rank, flat, persona=ev.get("persona", ""))

    _print_footer()


# ── Live report ───────────────────────────────────────────────────────────────

def run_report(args: argparse.Namespace) -> None:
    features_path = ROOT / os.getenv("FEATURES_PATH", "data/processed/real/features.parquet")
    if not features_path.exists():
        print(f"ERROR: {features_path} not found -- run scripts/prepare_data.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    alerts = df[df["risk_score"] >= args.threshold].sort_values("risk_score", ascending=False)
    if args.severity:
        alerts = alerts[alerts["severity"] == args.severity]
    alerts = alerts.head(args.top)

    print_header()
    print_summary(df, args.threshold)

    sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sev_labels = {
        "CRITICAL": ("Critical Alerts (Immediate Investigation)", RED),
        "HIGH":     ("High Alerts (Escalate within 1 hour)",     YELLOW),
        "MEDIUM":   ("Medium Alerts (Review within 24 hours)",   CYAN),
        "LOW":      ("Low Alerts (Monitor)",                     GREEN),
    }

    rank = 1
    for sev in sev_order:
        subset = alerts[alerts["severity"] == sev]
        if subset.empty:
            continue
        label, col = sev_labels[sev]
        hr("=")
        print(_c(col) + _c(BOLD) + label + _c(RESET))
        hr("=")
        blank()
        for _, row in subset.iterrows():
            print_alert(rank, row)
            rank += 1

    _print_footer()


def _print_footer() -> None:
    hr("=")
    print(_c(DIM) + "WatchDog v1.0  |  P=80.6%  R=83.3%  F1=0.820  |  PS4 PASS" + _c(RESET))
    print(_c(DIM) + "Dashboard --> uvicorn app.server:app --port 8000  then localhost:8000" + _c(RESET))
    hr("=")
    blank()


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global USE_COLOR, RED, YELLOW, CYAN, GREEN, BLUE, WHITE, DIM, BOLD, RESET

    p = argparse.ArgumentParser(description="WatchDog: terminal anomaly report")
    p.add_argument("--top",       type=int, default=20)
    p.add_argument("--threshold", type=int, default=65)
    p.add_argument("--severity",  choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    p.add_argument("--demo",      action="store_true", help="Show 4 scripted demo personas")
    p.add_argument("--no-color",  action="store_true")
    args = p.parse_args()

    if args.no_color:
        USE_COLOR = False
        RED = YELLOW = CYAN = GREEN = BLUE = WHITE = DIM = BOLD = RESET = ""

    if args.demo:
        run_demo()
    else:
        run_report(args)


if __name__ == "__main__":
    main()
