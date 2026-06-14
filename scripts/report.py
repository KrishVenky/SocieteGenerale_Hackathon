"""
WatchDog: Terminal Anomaly Report
----------------------------------
Usage:
  python scripts/report.py                   # top 20 alerts, threshold 65
  python scripts/report.py --top 10          # top 10
  python scripts/report.py --threshold 70    # stricter threshold
  python scripts/report.py --severity CRITICAL
  python scripts/report.py --no-color        # plain text (for piping)
  python scripts/report.py --demo            # show the 4 scripted personas
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

# Force UTF-8 output on Windows
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

SEV_COLOR = {
    "CRITICAL": RED,
    "HIGH":     YELLOW,
    "MEDIUM":   CYAN,
    "LOW":      GREEN,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            pass
    return []


def _col(code: str) -> str:
    return code if USE_COLOR else ""


def _sev_col(sev: str) -> str:
    return _col(SEV_COLOR.get(sev, WHITE))


def _anomaly_label(signals: list[str]) -> str:
    if "off_hours_export" in signals or "high_sens_export" in signals:
        return "DATA EXFILTRATION RISK"
    if "dormant_activation" in signals:
        return "DORMANT ACCOUNT REACTIVATION"
    if "failure_burst" in signals:
        return "BRUTE FORCE PATTERN"
    if "new_resource_off_hours_admin" in signals or "off_hours_admin" in signals:
        return "OFF-HOURS ADMIN OPERATION"
    if "service_acct_anomaly" in signals or "service_acct_new_resource" in signals:
        return "SERVICE ACCOUNT SCOPE VIOLATION"
    if "resource_scope_violation" in signals:
        return "UNAUTHORIZED RESOURCE ACCESS"
    if "is_new_ip" in signals:
        return "NEW SOURCE IP / UNUSUAL LOCATION"
    return "SUSPICIOUS ACCESS PATTERN"


_SIGNAL_LABELS = {
    "off_hours_export":          "Export action during off-hours — exfiltration risk",
    "high_sens_export":          "Export of high-sensitivity / restricted data",
    "dormant_activation":        "Account dormant before this event — possible compromise",
    "is_new_ip":                 "Source IP not seen in 90-day baseline history",
    "resource_scope_violation":  "System not in user's approved access list",
    "service_acct_anomaly":      "Service account active outside business hours",
    "service_acct_new_resource": "Service account accessed a resource type it never has before",
    "new_resource_off_hours_admin": "Admin operation on a new resource after hours",
    "failure_burst":             "Multiple auth failures followed by success — brute force",
    "is_export":                 "Export action on a restricted-classification system",
    "cross_dept_access":         "Accessed resources outside department scope",
    "off_hours_admin":           "Admin operation during off-hours — elevated risk window",
    "is_admin":                  "Admin-level operation — elevated impact if malicious",
    "volume_spike":              "Unusually high event volume vs 30-day rolling baseline",
    "sensitivity_escalation":    "Accessed higher sensitivity than historical maximum",
}


def _recommendation(score: int) -> str:
    if score >= 85:
        return "BLOCK + INVESTIGATE IMMEDIATELY + preserve audit trail"
    if score >= 70:
        return "ESCALATE to CISO + investigate within 1 hour"
    if score >= 50:
        return "FLAG for review + monitor user activity 24h"
    return "MONITOR — low risk, log for compliance"


# ── Rendering ─────────────────────────────────────────────────────────────────

def print_header() -> None:
    date_str = datetime.now().strftime("%Y-%m-%d  %H:%M")
    width = 62
    print()
    print(_col(BOLD) + _col(WHITE) + "=" * width + _col(RESET))
    print(_col(BOLD) + _col(WHITE) +
          f"  WATCHDOG: DATA ACCESS ANOMALY REPORT".ljust(width - 1) + _col(RESET))
    print(_col(DIM) +
          f"  {date_str}  ·  GCN + Bi-LSTM + Peer Group Analysis".ljust(width - 1) + _col(RESET))
    print(_col(BOLD) + _col(WHITE) + "=" * width + _col(RESET))
    print()


def print_summary(df: pd.DataFrame, threshold: int) -> None:
    n_alerts = int((df["risk_score"] >= threshold).sum())
    n_crit   = int((df["risk_score"] >= 85).sum())
    n_high   = int(((df["risk_score"] >= 70) & (df["risk_score"] < 85)).sum())
    n_med    = int(((df["risk_score"] >= 50) & (df["risk_score"] < 70)).sum())

    print(f"  {_col(WHITE)}Events processed :{_col(RESET)}  {len(df):,}")
    print(f"  {_col(WHITE)}Alerts (>={threshold})  :{_col(RESET)}  {n_alerts:,}")
    print(f"  {_col(RED  )}Critical  (>=85) :{_col(RESET)}  {n_crit}")
    print(f"  {_col(YELLOW)}High      (70-84):{_col(RESET)}  {n_high}")
    print(f"  {_col(CYAN )}Medium    (50-69):{_col(RESET)}  {n_med}")
    print()


def print_alert(rank: int, row: pd.Series | dict) -> None:
    if isinstance(row, dict):
        get = row.get
    else:
        get = lambda k, d=None: row.get(k, d) if hasattr(row, "get") else getattr(row, k, d)

    sev      = str(get("severity", "LOW"))
    score    = int(get("risk_score", 0))
    color    = _sev_col(sev)
    username = str(get("username", ""))
    action   = str(get("action", ""))
    resource = str(get("resource", ""))
    sg_name, sg_class = SG_RESOURCE_MAP.get(resource, (resource, "Unknown"))
    dept       = str(get("department", ""))
    sg_dept    = SG_DEPT_MAP.get(dept, dept)
    job_title  = str(get("job_title", ""))
    priv       = str(get("privilege_level", ""))
    ts         = str(get("timestamp", ""))
    time_class = str(get("time_classification", ""))
    status     = str(get("status", ""))
    source_ip  = str(get("source_ip", ""))
    inactive   = int(get("days_inactive", 0) or 0)
    peer_dev   = float(get("peer_deviation_pct", 0) or 0)
    signals    = _safe_list(get("triggered_signals"))
    mitre      = _safe_list(get("mitre_techniques"))

    anomaly_lbl = _anomaly_label(signals)

    print(f"  {color}{BOLD}Alert {rank}: {anomaly_lbl}{RESET}")
    print(f"  {_col(WHITE)}User       :{_col(RESET)} {_col(BOLD)}{username}{_col(RESET)}"
          f"  ({sg_dept} · {job_title} · {priv})")
    print(f"  {_col(WHITE)}Action     :{_col(RESET)} {action}")
    print(f"  {_col(WHITE)}System     :{_col(RESET)} {sg_name}"
          f"  {color}[{sg_class}]{_col(RESET)}")
    print(f"  {_col(WHITE)}Time       :{_col(RESET)} {ts}  {color}({time_class}){_col(RESET)}")
    print(f"  {_col(WHITE)}Source IP  :{_col(RESET)} {source_ip}")
    status_col = _col(RED) if status == "failure" else _col(GREEN)
    print(f"  {_col(WHITE)}Status     :{_col(RESET)} {status_col}{status}{_col(RESET)}")
    print(f"  {_col(WHITE)}Risk Score :{_col(RESET)} {color}{BOLD}{score}/100  {sev}{_col(RESET)}")
    print()

    # Context bullets
    context_lines: list[str] = []
    for sig in signals[:5]:
        label = _SIGNAL_LABELS.get(sig, sig.replace("_", " "))
        context_lines.append(label)
    if inactive > 0:
        context_lines.append(f"Account inactive {inactive}d before this event")
    if peer_dev > 0:
        context_lines.append(
            f"Peer deviation: +{peer_dev:.0f}% vs {sg_dept} department baseline"
        )
    if mitre:
        m_str = ", ".join(
            f"{m['technique_id']} ({m.get('tactic', '?')})" for m in mitre[:3]
        )
        context_lines.append(f"MITRE ATT&CK: {m_str}")

    if context_lines:
        print(f"  {_col(DIM)}Context:{_col(RESET)}")
        for line in context_lines:
            print(f"  {_col(DIM)}  - {line}{_col(RESET)}")
        print()

    rec = _recommendation(score)
    print(f"  {_col(WHITE)}Recommendation:{_col(RESET)} {color}{_col(BOLD)}{rec}{_col(RESET)}")
    print(f"  {_col(DIM)}{'─' * 58}{_col(RESET)}")
    print()


# ── Demo mode (4 scripted personas) ───────────────────────────────────────────

def run_demo() -> None:
    demo_path = ROOT / os.getenv(
        "DEMO_EVENTS_PATH", "data/processed/real/demo_replay/scripted_events.jsonl"
    )
    if not demo_path.exists():
        print(f"  ERROR: demo events not found at {demo_path}", file=sys.stderr)
        sys.exit(1)

    events: list[dict] = []
    with open(demo_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    events = sorted(
        events, key=lambda e: e.get("scored", {}).get("risk_score", 0), reverse=True
    )

    print_header()
    print(f"  {_col(BOLD)}Demo Replay — 4 Pre-Scripted Anomaly Personas{_col(RESET)}")
    print(f"  {_col(DIM)}Ground-truth labelled events · sorted by risk score{_col(RESET)}")
    print()

    for rank, ev in enumerate(events, 1):
        event   = ev.get("event", {})
        profile = ev.get("profile", {})
        scored  = ev.get("scored", {})

        flat = {
            **event,
            **profile,
            "risk_score":        scored.get("risk_score", 0),
            "severity":          scored.get("severity", "LOW"),
            "triggered_signals": scored.get("triggered_signals", []),
            "mitre_techniques":  scored.get("mitre_techniques", []),
            "peer_deviation_pct": ev.get("peer_deviation_pct", 0),
        }
        persona = ev.get("persona", "")
        if persona:
            print(f"  {_col(BLUE)}{_col(BOLD)}[ {persona} ]{_col(RESET)}")
        print_alert(rank, flat)

    _print_footer()


# ── Main report ───────────────────────────────────────────────────────────────

def run_report(args: argparse.Namespace) -> None:
    features_path = ROOT / os.getenv(
        "FEATURES_PATH", "data/processed/real/features.parquet"
    )
    if not features_path.exists():
        print(f"  ERROR: features not found at {features_path}", file=sys.stderr)
        print("  Run: python scripts/prepare_data.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    alerts = df[df["risk_score"] >= args.threshold].sort_values(
        "risk_score", ascending=False
    )
    if args.severity:
        alerts = alerts[alerts["severity"] == args.severity]
    alerts = alerts.head(args.top)

    print_header()
    print_summary(df, args.threshold)

    sev_order  = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sev_header = {
        "CRITICAL": f"{_col(RED)}{_col(BOLD)}[!!] CRITICAL ALERTS -- Immediate Investigation{_col(RESET)}",
        "HIGH":     f"{_col(YELLOW)}{_col(BOLD)}[!]  HIGH ALERTS -- Escalate within 1 hour{_col(RESET)}",
        "MEDIUM":   f"{_col(CYAN)}{_col(BOLD)}[~]  MEDIUM ALERTS -- Review within 24 hours{_col(RESET)}",
        "LOW":      f"{_col(GREEN)}{_col(BOLD)}[-]  LOW ALERTS -- Monitor{_col(RESET)}",
    }

    rank = 1
    for sev in sev_order:
        subset = alerts[alerts["severity"] == sev]
        if subset.empty:
            continue
        print(f"  {'─' * 58}")
        print(f"  {sev_header[sev]}")
        print(f"  {'─' * 58}")
        print()
        for _, row in subset.iterrows():
            print_alert(rank, row)
            rank += 1

    _print_footer()


def _print_footer() -> None:
    width = 62
    print(_col(BOLD) + _col(WHITE) + "=" * width + _col(RESET))
    print(_col(DIM) +
          "  WatchDog v1.0  ·  PS4 Evaluation: P=80.6%  R=83.3%  F1=0.820" + _col(RESET))
    print(_col(DIM) +
          "  Dashboard: http://localhost:8000  ·  uvicorn app.server:app --port 8000" + _col(RESET))
    print(_col(BOLD) + _col(WHITE) + "=" * width + _col(RESET))
    print()


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global USE_COLOR, RED, YELLOW, CYAN, GREEN, BLUE, WHITE, DIM, BOLD, RESET

    p = argparse.ArgumentParser(
        description="WatchDog: terminal anomaly report"
    )
    p.add_argument("--top",       type=int, default=20,
                   help="Top N alerts (default: 20)")
    p.add_argument("--threshold", type=int, default=65,
                   help="Risk score threshold (default: 65)")
    p.add_argument("--severity",  choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                   help="Filter to one severity level")
    p.add_argument("--demo",      action="store_true",
                   help="Show the 4 scripted demo personas instead of live queue")
    p.add_argument("--no-color",  action="store_true",
                   help="Disable ANSI colours (for piping/logging)")
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
