"""
prepare_data.py , PS4 data pipeline

Run on real PS4 data (default):
    python scripts/prepare_data.py

Run on synthetic data (separate outputs, separate evaluation):
    python scripts/prepare_data.py --synthetic

Outputs (all gitignored):
    data/processed/real/features.parquet        real ML-ready scored events
    data/processed/real/user_baselines.parquet  real per-user rolling stats
    data/processed/real/demo_replay/scripted_events.jsonl
    data/processed/synthetic/features.parquet   synthetic scored events
    data/processed/synthetic/user_baselines.parquet
    models/checkpoints/detector_real.pt         BiLSTM trained on real data
    models/checkpoints/detector_synthetic.pt    BiLSTM trained on synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.baseline import BaselineEngine
from src.graph_builder import GraphBuilder
from src.detector import AnomalyDetector
from src.ranker import RiskRanker

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PS4 data pipeline")
parser.add_argument("--synthetic", action="store_true",
                    help="Run on synthetic data (data/synthetic/) with separate outputs")
args, _ = parser.parse_known_args()

SYNTHETIC_MODE = args.synthetic

# ── Config , paths diverge by mode ───────────────────────────────────────────
if SYNTHETIC_MODE:
    LOGS_PATH     = ROOT / "data/synthetic/synthetic_logs.csv"
    PROFILES_PATH = ROOT / "data/synthetic/synthetic_profiles.csv"
    LABELS_PATH   = ROOT / "data/synthetic/synthetic_labels.csv"
    OUT_DIR       = ROOT / "data/processed/synthetic"
    OUT_MODEL     = ROOT / "models/checkpoints/detector_synthetic.pt"
    DATA_TAG      = "synthetic"
else:
    LOGS_PATH = ROOT / os.getenv(
        "PS4_LOGS_PATH",
        "data/Problem_04_Data_Access/sample_data/data_access_logs.csv"
    )
    PROFILES_PATH = ROOT / os.getenv(
        "PS4_PROFILES_PATH",
        "data/Problem_04_Data_Access/sample_data/user_profiles.csv"
    )
    LABELS_PATH = ROOT / os.getenv(
        "PS4_LABELS_PATH",
        "data/Problem_04_Data_Access/sample_data/data_access_labels.csv"
    )
    OUT_DIR   = ROOT / "data/processed/real"
    OUT_MODEL = ROOT / "models/checkpoints/detector_real.pt"
    DATA_TAG  = "real"

OUT_FEATURES  = OUT_DIR / "features.parquet"
OUT_BASELINES = OUT_DIR / "user_baselines.parquet"
OUT_DEMO      = OUT_DIR / "demo_replay/scripted_events.jsonl"

# ── Demo personas ─────────────────────────────────────────────────────────────
# Four injected events with is_anomaly=True, used as labelled ground truth.

DEMO_EVENTS = [
    {
        "timestamp": "2026-04-16 02:14:00",
        "user_id": "USR00044",
        "username": "pooja.mishra",
        "action": "export_data",
        "resource": "Customer_Vault",
        "resource_sensitivity": "high",
        "status": "success",
        "source_ip": "185.234.11.47",      # external IP , never seen in baseline
        "time_classification": "night",
        "is_anomaly": True,
        "anomaly_type": "pre_resignation_exfiltration",
        "persona": "Finance Pre-Resignation",
        "mitre_label": "T1048 / T1078",
        "risk_label": 96,
    },
    {
        "timestamp": "2026-04-16 23:47:00",
        "user_id": "USR00015",
        "username": "sophia.white",
        "action": "admin_operation",
        "resource": "Admin_Console",
        "resource_sensitivity": "medium",
        "status": "success",
        "source_ip": "192.168.16.233",
        "time_classification": "night",
        "is_anomaly": True,
        "anomaly_type": "audit_log_tamper",
        "persona": "IT Audit Evasion",
        "mitre_label": "T1562.001",
        "risk_label": 91,
    },
    {
        "timestamp": "2026-04-16 03:22:00",
        "user_id": "USR00005",
        "username": "george.lim",
        "action": "export_data",
        "resource": "PROD_DB",
        "resource_sensitivity": "high",
        "status": "success",
        "source_ip": "192.168.69.154",
        "time_classification": "night",
        "is_anomaly": True,
        "anomaly_type": "dormant_account_reactivation",
        "persona": "Dormant Account",
        "mitre_label": "T1078",
        "risk_label": 94,
    },
    {
        "timestamp": "2026-04-16 03:05:00",
        "user_id": "USR00002",
        "username": "kenneth.moore",
        "action": "sql_query",
        "resource": "HRIS",
        "resource_sensitivity": "high",
        "status": "success",
        "source_ip": "192.168.63.221",
        "time_classification": "night",
        "is_anomaly": True,
        "anomaly_type": "service_acct_scope_violation",
        "persona": "Service Account Cross-Dept",
        "mitre_label": "T1530 / T1078.004",
        "risk_label": 88,
    },
]


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[load] logs  : {LOGS_PATH}")
    print(f"[load] profiles: {PROFILES_PATH}")

    logs = pd.read_csv(LOGS_PATH, parse_dates=["timestamp"])
    profiles = pd.read_csv(PROFILES_PATH)

    logs = logs.sort_values("timestamp").reset_index(drop=True)

    print(f"[load] {len(logs):,} events  |  {profiles['user_id'].nunique()} users")
    print(f"[load] date range: {logs['timestamp'].min().date()} to {logs['timestamp'].max().date()}")
    return logs, profiles


# ── Feature computation ───────────────────────────────────────────────────────

def compute_all_features(
    logs: pd.DataFrame,
    profiles: pd.DataFrame,
    baseline_engine: BaselineEngine,
    graph_builder: GraphBuilder,
    detector: AnomalyDetector,
    ranker: RiskRanker,
) -> pd.DataFrame:
    """Score every event in the log. Returns a DataFrame with feature columns appended."""
    logs = logs.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    profile_map = profiles.set_index("user_id").to_dict("index")

    rows = []
    for user_id, user_events in logs.groupby("user_id"):
        user_events = user_events.sort_values("timestamp")
        profile_row = profile_map.get(user_id, {})
        profile_series = pd.Series(profile_row)
        dept = profile_row.get("department", "")

        event_list = [row for _, row in user_events.iterrows()]

        for i, event in enumerate(event_list):
            prior_events = user_events.iloc[:i].copy()

            # Behavioral features
            feat = baseline_engine.score_event(event, prior_events, profile_series)

            # LSTM score on last seq_len events
            window = event_list[max(0, i - detector.seq_len + 1) : i + 1]
            lstm_score = detector.score_sequence(window)

            # Graph divergence
            graph_div = graph_builder.get_graph_divergence(user_id, dept, profiles)

            # Composite risk
            scored = ranker.score(feat, lstm_score, graph_div)

            # Peer deviation
            peer_dev = baseline_engine.peer_deviation_pct(user_id, dept, logs)

            row = {
                **event.to_dict(),
                "department": profile_row.get("department", ""),
                "job_title": profile_row.get("job_title", ""),
                "privilege_level": profile_row.get("privilege_level", "user"),
                "days_inactive": profile_row.get("days_inactive", 0),
                "hire_date": profile_row.get("hire_date", ""),
                "systems_access": profile_row.get("systems_access", ""),
                # Features
                **{f"feat_{k}": v for k, v in feat.items()},
                # Scores
                "lstm_score": lstm_score,
                "graph_divergence": graph_div,
                "risk_score": scored["risk_score"],
                "severity": scored["severity"],
                "triggered_signals": json.dumps(scored["triggered_signals"]),
                "mitre_techniques": json.dumps(scored["mitre_techniques"]),
                "behavioral_score": scored["component_scores"]["behavioral"],
                "lstm_component": scored["component_scores"]["lstm_sequence"],
                "graph_component": scored["component_scores"]["graph_divergence"],
                "peer_deviation_pct": peer_dev,
                # Ground truth (empty for baseline events)
                "is_anomaly": False,
                "anomaly_type": "",
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ── Inject demo personas ──────────────────────────────────────────────────────

def inject_demo_personas(
    features_df: pd.DataFrame,
    profiles: pd.DataFrame,
    baseline_engine: BaselineEngine,
    graph_builder: GraphBuilder,
    detector: AnomalyDetector,
    ranker: RiskRanker,
) -> pd.DataFrame:
    """Score and inject the 4 labelled demo anomaly events."""
    profile_map = profiles.set_index("user_id").to_dict("index")
    demo_rows = []

    for demo in DEMO_EVENTS:
        demo_copy = {**demo, "timestamp": pd.to_datetime(demo["timestamp"])}
        event = pd.Series(demo_copy)
        user_id = demo["user_id"]
        profile_row = profile_map.get(user_id, {})
        profile_series = pd.Series(profile_row)
        dept = profile_row.get("department", "")

        # All existing events for this user = prior history
        prior_events = features_df[features_df["user_id"] == user_id][
            ["timestamp", "action", "resource", "resource_sensitivity",
             "status", "source_ip", "time_classification", "user_id"]
        ].copy()
        prior_events["timestamp"] = pd.to_datetime(prior_events["timestamp"])

        feat = baseline_engine.score_event(event, prior_events, profile_series)
        lstm_score = detector.score_sequence([event])
        graph_div = graph_builder.get_graph_divergence(user_id, dept, profiles)
        scored = ranker.score(feat, lstm_score, graph_div)
        peer_dev = baseline_engine.peer_deviation_pct(user_id, dept, features_df)

        row = {
            **demo_copy,
            "department": dept,
            "job_title": profile_row.get("job_title", ""),
            "privilege_level": profile_row.get("privilege_level", "user"),
            "days_inactive": profile_row.get("days_inactive", 0),
            "hire_date": profile_row.get("hire_date", ""),
            "systems_access": profile_row.get("systems_access", ""),
            **{f"feat_{k}": v for k, v in feat.items()},
            "lstm_score": lstm_score,
            "graph_divergence": graph_div,
            "risk_score": scored["risk_score"],
            "severity": scored["severity"],
            "triggered_signals": json.dumps(scored["triggered_signals"]),
            "mitre_techniques": json.dumps(scored["mitre_techniques"]),
            "behavioral_score": scored["component_scores"]["behavioral"],
            "lstm_component": scored["component_scores"]["lstm_sequence"],
            "graph_component": scored["component_scores"]["graph_divergence"],
            "peer_deviation_pct": peer_dev,
        }
        demo_rows.append(row)

    demo_df = pd.DataFrame(demo_rows)
    return pd.concat([features_df, demo_df], ignore_index=True)


# ── Save demo replay ──────────────────────────────────────────────────────────

def save_demo_replay(features_df: pd.DataFrame) -> None:
    """Write the 4 demo events as JSONL for the dashboard demo replay."""
    OUT_DEMO.parent.mkdir(parents=True, exist_ok=True)
    demo_events = features_df[features_df["is_anomaly"] == True].copy()
    demo_events = demo_events.sort_values("risk_score", ascending=False)

    with open(OUT_DEMO, "w") as f:
        for _, row in demo_events.iterrows():
            record = {
                "event": {
                    k: str(row[k]) for k in
                    ["timestamp", "user_id", "username", "action", "resource",
                     "resource_sensitivity", "status", "source_ip", "time_classification"]
                },
                "profile": {
                    k: str(row.get(k, "")) for k in
                    ["department", "job_title", "privilege_level",
                     "days_inactive", "hire_date", "systems_access"]
                },
                "scored": {
                    "risk_score": int(row["risk_score"]),
                    "severity": str(row["severity"]),
                    "triggered_signals": json.loads(row["triggered_signals"]),
                    "mitre_techniques": json.loads(row["mitre_techniques"]),
                    "component_scores": {
                        "behavioral": float(row["behavioral_score"]),
                        "lstm_sequence": float(row["lstm_component"]),
                        "graph_divergence": float(row["graph_component"]),
                    },
                },
                "peer_deviation_pct": float(row["peer_deviation_pct"]),
                "anomaly_type": str(row.get("anomaly_type", "")),
                "persona": str(row.get("persona", "")),
                "mitre_label": str(row.get("mitre_label", "")),
            }
            f.write(json.dumps(record) + "\n")
    print(f"[demo] saved {len(demo_events)} demo events -> {OUT_DEMO}")


# ── Evaluation stub ───────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> None:
    """
    If a labels file is available, run precision/recall against it.
    Synthetic mode has labels from the generator. Real mode falls back
    to the 4 injected demo anomaly labels if official labels aren't present.
    """
    labels_path = LABELS_PATH
    threshold = int(os.getenv("RISK_THRESHOLD", "70"))

    if labels_path.exists():
        from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
        labels_df = pd.read_csv(labels_path)
        merged = features_df.merge(labels_df, on=["timestamp", "user_id"], how="inner")
        if "is_anomaly" in merged.columns:
            y_true = merged["is_anomaly_y"].astype(int)
            y_pred = (merged["risk_score"] >= threshold).astype(int)
            print("\n[eval] -- Official labels ---")
            print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"]))
    else:
        # Evaluate on demo events only
        demo_mask = features_df["is_anomaly"] == True
        if demo_mask.sum() == 0:
            print("[eval] no labelled events found - skipping evaluation")
            return
        y_true = features_df["is_anomaly"].astype(int)
        y_pred = (features_df["risk_score"] >= threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\n[eval] -- Demo label evaluation (threshold={threshold})")
        print(f"  Precision : {p:.2%}")
        print(f"  Recall    : {r:.2%}")
        print(f"  F1        : {f1:.3f}")
        print(f"  Alerts raised: {y_pred.sum()} / {len(y_pred)} events")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[mode] running in {'SYNTHETIC' if SYNTHETIC_MODE else 'REAL'} mode")
    print(f"[mode] outputs -> {OUT_DIR}")

    # Create output dirs
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load
    logs, profiles = load_data()

    # 2. Behavioral baselines
    print("\n[baseline] fitting per-user baselines...")
    baseline_engine = BaselineEngine()
    baseline_engine.fit(logs, profiles)
    baseline_df = baseline_engine.to_dataframe()
    baseline_df.to_parquet(OUT_BASELINES, index=False)
    print(f"[baseline] saved -> {OUT_BASELINES}")

    # 3. Graph
    print("\n[graph] building user-resource graphs...")
    graph_builder = GraphBuilder()
    graph_builder.build(logs, profiles)
    print(f"[graph] explicit: {graph_builder.explicit.number_of_nodes()} nodes, "
          f"{graph_builder.explicit.number_of_edges()} edges")
    print(f"[graph] implicit: {graph_builder.implicit.number_of_nodes()} nodes, "
          f"{graph_builder.implicit.number_of_edges()} edges")

    # 4. BiLSTM autoencoder
    print("\n[detector] training BiLSTM autoencoder...")
    detector = AnomalyDetector(seq_len=10, hidden_dim=32, latent_dim=16, epochs=60)
    detector.fit(logs)
    if detector.model is not None:
        detector.save(str(OUT_MODEL))
        print(f"[detector] saved -> {OUT_MODEL}")

    # 5. Score all events
    ranker = RiskRanker()
    print("\n[ranker] scoring all events...")
    features_df = compute_all_features(logs, profiles, baseline_engine, graph_builder, detector, ranker)
    print(f"[ranker] scored {len(features_df):,} events")

    # 6. Inject demo anomalies , only on real data (synthetic has its own labels)
    if not SYNTHETIC_MODE:
        print("\n[demo] injecting 4 labelled anomaly personas...")
        features_df = inject_demo_personas(
            features_df, profiles, baseline_engine, graph_builder, detector, ranker
        )
    else:
        # Synthetic: merge is_anomaly from the generator labels
        labels_df = pd.read_csv(LABELS_PATH)
        labels_df["timestamp"] = labels_df["timestamp"].astype(str)
        features_df["timestamp"] = features_df["timestamp"].astype(str)
        merged_labels = features_df.merge(
            labels_df[["timestamp", "user_id", "is_anomaly", "anomaly_type"]],
            on=["timestamp", "user_id"], how="left", suffixes=("", "_lab")
        )
        features_df["is_anomaly"]  = merged_labels["is_anomaly_lab"].fillna(False).astype(bool)
        features_df["anomaly_type"] = merged_labels["anomaly_type_lab"].fillna("")

    # 7. Save
    features_df.to_parquet(OUT_FEATURES, index=False)
    print(f"[save] features -> {OUT_FEATURES}")

    # 8. Demo replay JSONL , only for real data (scripted demo uses real personas)
    if not SYNTHETIC_MODE:
        save_demo_replay(features_df)

    # 9. Evaluation
    evaluate(features_df)

    # 10. Summary
    threshold = int(os.getenv("RISK_THRESHOLD", "70"))
    alerts = features_df[features_df["risk_score"] >= threshold]
    print(f"\n{'-'*50}")
    print(f"  Total events scored : {len(features_df):,}")
    print(f"  Alerts (risk>={threshold})  : {len(alerts)}")
    print(f"  CRITICAL (>=85)     : {(features_df['risk_score'] >= 85).sum()}")
    print(f"  HIGH (70-84)        : {((features_df['risk_score'] >= 70) & (features_df['risk_score'] < 85)).sum()}")
    print(f"  Demo anomalies      : {(features_df['is_anomaly'] == True).sum()}")
    print(f"{'-'*50}")
    next_cmd = ("streamlit run app/dashboard.py" if not SYNTHETIC_MODE
                else "python scripts/evaluate.py")
    print(f"  Done. Next: {next_cmd}")


if __name__ == "__main__":
    main()
