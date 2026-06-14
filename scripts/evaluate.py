"""
evaluate.py , Precision / Recall / F1 reporter

Loads scored features from BOTH real and synthetic processed directories
and prints a side-by-side comparison table.

Usage:
    python scripts/evaluate.py              # compare real vs synthetic
    python scripts/evaluate.py --real-only
    python scripts/evaluate.py --syn-only
    python scripts/evaluate.py --threshold 65
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

REAL_FEATURES = ROOT / "data/processed/real/features.parquet"
SYN_FEATURES  = ROOT / "data/processed/synthetic/features.parquet"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate precision/recall/F1")
    p.add_argument("--real-only",  action="store_true")
    p.add_argument("--syn-only",   action="store_true")
    p.add_argument("--threshold",  type=int, default=70,
                   help="Risk score threshold for positive prediction (default 70)")
    p.add_argument("--thresholds", type=str, default=None,
                   help="Comma-separated list of thresholds to sweep (e.g. 50,60,70,80,85)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] {label} features not found at {path}")
        print(f"       Run: python scripts/prepare_data.py"
              + (" --synthetic" if "synthetic" in str(path) else ""))
        return None
    df = pd.read_parquet(path)
    print(f"[load] {label}: {len(df):,} events  |  "
          f"labelled anomalies: {df['is_anomaly'].sum():,}")
    return df


def _metrics_at_threshold(df: pd.DataFrame, threshold: int) -> dict:
    if "is_anomaly" not in df.columns or df["is_anomaly"].sum() == 0:
        return {"error": "no labelled anomalies"}
    y_true = df["is_anomaly"].astype(int)
    y_pred = (df["risk_score"] >= threshold).astype(int)
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": p, "recall": r, "f1": f1,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "alerts":  int(y_pred.sum()),
        "positives": int(y_true.sum()),
        "total": len(df),
    }


def _print_single(df: pd.DataFrame, label: str, threshold: int) -> dict:
    m = _metrics_at_threshold(df, threshold)
    if "error" in m:
        print(f"\n  [{label}] {m['error']}")
        return m
    print(f"\n  [{label}]  threshold={threshold}")
    print(f"    Precision : {m['precision']:.2%}")
    print(f"    Recall    : {m['recall']:.2%}")
    print(f"    F1        : {m['f1']:.3f}")
    print(f"    TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"    Alerts raised  : {m['alerts']:,} / {m['total']:,} events")
    print(f"    True anomalies : {m['positives']:,}")
    return m


def _print_side_by_side(real_df: pd.DataFrame | None,
                         syn_df:  pd.DataFrame | None,
                         threshold: int) -> None:
    real_m = _metrics_at_threshold(real_df, threshold) if real_df is not None else {}
    syn_m  = _metrics_at_threshold(syn_df,  threshold) if syn_df  is not None else {}

    ROWS = [
        ("Precision",   lambda m: f"{m.get('precision', 0):.2%}"),
        ("Recall",      lambda m: f"{m.get('recall', 0):.2%}"),
        ("F1",          lambda m: f"{m.get('f1', 0):.3f}"),
        ("TP",          lambda m: str(m.get("tp", "-"))),
        ("FP",          lambda m: str(m.get("fp", "-"))),
        ("FN",          lambda m: str(m.get("fn", "-"))),
        ("Alerts",      lambda m: f"{m.get('alerts', 0):,} / {m.get('total', 0):,}"),
        ("True anom.",  lambda m: str(m.get("positives", "-"))),
    ]

    W = 18
    header = f"  {'Metric':<14} {'Real':>{W}} {'Synthetic':>{W}}"
    sep    = f"  {'-'*14} {'-'*W} {'-'*W}"
    print(f"\n  {'-'*50}")
    print(f"  Side-by-side comparison  (threshold={threshold})")
    print(sep)
    print(header)
    print(sep)
    for name, fn in ROWS:
        rv = fn(real_m) if real_m and "error" not in real_m else "-"
        sv = fn(syn_m)  if syn_m  and "error" not in syn_m  else "-"
        print(f"  {name:<14} {rv:>{W}} {sv:>{W}}")
    print(sep)

    # Pass/fail vs PS4 criteria
    PS4_P, PS4_R, PS4_F1 = 0.75, 0.70, 0.72
    for label, m in [("Real", real_m), ("Synthetic", syn_m)]:
        if not m or "error" in m:
            continue
        p_ok  = "PASS" if m["precision"] >= PS4_P  else "FAIL"
        r_ok  = "PASS" if m["recall"]    >= PS4_R  else "FAIL"
        f1_ok = "PASS" if m["f1"]        >= PS4_F1 else "FAIL"
        print(f"\n  PS4 criteria check [{label}]:")
        print(f"    Precision >={PS4_P:.0%}  : {p_ok}  ({m['precision']:.2%})")
        print(f"    Recall    >={PS4_R:.0%}  : {r_ok}  ({m['recall']:.2%})")
        print(f"    F1        >={PS4_F1}  : {f1_ok}  ({m['f1']:.3f})")


def _threshold_sweep(df: pd.DataFrame, label: str, thresholds: list[int]) -> None:
    print(f"\n  Event-level threshold sweep [{label}]")
    print(f"  {'Thresh':>7} {'Precision':>10} {'Recall':>8} {'F1':>7} {'Alerts':>8}")
    print(f"  {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
    for t in thresholds:
        m = _metrics_at_threshold(df, t)
        if "error" in m:
            break
        print(f"  {t:>7}  {m['precision']:>9.2%}  {m['recall']:>7.2%}  {m['f1']:>6.3f}  {m['alerts']:>7,}")


def _user_level_sweep(df: pd.DataFrame, label: str, thresholds: list[int]) -> None:
    """User-level threshold sweep , most meaningful for UEBA evaluation."""
    if "user_id" not in df.columns or "is_anomaly" not in df.columns:
        return
    user_label = df.groupby("user_id")["is_anomaly"].any().astype(int)
    user_max   = df.groupby("user_id")["risk_score"].max()

    PS4_P, PS4_R, PS4_F1 = 0.75, 0.70, 0.72
    print(f"\n  User-level threshold sweep [{label}]")
    print(f"  {'Thresh':>7} {'Precision':>10} {'Recall':>8} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'PS4':>6}")
    print(f"  {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
    for t in thresholds:
        user_pred = (user_max >= t).astype(int)
        p  = precision_score(user_label, user_pred, zero_division=0)
        r  = recall_score(user_label, user_pred, zero_division=0)
        f1 = f1_score(user_label, user_pred, zero_division=0)
        tp = int((user_pred * user_label).sum())
        fp = int(user_pred.sum() - tp)
        fn = int(user_label.sum() - tp)
        ps4 = "PASS" if p >= PS4_P and r >= PS4_R and f1 >= PS4_F1 else "----"
        print(f"  {t:>7}  {p:>9.2%}  {r:>7.2%}  {f1:>6.3f}  {tp:>5} {fp:>5} {fn:>5} {ps4:>6}")


def _anomaly_breakdown(df: pd.DataFrame, label: str, threshold: int) -> None:
    if "anomaly_type" not in df.columns:
        return
    anomalies = df[df["is_anomaly"] == True].copy()
    if anomalies.empty:
        return
    anomalies["detected"] = anomalies["risk_score"] >= threshold
    print(f"\n  Anomaly type breakdown [{label}] (threshold={threshold}):")
    grp = anomalies.groupby("anomaly_type")["detected"].agg(["sum", "count"])
    grp.columns = ["detected", "total"]
    grp["recall"] = grp["detected"] / grp["total"]
    for atype, row in grp.iterrows():
        bar = "#" * int(row["recall"] * 20) + "." * (20 - int(row["recall"] * 20))
        print(f"    {atype:<30} {int(row['detected'])}/{int(row['total'])}  {bar}  {row['recall']:.0%}")


def _user_level_metrics(df: pd.DataFrame, label: str, threshold: int) -> None:
    """User-level evaluation: flag user if any event >= threshold."""
    if "user_id" not in df.columns or "is_anomaly" not in df.columns:
        return
    user_label = df.groupby("user_id")["is_anomaly"].any().astype(int)
    user_pred  = (df.groupby("user_id")["risk_score"].max() >= threshold).astype(int)
    p  = precision_score(user_label, user_pred, zero_division=0)
    r  = recall_score(user_label, user_pred, zero_division=0)
    f1 = f1_score(user_label, user_pred, zero_division=0)
    tp = int((user_pred * user_label).sum())
    fp = int(user_pred.sum() - tp)
    fn = int(user_label.sum() - tp)

    PS4_P, PS4_R, PS4_F1 = 0.75, 0.70, 0.72
    p_ok  = "PASS" if p  >= PS4_P  else "FAIL"
    r_ok  = "PASS" if r  >= PS4_R  else "FAIL"
    f1_ok = "PASS" if f1 >= PS4_F1 else "FAIL"

    print(f"\n  User-level metrics [{label}] (threshold={threshold}):")
    print(f"    Users total:  {len(user_label)}")
    print(f"    Anomaly users:{int(user_label.sum())}  |  Flagged: {int(user_pred.sum())}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")
    print(f"    Precision : {p:.2%}   PS4 check: {p_ok}")
    print(f"    Recall    : {r:.2%}   PS4 check: {r_ok}")
    print(f"    F1        : {f1:.3f}  PS4 check: {f1_ok}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    threshold = args.threshold

    print(f"\n{'='*52}")
    print(f"  PS4 Insider Threat - Evaluation Report")
    print(f"  Threshold: {threshold}  |  PS4 criteria: P>=75%, R>=70%, F1>=0.72")
    print(f"{'='*52}")

    real_df = None
    syn_df  = None

    if not args.syn_only:
        real_df = _load(REAL_FEATURES, "real")
    if not args.real_only:
        syn_df = _load(SYN_FEATURES, "synthetic")

    if real_df is None and syn_df is None:
        print("\n  Nothing to evaluate - run prepare_data.py first.")
        sys.exit(1)

    # Single-mode or side-by-side
    if args.real_only and real_df is not None:
        _print_single(real_df, "REAL", threshold)
        _anomaly_breakdown(real_df, "real", threshold)
        _user_level_metrics(real_df, "real", threshold)
    elif args.syn_only and syn_df is not None:
        _print_single(syn_df, "SYNTHETIC", threshold)
        _anomaly_breakdown(syn_df, "synthetic", threshold)
        _user_level_metrics(syn_df, "synthetic", threshold)
    else:
        _print_side_by_side(real_df, syn_df, threshold)
        if real_df is not None:
            _anomaly_breakdown(real_df, "real", threshold)
        if syn_df is not None:
            _anomaly_breakdown(syn_df, "synthetic", threshold)
        # User-level is most meaningful for synthetic (many labeled users)
        if syn_df is not None:
            _user_level_metrics(syn_df, "synthetic", threshold)

    # Threshold sweeps if requested (both event-level and user-level)
    if args.thresholds:
        sweep = [int(t.strip()) for t in args.thresholds.split(",")]
        if real_df is not None and not args.syn_only:
            _threshold_sweep(real_df, "real", sweep)
        if syn_df is not None and not args.real_only:
            _threshold_sweep(syn_df, "synthetic", sweep)
            _user_level_sweep(syn_df, "synthetic", sweep)

    # Full sklearn report on real data (compact, useful for slides)
    if real_df is not None and "is_anomaly" in real_df.columns and real_df["is_anomaly"].sum() > 0:
        y_true = real_df["is_anomaly"].astype(int)
        y_pred = (real_df["risk_score"] >= threshold).astype(int)
        print(f"\n  Sklearn classification report [real, threshold={threshold}]:")
        print(classification_report(y_true, y_pred,
                                    target_names=["normal", "anomaly"],
                                    digits=3))

    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
