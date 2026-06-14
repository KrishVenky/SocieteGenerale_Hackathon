"""
Behavioral baseline engine.

Derives per-user rolling statistics from raw log history.
No pre-computed profiles. Everything comes from event data.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd


SENSITIVITY_SCORE = {"low": 1, "medium": 2, "high": 3}
TIME_RISK = {"business_hours": 0, "weekend": 1, "unusual_hours": 2, "night": 3}
ACTION_RISK = {
    "login": 0, "file_access": 1, "sql_query": 1,
    "api_call": 2, "admin_operation": 3, "export_data": 3,
}

# Which resource names map to which systems_access token in user_profiles
RESOURCE_TO_ACCESS_TOKEN = {
    "PROD_DB":       "PROD_DB",
    "SIEM":          "SIEM",
    "Admin_Console": "ADMIN_SYS",
}


class BaselineEngine:
    def __init__(self, window_days: int = 30, ip_window_days: int = 7,
                 history_fraction: float = 0.8):
        self.window_days = window_days
        self.ip_window_days = ip_window_days
        self.history_fraction = history_fraction
        self._baselines: dict = {}
        self._peer_stats: dict = {}  # department -> stats

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, logs: pd.DataFrame, profiles: pd.DataFrame) -> "BaselineEngine":
        """Build per-user and per-department baselines from historical log data."""
        logs = logs.copy()
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])
        logs = logs.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # Merge in privilege_level and systems_access for scope checks
        merged = logs.merge(
            profiles[["user_id", "department", "privilege_level",
                       "systems_access", "days_inactive"]],
            on="user_id", how="left"
        )

        for user_id, events in merged.groupby("user_id"):
            self._baselines[user_id] = self._compute_user_baseline(events)

        # Department-level peer group stats (merged already has department)
        for dept, group in merged.groupby("department"):
            self._peer_stats[dept] = self._compute_peer_stats(group)

        return self

    def _compute_user_baseline(self, events: pd.DataFrame) -> dict:
        events = events.sort_values("timestamp")
        cutoff = max(1, int(len(events) * self.history_fraction))
        hist = events.iloc[:cutoff]

        daily = hist.groupby(hist["timestamp"].dt.date).size()
        daily_mean = float(daily.mean()) if len(daily) > 0 else 1.0
        daily_std = float(daily.std()) if len(daily) > 1 else 0.5
        daily_std = max(daily_std, 0.1)

        hist_sens = hist["resource_sensitivity"].map(SENSITIVITY_SCORE).fillna(1)

        return {
            "typical_resources": set(hist["resource"].unique()),
            "typical_ips": set(hist["source_ip"].unique()),
            "typical_actions": hist["action"].value_counts(normalize=True).to_dict(),
            "typical_time": hist["time_classification"].value_counts(normalize=True).to_dict(),
            "daily_mean": daily_mean,
            "daily_std": daily_std,
            "max_sensitivity": int(hist_sens.max()) if len(hist_sens) > 0 else 1,
            "event_count": len(hist),
            "first_seen": hist["timestamp"].min(),
            "last_seen": hist["timestamp"].max(),
        }

    def _compute_peer_stats(self, events: pd.DataFrame) -> dict:
        daily = events.groupby([
            events["user_id"], events["timestamp"].dt.date
        ]).size()
        return {
            "daily_mean": float(daily.mean()) if len(daily) > 0 else 1.0,
            "daily_std": max(float(daily.std()) if len(daily) > 1 else 0.5, 0.1),
            "export_rate": float((events["action"] == "export_data").mean()),
            "high_sens_rate": float((events["resource_sensitivity"] == "high").mean()),
        }

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_event(self, event: pd.Series, prior_user_events: pd.DataFrame,
                    profile: pd.Series) -> dict:
        """
        Compute behavioral feature dict for a single event.

        Args:
            event: the event row being scored
            prior_user_events: all events for this user before this one
            profile: user_profiles row for this user

        Returns:
            feature dict consumed by RiskRanker
        """
        user_id = event["user_id"]
        baseline = self._baselines.get(user_id, self._empty_baseline())

        f = {}

        # ── Time ─────────────────────────────────────────────────────────────
        f["time_risk"] = TIME_RISK.get(event["time_classification"], 0)
        f["is_off_hours"] = f["time_risk"] >= 2  # unusual_hours or night

        # ── Sensitivity ───────────────────────────────────────────────────────
        f["sensitivity_score"] = SENSITIVITY_SCORE.get(event["resource_sensitivity"], 1)
        f["is_high_sens"] = f["sensitivity_score"] >= 3

        # ── Action ───────────────────────────────────────────────────────────
        f["action_risk"] = ACTION_RISK.get(event["action"], 1)
        f["is_export"] = event["action"] == "export_data"
        f["is_admin"] = event["action"] == "admin_operation"

        # ── Status ───────────────────────────────────────────────────────────
        f["is_failure"] = event["status"] == "failure"

        # ── Novel resource ────────────────────────────────────────────────────
        f["is_new_resource"] = event["resource"] not in baseline["typical_resources"]

        # ── Novel IP ─────────────────────────────────────────────────────────
        f["is_new_ip"] = event["source_ip"] not in baseline["typical_ips"]

        # ── IP entropy over ip_window_days ────────────────────────────────────
        f["ip_entropy"] = self._ip_entropy(event, prior_user_events)

        # ── Daily volume deviation (Z-score) ──────────────────────────────────
        today = pd.Timestamp(event["timestamp"]).date()
        window_start = pd.Timestamp(event["timestamp"]) - pd.Timedelta(days=self.window_days)
        window_events = prior_user_events[prior_user_events["timestamp"] >= window_start]
        today_count = int((prior_user_events["timestamp"].dt.date == today).sum()) + 1
        z = (today_count - baseline["daily_mean"]) / baseline["daily_std"]
        f["volume_zscore"] = float(z)

        # ── Failure burst (≥3 failures in past hour) ──────────────────────────
        one_hour_ago = pd.Timestamp(event["timestamp"]) - pd.Timedelta(hours=1)
        recent_failures = prior_user_events[
            (prior_user_events["timestamp"] >= one_hour_ago) &
            (prior_user_events["status"] == "failure")
        ]
        f["failure_burst"] = len(recent_failures) >= 3

        # ── Sensitivity escalation ────────────────────────────────────────────
        f["sensitivity_escalation"] = f["sensitivity_score"] > baseline["max_sensitivity"]

        # ── Dormant account ───────────────────────────────────────────────────
        # Two signals: profile days_inactive OR actual log gap > 30 days
        days_inactive = int(profile.get("days_inactive", 0)) if profile is not None else 0
        if len(prior_user_events) > 0:
            last_ts = prior_user_events["timestamp"].max()
            log_gap_days = (pd.Timestamp(event["timestamp"]) - last_ts).days
        else:
            log_gap_days = 999
        f["dormant_activation"] = (days_inactive > 30) or (log_gap_days > 30)

        # ── Resource scope violation ──────────────────────────────────────────
        f["resource_scope_violation"] = self._scope_violation(event, profile)

        # ── Service account night access ──────────────────────────────────────
        priv = str(profile.get("privilege_level", "user")) if profile is not None else "user"
        f["service_acct_anomaly"] = (
            priv == "service-account" and event["time_classification"] != "business_hours"
        )

        # ── Compound flags ────────────────────────────────────────────────────
        f["off_hours_export"] = f["is_off_hours"] and f["is_export"]
        f["high_sens_export"] = f["is_high_sens"] and f["is_export"]
        # admin_operation at off-hours (T1562.001, Defense Evasion)
        f["off_hours_admin"] = f["is_off_hours"] and f["is_admin"]
        # new resource + admin + off-hours combined (high confidence T1562.001)
        f["new_resource_off_hours_admin"] = f["is_new_resource"] and f["is_off_hours"] and f["is_admin"]
        # service account accessing a resource it has never touched before
        priv_for_compound = str(profile.get("privilege_level", "user")) if profile is not None else "user"
        f["service_acct_new_resource"] = priv_for_compound == "service-account" and f["is_new_resource"]

        return f

    def _scope_violation(self, event: pd.Series, profile: pd.Series) -> bool:
        """True if the resource accessed maps to a system not in the user's systems_access."""
        if profile is None:
            return False
        required_token = RESOURCE_TO_ACCESS_TOKEN.get(event["resource"])
        if required_token is None:
            return False
        approved = str(profile.get("systems_access", "")).split("|")
        return required_token not in approved

    def _ip_entropy(self, event: pd.Series, prior_events: pd.DataFrame) -> float:
        """Shannon entropy of IPs used in past ip_window_days."""
        window_start = pd.Timestamp(event["timestamp"]) - pd.Timedelta(days=self.ip_window_days)
        window = prior_events[prior_events["timestamp"] >= window_start]
        if len(window) == 0:
            return 0.0
        p = window["source_ip"].value_counts(normalize=True)
        return float(-sum(v * math.log2(v) for v in p if v > 0))

    # ── Peer group ────────────────────────────────────────────────────────────

    def get_peer_stats(self, department: str) -> dict:
        return self._peer_stats.get(department, {
            "daily_mean": 1.0, "daily_std": 0.5,
            "export_rate": 0.1, "high_sens_rate": 0.2,
        })

    def peer_deviation_pct(self, user_id: str, department: str,
                           all_logs: pd.DataFrame) -> float:
        """
        How many standard deviations above the department mean is this user's
        daily event volume? Returns as percentage deviation.
        """
        peer = self.get_peer_stats(department)
        baseline = self._baselines.get(user_id, self._empty_baseline())
        if peer["daily_std"] == 0:
            return 0.0
        z = (baseline["daily_mean"] - peer["daily_mean"]) / peer["daily_std"]
        return float(z * 100)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Export baselines as a DataFrame for storage."""
        rows = []
        for user_id, b in self._baselines.items():
            rows.append({
                "user_id": user_id,
                "daily_mean": b["daily_mean"],
                "daily_std": b["daily_std"],
                "max_sensitivity": b["max_sensitivity"],
                "event_count": b["event_count"],
                "typical_resource_count": len(b["typical_resources"]),
                "typical_ip_count": len(b["typical_ips"]),
                "first_seen": b["first_seen"],
                "last_seen": b["last_seen"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _empty_baseline() -> dict:
        return {
            "typical_resources": set(),
            "typical_ips": set(),
            "typical_actions": {},
            "typical_time": {},
            "daily_mean": 1.0,
            "daily_std": 0.5,
            "max_sensitivity": 1,
            "event_count": 0,
            "first_seen": None,
            "last_seen": None,
        }
