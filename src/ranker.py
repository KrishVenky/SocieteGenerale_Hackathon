"""
Risk ranker: composes behavioral Z-scores, LSTM reconstruction error, and
graph divergence into a 0-100 risk score, then maps to MITRE ATT&CK.
"""

from __future__ import annotations

# ── MITRE rules ───────────────────────────────────────────────────────────────
# Each rule: condition(features) → bool. First match wins per technique.

MITRE_RULES = [
    {
        "technique_id": "T1048",
        "technique_name": "Exfiltration Over Alternative Protocol",
        "tactic": "Exfiltration",
        "condition": lambda f: f.get("off_hours_export") and f.get("is_high_sens"),
    },
    {
        "technique_id": "T1078",
        "technique_name": "Valid Accounts , New Location",
        "tactic": "Initial Access",
        "condition": lambda f: f.get("is_new_ip") and f.get("is_high_sens"),
    },
    {
        "technique_id": "T1078",
        "technique_name": "Valid Accounts , Dormant Account",
        "tactic": "Initial Access",
        "condition": lambda f: f.get("dormant_activation"),
    },
    {
        "technique_id": "T1530",
        "technique_name": "Data from Cloud Storage Object",
        "tactic": "Collection",
        "condition": lambda f: f.get("resource_scope_violation") and not f.get("is_admin"),
    },
    {
        "technique_id": "T1562.001",
        "technique_name": "Disable or Modify Tools",
        "tactic": "Defense Evasion",
        "condition": lambda f: f.get("is_admin") and f.get("is_off_hours"),
    },
    {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access",
        "condition": lambda f: f.get("failure_burst"),
    },
    {
        "technique_id": "T1078.004",
        "technique_name": "Cloud Accounts , Service Account Abuse",
        "tactic": "Privilege Escalation",
        "condition": lambda f: f.get("service_acct_anomaly"),
    },
    {
        "technique_id": "T1530",
        "technique_name": "Data from Cloud Storage Object , Cross-Dept",
        "tactic": "Collection",
        "condition": lambda f: f.get("resource_scope_violation") and f.get("service_acct_anomaly"),
    },
]

# ── Point weights for behavioral signals ──────────────────────────────────────
# These sum at most to 100 before LSTM/graph components are added.

SIGNAL_WEIGHTS: dict[str, int] = {
    "off_hours_export":             25,
    "high_sens_export":             20,
    "resource_scope_violation":     30,   # raised: scope violations are high confidence (T1530)
    "service_acct_anomaly":         25,   # service accounts off-hours = critical
    "new_resource_off_hours_admin": 30,   # T1562.001 compound , very high confidence
    "failure_burst":                40,   # raised: brute-force pattern is unambiguous (T1110)
    "dormant_activation":           15,
    "service_acct_new_resource":    25,   # service acct accessing new resource
    "is_new_ip":                    14,
    "off_hours_admin":              12,
    "sensitivity_escalation":       12,
    "is_new_resource":              10,
    "is_admin":                      5,
    "is_export":                     5,
}

# High-confidence anchor signals (weight >= 18).
# Real UEBA tools (Splunk UBA, Sentinel) require at least one anchor signal for
# a confident alert , this corroboration check dramatically reduces FP from
# minor-signal accumulation on normal users.
_ANCHOR_SIGNALS: frozenset[str] = frozenset(
    s for s, w in SIGNAL_WEIGHTS.items() if w >= 18
)

# ── Severity thresholds ───────────────────────────────────────────────────────

def _severity(score: int) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 70:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


# ── Ranker ────────────────────────────────────────────────────────────────────

class RiskRanker:
    def __init__(
        self,
        behavioral_weight: float = 0.55,
        lstm_weight: float = 0.25,
        graph_weight: float = 0.20,
    ):
        self.behavioral_weight = behavioral_weight
        self.lstm_weight = lstm_weight
        self.graph_weight = graph_weight

    def score(
        self,
        features: dict,
        lstm_score: float = 0.0,
        graph_divergence: float = 0.0,
    ) -> dict:
        """
        Compute a 0-100 composite risk score.

        Args:
            features: dict from BaselineEngine.score_event()
            lstm_score: float [0, 1] from AnomalyDetector.score_sequence()
            graph_divergence: float from GraphBuilder.get_graph_divergence()

        Returns:
            dict with risk_score, severity, triggered_signals, mitre_techniques,
                 component_scores
        """
        # ── Behavioral signal ─────────────────────────────────────────────────
        behavioral_raw = 0
        triggered = []

        for signal, weight in SIGNAL_WEIGHTS.items():
            if features.get(signal):
                behavioral_raw += weight
                triggered.append(signal)

        # Volume Z-score: require z > 3.0 to reduce FP from normal busy days
        z = features.get("volume_zscore", 0.0)
        if z > 3.0:
            vol_pts = min(12, int((z - 3.0) * 6))
            behavioral_raw += vol_pts
            triggered.append(f"volume_z{z:.1f}")

        behavioral_raw = min(100, behavioral_raw)

        # Corroboration adjustment , mirrors real UEBA (Splunk UBA / Sentinel) behaviour.
        # No high-confidence anchor signal: likely minor-signal noise; dampen to cut FP.
        # (No bonus for 2+ anchors: off_hours_export + high_sens_export are correlated and
        #  fire together on the same event, so they don't represent independent evidence.)
        n_anchor = sum(1 for s in triggered if s in _ANCHOR_SIGNALS)
        if n_anchor == 0:
            behavioral_raw = int(behavioral_raw * 0.60)

        # ── Composite ─────────────────────────────────────────────────────────
        # graph_divergence is roughly in [0, 5]; scale to [0, 100]
        graph_pts = min(100.0, graph_divergence * 20.0)

        composite = (
            self.behavioral_weight * behavioral_raw
            + self.lstm_weight * lstm_score * 100.0
            + self.graph_weight * graph_pts
        )
        risk_score = int(min(100, max(0, round(composite))))

        # ── MITRE mapping ─────────────────────────────────────────────────────
        seen_ids: set[str] = set()
        mitre_hits = []
        for rule in MITRE_RULES:
            try:
                if rule["condition"](features):
                    tid = rule["technique_id"]
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        mitre_hits.append({
                            "technique_id": tid,
                            "technique_name": rule["technique_name"],
                            "tactic": rule["tactic"],
                        })
            except Exception:
                pass

        return {
            "risk_score": risk_score,
            "severity": _severity(risk_score),
            "triggered_signals": triggered,
            "mitre_techniques": mitre_hits,
            "component_scores": {
                "behavioral": behavioral_raw,
                "lstm_sequence": round(lstm_score * 100, 1),
                "graph_divergence": round(graph_pts, 1),
            },
        }

    def score_batch(
        self,
        feature_rows: list[dict],
        lstm_scores: list[float] | None = None,
        graph_scores: list[float] | None = None,
    ) -> list[dict]:
        """Score a list of events. lstm_scores and graph_scores are optional."""
        n = len(feature_rows)
        lstm_scores = lstm_scores or [0.0] * n
        graph_scores = graph_scores or [0.0] * n
        return [
            self.score(f, l, g)
            for f, l, g in zip(feature_rows, lstm_scores, graph_scores)
        ]
