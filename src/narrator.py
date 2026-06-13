"""
Narrative engine: generates streaming incident reports via Groq LLaMA 3.3.

Groq is used for the narrative layer only — all anomaly detection is done
by the ML pipeline (baseline + BiLSTM + graph). The LLM just writes the
human-readable incident report from the structured alert dict.
"""

from __future__ import annotations

import os
from typing import Generator

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Context maps ──────────────────────────────────────────────────────────────

SG_RESOURCE_MAP = {
    "Customer_Vault":  ("Client Portfolio System (Calypso CRM)", "Restricted"),
    "GL_System":       ("General Ledger (SAP FI/CO)",            "Restricted"),
    "HRIS":            ("HR Information System (Workday)",        "Confidential"),
    "PROD_DB":         ("Trading Production DB (Murex MX.3)",    "Restricted"),
    "Admin_Console":   ("Infrastructure Admin Console",           "Confidential"),
    "SIEM":            ("Security Platform (Splunk)",             "Confidential"),
    "Data_Lake":       ("Analytics Data Lake (Databricks)",       "Internal"),
    "BI_Tool":         ("Business Intelligence (Power BI)",       "Internal"),
    "File_Share":      ("SharePoint Document Library",            "Internal"),
    "Email_Archive":   ("Email Archive (Microsoft 365)",          "Internal"),
}

SG_DEPT_MAP = {
    "Finance": "Finance & Risk", "Engineering": "IT Infrastructure",
    "Security": "Information Security", "Legal": "Legal & Compliance",
    "HR": "Human Resources", "Sales": "Client Coverage",
    "Marketing": "Investment Research", "Compliance": "Regulatory Compliance",
    "Operations": "Operations & Settlement", "Executive": "Senior Management",
    "IT": "Technology", "Support": "IT Support",
}

_SYSTEM_PROMPT = """\
You are SentinelAI, Societe Generale's insider threat detection system.
Write a concise, professional incident report for a Tier-2 security analyst.

Format:
1. Threat summary (2-3 sentences) — who, what, when, why it's suspicious
2. Key evidence (3-4 bullet points, each starting with •)
3. Recommended Actions: (3 bullets, starting with ▶)

Use the exact names, timestamps, and statistics provided. Do not hedge or add disclaimers.
Write as if this is a live, confirmed alert requiring immediate analyst review.\
"""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(alert: dict) -> str:
    event   = alert.get("event", {})
    profile = alert.get("profile", {})
    scored  = alert.get("scored", {})
    signals = scored.get("triggered_signals", [])
    mitre   = scored.get("mitre_techniques", [])
    peer_dev = alert.get("peer_deviation_pct", 0.0)

    resource = str(event.get("resource", "Unknown"))
    sg_name, sg_class = SG_RESOURCE_MAP.get(resource, (resource, "Unknown"))
    sg_dept = SG_DEPT_MAP.get(str(profile.get("department", "")), str(profile.get("department", "")))

    mitre_str = (
        ", ".join(f"{m['technique_id']} ({m['tactic']})" for m in mitre)
        if mitre else "None"
    )
    signals_str = "\n".join(f"  - {s}" for s in signals) if signals else "  - None"
    new_ip_note = (
        "NEW IP — not in 90-day baseline history"
        if "is_new_ip" in signals else "Known IP"
    )

    return f"""\
INCIDENT ALERT — Risk Score: {scored.get('risk_score', 0)}/100  Severity: {scored.get('severity', 'UNKNOWN')}

User:        {profile.get('username', event.get('username', 'Unknown'))}
Role:        {profile.get('job_title', 'Unknown')} | Department: {sg_dept}
Privilege:   {profile.get('privilege_level', 'Unknown')}
Dormancy:    {profile.get('days_inactive', 0)} days inactive before this event
Tenure:      hired {profile.get('hire_date', 'Unknown')}

Event:
  Timestamp:    {event.get('timestamp', 'Unknown')}
  Action:       {event.get('action', 'Unknown')}
  System:       {sg_name}  [{sg_class}]
  Raw resource: {resource}
  Status:       {event.get('status', 'Unknown')}
  Source IP:    {event.get('source_ip', 'Unknown')}  ({new_ip_note})
  Time window:  {event.get('time_classification', 'Unknown')}

Detection signals:
{signals_str}

Peer group deviation: {peer_dev:+.0f}% vs {sg_dept} department baseline
MITRE ATT&CK:         {mitre_str}

Component risk scores:
  Behavioral baseline:   {scored.get('component_scores', {}).get('behavioral', 0)}/100
  LSTM sequence anomaly: {scored.get('component_scores', {}).get('lstm_sequence', 0)}/100
  Peer graph divergence: {scored.get('component_scores', {}).get('graph_divergence', 0)}/100\
"""


# ── Engine ────────────────────────────────────────────────────────────────────

class NarratorEngine:
    def __init__(self):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._threshold = int(os.getenv("RISK_THRESHOLD", "70"))

    def triage_alerts(self, alerts: list[dict]) -> list[dict]:
        """Mark each alert as escalate=True/False based on numeric threshold."""
        for alert in alerts:
            score = alert.get("scored", {}).get("risk_score", 0)
            alert["escalate"] = score >= self._threshold
        return alerts

    def stream(self, alert: dict) -> Generator[str, None, None]:
        """Stream the incident narrative token by token."""
        context = build_context(alert)
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
            max_tokens=450,
            stream=True,
        )
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate(self, alert: dict) -> str:
        """Non-streaming version — returns full narrative string."""
        return "".join(self.stream(alert))

    def generate_batch(self, alerts: list[dict]) -> list[str]:
        return [self.generate(a) for a in alerts if a.get("escalate")]
