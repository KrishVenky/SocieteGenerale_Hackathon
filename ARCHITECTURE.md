# Architecture

## System Overview

Two-stage pipeline: statistical/ML detection first, LLM narrative second. The LLM never touches raw logs — it receives structured, pre-filtered anomaly context only.

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVENT INGESTION LAYER                        │
│  logon.csv | email.csv | file.csv | http.csv | device.csv       │
│  + injected financial context (Murex, Bloomberg, SWIFT, etc.)   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                   BEHAVIORAL BASELINE ENGINE                     │
│                                                                  │
│  Per-user rolling stats (30-day window):                        │
│    • hourly access counts (time-of-day distribution)            │
│    • daily data volume (records accessed)                        │
│    • resource cluster fingerprint (which systems touched)       │
│    • geo/IP profile                                              │
│                                                                  │
│  Peer group baselines (by role + business unit):               │
│    • M&A Analyst peer group norms                               │
│    • Trader peer group norms                                     │
│    • IT Admin peer group norms                                  │
│    • Contractor peer group norms                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                     GRAPH CONSTRUCTION                           │
│                                                                  │
│  Explicit graph (user → resource):                              │
│    Nodes: users + resources                                      │
│    Edges: access events, weighted by frequency                  │
│                                                                  │
│  Implicit graph (user → user via shared resources):            │
│    Nodes: users                                                  │
│    Edges: users who access same resources are connected         │
│    → This IS the peer group, captured structurally              │
│                                                                  │
│  Both graphs → separate GCNs → node embeddings                 │
│  Embeddings concatenated + attention mechanism                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                   ANOMALY DETECTION                              │
│                                                                  │
│  Bi-LSTM on user event sequences:                               │
│    Input: (resource_id, action, hour, sensitivity, volume)      │
│    Predict next event probability → low prob = anomaly          │
│    Temporal window: 24h rolling sessions                        │
│                                                                  │
│  Combined signal:                                                │
│    anomaly_score = α × sequence_perplexity                      │
│                  + β × graph_embedding_distance                 │
│                  + γ × peer_group_percentile_deviation          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                      RISK RANKER                                 │
│                                                                  │
│  composite_risk = anomaly_score                                 │
│                 × sensitivity_multiplier  (1.0x–3.0x)          │
│                 × time_multiplier         (1.5x if off-hours)   │
│                 × geo_multiplier          (2.0x if new IP/geo)  │
│                 × velocity_multiplier     (burst access)        │
│                                                                  │
│  Confidence: isotonic regression calibration on CERT labels     │
│  Output: risk_score (0-100), confidence (0.0-1.0)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                  MITRE ATT&CK MAPPER                            │
│                                                                  │
│  Rules engine maps risk patterns to specific techniques:        │
│    bulk export off-hours  → T1048 (Exfiltration)               │
│    new geo + valid creds  → T1078 (Valid Accounts)             │
│    audit log disabled     → T1562.001 (Defense Evasion)        │
│    USB bulk copy          → T1052 (Exfil via Physical)         │
│    cross-domain access    → T1530 (Cloud Storage Collection)   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│              CLAUDE FABLE 5 NARRATIVE GENERATOR                 │
│                                                                  │
│  Input (structured, pre-filtered):                              │
│    {user, role, bu, risk_score, confidence,                     │
│     deviation_from_peer, mitre_technique,                       │
│     top_5_anomalous_events, baseline_summary}                   │
│                                                                  │
│  Output (streaming to dashboard):                               │
│    • Plain-English incident description                         │
│    • Why it's suspicious (evidence list)                        │
│    • Kill chain stage assessment                                │
│    • Confidence explanation                                     │
│    • Recommended response steps (tiered by severity)           │
│                                                                  │
│  Claude Haiku 4.5: bulk event triage ($0.0001/event)           │
│  Claude Fable 5: per-incident narrative ($0.002-0.005/incident)│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                            │
│                                                                  │
│  Left panel:  Live incident queue (sorted by risk score)        │
│  Center:      NetworkX graph viz (user-resource graph,          │
│               anomalous nodes highlighted red)                  │
│  Right panel: Streaming Fable 5 narrative (types out live)     │
│  Bottom:      Behavioral baseline chart vs today's spike        │
│               MITRE ATT&CK technique tag                        │
│  Buttons:     Escalate to CISO | Freeze Account | Audit Trail  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Contracts

### Event ingestion schema
```python
{
  "event_id": str,
  "timestamp": datetime,
  "user_id": str,
  "user_role": str,           # injected: Trader | Analyst | IT Admin | Contractor
  "business_unit": str,       # injected: Equities | FICC | M&A | Compliance | Risk
  "resource_id": str,
  "system_name": str,         # injected: Murex | Bloomberg | SWIFT | Calypso
  "action": str,              # logon | file_read | file_copy | email_send | usb_connect
  "data_sensitivity": str,    # injected: Public | Internal | Confidential | Restricted
  "volume_records": int,
  "source_ip": str,
  "geo_region": str
}
```

### Anomaly output schema (input to Fable 5)
```python
{
  "user_id": str,
  "risk_score": float,        # 0-100
  "confidence": float,        # 0.0-1.0
  "mitre_technique": str,     # e.g. "T1078"
  "mitre_tactic": str,        # e.g. "Initial Access"
  "peer_deviation_pct": float,# e.g. 2840.0 (2840% above peer group)
  "self_deviation_pct": float,
  "anomalous_events": list,   # top 5 most anomalous events
  "baseline_summary": dict,   # user's normal behavior stats
  "kill_chain_stage": str     # Reconnaissance | Access | Collection | Exfiltration
}
```

## Scaling Notes

- Demo runs on CERT r5.2 sample (10GB, 4k users)
- Architecture designed for Kafka stream ingestion at enterprise scale
- Federated deployment: each business unit trains local GCN, only model weights shared (not raw logs)
- Baseline recalibration: scheduled nightly, handles role changes and seasonality
