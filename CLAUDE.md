# CLAUDE.md — SG Hackathon: Insider Threat Detection System

## Project Context

**Event**: Societe Generale Hackathon
**Problem Statement chosen**: PS4 — Data Access Audit & Insider Threat Detection
**Goal**: Win the hackathon. Everyone is using AI agents. The edge is technical depth (graph ML + LLM narrative), real labeled data (CERT r5.2), financial-sector context injection, and a demo scripted to feel like a real SG deployment.

## Git Rules

- Remote: `https://github.com/KrishVenky/SocieteGenerale_Hackathon.git`
- **Never add Co-Authored-By lines to commits**
- Never commit files in `data/`, `models/checkpoints/`, `.env`, or any CSV/parquet
- Never push unless explicitly asked

---

## The Winning Strategy

### Why PS4 Wins

- "Data ingenuity over corpus size" — they want clever ML, not prompt engineering
- Most demo-friendly: "user pulled 14,847 records at 2am from a new IP" — judges instantly get it
- Quantifiable output: AUC, precision, recall on labeled CERT dataset
- Optional DLP prevention = bonus points nobody else will have

### What Everyone Else Will Build (and Why We Beat It)

| Approach | AUC on CERT r5.2 | Who builds it |
|---|---|---|
| Isolation Forest on raw features | ~85% | Most hackathon teams |
| DeepLog (LSTM on sequences) | 86.41% | Teams that read one paper |
| **Our system: GCN + Bi-LSTM + peer group** | **target 92-96%** | Us |
| SOTA (Dec 2025, arxiv 2512.18483) | 98.62% | Research lab with full infra |

The gap between 85% and 96% is the moat.

---

## Architecture

### Two-Stage Pipeline (Critical — Do Not Collapse Into One)

```
[Synthetic/CERT Event Stream]
         |
         v
[1. Behavioral Baseline Engine]     <- pandas, rolling stats, peer groups
         |
         v
[2. Graph Construction]             <- NetworkX: explicit (user→resource) +
         |                              implicit (shared-resource peer graph)
         v
[3. Anomaly Detection]              <- GCN embeddings + Bi-LSTM sequence model
         |                              OR simplified: node2vec + LSTM + Z-score
         v
[4. Risk Ranker]                    <- composite score: deviation × sensitivity × time
         |
         v
[5. MITRE ATT&CK Mapper]           <- map alert to specific technique
         |
         v
[6. Claude Fable 5 Narrative]      <- streaming incident report, confidence, response
         |
         v
[7. Streamlit Dashboard]           <- live incident queue, graph viz, narrative panel
```

**Rule**: The statistical/ML layers (1-4) do the heavy lifting. Claude only sees pre-filtered anomalies with structured context. This proves technical depth and conserves tokens.

### Model Roles

| Task | Model |
|---|---|
| Coding the system | Claude Sonnet 4.6 (via Claude Code) |
| Bulk event classification / tagging | `claude-haiku-4-5-20251001` |
| Final incident narrative + response steps | `claude-fable-5` |
| Architecture decisions during build | Claude Sonnet 4.6 |

### Token Conservation

- Haiku handles high-volume triage (thousands of events)
- Fable 5 handles only flagged anomalies (~20-50 per demo run)
- Estimated cost per incident narrative: ~$0.003
- Cache user baseline context using Anthropic prompt caching (5-min TTL)

---

## Tech Stack

```
Backend:     Python 3.11 + FastAPI
ML:          pandas, scikit-learn, PyTorch (Bi-LSTM), NetworkX (graphs)
Embeddings:  node2vec or simple degree/centrality features (fast to implement)
Frontend:    Streamlit (single file, real-time with st.empty())
AI:          anthropic Python SDK, streaming responses
Data:        CERT r5.2 base + financial context injection + SDV synthetic scale-up
```

---

## Dataset Strategy

### Source Datasets

1. **CERT r5.2** (primary, labeled)
   - Download: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
   - Size: 10.37 GB
   - Files: `logon.csv`, `email.csv`, `http.csv`, `file.csv`, `device.csv`
   - Ground truth: 3,995 benign users + 5 labeled malicious insiders
   - The standard benchmark — all paper AUCs use this, so your numbers are comparable

2. **CERT Paper GitHub** (preprocessed data, feature extraction code)
   - https://github.com/Yumlembam/Insider-Threat
   - Has `data/preprocessed/r5_data/` already done
   - Uses this exact dataset with GCN + Bi-LSTM (AUC 98.62 on r5.2)

3. **LANL Unified Host & Network** (scale validation only)
   - Download: https://csr.lanl.gov/data/2017/
   - 90 days, 1B+ events, real enterprise Windows auth logs
   - Use for "enterprise scale" architecture slide — don't need to run full model on it

4. **Feature extraction reference repos**
   - https://github.com/liujie40/feature-extraction-for-CERT-insider-threat-test-dataset
   - https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets

### Financial Context Injection

After loading CERT, run `scripts/inject_financial_context.py` to:
- Map generic resources → SG system names (Murex MX.3, Bloomberg Terminal, SWIFT gateway, Calypso)
- Add columns: `business_unit`, `data_sensitivity`, `system_name`, `transaction_volume`
- Add `counterparty_region` for cross-border data move detection

**Business units**: Equities, FICC, M&A Advisory, Compliance, Risk, IT Admin, Contractor

**Sensitivity tiers**:
- Public
- Internal
- Confidential (client portfolios, research reports)
- Restricted (trading positions, M&A deal data, SWIFT messages)

### Injected Anomaly Scenarios (Pre-Scripted for Demo)

| Persona | Pattern | MITRE | Risk Score |
|---|---|---|---|
| Alice Chen, M&A Analyst | 2am access, Paris IP (new), 3,200 client records from Murex, 3 days before resignation | T1078 | 96/100 |
| Bob Sharma, IT Admin | Disabled audit logging on SWIFT gateway at 11pm Friday, never done before | T1562 | 91/100 |
| Contractor acc SA-EXT-047 | USB copy of 14,847 Restricted files, access never seen in 18 months | T1052 | 94/100 |
| Service acct SA-MUREX-PROD | Suddenly querying HR database, first time ever | T1530 | 88/100 |

### Synthetic Scale-Up

Use SDV (Synthetic Data Vault) to scale from 4k users to 50k+:
```python
from sdv.tabular import GaussianCopula
model = GaussianCopula()
model.fit(enriched_cert_df)
synthetic_df = model.sample(50000)
```

---

## Key Research to Cite

| Paper | Result | Use in slides |
|---|---|---|
| GCN + Bi-LSTM (arxiv 2512.18483, Dec 2025) | AUC 98.62, 100% DR, 0.05% FPR on CERT r5.2 | Architecture reference |
| Federated Learning for Insider Threat (Scientific Reports 2025) | >90% accuracy, privacy loss <5% | Forward-looking architecture note |
| Mastercard GenAI + Graph (May 2024) | Doubled fraud detection, 3B cards | Real-world validation hook |
| DeepLog (baseline) | AUC 86.41 on CERT r5.2 | Baseline to beat |
| MITRE ATT&CK for Insider Threats — Securonix | Framework | Technique mapping |

**GitHub of primary paper**: https://github.com/Yumlembam/Insider-Threat

---

## Demo Script (5 Minutes, Pre-Scripted)

Do not do live inference on random data during the demo. Pre-script a replay:

```
T+0:00  Dashboard: 847 normal events streaming, all green
T+0:45  Alice Chen event surfaces: 2am, Paris IP, Murex Restricted, 3,200 records
T+1:00  System flags: risk 96/100, +2,840% peer deviation, MITRE T1078
T+1:15  Fable 5 narrative streams in LIVE (streaming API, types out character by character)
T+1:45  Alice's behavioral baseline graph vs today's spike (chart)
T+2:00  Kill chain: matches pre-exfiltration pattern (recon → access → bulk export)
T+2:30  Mock buttons: "Escalate to CISO" | "Freeze Account" | "Preserve Audit Trail"
T+3:00  Show second scenario (Bob Sharma, audit log disabled)
T+4:00  Numbers slide: AUC, FPR, time-to-alert, token cost per incident
T+4:30  Architecture + federated learning note (enterprise readiness)
```

---

## Numbers Slide

```
Benchmark: CERT Insider Threat Dataset r5.2 (Carnegie Mellon, industry standard)

Our system:        AUC [X]    Detection rate [Y]%    FPR [Z]%
DeepLog (2017):    AUC 86.41  DR 81.89%              FPR 0.19%
SOTA (Dec 2025):   AUC 98.62  DR 100%                FPR 0.05%

Time to alert:          < 3 seconds from event ingestion
Token cost/incident:    ~$0.003 (Haiku triage + Fable 5 narrative)
False positive reduction vs rule-based:  ~60-70% (Gartner UEBA benchmark)
```

---

## MITRE ATT&CK Mapping

| Detected pattern | Technique ID | Tactic |
|---|---|---|
| Bulk export after-hours | T1048 | Exfiltration |
| New geo / IP for valid user | T1078 | Initial Access |
| Access to new resource cluster | T1530 | Collection |
| Audit log disabled | T1562.001 | Defense Evasion |
| USB data transfer | T1052 | Exfiltration |
| Privilege escalation pattern | T1078.003 | Privilege Escalation |

---

## Peer Group Analysis (Key Differentiator)

Compare users not just to their own baseline but to their **role peer group**:
- "Is this M&A analyst accessing more Restricted data than other M&A analysts at their seniority?"
- Much lower false positive rate than self-comparison alone
- How enterprise UEBA tools (Splunk UBA, Microsoft Sentinel) actually work

Implicit graph in GCN captures this automatically: users who access the same resources are connected, so their embeddings are similar. Divergence from the cluster = anomaly.

---

## Federated Learning Note (Slides Only — Don't Implement)

Architecture is designed for federated deployment: each business unit trains locally, no raw logs leave the department. Reference: Scientific Reports 2025 federated insider threat paper. SG has strict data sovereignty requirements across jurisdictions — this is directly relevant.

---

## What NOT To Do

- Don't make the LLM do the anomaly detection — it's a narrative layer only
- Don't use only self-comparison baselines (use peer groups)
- Don't demo on live random data — pre-script the replay
- Don't commit any CSVs or model weights to git
- Don't use Isolation Forest as the sole detector (everyone will)
- Don't add Co-Authored-By to git commits
