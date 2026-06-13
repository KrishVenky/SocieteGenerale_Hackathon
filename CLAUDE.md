# CLAUDE.md — SG Hackathon: Insider Threat Detection System

## Project Context

**Event**: Societe Generale Hackathon
**Problem Statement chosen**: PS4 — Data Access Audit & Insider Threat Detection
**Goal**: Win the hackathon. Everyone is using AI agents. The edge is technical depth (graph ML + LLM narrative), behavioral baseline derivation from raw logs, and a demo scripted to feel like a real SG deployment.

## Git Rules

- Remote: `https://github.com/KrishVenky/SocieteGenerale_Hackathon.git`
- **Never add Co-Authored-By lines to commits**
- Never commit files in `data/`, `models/checkpoints/`, `.env`, or any CSV/parquet
- Never push unless explicitly asked

---

## The Winning Strategy

### Why PS4 Wins

- "Data ingenuity over corpus size" — they want clever ML, not prompt engineering
- Most demo-friendly: "user pulled customer vault data at 2am from a new IP" — judges instantly get it
- Quantifiable output: precision, recall, F1 on the provided labeled dataset
- Optional DLP prevention = bonus points nobody else will have

### What Everyone Else Will Build (and Why We Beat It)

| Approach | F1 on PS4 data | Who builds it |
|---|---|---|
| Rule engine (flag all night access) | ~0.40 | Beginner teams |
| Isolation Forest on raw features | ~0.65 | Most teams |
| Z-score per user with static profiles | ~0.70 | Intermediate teams |
| **Our system: GCN + Bi-LSTM + peer group** | **target 0.85+** | Us |

PS4 target: F1 > 0.72. We aim for 0.85+ by combining graph embeddings with sequence modeling and peer group context.

---

## Architecture

### Two-Stage Pipeline (Critical — Do Not Collapse Into One)

```
[PS4 Event Stream — data_access_logs.csv]
         |
         v
[1. Behavioral Baseline Engine]     <- pandas, rolling stats, peer groups by department
         |                              Derived from log history, NOT pre-computed profiles
         v
[2. Graph Construction]             <- NetworkX: explicit (user→resource) +
         |                              implicit (shared-resource peer graph)
         v
[3. Anomaly Detection]              <- GCN embeddings + Bi-LSTM sequence model
         |                              node2vec for graph embeddings; Z-score for fast baseline
         v
[4. Risk Ranker]                    <- composite: deviation × sensitivity × time_class × cross_dept
         |
         v
[5. MITRE ATT&CK Mapper]           <- map alert to specific technique
         |
         v
[6. Claude Fable 5 Narrative]      <- streaming incident report, confidence, response steps
         |
         v
[7. Streamlit Dashboard]           <- live incident queue, graph viz, narrative panel
```

**Rule**: The statistical/ML layers (1-4) do the heavy lifting. Claude only sees pre-filtered anomalies with structured context. This proves technical depth and conserves tokens.

### Model Roles

| Task | Model |
|---|---|
| Coding the system | Claude Sonnet 4.6 (via Claude Code) |
| Incident narrative + response steps | Groq `llama-3.3-70b-versatile` |
| Architecture decisions during build | Claude Sonnet 4.6 |

### Why Groq

- Free tier, no credit card required for the API key
- llama-3.3-70b on Groq is fast (~200 tok/s) — streaming looks live in the dashboard
- Narrative generation is the only LLM call in the app; all detection is ML-only
- Estimated cost per narrative: ~$0.00 on free tier

---

## Tech Stack

```
Backend:     Python 3.11 + FastAPI
ML:          pandas, scikit-learn, PyTorch (Bi-LSTM), NetworkX (graphs)
Embeddings:  node2vec or simple degree/centrality features (fast to implement)
Frontend:    Streamlit (single file, real-time with st.empty())
AI:          anthropic Python SDK, streaming responses
Data:        PS4 official data + financial context injection + SDV synthetic scale-up
```

---

## Dataset Strategy

### Primary: PS4 Official Sample Data

**Location**: `data/Problem_04_Data_Access/sample_data/`

**Files provided**:
- `data_access_logs.csv` — 1,201 events, Apr 2025 – Apr 2026 (full year)
- `user_profiles.csv` — 100 users

**`data_access_logs.csv` — actual columns**:

| Column | Values | Notes |
|---|---|---|
| `timestamp` | datetime | Full year coverage |
| `user_id` | USR00000–USR00099 | 100 unique users |
| `username` | firstname.lastname | |
| `action` | `export_data`, `sql_query`, `api_call`, `admin_operation`, `login`, `file_access` | |
| `resource` | `Customer_Vault`, `GL_System`, `HRIS`, `PROD_DB`, `Admin_Console`, `SIEM`, `Data_Lake`, `BI_Tool`, `File_Share`, `Email_Archive` | |
| `resource_sensitivity` | `low`, `medium`, `high` | No `restricted` tier in raw data |
| `status` | `success`, `failure` | |
| `source_ip` | 192.168.x.x | Internal IPs; detect IP entropy per user |
| `time_classification` | `business_hours`, `night`, `unusual_hours`, `weekend` | Pre-computed |

**`user_profiles.csv` — actual columns**:

| Column | Values | Notes |
|---|---|---|
| `user_id` | USR00000–USR00099 | |
| `username` | firstname.lastname | |
| `email` | @company.com | |
| `department` | Finance, Security, IT, Marketing, Engineering, Compliance, Legal, Sales, HR, Operations, Support, Executive | Defines peer groups |
| `job_title` | Developer, Director, Architect, Engineer, etc. | |
| `privilege_level` | `user`, `power-user`, `admin`, `service-account` | Key risk signal |
| `systems_access` | Pipe-separated: `AD|Azure_AD|SIEM|EMAIL|PROD_DB|GCP|...` | Cross-reference vs actual resource hit |
| `last_login` | date | |
| `days_inactive` | integer | >30 days then active = high risk |
| `is_active` | true/false | |
| `hire_date` | date | Tenure = shorter tenure → higher weight on anomalies |

**Important**: The profiles do NOT include pre-computed `typical_access_hours`, `avg_queries_per_day`, or `avg_rowcount_per_query`. We derive these from log history — which is the harder, more impressive ML problem.

**Missing from sample**: `data_access_labels.csv` and `user_profile_labels.csv` (ground truth). These may be released at evaluation time OR we evaluate against our injected anomaly labels.

### Label Strategy

Since ground truth labels are absent from sample data, our approach:

1. **Train behavioral baselines on full 1,201 events** (treat as historical "normal" with embedded anomalies)
2. **Inject 4 labeled anomaly events** (the demo personas below) with `is_anomaly=True`
3. **Evaluate precision/recall on injected + naturally detectable anomalies**
4. **If labels are released at eval time**, run our scorer against those labels immediately

### Financial Context Injection

Map PS4 generic resource names → SG-realistic system names for narrative layer:

```python
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
    "Finance":     "Finance & Risk",
    "Engineering": "IT Infrastructure",
    "Security":    "Information Security",
    "Legal":       "Legal & Compliance",
    "HR":          "Human Resources",
    "Sales":       "Client Coverage",
    "Marketing":   "Investment Research",
    "Compliance":  "Regulatory Compliance",
    "Operations":  "Operations & Settlement",
    "Executive":   "Senior Management",
    "IT":          "Technology",
    "Support":     "IT Support",
}
```

### Feature Engineering (No rowcount/destination — Adapt)

Features derived from the actual columns available:

| Feature | Derived from | Anomaly signal |
|---|---|---|
| `off_hours_export` | `action=export_data` + `time_class=night/unusual_hours` | Pre-exfiltration |
| `high_sens_export` | `action=export_data` + `resource_sensitivity=high` | Data theft |
| `cross_dept_access` | resource not in user's `systems_access` list | Unauthorized access |
| `dormant_activation` | `days_inactive > 30` + recent activity | Compromised account |
| `ip_entropy` | Shannon entropy of source_ip per user over rolling 7d | New IP location |
| `failure_burst` | ≥3 `status=failure` in 1h then `success` | Brute force |
| `new_resource` | First-ever access to this resource type for this user | Recon/expansion |
| `volume_deviation` | Events per day vs user's 30d rolling mean, Z-score | Bulk operation |
| `sensitivity_escalation` | Access sensitivity higher than user's historical max | Privilege abuse |
| `service_acct_anomaly` | `privilege_level=service-account` + `time_class≠business_hours` | Automated attack |

### Injected Anomaly Personas (Demo Replay)

Using real user IDs from the provided dataset, adapted to PS4 resource names:

| Persona | User ID | Profile | Pattern | MITRE | Risk |
|---|---|---|---|---|---|
| Finance pre-resignation | `USR00044` (pooja.mishra, Finance/Developer, admin, 59d inactive) | export_data on Customer_Vault (high) at 02:00, from new source IP never seen before | T1078 | 72/100 |
| IT audit evasion | `USR00015` (sophia.white, Security/Developer, admin) | admin_operation on Admin_Console at 23:47, action never in her 90d history | T1562.001 | 68/100 |
| Dormant account reactivation | `USR00005` (george.lim, Engineering/Admin, 57d inactive) | export_data on PROD_DB at 03:00, first event in 57 days | T1078 | 81/100 |
| Service account cross-dept | `USR00002` (kenneth.moore, service-account, approved: EMAIL\|PROD_DB only) | sql_query on HRIS — resource not in approved `systems_access` list | T1530 | 68/100 |

### Synthetic Scale-Up (SDV)

```python
from sdv.tabular import GaussianCopula

model = GaussianCopula()
model.fit(enriched_df)
synthetic_df = model.sample(num_rows=50000)
```

Scale from 100 users to 1,000+ for "enterprise scale" architecture slide. Don't run full model on synthetic — use for throughput benchmark.

---

## Key Research to Cite

| Paper | Result | Use in slides |
|---|---|---|
| GCN + Bi-LSTM (arxiv 2512.18483, Dec 2025) | AUC 98.62, 100% DR, 0.05% FPR on CERT r5.2 | Architecture reference |
| Federated Learning for Insider Threat (Scientific Reports 2025) | >90% accuracy, privacy loss <5% | Forward-looking architecture note |
| Mastercard GenAI + Graph (May 2024) | Doubled fraud detection, 3B cards | Real-world validation hook |
| DeepLog (baseline) | AUC 86.41 on CERT r5.2 | Baseline to beat |
| MITRE ATT&CK for Insider Threats — Securonix | Framework | Technique mapping |

**GitHub of primary architecture paper**: https://github.com/Yumlembam/Insider-Threat

---

## Demo Script (5 Minutes, Pre-Scripted)

Do not do live inference on random data during the demo. Pre-script a replay:

```
T+0:00  Dashboard: normal events streaming (green), 1,200 baseline events processed
T+0:45  USR00044 event surfaces: 02:00, new IP, Customer_Vault (Restricted), export_data
T+1:00  System flags: risk 96/100, +1,840% peer deviation vs Finance dept, MITRE T1078
T+1:15  Fable 5 narrative streams LIVE (streaming API, types out character by character)
         "pooja.mishra, Finance Developer, accessed Client Portfolio System at 2:14am from
          an IP address not seen in 90 days of baseline history..."
T+1:45  Behavioral baseline chart: 90-day history vs tonight's spike
T+2:00  Kill chain: recon (new resource access) → export (bulk export at night)
T+2:30  Mock action buttons: "Escalate to CISO" | "Freeze Account" | "Preserve Audit Trail"
T+3:00  Second scenario: USR00002 (service account) queries HRIS — out-of-scope access
T+4:00  Numbers slide: Precision/Recall/F1, time-to-alert, token cost
T+4:30  Architecture + federated learning note (enterprise readiness across BUs)
```

---

## Numbers Slide

```
Target benchmark: PS4 Evaluation Criteria (user-level, threshold=78)

Naive rule-based (flag all night access):
  Precision 40%    Recall 35%    F1 0.37

Isolation Forest (raw features):
  Precision 65%    Recall 60%    F1 0.62

Z-score self-comparison only:
  Precision 70%    Recall 68%    F1 0.69

Our system (GCN + Bi-LSTM + peer group) — MEASURED:
  Precision 80.6%  Recall 83.3%  F1 0.820

PS4 passing threshold: Precision > 75%, Recall > 70%, F1 > 0.72
Our result:           Precision 80.6%, Recall 83.3%, F1 0.820  [PASS]

Evaluation: 150-user synthetic benchmark, 30 labelled anomaly users (2 types each)
Real data: 100% recall on all 4 scripted demo personas at threshold 65

Time to alert:          < 3 seconds from event ingestion
Token cost/incident:    ~$0.003 (Groq llama-3.3-70b free tier)
False positive reduction vs rule-based:  ~70% (FP=6 vs ~40+ for rule engine)

Demo event scores (real PS4 data):
  USR00005 george.lim    81/100  HIGH     dormant + off-hours export + scope violation
  USR00044 pooja.mishra  72/100  HIGH     off-hours export + new IP + dormant
  USR00015 sophia.white  68/100  MEDIUM   admin op at night on new resource (T1562.001)
  USR00002 kenneth.moore 68/100  MEDIUM   service account scope violation
```

---

## MITRE ATT&CK Mapping

| Detected pattern | Technique ID | Tactic |
|---|---|---|
| `export_data` after-hours on `high` sensitivity | T1048 | Exfiltration |
| New source IP for established user | T1078 | Initial Access |
| Access to resource not in `systems_access` | T1530 | Collection |
| `admin_operation` on Admin_Console/SIEM at night | T1562.001 | Defense Evasion |
| Dormant account (days_inactive > 30) suddenly active | T1078 | Initial Access |
| Brute-force pattern (failures → success) | T1110 | Credential Access |
| Service account accessing off-scope resources | T1078.004 | Privilege Escalation |

---

## Peer Group Analysis (Key Differentiator)

Compare users not just to their own baseline but to their **department peer group**:
- "Is this Finance user exporting more high-sensitivity data than other Finance users?"
- Much lower false positive rate than self-comparison alone
- How enterprise UEBA tools (Splunk UBA, Microsoft Sentinel) actually work

The implicit graph in our GCN captures this automatically: users who access the same resources share edges, so their embeddings are similar. Divergence from the cluster = anomaly signal.

Peer groups defined by `department` column in `user_profiles.csv`.

---

## Federated Learning Note (Slides Only — Don't Implement)

Architecture is designed for federated deployment: each department trains locally, no raw logs cross department boundaries. Reference: Scientific Reports 2025 federated insider threat paper. SG has strict data sovereignty requirements across jurisdictions — this is directly relevant and no other team will mention it.

---

## What NOT To Do

- Don't make the LLM do the anomaly detection — it's a narrative layer only
- Don't use only self-comparison baselines (use peer groups)
- Don't demo on live random data — pre-script the replay
- Don't commit any CSVs or model weights to git
- Don't use Isolation Forest as the sole detector (everyone will)
- Don't add Co-Authored-By to git commits
- Don't claim `rowcount` or `destination` features — they don't exist in the provided data
- Don't pre-compute baselines from the profile CSV — derive them from log history (the harder, more impressive ML)
