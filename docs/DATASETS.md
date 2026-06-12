# Datasets

## Primary: CERT Insider Threat Dataset r5.2

**Download**: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
**Size**: 10.37 GB (r5.2), 22.18 GB (r6.2)
**License**: Free, registration required

**Files**:
| File | Contents |
|---|---|
| `logon.csv` | Workstation logon/logoff events |
| `email.csv` | Email send/receive (internal + external) |
| `http.csv` | Web browsing history |
| `file.csv` | File access, copy, move events |
| `device.csv` | USB/removable media connections |
| `psychometric.csv` | Employee survey scores (use as sensitivity proxy) |

**Ground truth**: `answers/` folder contains labeled malicious user IDs and incident dates

**Stats (r5.2)**:
- 3,995 benign employees
- 5 labeled malicious insiders
- ~32M total events

This is the industry-standard benchmark. All AUC numbers in research papers use this dataset, so your metrics are directly comparable.

---

## Paper GitHub (Preprocessed CERT Data + Code)

**Primary paper**: arxiv 2512.18483 — GCN + Bi-LSTM, AUC 98.62 on CERT r5.2
**GitHub**: https://github.com/Yumlembam/Insider-Threat

Contains:
- `data/preprocessed/r5_data/` — preprocessed CERT r5.2 ready to use
- `data/preprocessed/r6_data/` — preprocessed CERT r6.2
- Notebooks for data preparation and feature extraction
- GCN-NN, GCN-BiLSTM, GCN-BiLSTM+Attention implementations

**Feature extraction reference repos**:
- https://github.com/liujie40/feature-extraction-for-CERT-insider-threat-test-dataset
- https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets

---

## Scale Validation: LANL Unified Host & Network Dataset

**Download**: https://csr.lanl.gov/data/2017/
**License**: Public domain (CC0), LANL approval LA-UR-17-20763
**Size**: ~12 GB compressed

**Contents**:
- 90 days of real enterprise Windows authentication logs
- Authentication events: user, source host, destination host, auth type, result
- Process events (start/stop)
- Network flow data

**Use case**: Show your ingestion pipeline handles 1B+ events. Don't run the full model — use this for the architecture slide and a benchmark throughput number ("processes X events/second").

---

## Financial Context Injection

After downloading CERT, run `scripts/inject_financial_context.py` to transform generic CERT data into SG-realistic data.

### Resource → System Name Mapping

```python
SYSTEM_MAP = {
    # File server patterns → trading/risk systems
    "PC-.*\\\\share\\\\finance": "Murex MX.3 (Trading Positions)",
    "PC-.*\\\\share\\\\risk": "Calypso (Risk Analytics)",
    "PC-.*\\\\share\\\\client": "CRM (Client Portfolios)",
    "PC-.*\\\\share\\\\research": "Research Repository",
    # External HTTP → financial data terminals
    "bloomberg.com": "Bloomberg Terminal",
    "reuters.com": "Refinitiv Eikon",
    "factset.com": "FactSet Analytics",
    # Email external domains → SWIFT gateway proxy
    ".*@swift.com": "SWIFT Message Gateway",
    ".*@clearinghouse": "Settlement System",
}

SENSITIVITY_MAP = {
    "Murex MX.3": "Restricted",
    "Bloomberg Terminal": "Confidential",
    "SWIFT Message Gateway": "Restricted",
    "Calypso": "Confidential",
    "CRM": "Restricted",
    "Research Repository": "Confidential",
    "HR System": "Restricted",
    "SharePoint": "Internal",
}

BUSINESS_UNIT_MAP = {
    # Assign based on user ID patterns in CERT data
    # or map randomly by role
    "Trader": ["Equities", "FICC"],
    "Analyst": ["M&A Advisory", "Research", "Risk"],
    "IT Admin": ["IT Infrastructure"],
    "Contractor": ["External/Contractor"],
    "Manager": ["Compliance", "Risk Management"],
}
```

### Anomaly Personas (Inject into Dataset)

Four pre-scripted anomaly personas to inject into the CERT user pool for the demo replay:

```python
DEMO_PERSONAS = [
    {
        "user_id": "ALICE.CHEN",
        "role": "M&A Analyst",
        "bu": "M&A Advisory",
        "anomaly": {
            "timestamp": "2024-03-15 02:14:00",
            "source_ip": "185.234.XXX.XXX",  # Paris IP, new
            "system": "Murex MX.3",
            "action": "file_read",
            "volume": 3247,
            "sensitivity": "Restricted",
            "mitre": "T1078",
            "days_before_resignation": 3
        }
    },
    {
        "user_id": "BOB.SHARMA",
        "role": "IT Admin",
        "bu": "IT Infrastructure",
        "anomaly": {
            "timestamp": "2024-03-08 23:47:00",
            "action": "audit_log_disabled",
            "system": "SWIFT Message Gateway",
            "mitre": "T1562.001",
            "prior_occurrences": 0
        }
    },
    {
        "user_id": "SA-EXT-047",
        "role": "Contractor",
        "bu": "External/Contractor",
        "anomaly": {
            "timestamp": "2024-03-12 18:02:00",
            "action": "usb_copy",
            "volume": 14847,
            "sensitivity": "Restricted",
            "mitre": "T1052",
            "account_inactive_days": 547
        }
    },
    {
        "user_id": "SA-MUREX-PROD",
        "role": "Service Account",
        "bu": "IT Infrastructure",
        "anomaly": {
            "timestamp": "2024-03-20 03:22:00",
            "system": "HR System",
            "action": "db_query",
            "sensitivity": "Restricted",
            "mitre": "T1530",
            "prior_access_to_system": False
        }
    }
]
```

---

## Synthetic Scale-Up (SDV)

```bash
pip install sdv
```

```python
from sdv.tabular import GaussianCopula

# Train on enriched CERT data
model = GaussianCopula()
model.fit(enriched_df)

# Generate 50k synthetic users with same statistical properties
synthetic_users = model.sample(50000)

# Inject demo personas into synthetic population
full_dataset = pd.concat([synthetic_users, persona_events])
```

---

## Data Directory Structure (gitignored)

```
data/
├── raw/
│   ├── cert_r5.2/
│   │   ├── logon.csv
│   │   ├── email.csv
│   │   ├── http.csv
│   │   ├── file.csv
│   │   ├── device.csv
│   │   └── answers/
│   └── lanl_2017/
│       ├── auth.txt.gz
│       └── proc.txt.gz
├── processed/
│   ├── enriched_cert.parquet
│   ├── user_baselines.parquet
│   ├── graphs/
│   │   ├── explicit_graph.pkl
│   │   └── implicit_graph.pkl
│   └── demo_replay/
│       └── scripted_events.jsonl
└── synthetic/
    └── scaled_users.parquet
```
