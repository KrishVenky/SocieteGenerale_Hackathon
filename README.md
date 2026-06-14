# SG Hackathon - Insider Threat Detection System

**Societe Generale Hackathon | PS4: Data Access Audit & Insider Threat Detection**

A two-stage insider threat detection system combining graph ML anomaly detection with LLM-generated incident narratives, benchmarked on the CMU CERT r5.2 dataset with financial-sector context injection.

---

## What It Does

Ingests multi-source enterprise access logs, builds per-user and peer-group behavioral baselines, detects anomalies using a GCN + Bi-LSTM model, risk-ranks incidents, maps them to MITRE ATT&CK techniques, and generates streaming incident narratives via Claude Fable 5.

**Key output per incident**: risk score, confidence, behavioral deviation from peer group, kill chain stage, MITRE technique, and a plain-English incident report with recommended response steps.

---

## Architecture

```
Access Event Stream
    → Behavioral Baseline Engine (rolling stats + peer group)
    → Graph Construction (explicit user→resource + implicit peer graph)
    → GCN + Bi-LSTM Anomaly Detector
    → Risk Ranker (deviation × sensitivity × time-of-day)
    → MITRE ATT&CK Mapper
    → Claude Fable 5 Incident Narrative (streaming)
    → Streamlit Dashboard
```

---

## Dataset

- **Base**: CMU CERT Insider Threat Dataset r5.2 : industry standard benchmark (3,995 users, 5 labeled insiders)
- **Enriched**: Financial context injection (Murex, Bloomberg, SWIFT, Calypso system names; sensitivity tiers; business units)
- **Scale**: SDV synthetic scale-up to 50k users preserving CERT distributions
- **Reference**: LANL Unified Host & Network Dataset for enterprise-scale architecture validation

---

## Performance Target

| Model | AUC | Detection Rate | FPR |
|---|---|---|---|
| Our system (target) | 92-96% | >95% | <0.10% |
| DeepLog baseline | 86.41% | 81.89% | 0.19% |
| SOTA (arxiv 2512.18483) | 98.62% | 100% | 0.05% |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | Python 3.11 + FastAPI |
| ML | scikit-learn, PyTorch, NetworkX |
| AI Narrative | Claude Fable 5 (streaming) |
| AI Triage | Claude Haiku 4.5 |
| Frontend | Streamlit |
| Data | pandas + SDV |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Download CERT r5.2 dataset
# https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
# Place extracted files in data/raw/cert_r5.2/

# Run data pipeline
python scripts/prepare_data.py

# Launch dashboard
streamlit run app/dashboard.py
```

---

## Project Structure

```
SG_Hackathon/
├── app/
│   └── dashboard.py          # Streamlit UI
├── src/
│   ├── baseline.py           # Behavioral baseline engine
│   ├── graph_builder.py      # NetworkX graph construction
│   ├── detector.py           # GCN + Bi-LSTM anomaly detector
│   ├── ranker.py             # Risk scorer and MITRE mapper
│   └── narrator.py           # Claude Fable 5 narrative generator
├── scripts/
│   ├── prepare_data.py       # CERT preprocessing + context injection
│   ├── inject_financial_context.py
│   └── generate_demo_replay.py
├── data/                     # gitignored : download separately
│   ├── raw/
│   └── processed/
├── models/                   # gitignored
├── CLAUDE.md
├── ARCHITECTURE.md
├── DATASETS.md
├── RESEARCH.md
└── requirements.txt
```

---

## Key Research

- [GCN + Bi-LSTM Insider Threat Detection : arxiv 2512.18483](https://arxiv.org/abs/2512.18483) : architecture reference, AUC 98.62 on CERT r5.2
- [Paper GitHub](https://github.com/Yumlembam/Insider-Threat) : preprocessed CERT data + feature extraction code
- [Federated Learning for Insider Threat : Scientific Reports 2025](https://www.nature.com/articles/s41598-025-04029-w)
- [Mastercard GenAI + Graph Fraud Detection](https://newsroom.mastercard.com/news/press/2024/may/mastercard-accelerates-card-fraud-detection-with-generative-ai-technology/) : real-world validation
- [MITRE ATT&CK for Insider Threats : Securonix](https://www.securonix.com/blog/applying-the-mitre-attck-framework-to-insider-threats/)
