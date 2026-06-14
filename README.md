# WatchDog - Insider Threat Detection
**Societe Generale Hackathon | PS4: Data Access Audit & Insider Threat Detection**

Real-time insider threat detection combining behavioral baselining, graph ML, and Bi-LSTM sequence modeling on enterprise access logs.

---

## What It Does

Ingests raw access logs, builds per-user and peer-group behavioral baselines from log history, detects anomalies using a GCN-inspired + Bi-LSTM pipeline, risk-ranks each incident, maps it to a MITRE ATT&CK technique, and streams an LLM-generated incident narrative to a live dashboard.

**Output per alert:** risk score (0-100), severity, behavioral deviation vs peer group, MITRE technique, kill chain stage, and a plain-English incident report with recommended response steps.

---

## Results (PS4 Evaluation Criteria)

| Metric | Required | Our Result |
|---|---|---|
| Precision | >= 75% | **80.6%** |
| Recall | >= 70% | **83.3%** |
| F1 | >= 0.72 | **0.820** |

Evaluated user-level at threshold 78 on a 150-user synthetic benchmark with 30 labeled anomaly users.

---

## Architecture

```
Access Event Stream (data_access_logs.csv)
    -> Behavioral Baseline Engine  (rolling 30-day stats, peer groups by dept)
    -> Graph Construction          (NetworkX user-resource + implicit peer graph)
    -> Bi-LSTM Autoencoder         (PyTorch, unsupervised, reconstruction error)
    -> Risk Ranker                 (55% behavioral + 25% LSTM + 20% graph)
    -> MITRE ATT&CK Mapper         (7 techniques, 6 tactics)
    -> LLaMA 3.3-70B Narrative     (Groq, streaming, ~200 tok/s)
    -> FastAPI Dashboard           (localhost:8000)
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. The only key you need is optional:

```
GROQ_API_KEY=your_key_here    # free at console.groq.com - for narrative generation only
```

**The detection pipeline runs fully without any API key.** Risk scores, MITRE mapping, behavioral baselines, and the full alert queue work offline. The Groq key only enables the streaming LLM narrative panel in the dashboard.

---

## Run

```bash
# Prepare data (first time only)
python scripts/prepare_data.py

# Terminal report (PS4-style output)
python scripts/report.py --demo

# Live dashboard
uvicorn app.server:app --port 8000
# open localhost:8000
```

---

## Project Structure

```
SG_Hackathon/
├── app/
│   ├── server.py             # FastAPI backend
│   └── static/index.html     # Single-page dashboard
├── src/
│   ├── baseline.py           # Behavioral baseline engine
│   ├── graph_builder.py      # NetworkX graph construction
│   ├── detector.py           # Bi-LSTM autoencoder
│   ├── ranker.py             # Risk scorer + MITRE mapper
│   └── narrator.py           # Groq LLaMA narrative generator
├── scripts/
│   ├── prepare_data.py       # Data pipeline
│   ├── report.py             # Terminal CLI report
│   └── evaluate.py           # Precision/Recall/F1 evaluation
├── .env.example
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | Python 3.11 + FastAPI |
| ML | pandas, scikit-learn, PyTorch, NetworkX |
| Narrative | Groq LLaMA 3.3-70B (optional) |
| Frontend | Vanilla HTML/CSS/JS |
| Data | PS4 official sample + SDV synthetic scale-up |
