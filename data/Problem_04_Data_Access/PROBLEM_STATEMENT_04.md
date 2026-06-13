#  Problem Statement 04: Data Access Audit & Insider Threat Detection

> **Enterprise Challenge:** Detect abnormal data access patterns before sensitive information leaves the building.

---

## The Business Problem

**Scenario:** Enterprise processes **1M+ daily data access events** across:
- SQL databases (financial, HR, customer data)
- Data lakes (analytics, machine learning datasets)
- BI/reporting tools (Tableau, Power BI, Qlik)
- File shares & cloud storage (OneDrive, SharePoint, Google Drive)
- APIs & data exports

**The Threat:**
```
Tuesday 2:47 AM - Junior Developer queries customer database:
  SELECT * FROM customers WHERE salary > $200K

Expected: No access at this hour
Problem: 50,000 customer records exported to USB drive

Result: Data breach discovered 6 months later
```

**Real Incidents:**
- _Insider A:_ Finance analyst downloaded entire GL ledger before resignation (Monday), data sold to competitor
- _Insider B:_ HR analyst accessed salaries for 500 employees with jealous motive
- _Compromised Cred:_ Hacker used stolen admin password to access customer database at 3 AM (no one noticed)
- _Negligence:_ Developer configured app to export unencrypted customer data to test environment

**The Challenge:**
-  Too many access events to review manually (1M+ daily)
-  Naive alerting creates alert fatigue ("80% false positives")
- ⏱ Attacks detected weeks/months after data access
-  Hard to tell "normal work" from "suspicious" without context

**Compliance Gaps:**
- GDPR Article 32: Track data access, detect unauthorized use
- NIST IR-4: Incident detection response procedures
- SOX 302: Controls over sensitive financial data access

---

## Challenge Overview

Build a system to:
1. **Ingest** data access logs from multiple sources
2. **Establish** baseline behavior per user/role
3. **Detect** anomalies with high accuracy
4. **Classify** risk level (low, medium, high, critical)
5. **Investigate** incidents efficiently (provide context)
6. **Prevent** exfiltration (integrate with DLP if possible)

---

##  Data Reality & Edge Cases

**Behavioral Complexity:**
- Users have legitimate seasonal patterns (Finance team: busier month-end)
- Role changes create false positives (new admin has unusual access)
- On-call duties (engineer with elevated access 1 week/month)
- Contractors have short tenure (limited baseline history)
- Service accounts have no "normal" pattern
- Legitimate bulk exports (data warehouse refreshes, backups)

**Data Quality Issues:**
- Missing context (why did user access this data?)
- Timezone inconsistencies
- Access method variations (SQL, API, UI, download)
- Data sensitivity classifications incomplete
- User employment status data stale

**Evaluation Challenges:**
- Define "anomaly" precisely (statistical outlier vs risky activity)
- Distinguish insider threat from legitimate work
- Handle approved exceptions (exceptions that should suppress alerts)
- False positives destroy trust in system

---

##  Approach Options

### Option A: Behavioral ML + LLM Narratives (Advanced)
**Best for:** ML engineers, behavioral analysts

**Technical Approach:**
- Extract features per user per day:
  - Time of access (typical hours: 9-5 vs 2-3 AM?)
  - Data sensitivity accessed (public vs confined data?)
  - Volume accessed (10 rows vs 100k rows?)
  - Destination (local machine vs external email?)
  - Frequency (first time accessing this system?)
- Train isolation forest / autoencoder on "normal" behavior
- Score each access: 0-100 risk score
- LLM-powered investigation:
  - Input: User access pattern, their role, data sensitivity
  - Output: "Finance analyst accessed 50k customer records at 3 AM to personal email = HIGH RISK"
- Output: Risk dashboard with explanations and recommended actions

**Stack:** Python, Scikit-learn/PyTorch, Pandas, LLM API (OpenAI/HuggingFace), Plotly
**Complexity:**  (5/5)
**Effort:** 40-50 hours

---

### Option B: Statistical Anomaly Detection (Intermediate)
**Best for:** Data engineers, statisticians

**Technical Approach:**
- Build statistical profiles per user:
  - Distribution of access times (histogram)
  - Typical systems/databases accessed
  - Typical data volume per day
  - Typical destinations (local file system vs cloud)
- Flag deviations using Z-score or IQR (outlier detection)
- Context enrichment:
  - Is user on vacation? (check HR calendar)
  - Is this a job change? (elevated access expected)
  - Is there a business event? (month-end closing = more access)
- Risk scoring: Combined deviation score across features
- Output: Alert list with "deviation factor" explaining the anomaly

**Stack:** Python, NumPy/SciPy, Pandas, SQLite, Plotly
**Complexity:**  (3/5)
**Effort:** 25-35 hours

---

### Option C: Rule-Based Alerting Engine (Beginner-Intermediate)
**Best for:** Full-stack developers, security analysts

**Technical Approach:**
- Define simple rules in JSON:
  ```json
  {
    "rule_id": "off_hours_restricted_data",
    "condition": "access_time NOT IN (8-18) AND data_sensitivity='restricted'",
    "severity": "high",
    "action": "alert, block"
  },
  {
    "rule_id": "bulk_export",
    "condition": "rowcount > 10000",
    "severity": "medium",
    "action": "alert"
  }
  ```
- Build rule engine to evaluate each access event
- Track exceptions (e.g., "Finance can do bulk exports on Fridays")
- Generate daily alert summary reports
- Simple dashboard showing flagged accesses, user profiles, context

**Stack:** Python, REST API (Flask), PostgreSQL, JSON rule format, basic UI
**Complexity:**  (2/5)
**Effort:** 15-25 hours

---

## Sample Data Provided

**Files in `sample_data/`:**

| File | Records | Coverage | Description |
|------|---------|----------|-------------|
| `user_profiles.csv` | 100 | All tracked users | Baseline: department, role, privilege, typical hours, systems |
| `data_access_logs.csv` | 1,200 | Full 365 days (Apr 2025 – Apr 2026) | Every access: what, when, where, sensitivity, time classification |
| `user_profile_labels.csv` | 100 | All user profiles | Ground truth: is_anomaly, account risk, severity, explanation |
| `data_access_labels.csv` | 1,200 | All access events | Ground truth: is_anomaly, anomaly_type, severity, explanation |

**Anomaly distribution in labels:**
- Event-level anomalies: ~46% (bulk exports, after-hours, off-hours admin, cross-dept access)
- User-level risk: ~17% (stale accounts, over-privileged users)

**Self-Evaluation:**
```python
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

events = pd.read_csv('data_access_logs.csv')
labels = pd.read_csv('data_access_labels.csv')
# labels['predicted_anomaly'] = your_model.predict(events)

y_true = labels['is_anomaly'].astype(int)
y_pred = labels['predicted_anomaly'].astype(int)

print(f"Precision: {precision_score(y_true, y_pred):.2%}")
print(f"Recall:    {recall_score(y_true, y_pred):.2%}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.2f}")

# Severity breakdown
for sev in ['CRITICAL', 'HIGH', 'MEDIUM']:
    subset = labels[labels['severity'] == sev]
    subset_pred = labels.loc[subset.index]
    print(f"{sev} events in ground truth: {len(subset)}")
```

**Design for scale:** The sample dataset has 1,200 events across 365 days. Your solution architecture should handle 1M+ daily events. Document *how* you would scale it (streaming, partitioning, distributed compute) even if the sample runs locally.

**Sample Access Log:**
```csv
timestamp,user_id,username,department,action,database,table,rowcount,destination,sensitivity
2026-04-15 14:23:45,USR-0245,alice.smith,Finance,SELECT,GL_Ledger,accounts,150,local_workstation,high
2026-04-15 22:47:12,USR-0245,alice.smith,Finance,EXPORT,GL_Ledger,accounts,50000,external_email,high
2026-04-16 03:15:33,USR-1847,bob.jones,IT,SELECT,HR_Database,salaries,500,usb_drive,restricted
```

**Sample User Profile:**
```json
{
  "user_id": "USR-0245",
  "username": "alice.smith",
  "department": "Finance",
  "role": "Senior Analyst",
  "tenure_months": 48,
  "approved_systems": ["GL_Ledger", "AR_System"],
  "typical_access_hours": "9-17",
  "avg_queries_per_day": 12.5,
  "avg_rowcount_per_query": 450,
  "high_risk_event": false,
  "equipment": "company_laptop"
}
```

---

##  Success Criteria

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Precision** | > 75% | Minimize alert fatigue for analysts |
| **Recall** | > 70% | Don't miss actual threats |
| **F1 Score** | > 0.72 | Overall detection accuracy |
| **Detection Speed** | < 5 minutes | Timely alerting |
| **Explainability** | 4/5 stars | Investigators understand alerts |

**Baseline Comparison:**
- Naive approach (flag all night access): Precision 40%, Recall 35%
- **Your target:** Precision/Recall both > 70%

---

##  Deliverables

-  **Access log ingestion** (supports CSV, API formats)
-  **Anomaly detection model** (trained on provided data)
-  **Risk scoring engine** (ranks alerts by severity)
-  **Dashboard** (show top alerts, user profiles, data assets)
-  **Investigation toolkit** (contextual info for each alert)
-  **Sample incident report** (10-15 detected threats with narratives)
-  **Evaluation metrics** (metrics on ground truth labels)

---

##  Regulatory Alignment

**GDPR Article 32:** Security measures for personal data processing
-  Monitor unauthorized access
-  Detect data exfiltration attempts

**NIST IR-4:** Incident handling procedures
-  Detection capability (anomaly detection)
-  Response procedures (alerting + investigation)

**SOX 302:** Internal Controls over financial reporting
-  Unauthorized access to GL, AR, AP systems
-  Audit trail of data access

---

##  Tips to Win

1. **Feature engineering is key:** Think like a data scientist. What features distinguish "normal" from "suspicious"?
2. **Know your baseline:** Spend time understanding typical patterns before flagging anomalies
3. **Test on ground truth:** Use the 50 labeled scenarios to validate your approach
4. **Consider business context:** Access patterns change by time/season/role
5. **Make it operational:** Would a security analyst actually use this?
6. **Think explainability:** Can you tell an auditor why an alert triggered?

---

##  Example Output

```
DATA ACCESS ANOMALY REPORT - 2026-04-15
========================================

Critical Alerts (Immediate Investigation)


Alert 1: BULK EXPORT OF RESTRICTED DATA

User: bob.jones (IT, 3 months tenure)
Action: Export customer PII
Records: 50,000 customers with SSN, email
Destination: Personal USB drive
Time: 2026-04-15 03:47 AM (off-hours)
Risk Score: 96/100  CRITICAL

Context:
- First time accessing this table
- Never exports data outside work hours
- No business justification in tickets
- Employee marked "high risk": Termination notice filed yesterday

Recommendation: BLOCK + INVESTIGATE + audit logs from 72 hours


Medium Alerts (Review)


Alert 2: UNUSUAL TIMING
User: alice.smith (Finance, 4 years tenure)
Action: GL_Ledger query (normal access)
Time: 21:30 (beyond normal 9-17 hours)
Risk Score: 42/100
Context: Month-end close in progress (expected)
Recommendation: MONITOR
```

---

##  Example Walkthrough

**Input Data:**
```csv
2026-04-15 03:47:12,USR-0847,bob.jones,Export,PII_Database,critical,50000,personal_usb,ABNORMAL
```

**Expected Output:**
```json
{
  "alert_id": "ALERT-20260415-001",
  "user_id": "USR-0847",
  "risk_score": 96,
  "severity": "CRITICAL",
  "anomalies_detected": [
    "Off-hours access (03:47 vs normal 9-17)",
    "First-time access to restricted table",
    "Bulk export (50k records vs typical <100)",
    "USB export (exfiltration risk)"
  ],
  "business_context": "Employee filed termination notice yesterday",
  "recommendation": "BLOCK + INVESTIGATE IMMEDIATELY"
}
```

---

##  Evaluation Rubric (100 pts)

- **Detection Accuracy (30 pts):** Precision >75%, Recall >70%, distinguishes risky from legitimate
- **Risk Scoring (25 pts):** Combines multiple factors intelligently, explanations clear
- **False Positive Control (20 pts):** Contexts understood (month-end, role changes, on-call, etc.)
- **Performance (15 pts):** Analyzes 1M events in <120 sec
- **Presentation (10 pts):** Dashboard/report shows top risks with context
- **Bonus (5 pts):** ML-based profiling, DLP integration, trend analysis

---

##  Deliverables Checklist

- [ ] **GitHub Repo** - code, requirements.txt, clear README
- [ ] **Jupyter Notebook** - baseline analysis, feature importance
- [ ] **Anomaly Detection Output** - 20+ flagged accesses with explanations
- [ ] **Risk Dashboard** - CLI or web interface showing top alerts
- [ ] **False Positive Analysis** - document how you handle edge cases
- [ ] **Technical Docs** - ML approach, feature engineering, scaling
- [ ] **5-Min Presentation** - problem → solution → example alerts

---

##  Timeline

- **Day 1:** EDA → establish baselines → build core algorithm
- **Day 2:** Evaluate on test set → refine → add explanations → build UI
- **Day 3 (opt):** Improve accuracy → bonus features → polish demo

---

##  Bonus Features

- ML-based behavioral profiling (+5)
- DLP integration (prevent exfiltration) (+4)
- User context enrichment (HR status, on-call, etc.) (+3)
- Trend analysis (insider risk increasing?) (+2)

---

##  FAQ

**Q: How do we get user baseline?** A: Use first 30 days as "normal", then learn.
**Q: What about month-end seasonality?** A: Great question! Build seasonal profiles.
**Q: Can we use clustering?** A: Yes, excellent for grouping similar users.
**Q: False positive rate target?** A: <20% for system to be trusted.

---

##  Judge Guide

**Green Flags:** Low false positives, understands context (role changes, seasonality), clear narratives
**Red Flags:** Flags all night access, generic explanations, precision <50%
**Questions:** "Why is this person flagged?", "Who has legitimate off-hours access?", "How do you handle contractors?"

---

**Download starter code & data:** [DOWNLOAD_LINK]


