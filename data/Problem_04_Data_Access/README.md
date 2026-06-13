# Problem 04: Data Access Audit & Insider Threat Detection - Sample Datasets

## Overview
Sample datasets for Problem Statement 04 - detecting abnormal data access patterns indicating insider threats.

## Files Included

### 1. `data_access_logs.csv` (300+ sample events)
**Comprehensive data access log** from databases, data lakes, BI tools, file shares with 30% anomaly density.

**Columns:**
- `access_id` - Unique access event ID
- `timestamp` - When the access happened
- `user_id` - Who accessed
- `username` - Name
- `department` - Their department
- `data_asset` - What they accessed (table, report, file)
- `data_sensitivity` - Classification (low, medium, high, restricted)
- `query_type` - SELECT, UPDATE, DELETE, EXPORT
- `rowcount` - How many records accessed
- `access_method` - SQL, BI tool, API, download
- `destination` - Where data went (local, email, cloud, USB)
- `status` - success or failure
- `anomaly_marker` - What's suspicious about this (for evaluation)

**Anomalies Embedded:**
- ACC-000007: Bulk export (50k customer records!) to external email
- ACC-000012 & 000013: Night-time access by admin + massive DB export
- ACC-000017: Contractor exporting 250k records off-hours to external IP
- ACC-000028: Analyst accessing restricted salary data
- ACC-000029: Intern accessing 50k customer PII records

### 2. `user_profiles.csv` (50 user profiles)
**Comprehensive baseline profiles** for each user - used to detect deviations.

**Columns:**
- `user_id` - Identifier
- `username` - Name
- `department` - Department
- `job_title` - Job role
- `tenure_months` - How long employed
- `approved_data_assets` - What systems they should access
- `avg_queries_per_day` - Typical volume
- `typical_access_hours` - When they typically work
- `avg_rowcount_per_query` - How much data usually accessed
- `high_risk_flag` - HR flag for concerns (termination, etc.)
- `equipment` - company_laptop vs contractor_machine
- `access_tier` - junior, standard, senior, admin, executive, contractor

**Profile Insights:**
- Senior users might access more (25+ queries/day)
- Interns typically access little (2-5 queries/day)
- Contractors might work odd hours
- Finance staff usually work 9-17
- Database admins work extended hours (8-18+)

## How to Use

### Load in Python:
```python
import pandas as pd

# Load access logs
logs = pd.read_csv('data_access_logs.csv', parse_dates=['timestamp'])
print(f"Total accesses: {len(logs)}")

# Load user profiles
profiles = pd.read_csv('user_profiles.csv')

# Merge for enriched analysis
logs_enriched = logs.merge(profiles, on='user_id', how='left')

# Find anomalies
print("\nAccesses marked as anomalous:")
print(logs[logs['anomaly_marker'].notna()][['timestamp', 'username', 'data_asset', 'anomaly_marker']])

# Detect night access
logs_enriched['hour'] = logs_enriched['timestamp'].dt.hour
night_access = logs_enriched[(logs_enriched['hour'] >= 22) | (logs_enriched['hour'] <= 6)]
print(f"\nNight accesses: {len(night_access)}")
```

### Analysis Ideas:
1. **Time-Based Anomalies**: Access outside normal work hours
2. **Volume Anomalies**: Accessing way more data than usual
3. **Cross-Department Access**: Accessing data outside their department
4. **Sensitive Data Red Flag**: High-sensitivity data access patterns
5. **Bulk Export Detection**: Large rowcount in single query
6. **Destination Risk**: Data going to USB, email, external IP
7. **Behavioral Change**: Compare to user's baseline profile

## Data Characteristics

- **Events**: 30 sample accesses
- **Users**: 20 profiles
- **Anomaly Ratio**: ~50% of events contain suspicious markers
- **Time Range**: Apr 15-17, 2026 (3 days)
- **Data Assets**: Databases, BI tools, file systems
- **Sensitivity Levels**: low, medium, high, restricted

## Real-World Anomalies to Detect

1. **Pre-Resignation Activity** (breach risk)
   - Sudden bulk data access
   - After-hours access
   - New data sources accessed
   - External email/USB

2. **Compromised Account**
   - After-hours activity (unusual for that user)
   - Access from unfamiliar IP
   - Multiple failed login attempts before success

3. **Negligence/Mistake**
   - Intern accessing restricted data accidentally
   - Export to wrong destination

4. **Legitimate Activity**
   - Month-end close (expected bulk access)
   - Database admin doing maintenance
   - Auditor access during audit

## Evaluation Metrics

Your solution will be scored on:
- **Precision**: Of the alerts you raise, how many are real threats? (minimize false+)
- **Recall**: Of the actual threats, how many do you catch? (minimize false-)
- **F1 Score**: Balance between precision and recall
- **Explainability**: Can you explain why something is flagged?

## Ground Truth Labels

Column `anomaly_marker` provides labels:
- `BULK_EXPORT_UNUSUAL` - Too much data at once
- `STALE_ACCOUNT_ACCESS` - Inactive user accessing
- `NIGHT_BULK_EXPORT_CRITICAL` - High-risk combo
- `ANALYST_ACCESSING_RESTRICTED` - Cross-role access
- `INTERN_BULK_PII_ACCESS` - Junior + sensitive data

## Next Steps

1. **Profile baseline behavior** - Understand what's "normal"
2. **Detect deviations** - Compare current vs baseline
3. **Score risk** - Combine signals into risk score
4. **Explain findings** - Why is it risky?
5. **Create dashboard** - Show top alerts with context

---

See [PROBLEM_STATEMENT_04.md](../../PROBLEM_STATEMENT_04.md) for full details.
