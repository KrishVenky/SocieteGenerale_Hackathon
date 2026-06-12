# Research References

## Primary Architecture Reference

### GCN + Bi-LSTM Insider Threat Detection (Dec 2025)
- **Paper**: arxiv 2512.18483
- **Title**: "Insider Threat Detection Using GCN and Bi-LSTM with Explicit and Implicit Graph Representations"
- **Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang, Deepa Krishnan
- **GitHub**: https://github.com/Yumlembam/Insider-Threat
- **Results on CERT r5.2**: AUC **98.62**, Detection Rate **100%**, FPR **0.05%**
- **Results on CERT r6.2**: AUC 88.48, DR 80.15%, FPR 0.15%
- **Key idea**: Two separate GCNs process explicit (user→resource) and implicit (peer group) graphs; embeddings concatenated + attention; fed into Bi-LSTM for temporal modeling
- **Use in project**: Direct architecture reference. Cite on architecture slide.

---

## Baseline (What to Beat)

### DeepLog (2017)
- **Result on CERT r5.2**: AUC 86.41, TPR 81.89%, FPR 0.19%
- **Method**: LSTM predicts next log entry; low probability = anomaly
- **Why cite it**: Establishes the "naive sequence model" baseline. Teams using only sequence modeling without graphs will land here.

---

## Supporting Research

### Federated Learning for Insider Threat (2025)
- **Paper**: Scientific Reports 2025
- **Link**: https://www.nature.com/articles/s41598-025-04029-w
- **Result**: Detection accuracy >90%, privacy loss <5%, communication efficiency +25%
- **Key idea**: Each org unit trains locally, only model weights shared — no raw logs leave the department
- **Use in project**: Architecture slide note on enterprise readiness and data sovereignty. Directly relevant to SG's multi-jurisdiction requirements. Don't implement, just design for it.

### Transformer-Based User Sequence Modeling (2025)
- **Paper**: arxiv 2506.23446
- **Link**: https://arxiv.org/html/2506.23446v1
- **Result**: Recall 99.43%, AUROC 95% on CERT
- **Method**: Multi-layer Transformer encoder (6 layers, 512 d_model, 8 attention heads) on user event sequences
- **Use**: Alternative to Bi-LSTM if implementation runs short on time

### Session-Graph GNN (2025)
- **Result**: 99.56% TPR, 0% FPR on CERT AB-II subset
- **Method**: 7 heuristic rules build Associated Session Graph (ASG) with session nodes, core/boundary nodes, inter-session edges
- **Use**: Cite for "near-perfect detection is achievable" framing

### Heterogeneous GNN Survey for Cybersecurity (Oct 2025)
- **Paper**: arxiv 2510.26307
- **Link**: https://arxiv.org/pdf/2510.26307
- **Use**: Shows you surveyed the field before choosing architecture

### Memory-Augmented Log Analysis with Phi-4-mini (2025)
- **Paper**: arxiv 2510.00529
- **Use**: Shows LLM-augmented log analysis is an active research area; validates using Fable 5 for narrative layer

---

## Real-World Industry Validation

### Mastercard GenAI + Graph Fraud Detection (May 2024)
- **Source**: https://newsroom.mastercard.com/news/press/2024/may/mastercard-accelerates-card-fraud-detection-with-generative-ai-technology/
- **Result**: Doubled fraud detection speed, deployed across 3 billion cards globally
- **Method**: Generative AI + graph technology modeling relationships between cards, merchants, transaction patterns
- **Use in slides**: Opening hook — "We applied the same graph + GenAI architecture Mastercard deployed in production to insider threat detection."

### BGL BNP Paribas ML Fraud Detection
- **Result**: ML cut false positives by 40%
- **Use**: FP reduction benchmark citation

### Gartner UEBA Benchmark
- **Result**: Behavioral analytics reduces false positives 60-70% vs rule-based systems
- **Use**: "False positive reduction vs rule-based: ~60-70% (Gartner UEBA benchmark)" on numbers slide

---

## Framework References

### MITRE ATT&CK for Insider Threats
- **Source**: https://www.securonix.com/blog/applying-the-mitre-attck-framework-to-insider-threats/
- **Use**: Technique mapping in risk ranker

### NIST Controls Referenced in PS4
- **GDPR Article 32**: Security of processing — data access controls
- **NIST IR-4**: Incident handling
- **SOX**: Controls over sensitive financial data access

---

## Benchmark Comparison Table (For Slides)

| Model | Year | AUC | DR | FPR | Dataset |
|---|---|---|---|---|---|
| Session-Graph GNN | 2025 | ~99% | 99.56% | 0% | CERT AB-II |
| **GCN + Bi-LSTM** | **2025** | **98.62%** | **100%** | **0.05%** | **CERT r5.2** |
| Transformer (6L) | 2025 | 95% AUROC | 99.43% | — | CERT |
| **Our system** | **2025** | **[X]** | **[Y]** | **[Z]** | **CERT r5.2** |
| DeepLog | 2017 | 86.41% | 81.89% | 0.19% | CERT r5.2 |
| Isolation Forest | — | ~85% | varies | high | CERT |

Fill in our numbers after evaluation on CERT r5.2 ground truth.

---

## Feature Extraction Repositories

- https://github.com/liujie40/feature-extraction-for-CERT-insider-threat-test-dataset
- https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets
- https://github.com/AymanMansur/Insider-threat-detection-using-cert-dataset-Logon-

---

## Datasets

- **CERT r5.2**: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
- **LANL 2017**: https://csr.lanl.gov/data/2017/
- **Splunk BOTS v3**: https://github.com/splunk/botsv3
- **PaySim (financial transactions)**: https://www.kaggle.com/datasets/ealaxi/paysim1
- **Awesome Cybersecurity Datasets**: https://github.com/shramos/Awesome-Cybersecurity-Datasets
