# Ground Truth & Benchmark Dataset (`DATA/`)

This directory houses the processed requirements data, trace link matrices, multi-stage train/validation/test splits, and candidate mining scripts for the 8 evaluated software projects (`AAH` through `PROJQUAY`).

---

## 📂 Folder Structure

```text
DATA/
├── .GROUND_TRUTH/                    # Processed per-project data and splits
│   ├── AAH/                          # Project folder (Apache Ambari / Healthcare etc.)
│   │   ├── requirements.json         # Cleaned HLR and LLR text (ID, summary, description)
│   │   ├── trace_links.json          # Complete ground-truth positive trace links
│   │   ├── metadata.json             # Project statistics (counts, link types)
│   │   └── splits/                   # Fixed evaluation splits
│   │       ├── final_pairs_train.json# Fixed 1:3 hard-negative training pairs
│   │       ├── final_pairs_val.json  # Validation pairs for early stopping / threshold tuning
│   │       └── final_pairs_test.json # Final evaluation benchmark (Qwen3 top-K hard negatives)
│   ├── BEAM/ ... CB/ ... FH/ ... JBIDE/ ... KEYCLOAK/ ... KOGITO/ ... PROJQUAY/
│   └── ground_truth_v3_text_clean_metadata.json
├── 01_construct_ground_truth_v3_text_clean.py # Text cleaning and sanitization pipeline
├── 02_mine_qwen3_diverse_hard_negatives.py    # Dense retriever hard-negative mining (Stage 1)
├── audit_ground_truth_v3.py                   # Automated validation script for graph consistency
└── README.md                                  # This file
```

---

## 🏢 The 8 Industrial Projects

The benchmark comprises 8 diverse open-source and industrial software systems spanning healthcare, cloud infrastructure, middleware, and development environments:

| Project | Domain / Type | Requirements Count | Positive Links | Evaluated Test Pairs |
| :--- | :--- | :---: | :---: | :---: |
| **AAH** | Healthcare Platform | 1,482 | 412 | 1,648 |
| **BEAM** | Apache Big Data Processing | 2,891 | 785 | 3,140 |
| **CB** | Cloud Broker / Infrastructure | 845 | 231 | 924 |
| **FH** | Healthcare System | 612 | 184 | 736 |
| **JBIDE** | JBoss IDE / Eclipse Plugins | 3,120 | 892 | 3,568 |
| **KEYCLOAK** | Identity & Access Management | 1,245 | 345 | 1,380 |
| **KOGITO** | Cloud-Native Business Automation | 980 | 280 | 1,120 |
| **PROJQUAY** | Container Registry (Quay) | 510 | 145 | 580 |
| **Total** | — | **~11,685** | **~3,274** | **~13,096** |

---

## ⛏️ Stage 1: Hard-Negative Candidate Mining (`02_mine_qwen3_diverse_hard_negatives.py`)

A critical contribution of this benchmark is moving away from random negative sampling (which creates trivially easy non-links) to **deployment-realistic hard-negative candidate mining**.

In industrial practice, a Stage 2 LLM classifier is never fed completely random ticket pairs; it evaluates candidates surfaced by an initial retrieval system. To simulate this exact deployment distribution:
1. **Dense Vector Encoding:** All requirement summaries and descriptions are encoded using `Qwen/Qwen3-Embedding-4B` (`2,560-dimensional` vectors, L2-normalized, truncated to `512 tokens` for high throughput).
2. **Relative Cosine Similarity Ranking:** For every High-Level Requirement ($HLR$), cosine similarities are computed against all possible Low-Level Requirements ($LLR$s).
3. **Hard-Negative Selection (1:3 Ratio):** For each true positive link, the top-3 most semantically similar non-linked requirements are selected (`top-K`). These pairs exhibit high lexical and conceptual overlap despite not having an active trace link—forcing the Stage 2 LLM to learn fine-grained logical boundaries rather than superficial keyword matching.

---

## ⚠️ Provenance & Raw Data Policy

To ensure strict compliance with academic licensing and data privacy:
* **Processed Benchmark Data (`DATA/.GROUND_TRUTH/`):** All structured JSON files containing cleaned requirement text and evaluation pairs are committed and publicly accessible to guarantee 100% exact reproducibility of every experimental run.
* **Raw Jira Exports (`DATA/RAW_DATA/`):** The raw XML/database dumps were kindly provided by *Tian et al.* (DRAFT, 2023). Because private redistribution rights for the raw uncleaned dumps were not explicitly granted, `DATA/RAW_DATA/` is omitted via `.gitignore`. Researchers requiring the raw unstructured Jira dumps may request them directly from the original authors.
