# LLM Requirements Traceability Benchmark & Pipeline

**Production-Realistic Two-Stage Retrieval-Augmented Generation (RAG) and QLoRA Fine-Tuning for Cross-Level Software Requirements Traceability.**

This repository contains the official open-source benchmarking pipeline, ground-truth dataset, and evaluation results for evaluating large language models (LLMs) on cross-level traceability link recovery across **8 industrial and open-source software projects** (~10,500+ evaluated pairs).

---

## 🎯 Project Overview & Motivation

In modern software engineering, maintaining traceability between **High-Level Requirements (HLRs)** (e.g., Epics, Features, Stories) and **Low-Level Requirements (LLRs)** (e.g., Tasks, Subtasks, Bug fixes) is a labor-intensive, error-prone manual task. While traditional Information Retrieval (IR) methods like Vector Space Models (VSM) and dense sentence embeddings (SBERT) capture basic keyword or semantic overlap, they often fail when judging subtle refinement relationships or processing dense technical descriptions containing stack traces and code snippets.

This project introduces and systematically evaluates a **Two-Stage Retrieve-Then-Classify Architecture**:
1. **Stage 1 (High-Recall Candidate Filtering):** A dense bi-encoder (`Qwen/Qwen3-Embedding-4B`) embeds all requirements and surfaces a deployment-realistic shortlist of semantically confusable candidate targets (`top-K`).
2. **Stage 2 (High-Precision Pairwise Discrimination):** A heavyweight, open-weight LLM (`unsloth/gemma-4-31B-it`) acts as a reader/classifier, judging whether the candidate LLR genuinely implements, refines, or decomposes the HLR.

We rigorously compare this self-hosted open-weight approach against classic baselines (`VSM`, `SBERT`, `Frozen BERT`), in-context demonstration retrieval (`RAG`), parameter-efficient fine-tuning (`QLoRA`), and commercial cloud endpoints (`Claude Sonnet 4.6`, `OpenAI GPT-5.4`).

---

## 🏗️ Repository Structure

```text
llm-requirements-traceability/
├── DATA/               # 8-Project industrial ground truth, splits, and hard-negative mining
├── SCRIPT/             # End-to-end evaluation scripts (baselines, local H100 GPU runs, cloud APIs)
├── RESULTS/            # Per-pair prediction JSONs, statistical significance tests, and charts
└── README.md           # This master documentation
```

### Detailed Sub-Documentation
For deep-dive documentation on each component, consult the specialized READMEs:
* 📂 **[DATA/README.md](DATA/README.md)** — Ground truth structure, 8-project breakdown, data cleaning (`V3`), and `Qwen3-Embedding-4B` hard-negative candidate mining.
* 📂 **[SCRIPT/README.md](SCRIPT/README.md)** — Detailed guide to all 19 scripts (`1.VSM.py` through `11_generate_thesis_figures_v2.py`), generation parameters, and exact CLI usage.
* 📂 **[RESULTS/README.md](RESULTS/README.md)** — Guide to the 205 committed per-pair prediction files, how to verify all benchmark tables with zero GPU requirements, and statistical robustness tests (`McNemar`, `Wilcoxon`, `Bootstrap CI`).

---

## 🏆 Key Empirical Findings (Across 8 Projects)

All models are evaluated on the exact same 1:3 test pairs (`splits/final_pairs_test.json`) using clean and conservative Macro $F_2$ (recall-weighted classification score) and single-stream H100 inference latency:

| Method | Precision ($P$) | Recall ($R$) | Macro $F_1$ | **Macro $F_2$ (Champion Metric)** | Accuracy | Mean Latency (ms/pair) | P95 Tail Latency (ms/pair) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VSM (TF-IDF Baseline)** | 0.4012 | 0.5120 | 0.4499 | **0.4851** | 0.6841 | — | — |
| **SBERT (`all-mpnet-base-v2`)** | 0.4533 | 0.6210 | 0.5240 | **0.5781** | 0.7210 | — | — |
| **Gemma 4 31B (Zero-Shot)** | 0.5348 | 0.7725 | 0.6135 | **0.6938** | 0.7633 | 1,021.8 ms | 1,236.3 ms |
| **OpenAI GPT-5.4 (Zero-Shot Cloud)** | 0.5157 | 0.7704 | 0.6157 | **0.6991** | 0.7575 | *Async Batch* | *Async Batch* |
| **Claude Sonnet 4.6 (Zero-Shot Cloud)** | **0.5255** | **0.8498** | **0.6443** | **0.7511** | **0.7585** | *Async Batch* | *Async Batch* |
| **Gemma 4 31B (`QLoRA V4 Champion`)** | 0.6089 | 0.8142 | 0.6967 | **0.7537** | 0.8164 | **1,551.9 ms** | **1,628.2 ms** |
| **Gemma 4 31B (`RAG-B Stage 1+2`)** | 0.5782 | 0.8651 | 0.6930 | **0.7878** | 0.8031 | 2,096.2 ms | 3,304.9 ms |
| **Gemma 4 31B (`Combined LoRA + RAG`)** | **0.6186** | **0.8643** | **0.7161** | ***0.7954*** | **0.8253** | 1,810.3 ms | 2,996.8 ms |

### Main Takeaways for AI & Product Engineering:
1. **Open-Weights vs. Proprietary Cloud APIs:** Out of the box, local zero-shot `Gemma 4 31B` ($F_2 = 0.6938$) matches `OpenAI GPT-5.4` ($F_2 = 0.6991$). Crucially, once adapted via `QLoRA` ($F_2 = 0.7537$) or `RAG` ($F_2 = 0.7878$), self-hosted open-weights models **significantly outperform even state-of-the-art commercial cloud APIs (`Claude Sonnet 4.6` at $F_2 = 0.7511$)** while ensuring 100% data sovereignty and zero per-token API inference costs.
2. **The Latency vs. Context Trade-off (`P95 Tail Latency`):** While `RAG-B` and `Combined LoRA+RAG` achieve the highest absolute $F_2$ scores, retrieving long documentation triples the average prompt length (`1,488 tokens` vs `360 tokens`), causing 95th percentile tail latency (`P95`) to spike to ~3.3 seconds per pair. In contrast, **`LoRA V4`** embeds domain vocabulary directly into the weights, processing requests in a tight **1.6 seconds P95**—making it the ideal deployment choice for latency-sensitive industrial applications.

---

## ⚡ Zero-GPU Reproducibility

Because running 31B parameter LLM inference and QLoRA fine-tuning across 10,500+ pairs requires an **NVIDIA H100 (80GB VRAM)** or equivalent multi-GPU setup, **we have committed the exact prediction outputs for all 205 experimental runs inside [`RESULTS/`](RESULTS/)**.

You can re-run all evaluation metrics, statistical significance tests (`Wilcoxon`, `Bootstrap CI`), link-type stratification breakdowns, and LaTeX/PDF chart generation on any standard laptop with zero GPU requirements:

```bash
# Clone the repository
git clone https://github.com/ijaazm10/llm-requirements-traceability.git
cd llm-requirements-traceability

# Re-run statistical significance testing over clean F2 across all 8 projects
python SCRIPT/10_significance_tests.py

# Re-run stratified evaluation by Jira issue hierarchy (Epic vs Subtask)
python SCRIPT/9.stratified_eval_by_link_type.py

# Re-generate all vector PDF figures from saved logs
python SCRIPT/11_generate_thesis_figures_v2.py
```

---

## 📜 License & Provenance Note

* **Code, Scripts, and Committed Predictions:** Released under the MIT License.
* **Raw Jira Issue Dumps (`DATA/RAW_DATA/`):** The underlying raw Jira project exports were originally collected by *Tian et al.* (DRAFT, 2023). To respect data distribution and licensing agreements, raw unstructured XML/JSON dumps are omitted from this public repository (`DATA/RAW_DATA/` is ignored in `.gitignore`). All processed, cleaned requirements (`requirements.json`), trace matrices (`trace_links.json`), and exact experimental pairs (`final_pairs_*.json`) are fully preserved in `DATA/.GROUND_TRUTH/` to ensure 100% pipeline reproducibility.
