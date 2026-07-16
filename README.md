# LLM Requirements Traceability Benchmark & Pipeline

**A two-stage (retrieve-then-classify) evaluation pipeline with QLoRA fine-tuning and RAG for cross-level software requirements traceability, benchmarked under a controlled hard-negative protocol.**

This repository contains the official open-source benchmarking pipeline, ground-truth dataset, and evaluation results for evaluating large language models (LLMs) on cross-level traceability link recovery across **8 real-world open-source software projects** (~10,500+ evaluated pairs).

---

## 🎯 Project Overview & Motivation

In modern software engineering, maintaining traceability between **High-Level Requirements (HLRs)** (e.g., Epics, Features, Stories) and **Low-Level Requirements (LLRs)** (e.g., Tasks, Subtasks, Bug fixes) is a labor-intensive, error-prone manual task. While traditional Information Retrieval (IR) methods like Vector Space Models (VSM) and dense sentence embeddings (SBERT) capture basic keyword or semantic overlap, they often fail when judging subtle refinement relationships or processing dense technical descriptions containing stack traces and code snippets.

This project introduces and systematically evaluates a **Two-Stage Retrieve-Then-Classify Architecture**:
1. **Stage 1 (High-Recall Candidate Filtering):** A dense bi-encoder (`Qwen/Qwen3-Embedding-4B`) embeds all requirements and surfaces the semantically confusable candidate targets that make up the hard-negative benchmark (a controlled reader-stage protocol, not a deployment simulation) (`top-K`).
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
* 📂 **[RESULTS/README.md](RESULTS/README.md)** — Guide to the 205 committed per-pair prediction files, how to verify all benchmark tables with zero GPU requirements, and statistical robustness tests (`sign test`, `Wilcoxon`, `Bootstrap CI`).

---

## 🏆 Key Empirical Findings (Across 8 Projects)

All models are evaluated on the exact same 1:3 test pairs (`splits/final_pairs_test.json`) using clean and conservative Macro $F_2$ (recall-weighted classification score) and single-stream H100 inference latency:

| Method | Precision ($P$) | Recall ($R$) | Macro $F_1$ | **Macro $F_2$ (Primary Metric)** | Accuracy | Mean Latency (ms/pair) | P95 Tail Latency (ms/pair) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| All-positive sanity check | 0.2500 | 1.0000 | 0.4000 | **0.6250** | 0.2500 | — | — |
| **VSM (TF-IDF Baseline)** | 0.2984 | 0.6941 | 0.4139 | **0.5433** | — | — | — |
| **SBERT (`all-mpnet-base-v2`)** | 0.2528 | 0.9663 | 0.4006 | **0.6174** | — | — | — |
| **Frozen BERT + MLP** | 0.2758 | 0.9304 | 0.4219 | **0.6238** | — | — | — |
| **Gemma 4 31B (Zero-Shot)** | 0.4399 | 0.8148 | 0.5693 | **0.6938** | 0.6888 | 1,021.8 | 1,236.3 |
| **OpenAI GPT-5.4 (Zero-Shot Cloud)** | 0.5157 | 0.7704 | 0.6157 | **0.6991** | 0.7575 | *async batch* | *async batch* |
| **Claude Sonnet 4.6 (Zero-Shot Cloud)** | 0.5255 | 0.8498 | 0.6443 | **0.7511** | 0.7585 | *async batch* | *async batch* |
| **Gemma 4 31B (QLoRA V4)** | 0.6872 | 0.7757 | 0.7253 | **0.7537** | 0.8514 | 1,551.9 | 1,628.2 |
| **Gemma 4 31B (RAG-B, Qwen3 2+2)** | 0.6202 | 0.8491 | 0.7135 | **0.7878** | 0.8242 | 2,096.2 | 3,304.9 |
| **Gemma 4 31B (Combined LoRA + RAG)** | **0.7170** | 0.8212 | **0.7612** | **0.7947** | **0.8670** | 1,810.3 | 2,996.8 |

All values are macro-averaged over the 8 projects (clean scoring) and reproduce exactly from the committed prediction files.

### Main Takeaways:
1. **Open-weight vs. proprietary cloud (zero-shot, like-for-like):** local zero-shot Gemma 4 31B ($F_2 = 0.6938$) performs at the level of OpenAI GPT-5.4 ($F_2 = 0.6991$); Claude Sonnet 4.6 is the strongest zero-shot model ($F_2 = 0.7511$). The cloud models were evaluated **zero-shot only** — the adapted local results below are *not* a ranking against them, since applying the same adaptation to cloud models could be expected to improve them as well. The open-weight model is used for the adaptation study because it offers weight access for fine-tuning, no per-request cost at experiment scale, and full data locality.
2. **Adaptation pays:** RAG with balanced retrieved demonstrations lifts the open-weight model to $F_2 = 0.7878$ and the combined LoRA+RAG configuration to $F_2 = 0.7947$ — although the combined method's advantage over RAG alone is **not statistically robust** across the 8 projects (Wilcoxon $p = 1.0$).
3. **Latency vs. context trade-off:** RAG roughly doubles prompt length (~373 → ~1,488 tokens) and pushes P95 latency to ~3.3 s/pair, while LoRA V4 keeps a tight 1.6 s P95 with the strongest precision (0.687) — the better fit when review effort and latency dominate.

---

## ⚡ Zero-GPU Reproducibility

Because running 31B parameter LLM inference and QLoRA fine-tuning across 10,500+ pairs requires an **NVIDIA H100 (80GB VRAM)** or equivalent multi-GPU setup, **we have committed the exact prediction outputs for all 178 committed per-pair prediction files (the model-selection checkpoint predictions are excluded with the checkpoints directory) inside [`RESULTS/`](RESULTS/)**.

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
