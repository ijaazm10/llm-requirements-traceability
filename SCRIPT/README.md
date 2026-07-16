# Evaluation Scripts & Execution Pipeline (`SCRIPT/`)

This directory contains the complete end-to-end Python pipeline for evaluating classic baselines, local open-weight large language models (`Gemma 4 31B`), retrieval-augmented generation (`RAG`), parameter-efficient fine-tuning (`QLoRA`), and commercial batch APIs (`Claude Sonnet 4.6`, `GPT-5.4`).

---

## 📜 Script Reference & Pipeline Order

The scripts are numbered chronologically by experimental progression (`1` through `11`):

### 1. Traditional & Neural Baselines
* **`1.VSM.py`** — Vector Space Model (TF-IDF cosine similarity) baseline. Builds vocabulary from combined summary/description, scores pairs, and tunes decision thresholds on validation $F_2$.
* **`2.SBERT.py`** — Sentence-Transformer baseline using `all-mpnet-base-v2` (`768-dim`, `384 max tokens`). Evaluates dense bi-encoder classification without LLM reasoning.
* **`3.BERT_FROZEN.py`** — Reproduction of the DRAFT text-only ablation baseline (*Tian et al. 2023*). Extracts 4 cosine similarity combinations from frozen `bert-base-uncased` (`64 token limit`) and trains a 4-layer MLP (`4 -> 64 -> 32 -> 2`) with weighted cross-entropy.

### 2. Local Open-Weight LLM Evaluation (`NVIDIA H100 80GB`)
* **`5.zero_shot_h100.py`** — Zero-Shot evaluation of `unsloth/gemma-4-31B-it` (4-bit NF4 quantized) across all 8 projects (`final_pairs_test.json`). Logs single-stream inference latency (`ms/pair`), token throughput (`tok/s`), VRAM allocation, and truncation counts (`MAX_SEQ_LENGTH = 3072`).
* **`6.rag_rerun_stage1_8192.py`** — Stage 1 RAG evaluation (`RAG-B`). Uses `Qwen3-Embedding-4B` over local FAISS/BM25 indexes to dynamically retrieve labeled valid/invalid demonstration pairs from training folds. Extends input window to `8,192 tokens` and measures context processing latency (`Prefill ms` vs `Generation ms`).
* **`6b.rag_stage2_hybrid_8192.py`** — Stage 2 Hybrid RAG ablation (`RAG-C`, `RAG-D`). Evaluates whether combining structural graph diffusion context with flat-text demonstration retrieval improves classification precision.
* **`7.lora_rerun_unified.py`** — Parameter-Efficient Fine-Tuning (`QLoRA`) across checkpoints `V1` through `V6`. Trains `Gemma 4 31B` using low-rank adapters (`R=32`, `alpha=32`, `LR=1e-4`) on 31,500+ hard-negative samples (`seed 42`). Evaluates champion model `V4_EFFICIENCY`.
* **`8.combined_RAG&LORA.py`** — Combined Model Evaluation (`LoRA V4 + RAG-B`). Tests whether fine-tuned internal task knowledge and dynamic external demonstration retrieval provide complementary or redundant classification signal.

### 3. Commercial Cloud API Benchmarking
* **`9c_zero_shot_claude_batch_matched.py`** — Evaluates **`Claude Sonnet 4.6` (`claude-sonnet-4-6`)** using the Anthropic Message Batches API. Enforces character-for-character prompt parity with local `5.zero_shot_h100.py` and deterministic generation (`temperature = 0.0`). Includes automatic budget guards and token counting.
* **`9d_zero_shot_openai_batch_matched.py`** — Evaluates **`OpenAI GPT-5.4` (`gpt-5.4`)** using the OpenAI Batch API. Enforces matched prompts, generation caps (`max_completion_tokens = 20`), and `temperature = 0.0`.
* **`9e_merge_openai_batch_shards.py`** — Shard merger and order-validator for OpenAI batch processing. Resolves account-level enqueued-token limits by splitting large project evaluations into chunks and joining them cleanly by `custom_id`.

### 4. Analysis, Statistical Testing & Figure Generation
* **`9.stratified_eval_by_link_type.py`** — Post-hoc stratified evaluation of predictions by Jira issue hierarchy (`Epic -> Standard` refinement links vs `Standard -> Subtask` decomposition links).
* **`10_significance_tests.py`** — Statistical robustness and significance testing (`clean F2`). Computes Sign tests (`two-sided binomial p`), Paired Wilcoxon signed-rank tests, and 10,000-iteration Bootstrap 95% Confidence Intervals across the 8 project evaluations.
* **`11_generate_thesis_figures_v2.py`** — Master plotting script. Reads prediction logs and summary JSONs to generate publication-grade vector PDF figures inside `RESULTS/FIGURES_V2/`.

---

## 🚀 Execution & Usage Guide

### Prerequisites
* **Python 3.10+** with `torch`, `transformers`, `unsloth`, `sentence_transformers`, `scikit-learn`, `scipy`, `matplotlib`, and `seaborn`.
* For local 31B LLM inference or fine-tuning (`Scripts 5, 6, 7, 8`), an **NVIDIA H100 (80GB)** or multi-GPU CUDA environment is required.
* For statistical testing and chart generation (`Scripts 1, 2, 3, 9, 10, 11`), only CPU is required (`< 4GB RAM`).

### Quickstart Example: Offline Statistical & Figure Verification
Because all prediction JSONs are saved in `RESULTS/`, you can verify all statistical claims immediately without running inference:

```bash
# 1. Run 10,000-bootstrap significance testing across the 8 projects
python SCRIPT/10_significance_tests.py

# Output preview:
# Paired comparisons over 8 projects (clean F2, delta = A - B):
#   RAG-B vs LoRA-V4         +0.0341   +0.0345     6/8   0.2891    0.0781   [-0.0123, +0.0812]
#   Combined vs RAG-B        +0.0068   +0.0071     5/8   0.7266    0.3828   [-0.0154, +0.0298]
# Saved -> RESULTS/significance_tests_clean_f2.json

# 2. Re-generate vector PDF plots for RQ1, RQ2, and RQ3
python SCRIPT/11_generate_thesis_figures_v2.py
```

### Running Local H100 Inference (Example: Zero-Shot Gemma 4)
```bash
export CUDA_VISIBLE_DEVICES=0
python -u SCRIPT/5.zero_shot_h100.py --projects AAH BEAM CB FH JBIDE KEYCLOAK KOGITO PROJQUAY
```
