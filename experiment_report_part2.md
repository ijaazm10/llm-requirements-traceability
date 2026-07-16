# Experiment Report - Requirements Traceability with LLMs
## Part 2: Methods, Final Experiments, and Results

---

## 6. Evaluation Setup

### 6.1 Hardware and Execution Environments

| Resource | Specification / Use |
|---|---|
| GPU | NVIDIA H100 NVL, 93 GiB VRAM |
| Local LLM model | Gemma 4 31B instruction model, loaded through Unsloth |
| Model selection serving | Ollama API at `https://ymir-api.ifak.eu` |
| Dense retrieval embedding service | Qwen3 embedding model via Ollama API |
| Cloud models | Claude Sonnet 4.6 through Anthropic Message Batches; GPT-5.4 through OpenAI Batch API |
| Input data | `DATA/GROUND_TRUTH/{PROJECT}/splits/final_pairs_test.json` |
| Output data | `RESULTS/` numbered files `1` through `9d` |

All final experiments use the same held-out hard-negative test set constructed in Part 1. The final test set contains **10,668 pairs** across eight projects, with a strict 1:3 positive/negative ratio.

### 6.2 Metrics

The primary metric is **macro F2**, computed as the unweighted mean of project-level F2 scores across the eight projects:

```text
F2 = (5 * precision * recall) / (4 * precision + recall)
```

F2 is used because recall is more important than precision in traceability recovery: a missed true link can break impact analysis, compliance checking, or downstream change reasoning, whereas a false positive can still be reviewed by an engineer.

The report also records precision, recall, F1, and accuracy where applicable. Generative methods additionally record parse failures and, where available, truncation rates and timing metrics.

Two scoring variants are used for generative methods:

| Variant | Definition | Purpose |
|---|---|---|
| Clean | Metrics computed on parseable outputs | Primary performance comparison |
| Conservative | Parse failures treated as negative predictions | Reliability lower bound |

For the final cloud runs, parse failures and API failures were zero, so clean and conservative scores are identical.

---

## 7. Non-LLM Baselines

Three non-LLM baselines establish how far lexical, embedding, and shallow learned approaches can go on the hard-negative protocol.

### 7.1 VSM / TF-IDF Baseline

**Script:** `SCRIPT/1.VSM.py`  
**Result file:** `RESULTS/1.vsm_final_pairs_results.json`

The Vector Space Model baseline represents each requirement pair using TF-IDF features and cosine similarity. A project-specific validation threshold is tuned, then applied to `final_pairs_test.json`.

This baseline tests whether lexical overlap is sufficient for cross-level traceability once negatives are mined to be semantically close to the source.

**Final macro result:**

| Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|
| 0.3807 | 0.4835 | 0.4087 | 0.4460 |

### 7.2 SBERT / MPNet Baseline

**Script:** `SCRIPT/2.SBERT.py`  
**Result file:** `RESULTS/2.sbert_final_pairs_results.json`

The SBERT baseline embeds requirements using a sentence-transformer model and applies cosine similarity with a tuned threshold. It tests whether pretrained dense semantic similarity is more robust than sparse lexical similarity.

**Final macro result:**

| Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|
| 0.2731 | 0.9016 | 0.4116 | 0.6045 |

The high recall and low precision indicate that dense semantic similarity retrieves many true links, but also over-predicts links under hard-negative conditions.

### 7.3 Frozen BERT Classifier

**Script:** `SCRIPT/3.BERT_FROZEN.py`  
**Result file:** `RESULTS/3.frozen_bert_final_pairs_results.json`

This method uses `bert-base-uncased` as a frozen encoder and trains only a small classification head. It is a lightweight learned baseline: the encoder is not fine-tuned, so the model can learn a decision boundary over BERT representations without adapting the language model itself.

**Final macro result:**

| Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|
| 0.2965 | 0.8058 | 0.4265 | 0.5882 |

Frozen BERT improves recall relative to VSM but remains precision-limited, showing that a shallow classifier over fixed representations is not enough to separate true links from Qwen3-mined hard negatives.

---

## 8. Open-Weight Model Selection

**Script:** `SCRIPT/4.ModelSelection.py`  
**Result file:** `RESULTS/4.model_selection_v3_hard_results.json`

Before running full LLM experiments, nine open-weight models were compared on a validation subset drawn from three representative projects. The selection protocol used up to 400 validation pairs per project, deterministic zero-shot prompting, and macro F2 as the primary ranking metric.

### 8.1 Candidate Models

| Size tier | Models |
|---|---|
| 7-8B | `llama3.1:8b`, `qwen3:8b` |
| 12-14B | `gemma3:12b`, `phi4:14b`, `qwen2.5-coder:14b-instruct-fp16`, `ministral-3:14b` |
| 27-31B | `gemma3:27b`, `qwen3-coder:30b`, `gemma4:31b` |

### 8.2 Top Five Selection Results

| Rank | Model | Macro P | Macro R | Macro F1 | Macro F2 | Failed % | Time / pair |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `gemma4:31b` | 0.4858 | 0.7767 | 0.5915 | 0.6871 | 0.1 | 0.88s |
| 2 | `gemma3:12b` | 0.3415 | 0.7700 | 0.4721 | 0.6140 | 0.0 | 0.74s |
| 3 | `llama3.1:8b` | 0.3518 | 0.7267 | 0.4734 | 0.5982 | 0.0 | 0.46s |
| 4 | `gemma3:27b` | 0.3131 | 0.7667 | 0.4416 | 0.5898 | 0.0 | 0.86s |
| 5 | `qwen3:8b` | 0.3932 | 0.6367 | 0.4803 | 0.5604 | 0.0 | 0.45s |

**Selected open-weight model:** `gemma4:31b`

All subsequent local LLM experiments use Gemma 4 31B.

---

## 9. Zero-Shot LLM Evaluation

### 9.1 Local Gemma 4 31B Zero-Shot

**Script:** `SCRIPT/5.zero_shot_h100.py`  
**Result file:** `RESULTS/5.zero_shot_qwen_hard_results.json`

Gemma 4 31B is prompted with a single HLR/LLR pair and no examples. The model returns only:

```json
{"is_linked": true}
```

or

```json
{"is_linked": false}
```

The prompt contains the traceability task definition, the HLR summary and description, the LLR summary and description, and the JSON-only output instruction.

**Generation setup:** greedy decoding, `MAX_NEW_TOKENS=20`, seed 42, H100 inference.

**Final macro result:**

| Precision | Recall | F1 | F2 | Accuracy |
|---:|---:|---:|---:|---:|
| 0.4399 | 0.8148 | 0.5693 | 0.6938 | 0.6888 |

This is the local zero-shot LLM baseline against which RAG, LoRA, and combined adaptation are compared.

### 9.2 Claude Sonnet 4.6 Zero-Shot Batch

**Script:** `SCRIPT/9c_zero_shot_claude_batch_matched.py`  
**Result file:** `RESULTS/9c.claude-sonnet-4-6_results.json`

Claude Sonnet 4.6 was evaluated using the Anthropic Message Batches API. The cloud run was designed as a matched-prompt comparison against Gemma:

| Factor | Setting |
|---|---|
| Prompt content | Same semantic zero-shot prompt as Gemma |
| Message format | Single user message, no system message |
| Tools / JSON mode | None |
| Max output | 20 tokens |
| Temperature | 0.0 |
| Test pairs | 10,668 |
| API failures | 0 |
| Parse failures | 0 |
| Prompt hash | `209a9beb289a5172` |

Batch mode changes only the execution mechanism. It is valid for classification quality, but the asynchronous batch wall-clock time is not used for single-stream latency comparisons.

**Final macro result:**

| Precision | Recall | F1 | F2 | Accuracy |
|---:|---:|---:|---:|---:|
| 0.5255 | 0.8498 | 0.6443 | 0.7511 | 0.7585 |

Claude is the strongest zero-shot model in this final comparison.

### 9.3 GPT-5.4 Zero-Shot Batch

**Scripts:** `SCRIPT/9d_zero_shot_openai_batch_matched.py`, `SCRIPT/9e_merge_openai_batch_shards.py`  
**Result file:** `RESULTS/9d.openai_gpt54_results.json`

GPT-5.4 was evaluated through the OpenAI Batch API using the same matched prompt. The first full-size batch could not be submitted as a single job because the OpenAI organization had a 900,000 enqueued-token limit for the model. Therefore, the run was split into 18 shards, each submitted and collected separately.

The merge script then:

1. Auto-discovers the shard folders.
2. Loads all project prediction JSON files.
3. Concatenates shard rows in shard-name order.
4. Validates the merged sequence against `final_pairs_test.json`.
5. Refuses silent deduplication.
6. Recomputes project and macro metrics.

**Final merge integrity:**

| Check | Result |
|---|---:|
| Total expected pairs | 10,668 |
| Total merged pairs | 10,668 |
| Missing pairs | 0 |
| API failures | 0 |
| Parse failures | 0 |
| Prompt hash | `209a9beb289a5172` |

**Final macro result:**

| Precision | Recall | F1 | F2 | Accuracy |
|---:|---:|---:|---:|---:|
| 0.5157 | 0.7704 | 0.6157 | 0.6991 | 0.7575 |

GPT-5.4 improves over local Gemma zero-shot but remains below Claude Sonnet 4.6 and below all adapted Gemma methods.

### 9.4 Zero-Shot Summary

| Method | Precision | Recall | F1 | F2 | Accuracy | Parse/API failures |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 31B zero-shot | 0.4399 | 0.8148 | 0.5693 | 0.6938 | 0.6888 | Reported in result file |
| GPT-5.4 zero-shot batch | 0.5157 | 0.7704 | 0.6157 | 0.6991 | 0.7575 | 0 |
| Claude Sonnet 4.6 zero-shot batch | 0.5255 | 0.8498 | 0.6443 | 0.7511 | 0.7585 | 0 |

---

## 10. Retrieval-Augmented Generation

**Scripts:** `SCRIPT/6.rag_rerun_stage1_8192.py`, `SCRIPT/6b.rag_stage2_hybrid_8192.py`  
**Result file:** `RESULTS/6.RAG_ABCD_COMPARISON.json`

RAG adds dynamically retrieved training examples to the prompt. Each test pair is first used to retrieve similar labelled training pairs, then the LLM classifies the target pair after seeing those examples.

### 10.1 Shared RAG Setup

| Component | Setting |
|---|---|
| Base model | `unsloth/gemma-4-31B-it` |
| Inference context length | 8,192 tokens |
| Max new tokens | 20 |
| Decoding | Greedy |
| Retrieval corpus | `final_pairs_train.json` |
| Query text | Full HLR + LLR pair text |
| Demonstration format | HLR, LLR, and known label |
| Timing | CUDA synchronization around generation; first 10 pairs excluded from steady-state latency aggregation |

The retrieval query uses the full pair text. This is intentional: RAG here retrieves demonstration examples for pairwise classification, not candidate targets for a production Stage-1 retriever.

### 10.2 Final RAG Configurations

Only four RAG configurations belong to the final report.

| Config | Retriever | Demonstrations | Purpose |
|---|---|---:|---|
| RAG-A | MPNet dense | 2 positive + 2 negative | Dense retriever comparison |
| RAG-B | Qwen3 dense | 2 positive + 2 negative | Main dense retriever anchor |
| RAG-C | Qwen3 dense | 4 positive + 0 negative | Positive-only composition test at four demos |
| RAG-D | Hybrid Qwen3 + BM25 via RRF | 2 positive + 2 negative | Lexical+dense hybrid test |

### 10.3 RAG Results

| Config | Precision | Recall | F1 | F2 | Accuracy | Failed parses |
|---|---:|---:|---:|---:|---:|---:|
| RAG-A MPNet 2+2 | 0.5910 | 0.8602 | 0.6965 | 0.7846 | 0.8062 | 157 |
| RAG-B Qwen3 2+2 | **0.6202** | 0.8491 | **0.7135** | **0.7878** | **0.8242** | 206 |
| RAG-C Qwen3 4+0 | 0.5007 | **0.9053** | 0.6410 | 0.7751 | 0.7416 | 377 |
| RAG-D Hybrid 2+2 | 0.6063 | 0.8241 | 0.6959 | 0.7664 | 0.8164 | 272 |

**RAG winner:** RAG-B Qwen3 2+2.

RAG-C achieves the highest recall but loses too much precision. RAG-D shows that adding BM25 through reciprocal-rank fusion does not improve the final classification score and increases retrieval latency substantially.

### 10.4 RAG Deployment Metrics

| Config | ms / pair | Retrieval ms | Generation ms | tok/s | Fail % | Trunc % | Load GB | Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG-A MPNet 2+2 | 2047.4 | 1.04 | 2036.1 | 6.0 | 0.85 | 0.88 | 18.22 | 21.49 |
| RAG-B Qwen3 2+2 | 2058.8 | 2.96 | 2044.8 | 6.0 | 1.21 | 1.21 | 18.22 | 21.16 |
| RAG-C Qwen3 4+0 | 2073.8 | 1.64 | 2060.9 | 6.0 | 2.82 | 2.85 | 18.22 | 21.18 |
| RAG-D Hybrid 2+2 | 2185.7 | 134.11 | 2040.0 | 6.0 | 1.53 | 1.56 | 17.78 | 20.66 |

---

## 11. LoRA Fine-Tuning

**Script:** `SCRIPT/7.lora_rerun_unified.py`  
**Result file:** `RESULTS/7.LORA_MASTER_COMPARISON.json`

LoRA fine-tunes lightweight adapter weights on top of the frozen Gemma 4 31B model. A separate adapter is trained for each project and configuration, then evaluated on that project's `final_pairs_test.json`.

### 11.1 Shared LoRA Setup

| Setting | Value |
|---|---|
| Base model | `unsloth/gemma-4-31B-it` |
| Quantisation | 4-bit |
| Max sequence length | 3,072 tokens |
| Max new tokens at evaluation | 20 |
| Epochs | Up to 5 |
| Batch setup | Per-device batch 2, gradient accumulation 8, effective batch 16 |
| Scheduler | Cosine |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Evaluation interval | 100 steps |
| Early stopping patience | 5 |
| LoRA dropout | 0.05 |
| Loss masking | Loss computed only on assistant response |
| Seed | 42 |

### 11.2 LoRA Ablation Versions

| Version | Learning rate | Rank | Alpha | Upsampling | Target modules | Purpose |
|---|---:|---:|---:|---:|---|---|
| V1_NAIVE | 1e-4 | 64 | 128 | 1x | Attention | Natural 1:3 imbalance |
| V2_BALANCED | 1e-4 | 64 | 128 | 3x | Attention | Balanced training baseline |
| V3_STABILIZED | 5e-5 | 64 | 128 | 3x | Attention | Lower learning rate |
| V4_EFFICIENCY | 1e-4 | 32 | 64 | 3x | Attention | Lower adapter rank |
| V5_SYNTHESIS | 5e-5 | 32 | 64 | 3x | Attention | Lower rank + lower learning rate |
| V6_MLP | 5e-5 | 32 | 64 | 3x | Attention + MLP | Adds `gate_proj`, `up_proj`, `down_proj` |

Attention modules are `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

### 11.3 LoRA Results

| Version | Precision | Recall | F1 | F2 | Accuracy | Failed parses |
|---|---:|---:|---:|---:|---:|---:|
| V1_NAIVE | **0.7152** | 0.6312 | 0.6338 | 0.6234 | 0.8264 | 27 |
| V2_BALANCED | 0.6805 | 0.7662 | 0.7088 | 0.7395 | 0.8440 | 27 |
| V3_STABILIZED | 0.6581 | **0.7812** | 0.7094 | 0.7491 | 0.8368 | 27 |
| V4_EFFICIENCY | 0.6872 | 0.7757 | **0.7253** | **0.7537** | **0.8514** | 27 |
| V5_SYNTHESIS | 0.6766 | 0.7710 | 0.7143 | 0.7452 | 0.8417 | 27 |
| V6_MLP | 0.6901 | 0.7625 | 0.7198 | 0.7433 | 0.8512 | 27 |

**LoRA winner:** V4_EFFICIENCY.

The strongest LoRA result comes from reducing adapter rank from 64 to 32 while keeping the higher learning rate. MLP adaptation does not improve F2 over the attention-only V4 setting.

---

## 12. Combined RAG + LoRA

**Script:** `SCRIPT/8.combined_RAG&LORA.py`  
**Result file:** `RESULTS/8.MASTER_COMBINED__RAG&LORA.json`

The combined experiment applies the best RAG configuration to the best LoRA-adapted model:

| Component | Selected configuration |
|---|---|
| LoRA | V4_EFFICIENCY: attention-only, LR=1e-4, rank=32, alpha=64 |
| RAG | RAG-B: Qwen3 dense retriever, 2 positive + 2 negative demonstrations |
| Context length | 8,192 tokens |

The experiment tests whether parametric adaptation (LoRA) and non-parametric in-context adaptation (RAG) are complementary.

### 12.1 Combined Results

| Method | Precision | Recall | F1 | F2 | Accuracy |
|---|---:|---:|---:|---:|---:|
| LoRA V4 alone | 0.6872 | 0.7757 | 0.7253 | 0.7537 | 0.8514 |
| RAG-B alone | 0.6202 | **0.8491** | 0.7135 | 0.7878 | 0.8242 |
| LoRA V4 + RAG-B | **0.7170** | 0.8212 | **0.7612** | **0.7947** | **0.8670** |

**Overall winner:** LoRA V4 + RAG-B.

The combined model improves precision substantially over standalone RAG while retaining much of RAG's recall. It also improves recall and F2 over standalone LoRA. This supports the complementarity hypothesis: LoRA improves the model's task-specific decision boundary, while RAG supplies local labelled examples at inference time.

### 12.2 Combined Deployment Metrics

| Metric | Value |
|---|---:|
| Time / pair | 1766.9 ms |
| Retrieval time / pair | 3.19 ms |
| Generation time / pair | 1752.2 ms |
| Format failure rate | 1.19% |
| Truncation rate | 1.21% |
| Load VRAM | 18.17 GB |
| Peak VRAM | 21.12 GB |
| Failed parses | 201 / 10,668 |

---

## 13. Final Cross-Method Result Table

| Method | Precision | Recall | F1 | F2 | Accuracy | Main result file |
|---|---:|---:|---:|---:|---:|---|
| VSM / TF-IDF | 0.3807 | 0.4835 | 0.4087 | 0.4460 | - | `1.vsm_final_pairs_results.json` |
| SBERT / MPNet | 0.2731 | 0.9016 | 0.4116 | 0.6045 | - | `2.sbert_final_pairs_results.json` |
| Frozen BERT | 0.2965 | 0.8058 | 0.4265 | 0.5882 | - | `3.frozen_bert_final_pairs_results.json` |
| Gemma 4 zero-shot | 0.4399 | 0.8148 | 0.5693 | 0.6938 | 0.6888 | `5.zero_shot_qwen_hard_results.json` |
| GPT-5.4 zero-shot batch | 0.5157 | 0.7704 | 0.6157 | 0.6991 | 0.7575 | `9d.openai_gpt54_results.json` |
| Claude Sonnet 4.6 zero-shot batch | 0.5255 | 0.8498 | 0.6443 | 0.7511 | 0.7585 | `9c.claude-sonnet-4-6_results.json` |
| RAG-B Qwen3 2+2 | 0.6202 | 0.8491 | 0.7135 | 0.7878 | 0.8242 | `6.RAG_ABCD_COMPARISON.json` |
| LoRA V4 | 0.6872 | 0.7757 | 0.7253 | 0.7537 | 0.8514 | `7.LORA_MASTER_COMPARISON.json` |
| Combined LoRA V4 + RAG-B | **0.7170** | 0.8212 | **0.7612** | **0.7947** | **0.8670** | `8.MASTER_COMBINED__RAG&LORA.json` |

---

## 14. Research Question Mapping

| Research question | Evidence from final experiments |
|---|---|
| RQ1: Zero-shot open-weight LLM vs classical/neural baselines | Gemma 4 zero-shot outperforms VSM, SBERT, and Frozen BERT on F2. |
| RQ2: Do RAG and LoRA improve beyond zero-shot? | RAG-B improves F2 from 0.6938 to 0.7878; LoRA V4 improves to 0.7537. |
| RQ2.1: Which adaptation strategy performs best? | RAG-B beats LoRA V4 on F2, but LoRA V4 has higher precision and accuracy. |
| RQ2.2: Which design choices matter? | Qwen3 2+2 is the best RAG design; LoRA rank 32 with attention-only modules is the best LoRA design. |
| RQ2.3: Are RAG and LoRA complementary? | Combined LoRA V4 + RAG-B is the overall best method, F2 = 0.7947. |
| RQ3.1: Inference latency by stage | RAG and combined results include retrieval and generation timing; cloud batch latency is excluded from single-stream latency comparisons. |
| RQ3.2: One-time preparation cost | RAG records index build and embedding preparation; LoRA requires per-project adapter training. |
| RQ3.3: Output reliability | Parse failures, truncation rates, and clean/conservative deltas are recorded for generative methods. |

---

## 15. Final Interpretation

The final result pattern is coherent:

1. Similarity-only methods struggle under Qwen3 hard-negative mining. SBERT has very high recall but poor precision.
2. Zero-shot LLMs are stronger than non-LLM baselines, especially Claude Sonnet 4.6.
3. RAG is the strongest standalone adaptation by F2 because retrieved demonstrations increase recall while maintaining usable precision.
4. LoRA is more precision-oriented and produces the highest standalone accuracy among adapted local methods.
5. The combined method is best overall, indicating that RAG and LoRA are complementary rather than redundant.

The final method recommendation is therefore:

```text
Use Qwen3 hard-negative mining for evaluation,
Gemma 4 31B as the open-weight base model,
RAG-B for the strongest standalone adaptation,
LoRA V4 when precision and compact deployment are priorities,
and LoRA V4 + RAG-B for the best overall F2.
```

