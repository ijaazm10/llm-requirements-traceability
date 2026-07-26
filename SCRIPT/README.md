# Evaluation Scripts and Execution Pipeline (`SCRIPT/`)

This directory contains the scripts used for baselines, local LLM inference, RAG, QLoRA fine-tuning, cloud batch evaluation, statistical checks, and figure generation.

The scripts are retained as implementation artefacts. The current repository does not package the model weights, adapters, retrieval indexes, API credentials, compute environment, or complete dependency lock needed to rerun every experiment.

## Script-to-Result Map

| Script | High-level role | Primary input | Archived output |
| :--- | :--- | :--- | :--- |
| `1.VSM.py` | TF-IDF/cosine baseline with a validation-selected threshold | `DATA/.GROUND_TRUTH/*/splits/` | `RESULTS/vsm_final_pairs_results (1).json` |
| `2.SBERT.py` | `all-mpnet-base-v2` sentence-embedding baseline | `DATA/.GROUND_TRUTH/*/splits/` | `RESULTS/sbert_final_pairs_results (1).json` |
| `3.BERT_FROZEN.py` | Frozen `bert-base-uncased` representations with a trained MLP classifier | `DATA/.GROUND_TRUTH/*/splits/` | `RESULTS/frozen_bert_final_pairs_results (1).json` |
| `4.ModelSelection.py` | Initial open-weight model screening | Validation pairs for CB, KEYCLOAK, and JBIDE | `RESULTS/model_selection_v3_hard_results.json` |
| `4.ModelSelection-FINAL.ipynb` | Final all-4-bit model-selection record | The same three-project validation subset | `RESULTS/model_selection_v3_hard_results_FINAL (1).json` |
| `5.zero_shot_h100.py` | Gemma 4 31B pair classification without project-specific adaptation | Held-out test pairs | `RESULTS/ZERO_SHOT_QWEN_HARD/` |
| `6.rag_rerun_stage1_8192.py` | RAG-A/B/C ablation: MPNet or Qwen3 retrieval and balanced or positive-only demonstrations | Training pairs as demonstration pool; test pairs as queries | `RESULTS/RAG_STAGE1_V3_8192/` |
| `6b.rag_stage2_hybrid_8192.py` | RAG-D hybrid Qwen3+BM25 retrieval with reciprocal-rank fusion | Training and test pairs | `RESULTS/RAG_STAGE2_HYBRID_V3_8192/` |
| `7.lora_rerun_unified.py` | Project-specific QLoRA V1-V6 ablation | Training, validation, and test pairs | `RESULTS/LORA_RERUN_V3/` |
| `8.combined_RAG&LORA.py` | LoRA V4 adapter evaluated with RAG-B demonstrations | LoRA V4 artefacts, RAG-B retrieval pool, and test pairs | `RESULTS/COMBINED_RERUN_V3/` |
| `9c_zero_shot_claude_batch_matched.py` | Matched zero-shot Claude Sonnet 4.6 evaluation | Held-out test pairs | `RESULTS/CLAUDE_ZERO_SHOT_BATCH_V3/` |
| `9d_zero_shot_openai_batch_matched.py` | Matched zero-shot GPT-5.4 batch submission and collection | Held-out test pairs | Shards under `RESULTS/OPENAI_ZERO_SHOT_BATCH_V3/` |
| `9e_merge_openai_batch_shards.py` | Merges and validates the GPT-5.4 batch shards | OpenAI shard outputs | `RESULTS/OPENAI_ZERO_SHOT_BATCH_V3/gpt-5_4_merged_matched_single_user_prompt_v1_batch/` |
| `9.stratified_eval_by_link_type.py` | Separates results into refinement and subtask strata | Saved method predictions | `RESULTS/stratified_by_link_type_v3.json` |
| `10_significance_tests.py` | Project-level sign tests, Wilcoxon tests, and bootstrap intervals | Saved predictions for the four main local methods | `RESULTS/significance_tests_clean_f2.json` |
| `11_generate_thesis_figures_v2.py` | Generates the current thesis result figures | Consolidated result and statistical JSON files | `RESULTS/FIGURES_V2/` |

The shared `9` prefix does not indicate that these files are one combined stage. `9c`--`9e` implement the proprietary zero-shot batch workflow, while `9.stratified_eval_by_link_type.py` is a separate post-hoc analysis over saved predictions.

## Method Configuration Notes

- **RAG:** RAG-A uses MPNet with balanced demonstrations; RAG-B replaces MPNet with Qwen3 embeddings; RAG-C retrieves positive demonstrations only; RAG-D combines Qwen3 and BM25 rankings through reciprocal-rank fusion.
- **QLoRA:** V1-V6 vary class balancing, learning rate, LoRA rank/alpha, and target modules. The selected V4 configuration uses rank 32, alpha 64, learning rate `1e-4`, threefold positive upsampling, and attention-only target modules.
- **Combined:** the selected LoRA V4 adapter supplies supervised adaptation while the selected RAG-B configuration supplies project-specific in-context examples.
- **Cloud comparison:** GPT-5.4 and Claude Sonnet 4.6 use the same pair-classification task and matched zero-shot prompt contract as the local zero-shot comparison.

## Current Statistical Reference

The current committed `RESULTS/significance_tests_clean_f2.json` reports:

| Comparison | Mean delta | Median delta | Wins | Sign p | Wilcoxon p | Bootstrap 95% CI |
| :--- | ---: | ---: | :---: | ---: | ---: | :--- |
| RAG-B vs LoRA-V4 | 0.0341 | 0.0139 | 7/8 | 0.0703 | 0.1094 | [-0.0077, 0.0813] |
| Combined vs RAG-B | 0.0069 | 0.0037 | 5/8 | 0.7266 | 1.0000 | [-0.0102, 0.0309] |
| Combined vs LoRA-V4 | 0.0409 | 0.0210 | 8/8 | 0.0078 | 0.0078 | [0.0150, 0.0743] |
| RAG-B vs ZeroShot | 0.0940 | 0.0883 | 8/8 | 0.0078 | 0.0078 | [0.0595, 0.1289] |
| LoRA-V4 vs ZeroShot | 0.0600 | 0.0743 | 7/8 | 0.0703 | 0.0547 | [0.0080, 0.1058] |

With only eight projects, these tests are reported as project-level robustness checks. Effect sizes and project-level win counts are at least as important as the p-values.
