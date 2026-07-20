# Evaluation Scripts and Execution Pipeline (`SCRIPT/`)

This directory contains the scripts used for baselines, local LLM inference, RAG, QLoRA fine-tuning, cloud batch evaluation, statistical checks, and figure generation.

Not every script is expected to run on an ordinary laptop. The CPU verification scripts can be run from the committed prediction files. The local LLM and QLoRA scripts require H100-class hardware or an equivalent multi-GPU environment; the cloud scripts require API access.

## Script Reference

### Baselines

- `1.VSM.py`: TF-IDF / cosine-similarity baseline with validation-threshold tuning.
- `2.SBERT.py`: sentence-transformer baseline using `all-mpnet-base-v2`.
- `3.BERT_FROZEN.py`: frozen `bert-base-uncased` feature extractor with a small MLP classifier.

### Local Open-Weight LLM Experiments

- `4.ModelSelection.py`: open-weight model screening.
- `4.ModelSelection-FINAL.ipynb`: notebook record of the final model-selection rerun.
- `5.zero_shot_h100.py`: zero-shot Gemma 4 31B evaluation on held-out hard-negative test pairs.
- `6.rag_rerun_stage1_8192.py`: RAG-A/B/C evaluation with retrieved labelled demonstrations.
- `6b.rag_stage2_hybrid_8192.py`: additional RAG-D hybrid retrieval evaluation.
- `7.lora_rerun_unified.py`: QLoRA ablation runs V1-V6. Rank, alpha, learning rate, upsampling, and target modules vary by version; the selected V4 configuration uses rank 32, alpha 64, learning rate 1e-4, 3x positive upsampling, and attention-only LoRA modules.
- `8.combined_RAG&LORA.py`: combined LoRA V4 + RAG-B inference.

### Cloud Batch Evaluation

- `9c_zero_shot_claude_batch_matched.py`: Claude Sonnet 4.6 zero-shot batch evaluation.
- `9d_zero_shot_openai_batch_matched.py`: GPT-5.4 zero-shot batch evaluation.
- `9e_merge_openai_batch_shards.py`: OpenAI shard merger and validation utility.

### Analysis and Figures

- `9.stratified_eval_by_link_type.py`: recomputes clean/conservative performance by hierarchy stratum.
- `10_significance_tests.py`: recomputes per-project clean F2, sign tests, Wilcoxon tests, and bootstrap confidence intervals.
- `11_generate_thesis_figures_v2.py`: regenerates the main result figures from saved JSON artifacts.

## CPU-Only Verification

Install the CPU verification dependencies:

```bash
pip install -r requirements-cpu.txt
```

Then run:

```bash
python SCRIPT/10_significance_tests.py
python SCRIPT/9.stratified_eval_by_link_type.py
python SCRIPT/11_generate_thesis_figures_v2.py
```

These commands operate on the committed files under `RESULTS/` and do not rerun LLM inference.

## Current Statistical Reference

The current committed `RESULTS/significance_tests_clean_f2.json` reports:

| Comparison | Mean delta | Median delta | Wins | Sign p | Wilcoxon p | Bootstrap 95% CI |
| :--- | ---: | ---: | :---: | ---: | ---: | :--- |
| RAG-B vs LoRA-V4 | 0.0341 | 0.0139 | 7/8 | 0.0703 | 0.1094 | [-0.0077, 0.0813] |
| Combined vs RAG-B | 0.0069 | 0.0037 | 5/8 | 0.7266 | 1.0000 | [-0.0102, 0.0309] |
| Combined vs LoRA-V4 | 0.0409 | 0.0210 | 8/8 | 0.0078 | 0.0078 | [0.0150, 0.0743] |
| RAG-B vs ZeroShot | 0.0940 | 0.0883 | 8/8 | 0.0078 | 0.0078 | [0.0595, 0.1289] |
| LoRA-V4 vs ZeroShot | 0.0600 | 0.0743 | 7/8 | 0.0703 | 0.0547 | [0.0080, 0.1058] |

With only eight projects, these tests should be read as robustness checks. Effect sizes and project-level win counts are at least as important as p-values.

## GPU/API Reruns

Example local zero-shot rerun:

```bash
CUDA_VISIBLE_DEVICES=0 python -u SCRIPT/5.zero_shot_h100.py --projects AAH BEAM CB FH JBIDE KEYCLOAK KOGITO PROJQUAY
```

The GPU scripts assume that model weights, CUDA libraries, and any required adapters/indexes are available locally or can be regenerated. They are included primarily for transparency and specialist reruns.
