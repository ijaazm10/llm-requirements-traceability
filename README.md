# LLM Requirements Traceability Benchmark & Pipeline

This repository contains the code, processed benchmark data, saved predictions, and analysis artifacts for a master's thesis on large language models for cross-level requirements traceability in Jira issue hierarchies.

The study evaluates traceability link recovery as the second stage of a retrieve-then-classify pipeline. Instead of testing on random non-links, each positive hierarchy link is paired with semantically similar hard negatives mined with Qwen3 embeddings. This makes the classification task deliberately difficult: models must distinguish true refinement or decomposition links from issue pairs that look plausible under lexical or embedding similarity.

## Repository Scope

This is a transparent research artifact, not a fully raw-data-to-model reproduction package.

- The raw Jira exports are not redistributed because the original data was provided by Tian et al. and redistribution rights for the raw dumps were not cleared.
- The processed benchmark under `DATA/.GROUND_TRUTH/` is included.
- The exact saved predictions under `RESULTS/` are included, so the reported metrics, significance checks, link-type breakdowns, and figures can be recomputed without a GPU.
- Full LLM inference and QLoRA fine-tuning scripts are included for inspection and reruns, but require suitable hardware/API access.

## Repository Structure

```text
llm-requirements-traceability/
|-- DATA/       # Processed benchmark data, split files, and data-construction scripts
|-- SCRIPT/     # Baselines, LLM/RAG/LoRA evaluation scripts, statistics, and figure generation
|-- RESULTS/    # Saved predictions, result summaries, significance tests, and generated figures
|-- docs/       # README figures
|-- README.md
```

Detailed documentation:

- [DATA/README.md](DATA/README.md): processed benchmark structure, project-level counts, and raw-data policy.
- [SCRIPT/README.md](SCRIPT/README.md): script order, what is CPU-verifiable, and what requires GPU/API access.
- [RESULTS/README.md](RESULTS/README.md): saved prediction files, statistical JSONs, and figure artifacts.

## Benchmark Summary

The committed processed benchmark contains:

| Quantity | Count |
| :--- | ---: |
| Projects | 8 |
| Requirements | 11,680 |
| Positive trace links | 10,503 |
| Training pairs | 27,308 |
| Validation pairs | 4,036 |
| Held-out test pairs | 10,668 |
| Committed per-pair prediction files | 178 |

## Main Results

All values below are macro-averaged over the eight projects using clean scoring unless noted otherwise.

| Method | Precision | Recall | Macro F1 | Macro F2 | Accuracy | Mean latency (ms/pair) | P95 latency (ms/pair) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All-positive sanity check | 0.2500 | 1.0000 | 0.4000 | 0.6250 | 0.2500 | - | - |
| VSM / TF-IDF | 0.2984 | 0.6941 | 0.4139 | 0.5433 | - | - | - |
| SBERT / all-mpnet-base-v2 | 0.2528 | 0.9663 | 0.4006 | 0.6174 | - | - | - |
| Frozen BERT + MLP | 0.2758 | 0.9304 | 0.4219 | 0.6238 | - | - | - |
| Gemma 4 31B zero-shot | 0.4399 | 0.8148 | 0.5693 | 0.6938 | 0.6888 | 1,021.8 | 1,236.3 |
| OpenAI GPT-5.4 zero-shot | 0.5157 | 0.7704 | 0.6157 | 0.6991 | 0.7575 | async batch | async batch |
| Claude Sonnet 4.6 zero-shot | 0.5255 | 0.8498 | 0.6443 | 0.7511 | 0.7585 | async batch | async batch |
| Gemma 4 31B QLoRA V4 | 0.6872 | 0.7757 | 0.7253 | 0.7537 | 0.8514 | 1,551.9 | 1,628.2 |
| Gemma 4 31B RAG-B | 0.6202 | 0.8491 | 0.7135 | 0.7878 | 0.8242 | 2,096.2 | 3,304.9 |
| Gemma 4 31B LoRA + RAG | 0.7170 | 0.8212 | 0.7612 | 0.7947 | 0.8670 | 1,810.3 | 2,996.8 |

<p align="center">
  <img src="docs/figures/pr_scatter_f2_isocurves.png" alt="All evaluated methods in macro precision-recall space with F2 iso-curves" width="640">
</p>

<p align="center">
  <img src="docs/figures/per_project_f2_heatmap.png" alt="Per-project F2 scores for the main methods" width="720">
</p>

## Main Takeaways

1. Similarity-oriented baselines struggle under the hard-negative protocol. SBERT and Frozen BERT approach the all-positive sanity check but do not provide a strong discriminative signal.
2. Zero-shot LLMs provide a stronger second-stage traceability signal. Claude Sonnet 4.6 is the strongest zero-shot model among the evaluated cloud/local models.
3. Project-specific adaptation improves the local open-weight model. RAG-B reaches macro F2 = 0.7878, LoRA V4 reaches macro F2 = 0.7537, and the combined LoRA+RAG configuration reaches macro F2 = 0.7947.
4. The combined method has the highest absolute macro F2, but its advantage over RAG-B is not statistically robust across the eight projects (Wilcoxon p = 1.0000; bootstrap 95% CI [-0.0102, 0.0309]).

## Zero-GPU Verification

The expensive model outputs are committed as prediction JSON files. On a standard machine, you can recompute the statistical and figure artifacts without rerunning LLM inference:

```bash
python SCRIPT/10_significance_tests.py
python SCRIPT/9.stratified_eval_by_link_type.py
python SCRIPT/11_generate_thesis_figures_v2.py
```

For the CPU-only verification scripts, install:

```bash
pip install -r requirements-cpu.txt
```

Full local LLM inference/fine-tuning requires additional GPU dependencies and an H100-class setup or equivalent multi-GPU environment.

## License and Provenance

- Code, scripts, and committed prediction files are released under the MIT License.
- The processed benchmark in `DATA/.GROUND_TRUTH/` is derived from Jira exports of public open-source projects originally collected by Tian et al.
- Raw Jira exports are not included. `DATA/RAW_DATA/` is intentionally ignored.
