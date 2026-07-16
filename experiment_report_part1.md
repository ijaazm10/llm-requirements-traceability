> **SUPERSEDED (historical working notes).** This document describes an intermediate V2->V3 state of the pipeline and contains outdated figures (e.g., 551 both-empty issues, RAG configurations A-F). The authoritative record is the thesis and the current scripts/results in this repository. Kept for provenance only.

# Experiment Report — Requirements Traceability with LLMs
## Part 1: Task, Dataset, and Ground Truth Construction

---

## 1. Research Task

### 1.1 What We Are Trying to Solve

Requirements traceability is the practice of establishing and maintaining links between requirements at different levels of abstraction in a software project. Specifically, this work addresses **cross-level requirements traceability link recovery**: given a High-Level Requirement (HLR) and a Low-Level Requirement (LLR), determine whether a traceability link exists between them — that is, whether the LLR *implements*, *refines*, or *decomposes* the HLR.

This is cast as a **binary classification** task:

```
Input:  (HLR text, LLR text)
Output: 1 if a link exists, 0 if no link exists
```

In industrial Jira-based projects, this corresponds to the hierarchy:

| Level | Jira Issue Type | Role |
|-------|----------------|------|
| High-Level (parent) | Epic | System-level or feature-level requirement |
| Low-Level (standard) | Story, Task, Feature, Bug, Enhancement, Improvement | Implementation-level work item |
| Child (sub-task) | Sub-task | Fine-grained decomposition of a standard issue |

Two link types are considered:
- **Refinement**: Epic → Story/Task (parent decomposes to standard)
- **Subtask**: Story/Task → Sub-task (standard decomposes to child)

Same-level links, transitive links (Epic → Sub-task skipping Standard), and links involving unknown issue types are excluded by design.

### 1.2 Why This Is Hard

The dataset is naturally **class-imbalanced**: for every genuine link, many plausible non-links exist (a source requirement can have hundreds of possible target candidates but only 1–5 actual links). A naive classifier that predicts "no link" for everything achieves very high accuracy but is useless in practice. The primary evaluation metric is therefore **F2 score**, which weights recall twice as heavily as precision — in a traceability assistant, missing a true link (false negative) is more costly than flagging a false positive.

A further complication is that in this dataset, **hard negatives can be semantically more similar to the source requirement than true positives**, because the Jira issues within a single project share vocabulary, domain terms, and structural patterns. This was confirmed empirically (see Section 3.5).

---

### 1.3 Research Questions

The final experimental design is organized around three main research questions and supporting sub-questions.

**RQ1 — Zero-shot capability against non-LLM baselines**  
How does the zero-shot capability of a selected open-weight LLM compare against classical, embedding-based, and learned-classifier baselines for cross-level requirements traceability under a hard-negative evaluation protocol?

This question establishes whether a modern open-weight LLM can solve the traceability decision task without task-specific examples or fine-tuning, and whether it improves over traditional similarity-based and learned non-generative baselines. The selected open-weight model is chosen through the model-selection experiment described in Part 2.

**RQ1.1 — Open-weight versus proprietary cloud LLMs**  
How does the selected open-weight LLM compare with cost-tier-matched proprietary cloud LLMs from different vendors under the same zero-shot hard-negative traceability protocol?

This sub-question evaluates whether the local open-weight model is competitive with proprietary cloud models when all are given the same semantic task prompt and evaluated on the same hard-negative test pairs. The cloud comparison is restricted to zero-shot inference because LoRA-style weight adaptation is not available for closed proprietary models in the same way.

**RQ2 — Adaptation beyond zero-shot performance**  
To what extent do Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning (LoRA) improve LLM-based trace link recovery beyond zero-shot performance?

This question evaluates two adaptation strategies for the selected open-weight LLM: RAG, which supplies dynamically retrieved examples at inference time, and LoRA, which modifies a small number of adapter parameters through supervised fine-tuning.

**RQ2.1 — Best adaptation strategy**  
Which adaptation strategy achieves superior classification performance in terms of Precision, Recall, F1, and the recall-weighted F2?

**RQ2.2 — Important design choices**  
Which design choices within RAG and LoRA materially affect classification performance, and which configurations achieve the highest F2?

For RAG, the design choices include retriever type and demonstration composition. For LoRA, the design choices include rank, learning rate, class-balance handling, and target modules.

**RQ2.3 — Combined adaptation**  
Are RAG and LoRA complementary, redundant, or antagonistic when combined?

This question tests whether adding retrieved examples to the best LoRA-adapted model improves performance beyond either method alone.

**RQ3 — Computational profile under matched local conditions**  
Under matched local experimental conditions, how do the evaluated methods differ in computational profile?

This question focuses on local single-stream inference and preparation costs. Cloud batch runs are reported separately as zero-shot comparison experiments and are not treated as latency-comparable with local H100 single-stream runs.

**RQ3.1 — Inference latency**  
How do local methods differ in single-stream inference time per requirement pair, broken down by stage such as retrieval, prompt construction, generation, and parsing?

**RQ3.2 — Preparation cost**  
What one-time preparation cost is required by each method, and how does this cost scale with the number of projects?

**RQ3.3 — Output reliability**  
How reliably does each method produce parseable output under identical generation settings?

## 2. Dataset Origin and Provenance

### 2.1 Source Paper

The raw data originates from the paper:

> **"A Cross-Level Requirement Trace Link Update Model Based on Bidirectional Encoder Representations from Transformers"**  
> Jiahao Tian, Li Zhang, Xiaoli Lian  
> (IEEE/ACM, Software Engineering domain)

The paper describes a BERT-based model for maintaining traceability links as requirements evolve. It reports experiments on eight real-world open-source software projects whose issue data was exported from Jira.

### 2.2 Data Acquisition

The paper contained neither a public repository nor a data release. No preprocessing code was described. We contacted the authors directly by email and they provided the **raw Jira export dumps** for all eight projects.

What we received:
- Raw JSON files per project: issue records and link records
- No processing scripts, no pipeline, no documentation of how the paper's authors converted this into their own dataset
- No mention of how empty requirements were handled, how hierarchical levels were determined, or how splits were created

We therefore **designed, implemented, and documented the entire preprocessing and ground truth construction pipeline from scratch**, described in Section 3 below from the received RAW Jira export Dumps.

### 2.3 Projects

| ID | Project | Domain |
|----|---------|--------|
| AAH | AAH | Healthcare IT |
| BEAM | Apache BEAM | Distributed data processing |
| CB | Couchbase | Distributed database |
| FH | Fuse/Hawkular | Middleware / monitoring |
| JBIDE | JBoss IDE | Java IDE tooling |
| KEYCLOAK | Keycloak | Identity and access management |
| KOGITO | Kogito | Business automation / rules engine |
| PROJQUAY | Project Quay | Container registry |

---

## 3. Ground Truth Construction Pipeline (V3)

### 3.1 Overview

The pipeline has two stages, each implemented as a standalone Python script:

```
RAW_DATA/
  {PROJECT}_issue_denoised.json   ← raw Jira issue records
  {PROJECT}_link.json             ← raw Jira link records
        │
        ▼
  Stage 1: 01_construct_ground_truth_v3_text_clean.py
        │   - Text filter (remove both-empty issues)
        │   - Level classification (parent / standard / child)
        │   - Link type filter (refinement / subtask only)
        │   - Source-stratified 70/10/20 split
        │   → requirements.json, trace_links.json, splits/{train,val,test}_links.json
        │
        ▼
  Stage 2: 02_mine_qwen3_diverse_hard_negatives.py
        │   - Embed all requirements with Qwen3-Embedding-4B
        │   - For each source: rank all non-linked same-level candidates by cosine similarity
        │   - Assign disjoint windows of top-K negatives per positive (1:3 ratio)
        │   → splits/final_pairs_{train,val,test}.json
```

### 3.2 Stage 1: Ground Truth Constructor

**Issue type to hierarchy level mapping:**

| Jira Issue Type | Assigned Level |
|----------------|---------------|
| Epic | parent |
| Story, Task, Feature, Enhancement, Bug, Improvement | standard |
| Sub-task | child |
| All others | excluded |

**Text filter rule (`both_empty` strategy):**
```python
keep_issue = bool(summary.strip() or description.strip())
```
A requirement is removed only if *both* the summary and description are empty strings. A requirement with either a title or a description is retained — it still provides some semantic signal to any model. This is the most conservative defensible filter.

**Splitting strategy:** Source-level stratified split by dominant link type. All positive links belonging to a given source requirement are assigned to the *same* split (train, val, or test) to prevent data leakage. The split ratio is 70% train / 10% val / 20% test, with random seed 42.

**Validation checks run after construction:**
- No both-empty requirement may survive into `requirements.json`
- No duplicate trace link in `trace_links.json`
- No source appears in more than one split (strict leakage check)

### 3.3 Data Quality Issues Found and Fixed

Two issues were identified during auditing and corrected before final experiments.

#### Fix 1 — Both-Empty Requirements

**Problem:** Earlier runs retained 551 Jira issues with no text in either field across all eight projects. These introduced meaningless trace pairs where the model is asked to decide from no evidence.

**Fix:** Applied `both_empty` filter in Stage 1. All links touching filtered issues are discarded before split construction.

| Project | Old requirements | Removed (both-empty) | V3 requirements | Old trace links | V3 trace links |
|---------|-----------------|---------------------|----------------|----------------|----------------|
| AAH | 468 | 18 | 360 | 412 | 313 |
| BEAM | 1,271 | 52 | 1,188 | 1,107 | 1,036 |
| CB | 1,749 | 44 | 1,664 | 1,622 | 1,541 |
| FH | 1,311 | 106 | 1,037 | 1,151 | 915 |
| JBIDE | 3,363 | 116 | 3,235 | 3,080 | 2,954 |
| KEYCLOAK | 2,475 | 177 | 1,996 | 2,274 | 1,817 |
| KOGITO | 1,880 | 30 | 1,746 | 1,658 | 1,531 |
| PROJQUAY | 474 | 8 | 454 | 412 | 396 |
| **Total** | **12,991** | **551** | **11,680** | **11,716** | **10,503** |

#### Fix 2 — Duplicate Negative Pair Rows

**Problem:** The previous miner selected the same top-3 Qwen3-nearest candidates for every positive of a given source. A source with N positives produced N identical copies of the same 3 negatives, inflating counts with no diversity gain. Total duplicate extra rows: 20,628 train / 2,898 val / 6,564 test.

**Fix:** Disjoint ranked windows per positive (see Section 3.4).

| Project | Old train dups | Old val dups | Old test dups | V3 train dups | V3 val dups | V3 test dups |
|---------|---------------|-------------|--------------|--------------|------------|-------------|
| AAH | 528 | 144 | 366 | 0 | 0 | 0 |
| BEAM | 1,818 | 240 | 639 | 0 | 0 | 0 |
| CB | 2,889 | 393 | 927 | 0 | 0 | 0 |
| FH | 1,776 | 198 | 870 | 0 | 0 | 0 |
| JBIDE | 5,769 | 807 | 1,698 | 0 | 0 | 0 |
| KEYCLOAK | 4,176 | 486 | 1,104 | 0 | 0 | 0 |
| KOGITO | 2,886 | 543 | 804 | 0 | 0 | 0 |
| PROJQUAY | 786 | 87 | 156 | **8** | 0 | 0 |
| **Total** | **20,628** | **2,898** | **6,564** | **8** | **0** | **0** |

The 8 remaining duplicate rows in PROJQUAY train are mathematically unavoidable: source `PROJQUAY-1644` has 5 positive links requiring 15 unique negatives, but only 7 hierarchy-valid non-linked candidates exist in the entire project. This exception affects training data only and is fully documented in the mining metadata.

### 3.4 Stage 2: Hard Negative Mining

**Model:** Qwen/Qwen3-Embedding-4B (4-billion-parameter dense text embedding model, 2,560-dimensional output, float16, run on H100 GPU).

**Candidate pool for a given source:**
All requirements at the correct adjacent hierarchy level (same level as true targets) that are *not* linked to this source in `trace_links.json` and have at least one non-empty text field.

**Ranking:** The source requirement's `full_text` (summary + description) is embedded. All candidates are embedded. Cosine similarity is computed between the source vector and every candidate vector. Candidates are sorted descending by similarity — most-similar first.

**Disjoint window assignment:**

```
Ranked candidates (most → least similar to source):
 C1  C2  C3 | C4  C5  C6 | C7  C8  C9 | C10 C11 C12 ...
 ─── window 0 ─── ─── window 1 ─── ─── window 2 ───

Positive 0 → negatives: C1, C2, C3   (hardest — most similar to source)
Positive 1 → negatives: C4, C5, C6   (next band)
Positive 2 → negatives: C7, C8, C9   (next band)
```

Each positive gets exactly 3 negatives from a different, non-overlapping slice of the ranked list. No negative is reused between positives of the same source as long as the candidate pool is large enough.

**Shortage policy:** If the pool is exhausted, the miner re-uses the best available candidates only for the unavoidable remainder, and records the shortage in `hard_negative_mining_metadata_v3.json`.

### 3.5 Similarity Inversion Finding

An important dataset characteristic confirmed during mining: in **5 of 8 projects**, the mean cosine similarity between a source and its hard negatives is *higher* than between the source and its true positive targets.

| Project | Mean neg similarity | Mean pos similarity | Inverted? |
|---------|--------------------|--------------------|-----------|
| AAH (test) | 0.7779 | 0.7799 | No (≈ tie) |
| BEAM (test) | 0.7924 | 0.8071 | No |
| CB (test) | **0.8099** | **0.7745** | ✅ Yes (+0.035) |
| FH (test) | 0.7942 | 0.7945 | No (≈ tie) |
| JBIDE (test) | **0.8912** | **0.8829** | ✅ Yes (+0.008) |
| KEYCLOAK (test) | **0.7743** | **0.7635** | ✅ Yes (+0.011) |
| KOGITO (test) | **0.8108** | **0.7571** | ✅ Yes (+0.054) |
| PROJQUAY (test) | **0.8433** | **0.8095** | ✅ Yes (+0.034) |

Overall test-set mean: neg sim = 0.812, pos sim = 0.796. This confirms the V3 dataset is adversarially difficult: superficial semantic similarity alone cannot discriminate true links from hard negatives.

---

## 4. Final V3 Dataset Statistics

### 4.1 Pair Counts

| Project | Train pairs | Val pairs | Test pairs | Total pairs |
|---------|------------|-----------|-----------|-------------|
| AAH | 804 | 104 | 344 | 1,252 |
| BEAM | 2,528 | 360 | 1,256 | 4,144 |
| CB | 3,936 | 548 | 1,680 | 6,164 |
| FH | 2,344 | 432 | 884 | 3,660 |
| JBIDE | 7,880 | 1,272 | 2,664 | 11,816 |
| KEYCLOAK | 4,680 | 648 | 1,940 | 7,268 |
| KOGITO | 4,068 | 608 | 1,448 | 6,124 |
| PROJQUAY | 1,068 | 64 | 452 | 1,584 |
| **Total** | **27,308** | **4,036** | **10,668** | **42,012** |

### 4.2 Class Ratio

All splits maintain a strict **1:3 positive:negative ratio** (25% positive, 75% negative). Every pair is labelled 1 (link exists) or 0 (no link).

### 4.3 Data Integrity Checks (V3 final state)

| Check | Result |
|-------|--------|
| Both-empty requirements retained | **0** |
| Duplicate pair keys in validation | **0** |
| Duplicate pair keys in test | **0** |
| Duplicate pair keys in train | **3** (PROJQUAY-1644 only) |
| Duplicate extra rows in train | **8** (PROJQUAY-1644 only) |
| Label conflicts | **0** |
| Source leakage between splits | **0** |

---

## 5. Folder Structure

```
ground_truth_v3_clean_pipeline/
│
├── DATA/                                  ← All data artifacts
│   ├── RAW_DATA/                          ← Original Jira dumps (from authors)
│   │   ├── AAH/
│   │   │   ├── AAH_issue_denoised.json    ← Raw Jira issue records
│   │   │   └── AAH_link.json             ← Raw Jira link records
│   │   └── [BEAM, CB, FH, JBIDE, KEYCLOAK, KOGITO, PROJQUAY]/
│   │       └── (same structure)
│   │
│   ├── .GROUND_TRUTH/                     ← Processed ground truth (local copy)
│   │   ├── {PROJECT}/
│   │   │   ├── requirements.json          ← Clean requirements with text
│   │   │   ├── trace_links.json           ← Valid hierarchical links
│   │   │   ├── metadata.json             ← Per-project construction metadata
│   │   │   └── splits/
│   │   │       ├── train_links.json       ← Positive links for training
│   │   │       ├── val_links.json         ← Positive links for validation
│   │   │       ├── test_links.json        ← Positive links for test
│   │   │       ├── final_pairs_train.json ← Pos + hard neg pairs (train)
│   │   │       ├── final_pairs_val.json   ← Pos + hard neg pairs (val)
│   │   │       └── final_pairs_test.json  ← Pos + hard neg pairs (test)
│   │   └── hard_negative_mining_metadata_v3.json  ← Mining run statistics
│   │
│   ├── 01_construct_ground_truth_v3_text_clean.py  ← Stage 1 constructor
│   ├── 02_mine_qwen3_diverse_hard_negatives.py     ← Stage 2 miner
│   ├── audit_ground_truth_v3.py                    ← Post-construction audit script
│   ├── validate_server_gt.py                       ← Server-side path validator
│   ├── GROUND_TRUTH_V3_AUDIT_REPORT.json           ← Machine-readable audit output
│   └── Ground_Truth_V3_Fix_Audit_Report.md         ← Human-readable audit report
│
├── SCRIPT/                                ← All evaluation scripts
│   ├── 1.VSM.py                           ← Baseline: Vector Space Model (TF-IDF)
│   ├── 2.SBERT.py                         ← Baseline: SBERT cosine similarity
│   ├── 3.BERT_FROZEN.py                   ← Baseline: Frozen BERT classifier
│   ├── 4.ModelSelection.py                ← LLM model selection (9 models, Ollama)
│   ├── 5.zero_shot_h100.py                ← Zero-shot Gemma 4 31B (H100)
│   ├── 6.rag_rerun_unified.py             ← RAG ablation (initial A/B/C/D/E/F)
│   ├── 6b.rag_extra_ablation_v3.py        ← RAG ablation (next 3 configs)
│   ├── 7.lora_rerun_unified.py            ← LoRA fine-tuning (6 versions, H100)
│   └── 8.combined_rerun.py                ← Combined RAG + LoRA champion
│
├── RESULTS/                               ← All experimental outputs
│   ├── ZERO_SHOT_QWEN_HARD/               ← Zero-shot results (per project)
│   ├── RAG_RERUN_V3/                      ← RAG ablation results (A–F)
│   │   ├── RAG_A/ ... RAG_F/              ← Per-config result folders
│   │   └── MASTER_RAG_COMPARISON.json     ← Cross-config comparison table
│   ├── checkpoints/                       ← Model selection checkpoints
│   ├── model_selection_v3_hard_results.json  ← Champion model selection output
│   ├── vsm_final_pairs_results.json       ← VSM baseline results
│   ├── sbert_final_pairs_results.json     ← SBERT baseline results
│   └── frozen_bert_final_pairs_results.json  ← Frozen BERT baseline results
│
└── REPORT/                                ← Thesis report artifacts
```

> **Server path:** All scripts that run on the H100 use the absolute server path:
> `/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/`
> with `DATA/GROUND_TRUTH/` for input and `RESULTS/` for output.
