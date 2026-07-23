# Source Data and Constructed Benchmark (`DATA/`)

This directory contains the Jira source exports used by the thesis and the resulting processed benchmark: retained requirements, positive trace links, fixed train/validation/test splits, and hard-negative pair files for eight projects.

## Folder Structure

```text
DATA/
|-- RAW_DATA/
|   |-- AAH/
|   |-- BEAM/
|   |-- CB/
|   |-- FH/
|   |-- JBIDE/
|   |-- KEYCLOAK/
|   |-- KOGITO/
|   `-- PROJQUAY/
|-- .GROUND_TRUTH/
|   |-- AAH/
|   |   |-- requirements.json
|   |   |-- trace_links.json
|   |   |-- metadata.json
|   |   `-- splits/
|   |       |-- final_pairs_train.json
|   |       |-- final_pairs_val.json
|   |       |-- final_pairs_test.json
|   |       |-- train_links.json
|   |       |-- val_links.json
|   |       `-- test_links.json
|   |-- BEAM/
|   |-- CB/
|   |-- FH/
|   |-- JBIDE/
|   |-- KEYCLOAK/
|   |-- KOGITO/
|   |-- PROJQUAY/
|   |-- ground_truth_v3_text_clean_metadata.json
|   `-- hard_negative_mining_metadata_v3.json
|-- 01_construct_ground_truth_v3_text_clean.py
|-- 02_mine_qwen3_diverse_hard_negatives.py
`-- README.md
```

## Source Exports

Each project folder under `RAW_DATA/` contains the Jira files supplied with the source dataset. Ground-truth construction uses two files:

- `{PROJECT}_issue_denoised.json`: issue identifiers, types, summaries, and descriptions after the inherited denoising step.
- `{PROJECT}_link.json`: recorded Jira links used to reconstruct the adjacent-level hierarchy.

The other issue, component, version, and preselected-link exports are retained as source artefacts but are not inputs to the final construction script. The thesis does not independently reproduce the denoising process that produced the denoised issue files.

The exports retain the provenance and terms of their originating public Jira projects; they are not relicensed under the repository's MIT licence.

## Project Counts

These counts are computed from the committed JSON files in `DATA/.GROUND_TRUTH/`.

| Project | Project family | Requirements | Positive links | Train pairs | Val pairs | Test pairs |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| AAH | Ansible Automation Hub (Red Hat) | 360 | 313 | 804 | 104 | 344 |
| BEAM | Apache Beam data-processing framework | 1,188 | 1,036 | 2,528 | 360 | 1,256 |
| CB | Couchbase database platform | 1,664 | 1,541 | 3,936 | 548 | 1,680 |
| FH | Fuse / Hawkular middleware-related Jira export | 1,037 | 915 | 2,344 | 432 | 884 |
| JBIDE | JBoss Tools / IDE-related Jira export | 3,235 | 2,954 | 7,880 | 1,272 | 2,664 |
| KEYCLOAK | Keycloak identity and access management | 1,996 | 1,817 | 4,680 | 648 | 1,940 |
| KOGITO | Kogito business automation platform | 1,746 | 1,531 | 4,068 | 608 | 1,448 |
| PROJQUAY | Quay container registry | 454 | 396 | 1,068 | 64 | 452 |
| Total | - | 11,680 | 10,503 | 27,308 | 4,036 | 10,668 |

## Ground-Truth Construction

`01_construct_ground_truth_v3_text_clean.py` reads the denoised issue and raw link files and writes the requirements, positive links, source-stratified splits, and construction metadata under `.GROUND_TRUTH/`.

The processed benchmark keeps issue types mapped to the studied Jira hierarchy:

- Parent level: Epic / Feature
- Standard level: Story / Task / Bug / Improvement / Enhancement
- Child level: Sub-task

Positive links are retained only when both endpoints exist in the processed issue exports, both endpoints have usable text, and the pair connects adjacent hierarchy levels. The resulting task is therefore a cross-level traceability benchmark over the mapped issue-type subset, not a claim to cover every relation present in the raw Jira instances.

## Hard-Negative Pair Construction

`02_mine_qwen3_diverse_hard_negatives.py` constructs the fixed 1:3 labelled pair files used by the experiments:

1. Requirements are embedded with Qwen3-Embedding-4B.
2. Candidate targets are restricted to valid adjacent hierarchy levels.
3. For each positive link, three semantically similar non-linked candidates are selected from ranked windows.

The goal is to evaluate the classifier/reader stage under difficult candidate conditions. The mined negatives are not random non-links; they are intentionally similar candidates that a first-stage retriever could plausibly surface.

## Artefact Boundary

- `RAW_DATA/` preserves the source exports associated with the study.
- `.GROUND_TRUTH/` preserves the exact constructed benchmark used by all methods.
- The two construction scripts document the transformation between these stages.
- Diagnostic notebooks, one-off validation scripts, and internal audit reports are intentionally excluded from the public artefact archive.
