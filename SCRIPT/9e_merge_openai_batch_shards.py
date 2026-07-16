"""
Merge OpenAI GPT-5.4 Batch Shards
=================================

Merges prediction JSON files produced by 9d_zero_shot_openai_batch_matched.py
when the full run is split into multiple OpenAI Batch shards because of the
organization's enqueued-token limit.

The merger concatenates shard prediction rows in shard-name order and validates
the merged sequence against final_pairs_test.json for every project. Rows are
not deduplicated; the merge is instance-order preserving.

Usage:
    python 9e_merge_openai_batch_shards.py

Author: Thesis Work
Date: 2026-06-04
"""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np


DATA_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/DATA/GROUND_TRUTH"
SHARD_ROOT = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS/OPENAI_ZERO_SHOT_BATCH_V3"
OUTPUT_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS/OPENAI_ZERO_SHOT_BATCH_V3/gpt-5_4_merged_matched_single_user_prompt_v1_batch"

MODEL = "gpt-5.4"
PROMPT_TEMPLATE = """You are analyzing hierarchical software requirements for cross level traceability.
Determine whether a traceability link exists between a High-Level Requirement (HLR) and a Low-Level Requirement (LLR).
A link exists if the LLR implements, refines, or decomposes the HLR.
High-Level Requirement:
  Summary: {hlr_summary}
  Description: {hlr_description}
Low-Level Requirement:
  Summary: {llr_summary}
  Description: {llr_description}
Return only:
{{"is_linked": true}}
or
{{"is_linked": false}}
No additional text."""
PROMPT_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]

ALL_PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]

# Leave empty to auto-discover shard folders. If you want explicit control,
# fill this list with folder names in the desired merge order.
SHARD_DIRS = []


def discover_shard_dirs(shard_root):
    root = Path(shard_root)
    if SHARD_DIRS:
        return SHARD_DIRS
    candidates = []
    for p in (root.iterdir() if root.exists() else []):
        if not p.is_dir():
            continue
        name = p.name
        if "matched_single_user_prompt_v1_batch" not in name:
            continue
        if "shard" not in name.lower():
            continue
        if name.startswith("gpt-5_4_merged"):
            continue
        candidates.append(name)
    return sorted(candidates)


def atomic_save_json(data, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_expected_pairs(project):
    pairs_file = Path(DATA_DIR) / project / "splits" / "final_pairs_test.json"
    return load_json(pairs_file)


def compute_metrics(predictions, labels):
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_conservative_metrics(predictions_log):
    preds, labels = [], []
    for row in predictions_log:
        pred = row["prediction"]
        preds.append(0 if pred is None else pred)
        labels.append(row["label"])
    return compute_metrics(preds, labels) if preds else compute_metrics([], [])


def summarize_project(project, predictions, expected_count):
    api_successes = sum(1 for e in predictions if e.get("api_success", True))
    api_failures = len(predictions) - api_successes
    parse_failures = sum(1 for e in predictions
                         if e.get("api_success", True) and not e.get("parse_success", True))

    clean_preds = [e["prediction"] for e in predictions
                   if e.get("api_success", True) and e["prediction"] is not None]
    clean_labels = [e["label"] for e in predictions
                    if e.get("api_success", True) and e["prediction"] is not None]

    metrics_clean = compute_metrics(clean_preds, clean_labels) if clean_preds else compute_metrics([], [])
    metrics_conservative = compute_conservative_metrics(predictions)

    return {
        "project": project,
        "n_pairs": len(predictions),
        "n_pairs_total_expected": expected_count,
        "n_pos": sum(e["label"] for e in predictions),
        "n_neg": len(predictions) - sum(e["label"] for e in predictions),
        "api_successes": api_successes,
        "api_failures": api_failures,
        "parse_failures": parse_failures,
        "metrics_clean": metrics_clean,
        "metrics_conservative": metrics_conservative,
        "total_prompt_tokens": sum(e.get("prompt_tokens", 0) for e in predictions),
        "total_completion_tokens": sum(e.get("completion_tokens", 0) for e in predictions),
        "avg_latency_ms": None,
    }


def main():
    print("=" * 95)
    print("MERGE OPENAI GPT-5.4 ZERO-SHOT BATCH SHARDS")
    print("=" * 95)

    shard_root = Path(SHARD_ROOT)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_rows = {project: [] for project in ALL_PROJECTS}
    loaded_files = []

    shard_dirs = discover_shard_dirs(shard_root)
    if not shard_dirs:
        raise SystemExit(f"No shard directories found under {shard_root}")

    print("Shard directories:")
    for s in shard_dirs:
        print(f"  {s}")

    for shard_dir in shard_dirs:
        shard_path = shard_root / shard_dir
        if not shard_path.exists():
            print(f"MISS shard directory: {shard_path}")
            continue
        loaded_any = False
        for project in ALL_PROJECTS:
            pred_file = shard_path / f"{project}_predictions.json"
            if not pred_file.exists():
                continue
            rows = load_json(pred_file)
            loaded_any = True
            loaded_files.append(str(pred_file))
            print(f"LOAD {project:<10} {len(rows):>5} rows from {shard_dir}")
            available_rows[project].extend(rows)
        if not loaded_any:
            print(f"WARN no prediction files found in: {shard_path}")

    all_results = []
    missing = {}

    for project in ALL_PROJECTS:
        expected = load_expected_pairs(project)
        merged_rows = available_rows[project]
        missing_rows = []
        if len(merged_rows) < len(expected):
            for pair in expected[len(merged_rows):]:
                missing_rows.append({
                    "source_id": pair["source_id"],
                    "target_id": pair["target_id"],
                    "label": pair["label"],
                })

        order_mismatches = []
        for idx, (exp, got) in enumerate(zip(expected, merged_rows)):
            expected_key = (exp["source_id"], exp["target_id"], exp["label"])
            got_key = (got["source_id"], got["target_id"], got["label"])
            if expected_key != got_key:
                order_mismatches.append({
                    "index": idx,
                    "expected": {
                        "source_id": exp["source_id"],
                        "target_id": exp["target_id"],
                        "label": exp["label"],
                    },
                    "got": {
                        "source_id": got["source_id"],
                        "target_id": got["target_id"],
                        "label": got["label"],
                    },
                })
                if len(order_mismatches) >= 20:
                    break

        missing[project] = missing_rows
        if order_mismatches:
            mismatch_path = output_dir / f"{project}_order_mismatches.json"
            atomic_save_json(order_mismatches, mismatch_path)
            print(f"CHECK {project:<10} order mismatches detected; see {mismatch_path}")
        atomic_save_json(merged_rows, output_dir / f"{project}_predictions.json")
        result = summarize_project(project, merged_rows, len(expected))
        all_results.append(result)

        status = "OK" if len(missing_rows) == 0 and len(merged_rows) == len(expected) else "CHECK"
        print(f"{status} {project:<10} merged={len(merged_rows):>5} expected={len(expected):>5} missing={len(missing_rows):>4}")

    macro_p = float(np.mean([r["metrics_clean"]["precision"] for r in all_results]))
    macro_r = float(np.mean([r["metrics_clean"]["recall"] for r in all_results]))
    macro_f1 = float(np.mean([r["metrics_clean"]["f1"] for r in all_results]))
    macro_f2 = float(np.mean([r["metrics_clean"]["f2"] for r in all_results]))
    macro_acc = float(np.mean([r["metrics_clean"]["accuracy"] for r in all_results]))

    macro_p_c = float(np.mean([r["metrics_conservative"]["precision"] for r in all_results]))
    macro_r_c = float(np.mean([r["metrics_conservative"]["recall"] for r in all_results]))
    macro_f1_c = float(np.mean([r["metrics_conservative"]["f1"] for r in all_results]))
    macro_f2_c = float(np.mean([r["metrics_conservative"]["f2"] for r in all_results]))

    total_pairs = sum(r["n_pairs"] for r in all_results)
    total_expected = sum(r["n_pairs_total_expected"] for r in all_results)
    total_api_fails = sum(r["api_failures"] for r in all_results)
    total_parse_fails = sum(r["parse_failures"] for r in all_results)
    total_prompt = sum(r["total_prompt_tokens"] for r in all_results)
    total_completion = sum(r["total_completion_tokens"] for r in all_results)

    print(f"\n\n{'=' * 95}")
    print("OPENAI GPT-5.4 ZERO-SHOT BATCH SHARDS - MERGED CLEAN macro")
    print("=" * 95)
    print(f"{'Project':<12} {'Pairs':>6} {'F1':>7} {'F2':>7} {'P':>7} {'R':>7} "
          f"{'Acc':>7} {'Parse':>6} {'API':>5} {'InTok':>10} {'OutTok':>8}")
    print("-" * 95)
    for r in all_results:
        m = r["metrics_clean"]
        print(f"{r['project']:<12} {r['n_pairs']:>6} {m['f1']:>7.4f} {m['f2']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['accuracy']:>7.4f} "
              f"{r['parse_failures']:>6} {r['api_failures']:>5} "
              f"{r['total_prompt_tokens']:>10,} {r['total_completion_tokens']:>8,}")
    print("-" * 95)
    print(f"{'MACRO AVG':<12} {total_pairs:>6} {macro_f1:>7.4f} {macro_f2:>7.4f} "
          f"{macro_p:>7.4f} {macro_r:>7.4f} {macro_acc:>7.4f} "
          f"{total_parse_fails:>6} {total_api_fails:>5}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "experiment": "openai_gpt54_zero_shot_matched_prompt_batch_merged_shards",
        "prompt_template_hash": PROMPT_HASH,
        "source_shard_dirs": shard_dirs,
        "loaded_prediction_files": loaded_files,
        "merge_strategy": "concatenate shard prediction rows in shard-name order; validate against final_pairs_test order; do not deduplicate",
        "missing": missing,
        "macro_clean": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
            "f2": round(macro_f2, 4),
            "accuracy": round(macro_acc, 4),
        },
        "macro_conservative": {
            "precision": round(macro_p_c, 4),
            "recall": round(macro_r_c, 4),
            "f1": round(macro_f1_c, 4),
            "f2": round(macro_f2_c, 4),
        },
        "total_pairs": total_pairs,
        "total_expected_pairs": total_expected,
        "total_api_failures": total_api_fails,
        "total_parse_failures": total_parse_fails,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "per_project": all_results,
    }
    atomic_save_json(output, output_dir / "results.json")

    total_missing = sum(len(v) for v in missing.values())
    if total_missing:
        atomic_save_json(missing, output_dir / "missing.json")
        print(f"\nWARNING: missing predictions: {total_missing}")

    print(f"\nSaved merged results to: {output_dir / 'results.json'}")
    print("=" * 95)


if __name__ == "__main__":
    main()
