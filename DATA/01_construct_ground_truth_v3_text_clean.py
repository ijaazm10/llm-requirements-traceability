"""
Ground Truth V3 Construction - Text-Clean Traceability Dataset
==============================================================

Constructs a frozen requirements traceability dataset from raw Jira exports.

This is the cleaned successor to construct_ground_truth_v2.py. It preserves the
original adjacent-level traceability definition and source-level stratified
splitting, but removes requirements with insufficient text before extracting
trace links.

Default text filter:
  both_empty: remove an issue only when summary == "" AND description == "".

Why this default:
  A requirement with either a title or a description still gives the model some
  semantic signal. A requirement with neither field gives no meaningful input.

Outputs per project:
  requirements.json
  trace_links.json
  splits/train_links.json
  splits/val_links.json
  splits/test_links.json
  metadata.json

Usage:
  python 01_construct_ground_truth_v3_text_clean.py --dry-run

  python 01_construct_ground_truth_v3_text_clean.py ^
    --raw-data-dir "DATA/RAW_DATA" ^
    --output-dir "DATA/.GROUND_TRUTH" ^
    --filter-strategy both_empty

Author: Thesis Work
Date: 2026-06-05
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]
ROOT = Path(__file__).resolve().parents[1]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_SEED = 42


def norm_text(value):
    return (value or "").strip()


def get_level(itype):
    """Map Jira issue type to abstraction level. Matches the v2 constructor."""
    if itype == "Epic":
        return "parent"
    if itype in ["Story", "Task", "Feature", "Enhancement", "Bug", "Improvement"]:
        return "standard"
    if itype == "Sub-task":
        return "child"
    return None


def get_ltype(source_level, target_level):
    if source_level == "parent" and target_level == "standard":
        return "refinement"
    if source_level == "standard" and target_level == "child":
        return "subtask"
    return None


def should_keep_issue(issue, strategy):
    summary = norm_text(issue.get("summary", ""))
    description = norm_text(issue.get("description", ""))

    if strategy == "both_empty":
        return bool(summary or description)
    if strategy == "summary_empty":
        return bool(summary)
    if strategy == "any_empty":
        return bool(summary and description)
    raise ValueError(f"Unknown filter strategy: {strategy}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_issues(raw_data_dir, project):
    issue_file = Path(raw_data_dir) / project / f"{project}_issue_denoised.json"
    if not issue_file.exists():
        raise FileNotFoundError(f"Missing issue file: {issue_file}")

    issues = load_json(issue_file)
    id_to_issue = {}
    skipped_malformed = 0

    for issue in issues:
        if not isinstance(issue, dict) or "id" not in issue or "itype" not in issue:
            skipped_malformed += 1
            continue
        id_to_issue[issue["id"]] = issue

    return id_to_issue, skipped_malformed, len(issues)


def load_links(raw_data_dir, project):
    link_file = Path(raw_data_dir) / project / f"{project}_link.json"
    if not link_file.exists():
        raise FileNotFoundError(f"Missing link file: {link_file}")
    return load_json(link_file)


def extract_link_pair(link):
    if isinstance(link, dict):
        source = link.get("source")
        target = link.get("target")
    elif isinstance(link, list) and len(link) >= 2:
        source = link[0]
        target = link[1]
    else:
        return None

    if source and target:
        return source, target
    return None


def filter_links(raw_links, id_to_issue, kept_issue_ids):
    valid_links = []
    seen = set()
    stats = Counter()

    for link in raw_links:
        pair = extract_link_pair(link)
        if pair is None:
            stats["malformed_links"] += 1
            continue

        source_id, target_id = pair
        link_key = (source_id, target_id)
        if link_key in seen:
            stats["duplicate_raw_links"] += 1
            continue
        seen.add(link_key)

        source_issue = id_to_issue.get(source_id)
        target_issue = id_to_issue.get(target_id)
        if source_issue is None or target_issue is None:
            stats["missing_issue_links"] += 1
            continue

        if source_id not in kept_issue_ids or target_id not in kept_issue_ids:
            stats["text_filtered_links"] += 1
            continue

        source_type = source_issue.get("itype", "")
        target_type = target_issue.get("itype", "")
        source_level = get_level(source_type)
        target_level = get_level(target_type)
        if source_level is None or target_level is None:
            stats["unknown_type_links"] += 1
            continue

        ltype = get_ltype(source_level, target_level)
        if ltype is None:
            stats["non_adjacent_links"] += 1
            continue

        valid_links.append({
            "source_id": source_id,
            "target_id": target_id,
            "source_type": source_type,
            "target_type": target_type,
            "ltype": ltype,
        })
        stats["valid_links"] += 1

    valid_links.sort(key=lambda x: (x["source_id"], x["target_id"]))
    return valid_links, dict(stats)


def extract_participating_requirements(valid_links, id_to_issue, project):
    participating_ids = set()
    for link in valid_links:
        participating_ids.add(link["source_id"])
        participating_ids.add(link["target_id"])

    requirements = []
    for req_id in sorted(participating_ids):
        issue = id_to_issue.get(req_id)
        if issue is None:
            continue

        itype = issue.get("itype", "")
        level = get_level(itype)
        if level is None:
            continue

        summary = norm_text(issue.get("summary", ""))
        description = norm_text(issue.get("description", ""))
        full_text = f"{summary}\n\n{description}".strip()

        requirements.append({
            "id": req_id,
            "level": level,
            "issue_type": itype,
            "summary": summary,
            "description": description,
            "full_text": full_text,
            "project": project,
        })

    return requirements


def stratified_split_by_source_ltype(valid_links):
    source_to_links = defaultdict(list)
    for link in valid_links:
        source_to_links[link["source_id"]].append(link)

    source_dominant_ltype = {}
    for source_id, links in source_to_links.items():
        counts = Counter(link["ltype"] for link in links)
        if len(counts) == 1:
            dominant = next(iter(counts))
        elif counts["refinement"] > counts.get("subtask", 0):
            dominant = "refinement"
        elif counts.get("subtask", 0) > counts["refinement"]:
            dominant = "subtask"
        else:
            dominant = "mixed"
        source_dominant_ltype[source_id] = dominant

    sources_by_ltype = defaultdict(list)
    for source_id, dominant in source_dominant_ltype.items():
        sources_by_ltype[dominant].append(source_id)

    random.seed(RANDOM_SEED)
    train_sources, val_sources, test_sources = set(), set(), set()

    for _, sources in sorted(sources_by_ltype.items()):
        sources_copy = sorted(sources)
        random.shuffle(sources_copy)
        n = len(sources_copy)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        train_sources.update(sources_copy[:train_end])
        val_sources.update(sources_copy[train_end:val_end])
        test_sources.update(sources_copy[val_end:])

    train_links, val_links, test_links = [], [], []
    for source_id, links in source_to_links.items():
        if source_id in train_sources:
            train_links.extend(links)
        elif source_id in val_sources:
            val_links.extend(links)
        elif source_id in test_sources:
            test_links.extend(links)
        else:
            raise RuntimeError(f"Source not assigned to a split: {source_id}")

    train_links.sort(key=lambda x: (x["source_id"], x["target_id"]))
    val_links.sort(key=lambda x: (x["source_id"], x["target_id"]))
    test_links.sort(key=lambda x: (x["source_id"], x["target_id"]))

    return train_links, val_links, test_links


def compute_distribution(links):
    total = len(links)
    counts = Counter(link["ltype"] for link in links)
    if total == 0:
        return {"refinement": 0, "subtask": 0, "refinement_pct": 0.0, "subtask_pct": 0.0}
    return {
        "refinement": counts.get("refinement", 0),
        "subtask": counts.get("subtask", 0),
        "refinement_pct": 100 * counts.get("refinement", 0) / total,
        "subtask_pct": 100 * counts.get("subtask", 0) / total,
    }


def validate_project(requirements, valid_links, splits, strategy):
    req_ids = {r["id"] for r in requirements}
    errors = []

    for req in requirements:
        summary = norm_text(req.get("summary", ""))
        description = norm_text(req.get("description", ""))
        if strategy == "both_empty" and not (summary or description):
            errors.append(f"Both-empty requirement retained: {req['id']}")
        if strategy == "summary_empty" and not summary:
            errors.append(f"Summary-empty requirement retained: {req['id']}")
        if strategy == "any_empty" and (not summary or not description):
            errors.append(f"Partially empty requirement retained: {req['id']}")

    seen_links = set()
    for link in valid_links:
        key = (link["source_id"], link["target_id"])
        if key in seen_links:
            errors.append(f"Duplicate trace link retained: {key}")
        seen_links.add(key)
        if link["source_id"] not in req_ids or link["target_id"] not in req_ids:
            errors.append(f"Trace link touches missing requirement: {key}")

    split_source_sets = {}
    for split_name, links in splits.items():
        split_source_sets[split_name] = {link["source_id"] for link in links}

    for left in split_source_sets:
        for right in split_source_sets:
            if left >= right:
                continue
            overlap = split_source_sets[left] & split_source_sets[right]
            if overlap:
                errors.append(f"Source leakage between {left} and {right}: {sorted(overlap)[:5]}")

    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors[:50]))


def process_project(project, args):
    print(f"\nProcessing {project}...")
    id_to_issue, skipped_malformed, raw_issue_count = load_issues(args.raw_data_dir, project)
    raw_links = load_links(args.raw_data_dir, project)

    kept_issue_ids = {
        issue_id for issue_id, issue in id_to_issue.items()
        if get_level(issue.get("itype", "")) is not None and should_keep_issue(issue, args.filter_strategy)
    }
    known_type_ids = {
        issue_id for issue_id, issue in id_to_issue.items()
        if get_level(issue.get("itype", "")) is not None
    }
    text_filtered_issue_ids = sorted(known_type_ids - kept_issue_ids)

    valid_links, link_filter_stats = filter_links(raw_links, id_to_issue, kept_issue_ids)
    requirements = extract_participating_requirements(valid_links, id_to_issue, project)
    train_links, val_links, test_links = stratified_split_by_source_ltype(valid_links)
    splits = {"train": train_links, "val": val_links, "test": test_links}

    validate_project(requirements, valid_links, splits, args.filter_strategy)

    overall_dist = compute_distribution(valid_links)
    split_dist = {name: compute_distribution(links) for name, links in splits.items()}

    metadata = {
        "project": project,
        "timestamp": args.timestamp,
        "version": "v3_text_clean",
        "filter_strategy": args.filter_strategy,
        "num_raw_issues": raw_issue_count,
        "num_loaded_issues": len(id_to_issue),
        "num_malformed_issues_skipped": skipped_malformed,
        "num_known_type_issues": len(known_type_ids),
        "num_text_filtered_known_type_issues": len(text_filtered_issue_ids),
        "text_filtered_issue_ids": text_filtered_issue_ids,
        "num_requirements": len(requirements),
        "num_raw_links": len(raw_links),
        "num_trace_links": len(valid_links),
        "num_train_links": len(train_links),
        "num_val_links": len(val_links),
        "num_test_links": len(test_links),
        "link_filter_stats": link_filter_stats,
        "definition": "Direct adjacent-level hierarchical links only",
        "included_links": ["Epic -> Standard (refinement)", "Standard -> Sub-task (subtask)"],
        "excluded_links": [
            "Epic -> Sub-task transitive links",
            "Same-level links",
            "Links with unknown issue types",
            "Links touching text-filtered requirements",
        ],
        "split_strategy": "Source-level stratified by dominant ltype",
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "class_distribution": {
            "overall": overall_dist,
            "train": split_dist["train"],
            "val": split_dist["val"],
            "test": split_dist["test"],
        },
        "random_seed": RANDOM_SEED,
    }

    if not args.dry_run:
        project_output = Path(args.output_dir) / project
        split_output = project_output / "splits"
        save_json(requirements, project_output / "requirements.json")
        save_json(valid_links, project_output / "trace_links.json")
        save_json(train_links, split_output / "train_links.json")
        save_json(val_links, split_output / "val_links.json")
        save_json(test_links, split_output / "test_links.json")
        save_json(metadata, project_output / "metadata.json")

    print(
        f"  Requirements: {len(requirements)} "
        f"(text-filtered known-type issues: {len(text_filtered_issue_ids)})"
    )
    print(
        f"  Trace links:  {len(valid_links)} "
        f"(removed by text filter: {link_filter_stats.get('text_filtered_links', 0)})"
    )
    print(f"  Train/Val/Test links: {len(train_links)} / {len(val_links)} / {len(test_links)}")

    return metadata


def ensure_output_safe(output_dir, overwrite, dry_run):
    path = Path(output_dir)
    if dry_run:
        return
    if path.exists() and any(path.iterdir()) and not overwrite:
        print(f"ERROR: output directory exists and is not empty: {path}")
        print("Use --overwrite only when you intentionally want to replace files in that directory.")
        sys.exit(1)
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Construct text-clean ground_truth_3 dataset.")
    parser.add_argument("--raw-data-dir", default=str(ROOT / "DATA" / "RAW_DATA"))
    parser.add_argument("--output-dir", default=str(ROOT / "DATA" / ".GROUND_TRUTH"))
    parser.add_argument("--projects", nargs="+", choices=PROJECTS, default=PROJECTS)
    parser.add_argument(
        "--filter-strategy",
        choices=["both_empty", "summary_empty", "any_empty"],
        default="both_empty",
        help="both_empty is recommended: remove only artifacts with no summary and no description.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.timestamp = datetime.now().isoformat()

    print("=" * 80)
    print("GROUND TRUTH V3 - TEXT-CLEAN CONSTRUCTION")
    print("=" * 80)
    print(f"Raw data:        {args.raw_data_dir}")
    print(f"Output:          {args.output_dir}")
    print(f"Projects:        {args.projects}")
    print(f"Filter strategy: {args.filter_strategy}")
    print(f"Dry run:         {args.dry_run}")
    print(f"Overwrite:       {args.overwrite}")

    ensure_output_safe(args.output_dir, args.overwrite, args.dry_run)

    all_metadata = []
    for project in args.projects:
        all_metadata.append(process_project(project, args))

    total = {
        "requirements": sum(m["num_requirements"] for m in all_metadata),
        "trace_links": sum(m["num_trace_links"] for m in all_metadata),
        "train_links": sum(m["num_train_links"] for m in all_metadata),
        "val_links": sum(m["num_val_links"] for m in all_metadata),
        "test_links": sum(m["num_test_links"] for m in all_metadata),
        "text_filtered_known_type_issues": sum(m["num_text_filtered_known_type_issues"] for m in all_metadata),
        "text_filtered_links": sum(m["link_filter_stats"].get("text_filtered_links", 0) for m in all_metadata),
    }

    summary = {
        "timestamp": args.timestamp,
        "version": "v3_text_clean",
        "filter_strategy": args.filter_strategy,
        "raw_data_dir": args.raw_data_dir,
        "output_dir": args.output_dir,
        "projects": args.projects,
        "random_seed": RANDOM_SEED,
        "totals": total,
        "per_project": all_metadata,
    }

    if not args.dry_run:
        save_json(summary, Path(args.output_dir) / "ground_truth_v3_text_clean_metadata.json")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Project':<12} {'Reqs':>7} {'Links':>7} {'Train':>7} {'Val':>7} {'Test':>7} {'TextRm':>7}")
    for m in all_metadata:
        print(
            f"{m['project']:<12} {m['num_requirements']:>7} {m['num_trace_links']:>7} "
            f"{m['num_train_links']:>7} {m['num_val_links']:>7} {m['num_test_links']:>7} "
            f"{m['num_text_filtered_known_type_issues']:>7}"
        )
    print("-" * 80)
    print(
        f"{'TOTAL':<12} {total['requirements']:>7} {total['trace_links']:>7} "
        f"{total['train_links']:>7} {total['val_links']:>7} {total['test_links']:>7} "
        f"{total['text_filtered_known_type_issues']:>7}"
    )
    if not args.dry_run:
        print(f"\nSaved metadata: {Path(args.output_dir) / 'ground_truth_v3_text_clean_metadata.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
