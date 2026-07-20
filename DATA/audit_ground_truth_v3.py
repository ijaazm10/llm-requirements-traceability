"""
Audit Ground Truth V3 Dataset
=============================

Validates the cleaned V3 ground truth and final hard-negative pair files.

Checks:
  - required files exist for all projects
  - no both-empty requirements
  - trace links touch known requirements only
  - trace links are adjacent-level valid
  - no duplicate/conflicting trace links
  - source-level split isolation for positive links
  - final_pairs split ratio is 1:3
  - final_pairs positives exactly match split links
  - final_pairs negatives are not known positives
  - final_pairs touch no empty/unknown requirements
  - duplicate pair keys and label conflicts
  - train/val/test overlap by pair key

Usage:
  python audit_ground_truth_v3.py

Author: Thesis Work
Date: 2026-06-05
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "DATA" / ".GROUND_TRUTH"
OUT = REPO_ROOT / "DATA" / "GROUND_TRUTH_V3_AUDIT_REPORT.json"
PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]
SPLITS = ["train", "val", "test"]
NEG_PER_POS = 3


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def norm(value):
    return (value or "").strip()


def level(req):
    value = norm(req.get("level", "")).lower()
    if value in {"parent", "standard", "child"}:
        return value
    issue_type = norm(req.get("issue_type", ""))
    if issue_type == "Epic":
        return "parent"
    if issue_type in {"Story", "Task", "Feature", "Enhancement", "Bug", "Improvement"}:
        return "standard"
    if issue_type in {"Sub-task", "Sub-Task"}:
        return "child"
    return None


def valid_ltype(src_level, tgt_level):
    if src_level == "parent" and tgt_level == "standard":
        return "refinement"
    if src_level == "standard" and tgt_level == "child":
        return "subtask"
    return None


def pair_key(row):
    return (row["source_id"], row["target_id"])


def is_empty(req):
    return not norm(req.get("summary", "")) and not norm(req.get("description", ""))


def count_duplicates(rows):
    counts = Counter(pair_key(r) for r in rows)
    dup_keys = sum(1 for c in counts.values() if c > 1)
    dup_extra = sum(c - 1 for c in counts.values() if c > 1)
    return dup_keys, dup_extra


def audit_project(project):
    project_dir = ROOT / project
    errors = []
    warnings = []
    result = {"project": project}

    required = [
        project_dir / "requirements.json",
        project_dir / "trace_links.json",
        project_dir / "metadata.json",
    ]
    required += [project_dir / "splits" / f"{s}_links.json" for s in SPLITS]
    required += [project_dir / "splits" / f"final_pairs_{s}.json" for s in SPLITS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return {"project": project, "status": "FAIL", "errors": [f"Missing files: {missing}"]}

    reqs = load_json(project_dir / "requirements.json")
    trace_links = load_json(project_dir / "trace_links.json")
    req_map = {r["id"]: r for r in reqs}
    all_positive = {(l["source_id"], l["target_id"]) for l in trace_links}

    empty_ids = [rid for rid, req in req_map.items() if is_empty(req)]
    if empty_ids:
        errors.append(f"Both-empty requirements retained: {empty_ids[:20]}")

    trace_counts = Counter((l["source_id"], l["target_id"]) for l in trace_links)
    trace_dup_extra = sum(c - 1 for c in trace_counts.values() if c > 1)
    if trace_dup_extra:
        errors.append(f"Duplicate trace links: {trace_dup_extra}")

    for link in trace_links:
        src = req_map.get(link["source_id"])
        tgt = req_map.get(link["target_id"])
        if src is None or tgt is None:
            errors.append(f"Trace link touches unknown requirement: {link}")
            continue
        expected_ltype = valid_ltype(level(src), level(tgt))
        if expected_ltype is None:
            errors.append(f"Non-adjacent trace link retained: {link}")
        elif link.get("ltype") != expected_ltype:
            errors.append(f"Wrong ltype for trace link: {link}, expected={expected_ltype}")

    split_links = {}
    split_source_sets = {}
    split_pair_sets = {}
    result["splits"] = {}

    for split in SPLITS:
        links = load_json(project_dir / "splits" / f"{split}_links.json")
        pairs = load_json(project_dir / "splits" / f"final_pairs_{split}.json")
        split_links[split] = links
        split_source_sets[split] = {l["source_id"] for l in links}
        split_pair_sets[split] = {(l["source_id"], l["target_id"]) for l in links}

        link_set = {(l["source_id"], l["target_id"]) for l in links}
        pos_pairs = [p for p in pairs if p["label"] == 1]
        neg_pairs = [p for p in pairs if p["label"] == 0]
        pos_set = {pair_key(p) for p in pos_pairs}

        if len(pos_pairs) != len(links):
            errors.append(f"{split}: positive count mismatch pairs={len(pos_pairs)} links={len(links)}")
        if pos_set != link_set:
            errors.append(f"{split}: positive final pairs do not exactly match split links")
        if len(neg_pairs) != NEG_PER_POS * len(pos_pairs):
            errors.append(f"{split}: negative count is not 3x positive")

        label_map = defaultdict(set)
        for p in pairs:
            label_map[pair_key(p)].add(p["label"])
            if p["source_id"] not in req_map or p["target_id"] not in req_map:
                errors.append(f"{split}: pair touches unknown requirement {p}")
                continue
            if is_empty(req_map[p["source_id"]]) or is_empty(req_map[p["target_id"]]):
                errors.append(f"{split}: pair touches both-empty requirement {p}")
            if p["label"] == 0 and pair_key(p) in all_positive:
                errors.append(f"{split}: negative pair is known positive {p}")
            if p["label"] == 1 and pair_key(p) not in all_positive:
                errors.append(f"{split}: positive pair not in trace links {p}")

        conflicts = [k for k, labels in label_map.items() if len(labels) > 1]
        if conflicts:
            errors.append(f"{split}: conflicting labels for pair keys {conflicts[:20]}")

        dup_keys, dup_extra = count_duplicates(pairs)
        if dup_extra:
            warnings.append(f"{split}: duplicate final-pair keys dup_keys={dup_keys}, dup_extra={dup_extra}")

        result["splits"][split] = {
            "positive_links": len(links),
            "final_pairs": len(pairs),
            "final_pos": len(pos_pairs),
            "final_neg": len(neg_pairs),
            "ratio_neg_per_pos": round(len(neg_pairs) / max(len(pos_pairs), 1), 4),
            "duplicate_pair_keys": dup_keys,
            "duplicate_extra_rows": dup_extra,
            "unique_pair_keys": len(label_map),
            "source_count": len(split_source_sets[split]),
        }

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            source_overlap = split_source_sets[left] & split_source_sets[right]
            if source_overlap:
                errors.append(f"Source leakage {left}-{right}: {sorted(source_overlap)[:20]}")
            pair_overlap = split_pair_sets[left] & split_pair_sets[right]
            if pair_overlap:
                errors.append(f"Positive link overlap {left}-{right}: {sorted(pair_overlap)[:20]}")

    final_pair_sets = {}
    for split in SPLITS:
        pairs = load_json(project_dir / "splits" / f"final_pairs_{split}.json")
        final_pair_sets[split] = {pair_key(p) for p in pairs}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = final_pair_sets[left] & final_pair_sets[right]
            if overlap:
                warnings.append(f"Final pair-key overlap {left}-{right}: {len(overlap)}")

    result.update({
        "requirements": len(reqs),
        "trace_links": len(trace_links),
        "empty_requirements": len(empty_ids),
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
    })
    return result


def main():
    print("=" * 100)
    print("GROUND TRUTH V3 AUDIT")
    print("=" * 100)
    print(f"Root: {ROOT}")

    results = [audit_project(project) for project in PROJECTS]
    totals = {
        "requirements": sum(r.get("requirements", 0) for r in results),
        "trace_links": sum(r.get("trace_links", 0) for r in results),
        "train_pairs": sum(r.get("splits", {}).get("train", {}).get("final_pairs", 0) for r in results),
        "val_pairs": sum(r.get("splits", {}).get("val", {}).get("final_pairs", 0) for r in results),
        "test_pairs": sum(r.get("splits", {}).get("test", {}).get("final_pairs", 0) for r in results),
        "duplicate_extra_rows": sum(
            split.get("duplicate_extra_rows", 0)
            for r in results for split in r.get("splits", {}).values()
        ),
        "errors": sum(len(r.get("errors", [])) for r in results),
        "warnings": sum(len(r.get("warnings", [])) for r in results),
    }

    for r in results:
        print(f"\n{r['status']} {r['project']}")
        if "splits" in r:
            for split in SPLITS:
                s = r["splits"][split]
                print(
                    f"  {split:<5} pos={s['final_pos']:>5} neg={s['final_neg']:>5} "
                    f"total={s['final_pairs']:>5} ratio={s['ratio_neg_per_pos']:.1f} "
                    f"dup_extra={s['duplicate_extra_rows']:>3}"
                )
        for e in r.get("errors", [])[:10]:
            print(f"  ERROR: {e}")
        for w in r.get("warnings", [])[:10]:
            print(f"  WARN:  {w}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "root": str(ROOT),
        "projects": results,
        "totals": totals,
        "overall_status": "PASS" if totals["errors"] == 0 else "FAIL",
    }
    save_json(report, OUT)

    print("\n" + "=" * 100)
    print("TOTALS")
    print("=" * 100)
    for k, v in totals.items():
        print(f"{k}: {v}")
    print(f"overall_status: {report['overall_status']}")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
