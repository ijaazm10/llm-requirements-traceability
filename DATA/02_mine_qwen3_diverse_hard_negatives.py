"""
Qwen3 Diverse Top-K Hard Negative Mining - V3
============================================

Mines 1:3 hard-negative pair files for the text-clean ground_truth_3 dataset.

This script fixes the duplicate-negative weighting issue in the old miner. For
each source requirement, candidates are ranked once by Qwen3 cosine similarity.
Then the ranked list is sliced into disjoint windows:

  positive 1 -> ranks 0, 1, 2
  positive 2 -> ranks 3, 4, 5
  positive 3 -> ranks 6, 7, 8

This preserves the 1:3 positive:negative ratio while preventing repeated
(source_id, target_id) negative pairs within a split whenever the source has
enough hierarchy-valid non-linked candidates. If a source lacks enough
candidates, exact 1:3 and zero duplicates are mathematically incompatible; the
script then reuses the best candidates only for the unavoidable remainder and
records the shortage in metadata. Use --fail-on-shortage to make such cases
fatal.

Expected input layout:
  DATA/.GROUND_TRUTH/{PROJECT}/requirements.json
  DATA/.GROUND_TRUTH/{PROJECT}/trace_links.json
  DATA/.GROUND_TRUTH/{PROJECT}/splits/train_links.json
  DATA/.GROUND_TRUTH/{PROJECT}/splits/val_links.json
  DATA/.GROUND_TRUTH/{PROJECT}/splits/test_links.json

Outputs:
  final_pairs_train.json
  final_pairs_val.json
  final_pairs_test.json
  hard_negative_mining_metadata_v3.json

Usage on H100:
  python 02_mine_qwen3_diverse_hard_negatives.py --preflight

  CUDA_VISIBLE_DEVICES=0 python 02_mine_qwen3_diverse_hard_negatives.py --overwrite

Single project:
  CUDA_VISIBLE_DEVICES=0 python 02_mine_qwen3_diverse_hard_negatives.py --projects AAH --overwrite

Author: Thesis Work
Date: 2026-06-05
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]
SPLITS = ["train", "val", "test"]
NEG_PER_POS = 3
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = str(ROOT / "DATA" / ".GROUND_TRUTH")
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 512


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def norm_text(value):
    return (value or "").strip()


def has_text(req):
    return bool(norm_text(req.get("summary", "")) or norm_text(req.get("description", "")))


def canonical_level(req):
    level = norm_text(req.get("level", "")).lower()
    if level in {"parent", "standard", "child"}:
        return level

    issue_type = norm_text(req.get("issue_type", "") or req.get("type", ""))
    if issue_type == "Epic":
        return "parent"
    if issue_type in {"Story", "Task", "Feature", "Enhancement", "Bug", "Improvement"}:
        return "standard"
    if issue_type in {"Sub-task", "Sub-Task"}:
        return "child"
    return None


def valid_target_level_for_source(source_id, levels):
    level = levels.get(source_id)
    if level == "parent":
        return "standard"
    if level == "standard":
        return "child"
    return None


def load_requirements(project_path):
    reqs = load_json(project_path / "requirements.json")
    req_map = {}
    levels = {}
    for req in reqs:
        req_id = req["id"]
        summary = norm_text(req.get("summary", ""))
        description = norm_text(req.get("description", ""))
        level = canonical_level(req)
        req_map[req_id] = {
            "id": req_id,
            "summary": summary,
            "description": description,
            "full_text": f"{summary}\n\n{description}".strip(),
            "level": level,
            "issue_type": req.get("issue_type", req.get("type", "")),
        }
        if level:
            levels[req_id] = level
    return req_map, levels


def load_links(path):
    links = load_json(path)
    pairs = []
    for link in links:
        src = link.get("source_id") or link.get("source")
        tgt = link.get("target_id") or link.get("target")
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


def load_all_positive_set(project_path):
    links = load_json(project_path / "trace_links.json")
    positives = set()
    for link in links:
        src = link.get("source_id") or link.get("source")
        tgt = link.get("target_id") or link.get("target")
        if src and tgt:
            positives.add((src, tgt))
    return positives


def validate_text_clean(req_map):
    empty = [
        req_id for req_id, req in req_map.items()
        if not has_text(req)
    ]
    if empty:
        raise RuntimeError(
            f"Input dataset still contains both-empty requirements. First examples: {empty[:10]}"
        )


def group_links_by_source(split_links, req_map):
    groups = defaultdict(list)
    seen = set()
    for src, tgt in split_links:
        if src not in req_map or tgt not in req_map:
            continue
        key = (src, tgt)
        if key in seen:
            raise RuntimeError(f"Duplicate positive link in split input: {key}")
        seen.add(key)
        groups[src].append(tgt)
    for src in groups:
        groups[src] = sorted(groups[src])
    return dict(sorted(groups.items()))


def candidate_ids_for_source(src_id, req_map, levels, all_positive_set):
    target_level = valid_target_level_for_source(src_id, levels)
    if target_level is None:
        return []

    candidates = []
    for req_id, req in req_map.items():
        if req_id == src_id:
            continue
        if levels.get(req_id) != target_level:
            continue
        if (src_id, req_id) in all_positive_set:
            continue
        if not has_text(req):
            continue
        candidates.append(req_id)
    return sorted(candidates)


def preflight_project(project, args):
    project_path = Path(args.base_dir) / project
    req_map, levels = load_requirements(project_path)
    validate_text_clean(req_map)
    all_positive_set = load_all_positive_set(project_path)

    result = {"project": project, "splits": {}, "capacity_failures": []}
    for split in args.splits:
        split_links = load_links(project_path / "splits" / f"{split}_links.json")
        groups = group_links_by_source(split_links, req_map)
        split_info = {
            "positive_links": sum(len(v) for v in groups.values()),
            "sources": len(groups),
            "required_negatives": 0,
            "minimum_candidate_pool": None,
        }

        min_pool = None
        for src_id, target_ids in groups.items():
            candidates = candidate_ids_for_source(src_id, req_map, levels, all_positive_set)
            required = len(target_ids) * args.neg_per_pos
            split_info["required_negatives"] += required
            min_pool = len(candidates) if min_pool is None else min(min_pool, len(candidates))
            if len(candidates) < required:
                result["capacity_failures"].append({
                    "split": split,
                    "source_id": src_id,
                    "positives": len(target_ids),
                    "required_negatives": required,
                    "candidate_pool": len(candidates),
                })
        split_info["minimum_candidate_pool"] = min_pool or 0
        result["splits"][split] = split_info
    return result


def load_embedding_model(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading embedding model: {args.embedding_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(
        args.embedding_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()
    vram = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  Embedding model ready. VRAM: {vram:.1f} GB")
    return model, tokenizer, torch


def embed_batch(texts, model, tokenizer, torch_module, max_length):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch_module.no_grad():
        outputs = model(**inputs)

    mask = inputs["attention_mask"].unsqueeze(-1)
    token_embeddings = outputs.last_hidden_state
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    embeddings = summed / counts
    embeddings = torch_module.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float32)


def embed_all_requirements(req_map, model, tokenizer, torch_module, args):
    req_ids = sorted(req_map)
    texts = []
    for req_id in req_ids:
        text = req_map[req_id]["full_text"].strip()
        if not text:
            raise RuntimeError(f"Empty text reached embedding stage: {req_id}")
        texts.append(text)

    print(f"    Embedding {len(texts)} requirements...")
    all_embeddings = []
    n_batches = (len(texts) - 1) // args.embedding_batch_size + 1

    for batch_idx, start in enumerate(range(0, len(texts), args.embedding_batch_size), start=1):
        batch_texts = texts[start:start + args.embedding_batch_size]
        all_embeddings.append(
            embed_batch(batch_texts, model, tokenizer, torch_module, args.max_length)
        )
        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == n_batches:
            print(f"      Batch {batch_idx}/{n_batches}", end="\r")
    print()

    matrix = np.vstack(all_embeddings)
    return {req_id: matrix[idx] for idx, req_id in enumerate(req_ids)}


def mine_split(project, split, split_links, req_map, levels, all_positive_set, emb_map, args):
    source_groups = group_links_by_source(split_links, req_map)
    pairs = []
    stats = {
        "sources": len(source_groups),
        "total_positives": 0,
        "total_negatives": 0,
        "successful_sources": 0,
        "capacity_failures": [],
        "avg_pool_size": [],
        "pos_similarities": [],
        "neg_similarities": [],
    }

    for source_index, (src_id, target_ids) in enumerate(source_groups.items(), start=1):
        if source_index == 1 or source_index % 20 == 0 or source_index == len(source_groups):
            print(f"      [{project}/{split}] Source {source_index}/{len(source_groups)}", end="\r")

        candidates = candidate_ids_for_source(src_id, req_map, levels, all_positive_set)
        required = len(target_ids) * args.neg_per_pos
        if len(candidates) < required:
            failure = {
                "source_id": src_id,
                "positives": len(target_ids),
                "required_negatives": required,
                "candidate_pool": len(candidates),
            }
            stats["capacity_failures"].append(failure)
            if args.fail_on_shortage:
                raise RuntimeError(
                    f"Not enough unique negatives for {project}/{split}/{src_id}: "
                    f"required={required}, candidates={len(candidates)}"
                )

        src_emb = emb_map[src_id]
        candidate_matrix = np.vstack([emb_map[cid] for cid in candidates])
        similarities = candidate_matrix @ src_emb
        ranked_order = np.argsort(-similarities)
        ranked_candidates = [candidates[i] for i in ranked_order]
        ranked_sims = [float(similarities[i]) for i in ranked_order]
        stats["avg_pool_size"].append(len(candidates))

        used_for_source = set()
        for pos_idx, tgt_id in enumerate(target_ids):
            stats["total_positives"] += 1
            pairs.append({"source_id": src_id, "target_id": tgt_id, "label": 1})

            if tgt_id in emb_map:
                stats["pos_similarities"].append(float(emb_map[tgt_id] @ src_emb))

            start = pos_idx * args.neg_per_pos
            end = start + args.neg_per_pos
            window = ranked_candidates[start:end]
            window_sims = ranked_sims[start:end]

            if len(window) < args.neg_per_pos:
                for cid, sim in zip(ranked_candidates, ranked_sims):
                    if len(window) >= args.neg_per_pos:
                        break
                    if cid not in window:
                        window.append(cid)
                        window_sims.append(sim)

            for neg_id, neg_sim in zip(window, window_sims):
                key = (src_id, neg_id)
                if key in used_for_source and args.fail_on_shortage:
                    raise RuntimeError(f"Duplicate negative within source despite strict mode: {key}")
                used_for_source.add(key)
                pairs.append({"source_id": src_id, "target_id": neg_id, "label": 0})
                stats["neg_similarities"].append(float(neg_sim))
                stats["total_negatives"] += 1

        stats["successful_sources"] += 1

    print()
    validate_pairs(project, split, pairs, split_links, all_positive_set, req_map, args)
    return pairs, stats


def duplicate_pair_count(pairs):
    counts = Counter((p["source_id"], p["target_id"]) for p in pairs)
    return sum(1 for count in counts.values() if count > 1), sum(count - 1 for count in counts.values())


def validate_pairs(project, split, pairs, split_links, all_positive_set, req_map, args):
    pos = sum(1 for p in pairs if p["label"] == 1)
    neg = sum(1 for p in pairs if p["label"] == 0)
    expected_pos = len(split_links)
    expected_neg = expected_pos * args.neg_per_pos

    errors = []
    if pos != expected_pos:
        errors.append(f"positive count mismatch: got={pos}, expected={expected_pos}")
    if neg != expected_neg:
        errors.append(f"negative count mismatch: got={neg}, expected={expected_neg}")

    dup_keys, dup_extra = duplicate_pair_count(pairs)
    if dup_extra != 0 and args.fail_on_shortage:
        errors.append(f"duplicate pair keys found: dup_keys={dup_keys}, extra_rows={dup_extra}")

    seen_labels = {}
    for pair in pairs:
        key = (pair["source_id"], pair["target_id"])
        label = pair["label"]
        if key in seen_labels and seen_labels[key] != label:
            errors.append(f"conflicting labels for {key}")
        seen_labels[key] = label

        if pair["source_id"] not in req_map or pair["target_id"] not in req_map:
            errors.append(f"pair touches unknown requirement: {key}")
        elif not has_text(req_map[pair["source_id"]]) or not has_text(req_map[pair["target_id"]]):
            errors.append(f"pair touches empty requirement: {key}")

        if label == 1 and key not in all_positive_set:
            errors.append(f"positive pair not in trace_links: {key}")
        if label == 0 and key in all_positive_set:
            errors.append(f"negative pair is a known positive link: {key}")

        if len(errors) >= 20:
            break

    if errors:
        raise RuntimeError(f"Validation failed for {project}/{split}:\n" + "\n".join(errors))


def process_project(project, model, tokenizer, torch_module, args):
    project_path = Path(args.base_dir) / project
    splits_dir = project_path / "splits"

    print("\n" + "=" * 80)
    print(f"Project: {project}")
    print("=" * 80)

    req_map, levels = load_requirements(project_path)
    validate_text_clean(req_map)
    all_positive_set = load_all_positive_set(project_path)

    print(f"  Requirements: {len(req_map)}")
    print(f"  Trace links:  {len(all_positive_set)}")
    print(f"  Levels:       parent={sum(v == 'parent' for v in levels.values())}, "
          f"standard={sum(v == 'standard' for v in levels.values())}, "
          f"child={sum(v == 'child' for v in levels.values())}")

    emb_map = embed_all_requirements(req_map, model, tokenizer, torch_module, args)
    torch_module.cuda.empty_cache()

    project_summary = {}
    for split in args.splits:
        links_path = splits_dir / f"{split}_links.json"
        if not links_path.exists():
            raise FileNotFoundError(f"Missing split links: {links_path}")
        split_links = load_links(links_path)
        print(f"\n  [{split.upper()}] positives={len(split_links)}")

        pairs, stats = mine_split(
            project, split, split_links, req_map, levels, all_positive_set, emb_map, args
        )
        n_pos = sum(1 for p in pairs if p["label"] == 1)
        n_neg = sum(1 for p in pairs if p["label"] == 0)
        dup_keys, dup_extra = duplicate_pair_count(pairs)

        out_file = splits_dir / f"final_pairs_{split}.json"
        if out_file.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"{out_file} already exists. Use --overwrite to replace it.")
        if not args.dry_run:
            save_json(pairs, out_file)

        project_summary[split] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "total": len(pairs),
            "ratio": round(n_neg / max(n_pos, 1), 4),
            "duplicate_pair_keys": dup_keys,
            "duplicate_extra_rows": dup_extra,
            "sources": stats["sources"],
            "capacity_failures": stats["capacity_failures"],
            "avg_pool_size": round(float(np.mean(stats["avg_pool_size"])), 2) if stats["avg_pool_size"] else 0,
            "neg_sim_mean": round(float(np.mean(stats["neg_similarities"])), 4) if stats["neg_similarities"] else 0,
            "pos_sim_mean": round(float(np.mean(stats["pos_similarities"])), 4) if stats["pos_similarities"] else 0,
        }

        print(
            f"    Generated: pos={n_pos}, neg={n_neg}, total={len(pairs)}, "
            f"ratio=1:{n_neg / max(n_pos, 1):.1f}, duplicate_extra={dup_extra}"
        )
        if not args.dry_run:
            print(f"    Saved: {out_file}")

    return project_summary


def run_preflight(args):
    print("=" * 80)
    print("PREFLIGHT - UNIQUE NEGATIVE CAPACITY CHECK")
    print("=" * 80)
    failures = []
    summaries = {}
    for project in args.projects:
        result = preflight_project(project, args)
        summaries[project] = result
        project_failures = result["capacity_failures"]
        failures.extend({"project": project, **f} for f in project_failures)
        print(f"\n{project}")
        for split, info in result["splits"].items():
            print(
                f"  {split:<5} positives={info['positive_links']:>5} "
                f"sources={info['sources']:>4} required_neg={info['required_negatives']:>6} "
                f"min_pool={info['minimum_candidate_pool']:>5}"
            )
        if project_failures:
            print(f"  CAPACITY FAILURES: {len(project_failures)}")

    if failures and args.fail_on_shortage:
        print("\nERROR: at least one source lacks enough unique negatives.")
        print("First failures:")
        for failure in failures[:20]:
            print(f"  {failure}")
        sys.exit(2)

    if failures:
        print("\nWARNING: some sources lack enough unique negatives.")
        print("The miner will reuse negatives only for these unavoidable shortages.")
        print("First shortages:")
        for failure in failures[:20]:
            print(f"  {failure}")
        print(f"Total shortages: {len(failures)}")
    else:
        print("\nPreflight OK: every source has enough unique non-linked candidates.")

    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Mine duplicate-free Qwen3 hard negatives for ground_truth_3.")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--projects", nargs="+", choices=PROJECTS, default=PROJECTS)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--preflight", action="store_true", help="Check candidate capacity without loading the GPU model.")
    parser.add_argument("--dry-run", action="store_true", help="Run mining and validation, but do not write final_pairs files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing final_pairs_{split}.json files.")
    parser.add_argument("--fail-on-shortage", action="store_true",
                        help="Fail if a source lacks enough unique candidates for exact duplicate-free 1:3 mining.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().isoformat()
    np.random.seed(SEED)

    print("=" * 80)
    print("QWEN3 DIVERSE TOP-K HARD NEGATIVE MINING - V3")
    print("=" * 80)
    print(f"Timestamp:       {timestamp}")
    print(f"Base dir:        {args.base_dir}")
    print(f"Projects:        {args.projects}")
    print(f"Splits:          {args.splits}")
    print(f"Neg per pos:     {args.neg_per_pos}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Fail shortage:   {args.fail_on_shortage}")
    print(f"Dry run:         {args.dry_run}")
    print(f"Overwrite:       {args.overwrite}")

    if args.preflight:
        run_preflight(args)
        return

    run_preflight(args)

    model, tokenizer, torch_module = load_embedding_model(args)
    start_time = time.time()
    all_results = {}
    for project in args.projects:
        all_results[project] = process_project(project, model, tokenizer, torch_module, args)

    elapsed = time.time() - start_time
    metadata = {
        "timestamp": timestamp,
        "version": "v3_diverse_qwen3_hard_negatives",
        "base_dir": args.base_dir,
        "projects": args.projects,
        "splits": args.splits,
        "neg_per_pos": args.neg_per_pos,
        "embedding_model": args.embedding_model,
        "strategy": "disjoint ranked windows per source: ranks [i*k:(i+1)*k] for positive i",
        "fail_on_shortage": args.fail_on_shortage,
        "shortage_policy": "reuse ranked candidates only when exact 1:3 exceeds the available unique candidate pool",
        "seed": SEED,
        "total_time_minutes": round(elapsed / 60, 2),
        "results": all_results,
    }

    if not args.dry_run:
        save_json(metadata, Path(args.base_dir) / "hard_negative_mining_metadata_v3.json")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for project, split_results in all_results.items():
        parts = []
        for split in args.splits:
            info = split_results.get(split, {})
            parts.append(
                f"{split}:total={info.get('total', 0)},dup_extra={info.get('duplicate_extra_rows', 0)}"
            )
        print(f"  {project:<10} " + " | ".join(parts))
    print(f"\nElapsed: {elapsed / 60:.1f} minutes")
    if not args.dry_run:
        print(f"Metadata: {Path(args.base_dir) / 'hard_negative_mining_metadata_v3.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
