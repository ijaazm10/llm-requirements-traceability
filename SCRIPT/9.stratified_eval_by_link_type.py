"""
Stratified Evaluation by Link Type - V3
=======================================
Splits held-out test predictions into Refinement (Parent->Standard) vs
Subtask (Standard->Child) strata and computes P, R, F2 independently per
stratum, per project and macro-averaged.

The stratum of a pair is derived from the abstraction levels of its
endpoints in requirements.json, so mined hard negatives inherit the
stratum of their (source, candidate) pair.

Scoring:
  clean        : parse failures (prediction is None) excluded  [thesis primary]
  conservative : parse failures scored as prediction 0

Macro-averaging is computed over projects that contain at least one
positive pair in the stratum (refinement: 6 projects, since BEAM and CB
retain no Epic-level links; subtask: 8 projects).

Usage:
  python 9.stratified_eval_by_link_type.py
Outputs:
  RESULTS/stratified_by_link_type_v3.json
"""

import json
import os
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GT = os.path.join(BASE, "DATA", ".GROUND_TRUTH")
RESULTS = os.path.join(BASE, "RESULTS")
PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]

METHODS = {
    "Zero-Shot": os.path.join(RESULTS, "ZERO_SHOT_QWEN_HARD"),
    "RAG-B": os.path.join(RESULTS, "RAG_STAGE1_V3_8192", "RAG_B", "PREDICTIONS"),
    "LoRA V4": os.path.join(RESULTS, "LORA_RERUN_V3", "V4_EFFICIENCY", "PREDICTIONS"),
    "Combined": os.path.join(RESULTS, "COMBINED_RERUN_V3", "V4_EFFICIENCY_RAG_B_8192", "PREDICTIONS"),
    "GPT-5.4": os.path.join(RESULTS, "OPENAI_ZERO_SHOT_BATCH_V3", "gpt-5_4_merged_matched_single_user_prompt_v1_batch"),
    "Claude Sonnet 4.6": os.path.join(RESULTS, "CLAUDE_ZERO_SHOT_BATCH_V3", "claude-sonnet-4-6_matched_single_user_prompt_v1_batch"),
}


def metrics(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * pr * rc / (4 * pr + rc) if pr + rc else 0.0
    return {"P": round(pr, 4), "R": round(rc, 4), "F2": round(f2, 4),
            "n": tp + fp + fn + tn, "n_pos": tp + fn}


def main():
    levels = {}
    for p in PROJECTS:
        with open(os.path.join(GT, p, "requirements.json"), encoding="utf-8") as f:
            levels[p] = {r["id"]: r["level"] for r in json.load(f)}

    def stratum(p, s, t):
        sl, tl = levels[p].get(s), levels[p].get(t)
        if sl == "parent" and tl == "standard":
            return "refinement"
        if sl == "standard" and tl == "child":
            return "subtask"
        return "other"

    out = {}
    for scoring in ["clean", "conservative"]:
        out[scoring] = {}
        print("\n" + "=" * 90)
        print(f"STRATIFIED EVALUATION BY LINK TYPE - {scoring.upper()} SCORING")
        print("=" * 90)
        print(f"{'Method':<20} {'Stratum':<11} {'macroF2':>8} {'microF2':>8} {'nProj':>6} {'pairs':>7} {'pos':>6}")
        for m, root in METHODS.items():
            buckets = defaultdict(lambda: defaultdict(lambda: ([], [])))
            for p in PROJECTS:
                f = os.path.join(root, f"{p}_predictions.json")
                if not os.path.exists(f):
                    print(f"  [MISSING] {f}")
                    continue
                with open(f, encoding="utf-8") as fh:
                    for e in json.load(fh):
                        st = stratum(p, e["source_id"], e["target_id"])
                        if st == "other":
                            continue
                        pred = e["prediction"]
                        if pred is None:
                            if scoring == "clean":
                                continue
                            pred = 0
                        P, L = buckets[st][p]
                        P.append(pred)
                        L.append(e["label"])
            out[scoring][m] = {}
            for st in ["refinement", "subtask"]:
                allP, allL, f2s, per_project = [], [], [], {}
                for p, (P, L) in sorted(buckets[st].items()):
                    if sum(L) == 0:
                        continue
                    allP += P
                    allL += L
                    pm = metrics(P, L)
                    f2s.append(pm["F2"])
                    per_project[p] = pm
                micro = metrics(allP, allL)
                macro = round(sum(f2s) / len(f2s), 4) if f2s else 0.0
                out[scoring][m][st] = {
                    "macro_f2": macro, "micro": micro,
                    "n_projects": len(f2s), "per_project": per_project,
                }
                print(f"{m:<20} {st:<11} {macro:>8.4f} {micro['F2']:>8.4f} "
                      f"{len(f2s):>6} {micro['n']:>7} {micro['n_pos']:>6}")

    dest = os.path.join(RESULTS, "stratified_by_link_type_v3.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: {dest}")


if __name__ == "__main__":
    main()
