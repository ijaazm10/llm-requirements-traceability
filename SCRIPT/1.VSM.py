"""
VSM Baseline â€” Hard 1:3 Pairs Evaluation
==========================================
Evaluates VSM on the same fixed 1:3 pairs used by BERT/RAG/LoRA.

"""

import json
import os
import time
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import precision_recall_fscore_support, fbeta_score

# ==================== CONFIG ====================
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = str(ROOT / "DATA" / ".GROUND_TRUTH")
OUTPUT_DIR = str(ROOT / "RESULTS")
PROJECTS  = ['AAH', 'BEAM', 'CB', 'FH',
             'JBIDE', 'KEYCLOAK', 'KOGITO', 'PROJQUAY']

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
# ================================================


def load_project(project_path):
    """Load requirements and build TF-IDF vectors."""
    req_file = os.path.join(project_path, "requirements.json")
    with open(req_file, 'r', encoding='utf-8') as f:
        reqs = json.load(f)

    id_to_req = {}
    for r in reqs:
        rid = r['id']
        summary     = (r.get('summary', '') or '').strip()
        description = (r.get('description', '') or '').strip()
        r['full_text'] = f"{summary}\n{description}".strip()
        id_to_req[rid] = r

    return reqs, id_to_req


def evaluate_fixed_pairs(pairs, tfidf_matrix, id_to_idx, threshold):
    """
    Evaluate VSM on hard pairs at a given threshold.

    For each (source, target) pair:
      - Compute cosine similarity of their TF-IDF vectors
      - Predict positive if sim >= threshold
      - Compare against the label
    """
    y_true = []
    y_pred = []

    for pair in pairs:
        src_id = pair['source_id']
        tgt_id = pair['target_id']
        label  = pair['label']

        if src_id not in id_to_idx or tgt_id not in id_to_idx:
            continue

        src_vec = tfidf_matrix[id_to_idx[src_id]]
        tgt_vec = tfidf_matrix[id_to_idx[tgt_id]]
        score = cos_sim(src_vec, tgt_vec)[0, 0]

        y_true.append(label)
        y_pred.append(1 if score >= threshold else 0)

    if not y_true:
        return 0.0, 0.0, 0.0, 0.0

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, average='binary', zero_division=0)

    return p, r, f1, f2


def evaluate_project(project_name):
    """Full pipeline for one project."""
    print(f"\n{'='*64}")
    print(f"  Project: {project_name}")
    print(f"{'='*64}")

    proj_path = os.path.join(BASE_DIR, project_name)
    start_time = time.time()

    reqs, id_to_req = load_project(proj_path)

    # Build TF-IDF on ALL requirements
    ids   = [r['id'] for r in reqs]
    texts = [r['full_text'] for r in reqs]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    id_to_idx = {rid: i for i, rid in enumerate(ids)}

    print(f"  Requirements: {len(reqs):,}  |  Vocab: {len(vectorizer.vocabulary_):,}")

    # Load fixed pairs
    splits_dir = os.path.join(proj_path, "splits")
    val_file  = os.path.join(splits_dir, "final_pairs_val.json")
    test_file = os.path.join(splits_dir, "final_pairs_test.json")

    with open(val_file, 'r', encoding='utf-8') as f:
        val_pairs = json.load(f)
    with open(test_file, 'r', encoding='utf-8') as f:
        test_pairs = json.load(f)

    val_pos = sum(1 for p in val_pairs if p['label'] == 1)
    val_neg = len(val_pairs) - val_pos
    test_pos = sum(1 for p in test_pairs if p['label'] == 1)
    test_neg = len(test_pairs) - test_pos

    print(f"  Val:  {val_pos:>5} pos / {val_neg:>5} neg  (ratio 1:{val_neg/val_pos:.1f})")
    print(f"  Test: {test_pos:>5} pos / {test_neg:>5} neg  (ratio 1:{test_neg/test_pos:.1f})")

    # Threshold tuning on validation pairs
    print(f"\n  Threshold tuning (optimising F2 on validation set):")
    print(f"  {'t':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'F2':>7}")
    print("  " + "-" * 42)

    best_f2 = 0.0
    best_t  = 0.0

    for t in THRESHOLDS:
        p, r, f1, f2 = evaluate_fixed_pairs(
            val_pairs, tfidf_matrix, id_to_idx, t)
        mark = ""
        if f2 > best_f2:
            best_f2, best_t = f2, t
            mark = "  <-"
        print(f"  {t:>5.2f}  {p:>7.4f}  {r:>7.4f}  {f1:>7.4f}  {f2:>7.4f}{mark}")

    print(f"\n  Best threshold: {best_t}  (Val F2: {best_f2:.4f})")

    # Final test evaluation
    p, r, f1, f2 = evaluate_fixed_pairs(
        test_pairs, tfidf_matrix, id_to_idx, best_t)

    elapsed = time.time() - start_time

    print(f"\n  {'='*56}")
    print(f"  TEST RESULTS (threshold={best_t:.2f}):")
    print(f"  Precision: {p:.4f}  |  Recall: {r:.4f}")
    print(f"  F1: {f1:.4f}  |  F2: {f2:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  {'='*56}")

    return {
        "project": project_name,
        "best_threshold": best_t,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "f2": float(f2),
        "time_seconds": float(elapsed)
    }


def main():
    print("=" * 72)
    print("VSM BASELINE â€” Hard 1:3 Pairs Evaluation")
    print("Same pairs as BERT/RAG/LoRA for direct comparison")
    print("=" * 72)

    all_results = []

    for project in PROJECTS:
        try:
            result = evaluate_project(project)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing {project}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n\n" + "=" * 72)
    print("FINAL SUMMARY â€” VSM Baseline (Fixed 1:3 Pairs)")
    print("=" * 72)
    print(f"{'Project':<12} {'P':>8} {'R':>8} {'F1':>8} {'F2':>8} {'Thresh':>8}")
    print("-" * 72)

    ps, rs, f1s, f2s = [], [], [], []
    for r in all_results:
        print(f"{r['project']:<12} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['f2']:>8.4f} "
              f"{r['best_threshold']:>8.2f}")
        ps.append(r['precision'])
        rs.append(r['recall'])
        f1s.append(r['f1'])
        f2s.append(r['f2'])

    if ps:
        print("-" * 72)
        print(f"{'Average':<12} {np.mean(ps):>8.4f} {np.mean(rs):>8.4f} "
              f"{np.mean(f1s):>8.4f} {np.mean(f2s):>8.4f}")
    print("=" * 72)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, "vsm_final_pairs_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_info": {
                "model": "VSM (TF-IDF + Cosine Similarity)",
                "evaluation_strategy": "fixed_1_to_3_pairs",
                "vectorizer": "TfidfVectorizer(stop_words='english')",
                "threshold_tuning": "validation_f2",
                "note": "Same fixed pairs as BERT/RAG/LoRA"
            },
            "results": all_results,
            "summary": {
                "avg_precision": float(np.mean(ps)) if ps else None,
                "avg_recall": float(np.mean(rs)) if rs else None,
                "avg_f1": float(np.mean(f1s)) if f1s else None,
                "avg_f2": float(np.mean(f2s)) if f2s else None,
                "num_projects": len(all_results)
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
