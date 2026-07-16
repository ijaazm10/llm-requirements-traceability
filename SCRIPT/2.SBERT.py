"""
Sentence-Transformer Embedding Baseline
========================================
Evaluates pretrained semantic similarity as a traceability baseline.

Method:
  1. Encode all requirements using all-mpnet-base-v2 (768-dim vectors)
  2. For each pair, compute cosine similarity between HLR and LLR vectors
  3. Tune threshold on validation pairs (maximize F2)
  4. Evaluate on test pairs with tuned threshold

This sits between VSM and LLM in the method progression:
  VSM (keyword matching) â†’ Embedding (semantic similarity) â†’ LLM (reasoning)

Key design decisions:
  - all-mpnet-base-v2: best general-purpose sentence-transformer, trained for
    cosine similarity via contrastive learning. 768-dim, 384 max tokens.
  - Concatenate summary + description per requirement (single vector per req)
  - Cosine similarity between HLR and LLR vectors (not concatenated pair)
- Threshold tuned on validation F2 (matches recall-weighted thesis metric)
  - Same fixed 1:3 test pairs as all other methods

Runs locally on GPU/CPU â€” does NOT use Ollama server.
Can run in parallel with LLM evaluation.

Author: Thesis Work
Date: 2026-03
"""

import json
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, fbeta_score

# ==================== CONFIG ====================
DATA_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/DATA/GROUND_TRUTH"
OUTPUT_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS"

# Model: best general-purpose sentence-transformer
# - 768-dimensional embeddings
# - Max 384 tokens (vs BERT-DRAFT's 64 token truncation)
# - Trained with contrastive learning for cosine similarity
EMBEDDING_MODEL = "all-mpnet-base-v2"

PROJECTS = ['AAH', 'BEAM', 'CB', 'FH', 'JBIDE', 'KEYCLOAK', 'KOGITO', 'PROJQUAY']

# Threshold search grid: 0.05 to 0.95 in steps of 0.05
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
# ================================================


def load_requirements(project_path):
    """
    Load all requirements and build:
      - id_to_text: maps requirement ID â†’ concatenated "summary + description"
      - id_list: ordered list of IDs (needed to align with embedding matrix)
    
    We concatenate summary and description because sentence-transformers
    encode the full text into a single vector. This captures both the
    brief intent (summary) and detailed context (description).
    """
    req_file = os.path.join(project_path, "requirements.json")
    with open(req_file, 'r', encoding='utf-8') as f:
        reqs = json.load(f)

    id_to_text = {}
    for r in reqs:
        rid = r['id']
        summary = (r.get('summary', '') or '').strip()
        description = (r.get('description', '') or '').strip()
        # Concatenate with newline separator
        # If description is empty, we just get the summary
        full_text = f"{summary}\n{description}".strip()
        id_to_text[rid] = full_text

    return id_to_text


def encode_requirements(model, id_to_text):
    """
    Encode all requirements into dense vectors.
    
    This is done ONCE per project â€” not per pair. With ~3000 requirements
    in the largest project (JBIDE), this takes a few seconds on GPU.
    
    Returns:
      - id_list: ordered list of requirement IDs
      - embeddings: numpy array of shape (n_requirements, 768)
      - id_to_idx: maps requirement ID â†’ row index in embeddings matrix
    
    The embeddings are L2-normalized by default in sentence-transformers,
    so cosine similarity = dot product (faster computation).
    """
    id_list = list(id_to_text.keys())
    texts = [id_to_text[rid] for rid in id_list]

    # batch_size=64 balances GPU memory and speed
    # show_progress_bar for visibility during encoding
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize â†’ cosine sim = dot product
    )

    id_to_idx = {rid: i for i, rid in enumerate(id_list)}

    return id_list, embeddings, id_to_idx


def load_pairs(pairs_file):
    """Load pairs from JSON file."""
    with open(pairs_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_pair_similarities(pairs, embeddings, id_to_idx):
    """
    Compute cosine similarity for each pair.
    
    For each (source_id, target_id) pair:
      1. Look up the source embedding vector
      2. Look up the target embedding vector
      3. Compute cosine similarity (dot product since vectors are normalized)
    
    Returns:
      - similarities: list of float scores
      - labels: list of ground truth labels (0 or 1)
      - valid_pairs: list of pairs that had valid embeddings
    
    Pairs where either requirement is missing from embeddings are skipped.
    """
    similarities = []
    labels = []
    valid_pairs = []
    skipped = 0

    for p in pairs:
        src_id = p['source_id']
        tgt_id = p['target_id']

        if src_id not in id_to_idx or tgt_id not in id_to_idx:
            skipped += 1
            continue

        # Get embedding vectors
        src_vec = embeddings[id_to_idx[src_id]]  # shape: (768,)
        tgt_vec = embeddings[id_to_idx[tgt_id]]  # shape: (768,)

        # Cosine similarity = dot product (vectors are L2-normalized)
        sim = float(np.dot(src_vec, tgt_vec))

        similarities.append(sim)
        labels.append(p['label'])
        valid_pairs.append(p)

    if skipped > 0:
        print(f"    âš  Skipped {skipped} pairs (missing embeddings)")

    return similarities, labels, valid_pairs


def evaluate_at_threshold(similarities, labels, threshold):
    """
    Classify pairs using a similarity threshold and compute metrics.
    
    If cosine_similarity >= threshold â†’ predict linked (1)
    If cosine_similarity <  threshold â†’ predict unlinked (0)
    
    This is identical to how VSM works, just with dense embeddings
    instead of sparse TF-IDF vectors.
    """
    y_true = labels
    y_pred = [1 if sim >= threshold else 0 for sim in similarities]

    if not y_true or sum(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, average='binary', zero_division=0)

    return float(p), float(r), float(f1), float(f2)


def tune_threshold(similarities, labels):
    """
    Find optimal threshold by maximizing F2 on validation data.
    
    F2 is used because recall-weighted trace recovery is the primary
    thesis metric. The threshold is therefore selected at the same
    operating point used for the main comparison.
    
    Returns: (best_threshold, best_f2, all_results)
    """
    best_f2 = 0.0
    best_t = 0.0
    all_results = []

    for t in THRESHOLDS:
        p, r, f1, f2 = evaluate_at_threshold(similarities, labels, t)
        all_results.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f1, 'f2': f2})

        if f2 > best_f2:
            best_f2 = f2
            best_t = t

    return best_t, best_f2, all_results


def evaluate_project(model, project_name):
    """
    Full pipeline for one project:
      1. Load requirements â†’ encode into vectors
      2. Load validation pairs â†’ tune threshold
      3. Load test pairs â†’ evaluate at tuned threshold
    """
    print(f"\n{'=' * 70}")
    print(f"  Project: {project_name}")
    print(f"{'=' * 70}")

    proj_path = os.path.join(DATA_DIR, project_name)
    start_time = time.time()

    # â”€â”€ Step 1: Encode all requirements â”€â”€
    print(f"  Loading and encoding requirements...")
    id_to_text = load_requirements(proj_path)
    id_list, embeddings, id_to_idx = encode_requirements(model, id_to_text)
    print(f"  Encoded {len(id_list)} requirements â†’ {embeddings.shape[1]}-dim vectors")

    # â”€â”€ Step 2: Tune threshold on validation pairs â”€â”€
    val_file = os.path.join(proj_path, "splits", "final_pairs_val.json")
    val_pairs = load_pairs(val_file)
    val_sims, val_labels, _ = compute_pair_similarities(val_pairs, embeddings, id_to_idx)

    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    print(f"  Val pairs: {len(val_labels)} ({val_pos} pos / {val_neg} neg)")

    print(f"\n  Threshold tuning (optimizing F2 on validation):")
    print(f"  {'t':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'F2':>7}")
    print(f"  {'-' * 42}")

    best_t, best_val_f2, val_results = tune_threshold(val_sims, val_labels)

    for vr in val_results:
        mark = "  <-" if vr['threshold'] == best_t else ""
        print(f"  {vr['threshold']:>5.2f}  {vr['precision']:>7.4f}  {vr['recall']:>7.4f}  "
              f"{vr['f1']:>7.4f}  {vr['f2']:>7.4f}{mark}")

    print(f"\n  Best threshold: {best_t}  (Val F2: {best_val_f2:.4f})")

    # â”€â”€ Step 3: Evaluate on test pairs â”€â”€
    test_file = os.path.join(proj_path, "splits", "final_pairs_test.json")
    test_pairs = load_pairs(test_file)
    test_sims, test_labels, _ = compute_pair_similarities(test_pairs, embeddings, id_to_idx)

    test_pos = sum(test_labels)
    test_neg = len(test_labels) - test_pos
    print(f"  Test pairs: {len(test_labels)} ({test_pos} pos / {test_neg} neg)")

    p, r, f1, f2 = evaluate_at_threshold(test_sims, test_labels, best_t)

    elapsed = time.time() - start_time

    # â”€â”€ Similarity distribution analysis â”€â”€
    # Useful for understanding how well the embeddings separate classes
    pos_sims = [s for s, l in zip(test_sims, test_labels) if l == 1]
    neg_sims = [s for s, l in zip(test_sims, test_labels) if l == 0]
    sim_gap = np.mean(pos_sims) - np.mean(neg_sims)

    print(f"\n  {'=' * 56}")
    print(f"  TEST RESULTS (threshold={best_t:.2f}):")
    print(f"  Precision: {p:.4f}  |  Recall: {r:.4f}")
    print(f"  F1: {f1:.4f}  |  F2: {f2:.4f}")
    print(f"  Similarity gap: {sim_gap:.4f} (pos_mean={np.mean(pos_sims):.4f}, "
          f"neg_mean={np.mean(neg_sims):.4f})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  {'=' * 56}")

    return {
        'project': project_name,
        'n_requirements': len(id_list),
        'embedding_dim': int(embeddings.shape[1]),
        'best_threshold': float(best_t),
        'val_pairs': len(val_labels),
        'test_pairs': len(test_labels),
        'test_positives': test_pos,
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'f2': float(f2),
        'similarity_stats': {
            'pos_mean': float(np.mean(pos_sims)),
            'pos_std': float(np.std(pos_sims)),
            'neg_mean': float(np.mean(neg_sims)),
            'neg_std': float(np.std(neg_sims)),
            'gap': float(sim_gap),
        },
        'time_seconds': round(elapsed, 1),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 75)
    print("SENTENCE-TRANSFORMER EMBEDDING BASELINE")
    print("=" * 75)
    print(f"Model:      {EMBEDDING_MODEL}")
    print(f"Projects:   {PROJECTS}")
    print(f"Method:     Cosine similarity with threshold tuning")
    print(f"Threshold:  Tuned on validation F2, evaluated on test")
    print(f"Split:      Same fixed 1:3 pairs as all other methods")

    # â”€â”€ Load model â”€â”€
    print(f"\nLoading sentence-transformer model...")
    load_start = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Model loaded in {time.time() - load_start:.1f}s")
    print(f"  Max seq length: {model.max_seq_length}")
    print(f"  Embedding dim:  {model.get_sentence_embedding_dimension()}")

    # â”€â”€ Evaluate all projects â”€â”€
    all_results = []

    for project in PROJECTS:
        try:
            result = evaluate_project(model, project)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing {project}: {e}")
            import traceback
            traceback.print_exc()

    # â”€â”€ Summary table â”€â”€
    print("\n\n" + "=" * 85)
    print(f"FINAL SUMMARY â€” Embedding Baseline ({EMBEDDING_MODEL})")
    print("=" * 85)
    print(f"{'Project':<12} {'P':>8} {'R':>8} {'F1':>8} {'F2':>8} "
          f"{'Thresh':>8} {'SimGap':>8} {'Time':>6}")
    print("-" * 85)

    ps, rs, f1s, f2s = [], [], [], []
    for r in all_results:
        print(f"{r['project']:<12} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['f2']:>8.4f} "
              f"{r['best_threshold']:>8.2f} {r['similarity_stats']['gap']:>8.4f} "
              f"{r['time_seconds']:>5.1f}s")
        ps.append(r['precision'])
        rs.append(r['recall'])
        f1s.append(r['f1'])
        f2s.append(r['f2'])

    if ps:
        print("-" * 85)
        print(f"{'MACRO AVG':<12} {np.mean(ps):>8.4f} {np.mean(rs):>8.4f} "
              f"{np.mean(f1s):>8.4f} {np.mean(f2s):>8.4f}")

    # â”€â”€ Comparison with other baselines â”€â”€
    print(f"\n{'â”€' * 85}")
    print("COMPARISON WITH OTHER METHODS (Macro F1 on 1:3 fixed pairs):")
    print(f"  VSM (TF-IDF):          F1 = 0.6162")
    print(f"  Embedding (this):      F1 = {np.mean(f1s):.4f}")
    print(f"  BERT-DRAFT (frozen):   F1 = 0.4347")
    print(f"  LLM Zero-Shot (old):   F1 = 0.6839  (gemma3:27b)")
    print(f"  LLM Zero-Shot (new):   F1 = running  (gemma3:12b)")

    print("=" * 85)

    # â”€â”€ Save results â”€â”€
    results_file = os.path.join(OUTPUT_DIR, "sbert_final_pairs_results.json")
    output = {
        'model': EMBEDDING_MODEL,
        'embedding_dim': model.get_sentence_embedding_dimension(),
        'max_seq_length': model.max_seq_length,
        'method': 'cosine_similarity_with_threshold',
        'threshold_tuning': 'validation_f2',
        'evaluation': 'fixed_1_to_3_test_pairs',
        'results': all_results,
        'summary': {
            'avg_precision': float(np.mean(ps)) if ps else None,
            'avg_recall': float(np.mean(rs)) if rs else None,
            'avg_f1': float(np.mean(f1s)) if f1s else None,
            'avg_f2': float(np.mean(f2s)) if f2s else None,
            'num_projects': len(all_results),
        }
    }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nâœ“ Results saved to: {results_file}")
    total_time = sum(r['time_seconds'] for r in all_results)
    print(f"âœ“ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
 

if __name__ == "__main__":
    main()
