
"""
Model Selection V3 — Hard-Negative Pairs, 9 Models
====================================================
Systematic evaluation of 9 LLMs on stratified hard-negative pairs
to select the optimal model for all subsequent experiments.

Changes from V2:
  - Uses final_pairs_val.json instead of fixed_pairs_val.json
  - 9 models (added gemma4:31b)
  - Clean prompt (no JIRA mention)
  - Same 3 validation projects, up to 400 pairs each (BEAM has 360)

Selection Grid:
  - 9 models across 5 families (Google, Alibaba, Meta, Microsoft, Mistral)
  - 3 size tiers: Small (7-8B), Medium (12-14B), Large (27-31B)
  - Hypotheses:
      H1: Size scaling    → gemma3:12b vs gemma3:27b
      H2: Code-specialist → qwen2.5-coder:14b vs phi4:14b
      H3: Architecture    → 5 families prevent vendor bias
      H4: Generation leap → gemma3:12b vs gemma4:31b

Projects:
  - BEAM      — smallest viable
  - KEYCLOAK  — medium, mixed link types
  - JBIDE     — largest

Evaluation:
  - Up to 400 pairs per project from final_pairs_val.json (max 100 pos + 300 neg)
  - Deterministic: temp=0, num_predict=20, format=json
  - Primary metric: F2 (recall-weighted)
  - Total: 10,800 evaluations (9 models × 3 projects × 400 pairs)

Author: Thesis Work
Date: 2026-04-18
"""

import requests
import json
import time
import random
import re
import os
import sys
from datetime import datetime

# ==================== CONFIG ====================
OLLAMA_URL = "https://ymir-api.ifak.eu"
DATA_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/DATA/GROUND_TRUTH"
OUTPUT_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS"

SELECTION_PROJECTS = ["BEAM", "KEYCLOAK", "JBIDE"]

MODELS_TO_TEST = [
    # ── Small (7-8B) ──
    "llama3.1:8b",
    "qwen3:8b",

    # ── Medium (12-14B) ──
    "gemma3:12b",
    "phi4:14b",
    "qwen2.5-coder:14b-instruct-fp16",
    "ministral-3:14b",

    # ── Large (27-31B) ──
    "gemma3:27b",
    "qwen3-coder:30b",
    "gemma4:31b",              # NEW — Gemma 4
]

# Model-specific system prompts
MODEL_SYSTEM_PROMPTS = {
    "qwen3:8b": "You are a classifier. Respond ONLY with valid JSON. No explanation. /no_think",
}
DEFAULT_SYSTEM_PROMPT = "You are a classifier. Respond ONLY with valid JSON. No explanation."

SAMPLE_SIZE_PER_PROJECT = 400  # max 100 pos + 300 neg
SEED = 42
# ================================================


# =================================================
# 1. DATA LOADING
# =================================================

def load_requirements(project_path):
    req_file = os.path.join(project_path, "requirements.json")
    with open(req_file, 'r', encoding='utf-8') as f:
        reqs = json.load(f)
    id_map = {}
    for r in reqs:
        summary = (r.get('summary', '') or '').strip()
        description = (r.get('description', '') or '').strip()
        id_map[r['id']] = (summary, description)
    return id_map


def load_hard_validation_pairs(project_path, id_map, sample_size=400):
    """
    Load validation pairs from final_pairs_val.json with stratified sampling.
    Target: max 100 positive + 300 negative = 400 pairs per project.
    """
    pairs_file = os.path.join(project_path, "splits", "final_pairs_val.json")

    if not os.path.exists(pairs_file):
        print(f"    ✗ {pairs_file} not found!")
        print(f"      Run generate_hard_pairs.py first.")
        sys.exit(1)

    with open(pairs_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    enriched = []
    for p in pairs:
        src_id = p['source_id']
        tgt_id = p['target_id']

        if src_id not in id_map or tgt_id not in id_map:
            continue

        hlr_sum, hlr_desc = id_map[src_id]
        llr_sum, llr_desc = id_map[tgt_id]

        enriched.append({
            'hlr_summary': hlr_sum,
            'hlr_description': hlr_desc,
            'llr_summary': llr_sum,
            'llr_description': llr_desc,
            'label': p['label'],
            'source_id': src_id,
            'target_id': tgt_id,
        })

    # Stratified sampling: maintain 1:3 ratio
    positives = [p for p in enriched if p['label'] == 1]
    negatives = [p for p in enriched if p['label'] == 0]

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_pos_target = sample_size // 4   # 100
    n_neg_target = sample_size - n_pos_target  # 300

    n_pos = min(n_pos_target, len(positives))
    n_neg = min(n_pos * 3, len(negatives))

    sampled = positives[:n_pos] + negatives[:n_neg]
    rng.shuffle(sampled)

    return sampled


# =================================================
# 2. PROMPT — Clean champion prompt (no JIRA)
# =================================================

def create_prompt(pair):
    """The champion zero-shot prompt — tool-agnostic, no JIRA mention."""
    hlr_desc = pair['hlr_description'] if pair['hlr_description'] else 'N/A'
    llr_desc = pair['llr_description'] if pair['llr_description'] else 'N/A'

    return f"""You are analyzing hierarchical software requirements for cross level traceability.
Determine whether a traceability link exists between a High-Level Requirement (HLR) and a Low-Level Requirement (LLR).
A link exists if the LLR implements, refines, or decomposes the HLR.
High-Level Requirement:
  Summary: {pair['hlr_summary']}
  Description: {hlr_desc}
Low-Level Requirement:
  Summary: {pair['llr_summary']}
  Description: {llr_desc}
Return only:
{{"is_linked": true}}
or
{{"is_linked": false}}
No additional text."""


# =================================================
# 3. LLM QUERY & PARSING
# =================================================

def query_ollama(model_name, prompt):
    url = f"{OLLAMA_URL}/api/generate"
    system_prompt = MODEL_SYSTEM_PROMPTS.get(model_name, DEFAULT_SYSTEM_PROMPT)

    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 20,
            "repeat_penalty": 1.0,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get('response', '').strip()
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def parse_response(response_text):
    if not response_text:
        return None

    # Strip <think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    if not cleaned or cleaned == response_text:
        cleaned = re.sub(r'<think>.*', '', response_text, flags=re.DOTALL).strip()
    text = cleaned if cleaned else response_text

    # JSON parse
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            val = data.get('is_linked')
            if val is not None:
                return 1 if val else 0
    except (json.JSONDecodeError, ValueError):
        pass

    # Keyword fallback
    lower = text.lower()
    if '"is_linked": true' in lower or '"is_linked":true' in lower:
        return 1
    elif '"is_linked": false' in lower or '"is_linked":false' in lower:
        return 0
    return None


# =================================================
# 4. METRICS
# =================================================

def compute_metrics(predictions, labels):
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'f1': round(f1, 4), 'f2': round(f2, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# =================================================
# 5. CHECKPOINTING
# =================================================

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_file, predictions):
    with open(checkpoint_file, 'w') as f:
        json.dump(predictions, f)


# =================================================
# 6. EVALUATION
# =================================================

def evaluate_model_on_project(model_name, project_name, pairs, checkpoint_dir):
    """Evaluate one model on one project's hard-negative validation pairs."""
    # Checkpoint per model+project
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    checkpoint_file = os.path.join(
        checkpoint_dir, f"{safe_model_name}_{project_name}_predictions.json")

    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        start_idx = len(checkpoint)
        predictions_log = checkpoint
        print(f"      Resuming from pair {start_idx}/{len(pairs)}")
    else:
        start_idx = 0
        predictions_log = []

    start_time = time.time()

    for i in range(start_idx, len(pairs)):
        pair = pairs[i]

        if (i + 1) % 25 == 0 or i == start_idx:
            elapsed = time.time() - start_time
            done = i - start_idx + 1
            rate = elapsed / done if done > 0 else 0
            remaining = rate * (len(pairs) - i - 1)
            print(f"      [{project_name}] {i+1}/{len(pairs)}  "
                  f"(~{remaining/60:.1f}min remaining)", end='\r')

        prompt = create_prompt(pair)
        response = query_ollama(model_name, prompt)

        if response is None:
            time.sleep(2)
            response = query_ollama(model_name, prompt)

        pred = parse_response(response)

        predictions_log.append({
            'source_id': pair['source_id'],
            'target_id': pair['target_id'],
            'label': pair['label'],
            'prediction': pred,
        })

        # Save checkpoint every 50 pairs
        if (i + 1) % 50 == 0:
            save_checkpoint(checkpoint_file, predictions_log)

    # Final save
    save_checkpoint(checkpoint_file, predictions_log)

    elapsed = time.time() - start_time

    # Compute metrics
    all_preds, all_labels = [], []
    clean_preds, clean_labels = [], []
    failed = 0

    for entry in predictions_log:
        all_labels.append(entry['label'])
        if entry['prediction'] is None:
            failed += 1
            all_preds.append(0)
        else:
            all_preds.append(entry['prediction'])
            clean_preds.append(entry['prediction'])
            clean_labels.append(entry['label'])

    metrics_clean = compute_metrics(clean_preds, clean_labels) if clean_preds else {
        'precision': 0, 'recall': 0, 'f1': 0, 'f2': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
    }
    metrics_all = compute_metrics(all_preds, all_labels)

    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos

    print(f"      [{project_name}] DONE  F1={metrics_clean['f1']:.4f}  "
          f"F2={metrics_clean['f2']:.4f}  P={metrics_clean['precision']:.4f}  "
          f"R={metrics_clean['recall']:.4f}  "
          f"Fail={failed}/{len(pairs)}  Time={elapsed:.1f}s" + " " * 20)

    return {
        'project': project_name,
        'n_pairs': len(pairs),
        'n_pos': n_pos,
        'n_neg': n_neg,
        'failed_parses': failed,
        'failed_pct': round(100 * failed / len(pairs), 1) if pairs else 0,
        'time_seconds': round(elapsed, 1),
        'time_per_pair': round(elapsed / len(pairs), 2) if pairs else 0,
        'metrics_clean': metrics_clean,
        'metrics_all': metrics_all,
    }


# =================================================
# 7. MAIN
# =================================================

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 85)
    print("MODEL SELECTION V3 — HARD-NEGATIVE PAIRS, 9 MODELS")
    print("=" * 85)
    print(f"Timestamp:   {timestamp}")
    print(f"Projects:    {SELECTION_PROJECTS}")
    print(f"Models:      {len(MODELS_TO_TEST)}")
    print(f"Pairs/proj:  up to {SAMPLE_SIZE_PER_PROJECT} (max 100 pos + 300 neg, seed={SEED})")
    print(f"Pair source: final_pairs_val.json (stratified negatives)")
    print(f"Total evals: {SAMPLE_SIZE_PER_PROJECT * len(SELECTION_PROJECTS) * len(MODELS_TO_TEST)}")
    print(f"Config:      temp=0, num_predict=20, format=json")
    print(f"Prompt:      Clean champion (no JIRA mention)")
    print(f"Primary:     F2 (recall-weighted)")
    print()
    print("Models under test:")
    for i, m in enumerate(MODELS_TO_TEST, 1):
        tag = ""
        if m in MODEL_SYSTEM_PROMPTS:
            tag = " [/no_think]"
        if m == "gemma4:31b":
            tag = " [NEW — Gemma 4]"
        print(f"  {i:>2}. {m}{tag}")

    # ── Load data ──
    print(f"\n{'─' * 85}")
    print("Loading hard-negative validation data...")
    project_data = {}
    for project in SELECTION_PROJECTS:
        project_path = os.path.join(DATA_DIR, project)
        id_map = load_requirements(project_path)
        pairs = load_hard_validation_pairs(project_path, id_map, SAMPLE_SIZE_PER_PROJECT)
        project_data[project] = pairs
        n_pos = sum(1 for p in pairs if p['label'] == 1)
        n_neg = len(pairs) - n_pos
        print(f"  {project}: {len(pairs)} pairs ({n_pos} pos / {n_neg} neg)")

    total_pairs_per_model = sum(len(v) for v in project_data.values())
    print(f"  Total per model: {total_pairs_per_model}")



    # ── Evaluate all models ──
    all_results = {}

    for model_idx, model_name in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'=' * 85}")
        print(f"  [{model_idx}/{len(MODELS_TO_TEST)}] MODEL: {model_name}")
        if model_name in MODEL_SYSTEM_PROMPTS:
            print(f"  System prompt: /no_think")
        if model_name == "gemma4:31b":
            print(f"  ★ NEW MODEL — Gemma 4 generation")
        print(f"{'=' * 85}")

        model_results = []
        for project in SELECTION_PROJECTS:
            result = evaluate_model_on_project(
                model_name, project, project_data[project], checkpoint_dir)
            model_results.append(result)

        # Aggregate
        total_failed = sum(r['failed_parses'] for r in model_results)
        total_time = sum(r['time_seconds'] for r in model_results)
        total_n = sum(r['n_pairs'] for r in model_results)

        macro_f1 = sum(r['metrics_clean']['f1'] for r in model_results) / len(model_results)
        macro_f2 = sum(r['metrics_clean']['f2'] for r in model_results) / len(model_results)
        macro_p = sum(r['metrics_clean']['precision'] for r in model_results) / len(model_results)
        macro_r = sum(r['metrics_clean']['recall'] for r in model_results) / len(model_results)

        all_results[model_name] = {
            'per_project': model_results,
            'macro_clean': {
                'precision': round(macro_p, 4),
                'recall': round(macro_r, 4),
                'f1': round(macro_f1, 4),
                'f2': round(macro_f2, 4),
            },
            'total_failed': total_failed,
            'total_failed_pct': round(100 * total_failed / total_n, 1) if total_n else 0,
            'total_time': round(total_time, 1),
            'avg_time_per_pair': round(total_time / total_n, 2) if total_n else 0,
        }

        print(f"\n  ── {model_name} SUMMARY ──")
        print(f"    Macro F2: {macro_f2:.4f}  |  F1: {macro_f1:.4f}")
        print(f"    Macro P:  {macro_p:.4f}  |  R:  {macro_r:.4f}")
        print(f"    Failed:   {total_failed}/{total_n} ({100*total_failed/total_n:.1f}%)")
        print(f"    Speed:    {total_time/total_n:.2f}s/pair  ({total_time:.1f}s total)")

    # ── Final Rankings ──
    print("\n\n" + "=" * 95)
    print("FINAL RANKINGS — HARD-NEGATIVE PAIRS (sorted by Macro F2)")
    print("=" * 95)
    print(f"\n{'Rank':<5} {'Model':<40} {'P':>7} {'R':>7} {'F1':>7} {'F2':>7} "
          f"{'Fail%':>6} {'s/pair':>7}")
    print("-" * 95)

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['macro_clean']['f2'], reverse=True)

    for rank, (model_name, r) in enumerate(ranked, 1):
        m = r['macro_clean']
        marker = "  ◄ BEST" if rank == 1 else ""
        print(f"{rank:<5} {model_name:<40} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {m['f2']:>7.4f} "
              f"{r['total_failed_pct']:>5.1f}% {r['avg_time_per_pair']:>6.2f}s{marker}")




    # ── Per-Project Breakdown ──
    print(f"\n{'─' * 95}")
    print("PER-PROJECT BREAKDOWN (sorted by F2):")

    for project in SELECTION_PROJECTS:
        print(f"\n  {project}:")
        print(f"    {'Model':<40} {'F2':>7} {'F1':>7} {'P':>7} {'R':>7} {'Fail':>6}")
        print(f"    {'-'*74}")

        proj_ranked = []
        for model_name, r in all_results.items():
            proj_r = next(p for p in r['per_project'] if p['project'] == project)
            proj_ranked.append((model_name, proj_r))
        proj_ranked.sort(key=lambda x: x[1]['metrics_clean']['f2'], reverse=True)

        for model_name, proj_r in proj_ranked:
            mc = proj_r['metrics_clean']
            print(f"    {model_name:<40} {mc['f2']:>7.4f} {mc['f1']:>7.4f} "
                  f"{mc['precision']:>7.4f} {mc['recall']:>7.4f} "
                  f"{proj_r['failed_parses']:>3}/{proj_r['n_pairs']}")

    # ── Champion ──
    best_model = ranked[0][0]
    best_r = ranked[0][1]

    print(f"\n{'=' * 95}")
    print(f"CHAMPION MODEL: {best_model}")
    print(f"  Highest Macro F2: {best_r['macro_clean']['f2']:.4f}")
    print("=" * 95)
    est_full_pairs = 10155
    est_time_hrs = (est_full_pairs * best_r['avg_time_per_pair']) / 3600
    print(f"\n  Estimated full evaluation time (10,155 test pairs): {est_time_hrs:.1f} hours")
    print("=" * 95)

    # ── Save Results ──
    output = {
        'timestamp': timestamp,
        'evaluation_type': 'final_negative_pairs',
        'pair_source': 'final_pairs_val.json',
        'prompt': 'clean_champion_no_jira',
        'config': {
            'selection_projects': SELECTION_PROJECTS,
            'models_tested': MODELS_TO_TEST,
            'sample_size_per_project': SAMPLE_SIZE_PER_PROJECT,
            'actual_pairs_per_project': {p: len(project_data[p])
                                          for p in SELECTION_PROJECTS},
            'seed': SEED,
            'generation': {
                'temperature': 0.0, 'top_p': 1.0,
                'num_predict': 20, 'format': 'json',
                'repeat_penalty': 1.0,
            },
        },
        'results': {},
        'ranking': [],
        'recommendation': best_model,
    }

    for model_name, r in all_results.items():
        output['results'][model_name] = r

    for rank, (model_name, r) in enumerate(ranked, 1):
        output['ranking'].append({
            'rank': rank,
            'model': model_name,
            'macro_f2': r['macro_clean']['f2'],
            'macro_f1': r['macro_clean']['f1'],
            'macro_precision': r['macro_clean']['precision'],
            'macro_recall': r['macro_clean']['recall'],
            'failed_pct': r['total_failed_pct'],
            'avg_time_per_pair': r['avg_time_per_pair'],
        })

    results_file = os.path.join(OUTPUT_DIR, "model_selection_v3_hard_results.json")
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Checkpoints in: {checkpoint_dir}/")
    print(f"✓ Total time: {sum(r['total_time'] for r in all_results.values())/60:.1f} minutes")


if __name__ == "__main__":
    main()
