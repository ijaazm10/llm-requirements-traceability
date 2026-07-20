"""
Zero-Shot — LOCAL H100 Inference (Qwen3 Top-K Hard Negatives)
=============================================================
Runs zero-shot Gemma 4 31B LOCALLY on H100.
Evaluates on final_pairs_test.json (Qwen3 top-K deployment-realistic negatives).

All RQ3 deployment metrics captured:
  - Inference latency (ms/pair, s/pair)
  - Token throughput (tok/s, total input/generated tokens)
  - Peak VRAM usage
  - Truncation count
  - Per-pair token counts (avg input, avg generated)

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u zero_shot_h100.py
  CUDA_VISIBLE_DEVICES=0 python -u zero_shot_h100.py --projects AAH BEAM

Author: Thesis Work
Date: 2026-05-03
"""

import json
import os
import sys
import time
import gc
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# ==================== CONFIG ====================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT / "DATA" / ".GROUND_TRUTH")
OUTPUT_DIR = str(ROOT / "RESULTS" / "ZERO_SHOT_QWEN_HARD")

BASE_MODEL = "unsloth/gemma-4-31B-it"

ALL_PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]

MAX_NEW_TOKENS = 20
MAX_SEQ_LENGTH = 3072     # Gemma 4 supports 256K, but our prompts are short
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


def load_test_pairs(project_path, id_map):
    """Load final_pairs_test.json (Qwen3 top-K hard negatives)."""
    pairs_file = os.path.join(project_path, "splits", "final_pairs_test.json")
    if not os.path.exists(pairs_file):
        print(f"    ERROR: {pairs_file} not found!")
        return []
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
            'source_id': src_id, 'target_id': tgt_id,
            'hlr_summary': hlr_sum,
            'hlr_description': hlr_desc if hlr_desc else 'N/A',
            'llr_summary': llr_sum,
            'llr_description': llr_desc if llr_desc else 'N/A',
            'label': p['label'],
        })
    return enriched


# =================================================
# 2. PROMPT & LLM
# =================================================

def create_prompt_text(pair):
    """Same prompt as LoRA eval and Ollama zero-shot."""
    return f"""You are analyzing hierarchical software requirements for cross level traceability.
Determine whether a traceability link exists between a High-Level Requirement (HLR) and a Low-Level Requirement (LLR).
A link exists if the LLR implements, refines, or decomposes the HLR.
High-Level Requirement:
  Summary: {pair['hlr_summary']}
  Description: {pair['hlr_description']}
Low-Level Requirement:
  Summary: {pair['llr_summary']}
  Description: {pair['llr_description']}
Return only:
{{"is_linked": true}}
or
{{"is_linked": false}}
No additional text."""


def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"  Loading LLM: {BASE_MODEL}...")
    print(f"  Using 4-bit quantization (NF4) to fit in VRAM")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 4-bit NF4 quantization — reduces 31B from ~58 GB to ~18 GB
    # Comparable to Ollama's Q4_K_M used in model selection
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # Nested quantization for extra savings
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    vram = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  ✓ LLM ready ({BASE_MODEL}). VRAM: {vram:.1f} GB (4-bit)")
    return model, tokenizer


def generate_response(model, tokenizer, prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)  # Disable thinking to avoid CoT overhead
    inputs = tokenizer(input_text, return_tensors="pt",
                       truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, use_cache=True,
        )

    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response, len(generated), input_len


def parse_response(response_text):
    if not response_text:
        return None
    text = response_text.strip()
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
    lower = text.lower()
    if '"is_linked": true' in lower or '"is_linked":true' in lower:
        return 1
    elif '"is_linked": false' in lower or '"is_linked":false' in lower:
        return 0
    return None


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
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'f1': round(f1, 4), 'f2': round(f2, 4),
        'accuracy': round(accuracy, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# =================================================
# 3. EVALUATION
# =================================================

def compute_conservative_metrics(predictions_log):
    """
    Conservative metrics: parse failures count as WRONG predictions.
    - If true label=1 and prediction=None → FN (missed a positive)
    - If true label=0 and prediction=None → TN (conservative: assume correct rejection)
    This gives a lower bound on recall and F1.
    """
    all_preds = []
    all_labels = []
    for e in predictions_log:
        label = e['label']
        pred = e['prediction']
        if pred is None:
            # Parse failure → treat as negative prediction (conservative)
            pred = 0
        all_preds.append(pred)
        all_labels.append(label)
    return compute_metrics(all_preds, all_labels)


def evaluate_project(model, tokenizer, test_pairs, project_name):
    checkpoint_file = os.path.join(OUTPUT_DIR, f"{project_name}_predictions.json")

    # Resume from checkpoint
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            predictions_log = json.load(f)
        start_idx = len(predictions_log)
        if start_idx >= len(test_pairs):
            print(f"    ✓ Already evaluated ({start_idx} pairs)")
            print(f"    ⚠ WARNING: Timing data not available from checkpoint.")
            print(f"    ⚠ For accurate RQ3 deployment metrics, delete checkpoint and re-run.")

            clean_preds = [e['prediction'] for e in predictions_log if e['prediction'] is not None]
            clean_labels = [e['label'] for e in predictions_log if e['prediction'] is not None]
            failed = sum(1 for e in predictions_log if e['prediction'] is None)
            metrics = compute_metrics(clean_preds, clean_labels)
            metrics_conservative = compute_conservative_metrics(predictions_log)

            # Recover token counts and timing from per-pair checkpoint data
            total_input = sum(e.get('input_tokens', 0) for e in predictions_log)
            total_gen = sum(e.get('gen_tokens', 0) for e in predictions_log)
            total_time_ms = sum(e.get('gen_time_ms', 0) for e in predictions_log)
            total_time_s = total_time_ms / 1000.0

            return {
                'project': project_name, 'n_pairs': len(test_pairs),
                'n_pos': sum(e['label'] for e in predictions_log),
                'n_neg': len(predictions_log) - sum(e['label'] for e in predictions_log),
                'failed_parses': failed,
                'time_seconds': round(total_time_s, 1),
                'time_per_pair': round(total_time_s / len(test_pairs), 3),
                'metrics': metrics,
                'metrics_conservative': metrics_conservative,
                'deployment_metrics': {
                    'note': 'timing recovered from per-pair gen_time_ms in checkpoint',
                    'total_input_tokens': total_input,
                    'total_generated_tokens': total_gen,
                    'avg_input_tokens': round(total_input / max(len(test_pairs), 1), 1),
                    'avg_gen_tokens': round(total_gen / max(len(test_pairs), 1), 1),
                    'avg_gen_ms': round(total_time_ms / max(len(test_pairs), 1), 2),
                    'tokens_per_second': round(total_gen / max(total_time_s, 0.001), 1),
                },
            }
        print(f"    Resuming from pair {start_idx}/{len(test_pairs)}")
    else:
        start_idx = 0
        predictions_log = []

    # Accumulate token counts from PREVIOUSLY checkpointed pairs too
    total_generated_tokens = sum(e.get('gen_tokens', 0) for e in predictions_log)
    total_input_tokens = sum(e.get('input_tokens', 0) for e in predictions_log)
    total_gen_time = sum(e.get('gen_time_ms', 0) for e in predictions_log) / 1000.0
    truncated_count = 0
    torch.cuda.reset_peak_memory_stats(0)

    start_time = time.time()

    for i in range(start_idx, len(test_pairs)):
        pair = test_pairs[i]

        if (i + 1) % 25 == 0 or i == start_idx:
            elapsed = time.time() - start_time
            done = i - start_idx + 1
            rate = elapsed / done if done > 0 else 0
            remaining = rate * (len(test_pairs) - i - 1)
            print(f"    [{project_name}] {i+1}/{len(test_pairs)}  "
                  f"(~{remaining/60:.1f}min remaining)", end='\r')

        prompt_text = create_prompt_text(pair)

        # Check truncation
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs_check = tokenizer(input_text, return_tensors="pt", truncation=False)
        if inputs_check['input_ids'].shape[1] > MAX_SEQ_LENGTH:
            truncated_count += 1

        t0 = time.time()
        response, n_tokens, n_input = generate_response(model, tokenizer, prompt_text)
        total_input_tokens += n_input
        t1 = time.time()
        gen_time_this = t1 - t0
        total_gen_time += gen_time_this
        total_generated_tokens += n_tokens

        pred = parse_response(response)

        predictions_log.append({
            'source_id': pair['source_id'],
            'target_id': pair['target_id'],
            'label': pair['label'],
            'prediction': pred,
            'response': response[:100],
            'input_tokens': n_input,
            'gen_tokens': n_tokens,
            'gen_time_ms': round(gen_time_this * 1000, 1),
        })

        if (i + 1) % 50 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(predictions_log, f)

    with open(checkpoint_file, 'w') as f:
        json.dump(predictions_log, f)

    elapsed_wall = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
    n_all = len(predictions_log)  # total pairs (resumed + new)

    # Primary metrics (excluding parse failures)
    clean_preds = [e['prediction'] for e in predictions_log if e['prediction'] is not None]
    clean_labels = [e['label'] for e in predictions_log if e['prediction'] is not None]
    failed = sum(1 for e in predictions_log if e['prediction'] is None)
    metrics = compute_metrics(clean_preds, clean_labels) if clean_preds else {
        'precision': 0, 'recall': 0, 'f1': 0, 'f2': 0, 'accuracy': 0,
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
    }

    # Conservative metrics (parse failures → predicted negative)
    metrics_conservative = compute_conservative_metrics(predictions_log)

    n_pos = sum(e['label'] for e in predictions_log)
    n_neg = n_all - n_pos
    # Use total_gen_time which includes ALL pairs (resumed + new)
    tokens_per_sec = total_generated_tokens / total_gen_time if total_gen_time > 0 else 0

    print(f"    [{project_name}] DONE  |  F1={metrics['f1']:.4f}  "
          f"F2={metrics['f2']:.4f}  P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"Failed={failed}/{n_all}  Time={elapsed_wall:.1f}s"
          + " " * 20)
    if failed > 0:
        print(f"    Conservative (failures=neg): F1={metrics_conservative['f1']:.4f}  "
              f"R={metrics_conservative['recall']:.4f}")
    print(f"    Deployment: {tokens_per_sec:.1f} gen_tok/s  Peak VRAM={peak_vram:.1f}GB  "
          f"Truncated={truncated_count}/{n_all}")

    return {
        'project': project_name,
        'n_pairs': n_all, 'n_pos': n_pos, 'n_neg': n_neg,
        'failed_parses': failed,
        'time_seconds': round(total_gen_time, 1),  # total across ALL pairs
        'time_per_pair': round(total_gen_time / max(n_all, 1), 3),
        'metrics': metrics,
        'metrics_conservative': metrics_conservative,
        'deployment_metrics': {
            'generated_tokens_per_second': round(tokens_per_sec, 1),
            'total_generated_tokens': total_generated_tokens,
            'total_input_tokens': total_input_tokens,
            'peak_inference_vram_gb': round(peak_vram, 2),
            'avg_gen_ms': round(1000 * total_gen_time / max(n_all, 1), 2),
            'truncated_pairs': truncated_count,
            'avg_input_tokens': round(total_input_tokens / max(n_all, 1), 1),
            'avg_gen_tokens': round(total_generated_tokens / max(n_all, 1), 1),
        },
    }


# =================================================
# 4. MAIN
# =================================================

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    projects = ALL_PROJECTS
    if '--projects' in sys.argv:
        idx = sys.argv.index('--projects')
        projects = [p for p in sys.argv[idx + 1:] if p in ALL_PROJECTS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 85)
    print("ZERO-SHOT — LOCAL H100 (Gemma 4 31B, Qwen3 Top-K Hard Negatives)")
    print("=" * 85)
    print(f"Timestamp:     {timestamp}")
    print(f"LLM:           {BASE_MODEL} (LOCAL)")
    print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print(f"VRAM:          {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print(f"Projects:      {projects}")
    print(f"Test data:     final_pairs_test.json (Qwen3 top-K)")
    print(f"Neg strategy:  Qwen3-ranked disjoint windows per source")

    print(f"\n{'─' * 85}")
    print("Loading model...")
    model, tokenizer = load_llm()

    all_results = []

    for proj_idx, project in enumerate(projects, 1):
        print(f"\n{'=' * 85}")
        print(f"  [{proj_idx}/{len(projects)}] PROJECT: {project}")
        print(f"{'=' * 85}")

        project_path = os.path.join(DATA_DIR, project)
        id_map = load_requirements(project_path)
        test_pairs = load_test_pairs(project_path, id_map)

        if not test_pairs:
            print(f"    SKIPPED — no test pairs")
            continue

        n_pos = sum(1 for p in test_pairs if p['label'] == 1)
        n_neg = len(test_pairs) - n_pos
        print(f"    Test pairs: {len(test_pairs)} ({n_pos} pos / {n_neg} neg)")

        # Clear VRAM cache between projects to prevent fragmentation
        torch.cuda.empty_cache()
        gc.collect()

        result = evaluate_project(model, tokenizer, test_pairs, project)
        all_results.append(result)

    if not all_results:
        print("\nNo projects evaluated.")
        return

    # ── Final Report ──
    macro_p = np.mean([r['metrics']['precision'] for r in all_results])
    macro_r = np.mean([r['metrics']['recall'] for r in all_results])
    macro_f1 = np.mean([r['metrics']['f1'] for r in all_results])
    macro_f2 = np.mean([r['metrics']['f2'] for r in all_results])
    macro_acc = np.mean([r['metrics']['accuracy'] for r in all_results])
    total_pairs = sum(r['n_pairs'] for r in all_results)
    total_failed = sum(r['failed_parses'] for r in all_results)
    total_time = sum(r['time_seconds'] for r in all_results)

    print("\n\n" + "=" * 85)
    print("ZERO-SHOT (H100 LOCAL) — final_pairs_test (Qwen3 Top-K)")
    print("=" * 85)
    print(f"\n{'Project':<12} {'Pairs':>6} {'F1':>7} {'F2':>7} {'P':>7} {'R':>7} "
          f"{'Acc':>7} {'Fail':>6} {'s/pair':>7}")
    print("-" * 85)

    for r in all_results:
        m = r['metrics']
        print(f"{r['project']:<12} {r['n_pairs']:>6} {m['f1']:>7.4f} {m['f2']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['accuracy']:>7.4f} "
              f"{r['failed_parses']:>4}/{r['n_pairs']:<4} {r['time_per_pair']:>6.3f}s")

    print("-" * 85)
    print(f"{'MACRO AVG':<12} {total_pairs:>6} {macro_f1:>7.4f} {macro_f2:>7.4f} "
          f"{macro_p:>7.4f} {macro_r:>7.4f} "
          f"{macro_acc:>7.4f} "
          f"{total_failed:>4}/{total_pairs:<4} {total_time/max(total_pairs,1):>6.3f}s")

    # Deployment summary
    print(f"\n{'─' * 85}")
    print("DEPLOYMENT METRICS (RQ3)")
    print(f"{'─' * 85}")
    for r in all_results:
        dm = r.get('deployment_metrics', {})
        print(f"  {r['project']:<12}  s/pair={r['time_per_pair']:.3f}  "
              f"tok/s={dm.get('tokens_per_second', 'N/A')}  "
              f"VRAM={dm.get('peak_inference_vram_gb', 'N/A')}GB  "
              f"gen={dm.get('avg_gen_ms', 'N/A')}ms  "
              f"in_tok={dm.get('avg_input_tokens', 'N/A')}  "
              f"gen_tok={dm.get('avg_gen_tokens', 'N/A')}")

    print(f"\n  Average latency: {total_time/max(total_pairs,1)*1000:.0f} ms/pair")
    print(f"  Total time: {total_time/60:.1f} minutes")

    # Save
    output = {
        'timestamp': timestamp,
        'model': BASE_MODEL,
        'hardware': torch.cuda.get_device_name(0),
        'experiment': 'zero_shot_qwen_hard',
        'test_type': 'final_pairs_test',
        'negative_strategy': 'Qwen3-ranked disjoint windows per source',
        'macro_metrics': {
            'precision': round(float(macro_p), 4),
            'recall': round(float(macro_r), 4),
            'f1': round(float(macro_f1), 4),
            'f2': round(float(macro_f2), 4),
            'accuracy': round(float(macro_acc), 4),
        },
        'total_pairs': total_pairs,
        'total_failed': total_failed,
        'total_time_seconds': round(total_time, 1),
        'avg_time_per_pair': round(total_time / max(total_pairs, 1), 4),
        'per_project': all_results,
    }

    results_file = os.path.join(OUTPUT_DIR, "zero_shot_qwen_hard_results.json")
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {results_file}")
    print("=" * 85)


if __name__ == "__main__":
    main()
