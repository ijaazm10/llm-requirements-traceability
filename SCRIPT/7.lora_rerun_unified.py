"""
LoRA Re-run (Unified) — Gemma 4 31B on Qwen3 Top-K Hard Negatives
==================================================================
Trains AND evaluates 6 LoRA configurations across 8 projects in a single script.
Single source of truth for prompts, configs, and metrics.

Replaces:
  - lora_train.py (V2 baseline)
  - lora_evaluate.py (V2 baseline)
  - lora_ablation_runner.py (V1/V3/V4)
  - lora_ablation_evaluator.py (V1/V3/V4)
  - lora_v4_runner_py.py (V5)
  - lora_v4_evaluator.py (V5)

Fixes from audit:
  1. Same SFTConfig structure for ALL versions (no V5 divergence)
  2. Same metric computation for ALL versions (both clean & conservative reported)
  3. Per-prediction logging so metrics can be recomputed without re-running
  4. Identical prompt for training and evaluation (single PROMPT_TEMPLATE constant)
  5. Deterministic seeding everywhere
  6. Sanity checks before any training starts

Usage:
  # Train + eval everything
  CUDA_VISIBLE_DEVICES=0 python -u lora_rerun_unified.py

  # Restrict scope
  CUDA_VISIBLE_DEVICES=0 python -u lora_rerun_unified.py --versions V1 V2
  CUDA_VISIBLE_DEVICES=0 python -u lora_rerun_unified.py --projects AAH BEAM
  CUDA_VISIBLE_DEVICES=0 python -u lora_rerun_unified.py --versions V5 V6 --projects KEYCLOAK

  # Re-evaluate only (skip training if adapter exists)
  CUDA_VISIBLE_DEVICES=0 python -u lora_rerun_unified.py --eval-only

  # Force re-train and re-eval
  CUDA_VISIBLE_DEVICES=0 python -u lora_rerun_unified.py --force

Date: 2026-05-13
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ==================== PATHS ====================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT / "DATA" / ".GROUND_TRUTH")
RERUN_ROOT = str(ROOT / "RESULTS" / "LORA_RERUN_V3")


# ==================== MODEL / TASK CONSTANTS ====================
BASE_MODEL = "unsloth/gemma-4-31B-it"
MAX_SEQ_LENGTH = 3072
MAX_NEW_TOKENS = 20
SEED = 42

ALL_PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]


# ==================== STATIC TRAINING HYPERPARAMETERS ====================
# These do NOT vary across V1-V6. Only the values in ABLATIONS vary.
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8  # effective batch = 16
NUM_EPOCHS = 5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
EVAL_STEPS = 100
SAVE_STEPS = 100
EARLY_STOPPING_PATIENCE = 5
LORA_DROPOUT = 0.05


# ==================== THE 6 ABLATIONS ====================
# Single source of truth for what changes between versions.
# Anything not listed here is identical across all versions.
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_MLP_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

ABLATIONS = [
    {
        "version": "V1_NAIVE",
        "label": "V1 Naive (no upsample)",
        "learning_rate": 1e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "upsample": 1,
        "target_modules": ATTN_MODULES,
        "finetune_mlp_modules": False,
        "desc": "Natural 1:3 imbalance, no upsampling. Tests recall collapse.",
    },
    {
        "version": "V2_BALANCED",
        "label": "V2 Balanced (baseline)",
        "learning_rate": 1e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "upsample": 3,
        "target_modules": ATTN_MODULES,
        "finetune_mlp_modules": False,
        "desc": "Baseline: 3x upsample, LR=1e-4, R=64, attention-only.",
    },
    {
        "version": "V3_STABILIZED",
        "label": "V3 Stabilized (LR halved)",
        "learning_rate": 5e-5,
        "lora_r": 64,
        "lora_alpha": 128,
        "upsample": 3,
        "target_modules": ATTN_MODULES,
        "finetune_mlp_modules": False,
        "desc": "Halve LR. Tests training stability.",
    },
    {
        "version": "V4_EFFICIENCY",
        "label": "V4 Efficiency (rank halved)",
        "learning_rate": 1e-4,
        "lora_r": 32,
        "lora_alpha": 64,
        "upsample": 3,
        "target_modules": ATTN_MODULES,
        "finetune_mlp_modules": False,
        "desc": "Halve rank with original LR. Tests capacity reduction.",
    },
    {
        "version": "V5_SYNTHESIS",
        "label": "V5 Synthesis (rank+LR halved)",
        "learning_rate": 5e-5,
        "lora_r": 32,
        "lora_alpha": 64,
        "upsample": 3,
        "target_modules": ATTN_MODULES,
        "finetune_mlp_modules": False,
        "desc": "V3 + V4 combined. Tests if V4 failed from instability not capacity.",
    },
    {
        "version": "V6_MLP",
        "label": "V6 MLP (attn + MLP modules)",
        "learning_rate": 5e-5,
        "lora_r": 32,
        "lora_alpha": 64,
        "upsample": 3,
        "target_modules": ATTN_MLP_MODULES,
        "finetune_mlp_modules": True,
        "desc": "V5 + MLP modules (gate/up/down_proj). Replicates Gemma 3 MLP test on Gemma 4.",
    },
]


# ==================== PROMPT (single source of truth) ====================
# Used for both training (with answer appended) and evaluation (without answer).
# IMPORTANT: must be byte-for-byte identical across train and eval, or the
# adapter sees out-of-distribution prompts at inference time.

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


def render_prompt(pair):
    """Render the prompt for one pair. Used by both train and eval paths."""
    return PROMPT_TEMPLATE.format(
        hlr_summary=pair["hlr_summary"],
        hlr_description=pair["hlr_description"],
        llr_summary=pair["llr_summary"],
        llr_description=pair["llr_description"],
    )


def format_as_chat(pair):
    """Render prompt + label as a 2-turn chat for SFT training."""
    user_message = render_prompt(pair)
    label_json = '{"is_linked": true}' if pair["label"] == 1 else '{"is_linked": false}'
    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": label_json},
        ]
    }


# ==================== DATA LOADING ====================

def load_requirements(project_path):
    req_file = os.path.join(project_path, "requirements.json")
    with open(req_file, "r", encoding="utf-8") as f:
        reqs = json.load(f)
    id_map = {}
    for r in reqs:
        summary = (r.get("summary", "") or "").strip()
        description = (r.get("description", "") or "").strip()
        id_map[r["id"]] = (summary, description)
    return id_map


def load_pairs(project_path, split, id_map):
    """Load final_pairs_{split}.json and enrich with requirement text."""
    pairs_file = os.path.join(project_path, "splits", f"final_pairs_{split}.json")
    if not os.path.exists(pairs_file):
        return []
    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    enriched = []
    for p in pairs:
        src_id = p["source_id"]
        tgt_id = p["target_id"]
        if src_id not in id_map or tgt_id not in id_map:
            continue
        hlr_sum, hlr_desc = id_map[src_id]
        llr_sum, llr_desc = id_map[tgt_id]
        enriched.append({
            "source_id": src_id,
            "target_id": tgt_id,
            "hlr_summary": hlr_sum,
            "hlr_description": hlr_desc if hlr_desc else "N/A",
            "llr_summary": llr_sum,
            "llr_description": llr_desc if llr_desc else "N/A",
            "label": p["label"],
        })
    return enriched


def upsample_positives(pairs, factor, seed=SEED):
    """Duplicate positives `factor` times and shuffle. factor=1 means no upsampling."""
    if factor <= 1:
        balanced = list(pairs)
        random.Random(seed).shuffle(balanced)
        return balanced
    positives = [p for p in pairs if p["label"] == 1]
    negatives = [p for p in pairs if p["label"] == 0]
    balanced = positives * factor + negatives
    random.Random(seed).shuffle(balanced)
    return balanced


# ==================== METRICS (single source of truth) ====================

def compute_metrics(predictions, labels):
    """Compute precision/recall/F1/F2/accuracy + raw counts.

    predictions and labels must be same length. predictions must contain
    only 0/1 (no None) — caller is responsible for handling failures.
    """
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_both_metrics(predictions_log):
    """Given a per-pair log of {prediction, label}, compute BOTH:
      - clean: drop pairs where prediction is None
      - conservative: treat None as 0 (false negative)
    """
    labels_all = [e["label"] for e in predictions_log]

    # Clean: drop None
    clean_preds = [e["prediction"] for e in predictions_log if e["prediction"] is not None]
    clean_labels = [e["label"] for e in predictions_log if e["prediction"] is not None]
    if clean_preds:
        clean = compute_metrics(clean_preds, clean_labels)
    else:
        clean = compute_metrics([], [])

    # Conservative: None -> 0
    conservative_preds = [0 if e["prediction"] is None else e["prediction"] for e in predictions_log]
    conservative = compute_metrics(conservative_preds, labels_all)

    return {
        "clean": clean,
        "conservative": conservative,
        "n_total": len(predictions_log),
        "n_failed_parses": sum(1 for e in predictions_log if e["prediction"] is None),
        "n_pos": sum(1 for e in predictions_log if e["label"] == 1),
        "n_neg": sum(1 for e in predictions_log if e["label"] == 0),
    }


# ==================== RESPONSE PARSING ====================

def parse_response(response_text):
    """Parse model output to 0/1 or None on failure. Identical logic to all prior eval scripts."""
    if not response_text:
        return None
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    if not text:
        text = response_text.strip()
    # Try JSON parse
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s >= 0 and e > s:
            data = json.loads(text[s:e])
            val = data.get("is_linked")
            if val is not None:
                return 1 if val else 0
    except (json.JSONDecodeError, ValueError):
        pass
    # Substring fallback
    lo = text.lower()
    if '"is_linked": true' in lo or '"is_linked":true' in lo:
        return 1
    if '"is_linked": false' in lo or '"is_linked":false' in lo:
        return 0
    return None


# ==================== SANITY CHECKS ====================

def sanity_check_data(projects):
    """Verify all required data files exist before any training starts."""
    print("\n[SANITY] Checking data files...")
    missing = []
    for p in projects:
        proj_path = os.path.join(DATA_DIR, p)
        if not os.path.exists(os.path.join(proj_path, "requirements.json")):
            missing.append(f"{p}/requirements.json")
        for split in ["train", "val", "test"]:
            f = os.path.join(proj_path, "splits", f"final_pairs_{split}.json")
            if not os.path.exists(f):
                missing.append(f"{p}/splits/final_pairs_{split}.json")
    if missing:
        print(f"[SANITY] MISSING FILES:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)
    print(f"[SANITY] All data files present for {len(projects)} projects.")


def sanity_check_pair_counts(projects):
    """Print pair counts for each project/split to confirm expected dataset sizes."""
    print("\n[SANITY] Pair counts per project/split:")
    print(f"  {'Project':<10} {'Train':>8} {'Val':>6} {'Test':>6} "
          f"{'TrPos':>6} {'TrNeg':>6}")
    for p in projects:
        proj_path = os.path.join(DATA_DIR, p)
        id_map = load_requirements(proj_path)
        tr = load_pairs(proj_path, "train", id_map)
        vl = load_pairs(proj_path, "val", id_map)
        te = load_pairs(proj_path, "test", id_map)
        tr_pos = sum(1 for x in tr if x["label"] == 1)
        tr_neg = sum(1 for x in tr if x["label"] == 0)
        print(f"  {p:<10} {len(tr):>8} {len(vl):>6} {len(te):>6} "
              f"{tr_pos:>6} {tr_neg:>6}")


def sanity_check_prompt_render():
    """Verify the prompt renders correctly with a synthetic pair."""
    print("\n[SANITY] Prompt template render check...")
    dummy = {
        "hlr_summary": "TEST HLR",
        "hlr_description": "TEST HLR DESC",
        "llr_summary": "TEST LLR",
        "llr_description": "TEST LLR DESC",
        "label": 1,
    }
    rendered = render_prompt(dummy)
    chat = format_as_chat(dummy)
    assert "TEST HLR" in rendered
    assert "TEST LLR" in rendered
    assert chat["messages"][1]["content"] == '{"is_linked": true}'
    print("[SANITY] Prompt renders correctly.")


# ==================== TRAINING ====================

def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_version_dir(version_name):
    return os.path.join(RERUN_ROOT, version_name)


def get_adapter_path(version_name, project_name):
    return os.path.join(get_version_dir(version_name), "ADAPTERS",
                        project_name, "best_adapter")


def get_training_log_path(version_name, project_name):
    return os.path.join(get_version_dir(version_name), "LOGS",
                        f"{project_name}_training_stats.json")


def get_predictions_path(version_name, project_name):
    return os.path.join(get_version_dir(version_name), "PREDICTIONS",
                        f"{project_name}_predictions.json")


def get_version_results_path(version_name):
    return os.path.join(get_version_dir(version_name), "results.json")


def train_one(version_cfg, project_name, force=False):
    """Train one (version, project) adapter. Skips if adapter exists unless force=True."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback

    v_name = version_cfg["version"]
    adapter_path = get_adapter_path(v_name, project_name)

    if os.path.exists(adapter_path) and not force:
        print(f"    [TRAIN] {v_name}/{project_name}: adapter exists, skipping.")
        return None

    project_path = os.path.join(DATA_DIR, project_name)
    id_map = load_requirements(project_path)
    train_pairs = load_pairs(project_path, "train", id_map)
    val_pairs = load_pairs(project_path, "val", id_map)

    if not train_pairs or not val_pairs:
        print(f"    [TRAIN] {v_name}/{project_name}: empty data, skipping.")
        return None

    n_tr_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_tr_neg = sum(1 for p in train_pairs if p["label"] == 0)
    n_val_pos = sum(1 for p in val_pairs if p["label"] == 1)
    n_val_neg = sum(1 for p in val_pairs if p["label"] == 0)

    # Upsample
    train_balanced = upsample_positives(train_pairs, factor=version_cfg["upsample"])

    # Set seeds before model load
    set_all_seeds(SEED)

    # Clean VRAM before load
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # Load base model
    print(f"    [TRAIN] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # Add LoRA adapter
    print(f"    [TRAIN] Injecting LoRA adapter (r={version_cfg['lora_r']}, "
          f"alpha={version_cfg['lora_alpha']}, "
          f"modules={'attn+MLP' if version_cfg['finetune_mlp_modules'] else 'attn'})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=version_cfg["lora_r"],
        lora_alpha=version_cfg["lora_alpha"],
        lora_dropout=LORA_DROPOUT,
        target_modules=version_cfg["target_modules"],
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=version_cfg["finetune_mlp_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    [TRAIN] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # Format datasets
    train_dataset = Dataset.from_list([format_as_chat(p) for p in train_balanced])
    val_dataset = Dataset.from_list([format_as_chat(p) for p in val_pairs])

    def apply_template(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}

    train_dataset = train_dataset.map(apply_template, batched=True)
    val_dataset = val_dataset.map(apply_template, batched=True)

    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION
    steps_per_epoch = max(len(train_dataset) // effective_batch, 1)

    print(f"    [TRAIN] Train (bal): {len(train_dataset)} | Val: {len(val_dataset)} "
          f"| eff_batch={effective_batch} | steps/epoch={steps_per_epoch}")

    # SFTConfig (identical structure for all versions)
    training_args = SFTConfig(
        output_dir=os.path.dirname(adapter_path),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=version_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    # Loss only on assistant response, not on prompt
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print(f"    [TRAIN] Training start...")
    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3

    # Save adapter
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    adapter_size_mb = sum(
        os.path.getsize(os.path.join(adapter_path, f))
        for f in os.listdir(adapter_path)
        if os.path.isfile(os.path.join(adapter_path, f))
    ) / (1024**2)

    log_history = trainer.state.log_history
    eval_losses = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]

    stats = {
        "version": v_name,
        "project": project_name,
        "config_used": {k: v for k, v in version_cfg.items() if k != "target_modules"} | {
            "target_modules": version_cfg["target_modules"],
            "lora_dropout": LORA_DROPOUT,
            "max_seq_length": MAX_SEQ_LENGTH,
            "effective_batch": effective_batch,
            "num_epochs": NUM_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": SEED,
        },
        "data": {
            "n_train_original": len(train_pairs),
            "n_train_balanced": len(train_dataset),
            "n_train_positives_original": n_tr_pos,
            "n_train_negatives_original": n_tr_neg,
            "n_val": len(val_dataset),
            "n_val_positives": n_val_pos,
            "n_val_negatives": n_val_neg,
        },
        "training": {
            "time_minutes": round(train_time / 60, 1),
            "final_train_loss": round(train_result.training_loss, 4),
            "best_eval_loss": round(min(l[1] for l in eval_losses), 4) if eval_losses else None,
            "best_eval_step": min(eval_losses, key=lambda x: x[1])[0] if eval_losses else None,
            "total_steps": train_result.global_step,
            "epochs_completed": round(train_result.global_step / steps_per_epoch, 2),
            "peak_vram_gb": round(peak_vram, 1),
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "adapter_size_mb": round(adapter_size_mb, 1),
            "eval_loss_history": eval_losses,
        },
    }

    log_path = get_training_log_path(v_name, project_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"    [TRAIN] {v_name}/{project_name}: DONE in {train_time/60:.1f}min "
          f"| train_loss={train_result.training_loss:.4f} "
          f"| best_eval={stats['training']['best_eval_loss']} "
          f"| peak_vram={peak_vram:.1f}GB")

    # Cleanup
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return stats


# ==================== EVALUATION ====================

def evaluate_one(version_cfg, project_name, force=False):
    """Evaluate one (version, project) adapter. Skips if predictions exist unless force=True."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    v_name = version_cfg["version"]
    adapter_path = get_adapter_path(v_name, project_name)
    pred_path = get_predictions_path(v_name, project_name)

    if not os.path.exists(adapter_path):
        print(f"    [EVAL]  {v_name}/{project_name}: adapter not found, skipping.")
        return None

    # If predictions exist and not forced, just compute metrics from them
    if os.path.exists(pred_path) and not force:
        with open(pred_path, "r") as f:
            predictions_log = json.load(f)
        metrics_block = compute_both_metrics(predictions_log)
        print(f"    [EVAL]  {v_name}/{project_name}: cached "
              f"F2_clean={metrics_block['clean']['f2']:.4f} "
              f"F2_cons={metrics_block['conservative']['f2']:.4f} "
              f"failed={metrics_block['n_failed_parses']}")
        return _assemble_eval_result(v_name, project_name, predictions_log, metrics_block,
                                     elapsed=None, peak_vram=None, from_cache=True,
                                     adapter_path=adapter_path)

    project_path = os.path.join(DATA_DIR, project_name)
    id_map = load_requirements(project_path)
    test_pairs = load_pairs(project_path, "test", id_map)
    if not test_pairs:
        print(f"    [EVAL]  {v_name}/{project_name}: empty test set, skipping.")
        return None

    # Clean VRAM
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    print(f"    [EVAL]  Loading adapter: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    FastLanguageModel.for_inference(model)

    # Resume from partial checkpoint if it exists
    predictions_log = []
    start_idx = 0
    if os.path.exists(pred_path) and not force:
        with open(pred_path, "r") as f:
            predictions_log = json.load(f)
        if len(predictions_log) >= len(test_pairs):
            # Already complete — handled above, but guard anyway
            metrics_block = compute_both_metrics(predictions_log)
            return _assemble_eval_result(v_name, project_name, predictions_log, metrics_block,
                                         elapsed=None, peak_vram=None, from_cache=True,
                                         adapter_path=adapter_path)
        start_idx = len(predictions_log)
        print(f"    [EVAL]  Resuming from pair {start_idx}/{len(test_pairs)}")

    start_time = time.time()
    total_input_tokens = sum(e.get("input_tokens", 0) for e in predictions_log)
    total_gen_tokens = sum(e.get("gen_tokens", 0) for e in predictions_log)
    total_gen_time = sum(e.get("gen_time_ms", 0) for e in predictions_log) / 1000.0
    truncated = sum(1 for e in predictions_log if e.get("truncated", False))

    for i in range(start_idx, len(test_pairs)):
        pair = test_pairs[i]
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / (i + 1)
            remaining = rate * (len(test_pairs) - i - 1)
            print(f"      {v_name}/{project_name} {i+1}/{len(test_pairs)} "
                  f"(~{remaining/60:.1f}min)", end="\r")

        prompt_text = render_prompt(pair)
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Truncation check (before clipping)
        inputs_unclipped = tokenizer(text=input_text, return_tensors="pt", truncation=False)
        is_truncated = inputs_unclipped["input_ids"].shape[1] > MAX_SEQ_LENGTH
        if is_truncated:
            truncated += 1

        # Actual inference (with truncation)
        inputs = tokenizer(text=input_text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        t_gen = time.time() - t0

        generated = outputs[0][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        pred = parse_response(response)

        n_gen = len(generated)
        total_input_tokens += input_len
        total_gen_tokens += n_gen
        total_gen_time += t_gen

        predictions_log.append({
            "source_id": pair["source_id"],
            "target_id": pair["target_id"],
            "label": pair["label"],
            "prediction": pred,
            "response": response[:120],
            "input_tokens": input_len,
            "gen_tokens": n_gen,
            "gen_time_ms": round(t_gen * 1000, 1),
            "truncated": is_truncated,
        })

        # Checkpoint every 50 pairs
        if (i + 1) % 50 == 0:
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            with open(pred_path, "w") as f:
                json.dump(predictions_log, f)

    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3

    # Persist predictions
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions_log, f)

    metrics_block = compute_both_metrics(predictions_log)
    print(f"    [EVAL]  {v_name}/{project_name}: DONE in {elapsed:.1f}s "
          f"F2_clean={metrics_block['clean']['f2']:.4f} "
          f"F2_cons={metrics_block['conservative']['f2']:.4f} "
          f"failed={metrics_block['n_failed_parses']}/{len(test_pairs)} "
          f"trunc={truncated}")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return _assemble_eval_result(
        v_name, project_name, predictions_log, metrics_block,
        elapsed=elapsed, peak_vram=peak_vram, from_cache=False,
        adapter_path=adapter_path,
        total_input_tokens=total_input_tokens,
        total_gen_tokens=total_gen_tokens,
        total_gen_time=total_gen_time,
        truncated=truncated,
    )


def _assemble_eval_result(version_name, project_name, predictions_log, metrics_block,
                          elapsed=None, peak_vram=None, from_cache=False, adapter_path=None,
                          total_input_tokens=None, total_gen_tokens=None,
                          total_gen_time=None, truncated=None):
    """Build the per-project result dict. Pulls deployment metrics from log if not passed."""
    n = len(predictions_log)
    if total_input_tokens is None:
        total_input_tokens = sum(e.get("input_tokens", 0) for e in predictions_log)
    if total_gen_tokens is None:
        total_gen_tokens = sum(e.get("gen_tokens", 0) for e in predictions_log)
    if total_gen_time is None:
        total_gen_time = sum(e.get("gen_time_ms", 0) for e in predictions_log) / 1000.0
    if truncated is None:
        truncated = sum(1 for e in predictions_log if e.get("truncated", False))

    tok_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0.0

    return {
        "version": version_name,
        "project": project_name,
        "n_pairs": n,
        "n_pos": metrics_block["n_pos"],
        "n_neg": metrics_block["n_neg"],
        "failed_parses": metrics_block["n_failed_parses"],
        "metrics_clean": metrics_block["clean"],
        "metrics_conservative": metrics_block["conservative"],
        "deployment": {
            "from_cache": from_cache,
            "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
            "time_per_pair_ms": round(elapsed * 1000 / n, 1) if (elapsed and n) else None,
            "peak_vram_gb": round(peak_vram, 2) if peak_vram is not None else None,
            "total_input_tokens": total_input_tokens,
            "total_gen_tokens": total_gen_tokens,
            "avg_input_tokens": round(total_input_tokens / max(n, 1), 1),
            "avg_gen_tokens": round(total_gen_tokens / max(n, 1), 1),
            "tokens_per_second_generated": round(tok_per_sec, 1),
            "truncated_pairs": truncated,
            "adapter_path": adapter_path,
        },
    }


# ==================== AGGREGATION ====================

def macro_average(per_project_results, which="clean"):
    """Compute unweighted macro-average across projects."""
    if not per_project_results:
        return None
    keys = ["precision", "recall", "f1", "f2", "accuracy"]
    return {k: round(float(np.mean([r[f"metrics_{which}"][k] for r in per_project_results])), 4)
            for k in keys}


def write_version_results(version_cfg, per_project_results):
    """Write per-version results.json with macro + per-project breakdown."""
    v_name = version_cfg["version"]
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": v_name,
        "version_label": version_cfg["label"],
        "model": BASE_MODEL,
        "config": {k: v for k, v in version_cfg.items() if k != "target_modules"} | {
            "target_modules": version_cfg["target_modules"],
        },
        "n_projects_evaluated": len(per_project_results),
        "macro_clean": macro_average(per_project_results, "clean"),
        "macro_conservative": macro_average(per_project_results, "conservative"),
        "total_failed_parses": sum(r["failed_parses"] for r in per_project_results),
        "total_pairs": sum(r["n_pairs"] for r in per_project_results),
        "per_project": per_project_results,
    }
    out_path = get_version_results_path(v_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def write_master_comparison(all_version_results):
    """Write the cross-version comparison table."""
    master = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": BASE_MODEL,
        "test_set": "final_pairs_test (Qwen3 top-K hard negatives)",
        "versions": {},
    }
    for vr in all_version_results:
        master["versions"][vr["version"]] = {
            "label": vr["version_label"],
            "config": vr["config"],
            "macro_clean": vr["macro_clean"],
            "macro_conservative": vr["macro_conservative"],
            "n_projects": vr["n_projects_evaluated"],
            "total_failed_parses": vr["total_failed_parses"],
        }
    out_path = os.path.join(RERUN_ROOT, "MASTER_COMPARISON.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(master, f, indent=2, default=str)
    return master


def print_master_table(master):
    print("\n\n" + "=" * 110)
    print("MASTER COMPARISON TABLE — both clean (drop failures) and conservative (failures=FN)")
    print("=" * 110)
    header = f"{'Version':<20} {'Metric':<14} {'P':>7} {'R':>7} {'F1':>7} {'F2':>7} {'Acc':>7} {'Fails':>7}"
    print(header)
    print("-" * 110)
    for vname, v in master["versions"].items():
        clean = v["macro_clean"] or {}
        cons = v["macro_conservative"] or {}
        fails = v["total_failed_parses"]
        print(f"{v['label']:<20} {'clean':<14} "
              f"{clean.get('precision', 0):>7.4f} {clean.get('recall', 0):>7.4f} "
              f"{clean.get('f1', 0):>7.4f} {clean.get('f2', 0):>7.4f} "
              f"{clean.get('accuracy', 0):>7.4f} {fails:>7}")
        print(f"{'':<20} {'conservative':<14} "
              f"{cons.get('precision', 0):>7.4f} {cons.get('recall', 0):>7.4f} "
              f"{cons.get('f1', 0):>7.4f} {cons.get('f2', 0):>7.4f} "
              f"{cons.get('accuracy', 0):>7.4f}")
        print("-" * 110)


# ==================== MAIN ====================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+",
                    choices=[a["version"] for a in ABLATIONS],
                    help="Subset of versions to run (default: all)")
    ap.add_argument("--projects", nargs="+", choices=ALL_PROJECTS,
                    help="Subset of projects to run (default: all)")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training (use existing adapters)")
    ap.add_argument("--train-only", action="store_true",
                    help="Skip evaluation (just train adapters)")
    ap.add_argument("--force", action="store_true",
                    help="Re-train and re-eval even if adapters/predictions exist")
    return ap.parse_args()


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = parse_args()
    versions = [a for a in ABLATIONS if (not args.versions or a["version"] in args.versions)]
    projects = args.projects or ALL_PROJECTS

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 110)
    print("LoRA RE-RUN UNIFIED — GEMMA 4 31B (Qwen3 Top-K Hard Negatives)")
    print("=" * 110)
    print(f"Timestamp:    {timestamp}")
    print(f"Output root:  {RERUN_ROOT}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print(f"Versions:     {[v['version'] for v in versions]}")
    print(f"Projects:     {projects}")
    print(f"Mode:         "
          f"{'EVAL-ONLY' if args.eval_only else 'TRAIN-ONLY' if args.train_only else 'TRAIN+EVAL'}"
          f"{' (FORCE)' if args.force else ''}")

    # Sanity checks
    sanity_check_data(projects)
    sanity_check_pair_counts(projects)
    sanity_check_prompt_render()

    os.makedirs(RERUN_ROOT, exist_ok=True)

    all_version_results = []

    for v_idx, version_cfg in enumerate(versions, 1):
        v_name = version_cfg["version"]
        print(f"\n\n{'#' * 110}")
        print(f"# [{v_idx}/{len(versions)}] {version_cfg['label']}")
        print(f"# {version_cfg['desc']}")
        print(f"# LR={version_cfg['learning_rate']} R={version_cfg['lora_r']} "
              f"alpha={version_cfg['lora_alpha']} upsample={version_cfg['upsample']}x "
              f"modules={'attn+MLP' if version_cfg['finetune_mlp_modules'] else 'attn'}")
        print("#" * 110)

        per_project_results = []

        for p_idx, project in enumerate(projects, 1):
            print(f"\n  [{p_idx}/{len(projects)}] {v_name} -> {project}")

            # Train (unless --eval-only)
            if not args.eval_only:
                try:
                    train_one(version_cfg, project, force=args.force)
                except Exception as e:
                    print(f"    [TRAIN] ERROR on {v_name}/{project}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Eval (unless --train-only)
            if not args.train_only:
                try:
                    result = evaluate_one(version_cfg, project, force=args.force)
                    if result:
                        per_project_results.append(result)
                except Exception as e:
                    print(f"    [EVAL]  ERROR on {v_name}/{project}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Per-version aggregation
        if per_project_results and not args.train_only:
            version_results = write_version_results(version_cfg, per_project_results)
            all_version_results.append(version_results)

            print(f"\n  {'-' * 100}")
            print(f"  {v_name} MACRO RESULTS ({len(per_project_results)} projects)")
            print(f"  {'-' * 100}")
            mc = version_results["macro_clean"]
            mcs = version_results["macro_conservative"]
            print(f"    Clean (drop failures):        "
                  f"P={mc['precision']:.4f} R={mc['recall']:.4f} "
                  f"F1={mc['f1']:.4f} F2={mc['f2']:.4f} Acc={mc['accuracy']:.4f}")
            print(f"    Conservative (failures=FN):  "
                  f"P={mcs['precision']:.4f} R={mcs['recall']:.4f} "
                  f"F1={mcs['f1']:.4f} F2={mcs['f2']:.4f} Acc={mcs['accuracy']:.4f}")
            print(f"    Total failed parses: {version_results['total_failed_parses']}/"
                  f"{version_results['total_pairs']}")

    # Master comparison
    if all_version_results and not args.train_only:
        master = write_master_comparison(all_version_results)
        print_master_table(master)
        print(f"\nMaster comparison saved to: {os.path.join(RERUN_ROOT, 'MASTER_COMPARISON.json')}")

    print("\n" + "=" * 110)
    print("ALL DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()
