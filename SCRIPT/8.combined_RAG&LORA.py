"""
RAG + LoRA Combined Run (Champion) — Gemma 4 31B
==================================================
Loads the LoRA V4_EFFICIENCY adapter for each project and runs inference with
RAG-B demonstrations (Qwen3 dense retrieval, 2 positives, 2 negatives).

Champions used:
  LoRA: V4_EFFICIENCY (LR=1e-4, R=32, A=64, attention-only, 3x upsample)
        Champion in the LoRA ablation re-run.
  RAG:  RAG_B Qwen3 2+2 (Qwen3 dense, 2 positive demos, 2 negative demos)
        Champion in the RAG Stage-1 rerun.

Combined config tests whether the two adaptation strategies are
complementary (additive gain), redundant (small/no gain), or
antagonistic (combined worse than either alone).

All assets are cached from previous runs:
  - V4_EFFICIENCY adapters in LORA_RERUN_V3/V4_EFFICIENCY/ADAPTERS/{project}/best_adapter/
  - Qwen3 dense indexes in RAG_STAGE1_V3_8192/INDEXES/qwen3/{project}/
  - final_pairs_train.json and final_pairs_test.json (input data)

Outputs go to a fresh directory:
  ../RESULTS/COMBINED_RERUN_V3/V4_RAGB_8192/
    PREDICTIONS/{project}_predictions.json
    results.json
  ../RESULTS/COMBINED_RERUN_V3/MASTER_COMBINED_COMPARISON.json

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u combined_rerun.py
  CUDA_VISIBLE_DEVICES=0 python -u combined_rerun.py --projects AAH BEAM
  CUDA_VISIBLE_DEVICES=0 python -u combined_rerun.py --force

Date: 2026-06-16
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch


# ==================== PATHS ====================
DATA_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/DATA/GROUND_TRUTH"
LORA_ROOT = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS/LORA_RERUN_V3"
RAG_ROOT = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS/RAG_STAGE1_V3_8192"
COMBINED_ROOT = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS/COMBINED_RERUN_V3"


# ==================== MODEL / TASK CONSTANTS ====================
BASE_MODEL = "unsloth/gemma-4-31B-it"
MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 20
SEED = 42

ALL_PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]


# ==================== CHAMPION CONFIG ====================
LORA_CHAMPION_VERSION = "V4_EFFICIENCY"
LORA_CHAMPION_LABEL = "V4 Efficiency (attention-only, LR=1e-4, R=32, A=64)"

# RAG-B matches: Qwen3 dense, k_pos=2, k_neg=2
RAG_CHAMPION_LABEL = "RAG-B Qwen3 2+2"
RAG_RETRIEVER = "qwen3"
RAG_K_POS = 2
RAG_K_NEG = 2


# ==================== RETRIEVER CONFIG ====================
OLLAMA_URL = "https://ymir-api.ifak.eu"
QWEN3_MODEL = "qwen3-embedding:4b"
EMBED_BATCH_SIZE = 32
EMBED_PARALLEL_WORKERS = 4
EMBED_RETRY_MAX = 5
EMBED_TIMEOUT_SECONDS = 180


# ==================== PROMPT (same as RAG script — single source of truth) ====================
PROMPT_INSTRUCTION = """You are analyzing hierarchical software requirements for cross level traceability.
Determine whether a traceability link exists between a High-Level Requirement (HLR) and a Low-Level Requirement (LLR).
A link exists if the LLR implements, refines, or decomposes the HLR."""

PROMPT_PAIR_BLOCK = """High-Level Requirement:
  Summary: {hlr_summary}
  Description: {hlr_description}
Low-Level Requirement:
  Summary: {llr_summary}
  Description: {llr_description}"""

PROMPT_OUTPUT_INSTRUCTION = """Return only:
{"is_linked": true}
or
{"is_linked": false}
No additional text."""


def render_pair_block(pair):
    return PROMPT_PAIR_BLOCK.format(
        hlr_summary=pair["hlr_summary"],
        hlr_description=pair["hlr_description"] or "N/A",
        llr_summary=pair["llr_summary"],
        llr_description=pair["llr_description"] or "N/A",
    )


def render_demo_block(demo, demo_index):
    block = render_pair_block(demo)
    answer = '{"is_linked": true}' if demo["label"] == 1 else '{"is_linked": false}'
    return f"Example {demo_index}:\n{block}\nAnswer: {answer}"


def render_rag_prompt(demos, query_pair):
    parts = [PROMPT_INSTRUCTION]
    if demos:
        parts.append("\nHere are similar examples:")
        for i, demo in enumerate(demos, 1):
            parts.append("\n" + render_demo_block(demo, i))
        parts.append("\nNow classify the following:")
    parts.append("\n" + render_pair_block(query_pair))
    parts.append("\n" + PROMPT_OUTPUT_INSTRUCTION)
    return "\n".join(parts)


# ==================== DATA ====================

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
    pairs_file = os.path.join(project_path, "splits", f"final_pairs_{split}.json")
    if not os.path.exists(pairs_file):
        return []
    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    enriched = []
    for p in pairs:
        s, t = p["source_id"], p["target_id"]
        if s not in id_map or t not in id_map:
            continue
        hsum, hdesc = id_map[s]
        lsum, ldesc = id_map[t]
        enriched.append({
            "source_id": s, "target_id": t,
            "hlr_summary": hsum,
            "hlr_description": hdesc if hdesc else "N/A",
            "llr_summary": lsum,
            "llr_description": ldesc if ldesc else "N/A",
            "label": p["label"],
        })
    return enriched


def query_text(pair):
    return (f"{pair['hlr_summary']}\n{pair['hlr_description']}\n"
            f"{pair['llr_summary']}\n{pair['llr_description']}")


# ==================== METRICS ====================

def compute_metrics(predictions, labels):
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "f2": round(f2, 4), "accuracy": round(accuracy, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_both_metrics(predictions_log):
    labels_all = [e["label"] for e in predictions_log]
    clean_preds = [e["prediction"] for e in predictions_log if e["prediction"] is not None]
    clean_labels = [e["label"] for e in predictions_log if e["prediction"] is not None]
    clean = compute_metrics(clean_preds, clean_labels) if clean_preds else compute_metrics([], [])
    conservative_preds = [0 if e["prediction"] is None else e["prediction"]
                          for e in predictions_log]
    conservative = compute_metrics(conservative_preds, labels_all)
    return {
        "clean": clean, "conservative": conservative,
        "n_total": len(predictions_log),
        "n_failed_parses": sum(1 for e in predictions_log if e["prediction"] is None),
        "n_pos": sum(1 for e in predictions_log if e["label"] == 1),
        "n_neg": sum(1 for e in predictions_log if e["label"] == 0),
    }


# ==================== RESPONSE PARSING ====================

def parse_response(response_text):
    if not response_text:
        return None
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    if not text:
        text = response_text.strip()
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
    lo = text.lower()
    if '"is_linked": true' in lo or '"is_linked":true' in lo:
        return 1
    if '"is_linked": false' in lo or '"is_linked":false' in lo:
        return 0
    return None


# ==================== QWEN3 EMBEDDER ====================

class Qwen3Embedder:
    def __init__(self, url=OLLAMA_URL, model=QWEN3_MODEL,
                 batch_size=EMBED_BATCH_SIZE,
                 parallel_workers=EMBED_PARALLEL_WORKERS):
        import requests
        self.url = url.rstrip("/") + "/api/embed"
        self.model = model
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self._requests = requests

    def _embed_batch(self, texts, retry_max=EMBED_RETRY_MAX):
        for attempt in range(retry_max):
            try:
                r = self._requests.post(
                    self.url,
                    json={"model": self.model, "input": texts},
                    timeout=EMBED_TIMEOUT_SECONDS,
                )
                r.raise_for_status()
                embs = r.json().get("embeddings", [])
                if len(embs) == len(texts):
                    return np.array(embs, dtype=np.float32)
                raise ValueError(
                    f"Ollama returned {len(embs)} embeddings for {len(texts)} inputs"
                )
            except Exception as ex:
                wait = 5 * (2 ** attempt)
                if attempt < retry_max - 1:
                    print(f"      Embed batch failed (attempt {attempt+1}/{retry_max}, "
                          f"retry in {wait}s): {ex}")
                    time.sleep(wait)
                else:
                    raise

    def _embed_batch_with_split(self, texts):
        try:
            return self._embed_batch(texts)
        except Exception:
            if len(texts) <= 1:
                raise
            mid = len(texts) // 2
            print(f"      Splitting batch {len(texts)} -> {mid}+{len(texts)-mid}")
            a = self._embed_batch_with_split(texts[:mid])
            b = self._embed_batch_with_split(texts[mid:])
            return np.concatenate([a, b], axis=0)

    def embed(self, texts):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batches.append(texts[i:i + self.batch_size])
        results = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as ex:
            futures = {
                ex.submit(self._embed_batch_with_split, batch_texts): batch_idx
                for batch_idx, batch_texts in enumerate(batches)
            }
            for fut in as_completed(futures):
                batch_idx = futures[fut]
                results[batch_idx] = fut.result()
        return np.concatenate(results, axis=0)


# ==================== RETRIEVAL ====================

def load_cached_qwen3_index(project):
    """Load the qwen3 FAISS index built by the RAG ablation script."""
    import faiss
    idx_dir = os.path.join(RAG_ROOT, "INDEXES", "qwen3", project)
    idx_path = os.path.join(idx_dir, "index.faiss")
    docs_path = os.path.join(idx_dir, "docs.json")
    manifest_path = os.path.join(idx_dir, "manifest.json")

    if not (os.path.exists(idx_path) and os.path.exists(docs_path)
            and os.path.exists(manifest_path)):
        raise FileNotFoundError(
            f"Qwen3 index for {project} missing at {idx_dir}. "
            f"Run rag_rerun_unified.py first to build it."
        )

    index = faiss.read_index(idx_path)
    with open(docs_path, "r") as f:
        docs = json.load(f)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"      [qwen3/{project}] loaded cached index "
          f"(n={len(docs)}, hash={manifest['pairs_hash'][:12]})")
    return index, docs, manifest


def retrieve_dense_topk(index, docs, query_emb, k, label_filter=None):
    """Top-k docs filtered by label. Returns (hits, elapsed_ms)."""
    search_k = max(k * 5, 50)
    if search_k > len(docs):
        search_k = len(docs)
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    q = q.reshape(1, -1).astype(np.float32)
    t0 = time.time()
    scores, idxs = index.search(q, search_k)
    elapsed_ms = (time.time() - t0) * 1000
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        doc = docs[i]
        if label_filter is not None and doc["label"] != label_filter:
            continue
        out.append((float(s), int(i), doc))
        if len(out) >= k:
            break
    return out, elapsed_ms


def select_demos(dense_index, docs, query_emb):
    """RAG-B selection: 2 positives, 2 negatives."""
    t_overall = time.time()
    selected = []
    dense_ms = 0.0
    if RAG_K_POS > 0:
        pos_hits, dms = retrieve_dense_topk(
            dense_index, docs, query_emb, RAG_K_POS, label_filter=1)
        dense_ms += dms
        for score, idx, doc in pos_hits:
            selected.append({
                **{k: doc[k] for k in
                   ["source_id", "target_id", "hlr_summary", "hlr_description",
                    "llr_summary", "llr_description", "label"]},
                "_doc_index": idx, "_dense_score": score,
            })
    if RAG_K_NEG > 0:
        neg_hits, dms = retrieve_dense_topk(
            dense_index, docs, query_emb, RAG_K_NEG, label_filter=0)
        dense_ms += dms
        for score, idx, doc in neg_hits:
            selected.append({
                **{k: doc[k] for k in
                   ["source_id", "target_id", "hlr_summary", "hlr_description",
                    "llr_summary", "llr_description", "label"]},
                "_doc_index": idx, "_dense_score": score,
            })
    total_ms = (time.time() - t_overall) * 1000
    return selected, {
        "dense_search_ms": round(dense_ms, 3),
        "bm25_search_ms": 0.0,
        "retrieval_total_ms": round(total_ms, 3),
    }


# ==================== SANITY CHECKS ====================

def sanity_check_data(projects):
    print("\n[SANITY] Checking data files...")
    missing = []
    for p in projects:
        proj_path = os.path.join(DATA_DIR, p)
        if not os.path.exists(os.path.join(proj_path, "requirements.json")):
            missing.append(f"{p}/requirements.json")
        for split in ["train", "test"]:
            f = os.path.join(proj_path, "splits", f"final_pairs_{split}.json")
            if not os.path.exists(f):
                missing.append(f"{p}/splits/final_pairs_{split}.json")
    if missing:
        print("[SANITY] MISSING:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)
    print(f"[SANITY] Data files OK for {len(projects)} projects.")


def sanity_check_adapters(projects):
    """Verify all selected LoRA adapters exist before any inference starts."""
    print(f"\n[SANITY] Checking LoRA adapters ({LORA_CHAMPION_VERSION})...")
    missing = []
    for p in projects:
        adapter_path = os.path.join(LORA_ROOT, LORA_CHAMPION_VERSION,
                                    "ADAPTERS", p, "best_adapter")
        if not os.path.isdir(adapter_path):
            missing.append(adapter_path)
        else:
            # Look for the actual weights file
            files = os.listdir(adapter_path)
            has_weights = any(f.endswith(".safetensors") or f.endswith(".bin")
                              for f in files)
            if not has_weights:
                missing.append(f"{adapter_path} (no weights file)")
    if missing:
        print("[SANITY] MISSING ADAPTERS:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)
    print(f"[SANITY] All {len(projects)} {LORA_CHAMPION_VERSION} adapters found.")


def sanity_check_indexes(projects):
    """Verify all qwen3 indexes exist before any inference starts."""
    print(f"\n[SANITY] Checking Qwen3 FAISS indexes...")
    missing = []
    hashes = {}
    for p in projects:
        idx_dir = os.path.join(RAG_ROOT, "INDEXES", "qwen3", p)
        idx_path = os.path.join(idx_dir, "index.faiss")
        docs_path = os.path.join(idx_dir, "docs.json")
        manifest_path = os.path.join(idx_dir, "manifest.json")
        if not all(os.path.exists(x) for x in [idx_path, docs_path, manifest_path]):
            missing.append(idx_dir)
            continue
        with open(manifest_path, "r") as f:
            m = json.load(f)
        hashes[p] = m.get("pairs_hash", "?")[:12]
    if missing:
        print("[SANITY] MISSING INDEXES:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)
    print(f"[SANITY] All {len(projects)} Qwen3 indexes found.")
    print(f"[SANITY] Index hashes (train pair fingerprints):")
    for p, h in hashes.items():
        print(f"    {p}: {h}")


def sanity_check_prompt():
    print("\n[SANITY] Prompt render check...")
    dummy_query = {"hlr_summary": "Q-HLR", "hlr_description": "Q-HLR-DESC",
                   "llr_summary": "Q-LLR", "llr_description": "Q-LLR-DESC",
                   "label": 1, "source_id": "X", "target_id": "Y"}
    n_demos = RAG_K_POS + RAG_K_NEG
    dummy_demos = [
        {"hlr_summary": f"D{i}-HLR", "hlr_description": f"D{i}-HLR-DESC",
         "llr_summary": f"D{i}-LLR", "llr_description": f"D{i}-LLR-DESC",
         "label": 1 if i < RAG_K_POS else 0, "source_id": f"S{i}", "target_id": f"T{i}"}
        for i in range(n_demos)
    ]
    p = render_rag_prompt(dummy_demos, dummy_query)
    assert "Q-HLR" in p and "Q-LLR" in p
    assert all(f"D{i}-HLR" in p for i in range(n_demos))
    assert '{"is_linked": true}' in p
    if RAG_K_NEG > 0:
        assert '{"is_linked": false}' in p
    print(f"[SANITY] Prompt renders correctly with {RAG_K_POS}+{RAG_K_NEG} demos.")


def sanity_check_ollama():
    print(f"\n[SANITY] Probing Ollama at {OLLAMA_URL}/api/embed ...")
    try:
        embedder = Qwen3Embedder()
        v = embedder.embed(["sanity probe one", "sanity probe two"])
        assert v.shape[0] == 2 and v.shape[1] > 0
        print(f"[SANITY] Qwen3 OK (dim={v.shape[1]})")
    except Exception as e:
        print(f"[SANITY] Ollama check FAILED: {e}")
        sys.exit(1)


# ==================== PATHS ====================

CONFIG_ID = f"{LORA_CHAMPION_VERSION}_RAG_B_8192"
CONFIG_LABEL = f"{LORA_CHAMPION_LABEL} + {RAG_CHAMPION_LABEL}"


def predictions_path(project):
    return os.path.join(COMBINED_ROOT, CONFIG_ID, "PREDICTIONS",
                        f"{project}_predictions.json")


def results_path():
    return os.path.join(COMBINED_ROOT, CONFIG_ID, "results.json")


# ==================== EVALUATION ====================

def evaluate_one_project(project, force=False):
    pred_path = predictions_path(project)
    if os.path.exists(pred_path) and not force:
        with open(pred_path, "r") as f:
            predictions_log = json.load(f)
        metrics_block = compute_both_metrics(predictions_log)
        print(f"  [{project}] cached: "
              f"F2_clean={metrics_block['clean']['f2']:.4f} "
              f"F2_cons={metrics_block['conservative']['f2']:.4f} "
              f"failed={metrics_block['n_failed_parses']}")
        return _assemble_result(project, predictions_log, metrics_block,
                                from_cache=True)

    proj_path = os.path.join(DATA_DIR, project)
    id_map = load_requirements(proj_path)
    test_pairs = load_pairs(proj_path, "test", id_map)
    if not test_pairs:
        print(f"  [{project}] empty test set, skipping.")
        return None

    # Load cached qwen3 index
    print(f"  [{project}] loading cached resources...")
    dense_index, dense_docs, dense_manifest = load_cached_qwen3_index(project)

    # Pre-embed all test queries
    print(f"  [{project}] embedding {len(test_pairs)} test queries...")
    embedder = Qwen3Embedder()
    test_query_texts = [query_text(p) for p in test_pairs]
    t_embed_start = time.time()
    test_query_embs = embedder.embed(test_query_texts)
    test_query_embed_seconds = time.time() - t_embed_start
    norms = np.linalg.norm(test_query_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    test_query_embs = test_query_embs / norms
    avg_embed_ms = (test_query_embed_seconds / len(test_pairs)) * 1000
    print(f"  [{project}] embeddings ready "
          f"({test_query_embed_seconds:.1f}s, avg {avg_embed_ms:.1f}ms/query)")

    # Load LoRA-adapted model (NOT cached across projects — adapter differs per project)
    adapter_path = os.path.join(LORA_ROOT, LORA_CHAMPION_VERSION,
                                "ADAPTERS", project, "best_adapter")
    print(f"  [{project}] loading model with adapter from: {adapter_path}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # Loading from adapter_path auto-loads base model + adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    FastLanguageModel.for_inference(model)

    torch.cuda.synchronize()
    vram_at_load_gb = torch.cuda.memory_allocated(0) / 1024 ** 3
    torch.cuda.reset_peak_memory_stats(0)
    print(f"  [{project}] model+adapter loaded. VRAM at load: {vram_at_load_gb:.2f} GB")

    # Inference loop
    print(f"  [{project}] running inference...")
    predictions_log = []
    start_time = time.time()
    total_input_tokens = 0
    total_gen_tokens = 0
    total_gen_time = 0.0
    truncated = 0

    for i, pair in enumerate(test_pairs):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / (i + 1)
            remaining = rate * (len(test_pairs) - i - 1)
            print(f"    {project} {i+1}/{len(test_pairs)} "
                  f"(~{remaining/60:.1f}min)", end="\r")

        q_emb = test_query_embs[i]
        demos, retrieval_timing = select_demos(dense_index, dense_docs, q_emb)

        t_prompt = time.time()
        prompt_text = render_rag_prompt(demos, pair)
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_unclipped = tokenizer(text=input_text, return_tensors="pt",
                                      truncation=False)
        is_truncated = inputs_unclipped["input_ids"].shape[1] > MAX_SEQ_LENGTH
        if is_truncated:
            truncated += 1
        inputs = tokenizer(text=input_text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        prompt_ms = (time.time() - t_prompt) * 1000

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
            )
        torch.cuda.synchronize()
        t_gen = time.time() - t0

        generated = outputs[0][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        pred = parse_response(response)

        n_gen = len(generated)
        total_input_tokens += input_len
        total_gen_tokens += n_gen
        total_gen_time += t_gen

        demo_summary = [
            {"source_id": d["source_id"], "target_id": d["target_id"],
             "label": d["label"], "dense_score": round(d["_dense_score"], 4)}
            for d in demos
        ]

        predictions_log.append({
            "source_id": pair["source_id"],
            "target_id": pair["target_id"],
            "label": pair["label"],
            "prediction": pred,
            "response": response[:120],
            "input_tokens": input_len,
            "gen_tokens": n_gen,
            "gen_time_ms": round(t_gen * 1000, 1),
            "prompt_build_ms": round(prompt_ms, 1),
            "retrieval": retrieval_timing,
            "truncated": is_truncated,
            "retrieved_demos": demo_summary,
        })

    elapsed = time.time() - start_time
    peak_vram_inference_gb = torch.cuda.max_memory_allocated(0) / 1024 ** 3

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions_log, f)

    metrics_block = compute_both_metrics(predictions_log)
    print(f"\n  [{project}] DONE in {elapsed:.1f}s "
          f"F2_clean={metrics_block['clean']['f2']:.4f} "
          f"F2_cons={metrics_block['conservative']['f2']:.4f} "
          f"failed={metrics_block['n_failed_parses']}/{len(test_pairs)} "
          f"trunc={truncated} "
          f"vram_load={vram_at_load_gb:.1f}GB peak={peak_vram_inference_gb:.1f}GB")

    # Free this project's adapter so the next project can load its own
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return _assemble_result(
        project, predictions_log, metrics_block,
        elapsed=elapsed, vram_at_load_gb=vram_at_load_gb,
        vram_peak_inference_gb=peak_vram_inference_gb,
        from_cache=False,
        total_input_tokens=total_input_tokens,
        total_gen_tokens=total_gen_tokens,
        total_gen_time=total_gen_time,
        truncated=truncated,
        test_query_embed_seconds=test_query_embed_seconds,
        avg_embed_ms=avg_embed_ms,
        adapter_path=adapter_path,
        index_hash=dense_manifest.get("pairs_hash", "?"),
    )


def _assemble_result(project, predictions_log, metrics_block,
                     elapsed=None, vram_at_load_gb=None,
                     vram_peak_inference_gb=None, from_cache=False,
                     total_input_tokens=None, total_gen_tokens=None,
                     total_gen_time=None, truncated=None,
                     test_query_embed_seconds=None, avg_embed_ms=None,
                     adapter_path=None, index_hash=None):
    n = len(predictions_log)
    if total_input_tokens is None:
        total_input_tokens = sum(e.get("input_tokens", 0) for e in predictions_log)
    if total_gen_tokens is None:
        total_gen_tokens = sum(e.get("gen_tokens", 0) for e in predictions_log)
    if total_gen_time is None:
        total_gen_time = sum(e.get("gen_time_ms", 0) for e in predictions_log) / 1000.0
    if truncated is None:
        truncated = sum(1 for e in predictions_log if e.get("truncated", False))

    retrievals = [e.get("retrieval", {}) for e in predictions_log if "retrieval" in e]
    avg_dense_ms = float(np.mean([r.get("dense_search_ms", 0.0) for r in retrievals])) if retrievals else 0.0
    avg_retrieval_total_ms = float(np.mean([r.get("retrieval_total_ms", 0.0) for r in retrievals])) if retrievals else 0.0
    avg_prompt_ms = float(np.mean([e.get("prompt_build_ms", 0.0) for e in predictions_log]))
    avg_gen_ms = float(np.mean([e.get("gen_time_ms", 0.0) for e in predictions_log]))

    tok_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0.0
    format_failure_rate_pct = 100.0 * metrics_block["n_failed_parses"] / max(n, 1)
    truncation_rate_pct = 100.0 * truncated / max(n, 1)

    return {
        "config_id": CONFIG_ID,
        "project": project,
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
            "vram_at_load_gb": vram_at_load_gb,
            "vram_peak_inference_gb": round(vram_peak_inference_gb, 2)
                                      if vram_peak_inference_gb is not None else None,
            "vram_growth_gb": round(vram_peak_inference_gb - vram_at_load_gb, 2)
                              if (vram_peak_inference_gb is not None
                                  and vram_at_load_gb is not None) else None,
            "total_input_tokens": total_input_tokens,
            "total_gen_tokens": total_gen_tokens,
            "avg_input_tokens": round(total_input_tokens / max(n, 1), 1),
            "avg_gen_tokens": round(total_gen_tokens / max(n, 1), 1),
            "tokens_per_second_generated": round(tok_per_sec, 1),
            "truncated_pairs": truncated,
            "truncation_rate_pct": round(truncation_rate_pct, 2),
            "format_failure_rate_pct": round(format_failure_rate_pct, 2),
            "latency_breakdown_ms": {
                "avg_retrieval_total_ms": round(avg_retrieval_total_ms, 2),
                "avg_dense_search_ms": round(avg_dense_ms, 3),
                "avg_prompt_build_ms": round(avg_prompt_ms, 2),
                "avg_generation_ms": round(avg_gen_ms, 2),
            },
            "test_query_embedding": {
                "total_seconds": round(test_query_embed_seconds, 2)
                                 if test_query_embed_seconds else None,
                "avg_ms_per_query": round(avg_embed_ms, 2)
                                    if avg_embed_ms else None,
            },
            "adapter_path": adapter_path,
            "index_hash": index_hash,
        },
    }


# ==================== AGGREGATION ====================

def macro_average(per_project_results, which="clean"):
    if not per_project_results:
        return None
    keys = ["precision", "recall", "f1", "f2", "accuracy"]
    return {k: round(float(np.mean([r[f"metrics_{which}"][k]
                                    for r in per_project_results])), 4)
            for k in keys}


def deployment_rollup(per_project_results):
    if not per_project_results:
        return None

    def avg(getter):
        vals = [getter(r) for r in per_project_results]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    def total(getter):
        vals = [getter(r) for r in per_project_results]
        vals = [v for v in vals if v is not None]
        return round(float(np.sum(vals)), 3) if vals else None

    return {
        "avg_time_per_pair_ms": avg(lambda r: r["deployment"].get("time_per_pair_ms")),
        "avg_vram_at_load_gb": avg(lambda r: r["deployment"].get("vram_at_load_gb")),
        "avg_vram_peak_inference_gb": avg(lambda r: r["deployment"].get("vram_peak_inference_gb")),
        "avg_vram_growth_gb": avg(lambda r: r["deployment"].get("vram_growth_gb")),
        "avg_input_tokens_per_pair": avg(lambda r: r["deployment"].get("avg_input_tokens")),
        "avg_gen_tokens_per_pair": avg(lambda r: r["deployment"].get("avg_gen_tokens")),
        "avg_tokens_per_second_generated":
            avg(lambda r: r["deployment"].get("tokens_per_second_generated")),
        "macro_truncation_rate_pct":
            avg(lambda r: r["deployment"].get("truncation_rate_pct")),
        "macro_format_failure_rate_pct":
            avg(lambda r: r["deployment"].get("format_failure_rate_pct")),
        "avg_retrieval_total_ms":
            avg(lambda r: (r["deployment"].get("latency_breakdown_ms") or {})
                .get("avg_retrieval_total_ms")),
        "avg_dense_search_ms":
            avg(lambda r: (r["deployment"].get("latency_breakdown_ms") or {})
                .get("avg_dense_search_ms")),
        "avg_generation_ms":
            avg(lambda r: (r["deployment"].get("latency_breakdown_ms") or {})
                .get("avg_generation_ms")),
        "total_test_query_embed_seconds":
            total(lambda r: (r["deployment"].get("test_query_embedding") or {})
                  .get("total_seconds")),
    }


def write_results(per_project_results):
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_id": CONFIG_ID,
        "label": CONFIG_LABEL,
        "model": BASE_MODEL,
        "lora_champion": {
            "version": LORA_CHAMPION_VERSION,
            "label": LORA_CHAMPION_LABEL,
        },
        "rag_champion": {
            "label": RAG_CHAMPION_LABEL,
            "retriever": RAG_RETRIEVER,
            "k_pos": RAG_K_POS,
            "k_neg": RAG_K_NEG,
        },
        "n_projects_evaluated": len(per_project_results),
        "macro_clean": macro_average(per_project_results, "clean"),
        "macro_conservative": macro_average(per_project_results, "conservative"),
        "deployment_rollup": deployment_rollup(per_project_results),
        "total_failed_parses": sum(r["failed_parses"] for r in per_project_results),
        "total_pairs": sum(r["n_pairs"] for r in per_project_results),
        "per_project": per_project_results,
    }
    out_path = results_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def write_master(combined_results):
    """Write a small master file that compares champion combined against
    the individual champions (LoRA V4 alone, RAG-B alone) for easy reference.
    Reads the V4 and RAG-B macro numbers from their result files
    if available; otherwise just writes the combined result.
    """
    master = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": BASE_MODEL,
        "test_set": "final_pairs_test (Qwen3 top-K hard negatives)",
        "configurations": {
            CONFIG_ID: {
                "label": CONFIG_LABEL,
                "macro_clean": combined_results["macro_clean"],
                "macro_conservative": combined_results["macro_conservative"],
                "deployment_rollup": combined_results["deployment_rollup"],
                "n_projects": combined_results["n_projects_evaluated"],
                "total_failed_parses": combined_results["total_failed_parses"],
                "total_pairs": combined_results["total_pairs"],
            },
        },
    }

    # Try to pull in LoRA V4 and RAG-B solo numbers for context
    lora_v4_path = os.path.join(LORA_ROOT, LORA_CHAMPION_VERSION, "results.json")
    if os.path.exists(lora_v4_path):
        with open(lora_v4_path, "r") as f:
            v4 = json.load(f)
        master["configurations"]["LORA_V4_ALONE"] = {
            "label": LORA_CHAMPION_LABEL + " (alone)",
            "macro_clean": v4.get("macro_clean"),
            "macro_conservative": v4.get("macro_conservative"),
        }

    rag_b_path = os.path.join(RAG_ROOT, "RAG_B", "results.json")
    if os.path.exists(rag_b_path):
        with open(rag_b_path, "r") as f:
            rb = json.load(f)
        master["configurations"]["RAG_B_ALONE"] = {
            "label": RAG_CHAMPION_LABEL + " (alone)",
            "macro_clean": rb.get("macro_clean"),
            "macro_conservative": rb.get("macro_conservative"),
        }
    else:
        rag_master_path = os.path.join(RAG_ROOT, "MASTER_RAG_COMPARISON.json")
        if os.path.exists(rag_master_path):
            with open(rag_master_path, "r") as f:
                rm = json.load(f)
            rb = (rm.get("configs") or {}).get("RAG_B")
            if rb:
                master["configurations"]["RAG_B_ALONE"] = {
                    "label": RAG_CHAMPION_LABEL + " (alone)",
                    "macro_clean": rb.get("macro_clean"),
                    "macro_conservative": rb.get("macro_conservative"),
                    "deployment_rollup": rb.get("deployment_rollup"),
                }

    out_path = os.path.join(COMBINED_ROOT, "MASTER_COMBINED_COMPARISON.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(master, f, indent=2, default=str)
    return master


def print_comparison_table(master):
    print("\n\n" + "=" * 120)
    print("COMBINED RUN — CHAMPION-VS-CHAMPION COMPARISON")
    print("=" * 120)
    print(f"{'Configuration':<42} {'Metric':<14} "
          f"{'P':>8} {'R':>8} {'F1':>8} {'F2':>8} {'Acc':>8}")
    print("-" * 120)
    for cid, c in master["configurations"].items():
        clean = c.get("macro_clean") or {}
        cons = c.get("macro_conservative") or {}
        print(f"{c['label']:<42} {'clean':<14} "
              f"{(clean.get('precision') or 0):>8.4f} "
              f"{(clean.get('recall') or 0):>8.4f} "
              f"{(clean.get('f1') or 0):>8.4f} "
              f"{(clean.get('f2') or 0):>8.4f} "
              f"{(clean.get('accuracy') or 0):>8.4f}")
        print(f"{'':<42} {'conservative':<14} "
              f"{(cons.get('precision') or 0):>8.4f} "
              f"{(cons.get('recall') or 0):>8.4f} "
              f"{(cons.get('f1') or 0):>8.4f} "
              f"{(cons.get('f2') or 0):>8.4f} "
              f"{(cons.get('accuracy') or 0):>8.4f}")
        print("-" * 120)


# ==================== MAIN ====================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="+", choices=ALL_PROJECTS,
                    help="Subset of projects to run (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run inference even if predictions cached")
    return ap.parse_args()


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args()
    projects = args.projects or ALL_PROJECTS

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 120)
    print("RAG + LoRA COMBINED (CHAMPION) — GEMMA 4 31B")
    print("=" * 120)
    print(f"Timestamp:    {timestamp}")
    print(f"Output root:  {COMBINED_ROOT}/{CONFIG_ID}/")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print(f"Max seq len:  {MAX_SEQ_LENGTH}")
    print(f"LoRA champ:   {LORA_CHAMPION_LABEL}  (adapters in {LORA_ROOT})")
    print(f"RAG champ:    {RAG_CHAMPION_LABEL}  (indexes in {RAG_ROOT})")
    print(f"Combo label:  {CONFIG_LABEL}")
    print(f"Projects:     {projects}")
    print(f"Ollama URL:   {OLLAMA_URL}/api/embed  (model: {QWEN3_MODEL})")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    sanity_check_data(projects)
    sanity_check_adapters(projects)
    sanity_check_indexes(projects)
    sanity_check_prompt()
    sanity_check_ollama()

    os.makedirs(COMBINED_ROOT, exist_ok=True)

    per_project_results = []
    for p_idx, project in enumerate(projects, 1):
        print(f"\n{'#' * 120}")
        print(f"# [{p_idx}/{len(projects)}] {project}")
        print(f"{'#' * 120}")
        try:
            result = evaluate_one_project(project, force=args.force)
            if result:
                per_project_results.append(result)
        except Exception as e:
            print(f"  [{project}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not per_project_results:
        print("\nNo results produced. Aborting.")
        sys.exit(1)

    combined = write_results(per_project_results)

    mc = combined["macro_clean"]
    mcs = combined["macro_conservative"]
    d = combined["deployment_rollup"] or {}
    print(f"\n{'=' * 120}")
    print(f"COMBINED MACRO RESULTS ({len(per_project_results)} projects)")
    print(f"{'=' * 120}")
    print(f"  Clean:        P={mc['precision']:.4f} R={mc['recall']:.4f} "
          f"F1={mc['f1']:.4f} F2={mc['f2']:.4f}")
    print(f"  Conservative: P={mcs['precision']:.4f} R={mcs['recall']:.4f} "
          f"F1={mcs['f1']:.4f} F2={mcs['f2']:.4f}")
    print(f"  Time/pair:    {(d.get('avg_time_per_pair_ms') or 0):.1f}ms "
          f"(retrieval {(d.get('avg_retrieval_total_ms') or 0):.2f}ms, "
          f"gen {(d.get('avg_generation_ms') or 0):.1f}ms)")
    print(f"  VRAM:         load={(d.get('avg_vram_at_load_gb') or 0):.2f}GB, "
          f"peak={(d.get('avg_vram_peak_inference_gb') or 0):.2f}GB, "
          f"growth={(d.get('avg_vram_growth_gb') or 0):.2f}GB")
    print(f"  Reliability:  format_fail={(d.get('macro_format_failure_rate_pct') or 0):.2f}%, "
          f"trunc={(d.get('macro_truncation_rate_pct') or 0):.2f}%")
    print(f"  Total failed parses: {combined['total_failed_parses']}/{combined['total_pairs']}")

    master = write_master(combined)
    print_comparison_table(master)
    print(f"\nResults saved to: {results_path()}")
    print(f"Master saved to:  {os.path.join(COMBINED_ROOT, 'MASTER_COMBINED_COMPARISON.json')}")

    print("\n" + "=" * 120)
    print("ALL DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
