"""
RAG Stage-1 Re-run — Gemma 4 31B on Qwen3 Top-K Hard Negatives
===============================================================
Implementation notes inherited from the unified RAG runner:
  - Ollama endpoint corrected to /api/embed with "input" key (was /api/embeddings + "prompt")
  - Native list batching + parallel HTTP workers for embedding throughput
  - Retrieval timing broken out (embed / dense / bm25 / RRF) for RQ3.1
  - VRAM snapshot at model load (separate from peak inference VRAM) for RQ3.3
  - Index storage and build-time measurement for RQ3.2

Configs:
  RAG-A: MPNet dense  | 2 pos + 2 neg
  RAG-B: Qwen3 dense  | 2 pos + 2 neg
  RAG-C: Qwen3 dense  | 4 pos + 0 neg

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u 6.rag_rerun_stage1_8192.py
  CUDA_VISIBLE_DEVICES=0 python -u 6.rag_rerun_stage1_8192.py --configs RAG_A RAG_B
  CUDA_VISIBLE_DEVICES=0 python -u 6.rag_rerun_stage1_8192.py --projects AAH BEAM
  CUDA_VISIBLE_DEVICES=0 python -u 6.rag_rerun_stage1_8192.py --force

Date: 2026-06-14
"""

import argparse
import gc
import hashlib
import json
import os
import random
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ==================== PATHS ====================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT / "DATA" / ".GROUND_TRUTH")
RERUN_ROOT = str(ROOT / "RESULTS" / "RAG_STAGE1_V3_8192")
INDEXES_ROOT = os.path.join(RERUN_ROOT, "INDEXES")


# ==================== MODEL / TASK CONSTANTS ====================
BASE_MODEL = "unsloth/gemma-4-31B-it"
MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 20
SEED = 42
WARMUP_PAIRS_FOR_LATENCY = 10

ALL_PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]


# ==================== RETRIEVER CONFIG ====================
OLLAMA_URL = "https://ymir-api.ifak.eu"
QWEN3_MODEL = "qwen3-embedding:4b"
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
RRF_K = 60

EMBED_BATCH_SIZE = 32
EMBED_PARALLEL_WORKERS = 4
EMBED_RETRY_MAX = 5
EMBED_TIMEOUT_SECONDS = 180


# ==================== STAGE-1 RAG CONFIGS ====================
RAG_CONFIGS = [
    {"id": "RAG_A", "label": "RAG-A MPNet 2+2",  "retriever": "mpnet",
     "k_pos": 2, "k_neg": 2, "desc": "Dense (MPNet), balanced contrastive."},
    {"id": "RAG_B", "label": "RAG-B Qwen3 2+2",  "retriever": "qwen3",
     "k_pos": 2, "k_neg": 2, "desc": "Dense (Qwen3), balanced contrastive."},
    {"id": "RAG_C", "label": "RAG-C Qwen3 4+0",  "retriever": "qwen3",
     "k_pos": 4, "k_neg": 0, "desc": "Dense (Qwen3), positive-only, k=4."},
]


# ==================== PROMPT ====================
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
    skipped_missing_endpoint = []
    for p in pairs:
        s, t = p["source_id"], p["target_id"]
        if s not in id_map or t not in id_map:
            skipped_missing_endpoint.append((s, t))
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
    if skipped_missing_endpoint:
        preview = ", ".join(f"{s}->{t}" for s, t in skipped_missing_endpoint[:5])
        raise ValueError(
            f"{os.path.basename(project_path)}/{split}: skipped "
            f"{len(skipped_missing_endpoint)} pairs because source_id/target_id "
            f"was missing from requirements.json. First examples: {preview}"
        )
    if len(enriched) != len(pairs):
        raise AssertionError(
            f"{os.path.basename(project_path)}/{split}: loaded {len(enriched)} "
            f"pairs from {len(pairs)} raw pairs"
        )
    return enriched


def demo_text_for_indexing(pair):
    return (f"{pair['hlr_summary']}\n{pair['hlr_description']}\n"
            f"{pair['llr_summary']}\n{pair['llr_description']}")


def query_text(pair):
    return demo_text_for_indexing(pair)


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
    conservative_preds = [0 if e["prediction"] is None else e["prediction"] for e in predictions_log]
    conservative = compute_metrics(conservative_preds, labels_all)
    return {
        "clean": clean, "conservative": conservative,
        "n_total": len(predictions_log),
        "n_failed_parses": sum(1 for e in predictions_log if e["prediction"] is None),
        "n_pos": sum(1 for e in predictions_log if e["label"] == 1),
        "n_neg": sum(1 for e in predictions_log if e["label"] == 0),
    }


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


# ==================== BM25 TOKENIZER ====================
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def bm25_tokenize(text):
    if not text:
        return []
    text = text.lower().translate(_PUNCT_TABLE)
    return [t for t in text.split() if len(t) >= 2]


# ==================== EMBEDDING PROVIDERS ====================

class Qwen3Embedder:
    """Ollama embeddings via /api/embed with native list batching + parallel HTTP.

    Endpoint: POST {OLLAMA_URL}/api/embed
    Payload:  {"model": "qwen3-embedding:4b", "input": [text1, text2, ...]}
    Response: {"embeddings": [[...], [...], ...]}
    """

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
        """One HTTP request, retried with exponential backoff."""
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
        """Recursive split-on-failure (mirrors rag_ablation_abc.py)."""
        try:
            return self._embed_batch(texts)
        except Exception:
            if len(texts) <= 1:
                raise
            mid = len(texts) // 2
            print(f"      Splitting batch {len(texts)} -> {mid}+{len(texts)-mid} "
                  f"after persistent failure")
            a = self._embed_batch_with_split(texts[:mid])
            b = self._embed_batch_with_split(texts[mid:])
            return np.concatenate([a, b], axis=0)

    def embed(self, texts):
        """Parallel batched HTTP embedding. Returns np.ndarray [n, dim]."""
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


class MPNetEmbedder:
    def __init__(self, model_name=MPNET_MODEL):
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts, batch_size=32):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        embs = self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=False, convert_to_numpy=True)
        return embs.astype(np.float32)


# ==================== INDEX MANAGEMENT ====================

def _hash_pairs(pairs):
    keys = sorted([(p["source_id"], p["target_id"], p["label"]) for p in pairs])
    return hashlib.sha1(json.dumps(keys).encode()).hexdigest()


def index_dir(retriever_name, project):
    return os.path.join(INDEXES_ROOT, retriever_name, project)


def index_manifest_path(retriever_name, project):
    return os.path.join(index_dir(retriever_name, project), "manifest.json")


def _dir_size_bytes(path):
    if not os.path.isdir(path):
        return 0
    total = 0
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total


def build_dense_index(project, embedder, retriever_name, train_pairs, force=False):
    """Returns (index, docs, build_stats)."""
    import faiss
    out_dir = index_dir(retriever_name, project)
    idx_path = os.path.join(out_dir, "index.faiss")
    docs_path = os.path.join(out_dir, "docs.json")
    manifest_path = index_manifest_path(retriever_name, project)
    current_pairs_hash = _hash_pairs(train_pairs)

    if (os.path.exists(idx_path) and os.path.exists(docs_path)
            and os.path.exists(manifest_path) and not force):
        with open(docs_path, "r") as f:
            docs = json.load(f)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        cached_pairs_hash = manifest.get("pairs_hash")
        docs_pairs_hash = _hash_pairs(docs)
        if (cached_pairs_hash == current_pairs_hash and
                docs_pairs_hash == current_pairs_hash and
                len(docs) == len(train_pairs)):
            index = faiss.read_index(idx_path)
            size_mb = _dir_size_bytes(out_dir) / (1024 ** 2)
            print(f"      [{retriever_name}/{project}] loaded cached index "
                  f"(n={len(docs)}, {size_mb:.2f} MB, hash OK)")
            return index, docs, {
                "from_cache": True, "build_seconds": 0.0,
                "size_mb": round(size_mb, 2),
                "n_documents": len(docs),
                "embedding_dim": manifest.get("embedding_dim"),
                "pairs_hash": current_pairs_hash,
            }
        print(f"      [{retriever_name}/{project}] cached index hash mismatch; rebuilding.")

    os.makedirs(out_dir, exist_ok=True)
    texts = [demo_text_for_indexing(p) for p in train_pairs]
    print(f"      [{retriever_name}/{project}] embedding {len(texts)} train pairs...")
    t0 = time.time()
    embeddings = embedder.embed(texts)
    embed_seconds = time.time() - t0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, idx_path)
    with open(docs_path, "w") as f:
        json.dump(train_pairs, f)

    manifest = {
        "retriever": retriever_name, "project": project,
        "source_file": "final_pairs_train.json",
        "n_documents": len(train_pairs),
        "n_positives": sum(1 for p in train_pairs if p["label"] == 1),
        "n_negatives": sum(1 for p in train_pairs if p["label"] == 0),
        "embedding_dim": dim,
        "pairs_hash": current_pairs_hash,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "build_embed_seconds": round(embed_seconds, 2),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = _dir_size_bytes(out_dir) / (1024 ** 2)
    print(f"      [{retriever_name}/{project}] built ({len(texts)} docs, dim={dim}, "
          f"{size_mb:.2f} MB, embed={embed_seconds:.1f}s)")
    return index, train_pairs, {
        "from_cache": False, "build_seconds": round(embed_seconds, 2),
        "size_mb": round(size_mb, 2),
        "n_documents": len(train_pairs), "embedding_dim": dim,
        "pairs_hash": current_pairs_hash,
    }


def build_bm25_index(project, train_pairs):
    """Returns (bm25, docs, build_stats). rank_bm25 is in-memory; size estimated via pickle."""
    from rank_bm25 import BM25Okapi
    import pickle
    out_dir = index_dir("bm25", project)
    manifest_path = index_manifest_path("bm25", project)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    tokenized = [bm25_tokenize(demo_text_for_indexing(p)) for p in train_pairs]
    bm25 = BM25Okapi(tokenized)
    build_seconds = time.time() - t0

    pickle_bytes = len(pickle.dumps(bm25))
    size_mb = pickle_bytes / (1024 ** 2)

    manifest = {
        "retriever": "bm25", "project": project,
        "source_file": "final_pairs_train.json",
        "tokenizer": "lowercase+punct_split+min_len_2",
        "n_documents": len(train_pairs),
        "n_positives": sum(1 for p in train_pairs if p["label"] == 1),
        "n_negatives": sum(1 for p in train_pairs if p["label"] == 0),
        "pairs_hash": _hash_pairs(train_pairs),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bm25_k1": 1.5, "bm25_b": 0.75,
        "build_seconds": round(build_seconds, 2),
        "in_memory_size_mb_estimate": round(size_mb, 2),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"      [bm25/{project}] built ({len(train_pairs)} docs, "
          f"~{size_mb:.2f} MB in-mem, build={build_seconds:.1f}s)")
    return bm25, train_pairs, {
        "from_cache": False, "build_seconds": round(build_seconds, 2),
        "size_mb": round(size_mb, 2), "n_documents": len(train_pairs),
    }


# ==================== RETRIEVAL (with timing) ====================

def retrieve_dense_topk(index, docs, query_emb, k, label_filter=None):
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


def retrieve_dense_ranked_ids(index, docs, query_emb, top_n, label_filter=None):
    search_n = max(top_n * 5, 100)
    if search_n > len(docs):
        search_n = len(docs)
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    q = q.reshape(1, -1).astype(np.float32)
    t0 = time.time()
    scores, idxs = index.search(q, search_n)
    elapsed_ms = (time.time() - t0) * 1000
    ranked = []
    for s, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        if label_filter is not None and docs[i]["label"] != label_filter:
            continue
        ranked.append(int(i))
        if len(ranked) >= top_n:
            break
    return ranked, elapsed_ms


def retrieve_bm25_ranked_ids(bm25, docs, query_tokens, top_n, label_filter=None):
    t0 = time.time()
    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores)
    elapsed_ms = (time.time() - t0) * 1000
    ranked = []
    for i in order:
        if label_filter is not None and docs[i]["label"] != label_filter:
            continue
        ranked.append(int(i))
        if len(ranked) >= top_n:
            break
    return ranked, elapsed_ms


def reciprocal_rank_fusion(rank_lists, k=RRF_K, top_n=10):
    rrf_scores = {}
    for ranks in rank_lists:
        for rank, doc_idx in enumerate(ranks, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [doc_idx for doc_idx, _ in fused[:top_n]]


def select_demos_for_config(config, dense_index, bm25, docs, query_emb, query_tokens):
    """Returns (demos, retrieval_timing_dict)."""
    retriever = config["retriever"]
    k_pos = config["k_pos"]
    k_neg = config["k_neg"]
    selected = []
    dense_ms_total = 0.0
    bm25_ms_total = 0.0
    t_overall = time.time()

    if retriever in ("mpnet", "qwen3"):
        if k_pos > 0:
            pos_hits, dms = retrieve_dense_topk(dense_index, docs, query_emb, k_pos, label_filter=1)
            dense_ms_total += dms
            for score, idx, doc in pos_hits:
                selected.append({**{k: doc[k] for k in
                                  ["source_id", "target_id", "hlr_summary",
                                   "hlr_description", "llr_summary", "llr_description", "label"]},
                                "_doc_index": idx, "_dense_score": score})
        if k_neg > 0:
            neg_hits, dms = retrieve_dense_topk(dense_index, docs, query_emb, k_neg, label_filter=0)
            dense_ms_total += dms
            for score, idx, doc in neg_hits:
                selected.append({**{k: doc[k] for k in
                                  ["source_id", "target_id", "hlr_summary",
                                   "hlr_description", "llr_summary", "llr_description", "label"]},
                                "_doc_index": idx, "_dense_score": score})

    elif retriever == "hybrid":
        for label_filter, k_take in [(1, k_pos), (0, k_neg)]:
            if k_take <= 0:
                continue
            fusion_pool = 50
            dense_ranks, dms = retrieve_dense_ranked_ids(
                dense_index, docs, query_emb, fusion_pool, label_filter=label_filter)
            bm25_ranks, bms = retrieve_bm25_ranked_ids(
                bm25, docs, query_tokens, fusion_pool, label_filter=label_filter)
            dense_ms_total += dms
            bm25_ms_total += bms
            fused = reciprocal_rank_fusion([dense_ranks, bm25_ranks], k=RRF_K, top_n=k_take)
            for doc_idx in fused:
                doc = docs[doc_idx]
                selected.append({**{k: doc[k] for k in
                                  ["source_id", "target_id", "hlr_summary",
                                   "hlr_description", "llr_summary", "llr_description", "label"]},
                                "_doc_index": doc_idx,
                                "_dense_rank": dense_ranks.index(doc_idx) + 1 if doc_idx in dense_ranks else None,
                                "_bm25_rank": bm25_ranks.index(doc_idx) + 1 if doc_idx in bm25_ranks else None})
    else:
        raise ValueError(f"Unknown retriever: {retriever}")

    total_ms = (time.time() - t_overall) * 1000
    return selected, {
        "dense_search_ms": round(dense_ms_total, 3),
        "bm25_search_ms": round(bm25_ms_total, 3),
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
        print("[SANITY] MISSING FILES:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)
    print(f"[SANITY] Data files OK for {len(projects)} projects.")


def sanity_check_pair_counts(projects):
    print("\n[SANITY] Pair counts per project:")
    print(f"  {'Project':<10} {'Train':>8} {'Test':>6} {'TrPos':>6} {'TrNeg':>6}")
    for p in projects:
        proj_path = os.path.join(DATA_DIR, p)
        id_map = load_requirements(proj_path)
        tr = load_pairs(proj_path, "train", id_map)
        te = load_pairs(proj_path, "test", id_map)
        tr_pos = sum(1 for x in tr if x["label"] == 1)
        tr_neg = sum(1 for x in tr if x["label"] == 0)
        print(f"  {p:<10} {len(tr):>8} {len(te):>6} {tr_pos:>6} {tr_neg:>6}")


def sanity_check_prompt():
    print("\n[SANITY] Prompt render check...")
    dummy_query = {"hlr_summary": "Q-HLR", "hlr_description": "Q-HLR-DESC",
                   "llr_summary": "Q-LLR", "llr_description": "Q-LLR-DESC",
                   "label": 1, "source_id": "X", "target_id": "Y"}
    dummy_demos = [
        {"hlr_summary": "D-HLR", "hlr_description": "D-HLR-DESC",
         "llr_summary": "D-LLR", "llr_description": "D-LLR-DESC",
         "label": 1, "source_id": "A", "target_id": "B"},
        {"hlr_summary": "E-HLR", "hlr_description": "E-HLR-DESC",
         "llr_summary": "E-LLR", "llr_description": "E-LLR-DESC",
         "label": 0, "source_id": "C", "target_id": "D"},
    ]
    p1 = render_rag_prompt([], dummy_query)
    p2 = render_rag_prompt(dummy_demos, dummy_query)
    assert "Q-HLR" in p1 and "Q-LLR" in p1
    assert "D-HLR" in p2 and "E-LLR" in p2 and "Q-HLR" in p2
    assert '{"is_linked": true}' in p2
    assert '{"is_linked": false}' in p2
    print("[SANITY] Prompts render correctly.")


def sanity_check_ollama():
    """Verify /api/embed endpoint with native list batching."""
    print(f"\n[SANITY] Probing Ollama at {OLLAMA_URL}/api/embed ...")
    try:
        embedder = Qwen3Embedder()
        v1 = embedder.embed(["sanity probe"])
        assert v1.shape[0] == 1 and v1.shape[1] > 0
        v2 = embedder.embed(["probe one", "probe two", "probe three"])
        assert v2.shape[0] == 3
        print(f"[SANITY] Qwen3 OK (dim={v1.shape[1]}, list-batch verified)")
    except Exception as e:
        print(f"[SANITY] Ollama check FAILED: {e}")
        print(f"[SANITY] Verify {OLLAMA_URL}/api/embed accepts "
              f"{{'model': '{QWEN3_MODEL}', 'input': [...]}}")
        sys.exit(1)


# ==================== PATHS PER CONFIG ====================

def config_dir(config_id):
    return os.path.join(RERUN_ROOT, config_id)


def predictions_path(config_id, project):
    return os.path.join(config_dir(config_id), "PREDICTIONS", f"{project}_predictions.json")


def config_results_path(config_id):
    return os.path.join(config_dir(config_id), "results.json")


# ==================== LLM (cached, with VRAM-at-load snapshot) ====================

_LLM_CACHE = {"model": None, "tokenizer": None, "vram_at_load_gb": None}


def get_llm():
    if _LLM_CACHE["model"] is not None:
        return _LLM_CACHE["model"], _LLM_CACHE["tokenizer"], _LLM_CACHE["vram_at_load_gb"]
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    print(f"\n[LLM] Loading {BASE_MODEL} (4-bit NF4)...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True, dtype=torch.bfloat16, device_map={"": 0},
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    FastLanguageModel.for_inference(model)
    torch.cuda.synchronize()
    vram_at_load = torch.cuda.memory_allocated(0) / 1024 ** 3
    _LLM_CACHE["model"] = model
    _LLM_CACHE["tokenizer"] = tokenizer
    _LLM_CACHE["vram_at_load_gb"] = round(vram_at_load, 2)
    print(f"[LLM] Loaded. VRAM at load: {vram_at_load:.2f} GB")
    return model, tokenizer, _LLM_CACHE["vram_at_load_gb"]


# ==================== EVALUATION ====================

def evaluate_config_on_project(config, project, force=False, force_rebuild_indexes=False):
    config_id = config["id"]
    pred_path = predictions_path(config_id, project)

    if os.path.exists(pred_path) and not force:
        with open(pred_path, "r") as f:
            predictions_log = json.load(f)
        metrics_block = compute_both_metrics(predictions_log)
        print(f"    [{config_id}/{project}] cached: "
              f"F2_clean={metrics_block['clean']['f2']:.4f} "
              f"F2_cons={metrics_block['conservative']['f2']:.4f} "
              f"failed={metrics_block['n_failed_parses']}")
        return _assemble_result(config_id, project, predictions_log, metrics_block,
                                from_cache=True)

    proj_path = os.path.join(DATA_DIR, project)
    id_map = load_requirements(proj_path)
    train_pairs = load_pairs(proj_path, "train", id_map)
    test_pairs = load_pairs(proj_path, "test", id_map)
    if not train_pairs or not test_pairs:
        print(f"    [{config_id}/{project}] empty data, skipping.")
        return None

    retriever = config["retriever"]
    dense_index = None
    dense_docs = None
    bm25 = None
    bm25_docs = None
    dense_build_stats = None
    bm25_build_stats = None

    if retriever == "mpnet":
        print(f"    [{config_id}/{project}] preparing MPNet index...")
        emb = MPNetEmbedder()
        dense_index, dense_docs, dense_build_stats = build_dense_index(
            project, emb, "mpnet", train_pairs, force=force_rebuild_indexes)
        query_embedder = emb
    elif retriever == "qwen3":
        print(f"    [{config_id}/{project}] preparing Qwen3 index...")
        emb = Qwen3Embedder()
        dense_index, dense_docs, dense_build_stats = build_dense_index(
            project, emb, "qwen3", train_pairs, force=force_rebuild_indexes)
        query_embedder = emb
    elif retriever == "hybrid":
        print(f"    [{config_id}/{project}] preparing Qwen3 + BM25 indexes...")
        emb = Qwen3Embedder()
        dense_index, dense_docs, dense_build_stats = build_dense_index(
            project, emb, "qwen3", train_pairs, force=force_rebuild_indexes)
        bm25, bm25_docs, bm25_build_stats = build_bm25_index(project, train_pairs)
        assert _hash_pairs(dense_docs) == _hash_pairs(bm25_docs), \
            "dense and BM25 must use identical pair ordering"
        query_embedder = emb
    else:
        raise ValueError(f"Unknown retriever: {retriever}")

    # Pre-embed all test queries in one batched pass
    print(f"    [{config_id}/{project}] embedding {len(test_pairs)} test queries...")
    test_query_texts = [query_text(p) for p in test_pairs]
    t_embed_start = time.time()
    test_query_embs = query_embedder.embed(test_query_texts)
    test_query_embed_seconds = time.time() - t_embed_start
    norms = np.linalg.norm(test_query_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    test_query_embs = test_query_embs / norms
    avg_embed_ms_per_query = (test_query_embed_seconds / len(test_pairs)) * 1000
    print(f"    [{config_id}/{project}] embeddings ready "
          f"({test_query_embed_seconds:.1f}s, avg {avg_embed_ms_per_query:.1f}ms/query)")

    model, tokenizer, vram_at_load_gb = get_llm()
    torch.cuda.reset_peak_memory_stats(0)

    print(f"    [{config_id}/{project}] running inference...")
    predictions_log = []
    start_time = time.time()
    total_input_tokens = 0
    total_gen_tokens = 0
    total_gen_time = 0.0
    truncated = 0
    docs_for_retrieval = dense_docs

    for i, pair in enumerate(test_pairs):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / (i + 1)
            remaining = rate * (len(test_pairs) - i - 1)
            print(f"      {config_id}/{project} {i+1}/{len(test_pairs)} "
                  f"(~{remaining/60:.1f}min)", end="\r")

        q_emb = test_query_embs[i]
        q_tokens = bm25_tokenize(test_query_texts[i]) if retriever == "hybrid" else None

        demos, retrieval_timing = select_demos_for_config(
            config, dense_index, bm25, docs_for_retrieval, q_emb, q_tokens
        )

        t_prompt = time.time()
        prompt_text = render_rag_prompt(demos, pair)
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_unclipped = tokenizer(text=input_text, return_tensors="pt", truncation=False)
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

        demo_summary = []
        for d in demos:
            entry = {"source_id": d["source_id"], "target_id": d["target_id"],
                     "label": d["label"]}
            if "_dense_score" in d:
                entry["dense_score"] = round(d["_dense_score"], 4)
            if "_dense_rank" in d:
                entry["dense_rank"] = d["_dense_rank"]
                entry["bm25_rank"] = d["_bm25_rank"]
            demo_summary.append(entry)

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
    print(f"    [{config_id}/{project}] DONE in {elapsed:.1f}s "
          f"F2_clean={metrics_block['clean']['f2']:.4f} "
          f"F2_cons={metrics_block['conservative']['f2']:.4f} "
          f"failed={metrics_block['n_failed_parses']}/{len(test_pairs)} "
          f"trunc={truncated} "
          f"vram_load={vram_at_load_gb:.1f}GB peak={peak_vram_inference_gb:.1f}GB")

    return _assemble_result(
        config_id, project, predictions_log, metrics_block,
        elapsed=elapsed, vram_at_load_gb=vram_at_load_gb,
        vram_peak_inference_gb=peak_vram_inference_gb,
        from_cache=False,
        total_input_tokens=total_input_tokens,
        total_gen_tokens=total_gen_tokens,
        total_gen_time=total_gen_time,
        truncated=truncated,
        test_query_embed_seconds=test_query_embed_seconds,
        avg_embed_ms_per_query=avg_embed_ms_per_query,
        dense_build_stats=dense_build_stats,
        bm25_build_stats=bm25_build_stats,
    )


def _assemble_result(config_id, project, predictions_log, metrics_block,
                     elapsed=None, vram_at_load_gb=None,
                     vram_peak_inference_gb=None, from_cache=False,
                     total_input_tokens=None, total_gen_tokens=None,
                     total_gen_time=None, truncated=None,
                     test_query_embed_seconds=None, avg_embed_ms_per_query=None,
                     dense_build_stats=None, bm25_build_stats=None):
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
    timed_entries = (predictions_log[WARMUP_PAIRS_FOR_LATENCY:]
                     if len(predictions_log) > WARMUP_PAIRS_FOR_LATENCY
                     else predictions_log)
    timed_retrievals = [e.get("retrieval", {}) for e in timed_entries if "retrieval" in e]

    def latency_stats(values):
        values = [float(v) for v in values if v is not None]
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0}
        return {
            "mean": round(float(np.mean(values)), 2),
            "median": round(float(np.median(values)), 2),
            "std": round(float(np.std(values)), 2),
            "p95": round(float(np.percentile(values, 95)), 2),
        }

    avg_dense_ms = float(np.mean([r.get("dense_search_ms", 0.0) for r in retrievals])) if retrievals else 0.0
    avg_bm25_ms = float(np.mean([r.get("bm25_search_ms", 0.0) for r in retrievals])) if retrievals else 0.0
    avg_retrieval_total_ms = float(np.mean([r.get("retrieval_total_ms", 0.0) for r in retrievals])) if retrievals else 0.0
    avg_prompt_ms = float(np.mean([e.get("prompt_build_ms", 0.0) for e in predictions_log])) if predictions_log else 0.0
    avg_gen_ms = float(np.mean([e.get("gen_time_ms", 0.0) for e in predictions_log])) if predictions_log else 0.0

    steady_retrieval_total = latency_stats([r.get("retrieval_total_ms", 0.0) for r in timed_retrievals])
    steady_dense = latency_stats([r.get("dense_search_ms", 0.0) for r in timed_retrievals])
    steady_bm25 = latency_stats([r.get("bm25_search_ms", 0.0) for r in timed_retrievals])
    steady_prompt = latency_stats([e.get("prompt_build_ms", 0.0) for e in timed_entries])
    steady_generation = latency_stats([e.get("gen_time_ms", 0.0) for e in timed_entries])
    all_generation = latency_stats([e.get("gen_time_ms", 0.0) for e in predictions_log])

    tok_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0.0
    format_failure_rate_pct = 100.0 * metrics_block["n_failed_parses"] / max(n, 1)
    truncation_rate_pct = 100.0 * truncated / max(n, 1)

    return {
        "config_id": config_id,
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
                              if (vram_peak_inference_gb is not None and
                                  vram_at_load_gb is not None) else None,
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
                "avg_bm25_search_ms": round(avg_bm25_ms, 3),
                "avg_prompt_build_ms": round(avg_prompt_ms, 2),
                "avg_generation_ms": round(avg_gen_ms, 2),
                "warmup_pairs_excluded": min(WARMUP_PAIRS_FOR_LATENCY, max(n - 1, 0)),
                "steady_retrieval_total": steady_retrieval_total,
                "steady_dense_search": steady_dense,
                "steady_bm25_search": steady_bm25,
                "steady_prompt_build": steady_prompt,
                "steady_generation": steady_generation,
                "all_generation": all_generation,
            },
            "test_query_embedding": {
                "total_seconds": round(test_query_embed_seconds, 2)
                                 if test_query_embed_seconds else None,
                "avg_ms_per_query": round(avg_embed_ms_per_query, 2)
                                    if avg_embed_ms_per_query else None,
            },
            "index_stats": {
                "dense": dense_build_stats,
                "bm25": bm25_build_stats,
            },
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

    dense_size_total = 0.0
    bm25_size_total = 0.0
    dense_build_total = 0.0
    bm25_build_total = 0.0
    for r in per_project_results:
        idx = r["deployment"].get("index_stats", {}) or {}
        if idx.get("dense"):
            dense_size_total += idx["dense"].get("size_mb", 0) or 0
            dense_build_total += idx["dense"].get("build_seconds", 0) or 0
        if idx.get("bm25"):
            bm25_size_total += idx["bm25"].get("size_mb", 0) or 0
            bm25_build_total += idx["bm25"].get("build_seconds", 0) or 0

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
        "avg_bm25_search_ms":
            avg(lambda r: (r["deployment"].get("latency_breakdown_ms") or {})
                .get("avg_bm25_search_ms")),
        "avg_generation_ms":
            avg(lambda r: (r["deployment"].get("latency_breakdown_ms") or {})
                .get("avg_generation_ms")),
        "total_dense_index_size_mb": round(dense_size_total, 2),
        "total_bm25_index_size_mb_estimate": round(bm25_size_total, 2),
        "total_dense_index_build_seconds": round(dense_build_total, 2),
        "total_bm25_index_build_seconds": round(bm25_build_total, 2),
        "total_test_query_embed_seconds":
            total(lambda r: (r["deployment"].get("test_query_embedding") or {})
                  .get("total_seconds")),
    }


def write_config_results(config, per_project_results):
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_id": config["id"],
        "label": config["label"],
        "model": BASE_MODEL,
        "config": {k: v for k, v in config.items()},
        "rrf_k": RRF_K if config["retriever"] == "hybrid" else None,
        "n_projects_evaluated": len(per_project_results),
        "macro_clean": macro_average(per_project_results, "clean"),
        "macro_conservative": macro_average(per_project_results, "conservative"),
        "deployment_rollup": deployment_rollup(per_project_results),
        "total_failed_parses": sum(r["failed_parses"] for r in per_project_results),
        "total_pairs": sum(r["n_pairs"] for r in per_project_results),
        "per_project": per_project_results,
    }
    out_path = config_results_path(config["id"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def write_master_comparison(all_results):
    master = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": BASE_MODEL,
        "test_set": "final_pairs_test (Qwen3 top-K hard negatives)",
        "configs": {},
    }
    for r in all_results:
        master["configs"][r["config_id"]] = {
            "label": r["label"], "config": r["config"],
            "macro_clean": r["macro_clean"],
            "macro_conservative": r["macro_conservative"],
            "deployment_rollup": r["deployment_rollup"],
            "n_projects": r["n_projects_evaluated"],
            "total_failed_parses": r["total_failed_parses"],
        }
    out_path = os.path.join(RERUN_ROOT, "MASTER_RAG_COMPARISON.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(master, f, indent=2, default=str)
    return master


def print_master_table(master):
    print("\n\n" + "=" * 110)
    print("RAG MASTER COMPARISON — accuracy")
    print("=" * 110)
    print(f"{'Config':<22} {'Metric':<14} {'P':>7} {'R':>7} {'F1':>7} {'F2':>7} {'Acc':>7} {'Fails':>7}")
    print("-" * 110)
    for cid, c in master["configs"].items():
        clean = c["macro_clean"] or {}
        cons = c["macro_conservative"] or {}
        fails = c["total_failed_parses"]
        print(f"{c['label']:<22} {'clean':<14} "
              f"{clean.get('precision', 0):>7.4f} {clean.get('recall', 0):>7.4f} "
              f"{clean.get('f1', 0):>7.4f} {clean.get('f2', 0):>7.4f} "
              f"{clean.get('accuracy', 0):>7.4f} {fails:>7}")
        print(f"{'':<22} {'conservative':<14} "
              f"{cons.get('precision', 0):>7.4f} {cons.get('recall', 0):>7.4f} "
              f"{cons.get('f1', 0):>7.4f} {cons.get('f2', 0):>7.4f} "
              f"{cons.get('accuracy', 0):>7.4f}")
        print("-" * 110)

    print("\n" + "=" * 110)
    print("RAG MASTER COMPARISON — deployment / RQ3 metrics")
    print("=" * 110)
    print(f"{'Config':<22} {'ms/pair':>9} {'retr_ms':>9} {'gen_ms':>9} "
          f"{'tok/s':>7} {'fail%':>6} {'trunc%':>7} {'load GB':>8} {'peak GB':>8}")
    print("-" * 110)
    for cid, c in master["configs"].items():
        d = c.get("deployment_rollup") or {}
        print(f"{c['label']:<22} "
              f"{(d.get('avg_time_per_pair_ms') or 0):>9.1f} "
              f"{(d.get('avg_retrieval_total_ms') or 0):>9.2f} "
              f"{(d.get('avg_generation_ms') or 0):>9.1f} "
              f"{(d.get('avg_tokens_per_second_generated') or 0):>7.1f} "
              f"{(d.get('macro_format_failure_rate_pct') or 0):>6.2f} "
              f"{(d.get('macro_truncation_rate_pct') or 0):>7.2f} "
              f"{(d.get('avg_vram_at_load_gb') or 0):>8.2f} "
              f"{(d.get('avg_vram_peak_inference_gb') or 0):>8.2f}")


# ==================== MAIN ====================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    choices=[c["id"] for c in RAG_CONFIGS],
                    help="Subset of RAG configs (default: all)")
    ap.add_argument("--projects", nargs="+", choices=ALL_PROJECTS,
                    help="Subset of projects (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run inference even if predictions cached")
    ap.add_argument("--force-rebuild-indexes", action="store_true",
                    help="Rebuild dense+BM25 indexes even if cached")
    return ap.parse_args()


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args()
    configs = [c for c in RAG_CONFIGS if (not args.configs or c["id"] in args.configs)]
    projects = args.projects or ALL_PROJECTS

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 110)
    print("RAG STAGE-1 8192 — GEMMA 4 31B (Qwen3 Top-K Hard Negatives)")
    print("=" * 110)
    print(f"Timestamp:    {timestamp}")
    print(f"Output root:  {RERUN_ROOT}")
    print(f"Indexes:      {INDEXES_ROOT}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print(f"Max seq len:  {MAX_SEQ_LENGTH}")
    print(f"Configs:      {[c['id'] for c in configs]}")
    print(f"Projects:     {projects}")
    print(f"Ollama URL:   {OLLAMA_URL}/api/embed  (model: {QWEN3_MODEL})")
    print(f"Embed batch:  {EMBED_BATCH_SIZE} texts/req x {EMBED_PARALLEL_WORKERS} parallel workers")
    print(f"RRF k:        {RRF_K}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    sanity_check_data(projects)
    sanity_check_pair_counts(projects)
    sanity_check_prompt()
    sanity_check_ollama()

    os.makedirs(RERUN_ROOT, exist_ok=True)
    os.makedirs(INDEXES_ROOT, exist_ok=True)

    all_results = []

    for c_idx, config in enumerate(configs, 1):
        cid = config["id"]
        print(f"\n\n{'#' * 110}")
        print(f"# [{c_idx}/{len(configs)}] {config['label']}")
        print(f"# {config['desc']}")
        print(f"# retriever={config['retriever']}  "
              f"k_pos={config['k_pos']}  k_neg={config['k_neg']}")
        print("#" * 110)

        per_project_results = []
        for p_idx, project in enumerate(projects, 1):
            print(f"\n  [{p_idx}/{len(projects)}] {cid} -> {project}")
            try:
                result = evaluate_config_on_project(
                    config, project,
                    force=args.force,
                    force_rebuild_indexes=args.force_rebuild_indexes,
                )
                if result:
                    per_project_results.append(result)
            except Exception as e:
                print(f"    [{cid}/{project}] ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        if per_project_results:
            cr = write_config_results(config, per_project_results)
            all_results.append(cr)
            mc = cr["macro_clean"]
            mcs = cr["macro_conservative"]
            d = cr["deployment_rollup"] or {}
            print(f"\n  {'-' * 100}")
            print(f"  {cid} MACRO RESULTS ({len(per_project_results)} projects)")
            print(f"  {'-' * 100}")
            print(f"    Clean:        P={mc['precision']:.4f} R={mc['recall']:.4f} "
                  f"F1={mc['f1']:.4f} F2={mc['f2']:.4f}")
            print(f"    Conservative: P={mcs['precision']:.4f} R={mcs['recall']:.4f} "
                  f"F1={mcs['f1']:.4f} F2={mcs['f2']:.4f}")
            print(f"    Time/pair:    {(d.get('avg_time_per_pair_ms') or 0):.1f}ms "
                  f"(retrieval {(d.get('avg_retrieval_total_ms') or 0):.2f}ms, "
                  f"gen {(d.get('avg_generation_ms') or 0):.1f}ms)")
            print(f"    VRAM:         load={(d.get('avg_vram_at_load_gb') or 0):.2f}GB, "
                  f"peak={(d.get('avg_vram_peak_inference_gb') or 0):.2f}GB, "
                  f"growth={(d.get('avg_vram_growth_gb') or 0):.2f}GB")
            print(f"    Reliability:  format_fail={(d.get('macro_format_failure_rate_pct') or 0):.2f}%, "
                  f"trunc={(d.get('macro_truncation_rate_pct') or 0):.2f}%")
            print(f"    Indexing:     dense {(d.get('total_dense_index_build_seconds') or 0):.1f}s "
                  f"({(d.get('total_dense_index_size_mb') or 0):.2f} MB), "
                  f"bm25 {(d.get('total_bm25_index_build_seconds') or 0):.1f}s "
                  f"(~{(d.get('total_bm25_index_size_mb_estimate') or 0):.2f} MB est)")
            print(f"    Total failed parses: {cr['total_failed_parses']}/{cr['total_pairs']}")

    if all_results:
        master = write_master_comparison(all_results)
        print_master_table(master)
        print(f"\nMaster comparison saved to: "
              f"{os.path.join(RERUN_ROOT, 'MASTER_RAG_COMPARISON.json')}")

    print("\n" + "=" * 110)
    print("ALL DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()
