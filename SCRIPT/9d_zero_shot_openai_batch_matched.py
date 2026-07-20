"""
Zero-Shot - OpenAI Batch API (Matched-Prompt GPT-5.4 Comparison)
================================================================

Methodologically matched counterpart to:
  - 5.zero_shot_h100.py                    (Gemma 4 31B local H100)
  - 9a_zero_shot_openai_matched.py         (GPT-5.4-mini synchronous OpenAI)
  - 9c_zero_shot_claude_batch_matched.py   (Claude Sonnet 4.6 Anthropic Batch)

What is held constant against the Gemma/OpenAI/Claude reference runs
--------------------------------------------------------------------
* Same test data:        splits/final_pairs_test.json (Qwen3 top-K hard negs)
* Same data loader:      reads requirements.json as a list, builds id_map,
                         joins by source_id/target_id, empty description -> "N/A"
* Same prompt text:      character-for-character identical to create_prompt_text
                         in 5.zero_shot_h100.py and 9a_zero_shot_openai_matched.py
* Same role structure:   single user-role message; no system role; no JSON mode
* Same generation:       max_completion_tokens=20, temperature=0.0, seed=42
* Same parser:           json-window extract -> substring fallback
* Same metrics:          P, R, F1, F2(beta=2), accuracy; clean + conservative

What differs by necessity
-------------------------
* Execution mode:        OpenAI Batch API is asynchronous. This is valid for
                         classification quality, but NOT comparable for
                         interactive latency. Do not use batch wall-clock time in
                         RQ3 single-stream latency tables.
* Token counts:          OpenAI tokenization differs from Gemma and Claude.
* Chat formatting:       OpenAI receives the same semantic payload as one user
                         message and applies native chat formatting server-side.

Operational safety
------------------
* No paid API submission unless --submit is passed.
* --submit requires --budget-usd.
* Budget guard uses local token estimation plus the max output-token cap.
* Saves JSONL request file and manifest before submission.
* Result collection maps unordered batch output by custom_id.
* Result files use the same per-project prediction JSON format as the OpenAI
  synchronous script.

Usage
-----
Dry run, no API calls:
    python 9d_zero_shot_openai_batch_matched.py --dry-run

Local estimate only, no API calls:
    python 9d_zero_shot_openai_batch_matched.py --estimate-only --budget-usd 9.00

Tiny paid synchronous smoke test, no batch:
    python 9d_zero_shot_openai_batch_matched.py --smoke-test --projects AAH --max-pairs 1 --budget-usd 0.05

Submit the full GPT-5.4 batch:
    python 9d_zero_shot_openai_batch_matched.py --submit --budget-usd 9.00

Submit a sharded project/range when the organization has a low enqueued-token cap:
    python 9d_zero_shot_openai_batch_matched.py --submit --projects JBIDE --start-index 0 --max-pairs 650 --run-suffix jbide_0000_0649 --budget-usd 9.00

Poll until finished and collect results:
    python 9d_zero_shot_openai_batch_matched.py --collect --poll

Collect an existing batch id:
    python 9d_zero_shot_openai_batch_matched.py --collect --batch-id batch_...

Author: Thesis Work
Date: 2026-06-04
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT / "DATA" / ".GROUND_TRUTH")
BASE_OUTPUT_DIR = str(ROOT / "RESULTS" / "OPENAI_ZERO_SHOT_BATCH_V3")

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_MAX_COMPLETION_TOKENS = 20
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42

ALL_PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]

PROMPT_MODE = "matched_single_user_prompt_v1_batch"

# Character-for-character identical to the prompt used in the Gemma/OpenAI
# matched zero-shot scripts. Do not reformat. The hash is saved in run_config.
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

# Batch prices used only for budget-guard estimates. Provider invoice is
# authoritative. Values are 50% of the corresponding synchronous prices used in
# 9a_zero_shot_openai_matched.py for the GPT-5.4 family.
PRICING_PER_1M_TOKENS = {
    "gpt-5.4":       {"input": 1.25,  "output": 7.50},
    "gpt-5.4-mini":  {"input": 0.375, "output": 2.25},
    "gpt-5.4-nano":  {"input": 0.10,  "output": 0.625},
    "gpt-4.1":       {"input": 1.00,  "output": 4.00},
    "gpt-4.1-mini":  {"input": 0.20,  "output": 0.80},
    "gpt-4.1-nano":  {"input": 0.05,  "output": 0.20},
    "gpt-4o":        {"input": 1.25,  "output": 5.00},
    "gpt-4o-mini":   {"input": 0.075, "output": 0.30},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sanitize_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", model_name.replace(".", "_"))


def prompt_template_hash():
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]


def atomic_save_json(data, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_cost_usd(model, prompt_tokens, completion_tokens):
    pricing = PRICING_PER_1M_TOKENS.get(model)
    if not pricing:
        return None
    return (prompt_tokens / 1e6) * pricing["input"] + (completion_tokens / 1e6) * pricing["output"]


def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def obj_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "json"):
        return json.loads(obj.json())
    return obj


# ---------------------------------------------------------------------------
# Data loading - matched to 5.zero_shot_h100.py
# ---------------------------------------------------------------------------

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


def load_test_pairs(project_path, id_map):
    pairs_file = os.path.join(project_path, "splits", "final_pairs_test.json")
    if not os.path.exists(pairs_file):
        print(f"    ERROR: {pairs_file} not found!")
        return []
    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    enriched = []
    skipped_missing_endpoint = []
    for p in pairs:
        src_id = p["source_id"]
        tgt_id = p["target_id"]
        if src_id not in id_map or tgt_id not in id_map:
            skipped_missing_endpoint.append((src_id, tgt_id))
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
    if skipped_missing_endpoint:
        preview = ", ".join(f"{s}->{t}" for s, t in skipped_missing_endpoint[:5])
        raise ValueError(
            f"{os.path.basename(project_path)}/test: skipped "
            f"{len(skipped_missing_endpoint)} pairs because source_id/target_id "
            f"was missing from requirements.json. First examples: {preview}"
        )
    if len(enriched) != len(pairs):
        raise AssertionError(
            f"{os.path.basename(project_path)}/test: loaded {len(enriched)} "
            f"pairs from {len(pairs)} raw pairs"
        )
    return enriched


def create_prompt_text(pair):
    return PROMPT_TEMPLATE.format(
        hlr_summary=pair["hlr_summary"],
        hlr_description=pair["hlr_description"],
        llr_summary=pair["llr_summary"],
        llr_description=pair["llr_description"],
    )


def load_all_project_pairs(args):
    all_project_pairs = {}
    for project in args.projects:
        project_path = os.path.join(args.data_dir, project)
        id_map = load_requirements(project_path)
        pairs = load_test_pairs(project_path, id_map)
        if args.start_index < 0:
            print("ERROR: --start-index must be >= 0")
            sys.exit(1)
        if args.start_index:
            pairs = pairs[args.start_index:]
        if args.max_pairs is not None:
            pairs = pairs[: args.max_pairs]
        if not pairs:
            print(f"    SKIP {project} - no test pairs loaded")
            continue
        all_project_pairs[project] = pairs
        n_pos = sum(1 for p in pairs if p["label"] == 1)
        n_neg = len(pairs) - n_pos
        ratio_ok = "PARTIAL" if args.max_pairs is not None else ("OK" if n_neg == 3 * n_pos else "CHECK")
        print(f"    {project:<10} pairs={len(pairs):>5}  pos={n_pos:>4}  neg={n_neg:>5}  ratio={ratio_ok}")
    return all_project_pairs


# ---------------------------------------------------------------------------
# Parser and metrics - matched to Gemma/OpenAI synchronous zero-shot behavior
# ---------------------------------------------------------------------------

def parse_response(response_text):
    if not response_text:
        return None
    text = response_text.strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            val = data.get("is_linked")
            if val is not None:
                return 1 if val else 0
    except (json.JSONDecodeError, ValueError):
        pass
    lower = text.lower()
    if '"is_linked": true' in lower or '"is_linked":true' in lower:
        return 1
    if '"is_linked": false' in lower or '"is_linked":false' in lower:
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
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_conservative_metrics(predictions_log):
    cons_preds, cons_labels = [], []
    for e in predictions_log:
        pred = e["prediction"]
        cons_preds.append(0 if pred is None else pred)
        cons_labels.append(e["label"])
    return compute_metrics(cons_preds, cons_labels) if cons_preds else compute_metrics([], [])


# ---------------------------------------------------------------------------
# Batch request construction and token estimation
# ---------------------------------------------------------------------------

def build_custom_id(project, idx):
    return f"{project}_{idx:05d}"


def build_chat_body(prompt_text, args):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }


def build_requests_and_manifest(all_project_pairs, args):
    requests = []
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "model": args.model,
        "endpoint": "/v1/chat/completions",
        "prompt_mode": PROMPT_MODE,
        "prompt_template_hash": prompt_template_hash(),
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "message_format": "single user-role message; no system prompt; no JSON mode; OpenAI Batch Chat Completions",
        "negative_protocol": "Qwen3 top-K hard negatives, 1:3 positive:negative test pairs",
        "projects": args.projects,
        "start_index": args.start_index,
        "max_pairs": args.max_pairs,
        "run_suffix": args.run_suffix,
        "requests": {},
    }

    for project, pairs in all_project_pairs.items():
        for idx, pair in enumerate(pairs):
            custom_id = build_custom_id(project, idx)
            prompt_text = create_prompt_text(pair)
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_chat_body(prompt_text, args),
            })
            manifest["requests"][custom_id] = {
                "project": project,
                "index": idx,
                "source_id": pair["source_id"],
                "target_id": pair["target_id"],
                "label": pair["label"],
                "prompt_chars": len(prompt_text),
            }
    return requests, manifest


def estimate_prompt_tokens_for_body(body):
    text = body["messages"][0]["content"]
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(body["model"])
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        # Approximate Chat Completions wrapper overhead for one user message.
        return len(enc.encode(text)) + 12
    except Exception:
        # Conservative fallback used when tiktoken is unavailable.
        return max(1, int(len(text) / 3.7) + 12)


def estimate_batch_tokens(requests, args):
    prompt_tokens = sum(estimate_prompt_tokens_for_body(req["body"]) for req in requests)
    completion_tokens_upper = len(requests) * args.max_completion_tokens
    return prompt_tokens, completion_tokens_upper


def write_jsonl(requests, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# OpenAI execution modes
# ---------------------------------------------------------------------------

def make_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def smoke_test(client, requests, args):
    print("\nRunning synchronous smoke test on the first request...")
    req = requests[0]
    t0 = time.time()
    response = client.chat.completions.create(**req["body"])
    elapsed_ms = (time.time() - t0) * 1000.0
    content = response.choices[0].message.content or ""
    pred = parse_response(content)
    usage = response.usage
    prompt_tokens = int(get_attr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(get_attr(usage, "completion_tokens", 0) or 0)
    batch_equiv_cost = estimate_cost_usd(args.model, prompt_tokens, completion_tokens)

    print(f"  custom_id:          {req['custom_id']}")
    print(f"  response:           {content.strip()!r}")
    print(f"  parsed:             {pred}")
    print(f"  prompt tokens:      {prompt_tokens}")
    print(f"  completion tokens:  {completion_tokens}")
    print(f"  latency:            {elapsed_ms:.0f} ms")
    if batch_equiv_cost is not None:
        print(f"  rough sync cost:    ${batch_equiv_cost * 2:.5f} "
              f"(batch-equivalent would be ${batch_equiv_cost:.5f})")


def submit_batch(client, requests, manifest, output_dir, args):
    if args.budget_usd is None:
        print("\nREFUSING TO SUBMIT: --budget-usd is required for paid batch submission.")
        print("Example: python 9d_zero_shot_openai_batch_matched.py --submit --budget-usd 9.00")
        sys.exit(1)

    prompt_est, completion_est = estimate_batch_tokens(requests, args)
    cost_est = estimate_cost_usd(args.model, prompt_est, completion_est)
    manifest["estimated_prompt_tokens"] = prompt_est
    manifest["estimated_completion_tokens_upper"] = completion_est
    manifest["estimated_batch_cost_usd_upper"] = round(cost_est, 4) if cost_est is not None else None
    manifest["cost_estimate_basis"] = "tiktoken/o200k_base if available, else char/3.7; max output token cap"
    manifest["pricing_per_1M_batch"] = PRICING_PER_1M_TOKENS.get(args.model)

    if cost_est is None:
        print("\nWARNING: Model is not in pricing dictionary, so budget guard cannot estimate cost.")
        print("Submission blocked. Add pricing to PRICING_PER_1M_TOKENS or pass a known --model.")
        sys.exit(3)

    if cost_est > args.budget_usd:
        atomic_save_json(manifest, output_dir / "batch_manifest_presubmit.json")
        print(f"\nBUDGET GUARD: estimated upper cost ${cost_est:.4f} > ${args.budget_usd:.2f}")
        print("Submission blocked. Increase --budget-usd only if you accept the risk.")
        sys.exit(3)

    input_jsonl = output_dir / "batch_requests.jsonl"
    write_jsonl(requests, input_jsonl)
    atomic_save_json(manifest, output_dir / "batch_manifest_presubmit.json")

    print("\nSubmitting OpenAI Batch...")
    print(f"  Requests:      {len(requests)}")
    print(f"  Model:         {args.model}")
    print(f"  Endpoint:      /v1/chat/completions")
    print(f"  Max tokens:    {args.max_completion_tokens}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Seed:          {args.seed}")
    print(f"  Est. upper $:  ${cost_est:.4f}")
    print(f"  Request file:  {input_jsonl}")

    with open(input_jsonl, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    uploaded_dict = obj_to_dict(uploaded)
    manifest["input_file_id"] = get_attr(uploaded, "id")
    manifest["input_file"] = uploaded_dict
    atomic_save_json(manifest, output_dir / "batch_manifest_uploaded.json")

    batch = client.batches.create(
        input_file_id=manifest["input_file_id"],
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "thesis_zero_shot_gpt54_matched",
            "prompt_hash": prompt_template_hash(),
            "method": "zero_shot_batch",
        },
    )
    batch_dict = obj_to_dict(batch)
    manifest["batch"] = batch_dict
    manifest["batch_id"] = get_attr(batch, "id")
    manifest["submitted_at"] = datetime.now().isoformat()
    atomic_save_json(manifest, output_dir / "batch_manifest.json")
    atomic_save_json(batch_dict, output_dir / "batch_submission.json")

    print(f"\nSubmitted batch: {manifest['batch_id']}")
    print(f"Manifest saved:  {output_dir / 'batch_manifest.json'}")
    print("\nLater, collect with:")
    print(f"  python {Path(__file__).name} --collect --batch-id {manifest['batch_id']} --poll")
    return manifest["batch_id"]


def list_recent_batches(client, args):
    print("\nRecent OpenAI Batches:")
    listed = 0
    for batch in client.batches.list(limit=args.list_limit):
        listed += 1
        d = obj_to_dict(batch)
        print(f"  {d.get('id')}  status={d.get('status')}  endpoint={d.get('endpoint')}  "
              f"created={d.get('created_at')}  counts={d.get('request_counts')}  "
              f"output={d.get('output_file_id')}  error={d.get('error_file_id')}")
    if listed == 0:
        print("  No batches returned.")


def poll_batch(client, batch_id, args):
    while True:
        batch = client.batches.retrieve(batch_id)
        d = obj_to_dict(batch)
        status = d.get("status")
        counts = d.get("request_counts")
        print(f"  batch {batch_id}: status={status} counts={counts}")
        if status in {"completed", "failed", "expired", "cancelled", "canceled"}:
            return batch
        if not args.poll:
            return batch
        time.sleep(args.poll_seconds)


def iter_jsonl_file_content(client, file_id):
    content = client.files.content(file_id)
    raw = None
    if hasattr(content, "read"):
        raw = content.read()
    elif hasattr(content, "content"):
        raw = content.content
    elif isinstance(content, (bytes, bytearray)):
        raw = bytes(content)
    else:
        raw = str(content).encode("utf-8")

    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)

    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def reconstruct_manifest_if_missing(output_dir, all_project_pairs, args):
    manifest_path = output_dir / "batch_manifest.json"
    if manifest_path.exists():
        return load_json(manifest_path)
    print(f"\nWARNING: {manifest_path} not found. Reconstructing manifest from dataset.")
    _, manifest = build_requests_and_manifest(all_project_pairs, args)
    manifest["reconstructed_for_collect"] = True
    atomic_save_json(manifest, output_dir / "batch_manifest_reconstructed.json")
    return manifest


def extract_chat_response_and_usage(body):
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        response_text = (message.get("content") or "").strip()
    else:
        response_text = ""
    usage = body.get("usage") or {}
    return response_text, {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def collect_results(client, batch_id, output_dir, all_project_pairs, args):
    manifest = reconstruct_manifest_if_missing(output_dir, all_project_pairs, args)
    request_meta = manifest["requests"]

    batch = poll_batch(client, batch_id, args)
    batch_dict = obj_to_dict(batch)
    atomic_save_json(batch_dict, output_dir / "batch_status_at_collect.json")

    status = batch_dict.get("status")
    if status != "completed":
        print(f"Batch is not completed. Current status={status}.")
        if batch_dict.get("error_file_id"):
            print(f"Error file id: {batch_dict.get('error_file_id')}")
        return

    output_file_id = batch_dict.get("output_file_id")
    error_file_id = batch_dict.get("error_file_id")
    if not output_file_id:
        print("ERROR: completed batch has no output_file_id.")
        sys.exit(1)

    raw_output_path = output_dir / "batch_results_raw.jsonl"
    raw_error_path = output_dir / "batch_errors_raw.jsonl"

    by_project = {p: [] for p in ALL_PROJECTS}
    total_results = 0

    output_lines = list(iter_jsonl_file_content(client, output_file_id))
    with open(raw_output_path, "w", encoding="utf-8") as f:
        for item in output_lines:
            total_results += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

            custom_id = item.get("custom_id")
            meta = request_meta.get(custom_id)
            if not meta:
                continue

            response = item.get("response")
            error = item.get("error")

            response_text = ""
            prediction = None
            parse_success = False
            api_success = False
            api_error = error
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            status_code = None

            if response is not None and response.get("status_code") == 200:
                api_success = True
                status_code = response.get("status_code")
                body = response.get("body") or {}
                response_text, usage = extract_chat_response_and_usage(body)
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                total_tokens = usage["total_tokens"]
                prediction = parse_response(response_text)
                parse_success = prediction is not None
            else:
                status_code = response.get("status_code") if response else None
                if api_error is None and response is not None:
                    api_error = response.get("body")

            by_project[meta["project"]].append({
                "source_id": meta["source_id"],
                "target_id": meta["target_id"],
                "label": meta["label"],
                "prediction": prediction,
                "response": response_text[:200],
                "api_success": api_success,
                "api_error": api_error,
                "parse_success": parse_success,
                "latency_ms": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "custom_id": custom_id,
                "status_code": status_code,
            })

    if error_file_id:
        error_lines = list(iter_jsonl_file_content(client, error_file_id))
        with open(raw_error_path, "w", encoding="utf-8") as f:
            for item in error_lines:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    all_results = []
    for project, rows in by_project.items():
        if not rows:
            continue
        rows.sort(key=lambda e: int(e["custom_id"].split("_")[-1]))
        atomic_save_json(rows, output_dir / f"{project}_predictions.json")
        all_results.append(summarize_project(project, rows, len(rows)))

    print(f"\nCollected {total_results} batch output records.")
    write_final_report(all_results, output_dir, args, manifest, batch_dict)


def summarize_project(project, predictions, n_pairs_total):
    api_successes = sum(1 for e in predictions if e.get("api_success", True))
    api_failures = len(predictions) - api_successes
    parse_failures = sum(1 for e in predictions
                         if e.get("api_success", True) and not e.get("parse_success", True))

    clean_preds = [e["prediction"] for e in predictions
                   if e.get("api_success", True) and e["prediction"] is not None]
    clean_labels = [e["label"] for e in predictions
                    if e.get("api_success", True) and e["prediction"] is not None]

    metrics_clean = compute_metrics(clean_preds, clean_labels) if clean_preds else compute_metrics([], [])
    metrics_conservative = compute_conservative_metrics(predictions)

    total_prompt = sum(e.get("prompt_tokens", 0) for e in predictions)
    total_completion = sum(e.get("completion_tokens", 0) for e in predictions)

    n_pos = sum(1 for e in predictions if e["label"] == 1)
    n_neg = len(predictions) - n_pos

    return {
        "project": project,
        "n_pairs": len(predictions),
        "n_pairs_total_expected": n_pairs_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "api_successes": api_successes,
        "api_failures": api_failures,
        "parse_failures": parse_failures,
        "metrics_clean": metrics_clean,
        "metrics_conservative": metrics_conservative,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "avg_latency_ms": None,
    }


def write_final_report(all_results, output_dir, args, manifest, batch_dict):
    if not all_results:
        print("No successful project results to summarize.")
        return

    macro_p = float(np.mean([r["metrics_clean"]["precision"] for r in all_results]))
    macro_r = float(np.mean([r["metrics_clean"]["recall"] for r in all_results]))
    macro_f1 = float(np.mean([r["metrics_clean"]["f1"] for r in all_results]))
    macro_f2 = float(np.mean([r["metrics_clean"]["f2"] for r in all_results]))
    macro_acc = float(np.mean([r["metrics_clean"]["accuracy"] for r in all_results]))

    macro_p_c = float(np.mean([r["metrics_conservative"]["precision"] for r in all_results]))
    macro_r_c = float(np.mean([r["metrics_conservative"]["recall"] for r in all_results]))
    macro_f1_c = float(np.mean([r["metrics_conservative"]["f1"] for r in all_results]))
    macro_f2_c = float(np.mean([r["metrics_conservative"]["f2"] for r in all_results]))

    total_pairs = sum(r["n_pairs"] for r in all_results)
    total_api_fails = sum(r["api_failures"] for r in all_results)
    total_parse_fails = sum(r["parse_failures"] for r in all_results)
    total_prompt = sum(r["total_prompt_tokens"] for r in all_results)
    total_completion = sum(r["total_completion_tokens"] for r in all_results)
    cost = estimate_cost_usd(args.model, total_prompt, total_completion)

    print(f"\n\n{'=' * 95}")
    print(f"ZERO-SHOT (MATCHED PROMPT) - {args.model} - OPENAI BATCH - CLEAN macro")
    print("=" * 95)
    print(f"{'Project':<12} {'Pairs':>6} {'F1':>7} {'F2':>7} {'P':>7} {'R':>7} "
          f"{'Acc':>7} {'Parse':>6} {'API':>5} {'InTok':>10} {'OutTok':>8}")
    print("-" * 95)
    for r in all_results:
        m = r["metrics_clean"]
        print(f"{r['project']:<12} {r['n_pairs']:>6} {m['f1']:>7.4f} {m['f2']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['accuracy']:>7.4f} "
              f"{r['parse_failures']:>6} {r['api_failures']:>5} "
              f"{r['total_prompt_tokens']:>10,} {r['total_completion_tokens']:>8,}")
    print("-" * 95)
    print(f"{'MACRO AVG':<12} {total_pairs:>6} {macro_f1:>7.4f} {macro_f2:>7.4f} "
          f"{macro_p:>7.4f} {macro_r:>7.4f} {macro_acc:>7.4f} "
          f"{total_parse_fails:>6} {total_api_fails:>5}")
    print(f"\n  Total tokens:   {total_prompt:>10,} input + {total_completion:>8,} output")
    if cost is not None:
        print(f"  Est. batch cost: ${cost:.4f} ({args.model} batch pricing - invoice is authoritative)")
    print("  Latency note:   batch mode is async; exclude from single-stream latency comparison.")

    output = {
        "timestamp": manifest.get("created_at"),
        "completed_at": datetime.now().isoformat(),
        "model": args.model,
        "experiment": "openai_gpt54_zero_shot_matched_prompt_batch_v3",
        "batch_id": manifest.get("batch_id") or batch_dict.get("id"),
        "matched_against": [
            "5.zero_shot_h100.py (Gemma 4 31B local H100)",
            "9a_zero_shot_openai_matched.py (GPT-5.4-mini synchronous OpenAI API)",
            "9c_zero_shot_claude_batch_matched.py (Claude Sonnet 4.6 Anthropic Batch)",
        ],
        "prompt_mode": PROMPT_MODE,
        "prompt_template_hash": prompt_template_hash(),
        "message_format": "single user-role message; no system prompt; no JSON mode; OpenAI native Batch Chat Completions",
        "execution_mode": "OpenAI Batch API",
        "latency_comparability": "Not comparable to single-stream synchronous runs; use only for classification metrics and batch token cost.",
        "pricing_per_1M_batch": PRICING_PER_1M_TOKENS.get(args.model),
        "generation": {
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "macro_clean": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
            "f2": round(macro_f2, 4),
            "accuracy": round(macro_acc, 4),
        },
        "macro_conservative": {
            "precision": round(macro_p_c, 4),
            "recall": round(macro_r_c, 4),
            "f1": round(macro_f1_c, 4),
            "f2": round(macro_f2_c, 4),
        },
        "total_pairs": total_pairs,
        "total_api_failures": total_api_fails,
        "total_parse_failures": total_parse_fails,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "estimated_batch_cost_usd": round(cost, 4) if cost is not None else None,
        "per_project": all_results,
        "batch": batch_dict,
    }
    atomic_save_json(output, output_dir / "results.json")
    print(f"\nSaved results to: {output_dir / 'results.json'}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched-prompt OpenAI GPT-5.4 zero-shot run via Batch API."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-root", default=BASE_OUTPUT_DIR)
    parser.add_argument("--projects", nargs="+", choices=ALL_PROJECTS, default=ALL_PROJECTS)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--budget-usd", type=float, default=None,
                        help="Block batch submission if estimated batch cost exceeds this amount.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Per-project cap for smoke tests or partial batches.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Per-project start offset before applying --max-pairs. Useful for sharding large projects.")
    parser.add_argument("--run-suffix", default=None,
                        help="Append a suffix to the output directory so sharded runs do not overwrite each other.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--list-limit", type=int, default=10)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print first prompt and request shape. No API calls.")
    mode.add_argument("--estimate-only", action="store_true",
                      help="Estimate batch cost locally and write presubmit manifest. No API calls.")
    mode.add_argument("--smoke-test", action="store_true",
                      help="Run one synchronous Chat Completions API call on first request.")
    mode.add_argument("--submit", action="store_true",
                      help="Upload JSONL and submit an OpenAI Batch.")
    mode.add_argument("--collect", action="store_true",
                      help="Collect and score a completed batch.")
    mode.add_argument("--list-batches", action="store_true",
                      help="List recent OpenAI batches.")

    parser.add_argument("--batch-id", default=None,
                        help="Existing OpenAI batch id for collect/poll.")
    parser.add_argument("--poll", action="store_true",
                        help="When collecting, poll until the batch reaches a terminal state.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_slug = sanitize_model_name(args.model)
    run_name = f"{model_slug}_{PROMPT_MODE}"
    if args.run_suffix:
        run_name = f"{run_name}_{sanitize_model_name(args.run_suffix)}"
    output_dir = Path(args.output_root) / run_name
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    pricing = PRICING_PER_1M_TOKENS.get(args.model)
    pricing_str = (f"${pricing['input']}/1M in, ${pricing['output']}/1M out"
                   if pricing else "UNKNOWN - budget guard disabled")

    print("=" * 90)
    print("ZERO-SHOT (MATCHED PROMPT) - OpenAI Batch API")
    print("=" * 90)
    print(f"Timestamp:           {timestamp}")
    print(f"Model:               {args.model}")
    print(f"Batch pricing:       {pricing_str}")
    print(f"Prompt mode:         {PROMPT_MODE}")
    print(f"Prompt hash:         {prompt_template_hash()}")
    print("Message format:      single user-role message; no system; no JSON mode")
    print(f"Max completion tok:  {args.max_completion_tokens}")
    print(f"Temperature:         {args.temperature}")
    print(f"Seed:                {args.seed}")
    print("Execution mode:      OpenAI Batch API (/v1/chat/completions)")
    print(f"Budget guard:        " + (f"${args.budget_usd:.2f}" if args.budget_usd else "OFF"))
    print(f"Projects:            {args.projects}")
    print(f"Start index:         {args.start_index}")
    print(f"Max pairs/project:   {args.max_pairs if args.max_pairs is not None else 'ALL'}")
    print(f"Run suffix:          {args.run_suffix if args.run_suffix else 'NONE'}")
    print(f"Output:              {output_dir}")
    print("=" * 90)

    all_project_pairs = load_all_project_pairs(args)
    if not all_project_pairs:
        print("\nNo projects with valid data. Exiting.")
        sys.exit(1)

    requests, manifest = build_requests_and_manifest(all_project_pairs, args)

    if args.dry_run or not any([args.estimate_only, args.smoke_test, args.submit, args.collect, args.list_batches]):
        first_project = next(iter(all_project_pairs))
        first_pair = all_project_pairs[first_project][0]
        prompt = create_prompt_text(first_pair)
        print("\n" + "-" * 90)
        print(f"DRY RUN - first prompt from {first_project} (label={first_pair['label']})")
        print("-" * 90)
        print(prompt)
        print("-" * 90)
        print(f"Prompt char length: {len(prompt)}")
        print(f"Prompt SHA256 (16): {prompt_template_hash()}")
        print("\nFirst batch request shape:")
        print(json.dumps(requests[0], indent=2, ensure_ascii=False)[:2200])
        print("\nNo API calls were made.")
        return

    if args.estimate_only:
        prompt_est, completion_est = estimate_batch_tokens(requests, args)
        cost_est = estimate_cost_usd(args.model, prompt_est, completion_est)
        manifest["estimated_prompt_tokens"] = prompt_est
        manifest["estimated_completion_tokens_upper"] = completion_est
        manifest["estimated_batch_cost_usd_upper"] = round(cost_est, 4) if cost_est is not None else None
        manifest["pricing_per_1M_batch"] = PRICING_PER_1M_TOKENS.get(args.model)
        atomic_save_json(manifest, output_dir / "batch_manifest_presubmit.json")
        print("\nLocal estimate only:")
        print(f"  Requests:        {len(requests)}")
        print(f"  Prompt tokens:   {prompt_est:,}")
        print(f"  Output cap:      {completion_est:,}")
        if cost_est is not None:
            print(f"  Est. upper cost: ${cost_est:.4f}")
            if args.budget_usd is not None:
                print(f"  Budget status:   {'OK' if cost_est <= args.budget_usd else 'OVER'} "
                      f"(${args.budget_usd:.2f})")
        else:
            print("  Est. upper cost: UNKNOWN - model missing from pricing dict")
        print(f"  Manifest saved:  {output_dir / 'batch_manifest_presubmit.json'}")
        return

    client = make_client()

    if args.list_batches:
        list_recent_batches(client, args)
        return

    if args.smoke_test:
        smoke_test(client, requests, args)
        return

    if args.submit:
        submit_batch(client, requests, manifest, output_dir, args)
        return

    if args.collect:
        batch_id = args.batch_id
        if batch_id is None:
            manifest_path = output_dir / "batch_manifest.json"
            if manifest_path.exists():
                batch_id = load_json(manifest_path).get("batch_id")
        if not batch_id:
            print("ERROR: Provide --batch-id or run from an output folder with batch_manifest.json.")
            sys.exit(1)
        collect_results(client, batch_id, output_dir, all_project_pairs, args)
        return


if __name__ == "__main__":
    main()
