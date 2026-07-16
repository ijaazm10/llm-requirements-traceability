"""
Zero-Shot - Anthropic Claude Batch API (Matched-Prompt Cloud Comparison)
========================================================================

Methodologically matched counterpart to:
  - 5.zero_shot_h100.py                    (Gemma 4 31B local H100)
  - 9a_zero_shot_openai_matched.py         (GPT-5.4-mini OpenAI API)

What is held constant against the Gemma/OpenAI reference runs
-------------------------------------------------------------
* Same test data:        splits/final_pairs_test.json (Qwen3 top-K hard negs)
* Same data loader:      reads requirements.json as a list, builds id_map,
                         joins by source_id/target_id, empty description -> "N/A"
* Same prompt text:      character-for-character identical to create_prompt_text
                         in 5.zero_shot_h100.py and 9a_zero_shot_openai_matched.py
* Same role structure:   single user-role message; no system role; no tools
* Same generation:       max_tokens=20, temperature=0.0
* Same parser:           json-window extract -> substring fallback
* Same metrics:          P, R, F1, F2(beta=2), accuracy; clean + conservative

What differs by necessity
-------------------------
* Execution mode:        Anthropic Message Batches API is asynchronous. This is
                         valid for classification quality, but NOT comparable for
                         interactive latency. Do not use batch wall-clock time in
                         RQ3 single-stream latency tables.
* Token counts:          Claude uses Anthropic tokenization. Token counts differ
                         from Gemma/OpenAI even when the semantic prompt is the same.
* Chat formatting:       Claude receives the same semantic payload as one user
                         message and Anthropic applies native chat formatting.

Operational safety
------------------
* No full API submission unless --submit is passed.
* Optional free Anthropic token counting via --count-tokens before submission.
* Budget guard blocks submission when the estimated batch cost exceeds --budget-usd.
* Saves request manifest with pair metadata and prompt hash before submission.
* Collect/scoring phase maps unordered batch results by custom_id.
* Result files use the same per-project prediction JSON format as the OpenAI run.

Usage
-----
Dry run, no API calls:
    python 9c_zero_shot_claude_batch_matched.py --dry-run

Free exact input-token estimate using Anthropic token counting:
    python 9c_zero_shot_claude_batch_matched.py --count-tokens --budget-usd 6.50

Tiny paid synchronous smoke test, no batch:
    python 9c_zero_shot_claude_batch_matched.py --smoke-test --projects AAH --max-pairs 2 --budget-usd 0.05

Submit the full batch. By default this first runs free Anthropic token counting
so the budget guard uses exact input tokens plus the max output cap:
    python 9c_zero_shot_claude_batch_matched.py --submit --budget-usd 6.50

Poll until finished and collect results:
    python 9c_zero_shot_claude_batch_matched.py --collect --poll

Collect an existing batch id:
    python 9c_zero_shot_claude_batch_matched.py --collect --batch-id msgbatch_...

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

DATA_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/DATA/GROUND_TRUTH"
BASE_OUTPUT_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS/CLAUDE_ZERO_SHOT_BATCH_V3"

# Anthropic 4.6+ model IDs use the dateless canonical format. Verified against
# Anthropic model-ID docs on 2026-06-04.
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 20
DEFAULT_TEMPERATURE = 0.0

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

# Batch prices from Anthropic pricing docs. Provider invoice is authoritative.
# Verified against Anthropic pricing docs on 2026-06-04.
PRICING_PER_1M_TOKENS = {
    "claude-sonnet-4-6": {"input": 1.50, "output": 7.50},
    "claude-sonnet-4-5": {"input": 1.50, "output": 7.50},
    "claude-haiku-4-5-20251001": {"input": 0.50, "output": 2.50},
    "claude-opus-4-7": {"input": 2.50, "output": 12.50},
    "claude-opus-4-6": {"input": 2.50, "output": 12.50},
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


def estimate_cost_usd(model, input_tokens, output_tokens):
    pricing = PRICING_PER_1M_TOKENS.get(model)
    if not pricing:
        return None
    return (input_tokens / 1e6) * pricing["input"] + (output_tokens / 1e6) * pricing["output"]


def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def result_to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return json.loads(obj.json())


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
# Parser and metrics - matched to Gemma/OpenAI zero-shot behavior
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
# Anthropic request construction
# ---------------------------------------------------------------------------

def build_custom_id(project, idx):
    return f"{project}_{idx:05d}"


def build_message_params(prompt_text, args):
    params = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    # Do not set thinking/effort. The matched zero-shot condition is ordinary
    # native chat inference, not an extended-thinking run.
    return params


def build_requests_and_manifest(all_project_pairs, args):
    requests = []
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "model": args.model,
        "prompt_mode": PROMPT_MODE,
        "prompt_template_hash": prompt_template_hash(),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "message_format": "single user-role message; no system prompt; no tools; batch Messages API",
        "negative_protocol": "Qwen3 top-K hard negatives, 1:3 positive:negative test pairs",
        "projects": args.projects,
        "requests": {},
    }

    for project, pairs in all_project_pairs.items():
        for idx, pair in enumerate(pairs):
            custom_id = build_custom_id(project, idx)
            prompt_text = create_prompt_text(pair)
            requests.append({
                "custom_id": custom_id,
                "params": build_message_params(prompt_text, args),
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


def crude_token_estimate(requests, args):
    # Conservative-ish local estimate when Anthropic count_tokens is not used.
    # This is not used for final cost reporting.
    input_tokens = 0
    for req in requests:
        text = req["params"]["messages"][0]["content"]
        input_tokens += max(1, int(len(text) / 3.7) + 10)
    output_tokens = len(requests) * args.max_tokens
    return input_tokens, output_tokens


def exact_count_tokens(client, requests, args):
    total = 0
    counts = {}
    t0 = time.time()
    for i, req in enumerate(requests, 1):
        params = req["params"]
        response = None
        for attempt in range(5):
            try:
                response = client.messages.count_tokens(
                    model=params["model"],
                    messages=params["messages"],
                )
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "rate" in msg or "429" in msg or "too many" in msg:
                    wait = 30 * (attempt + 1)
                    print(f"\n    token count rate limit at {i}/{len(requests)}; waiting {wait}s...")
                    time.sleep(wait)
                    continue
                raise
        if response is None:
            raise RuntimeError(f"Token counting failed repeatedly at request {i}/{len(requests)}")
        n = get_attr(response, "input_tokens")
        counts[req["custom_id"]] = int(n)
        total += int(n)
        if i == 1 or i % args.print_every == 0:
            elapsed = time.time() - t0
            rate = elapsed / i
            remaining = rate * (len(requests) - i)
            print(f"    token count {i}/{len(requests)} (~{remaining/60:.1f} min)", end="\r")
        if args.token_count_delay > 0:
            time.sleep(args.token_count_delay)
    print(" " * 90, end="\r")
    return total, counts


# ---------------------------------------------------------------------------
# Anthropic execution modes
# ---------------------------------------------------------------------------

def make_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def smoke_test(client, requests, args):
    print("\nRunning synchronous smoke test on the first request...")
    req = requests[0]
    t0 = time.time()
    message = client.messages.create(**req["params"])
    elapsed_ms = (time.time() - t0) * 1000.0
    response_text, usage = extract_message_text_and_usage(message)
    pred = parse_response(response_text)
    print(f"  custom_id:     {req['custom_id']}")
    print(f"  response:      {response_text!r}")
    print(f"  parsed:        {pred}")
    print(f"  input tokens:  {usage['input_tokens']}")
    print(f"  output tokens: {usage['output_tokens']}")
    print(f"  latency:       {elapsed_ms:.0f} ms")
    cost = estimate_cost_usd(args.model, usage["input_tokens"], usage["output_tokens"])
    if cost is not None:
        # Synchronous smoke test is standard price, not batch price. Double batch
        # estimate to approximate sync price for this tiny test.
        print(f"  rough sync cost: ${cost * 2:.5f} (batch-equivalent would be ${cost:.5f})")


def submit_batch(client, requests, manifest, output_dir, args):
    if args.budget_usd is None:
        print("\nREFUSING TO SUBMIT: --budget-usd is required for paid batch submission.")
        print("Example: python 9c_zero_shot_claude_batch_matched.py --submit --budget-usd 6.50")
        sys.exit(1)

    est_input = manifest.get("exact_input_tokens")
    est_output = len(requests) * args.max_tokens
    if est_input is None:
        est_input, est_output = crude_token_estimate(requests, args)
        manifest["crude_input_tokens"] = est_input
        manifest["crude_output_tokens_max"] = est_output
        manifest["cost_estimate_basis"] = "local char/3.7 input estimate + max output tokens"
    else:
        manifest["max_output_tokens_if_all_use_cap"] = est_output
        manifest["cost_estimate_basis"] = "Anthropic count_tokens input + max output tokens"

    est_cost = estimate_cost_usd(args.model, est_input, est_output)
    manifest["estimated_batch_cost_usd_upper"] = round(est_cost, 4) if est_cost is not None else None
    atomic_save_json(manifest, output_dir / "batch_manifest.json")

    if args.budget_usd is not None and est_cost is not None and est_cost > args.budget_usd:
        print(f"\nBUDGET GUARD: estimated upper cost ${est_cost:.4f} > ${args.budget_usd:.2f}")
        print("Submission blocked. Use --budget-usd with a higher cap only if you accept the risk.")
        sys.exit(3)

    print("\nSubmitting Anthropic Message Batch...")
    print(f"  Requests:      {len(requests)}")
    print(f"  Model:         {args.model}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Temperature:   {args.temperature}")
    if est_cost is not None:
        print(f"  Est. upper $:  ${est_cost:.4f} ({manifest['cost_estimate_basis']})")

    batch = client.messages.batches.create(requests=requests)
    batch_dict = result_to_dict(batch)
    manifest["batch"] = batch_dict
    manifest["batch_id"] = get_attr(batch, "id")
    manifest["submitted_at"] = datetime.now().isoformat()
    atomic_save_json(manifest, output_dir / "batch_manifest.json")
    atomic_save_json(batch_dict, output_dir / "batch_submission.json")

    print(f"\nSubmitted batch: {manifest['batch_id']}")
    print(f"Manifest saved:  {output_dir / 'batch_manifest.json'}")
    print("\nLater, collect with:")
    print(f"  python {Path(__file__).name} --collect --batch-id {manifest['batch_id']}")
    return manifest["batch_id"]


def list_recent_batches(client, args):
    print("\nRecent Anthropic Message Batches:")
    listed = 0
    for batch in client.messages.batches.list(limit=args.list_limit):
        listed += 1
        batch_dict = result_to_dict(batch)
        batch_id = batch_dict.get("id")
        status = batch_dict.get("processing_status")
        created_at = batch_dict.get("created_at")
        ended_at = batch_dict.get("ended_at")
        counts = batch_dict.get("request_counts")
        print(f"  {batch_id}  status={status}  created={created_at}  ended={ended_at}  counts={counts}")
    if listed == 0:
        print("  No batches returned.")


def poll_batch(client, batch_id, args):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = get_attr(batch, "processing_status")
        counts = get_attr(batch, "request_counts", None)
        print(f"  batch {batch_id}: status={status} counts={counts}")
        if status == "ended":
            return batch
        if not args.poll:
            return batch
        time.sleep(args.poll_seconds)


def extract_message_text_and_usage(message):
    content = get_attr(message, "content", []) or []
    parts = []
    for block in content:
        block_type = get_attr(block, "type")
        if block_type == "text":
            parts.append(get_attr(block, "text", "") or "")
    usage_obj = get_attr(message, "usage", None)
    usage = {
        "input_tokens": int(get_attr(usage_obj, "input_tokens", 0) or 0),
        "output_tokens": int(get_attr(usage_obj, "output_tokens", 0) or 0),
    }
    return "".join(parts).strip(), usage


def collect_results(client, batch_id, output_dir, args):
    manifest_path = output_dir / "batch_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        print("Run from the same output directory/root, or pass matching --model/--output-root.")
        sys.exit(1)

    manifest = load_json(manifest_path)
    request_meta = manifest["requests"]

    batch = poll_batch(client, batch_id, args)
    status = get_attr(batch, "processing_status")
    atomic_save_json(result_to_dict(batch), output_dir / "batch_status_at_collect.json")
    if status != "ended":
        print("Batch has not ended yet. Re-run with --collect --poll, or collect later.")
        return

    by_project = {p: [] for p in ALL_PROJECTS}
    raw_results_path = output_dir / "batch_results_raw.jsonl"
    total_results = 0

    with open(raw_results_path, "w", encoding="utf-8") as raw_f:
        for result in client.messages.batches.results(batch_id):
            total_results += 1
            raw_f.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")

            custom_id = get_attr(result, "custom_id")
            meta = request_meta.get(custom_id)
            if not meta:
                continue

            result_obj = get_attr(result, "result")
            result_type = get_attr(result_obj, "type")

            response_text = ""
            prediction = None
            parse_success = False
            api_success = False
            api_error = None
            input_tokens = 0
            output_tokens = 0
            stop_reason = None

            if result_type == "succeeded":
                api_success = True
                message = get_attr(result_obj, "message")
                response_text, usage = extract_message_text_and_usage(message)
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]
                stop_reason = get_attr(message, "stop_reason")
                prediction = parse_response(response_text)
                parse_success = prediction is not None
            else:
                err = get_attr(result_obj, "error", None)
                api_error = result_to_dict(err) if err is not None else {"type": result_type}

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
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "custom_id": custom_id,
                "stop_reason": stop_reason,
            })

    # Sort back into original test order and save per-project prediction files.
    all_results = []
    for project, rows in by_project.items():
        if not rows:
            continue
        rows.sort(key=lambda e: int(e["custom_id"].split("_")[-1]))
        atomic_save_json(rows, output_dir / f"{project}_predictions.json")
        all_results.append(summarize_project(project, rows, len(rows)))

    print(f"\nCollected {total_results} batch results.")
    write_final_report(all_results, output_dir, args, manifest)


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


def write_final_report(all_results, output_dir, args, manifest):
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
    print(f"ZERO-SHOT (MATCHED PROMPT) - {args.model} - ANTHROPIC BATCH - CLEAN macro")
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
        "experiment": "anthropic_claude_zero_shot_matched_prompt_batch_v3",
        "batch_id": manifest.get("batch_id"),
        "matched_against": [
            "5.zero_shot_h100.py (Gemma 4 31B local H100)",
            "9a_zero_shot_openai_matched.py (GPT-5.4-mini OpenAI API)",
        ],
        "prompt_mode": PROMPT_MODE,
        "prompt_template_hash": prompt_template_hash(),
        "message_format": "single user-role message; no system prompt; no tools; Anthropic native batch Messages API",
        "execution_mode": "Anthropic Message Batches API",
        "latency_comparability": "Not comparable to single-stream synchronous runs; use only for classification metrics and batch token cost.",
        "pricing_per_1M_batch": PRICING_PER_1M_TOKENS.get(args.model),
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "thinking": "not set",
            "tools": "not set",
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
    }
    atomic_save_json(output, output_dir / "results.json")
    print(f"\nSaved results to: {output_dir / 'results.json'}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched-prompt Anthropic Claude zero-shot run via Message Batches API."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-root", default=BASE_OUTPUT_DIR)
    parser.add_argument("--projects", nargs="+", choices=ALL_PROJECTS, default=ALL_PROJECTS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--budget-usd", type=float, default=None,
                        help="Block batch submission if estimated batch cost exceeds this amount.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Per-project cap for smoke tests or partial batches.")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--token-count-delay", type=float, default=0.02,
                        help="Delay between token counting requests to be gentle on RPM limits.")
    parser.add_argument("--skip-token-count", action="store_true",
                        help="For --submit only: skip Anthropic token counting and use crude local estimate.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print first prompt and request shape. No API calls.")
    mode.add_argument("--count-tokens", action="store_true",
                      help="Use Anthropic free token counting endpoint, then exit unless --submit is also used.")
    mode.add_argument("--smoke-test", action="store_true",
                      help="Run one synchronous Messages API call on first request.")
    mode.add_argument("--submit", action="store_true",
                      help="Submit a Message Batch.")
    mode.add_argument("--collect", action="store_true",
                      help="Collect and score an ended batch.")
    mode.add_argument("--list-batches", action="store_true",
                      help="List recent Anthropic batches. Useful if submission succeeded but local save failed.")

    parser.add_argument("--batch-id", default=None,
                        help="Existing Anthropic batch id for collect/poll.")
    parser.add_argument("--poll", action="store_true",
                        help="When collecting, poll until the batch ends.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--list-limit", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_slug = sanitize_model_name(args.model)
    output_dir = Path(args.output_root) / f"{model_slug}_{PROMPT_MODE}"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    pricing = PRICING_PER_1M_TOKENS.get(args.model)
    pricing_str = (f"${pricing['input']}/1M in, ${pricing['output']}/1M out"
                   if pricing else "UNKNOWN - budget guard disabled")

    print("=" * 90)
    print("ZERO-SHOT (MATCHED PROMPT) - Anthropic Claude Batch API")
    print("=" * 90)
    print(f"Timestamp:           {timestamp}")
    print(f"Model:               {args.model}")
    print(f"Batch pricing:       {pricing_str}")
    print(f"Prompt mode:         {PROMPT_MODE}")
    print(f"Prompt hash:         {prompt_template_hash()}")
    print("Message format:      single user-role message; no system; no tools")
    print(f"Max tokens:          {args.max_tokens}")
    print(f"Temperature:         {args.temperature}")
    print("Thinking/effort:     not set")
    print("Execution mode:      Anthropic Message Batches API")
    print(f"Budget guard:        " + (f"${args.budget_usd:.2f}" if args.budget_usd else "OFF"))
    print(f"Projects:            {args.projects}")
    print(f"Output:              {output_dir}")
    print("=" * 90)

    all_project_pairs = load_all_project_pairs(args)
    if not all_project_pairs:
        print("\nNo projects with valid data. Exiting.")
        sys.exit(1)

    requests, manifest = build_requests_and_manifest(all_project_pairs, args)

    if args.dry_run or not any([args.count_tokens, args.smoke_test, args.submit, args.collect, args.list_batches]):
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
        print(json.dumps(requests[0], indent=2, ensure_ascii=False)[:2000])
        print("\nNo API calls were made.")
        return

    client = make_client()

    if args.list_batches:
        list_recent_batches(client, args)
        return

    if args.count_tokens or (args.submit and not args.skip_token_count):
        print("\nCounting exact Anthropic input tokens...")
        exact_input, counts = exact_count_tokens(client, requests, args)
        manifest["exact_input_tokens"] = exact_input
        manifest["token_counts_by_custom_id"] = counts
        max_output = len(requests) * args.max_tokens
        upper_cost = estimate_cost_usd(args.model, exact_input, max_output)
        print(f"  Exact input tokens:       {exact_input:,}")
        print(f"  Max output token cap:     {max_output:,}")
        if upper_cost is not None:
            print(f"  Batch upper cost estimate: ${upper_cost:.4f}")
        atomic_save_json(manifest, output_dir / "batch_manifest_presubmit.json")
        if args.count_tokens and not args.submit:
            print(f"\nToken-count manifest saved: {output_dir / 'batch_manifest_presubmit.json'}")
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
        collect_results(client, batch_id, output_dir, args)
        return


if __name__ == "__main__":
    main()
