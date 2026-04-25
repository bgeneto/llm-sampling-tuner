"""
Parameter Sweep Runner
======================
Runs model through the parameter grid and collects graded results.

Strategy:
  Phase 1 (COARSE): Test a strategic subset of the full grid using 2 prompts per mode.
          Pick top-10 param combos per mode.
  Phase 2 (FOCUSED): Test top-10 combos across ALL prompts with full N=5 sampling.
  Phase 3 (FINE): Zoom into the best region and test ±small deltas.

This approach keeps total API calls manageable:
  Coarse: ~600 combos × 2 prompts × 3 samples = ~3,600 calls
  Focused: 10 combos × 9 prompts × 5 samples = ~450 calls per mode
  Fine:    ~30 combos × 9 prompts × 5 samples = ~1,350 calls per mode

Total: ~7,000ish calls — feasible on local inference.
"""

import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import requests

from config import (API_BASE, API_KEY, DEFAULT_PARALLEL_REQUESTS,
                    DEFAULT_REASONING_PROFILES,
                    DEFAULT_USE_REASONING_AS_RESPONSE, MAX_TOKENS_CODER,
                    MAX_TOKENS_PLANNER, MODEL_ID, PARAM_COMBOS_STRATEGIC,
                    QWEN3_RECOMMENDED_COMBOS, SAMPLES_PER_COMBO,
                    expand_param_combos, normalize_reasoning_params,
                    resolve_reasoning_profiles, should_skip)
from grader import GradeResult, grade_coder, grade_planner, grade_stability
from prompts.coder_prompts import CODER_PROMPTS
from prompts.planner_prompts import PLANNER_PROMPTS

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
INTERNAL_PARAM_KEYS = {
    "reasoning_profile",
    "use_reasoning_as_response",
    "chat_template_kwargs",
    "thinking_token_budget",
}


def param_hash(params: dict) -> str:
    """Short hash for a param combo, for filenames."""
    s = json.dumps(normalize_reasoning_params(params), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def run_key(prompt_id: str, params: dict, sample_idx: int) -> tuple[str, str, int]:
    """Stable key for one prompt/parameter/sample execution."""
    return (prompt_id, param_hash(params), sample_idx)


def expected_run_keys(
    prompts: list[dict],
    param_combos: list[dict],
    n_samples: int,
) -> set[tuple[str, str, int]]:
    """Build the exact run-key set for a sweep request."""
    return {
        run_key(prompt["id"], params, sample_idx)
        for prompt in prompts
        for params in param_combos
        for sample_idx in range(n_samples)
    }


def format_param_combo(params: dict) -> str:
    """Compact human-readable representation of a full experiment combo."""
    params = normalize_reasoning_params(params)
    parts = [
        f"profile={params.get('reasoning_profile', 'unprofiled')}",
        f"t={params['temperature']}",
        f"top_p={params['top_p']}",
        f"top_k={params['top_k']}",
        f"min_p={params['min_p']}",
        f"rep={params['repeat_penalty']}",
    ]
    budget = params.get("thinking_token_budget")
    if budget is not None:
        parts.append(f"budget={budget}")
    return " ".join(parts)


def parse_reasoning_profiles_arg(raw_value: str | None) -> list[str]:
    """Parse comma-separated reasoning profile names."""
    if not raw_value:
        return list(DEFAULT_REASONING_PROFILES)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_request_max_tokens(answer_max_tokens: int, params: dict) -> int:
    """Return completion max_tokens with room for reasoning plus visible answer.

    Reasoning tokens consume the same completion budget as visible content.
    Treat the mode max token setting as the visible-answer budget, then add the
    thinking budget when thinking is enabled.
    """
    chat_template_kwargs = params.get("chat_template_kwargs") or {}
    thinking_budget = params.get("thinking_token_budget")
    if not chat_template_kwargs.get("enable_thinking") or thinking_budget is None:
        return answer_max_tokens

    thinking_budget = int(thinking_budget)
    if thinking_budget < 1:
        raise ValueError("thinking_token_budget must be at least 1")
    if answer_max_tokens < 1:
        raise ValueError("answer max_tokens must be at least 1")
    return thinking_budget + answer_max_tokens


def call_lmstudio(messages: list[dict], params: dict, max_tokens: int,
                   timeout: int = 180) -> dict:
    """Call LM Studio chat completions API. Returns raw response dict."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": resolve_request_max_tokens(max_tokens, params),
        "stream": False,
    }
    for key, value in params.items():
        if key in INTERNAL_PARAM_KEYS or value is None:
            continue
        payload[key] = value

    chat_template_kwargs = params.get("chat_template_kwargs")
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        thinking_token_budget = params.get("thinking_token_budget")
        if thinking_token_budget is not None:
            payload["chat_template_kwargs"]["thinking_token_budget"] = thinking_token_budget

    # Remove params that are 0/disabled to let LM Studio use defaults
    if payload.get("top_k") == 0:
        del payload["top_k"]

    headers = None
    if API_KEY:
        headers = {"Authorization": f"Bearer {API_KEY}"}

    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    body_preview = resp.text[:500].replace("\n", "\\n")
    content_type = resp.headers.get("content-type", "")

    if not resp.text.strip():
        raise RuntimeError(
            f"Empty response from API: status={resp.status_code}, "
            f"content_type={content_type!r}, payload_keys={sorted(payload)}"
        )

    try:
        raw = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Non-JSON response from API: status={resp.status_code}, "
            f"content_type={content_type!r}, body={body_preview!r}, "
            f"payload_keys={sorted(payload)}"
        ) from exc

    if resp.status_code >= 400:
        raise RuntimeError(
            f"API error: status={resp.status_code}, body={json.dumps(raw)[:500]}, "
            f"payload_keys={sorted(payload)}"
        )

    return raw


def extract_response_text(raw: dict, use_reasoning_as_response: bool = False) -> tuple[str, dict]:
    """Normalize assistant message content to text and capture response metadata."""
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning")
    reasoning_text = reasoning if isinstance(reasoning, str) else ""

    if isinstance(content, str):
        text = content
        content_type = "str"
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                part_text = part.get("text")
                if isinstance(part_text, str):
                    text_parts.append(part_text)
        text = "".join(text_parts)
        content_type = "list"
    elif content is None:
        text = ""
        content_type = "none"
    else:
        text = str(content)
        content_type = type(content).__name__

    used_reasoning_as_response = False
    if not text.strip() and use_reasoning_as_response and reasoning_text.strip():
        text = reasoning_text
        content_type = "reasoning"
        used_reasoning_as_response = True

    return text, {
        "finish_reason": choice.get("finish_reason"),
        "content_type": content_type,
        "has_reasoning": bool(reasoning_text),
        "reasoning_length": len(reasoning_text),
        "used_reasoning_as_response": used_reasoning_as_response,
    }


def run_single(prompt_data: dict, params: dict, mode: str) -> dict:
    """Run one prompt with one param set, grade it, return result dict."""
    params = normalize_reasoning_params(params)
    system_msg = {
        "planner": "You are an expert technical planner and architect. Produce structured, actionable plans. Do not write code unless explicitly asked.",
        "coder": "You are an expert software engineer. Write clean, correct, well-tested code. Follow the requirements exactly.",
    }[mode]

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_data["prompt"]},
    ]

    max_tokens = MAX_TOKENS_PLANNER if mode == "planner" else MAX_TOKENS_CODER

    t0 = time.time()
    try:
        raw = call_lmstudio(messages, params, max_tokens)
        elapsed = time.time() - t0
    except Exception as e:
        return {
            "error": str(e),
            "elapsed": time.time() - t0,
            "params": params,
            "reasoning_profile": params.get("reasoning_profile", "unprofiled"),
            "prompt_id": prompt_data["id"],
            "mode": mode,
        }

    content, response_meta = extract_response_text(
        raw,
        use_reasoning_as_response=params.get(
            "use_reasoning_as_response",
            DEFAULT_USE_REASONING_AS_RESPONSE,
        ),
    )
    usage = raw.get("usage", {})

    # Grade
    if mode == "planner":
        grade = grade_planner(content, prompt_data)
    else:
        grade = grade_coder(content, prompt_data)

    if not content.strip():
        if "empty_response" not in grade.flags:
            grade.flags.append("empty_response")
        if response_meta["content_type"] == "none":
            grade.flags.append("null_message_content")
        if response_meta["has_reasoning"]:
            grade.flags.append("reasoning_without_answer")
        finish_reason = response_meta.get("finish_reason")
        if finish_reason:
            grade.flags.append(f"finish_reason_{finish_reason}")

    return {
        "prompt_id": prompt_data["id"],
        "mode": mode,
        "reasoning_profile": params.get("reasoning_profile", "unprofiled"),
        "params": params,
        "response": content,
        "response_meta": response_meta,
        "grade": {
            "dimensions": grade.dimensions,
            "weighted_score": grade.weighted_score,
            "flags": grade.flags,
            "raw_length": grade.raw_length,
            "code_blocks": grade.code_blocks,
            "has_structure": grade.has_structure,
            "exec_result": grade.exec_result if grade.exec_result else None,
        },
        "usage": usage,
        "elapsed": round(elapsed, 2),
    }


def run_sweep_phase(
    mode: str,
    prompts: list[dict],
    param_combos: list[dict],
    n_samples: int,
    phase_name: str,
    resume_from: str | None = None,
    parallel_requests: int = DEFAULT_PARALLEL_REQUESTS,
    retry_errors: bool = True,
) -> list[dict]:
    """Run a full sweep: prompts × param_combos × n_samples.

    Saves results incrementally to disk for crash recovery.
    """
    parallel_requests = max(1, parallel_requests)
    results_file = RESULTS_DIR / f"{phase_name}_{mode}.jsonl"
    target_keys = expected_run_keys(prompts, param_combos, n_samples)

    # Load existing results for resume
    existing = set()
    existing_error_keys = set()
    ignored_existing = 0
    if resume_from and Path(resume_from).exists():
        results_file = Path(resume_from)
    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = run_key(r["prompt_id"], r["params"], r.get("sample_idx", 0))
                    if key not in target_keys:
                        ignored_existing += 1
                        continue
                    if "error" in r and retry_errors:
                        existing_error_keys.add(key)
                        continue
                    existing.add(key)
                except:
                    pass
        existing_errors = len(existing_error_keys - existing)
        print(f"  Resuming: {len(existing)} matching result(s) already collected")
        if existing_errors:
            print(f"  Retrying {existing_errors} previous error result(s)")
        if ignored_existing:
            print(f"  Ignoring {ignored_existing} result(s) from other prompt/param/sample sets")

    total = len(target_keys)
    done = len(existing)
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name} | Mode: {mode}")
    print(f"  Prompts: {len(prompts)} | Param combos: {len(param_combos)} | Samples/combo: {n_samples}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"  Total calls: {total} | Already done: {done} | Remaining: {total - done}")
    print(f"  Results file: {results_file}")
    print(f"{'='*60}\n")

    all_results = []
    scheduled_tasks = []
    call_count = done

    for prompt_data in prompts:
        for params in param_combos:
            for si in range(n_samples):
                key = (prompt_data["id"], param_hash(params), si)
                if key in existing:
                    continue
                call_count += 1
                scheduled_tasks.append({
                    "call_idx": call_count,
                    "prompt_data": prompt_data,
                    "params": params,
                    "sample_idx": si,
                })

    def finalize_result(task: dict, result: dict):
        prefix = (
            f"[{task['call_idx']}/{total}] {task['prompt_data']['id']} | "
            f"params={param_hash(task['params'])} | sample={task['sample_idx'] + 1}/{n_samples}"
        )
        print(f"  {prefix} ...", end="", flush=True)

        result["sample_idx"] = task["sample_idx"]
        result["phase"] = phase_name

        if "error" in result:
            print(f" ERROR: {result['error'][:60]}")
        else:
            score = result["grade"]["weighted_score"]
            elapsed = result["elapsed"]
            flags = result["grade"].get("flags", [])
            if "empty_response" in flags:
                print(f" score={score:.3f} | empty_response | {elapsed:.1f}s")
            else:
                print(f" score={score:.3f} | {elapsed:.1f}s")

        with open(results_file, "a") as f:
            save_result = result.copy()
            if phase_name.startswith("coarse"):
                save_result["response"] = save_result.get("response", "")[:500]
            f.write(json.dumps(save_result) + "\n")

        all_results.append(result)

    if parallel_requests == 1:
        for task in scheduled_tasks:
            finalize_result(task, run_single(task["prompt_data"], task["params"], mode))
            time.sleep(0.2)
        return all_results

    def submit_task(executor: ThreadPoolExecutor, task: dict):
        future = executor.submit(run_single, task["prompt_data"], task["params"], mode)
        return future

    with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        future_to_task = {}
        task_iter = iter(scheduled_tasks)

        for _ in range(min(parallel_requests, len(scheduled_tasks))):
            task = next(task_iter, None)
            if task is None:
                break
            future_to_task[submit_task(executor, task)] = task

        while future_to_task:
            done_futures, _ = wait(future_to_task, return_when=FIRST_COMPLETED)
            for future in done_futures:
                task = future_to_task.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "error": str(exc),
                        "elapsed": 0.0,
                        "params": task["params"],
                        "prompt_id": task["prompt_data"]["id"],
                        "mode": mode,
                        "reasoning_profile": task["params"].get("reasoning_profile", "unprofiled"),
                    }
                finalize_result(task, result)

                next_task = next(task_iter, None)
                if next_task is not None:
                    future_to_task[submit_task(executor, next_task)] = next_task

    return all_results


def analyze_coarse_results(
    mode: str,
    phase_name: str,
    expected_prompts: list[dict] | None = None,
    expected_param_combos: list[dict] | None = None,
    n_samples: int | None = None,
    require_complete: bool = False,
) -> list[dict]:
    """Analyze coarse results and return top-N param combos."""
    results_file = RESULTS_DIR / f"{phase_name}_{mode}.jsonl"
    if not results_file.exists():
        print(f"No results file: {results_file}")
        return []

    # Load all results
    results = []
    with open(results_file) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass

    expected_keys = None
    expected_params_by_hash = {}
    expected_runs_by_hash = {}
    if expected_prompts is not None and expected_param_combos is not None and n_samples is not None:
        expected_keys = expected_run_keys(expected_prompts, expected_param_combos, n_samples)
        expected_params_by_hash = {
            param_hash(params): normalize_reasoning_params(params)
            for params in expected_param_combos
        }
        expected_runs_by_hash = defaultdict(int)
        for _prompt_id, combo_hash, _sample_idx in expected_keys:
            expected_runs_by_hash[combo_hash] += 1

    # Group by param combo. Keep the latest successful row for duplicate run keys
    # so retried JSONL rows cannot overweight a prompt/sample.
    valid_by_key = {}
    errors_matching_expected = 0
    error_keys_matching_expected = set()
    unexpected_valid = 0
    by_params = defaultdict(list)
    for r in results:
        if "params" not in r or "prompt_id" not in r:
            continue
        key = run_key(r["prompt_id"], r["params"], r.get("sample_idx", 0))
        if expected_keys is not None and key not in expected_keys:
            if "error" not in r:
                unexpected_valid += 1
            continue
        if "error" in r:
            if expected_keys is None or key in expected_keys:
                errors_matching_expected += 1
                error_keys_matching_expected.add(key)
            continue
        valid_by_key[key] = r

    if expected_keys is not None:
        missing_keys = expected_keys - set(valid_by_key)
        missing_by_hash = defaultdict(int)
        for _prompt_id, combo_hash, _sample_idx in missing_keys:
            missing_by_hash[combo_hash] += 1
        complete_combo_count = sum(
            1 for combo_hash, expected_n in expected_runs_by_hash.items()
            if expected_n - missing_by_hash.get(combo_hash, 0) == expected_n
        )
        incomplete_combo_count = len(expected_runs_by_hash) - complete_combo_count

        print(
            f"  Analysis coverage: analyzed {len(valid_by_key)}/{len(expected_keys)} "
            f"expected successful run(s)"
        )
        print(
            f"  Combo coverage: {complete_combo_count}/{len(expected_runs_by_hash)} "
            f"complete, {incomplete_combo_count} partial/incomplete"
        )
        if missing_keys:
            print(
                f"  Missing or errored expected run(s): {len(missing_keys)} "
                f"across {len(missing_by_hash)} combo(s)"
            )
        if unexpected_valid:
            print(f"  Ignored {unexpected_valid} successful result(s) outside the requested analysis set")
        if errors_matching_expected:
            print(
                f"  Found {errors_matching_expected} error row(s) "
                f"covering {len(error_keys_matching_expected)} expected run key(s)"
            )
        if require_complete:
            print("  Strict mode: ranking complete combos only")
        else:
            print("  Partial mode: ranking all combos with at least one successful run")
    else:
        print(f"  Analysis coverage: analyzed {len(valid_by_key)} successful run(s)")
        if errors_matching_expected:
            print(f"  Found {errors_matching_expected} error row(s)")

    for key, r in valid_by_key.items():
        _prompt_id, combo_hash, _sample_idx = key
        by_params[combo_hash].append(r)

    # Score each param combo: aggregate across all prompts and samples
    combo_scores = []
    for phash, runs in by_params.items():
        expected_n = expected_runs_by_hash.get(phash)
        if expected_keys is not None and require_complete:
            if len(runs) < expected_n:
                continue
        scores = [r["grade"]["weighted_score"] for r in runs]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        min_score = min(scores)
        max_score = max(scores)

        # Combined score: mean - 0.5*std (penalize variance) + 0.1*min (reward floor)
        combined = mean - 0.5 * std + 0.1 * min_score
        combo_scores.append({
            "params": expected_params_by_hash.get(phash, runs[0]["params"]),
            "param_hash": phash,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min_score, 4),
            "max": round(max_score, 4),
            "combined": round(combined, 4),
            "n_runs": len(runs),
            "expected_runs": expected_n,
            "missing_runs": max((expected_n or len(runs)) - len(runs), 0),
            "complete": expected_n is None or len(runs) >= expected_n,
        })

    combo_scores.sort(key=lambda x: x["combined"], reverse=True)

    if not combo_scores:
        print("\n  No matching successful runs to analyze; analysis file was not updated")
        return []

    # Save analysis
    analysis_file = RESULTS_DIR / f"analysis_{phase_name}_{mode}.json"
    with open(analysis_file, "w") as f:
        json.dump(combo_scores, f, indent=2)
    print(f"\n  Analysis saved: {analysis_file}")
    print(f"  Total param combos evaluated: {len(combo_scores)}")

    # Print top 10
    print(f"\n  Top 10 param combos for {mode}:")
    print(f"  {'Rank':<5} {'Combined':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Runs':<11} {'Params'}")
    print(f"  {'-'*94}")
    for i, cs in enumerate(combo_scores[:10]):
        p = cs["params"]
        param_str = format_param_combo(p)
        expected_n = cs.get("expected_runs")
        run_str = f"{cs['n_runs']}/{expected_n}" if expected_n else str(cs["n_runs"])
        print(
            f"  {i+1:<5} {cs['combined']:<10.4f} {cs['mean']:<8.4f} "
            f"{cs['std']:<8.4f} {cs['min']:<8.4f} {run_str:<11} {param_str}"
        )

    return combo_scores[:10]


def generate_fine_grid(top_combo: dict, step_sizes: dict | None = None) -> list[dict]:
    """Generate a fine grid around a top combo with ±deltas."""
    if step_sizes is None:
        step_sizes = {
            "temperature": [-.05, 0, .05, .1],
            "top_p":       [-.05, 0, .05],
            "top_k":       [-5, 0, 5, 10],
            "min_p":       [-.02, 0, .02],
            "repeat_penalty": [-.02, 0, .02],
        }

    base = top_combo["params"]
    combos = []

    import itertools
    keys = list(step_sizes.keys())
    for deltas in itertools.product(*[step_sizes[k] for k in keys]):
        new_params = {k: v for k, v in base.items() if k not in step_sizes}
        for k, d in zip(keys, deltas):
            val = base[k] + d
            # Clamp
            if k == "temperature":
                val = max(0.0, min(2.0, round(val, 3)))
            elif k == "top_p":
                val = max(0.1, min(1.0, round(val, 3)))
            elif k == "top_k":
                val = max(0, int(val))
            elif k == "min_p":
                val = max(0.0, min(0.5, round(val, 3)))
            elif k == "repeat_penalty":
                val = max(1.0, min(1.5, round(val, 3)))
            new_params[k] = val
        if not should_skip(new_params):
            combos.append(new_params)

    # Deduplicate
    seen = set()
    unique = []
    for c in combos:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ── Main entry points ──

def run_coarse(mode: str, reasoning_profiles: list[str] | None = None,
               parallel_requests: int = DEFAULT_PARALLEL_REQUESTS,
               thinking_token_budget: int | None = None):
    """Phase 1: Coarse sweep with 3 representative prompts using strategic combos."""
    if mode == "planner":
        # Pick 1 hard arch + 1 medium debug + 1 feature as representatives
        target_ids = ("plan_arch_01", "plan_debug_02", "plan_feat_01")
        prompts = [p for p in PLANNER_PROMPTS if p["id"] in target_ids]
    else:
        target_ids = ("code_algo_01", "code_fix_01", "code_spec_01")
        prompts = [p for p in CODER_PROMPTS if p["id"] in target_ids]

    combos = expand_param_combos(
        PARAM_COMBOS_STRATEGIC,
        reasoning_profiles,
        thinking_token_budget=thinking_token_budget,
    )
    print(f"Using {len(combos)} strategic param combos for coarse sweep")

    # n_samples=2 for coarse to keep time reasonable (~100s/call on IQ4_SX)
    return run_sweep_phase(
        mode,
        prompts,
        combos,
        n_samples=2,
        phase_name="coarse",
        parallel_requests=parallel_requests,
    )


def run_focused(mode: str, top_combos: list[dict],
                parallel_requests: int = DEFAULT_PARALLEL_REQUESTS):
    """Phase 2: Test top combos on ALL prompts with full sampling."""
    prompts = PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS
    param_list = [tc["params"] for tc in top_combos]
    return run_sweep_phase(
        mode,
        prompts,
        param_list,
        n_samples=SAMPLES_PER_COMBO,
        phase_name="focused",
        parallel_requests=parallel_requests,
    )


def run_fine(mode: str, top_combo: dict,
             parallel_requests: int = DEFAULT_PARALLEL_REQUESTS):
    """Phase 3: Fine grid around the best combo."""
    prompts = PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS
    fine_combos = generate_fine_grid(top_combo)
    print(f"Generated {len(fine_combos)} fine-grid combos around best")
    return run_sweep_phase(
        mode,
        prompts,
        fine_combos,
        n_samples=SAMPLES_PER_COMBO,
        phase_name="fine",
        parallel_requests=parallel_requests,
    )


def run_quickscan(mode: str, reasoning_profiles: list[str] | None = None,
                  parallel_requests: int = DEFAULT_PARALLEL_REQUESTS,
                  thinking_token_budget: int | None = None):
    """Low-cost scan that trades breadth for faster directional feedback.

    Planner stays minimal.
    Coder keeps broader prompt coverage, but on a much smaller combo shortlist.
    """
    if mode == "planner":
        prompts = [p for p in PLANNER_PROMPTS if p["id"] == "plan_arch_01"]
        n_samples = 1
        combos = [
            # Greedy
            {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
            {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.1},
            # Bridge temp
            dict(QWEN3_RECOMMENDED_COMBOS["instruct"]),
            {"temperature": 0.7, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
            {"temperature": 0.7, "top_p": 0.95, "top_k": 10, "min_p": 0.05, "repeat_penalty": 1.0},
            {"temperature": 0.7, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
            # Med-low
            {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
            {"temperature": 0.4, "top_p": 0.80, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
            {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
            {"temperature": 0.4, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
            {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
            {"temperature": 0.4, "top_p": 0.80, "top_k": 10, "min_p": 0.05, "repeat_penalty": 1.05},
            {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
            {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
            {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
            # Medium
            dict(QWEN3_RECOMMENDED_COMBOS["thinking_coding"]),
            {"temperature": 0.6, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
            {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 10,  "min_p": 0.05, "repeat_penalty": 1.05},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
            {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
            # Med-high
            {"temperature": 0.8, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
            {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
            {"temperature": 0.8, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
            {"temperature": 0.8, "top_p": 0.95, "top_k": 10, "min_p": 0.1,  "repeat_penalty": 1.1},
            # High
            dict(QWEN3_RECOMMENDED_COMBOS["default"]),
            {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
            {"temperature": 1.0, "top_p": 0.80, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
            {"temperature": 1.0, "top_p": 0.95, "top_k": 10, "min_p": 0.1,  "repeat_penalty": 1.15},
            {"temperature": 1.0, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
        ]
    else:
        quickscan_prompt_ids = {"code_algo_01", "code_fix_02", "code_spec_02"}
        prompts = [p for p in CODER_PROMPTS if p["id"] in quickscan_prompt_ids]
        n_samples = 2
        combos = [
            {"temperature": 0.0, "top_p": 1.0, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
            {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
            dict(QWEN3_RECOMMENDED_COMBOS["thinking_coding"]),
            {"temperature": 0.8, "top_p": 0.95, "top_k": 10, "min_p": 0.1,  "repeat_penalty": 1.1},
            {"temperature": 1.0, "top_p": 0.95, "top_k": 10, "min_p": 0.1,  "repeat_penalty": 1.15},
        ]
    expanded = expand_param_combos(
        combos,
        reasoning_profiles,
        thinking_token_budget=thinking_token_budget,
    )
    total_calls = len(expanded) * len(prompts) * n_samples
    print(
        f"Quickscan: {len(expanded)} combos × {len(prompts)} prompt(s) × "
        f"{n_samples} sample(s) = {total_calls} calls"
    )
    print(f"  Prompt ids: {', '.join(prompt['id'] for prompt in prompts)}")
    return run_sweep_phase(
        mode,
        prompts,
        expanded,
        n_samples=n_samples,
        phase_name="quickscan",
        parallel_requests=parallel_requests,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Devstral Parameter Tuning Runner")
    parser.add_argument("phase", choices=["quickscan", "coarse", "analyze_coarse", "focused", "fine", "full"],
                        help="Which phase to run")
    parser.add_argument("--mode", choices=["planner", "coder", "both"], default="both",
                        help="Which mode to benchmark")
    parser.add_argument("--top-n", type=int, default=10, help="Top N combos for focused phase")
    parser.add_argument("--reasoning-profiles",
                        help="Comma-separated reasoning profile names from config.REASONING_PROFILES")
    parser.add_argument("--thinking-token-budget", type=int,
                        help="Custom thinking budget. Use with thinking_custom or profiles like thinking_<N>")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL_REQUESTS,
                        help="Number of concurrent requests to keep in flight")
    args = parser.parse_args()

    try:
        requested_reasoning_profiles = (
            parse_reasoning_profiles_arg(args.reasoning_profiles)
            if args.reasoning_profiles
            else None
        )
        reasoning_profiles = resolve_reasoning_profiles(
            requested_reasoning_profiles,
            thinking_token_budget=args.thinking_token_budget,
        )
    except ValueError as exc:
        parser.error(str(exc))

    modes = ["planner", "coder"] if args.mode == "both" else [args.mode]

    if args.phase == "quickscan":
        for m in modes:
            run_quickscan(
                m,
                reasoning_profiles=reasoning_profiles,
                parallel_requests=args.parallel,
                thinking_token_budget=args.thinking_token_budget,
            )
            print("\n>>> Quickscan analysis:")
            analyze_coarse_results(m, "quickscan")

    elif args.phase == "coarse":
        for m in modes:
            run_coarse(
                m,
                reasoning_profiles=reasoning_profiles,
                parallel_requests=args.parallel,
                thinking_token_budget=args.thinking_token_budget,
            )

    elif args.phase == "analyze_coarse":
        for m in modes:
            analyze_coarse_results(m, "coarse")

    elif args.phase == "focused":
        for m in modes:
            top = analyze_coarse_results(m, "coarse")[:args.top_n]
            if top:
                run_focused(m, top, parallel_requests=args.parallel)

    elif args.phase == "fine":
        for m in modes:
            top = analyze_coarse_results(m, "focused")
            if top:
                run_fine(m, top[0], parallel_requests=args.parallel)

    elif args.phase == "full":
        for m in modes:
            print(f"\n{'#'*60}")
            print(f"  FULL PIPELINE: {m.upper()}")
            print(f"{'#'*60}")

            print("\n>>> Phase 1: Coarse sweep")
            run_coarse(
                m,
                reasoning_profiles=reasoning_profiles,
                parallel_requests=args.parallel,
                thinking_token_budget=args.thinking_token_budget,
            )

            print("\n>>> Phase 2: Analyze coarse + Focused sweep")
            top = analyze_coarse_results(m, "coarse")[:args.top_n]
            if top:
                run_focused(m, top, parallel_requests=args.parallel)

                print("\n>>> Phase 3: Analyze focused + Fine sweep")
                top_focused = analyze_coarse_results(m, "focused")
                if top_focused:
                    run_fine(m, top_focused[0], parallel_requests=args.parallel)

                    print("\n>>> Final analysis")
                    analyze_coarse_results(m, "fine")
