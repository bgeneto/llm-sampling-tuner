"""
Parameter Sweep Runner
======================
Runs Devstral through the parameter grid and collects graded results.

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
from datetime import datetime
from pathlib import Path

import requests

from config import (API_BASE, MAX_TOKENS_CODER, MAX_TOKENS_PLANNER, MODEL_ID,
                    PARAM_COMBOS_STRATEGIC, PARAM_GRID_COARSE,
                    SAMPLES_PER_COMBO, generate_combos, should_skip)
from grader import GradeResult, grade_coder, grade_planner, grade_stability
from prompts.coder_prompts import CODER_PROMPTS
from prompts.planner_prompts import PLANNER_PROMPTS

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def param_hash(params: dict) -> str:
    """Short hash for a param combo, for filenames."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def call_lmstudio(messages: list[dict], params: dict, max_tokens: int,
                   timeout: int = 180) -> dict:
    """Call LM Studio chat completions API. Returns raw response dict."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        **params,
    }
    # Remove params that are 0/disabled to let LM Studio use defaults
    if payload.get("top_k") == 0:
        del payload["top_k"]

    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_response_text(raw: dict) -> tuple[str, dict]:
    """Normalize assistant message content to text and capture response metadata."""
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")

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

    reasoning = message.get("reasoning")
    return text, {
        "finish_reason": choice.get("finish_reason"),
        "content_type": content_type,
        "has_reasoning": bool(reasoning),
        "reasoning_length": len(reasoning) if isinstance(reasoning, str) else 0,
    }


def run_single(prompt_data: dict, params: dict, mode: str) -> dict:
    """Run one prompt with one param set, grade it, return result dict."""
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
            "prompt_id": prompt_data["id"],
            "mode": mode,
        }

    content, response_meta = extract_response_text(raw)
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
    resume_from: str = None,
) -> list[dict]:
    """Run a full sweep: prompts × param_combos × n_samples.

    Saves results incrementally to disk for crash recovery.
    """
    results_file = RESULTS_DIR / f"{phase_name}_{mode}.jsonl"

    # Load existing results for resume
    existing = set()
    if resume_from and Path(resume_from).exists():
        results_file = Path(resume_from)
    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r["prompt_id"], param_hash(r["params"]), r.get("sample_idx", 0))
                    existing.add(key)
                except:
                    pass
        print(f"  Resuming: {len(existing)} results already collected")

    total = len(prompts) * len(param_combos) * n_samples
    done = len(existing)
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name} | Mode: {mode}")
    print(f"  Prompts: {len(prompts)} | Param combos: {len(param_combos)} | Samples/combo: {n_samples}")
    print(f"  Total calls: {total} | Already done: {done} | Remaining: {total - done}")
    print(f"  Results file: {results_file}")
    print(f"{'='*60}\n")

    all_results = []
    call_count = done

    for pi, prompt_data in enumerate(prompts):
        for ci, params in enumerate(param_combos):
            for si in range(n_samples):
                key = (prompt_data["id"], param_hash(params), si)
                if key in existing:
                    continue

                call_count += 1
                prefix = f"[{call_count}/{total}] {prompt_data['id']} | params={param_hash(params)} | sample={si+1}/{n_samples}"
                print(f"  {prefix} ...", end="", flush=True)

                result = run_single(prompt_data, params, mode)
                result["sample_idx"] = si
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

                # Save incrementally
                with open(results_file, "a") as f:
                    # Don't save full response in coarse phase to save disk
                    save_result = result.copy()
                    if phase_name.startswith("coarse"):
                        save_result["response"] = save_result.get("response", "")[:500]
                    f.write(json.dumps(save_result) + "\n")

                all_results.append(result)

                # Small delay to avoid overwhelming LM Studio
                time.sleep(0.2)

    return all_results


def analyze_coarse_results(mode: str, phase_name: str) -> list[dict]:
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

    # Group by param combo
    from collections import defaultdict
    by_params = defaultdict(list)
    for r in results:
        if "error" not in r:
            key = param_hash(r["params"])
            by_params[key].append(r)

    # Score each param combo: aggregate across all prompts and samples
    combo_scores = []
    for phash, runs in by_params.items():
        scores = [r["grade"]["weighted_score"] for r in runs]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        min_score = min(scores)
        max_score = max(scores)

        # Combined score: mean - 0.5*std (penalize variance) + 0.1*min (reward floor)
        combined = mean - 0.5 * std + 0.1 * min_score
        combo_scores.append({
            "params": runs[0]["params"],
            "param_hash": phash,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min_score, 4),
            "max": round(max_score, 4),
            "combined": round(combined, 4),
            "n_runs": len(runs),
        })

    combo_scores.sort(key=lambda x: x["combined"], reverse=True)

    # Save analysis
    analysis_file = RESULTS_DIR / f"analysis_{phase_name}_{mode}.json"
    with open(analysis_file, "w") as f:
        json.dump(combo_scores, f, indent=2)
    print(f"\n  Analysis saved: {analysis_file}")
    print(f"  Total param combos evaluated: {len(combo_scores)}")

    # Print top 10
    print(f"\n  Top 10 param combos for {mode}:")
    print(f"  {'Rank':<5} {'Combined':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Params'}")
    print(f"  {'-'*80}")
    for i, cs in enumerate(combo_scores[:10]):
        p = cs["params"]
        param_str = f"t={p['temperature']} top_p={p['top_p']} top_k={p['top_k']} min_p={p['min_p']} rep={p['repeat_penalty']}"
        print(f"  {i+1:<5} {cs['combined']:<10.4f} {cs['mean']:<8.4f} {cs['std']:<8.4f} {cs['min']:<8.4f} {param_str}")

    return combo_scores[:10]


def generate_fine_grid(top_combo: dict, step_sizes: dict = None) -> list[dict]:
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
        new_params = {}
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

def run_coarse(mode: str):
    """Phase 1: Coarse sweep with 3 representative prompts using strategic combos."""
    if mode == "planner":
        # Pick 1 hard arch + 1 medium debug + 1 feature as representatives
        target_ids = ("plan_arch_01", "plan_debug_02", "plan_feat_01")
        prompts = [p for p in PLANNER_PROMPTS if p["id"] in target_ids]
    else:
        target_ids = ("code_algo_01", "code_fix_01", "code_spec_01")
        prompts = [p for p in CODER_PROMPTS if p["id"] in target_ids]

    combos = PARAM_COMBOS_STRATEGIC
    print(f"Using {len(combos)} strategic param combos for coarse sweep")

    # n_samples=2 for coarse to keep time reasonable (~100s/call on IQ4_SX)
    return run_sweep_phase(mode, prompts, combos, n_samples=2, phase_name="coarse")


def run_focused(mode: str, top_combos: list[dict]):
    """Phase 2: Test top combos on ALL prompts with full sampling."""
    prompts = PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS
    param_list = [tc["params"] for tc in top_combos]
    return run_sweep_phase(mode, prompts, param_list, n_samples=SAMPLES_PER_COMBO, phase_name="focused")


def run_fine(mode: str, top_combo: dict):
    """Phase 3: Fine grid around the best combo."""
    prompts = PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS
    fine_combos = generate_fine_grid(top_combo)
    print(f"Generated {len(fine_combos)} fine-grid combos around best")
    return run_sweep_phase(mode, prompts, fine_combos, n_samples=SAMPLES_PER_COMBO, phase_name="fine")


def run_quickscan(mode: str):
    """Ultra-fast scan: 30 key combos × 1 prompt × 1 sample.
    ~30 calls × ~100s = ~50 min. Gives directional data fast."""
    # Pick the MOST discriminating prompt per mode
    if mode == "planner":
        prompts = [p for p in PLANNER_PROMPTS if p["id"] == "plan_arch_01"]
    else:
        prompts = [p for p in CODER_PROMPTS if p["id"] == "code_algo_01"]

    # 30 maximally diverse combos spanning the full space
    combos = [
        # Greedy
        {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
        {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.1},
        # Low temp
        {"temperature": 0.2, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
        {"temperature": 0.2, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
        {"temperature": 0.2, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
        {"temperature": 0.2, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
        # Med-low
        {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
        {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
        {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
        {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
        {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
        {"temperature": 0.4, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
        {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
        {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
        {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
        # Medium
        {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
        {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
        {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
        {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
        {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
        {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
        # Med-high
        {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
        {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
        {"temperature": 0.8, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
        {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.1},
        # High
        {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
        {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
        {"temperature": 1.0, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.15},
        {"temperature": 1.0, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
    ]
    print(f"Quickscan: {len(combos)} combos × 1 prompt × 1 sample = {len(combos)} calls")
    return run_sweep_phase(mode, prompts, combos, n_samples=1, phase_name="quickscan")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Devstral Parameter Tuning Runner")
    parser.add_argument("phase", choices=["quickscan", "coarse", "analyze_coarse", "focused", "fine", "full"],
                        help="Which phase to run")
    parser.add_argument("--mode", choices=["planner", "coder", "both"], default="both",
                        help="Which mode to benchmark")
    parser.add_argument("--top-n", type=int, default=10, help="Top N combos for focused phase")
    args = parser.parse_args()

    modes = ["planner", "coder"] if args.mode == "both" else [args.mode]

    if args.phase == "quickscan":
        for m in modes:
            run_quickscan(m)
            print("\n>>> Quickscan analysis:")
            analyze_coarse_results(m, "quickscan")

    elif args.phase == "coarse":
        for m in modes:
            run_coarse(m)

    elif args.phase == "analyze_coarse":
        for m in modes:
            analyze_coarse_results(m, "coarse")

    elif args.phase == "focused":
        for m in modes:
            top = analyze_coarse_results(m, "coarse")[:args.top_n]
            if top:
                run_focused(m, top)

    elif args.phase == "fine":
        for m in modes:
            top = analyze_coarse_results(m, "focused")
            if top:
                run_fine(m, top[0])

    elif args.phase == "full":
        for m in modes:
            print(f"\n{'#'*60}")
            print(f"  FULL PIPELINE: {m.upper()}")
            print(f"{'#'*60}")

            print("\n>>> Phase 1: Coarse sweep")
            run_coarse(m)

            print("\n>>> Phase 2: Analyze coarse + Focused sweep")
            top = analyze_coarse_results(m, "coarse")[:args.top_n]
            if top:
                run_focused(m, top)

                print("\n>>> Phase 3: Analyze focused + Fine sweep")
                top_focused = analyze_coarse_results(m, "focused")
                if top_focused:
                    run_fine(m, top_focused[0])

                    print("\n>>> Final analysis")
                    analyze_coarse_results(m, "fine")
