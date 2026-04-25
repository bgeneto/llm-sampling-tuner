"""
Focused Coarse Sweep
====================
Runs 15 strategically chosen param combos across ALL prompts with 2 samples each.

Planner: 15 combos × 9 prompts × 2 samples = 270 calls (~7.5 hours at 100s/call)
Coder:   15 combos × 11 prompts × 2 samples = 330 calls (~9 hours at 100s/call)

Total: ~600 calls (~16 hours). Run one mode at a time.

Usage:
  python run_coarse.py planner
  python run_coarse.py coder
    python run_coarse.py planner --exclude-prompt-ids plan_arch_02,plan_edge_02
    python run_coarse.py planner --analysis-phase coarse_v2 --top-n 5 --prompt-ids plan_arch_02,plan_edge_02
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

from config import (DEFAULT_PARALLEL_REQUESTS, DEFAULT_REASONING_PROFILES,
                    expand_param_combos, resolve_reasoning_profiles)
from prompts.coder_prompts import CODER_PROMPTS
from prompts.planner_prompts import PLANNER_PROMPTS
from runner import (analyze_coarse_results, format_param_combo, param_hash,
                    parse_reasoning_profiles_arg, run_sweep_phase)

# 15 param combos spanning the full interesting space
# Selected to maximize information gain based on quickscan data
FOCUSED_COMBOS = [
    # Greedy baselines (deterministic reference points)
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.1},

    # Bridge temp check (T=0.7)
    {"temperature": 0.7, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 0,  "min_p": 0.0, "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.00, "repeat_penalty": 1.00},
    {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1},

    # Med-low (T=0.4) — densely sampled, likely optimal region
    {"temperature": 0.4, "top_p": 0.80, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.1},

    # Medium (T=0.6) — balanced
    {"temperature": 0.6, "top_p": 0.80, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.80, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},

    # Med-high (T=0.8) — pushing creativity
    {"temperature": 0.8, "top_p": 0.80, "top_k": 0, "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.95, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},

    # High (T=1.0) — stress test with guardrails
    {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.15},
]


def parse_csv_arg(raw_value: str | None) -> list[str]:
    """Parse a comma-separated CLI argument into a cleaned list."""
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def filter_prompts(prompts: list[dict], include_ids: list[str], exclude_ids: list[str]) -> list[dict]:
    """Filter prompts by id while validating all referenced ids."""
    known_ids = {prompt["id"] for prompt in prompts}
    requested_ids = set(include_ids) | set(exclude_ids)
    missing = sorted(requested_ids - known_ids)
    if missing:
        raise ValueError(f"Unknown prompt id(s): {', '.join(missing)}")

    filtered = [
        prompt for prompt in prompts
        if (not include_ids or prompt["id"] in include_ids)
        and prompt["id"] not in exclude_ids
    ]
    if not filtered:
        raise ValueError("Prompt filters excluded every prompt for this mode.")
    return filtered


def resolve_analysis_path(mode: str, analysis_phase: str | None, analysis_file: str | None) -> Path | None:
    """Resolve the analysis JSON path used to source finalist combos."""
    if analysis_file:
        return Path(analysis_file)
    if analysis_phase:
        return Path("results") / f"analysis_{analysis_phase}_{mode}.json"
    return None


def load_param_combos_from_analysis(
    mode: str,
    analysis_phase: str | None,
    analysis_file: str | None,
    top_n: int,
    param_hashes: list[str],
) -> tuple[list[dict] | None, Path | None]:
    """Load finalist combos from a saved analysis JSON file."""
    analysis_path = resolve_analysis_path(mode, analysis_phase, analysis_file)
    if analysis_path is None:
        return None, None

    if not analysis_path.exists():
        raise ValueError(f"Analysis file not found: {analysis_path}")

    try:
        entries = json.loads(analysis_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in analysis file {analysis_path}: {exc}") from exc

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Analysis file has no combo entries: {analysis_path}")

    if param_hashes:
        by_hash = {
            entry.get("param_hash"): entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("param_hash") and isinstance(entry.get("params"), dict)
        }
        missing_hashes = [combo_hash for combo_hash in param_hashes if combo_hash not in by_hash]
        if missing_hashes:
            raise ValueError(
                f"Unknown param_hash value(s) in {analysis_path}: {', '.join(missing_hashes)}"
            )
        selected_entries = [by_hash[combo_hash] for combo_hash in param_hashes]
    else:
        if top_n < 1:
            raise ValueError("--top-n must be at least 1 when loading finalists from analysis.")
        selected_entries = entries[:top_n]

    combos = []
    for entry in selected_entries:
        params = entry.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"Malformed analysis entry in {analysis_path}: missing params dict")
        combos.append(params)

    if not combos:
        raise ValueError(f"No parameter combos selected from analysis file: {analysis_path}")
    return combos, analysis_path


def build_phase_name(
    explicit_phase_name: str | None,
    mode: str,
    prompts: list[dict],
    analysis_path: Path | None,
    top_n: int,
    param_hashes: list[str],
    reasoning_profiles: list[str],
) -> str:
    """Create a safe phase name for default or holdout/custom runs."""
    if explicit_phase_name:
        return explicit_phase_name

    if analysis_path is None and len(prompts) == len(PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS):
        return "coarse_v2"

    descriptor = {
        "mode": mode,
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "analysis_path": str(analysis_path) if analysis_path else None,
        "top_n": top_n,
        "param_hashes": param_hashes,
        "reasoning_profiles": reasoning_profiles,
    }
    digest = hashlib.md5(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:8]
    prefix = "holdout" if analysis_path else "coarse_custom"
    return f"{prefix}_{digest}"


def print_prompt_catalog(mode: str, prompts: list[dict]):
    """Print prompt ids available for a mode and exit."""
    print(f"\nAvailable prompts for {mode.upper()}:")
    for prompt in prompts:
        print(
            f"  {prompt['id']:<18} category={prompt.get('category', 'n/a'):<12} "
            f"difficulty={prompt.get('difficulty', 'n/a')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Focused coarse sweep runner")
    parser.add_argument("mode", choices=["planner", "coder"])
    parser.add_argument("--analyze", action="store_true", help="Analyze only; do not run new requests")
    parser.add_argument("--list-prompts", action="store_true",
                        help="List prompt ids for the selected mode and exit")
    parser.add_argument("--reasoning-profiles", default=",".join(DEFAULT_REASONING_PROFILES),
                        help="Comma-separated reasoning profile names from config.REASONING_PROFILES")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL_REQUESTS,
                        help="Number of concurrent requests to keep in flight")
    parser.add_argument("--prompt-ids",
                        help="Comma-separated prompt ids to include (useful for holdout evaluation)")
    parser.add_argument("--exclude-prompt-ids",
                        help="Comma-separated prompt ids to exclude (useful for training split runs)")
    analysis_group = parser.add_mutually_exclusive_group()
    analysis_group.add_argument("--analysis-phase",
                                help="Load finalist combos from results/analysis_<phase>_<mode>.json")
    analysis_group.add_argument("--analysis-file",
                                help="Load finalist combos from a specific analysis JSON file")
    parser.add_argument("--top-n", type=int, default=5,
                        help="When loading finalists from analysis, use the top N combos")
    parser.add_argument("--param-hashes",
                        help="Comma-separated param_hash values to select exact finalist combos from analysis")
    parser.add_argument("--phase-name",
                        help="Custom phase/result prefix. Defaults to coarse_v2 or an auto-generated holdout name")
    args = parser.parse_args()

    try:
        reasoning_profiles = resolve_reasoning_profiles(
            parse_reasoning_profiles_arg(args.reasoning_profiles)
        )
    except ValueError as exc:
        parser.error(str(exc))

    base_prompts = PLANNER_PROMPTS if args.mode == "planner" else CODER_PROMPTS
    if args.list_prompts:
        print_prompt_catalog(args.mode, base_prompts)
        return

    include_prompt_ids = parse_csv_arg(args.prompt_ids)
    exclude_prompt_ids = parse_csv_arg(args.exclude_prompt_ids)
    selected_hashes = parse_csv_arg(args.param_hashes)

    if selected_hashes and not (args.analysis_phase or args.analysis_file):
        parser.error("--param-hashes requires --analysis-phase or --analysis-file")

    try:
        prompts = filter_prompts(base_prompts, include_prompt_ids, exclude_prompt_ids)
        loaded_combos, analysis_path = load_param_combos_from_analysis(
            args.mode,
            args.analysis_phase,
            args.analysis_file,
            args.top_n,
            selected_hashes,
        )
    except ValueError as exc:
        parser.error(str(exc))

    n_samples = 2
    if loaded_combos is not None:
        expanded_combos = loaded_combos
        combo_source = f"analysis file {analysis_path}"
    else:
        expanded_combos = expand_param_combos(FOCUSED_COMBOS, reasoning_profiles)
        combo_source = "built-in focused combo set"

    phase_name = build_phase_name(
        args.phase_name,
        args.mode,
        prompts,
        analysis_path,
        args.top_n,
        selected_hashes,
        reasoning_profiles,
    )

    total_calls = len(expanded_combos) * len(prompts) * n_samples
    est_hours = total_calls * 100 / 3600
    print(f"\nFocused Coarse Sweep: {args.mode.upper()}")
    if analysis_path is None:
        print(f"  Reasoning profiles: {', '.join(reasoning_profiles)}")
    else:
        selected_profiles = sorted({combo.get('reasoning_profile', 'unprofiled') for combo in expanded_combos})
        print(f"  Finalists loaded from: {analysis_path}")
        print(f"  Finalist reasoning profiles: {', '.join(selected_profiles)}")
        if selected_hashes:
            print(f"  Selected param_hashes: {', '.join(selected_hashes)}")
        else:
            print(f"  Finalists from analysis: top {min(args.top_n, len(expanded_combos))}")
    print(f"  Phase name: {phase_name}")
    print(f"  Combo source: {combo_source}")
    print(f"  Combos: {len(expanded_combos)}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Prompt ids: {', '.join(prompt['id'] for prompt in prompts)}")
    print(f"  Samples/combo: {n_samples}")
    print(f"  Parallel requests: {args.parallel}")
    print(f"  Total calls: {total_calls}")
    print(f"  Estimated time: ~{est_hours:.1f} hours")

    if expanded_combos:
        print("  Selected combos:")
        for combo in expanded_combos:
            print(f"    {param_hash(combo)}  {format_param_combo(combo)}")

    if not args.analyze:
        run_sweep_phase(
            args.mode,
            prompts,
            expanded_combos,
            n_samples,
            phase_name,
            parallel_requests=args.parallel,
        )

    print("\n>>> Analysis:")
    analyze_coarse_results(args.mode, phase_name)


if __name__ == "__main__":
    main()
