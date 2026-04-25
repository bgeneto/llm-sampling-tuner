"""
Full sweep: planner then coder, then analyze both.
Run this and walk away — it handles everything.
"""
import argparse
import sys

sys.path.insert(0, '.')

from config import (DEFAULT_PARALLEL_REQUESTS, DEFAULT_REASONING_PROFILES,
                    expand_param_combos, resolve_reasoning_profiles)
from prompts.coder_prompts import CODER_PROMPTS
from prompts.planner_prompts import PLANNER_PROMPTS
from report import generate_report
from run_coarse import FOCUSED_COMBOS
from runner import (analyze_coarse_results, parse_reasoning_profiles_arg,
                    run_sweep_phase)

N_SAMPLES = 2

def run_mode(mode, reasoning_profiles, parallel_requests):
    prompts = PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS
    combos = expand_param_combos(FOCUSED_COMBOS, reasoning_profiles)
    total = len(combos) * len(prompts) * N_SAMPLES
    est_hours = total * 100 / 3600
    print(f"\n{'#'*60}")
    print(f"  SWEEP: {mode.upper()} ({total} calls, ~{est_hours:.1f}h)")
    print(f"  Reasoning profiles: {', '.join(reasoning_profiles)}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"{'#'*60}")
    run_sweep_phase(
        mode,
        prompts,
        combos,
        N_SAMPLES,
        "coarse_v2",
        parallel_requests=parallel_requests,
    )
    print(f"\n>>> Analysis for {mode}:")
    analyze_coarse_results(mode, "coarse_v2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run planner/coder sweeps end-to-end")
    parser.add_argument("modes", nargs="*", choices=["planner", "coder"], default=["planner", "coder"])
    parser.add_argument("--reasoning-profiles", default=",".join(DEFAULT_REASONING_PROFILES),
                        help="Comma-separated reasoning profile names from config.REASONING_PROFILES")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL_REQUESTS,
                        help="Number of concurrent requests to keep in flight")
    args = parser.parse_args()

    try:
        reasoning_profiles = resolve_reasoning_profiles(
            parse_reasoning_profiles_arg(args.reasoning_profiles)
        )
    except ValueError as exc:
        parser.error(str(exc))

    modes = args.modes
    for m in modes:
        run_mode(m, reasoning_profiles, args.parallel)

    print("\n\n>>> GENERATING FINAL REPORTS <<<")
    for m in modes:
        generate_report(m)
