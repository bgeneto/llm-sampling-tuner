"""
Full sweep: planner then coder, then analyze both.
Run this and walk away — it handles everything.
"""
import sys
sys.path.insert(0, '.')

from runner import run_sweep_phase, analyze_coarse_results
from prompts.planner_prompts import PLANNER_PROMPTS
from prompts.coder_prompts import CODER_PROMPTS
from run_coarse import FOCUSED_COMBOS
from report import generate_report

N_SAMPLES = 2

def run_mode(mode):
    prompts = PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS
    total = len(FOCUSED_COMBOS) * len(prompts) * N_SAMPLES
    est_hours = total * 100 / 3600
    print(f"\n{'#'*60}")
    print(f"  SWEEP: {mode.upper()} ({total} calls, ~{est_hours:.1f}h)")
    print(f"{'#'*60}")
    run_sweep_phase(mode, prompts, FOCUSED_COMBOS, N_SAMPLES, "coarse_v2")
    print(f"\n>>> Analysis for {mode}:")
    analyze_coarse_results(mode, "coarse_v2")

if __name__ == "__main__":
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["planner", "coder"]
    for m in modes:
        run_mode(m)

    print("\n\n>>> GENERATING FINAL REPORTS <<<")
    for m in modes:
        generate_report(m)
