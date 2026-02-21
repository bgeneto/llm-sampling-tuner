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
"""

import sys
sys.path.insert(0, '.')

from runner import run_sweep_phase, analyze_coarse_results
from prompts.planner_prompts import PLANNER_PROMPTS
from prompts.coder_prompts import CODER_PROMPTS

# 15 param combos spanning the full interesting space
# Selected to maximize information gain based on quickscan data
FOCUSED_COMBOS = [
    # Greedy baselines (deterministic reference points)
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.1},

    # Low temp sweet spot (T=0.2)
    {"temperature": 0.2, "top_p": 0.95, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},

    # Med-low (T=0.4) — densely sampled, likely optimal region
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.1},

    # Medium (T=0.6) — balanced
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},

    # Med-high (T=0.8) — pushing creativity
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.95, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},

    # High (T=1.0) — stress test with guardrails
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.1,  "repeat_penalty": 1.15},
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_coarse.py <planner|coder> [--analyze]")
        sys.exit(1)

    mode = sys.argv[1]
    analyze_only = "--analyze" in sys.argv

    if mode not in ("planner", "coder"):
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    prompts = PLANNER_PROMPTS if mode == "planner" else CODER_PROMPTS
    n_samples = 2

    total_calls = len(FOCUSED_COMBOS) * len(prompts) * n_samples
    est_hours = total_calls * 100 / 3600
    print(f"\nFocused Coarse Sweep: {mode.upper()}")
    print(f"  Combos: {len(FOCUSED_COMBOS)}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Samples/combo: {n_samples}")
    print(f"  Total calls: {total_calls}")
    print(f"  Estimated time: ~{est_hours:.1f} hours")

    if not analyze_only:
        run_sweep_phase(mode, prompts, FOCUSED_COMBOS, n_samples, "coarse_v2")

    print("\n>>> Analysis:")
    analyze_coarse_results(mode, "coarse_v2")


if __name__ == "__main__":
    main()
