"""
LLM Sampling Parameter Tuning Configuration
============================================
Change MODEL_ID and API_BASE below to benchmark any model on any
OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, text-generation-webui, etc.)
"""

import itertools

# ═══════════════════════════════════════════════════════════════════════════════
#  CHANGE THESE TWO VALUES to target a different model / API endpoint
# ═══════════════════════════════════════════════════════════════════════════════
API_BASE  = "https://vllm1.webonly.app/v1"                           # OpenAI-compatible endpoint
MODEL_ID  = "qwen-gpu"      # exact model ID served by the endpoint
MAX_CTX   = 61440                                                # context window (tokens)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Repetitions per (prompt, param_combo) to handle stochastic variance ──
SAMPLES_PER_COMBO = 5

# ── Max tokens by mode ──
MAX_TOKENS_PLANNER = 2048
MAX_TOKENS_CODER = 4096

# ── Parameter Grid ──
# We sweep across a focused grid. The grid is intentionally asymmetric:
# - Planner needs coherent, structured reasoning → lower temp, tighter nucleus
# - Coder needs precise syntax + creative problem solving → balanced temp, strict sampling
#
# Each parameter has a "coarse" sweep first, then we zoom into the best region.

PARAM_GRID_COARSE = {
    "temperature":    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "top_p":          [0.7, 0.85, 0.95, 1.0],
    "top_k":          [0, 20, 50],                 # 0 = disabled
    "min_p":          [0.0, 0.05, 0.1],
    "repeat_penalty": [1.0, 1.05, 1.1, 1.15],
}

# ── Strategic subset for coarse phase ──
# Instead of full Cartesian (3000+ combos), we use curated combos
# that cover the parameter space efficiently: ~100 combos
PARAM_COMBOS_STRATEGIC = [
    # Greedy baseline (temp=0 makes top_p/k/min_p irrelevant)
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.05},
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.1},
    {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.15},

    # Low temp (0.2) — conservative but not greedy
    {"temperature": 0.2, "top_p": 0.85, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.2, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.2, "top_p": 0.85, "top_k": 20, "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.2, "top_p": 0.95, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.2, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 0.95, "top_k": 50, "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 0.95, "top_k": 50, "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.2, "top_p": 1.0,  "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.2, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},

    # Medium-low temp (0.4) — the "sweet spot" region to explore densely
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 20, "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 50, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 50, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},

    # Medium temp (0.6) — balanced creativity/coherence
    {"temperature": 0.6, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 50, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},

    # Medium-high temp (0.8) — more creative, higher derail risk
    {"temperature": 0.8, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 50, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},

    # High temp (1.0) — maximum creativity, strong guardrails needed
    {"temperature": 1.0, "top_p": 0.7,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.7,  "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 50, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
    {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
]

# ── Focused grids per mode (used after coarse sweep narrows regions) ──
# These will be populated by analysis phase
PARAM_GRID_PLANNER_FINE = {}
PARAM_GRID_CODER_FINE = {}

# ── Sensible combo pruning: not all combos are worth testing ──
# Skip combos where temp=0 and top_p/top_k/min_p vary (greedy is deterministic)
def should_skip(params):
    if params["temperature"] == 0.0:
        # For greedy, only one set of top_p/top_k/min_p matters
        if params["top_p"] != 1.0 or params["top_k"] != 0 or params["min_p"] != 0.0:
            return True
    return False

def generate_combos(grid):
    """Generate all non-skippable parameter combinations."""
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        combo = dict(zip(keys, vals))
        if not should_skip(combo):
            combos.append(combo)
    return combos

# ── Scoring weights by mode ──
SCORING_WEIGHTS = {
    "planner": {
        "structure":       0.20,  # Has clear sections, numbered steps, hierarchy
        "completeness":    0.20,  # Covers all aspects of the problem
        "actionability":   0.20,  # Steps are concrete and executable
        "coherence":       0.15,  # Logical flow, no contradictions
        "conciseness":     0.10,  # Not bloated or repetitive
        "no_hallucination": 0.15, # Doesn't invent APIs/tools/constraints
    },
    "coder": {
        "correctness":     0.30,  # Code would actually work
        "completeness":    0.15,  # Handles edge cases, full solution
        "code_quality":    0.15,  # Clean, idiomatic, no smells
        "follows_spec":    0.15,  # Does what was asked, not more/less
        "no_hallucination": 0.15, # Doesn't invent APIs/libs/syntax
        "parseable":       0.10,  # Valid syntax, extractable code blocks
    },
}

# ── Stability scoring (across N samples of same prompt+params) ──
STABILITY_WEIGHTS = {
    "consistency":  0.40,  # How similar are N runs to each other
    "no_derail":    0.30,  # None of N runs went off the rails
    "best_quality": 0.30,  # Quality of the best run in the set
}
