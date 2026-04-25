"""
LLM Sampling Parameter Tuning Configuration
============================================
Change MODEL_ID and API_BASE below to benchmark any model on any
OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, text-generation-webui, etc.)
"""

import os
import re

# ═══════════════════════════════════════════════════════════════════════════════
#  CHANGE THESE VALUES to target a different model / API endpoint
# ═══════════════════════════════════════════════════════════════════════════════
API_BASE  = "https://llm.sistema.pro.br/v1"                           # OpenAI-compatible endpoint
MODEL_ID  = "FUPiA"      # exact model ID served by the endpoint
API_KEY   = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")  # optional bearer token for protected endpoints
MAX_CTX   = 65536                                                # context window (tokens)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Repetitions per (prompt, param_combo) to handle stochastic variance ──
SAMPLES_PER_COMBO = 5

# ── Max tokens by mode ──
MAX_TOKENS_PLANNER = 2048
MAX_TOKENS_CODER = 4096

# ── Reasoning-mode sweep profiles ──
# Treat reasoning mode as a first-class benchmark axis. Each profile expands the
# sampling combos into a distinct serving configuration.
DEFAULT_USE_REASONING_AS_RESPONSE = False
REASONING_PROFILE_ALIASES = {
    "direct": "non_thinking",
    "thinking": "thinking_custom",
    "custom_thinking": "thinking_custom",
}
REASONING_PROFILES = {
    "non_thinking": {
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_token_budget": None,
        "use_reasoning_as_response": False,
        "description": "Disable thinking and score non-thinking answers only.",
    },
    "server_default": {
        "chat_template_kwargs": None,
        "thinking_token_budget": None,
        "use_reasoning_as_response": False,
        "description": "Send no reasoning override and benchmark the provider default.",
    },
    "thinking_512": {
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking_token_budget": 512,
        "use_reasoning_as_response": False,
        "description": "Enable thinking with a 512-token reasoning budget.",
    },
    "thinking_1024": {
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking_token_budget": 1024,
        "use_reasoning_as_response": False,
        "description": "Enable thinking with a 1024-token reasoning budget.",
    },
    "thinking_custom": {
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking_token_budget": None,
        "requires_thinking_token_budget": True,
        "use_reasoning_as_response": False,
        "description": "Enable thinking with a CLI-provided reasoning budget.",
    },
}
DEFAULT_REASONING_PROFILES = ["non_thinking"]

# ── Sweep execution controls ──
DEFAULT_PARALLEL_REQUESTS = 2

# ── Qwen3 reference presets ──
QWEN3_RECOMMENDED_COMBOS = {
    "default": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    },
    "thinking_coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    },
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

    # Bridge temp (0.7) — explicitly cover the gap between 0.6 and 0.8
    {"temperature": 0.7, "top_p": 0.80, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.7, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.7, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 60, "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 60, "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.7, "top_p": 1.0,  "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.7, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},

    # Medium-low temp (0.4) — the "sweet spot" region to explore densely
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.80, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.4, "top_p": 0.80, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.80, "top_k": 20, "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.80, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.85, "top_k": 60, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 0.95, "top_k": 60, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.4, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},

    # Medium temp (0.6) — balanced creativity/coherence
    {"temperature": 0.6, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.80, "top_k": 0,  "min_p": 0.0,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.6, "top_p": 0.80, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.80, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.85, "top_k": 60, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 0.95, "top_k": 60, "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.0},
    {"temperature": 0.6, "top_p": 1.0,  "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},

    # Medium-high temp (0.8) — more creative, higher derail risk
    {"temperature": 0.8, "top_p": 0.7,  "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.7,  "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.0},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.80, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05},
    {"temperature": 0.8, "top_p": 0.85, "top_k": 20, "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_p": 0.80, "top_k": 60, "min_p": 0.1,  "repeat_penalty": 1.05},
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
    {"temperature": 1.0, "top_p": 0.80, "top_k": 0,  "min_p": 0.05, "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.05},
    {"temperature": 1.0, "top_p": 0.80, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.1},
    {"temperature": 1.0, "top_p": 0.85, "top_k": 0,  "min_p": 0.1,  "repeat_penalty": 1.15},
    {"temperature": 1.0, "top_p": 0.80, "top_k": 60, "min_p": 0.1,  "repeat_penalty": 1.1},
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


def parse_thinking_token_budget(value):
    """Validate and normalize a user-provided thinking token budget."""
    if value is None:
        return None
    try:
        budget = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Thinking token budget must be an integer") from exc
    if budget < 1:
        raise ValueError("Thinking token budget must be at least 1")
    return budget


def parse_dynamic_thinking_profile(profile_name):
    """Return the budget from profile names like thinking_2048 or think_2048."""
    if not isinstance(profile_name, str):
        return None
    match = re.fullmatch(r"(?:thinking|think)_(\d+)", profile_name)
    if not match:
        return None
    return parse_thinking_token_budget(match.group(1))


def normalize_reasoning_profile(profile_name):
    """Map legacy reasoning profile names to their canonical form."""
    if profile_name is None:
        return None
    aliased = REASONING_PROFILE_ALIASES.get(profile_name, profile_name)
    dynamic_budget = parse_dynamic_thinking_profile(aliased)
    if dynamic_budget is not None:
        return f"thinking_{dynamic_budget}"
    return aliased


def normalize_reasoning_params(params):
    """Return a shallow copy of params with canonical reasoning profile names."""
    normalized = dict(params)
    if "reasoning_profile" in normalized:
        normalized["reasoning_profile"] = normalize_reasoning_profile(normalized["reasoning_profile"])
    return normalized


def resolve_reasoning_profiles(reasoning_profiles=None, thinking_token_budget=None):
    """Validate requested reasoning profiles and return them in order."""
    budget = parse_thinking_token_budget(thinking_token_budget)
    raw_profile_names = reasoning_profiles or (
        ["thinking_custom"] if budget is not None else DEFAULT_REASONING_PROFILES
    )
    profile_names = []
    seen = set()
    for name in raw_profile_names:
        normalized = normalize_reasoning_profile(name)
        dynamic_budget = parse_dynamic_thinking_profile(normalized)
        if budget is not None and dynamic_budget is not None and dynamic_budget != budget:
            raise ValueError(
                f"Profile {normalized} conflicts with --thinking-token-budget={budget}"
            )
        if normalized not in seen:
            profile_names.append(normalized)
            seen.add(normalized)

    missing = [
        name for name in profile_names
        if name not in REASONING_PROFILES and parse_dynamic_thinking_profile(name) is None
    ]
    if missing:
        known = ", ".join(sorted(REASONING_PROFILES)) + ", thinking_<N>"
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown reasoning profile(s): {missing_str}. Known profiles: {known}")

    requires_budget = [
        name for name in profile_names
        if REASONING_PROFILES.get(name, {}).get("requires_thinking_token_budget")
    ]
    if requires_budget and budget is None:
        raise ValueError(
            f"{', '.join(requires_budget)} requires --thinking-token-budget"
        )

    if budget is not None:
        has_budgeted_profile = any(
            name in requires_budget
            or parse_dynamic_thinking_profile(name) is not None
            or bool((REASONING_PROFILES.get(name, {}).get("chat_template_kwargs") or {}).get("enable_thinking"))
            for name in profile_names
        )
        if not has_budgeted_profile:
            raise ValueError("--thinking-token-budget requires a thinking profile")
    return list(profile_names)


def resolve_reasoning_profile_config(profile_name, thinking_token_budget=None):
    """Return a concrete profile config and display name for a profile request."""
    profile_name = normalize_reasoning_profile(profile_name)
    cli_budget = parse_thinking_token_budget(thinking_token_budget)
    dynamic_budget = parse_dynamic_thinking_profile(profile_name)

    if dynamic_budget is not None:
        if cli_budget is not None and cli_budget != dynamic_budget:
            raise ValueError(
                f"Profile {profile_name} conflicts with --thinking-token-budget={cli_budget}"
            )
        budget = dynamic_budget
        return f"thinking_{budget}", {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": budget,
            "use_reasoning_as_response": False,
        }

    profile = dict(REASONING_PROFILES[profile_name])
    if profile.get("requires_thinking_token_budget"):
        budget = parse_thinking_token_budget(cli_budget)
        profile["thinking_token_budget"] = budget
        return f"thinking_{budget}", profile

    if cli_budget is not None and (profile.get("chat_template_kwargs") or {}).get("enable_thinking"):
        profile["thinking_token_budget"] = cli_budget
        return f"thinking_{cli_budget}", profile

    return profile_name, profile


def expand_param_combos(param_combos, reasoning_profiles=None, thinking_token_budget=None):
    """Cross product sampling combos with the selected reasoning profiles."""
    expanded = []
    for combo in param_combos:
        for profile_name in resolve_reasoning_profiles(reasoning_profiles, thinking_token_budget):
            profile_name, profile = resolve_reasoning_profile_config(
                profile_name,
                thinking_token_budget,
            )
            expanded_combo = dict(combo)
            expanded_combo["reasoning_profile"] = profile_name
            expanded_combo["chat_template_kwargs"] = profile.get("chat_template_kwargs")
            expanded_combo["thinking_token_budget"] = profile.get("thinking_token_budget")
            expanded_combo["use_reasoning_as_response"] = profile.get(
                "use_reasoning_as_response",
                DEFAULT_USE_REASONING_AS_RESPONSE,
            )
            expanded.append(expanded_combo)
    return expanded

# ── Scoring weights by mode ──
SCORING_WEIGHTS = {
    "planner": {
        "structure":       0.10,  # Formatting matters, but less than substance
        "completeness":    0.24,  # Covers all aspects of the problem
        "actionability":   0.24,  # Steps are concrete and executable
        "coherence":       0.15,  # Logical flow, no contradictions
        "conciseness":     0.10,  # Not bloated or repetitive
        "no_hallucination": 0.17, # Doesn't invent APIs/tools/constraints
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
