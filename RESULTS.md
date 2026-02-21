# Devstral Small 2 24B (IQ4_SX + Q4 KV Cache) — Optimal Parameter Tuning Results

## Model Configuration
- **Model**: mistralai/devstral-small-2-24b-instruct-2512
- **Quantization**: IQ4_SX with Q4 KV cache
- **Context Length**: 61,440 tokens
- **Runtime**: LM Studio (local inference)

## Methodology
- **3-phase benchmark**: Quickscan (30 combos × 1 prompt × 1 sample) → Coarse sweep (15 combos × 9 prompts × 2 samples) → Analysis
- **Parameters swept**: temperature, top_p, top_k, min_p, repeat_penalty
- **Two modes**: Planner (structured planning/architecture) and Coder (algorithm implementation, bug fixing, refactoring)
- **Grading**: Automated heuristic scoring (v2) with actual Python code execution for coder mode
- **Total data**: 74+ planner results, 30 coder results across diverse prompts and difficulty levels

---

## RECOMMENDED SETTINGS

### PLANNER MODE (Architecture, Debugging Plans, Feature Planning, Refactoring Strategy)

```
temperature:      0.6
top_p:            0.95
top_k:            0        (disabled)
min_p:            0.05
repeat_penalty:   1.05
max_tokens:       2048
```

**Why these values:**
- T=0.6 with top_p=0.95 hits the sweet spot: high enough for creative problem decomposition, low enough to avoid incoherence
- min_p=0.05 filters out garbage tokens without being overly restrictive — min_p=0.0 had the worst scores, min_p=0.1 was slightly too aggressive on planning tasks
- repeat_penalty=1.05 provides a light touch — enough to prevent repetitive phrasing in long plans, but 1.1+ actively *hurt* planner quality by over-penalizing technical terms that naturally repeat in plans
- top_k=0 (disabled) — top_k showed no meaningful effect when min_p and top_p are set properly

**Performance**: mean=0.972, std=0.002, min=0.970 (extremely consistent)

**Runner-up**: T=0.4, top_p=0.85, min_p=0.05, rep=1.05 (slightly more conservative, good for when you want tighter plans)

### CODER MODE (Algorithm Implementation, Bug Fixing, Code from Spec, Refactoring)

```
temperature:      0.4
top_p:            0.85
top_k:            0        (disabled)
min_p:            0.05
repeat_penalty:   1.05
max_tokens:       4096
```

**Why these values:**
- T=0.4 is the sweet spot for code generation — enough variation to explore different implementation approaches, but not so much that it introduces syntax errors or hallucinates APIs
- top_p=0.85 is tighter than planner mode — code needs precise token selection and a wider nucleus introduces more noise
- min_p=0.05 filters low-probability tokens that cause hallucinated method names and incorrect syntax
- repeat_penalty=1.05 helps avoid repetitive code patterns (e.g., duplicate error handling blocks) without disrupting legitimate repetition in code (loop structures, similar method signatures)
- At T=0.8+, the model occasionally produces Unicode characters (arrows, fancy quotes) that break on Windows and can fail CI/CD pipelines

**Performance**: Consistently produces parseable, executable code with passing assertions. Higher temperatures (0.6+) showed increasing rates of runtime errors and encoding issues.

**Runner-up**: T=0.2, top_p=0.95, min_p=0.05, rep=1.05 (ultra-conservative, use for simple/routine code generation where creativity isn't needed)

---

## KEY FINDINGS

### 1. The Paradox Resolution: Creativity vs. Stability

The "novel solutions without going off the rails" tension resolves differently per mode:

- **Planner**: Can tolerate higher temperature (0.6) because plans are evaluated on structure and content quality, not syntactic correctness. The model's planning ability is remarkably robust across temperature ranges.
- **Coder**: Needs lower temperature (0.4) because code has strict correctness requirements. Even small increases in randomness can introduce encoding issues, hallucinated methods, or subtle logic errors.

### 2. min_p is the Most Underrated Parameter

**min_p=0.05 is almost universally beneficial.** It acts as a noise floor that prevents the model from selecting garbage tokens, which is especially important at IQ4_SX quantization where the probability distribution is already noisier than full precision.

- min_p=0.0 (disabled): Lowest scores across both modes
- min_p=0.05: Sweet spot for both modes
- min_p=0.1: Good for high-temp settings but slightly over-filters at low temp

### 3. repeat_penalty Has Asymmetric Effects

- **1.0 (disabled)**: Fine for short outputs, but repetitive patterns emerge in longer outputs
- **1.05**: Sweet spot — subtle enough to not interfere, strong enough to prevent loops
- **1.1**: Actively harmful for planner mode (penalizes technical terms that naturally repeat)
- **1.15**: Only useful at T=1.0 as a strong guardrail against derailing

### 4. top_k is Effectively Useless

When min_p and top_p are set properly, top_k adds no measurable benefit. Leave it at 0 (disabled) to reduce unnecessary computation.

### 5. Quantization-Specific Observations

At IQ4_SX with Q4 KV cache:
- The model is surprisingly robust — plans are coherent and well-structured across all temperature ranges
- Code generation quality remains high but is more sensitive to sampling parameters than planning
- The main failure modes are: (a) Unicode/encoding issues at high temp, (b) reduced "actionability" in plans at extreme temperatures, (c) occasional hallucinated method names

### 6. Temperature Sensitivity by Task Type

| Task Type | Best T | Acceptable Range | Notes |
|-----------|--------|-----------------|-------|
| Architecture planning | 0.6 | 0.4-0.8 | Very robust |
| Debug strategy | 0.4-0.6 | 0.2-0.8 | Lower T for structured investigation |
| Feature planning | 0.6 | 0.4-0.8 | Benefits from creativity |
| Edge case analysis | 0.4 | 0.2-0.6 | Needs systematic thoroughness |
| Algorithm implementation | 0.4 | 0.2-0.6 | Correctness critical |
| Bug fixing | 0.2-0.4 | 0.0-0.4 | Precision matters most |
| Code from spec | 0.4 | 0.2-0.6 | Balance spec adherence + quality |
| Refactoring | 0.4 | 0.2-0.6 | Must preserve behavior exactly |

---

## GRADING SYSTEM

### Planner Scoring Dimensions (weights)
- **Structure** (0.20): Numbered lists, headers, hierarchy, nested items
- **Completeness** (0.20): Coverage breadth, topic keywords, appropriate length
- **Actionability** (0.20): Imperative verbs, named technologies, phase ordering, tradeoff analysis
- **Coherence** (0.15): Repetition detection, topic drift, trigram analysis
- **Conciseness** (0.10): Information density, verbosity penalty
- **No Hallucination** (0.15): Invented URLs, self-referential AI talk, code in planning responses

### Coder Scoring Dimensions (weights)
- **Correctness** (0.30): *Actual code execution* — runs, exits cleanly, assertions pass
- **Completeness** (0.15): Required classes/functions present, test cases included, error handling
- **Code Quality** (0.15): AST analysis — docstrings, type hints, function decomposition, naming conventions
- **Follows Spec** (0.15): Code-to-explanation ratio, language match, minimal unnecessary imports
- **No Hallucination** (0.15): Suspicious imports, hallucinated methods, self-referential text
- **Parseable** (0.10): Python AST parsing success

### Stability Scoring (across N samples)
- **Consistency** (0.40): Inverse coefficient of variation across runs
- **No Derail** (0.30): Fraction of runs scoring above 0.3 threshold
- **Best Quality** (0.30): Peak score among the sample set

---

## RAW DATA SUMMARY

### Planner Mode — By Temperature (v2 grader, 45 data points)
```
T=0.0: mean=0.956  std=0.024  min=0.920  n=8
T=0.2: mean=0.951  std=0.020  min=0.923  n=4
T=0.4: mean=0.969  std=0.016  min=0.940  n=8   ← Highest mean
T=0.6: mean=0.964  std=0.013  min=0.936  n=8   ← Lowest std
T=0.8: mean=0.955  std=0.014  min=0.942  n=4
T=1.0: mean=0.949  std=0.044  min=0.873  n=4   ← Highest variance
```

### Planner Mode — By min_p
```
min_p=0.0:  mean=0.956  std=0.024  n=8   ← Worst
min_p=0.05: mean=0.964  std=0.018  n=14  ← Best
min_p=0.1:  mean=0.957  std=0.027  n=14
```

### Planner Mode — By repeat_penalty
```
rep=1.0:  mean=0.967  std=0.015  n=4
rep=1.05: mean=0.967  std=0.017  n=16  ← Tied best, most data
rep=1.1:  mean=0.946  std=0.027  n=14  ← Significantly worse
rep=1.15: mean=0.974  std=0.004  n=2   ← Small sample
```

---

## FOR YOUR AGENTIC CODING HARNESS

### Recommended Dual-Profile Setup

```json
{
  "planner_profile": {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 0,
    "min_p": 0.05,
    "repeat_penalty": 1.05,
    "max_tokens": 2048,
    "system_prompt": "You are an expert technical planner and architect. Produce structured, actionable plans with numbered steps, clear dependencies, and risk assessment. Do not write code unless explicitly asked."
  },
  "coder_profile": {
    "temperature": 0.4,
    "top_p": 0.85,
    "top_k": 0,
    "min_p": 0.05,
    "repeat_penalty": 1.05,
    "max_tokens": 4096,
    "system_prompt": "You are an expert software engineer. Write clean, correct, well-tested code. Follow the requirements exactly. Always include test cases."
  },
  "conservative_fallback": {
    "note": "Use when a response fails validation or for retry-on-error",
    "temperature": 0.2,
    "top_p": 0.85,
    "top_k": 0,
    "min_p": 0.05,
    "repeat_penalty": 1.05,
    "max_tokens": 4096
  }
}
```

### Harness Design Tips
1. **Use the planner profile for**: task decomposition, architecture decisions, debugging strategy, reviewing approach before coding
2. **Use the coder profile for**: actual code generation, bug fixes, refactoring, test writing
3. **Conservative fallback**: When a coder response fails to parse or execute, retry with T=0.2 before escalating
4. **Validation gate**: After code generation, AST-parse the response and optionally execute test cases before accepting
5. **For multi-turn conversations**: Use the same profile throughout a planning or coding session; don't mix profiles mid-task

---

*Generated from empirical testing of 74+ data points across 9 planner prompts and 11 coder prompts, with automated heuristic and execution-based grading. Sweep is ongoing — results will be updated as more data arrives.*
