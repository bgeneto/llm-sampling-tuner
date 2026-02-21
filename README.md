# LLM Sampling Tuner

**Find the real-world optimal settings for any local LLM — because published benchmarks won't tell you.**

---

## The Problem

You downloaded an open-source model. You quantized it to fit your GPU. Now what?

Every model ships with recommended sampling parameters — `temperature`, `top_p`, `repeat_penalty` — but those numbers were tested on **full-precision weights** running on A100 clusters. The moment you quantize to Q4 or Q5 to run locally, those recommendations no longer apply. The probability distributions shift, token selection becomes noisier, and the model behaves differently than the benchmarks suggest.

On top of that, published benchmarks (MMLU, HumanEval, etc.) are increasingly unreliable. Models are trained on the test sets. Scores go up while real-world performance stays flat. There is no benchmark for *"Can this model plan a system architecture without going off the rails at temperature 0.6?"*

**This tool fills that gap.** It runs your actual model, on your actual hardware, at your actual quantization level, against novel prompts that no model has been trained on — and tells you the exact sampling parameters that produce the best results for your use case.

---

## What It Does

LLM Sampling Tuner is an automated parameter sweep pipeline that:

1. **Sends diverse test prompts** to your model through any OpenAI-compatible API
2. **Grades every response** using heuristic analysis and actual code execution
3. **Sweeps the parameter space** across temperature, top_p, top_k, min_p, and repeat_penalty
4. **Accounts for stochastic variance** by running multiple samples per configuration
5. **Produces a ranked report** of optimal settings, broken down by task type

It benchmarks two distinct modes that matter for agentic coding workflows:

| Mode | What it tests | Example tasks |
|------|--------------|---------------|
| **Planner** | Structured reasoning, architecture, debugging strategy | System design, migration plans, failure mode analysis |
| **Coder** | Code generation correctness and quality | Algorithm implementation, bug fixing, code from spec |

### Grading

Responses are not graded with another LLM (which would add its own biases). Instead, the grader uses deterministic heuristics:

- **Planner mode**: Scores structure (headers, numbered steps, nesting), completeness (topic coverage), actionability (imperative verbs, named technologies, tradeoff analysis), coherence (trigram repetition detection), conciseness (information density), and hallucination resistance
- **Coder mode**: Actually **executes the generated Python code** in a sandboxed subprocess, checks if assertions pass, then scores code quality via AST analysis (docstrings, type hints, function decomposition, naming conventions), spec adherence, and hallucination detection

### Why Not Just Use an LLM-as-Judge?

Because the judge would need its own temperature settings tuned first. The grading must be deterministic and independent of the model under test. Heuristic scoring with code execution gives you ground truth for coder mode and highly correlated signal for planner mode — without circular dependencies.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A running OpenAI-compatible API endpoint (LM Studio, Ollama, vLLM, llama.cpp server, etc.)
- `requests` library (`pip install requests`)

### 1. Configure your model

Open `config.py` and set two values:

```python
API_BASE  = "http://localhost:1234/v1"                        # your API endpoint
MODEL_ID  = "mistralai_devstral-small-2-24b-instruct-2512"   # your model ID
MAX_CTX   = 61440                                             # your context window
```

That's it. Everything else is automatic.

### 2. Run a quickscan (~50 minutes)

Get directional results fast with 30 parameter combinations:

```bash
python runner.py quickscan --mode planner
python runner.py quickscan --mode coder
```

### 3. Run a full coarse sweep (~7–9 hours per mode)

Test 15 strategically chosen parameter combos across all prompts:

```bash
python run_coarse.py planner
python run_coarse.py coder
```

Or run both sequentially:

```bash
python run_full_sweep.py
```

### 4. Analyze results

```bash
python run_coarse.py planner --analyze
python report.py planner coder
```

All results save incrementally to `results/` as JSONL. Sweeps resume automatically if interrupted — no work is lost.

---

## What You Get

The pipeline produces a ranked list of parameter combinations scored by `mean - 0.5 * std + 0.1 * min` — rewarding both quality and consistency. A sample output:

```
 Rank  Combined   Mean±Std        Parameters
 1     1.0682     0.974±0.006     T=0.6  top_p=0.95  min_p=0.05  rep=1.05
 2     1.0525     0.967±0.018     T=0.0  top_p=1.00  min_p=0.00  rep=1.00
 3     1.0511     0.963±0.015     T=0.4  top_p=0.85  min_p=0.05  rep=1.05
```

The report includes:
- **Optimal settings** per mode with explanation of why each value was chosen
- **Parameter sensitivity analysis** — which knobs matter and which are noise
- **Per-prompt breakdown** — how the model performs across different task types
- **Stability analysis** — variance and derail rates across stochastic samples
- **Ready-to-use harness config** — JSON profiles for planner/coder/fallback modes

---

## Example Findings (Devstral Small 24B @ IQ4_SX)

These are real results from running this pipeline on Devstral Small 2 24B at IQ4_SX quantization with Q4 KV cache:

| Finding | Detail |
|---------|--------|
| **min_p=0.05 is universally beneficial** | Acts as a noise floor for quantized weights. min_p=0.0 scored worst across both modes. |
| **repeat_penalty=1.1+ hurts planning** | Over-penalizes technical terms that naturally repeat in structured plans. 1.05 is the sweet spot. |
| **top_k does nothing** | When min_p and top_p are set properly, top_k adds zero measurable benefit. |
| **Planner tolerates higher temperature** | T=0.6 is optimal for plans. T=0.4 is optimal for code. |
| **Coder mode needs tighter nucleus** | top_p=0.85 for code vs 0.95 for planning — code needs precise token selection. |

These findings are specific to this model at this quantization. **That's the entire point** — your model, your hardware, your results.

---

## Project Structure

```
config.py                  # API_BASE, MODEL_ID, parameter grids, scoring weights
grader.py                  # Automated grading with code execution (v2)
runner.py                  # Sweep engine with resume support
run_coarse.py              # Focused 15-combo sweep runner
run_full_sweep.py          # Combined planner + coder runner
report.py                  # Analysis and report generator
prompts/
  planner_prompts.py       # 9 test prompts (architecture, debugging, features, refactoring, edge cases)
  coder_prompts.py         # 11 test prompts (algorithms, systems, bug fixes, from-spec, refactoring)
results/                   # JSONL data files (git-ignored, regenerate locally)
RESULTS.md                 # Detailed findings for the example model
```

---

## Customization

### Adding your own prompts

Add entries to `prompts/planner_prompts.py` or `prompts/coder_prompts.py`:

```python
{
    "id": "my_custom_prompt",
    "category": "your_category",
    "difficulty": "hard",
    "prompt": "Your test prompt here...",
    "eval_notes": "What a good response should contain",
}
```

### Adjusting scoring weights

Edit the `SCORING_WEIGHTS` dict in `config.py` to prioritize the dimensions that matter to you:

```python
SCORING_WEIGHTS = {
    "planner": {
        "structure": 0.20,
        "completeness": 0.20,
        "actionability": 0.20,
        "coherence": 0.15,
        "conciseness": 0.10,
        "no_hallucination": 0.15,
    },
    "coder": {
        "correctness": 0.30,     # Code execution result
        "completeness": 0.15,
        "code_quality": 0.15,    # AST analysis
        "follows_spec": 0.15,
        "no_hallucination": 0.15,
        "parseable": 0.10,
    },
}
```

### Adjusting the parameter grid

Modify `PARAM_COMBOS_STRATEGIC` in `config.py` or `FOCUSED_COMBOS` in `run_coarse.py` to test different regions of the parameter space.

---

## Compatibility

Works with any OpenAI-compatible API that supports these sampling parameters:

| Endpoint | Status |
|----------|--------|
| **LM Studio** | Tested |
| **Ollama** | Compatible (use `http://localhost:11434/v1`) |
| **vLLM** | Compatible |
| **llama.cpp server** | Compatible |
| **text-generation-webui** (with openai ext) | Compatible |
| **OpenRouter / Together / Fireworks** | Compatible (set API_BASE + add API key handling) |

---

## License

MIT
