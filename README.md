# LLM Sampling Tuner

**Find strong real-world settings for any local LLM — because published benchmarks won't tell you.**

---

## The Problem

You downloaded an open-source model. You quantized it to fit your GPU. Now what?

Every model ships with recommended sampling parameters — `temperature`, `top_p`, `repeat_penalty` — but those numbers were tested on **full-precision weights** running on A100 clusters. The moment you quantize to Q4 or Q5 to run locally, those recommendations no longer apply. The probability distributions shift, token selection becomes noisier, and the model behaves differently than the benchmarks suggest.

On top of that, published benchmarks (MMLU, HumanEval, etc.) are increasingly unreliable. Models are trained on the test sets. Scores go up while real-world performance stays flat. There is no benchmark for *"Can this model plan a system architecture without going off the rails at temperature 0.6?"*

**This tool fills that gap.** It runs your actual model, on your actual hardware, at your actual quantization level, against your actual novel problem that no model has been trained on — and helps you identify the best-performing parameter candidates for your use case.

---

## What It Does

LLM Sampling Tuner is an automated parameter sweep pipeline that:

1. **Sends diverse test prompts** to your model through any OpenAI-compatible API
2. **Grades every response** using heuristic analysis and actual code execution
3. **Sweeps the parameter space** across temperature, top_p, top_k, min_p, and repeat_penalty
4. **Treats reasoning mode as a first-class axis** so you can compare direct-answer vs bounded-thinking profiles
5. **Accounts for stochastic variance** by running multiple samples per configuration
6. **Produces a ranked report** of strong candidate settings, broken down by task type

It benchmarks two distinct modes that matter for agentic coding workflows:

| Mode | What it tests | Example tasks |
|------|--------------|---------------|
| **Planner** | Structured reasoning, architecture, debugging strategy | System design, migration plans, failure mode analysis |
| **Coder** | Code generation correctness and quality | Algorithm implementation, bug fixing, code from spec |

### Grading

Responses are not graded with another LLM (which would add its own biases). Instead, the grader uses deterministic heuristics:

- **Planner mode**: Scores structure (headers, numbered steps, nesting), completeness (topic coverage), actionability (imperative verbs, named technologies, tradeoff analysis), coherence (trigram repetition detection), conciseness (information density), and hallucination resistance
- **Coder mode**: Actually **executes the generated Python code** in an isolated local sandbox, runs reference checks when the prompt defines them, then scores code quality via AST analysis (docstrings, type hints, function decomposition, naming conventions), spec adherence, and hallucination detection

### Why Not Just Use an LLM-as-Judge?

Because the judge would need its own temperature settings tuned first. The grading must be deterministic and independent of the model under test. Heuristic scoring with code execution gives you a strong correctness signal for coder mode and a useful proxy for planner mode — without circular dependencies.

### What This Tool Is And Isn't

This project is best used as a **workload-specific ranking harness**. It is good at finding bad parameter regions, surfacing strong candidates, and showing how different sampling settings behave on your prompts.

It is **not** a proof of a global optimum. The final ranking depends on your prompt set, your scoring weights, your serving stack, and the current heuristic grader. Treat the top result as a strong candidate to validate, not as a mathematically settled answer.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A running OpenAI-compatible API endpoint (LM Studio, Ollama, vLLM, llama.cpp server, etc.)
- Python dependencies from `requirements.txt`
- Linux `bubblewrap` (`bwrap`) for sandboxed coder grading. For production runs, do not disable this sandbox.

Create an environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Debian/Ubuntu, install the sandbox runtime with:

```bash
sudo apt-get install bubblewrap
```

Planner-only runs do not execute generated code. Coder runs fail closed if `bwrap`
is unavailable unless you explicitly set `GRADER_ALLOW_UNSANDBOXED=1`, which is
not recommended for production.

### 1. Configure your model

Open `config.py` and set the endpoint values you need:

```python
API_BASE  = "http://localhost:1234/v1"                        # your API endpoint
MODEL_ID  = "mistralai_devstral-small-2-24b-instruct-2512"   # your model ID
API_KEY   = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")  # optional bearer token
MAX_CTX   = 61440                                             # your context window
```

For protected hosted endpoints, export a token before running:

```bash
export LLM_API_KEY="your-provider-token"
```

If you already use `OPENAI_API_KEY`, that works too.

If you benchmark a reasoning model such as Qwen3 on vLLM, use
`--reasoning-profiles` to compare non-thinking and thinking modes explicitly.
Built-in profiles include `thinking_512` and `thinking_1024`; arbitrary budgets
are supported with either `thinking_<N>` or `thinking_custom` plus
`--thinking-token-budget N`. For Qwen3 on vLLM, unbounded thinking can spend all
completion tokens in `message.reasoning` and leave `message.content` empty.

That's it. Everything else is automatic.

### 2. Run a quickscan (~50 minutes)

Get directional results fast with a small parameter set:

```bash
python runner.py quickscan --mode planner
python runner.py quickscan --mode coder
```

To compare direct-answer mode with bounded thinking while keeping 3 requests in
flight:

```bash
python runner.py quickscan --mode planner --reasoning-profiles non_thinking,thinking_512 --parallel 3
```

To test a custom 2048-token thinking budget:

```bash
python runner.py quickscan --mode planner --reasoning-profiles thinking_2048
python runner.py quickscan --mode planner --reasoning-profiles thinking_custom --thinking-token-budget 2048
```

If you pass `--thinking-token-budget` without `--reasoning-profiles`, the CLI
uses the custom thinking profile automatically.

Reasoning tokens are part of the same generated-token allowance as the visible
answer. When a thinking budget is active, the runner treats
`MAX_TOKENS_PLANNER`/`MAX_TOKENS_CODER` as the visible-answer budget and sends
`max_tokens = thinking_token_budget + visible_answer_budget`, so thinking cannot
consume the entire response.

### 3. Run a full coarse sweep (~12–14 hours per mode)

Test 25 strategically chosen parameter combos across all prompts:

```bash
python run_coarse.py planner
python run_coarse.py coder
```

Or run both sequentially:

```bash
python run_full_sweep.py
```

You can also raise request concurrency when your backend supports it:

```bash
python run_coarse.py planner --reasoning-profiles non_thinking,thinking_512 --parallel 3
```

### 4. Analyze results

```bash
python run_coarse.py planner --analyze
python report.py planner coder
```

All results save incrementally to `results/` as JSONL. Sweeps resume automatically if interrupted, retry previous error rows by default, and analysis excludes incomplete prompt/parameter/sample sets when the expected sweep shape is known.

### 5. Validate with a holdout set

Before locking a default configuration, validate the finalists on prompts that were **not** used during tuning.

The current CLI supports this directly:

1. List the available prompt ids for each mode:

```bash
python run_coarse.py planner --list-prompts
python run_coarse.py coder --list-prompts
```

2. Pick a holdout split up front. Example planner holdout: `plan_arch_02,plan_edge_02`

3. Tune on the non-holdout prompts only:

```bash
python run_coarse.py planner --exclude-prompt-ids plan_arch_02,plan_edge_02
```

4. Validate the finalists on the holdout prompts without editing prompt files or `FOCUSED_COMBOS`:

```bash
python run_coarse.py planner --analysis-phase coarse_v2 --top-n 5 --prompt-ids plan_arch_02,plan_edge_02
```

5. If you want exact finalists instead of the top N rows, use the hashes from the analysis output:

```bash
python run_coarse.py planner \
    --analysis-phase coarse_v2 \
    --param-hashes 9c0a1abc,31fe88d2,77b0ef14 \
    --prompt-ids plan_arch_02,plan_edge_02
```

6. Repeat the same pattern for coder mode with its own holdout prompt ids.

7. Pick the default from holdout performance, not from the tuning split alone. If two settings are close, prefer the more stable one across repeated runs.

Custom holdout runs automatically use a separate phase name, so they do not mix results into the default `coarse_v2_*` files unless you explicitly override `--phase-name`.

---

## What You Get

The pipeline produces a ranked list of parameter combinations scored by `mean - 0.5 * std + 0.1 * min` — rewarding both quality and consistency. Treat that ranking as a candidate shortlist rather than as a formal proof of optimality. A sample output:

```
 Rank  Combined   Mean±Std        Parameters
 1     1.0682     0.974±0.006     T=0.6  top_p=0.95  min_p=0.05  rep=1.05
 2     1.0525     0.967±0.018     T=0.0  top_p=1.00  min_p=0.00  rep=1.00
 3     1.0511     0.963±0.015     T=0.4  top_p=0.85  min_p=0.05  rep=1.05
```

The report includes:
- **Best-performing candidate settings** per mode with explanation of why each value was chosen
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
run_coarse.py              # Focused 25-combo sweep runner
run_full_sweep.py          # Combined planner + coder runner
report.py                  # Analysis and report generator
prompts/
  planner_prompts.py       # 9 test prompts (architecture, debugging, features, refactoring, edge cases)
  coder_prompts.py         # 10 test prompts (algorithms, systems, bug fixes, from-spec, refactoring)
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

For coder prompts with deterministic expected behavior, you can also add:

```python
{
    "has_verifiable_output": True,
    "reference_tests": """
assert my_function(1) == 2
assert my_function(5) == 6
""",
}
```

These tests are not shown in the prompt. The grader compiles them inside the sandbox after executing the generated code and does not store the reference-test source in model-visible globals, which makes coder-mode scoring materially more trustworthy for algorithmic and bug-fix tasks.

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
| **OpenRouter / Together / Fireworks** | Compatible (set API_BASE + `LLM_API_KEY` or `OPENAI_API_KEY`) |

---

## License

MIT
