"""
Automated Grading System for Devstral Responses (v2)
=====================================================
Uses heuristic checks + actual code execution to score responses.
Scores are 0.0–1.0 per dimension, then weighted into a final score.

v2 improvements:
  - Code execution testing (sandboxed subprocess)
  - Higher-resolution planner scoring (more granular actionability, depth detection)
  - Deeper code quality metrics (complexity, naming, structure)
"""

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path

EVAL_NOTE_STOPWORDS = {
    "a", "an", "and", "all", "approach", "be", "brief", "briefly", "bug", "bugs",
    "by", "clear", "comprehensive", "consider", "correct", "cover", "covers", "describe",
    "edge", "exact", "exactly", "for", "full", "good", "handle", "have", "hint", "hints",
    "identify", "include", "incremental", "is", "it", "its", "mention", "must", "need",
    "needs", "no", "of", "on", "or", "proper", "provide", "raise", "separate", "should",
    "strategy", "suggest", "the", "them", "to", "use", "uses", "using", "valid", "working",
}

SANDBOX_HELPER_SOURCE = r"""
import json
import os
import resource
import socket
import sys


def _int_from_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _install_resource_limits():
    timeout = _int_from_env("GRADER_EXEC_TIMEOUT", 15)
    memory_mb = _int_from_env("GRADER_MEMORY_MB", 512)
    file_mb = _int_from_env("GRADER_FILE_MB", 10)
    processes = _int_from_env("GRADER_MAX_PROCESSES", 64)

    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 2))
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_mb * 1024 * 1024, file_mb * 1024 * 1024))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
    except (AttributeError, ValueError, OSError):
        pass


def _install_audit_guards():
    blocked_import_roots = {"ctypes", "cffi", "gc"}
    blocked_events = {
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.spawnve",
        "os.system",
        "pty.spawn",
        "subprocess.Popen",
    }
    network_families = {
        getattr(socket, "AF_INET", object()),
        getattr(socket, "AF_INET6", object()),
        getattr(socket, "AF_PACKET", object()),
    }

    def audit_hook(event, args):
        if event == "import" and args:
            root = str(args[0]).split(".", 1)[0]
            if root in blocked_import_roots:
                raise ImportError(f"blocked sandbox import: {root}")
        if event == "socket.__new__" and args:
            family = args[0]
            if family in network_families:
                raise RuntimeError("network sockets are disabled in grader sandbox")
        if event in blocked_events:
            raise RuntimeError(f"blocked sandbox operation: {event}")

    sys.addaudithook(audit_hook)


def _block_frame_introspection():
    def blocked_getframe(*_args, **_kwargs):
        raise RuntimeError("frame introspection is disabled in grader sandbox")

    sys._getframe = blocked_getframe


def _exec_model(model_source):
    namespace = {"__name__": "__grader__", "__builtins__": __builtins__}
    exec(compile(model_source, "<model>", "exec"), namespace)
    return namespace


def _exec_reference_code(namespace, reference_code):
    exec(reference_code, namespace)


def main():
    _install_resource_limits()
    _install_audit_guards()
    _block_frame_introspection()

    payload = json.loads(sys.stdin.read())
    model_source = payload.pop("code")
    reference_source = payload.pop("reference_tests", "") or ""
    del payload

    namespace = _exec_model(model_source)
    if reference_source:
        reference_code = compile(reference_source, "<reference_tests>", "exec")
        reference_source = None
        _exec_reference_code(namespace, reference_code)


if __name__ == "__main__":
    main()
"""


@dataclass
class GradeResult:
    dimensions: dict = field(default_factory=dict)  # dim_name → 0.0-1.0
    weighted_score: float = 0.0
    flags: list = field(default_factory=list)  # warning strings
    raw_length: int = 0
    code_blocks: int = 0
    has_structure: bool = False
    exec_result: dict = field(default_factory=dict)  # code execution details


def extract_code_blocks(text: str) -> list[str]:
    """Extract all fenced code blocks from a response."""
    code_pattern = r'```(?:python|py)?\s*\n(.*?)```'
    blocks = re.findall(code_pattern, text, re.DOTALL)
    if blocks:
        return blocks

    # Fallback: detect inline code (response is mostly code without fences)
    lines = text.split("\n")
    code_lines = sum(1 for l in lines if re.match(
        r'^(import |from |def |class |    |if |for |while |return |#|@)', l))
    if code_lines > len(lines) * 0.4:
        return [text]
    return []


def _contains_eval_token(text: str, token: str) -> bool:
    """Match alphanumeric tokens by word boundary and symbolic tokens by substring."""
    if re.fullmatch(r'[a-z0-9_]+', token):
        variants = {token}
        if len(token) > 3:
            variants.update({token + "s", token + "es", token + "ed", token + "ing"})
        if token.endswith("y") and len(token) > 3:
            variants.add(token[:-1] + "ies")
        if token.endswith("ing") and len(token) > 4:
            variants.add(token[:-3])
        return any(bool(re.search(rf'\b{re.escape(variant)}\b', text)) for variant in variants)
    return token in text


def score_eval_note_coverage(text: str, eval_notes: str) -> tuple[float, list[str]]:
    """Turn eval_notes into lightweight rubric checks and measure coverage."""
    if not isinstance(eval_notes, str) or not eval_notes.strip():
        return 0.0, []

    normalized = text.lower()
    requirements = []
    seen = set()

    for clause in re.split(r'[.,;:]', eval_notes.lower()):
        clause = clause.strip()
        if not clause:
            continue
        for alternative in re.split(r'\bor\b', clause):
            tokens = [
                token for token in re.findall(r"[a-z][a-z0-9_+-]*", alternative)
                if token not in EVAL_NOTE_STOPWORDS and len(token) > 1
            ]
            if not tokens:
                continue
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            requirements.append(tokens)

    if not requirements:
        return 0.0, []

    matched_requirements = []
    for tokens in requirements:
        matched_tokens = sum(1 for token in tokens if _contains_eval_token(normalized, token))
        threshold = 1 if len(tokens) <= 2 else 2
        if matched_tokens >= min(threshold, len(tokens)):
            matched_requirements.append(" ".join(tokens))

    return len(matched_requirements) / len(requirements), matched_requirements


def build_execution_payload(code: str, reference_tests: str | None = None) -> str:
    """Serialize model code and reference checks for the sandbox helper."""
    return json.dumps({
        "code": code,
        "reference_tests": textwrap.dedent(reference_tests or "").strip(),
    })


def _sandbox_env(timeout: int) -> dict[str, str]:
    """Return a sanitized environment for generated-code execution."""
    return {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "TMPDIR": "/tmp",
        "GRADER_EXEC_TIMEOUT": str(timeout),
        "GRADER_MEMORY_MB": os.getenv("GRADER_MEMORY_MB", "512"),
        "GRADER_FILE_MB": os.getenv("GRADER_FILE_MB", "10"),
        "GRADER_MAX_PROCESSES": os.getenv("GRADER_MAX_PROCESSES", "64"),
    }


def _sandbox_python() -> str:
    """Use the real interpreter path so a venv symlink is not required in bwrap."""
    return str(Path(sys.executable).resolve())


def _bwrap_command(sandbox_dir: str) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bwrap is required for coder execution; set GRADER_ALLOW_UNSANDBOXED=1 to opt out")

    cmd = [
        bwrap,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "--new-session",
    ]
    for path in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(path).exists():
            cmd.extend(["--ro-bind", path, path])
    cmd.extend([
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--ro-bind", sandbox_dir, "/runner",
        "--chdir", "/runner",
        _sandbox_python(),
        "-I",
        "-u",
        "/runner/harness.py",
    ])
    return cmd


def _unsandboxed_command(sandbox_dir: str) -> list[str]:
    if os.getenv("GRADER_ALLOW_UNSANDBOXED") != "1":
        raise RuntimeError("unsafe unsandboxed execution is disabled")
    return [_sandbox_python(), "-I", "-u", str(Path(sandbox_dir) / "harness.py")]


def _execution_command(sandbox_dir: str) -> tuple[str, list[str]]:
    backend = os.getenv("GRADER_SANDBOX", "bwrap").lower()
    if backend == "bwrap":
        return "bwrap", _bwrap_command(sandbox_dir)
    if backend in {"none", "disabled", "unsafe"}:
        return "unsandboxed", _unsandboxed_command(sandbox_dir)
    raise RuntimeError(f"Unknown GRADER_SANDBOX backend: {backend}")


def _run_command(cmd: list[str], payload: str, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def execute_code_safely(code: str, timeout: int = 15, reference_tests: str | None = None) -> dict:
    """Execute Python code in a sandboxed subprocess. Returns execution details."""
    result = {
        "ran": False,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "error_type": None,
        "sandbox_backend": os.getenv("GRADER_SANDBOX", "bwrap"),
        "tests_found": 0,
        "tests_passed": 0,
        "assertions_found": 0,
        "assertions_passed": 0,
        "reference_tests_found": 0,
        "reference_assertions_found": 0,
    }

    # Count assertions in the code before running
    result["assertions_found"] = len(re.findall(r'\bassert\b', code))
    result["tests_found"] = len(re.findall(r'\bdef\s+test_\w+', code))
    normalized_reference_tests = textwrap.dedent(reference_tests or "").strip()
    result["reference_assertions_found"] = len(re.findall(r'\bassert\b', normalized_reference_tests))
    result["reference_tests_found"] = len(re.findall(r'\bdef\s+test_\w+', normalized_reference_tests))

    try:
        payload = build_execution_payload(code, normalized_reference_tests)
        with tempfile.TemporaryDirectory(prefix="llm_grader_") as sandbox_dir:
            harness_path = Path(sandbox_dir) / "harness.py"
            harness_path.write_text(SANDBOX_HELPER_SOURCE, encoding="utf-8")
            backend, cmd = _execution_command(sandbox_dir)
            result["sandbox_backend"] = backend
            proc = _run_command(cmd, payload, _sandbox_env(timeout), timeout)

        result["ran"] = True
        result["exit_code"] = proc.returncode
        result["stdout"] = proc.stdout[:2000]
        result["stderr"] = proc.stderr[:2000]

        if proc.returncode != 0:
            # Classify the error
            stderr = proc.stderr
            if "SyntaxError" in stderr:
                result["error_type"] = "syntax_error"
            elif "NameError" in stderr:
                result["error_type"] = "name_error"
            elif "TypeError" in stderr:
                result["error_type"] = "type_error"
            elif "AssertionError" in stderr:
                result["error_type"] = "assertion_error"
            elif "ImportError" in stderr or "ModuleNotFoundError" in stderr:
                result["error_type"] = "import_error"
            elif "RecursionError" in stderr:
                result["error_type"] = "recursion_error"
            elif "AttributeError" in stderr:
                result["error_type"] = "attribute_error"
            elif "IndexError" in stderr or "KeyError" in stderr:
                result["error_type"] = "index_key_error"
            else:
                result["error_type"] = "runtime_error"

            # Without a dedicated test runner, we cannot infer how many assertions
            # passed before the first failure.
            result["assertions_passed"] = 0
        else:
            result["assertions_passed"] = result["assertions_found"]

    except subprocess.TimeoutExpired:
        result["ran"] = True
        result["exit_code"] = -1
        result["error_type"] = "timeout"
        result["stderr"] = f"Execution timed out after {timeout}s"
    except Exception as e:
        if "bwrap is required" in str(e) or "GRADER_SANDBOX" in str(e) or "unsandboxed" in str(e):
            result["error_type"] = "sandbox_unavailable"
        else:
            result["error_type"] = "execution_failed"
        result["stderr"] = str(e)

    return result


# ── Planner Grading (v2) ──

def grade_planner(response: str, prompt_meta: dict) -> GradeResult:
    """Grade a planner-mode response using high-resolution heuristics."""
    result = GradeResult()
    if not isinstance(response, str):
        result.flags.append("non_text_response")
        return result

    result.raw_length = len(response)
    text = response.strip()
    if not text:
        result.flags.append("empty_response")
        return result

    word_count = len(text.split())

    # ── Structure (0.20 weight) ──
    num_lists = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
    num_headers = len(re.findall(r'^#+\s|^[A-Z][A-Za-z\s]+:\s*$|^\*\*[^*]+\*\*', text, re.MULTILINE))
    num_bullets = len(re.findall(r'^\s*[-*]\s', text, re.MULTILINE))
    has_nested = bool(re.search(r'^\s{2,}\d+[\.\)]|^\s{2,}[-*]\s', text, re.MULTILINE))
    has_sections = num_headers >= 3

    result.has_structure = num_lists >= 3 or num_headers >= 2
    structure = 0.0
    structure += min(num_lists / 8.0, 0.25)      # Up to 0.25 for numbered items
    structure += min(num_headers / 5.0, 0.25)     # Up to 0.25 for headers
    structure += min(num_bullets / 6.0, 0.15)     # Up to 0.15 for bullets
    structure += 0.15 if has_nested else 0.0      # Hierarchy/nesting bonus
    structure += 0.10 if has_sections else 0.0    # Multiple clear sections
    structure += 0.10 if bool(re.search(r'---|\*\*\*|===', text)) else 0.0  # Separators
    result.dimensions["structure"] = min(structure, 1.0)

    # ── Completeness (0.20 weight) ──
    # More granular than v1: check for specific sections based on prompt
    completeness = 0.0

    # Base: word count mapping (more granular curve)
    if word_count < 80:
        completeness = 0.15
    elif word_count < 150:
        completeness = 0.3
    elif word_count < 300:
        completeness = 0.5
    elif word_count < 500:
        completeness = 0.7
    elif word_count < 800:
        completeness = 0.85
    elif word_count < 1200:
        completeness = 0.95
    elif word_count < 1800:
        completeness = 1.0
    else:
        completeness = 0.85  # verbosity penalty

    # Bonus: covers multiple distinct aspects (heuristic: unique topic keywords)
    prompt_lower = prompt_meta.get("prompt", "").lower()
    topic_keywords = set()
    for kw in ['risk', 'tradeoff', 'testing', 'security', 'performance', 'scalab',
               'monitoring', 'deploy', 'rollback', 'migration', 'error', 'edge case',
               'dependency', 'phase', 'timeline', 'cost', 'maintenance']:
        if re.search(rf'\b{kw}', text, re.IGNORECASE):
            topic_keywords.add(kw)
    completeness = min(completeness + len(topic_keywords) * 0.02, 1.0)

    eval_note_coverage, matched_eval_requirements = score_eval_note_coverage(
        text,
        prompt_meta.get("eval_notes", ""),
    )
    completeness = min(completeness + 0.2 * eval_note_coverage, 1.0)
    if prompt_meta.get("eval_notes") and eval_note_coverage < 0.34:
        result.flags.append("low_eval_note_coverage")

    result.dimensions["completeness"] = completeness

    # ── Actionability (0.20 weight) — most differentiating dimension ──
    action_score = 0.0

    # Imperative verbs (strong signal)
    imperative_verbs = re.findall(
        r'\b(implement|create|build|configure|set up|define|write|test|deploy|'
        r'migrate|extract|refactor|install|run|validate|monitor|measure|'
        r'design|establish|integrate|split|separate|introduce|add|remove|'
        r'replace|wrap|inject|mock|stub|benchmark|profile|audit|review)\b',
        text, re.IGNORECASE)
    action_score += min(len(imperative_verbs) / 25.0, 0.35)

    # Named technologies/tools (concrete, not vague)
    tech_mentions = re.findall(
        r'\b(Redis|PostgreSQL|MongoDB|WebSocket|REST|GraphQL|Docker|Kubernetes|'
        r'nginx|React|Vue|Express|FastAPI|Django|Node\.js|TypeScript|Zod|'
        r'Jest|pytest|Mocha|ESLint|Prettier|GitHub|GitLab|CI/CD|AWS|GCP|'
        r'S3|Lambda|CDN|CRDT|OT|JWT|OAuth|RBAC|mutex|semaphore|'
        r'queue|stack|heap|hash map|linked list|tree|graph|DAG|'
        r'worker|thread|process|event loop|promise|async|await)\b',
        text, re.IGNORECASE)
    action_score += min(len(tech_mentions) / 15.0, 0.25)

    # Phase/step references (shows ordering/dependency thinking)
    phase_refs = re.findall(r'\b(step\s*\d|phase\s*\d|stage\s*\d|first|then|next|finally|before|after|depends on|prerequisite|blocked by)\b',
                            text, re.IGNORECASE)
    action_score += min(len(phase_refs) / 10.0, 0.20)

    # Quantified estimates (time, effort, risk levels)
    estimates = re.findall(r'\b(\d+\s*(hours?|days?|weeks?|minutes?|ms|seconds?)|low|medium|high|critical)\s*(risk|effort|priority|impact|complexity)',
                           text, re.IGNORECASE)
    action_score += min(len(estimates) / 5.0, 0.10)

    # Tradeoff analysis (shows depth of thinking)
    tradeoffs = re.findall(r'\b(tradeoff|trade-off|pros?\s*(?:and|\/)\s*cons?|advantage|disadvantage|however|alternatively|on the other hand|downside|upside|versus|vs\.?)\b',
                           text, re.IGNORECASE)
    action_score += min(len(tradeoffs) / 5.0, 0.10)

    result.dimensions["actionability"] = min(action_score, 1.0)

    # ── Coherence (0.15 weight) ──
    sentences = re.split(r'[.!?]\s', text)
    sentences = [s for s in sentences if len(s.strip()) > 20]
    unique_prefixes = set(s.strip().lower()[:60] for s in sentences)
    repetition_ratio = len(unique_prefixes) / max(len(sentences), 1)

    coherence = min(repetition_ratio + 0.2, 1.0)

    # Penalty for off-topic drift
    drift_signals = re.findall(r'\b(by the way|unrelated|off topic|anyway|let me know|hope this helps|feel free)\b', text, re.IGNORECASE)
    coherence -= len(drift_signals) * 0.1

    # Penalty for verbatim repetition of phrases (3+ words repeated)
    words = text.lower().split()
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    trigram_counts = {}
    for t in trigrams:
        trigram_counts[t] = trigram_counts.get(t, 0) + 1
    repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 3)
    coherence -= repeated_trigrams * 0.03

    result.dimensions["coherence"] = max(min(coherence, 1.0), 0.0)

    # ── Conciseness (0.10 weight) ──
    # Reward density: good content per word
    if word_count > 2000:
        conciseness = max(0.2, 1.0 - (word_count - 1500) / 2000)
    elif word_count < 50:
        conciseness = 0.1
    elif word_count < 100:
        conciseness = 0.4
    else:
        # Compute information density: unique meaningful words / total words
        meaningful = set(w.lower() for w in words if len(w) > 3 and not w.startswith('#'))
        density = len(meaningful) / max(word_count, 1)
        conciseness = 0.5 + density * 2.0  # density ~0.15-0.30 → conciseness 0.8-1.1
    result.dimensions["conciseness"] = max(min(conciseness, 1.0), 0.0)

    # ── No Hallucination (0.15 weight) ──
    hall_score = 1.0

    # Invented URLs
    if re.search(r'https?://(?!example\.com|github\.com)[a-z]+\.[a-z]+/\w+', text):
        hall_score -= 0.1

    # Self-referential AI talk
    if re.search(r'\b(as an ai|i am a language model|i cannot|i don\'t have access)\b', text, re.IGNORECASE):
        hall_score -= 0.15
        result.flags.append("self_referential")

    # Excessive code in a planning response
    code_blocks = len(re.findall(r'```', text))
    result.code_blocks = code_blocks // 2
    if result.code_blocks > 3:
        hall_score -= 0.15
        result.flags.append("too_much_code_in_plan")
    elif result.code_blocks > 1:
        hall_score -= 0.05

    # Filler phrases that pad without adding value
    filler = re.findall(r'\b(it is important to note|it should be noted|it is worth mentioning|needless to say|as you know|obviously)\b',
                        text, re.IGNORECASE)
    hall_score -= len(filler) * 0.03

    result.dimensions["no_hallucination"] = max(hall_score, 0.0)

    # ── Weighted Score ──
    from config import SCORING_WEIGHTS
    weights = SCORING_WEIGHTS["planner"]
    result.weighted_score = sum(
        result.dimensions.get(dim, 0) * w for dim, w in weights.items()
    )
    return result


# ── Coder Grading (v2) ──

def grade_coder(response: str, prompt_meta: dict) -> GradeResult:
    """Grade a coder-mode response with code execution testing."""
    result = GradeResult()
    if not isinstance(response, str):
        result.flags.append("non_text_response")
        return result

    result.raw_length = len(response)
    text = response.strip()
    if not text:
        result.flags.append("empty_response")
        return result

    # Extract code blocks
    code_blocks = extract_code_blocks(text)
    result.code_blocks = len(code_blocks)
    all_code = "\n".join(code_blocks) if code_blocks else ""

    # ── Parseable (0.10 weight) ──
    parseable = 0.0
    if all_code:
        try:
            ast.parse(all_code)
            parseable = 1.0
        except SyntaxError:
            # Try individual blocks
            good = sum(1 for b in code_blocks if _try_parse(b))
            parseable = good / max(len(code_blocks), 1) * 0.8
    else:
        result.flags.append("no_code_found")
    result.dimensions["parseable"] = parseable

    # ── Correctness (0.30 weight) — execution-based ──
    correctness = 0.0
    exec_result = None

    requires_verifiable_checks = bool(prompt_meta.get("has_verifiable_output"))

    if all_code and parseable >= 0.5:
        # Try to execute the code
        exec_result = execute_code_safely(
            all_code,
            timeout=15,
            reference_tests=prompt_meta.get("reference_tests"),
        )
        result.exec_result = exec_result
        has_self_checks = bool(exec_result["assertions_found"] or exec_result["tests_found"])
        has_reference_checks = bool(
            exec_result["reference_assertions_found"] or exec_result["reference_tests_found"]
        )
        has_verification_checks = has_self_checks or has_reference_checks

        if exec_result["ran"]:
            if exec_result["exit_code"] == 0:
                # Clean execution!
                correctness = 0.55

                # Hidden reference checks count as strong verification; model-supplied
                # checks still count, but we no longer require them when hidden tests exist.
                if has_reference_checks or has_self_checks:
                    correctness += 0.25
                # Small bonus for no stderr warnings
                if not exec_result["stderr"].strip():
                    correctness += 0.05
                if requires_verifiable_checks and not has_verification_checks:
                    correctness = min(correctness, 0.35)
                    result.flags.append("unverified_clean_run")

            elif exec_result["error_type"] == "import_error":
                # Code structure is right but missing a dependency
                correctness = 0.25
                result.flags.append(f"import_error: {exec_result['stderr'][:80]}")

            elif exec_result["error_type"] == "assertion_error":
                # Logic error but code runs. Do not infer partially passed tests.
                correctness = 0.15
                if has_reference_checks:
                    result.flags.append("reference_test_failed")
                result.flags.append("assertion_failed")

            elif exec_result["error_type"] == "timeout":
                # Infinite loop or very slow
                correctness = 0.15
                result.flags.append("execution_timeout")

            else:
                # Runtime error
                correctness = 0.15
                result.flags.append(f"runtime_error: {exec_result['error_type']}")

        else:
            # Couldn't even run
            correctness = 0.10

    elif all_code:
        # Has code but doesn't parse
        correctness = 0.05

    result.dimensions["correctness"] = min(correctness, 1.0)

    # ── Completeness (0.15 weight) ──
    prompt_lower = prompt_meta.get("prompt", "").lower()
    completeness = 0.0
    checks = 0
    hits = 0

    # Check for class if requested
    if "class " in prompt_lower:
        checks += 1
        if re.search(r'\bclass\s+\w+', all_code):
            hits += 1

    # Check for specific methods/functions mentioned
    method_matches = re.findall(r'`(\w+)\(`', prompt_lower)
    func_sigs = re.findall(r'def (\w+)\(', prompt_lower)
    for name in set(m for m in method_matches + func_sigs if m not in ('def', 'class', 'self')):
        checks += 1
        if re.search(rf'\bdef\s+{re.escape(name)}\b', all_code):
            hits += 1

    # Check for test cases if requested
    if "test" in prompt_lower:
        checks += 1
        test_count = len(re.findall(r'\bassert\b', all_code))
        if test_count > 0:
            hits += 1
            # Bonus for having many tests
            if test_count >= 3:
                hits += 0.5

    # Check for error handling if requested
    if "raise" in prompt_lower or "error" in prompt_lower or "exception" in prompt_lower:
        checks += 1
        if re.search(r'\braise\b|\bexcept\b', all_code):
            hits += 1

    completeness = hits / max(checks, 1)

    # Code volume bonus
    code_lines = len([l for l in all_code.split("\n") if l.strip() and not l.strip().startswith("#")])
    if code_lines > 50:
        completeness = min(completeness + 0.15, 1.0)
    elif code_lines > 25:
        completeness = min(completeness + 0.08, 1.0)

    result.dimensions["completeness"] = min(completeness, 1.0)

    # ── Code Quality (0.15 weight) ──
    quality = 0.0
    if all_code:
        # Parse AST for deeper analysis
        try:
            tree = ast.parse(all_code)
            quality += 0.15  # Parseable baseline

            # Check function/class count (well-structured code has multiple)
            funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            if len(funcs) >= 2 or len(classes) >= 1:
                quality += 0.10

            # Check for docstrings
            has_docstring = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                        has_docstring = True
                        break
            if has_docstring:
                quality += 0.10

            # Type hints
            has_type_hints = any(
                f.returns is not None or any(a.annotation is not None for a in f.args.args)
                for f in funcs
            )
            if has_type_hints:
                quality += 0.10

            # Check function length (short functions = good decomposition)
            func_lengths = []
            for f in funcs:
                func_lengths.append(f.end_lineno - f.lineno if hasattr(f, 'end_lineno') else 10)
            if func_lengths:
                avg_len = sum(func_lengths) / len(func_lengths)
                if avg_len < 20:
                    quality += 0.10
                elif avg_len < 40:
                    quality += 0.05

        except SyntaxError:
            pass

        # Comments
        comment_lines = len(re.findall(r'#\s+\w+', all_code))
        if comment_lines >= 3:
            quality += 0.10
        elif comment_lines >= 1:
            quality += 0.05

        # No wildcard imports
        if not re.search(r'from \w+ import \*', all_code):
            quality += 0.05

        # Uses context managers where appropriate
        if "lock" in prompt_lower or "file" in prompt_lower:
            if re.search(r'\bwith\b', all_code):
                quality += 0.10

        # Good naming: snake_case for functions, PascalCase for classes
        func_names = re.findall(r'\bdef\s+(\w+)', all_code)
        class_names = re.findall(r'\bclass\s+(\w+)', all_code)
        good_func_names = sum(1 for n in func_names if re.match(r'^[a-z_]\w*$', n) or n.startswith('__'))
        good_class_names = sum(1 for n in class_names if re.match(r'^[A-Z]\w*$', n))
        naming_ratio = (good_func_names + good_class_names) / max(len(func_names) + len(class_names), 1)
        quality += naming_ratio * 0.10

    result.dimensions["code_quality"] = min(quality, 1.0)

    # ── Follows Spec (0.15 weight) ──
    follows = 0.0
    if result.code_blocks == 0:
        follows = 0.05
    else:
        follows = 0.4  # Has code

        # Code-to-explanation ratio (want mostly code)
        code_ratio = len(all_code) / max(len(text), 1)
        if code_ratio > 0.6:
            follows += 0.2
        elif code_ratio > 0.3:
            follows += 0.1

        # Check language matches what was asked
        lang = prompt_meta.get("language", "python")
        if lang == "python" and re.search(r'\bdef\s|\bclass\s|\bimport\s', all_code):
            follows += 0.15

        # Didn't add unrequested libraries (kept it simple)
        imports = re.findall(r'(?:from|import)\s+(\w+)', all_code)
        if len(imports) <= 5:
            follows += 0.10
        # Didn't go off on tangents about what it can/can't do
        if not re.search(r'\b(note that|please note|keep in mind|disclaimer)\b', text, re.IGNORECASE):
            follows += 0.10

        eval_note_coverage, matched_eval_requirements = score_eval_note_coverage(
            text,
            prompt_meta.get("eval_notes", ""),
        )
        follows += 0.15 * eval_note_coverage
        if prompt_meta.get("eval_notes") and eval_note_coverage < 0.34:
            result.flags.append("low_eval_note_coverage")

    result.dimensions["follows_spec"] = min(follows, 1.0)

    # ── No Hallucination (0.15 weight) ──
    hall_score = 1.0

    # Check for suspicious imports (only match real import lines, not comments/strings)
    real_libs = {
        'os', 'sys', 'json', 're', 'ast', 'math', 'random', 'time', 'datetime',
        'collections', 'itertools', 'functools', 'typing', 'dataclasses', 'enum',
        'threading', 'multiprocessing', 'queue', 'asyncio', 'socket', 'http',
        'urllib', 'pathlib', 'io', 'abc', 'copy', 'hashlib', 'hmac', 'logging',
        'unittest', 'pytest', 'heapq', 'bisect', 'contextlib', 'concurrent',
        'aiohttp', 'requests', 'flask', 'fastapi', 'pydantic', 'sqlalchemy',
        'numpy', 'pandas', 'redis', 'celery', 'django', 'starlette',
        'textwrap', 'string', 'struct', 'traceback', 'warnings', 'weakref',
    }
    # Only match imports at start of line (not in comments/strings)
    import_lines = re.findall(r'^\s*(?:from|import)\s+(\w+)', all_code, re.MULTILINE)
    for imp in import_lines:
        if imp.lower() not in real_libs and not imp.startswith('_'):
            hall_score -= 0.08
            result.flags.append(f"suspicious_import:{imp}")

    # Self-referential AI talk
    if re.search(r'\b(as an ai|i am a language model|i cannot)\b', text, re.IGNORECASE):
        hall_score -= 0.15
        result.flags.append("self_referential")

    # Hallucinated method names on built-in types
    # (e.g., calling str.to_list() which doesn't exist)
    hall_methods = re.findall(r'\.(\w+)\(', all_code)
    fake_methods = {'to_list', 'to_str', 'to_int', 'to_dict', 'to_set',
                    'contains', 'size', 'length', 'is_empty', 'to_array',
                    'add_all', 'remove_all', 'each', 'map_values'}
    for m in hall_methods:
        if m in fake_methods:
            hall_score -= 0.05
            result.flags.append(f"hallucinated_method:{m}")

    result.dimensions["no_hallucination"] = max(hall_score, 0.0)

    # ── Weighted Score ──
    from config import SCORING_WEIGHTS
    weights = SCORING_WEIGHTS["coder"]
    result.weighted_score = sum(
        result.dimensions.get(dim, 0) * w for dim, w in weights.items()
    )
    return result


def _try_parse(code: str) -> bool:
    """Try to parse a code string as Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ── Stability Grading (across N samples) ──

def grade_stability(grades: list[GradeResult], mode: str) -> dict:
    """Grade the stability of N runs of the same prompt+params."""
    if not grades:
        return {"consistency": 0, "no_derail": 0, "best_quality": 0, "stability_score": 0}

    scores = [g.weighted_score for g in grades]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5

    if mean_score > 0:
        cv = std_dev / mean_score
        consistency = max(0, 1.0 - cv)
    else:
        consistency = 0.0

    derail_threshold = 0.3
    no_derail = sum(1 for s in scores if s >= derail_threshold) / len(scores)
    best_quality = max(scores)

    from config import STABILITY_WEIGHTS
    stability_score = (
        STABILITY_WEIGHTS["consistency"] * consistency +
        STABILITY_WEIGHTS["no_derail"] * no_derail +
        STABILITY_WEIGHTS["best_quality"] * best_quality
    )

    return {
        "consistency": round(consistency, 4),
        "no_derail": round(no_derail, 4),
        "best_quality": round(best_quality, 4),
        "stability_score": round(stability_score, 4),
        "mean": round(mean_score, 4),
        "std_dev": round(std_dev, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "all_scores": [round(s, 4) for s in scores],
    }
