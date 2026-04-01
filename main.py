import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from sandbox_eval.evaluator import evaluate_sources

# Import the local MAIN_KERNEL_AGENT module from ./agents without colliding with the SDK package name.
AGENTS_DIR = Path(__file__).resolve().parent / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from MAIN_KERNEL_AGENT import (
    GuardrailRetriesExhausted,
    KernelRevision,
    build_agent,
    get_model_name,
    run_kernel_turn,
)
from SUMMARY_AGENT import KernelSummary, summarize_kernel_revision, summary_to_json


# Return the repository root directory for path building.
def project_root() -> Path:
    return Path(__file__).resolve().parent


# Load environment variables from the repo-local .env file.
def init_env() -> None:
    load_dotenv(project_root() / ".env")


def task_path(*parts: str) -> Path:
    return project_root().joinpath("task", *parts)


# Parse CPP_SOURCE from a task/*.py file that defines it.
def load_cpp_from_task_py(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'CPP_SOURCE\s*=\s*r?"""(.*?)"""', text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not parse CPP_SOURCE from {path}")
    return match.group(1).strip()


# Parse and return the baseline CPP_SOURCE string from task/base_kernel.py.
def load_baseline_cpp() -> str:
    return load_cpp_from_task_py(task_path("base_kernel.py"))


# File on disk that supplies the working C++ (best_kernel.py if present, else base_kernel.py).
def working_kernel_path() -> Path:
    best = task_path("best_kernel.py")
    if best.exists():
        return best
    return task_path("base_kernel.py")


def working_kernel_task_relpath() -> str:
    """Repo-relative path, e.g. `task/best_kernel.py` — same file `load_working_cpp()` reads."""
    return working_kernel_path().relative_to(project_root()).as_posix()


# Working kernel for the next agent turn: best promoted kernel if present, else baseline.
def load_working_cpp() -> str:
    return load_cpp_from_task_py(working_kernel_path())


def write_cpp_to_task_py(path: Path, cpp: str) -> None:
    """Replace CPP_SOURCE = r\"\"\" ... \"\"\" in a task kernel file."""
    text = path.read_text(encoding="utf-8")
    body = cpp.rstrip()
    if '"""' in body:
        raise ValueError('Generated C++ must not contain the sequence """ in the source.')

    def repl(m: re.Match[str]) -> str:
        return m.group(1) + "\n" + body + "\n" + '"""'

    pattern = r'(CPP_SOURCE\s*=\s*r?""")(.*?)(?:""")'
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(f"Could not replace CPP_SOURCE in {path}")
    path.write_text(new_text, encoding="utf-8")


def evaluate_candidate_kernel_sync() -> dict:
    """Run compile + correctness + benchmark: candidate vs reference (same as run_eval.py)."""
    root = project_root()
    ref = (root / "task" / "reference.py").read_text(encoding="utf-8")
    cand = (root / "task" / "candidate.py").read_text(encoding="utf-8")
    build_root = root / "build" / "eval"
    build_root.mkdir(parents=True, exist_ok=True)
    return evaluate_sources(
        ref_src=ref,
        candidate_src=cand,
        build_root=build_root,
        num_trials=10,
        seed_num=42,
    )


# Mean time (ms) for task/base_kernel.py naive GEMM under evaluate_sources (reference=torch.matmul).
# Profile once if N/flags change: python -c "..."  # see README or prior chat snippet
NAIVE_BASELINE_TIME_MS = 202.124


def enrich_eval_with_naive_baseline(result: dict) -> dict:
    """Add naive baseline + speedup_vs_naive; evaluator `speedup` remains ref_ms/cand_ms (library vs candidate)."""
    out = dict(result)
    out["naive_baseline_ms"] = NAIVE_BASELINE_TIME_MS
    cand = out.get("candidate_time_ms")
    if cand is not None and float(cand) > 0:
        out["speedup_vs_naive"] = NAIVE_BASELINE_TIME_MS / float(cand)
    else:
        out["speedup_vs_naive"] = None
    return out


def should_promote_to_best_kernel(eval_result: dict) -> bool:
    """True if compile + correctness pass and candidate is strictly faster than the profiled naive baseline."""
    if not eval_result.get("compile_pass") or not eval_result.get("correct_pass"):
        return False
    cand = eval_result.get("candidate_time_ms")
    if cand is None:
        return False
    return float(cand) < NAIVE_BASELINE_TIME_MS


def promote_cpp_to_best_kernel(cpp: str) -> None:
    write_cpp_to_task_py(task_path("best_kernel.py"), cpp)


def kernel_eval_skipped() -> bool:
    return os.environ.get("KERNEL_SKIP_EVAL", "").lower() in ("1", "true", "yes")


def should_save_kernel_history(eval_result: dict | None) -> bool:
    """
    Persist to kernel_history only when the evaluator proves compile + correctness.
    Slower-than-library-reference is still saved (evaluator speedup vs torch.matmul may be below 1; history cares about correctness).

    If KERNEL_SKIP_EVAL is set, no eval runs; saving requires KERNEL_SAVE_WITHOUT_EVAL=1 (dev escape hatch).
    """
    if kernel_eval_skipped():
        return os.environ.get("KERNEL_SAVE_WITHOUT_EVAL", "").lower() in ("1", "true", "yes")
    if eval_result is None:
        return False
    return bool(eval_result.get("compile_pass") and eval_result.get("correct_pass"))


def apply_generation_to_task_files(cpp: str) -> tuple[dict | None, bool]:
    """
    Write generated C++ to task/candidate.py, run evaluator vs reference, and if compile+correct
    and candidate is faster than the profiled naive baseline (base_kernel.py), copy C++ to task/best_kernel.py.
    Returns (eval_result_or_none_if_skipped, promoted).
    """
    write_cpp_to_task_py(task_path("candidate.py"), cpp)
    if kernel_eval_skipped():
        return None, False
    result = enrich_eval_with_naive_baseline(evaluate_candidate_kernel_sync())
    promoted = should_promote_to_best_kernel(result)
    if promoted:
        promote_cpp_to_best_kernel(cpp)
    return result, promoted


# Ensure the local storage directory for saved kernel revisions exists.
def storage_dir() -> Path:
    path = project_root() / "kernel_history"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Convert free-form prompt text into a safe filename component.
def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned[:40] or "revision"


def _eval_timing_lines(eval_result: dict | None, is_best: bool) -> str:
    """YAML-ish metadata lines for evaluator timing (ms) and best-kernel flag."""
    if eval_result is None:
        return (
            "- candidate_time_ms: null\n"
            "- reference_time_ms: null\n"
            "- speedup: null\n"
            "- naive_baseline_ms: null\n"
            "- speedup_vs_naive: null\n"
            f"- is_best: {json.dumps(is_best)}\n"
        )
    return (
        f"- candidate_time_ms: {json.dumps(eval_result.get('candidate_time_ms'))}\n"
        f"- reference_time_ms: {json.dumps(eval_result.get('reference_time_ms'))}\n"
        f"- speedup: {json.dumps(eval_result.get('speedup'))}\n"
        f"- naive_baseline_ms: {json.dumps(eval_result.get('naive_baseline_ms'))}\n"
        f"- speedup_vs_naive: {json.dumps(eval_result.get('speedup_vs_naive'))}\n"
        f"- is_best: {json.dumps(is_best)}\n"
    )


# Save a revision to disk as a readable markdown file (metadata + explanation + C++).
def save_kernel_revision(
    revision: KernelRevision,
    user_request: str,
    *,
    eval_result: dict | None = None,
    is_best: bool = False,
) -> Path:
    """Persist one revision; includes evaluator timing when available."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _sanitize_filename(user_request)
    filename = f"{ts}-{slug}.md"
    path = storage_dir() / filename
    run_id = _extract_run_id_from_filename(path)

    timing = _eval_timing_lines(eval_result, is_best)
    contents = (
        f"# Kernel Revision\n\n"
        f"- run_id: {run_id}\n"
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- user_request: {json.dumps(user_request)}\n"
        f"{timing}\n"
        "## Explanation\n\n"
        f"{revision.explanation.strip()}\n\n"
        "## C++ Code\n\n"
        "```cpp\n"
        f"{revision.cpp_code.rstrip()}\n"
        "```\n"
    )
    path.write_text(contents, encoding="utf-8")
    return path


def _extract_run_id_from_filename(path: Path) -> str:
    return path.stem


def _extract_summary_json_block(text: str) -> dict | None:
    match = re.search(r"## Kernel Summary\s*```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _extract_user_explanation_cpp(text: str) -> tuple[str, str]:
    explanation_match = re.search(
        r"## Explanation\s*(.*?)\s*## C\+\+ Code",
        text,
        re.DOTALL,
    )
    cpp_match = re.search(r"## C\+\+ Code\s*```cpp\s*(.*?)\s*```", text, re.DOTALL)
    if not explanation_match or not cpp_match:
        raise ValueError("Could not parse explanation/cpp from kernel history file.")
    return explanation_match.group(1).strip(), cpp_match.group(1).strip()


def _summary_markdown(summary: KernelSummary) -> str:
    return f"\n## Kernel Summary\n\n```json\n{summary_to_json(summary)}\n```\n"


def _render_summary_for_prompt(summary: KernelSummary) -> str:
    tags = ", ".join(summary.optimization_tags) if summary.optimization_tags else "(none)"
    notes = "; ".join(summary.notes) if summary.notes else "(none)"
    return (
        f"- run_id: {summary.run_id}\n"
        f"  - optimization_tags: {tags}\n"
        f"  - high_level_summary: {summary.high_level_summary}\n"
        f"  - notes: {notes}"
    )


async def _ensure_summary_on_file(path: Path) -> KernelSummary:
    text = path.read_text(encoding="utf-8")
    existing = _extract_summary_json_block(text)
    run_id = _extract_run_id_from_filename(path)

    if existing:
        return KernelSummary.model_validate(existing)

    explanation, cpp_code = _extract_user_explanation_cpp(text)
    summary = await summarize_kernel_revision(
        run_id=run_id,
        cpp_code=cpp_code,
        explanation=explanation,
        model_name=get_model_name(),
    )
    path.write_text(text.rstrip() + _summary_markdown(summary), encoding="utf-8")
    return summary


async def get_top_k_summary_context(k: int = 3) -> list[str]:
    """
    Return newest-first summary lines for prompt context.
    Missing summaries are generated and written into kernel_history files.
    """
    files = sorted(storage_dir().glob("*.md"), reverse=True)
    selected = files[:k]
    lines: list[str] = []
    for path in selected:
        summary = await _ensure_summary_on_file(path)
        lines.append(_render_summary_for_prompt(summary))
    return lines


async def save_kernel_revision_with_summary(
    revision: KernelRevision,
    user_request: str,
    *,
    eval_result: dict | None = None,
    is_best: bool = False,
) -> Path:
    path = save_kernel_revision(
        revision, user_request, eval_result=eval_result, is_best=is_best
    )
    await _ensure_summary_on_file(path)
    return path


def cpp_matches_best_kernel(cpp: str) -> bool:
    """True if this C++ matches the current `task/best_kernel.py` body."""
    best = task_path("best_kernel.py")
    if not best.exists():
        return False
    try:
        return load_cpp_from_task_py(best).strip() == cpp.strip()
    except (OSError, RuntimeError):
        return False


# Return saved revision filenames. sort_by: "recent" (filename ts, newest first) or "fastest" (lowest candidate_time_ms).
def list_saved_kernel_files(sort_by: str = "recent") -> list[str]:
    storage = storage_dir()
    paths = list(storage.glob("*.md"))
    names = [p.name for p in paths]
    if sort_by == "fastest":

        def candidate_ms(name: str) -> float:
            text = (storage / name).read_text(encoding="utf-8")
            m = re.search(r"- candidate_time_ms:\s*([^\n]+)", text)
            if not m:
                return float("inf")
            s = m.group(1).strip()
            if s in ("null", "None", ""):
                return float("inf")
            try:
                return float(s)
            except ValueError:
                return float("inf")

        return sorted(names, key=lambda n: (candidate_ms(n), n))

    return sorted(names, reverse=True)


# Load one saved markdown revision file and parse it into KernelRevision.
def load_saved_kernel_revision(filename: str) -> KernelRevision:
    path = storage_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Saved revision not found: {filename}")
    text = path.read_text(encoding="utf-8")

    explanation_match = re.search(
        r"## Explanation\s*(.*?)\s*## C\+\+ Code",
        text,
        re.DOTALL,
    )
    cpp_match = re.search(r"## C\+\+ Code\s*```cpp\s*(.*?)\s*```", text, re.DOTALL)
    if not explanation_match or not cpp_match:
        raise ValueError(f"Could not parse revision file: {filename}")

    return KernelRevision(
        explanation=explanation_match.group(1).strip(),
        cpp_code=cpp_match.group(1).strip(),
    )


# CLI entrypoint: interactive loop that iteratively improves and saves kernels.
async def main() -> None:
    init_env()

    current_cpp = load_working_cpp()
    agent = build_agent()

    while True:
        try:
            user_line = input("> ").strip()
        except EOFError:
            print()
            break

        if not user_line:
            continue
        if user_line.lower() == "exit":
            break

        prior_summaries = await get_top_k_summary_context(k=3)
        try:
            revision = await run_kernel_turn(
                agent,
                user_line,
                current_cpp,
                prior_summaries=prior_summaries,
            )
        except GuardrailRetriesExhausted as exc:
            print(f"[guardrail] {exc}")
            print(f"[guardrail] last_reason: {exc.last_reason}\n")
            continue

        print("\n--- cpp_code ---\n")
        print(revision.cpp_code)
        print("\n--- explanation ---\n")
        print(revision.explanation)
        print()

        try:
            eval_result, promoted = apply_generation_to_task_files(revision.cpp_code)
        except Exception as exc:
            print(f"[eval] failed: {exc}")
            eval_result, promoted = None, False
        else:
            if eval_result is not None:
                print(json.dumps(eval_result, indent=2))
                if promoted:
                    print("[eval] promoted to task/best_kernel.py (faster than naive baseline)")
                elif eval_result.get("compile_pass") and eval_result.get("correct_pass"):
                    print("[eval] correct but not faster than naive baseline; best_kernel.py unchanged")
                else:
                    print("[eval] compile/correctness did not pass; candidate.py holds last attempt")
            else:
                print("[eval] skipped (KERNEL_SKIP_EVAL)")

        if should_save_kernel_history(eval_result):
            saved_path = await save_kernel_revision_with_summary(
                revision,
                user_line,
                eval_result=eval_result,
                is_best=promoted,
            )
            print(f"[kernel_history] saved {saved_path}")
        else:
            print(
                "[kernel_history] not saved (needs compile+correct eval pass; "
                "or set KERNEL_SAVE_WITHOUT_EVAL=1 when using KERNEL_SKIP_EVAL)"
            )

        current_cpp = revision.cpp_code


if __name__ == "__main__":
    asyncio.run(main())
