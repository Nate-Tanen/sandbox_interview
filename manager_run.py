"""Manager session: multiple runs × parallel improvement cycles; one kernel_history append for global best."""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_AGENTS = _ROOT / "agents"
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from improvement_cycle import (  # noqa: E402
    ImprovementEvalFailure,
    ImprovementGuardrailFailure,
    ImprovementSuccess,
    run_improvement_cycle,
)
from main import (  # noqa: E402
    apply_generation_to_task_files,
    evaluate_candidate_cpp_isolated,
    enrich_eval_with_naive_baseline,
    save_kernel_revision_with_summary,
    should_promote_to_best_kernel,
    should_save_kernel_history,
)
from MAIN_KERNEL_AGENT import KernelRevision, build_agent  # noqa: E402
from MANAGER_AGENT import generate_manager_prompts  # noqa: E402


LogFn = Callable[[str], None]
ProgressFn = Callable[..., None]


def _dedupe_preserve_order(entries: list[tuple[str, float | None]]) -> list[tuple[str, float | None]]:
    seen: set[str] = set()
    out: list[tuple[str, float | None]] = []
    for cpp, ms in entries:
        key = cpp.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((cpp, ms))
    return out


def build_capped_context(
    seed_entries: list[tuple[str, float | None]],
    run_winners: list[tuple[str, dict]],
    num_cap: int,
) -> list[tuple[str, float | None]]:
    """Seeds + prior run winners, fastest first, length ≤ num_cap."""
    items: list[tuple[str, float | None]] = list(seed_entries)
    for cpp, er in run_winners:
        ms = er.get("candidate_time_ms")
        items.append((cpp, float(ms) if ms is not None else None))
    items = _dedupe_preserve_order(items)

    def sort_key(x: tuple[str, float | None]) -> tuple[bool, float]:
        ms = x[1]
        if ms is None:
            return (True, float("inf"))
        return (False, float(ms))

    items.sort(key=sort_key)
    return items[: max(1, num_cap)]


def _format_context_for_manager(stack: list[tuple[str, float | None]]) -> str:
    lines: list[str] = []
    for i, (cpp, ms) in enumerate(stack, start=1):
        head = f"- Kernel {i}"
        if ms is not None:
            head += f": ~{ms:.4g} ms (candidate_time_ms)"
        else:
            head += ": timing unknown"
        lines.append(head)
        lines.append(f"  (first {min(400, len(cpp))} chars) {cpp[:400].replace(chr(10), ' ')}…")
    return "\n".join(lines) if lines else "(no context)"


def _digest_run_failures(results: list) -> str:
    parts: list[str] = []
    g = 0
    e = 0
    for r in results:
        if isinstance(r, ImprovementGuardrailFailure):
            g += 1
        elif isinstance(r, ImprovementEvalFailure):
            e += 1
            er = r.eval_result
            parts.append(
                f"- eval: compile={er.get('compile_pass')} correct={er.get('correct_pass')} "
                f"candidate_time_ms={er.get('candidate_time_ms')}"
            )
    summary = f"Parallel branches: guardrail_failures={g}, eval_failures={e}."
    if parts:
        summary += "\n" + "\n".join(parts[:12])
    return summary


async def _eval_seed_entries(seed_cpps: list[str], log: LogFn) -> list[tuple[str, float | None]]:
    """Isolated eval per seed for timing; unknown ms if compile/correct fails."""

    async def one(cpp: str, i: int) -> tuple[str, float | None]:
        key = f"manager_seed_{i}_{uuid.uuid4().hex[:8]}"
        raw = await asyncio.to_thread(evaluate_candidate_cpp_isolated, cpp, key)
        er = enrich_eval_with_naive_baseline(raw)
        if er.get("compile_pass") and er.get("correct_pass"):
            ms = er.get("candidate_time_ms")
            return (cpp, float(ms) if ms is not None else None)
        log(
            f"Seed {i + 1} did not pass compile+correctness; ordering uses unknown timing for that body."
        )
        return (cpp, None)

    return list(await asyncio.gather(*[one(c, i) for i, c in enumerate(seed_cpps)]))


async def run_manager_session(
    *,
    seed_kernel_cpps: list[str],
    user_idea: str,
    num_parallel: int,
    num_runs: int,
    num_cap: int,
    max_retries: int,
    kernel_agent: Any | None = None,
    manager_model_name: str | None = None,
    log: LogFn | None = None,
    on_progress: ProgressFn | None = None,
) -> dict:
    """
    Run `num_runs` rounds; each round launches `num_parallel` improvement cycles.
    Appends **one** kernel_history file for the single fastest correct kernel across the whole
    session (if any). Does not append if every branch fails guardrails or eval.
    """
    log = log or (lambda _s: None)
    on_progress = on_progress or (lambda **_k: None)

    def prog(phase: str, run_idx: int = 0, detail: str = "") -> None:
        on_progress(phase=phase, run_idx=run_idx, num_runs=num_runs, detail=detail)

    agent = kernel_agent if kernel_agent is not None else build_agent(None)

    if not seed_kernel_cpps:
        raise ValueError("run_manager_session requires at least one seed kernel")

    n_seeds = len(seed_kernel_cpps)
    log(
        f"Evaluating {n_seeds} seed kernel(s) (isolated) for context ordering…"
    )
    prog(
        "seed_eval",
        run_idx=0,
        detail=f"Baseline timing for {n_seeds} seed(s)",
    )
    seed_entries = await _eval_seed_entries(seed_kernel_cpps, log)

    run_winners: list[tuple[str, dict]] = []
    failure_digest: str | None = None
    all_successes: list[tuple[KernelRevision, dict]] = []

    for run_idx in range(1, num_runs + 1):
        stack = build_capped_context(seed_entries, run_winners, num_cap)
        ctx_text = _format_context_for_manager(stack)

        prog("manager_llm", run_idx=run_idx, detail=f"Generating {num_parallel} prompts…")
        log(f"[run {run_idx}/{num_runs}] Manager LLM: {num_parallel} parallel prompts…")
        prompts = await generate_manager_prompts(
            num_parallel,
            user_idea,
            ctx_text,
            failure_digest,
            model_name=manager_model_name,
        )

        prog(
            "parallel_cycles",
            run_idx=run_idx,
            detail=f"Improvement cycles 1–{num_parallel} (parallel)…",
        )

        async def one_branch(i: int, prompt: str) -> Any:
            iso = f"r{run_idx}_b{i}_{uuid.uuid4().hex[:8]}"
            log(f"[run {run_idx}/{num_runs}] branch {i + 1}/{num_parallel}: start")
            return await run_improvement_cycle(
                prompt,
                stack,
                agent=agent,
                eval_isolate_key=iso,
                max_retries=max_retries,
            )

        tasks = [one_branch(i, prompts[i]) for i in range(num_parallel)]
        results = await asyncio.gather(*tasks)

        run_successes: list[ImprovementSuccess] = [
            r for r in results if isinstance(r, ImprovementSuccess)
        ]
        for s in run_successes:
            all_successes.append((s.revision, s.eval_result))

        if run_successes:
            best_in_run = min(
                run_successes,
                key=lambda s: float(s.eval_result.get("candidate_time_ms") or float("inf")),
            )
            run_winners.append((best_in_run.revision.cpp_code, best_in_run.eval_result))
            failure_digest = None
            log(
                f"[run {run_idx}/{num_runs}] winner: {best_in_run.eval_result.get('candidate_time_ms')} ms"
            )
        else:
            failure_digest = _digest_run_failures(list(results))
            log(f"[run {run_idx}/{num_runs}] no successful branch. {failure_digest}")

        prog("run_done", run_idx=run_idx, detail=f"Finished run {run_idx}/{num_runs}")

    saved_name: str | None = None
    best_eval: dict | None = None
    best_revision: KernelRevision | None = None

    if all_successes:
        best_revision, best_eval = min(
            all_successes,
            key=lambda t: float(t[1].get("candidate_time_ms") or float("inf")),
        )
        prog("save", run_idx=num_runs, detail="Saving global best to kernel_history…")
        log("Saving single best kernel from manager session to kernel_history…")
        user_req = f"[manager session] {user_idea.strip()[:500]}"
        if should_save_kernel_history(best_eval):
            path = await save_kernel_revision_with_summary(
                best_revision,
                user_req,
                eval_result=best_eval,
                is_best=should_promote_to_best_kernel(best_eval),
            )
            saved_name = path.name
        try:
            apply_generation_to_task_files(best_revision.cpp_code)
        except Exception as exc:
            log(f"apply_generation_to_task_files: {exc}")
    else:
        log("No compile+correct kernel in this manager session — kernel_history unchanged.")

    prog("done", run_idx=num_runs, detail="Manager session complete")

    return {
        "saved_kernel_history": saved_name,
        "best_eval": best_eval,
        "best_revision": best_revision,
        "run_winners_count": len(run_winners),
        "total_success_branches": len(all_successes),
    }
