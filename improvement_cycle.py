"""One improvement cycle: MAIN_KERNEL_AGENT → guardrails → isolated evaluator."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_AGENTS = _ROOT / "agents"
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))

from MAIN_KERNEL_AGENT import (  # noqa: E402
    GuardrailRetriesExhausted,
    KernelAgentContext,
    KernelRevision,
    run_kernel_turn,
)

from main import (  # noqa: E402
    enrich_eval_with_naive_baseline,
    evaluate_candidate_cpp_isolated,
    kernel_eval_skipped,
)


@dataclass
class ImprovementSuccess:
    revision: KernelRevision
    kernel_ctx: KernelAgentContext
    eval_result: dict


@dataclass
class ImprovementGuardrailFailure:
    last_reason: str
    failed_attempts: list[tuple[int, str, str]]


@dataclass
class ImprovementEvalFailure:
    revision: KernelRevision
    eval_result: dict


ImprovementCycleResult = ImprovementSuccess | ImprovementGuardrailFailure | ImprovementEvalFailure


async def run_improvement_cycle(
    user_request: str,
    manager_context_kernels: list[tuple[str, float | None]],
    *,
    agent: Any,
    eval_isolate_key: str,
    max_retries: int,
) -> ImprovementCycleResult:
    ctx = KernelAgentContext()
    try:
        revision, kctx = await run_kernel_turn(
            user_request,
            "",
            prior_summaries=None,
            agent=agent,
            max_retries=max_retries,
            context=ctx,
            manager_context_kernels=manager_context_kernels,
        )
    except GuardrailRetriesExhausted as exc:
        return ImprovementGuardrailFailure(
            last_reason=exc.last_reason,
            failed_attempts=list(exc.failed_attempts),
        )

    if kernel_eval_skipped():
        return ImprovementEvalFailure(
            revision=revision,
            eval_result={
                "compile_pass": False,
                "correct_pass": False,
                "skipped_eval": True,
            },
        )

    raw = await asyncio.to_thread(
        evaluate_candidate_cpp_isolated,
        revision.cpp_code,
        eval_isolate_key,
    )
    eval_result = enrich_eval_with_naive_baseline(raw)
    if not eval_result.get("compile_pass") or not eval_result.get("correct_pass"):
        return ImprovementEvalFailure(revision=revision, eval_result=eval_result)
    return ImprovementSuccess(revision=revision, kernel_ctx=kctx, eval_result=eval_result)
