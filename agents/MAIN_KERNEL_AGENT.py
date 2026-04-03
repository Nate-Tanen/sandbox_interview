import os
from dataclasses import dataclass
from typing import Annotated, Any

from agents import Agent, Runner, RunContextWrapper, WebSearchTool, function_tool
from pydantic import BaseModel, Field

from GUARDRAIL_AGENT import check_reward_hacking_cpp


@dataclass
class KernelAgentContext:
    """Shared across guardrail retries for one user message (one web research budget)."""

    web_research_consumed: bool = False


class KernelRevision(BaseModel):
    """Structured reply: full C++ body for CPP_SOURCE plus explanation."""

    cpp_code: str = Field(
        ...,
        description=(
            "Complete C++ translation unit for CPP_SOURCE: start with #include <torch/extension.h>, "
            "then implement torch::Tensor naive_gemm_cpu(torch::Tensor A, torch::Tensor B) for float32 "
            "square N×N inputs with the same numerical result as the reference float32 GEMM for those inputs."
        ),
    )
    explanation: str = Field(
        ...,
        description="What you changed compared to the current kernel, why, and any tradeoffs or risks.",
    )


MAIN_KERNEL_AGENT_INSTRUCTIONS = """You are a senior performance engineer improving a PyTorch C++ CPU extension.

## Task (interview sandbox)
- The product is a float32 **square GEMM** in C++ loaded via `torch.utils.cpp_extension` (see `task/base_kernel.py`).
- The goal is a **faster, still correct** implementation. An offline evaluator compares the candidate to a reference model on CPU (compile + numerical checks).
- In the **Streamlit UI**, the session keeps a rolling **current kernel**: each chat turn refines the last `CPP_SOURCE` unless the user resets to the baseline from `task/base_kernel.py`.

## What automated “guardrails” mean here (read this)
- A separate **GUARDRAIL_AGENT** pass looks only for **injection / sandbox-escape** patterns in your C++ (e.g. shell execution, suspicious dynamic loading, unambiguous instruction-injection text). It does **not** scan for “illegal” GEMM APIs — use any correct approach the evaluator accepts.
- **You** should still implement a real low-level GEMM strategy; the offline evaluator compares to the reference model. Do not rely on guardrails to enforce “no BLAS” — that is your design choice and the eval harness.

## Your job each turn
- You receive the **current** `CPP_SOURCE` body and a **user request** (what to change or try next).
- You output a **full replacement** `cpp_code` for that translation unit, plus an **explanation** of what changed and why.

## Correctness (must hold)
- Keep the public API exactly: `torch::Tensor naive_gemm_cpu(torch::Tensor A, torch::Tensor B)`.
- Float32, square N×N matrices only; result must match the reference GEMM for valid inputs within the project’s evaluation tolerances.
- Start `cpp_code` with `#include <torch/extension.h>` as the first non-empty line.
- Keep sensible `TORCH_CHECK` usage for device, dtypes, and shapes like the baseline unless the user explicitly asks otherwise.

## Legitimate optimizations (encouraged when requested or helpful)
- **Algorithmic**: different asymptotics or structure (e.g. Strassen/recursive splits), blocking/tiling, loop reordering (ijk vs ikj), unrolling, cache-friendly access.
- **Mechanistic**: nested loops, raw pointers from `data_ptr`, tensor slices/views, accumulation order changes — anything that stays correct and does not smuggle in a library GEMM.

## Quality and honesty (your responsibility)
- Prefer a **real** optimized implementation that passes the evaluator. Shortcuts that break correctness or the interview intent are evaluated by **tests**, not by the injection guardrail.
- Do **not** embed shell commands, process-spawning, or other host-escape patterns in C++ — the guardrail will reject those.

## Guardrail feedback (when present)
If prior attempts failed with **injection guardrail** reasons, fix the security issue and return a **new** full kernel. If the failure was a false positive, remove ambiguous constructs while preserving behavior.

## Optional web research (you decide — users cannot trigger this)
- You may call **`research_algorithm_summary`** only when you **lack concrete knowledge** needed to implement a **specific** algorithm, trick, or technique for this GEMM (e.g. an obscure loop order, a numeric stability detail, or a structured approach you are not sure about).
- **Do not** call it because the user asked for “research” or “look it up” — ignore such wording unless *you* still need technical grounding.
- **At most one successful research call per user message** (enforced by the tool). If research was already used this turn, proceed from memory and prior context only.
- After research, fold what you learned into `cpp_code` and `explanation` as usual.

## Output format (mandatory)
- Structured fields only: `cpp_code` (raw C++, no Markdown fences inside the string) and `explanation` (plain text).
"""


class GuardrailRetriesExhausted(RuntimeError):
    """Raised when guardrail checks fail after max retries."""

    def __init__(
        self,
        message: str,
        *,
        last_reason: str,
        failed_attempts: list[tuple[int, str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.last_reason = last_reason
        self.failed_attempts = failed_attempts or []


# Resolve the model name from env with a sensible default.
def get_model_name() -> str:
    return os.environ.get("KERNEL_AGENT_MODEL", "gpt-4o-mini")


def get_research_model_name() -> str:
    """Model for the nested web-search helper (defaults to main agent model)."""
    return os.environ.get("KERNEL_RESEARCH_MODEL") or get_model_name()


@function_tool
async def research_algorithm_summary(
    wrapper: RunContextWrapper[KernelAgentContext],
    topic: Annotated[
        str,
        "Specific algorithm, technique, or narrow technical question you need summarized to implement this GEMM correctly in C++.",
    ],
) -> str:
    """
    Fetch a one-time web-backed summary for implementation grounding.
    Enforces a single research call per user message (shared across guardrail retries).
    """
    if wrapper.context.web_research_consumed:
        return (
            "Web research was already used once for this user message. Do not call this tool again. "
            "Use your prior notes, training knowledge, and the current kernel to finish."
        )
    wrapper.context.web_research_consumed = True

    research_agent = Agent(
        name="WebResearchAssistant",
        instructions=(
            "You have the web_search tool. The downstream task is a float32 square GEMM in a PyTorch C++ CPU extension. "
            "Search as needed, then give a concise, factual summary: key ideas, common pitfalls, and implementation hints. "
            "Prefer correctness and numerical stability notes. Keep under ~800 words unless the topic demands more. "
            "Do not output a full production kernel—only guidance."
        ),
        tools=[WebSearchTool()],
        model=get_research_model_name(),
    )
    prompt = (
        "Produce a technical summary for someone implementing this in C++ inside torch.utils.cpp_extension:\n\n"
        f"{topic.strip()}"
    )
    result = await Runner.run(research_agent, prompt)
    out = result.final_output
    return str(out) if out is not None else "(no research output)"


def _format_guardrail_feedback(
    attempts: list[tuple[int, str, str]],
) -> str:
    """Format prior failures (attempt_no, reason, rejected_cpp) for the next prompt."""
    parts: list[str] = []
    for attempt_no, reason, rejected_cpp in attempts:
        parts.append(
            f"### Injection guardrail failure (attempt {attempt_no})\n"
            f"**Reason:** {reason}\n\n"
            "**Rejected candidate (reference only — do not ship; fix injection/sandbox issues):**\n"
            f"```cpp\n{rejected_cpp.rstrip()}\n```\n"
        )
    return "## Injection guardrail feedback — fix previous attempt(s)\n\n" + "\n".join(parts) + "\n"


def _sort_manager_context_kernels(
    kernels: list[tuple[str, float | None]],
) -> list[tuple[str, float | None]]:
    """Fastest (lowest ms) first; unknown timing last."""

    def sort_key(item: tuple[str, float | None]) -> tuple[bool, float]:
        ms = item[1]
        if ms is None:
            return (True, float("inf"))
        return (False, float(ms))

    return sorted(kernels, key=sort_key)


# Build the per-turn prompt: summaries, optional guardrail retries, user request, current kernel.
def build_turn_message(
    *,
    user_request: str,
    current_cpp: str,
    prior_summaries: list[str] | None = None,
    guardrail_attempts: list[tuple[int, str, str]] | None = None,
    manager_context_kernels: list[tuple[str, float | None]] | None = None,
) -> str:
    blocks: list[str] = []
    if manager_context_kernels is not None:
        ordered = _sort_manager_context_kernels(list(manager_context_kernels))
        parts: list[str] = []
        if len(ordered) > 1:
            parts.append(
                "## Additional context kernels (reference only — do not treat as the edit target unless asked)\n"
            )
            for i, (kcpp, ms) in enumerate(ordered[1:], start=2):
                label = f"Kernel {i}"
                if ms is not None:
                    label += f" — ~{ms:.4g} ms candidate_time_ms"
                else:
                    label += " — timing unknown"
                parts.append(f"### {label}\n```cpp\n{kcpp}\n```\n")
            blocks.append("\n".join(parts))
    elif prior_summaries:
        blocks.append(
            "## Prior kernel summaries (newest first)\n" + "\n\n".join(prior_summaries) + "\n"
        )
    if guardrail_attempts:
        blocks.append(_format_guardrail_feedback(guardrail_attempts))
    blocks.append(f"## User request\n{user_request.strip()}\n")
    blocks.append(f"## Current CPP_SOURCE body\n```cpp\n{current_cpp}\n```\n")
    return "\n".join(blocks)


# Construct the MAIN_KERNEL_AGENT with structured output (guardrails run in run_kernel_turn with retries).
def build_agent(model_name: str | None = None) -> Any:
    return Agent(
        name="MAIN_KERNEL_AGENT",
        instructions=MAIN_KERNEL_AGENT_INSTRUCTIONS,
        model=model_name or get_model_name(),
        output_type=KernelRevision,
        tools=[research_algorithm_summary],
    )


# Run one user turn: generate, check guardrails; on failure retry with feedback until pass or max retries.
# One KernelAgentContext per user message: web research budget is shared across guardrail retries.
async def run_kernel_turn(
    user_request: str,
    current_cpp: str,
    prior_summaries: list[str] | None = None,
    *,
    agent: Any | None = None,
    model_name: str | None = None,
    context: KernelAgentContext | None = None,
    max_retries: int | None = None,
    manager_context_kernels: list[tuple[str, float | None]] | None = None,
) -> tuple[KernelRevision, KernelAgentContext]:
    if max_retries is None:
        max_retries = int(os.environ.get("KERNEL_GUARD_MAX_RETRIES", "3"))
    else:
        max_retries = int(max_retries)
    ctx = context if context is not None else KernelAgentContext()
    agent = agent if agent is not None else build_agent(model_name)

    if manager_context_kernels is not None:
        ordered = _sort_manager_context_kernels(list(manager_context_kernels))
        if not ordered:
            raise ValueError("manager_context_kernels must contain at least one kernel")
        current_cpp = ordered[0][0]
        prior_summaries = None

    failed_attempts: list[tuple[int, str, str]] = []
    last_reason = ""

    for attempt in range(1, max_retries + 1):
        message = build_turn_message(
            user_request=user_request,
            current_cpp=current_cpp,
            prior_summaries=prior_summaries,
            guardrail_attempts=failed_attempts or None,
            manager_context_kernels=manager_context_kernels,
        )
        result = await Runner.run(agent, message, context=ctx)
        revision = result.final_output_as(KernelRevision, raise_if_incorrect_type=True)

        ok, reason = await check_reward_hacking_cpp(revision.cpp_code)
        if ok:
            return revision, ctx

        last_reason = reason
        failed_attempts.append((attempt, reason, revision.cpp_code))

    raise GuardrailRetriesExhausted(
        f"Injection guardrail blocked output after {max_retries} attempt(s). Last reason: {last_reason}",
        last_reason=last_reason,
        failed_attempts=failed_attempts,
    )
