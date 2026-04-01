import os
from typing import Any

from agents import Agent, Runner
from pydantic import BaseModel, Field

from GUARDRAIL_AGENT import check_reward_hacking_cpp


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

## What “guardrails” mean here (read this)
- They exist to stop **reward hacking**: calling a **pre-built matrix multiply** (framework matmul APIs, BLAS/LAPACK-style shortcuts) instead of implementing that multiply yourself in this extension.
- They are **not** there to forbid algorithmic ideas. **Strassen, blocking, tiling, recursion, Winograd-style structure, loop reordering**, etc. are all valid **if** you implement them without those forbidden shortcuts.
- Using **PyTorch tensors**, `Tensor` views/slices, `torch::zeros`, `contiguous()`, indexing, and scalar arithmetic in C++ is **expected** — normal extension code, not cheating.

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

## Reward hacking (must avoid) — narrow definition
- **Only** this class of cheat: delegating GEMM to a **library or framework matmul** you did not implement — e.g. `torch::matmul`, `at::matmul`, BLAS/LAPACK/MKL/Eigen **calls** that compute the product for you.
- Do **not** add extra `#include` beyond `<torch/extension.h>` to pull in external linear algebra to do the multiply, spawn processes to cheat, or alter the Python harness from generated C++.
- Do **not** put the exact substring `torch::matmul` inside quoted `TORCH_CHECK` strings (static tooling may flag it). Say “reference GEMM” or similar instead.

## Guardrail feedback (when present)
If prior attempts failed with reasons and rejected `cpp_code`, you must address those failures and return a **new** compliant full kernel. Do not repeat forbidden patterns.

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


def _format_guardrail_feedback(
    attempts: list[tuple[int, str, str]],
) -> str:
    """Format prior failures (attempt_no, reason, rejected_cpp) for the next prompt."""
    parts: list[str] = []
    for attempt_no, reason, rejected_cpp in attempts:
        parts.append(
            f"### Guardrail failure (attempt {attempt_no})\n"
            f"**Reason:** {reason}\n\n"
            "**Rejected candidate (reference only — do not ship; regenerate compliant code):**\n"
            f"```cpp\n{rejected_cpp.rstrip()}\n```\n"
        )
    return "## Guardrail feedback — fix previous attempt(s)\n\n" + "\n".join(parts) + "\n"


# Build the per-turn prompt: summaries, optional guardrail retries, user request, current kernel.
def build_turn_message(
    *,
    user_request: str,
    current_cpp: str,
    prior_summaries: list[str] | None = None,
    guardrail_attempts: list[tuple[int, str, str]] | None = None,
) -> str:
    blocks: list[str] = []
    if prior_summaries:
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
    )


# Run one user turn: generate, check guardrails; on failure retry with feedback until pass or max retries.
async def run_kernel_turn(
    agent: Any,
    user_request: str,
    current_cpp: str,
    prior_summaries: list[str] | None = None,
    *,
    max_retries: int | None = None,
) -> KernelRevision:
    if max_retries is None:
        max_retries = int(os.environ.get("KERNEL_GUARD_MAX_RETRIES", "3"))
    else:
        max_retries = int(max_retries)
    failed_attempts: list[tuple[int, str, str]] = []
    last_reason = ""

    for attempt in range(1, max_retries + 1):
        message = build_turn_message(
            user_request=user_request,
            current_cpp=current_cpp,
            prior_summaries=prior_summaries,
            guardrail_attempts=failed_attempts or None,
        )
        result = await Runner.run(agent, message)
        revision = result.final_output_as(KernelRevision, raise_if_incorrect_type=True)

        ok, reason = await check_reward_hacking_cpp(revision.cpp_code)
        if ok:
            return revision

        last_reason = reason
        failed_attempts.append((attempt, reason, revision.cpp_code))

    raise GuardrailRetriesExhausted(
        f"Guardrail blocked output after {max_retries} attempt(s). Last reason: {last_reason}",
        last_reason=last_reason,
        failed_attempts=failed_attempts,
    )
