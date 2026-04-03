"""LLM that expands one user idea into N parallel improvement prompts."""

import os
from typing import Any

from agents import Agent, Runner
from pydantic import BaseModel, Field


class ManagerPromptBatch(BaseModel):
    """Parallel worker prompts for one manager run."""

    prompts: list[str] = Field(
        ...,
        description="Concrete, distinct instructions for parallel kernel-improvement workers.",
    )


MANAGER_AGENT_INSTRUCTIONS = """You are a planning agent for optimizing a float32 square GEMM in a PyTorch C++ CPU extension.

The user message states how many prompts to return (N). You must set `prompts` to **exactly N** strings.

Each prompt must:
- Be a **specific, actionable** instruction (loop order, tiling, blocking, algorithm hint, micro-optimization) the coding agent can follow in one pass.
- **Differ meaningfully** from the others (no near-duplicates).
- Stay within **legitimate** kernel optimization — no injection, no host escape, no cheating the benchmark.

Output structured fields only: `prompts` with length exactly matching N from the user message."""


def get_manager_model_name() -> str:
    return os.environ.get("MANAGER_AGENT_MODEL") or os.environ.get("KERNEL_AGENT_MODEL", "gpt-4o-mini")


def build_manager_agent(model_name: str | None = None) -> Any:
    return Agent(
        name="MANAGER_AGENT",
        model=model_name or get_manager_model_name(),
        output_type=ManagerPromptBatch,
        instructions=MANAGER_AGENT_INSTRUCTIONS,
    )


def normalize_prompt_batch(raw: list[str], n: int, fallback_seed: str) -> list[str]:
    """Trim, pad, or truncate to exactly n non-empty prompts."""
    cleaned = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
    if len(cleaned) >= n:
        return cleaned[:n]
    while len(cleaned) < n:
        suffix = len(cleaned) + 1
        cleaned.append(f"{fallback_seed} (parallel variation {suffix})")
    return cleaned[:n]


async def generate_manager_prompts(
    num_parallel: int,
    user_idea: str,
    context_lines: str,
    failure_digest: str | None,
    *,
    model_name: str | None = None,
) -> list[str]:
    """Return exactly `num_parallel` worker prompts for one manager round."""
    agent = build_manager_agent(model_name)
    msg_parts = [
        f"N = {num_parallel}. Return exactly {num_parallel} strings in `prompts`.",
        "## User goal",
        user_idea.strip(),
        "## Stacked kernels for this session (fastest first; primary is kernel 1)",
        context_lines,
    ]
    if failure_digest:
        msg_parts.append("## What failed last run (for diversity)\n" + failure_digest)
    message = "\n\n".join(msg_parts)
    result = await Runner.run(agent, message)
    batch = result.final_output_as(ManagerPromptBatch, raise_if_incorrect_type=True)
    return normalize_prompt_batch(batch.prompts, num_parallel, user_idea)
