import json
import os
from typing import Any

from agents import Agent, Runner
from pydantic import BaseModel, Field


class KernelSummary(BaseModel):
    run_id: str = Field(..., description="Identifier of the kernel revision.")
    optimization_tags: list[str] = Field(
        default_factory=list,
        description="Current optimization tags that are truly present in this kernel now.",
    )
    high_level_summary: str = Field(
        ...,
        description="Short summary of what this kernel does and key optimization choices.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Extra notes, caveats, or regression risks.",
    )


SUMMARY_AGENT_INSTRUCTIONS = """You summarize kernel revisions for iterative optimization memory.

Requirements:
- Re-check the provided C++ code directly; do not trust prior labels.
- Only output optimization_tags that are currently present in this exact kernel.
- Remove stale/obsolete tags that no longer apply.
- Keep high_level_summary concise and factual.
- Include notes for caveats, risks, or suspicious regressions when relevant.
- Return strictly the structured fields in the output schema.
"""


def _build_summary_agent(model_name: str | None = None) -> Agent:
    return Agent(
        name="SUMMARY_AGENT",
        instructions=SUMMARY_AGENT_INSTRUCTIONS,
        model=model_name or os.environ.get("KERNEL_AGENT_MODEL", "gpt-4o-mini"),
        output_type=KernelSummary,
    )


async def summarize_kernel_revision(
    *,
    run_id: str,
    cpp_code: str,
    explanation: str,
    model_name: str | None = None,
) -> KernelSummary:
    prompt = (
        "Summarize this kernel revision.\n\n"
        f"run_id: {run_id}\n\n"
        "Explanation:\n"
        f"{explanation}\n\n"
        "C++ kernel:\n"
        f"```cpp\n{cpp_code}\n```\n\n"
        "Return optimization_tags that are truly present in the C++ now."
    )
    result = await Runner.run(_build_summary_agent(model_name), prompt)
    summary = result.final_output_as(KernelSummary, raise_if_incorrect_type=True)
    return KernelSummary(
        run_id=run_id,
        optimization_tags=summary.optimization_tags,
        high_level_summary=summary.high_level_summary,
        notes=summary.notes,
    )


def summary_to_json(summary: KernelSummary) -> str:
    return json.dumps(summary.model_dump(mode="json"), indent=2)
