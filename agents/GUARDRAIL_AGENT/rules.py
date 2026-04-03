"""Deterministic checks for obvious sandbox-escape / injection primitives (no matmul policy)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DisallowedPattern:
    pattern: str
    reason: str


# High-confidence patterns only: process execution, dynamic load of foreign code, obvious shell vectors.
# We do NOT match framework APIs (torch::matmul, BLAS, etc.) — those are not "injection" and are left to the LLM reviewer only if needed elsewhere.
DISALLOWED_CPP_PATTERNS: tuple[DisallowedPattern, ...] = (
    DisallowedPattern("system(", "Process/shell execution (injection vector)."),
    DisallowedPattern("popen(", "Shell pipeline (injection vector)."),
    DisallowedPattern("fork(", "Process fork (sandbox escape vector)."),
    DisallowedPattern("execl(", "Process execution."),
    DisallowedPattern("execv(", "Process execution."),
    DisallowedPattern("execve(", "Process execution."),
    DisallowedPattern("dlopen(", "Dynamic library load (code-injection vector)."),
    DisallowedPattern("dlsym(", "Dynamic symbol resolution after dlopen."),
    DisallowedPattern("winexec", "Windows process execution."),
    DisallowedPattern("createprocess", "Windows process creation."),
    DisallowedPattern("shellexecute", "Windows shell execution."),
)


def find_disallowed_hits(cpp_code: str) -> list[str]:
    source = cpp_code.lower()
    hits: list[str] = []
    for entry in DISALLOWED_CPP_PATTERNS:
        if entry.pattern.lower() in source:
            hits.append(f"{entry.pattern}: {entry.reason}")
    return hits
