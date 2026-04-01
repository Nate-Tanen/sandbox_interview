"""Deterministic denylist for generated C++ (no LLM)."""

from dataclasses import dataclass


# One blocked substring plus a short reason for error messages.
@dataclass(frozen=True)
class DisallowedPattern:
    pattern: str
    reason: str


# Substring checks (case-insensitive in find_disallowed_hits). Extend as needed.
DISALLOWED_CPP_PATTERNS: tuple[DisallowedPattern, ...] = (
    DisallowedPattern("torch::matmul", "Direct framework matmul call is disallowed."),
    DisallowedPattern("torch::mm", "ATen-style mm shortcut is disallowed."),
    DisallowedPattern("at::matmul", "ATen matmul shortcut is disallowed."),
    # 2D/batched matmul shortcuts (same policy as matmul; substring match).
    DisallowedPattern("at::mm", "ATen mm shortcut is disallowed."),
    DisallowedPattern("at::bmm", "ATen bmm shortcut is disallowed."),
    DisallowedPattern("at::addmm", "ATen addmm shortcut is disallowed."),
    DisallowedPattern("cblas_", "BLAS shortcuts are disallowed for this task."),
    DisallowedPattern("openblas", "External BLAS dependencies are disallowed."),
    DisallowedPattern("mkl_", "MKL shortcuts are disallowed for this task."),
    DisallowedPattern("#include <cblas", "BLAS include is disallowed."),
    DisallowedPattern("#include <mkl", "MKL include is disallowed."),
    DisallowedPattern("#include <eigen", "Eigen include is disallowed."),
    DisallowedPattern("#include <lapack", "LAPACK include is disallowed."),
    DisallowedPattern("pybind11", "Python/C++ bridge usage is disallowed."),
    DisallowedPattern("numpy", "NumPy usage in generated C++ is disallowed."),
    DisallowedPattern("system(", "OS shell/process execution is disallowed."),
    DisallowedPattern("popen(", "OS process execution is disallowed."),
    DisallowedPattern("fork(", "Process forking is disallowed."),
    DisallowedPattern("dlopen(", "Dynamic loading escapes are disallowed."),
)


# Return human-readable hit lines for any denylist pattern present in cpp_code.
def find_disallowed_hits(cpp_code: str) -> list[str]:
    source = cpp_code.lower()
    hits: list[str] = []
    for entry in DISALLOWED_CPP_PATTERNS:
        if entry.pattern.lower() in source:
            hits.append(f"{entry.pattern}: {entry.reason}")
    return hits
