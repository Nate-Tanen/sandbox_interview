# Sandbox Interview

This directory is a standalone take-home interview focused on sandbox and agent setup for local C++ matmul optimization.

## What this README is for

This document describes **what we implemented**: how the kernel improver loop works end-to-end, **what guardrails exist and why**, how the **offline evaluator** is wired in, how **`kernel_history/`** and promotion to **`best_kernel.py`** behave, and what the **Streamlit UI** adds on top of the REPL. It is meant to explain **workflow and design choices**, not to optimize for the single fastest kernel. The interview cares about setup, safety, and reasoning—not peak benchmark scores.

## Goal (interview brief)

Set up an agent or sandboxed workflow that can take the provided C++ matmul task, iterate on the candidate implementation, and improve it while preserving correctness.

## What we want to learn

We are evaluating how well a candidate can:

- set up an isolated environment for code generation or code editing
- wire an agent into that environment safely
- use an evaluator to catch compile and correctness regressions
- iterate toward a faster low-level implementation without reward hacking

## Kernel improver workflow

Both **`main.py`** (REPL) and **`UI/app.py`** (Streamlit) follow the same pipeline:

1. **Working kernel** — The session starts from `task/best_kernel.py` when it exists, otherwise `task/base_kernel.py` (`load_working_cpp()` in `main.py`). Each successful turn can update the in-memory “current” C++ for the next turn.
2. **Prior context** — The **K** newest files in `kernel_history/` are summarized (or summarized on demand) and injected as structured context so the agent sees what was tried before.
3. **Generation** — **`MAIN_KERNEL_AGENT`** (`agents/MAIN_KERNEL_AGENT.py`) returns a structured **`KernelRevision`**: full `cpp_code` for `CPP_SOURCE` plus an `explanation`.
4. **Guardrails** — Before any eval, generated C++ is checked for reward-hacking patterns (see **Guardrails** below). Failed checks trigger retries with the rejection reason and snippet fed back into the prompt, up to **`max_retries`** (env `KERNEL_GUARD_MAX_RETRIES` or the UI control). Exhaustion raises **`GuardrailRetriesExhausted`** (UI surfaces this in a dedicated tab).
5. **Staging** — Passing C++ is written to **`task/candidate.py`**.
6. **Evaluator** — **`evaluate_candidate_kernel_sync()`** runs **`sandbox_eval.evaluate_sources`** with **`task/reference.py`** vs **`task/candidate.py`**: compile, multi-trial correctness vs the reference model, then benchmark. Results are **enriched** in `main.py` with naive-baseline fields (see below).
7. **Persistence** — If compile and correctness pass, a markdown artifact may be written under **`kernel_history/`** (with timing metadata). Promotion to **`task/best_kernel.py`** uses a **separate rule** from the evaluator’s `speedup` vs `torch.matmul` (see **Evaluator and promotion choices**).
8. **Optional eval skip** — For local development, **`KERNEL_SKIP_EVAL=1`** skips the evaluator; **`KERNEL_SAVE_WITHOUT_EVAL=1`** allows appending history without eval.

## Guardrails

**Intent:** Block **reward hacking**—delegating the GEMM to framework/BLAS-style shortcuts instead of implementing the multiply in the extension—while still allowing legitimate algorithmic work (tiling, reordering, blocking, etc.). The agent system prompt in **`MAIN_KERNEL_AGENT`** spells out this narrow definition.

**Mechanics:**

- **`agents/GUARDRAIL_AGENT/rules.py`** — Deterministic substring denylist (e.g. `torch::matmul`, `at::mm`, BLAS/MKL includes, `system(`). Hits are cheap and reproducible.
- **`check_reward_hacking_cpp`** — Runs the denylist first; optionally an LLM check if **`KERNEL_GUARDRAIL_USE_LLM`** is set (see `GUARDRAIL_AGENT.py`).
- **`run_kernel_turn`** — Loops: generate → guardrail → on failure append structured **guardrail feedback** to the next message until pass or max retries.

This is independent of the **evaluator**: guardrails are a pre-filter on generated source; the evaluator checks compile/correctness/speed against the harness.

## Summaries and `kernel_history/`

After a revision is saved, a **kernel summary** (tags, notes, high-level description) can be produced and embedded in the markdown file. The **K** newest saved kernels’ summaries are used as **prior context** for the next turns (configurable in the UI). Saved files record metadata such as candidate/reference times, evaluator `speedup`, **`naive_baseline_ms`**, **`speedup_vs_naive`**, and **`is_best`** when relevant—so the UI can sort, label, and compare runs without re-running eval.

## Evaluator and promotion choices

**Correctness and timing** — The evaluator always uses **`task/reference.py`** (`torch.matmul`) as the numerical reference. That is the right contract for “does this extension match the reference GEMM?”

**Why `speedup` ≠ promotion rule** — The same run reports **`speedup` = reference_time_ms / candidate_time_ms** (library vs your kernel). With `torch.matmul` as the reference, naive C++ is vastly *slower*, so this ratio is often near zero. Using **`speedup > 1`** as “promote to best” would mean “beat `torch.matmul`,” which is the wrong bar for this task.

**Promotion bar we use** — We profiled the **naive** implementation in **`task/base_kernel.py`** once and store **`NAIVE_BASELINE_TIME_MS`** in `main.py`. We promote to **`task/best_kernel.py`** when compile and correctness pass and **`candidate_time_ms`** is **strictly below** that naive baseline—i.e. the candidate actually improves on the intentional starting point, not on PyTorch’s fused matmul.

**Enriched fields** — After each eval, `main.py` adds **`naive_baseline_ms`** and **`speedup_vs_naive`** (`NAIVE_BASELINE_TIME_MS / candidate_time_ms`) for display, history, and debugging.

**Refreshing the baseline** — If you change `N`, compiler flags, or benchmark env vars, re-profile `base_kernel.py` with the same `evaluate_sources` call and update **`NAIVE_BASELINE_TIME_MS`**.

```bash
python -c "
from pathlib import Path
from sandbox_eval.evaluator import evaluate_sources
import json
root = Path('.')
ref = (root / 'task' / 'reference.py').read_text(encoding='utf-8')
base = (root / 'task' / 'base_kernel.py').read_text(encoding='utf-8')
r = evaluate_sources(ref_src=ref, candidate_src=base, build_root=root / 'build' / 'eval_profile', num_trials=10, seed_num=42)
print(json.dumps({'candidate_time_ms': r['candidate_time_ms']}, indent=2))
"
```

Use the printed **`candidate_time_ms`** as the new constant in `main.py`.

## Streamlit UI (`UI/app.py`)

```bash
streamlit run UI/app.py
```

The UI mirrors the REPL pipeline (same agent, guardrails, eval, history). It adds:

- **Session** — Reset to baseline; open the **current working kernel** in a tab (same file as `load_working_cpp()`).
- **Prior summaries (K)** — Controls how many newest `kernel_history` summaries are injected per turn.
- **Guardrail max retries** — Per-session override for `run_kernel_turn` (default still available via `KERNEL_GUARD_MAX_RETRIES`).
- **Saved kernels** — Browse history with timing in labels; sort by recency or by fastest candidate time; **⭐** = among the K newest used for prompt context; **✅** = working kernel row; **🏆** = revision matches **`task/best_kernel.py`**.

## Folder layout

- `run_eval.py` — Standalone CLI around `evaluate_sources`
- `main.py` — REPL (`python main.py`); eval + `kernel_history` + promotion helpers
- `UI/app.py` — Streamlit front end
- `agents/` — `MAIN_KERNEL_AGENT`, `GUARDRAIL_AGENT`, `SUMMARY_AGENT`
- `sandbox_eval/` — Evaluator package
- `kernel_history/` — Saved markdown revisions (gitignored by default if configured)
- `task/reference.py` — Reference `torch.matmul` model (correctness + library timing)
- `task/base_kernel.py` — Naive C++ baseline; defines the profiled promotion threshold
- `task/candidate.py` — Latest proposed kernel from the agent (evaluated each turn)
- `task/best_kernel.py` — Last promoted kernel (beats naive baseline when promoted)

## Task

Setup an environment where an agent can go in and optimize the square GEMM task defined by `task/reference.py`.

The provided baseline in `task/base_kernel.py` is intentionally naive:

- a scalar triple loop computes the full matrix product
- no tiling or cache blocking
- no vectorization
- no multithreading
- no packing or layout transformations

This should be runnable on your local laptop.

Preferred approach:

- custom C++ loaded through `torch.utils.cpp_extension`

Word of advice: make sure the agent does not reward-hack. Agents often try to fall back to library calls such as `torch.matmul`, NumPy matmul, or BLAS-backed wrappers. Other times, the agent will try to change the testing harness, be careful with that. The goal here is to generate an actually improved low-level implementation, not a library shortcut or some other reward hack.

## Setup

### Create A New Environment

Use any local Python environment manager you prefer. Example with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install PyTorch

Install a CPU build of PyTorch appropriate for your machine.

Example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Install The Remaining Requirements

```bash
pip install -r requirements.txt
```

This installs the remaining Python packages needed by the evaluator, including:

- `numpy`
- `psutil`
- `ninja`

## Minimum Requirements

This evaluator expects:

- Python 3.12+
- a local C++ toolchain that PyTorch extensions can build against
- a CPU-compatible PyTorch install

For local builds this usually means:

- macOS: Xcode command line tools
- Windows: Visual Studio Build Tools
- Linux: GCC or Clang

## Evaluator reference (API contract)

The stock evaluator uses two Python source files:

- `reference.py`
- `candidate.py` or `base_kernel.py`

### Reference Contract

`reference.py` must define all of the following at module scope:

- `Model`
- `get_init_inputs()`
- `get_inputs()`

The evaluator expects:

- `Model` to be a `torch.nn.Module`
- `get_init_inputs()` to return a list of constructor inputs
- `get_inputs()` to return a list of runtime inputs for `forward`

### Candidate Contract

`candidate.py` or `base_kernel.py` must define:

- `ModelNew`

The evaluator expects:

- `ModelNew` to be a `torch.nn.Module`
- `ModelNew` to accept the same initialization inputs as `Model`
- `ModelNew.forward(...)` to accept the same runtime inputs as `Model.forward(...)`

### Compile Check

The compile stage passes only if:

- the candidate file is valid Python
- the candidate module executes successfully
- `ModelNew` is defined after execution

If the candidate builds a C++ extension during import or initialization, that is allowed.

### Correctness Check

The correctness stage:

- instantiates `Model` and `ModelNew` with the same initialization inputs
- generates runtime inputs from `get_inputs()` across multiple trials
- runs both models on CPU
- compares output structure, shape, and values

The correctness stage fails if:

- output tuple or list lengths differ
- output shapes differ
- non-tensor outputs appear where tensor outputs are expected
- output values deviate beyond the evaluator tolerance
- the candidate throws runtime errors

### Output Comparison Rules

For each output tensor pair, the evaluator checks:

- absolute tolerance: `1e-3`
- relative tolerance: `1e-3`
- pass threshold: at least `95%` of elements must be within tolerance

### Evaluator Output

The evaluator prints JSON with:

- `compile_pass`
- `correct_pass`
- `reference_time_ms`
- `candidate_time_ms`
- `speedup`
- `metadata`

Timing is only populated when the candidate passes both compile and correctness checks.

- `reference_time_ms`: mean runtime of the reference implementation in milliseconds
- `candidate_time_ms`: mean runtime of the candidate implementation in milliseconds
- `speedup`: `reference_time_ms / candidate_time_ms` (library reference vs your candidate; values near **0** are normal when the reference is `torch.matmul` and the candidate is a naive loop)

Detailed timing stats are also stored under `metadata["timing"]`.

## Running `run_eval.py` (standalone)

For ad-hoc runs without the agent:

```bash
python run_eval.py \
  --reference task/reference.py \
  --candidate task/candidate.py
```

Use **`task/base_kernel.py`** as `--candidate` to measure the naive kernel the same way the harness measures any candidate. The **integrated** agent flow (guardrails → `candidate.py` → `evaluate_candidate_kernel_sync()` → history / promotion) is described under **Kernel improver workflow** and **Evaluator and promotion choices** above.

## Suggested Deliverables

- the sandbox or agent setup used
- the improved C++ implementation
- a document explaining the workflow and design choices (the sections above are an example of that kind of write-up)

Reminder: we are not evaluating how fast you can get this matmul—we are evaluating how well you can set up the agent to optimize it safely and explain your choices.