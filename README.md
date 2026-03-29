# Sandbox Interview

This directory is a standalone take-home interview focused on sandbox and agent setup for local C++ matmul optimization.

## Goal

Set up an agent or sandboxed workflow that can take the provided C++ matmul task, iterate on the candidate implementation, and improve it while preserving correctness.

## What We Want To Learn

We are evaluating how well a candidate can:

- set up an isolated environment for code generation or code editing
- wire an agent into that environment safely
- use an evaluator to catch compile and correctness regressions
- iterate toward a faster low-level implementation without reward hacking

## Folder Layout

- `run_eval.py`: CLI for compile and correctness evaluation
- `requirements.txt`: Python dependencies for the standalone evaluator
- `sandbox_eval/`: standalone evaluator code
- `task/reference.py`: reference implementation
- `task/base_kernel.py`: intentionally naive C++ CPU starting point
- `task/candidate.py`: working area for the improved implementation

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

## Evaluator Contract

The evaluator uses two Python source files:

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
- `speedup`: `reference_time_ms / candidate_time_ms`

Detailed timing stats are also stored under `metadata["timing"]`.

## Running The Evaluator

```bash
python run_eval.py \
  --reference task/reference.py \
  --candidate task/base_kernel.py
```

## Suggested Deliverables

- the sandbox or agent setup used
- the improved C++ implementation
- a document explaining the workflow and design choices

Reminder, we are not evaluating how fast you can get this matmul, we are evaluating how well you can set up the agent to optimize it.