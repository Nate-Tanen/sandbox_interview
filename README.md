# Sandbox Interview

This directory is a standalone take-home interview focused on sandbox and agent setup for local C++ matmul optimization.

## Goal

Set up an agent or sandboxed workflow that can take the provided C++ matmul task, iterate on the candidate implementation, and improve it while preserving correctness.

## What we want to learn

We are evaluating how well a candidate can:

- set up an isolated environment for code generation or code editing
- wire an agent into that environment safely
- use an evaluator to catch compile and correctness regressions
- iterate toward a faster low-level implementation without reward hacking

## Architecture diagrams

Two concepts matter: a single **improvement cycle** (generate, guardrail checks, eval, with guardrail retries), and the **manager workflow** (many cycles in parallel, repeated rounds, one best artifact at the end of the session). Each subsection below shows the diagram first, then walks through it.

### Improvement cycle

![Improvement cycle diagram](Diagram%20Images/IMG_0038.jpg)

**What the diagram shows**

1. **Context** — The prompt bundles:
   - **Past kernels** with **summaries** from **`SUMMARY_AGENT`**: tags, notes, and a short recap. The default prompt uses summaries, not full C++ bodies; you can open a saved file in the UI to read the full code.
   - **Manager / user instructions** — What to try next, whether from chat, the REPL, or a worker prompt in a manager run.
   - **Guardrail feedback** — If a candidate fails injection checks, the next attempt includes the reasons and rejected snippets so the model can revise without discarding the rest of the context.

2. **`MAIN_KERNEL_AGENT`** — Produces a full **`KernelRevision`**: C++ plus an explanation. It may call **`research_algorithm_summary`** for web-backed notes at most once per user message when it needs concrete technical detail.

3. **Guardrail stack** before eval:
   - **Static checks** — Denylist-style patterns for obvious host-escape primitives in `rules.py` and the static pass inside `check_reward_hacking_cpp`.
   - **`GUARDRAIL_AGENT`** — Optional LLM pass for injection-style issues beyond the static list.
   - **Evaluator** — The offline harness: compile, numerical correctness against `task/reference.py`, then timing when both pass. This step is the task’s correctness and performance contract, not another injection pass.

4. **Outcomes** — On success you get a new kernel, timing fields, and text suitable for history or summarization. **Guardrail failures** feed back into the next attempt inside **`run_kernel_turn`** until success or **`max retries`**, which matches the diagram’s “summary + retry” loop for injection failures. **Eval failure** (compile or correctness) does not append `kernel_history` in the chat or REPL; you send a new message if you want another try. Inside the manager, a failed branch is simply recorded and does not retry automatically.

In code, guardrail retries live in **`run_kernel_turn`**. Manager workers use **`improvement_cycle.run_improvement_cycle`**, which runs the agent and then evaluates against an isolated candidate module.

### Manager workflow

![Manager workflow diagram](Diagram%20Images/IMG_0039.jpg)

**What the diagram shows**

1. **Context** — Seed kernels (session, on-disk working kernel, baseline, and/or saved revisions), the user’s goal, and the constraints baked into agent instructions: API shape, safety expectations, and how the evaluator behaves.

2. **`MANAGER_AGENT`** — Turns that goal into **`NUM_PARALLEL`** distinct worker prompts for one batch.

3. **Single run** — Each prompt runs a full improvement cycle—`MAIN_KERNEL_AGENT`, guardrails, then eval—with isolated builds so parallel runs do not overwrite `task/candidate.py`. The fastest compile-and-correct result in that batch, by **`candidate_time_ms`**, is merged into context for the next round together with seeds, subject to **`NUM_CAP`**.

4. **Repeat** — That batch is repeated **`NUM_RUNS`** times. A round can end with no successful branch; a short digest of failures can influence the next round’s prompts.

5. **Output** — When everything finishes, the fastest correct kernel over the whole session is written to **`kernel_history/`** once, and `task/` can be updated from that result. Details appear under **Manager session** and **Storing revisions and “best” kernels**.

### Why this structure?

The improvement path is split so responsibilities stay clear:

- **`improvement_cycle.py`** and **`MAIN_KERNEL_AGENT`** own one turn: prompt in, structured kernel out, then guardrails and eval. Chat, REPL, and manager workers all call that path.
- **`manager_run.py`** and **`MANAGER_AGENT`** schedule work only: prompt expansion, parallelism, merging winners, trimming context. They do not embed GEMM knowledge themselves.

Context is shared in three ways: **summaries** for long-running memory in chat, **multi-kernel stacks** for manager turns with a fixed **`NUM_CAP`**, and **failure digests** between manager rounds so later prompts are not copies of the first batch. The intent is to cap prompt size while still carrying forward what mattered.

## Kernel improver workflow (single chat / REPL turn)

The REPL in **`main.py`** and the Kernel Agent Chat tab in **`UI/app.py`** follow the same steps: generate, guardrails, write **`task/candidate.py`**, evaluate, then optionally write **`kernel_history/`** and maybe promote. The only behavioral difference is how prior **summaries** are chosen; see **Summaries and `kernel_history/`**.

1. **Working kernel** — Start from **`task/best_kernel.py`** if it exists, otherwise **`task/base_kernel.py`** via `load_working_cpp()` in `main.py`. A successful turn updates the current C++ for the next UI message or REPL input.
2. **Prior context** — Kernel summaries from earlier runs are injected so the agent sees what was tried. The REPL uses **`get_top_k_summary_context(k)`** on the **K** newest saved files, default **K = 3**. The Streamlit app uses **`get_summary_context_for_filenames`** on whichever files you select under Chat memory.
3. **Generation** — **`MAIN_KERNEL_AGENT`** returns a **`KernelRevision`**: full `cpp_code` for `CPP_SOURCE` and an `explanation`.
4. **Guardrails** — Before eval, C++ is scanned for injection and host-escape patterns, not for banning particular GEMM styles. Failures retry with structured feedback up to **`max_retries`** from **`KERNEL_GUARD_MAX_RETRIES`** or the UI global guardrail control. Too many failures raise **`GuardrailRetriesExhausted`**; the UI opens a dedicated tab for the trace.
5. **Staging** — Accepted C++ is written to **`task/candidate.py`**.
6. **Evaluator** — **`evaluate_candidate_kernel_sync()`** runs **`sandbox_eval.evaluate_sources`** on **`task/reference.py`** and **`task/candidate.py`**: compile, multi-trial correctness against the reference model, then benchmark. `main.py` adds naive-baseline fields for display.
7. **Persistence** — Compile- and correctness-passing runs can be saved under **`kernel_history/`** with timing metadata. Promotion to **`task/best_kernel.py`** follows the naive-baseline rule in **Evaluator and promotion choices**, not the raw **`speedup`** vs `torch.matmul`.
8. **Optional eval skip** — **`KERNEL_SKIP_EVAL=1`** skips evaluation for local development. **`KERNEL_SAVE_WITHOUT_EVAL=1`** allows saving history without a successful eval when eval is skipped.

## Guardrails

**Goal:** Block injection-style host escape in generated C++, not to police whether you use a particular GEMM API. Legitimate optimization code is left to the evaluator for correctness and speed.

**Implementation:**

- **`agents/GUARDRAIL_AGENT/rules.py`** — Static denylist for high-confidence escape primitives such as `system(`, `popen(`, `exec*`, `dlopen(`. There is no matmul substring ban.
- **`GUARDRAIL_AGENT`** — Optional small LLM that reads the full translation unit for semantic injection. Enabled by default; **`KERNEL_GUARDRAIL_USE_LLM=0`** restricts to static checks. **`KERNEL_GUARDRAIL_MODEL`** sets the reviewer model if needed; default is `gpt-4o-mini`.
- **`check_reward_hacking_cpp`** — Historical name; runs static checks first, then the LLM when enabled.
- **`run_kernel_turn`** — Generates, runs guardrails, and on failure appends structured feedback to the next attempt until pass or **`max_retries`**.

Guardrails run before the evaluator. They are a security-oriented filter; the evaluator remains the source of truth for compile, correctness, and timing.

## Manager session (autonomous multi-run)

The Manager entry point in the UI calls **`manager_run.run_manager_session`** in `manager_run.py`. Each worker runs **`improvement_cycle.run_improvement_cycle`**: **`MAIN_KERNEL_AGENT`**, injection guardrails, then evaluation in an isolated build. Nothing is written to **`kernel_history/`** until the full manager session completes.

**Purpose:** Run **`NUM_PARALLEL`** independent prompts per round for **`NUM_RUNS`** rounds, while keeping a **context stack** of C++ bodies: seeds plus the best result from each round, ordered by **`candidate_time_ms`**, truncated to **`NUM_CAP`**. **`MANAGER_AGENT`** in `agents/MANAGER_AGENT.py` expands one user goal into **`NUM_PARALLEL`** prompts per round. A short summary of failures can shape the next round’s prompts.

**Mechanics:**

- **Isolated evaluation** — Parallel workers cannot share **`task/candidate.py`**. **`evaluate_candidate_cpp_isolated(cpp, isolate_key)`** in `main.py` builds an in-memory module shaped like **`task/candidate.py`** with a unique **`load_inline`** name and build directory under **`build/eval_iso/`**, then calls **`evaluate_sources`** the same way as the normal path. Each isolate key gets its own build artifacts.
- **Multi-kernel context** — With **`manager_context_kernels`**, **`MAIN_KERNEL_AGENT`** sees extra kernels as reference; the body to replace is the fastest in the stack, repeated in the **Current CPP_SOURCE** block of the prompt.
- **Multi-seed** — Seeds can come from the session, the on-disk working kernel, baseline, or saved files. Each seed is timed once in isolation; seeds and per-round winners are merged, deduplicated, sorted fastest-first, then cut to **`NUM_CAP`**.
- **One history file per manager session** — The fastest compile-and-correct kernel across all branches and all rounds is saved once under **`kernel_history/`**. It may be slower than a seed; if no branch passes eval, nothing is saved. **`apply_generation_to_task_files`** then applies that kernel to **`task/`** so the tree matches the saved revision.

Optional **`MANAGER_AGENT_MODEL`** selects the planner model; if unset, it follows **`KERNEL_AGENT_MODEL`**.

## Summaries and `kernel_history/`

Saved revisions can include a **kernel summary** produced by **`SUMMARY_AGENT`**: tags, notes, and a short description embedded in the markdown. Files also store timing metadata such as **`candidate_time_ms`**, **`speedup`**, **`naive_baseline_ms`**, **`speedup_vs_naive`**, and **`is_best`** where applicable so the UI can sort and compare without re-running eval.

The agent’s prior context uses **summary text** by default. **`get_summary_context_for_filenames`** in `main.py` loads or generates summaries for the `kernel_history/*.md` names you select in Chat memory. Selected files are ordered **newest first** by modification time. The REPL uses **`get_top_k_summary_context(k)`** on the **K** newest files by filename.

### Storing revisions and “best” kernels

- **`kernel_history/*.md`** holds revisions that **passed compile and correctness**, except when you use the skip-eval development flags. Each file has metadata, an explanation, a C++ block, and optionally a **Kernel Summary** JSON from **`SUMMARY_AGENT`**.
- **`is_best`** in the metadata means the revision was **promoted** to **`task/best_kernel.py`** under **`should_promote_to_best_kernel`**: faster than the profiled naive baseline in **`task/base_kernel.py`**. It does not mean “fastest kernel ever stored,” only that this save promoted the working file. A run can be correct and still appear only in history if it is slower than that baseline.
- **`task/best_kernel.py`** is what **`load_working_cpp()`** reads by default. Chat, the REPL, and the final step after a manager session go through **`apply_generation_to_task_files`**, which promotes when the rule is satisfied. **`kernel_history/`** keeps a wider set of correct runs, including ones that did not promote, which is why Chat memory reads from history rather than only from **`best_kernel.py`**.
- **Manager sessions** add at most **one** new markdown file per session: the fastest correct kernel across every parallel branch and every round. **`is_best`** on that file uses the same promotion rule on its eval metrics.

## Evaluator and promotion choices

**Reference model** — Evaluation always compares against **`task/reference.py`**, which uses **`torch.matmul`**. That is the correctness and timing reference for “does this extension implement the same GEMM?”

**`speedup` vs promotion** — The evaluator reports **speedup** as `reference_time_ms / candidate_time_ms`. Because the reference is a highly optimized library call, naive C++ is often much slower than the reference, so **speedup** is often small. Using **speedup > 1** as the promotion rule would effectively require beating **`torch.matmul`**, which is not the goal of this exercise.

**Promotion rule** — We profiled the naive implementation in **`task/base_kernel.py`** once and store **`NAIVE_BASELINE_TIME_MS`** in `main.py`. **`task/best_kernel.py`** is updated when compile and correctness pass and **`candidate_time_ms`** is **strictly below** that naive baseline. The bar is “faster than our deliberate slow baseline,” not “faster than PyTorch’s matmul.”

**Extra fields** — After each eval, `main.py` adds **`naive_baseline_ms`** and **`speedup_vs_naive`** for dashboards and history.

**Updating the baseline constant** — If you change problem size **`N`**, compiler flags, or benchmark settings, re-profile **`base_kernel.py`** with the same **`evaluate_sources`** call and replace **`NAIVE_BASELINE_TIME_MS`** accordingly.

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

The UI uses the same agent, guardrails, and evaluator as the REPL. The sidebar groups global settings, single-turn chat options, and manager options separately.

### Session

- Reset to baseline; open the current working kernel in a tab, which is the same file **`load_working_cpp()`** uses.

### Global settings

- **Guardrail max retries** applies to both chat and manager runs. Override with **`KERNEL_GUARD_MAX_RETRIES`**; the LLM guardrail reviewer is controlled by **`KERNEL_GUARDRAIL_USE_LLM`**.

### Single kernel implementation cycle (chat)

- **Chat memory** — The **Choose memory…** control lists saved **`kernel_history`** files. You can select any number; their summaries are injected on the next turns. There is no fixed “top K” slider: you choose the files explicitly. An empty selection means only the current kernel body is in context, with no extra summaries.

### Manager session

- **`NUM_CAP`**: maximum C++ bodies in the manager context stack after sorting and truncation.
- **`NUM_PARALLEL`**: parallel improvement cycles per round, each with its own isolated eval.
- **`NUM_RUNS`**: how many rounds to run.
- **Choose seed kernels…** — Same pattern as chat memory: multi-select session kernel, on-disk working kernel, baseline, and/or saved revisions. Seeds are timed once, then combined with per-round winners.

### Saved kernels (sidebar list)

- Lists saved files with timing in the labels; sort by recency or by fastest candidate time.
- A **star** marker in the list means that file is selected for **chat memory** for the next message.
- The **working kernel** row opens the on-disk best or baseline file.
- File metadata can indicate when a revision matches **`task/best_kernel.py`**.

### Tabs

- **Kernel Agent Chat** — Single-turn loop described earlier.
- **Past kernels** — List with actions to open a detail tab or load a kernel into the session.
- **Manager** — Goal field, run control, progress, and session log.
- **Manager log** — Full text log for long runs.

Completed manager results also render **below the tab row** so they stay visible if you switch tabs; see the implementation in **`UI/app.py`** if you need the exact behavior.

## Folder layout

- `Diagram Images/` — README figures: improvement cycle (`IMG_0038.jpg`), manager workflow (`IMG_0039.jpg`)
- `run_eval.py` — Standalone CLI around `evaluate_sources`
- `main.py` — REPL (`python main.py`); eval + `kernel_history` + promotion helpers; **`evaluate_candidate_cpp_isolated`** / **`build_candidate_module_source`** for manager-safe parallel eval
- `improvement_cycle.py` — One cycle: `run_kernel_turn` → isolated eval result types
- `manager_run.py` — Manager session orchestration (`NUM_RUNS`, `NUM_PARALLEL`, capped context, single history save)
- `UI/app.py` — Streamlit front end
- `agents/` — `MAIN_KERNEL_AGENT`, `MANAGER_AGENT`, `GUARDRAIL_AGENT`, `SUMMARY_AGENT`
- `sandbox_eval/` — Evaluator package
- `kernel_history/` — Saved markdown revisions (gitignored by default if configured)
- `task/reference.py` — Reference `torch.matmul` model (correctness + library timing)
- `task/base_kernel.py` — Naive C++ baseline; defines the profiled promotion threshold
- `task/candidate.py` — Latest proposed kernel from the **chat** path (and final apply after manager); **not** shared concurrently with parallel manager branches (those use in-memory candidate sources)
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

Watch for reward hacking: models may try to call **`torch.matmul`**, NumPy, or BLAS-backed paths instead of keeping work in the extension, or may attempt to alter the harness. The intended outcome is a better low-level implementation in C++, not a shortcut around the task.

## Setup

Do this from the **repository root** (`sandbox_interview/`, where `requirements.txt` and `main.py` live). You need **Python 3.12+** (see **Minimum Requirements** below for C++ toolchain notes).

### 1. Create a virtual environment

**macOS / Linux (bash or zsh)**

```bash
cd /path/to/sandbox_interview
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your shell prompt. Use `deactivate` later to leave the environment.

**Windows (PowerShell)**

```powershell
cd C:\path\to\sandbox_interview
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks activation, run once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then try again.

**Windows (Command Prompt)**

```cmd
cd C:\path\to\sandbox_interview
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Upgrade pip (recommended)

```bash
python -m pip install --upgrade pip
```

(Same command in PowerShell or `cmd` after activation.)

### 3. Install PyTorch (CPU)

Install a **CPU** build suitable for laptops and for the evaluator:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use the same line on **macOS, Windows, and Linux** once the virtual environment is active.

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

This installs **`openai-agents`**, **Streamlit**, **pydantic**, **python-dotenv**, and the rest of **`requirements.txt`**, including **`numpy`**, **`psutil`**, and **`ninja`** for the evaluator.

### 5. API keys (agents)

Create a **`.env`** file in the repository root. It should stay out of version control. Put your OpenAI API key and any model overrides there, for example:

```env
OPENAI_API_KEY=sk-...
```

**`main.py`** and the UI load it through **`python-dotenv`**.

### 6. Start the Streamlit app

With the venv **activated** and your current directory still the repo root:

```bash
streamlit run UI/app.py
```

The browser should open to the app. Stop the server with **Ctrl+C** in the terminal.

### 7. Optional: interactive REPL

```bash
python main.py
```

This runs the interactive kernel improver in the terminal. Summary context uses the **top-K** newest files, as described under **Summaries and `kernel_history/`**.

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

## Suggested deliverables

- The sandbox or agent setup you used
- The improved C++ implementation
- A short write-up of workflow and design decisions; this README is one example of that style

The interview emphasizes safe setup and clear reasoning, not raw benchmark scores.
