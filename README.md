# Sandbox Interview

This directory is a standalone take-home interview focused on sandbox and agent setup for local C++ matmul optimization.

## What this README is for

This document describes **what we implemented**: how the **single-turn** kernel improver loop works end-to-end, how the optional **manager** (multi-run, parallel) session layers on top, **what guardrails exist and why**, how the **offline evaluator** is wired (including **isolated** eval for parallel branches), how **`kernel_history/`** and promotion to **`best_kernel.py`** behave, and what the **Streamlit UI** adds on top of the REPL. **Architecture diagrams** (`Diagram Images/IMG_0038.jpg`, `IMG_0039.jpg`) illustrate the **improvement cycle** and **manager workflow**. It is meant to explain **workflow and design choices**, not to optimize for the single fastest kernel. The interview cares about setup, safety, and reasoning—not peak benchmark scores.

## Goal (interview brief)

Set up an agent or sandboxed workflow that can take the provided C++ matmul task, iterate on the candidate implementation, and improve it while preserving correctness.

## What we want to learn

We are evaluating how well a candidate can:

- set up an isolated environment for code generation or code editing
- wire an agent into that environment safely
- use an evaluator to catch compile and correctness regressions
- iterate toward a faster low-level implementation without reward hacking

## Architecture diagrams

The system is split into two ideas: a **single improvement cycle** (one generate → guardrail → eval attempt, with retries) and a **manager workflow** (many cycles in parallel, many rounds, one best result at the end). Diagrams are shown **first**, then explained.

### Improvement cycle

![Improvement cycle — context, MAIN_KERNEL_AGENT, guardrails, output or retry](Diagram%20Images/IMG_0038.jpg)

**What the diagram shows**

1. **Context** — The prompt bundles:
   - **Past kernels** with **summaries** from **`SUMMARY_AGENT`** (structured tags, notes, high-level recap—not necessarily full C++ in the prompt unless you open a file).
   - **Manager / user instructions** — What to try next (from chat, REPL, or a worker prompt inside a manager run).
   - **Guardrail feedback** — When a candidate fails the injection checks, the next attempt receives the reasons and rejected snippets so the model can fix security issues without starting from scratch.

2. **`MAIN_KERNEL_AGENT`** — Produces a full **`KernelRevision`** (C++ + explanation). It may call **`research_algorithm_summary`** (web search) **at most once** per user message when it needs concrete technical grounding.

3. **Guardrail stack** (before trusting eval):
   - **Static checks** — Fast denylist-style patterns for obvious host-escape primitives (`rules.py` + static pass in `check_reward_hacking_cpp`).
   - **`GUARDRAIL_AGENT`** — Optional LLM review for injection / edge cases beyond the static list.
   - **Evaluator (correctness)** — The offline harness: compile, numerical correctness vs `task/reference.py`, then timing when both pass. This is **not** “more injection rules”; it is the task’s correctness and performance contract.

4. **Outcomes** — **Pass** → you get a **new kernel**, profiling fields, and text suitable for history/summary. **Fail guardrails** → the rejection reasons and snippets are **fed back into the next attempt** inside **`run_kernel_turn`** until **max retries** or success (this matches the diagram’s “Triggered = Summary + Retry” loop for injection failures). **Fail eval** (compile or correctness) → the candidate is **not** saved to history; the chat/REPL does **not** auto-retry the same user message—you issue a follow-up—while an improvement cycle inside the manager simply records that branch as failed.

In code, the guardrail loop is **`run_kernel_turn`**; full cycles with isolated eval are **`improvement_cycle.run_improvement_cycle`** (manager workers).

### Manager workflow

![Manager workflow — context, MANAGER_AGENT, parallel instruction cycles, NUM_RUNS, output](Diagram%20Images/IMG_0039.jpg)

**What the diagram shows**

1. **Context** — **Seed kernel(s)** (session, on-disk, baseline, and/or saved revisions), the **user goal** (high-level idea), and **guidelines** encoded in agent instructions (safety, API contract, evaluator behavior).

2. **`MANAGER_AGENT`** — Expands the goal into **`NUM_PARALLEL`** distinct **worker prompts** for a single “single run” batch.

3. **Single run** — Each prompt runs a full **improvement cycle** (MAIN_KERNEL_AGENT → guardrails → eval) **in parallel** with isolated builds. The **best** compile+correct result in that batch (by measured **`candidate_time_ms`**) is **merged into context** for the next round (along with seeds, capped by **`NUM_CAP`**).

4. **Repeat** — That **single-run** block is repeated **`NUM_RUNS`** times. Failures still consume a run; digest text can inform the next batch’s prompts.

5. **Output** — After all runs, the session’s **globally fastest** correct kernel is written to **`kernel_history/`** once (plus optional apply to `task/`), with profiling and explanation—see **Manager session** and **Storing revisions and “best” kernels** below.

### Why this structure?

We kept the design **compartmentalized** so each piece has a clear contract:

- **Improvement cycle** — One place (`improvement_cycle.py` + `MAIN_KERNEL_AGENT`) for “turn prompt → structured kernel → safety → eval,” reusable from chat, REPL, or manager workers.
- **Manager** — A thin orchestration layer (`manager_run.py`, `MANAGER_AGENT`) that does not reimplement GEMM logic; it only plans prompts, fans out parallel cycles, merges winners, and trims context.

**Shared context** is handled explicitly: **summaries** for narrative memory, **multi-kernel C++ stacks** for manager turns (fastest-first, **`NUM_CAP`**), and **failure digests** so later prompts stay diverse. That keeps prompts bounded while still passing forward what mattered from prior attempts.

## Kernel improver workflow (single chat / REPL turn)

**`main.py`** (REPL) and the **Kernel Agent Chat** tab in **`UI/app.py`** share the same _logical_ pipeline: generate → guardrails → write candidate → evaluate → optional `kernel_history` + promotion. Differences are only in **how prior summary context is chosen** (see **Summaries and `kernel_history/`**).

1. **Working kernel** — The session starts from `task/best_kernel.py` when it exists, otherwise `task/base_kernel.py` (`load_working_cpp()` in `main.py`). Each successful turn can update the in-memory “current” C++ for the next turn (UI) or the next REPL line (CLI).
2. **Prior context (summaries)** — Structured **kernel summaries** (tags, notes, high-level description) from selected past runs are injected so the agent can see what was tried before. **REPL:** `get_top_k_summary_context(k)` uses the **K newest** saved files (default `k=3`). **Streamlit:** you pick explicit files via **Chat memory** (see **Streamlit UI**); no fixed “top K.”
3. **Generation** — **`MAIN_KERNEL_AGENT`** (`agents/MAIN_KERNEL_AGENT.py`) returns a structured **`KernelRevision`**: full `cpp_code` for `CPP_SOURCE` plus an `explanation`.
4. **Guardrails** — Before any eval, generated C++ is checked for **injection / host-escape** patterns (see **Guardrails** below), not for “disallowed” GEMM APIs. Failed checks trigger retries with the rejection reason and snippet fed back into the prompt, up to **`max_retries`** (env `KERNEL_GUARD_MAX_RETRIES` or the UI **Global settings** control). Exhaustion raises **`GuardrailRetriesExhausted`** (UI surfaces this in a dedicated tab).
5. **Staging** — Passing C++ is written to **`task/candidate.py`**.
6. **Evaluator** — **`evaluate_candidate_kernel_sync()`** runs **`sandbox_eval.evaluate_sources`** with **`task/reference.py`** vs **`task/candidate.py`**: compile, multi-trial correctness vs the reference model, then benchmark. Results are **enriched** in `main.py` with naive-baseline fields (see below).
7. **Persistence** — If compile and correctness pass, a markdown artifact may be written under **`kernel_history/`** (with timing metadata). Promotion to **`task/best_kernel.py`** uses a **separate rule** from the evaluator’s `speedup` vs `torch.matmul` (see **Evaluator and promotion choices**).
8. **Optional eval skip** — For local development, **`KERNEL_SKIP_EVAL=1`** skips the evaluator; **`KERNEL_SAVE_WITHOUT_EVAL=1`** allows appending history without eval.

## Guardrails

**Intent:** Catch **injection attacks** and **sandbox / host escape** in generated C++—not to enforce a “no BLAS / no `torch::matmul`” policy. Normal optimization code (any legitimate API or algorithm) is allowed; the **evaluator** decides correctness and performance vs the reference.

**Mechanics:**

- **`agents/GUARDRAIL_AGENT/rules.py`** — Small static denylist for **high-confidence** escape primitives (e.g. `system(`, `popen(`, `exec*`, `dlopen(`). No framework matmul / BLAS substring bans.
- **`GUARDRAIL_AGENT` (LLM)** — Separate small model reviews the full translation unit for semantic injection (shell escape, suspicious dynamic loading, unambiguous instruction-injection text). **On by default**; set **`KERNEL_GUARDRAIL_USE_LLM=0`** to use only static checks. Optional **`KERNEL_GUARDRAIL_MODEL`** overrides the reviewer model (default `gpt-4o-mini`).
- **`check_reward_hacking_cpp`** (name kept for imports) — Static pass first, then LLM when enabled.
- **`run_kernel_turn`** — Loops: generate → guardrail → on failure append structured **guardrail feedback** to the next message until pass or max retries.

Independent of the **evaluator**: guardrails are a **security-style** pre-filter; the evaluator checks compile/correctness/speed against the harness.

## Manager session (autonomous multi-run)

Separate from the single chat turn, the **Manager** flow in the UI runs **`manager_run.run_manager_session`** (`manager_run.py`). It orchestrates **improvement cycles** (`improvement_cycle.py`): each cycle is **`MAIN_KERNEL_AGENT` → injection guardrails → isolated evaluator**, with **no** write to `kernel_history/` until the **entire** manager session finishes.

**Why it exists:** explore many directions in parallel ( **`NUM_PARALLEL`** prompts per round) across several rounds (**`NUM_RUNS`**), while growing a **context stack** of C++ bodies (seeds + per-round winners), sorted by fastest measured **`candidate_time_ms`**, capped at **`NUM_CAP`**. The **MANAGER_AGENT** (`agents/MANAGER_AGENT.py`) turns one user “goal” into **`NUM_PARALLEL`** distinct worker prompts each round; failures from a round are summarized for the next round’s prompt diversity.

**Design choices:**

- **Isolated evaluation** — Parallel branches cannot all write `task/candidate.py` at once. **`evaluate_candidate_cpp_isolated(cpp, isolate_key)`** in `main.py` builds a **`task/candidate.py`-shaped** module string with a **unique** `load_inline` name and build directory under `build/eval_iso/…`, then calls the same **`evaluate_sources`** contract as the normal path. Each branch gets a unique key so extension builds do not clobber each other.
- **Multi-kernel context** — When **`manager_context_kernels`** is set, **`MAIN_KERNEL_AGENT`** lists additional kernels as **reference-only**; the **primary** CPP to replace is the fastest in the stack (see **Current CPP_SOURCE** in the built prompt).
- **Multi-seed** — You can select several seeds (session, on-disk working, baseline, and/or saved files). Each seed is evaluated in isolation once for timing; seeds and run winners are merged, **deduped**, sorted fastest-first, then **`NUM_CAP`** is applied.
- **One `kernel_history` append per manager session** — Across all runs and all parallel successes, we persist **only the single fastest** compile+correct kernel (if any). It is OK if it is slower than a seed; if **nothing** passes compile+correctness, **nothing** is saved. After save, **`apply_generation_to_task_files`** applies that best kernel to `task/` so the workspace matches the recorded winner.

**Env:** optional **`MANAGER_AGENT_MODEL`** for the planner; defaults track **`KERNEL_AGENT_MODEL`**.

## Summaries and `kernel_history/`

After a revision is saved, a **kernel summary** (tags, notes, high-level description) can be produced and embedded in the markdown file via **`SUMMARY_AGENT`**. Saved files record metadata such as candidate/reference times, evaluator `speedup`, **`naive_baseline_ms`**, **`speedup_vs_naive`**, and **`is_best`** when relevant—so the UI can sort, label, and compare runs without re-running eval.

**Prior context for the agent is summary text, not full C++** (unless you open a tab and read the file). **`get_summary_context_for_filenames`** (`main.py`) loads or generates summaries for **user-selected** `kernel_history/*.md` names. In the UI, **Chat memory** controls which files are included; among those picks, summaries are ordered **newest file first** (by filesystem mtime) so the prompt order stays stable and aligned with “recent first” semantics. The **REPL** still uses **`get_top_k_summary_context(k)`** (newest-first by filename) for a fixed **K** for non-UI use.

### Storing revisions and “best” kernels

- **`kernel_history/*.md`** is the **append-only style archive** of revisions that **passed compile + correctness** (unless you use the dev escape hatches for skip-eval saving). Each file is human-readable: metadata lines, explanation, C++ block, then an optional **Kernel Summary** JSON from **`SUMMARY_AGENT`**.
- **`is_best` in metadata** — Set when the revision was **promoted** under our rule: faster than the profiled naive baseline in **`task/base_kernel.py`** (`should_promote_to_best_kernel`). It does **not** mean “fastest in all of history”; it means “this run earned **`task/best_kernel.py`** at save time.” Revisions can be correct and saved to history but **not** promoted if they are still slower than that naive baseline.
- **`task/best_kernel.py`** — The **live** promoted kernel used as the default **working** kernel (`load_working_cpp()`). Only the **chat/REPL path** and the **final apply after a manager session** write through **`apply_generation_to_task_files`**, which promotes when the bar is met. **`kernel_history/`** keeps a **broader record** of good runs (including correct-but-not-promoted kernels), which is why history is the right place to pick **chat memory** and compare runs over time.
- **Manager sessions** — Append **at most one** new markdown file per session: the **single fastest** correct kernel across **all** parallel branches and **all** **`NUM_RUNS`** (see **Manager session**). That file’s **`is_best`** flag still follows the same promotion rule on the global best’s eval metrics.

## Evaluator and promotion choices

**Correctness and timing** — The evaluator always uses **`task/reference.py`** (`torch.matmul`) as the numerical reference. That is the right contract for “does this extension match the reference GEMM?”

**Why `speedup` ≠ promotion rule** — The same run reports **`speedup` = reference_time_ms / candidate_time_ms** (library vs your kernel). With `torch.matmul` as the reference, naive C++ is vastly _slower_, so this ratio is often near zero. Using **`speedup > 1`** as “promote to best” would mean “beat `torch.matmul`,” which is the wrong bar for this task.

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

The UI uses the same **`MAIN_KERNEL_AGENT`**, guardrails, and evaluator path for **Kernel Agent Chat** as the REPL, but organizes controls so **global**, **single-turn**, and **manager** behavior are easy to tell apart.

### Session

- **Reset to baseline**; open the **current working kernel** in a tab (same file as `load_working_cpp()`).

### Global settings

- **Guardrail max retries** — Applies to **both** the chat pipeline and **manager** improvement cycles (still overridable via `KERNEL_GUARD_MAX_RETRIES`; LLM reviewer via `KERNEL_GUARDRAIL_USE_LLM`).

### Single kernel implementation cycle (chat)

- **Chat memory** — A **Choose memory…** popover lists **saved** `kernel_history` files. Pick any number; their **summaries** are injected into the next chat turns. There is no separate “K” slider: you explicitly choose which past runs matter. Empty selection means **no** extra summaries (only the current kernel body in the prompt).
- This keeps **intent** clear: memory is opt-in and inspectable (filenames), not an opaque “last K files.”

### Manager session

- **`NUM_CAP`** — Maximum number of C++ bodies in the manager context stack (seeds + per-run winners), fastest-first after measurement, then truncated.
- **`NUM_PARALLEL`** — Parallel improvement cycles per manager round (each with isolated eval).
- **`NUM_RUNS`** — Number of manager rounds in one session.
- **Choose seed kernels…** — Same interaction pattern as chat memory: multi-select **session**, **on-disk working**, **baseline**, and/or **saved** revisions. Multiple seeds are timed once each, then merged into the stack with winners.

### Saved kernels (sidebar list)

- Browse history with timing in labels; sort by recency or fastest candidate time.
- **⭐** = file is selected for **chat memory** (summaries for the next message), not “newest K by default.”
- **✅** = working kernel row; **🏆** in file metadata = revision matches **`task/best_kernel.py`**.

### Tabs

- **Kernel Agent Chat** — Single-turn loop described above.
- **Past kernels** — Table-style list with open tab / use kernel actions.
- **Manager** — Goal text, run button, progress text (run index / phase), session log expander.
- **Manager log** — Full log buffer for debugging long runs.

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

Word of advice: make sure the agent does not reward-hack. Agents often try to fall back to library calls such as `torch.matmul`, NumPy matmul, or BLAS-backed wrappers. Other times, the agent will try to change the testing harness, be careful with that. The goal here is to generate an actually improved low-level implementation, not a library shortcut or some other reward hack.

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

This pulls in **`openai-agents`**, **Streamlit**, **pydantic**, **python-dotenv**, and the other packages listed in `requirements.txt` (including **`numpy`**, **`psutil`**, **`ninja`** for the evaluator).

### 5. API keys (agents)

Create a **`.env`** file in the repository root (it is gitignored) with your OpenAI API key and any model overrides your setup needs, for example:

```env
OPENAI_API_KEY=sk-...
```

The agents read this via **`python-dotenv`** in `main.py` / the UI.

### 6. Start the Streamlit app

With the venv **activated** and your current directory still the repo root:

```bash
streamlit run UI/app.py
```

Your browser should open to the Kernel Agent UI. Use **Ctrl+C** in the terminal to stop the server.

### 7. Optional: interactive REPL

```bash
python main.py
```

This runs the text **kernel improver loop** in the terminal (same core pipeline as the chat tab, with **top-K** summary context as documented above).

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
