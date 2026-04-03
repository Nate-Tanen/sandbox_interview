import asyncio
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from agents import Agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import manager_run

from main import (
    GuardrailRetriesExhausted,
    apply_generation_to_task_files,
    build_agent,
    cpp_matches_best_kernel,
    should_save_kernel_history,
    get_model_name,
    get_summary_context_for_filenames,
    init_env,
    list_saved_kernel_files,
    load_baseline_cpp,
    load_saved_kernel_revision,
    load_working_cpp,
    run_kernel_turn,
    save_kernel_revision_with_summary,
    working_kernel_task_relpath,
)

_DEFAULT_GUARD_RETRIES = 3
_DEFAULT_NUM_CAP = 4
_DEFAULT_NUM_PARALLEL = 2
_DEFAULT_NUM_RUNS = 3


def _guardrail_max_retries() -> int:
    return int(st.session_state.get("guardrail_max_retries", _DEFAULT_GUARD_RETRIES))


def _manager_seed_token_options() -> list[str]:
    return ["session", "working", "baseline"] + [
        f"file:{f}" for f in list_saved_kernel_files("recent")
    ]


def _label_manager_seed_token(t: str) -> str:
    if t == "session":
        return "Current session kernel (chat)"
    if t == "working":
        return f"On-disk working · `{working_kernel_task_relpath()}`"
    if t == "baseline":
        return "Baseline · `task/base_kernel.py`"
    if t.startswith("file:"):
        return f"Saved · `{t[5:]}`"
    return t


def resolve_manager_seed_cpps() -> list[str]:
    """Deduped C++ list from sidebar token selection (session, working, baseline, saved files)."""
    tokens: list[str] = list(st.session_state.get("manager_seed_tokens") or ["working"])
    out: list[str] = []
    for t in tokens:
        if t == "session":
            out.append(st.session_state.current_cpp)
        elif t == "working":
            out.append(load_working_cpp())
        elif t == "baseline":
            out.append(load_baseline_cpp())
        elif t.startswith("file:"):
            fn = t[5:].strip()
            out.append(load_saved_kernel_revision(fn).cpp_code)
    seen: set[str] = set()
    deduped: list[str] = []
    for cpp in out:
        key = cpp.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cpp)
    return deduped


def working_kernel_tab_label() -> str:
    return f"Current best kernel · {working_kernel_task_relpath()}"


@st.cache_resource
def get_agent(model_name: str) -> Agent:
    return build_agent(model_name)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_cpp" not in st.session_state:
        st.session_state.current_cpp = load_working_cpp()
    if "open_summary_tabs" not in st.session_state:
        st.session_state.open_summary_tabs = []
    if "rejection_tabs" not in st.session_state:
        st.session_state.rejection_tabs = []
    if "run_in_progress" not in st.session_state:
        st.session_state.run_in_progress = False
    if "open_working_kernel_tab" not in st.session_state:
        st.session_state.open_working_kernel_tab = False
    if "manager_log" not in st.session_state:
        st.session_state.manager_log = []
    if "manager_last_result" not in st.session_state:
        st.session_state.manager_last_result = None
    if "manager_run_in_progress" not in st.session_state:
        st.session_state.manager_run_in_progress = False
    if "chat_memory_files" not in st.session_state:
        st.session_state.chat_memory_files = []
    if "manager_seed_tokens" not in st.session_state:
        st.session_state.manager_seed_tokens = ["working"]


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return

        if msg.get("type") == "guardrail_error":
            st.warning(msg.get("summary", "Injection guardrail blocked this generation."))
            if msg.get("tab_label"):
                st.caption(f"Details are in the **{msg['tab_label']}** tab.")
            return

        st.markdown("**Explanation**")
        st.write(msg["explanation"])
        if msg.get("web_research_used"):
            st.caption(
                "Internal **web research** ran once this turn (agent-decided). "
                "Not controlled by your prompt; at most one search per message."
            )
        with st.expander("Generated C++", expanded=True):
            st.code(msg["cpp_code"], language="cpp")

        if msg.get("eval_error"):
            st.error(f"Evaluator error: {msg['eval_error']}")
            st.warning(
                "Not saved to `kernel_history/`. `task/candidate.py` was updated before the eval failure."
            )
        elif msg.get("eval_skipped"):
            if msg.get("saved_path"):
                st.caption(
                    "Evaluator skipped (`KERNEL_SKIP_EVAL`); saved to `kernel_history/` because "
                    "`KERNEL_SAVE_WITHOUT_EVAL` is set. `task/candidate.py` was updated."
                )
            else:
                st.caption(
                    "Evaluator skipped (`KERNEL_SKIP_EVAL`). `task/candidate.py` was updated. "
                    "Set `KERNEL_SAVE_WITHOUT_EVAL=1` to append to `kernel_history/` without eval."
                )
        elif msg.get("eval_result") is not None:
            er = msg["eval_result"]
            st.markdown("**Evaluator**")
            st.json(
                {
                    "compile_pass": er.get("compile_pass"),
                    "correct_pass": er.get("correct_pass"),
                    "reference_time_ms": er.get("reference_time_ms"),
                    "candidate_time_ms": er.get("candidate_time_ms"),
                    "speedup": er.get("speedup"),
                    "naive_baseline_ms": er.get("naive_baseline_ms"),
                    "speedup_vs_naive": er.get("speedup_vs_naive"),
                }
            )
            if er.get("compile_pass") and er.get("correct_pass"):
                if msg.get("promoted"):
                    st.success(
                        "Promoted to `task/best_kernel.py` (faster than profiled naive baseline in "
                        "`task/base_kernel.py`)."
                    )
                else:
                    st.info(
                        "Correct vs reference; not promoted to `best_kernel.py` (not faster than naive baseline). "
                        "Slower-but-correct runs are still saved to `kernel_history/`."
                    )
            else:
                st.warning("Compile and/or correctness failed — see flags above.")

        if msg.get("saved_path"):
            st.caption(f"Saved to `kernel_history/{msg['saved_path']}` (eval passed compile + correctness).")
        elif msg.get("history_saved") is False and not msg.get("eval_error"):
            if not msg.get("eval_skipped") and msg.get("eval_result") is not None:
                er = msg["eval_result"]
                if not (er.get("compile_pass") and er.get("correct_pass")):
                    st.warning(
                        "Not saved to `kernel_history/`: needs compile + correctness. "
                        "`task/candidate.py` has this attempt."
                    )


def _parse_optional_float_line(text: str, field: str) -> float | None:
    m = re.search(rf"- {re.escape(field)}:\s*([^\n]+)", text)
    if not m:
        return None
    s = m.group(1).strip()
    if s in ("null", "None", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_history_file(filename: str) -> dict:
    path = PROJECT_ROOT / "kernel_history" / filename
    text = path.read_text(encoding="utf-8")

    run_id_match = re.search(r"- run_id:\s*(.+)", text)
    created_match = re.search(r"- created_at:\s*(.+)", text)
    request_match = re.search(r"- user_request:\s*(.+)", text)
    explanation_match = re.search(r"## Explanation\s*(.*?)\s*## C\+\+ Code", text, re.DOTALL)
    cpp_match = re.search(r"## C\+\+ Code\s*```cpp\s*(.*?)\s*```", text, re.DOTALL)
    summary_match = re.search(r"## Kernel Summary\s*```json\s*(.*?)\s*```", text, re.DOTALL)

    summary_obj = None
    if summary_match:
        try:
            summary_obj = json.loads(summary_match.group(1))
        except json.JSONDecodeError:
            summary_obj = {"parse_error": "Invalid JSON summary block."}

    raw_user_request = request_match.group(1).strip() if request_match else ""
    user_request = raw_user_request
    if raw_user_request:
        try:
            # History files write this field via json.dumps(...), so decode when possible.
            user_request = json.loads(raw_user_request)
        except json.JSONDecodeError:
            user_request = raw_user_request.strip('"')

    ib = re.search(r"- is_best:\s*([^\n]+)", text)
    is_best_meta = False
    if ib:
        is_best_meta = ib.group(1).strip().lower() in ("true", "1")

    cpp_code = cpp_match.group(1).strip() if cpp_match else ""
    is_best = is_best_meta or (bool(cpp_code) and cpp_matches_best_kernel(cpp_code))

    return {
        "filename": filename,
        "run_id": (run_id_match.group(1).strip() if run_id_match else path.stem),
        "created_at": (created_match.group(1).strip() if created_match else ""),
        "user_request": user_request,
        "candidate_time_ms": _parse_optional_float_line(text, "candidate_time_ms"),
        "reference_time_ms": _parse_optional_float_line(text, "reference_time_ms"),
        "speedup": _parse_optional_float_line(text, "speedup"),
        "naive_baseline_ms": _parse_optional_float_line(text, "naive_baseline_ms"),
        "speedup_vs_naive": _parse_optional_float_line(text, "speedup_vs_naive"),
        "is_best": is_best,
        "explanation": (explanation_match.group(1).strip() if explanation_match else ""),
        "cpp_code": cpp_code,
        "summary": summary_obj,
    }


def _shorten(text: str, max_len: int = 34) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _format_display_time(created_at: str, filename_fallback: str = "") -> str:
    for src in (created_at, filename_fallback):
        if not src:
            continue
        try:
            dt = datetime.fromisoformat(src)
            return dt.strftime("%m/%d %H:%M")
        except ValueError:
            pass
        match = re.match(r"^(\d{8})-(\d{6})-", src)
        if match:
            return f"{match.group(1)[4:6]}/{match.group(1)[6:8]} {match.group(2)[0:2]}:{match.group(2)[2:4]}"
    return "--/-- --:--"


def kernel_display_meta(filename: str) -> dict:
    data = parse_history_file(filename)
    time_label = _format_display_time(data["created_at"] or "", filename)
    req = _shorten(data.get("user_request") or data.get("run_id") or filename)
    ms = data.get("candidate_time_ms")
    prof = f"{ms:.2f} ms" if ms is not None else "— ms"
    best_mark = "🏆 " if data.get("is_best") else ""
    label = f"{best_mark}{prof} · {time_label} · {req}"
    unique_suffix = (data.get("run_id") or filename)[-4:]
    tab_label = f"{label} · {unique_suffix}"
    tooltip = (
        f"File: {filename}\n"
        f"Candidate time (mean): {data.get('candidate_time_ms')}\n"
        f"Reference time (mean): {data.get('reference_time_ms')}\n"
        f"Speedup (ref/cand): {data.get('speedup')}\n"
        f"Naive baseline ms: {data.get('naive_baseline_ms')}\n"
        f"Speedup vs naive: {data.get('speedup_vs_naive')}\n"
        f"Matches best_kernel.py: {data.get('is_best', False)}\n"
        f"Run ID: {data.get('run_id', '')}\n"
        f"Created: {data.get('created_at', '')}\n"
        f"Request: {data.get('user_request', '')}"
    )
    return {"label": label, "tab_label": tab_label, "tooltip": tooltip}


def render_working_kernel_tab() -> None:
    rel = working_kernel_task_relpath()
    cpp = load_working_cpp()
    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown(f"### Current best kernel · `{rel}`")
    with c2:
        if st.button("Close tab", key="close-working-kernel-tab"):
            st.session_state.open_working_kernel_tab = False
            st.rerun()
    st.caption(
        "This file is the C++ starting point for the next agent turn (same as **Session → Open current best kernel** "
        "and the ✅ row in **Saved kernels**)."
    )
    st.code(cpp, language="cpp")


def render_summary_tab(filename: str) -> None:
    data = parse_history_file(filename)
    display = kernel_display_meta(filename)
    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown(f"### {display['label']}")
    with c2:
        if st.button("Close tab", key=f"close-tab-inside-{filename}"):
            st.session_state.open_summary_tabs = [f for f in st.session_state.open_summary_tabs if f != filename]
            st.rerun()
    st.caption(
        f"file: `{data['filename']}`  |  run_id: `{data['run_id']}`  |  created_at: `{data['created_at']}`"
    )
    ct = data.get("candidate_time_ms")
    rt = data.get("reference_time_ms")
    sp = data.get("speedup")
    nb = data.get("naive_baseline_ms")
    svn = data.get("speedup_vs_naive")
    prof_parts: list[str] = []
    if ct is not None:
        prof_parts.append(f"candidate **{ct:.2f} ms**")
    if rt is not None:
        prof_parts.append(f"library ref **{rt:.2f} ms**")
    if sp is not None:
        prof_parts.append(f"ref/cand **{sp:.4g}×**")
    if nb is not None:
        prof_parts.append(f"naive baseline **{nb:.2f} ms**")
    if svn is not None:
        prof_parts.append(f"vs naive **{svn:.4g}×**")
    if prof_parts:
        st.caption("**Profiling (eval):** " + " · ".join(prof_parts))
    else:
        st.caption("**Profiling:** not recorded for this file.")
    if data.get("is_best"):
        st.success("This revision matches **`task/best_kernel.py`** (current best promoted kernel).")
    if data["user_request"]:
        st.markdown(f"**User request:** {data['user_request']}")

    if data["summary"]:
        st.markdown("**Kernel Summary**")
        st.json(data["summary"])
    else:
        st.warning("No summary JSON found yet for this file.")

    st.markdown("**Explanation**")
    st.write(data["explanation"] or "(none)")
    with st.expander("C++ Code", expanded=True):
        st.code(data["cpp_code"], language="cpp")


def render_rejection_tab(rej: dict) -> None:
    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown("### Blocked kernel")
    with c2:
        if st.button("Close tab", key=f"dismiss-top-{rej['id']}"):
            st.session_state.rejection_tabs = [r for r in st.session_state.rejection_tabs if r["id"] != rej["id"]]
            st.rerun()
    st.error(f"**Last reason:** {rej['last_reason']}")
    st.caption(
        "Each attempt below was rejected by the **injection guardrail** (static host-escape patterns "
        "and/or GUARDRAIL_AGENT). This does not police normal GEMM APIs—fix injection issues or ambiguity."
    )

    raw_attempts = rej.get("attempts", [])
    attempts: list[dict] = []
    for idx, item in enumerate(raw_attempts, start=1):
        if isinstance(item, dict):
            attempt_no = item.get("attempt", idx)
            reason = item.get("reason", "")
            cpp = item.get("cpp_code") or item.get("rejected_cpp") or item.get("cpp") or ""
        elif isinstance(item, (tuple, list)):
            attempt_no = item[0] if len(item) > 0 else idx
            reason = item[1] if len(item) > 1 else ""
            cpp = item[2] if len(item) > 2 else ""
        else:
            attempt_no = idx
            reason = str(item)
            cpp = ""
        attempts.append({"attempt": attempt_no, "reason": reason, "cpp_code": cpp})

    if not attempts:
        st.warning("No blocked attempt payload was captured for this run.")

    for i, item in enumerate(attempts):
        attempt = item.get("attempt", "?")
        reason = item.get("reason", "")
        cpp = item.get("cpp_code", "")
        with st.expander(
            f"Attempt {attempt} — blocked",
            expanded=(i == len(attempts) - 1),
        ):
            st.markdown(f"**Why:** {reason}")
            if cpp:
                st.code(cpp, language="cpp")
            else:
                st.caption("No rejected kernel source was attached for this attempt.")

def render_past_kernels_tab() -> None:
    st.markdown("### Past kernels")
    sort_label = st.session_state.get("kernel_sort", "Most recent")
    sort_key = "recent" if sort_label == "Most recent" else "fastest"
    saved_files = list_saved_kernel_files(sort_key)
    mem = set(st.session_state.get("chat_memory_files") or [])
    st.caption(
        "Sort order matches the sidebar. Open a file in its own tab to inspect summary + code. "
        "⭐ marks kernels you selected for **chat memory** (summaries in the next chat message) — "
        "use **Single kernel implementation cycle → Choose memory** in the sidebar."
    )
    if not saved_files:
        st.info("No saved kernels yet.")
        return

    for filename in saved_files:
        cols = st.columns([4, 1, 1])
        display = kernel_display_meta(filename)
        with cols[0]:
            prefix = "⭐ " if filename in mem else ""
            st.markdown(f"{prefix}`{display['label']}`")
        with cols[1]:
            if st.button(
                "Open tab",
                key=f"past-open-{filename}",
                help=display["tooltip"],
                disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
            ):
                if filename not in st.session_state.open_summary_tabs:
                    st.session_state.open_summary_tabs.append(filename)
                st.rerun()
        with cols[2]:
            if st.button(
                "Use kernel",
                key=f"past-use-{filename}",
                disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
            ):
                saved = load_saved_kernel_revision(filename)
                st.session_state.current_cpp = saved.cpp_code
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "explanation": f"Loaded kernel from `{filename}`.\n\n{saved.explanation}",
                        "cpp_code": saved.cpp_code,
                        "saved_path": filename,
                    }
                )
                st.rerun()


def _render_manager_last_run_panel() -> None:
    """
    Always rendered *below* the tab strip so output survives tab switches.

    Streamlit may not execute inactive `st.tabs` panels on a rerun; if the user switches
    to Past kernels or a kernel preview tab, the Manager tab body would not run and would
    look empty. Session state still holds results — we mirror them here every run.
    """
    res = st.session_state.get("manager_last_result")
    log_lines = st.session_state.get("manager_log") or []
    if not res and not log_lines:
        return
    st.divider()
    st.markdown("### Last manager run")
    st.caption(
        "This block is **outside** the tab strip so it stays on screen when you open other tabs "
        "(e.g. Past kernels or a saved kernel preview) while a run finishes or afterward."
    )
    if res:
        st.success(
            f"Saved `{res.get('saved_kernel_history') or 'nothing'}` · "
            f"successful branches (session): {res.get('total_success_branches', 0)}"
        )
        if res.get("saved_kernel_history"):
            st.caption(
                "Reload **Past kernels** for the new file. `task/candidate.py` / `best_kernel.py` "
                "were updated from the global best."
            )
    with st.expander("Manager session log (last run)", expanded=False):
        if log_lines:
            st.code("\n".join(log_lines), language="text")
        else:
            st.caption("No log lines for the last session.")


def render_manager_tab(agent) -> None:
    st.markdown("### Autonomous manager session")
    st.caption(
        "Runs **NUM_RUNS** rounds. Each round asks the manager model for **NUM_PARALLEL** prompts, "
        "then runs that many improvement cycles in parallel (isolated eval). "
        "Context is **seed + prior run winners**, sorted fastest-first, capped at **NUM_CAP**. "
        "At the end, **one** entry is appended to `kernel_history/` — the single fastest correct kernel "
        "from the whole session (if any). "
        "**Results and log** also appear under **Last manager run** below the tabs so they stay visible "
        "when you switch to another tab."
    )
    idea = st.text_area(
        "Goal / idea for the manager",
        height=120,
        key="manager_user_idea",
        placeholder="e.g. Explore cache-friendly blocking and k-dimension ordering…",
    )
    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button(
            "Run manager session",
            type="primary",
            disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
            key="manager_run_btn",
        )
    with c2:
        if st.button("Clear manager log", key="manager_clear_log"):
            st.session_state.manager_log = []
            st.session_state.manager_last_result = None
            st.rerun()

    if run_btn:
        if not (idea or "").strip():
            st.warning("Enter a goal for the manager.")
        else:
            st.session_state.run_in_progress = True
            st.session_state.manager_run_in_progress = True
            st.session_state.manager_log = []
            st.session_state.manager_last_result = None

            seed_cpps = resolve_manager_seed_cpps()
            cap = int(st.session_state.get("manager_num_cap", _DEFAULT_NUM_CAP))
            npar = int(st.session_state.get("manager_num_parallel", _DEFAULT_NUM_PARALLEL))
            nruns = int(st.session_state.get("manager_num_runs", _DEFAULT_NUM_RUNS))
            gr = _guardrail_max_retries()

            progress = st.progress(0.0, text="Starting…")
            status = st.empty()

            def log_fn(line: str) -> None:
                st.session_state.manager_log.append(line)

            def on_progress(*, phase: str = "", run_idx: int = 0, num_runs: int = 1, detail: str = "") -> None:
                nr = max(int(num_runs), 1)
                ri = int(run_idx)
                if phase == "seed_eval":
                    p = 0.04
                    label = detail or "Evaluating seed kernel(s) (isolated) for ordering…"
                elif phase == "manager_llm":
                    base = (ri - 1) / nr if ri else 0.0
                    p = min(0.05 + base * 0.72, 0.77)
                    label = f"Run {ri}/{nr}: manager LLM — {detail}"
                elif phase == "parallel_cycles":
                    base = (ri - 1) / nr if ri else 0.0
                    p = min(0.12 + base * 0.72, 0.82)
                    label = f"Run {ri}/{nr}: parallel improvement + eval — {detail}"
                elif phase == "run_done":
                    p = min(0.18 + (ri / nr) * 0.7, 0.88)
                    label = f"Run {ri}/{nr}: round complete"
                elif phase == "save":
                    p = 0.93
                    label = "Saving global best to kernel_history…"
                elif phase == "done":
                    p = 1.0
                    label = "Manager session finished"
                else:
                    p, label = 0.1, f"{phase} {detail}"
                progress.progress(min(float(p), 1.0), text=label[:120])
                status.markdown(f"**{label}**")

            try:
                result = asyncio.run(
                    manager_run.run_manager_session(
                        seed_kernel_cpps=seed_cpps,
                        user_idea=idea.strip(),
                        num_parallel=npar,
                        num_runs=nruns,
                        num_cap=cap,
                        max_retries=gr,
                        kernel_agent=agent,
                        log=log_fn,
                        on_progress=on_progress,
                    )
                )
                st.session_state.manager_last_result = result
            except Exception as exc:
                st.session_state.manager_log.append(f"[error] {exc}")
                st.error(f"Manager session failed: {exc}")
            finally:
                st.session_state.run_in_progress = False
                st.session_state.manager_run_in_progress = False
            st.rerun()


def render_manager_log_tab() -> None:
    st.markdown("### Manager debug log")
    st.caption("Full stdout-style log from the last manager sessions this browser session.")
    lines = st.session_state.get("manager_log") or []
    if lines:
        st.code("\n".join(lines), language="text")
    else:
        st.info("Empty — run a manager session from the **Manager** tab.")


def main() -> None:
    init_env()
    st.set_page_config(page_title="Kernel Agent UI", page_icon="UI", layout="wide")
    st.title("Kernel Agent")
    st.caption(
        "Chat with the kernel optimization agent. Type **reset** in chat to restart from baseline. "
        "Optional **chat memory** (saved kernels’ summaries) is chosen in the sidebar; **guardrail retries** "
        "are under global settings. The model may run **web research** at most once per reply (its own choice)."
    )

    model_name = get_model_name()
    agent = get_agent(model_name)
    init_state()

    with st.sidebar:
        st.subheader("Session")
        st.write(f"Model: `{model_name}`")
        if st.button(
            "Reset to baseline",
            use_container_width=True,
            disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
        ):
            st.session_state.current_cpp = load_baseline_cpp()
            st.session_state.messages = []
            st.session_state.open_summary_tabs = []
            st.session_state.rejection_tabs = []
            st.session_state.open_working_kernel_tab = False
            st.session_state.run_in_progress = False
            st.session_state.manager_run_in_progress = False
            st.rerun()
        wrel = working_kernel_task_relpath()
        if st.button(
            f"Open current best kernel · `{wrel}`",
            key="open-working-kernel-preview",
            use_container_width=True,
            disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
        ):
            st.session_state.open_working_kernel_tab = True
            st.rerun()

        st.divider()
        st.subheader("Global settings")
        st.number_input(
            "Guardrail max retries",
            min_value=1,
            max_value=20,
            value=_DEFAULT_GUARD_RETRIES,
            key="guardrail_max_retries",
            help="Applies to **Kernel Agent Chat** and **Manager** runs. Regenerations when the injection guardrail (static + GUARDRAIL_AGENT) blocks output.",
        )

        st.divider()
        st.subheader("Single kernel implementation cycle")
        st.caption(
            "Optional **prior summaries** from `kernel_history/` — not the full C++ unless you open a tab."
        )
        n_mem = len(st.session_state.get("chat_memory_files") or [])
        st.markdown(
            f"**Chat memory:** {n_mem} saved kernel(s) selected"
            if n_mem
            else "**Chat memory:** none (only the current kernel in chat)"
        )
        with st.popover("Choose memory…", use_container_width=True):
            st.markdown("##### Kernel memory for chat")
            st.caption(
                "Pick any number of saved kernels. Their **summaries** are added to the next agent turn "
                "(order follows filename sort: **newest saved files first** among your picks)."
            )
            mem_files = list_saved_kernel_files("recent")
            valid_cur = [f for f in (st.session_state.get("chat_memory_files") or []) if f in mem_files]
            pick = st.multiselect(
                "kernel_history",
                options=mem_files,
                default=valid_cur,
                disabled=not mem_files,
                label_visibility="collapsed",
            )
            if not mem_files:
                st.info("No saved kernels yet — run chat once with a successful eval, or use Manager.")
            if st.button("Apply selection", type="primary", key="apply_chat_memory"):
                st.session_state.chat_memory_files = pick
                st.rerun()

        st.divider()
        st.subheader("Manager session")
        st.caption("Autonomous multi-run optimizer — see **Manager** tab.")
        st.number_input(
            "NUM_CAP",
            min_value=1,
            max_value=32,
            value=_DEFAULT_NUM_CAP,
            key="manager_num_cap",
            help="Max kernels in manager context: seed + run winners, fastest first, then capped.",
        )
        st.number_input(
            "NUM_PARALLEL",
            min_value=1,
            max_value=16,
            value=_DEFAULT_NUM_PARALLEL,
            key="manager_num_parallel",
            help="Parallel improvement cycles per run (each gets an isolated evaluator build).",
        )
        st.number_input(
            "NUM_RUNS",
            min_value=1,
            max_value=50,
            value=_DEFAULT_NUM_RUNS,
            key="manager_num_runs",
            help="How many manager rounds in one session.",
        )
        n_tok = len(st.session_state.get("manager_seed_tokens") or [])
        st.markdown(f"**Seed kernels:** {n_tok} source(s) selected")
        with st.popover("Choose seed kernels…", use_container_width=True):
            st.markdown("##### Seeds for manager context")
            st.caption(
                "Each seed is evaluated once for timing, then combined with per-run winners (fastest first, "
                "capped by NUM_CAP). Include session, on-disk working, baseline, and/or any saved revision."
            )
            opt_keys = _manager_seed_token_options()
            cur = [x for x in (st.session_state.get("manager_seed_tokens") or ["working"]) if x in opt_keys]
            mgr_pick = st.multiselect(
                "Sources",
                options=opt_keys,
                default=cur,
                format_func=_label_manager_seed_token,
                label_visibility="collapsed",
            )
            if st.button("Apply seeds", type="primary", key="apply_manager_seeds"):
                if not mgr_pick:
                    st.warning("Select at least one seed.")
                else:
                    st.session_state.manager_seed_tokens = mgr_pick
                    st.rerun()

        st.divider()
        st.subheader("Saved kernels")
        st.selectbox(
            "Sort",
            ["Most recent", "Fastest (candidate time)"],
            key="kernel_sort",
            help="Fastest sorts by mean candidate runtime from the last evaluator run (lower is better).",
        )
        sort_label = st.session_state.get("kernel_sort", "Most recent")
        sort_key = "recent" if sort_label == "Most recent" else "fastest"
        saved_files = list_saved_kernel_files(sort_key)
        wrel_saved = working_kernel_task_relpath()
        if st.button(
            f"✅ Current best kernel · `{wrel_saved}`",
            key="sidebar-working-kernel",
            use_container_width=True,
            disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
        ):
            st.session_state.open_working_kernel_tab = True
            st.rerun()
        if not saved_files:
            st.caption("No saved revisions yet.")
        else:
            mem_set = set(st.session_state.get("chat_memory_files") or [])
            st.caption(
                "⭐ = in **chat memory** (summaries for the next chat message). "
                "Configure under **Single kernel implementation cycle**."
            )
            for filename in saved_files:
                display = kernel_display_meta(filename)
                label = f"⭐ {display['label']}" if filename in mem_set else display["label"]
                if st.button(
                    label,
                    key=f"open-{filename}",
                    use_container_width=True,
                    disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
                    help=display["tooltip"],
                ):
                    if filename not in st.session_state.open_summary_tabs:
                        st.session_state.open_summary_tabs.append(filename)
                    st.rerun()

    rejection_labels = [f"Blocked · {r['id'][-8:]}" for r in st.session_state.rejection_tabs]
    open_tab_labels = [kernel_display_meta(f)["tab_label"] for f in st.session_state.open_summary_tabs]
    working_tab_label = working_kernel_tab_label()
    show_working_tab = bool(st.session_state.get("open_working_kernel_tab"))
    tab_names = (
        ["Kernel Agent Chat", "Past kernels", "Manager", "Manager log"]
        + rejection_labels
        + ([working_tab_label] if show_working_tab else [])
        + open_tab_labels
    )
    tabs = st.tabs(tab_names)

    with tabs[0]:
        for msg in st.session_state.messages:
            render_message(msg)

        prompt = st.chat_input(
            "Ask the agent to improve the kernel...",
            disabled=st.session_state.run_in_progress or st.session_state.manager_run_in_progress,
        )
        if prompt:
            if prompt.strip().lower() == "reset":
                st.session_state.current_cpp = load_baseline_cpp()
                st.session_state.messages = []
                st.session_state.open_summary_tabs = []
                st.session_state.rejection_tabs = []
                st.session_state.open_working_kernel_tab = False
                st.session_state.run_in_progress = False
                st.session_state.manager_run_in_progress = False
                st.rerun()

            user_msg = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_msg)
            render_message(user_msg)

            with st.chat_message("assistant"):
                progress = st.progress(0, text="Preparing context...")
                status = st.empty()
                kernel_for_run = st.session_state.current_cpp
                st.session_state.run_in_progress = True
                try:
                    gr = _guardrail_max_retries()
                    mem_names = list(st.session_state.get("chat_memory_files") or [])
                    if mem_names:
                        status.info(
                            f"Loading summaries from {len(mem_names)} selected kernel(s) (chat memory)…"
                        )
                    else:
                        status.info("No chat memory selected — using current kernel only…")
                    summaries = asyncio.run(get_summary_context_for_filenames(mem_names))
                    progress.progress(30, text="Running kernel agent...")
                    status.info(f"Generating candidate kernel (guardrail retries ≤ {gr})…")
                    revision, turn_ctx = asyncio.run(
                        run_kernel_turn(
                            prompt,
                            kernel_for_run,
                            prior_summaries=summaries,
                            agent=agent,
                            max_retries=gr,
                        )
                    )
                    progress.progress(65, text="Guardrails passed — writing candidate + running evaluator…")
                    status.info("Evaluator: compile, correctness, benchmark (can take a few minutes)…")
                    eval_result: dict | None = None
                    promoted = False
                    eval_error = None
                    eval_skipped = False
                    try:
                        eval_result, promoted = asyncio.run(
                            asyncio.to_thread(apply_generation_to_task_files, revision.cpp_code)
                        )
                        if eval_result is None:
                            eval_skipped = True
                    except Exception as exc:
                        eval_error = str(exc)
                        eval_result = None
                        promoted = False
                    progress.progress(88, text="Saving to kernel_history if eval passed…")
                    saved_path = None
                    history_saved = False
                    if eval_error is None and should_save_kernel_history(eval_result):
                        saved_path = asyncio.run(
                            save_kernel_revision_with_summary(
                                revision,
                                prompt,
                                eval_result=eval_result,
                                is_best=promoted,
                            )
                        ).name
                        history_saved = True
                    elif eval_error is None:
                        history_saved = False
                    progress.progress(100, text="Done")
                    status.success(
                        "Kernel generated."
                        + (f" Saved to `kernel_history/{saved_path}`." if saved_path else "")
                    )
                except GuardrailRetriesExhausted as exc:
                    rid = f"rej-{uuid.uuid4().hex[:12]}"
                    tab_label = f"Blocked · {rid[-8:]}"
                    attempts_detail = [
                        {"attempt": a, "reason": r, "cpp_code": cpp, "rejected_cpp": cpp}
                        for a, r, cpp in exc.failed_attempts
                    ]
                    num_attempts = len(exc.failed_attempts)
                    st.session_state.rejection_tabs.insert(
                        0,
                        {
                            "id": rid,
                            "last_reason": exc.last_reason,
                            "attempts": attempts_detail,
                        },
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "type": "guardrail_error",
                            "summary": (
                                f"Injection guardrail blocked this generation after {num_attempts} "
                                f"attempt{'s' if num_attempts != 1 else ''}."
                            ),
                            "tab_label": tab_label,
                        }
                    )
                    progress.empty()
                    status.empty()
                    st.session_state.run_in_progress = False
                    st.rerun()
                except Exception as exc:
                    progress.empty()
                    status.empty()
                    st.session_state.run_in_progress = False
                    st.error(f"Agent call failed: {exc}")
                    return

                assistant_msg = {
                    "role": "assistant",
                    "explanation": revision.explanation,
                    "cpp_code": revision.cpp_code,
                    "saved_path": saved_path,
                    "history_saved": history_saved,
                    "eval_result": eval_result,
                    "promoted": promoted,
                    "eval_error": eval_error,
                    "eval_skipped": eval_skipped,
                    "web_research_used": turn_ctx.web_research_consumed,
                }
                st.session_state.messages.append(assistant_msg)
                st.markdown("**Explanation**")
                st.write(revision.explanation)
                if saved_path:
                    st.caption(f"Saved to `kernel_history/{saved_path}`")
                with st.expander("Generated C++", expanded=True):
                    st.code(revision.cpp_code, language="cpp")

                st.session_state.current_cpp = revision.cpp_code
                st.session_state.run_in_progress = False
                st.rerun()

    with tabs[1]:
        render_past_kernels_tab()

    with tabs[2]:
        render_manager_tab(agent)

    with tabs[3]:
        render_manager_log_tab()

    rej_idx = 4
    for rej in st.session_state.rejection_tabs:
        with tabs[rej_idx]:
            render_rejection_tab(rej)
        rej_idx += 1

    if st.session_state.get("open_working_kernel_tab"):
        with tabs[rej_idx]:
            render_working_kernel_tab()
        rej_idx += 1

    for filename in st.session_state.open_summary_tabs:
        with tabs[rej_idx]:
            render_summary_tab(filename)
        rej_idx += 1

    _render_manager_last_run_panel()


if __name__ == "__main__":
    main()
