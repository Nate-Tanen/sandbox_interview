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

from main import (
    GuardrailRetriesExhausted,
    apply_generation_to_task_files,
    build_agent,
    cpp_matches_best_kernel,
    should_save_kernel_history,
    get_model_name,
    get_top_k_summary_context,
    init_env,
    list_saved_kernel_files,
    load_baseline_cpp,
    load_saved_kernel_revision,
    load_working_cpp,
    run_kernel_turn,
    save_kernel_revision_with_summary,
    working_kernel_task_relpath,
)

# Default for newest-K summary context (overridden by sidebar `summary_context_k`).
_DEFAULT_SUMMARY_K = 3
_DEFAULT_GUARD_RETRIES = 3


def _summary_context_k() -> int:
    return int(st.session_state.get("summary_context_k", _DEFAULT_SUMMARY_K))


def _guardrail_max_retries() -> int:
    return int(st.session_state.get("guardrail_max_retries", _DEFAULT_GUARD_RETRIES))


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


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return

        if msg.get("type") == "guardrail_error":
            st.warning(msg.get("summary", "Guardrail blocked this generation."))
            if msg.get("tab_label"):
                st.caption(f"Details are in the **{msg['tab_label']}** tab.")
            return

        st.markdown("**Explanation**")
        st.write(msg["explanation"])
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
        "Each attempt below was rejected by the guardrail (static denylist or LLM guardrail). "
        "Regenerate without repeating these patterns."
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
    k = _summary_context_k()
    st.caption(
        f"Sort order matches the sidebar. Open a file in its own tab to inspect summary + code. "
        f"**K = {k}**: the **{k} newest** saved kernels’ summaries are used as agent context — ⭐ marks those (by recency, not sort order here)."
    )
    if not saved_files:
        st.info("No saved kernels yet.")
        return

    top_set = set(list_saved_kernel_files("recent")[:k])

    for filename in saved_files:
        cols = st.columns([4, 1, 1])
        display = kernel_display_meta(filename)
        with cols[0]:
            prefix = "⭐ " if filename in top_set else ""
            st.markdown(f"{prefix}`{display['label']}`")
        with cols[1]:
            if st.button("Open tab", key=f"past-open-{filename}", help=display["tooltip"]):
                if filename not in st.session_state.open_summary_tabs:
                    st.session_state.open_summary_tabs.append(filename)
                st.rerun()
        with cols[2]:
            if st.button("Use kernel", key=f"past-use-{filename}"):
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


def main() -> None:
    init_env()
    st.set_page_config(page_title="Kernel Agent UI", page_icon="UI", layout="wide")
    st.title("Kernel Agent")
    st.caption(
        "Chat with the kernel optimization agent. Type **reset** in chat to restart from baseline. "
        "Use the sidebar to set **K** (prior summaries) and **guardrail retries**."
    )

    model_name = get_model_name()
    agent = get_agent(model_name)
    init_state()

    with st.sidebar:
        st.subheader("Session")
        st.write(f"Model: `{model_name}`")
        if st.button("Reset to baseline", use_container_width=True, disabled=st.session_state.run_in_progress):
            st.session_state.current_cpp = load_baseline_cpp()
            st.session_state.messages = []
            st.session_state.open_summary_tabs = []
            st.session_state.rejection_tabs = []
            st.session_state.open_working_kernel_tab = False
            st.session_state.run_in_progress = False
            st.rerun()
        wrel = working_kernel_task_relpath()
        if st.button(
            f"Open current best kernel · `{wrel}`",
            key="open-working-kernel-preview",
            use_container_width=True,
            disabled=st.session_state.run_in_progress,
        ):
            st.session_state.open_working_kernel_tab = True
            st.rerun()
        st.number_input(
            "Prior summaries (K)",
            min_value=1,
            max_value=50,
            value=_DEFAULT_SUMMARY_K,
            key="summary_context_k",
            help="Number of newest `kernel_history` summaries injected into each agent turn.",
        )
        st.number_input(
            "Guardrail max retries",
            min_value=1,
            max_value=20,
            value=_DEFAULT_GUARD_RETRIES,
            key="guardrail_max_retries",
            help="How many times to regenerate when the reward-hacking guardrail blocks output.",
        )

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
            disabled=st.session_state.run_in_progress,
        ):
            st.session_state.open_working_kernel_tab = True
            st.rerun()
        if not saved_files:
            st.caption("No saved revisions yet.")
        else:
            k_ctx = _summary_context_k()
            top_k_files = set(list_saved_kernel_files("recent")[:k_ctx])
            st.caption(
                f"**K = {k_ctx}**: the **{k_ctx} newest** saved kernels’ summaries are included in the agent prompt each turn — ⭐ highlights them."
            )
            for filename in saved_files:
                display = kernel_display_meta(filename)
                label = f"⭐ {display['label']}" if filename in top_k_files else display["label"]
                if st.button(
                    label,
                    key=f"open-{filename}",
                    use_container_width=True,
                    disabled=st.session_state.run_in_progress,
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
        ["Kernel Agent Chat", "Past kernels"]
        + rejection_labels
        + ([working_tab_label] if show_working_tab else [])
        + open_tab_labels
    )
    tabs = st.tabs(tab_names)

    with tabs[0]:
        for msg in st.session_state.messages:
            render_message(msg)

        prompt = st.chat_input("Ask the agent to improve the kernel...")
        if prompt:
            if prompt.strip().lower() == "reset":
                st.session_state.current_cpp = load_baseline_cpp()
                st.session_state.messages = []
                st.session_state.open_summary_tabs = []
                st.session_state.rejection_tabs = []
                st.session_state.open_working_kernel_tab = False
                st.session_state.run_in_progress = False
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
                    k_ctx = _summary_context_k()
                    gr = _guardrail_max_retries()
                    status.info(
                        f"Loading summaries for the {k_ctx} newest kernels (prior prompt context)…"
                    )
                    summaries = asyncio.run(get_top_k_summary_context(k=k_ctx))
                    progress.progress(30, text="Running kernel agent...")
                    status.info(f"Generating candidate kernel (guardrail retries ≤ {gr})…")
                    revision = asyncio.run(
                        run_kernel_turn(
                            agent,
                            prompt,
                            kernel_for_run,
                            prior_summaries=summaries,
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
                                f"Guardrail blocked this generation after {num_attempts} "
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

    rej_idx = 2
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


if __name__ == "__main__":
    main()
