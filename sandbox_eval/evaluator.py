import copy
import gc
import multiprocessing
import os
import pathlib
import random
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from typing import Optional

import torch

from .util import debug_print, set_seed


LOAD_MODEL_BACKOFF_TIME = 1.0
RUN_MODEL_BACKOFF_TIME = 2.0
EVAL_RUN_TIMEOUT = float(os.getenv("EVAL_RUN_TIMEOUT", "300.0"))
EVAL_RUN_TIMEOUT_CHECK_COMPILE = float(os.getenv("EVAL_RUN_TIMEOUT_CHECK_COMPILE", "300.0"))
EVAL_RUN_TIMEOUT_CHECK_CORRECT = float(os.getenv("EVAL_RUN_TIMEOUT_CHECK_CORRECT", "600.0"))
BENCHMARK_NUM_WARMUPS = int(os.getenv("BENCHMARK_NUM_WARMUPS", "1"))
BENCHMARK_NUM_TRIALS = int(os.getenv("BENCHMARK_NUM_TRIALS", "3"))


@dataclass
class EvaluationResult:
    compile_pass: bool
    correct_pass: bool
    reference_time_ms: Optional[float] = None
    candidate_time_ms: Optional[float] = None
    speedup: Optional[float] = None
    metadata: dict = field(default_factory=dict)


def remove_build_directory(build_directory: str):
    if os.path.exists(build_directory):
        try:
            shutil.rmtree(build_directory, ignore_errors=True)
        except Exception as exc:
            debug_print(f"[EVALUATE] Failed to remove build directory {build_directory}: {exc}")


def clone_inputs(inputs):
    cloned = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            cloned.append(x.clone().detach())
        else:
            cloned.append(copy.deepcopy(x))
    return cloned


def tensors_pass(a, b, atol=1e-03, rtol=1e-03):
    diff = torch.abs(a.to(torch.float32) - b.to(torch.float32))
    max_diff = torch.max(diff).item()
    avg_diff = torch.mean(diff).item()
    threshold = atol + rtol * torch.abs(a.to(torch.float32))
    within_tolerance = diff <= threshold
    percent_within_tolerance = within_tolerance.float().mean().item() * 100.0
    return max_diff, avg_diff, percent_within_tolerance


def load_original_model_and_inputs(ref_src: str, context: dict, metadata: dict):
    try:
        compile(ref_src, "<string>", "exec")
    except SyntaxError as exc:
        metadata["compile"] = f"Syntax Error in original code {exc}"
        return None

    try:
        exec(ref_src, context)
    except Exception as exc:
        metadata["compile"] = f"Error in executing original code {exc}"
        return None

    return (
        context.get("Model"),
        context.get("get_init_inputs"),
        context.get("get_inputs"),
    )


def load_custom_model(candidate_src: str, context: dict, metadata: dict, build_directory: str):
    build_path = pathlib.Path(build_directory)
    build_path.mkdir(parents=True, exist_ok=True)

    context["BUILD_DIRECTORY"] = build_directory
    candidate_src = (
        "import os\nimport gc\n"
        f"os.environ['TORCH_EXTENSIONS_DIR'] = r'{build_directory}'\n"
    ) + candidate_src + "\ngc.collect()\n"

    model_file_path = build_path / "model_custom.py"
    model_file_path.write_text(candidate_src)

    read_fd, write_fd = os.pipe()
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    retval = True
    try:
        code = compile(model_file_path.read_text(), str(model_file_path), "exec")
        exec(code, context)
    except Exception as exc:
        metadata["compile"] = f"Syntax error in generated code: {exc}"
        retval = None

    try:
        model_new = context.get("ModelNew")
    except Exception as exc:
        metadata["compile"] = f"Error in executing generated code {exc}"
        retval = None
        model_new = None

    os.dup2(old_out, 1)
    os.dup2(old_err, 2)
    os.close(old_out)
    os.close(old_err)

    gc.collect()

    with os.fdopen(read_fd, "r") as log_file:
        error = log_file.read()

    if retval is not None:
        return model_new

    if error:
        metadata["compile"] += f"\nProgram output: {error}"
    return None


def _check_compile(ref_src, candidate_src, metadata, build_directory=None):
    context = {}
    if ref_src is not None:
        try:
            loaded = load_original_model_and_inputs(ref_src, context, metadata)
        except Exception:
            return False
        if loaded is None:
            return False

    try:
        model_new = load_custom_model(candidate_src, context, metadata, build_directory)
        if model_new is None:
            metadata["compile"] += "Loading ModelNew failed: ModelNew is None"
            return False
    except Exception:
        gc.collect()
        return False
    return True


def _build_model(model_cls, init_inputs):
    if len(init_inputs) == 2 and isinstance(init_inputs[0], list) and isinstance(init_inputs[1], dict):
        return model_cls(*init_inputs[0], **init_inputs[1])
    return model_cls(*init_inputs)


def _check_correct(
    ref_src,
    candidate_src,
    metadata,
    num_trials=10,
    seed_num=42,
    build_directory=None,
    device=None,
):
    metadata["hardware"] = str(device)
    metadata["device"] = str(device)

    ref_context = {}
    candidate_context = {}

    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, ref_context, metadata)
    ModelNew = load_custom_model(candidate_src, candidate_context, metadata, build_directory)

    try:
        set_seed(seed_num)
        init_inputs = get_init_inputs_fn()
        init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

        with torch.no_grad():
            set_seed(seed_num)
            original_model = _build_model(Model, init_inputs)

            set_seed(seed_num)
            custom_model = _build_model(ModelNew, init_inputs)

        pass_count = 0
        torch.manual_seed(seed_num)
        correctness_trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)]

        with torch.no_grad():
            for trial in range(num_trials):
                trial_seed = correctness_trial_seeds[trial]
                print(f"[correctness] trial {trial + 1}/{num_trials} starting", flush=True)

                set_seed(trial_seed)
                inputs = get_inputs_fn()
                inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

                try:
                    inputs_new = clone_inputs(inputs)
                    output_new = custom_model(*inputs_new)
                except Exception as exc:
                    metadata["correct"] = f"Runtime error when checking correctness: {exc}"
                    gc.collect()
                    return False

                output = original_model(*inputs)

                out_seq = list(output) if isinstance(output, (tuple, list)) else [output]
                out_new_seq = list(output_new) if isinstance(output_new, (tuple, list)) else [output_new]

                if len(out_seq) != len(out_new_seq):
                    metadata["correct"] = (
                        f"Output tuple/list length mismatch, expected {len(out_seq)}, got {len(out_new_seq)}"
                    )
                    print(f"[correctness] trial {trial + 1}/{num_trials} failed: {metadata['correct']}", flush=True)
                    continue

                shape_mismatch = False
                for i, (a, b) in enumerate(zip(out_seq, out_new_seq)):
                    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                        metadata["correct"] = f"Non-tensor output at output tensor {i}: types {type(a)} vs {type(b)}"
                        shape_mismatch = True
                        break
                    if a.shape != b.shape:
                        metadata["correct"] = f"Output shape mismatch at output tensor {i}: expected {a.shape}, got {b.shape}"
                        shape_mismatch = True
                        break

                if shape_mismatch:
                    print(f"[correctness] trial {trial + 1}/{num_trials} failed: {metadata['correct']}", flush=True)
                    continue

                any_mismatch = False
                max_diffs = []
                avg_diffs = []
                for i, (a, b) in enumerate(zip(out_seq, out_new_seq)):
                    max_diff, avg_diff, percent_within_tolerance = tensors_pass(a, b, atol=1e-03, rtol=1e-03)
                    num_elements = a.numel()
                    num_correct_elements = int((percent_within_tolerance / 100.0) * num_elements)
                    debug_print(
                        "[EVALUATE]",
                        f"trial {trial} output {i} max_diff={max_diff} avg_diff={avg_diff} ",
                        f"percent={percent_within_tolerance} num_elements={num_elements} ",
                        f"num_correct_elements={num_correct_elements}",
                    )
                    print(
                        f"[correctness] trial {trial + 1}/{num_trials} output {i}: "
                        f"{percent_within_tolerance:.2f}% within tolerance "
                        f"(max_diff={max_diff:.6g}, avg_diff={avg_diff:.6g})",
                        flush=True,
                    )
                    if percent_within_tolerance < 95:
                        metadata["correct"] = (
                            f"Output value mismatch at output tensor {i}, "
                            f"max diff: {max_diff}, avg diff: {avg_diff}, "
                            f"percent within tolerance: {percent_within_tolerance}"
                        )
                        print(f"[correctness] trial {trial + 1}/{num_trials} failed: {metadata['correct']}", flush=True)
                        any_mismatch = True
                        break
                    max_diffs.append(max_diff)
                    avg_diffs.append(avg_diff)

                if any_mismatch:
                    continue

                print(f"[correctness] trial {trial + 1}/{num_trials} passed", flush=True)
                if trial == num_trials - 1:
                    metadata["diff"] = f"Output value matched; max diffs: {max_diffs}, avg diffs: {avg_diffs}"
                pass_count += 1

    except Exception as exc:
        metadata["correct"] = f"Runtime error when checking correctness: {exc}"
        gc.collect()
        return False

    print(f"[correctness] completed: passed {pass_count} out of {num_trials} trials", flush=True)
    metadata["correct"] = f"Passed {pass_count} out of {num_trials} trials: {metadata.get('correct', 'ALL PASSED')}"
    if "diff" in metadata:
        metadata["correct"] += f"\n{metadata['diff']}"
        del metadata["diff"]
    return pass_count == num_trials


def _benchmark_model(model, inputs, num_warmups, num_trials, label):
    elapsed_times = []

    print(f"[benchmark] {label}: warmup starting ({num_warmups} warmups)", flush=True)
    with torch.no_grad():
        for warmup_idx in range(num_warmups):
            model(*inputs)
            print(
                f"[benchmark] {label}: completed warmup {warmup_idx + 1}/{num_warmups}",
                flush=True,
            )

        print(f"[benchmark] {label}: timing starting ({num_trials} trials)", flush=True)
        for trial_idx in range(num_trials):
            start = time.perf_counter()
            model(*inputs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            elapsed_times.append(elapsed_ms)
            print(
                f"[benchmark] {label}: trial {trial_idx + 1}/{num_trials} = {elapsed_ms:.6g} ms",
                flush=True,
            )

    mean_time = sum(elapsed_times) / len(elapsed_times)
    min_time = min(elapsed_times)
    max_time = max(elapsed_times)
    if len(elapsed_times) > 1:
        variance = sum((x - mean_time) ** 2 for x in elapsed_times) / len(elapsed_times)
        std_time = variance ** 0.5
    else:
        std_time = 0.0

    return {
        "mean_ms": float(f"{mean_time:.6g}"),
        "std_ms": float(f"{std_time:.6g}"),
        "min_ms": float(f"{min_time:.6g}"),
        "max_ms": float(f"{max_time:.6g}"),
        "num_trials": len(elapsed_times),
        "num_warmups": num_warmups,
    }


def benchmark_speeds(
    ref_src,
    candidate_src,
    metadata,
    build_directory=None,
    device=None,
    seed_num=42,
    num_warmups=BENCHMARK_NUM_WARMUPS,
    num_trials=BENCHMARK_NUM_TRIALS,
):
    metadata["hardware"] = str(device)
    metadata["device"] = str(device)

    ref_context = {}
    candidate_context = {}
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, ref_context, metadata)
    ModelNew = load_custom_model(candidate_src, candidate_context, metadata, build_directory)

    if Model is None or ModelNew is None:
        raise RuntimeError("Failed to load models for benchmarking.")

    set_seed(seed_num)
    init_inputs = get_init_inputs_fn()
    init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    set_seed(seed_num)
    benchmark_inputs = get_inputs_fn()
    benchmark_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in benchmark_inputs]

    with torch.no_grad():
        set_seed(seed_num)
        reference_model = _build_model(Model, init_inputs)

        set_seed(seed_num)
        candidate_model = _build_model(ModelNew, init_inputs)

    print("[benchmark] reference: preparing benchmark", flush=True)
    reference_stats = _benchmark_model(reference_model, benchmark_inputs, num_warmups, num_trials, "reference")
    print(
        f"[benchmark] reference: mean={reference_stats['mean_ms']} ms std={reference_stats['std_ms']} ms",
        flush=True,
    )

    print("[benchmark] candidate: preparing benchmark", flush=True)
    candidate_stats = _benchmark_model(candidate_model, benchmark_inputs, num_warmups, num_trials, "candidate")
    print(
        f"[benchmark] candidate: mean={candidate_stats['mean_ms']} ms std={candidate_stats['std_ms']} ms",
        flush=True,
    )

    candidate_mean = candidate_stats["mean_ms"]
    speedup = reference_stats["mean_ms"] / candidate_mean if candidate_mean > 0 else None
    if speedup is not None:
        print(f"[benchmark] speedup: {speedup:.6g}x", flush=True)

    return {
        "reference": reference_stats,
        "candidate": candidate_stats,
        "speedup": float(f"{speedup:.6g}") if speedup is not None else None,
    }


def _worker_wrapper(func, args: tuple, kwargs: dict, conn: Connection):
    import sys
    from io import StringIO

    stderr_capture = StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture
    metadata = kwargs.get("metadata", {})

    try:
        result = func(*args, **kwargs)
        stderr_content = stderr_capture.getvalue()
        if stderr_content:
            metadata["_stderr"] = stderr_content
        conn.send((True, result, metadata))
    except Exception as exc:
        metadata["_subproc_error"] = str(exc)
        metadata["_subproc_traceback"] = traceback.format_exc()
        stderr_content = stderr_capture.getvalue()
        if stderr_content:
            metadata["_stderr"] = stderr_content
        try:
            conn.send((False, None, metadata))
        except Exception:
            pass
    finally:
        sys.stderr = old_stderr
        stderr_capture.close()
        conn.close()


def run_in_subprocess(func, *args, stage_name: Optional[str] = None, **kwargs):
    if "metadata" not in kwargs:
        raise ValueError("metadata must be provided")

    stage = stage_name or getattr(func, "__name__", "unknown_stage")
    stage_timeout = EVAL_RUN_TIMEOUT
    if stage == "check_compile":
        stage_timeout = EVAL_RUN_TIMEOUT_CHECK_COMPILE
    elif stage == "check_correct":
        stage_timeout = EVAL_RUN_TIMEOUT_CHECK_CORRECT

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_worker_wrapper, args=(func, args, kwargs, child_conn))
    process.start()
    child_conn.close()

    if not parent_conn.poll(stage_timeout):
        process.terminate()
        process.join()
        kwargs["metadata"]["timeout"] = f"run_in_subprocess({stage}): timed out after {stage_timeout}s"
        kwargs["metadata"]["timeout_stage"] = stage
        if "correct" in stage:
            kwargs["metadata"]["correct"] = (
                kwargs["metadata"].get("correct", "") + f"\nTimeout in {stage} after {stage_timeout}s"
            ).strip()
        if "compile" in stage:
            kwargs["metadata"]["compile"] = (
                kwargs["metadata"].get("compile", "") + f"\nTimeout in {stage} after {stage_timeout}s"
            ).strip()
        return False

    try:
        success, result, new_meta = parent_conn.recv()
        process.join()

        if not result and new_meta.get("correct"):
            kwargs["metadata"]["correct"] = (
                kwargs["metadata"].get("correct", "") + f"\n{new_meta['correct']}"
            ).strip()
        if not result and new_meta.get("compile"):
            kwargs["metadata"]["compile"] = (
                kwargs["metadata"].get("compile", "") + f"\n{new_meta['compile']}"
            ).strip()

        if not result and "_stderr" in new_meta and new_meta["_stderr"].strip():
            target_key = "compile" if "compile" in stage else "correct"
            kwargs["metadata"][target_key] = (
                kwargs["metadata"].get(target_key, "") + f"\nStderr output: {new_meta['_stderr']}"
            ).strip()

        if "compile" in new_meta:
            kwargs["metadata"]["compile"] = new_meta["compile"]

        if not success:
            sub_err = str(new_meta.get("_subproc_error", "") or "").strip()
            sub_tb = str(new_meta.get("_subproc_traceback", "") or "").strip()
            if sub_err:
                target_key = "compile" if "compile" in stage else "correct"
                kwargs["metadata"][target_key] = (
                    kwargs["metadata"].get(target_key, "") + f"\nSubprocess error in {stage}: {sub_err}"
                ).strip()
            if sub_tb:
                tb_tail = "\n".join(sub_tb.splitlines()[-12:])
                target_key = "compile" if "compile" in stage else "correct"
                kwargs["metadata"][target_key] = (
                    kwargs["metadata"].get(target_key, "") + f"\nTraceback tail ({stage}):\n{tb_tail}"
                ).strip()

        return result if success else False
    except EOFError:
        process.join()
        kwargs["metadata"]["correct"] = (
            kwargs["metadata"].get("correct", "") + f"Process terminated unexpectedly with exit code: {process.exitcode}"
        ).strip()
        return False
    except Exception as exc:
        process.terminate()
        process.join()
        kwargs["metadata"]["other"] = f"Communication error: {exc}"
        return False


def check_compile(ref_src, candidate_src, metadata, build_directory):
    print("[stage] compile: starting candidate compile/load check", flush=True)
    result = run_in_subprocess(
        _check_compile,
        ref_src,
        candidate_src,
        stage_name="check_compile",
        metadata=metadata,
        build_directory=build_directory,
    )
    print(f"[stage] compile: {'passed' if result else 'failed'}", flush=True)
    return result


def check_correct(ref_src, candidate_src, metadata, num_trials=10, seed_num=42, build_directory=None, device=None):
    print("[stage] correctness: starting correctness evaluation", flush=True)
    result = run_in_subprocess(
        _check_correct,
        ref_src,
        candidate_src,
        stage_name="check_correct",
        metadata=metadata,
        num_trials=num_trials,
        seed_num=seed_num,
        build_directory=build_directory,
        device=device,
    )
    print(f"[stage] correctness: {'passed' if result else 'failed'}", flush=True)
    return result


def evaluate_sources(ref_src: str, candidate_src: str, build_root: pathlib.Path, num_trials: int = 10, seed_num: int = 42):
    print("[stage] evaluator: starting evaluation", flush=True)

    build_root.mkdir(parents=True, exist_ok=True)
    metadata = {"compile": "", "correct": ""}
    build_directory = tempfile.mkdtemp(prefix="eval-", dir=str(build_root))
    print(f"[stage] evaluator: build directory = {build_directory}", flush=True)

    compile_pass = check_compile(ref_src, candidate_src, metadata, build_directory=build_directory)
    correct_pass = False
    reference_time_ms = None
    candidate_time_ms = None
    speedup = None
    device = torch.device("cpu")

    if compile_pass:
        print(f"[stage] evaluator: using device {device}", flush=True)
        correct_pass = check_correct(
            ref_src,
            candidate_src,
            metadata,
            num_trials=num_trials,
            seed_num=seed_num,
            build_directory=build_directory,
            device=device,
        )

        if correct_pass:
            print("[stage] benchmark: starting benchmark runs", flush=True)
            try:
                timing = benchmark_speeds(
                    ref_src,
                    candidate_src,
                    metadata,
                    build_directory=build_directory,
                    device=device,
                    seed_num=seed_num,
                )
                metadata["timing"] = timing
                reference_time_ms = timing["reference"]["mean_ms"]
                candidate_time_ms = timing["candidate"]["mean_ms"]
                speedup = timing["speedup"]
            except Exception as exc:
                metadata["benchmark"] = f"Benchmark failed: {exc}"
                print(f"[stage] benchmark: failed ({exc})", flush=True)

    print("[stage] evaluator: cleaning up build directory", flush=True)
    remove_build_directory(build_directory)
    print("[stage] evaluator: finished evaluation", flush=True)
    return asdict(
        EvaluationResult(
            compile_pass=compile_pass,
            correct_pass=correct_pass,
            reference_time_ms=reference_time_ms,
            candidate_time_ms=candidate_time_ms,
            speedup=speedup,
            metadata=metadata,
        )
    )
