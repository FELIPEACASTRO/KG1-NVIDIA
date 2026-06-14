#!/usr/bin/env python3
"""Run a KG1 Colab job while streaming logs to disk and optional Hugging Face."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kg1_live_log_common import empty_state, ingest_line, write_status


def default_log_dir() -> Path:
    if Path("/content").exists():
        return Path("/content/kg1_live_logs")
    return Path("artifacts/live_logs")


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    run_id = os.environ.get("RUN_ID", f"kg1_colab_{int(time.time())}")
    base = default_log_dir()
    parser.add_argument("--log-path", type=Path, default=base / f"{run_id}.log")
    parser.add_argument("--status-path", type=Path, default=base / f"{run_id}.status.json")
    parser.add_argument("--hf-repo", default=os.environ.get("KG1_LIVE_LOG_HF_REPO", ""))
    parser.add_argument("--hf-path", default=os.environ.get("KG1_LIVE_LOG_HF_PATH", f"colab/{run_id}/train.log"))
    parser.add_argument("--hf-status-path", default=os.environ.get("KG1_LIVE_STATUS_HF_PATH", f"colab/{run_id}/status.json"))
    parser.add_argument("--hf-repo-type", default=os.environ.get("KG1_LIVE_LOG_HF_REPO_TYPE", "dataset"))
    parser.add_argument("--upload-every", type=float, default=env_float("KG1_LIVE_LOG_UPLOAD_EVERY", 60.0))
    parser.add_argument("--watchdog-stale-seconds", type=float, default=env_float("KG1_WATCHDOG_STALE_SECONDS", 1800.0))
    parser.add_argument(
        "--watchdog-max-runtime-seconds",
        type=float,
        default=env_float("KG1_WATCHDOG_MAX_RUNTIME_SECONDS", 0.0),
    )
    parser.add_argument(
        "--disable-health-watchdog",
        action="store_true",
        default=env_bool("KG1_DISABLE_HEALTH_WATCHDOG", False),
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--require-upload", action="store_true", default=env_bool("KG1_REQUIRE_LIVE_LOG_UPLOAD"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def normalize_command(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        raw = raw[1:]
    if raw:
        return raw
    return [sys.executable, "scripts/hf_job_train_v90.py"]


def upload_loop(
    *,
    stop_event: threading.Event,
    log_path: Path,
    status_path: Path,
    repo_id: str,
    path_in_repo: str,
    status_path_in_repo: str,
    repo_type: str,
    upload_every: float,
    strict_upload: bool,
    child_process: subprocess.Popen[str] | None,
    failure_reasons: list[str],
) -> None:
    def mark_failure(reason: str) -> None:
        failure_reasons.append(reason)
        print(f"KG1_LIVE_UPLOAD_FATAL {reason}", flush=True)
        if child_process is not None and child_process.poll() is None:
            print("KG1_LIVE_UPLOAD_FATAL terminating_child_process=true", flush=True)
            child_process.terminate()

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        message = f"reason=huggingface_hub_unavailable error={exc}"
        if strict_upload:
            mark_failure(message)
        else:
            print(f"KG1_LIVE_UPLOAD_DISABLED {message}", flush=True)
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        message = "reason=missing_HF_TOKEN"
        if strict_upload:
            mark_failure(message)
        else:
            print(f"KG1_LIVE_UPLOAD_DISABLED {message}", flush=True)
        return

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True, token=token)
    except Exception as exc:
        message = f"reason=create_repo_failed error={exc}"
        if strict_upload:
            mark_failure(message)
        else:
            print(f"KG1_LIVE_UPLOAD_DISABLED {message}", flush=True)
        return

    print(
        "KG1_LIVE_UPLOAD_ENABLED "
        f"repo={repo_id} repo_type={repo_type} log_path={path_in_repo} status_path={status_path_in_repo}",
        flush=True,
    )
    # Upload IMEDIATO (canario): fecha a janela cega de <upload_every segundos.
    # Garante que um crash rapido no load (RC!=0 em <60s) ainda deixe o log no HF,
    # e revela na hora (KG1_LIVE_UPLOAD_OK/WARN) se o upload esta funcionando.
    try:
        upload_snapshot(
            api=api,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            log_path=log_path,
            status_path=status_path,
            path_in_repo=path_in_repo,
            status_path_in_repo=status_path_in_repo,
            raise_on_error=False,
        )
    except Exception as exc:
        print(f"KG1_LIVE_UPLOAD_WARN reason=initial_upload_failed error={type(exc).__name__}: {exc}", flush=True)
    while not stop_event.wait(max(5.0, upload_every)):
        try:
            upload_snapshot(
                api=api,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                log_path=log_path,
                status_path=status_path,
                path_in_repo=path_in_repo,
                status_path_in_repo=status_path_in_repo,
                raise_on_error=strict_upload,
            )
        except Exception as exc:
            mark_failure(f"reason=periodic_upload_failed error={type(exc).__name__}: {exc}")
            stop_event.set()
            return
    try:
        upload_snapshot(
            api=api,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            log_path=log_path,
            status_path=status_path,
            path_in_repo=path_in_repo,
            status_path_in_repo=status_path_in_repo,
            raise_on_error=strict_upload,
        )
    except Exception as exc:
        mark_failure(f"reason=final_upload_failed error={type(exc).__name__}: {exc}")


def upload_snapshot(
    *,
    api: Any,
    repo_id: str,
    repo_type: str,
    token: str,
    log_path: Path,
    status_path: Path,
    path_in_repo: str,
    status_path_in_repo: str,
    raise_on_error: bool = False,
) -> None:
    errors: list[str] = []
    for local_path, remote_path in [(log_path, path_in_repo), (status_path, status_path_in_repo)]:
        if not local_path.exists():
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=f"kg1 live log update {remote_path}",
            )
            print(f"KG1_LIVE_UPLOAD_OK path={remote_path} bytes={local_path.stat().st_size}", flush=True)
        except Exception as exc:
            message = f"path={remote_path} error={type(exc).__name__}: {exc}"
            print(f"KG1_LIVE_UPLOAD_WARN {message}", flush=True)
            errors.append(message)
    if raise_on_error and errors:
        raise RuntimeError("required live-log upload failed: " + "; ".join(errors))


def require_upload_ready(
    *,
    log_path: Path,
    status_path: Path,
    repo_id: str,
    path_in_repo: str,
    status_path_in_repo: str,
    repo_type: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(f"live-log upload is required, but huggingface_hub is unavailable: {exc}") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("live-log upload is required, but HF_TOKEN/HUGGINGFACE_HUB_TOKEN is missing")
    if not repo_id:
        raise RuntimeError("live-log upload is required, but --hf-repo is empty")

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True, token=token)
    except Exception as exc:
        raise RuntimeError(f"live-log upload is required, but create_repo failed: {exc}") from exc

    upload_snapshot(
        api=api,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        log_path=log_path,
        status_path=status_path,
        path_in_repo=path_in_repo,
        status_path_in_repo=status_path_in_repo,
        raise_on_error=True,
    )
    print(
        "KG1_LIVE_UPLOAD_REQUIRED_OK "
        f"repo={repo_id} repo_type={repo_type} log_path={path_in_repo} status_path={status_path_in_repo}",
        flush=True,
    )


def final_upload_after_end(
    *,
    log_path: Path,
    status_path: Path,
    repo_id: str,
    path_in_repo: str,
    status_path_in_repo: str,
    repo_type: str,
    strict_upload: bool,
) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        message = f"reason=final_huggingface_hub_unavailable error={exc}"
        if strict_upload:
            print(f"KG1_LIVE_UPLOAD_FATAL {message}", flush=True)
        else:
            print(f"KG1_LIVE_UPLOAD_DISABLED {message}", flush=True)
        return [message] if strict_upload else []

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        message = "reason=final_missing_HF_TOKEN"
        if strict_upload:
            print(f"KG1_LIVE_UPLOAD_FATAL {message}", flush=True)
        else:
            print(f"KG1_LIVE_UPLOAD_DISABLED {message}", flush=True)
        return [message] if strict_upload else []

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True, token=token)
        upload_snapshot(
            api=api,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            log_path=log_path,
            status_path=status_path,
            path_in_repo=path_in_repo,
            status_path_in_repo=status_path_in_repo,
            raise_on_error=strict_upload,
        )
    except Exception as exc:
        message = f"reason=final_upload_after_end_failed error={type(exc).__name__}: {exc}"
        if strict_upload:
            print(f"KG1_LIVE_UPLOAD_FATAL {message}", flush=True)
        else:
            print(f"KG1_LIVE_UPLOAD_WARN {message}", flush=True)
        return [message] if strict_upload else []
    return []


def main() -> int:
    args = build_parser().parse_args()
    command = normalize_command(args.command)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("FRIENDLY_REALTIME_LOGS", "1")
    env.setdefault("FRIENDLY_LOG_SCORE_HINTS", "1")

    runner_start_lines = [
        "KG1_COLAB_REALTIME_RUNNER_START",
        f"command={' '.join(command)}",
        f"log_path={args.log_path}",
        f"status_path={args.status_path}",
        f"watchdog_stale_seconds={args.watchdog_stale_seconds}",
        f"watchdog_max_runtime_seconds={args.watchdog_max_runtime_seconds}",
        f"health_watchdog_enabled={not args.disable_health_watchdog}",
    ]
    for line in runner_start_lines:
        print(line, flush=True)
    args.log_path.write_text("\n".join(runner_start_lines) + "\n", encoding="utf-8")

    state = empty_state()
    for line in runner_start_lines:
        ingest_line(state, line)
    write_status(args.status_path, state)

    upload_enabled = bool(args.hf_repo and not args.no_upload)
    if args.require_upload:
        if not upload_enabled:
            raise RuntimeError("live-log upload is required, but upload is disabled or --hf-repo is empty")
        require_upload_ready(
            log_path=args.log_path,
            status_path=args.status_path,
            repo_id=args.hf_repo,
            path_in_repo=args.hf_path,
            status_path_in_repo=args.hf_status_path,
            repo_type=args.hf_repo_type,
        )

    stop_event = threading.Event()
    uploader: threading.Thread | None = None
    watchdog_thread: threading.Thread | None = None
    upload_failure_reasons: list[str] = []
    watchdog_failure_reasons: list[str] = []
    state_lock = threading.Lock()
    watchdog_state = {
        "started_at": time.monotonic(),
        "last_output_at": time.monotonic(),
        "triggered": False,
    }

    def append_runner_line(line: str) -> None:
        with state_lock:
            with args.log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(line + "\n")
            ingest_line(state, line)
            write_status(args.status_path, state)
        print(line, flush=True)

    def terminate_child(proc: subprocess.Popen[str], reason: str) -> None:
        if watchdog_state["triggered"]:
            return
        watchdog_state["triggered"] = True
        watchdog_failure_reasons.append(reason)
        append_runner_line(f"KG1_WATCHDOG_STOP reason={reason}")
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                append_runner_line(f"KG1_WATCHDOG_KILL reason={reason}")
                proc.kill()

    def watchdog_loop(proc: subprocess.Popen[str]) -> None:
        last_hb = time.monotonic()
        hb_every = 45.0
        while not stop_event.wait(5.0):
            if proc.poll() is not None:
                return
            now = time.monotonic()
            # HEARTBEAT visivel: pulso a cada ~45s mesmo em fase silenciosa (load do 30B).
            # last_output_age_s mostra travamento CRESCENDO antes do watchdog matar por stall.
            if now - last_hb >= hb_every:
                last_hb = now
                age = now - float(watchdog_state["last_output_at"])
                elapsed = now - float(watchdog_state["started_at"])
                hb = (f"KG1_WRAPPER_HEARTBEAT elapsed_s={elapsed:.0f} last_output_age_s={age:.0f} "
                      f"stall_limit_s={args.watchdog_stale_seconds:.0f} child_alive=true")
                try:
                    with open(args.log_path, "a", encoding="utf-8") as _hbf:
                        _hbf.write(hb + "\n")
                except Exception:
                    pass
                print(hb, flush=True)
            if args.watchdog_max_runtime_seconds > 0:
                runtime = now - float(watchdog_state["started_at"])
                if runtime > args.watchdog_max_runtime_seconds:
                    terminate_child(proc, f"max_runtime_exceeded elapsed_s={runtime:.1f}")
                    return
            if args.watchdog_stale_seconds > 0:
                silence = now - float(watchdog_state["last_output_at"])
                if silence > args.watchdog_stale_seconds:
                    terminate_child(proc, f"no_stdout_progress elapsed_s={silence:.1f}")
                    return
            if not args.disable_health_watchdog:
                with state_lock:
                    health = str(state.get("health") or "")
                    health_reason = str(state.get("health_reason") or "")
                    process_finished = bool(state.get("process_finished"))
                if health == "STOP" and not process_finished:
                    terminate_child(proc, f"health_stop {health_reason[:240]}")
                    return

    if not upload_enabled:
        print("KG1_LIVE_UPLOAD_DISABLED reason=no_hf_repo_or_no_upload", flush=True)
    rc = 1
    with args.log_path.open("a", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if upload_enabled:
            uploader = threading.Thread(
                target=upload_loop,
                kwargs={
                    "stop_event": stop_event,
                    "log_path": args.log_path,
                    "status_path": args.status_path,
                    "repo_id": args.hf_repo,
                    "path_in_repo": args.hf_path,
                    "status_path_in_repo": args.hf_status_path,
                    "repo_type": args.hf_repo_type,
                    "upload_every": args.upload_every,
                    "strict_upload": args.require_upload,
                    "child_process": proc,
                    "failure_reasons": upload_failure_reasons,
                },
                daemon=True,
            )
            uploader.start()
        watchdog_thread = threading.Thread(target=watchdog_loop, args=(proc,), daemon=True)
        watchdog_thread.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            with state_lock:
                watchdog_state["last_output_at"] = time.monotonic()
                ingest_line(state, line)
                write_status(args.status_path, state)
        rc = proc.wait()

    stop_event.set()
    if uploader is not None:
        uploader.join(timeout=max(10.0, args.upload_every + 5.0))
    if watchdog_thread is not None:
        watchdog_thread.join(timeout=10.0)
    if watchdog_failure_reasons and rc == 0:
        rc = 89
    if watchdog_failure_reasons:
        state["watchdog_failure_reasons"] = watchdog_failure_reasons
    if upload_failure_reasons and rc == 0:
        rc = 88
        failure_line = "KG1_COLAB_REALTIME_RUNNER_UPLOAD_FAILURE return_code=88 " + "; ".join(upload_failure_reasons)
        with args.log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(failure_line + "\n")
        ingest_line(state, failure_line)
        state["return_code"] = rc
        state["upload_failure_reasons"] = upload_failure_reasons
        write_status(args.status_path, state)
        print(failure_line, flush=True)
    if upload_enabled:
        final_upload_failures = final_upload_after_end(
            log_path=args.log_path,
            status_path=args.status_path,
            repo_id=args.hf_repo,
            path_in_repo=args.hf_path,
            status_path_in_repo=args.hf_status_path,
            repo_type=args.hf_repo_type,
            strict_upload=args.require_upload,
        )
        if final_upload_failures and rc == 0:
            rc = 88
            failure_line = (
                "KG1_COLAB_REALTIME_RUNNER_UPLOAD_FAILURE return_code=88 "
                + "; ".join(final_upload_failures)
            )
            with args.log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(failure_line + "\n")
            ingest_line(state, failure_line)
            state["return_code"] = rc
            state["upload_failure_reasons"] = list(upload_failure_reasons) + final_upload_failures
            write_status(args.status_path, state)
            print(failure_line, flush=True)
    end_line = f"KG1_COLAB_REALTIME_RUNNER_END return_code={rc}"
    with args.log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(end_line + "\n")
    state["return_code"] = rc
    state["process_finished"] = True
    ingest_line(state, end_line)
    write_status(args.status_path, state)
    print(end_line, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
