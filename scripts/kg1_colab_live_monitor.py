#!/usr/bin/env python3
"""Monitor KG1 Colab live logs from a local file or Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from kg1_live_log_common import parse_text, render_dashboard, write_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log-path", type=Path)
    source.add_argument("--hf-repo")
    parser.add_argument("--hf-path", default=os.environ.get("KG1_LIVE_LOG_HF_PATH", ""))
    parser.add_argument("--hf-repo-type", default=os.environ.get("KG1_LIVE_LOG_HF_REPO_TYPE", "dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/live_log_monitor_cache"))
    parser.add_argument("--status-out", type=Path, default=Path("artifacts/live_log_monitor_cache/latest_status.json"))
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--target-accuracy", type=float, default=float(os.environ.get("KG1_TARGET_ACCURACY", "0.89")))
    parser.add_argument("--full-rows", type=int, default=int(os.environ.get("KG1_FULL_ROWS", "947")))
    parser.add_argument("--baseline-correct", type=int, default=int(os.environ.get("KG1_BASELINE_CORRECT", "823")))
    return parser


def read_local(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_hf(repo_id: str, path_in_repo: str, repo_type: str, cache_dir: Path) -> str:
    if not path_in_repo:
        raise ValueError("--hf-path is required when --hf-repo is used")
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub unavailable: {exc}") from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type=repo_type,
        token=token or None,
        cache_dir=str(cache_dir),
        force_download=True,
    )
    return Path(local_path).read_text(encoding="utf-8", errors="replace")


def load_text(args: argparse.Namespace) -> str:
    if args.log_path is not None:
        return read_local(args.log_path)
    return read_hf(args.hf_repo, args.hf_path, args.hf_repo_type, args.cache_dir)


def main() -> int:
    args = build_parser().parse_args()
    last_line_count: int | None = None
    while True:
        try:
            text = load_text(args)
            state = parse_text(text)
            write_status(args.status_out, state)
            if state["line_count"] != last_line_count or args.once:
                print(
                    render_dashboard(
                        state,
                        target_accuracy=args.target_accuracy,
                        full_rows=args.full_rows,
                        baseline_correct=args.baseline_correct,
                    ),
                    flush=True,
                )
                last_line_count = int(state["line_count"])
        except Exception as exc:
            print(f"KG1_LIVE_MONITOR_WAIT error={type(exc).__name__}: {exc}", flush=True)

        if args.once:
            return 0
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

