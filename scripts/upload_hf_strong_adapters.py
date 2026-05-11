#!/usr/bin/env python3
"""Upload validated KG1 strong adapters to a private Hugging Face model repo.

This intentionally uploads file-by-file so an interruption does not discard
already committed files.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, create_repo


UPLOAD_ORDER = [
    "README.md",
    "strong_adapters_validation_manifest.json",
    "v226_checkpoint1/adapter_config.json",
    "v226_checkpoint1/README.md",
    "v226_checkpoint1/chat_template.jinja",
    "v226_checkpoint1/tokenizer_config.json",
    "v226_checkpoint1/tokenizer.json",
    "v226_checkpoint1/adapter_model.safetensors",
    "v194_protected/adapter_config.json",
    "v194_protected/adapter_model.safetensors",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage = Path(args.stage_dir)
    if not stage.is_dir():
        raise FileNotFoundError(stage)

    manifest_path = stage / "strong_adapters_validation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed = [row for row in manifest.get("adapters", []) if not row.get("validation_pass")]
    if failed:
        raise RuntimeError(f"local validation failed for: {[row.get('name') for row in failed]}")

    api = HfApi()
    print(f"[{utc_now()}] create_or_reuse_repo repo_id={args.repo_id} private={args.private}", flush=True)
    create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    remote_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))
    print(f"[{utc_now()}] remote_file_count_before={len(remote_files)}", flush=True)

    for rel in UPLOAD_ORDER:
        local = stage / rel
        if not local.is_file():
            raise FileNotFoundError(local)
        if args.skip_existing and rel in remote_files:
            print(f"[{utc_now()}] skip_existing path={rel}", flush=True)
            continue
        size = local.stat().st_size
        print(f"[{utc_now()}] upload_start path={rel} bytes={size}", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload {rel}",
        )
        remote_files.add(rel)
        print(f"[{utc_now()}] upload_done path={rel}", flush=True)

    final_files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type)
    print(f"[{utc_now()}] remote_file_count_after={len(final_files)}", flush=True)
    for rel in UPLOAD_ORDER:
        print(f"[{utc_now()}] remote_has {rel}={rel in set(final_files)}", flush=True)
    print(f"[{utc_now()}] upload_complete repo_id={args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
