#!/usr/bin/env python3
"""Create a Kaggle adapter-only submission zip from a gated HF adapter.

This script is intentionally conservative. It downloads exactly
`adapter_config.json` and `adapter_model.safetensors` from a Hugging Face model
repo/subfolder, validates the adapter contract, optionally validates a full-eval
manifest, and writes a root-level `submission.zip`.

It does not submit to Kaggle unless `--submit` is passed and
`KG1_ALLOW_KAGGLE_SUBMIT=1` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_full_manifest(path: Path | None, min_correct: int, max_trunc: int) -> dict[str, Any]:
    if path is None:
        raise RuntimeError("full manifest is required before packaging an adapter-only submission")
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_json(path)
    best = payload.get("best_full_candidate", {})
    if not isinstance(best, dict):
        raise RuntimeError("full manifest missing best_full_candidate object")
    correct = int(best.get("correct", -1))
    truncated = int(best.get("truncated", 999999))
    gate = bool(payload.get("full_candidate_gate", False))
    if correct < min_correct:
        raise RuntimeError(f"full candidate below required correct floor: {correct} < {min_correct}")
    if truncated > max_trunc:
        raise RuntimeError(f"full candidate above truncation cap: {truncated} > {max_trunc}")
    if not gate:
        raise RuntimeError("full manifest full_candidate_gate is false")
    controls = payload.get("eval_prompt_controls", {})
    if str(controls.get("prediction_postprocessor", "none")) not in {"", "none"}:
        raise RuntimeError("submission package cannot rely on external prediction postprocessor")
    return payload


def validate_adapter(adapter_dir: Path, expected_r: int, expected_alpha: int) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError("missing adapter files: " + json.dumps(missing))
    config = read_json(adapter_dir / "adapter_config.json")
    if int(config.get("r", -1)) != expected_r:
        raise RuntimeError(f"adapter r mismatch: expected {expected_r}, got {config.get('r')}")
    if int(config.get("lora_alpha", -1)) != expected_alpha:
        raise RuntimeError(f"adapter alpha mismatch: expected {expected_alpha}, got {config.get('lora_alpha')}")
    weight_path = adapter_dir / "adapter_model.safetensors"
    if weight_path.stat().st_size <= 0:
        raise RuntimeError("adapter_model.safetensors is empty")
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(weight_path),
        "adapter_model_bytes": int(weight_path.stat().st_size),
        "r": int(config.get("r", -1)),
        "lora_alpha": int(config.get("lora_alpha", -1)),
        "target_modules": config.get("target_modules"),
        "target_parameters": config.get("target_parameters"),
    }


def download_adapter(repo: str, subfolder: str, output_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        filename = f"{subfolder.strip('/')}/{name}" if subfolder.strip("/") else name
        src = Path(hf_hub_download(repo_id=repo, repo_type="model", filename=filename))
        dst = adapter_dir / name
        dst.write_bytes(src.read_bytes())
        print("downloaded_adapter_file =", dst, "bytes =", dst.stat().st_size, flush=True)
    return adapter_dir


def create_zip(adapter_dir: Path, zip_path: Path) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(adapter_dir / name, name)
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
    if sorted(names) != sorted(REQUIRED_FILES):
        raise RuntimeError("submission zip has wrong entries: " + json.dumps(names))
    return {
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_bytes": int(zip_path.stat().st_size),
        "zip_entries": names,
    }


def maybe_submit(zip_path: Path, message: str) -> None:
    allowed = os.environ.get("KG1_ALLOW_KAGGLE_SUBMIT", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not allowed:
        raise RuntimeError("Kaggle submit locked. Set KG1_ALLOW_KAGGLE_SUBMIT=1 only after manual approval.")
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(zip_path),
        "-m",
        message,
    ]
    print("kaggle_submit_cmd =", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=600, check=False)
    print(proc.stdout, flush=True)
    if proc.returncode:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"kaggle submit failed rc={proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--subfolder", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-manifest-json", type=Path, required=True)
    parser.add_argument("--min-full-correct", type=int, default=823)
    parser.add_argument("--max-full-trunc", type=int, default=4)
    parser.add_argument("--expected-r", type=int, default=32)
    parser.add_argument("--expected-alpha", type=int, default=32)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--message", default="KG1 gated adapter-only submission")
    args = parser.parse_args()

    print("=== KG1 PACKAGE HF ADAPTER SUBMISSION START ===", flush=True)
    print("repo =", args.repo, flush=True)
    print("subfolder =", args.subfolder, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    full_manifest = validate_full_manifest(args.full_manifest_json, args.min_full_correct, args.max_full_trunc)
    adapter_dir = download_adapter(args.repo, args.subfolder, args.output_dir)
    adapter_meta = validate_adapter(adapter_dir, args.expected_r, args.expected_alpha)
    zip_meta = create_zip(adapter_dir, args.output_dir / "submission.zip")
    manifest = {
        "schema_version": "kg1_package_hf_adapter_submission_v1",
        "repo": args.repo,
        "subfolder": args.subfolder,
        "adapter": adapter_meta,
        "submission_zip": zip_meta,
        "full_manifest_json": str(args.full_manifest_json) if args.full_manifest_json else "",
        "full_manifest_gate": {
            "min_full_correct": args.min_full_correct,
            "max_full_trunc": args.max_full_trunc,
            "validated": True,
        },
        "kaggle_submit_attempted": bool(args.submit),
    }
    manifest_path = args.output_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("package_manifest_path =", manifest_path, flush=True)
    if args.submit:
        maybe_submit(args.output_dir / "submission.zip", args.message)
    print("=== KG1 PACKAGE HF ADAPTER SUBMISSION END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
