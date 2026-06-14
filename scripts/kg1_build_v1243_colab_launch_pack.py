#!/usr/bin/env python3
"""Build a minimal V1243 Colab launch pack with data, scripts, and docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "v1243_colab_launch_pack"
ARTIFACTS_ROOT = ROOT / "artifacts"

REQUIRED_FILES = [
    "scripts/hf_job_train_v90.py",
    "scripts/kg1_colab_realtime_runner.py",
    "scripts/kg1_colab_live_monitor.py",
    "scripts/kg1_live_log_common.py",
    "scripts/kg1_colab_v1243_launcher.py",
    "scripts/kg1_v1243_dataset_logic_audit.py",
    "artifacts/v1243_solver_to_lora_graft/v1243_hf_env_preview.json",
    "artifacts/v1243_solver_to_lora_graft/kg1_v1243_solver_to_lora_graft_manifest.json",
    "artifacts/v1243_solver_to_lora_graft/V1243_SOLVER_TO_LORA_GRAFT.md",
    "artifacts/v1243_solver_to_lora_graft/v1243_bit_specialist_train.jsonl",
    "artifacts/v1243_solver_to_lora_graft/v1243_equation_specialist_train.jsonl",
    "artifacts/v1243_solver_to_lora_graft/v1243_val170.jsonl",
    "docs/COLAB_REALTIME_LOG_MONITORING.md",
    "docs/KG1_SCORE_DETERMINANTS_AUDIT_2026_06_13.md",
]

REQUIREMENTS = [
    "accelerate>=0.34,<2",
    "bitsandbytes>=0.43,<1",
    "hf-xet>=1.5,<2",
    "huggingface_hub>=0.25,<1",
    "peft>=0.13,<1",
    "safetensors>=0.4,<1",
    "transformers>=4.45,<5",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def ensure_safe_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    artifacts_root = ARTIFACTS_ROOT.resolve()
    if resolved == ROOT.resolve() or resolved == artifacts_root:
        raise ValueError(f"refusing unsafe launch-pack output dir: {resolved}")
    if artifacts_root not in resolved.parents:
        raise ValueError(f"launch-pack output dir must stay under {artifacts_root}: {resolved}")
    return resolved


def clean_output_dir(output_dir: Path) -> None:
    output_dir = ensure_safe_output_dir(output_dir)
    if not output_dir.exists():
        return
    marker = output_dir / "kg1_v1243_colab_launch_pack_manifest.json"
    if not marker.exists():
        raise ValueError(
            "refusing to delete output dir without V1243 launch-pack marker: "
            f"{output_dir}"
        )
    shutil.rmtree(output_dir)


def clean_zip_path(zip_path: Path) -> None:
    if not zip_path.exists():
        return
    resolved = zip_path.resolve()
    artifacts_root = ARTIFACTS_ROOT.resolve()
    if artifacts_root not in resolved.parents:
        raise ValueError(f"refusing to delete zip outside artifacts root: {resolved}")
    if resolved.name != "v1243_colab_launch_pack.zip":
        raise ValueError(f"refusing to delete unexpected launch-pack zip name: {resolved}")
    zip_path.unlink()


def copy_required_files(output_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for relative in REQUIRED_FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"missing required launch-pack file: {source}")
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "path": relative,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    return copied


def write_readme(output_dir: Path) -> None:
    readme = """# KG1 V1243 Colab Launch Pack

This pack contains the minimal scripts and V1243 data artifacts needed to run
the bit/equation specialist training path with real-time logs.

## Colab Setup

The preferred Colab path is the one-cell launcher notebook, which downloads
and verifies this pack automatically. If you need the manual fallback, unzip
the pack in Colab, then run:

```bash
pip install -q -r requirements_v1243_colab.txt
```

Tokenize-only dry run:

```bash
python scripts/kg1_colab_v1243_launcher.py \
  --phase bit_specialist \
  --run-mode tokenize_dryrun \
  --target-accuracy 0.89 \
  --live-log-repo "$KG1_LIVE_LOG_HF_REPO" \
  --require-live-log-upload
```

Watchdog defaults:

```bash
export KG1_WATCHDOG_STALE_SECONDS=1800
export KG1_WATCHDOG_MAX_RUNTIME_SECONDS=0
export KG1_DISABLE_HEALTH_WATCHDOG=0
```

The runner aborts if required live-log upload fails, if no stdout progress is
seen for the stale threshold, if max runtime is exceeded, or if live health
turns STOP.

GPU model-load dry run:

```bash
export KG1_ACCEPT_GPU_SPEND=1
pip install --progress-bar off --no-build-isolation mamba-ssm==2.3.1
export INIT_ADAPTER_REPO="owner/baseline-adapter-repo"
export INIT_ADAPTER_REVISION="pinned-commit-sha"
python scripts/kg1_colab_v1243_launcher.py \
  --phase bit_specialist \
  --run-mode model_dryrun \
  --accept-gpu-spend \
  --target-accuracy 0.89 \
  --live-log-repo "$KG1_LIVE_LOG_HF_REPO" \
  --require-live-log-upload
```

Real train, only after gates and explicit decision:

```bash
export KG1_ACCEPT_GPU_SPEND=1
export INIT_ADAPTER_REPO="owner/baseline-adapter-repo"
export INIT_ADAPTER_REVISION="pinned-commit-sha"
python scripts/kg1_colab_v1243_launcher.py \
  --phase bit_specialist \
  --run-mode real_train \
  --allow-real-train \
  --accept-gpu-spend \
  --target-accuracy 0.89 \
  --live-log-repo "$KG1_LIVE_LOG_HF_REPO" \
  --require-live-log-upload \
  --output-repo "$OUTPUT_REPO"
```

Do not submit anything from this pack directly. Score claims still require
vLLM greedy raw_output generation and V1241 full947 gate.
"""
    (output_dir / "README_COLAB_LAUNCH.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    output_dir = ensure_safe_output_dir(args.output_dir)
    clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_required_files(output_dir)
    requirements_path = output_dir / "requirements_v1243_colab.txt"
    requirements_path.write_text("\n".join(REQUIREMENTS) + "\n", encoding="utf-8")
    write_readme(output_dir)

    manifest = {
        "schema_version": "kg1_v1243_colab_launch_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT),
        "output_dir": str(output_dir),
        "files": copied,
        "requirements": REQUIREMENTS,
        "decision": "pack_ready_no_gpu_no_submit",
    }
    manifest_path = output_dir / "kg1_v1243_colab_launch_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    zip_path = output_dir.with_suffix(".zip")
    clean_zip_path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))
    zip_sha256 = sha256_file(zip_path)
    zip_manifest_path = output_dir / "kg1_v1243_colab_launch_pack_zip_manifest.json"
    zip_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "kg1_v1243_colab_launch_pack_zip_manifest_v1",
                "zip_path": str(zip_path),
                "zip_sha256": zip_sha256,
                "zip_bytes": zip_path.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("[v1243-colab-pack] START", flush=True)
    print(f"[v1243-colab-pack] output_dir={output_dir}", flush=True)
    print(f"[v1243-colab-pack] zip={zip_path}", flush=True)
    print(f"[v1243-colab-pack] files={len(copied)}", flush=True)
    print(f"[v1243-colab-pack] zip_sha256={zip_sha256}", flush=True)
    print("[v1243-colab-pack] decision=pack_ready_no_gpu_no_submit", flush=True)
    print("[v1243-colab-pack] END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
