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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import MODEL_NAME, MODEL_REVISION, PROMPT_SUFFIX  # noqa: E402


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
OFFICIAL_LIKE_SCHEMA_VERSION = "kg1_v284_hf_official_like_eval_gate_manifest_v1"
EXPECTED_FULL_ROWS = 947
EXPECTED_FULL_ROW_CONTRACT_SHA256 = "5441932fc270eb9621a32b4d7e85ff444c45aa31d75e2bb7aea0de96cd638f21"
OFFICIAL_MAX_TOKENS = 7680
OFFICIAL_MAX_MODEL_LEN = 8192
OFFICIAL_MAX_NUM_SEQS = 64
OFFICIAL_GPU_MEMORY_UTILIZATION = 0.85


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {observed!r}")


def _require_float_equal(label: str, observed: Any, expected: float, tolerance: float = 1e-9) -> None:
    observed_float = float(observed)
    if abs(observed_float - expected) > tolerance:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {observed_float!r}")


def _find_manifest_adapter(payload: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    adapters = payload.get("adapters", [])
    if not isinstance(adapters, list) or not adapters:
        raise RuntimeError("full manifest missing non-empty adapters list")
    matches = [item for item in adapters if isinstance(item, dict) and str(item.get("candidate_name", "")) == candidate_name]
    if not matches and len(adapters) == 1 and isinstance(adapters[0], dict):
        matches = [adapters[0]]
    if len(matches) != 1:
        raise RuntimeError(f"could not uniquely match best candidate {candidate_name!r} to manifest adapter")
    return matches[0]


def validate_full_manifest(
    path: Path | None,
    min_correct: int,
    max_trunc: int,
    *,
    expected_repo: str,
    expected_subfolder: str,
    expected_revision: str,
    expected_repo_commit: str,
    expected_full_row_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        raise RuntimeError("full manifest is required before packaging an adapter-only submission")
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_json(path)
    _require_equal("full manifest schema_version", payload.get("schema_version"), OFFICIAL_LIKE_SCHEMA_VERSION)
    manifest_repo_commit = str(payload.get("repo_commit", "")).strip()
    if not manifest_repo_commit:
        raise RuntimeError("full manifest missing repo_commit")
    if expected_repo_commit:
        _require_equal("full manifest repo_commit", manifest_repo_commit, expected_repo_commit)
    best = payload.get("best_full_candidate", {})
    if not isinstance(best, dict):
        raise RuntimeError("full manifest missing best_full_candidate object")
    correct = int(best.get("correct", -1))
    truncated = int(best.get("truncated", 999999))
    gate = payload.get("full_candidate_gate", False)
    if correct < min_correct:
        raise RuntimeError(f"full candidate below required correct floor: {correct} < {min_correct}")
    if truncated > max_trunc:
        raise RuntimeError(f"full candidate above truncation cap: {truncated} > {max_trunc}")
    if gate is not True:
        raise RuntimeError("full manifest full_candidate_gate is false")
    if str(best.get("status", "")) != "ok":
        raise RuntimeError(f"full manifest best candidate is not ok: {best.get('status')!r}")
    controls = payload.get("eval_prompt_controls", {})
    if str(controls.get("prediction_postprocessor", "none")) not in {"", "none"}:
        raise RuntimeError("submission package cannot rely on external prediction postprocessor")
    _require_equal("disable_thinking", bool(controls.get("disable_thinking", False)), False)
    _require_equal("no_prompt_suffix", bool(controls.get("no_prompt_suffix", False)), False)
    _require_equal("prompt_suffix", str(controls.get("prompt_suffix", "")), PROMPT_SUFFIX)

    official_like_control_gate = payload.get("official_like_control_gate")
    if not isinstance(official_like_control_gate, dict):
        raise RuntimeError("full manifest missing official_like_control_gate")
    _require_equal("official-like strict", bool(official_like_control_gate.get("strict", False)), True)
    _require_equal("official-like max_tokens", int(official_like_control_gate.get("max_tokens", -1)), OFFICIAL_MAX_TOKENS)
    _require_equal(
        "official-like max_model_len",
        int(official_like_control_gate.get("max_model_len", -1)),
        OFFICIAL_MAX_MODEL_LEN,
    )
    _require_equal("official-like max_num_seqs", int(official_like_control_gate.get("max_num_seqs", -1)), OFFICIAL_MAX_NUM_SEQS)
    _require_float_equal(
        "official-like gpu_memory_utilization",
        official_like_control_gate.get("gpu_memory_utilization", -1),
        OFFICIAL_GPU_MEMORY_UTILIZATION,
    )

    config = (payload.get("candidate_summary") or {}).get("config") or {}
    _require_equal("official max_tokens", int(config.get("max_tokens", -1)), OFFICIAL_MAX_TOKENS)
    _require_equal("official max_model_len", int(config.get("max_model_len", -1)), OFFICIAL_MAX_MODEL_LEN)
    _require_equal("official max_num_seqs", int(config.get("max_num_seqs", -1)), OFFICIAL_MAX_NUM_SEQS)
    _require_equal("official model_name", str(config.get("model_name", "")), MODEL_NAME)
    _require_equal("official model_revision", str(config.get("model_revision", "")), MODEL_REVISION)
    _require_equal("official temperature", float(config.get("temperature", -1)), 0.0)
    _require_equal("official top_p", float(config.get("top_p", -1)), 1.0)
    _require_float_equal("official gpu_memory_utilization", config.get("gpu_memory_utilization", -1), OFFICIAL_GPU_MEMORY_UTILIZATION)

    full_csv = payload.get("full_csv") or {}
    _require_equal("full rows", int(full_csv.get("rows", -1)), EXPECTED_FULL_ROWS)
    if expected_full_row_contract_sha256:
        _require_equal(
            "full row contract",
            str(full_csv.get("row_contract_sha256", "")),
            expected_full_row_contract_sha256,
        )

    adapter_meta = _find_manifest_adapter(payload, str(best.get("name", "")))
    _require_equal("adapter repo", str(adapter_meta.get("repo", "")), expected_repo)
    _require_equal("adapter subfolder", str(adapter_meta.get("subfolder", "")).strip("/"), expected_subfolder.strip("/"))
    resolved_revision = str(adapter_meta.get("resolved_revision") or adapter_meta.get("revision") or "").strip()
    if expected_revision:
        if expected_revision not in {resolved_revision, str(adapter_meta.get("revision", "")).strip()}:
            raise RuntimeError(
                "adapter revision mismatch: "
                f"expected {expected_revision!r}, manifest has revision={adapter_meta.get('revision')!r} "
                f"resolved_revision={adapter_meta.get('resolved_revision')!r}"
            )
    if not resolved_revision:
        raise RuntimeError("full manifest adapter is missing immutable revision/resolved_revision")
    for key in ("adapter_config_sha256", "adapter_model_sha256"):
        if not str(adapter_meta.get(key, "")).strip():
            raise RuntimeError(f"full manifest adapter missing {key}")
    return payload, adapter_meta


def validate_adapter(
    adapter_dir: Path,
    expected_r: int,
    expected_alpha: int,
    *,
    expected_config_sha256: str = "",
    expected_model_sha256: str = "",
) -> dict[str, Any]:
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
    config_sha = sha256_file(adapter_dir / "adapter_config.json")
    model_sha = sha256_file(weight_path)
    if expected_config_sha256 and config_sha != expected_config_sha256:
        raise RuntimeError(f"adapter_config sha mismatch: expected {expected_config_sha256}, got {config_sha}")
    if expected_model_sha256 and model_sha != expected_model_sha256:
        raise RuntimeError(f"adapter_model sha mismatch: expected {expected_model_sha256}, got {model_sha}")
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "adapter_model_bytes": int(weight_path.stat().st_size),
        "r": int(config.get("r", -1)),
        "lora_alpha": int(config.get("lora_alpha", -1)),
        "target_modules": config.get("target_modules"),
        "target_parameters": config.get("target_parameters"),
    }


def download_adapter(repo: str, subfolder: str, output_dir: Path, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        filename = f"{subfolder.strip('/')}/{name}" if subfolder.strip("/") else name
        src = Path(hf_hub_download(repo_id=repo, repo_type="model", filename=filename, revision=revision))
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
    parser.add_argument("--revision", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-manifest-json", type=Path, required=True)
    parser.add_argument("--min-full-correct", type=int, default=831)
    parser.add_argument("--max-full-trunc", type=int, default=4)
    parser.add_argument("--expected-full-row-contract-sha256", default=EXPECTED_FULL_ROW_CONTRACT_SHA256)
    parser.add_argument("--expected-repo-commit", default=os.environ.get("KG1_EXPECTED_COMMIT", "").strip())
    parser.add_argument("--expected-r", type=int, default=32)
    parser.add_argument("--expected-alpha", type=int, default=32)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--message", default="KG1 gated adapter-only submission")
    args = parser.parse_args()

    print("=== KG1 PACKAGE HF ADAPTER SUBMISSION START ===", flush=True)
    print("repo =", args.repo, flush=True)
    print("subfolder =", args.subfolder, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    full_manifest, manifest_adapter = validate_full_manifest(
        args.full_manifest_json,
        args.min_full_correct,
        args.max_full_trunc,
        expected_repo=args.repo,
        expected_subfolder=args.subfolder,
        expected_revision=args.revision,
        expected_repo_commit=args.expected_repo_commit,
        expected_full_row_contract_sha256=args.expected_full_row_contract_sha256,
    )
    revision = args.revision or str(manifest_adapter.get("resolved_revision") or manifest_adapter.get("revision"))
    adapter_dir = download_adapter(args.repo, args.subfolder, args.output_dir, revision)
    adapter_meta = validate_adapter(
        adapter_dir,
        args.expected_r,
        args.expected_alpha,
        expected_config_sha256=str(manifest_adapter.get("adapter_config_sha256", "")),
        expected_model_sha256=str(manifest_adapter.get("adapter_model_sha256", "")),
    )
    zip_meta = create_zip(adapter_dir, args.output_dir / "submission.zip")
    manifest = {
        "schema_version": "kg1_package_hf_adapter_submission_v1",
        "repo": args.repo,
        "subfolder": args.subfolder,
        "revision": revision,
        "adapter": adapter_meta,
        "submission_zip": zip_meta,
        "full_manifest_json": str(args.full_manifest_json) if args.full_manifest_json else "",
        "full_manifest_gate": {
            "min_full_correct": args.min_full_correct,
            "max_full_trunc": args.max_full_trunc,
            "expected_full_row_contract_sha256": args.expected_full_row_contract_sha256,
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
