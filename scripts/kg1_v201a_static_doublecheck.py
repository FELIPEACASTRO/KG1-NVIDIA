#!/usr/bin/env python3
"""Static/network double-check for the V201A Colab notebook.

This validates the active notebook path without downloading the 63GB base
model. It checks local scripts, notebook fragments, raw GitHub dependencies,
PyPI pins, the V198 pack SHA/content, and the HF model revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NOTEBOOK = Path("notebooks/KG1_V201A_H100_SOLVER_VERIFIED_MICRO_COLAB_PRO.ipynb")
RUN_DIR = Path("runs/v200_rank_hillclimb_20260504")
BRANCH_BASE = "https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir"
PACK_URL = (
    "https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/"
    "31d439bc4a9b33b7b3c772d3526149847103a9b1/"
    "runs/v198_micro_distill_colab_pack_20260503/kg1_v198_colab_pack.zip"
)
APPROVED_PACK_SHA256 = {
    "e61908c0f75018b0d265c3668600170f6fa99a1a4d559508f489cba9cd6b7c93",
    "7e3e41b55bb6f5736c3d5325c7b481f3b52ac918eb13c311e9a343f43f6dedca",
}
REQUIRED_PACK_MEMBERS = {
    "data/v198/v198_micro_train.strict.jsonl",
    "data/v198/v198_micro_val.strict.jsonl",
    "scripts/hf_job_train_v90.py",
}
LOCAL_SCRIPTS = [
    "scripts/hf_job_train_v90.py",
    "scripts/kg1_v198_posttrain_gate.py",
    "scripts/kg1_v201a_posttrain_gate.py",
    "scripts/nemotron_submission_preflight.py",
    "scripts/kg1_submission_gate.py",
    "scripts/kg1_v198_final_submit_doublecheck.py",
    "scripts/kg1_update_space_soup_stream.py",
]
RAW_SCRIPTS = [
    "scripts/hf_job_train_v90.py",
    "scripts/kg1_v198_posttrain_gate.py",
    "scripts/kg1_v201a_posttrain_gate.py",
    "scripts/nemotron_submission_preflight.py",
    "scripts/kg1_submission_gate.py",
    "scripts/kg1_v198_final_submit_doublecheck.py",
    "scripts/kg1_update_space_soup_stream.py",
]
PYPI_PINS = {
    "transformers": "5.7.0",
    "accelerate": "1.13.0",
    "peft": "0.19.1",
    "datasets": "4.8.5",
    "safetensors": "0.7.0",
    "huggingface_hub": "1.13.0",
    "sentencepiece": "0.2.1",
    "protobuf": "7.34.1",
    "ninja": "1.13.0",
    "causal-conv1d": "1.6.1",
    "mamba-ssm": "2.3.1",
    "kagglehub": "1.0.1",
    "kagglesdk": "0.1.23",
}
REQUIRED_NOTEBOOK_FRAGMENTS = [
    "DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V201')",
    "V199_DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199')",
    "V194_RANK19_BOOTSTRAP_TARGET = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V201/baseline_v194_rank19/submission.zip')",
    "pathlib.Path('/content/drive/MyDrive/Submit/submission.zip')",
    "pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199/baseline_v194_rank19/submission.zip')",
    "V194_RANK19_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'",
    "V194_RANK19_ADAPTER_MODEL_SHA256 = '01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f'",
    "OUT_BASE = DRIVE_ROOT / 'output_v201a_h100_solver_verified_micro_5'",
    "RUN_ID'] = 'v201a-h100-v194-rank19-solver-verified-micro-5s'",
    "SAMPLING_MODE'] = 'weighted_replacement'",
    "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5,cipher:2.0,cryptarithm_deduce:3.0,cryptarithm_guess:2.0,equation_numeric_deduce:3.0,equation_numeric_guess:2.0,equation_transform:1.5'",
    "SOURCE_WEIGHTS'] = 'v198_v196_wrong_anti_regression:2.0,v198_v197_strict_gain_distill:1.5,v198_v195_balanced_rehearsal:1.0'",
    "BASELINE_EVAL_BEFORE_TRAIN'] = '1'",
    "REQUIRE_FINAL_EVAL_LTE_BASELINE'] = '1'",
    "MAX_FINAL_EVAL_REGRESSION'] = '0.0'",
    "kg1_v201a_posttrain_gate.py",
    "V201A gated candidate ready. No Kaggle submit was performed.",
]
FORBIDDEN_NOTEBOOK_FRAGMENTS = [
    "kaggle competitions submit",
    "KaggleApi",
    "files.upload",
    "V198_FINAL_ADAPTER_SHA256",
    "ABORT_EVAL_LOSS_GT'] = '0.98'",
    "MAX_STEPS'] = '20'",
    "MAX_STEPS'] = '10'",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def check_notebook() -> dict[str, Any]:
    source = "\n".join(
        "".join(cell.get("source") or [])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8")).get("cells", [])
    )
    missing = [item for item in REQUIRED_NOTEBOOK_FRAGMENTS if item not in source]
    forbidden = [item for item in FORBIDDEN_NOTEBOOK_FRAGMENTS if item in source]
    return {
        "ok": not missing and not forbidden,
        "missing": missing,
        "forbidden": forbidden,
        "cell_count": len(json.loads(NOTEBOOK.read_text(encoding="utf-8")).get("cells", [])),
    }


def check_local_scripts() -> dict[str, Any]:
    results = []
    for script in LOCAL_SCRIPTS:
        path = Path(script)
        item = {"path": script, "exists": path.exists(), "compile_ok": False}
        if path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
                item["compile_ok"] = True
            except Exception as exc:
                item["error"] = repr(exc)
        results.append(item)
    return {"ok": all(item["exists"] and item["compile_ok"] for item in results), "items": results}


def check_raw_scripts() -> dict[str, Any]:
    results = []
    for script in RAW_SCRIPTS:
        url = f"{BRANCH_BASE}/{script}"
        try:
            data = read_url(url)
            results.append(
                {
                    "path": script,
                    "url": url,
                    "ok": len(data) > 100 and (b"def main" in data or b"def train" in data),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        except Exception as exc:
            results.append({"path": script, "url": url, "ok": False, "error": repr(exc)})
    return {"ok": all(item["ok"] for item in results), "items": results}


def check_pack(download_pack: bool) -> dict[str, Any]:
    if not download_pack:
        return {"ok": True, "skipped": True, "reason": "download_pack disabled"}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "kg1_v198_colab_pack.zip"
        path.write_bytes(read_url(PACK_URL))
        digest = sha256_path(path)
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        return {
            "ok": digest in APPROVED_PACK_SHA256 and REQUIRED_PACK_MEMBERS.issubset(names),
            "url": PACK_URL,
            "sha256": digest,
            "approved_sha": digest in APPROVED_PACK_SHA256,
            "required_members": {name: name in names for name in sorted(REQUIRED_PACK_MEMBERS)},
            "bytes": path.stat().st_size,
        }


def check_pypi() -> dict[str, Any]:
    results = []
    for package, version in PYPI_PINS.items():
        url = f"https://pypi.org/pypi/{package}/json"
        try:
            payload = json.loads(read_url(url).decode("utf-8"))
            releases = payload.get("releases") or {}
            results.append({"package": package, "version": version, "ok": version in releases})
        except Exception as exc:
            results.append({"package": package, "version": version, "ok": False, "error": repr(exc)})
    return {"ok": all(item["ok"] for item in results), "items": results}


def check_hf_model_revision() -> dict[str, Any]:
    url = (
        "https://huggingface.co/api/models/"
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/"
        "revision/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
    )
    try:
        payload = json.loads(read_url(url).decode("utf-8"))
        return {"ok": payload.get("id") == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "url": url}
    except Exception as exc:
        return {"ok": False, "url": url, "error": repr(exc)}


def check_tinker_nightly() -> dict[str, Any]:
    command = [
        "git",
        "ls-remote",
        "--heads",
        "https://github.com/thinking-machines-lab/tinker-cookbook.git",
        "nightly",
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    return {
        "ok": result.returncode == 0 and "refs/heads/nightly" in result.stdout,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "note": "Fallback-only dependency; normal V201A path requires exact V194 zip and does not rebuild via Tinker.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--skip-pack-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = {
        "notebook": check_notebook(),
        "local_scripts": check_local_scripts(),
        "raw_scripts": check_raw_scripts(),
        "v198_pack": check_pack(download_pack=not args.skip_pack_download),
        "pypi_pins": check_pypi(),
        "hf_model_revision": check_hf_model_revision(),
        "tinker_nightly_fallback": check_tinker_nightly(),
        "fallback_lineage": {
            "ok": True,
            "note": "aaitdads/huikang KaggleHub components are fallback-only; production path blocks unless exact V194 zip SHA is present.",
        },
    }
    errors = [name for name, payload in checks.items() if not payload.get("ok")]
    report = {
        "generated_at": utc_now(),
        "notebook": str(NOTEBOOK),
        "decision": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "errors": errors, "output": str(args.output_json)}, indent=2))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
