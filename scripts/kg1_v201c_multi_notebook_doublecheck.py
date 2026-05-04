#!/usr/bin/env python3
"""Double-check the V201C three-candidate Colab notebook contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FRAGMENTS = [
    "KG1 V201C H100/A100 three-candidate micro-train",
    "V194_RANK19_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'",
    "V194_RANK19_ADAPTER_MODEL_SHA256 = '01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f'",
    "V194_RANK19_BOOTSTRAP_TARGET = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V201/baseline_v194_rank19/submission.zip')",
    "pathlib.Path('/content/drive/MyDrive/Submit/submission.zip')",
    "pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199/baseline_v194_rank19/submission.zip')",
    "DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V201')",
    "V199_DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199')",
    "BEST_RANKING_BASELINE_RULE",
    "assert BEST_RANKING_BASELINE['rank'] == '19/2613'",
    "OUT_BASE = DRIVE_ROOT / 'output_v201c_h100_a100_multicandidate_3x'",
    "ALLOW_V194_REBUILD_FALLBACK = False",
    "Use H100 or A100 80GB High-RAM",
    "gpu_mem_mib >= 75000",
    "V201C_OUT",
    "A_neutral_shuffle_3s",
    "B_equation_crypt_low_2s",
    "C_bit_cipher_low_2s",
    "'MODEL_REVISION': 'cbd3fa9f933d55ef16a84236559f4ee2a0526848'",
    "'MAX_FINAL_EVAL_REGRESSION': '0.0'",
    "'BASELINE_EVAL_BEFORE_TRAIN': '1'",
    "'REQUIRE_FINAL_EVAL_LTE_BASELINE': '1'",
    "'TRAINABLE_LORA_MODULES': 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'",
    "TRAIN_SCRIPT.parent.mkdir(parents=True, exist_ok=True)",
    "'sampling_mode': 'shuffle'",
    "'sampling_mode': 'weighted_replacement'",
    "equation_transform=1.15",
    "cryptarithm_deduce=1.25",
    "bit_manipulation=1.20",
    "cipher=1.20",
    "kg1_convert_local_training_adapter_to_kaggle_zip.py",
    "kg1_v201c_posttrain_gate.py",
    "v201c_candidates_summary.json",
    "v201c_final_selection.json",
    "No Kaggle submit was performed.",
]


FORBIDDEN_FRAGMENTS = [
    "files.upload",
    "kaggle competitions submit",
    "KaggleApi",
    "kernels_output",
    "kg1_v199b_safe_kaggle_submit.py",
    "--submit",
    "V198_FINAL_ADAPTER_SHA256",
    "ABORT_EVAL_LOSS_GT'] = '0.98'",
    "output_v201a_h100_solver_verified_micro_5",
    "output_v201b_h100_baseline_neutral_micro_3",
    "kg1_v201a_posttrain_gate.py",
    "kg1_v201b_posttrain_gate.py",
    "bit_manipulation=2.5",
    "v198_v196_wrong_anti_regression=2.0",
    "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5",
    "SOURCE_WEIGHTS'] = 'v198_v196_wrong_anti_regression:2.0",
    "ALLOW_V194_REBUILD_FALLBACK=1",
    "V201C fallback rebuild requested",
    "Rebuilding exact V194 rank-19 adapter",
]


@dataclass
class Finding:
    severity: str
    code: str
    detail: str


def read_notebook_source(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source") or []) for cell in data.get("cells", []))


def add(findings: list[Finding], severity: str, code: str, detail: str) -> None:
    findings.append(Finding(severity=severity, code=code, detail=detail))


def gate_source(source: str, findings: list[Finding]) -> None:
    for fragment in REQUIRED_FRAGMENTS:
        if fragment not in source:
            add(findings, "error", "missing_required_fragment", fragment)

    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in source:
            add(findings, "error", "forbidden_fragment", fragment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []
    if not args.notebook.exists():
        add(findings, "error", "path_missing", str(args.notebook))
    else:
        gate_source(read_notebook_source(args.notebook), findings)

    payload = {
        "decision": "PASS" if not any(item.severity == "error" for item in findings) else "FAIL",
        "notebook": str(args.notebook),
        "findings": [item.__dict__ for item in findings],
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
