#!/usr/bin/env python3
"""Double-check the V201A H100 solver-verified micro-train notebook contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FRAGMENTS = [
    "KG1 V201A H100 solver-verified weak-category micro-train",
    "V194_RANK19_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'",
    "V194_RANK19_ADAPTER_MODEL_SHA256 = '01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f'",
    "V194_RANK19_BOOTSTRAP_TARGET = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199/baseline_v194_rank19/submission.zip')",
    "pathlib.Path('/content/drive/MyDrive/Submit/submission.zip')",
    "assert sha256_path_bootstrap(V194_RANK19_BOOTSTRAP_TARGET) == V194_RANK19_ZIP_SHA256",
    "BEST_RANKING_BASELINE_RULE",
    "assert BEST_RANKING_BASELINE['rank'] == '19/2613'",
    "assert BEST_RANKING_BASELINE['zip_sha256'] == V194_RANK19_ZIP_SHA256",
    "OUT_BASE = DRIVE_ROOT / 'output_v201a_h100_solver_verified_micro_5'",
    "print('V201A_OUT =', OUT)",
    "V201A_OUT'] = str(OUT)",
    "RUN_ID'] = 'v201a-h100-v194-rank19-solver-verified-micro-5s'",
    "FIXED_TRAIN_SCRIPT_URL = 'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir/scripts/hf_job_train_v90.py'",
    "'BASELINE_EVAL_BEFORE_TRAIN' not in script_text",
    "assert 'BASELINE_EVAL_BEFORE_TRAIN' in script_text",
    "assert 'REQUIRE_FINAL_EVAL_LTE_BASELINE' in script_text",
    "MODEL_NAME'] = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'",
    "INIT_ADAPTER_DIR'] = str(INIT_ADAPTER)",
    "PEFT_MANUAL_LOAD_METHOD'] = 'direct'",
    "MAX_LENGTH'] = '2048'",
    "BATCH_SIZE'] = '16'",
    "MICRO_BATCH_SIZE'] = '1'",
    "GRADIENT_CHECKPOINTING'] = '1'",
    "MAX_STEPS'] = '5'",
    "SAVE_EVERY_STEPS'] = '5'",
    "EVAL_EVERY_STEPS'] = '5'",
    "EVAL_MAX_EXAMPLES'] = '360'",
    "LEARNING_RATE'] = '3e-7'",
    "FINAL_LEARNING_RATE'] = '1e-7'",
    "ABORT_EVAL_LOSS_GT'] = '0'",
    "BASELINE_EVAL_BEFORE_TRAIN'] = '1'",
    "ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA'] = '0.005'",
    "REQUIRE_FINAL_EVAL_LTE_BASELINE'] = '1'",
    "MAX_FINAL_EVAL_REGRESSION'] = '0.0'",
    "SAMPLING_MODE'] = 'weighted_replacement'",
    "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5,cipher:2.0,cryptarithm_deduce:3.0,cryptarithm_guess:2.0,equation_numeric_deduce:3.0,equation_numeric_guess:2.0,equation_transform:1.5'",
    "SOURCE_WEIGHTS'] = 'v198_v196_wrong_anti_regression:2.0,v198_v197_strict_gain_distill:1.5,v198_v195_balanced_rehearsal:1.0'",
    "TRAINABLE_LORA_MODULES'] = 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'",
    "MAX_TRAINABLE_PARAM_RATIO'] = '0.035'",
    "kg1_v201a_posttrain_gate.py",
    "v201a_posttrain_gate_report.json",
    "assert primary_label == 'final'",
    "V201A gated candidate ready. No Kaggle submit was performed.",
]


FORBIDDEN_FRAGMENTS = [
    "files.upload",
    "kaggle competitions submit",
    "KaggleApi",
    "kernels_output",
    "kg1_v199b_safe_kaggle_submit.py",
    "--submit",
    "EXPECTED_ZIP_SHA256 = '19fa057ad55c569c490650b430b76809cacc18fd5f5fbb8f6c5bf8f65785e59c'",
    "v199-conservative-v198-final-20s",
    "V198_FINAL_ADAPTER_SHA256",
    "ABORT_EVAL_LOSS_GT'] = '0.98'",
    "LEARNING_RATE'] = '3e-6'",
    "FINAL_LEARNING_RATE'] = '8e-7'",
    "MAX_STEPS'] = '20'",
    "MAX_STEPS'] = '10'",
    "kg1_v199_posttrain_gate.py",
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
