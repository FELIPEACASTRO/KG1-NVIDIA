#!/usr/bin/env python3
"""Double-check the V199B H100 baseline-gated Colab notebook contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FRAGMENTS = [
    "KG1 V199B H100 baseline-gated continuation",
    "V194_RANK19_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'",
    "V194_RANK19_BOOTSTRAP_TARGET = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199/baseline_v194_rank19/submission.zip')",
    "pathlib.Path('/content/drive/MyDrive/Submit/submission.zip')",
    "assert sha256_path_bootstrap(V194_RANK19_BOOTSTRAP_TARGET) == V194_RANK19_ZIP_SHA256",
    "DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199')",
    "ROOT = pathlib.Path('/content/kg1_v199')",
    "OUT_BASE = DRIVE_ROOT / 'output_v199b_h100_baseline_gated_10'",
    "BEST_RANKING_BASELINE_RULE",
    "assert BEST_RANKING_BASELINE['rank'] == '19/2613'",
    "assert BEST_RANKING_BASELINE['adapter_model_sha256'] == V194_RANK19_ADAPTER_MODEL_SHA256",
    "assert BEST_RANKING_BASELINE['zip_sha256'] == V194_RANK19_ZIP_SHA256",
    "PACK_SHA256 = 'e61908c0f75018b0d265c3668600170f6fa99a1a4d559508f489cba9cd6b7c93'",
    "MASTER_PACK_SHA256 = '7e3e41b55bb6f5736c3d5325c7b481f3b52ac918eb13c311e9a343f43f6dedca'",
    "EXPECTED_TRAIN_SHA256",
    "EXPECTED_VAL_SHA256",
    "assert (ROOT / 'data/v198/v198_micro_train.strict.jsonl').exists()",
    "assert (ROOT / 'data/v198/v198_micro_val.strict.jsonl').exists()",
    "assert 'H100' in gpu_name",
    "assert gpu_mem_mib >= 75000",
    "assert host_mem_gib >= 50",
    "assert disk_free_gib >= 100",
    "transformers==5.7.0",
    "accelerate==1.13.0",
    "peft==0.19.1",
    "datasets==4.8.5",
    "safetensors==0.7.0",
    "huggingface_hub==1.13.0",
    "sentencepiece==0.2.1",
    "protobuf==7.34.1",
    "causal-conv1d==1.6.1",
    "mamba-ssm==2.3.1",
    "kagglesdk==0.1.23",
    "kagglehub==1.0.1",
    "assert importlib.util.find_spec('torchao') is None",
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
    "MAX_STEPS'] = '10'",
    "SAVE_EVERY_STEPS'] = '5'",
    "EVAL_EVERY_STEPS'] = '5'",
    "EVAL_MAX_EXAMPLES'] = '360'",
    "LEARNING_RATE'] = '1e-6'",
    "FINAL_LEARNING_RATE'] = '3e-7'",
    "ABORT_EVAL_LOSS_GT'] = '0'",
    "BASELINE_EVAL_BEFORE_TRAIN'] = '1'",
    "ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA'] = '0.02'",
    "REQUIRE_FINAL_EVAL_LTE_BASELINE'] = '1'",
    "MAX_FINAL_EVAL_REGRESSION'] = '0.0'",
    "TRAINABLE_LORA_MODULES'] = 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'",
    "MAX_TRAINABLE_PARAM_RATIO'] = '0.035'",
    "kg1_v199_posttrain_gate.py",
    "nemotron_submission_preflight.py",
    "kg1_submission_gate.py",
    "kg1_v198_final_submit_doublecheck.py",
    "assert primary_label == 'final'",
    "No Kaggle submit was performed",
]


FORBIDDEN_FRAGMENTS = [
    "files.upload",
    "kaggle competitions submit",
    "KaggleApi",
    "kernels_output",
    "felipe1983/tinker-adapter-to-ready-to-submit-adapter",
    "V198_FINAL_ADAPTER_SHA256",
    "v199-conservative-v198-final-20s",
    "ABORT_EVAL_LOSS_GT'] = '0.98'",
    "LEARNING_RATE'] = '3e-6'",
    "FINAL_LEARNING_RATE'] = '8e-7'",
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
