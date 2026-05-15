#!/usr/bin/env python3
"""Launch V440 H200 preference smoke over V439 final-answer-only pairs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v440_v439_final_answer_only_preference_h200"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
FLAVOR = "h200"
RUN_ID = "v440-v439-final-answer-pref-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATA_ROOT = "data/v439_final_answer_only_pairs/20260515T_v439_final_answer_only"
PREF_TRAIN_FILE = f"{DATA_ROOT}/v439_final_answer_only_pairs_train.jsonl"
PREF_VAL_FILE = f"{DATA_ROOT}/v439_final_answer_only_pairs_val.jsonl"
PREF_TRAIN_SHA256 = "bc032da2f7cada19aef295aa91aef6098e03c7b85215e7729f1ddd71b3e5079a"
PREF_VAL_SHA256 = "57321347f9293e9c0f2f17e6c9de1d88f1246fee4154125574b2e60251aee3a6"
PREF_TRAIN_ROWS = 109
PREF_VAL_ROWS = 24

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v440-v439-final-answer-v290ckpt6"

MAX_STEPS = 12
SAVE_EVERY_STEPS = 3
EVAL_EVERY_STEPS = 3
EVAL_MAX_EXAMPLES = 24

REQUIRED_SUBCATEGORIES = (
    "bit_adapter_exact_wrong,"
    "equation_numeric_operator_to_number,equation_numeric_operator_to_symbolic,"
    "equation_symbolic_sequence,equation_symbolic_short"
)


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
export MAX_JOBS=8
python -m pip install -q --upgrade pip
apt-get update -qq && apt-get install -y -qq git build-essential ninja-build >/dev/null
python - <<'PY'
import json, torch
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(json.dumps({
    "torch_before": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
}), flush=True)
PY
python -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' packaging wheel setuptools
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
python -m py_compile scripts/hf_job_train_v315_preference.py scripts/hf_job_train_v90.py scripts/hf_job_preflight_gate.py scripts/kg1_static_safety_gate.py scripts/kg1_pre_paid_job_integration_gate.py
python scripts/kg1_static_safety_gate.py scripts/hf_job_train_v315_preference.py scripts/run_v435f_adapter_probe_preference_gate.py scripts/kg1_pre_paid_job_integration_gate.py artifacts/v440_hf_h200_v439_final_answer_preference_launch/launch_v440_hf_h200_v439_final_answer_preference.py
python scripts/kg1_pre_paid_job_integration_gate.py \
  --launcher artifacts/v440_hf_h200_v439_final_answer_preference_launch/launch_v440_hf_h200_v439_final_answer_preference.py \
  --train-jsonl artifacts/v439_final_answer_only_pairs/20260515T_v439_final_answer_only/v439_final_answer_only_pairs_train.jsonl \
  --val-jsonl artifacts/v439_final_answer_only_pairs/20260515T_v439_final_answer_only/v439_final_answer_only_pairs_val.jsonl \
  --v438-audit-manifest artifacts/v438_preference_objective_audit/20260515T_v438_v439_final_answer_only/v438_v439_final_answer_only_audit_manifest.json \
  --expected-data-root data/v439_final_answer_only_pairs/20260515T_v439_final_answer_only \
  --expected-train-sha256 "$KG1_PREF_TRAIN_SHA" \
  --expected-val-sha256 "$KG1_PREF_VAL_SHA" \
  --expected-train-rows "$KG1_PREF_TRAIN_ROWS" \
  --expected-val-rows "$KG1_PREF_VAL_ROWS" \
  --expected-output-repo "$KG1_OUTPUT_REPO" \
  --expected-init-adapter-repo "$KG1_INIT_ADAPTER_REPO" \
  --expected-init-adapter-subfolder "$KG1_INIT_ADAPTER_SUBFOLDER"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
export MODEL_REVISION='cbd3fa9f933d55ef16a84236559f4ee2a0526848'
export DATA_REPO='felipesp1983/kg1-nemotron-training'
export DATA_FILE="$KG1_PREF_TRAIN_FILE"
export VAL_FILE="$KG1_PREF_VAL_FILE"
export EXPECTED_TRAIN_SHA256="$KG1_PREF_TRAIN_SHA"
export EXPECTED_VAL_SHA256="$KG1_PREF_VAL_SHA"
export MIN_TRAIN_EXAMPLES="$KG1_PREF_TRAIN_ROWS"
export MIN_VAL_EXAMPLES="$KG1_PREF_VAL_ROWS"
export PREF_TRAIN_FILE="$KG1_PREF_TRAIN_FILE"
export PREF_VAL_FILE="$KG1_PREF_VAL_FILE"
export EXPECTED_PREF_TRAIN_SHA256="$KG1_PREF_TRAIN_SHA"
export EXPECTED_PREF_VAL_SHA256="$KG1_PREF_VAL_SHA"
export MIN_PREF_TRAIN_EXAMPLES="$KG1_PREF_TRAIN_ROWS"
export MIN_PREF_VAL_EXAMPLES="$KG1_PREF_VAL_ROWS"
export MIN_TOKENIZED_TRAIN_EXAMPLES="$KG1_PREF_TRAIN_ROWS"
export MIN_TOKENIZED_VAL_EXAMPLES="$KG1_PREF_VAL_ROWS"
export ALLOW_FORMAT_NEGATIVES=0
export OUTPUT_DIR='/tmp/kg1_v440_output'
export OUTPUT_REPO="$KG1_OUTPUT_REPO"
export RUN_ID="$KG1_RUN_ID"
export UPLOAD_TO_HF=1
export UPLOAD_CHECKPOINTS_DURING_TRAINING=1
export INIT_ADAPTER_REPO="$KG1_INIT_ADAPTER_REPO"
export INIT_ADAPTER_SUBFOLDER="$KG1_INIT_ADAPTER_SUBFOLDER"
export INIT_ADAPTER_LOAD_MODE='manual'
export PEFT_MANUAL_LOAD_METHOD='auto'
export FAIL_ON_MISSING_ADAPTER_KEYS=1
export LORA_R=32
export LORA_ALPHA=32
export LORA_DROPOUT=0.0
export LORA_TARGET_MODULES='down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj'
export LORA_TARGET_PARAMETERS=''
export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'
export MAX_TRAINABLE_PARAM_RATIO=0.030
export MAX_LENGTH=1536
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=2e-8
export FINAL_LEARNING_RATE=4e-9
export NUM_EPOCHS=1
export MAX_STEPS=12
export SAVE_EVERY_STEPS=3
export EVAL_EVERY_STEPS=3
export EVAL_MAX_EXAMPLES=24
export LOG_EVERY_STEPS=1
export MICRO_LOG_EVERY=0
export PREF_BETA=0.20
export PREF_MARGIN=0.0
export PREF_LOSS_WEIGHT=1.5
export CHOSEN_CE_WEIGHT=0.30
export REJECTED_CE_WEIGHT=0.0
export PAIR_SCORE_MODE='mean_nll'
export PREFERENCE_SYSTEM_PROMPT='Solve the KG1 puzzle. End with exactly one final answer in \boxed{}.'
export SAMPLING_MODE='shuffle'
export ABORT_TRAIN_RISE_POINTS=0
export ABORT_MAX_RESERVED_GIB=118
export MAX_PROMPT_TRUNCATION_RATE=0.0
export REQUIRE_OFFSET_MASK=1
export TOKENIZE_ONLY_DRY_RUN=0
export DRY_RUN_VALIDATE_ONLY=0
export USE_BITSANDBYTES=0
export MODEL_DEVICE_MAP='auto'
export ATTN_IMPLEMENTATION='eager'
export TORCH_ALLOW_TF32=1
export TORCH_FLOAT32_MATMUL_PRECISION='high'
export GRADIENT_CHECKPOINTING=1
python scripts/hf_job_preflight_gate.py --phase preinstall
python scripts/hf_job_preflight_gate.py --phase artifacts
python -m pip install -q --no-cache-dir 'transformers>=4.56.0' 'peft>=0.17.0' 'accelerate>=1.10.0' safetensors sentencepiece protobuf hf_transfer ninja einops
python -m pip install -q --no-cache-dir --no-build-isolation --no-deps --no-binary=causal-conv1d 'causal-conv1d==1.6.1'
python -m pip install -q --no-cache-dir --no-build-isolation --no-deps --no-binary=mamba-ssm 'mamba-ssm==2.3.1'
python scripts/hf_job_preflight_gate.py --phase postinstall
python scripts/hf_job_train_v315_preference.py
"""


def hardware_to_dict(item: object) -> dict[str, object]:
    accelerator = getattr(item, "accelerator", None)
    return {
        "name": str(getattr(item, "name", "")),
        "pretty_name": str(getattr(item, "pretty_name", "")),
        "cpu": str(getattr(item, "cpu", "")),
        "ram": str(getattr(item, "ram", "")),
        "accelerator_model": str(getattr(accelerator, "model", "")) if accelerator else "",
        "accelerator_quantity": str(getattr(accelerator, "quantity", "")) if accelerator else "",
        "accelerator_vram": str(getattr(accelerator, "vram", "")) if accelerator else "",
        "unit_cost_usd": float(getattr(item, "unit_cost_usd", 0.0) or 0.0),
        "unit_label": str(getattr(item, "unit_label", "")),
    }


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to launch V440.")
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if hardware[FLAVOR]["unit_cost_usd"] > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware[FLAVOR]}")

    job_env = {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_TORCH_VERSION": "2.8.0+cu128",
        "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware[FLAVOR]["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "KG1_REQUIRE_OFFSET_MASK": "1",
        "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_RUN_ID": RUN_ID,
        "KG1_PREF_TRAIN_FILE": PREF_TRAIN_FILE,
        "KG1_PREF_VAL_FILE": PREF_VAL_FILE,
        "KG1_PREF_TRAIN_SHA": PREF_TRAIN_SHA256,
        "KG1_PREF_VAL_SHA": PREF_VAL_SHA256,
        "KG1_PREF_TRAIN_ROWS": str(PREF_TRAIN_ROWS),
        "KG1_PREF_VAL_ROWS": str(PREF_VAL_ROWS),
        "KG1_INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "KG1_INIT_ADAPTER_SUBFOLDER": INIT_ADAPTER_SUBFOLDER,
    }
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware[FLAVOR], indent=2, sort_keys=True), flush=True)
    job = api.run_job(
        image=IMAGE,
        command=["/bin/bash", "-lc", COMMAND_SCRIPT],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=3600,
        namespace=NAMESPACE,
    )
    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.id,
        "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
        "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware[FLAVOR],
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "output_repo": OUTPUT_REPO,
        "init_adapter": {"repo": INIT_ADAPTER_REPO, "subfolder": INIT_ADAPTER_SUBFOLDER},
        "dataset": {
            "data_repo": DATA_REPO,
            "preference_train_file": PREF_TRAIN_FILE,
            "preference_val_file": PREF_VAL_FILE,
            "preference_train_sha256": PREF_TRAIN_SHA256,
            "preference_val_sha256": PREF_VAL_SHA256,
            "preference_train_rows": PREF_TRAIN_ROWS,
            "preference_val_rows": PREF_VAL_ROWS,
            "format_negatives_allowed": False,
        },
        "recipe": {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_examples": EVAL_MAX_EXAMPLES,
            "trainable_lora_modules": "q_proj,k_proj,v_proj,o_proj,lm_head",
            "learning_rate": "2e-8",
            "final_learning_rate": "4e-9",
            "preference_loss": "single-policy contrastive chosen/rejected plus chosen CE",
            "preference_beta": 0.20,
            "preference_loss_weight": 1.5,
            "chosen_ce_weight": 0.30,
            "sampling_mode": "shuffle",
            "continuation_from": f"{INIT_ADAPTER_REPO}/{INIT_ADAPTER_SUBFOLDER}",
            "promotion_gate": "weak eval only; promote only if bit>=136 and equation improves or total>=193",
        },
        "gates": [
            "V439 final-answer-only structural audit hf_gpu_allowed_for_same_objective=true",
            "kg1_static_safety_gate over launcher and trainer",
            "hf_job_preflight_gate:preinstall",
            "hf_job_preflight_gate:artifacts over V439 final-answer-only preference rows",
            "hf_job_preflight_gate:postinstall",
            "hf_job_train_v315_preference blocks format_negative rows",
            "hf_job_train_v315_preference tokenization/truncation/offset-mask gate for chosen and rejected",
        ],
        "next_action": "Monitor every 40 seconds; cancel if first checkpoint V439 final-answer preference metric does not improve.",
    }
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{RUN_ID}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("launch_manifest =", manifest_path, flush=True)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
