#!/usr/bin/env python3
"""Launch V315 H200 preference smoke over V312 chosen/rejected pairs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v315_v312_preference_contrastive_from_v290_checkpoint6_h200"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
FLAVOR = "h200"
RUN_ID = "v315-v312-pref-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-nemotron-training"
SFT_TRAIN_FILE = "data/v312_verifier_synthetic_distill/20260512T1545Z/v312_verifier_synthetic_distill_sft_train.jsonl"
SFT_VAL_FILE = "data/v312_verifier_synthetic_distill/20260512T1545Z/v312_verifier_synthetic_distill_sft_val.jsonl"
SFT_TRAIN_SHA256 = "352fb0cdcf8bf1505e81a2cc2c0b24bae790cfe1c1441a811d12db0b27594f5c"
SFT_VAL_SHA256 = "7e9d76277fc5adab680c5a6e53877880c14bcc6265b6469ef9e9737ee26b4153"
SFT_TRAIN_ROWS = 204
SFT_VAL_ROWS = 51

PREF_TRAIN_FILE = "data/v312_verifier_synthetic_distill/20260512T1545Z/v312_verifier_synthetic_distill_preferences_train.jsonl"
PREF_VAL_FILE = "data/v312_verifier_synthetic_distill/20260512T1545Z/v312_verifier_synthetic_distill_preferences_val.jsonl"
PREF_TRAIN_SHA256 = "f923b465a29c634f90e6d9ddf9075a0f33c1d5a2f3914ce9c725f0a18804b871"
PREF_VAL_SHA256 = "55c9632b6d65cf475b4acab952a3947249b44365b2e72cd68083aee5445c57be"
PREF_TRAIN_ROWS = 816
PREF_VAL_ROWS = 204

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v315-v312-preference-v290ckpt6"

MAX_STEPS = 16
SAVE_EVERY_STEPS = 4
EVAL_EVERY_STEPS = 4
EVAL_MAX_EXAMPLES = 204


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
python -m py_compile scripts/hf_job_train_v315_preference.py scripts/hf_job_train_v90.py scripts/hf_job_preflight_gate.py
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
export MODEL_REVISION='cbd3fa9f933d55ef16a84236559f4ee2a0526848'
export DATA_REPO='felipesp1983/kg1-nemotron-training'
export DATA_FILE="$KG1_SFT_TRAIN_FILE"
export VAL_FILE="$KG1_SFT_VAL_FILE"
export EXPECTED_TRAIN_SHA256="$KG1_SFT_TRAIN_SHA"
export EXPECTED_VAL_SHA256="$KG1_SFT_VAL_SHA"
export MIN_TRAIN_EXAMPLES="$KG1_SFT_TRAIN_ROWS"
export MIN_VAL_EXAMPLES="$KG1_SFT_VAL_ROWS"
export PREF_TRAIN_FILE="$KG1_PREF_TRAIN_FILE"
export PREF_VAL_FILE="$KG1_PREF_VAL_FILE"
export EXPECTED_PREF_TRAIN_SHA256="$KG1_PREF_TRAIN_SHA"
export EXPECTED_PREF_VAL_SHA256="$KG1_PREF_VAL_SHA"
export MIN_PREF_TRAIN_EXAMPLES="$KG1_PREF_TRAIN_ROWS"
export MIN_PREF_VAL_EXAMPLES="$KG1_PREF_VAL_ROWS"
export MIN_TOKENIZED_TRAIN_EXAMPLES="$KG1_PREF_TRAIN_ROWS"
export MIN_TOKENIZED_VAL_EXAMPLES="$KG1_PREF_VAL_ROWS"
export OUTPUT_DIR='/tmp/kg1_v315_output'
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
export MAX_TRAINABLE_PARAM_RATIO=0.025
export MAX_LENGTH=1024
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=8e-9
export FINAL_LEARNING_RATE=1e-9
export NUM_EPOCHS=1
export MAX_STEPS=16
export SAVE_EVERY_STEPS=4
export EVAL_EVERY_STEPS=4
export EVAL_MAX_EXAMPLES=204
export LOG_EVERY_STEPS=1
export MICRO_LOG_EVERY=0
export PREF_BETA=0.10
export PREF_MARGIN=0.0
export PREF_LOSS_WEIGHT=1.0
export CHOSEN_CE_WEIGHT=0.15
export REJECTED_CE_WEIGHT=0.0
export PAIR_SCORE_MODE='mean_nll'
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
export KG1_REQUIRE_MAMBA_IMPORTS=0
python scripts/hf_job_preflight_gate.py --phase preinstall
python scripts/hf_job_preflight_gate.py --phase artifacts
python -m pip install -q --no-cache-dir 'transformers>=4.56.0' 'peft>=0.17.0' 'accelerate>=1.10.0' safetensors sentencepiece protobuf hf_transfer ninja einops
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
        raise RuntimeError("HF token is required to launch V315.")
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
        "KG1_REQUIRE_MAMBA_IMPORTS": "0",
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
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
            "bit_fullbyte_v311_rule_variant,"
            "equation_numeric_add_direct,"
            "equation_numeric_colon_absdiff,"
            "equation_numeric_minus_signed"
        ),
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_RUN_ID": RUN_ID,
        "KG1_SFT_TRAIN_FILE": SFT_TRAIN_FILE,
        "KG1_SFT_VAL_FILE": SFT_VAL_FILE,
        "KG1_SFT_TRAIN_SHA": SFT_TRAIN_SHA256,
        "KG1_SFT_VAL_SHA": SFT_VAL_SHA256,
        "KG1_SFT_TRAIN_ROWS": str(SFT_TRAIN_ROWS),
        "KG1_SFT_VAL_ROWS": str(SFT_VAL_ROWS),
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
        timeout=5400,
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
            "sft_train_file": SFT_TRAIN_FILE,
            "sft_val_file": SFT_VAL_FILE,
            "preference_train_file": PREF_TRAIN_FILE,
            "preference_val_file": PREF_VAL_FILE,
            "preference_train_sha256": PREF_TRAIN_SHA256,
            "preference_val_sha256": PREF_VAL_SHA256,
            "preference_train_rows": PREF_TRAIN_ROWS,
            "preference_val_rows": PREF_VAL_ROWS,
        },
        "recipe": {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_examples": EVAL_MAX_EXAMPLES,
            "trainable_lora_modules": "q_proj,k_proj,v_proj,o_proj,lm_head",
            "learning_rate": "8e-9",
            "final_learning_rate": "1e-9",
            "preference_loss": "single-policy contrastive chosen/rejected plus chosen CE",
            "preference_beta": 0.10,
            "chosen_ce_weight": 0.15,
            "sampling_mode": "shuffle",
            "continuation_from": f"{INIT_ADAPTER_REPO}/{INIT_ADAPTER_SUBFOLDER}",
            "promotion_gate": "weak eval only; promote only if bit>=136 and equation improves or total>=193",
        },
        "gates": [
            "hf_job_preflight_gate:preinstall",
            "hf_job_preflight_gate:artifacts over V312 SFT rows",
            "hf_job_preflight_gate:postinstall",
            "hf_job_train_v315_preference preference JSON validation",
            "hf_job_train_v315_preference tokenization/truncation/offset-mask gate for chosen and rejected",
        ],
        "next_action": "Monitor every 30 seconds; run weak eval only if training completes.",
    }
    out_dir = Path(__file__).resolve().parent
    manifest_path = out_dir / f"{RUN_ID}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("launch_manifest =", manifest_path, flush=True)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
