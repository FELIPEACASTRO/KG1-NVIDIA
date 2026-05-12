#!/usr/bin/env python3
"""Launch V307 V304 solver-trace distillation from V290 checkpoint-6."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v307_v304_solver_trace_distill_from_v290_checkpoint6_h200"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
EXPECTED_COMMIT = "a68064e5ebb983a8a131579e9ea4f581bb5932d4"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
FLAVOR = "h200"
RUN_ID = "v307-v304-solver-trace-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_UPLOAD_COMMIT = "41b8afa26ed15ec65a842e00fd0de059fe105526"
TRAIN_FILE = "data/v304_solver_trace_distill/20260512T1430Z/v304_solver_trace_distill_train.jsonl"
VAL_FILE = "data/v304_solver_trace_distill/20260512T1430Z/v304_solver_trace_distill_val.jsonl"
TRAIN_SHA256 = "7935ff999cdd8318de67538922de3651170c59baa2664a10beac3334dfcf9082"
VAL_SHA256 = "2b06224afe035c5085798f4a4be27e764ffaebde3ff7eee11c558c0cd5bdd29d"
TRAIN_ROWS = 12822
VAL_ROWS = 969

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v307-v304-solver-trace-v290ckpt6"

MAX_STEPS = 12
SAVE_EVERY_STEPS = 3
EVAL_EVERY_STEPS = 3
EVAL_MAX_EXAMPLES = 256


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
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
export MODEL_REVISION='cbd3fa9f933d55ef16a84236559f4ee2a0526848'
export DATA_REPO='felipesp1983/kg1-nemotron-training'
export DATA_FILE="$KG1_TRAIN_FILE"
export VAL_FILE="$KG1_VAL_FILE"
export EXPECTED_TRAIN_SHA256="$KG1_TRAIN_SHA"
export EXPECTED_VAL_SHA256="$KG1_VAL_SHA"
export MIN_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
export MIN_VAL_EXAMPLES="$KG1_VAL_ROWS"
export MIN_TOKENIZED_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
export MIN_TOKENIZED_VAL_EXAMPLES="$KG1_VAL_ROWS"
export OUTPUT_DIR='/tmp/kg1_v307_output'
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
export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'
export TRAINABLE_LORA_MODULES='lm_head'
export MAX_TRAINABLE_PARAM_RATIO=0.005
export MAX_LENGTH=1024
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=6e-9
export FINAL_LEARNING_RATE=2e-9
export NUM_EPOCHS=1
export MAX_STEPS=12
export SAVE_EVERY_STEPS=3
export EVAL_EVERY_STEPS=3
export EVAL_MAX_EXAMPLES=256
export LOG_EVERY_STEPS=1
export MICRO_LOG_EVERY=0
export BASELINE_EVAL_BEFORE_TRAIN=1
export REQUIRE_FINAL_EVAL_LTE_BASELINE=0
export ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA=0.05
export MAX_FINAL_EVAL_REGRESSION=0
export ABORT_TRAIN_RISE_POINTS=0
export ABORT_MAX_RESERVED_GIB=0
export SAMPLING_MODE='weighted_replacement'
export SUBCATEGORY_WEIGHTS='bit_fullbyte_v300_gain_pattern=12.00,bit_fullbyte_safe_ternary=8.00,bit_fullbyte_binary=6.00,equation_numeric_add_direct=3.00,equation_numeric_colon_absdiff=3.00,equation_numeric_minus_signed=3.00,equation_transform=1.10,bit_manipulation=1.25'
export SOURCE_WEIGHTS='v304_solver_trace_bit_fullbyte_distill_exact=12.00,v304_solver_trace_bit_fullbyte_distill_random=6.00,v282_v274_rule_synthetic=3.00,v216_base_clean_safe_strict_equation=1.10,v216_synthetic_kg1_symbolic_rules=1.05,v216_synthetic_kg1_numeric_rules=1.05,v216_base_clean_safe_strict_bit=1.25,v216_synthetic_kg1_bit_rules=1.25,v215_replay_anchor=0.85'
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
python scripts/hf_job_train_v90.py
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
        raise RuntimeError("HF token is required to launch V307.")
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
        "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform,gravity_constant,numeral_system,text_encryption,unit_conversion",
        "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform,gravity_constant,numeral_system,text_encryption,unit_conversion",
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": "bit_fullbyte_v300_gain_pattern,bit_fullbyte_safe_ternary,bit_fullbyte_binary,equation_numeric_add_direct,equation_numeric_colon_absdiff,equation_numeric_minus_signed",
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_RUN_ID": RUN_ID,
        "KG1_TRAIN_FILE": TRAIN_FILE,
        "KG1_VAL_FILE": VAL_FILE,
        "KG1_TRAIN_SHA": TRAIN_SHA256,
        "KG1_VAL_SHA": VAL_SHA256,
        "KG1_TRAIN_ROWS": str(TRAIN_ROWS),
        "KG1_VAL_ROWS": str(VAL_ROWS),
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
        timeout=7200,
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
            "dataset_upload_commit": DATASET_UPLOAD_COMMIT,
            "train_file": TRAIN_FILE,
            "val_file": VAL_FILE,
            "train_sha256": TRAIN_SHA256,
            "val_sha256": VAL_SHA256,
            "train_rows": TRAIN_ROWS,
            "val_rows": VAL_ROWS,
        },
        "recipe": {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_examples": EVAL_MAX_EXAMPLES,
            "trainable_lora_modules": "lm_head",
            "learning_rate": "6e-9",
            "final_learning_rate": "2e-9",
            "sampling_mode": "weighted_replacement",
            "continuation_from": f"{INIT_ADAPTER_REPO}/{INIT_ADAPTER_SUBFOLDER}",
            "objective": "distill V304 solver traces and V306 +15 local signal into adapter-only behavior",
            "continuation_gate": "continue only if weak bit >135 or equation >56 without side-family regression",
        },
        "gates": [
            "hf_job_preflight_gate:preinstall",
            "hf_job_preflight_gate:artifacts",
            "hf_job_preflight_gate:postinstall",
            "hf_job_train_v90 tokenization/truncation/offset-mask gate",
        ],
        "next_action": "Monitor every 30 seconds; weak-eval checkpoint-3/6/9/12/final if training completes.",
    }
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
