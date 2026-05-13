#!/usr/bin/env python3
"""Launch V322 V51-filtered hybrid answer-span distillation on HF A100.

V321 showed no complementarity versus V290. V322 changes the data source by
adding V51 solver-enhanced public-train rows after filtering all known weak/full
gate rows by id and prompt hash. It keeps attention+lm_head training, but uses a
longer checkpointed run and stronger equation/bit sampling. Promotion is still
weak-gate blocked.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v322_v51_filtered_hybrid_attn_lmhead_from_v290_checkpoint6_nemo_a100"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "a100-large"
RUN_ID = "v322-nemo-a100-v51-filtered-hybrid-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_UPLOAD_COMMIT = "c17219390b6b067e2464c4eaa7246c49adb394a3"
TRAIN_FILE = "data/v322_v51_filtered_hybrid/20260513T0620Z/v322_v51_filtered_hybrid_train.jsonl"
VAL_FILE = "data/v322_v51_filtered_hybrid/20260513T0620Z/v322_v51_filtered_hybrid_val.jsonl"
TRAIN_SHA256 = "2bafab700a3f6baebb8313644785d7ddfacbf629aeee2e2e122ef0396a85c21b"
VAL_SHA256 = "dacb20c8cff73159d14336abdc5d6fa505435643184033ac2d4a85f0408ce5f1"
TRAIN_ROWS = 21154
VAL_ROWS = 1445

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v322-nemo-a100-v51-filtered-hybrid-v290ckpt6"

MAX_STEPS = 16
SAVE_EVERY_STEPS = 4
EVAL_EVERY_STEPS = 4
EVAL_MAX_EXAMPLES = 96
ANSWER_SPAN_LOSS_WEIGHT = "9.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1000"


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
export MAX_JOBS=8
PYBIN=$(command -v python || command -v python3)
echo "python_bin=$PYBIN"
apt-get update -qq && apt-get install -y -qq git >/dev/null
$PYBIN - <<'PY'
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
$PYBIN -m pip install -q --no-cache-dir --upgrade pip
$PYBIN -m pip install -q --no-cache-dir --upgrade 'huggingface_hub>=0.36.0' packaging wheel setuptools 'peft>=0.17.0' 'accelerate>=1.10.0' safetensors sentencepiece protobuf hf_transfer ninja einops
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m py_compile scripts/hf_job_train_v90.py scripts/hf_job_preflight_gate.py
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
export OUTPUT_DIR='/tmp/kg1_v322_output'
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
export TRAINABLE_LORA_NAME_SUBSTRINGS=''
export REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='q_proj,k_proj,v_proj,o_proj,lm_head'
export REQUIRE_LORA_TARGET_PARAMETER_MATCH=0
export MAX_TRAINABLE_PARAM_RATIO=0.035
export MAX_LENGTH=1024
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=2.0e-8
export FINAL_LEARNING_RATE=5e-9
export NUM_EPOCHS=1
export MAX_STEPS=16
export SAVE_EVERY_STEPS=4
export EVAL_EVERY_STEPS=4
export EVAL_MAX_EXAMPLES=96
export LOG_EVERY_STEPS=1
export MICRO_LOG_EVERY=0
export ANSWER_SPAN_LOSS_WEIGHT="$KG1_ANSWER_SPAN_LOSS_WEIGHT"
export ANSWER_SPAN_MIN_WEIGHTED_TOKENS="$KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS"
export BASELINE_EVAL_BEFORE_TRAIN=1
export REQUIRE_FINAL_EVAL_LTE_BASELINE=0
export ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA=0.16
export MAX_FINAL_EVAL_REGRESSION=0
export ABORT_TRAIN_RISE_POINTS=0
export ABORT_MAX_RESERVED_GIB=78
export SAMPLING_MODE='weighted_replacement'
export SUBCATEGORY_WEIGHTS='v51_equation_solver_cot_compact=20.00,v51_bit_solver_cot_compact=14.00,bit_fullbyte_v311_rule_variant=24.00,bit_fullbyte_v300_gain_pattern=14.00,bit_fullbyte_safe_ternary=10.00,bit_fullbyte_binary=9.00,equation_numeric_add_direct=18.00,equation_numeric_colon_absdiff=18.00,equation_numeric_minus_signed=18.00,equation_symbolic_binary=4.00,equation_symbolic_unary=4.00,equation_transform=1.50,bit_manipulation=1.50'
export SOURCE_WEIGHTS='v51_complete_filtered=12.00,v312_verifier_synthetic=20.00,v304_solver_trace_bit_fullbyte_distill_exact=12.00,v304_solver_trace_bit_fullbyte_distill_random=7.00,v282_v274_rule_synthetic=8.00,v216_base_clean_safe_strict_equation=1.20,v216_synthetic_kg1_symbolic_rules=1.05,v216_synthetic_kg1_numeric_rules=1.10,v216_base_clean_safe_strict_bit=1.35,v216_synthetic_kg1_bit_rules=1.35,v215_replay_anchor=0.90'
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
$PYBIN scripts/hf_job_preflight_gate.py --phase preinstall
$PYBIN scripts/hf_job_preflight_gate.py --phase artifacts
$PYBIN - <<'PY'
import inspect, json
from peft import LoraConfig
import causal_conv1d, mamba_ssm
payload = {
    "peft_lora_config_has_target_parameters": "target_parameters" in inspect.signature(LoraConfig.__init__).parameters,
    "causal_conv1d": getattr(causal_conv1d, "__version__", "ok"),
    "mamba_ssm": getattr(mamba_ssm, "__version__", "ok"),
}
print("nemo_dependency_probe = " + json.dumps(payload, sort_keys=True), flush=True)
if not payload["peft_lora_config_has_target_parameters"]:
    raise SystemExit("PEFT LoraConfig does not support target_parameters after upgrade")
PY
$PYBIN scripts/hf_job_preflight_gate.py --phase postinstall
$PYBIN scripts/hf_job_train_v90.py
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
        raise RuntimeError("HF token is required to launch V322.")
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if hardware[FLAVOR]["unit_cost_usd"] > 0.05:
        raise RuntimeError(f"A100 unit cost above gate: {hardware[FLAVOR]}")

    job_env = {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_TORCH_VERSION": "",
        "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "70",
        "KG1_REQUIRED_GPU_NAME_REGEX": "A100",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware[FLAVOR]["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.05",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "KG1_REQUIRE_OFFSET_MASK": "1",
        "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform,gravity_constant,numeral_system,text_encryption,unit_conversion",
        "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform,gravity_constant,numeral_system,text_encryption,unit_conversion",
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": "v51_equation_solver_cot_compact,v51_bit_solver_cot_compact,bit_fullbyte_v311_rule_variant,bit_fullbyte_v300_gain_pattern,bit_fullbyte_safe_ternary,bit_fullbyte_binary,equation_numeric_add_direct,equation_numeric_colon_absdiff,equation_numeric_minus_signed",
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
        "KG1_ANSWER_SPAN_LOSS_WEIGHT": ANSWER_SPAN_LOSS_WEIGHT,
        "KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
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
            "trainable_lora_modules": "q_proj,k_proj,v_proj,o_proj,lm_head",
            "learning_rate": "2.0e-8",
            "final_learning_rate": "5e-9",
            "answer_span_loss_weight": ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "objective": "filtered V51 solver-enhanced rows plus V304/V312 focused replay into attention/lm_head behavior",
            "finops": "A100 first; H200 reserved for weak eval or A100 failure.",
            "promotion_gate": "reject if total<192 or bit<136; inspect only if equation>56 or total>=193; promote only if equation>=60 and bit>=136",
        },
        "next_action": "Monitor every 30 seconds; weak-eval checkpoint-4/8/12/16/final if training completes.",
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
