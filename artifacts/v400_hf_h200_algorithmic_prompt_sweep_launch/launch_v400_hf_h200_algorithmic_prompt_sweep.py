#!/usr/bin/env python3
"""Launch V400 no-training weak algorithmic prompt sweep for V290 checkpoint-6.

V392 locked the current best submit package:
  V291/V290 checkpoint-6, weak 192/315, equation 56/155, bit 136/160.

This job deliberately does not train. It tests two short algorithmic prompt
suffixes before spending money on another LoRA run. Any variant must beat the
locked baseline before full/package/submit work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v400_h200_algorithmic_prompt_sweep"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID_BASE = "v400-h200-algorithmic-prompt-sweep-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
ADAPTER_SUBFOLDER = "checkpoint-6"
ADAPTER_NAME = "v290_checkpoint_6_prompt_sweep"
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_BASE = f"evals/{RUN_ID_BASE}"

SYMBOLIC_EQUATION_SUFFIX = (
    "\nInfer the hidden rule from the examples before answering. "
    "For equation/transformation rows, test concise candidates in this order: "
    "concat, reverse concat, sum, difference/subtraction, product, +1/-1, "
    "integer division/modulo, and punctuation/symbol mapping. "
    "For bit rows, preserve exact 8-bit binary output. "
    "Return only one line: `\\boxed{answer}`. No explanation."
)
BIT_STRIDE_SUFFIX = (
    "\nInfer the exact rule from examples. For bit_manipulation, compare output bits "
    "against input bits, negated bits, constants, bit pairs, and stride/rotation/shift "
    "patterns; output exactly 8 bits. For equation/transformation, test arithmetic, "
    "concat/reverse-concat, and symbol/punctuation transforms. "
    "Return only one line: `\\boxed{answer}`. No explanation."
)

PROMPT_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "symbolic_equation_first",
        "disable_thinking": False,
        "no_prompt_suffix": False,
        "prompt_suffix": SYMBOLIC_EQUATION_SUFFIX,
        "max_tokens": 7680,
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "purpose": "Push equation_transform toward verified DSL operators without changing the adapter.",
    },
    {
        "name": "bit_stride_guarded",
        "disable_thinking": False,
        "no_prompt_suffix": False,
        "prompt_suffix": BIT_STRIDE_SUFFIX,
        "max_tokens": 2048,
        "max_model_len": 4096,
        "max_num_seqs": 64,
        "purpose": "Protect bit format while checking whether a shorter generation budget reduces drift.",
    },
]


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
PYBIN=$(command -v python || command -v python3)
echo "python_bin=$PYBIN"
$PYBIN - <<'PY'
import json, torch
try:
    import vllm
    vllm_version = getattr(vllm, "__version__", "unknown")
except Exception as exc:
    vllm_version = repr(exc)
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(json.dumps({
    "torch_before": getattr(torch, "__version__", "unknown"),
    "cuda": getattr(torch.version, "cuda", ""),
    "cuda_available": torch.cuda.is_available(),
    "device": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
    "vllm": vllm_version,
}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
$PYBIN -m pip install -q --no-cache-dir --upgrade pip
$PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging peft safetensors hf_transfer
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m py_compile scripts/hf_job_weak_eval_v245.py scripts/hf_job_preflight_gate.py scripts/evaluate_lora_adapters_batch.py
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
cat > /tmp/kg1_v400_prompt_sweep.py <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from huggingface_hub import HfApi


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


variants = json.loads(os.environ["KG1_PROMPT_VARIANTS_JSON"])
skip_variant_names = set(json.loads(os.environ.get("KG1_SKIP_VARIANT_NAMES_JSON", "[]")))
if skip_variant_names:
    variants = [variant for variant in variants if variant["name"] not in skip_variant_names]
base_run_id = os.environ["KG1_RUN_ID_BASE"]
output_path_base = os.environ["KG1_OUTPUT_PATH_BASE"]
output_repo = os.environ["KG1_OUTPUT_REPO"]
summary_rows = []
best_row = None

print("=== V400 PROMPT SWEEP START ===", flush=True)
print("variant_count =", len(variants), flush=True)
print("skip_variant_names =", json.dumps(sorted(skip_variant_names)), flush=True)
print("base_run_id =", base_run_id, flush=True)
print("output_repo =", output_repo, flush=True)
print("output_path_base =", output_path_base, flush=True)

for index, variant in enumerate(variants, start=1):
    name = variant["name"]
    run_id = f"{base_run_id}_{name}"
    output_path = f"{output_path_base}/{name}"
    env = os.environ.copy()
    env.update(
        {
            "KG1_RUN_ID": run_id,
            "KG1_OUTPUT_PATH_IN_REPO": output_path,
            "KG1_LABEL_PREFIX": f"v400_{name}",
            "KG1_DISABLE_THINKING": "1" if variant.get("disable_thinking") else "0",
            "KG1_NO_PROMPT_SUFFIX": "1" if variant.get("no_prompt_suffix") else "0",
            "KG1_PROMPT_SUFFIX": str(variant.get("prompt_suffix", "")),
            "KG1_MAX_TOKENS": str(variant.get("max_tokens", 7680)),
            "KG1_MAX_MODEL_LEN": str(variant.get("max_model_len", 8192)),
            "KG1_MAX_NUM_SEQS": str(variant.get("max_num_seqs", 64)),
        }
    )
    print("=== V400 VARIANT START ===", flush=True)
    print("variant_index =", index, flush=True)
    print("variant_name =", name, flush=True)
    print("variant_config =", json.dumps(variant, sort_keys=True), flush=True)
    print("variant_run_id =", run_id, flush=True)
    rc = subprocess.run([sys.executable, "scripts/hf_job_weak_eval_v245.py"], env=env).returncode
    print("variant_returncode =", rc, flush=True)
    if rc:
        raise RuntimeError(f"variant failed rc={rc}: {name}")

    summary_path = Path("/tmp/kg1_v245_weak_eval") / run_id / "eval" / "batch_candidate_summary.json"
    manifest_path = Path("/tmp/kg1_v245_weak_eval") / run_id / "v245_hf_weak_eval_manifest.json"
    payload = load_json(summary_path)
    if not payload:
        raise RuntimeError(f"empty candidate summary for {name}")
    if isinstance(payload, dict):
        payload_rows = payload.get("rows", [])
    else:
        payload_rows = payload
    if not payload_rows:
        raise RuntimeError(f"empty candidate summary rows for {name}: {summary_path}")
    row = dict(payload_rows[0])
    row["variant_name"] = name
    row["variant_run_id"] = run_id
    row["variant_output_path_in_repo"] = output_path
    row["variant_manifest_path"] = str(manifest_path)
    row["variant_config"] = variant
    row["weak_gate_improves_locked_baseline"] = (
        int(row.get("correct", -1)) > 192
        and int(row.get("equation_transform_correct", -1)) > 56
        and int(row.get("bit_manipulation_correct", -1)) >= 136
        and int(row.get("truncated", 999)) == 0
    )
    summary_rows.append(row)
    if best_row is None or (
        int(row.get("correct", -1)),
        int(row.get("equation_transform_correct", -1)),
        int(row.get("bit_manipulation_correct", -1)),
        -int(row.get("truncated", 999)),
    ) > (
        int(best_row.get("correct", -1)),
        int(best_row.get("equation_transform_correct", -1)),
        int(best_row.get("bit_manipulation_correct", -1)),
        -int(best_row.get("truncated", 999)),
    ):
        best_row = row
    print("variant_summary =", json.dumps(row, sort_keys=True), flush=True)
    print("=== V400 VARIANT END ===", flush=True)

sweep_dir = Path("/tmp/kg1_v400_prompt_sweep") / base_run_id
sweep_dir.mkdir(parents=True, exist_ok=True)
sweep_manifest = {
    "schema_version": "kg1_v400_algorithmic_prompt_sweep_manifest_v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "base_run_id": base_run_id,
    "output_repo": output_repo,
    "output_path_base": output_path_base,
    "adapter_repo": os.environ["KG1_ADAPTER_REPO"],
    "adapter_specs_json": json.loads(os.environ["KG1_ADAPTER_SPECS_JSON"]),
    "locked_baseline": {
        "weak_total": 192,
        "equation_transform": 56,
        "bit_manipulation": 136,
        "truncated": 0,
    },
    "promotion_gate": {
        "weak_total_gt": 192,
        "weak_equation_gt": 56,
        "weak_bit_gte": 136,
        "weak_truncated_eq": 0,
        "full_must_improve_current_package": True,
    },
    "variants": variants,
    "summary_rows": summary_rows,
    "best_row": best_row,
    "decision": "promote_to_full_eval" if best_row and best_row.get("weak_gate_improves_locked_baseline") else "reject_all_variants_no_full_eval",
}
manifest_path = sweep_dir / "v400_prompt_sweep_manifest.json"
manifest_path.write_text(json.dumps(sweep_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("v400_sweep_manifest_path =", manifest_path, flush=True)
print("v400_sweep_best_row =", json.dumps(best_row, sort_keys=True), flush=True)
print("v400_sweep_decision =", sweep_manifest["decision"], flush=True)

api = HfApi(token=os.environ.get("HF_TOKEN") or None)
upload_info = api.upload_folder(
    repo_id=output_repo,
    repo_type="model",
    folder_path=str(sweep_dir),
    path_in_repo=f"{output_path_base}/sweep_manifest",
    commit_message=f"Add {base_run_id} prompt sweep manifest",
)
print("v400_sweep_upload_info =", upload_info, flush=True)
print("=== V400 PROMPT SWEEP END ===", flush=True)
PY
$PYBIN /tmp/kg1_v400_prompt_sweep.py
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


def adapter_exists(api: HfApi, repo_id: str, subfolder: str) -> bool:
    files = set(api.list_repo_files(repo_id, repo_type="model"))
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    return {prefix + "adapter_config.json", prefix + "adapter_model.safetensors"}.issubset(files)


def build_job_env(hardware: dict[str, object], skip_variants: list[str]) -> dict[str, str]:
    specs = [{"repo": ADAPTER_REPO, "subfolder": ADAPTER_SUBFOLDER, "name": ADAPTER_NAME}]
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_RUN_ID_BASE": RUN_ID_BASE,
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_OUTPUT_PATH_BASE": OUTPUT_PATH_BASE,
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_EVAL_TIMEOUT_S": "4200",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_PROMPT_VARIANTS_JSON": json.dumps(PROMPT_VARIANTS, sort_keys=True),
        "KG1_SKIP_VARIANT_NAMES_JSON": json.dumps(skip_variants, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually launch the Hugging Face Job.")
    parser.add_argument(
        "--skip-variant",
        action="append",
        default=[],
        choices=[variant["name"] for variant in PROMPT_VARIANTS],
        help="Skip a known variant, useful for FinOps-safe resume after a failed launcher.",
    )
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V400.")
    api = HfApi(token=token)
    hardware_by_name = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    hardware = hardware_by_name.get(FLAVOR)
    if not hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if float(hardware["unit_cost_usd"]) > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware}")
    if not adapter_exists(api, ADAPTER_REPO, ADAPTER_SUBFOLDER):
        raise RuntimeError(f"Missing adapter {ADAPTER_REPO}/{ADAPTER_SUBFOLDER}")

    skip_variants = sorted(set(args.skip_variant))
    job_env = build_job_env(hardware, skip_variants)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "debug_only_no_job_launched",
        "job_id": "",
        "job_url": "",
        "job_status": "not_launched",
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id_base": RUN_ID_BASE,
        "adapter_repo": ADAPTER_REPO,
        "adapter_subfolder": ADAPTER_SUBFOLDER,
        "output_repo": OUTPUT_REPO,
        "output_path_base": OUTPUT_PATH_BASE,
        "prompt_variants": PROMPT_VARIANTS,
        "skip_variants": skip_variants,
        "comparison_vs_locked_baseline": {
            "locked_weak": {"total": 192, "equation_transform": 56, "bit_manipulation": 136, "truncated": 0},
            "v391_checkpoint4": {"total": 191, "equation_transform": 56, "bit_manipulation": 135, "truncated": 0},
            "v400_required_to_promote": {"total_min": 193, "equation_transform_min": 57, "bit_manipulation_min": 136, "truncated": 0},
        },
        "promotion_gate": {
            "weak_total_gt": 192,
            "weak_equation_gt": 56,
            "weak_bit_gte": 136,
            "weak_truncated_eq": 0,
            "full_must_improve_current_package": True,
        },
        "finops_note": "No training. Stop after sweep unless at least one variant beats V392 locked weak gate.",
    }

    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware, indent=2, sort_keys=True), flush=True)

    if args.launch:
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=7200,
            namespace=NAMESPACE,
        )
        manifest.update(
            {
                "mode": "launched",
                "job_id": job.id,
                "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
                "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
            }
        )

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID_BASE}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"] or "not_launched_debug_only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
