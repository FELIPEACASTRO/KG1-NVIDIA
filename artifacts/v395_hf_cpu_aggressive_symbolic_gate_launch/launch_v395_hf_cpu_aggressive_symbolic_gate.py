#!/usr/bin/env python3
"""Launch V395 HF CPU aggressive symbolic/equation gate.

This job is deliberately CPU-only. It widens the symbolic/equation verifier
search around the locked V290 checkpoint-6 weak predictions, then integrates
only no-loss candidates. It does not train, evaluate with vLLM, package, or
submit. GPU work is authorized only if this CPU gate finds new IDs beyond the
known V390/V394 signal while preserving bit_manipulation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v395_hf_cpu_aggressive_symbolic_gate"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "python:3.11"
FLAVOR = "cpu-upgrade"
RUN_ID = "v395-hf-cpu-aggressive-symbolic-gate-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_DATASET_REPO = "felipesp1983/kg1-v395-cpu-symbolic-gate-artifacts"


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
python - <<'PY'
import json, platform
print(json.dumps({"python": platform.python_version(), "platform": platform.platform()}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
python -m pip install -q --no-cache-dir --upgrade pip
python -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas ortools
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
python -m py_compile \
  scripts/run_v324_equation_expanded_solver_gate.py \
  scripts/run_v329_symbolic_cryptarithm_gate.py \
  scripts/run_v336_integrated_no_loss_solver_gate.py \
  scripts/analyze_v375_equation_residual_clustering.py \
  scripts/analyze_v394_equation_row_level_inventory.py

BASE=artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv
OUT=artifacts/v395_hf_cpu_aggressive_symbolic_gate/${KG1_RUN_ID}
mkdir -p "$OUT"
echo "=== V395 HF CPU GATE START ==="
echo "base_csv=$BASE"
echo "out_dir=$OUT"
echo "run_id=$KG1_RUN_ID"
echo "output_dataset_repo=$KG1_OUTPUT_DATASET_REPO"

echo "=== V395 V324 AGGRESSIVE START ==="
python scripts/run_v324_equation_expanded_solver_gate.py \
  --input-csv "$BASE" \
  --output-dir "$OUT/v324_aggressive" \
  --target-equation-gain 4 \
  --bit-guardrail-min 136 \
  --pair-mapping-cap 12000 \
  --global-mapping-cap 60000 \
  --max-char-subset-size 5 \
  --max-position-sources 10 \
  --min-same-operator-examples 1
echo "=== V395 V324 AGGRESSIVE END ==="

echo "=== V395 V329 WIDE START ==="
python scripts/run_v329_symbolic_cryptarithm_gate.py \
  --input-csv "$BASE" \
  --v324-manifest-json "$OUT/v324_aggressive/v324_equation_expanded_solver_manifest.json" \
  --output-dir "$OUT/v329_wide" \
  --target-new-symbolic-gain 1 \
  --bit-guardrail-min 136 \
  --max-operator-symbols 8 \
  --max-solutions-per-assignment 20 \
  --solver-time-limit-s 1.5
echo "=== V395 V329 WIDE END ==="

echo "=== V395 V336 INTEGRATED START ==="
python scripts/run_v336_integrated_no_loss_solver_gate.py \
  --input-csv "$BASE" \
  --v324-manifest-json "$OUT/v324_aggressive/v324_equation_expanded_solver_manifest.json" \
  --v324-accepted-csv "$OUT/v324_aggressive/v324_equation_expanded_solver_accepted_candidates.csv" \
  --v329-manifest-json "$OUT/v329_wide/v329_symbolic_cryptarithm_manifest.json" \
  --v329-accepted-csv "$OUT/v329_wide/v329_symbolic_cryptarithm_accepted_candidates.csv" \
  --output-dir "$OUT/v336_integrated" \
  --weak-total-min 193 \
  --weak-equation-min 61 \
  --weak-bit-min 136
echo "=== V395 V336 INTEGRATED END ==="

echo "=== V395 V375 RESIDUAL START ==="
python scripts/analyze_v375_equation_residual_clustering.py \
  --v366-predictions-csv "$BASE" \
  --v324-audit-csv "$OUT/v324_aggressive/v324_equation_expanded_solver_audit.csv" \
  --output-dir "$OUT/v375_after_v324"
echo "=== V395 V375 RESIDUAL END ==="

echo "=== V395 V394 INVENTORY START ==="
python scripts/analyze_v394_equation_row_level_inventory.py \
  --baseline-predictions-csv "$BASE" \
  --v324-accepted-candidates-csv "$OUT/v324_aggressive/v324_equation_expanded_solver_accepted_candidates.csv" \
  --v324-audit-csv "$OUT/v324_aggressive/v324_equation_expanded_solver_audit.csv" \
  --v375-residual-rows-csv "$OUT/v375_after_v324/v375_equation_residual_rows.csv" \
  --output-dir "$OUT/v394_inventory"
echo "=== V395 V394 INVENTORY END ==="

python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

out = Path("artifacts/v395_hf_cpu_aggressive_symbolic_gate") / os.environ["KG1_RUN_ID"]
repo_id = os.environ["KG1_OUTPUT_DATASET_REPO"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out), path_in_repo=os.environ["KG1_RUN_ID"])
manifest = json.loads((out / "v336_integrated" / "v336a_integrated_no_loss_solver_gate_manifest.json").read_text(encoding="utf-8"))
print("v395_integrated_decision =", json.dumps(manifest.get("decision", {}), sort_keys=True), flush=True)
print("v395_integrated_family_summary =", json.dumps(manifest.get("family_summary", []), sort_keys=True), flush=True)
print("v395_uploaded_repo =", repo_id, flush=True)
print("v395_uploaded_path =", os.environ["KG1_RUN_ID"], flush=True)
PY
echo "=== V395 HF CPU GATE END ==="
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


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_RUN_ID": RUN_ID,
        "KG1_OUTPUT_DATASET_REPO": OUTPUT_DATASET_REPO,
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_HF_MAX_UNIT_COST_USD": "0.001",
        "KG1_TRAIN_ALLOWED": "0",
        "KG1_GPU_ALLOWED": "0",
        "KG1_PACKAGE_ALLOWED": "0",
        "KG1_KAGGLE_SUBMIT_ALLOWED": "0",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V395.")
    api = HfApi(token=token)
    hardware_by_name = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    hardware = hardware_by_name.get(FLAVOR)
    if not hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if float(hardware["unit_cost_usd"]) > 0.001:
        raise RuntimeError(f"CPU flavor unit cost above gate: {hardware}")

    job_env = build_job_env(hardware)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "debug_only_no_job_launched",
        "job_id": "",
        "job_url": "",
        "job_status": "not_launched",
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "output_dataset_repo": OUTPUT_DATASET_REPO,
        "gates": {
            "cpu_only": True,
            "gpu_allowed": False,
            "train_allowed": False,
            "package_allowed": False,
            "kaggle_submit_allowed": False,
            "promote_to_gpu_only_if_new_ids_beyond_v390": True,
        },
        "comparison_baseline": {
            "locked_submit": "V291/V290 checkpoint-6",
            "weak_total": 192,
            "equation_transform": 56,
            "bit_manipulation": 136,
            "known_cpu_projection_v390_v394": {
                "weak_total": 198,
                "equation_transform": 62,
                "bit_manipulation": 136,
                "known_ids": ["274def88", "528ec0d8", "7688e06e", "c5b058d6", "d1bd7478", "fb623471"],
            },
        },
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
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"] or "not_launched_debug_only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
