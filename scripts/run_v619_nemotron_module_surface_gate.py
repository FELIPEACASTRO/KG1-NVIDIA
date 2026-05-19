#!/usr/bin/env python3
"""V619 Nemotron module-surface gate.

This is a CPU/static gate.  It does not download model weights.  It inspects
the Nemotron source/config and a planned launcher manifest to prove that an
output-policy experiment has real attention/output-module names available and
requested before another paid H200 run is allowed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
DEFAULT_LAUNCH_MANIFEST = (
    ROOT
    / "artifacts/v615_hf_h200_v613_answer_first_launch/"
    / "v615-nemo-h200-v613-answerfirst-v290ckpt6-20260518T103423Z_launch_manifest.json"
)
DEFAULT_OUTPUT_JSON = ROOT / "artifacts/v619_nemotron_module_surface_gate/v619_module_surface_report.json"
ATTENTION_REQUIRED = {"q_proj", "v_proj", "o_proj"}
ATTENTION_OPTIONAL = {"k_proj"}
OUTPUT_POLICY_RELATED = ATTENTION_REQUIRED | ATTENTION_OPTIONAL | {"lm_head"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_csv(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def download_model_source(model_id: str, revision: str | None) -> tuple[Path, Path]:
    kwargs: dict[str, Any] = {"repo_id": model_id}
    if revision:
        kwargs["revision"] = revision
    modeling = Path(hf_hub_download(filename="modeling_nemotron_h.py", **kwargs))
    config = Path(hf_hub_download(filename="config.json", **kwargs))
    return modeling, config


def scan_source(modeling_path: Path, config_path: Path) -> dict[str, Any]:
    text = modeling_path.read_text(encoding="utf-8")
    config = read_json(config_path)
    declared_linear = re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*nn\.Linear", text)
    counts = Counter(declared_linear)
    has = {name: counts.get(name, 0) for name in sorted(set(declared_linear) | OUTPUT_POLICY_RELATED | {"up_proj", "down_proj"})}
    source_snippets: dict[str, str] = {}
    for name in sorted(OUTPUT_POLICY_RELATED | {"up_proj", "down_proj"}):
        marker = f"self.{name}"
        idx = text.find(marker)
        if idx >= 0:
            source_snippets[name] = " ".join(text[max(0, idx - 120) : idx + 180].split())
    return {
        "modeling_path": str(modeling_path),
        "config_path": str(config_path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "declared_linear_counts": dict(sorted(has.items())),
        "source_snippets": source_snippets,
        "attention_surface_available": all(counts.get(name, 0) > 0 for name in ATTENTION_REQUIRED),
        "lm_head_available": "lm_head" in text,
    }


def check_launch(launch_manifest: Path, source_report: dict[str, Any]) -> dict[str, Any]:
    launch = read_json(launch_manifest)
    job_env = launch.get("job_env", {})
    recipe = launch.get("recipe", {})
    trainable_modules = set(split_csv(job_env.get("KG1_TRAINABLE_LORA_MODULES") or recipe.get("trainable_lora_modules")))
    target_parameters = split_csv(job_env.get("KG1_LORA_TARGET_PARAMETERS"))
    missing_required = sorted(ATTENTION_REQUIRED - trainable_modules)
    attention_requested = sorted(trainable_modules & (ATTENTION_REQUIRED | ATTENTION_OPTIONAL))
    only_moe_target_parameters = bool(target_parameters) and all(item.startswith("mlp.experts.") for item in target_parameters)
    blockers: list[str] = []
    warnings: list[str] = []
    observations: list[str] = []
    if not source_report["attention_surface_available"]:
        blockers.append("source_attention_surface_not_detected")
    if missing_required:
        blockers.append("planned_trainable_attention_modules_missing:" + ",".join(missing_required))
    if not attention_requested:
        blockers.append("planned_attention_surface_empty")
    if only_moe_target_parameters:
        observations.append("target_parameters_are_moe_expert_only_but_attention_surface_is_requested_and_verified")
    if "lm_head" in trainable_modules and not source_report["lm_head_available"]:
        blockers.append("planned_lm_head_missing_in_source")
    return {
        "launch_manifest": str(launch_manifest),
        "trainable_lora_modules": sorted(trainable_modules),
        "target_parameters": target_parameters,
        "attention_requested": attention_requested,
        "missing_required_attention_modules": missing_required,
        "only_moe_target_parameters": only_moe_target_parameters,
        "blockers": blockers,
        "warnings": warnings,
        "observations": observations,
    }


def run_gate(*, model_id: str, revision: str | None, launch_manifest: Path, output_json: Path) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    try:
        modeling, config = download_model_source(model_id, revision)
        source_report = scan_source(modeling, config)
    except Exception as exc:
        source_report = {"download_or_scan_error": repr(exc), "attention_surface_available": False, "lm_head_available": False}
        blockers.append("model_source_scan_failed")
    launch_report = check_launch(launch_manifest, source_report) if launch_manifest else {}
    blockers.extend(launch_report.get("blockers", []))
    warnings.extend(launch_report.get("warnings", []))
    payload = {
        "schema_version": "kg1_v619_nemotron_module_surface_gate_v1",
        "generated_at_utc": utc_now(),
        "model_id": model_id,
        "revision": revision,
        "decision": "surface_gate_passed" if not blockers else "blocked",
        "ok": not blockers,
        "finding_counts": {"blocker": len(sorted(set(blockers))), "warning": len(sorted(set(warnings)))},
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "observations": sorted(set(launch_report.get("observations", []))),
        "source_report": source_report,
        "launch_report": launch_report,
        "export_env_if_used_by_next_launcher": {
            "KG1_V618_MODULE_SURFACE_GATE_STATUS": "passed" if not blockers else "blocked",
            "KG1_V619_MODULE_SURFACE_REPORT": str(output_json),
            "KG1_V619_ATTENTION_SURFACE": ",".join(sorted(launch_report.get("attention_requested", []))),
        },
        "rules": {
            "no_gpu_if_source_attention_missing": True,
            "no_gpu_if_trainable_attention_missing": True,
            "moe_target_parameters_warning": "MoE target_parameters may coexist only after this gate proves attention trainable modules are requested.",
        },
    }
    write_json(output_json, payload)
    return payload


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_v619_") as tmp_name:
        tmp = Path(tmp_name)
        modeling = tmp / "modeling_nemotron_h.py"
        config = tmp / "config.json"
        modeling.write_text(
            "\n".join(
                [
                    "self.q_proj = nn.Linear(1, 1)",
                    "self.k_proj = nn.Linear(1, 1)",
                    "self.v_proj = nn.Linear(1, 1)",
                    "self.o_proj = nn.Linear(1, 1)",
                    "self.lm_head = nn.Linear(1, 1)",
                ]
            ),
            encoding="utf-8",
        )
        config.write_text(json.dumps({"model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"]}), encoding="utf-8")
        report = scan_source(modeling, config)
        assert report["attention_surface_available"], report
        launch = tmp / "launch.json"
        launch.write_text(
            json.dumps({"job_env": {"KG1_TRAINABLE_LORA_MODULES": "q_proj,v_proj,o_proj"}, "recipe": {}}),
            encoding="utf-8",
        )
        checked = check_launch(launch, report)
        assert not checked["blockers"], checked
    print("[SELF_TEST] OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--launch-manifest", type=Path, default=DEFAULT_LAUNCH_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = run_gate(
        model_id=args.model_id,
        revision=args.revision,
        launch_manifest=args.launch_manifest,
        output_json=args.output_json,
    )
    print(
        "[V619_SURFACE] decision={decision} blockers={blockers} warnings={warnings} output={output}".format(
            decision=result["decision"],
            blockers=",".join(result["blockers"]) or "none",
            warnings=",".join(result["warnings"]) or "none",
            output=args.output_json,
        ),
        flush=True,
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
