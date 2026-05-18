#!/usr/bin/env python3
"""Validate V620 local/remote decoding-drift gate policy sync before paid HF jobs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = ROOT / "scripts" / "hf_job_preflight_gate.py"


def load_gate_module() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_hf_job_preflight_gate", GATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load gate script: {GATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--require-current-head", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.launch_manifest.read_text(encoding="utf-8"))
    job_env = {str(k): str(v) for k, v in manifest.get("job_env", {}).items()}
    command_expected = {
        str(k): str(v)
        for k, v in manifest.get("command_export_audit", {}).get("expected", {}).items()
    }
    for name in ("MAX_STEPS", "SAVE_EVERY_STEPS", "EVAL_EVERY_STEPS"):
        if name in command_expected:
            job_env[name] = command_expected[name]
    recipe = manifest.get("recipe", {})
    if "MAX_STEPS" not in job_env and "max_steps" in recipe:
        job_env["MAX_STEPS"] = str(recipe["max_steps"])
    if "SAVE_EVERY_STEPS" not in job_env and "save_every_steps" in recipe:
        job_env["SAVE_EVERY_STEPS"] = str(recipe["save_every_steps"])
    if "EVAL_EVERY_STEPS" not in job_env and "eval_every_steps" in recipe:
        job_env["EVAL_EVERY_STEPS"] = str(recipe["eval_every_steps"])
    expected_commit = job_env.get("KG1_EXPECTED_COMMIT") or manifest.get("expected_commit", "")
    head = git_head()

    findings: list[dict[str, str]] = []
    if args.require_current_head and expected_commit != head:
        findings.append(
            {
                "severity": "error",
                "code": "expected_commit_not_current_head",
                "message": f"KG1_EXPECTED_COMMIT={expected_commit} current_head={head}",
            }
        )

    gate = load_gate_module()
    old_env = dict(os.environ)
    gate_payload: dict[str, Any] | None = None
    try:
        os.environ.clear()
        os.environ.update(old_env)
        os.environ.update(job_env)
        gate_payload = gate.check_decoding_vs_adapter_drift_gate()
        blockers = gate_payload.get("blockers", []) if isinstance(gate_payload, dict) else []
        if blockers:
            findings.append(
                {
                    "severity": "error",
                    "code": "decoding_vs_adapter_drift_blockers",
                    "message": ",".join(str(item) for item in blockers),
                }
            )
        limits = gate_payload.get("limits", {}) if isinstance(gate_payload, dict) else {}
        if limits.get("max_steps_lte") != 20 or limits.get("checkpoint_every_steps_lte") != 10:
            findings.append(
                {
                    "severity": "error",
                    "code": "v618_surface_limits_not_active",
                    "message": json.dumps(limits, sort_keys=True),
                }
            )
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    report = {
        "schema_version": "kg1_v622_remote_gate_policy_sync_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch_manifest": str(args.launch_manifest),
        "current_head": head,
        "expected_commit": expected_commit,
        "ok": not findings,
        "findings": findings,
        "decoding_vs_adapter_drift_gate": gate_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
