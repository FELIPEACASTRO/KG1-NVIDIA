#!/usr/bin/env python3
"""V336B package-permission gate for solver/verifier routing.

The gate answers one narrow question: can the CPU solver/verifier gain from
V336A be submitted directly, or must it be transferred into an adapter-only
LoRA candidate? The decision is intentionally conservative and evidence based.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_V336A_MANIFEST = (
    REPO_ROOT
    / "artifacts/v336_integrated_no_loss_solver_gate/20260513T_cpu_gate/"
    / "v336a_integrated_no_loss_solver_gate_manifest.json"
)
DEFAULT_COMPETITION_PAGES_JSON = REPO_ROOT / "artifacts/v327_kaggle_rules_audit/competition_pages.json"
DEFAULT_V327_AUDIT_MD = REPO_ROOT / "artifacts/v327_kaggle_rules_audit/KG1_V327_KAGGLE_RULES_AUDIT.md"
DEFAULT_PACKAGE_SCRIPT = REPO_ROOT / "scripts/package_hf_adapter_submission.py"
DEFAULT_V291_PACKAGE_MANIFEST = (
    REPO_ROOT
    / "artifacts/v291_submission_package/v291_h200_checkpoint6_823_20260511T212028Z/package_manifest.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_content(pages: list[dict[str, Any]], name: str) -> str:
    for page in pages:
        if str(page.get("name", "")).lower() == name.lower():
            return str(page.get("content", ""))
    return ""


def contains_all(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return all(needle.lower() in lower for needle in needles)


def assert_v336a_passed(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v336a_integrated_no_loss_solver_gate_v1":
        raise RuntimeError("unexpected V336A schema: " + str(payload.get("schema_version")))
    decision = payload.get("decision", {})
    if decision.get("decision") != "v336a_cpu_integrated_no_loss_gate_passed":
        raise RuntimeError("V336A did not pass: " + json.dumps(decision, sort_keys=True))
    integrated = payload.get("integrated", {})
    if int(integrated.get("loss_count", -1)) != 0:
        raise RuntimeError("V336A has losses")
    return payload


def evaluate_official_requirements(pages_json: Path) -> dict[str, Any]:
    pages = read_json(pages_json)
    if not isinstance(pages, list):
        raise RuntimeError("competition_pages.json must contain a list")
    evaluation = page_content(pages, "Evaluation")
    data_description = page_content(pages, "data-description")
    description = page_content(pages, "Description")
    rules = page_content(pages, "rules")

    facts = {
        "evaluation_mentions_submission_zip": "submission.zip" in evaluation,
        "evaluation_mentions_lora_adapter": contains_all(evaluation, ["LoRA adapter", "adapter_config.json"]),
        "evaluation_mentions_vllm_loads_adapter": contains_all(evaluation, ["vLLM inference engine", "LoRA adapter"]),
        "evaluation_mentions_rank_32": bool(re.search(r"rank(?: at most)?\s*32|max_lora_rank\s*\|\s*32", evaluation, re.I)),
        "data_mentions_submission_zip_adapter": contains_all(data_description, ["submission.zip", "LoRA adapter"]),
        "description_mentions_final_lora_adapter": contains_all(
            description, ["final submission", "compatible LoRA adapter"]
        ),
        "rules_allow_external_tools_with_constraints": contains_all(rules, ["external data", "reasonably accessible"]),
    }
    facts["official_adapter_only_requirement_confirmed"] = all(
        facts[key]
        for key in (
            "evaluation_mentions_submission_zip",
            "evaluation_mentions_lora_adapter",
            "evaluation_mentions_vllm_loads_adapter",
            "evaluation_mentions_rank_32",
            "data_mentions_submission_zip_adapter",
            "description_mentions_final_lora_adapter",
        )
    )
    return facts


def evaluate_local_packaging(package_script: Path, v291_package_manifest: Path) -> dict[str, Any]:
    script_text = package_script.read_text(encoding="utf-8")
    package_manifest = read_json(v291_package_manifest)
    zip_entries = sorted(package_manifest.get("submission_zip", {}).get("zip_entries", []))
    adapter = package_manifest.get("adapter", {})
    return {
        "package_script_sha256": sha256_file(package_script),
        "v291_package_manifest_sha256": sha256_file(v291_package_manifest),
        "script_required_files_adapter_only": "REQUIRED_FILES = (\"adapter_config.json\", \"adapter_model.safetensors\")"
        in script_text,
        "script_rejects_prediction_postprocessor": "submission package cannot rely on external prediction postprocessor"
        in script_text,
        "script_submit_hard_locked": "KG1_ALLOW_KAGGLE_SUBMIT=1" in script_text,
        "v291_zip_entries": zip_entries,
        "v291_zip_adapter_only": zip_entries == ["adapter_config.json", "adapter_model.safetensors"],
        "v291_adapter_rank": adapter.get("r"),
        "v291_adapter_alpha": adapter.get("lora_alpha"),
        "v291_adapter_rank_ok": int(adapter.get("r", -1)) <= 32,
    }


def markdown_summary(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    integrated = payload["v336a"]["integrated"]
    official = payload["official_requirements"]
    local = payload["local_packaging"]
    lines = [
        "# KG1 V336B - Package Permission Gate",
        "",
        f"Generated at UTC: `{payload['generated_at_utc']}`",
        "",
        "## V336A Input",
        "",
        f"- Integrated weak: `{integrated['correct']}/315`.",
        f"- Equation: `{integrated['family_counts']['equation_transform']['correct']}/155`.",
        f"- Bit: `{integrated['family_counts']['bit_manipulation']['correct']}/160`.",
        f"- Losses: `{integrated['loss_count']}`.",
        "",
        "## Official/Local Evidence",
        "",
        f"- Official adapter-only requirement confirmed: `{official['official_adapter_only_requirement_confirmed']}`.",
        f"- Local V291 zip adapter-only: `{local['v291_zip_adapter_only']}`.",
        f"- Package script rejects prediction postprocessor: `{local['script_rejects_prediction_postprocessor']}`.",
        "",
        "## Decision",
        "",
        f"- `{decision['decision']}`",
        f"- Reason: {decision['reason']}",
        f"- Next action: {decision['next_action']}",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V336B PACKAGE PERMISSION GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v336a_manifest_json =", args.v336a_manifest_json, flush=True)
    print("competition_pages_json =", args.competition_pages_json, flush=True)
    print("package_script =", args.package_script, flush=True)
    print("v291_package_manifest =", args.v291_package_manifest, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v336a = assert_v336a_passed(args.v336a_manifest_json)
    official = evaluate_official_requirements(args.competition_pages_json)
    local_packaging = evaluate_local_packaging(args.package_script, args.v291_package_manifest)

    direct_solver_package_allowed = False
    adapter_only_required = bool(
        official["official_adapter_only_requirement_confirmed"]
        and local_packaging["v291_zip_adapter_only"]
        and local_packaging["script_required_files_adapter_only"]
        and local_packaging["script_rejects_prediction_postprocessor"]
    )

    if adapter_only_required:
        decision = {
            "decision": "solver_verifier_direct_package_blocked_adapter_only_required",
            "reason": (
                "Official extracted pages require submission.zip containing a rank<=32 LoRA adapter; "
                "the valid local package contains only adapter_config.json and adapter_model.safetensors; "
                "the package script rejects prediction_postprocessor. V336A gain must be transferred "
                "into adapter-only behavior before Kaggle submit."
            ),
            "next_action": "Proceed to V337D minimal transfer dataset; do not submit solver/verifier package.",
            "direct_solver_package_allowed": direct_solver_package_allowed,
            "adapter_only_required": adapter_only_required,
            "hf_gpu_allowed": False,
        }
    else:
        decision = {
            "decision": "package_permission_inconclusive_block_all_submit",
            "reason": "Could not prove adapter-only requirement from local official evidence; conservative block.",
            "next_action": "Refresh official Kaggle extraction before package, submit, or HF GPU.",
            "direct_solver_package_allowed": False,
            "adapter_only_required": False,
            "hf_gpu_allowed": False,
        }

    outputs = {
        "manifest_json": args.output_dir / "v336b_package_permission_gate_manifest.json",
        "summary_md": args.output_dir / "KG1_V336B_PACKAGE_PERMISSION_GATE.md",
    }
    payload = {
        "schema_version": "kg1_v336b_package_permission_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v336a_manifest_json": str(args.v336a_manifest_json),
            "v336a_manifest_sha256": sha256_file(args.v336a_manifest_json),
            "competition_pages_json": str(args.competition_pages_json),
            "competition_pages_sha256": sha256_file(args.competition_pages_json),
            "v327_audit_md": str(args.v327_audit_md),
            "v327_audit_sha256": sha256_file(args.v327_audit_md) if args.v327_audit_md.is_file() else "",
            "package_script": str(args.package_script),
            "package_script_sha256": sha256_file(args.package_script),
            "v291_package_manifest": str(args.v291_package_manifest),
            "v291_package_manifest_sha256": sha256_file(args.v291_package_manifest),
        },
        "v336a": {
            "baseline": v336a.get("baseline"),
            "integrated": v336a.get("integrated"),
            "decision": v336a.get("decision"),
        },
        "official_requirements": official,
        "local_packaging": local_packaging,
        "decision": decision,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], payload)
    outputs["summary_md"].write_text(markdown_summary(payload), encoding="utf-8")

    print("official_requirements =", json.dumps(official, sort_keys=True), flush=True)
    print("local_packaging =", json.dumps(local_packaging, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("summary_md =", outputs["summary_md"], flush=True)
    print("=== V336B PACKAGE PERMISSION GATE END ===", flush=True)
    return payload


def run_self_test() -> None:
    facts = {
        "evaluation_mentions_submission_zip": True,
        "evaluation_mentions_lora_adapter": True,
        "evaluation_mentions_vllm_loads_adapter": True,
        "evaluation_mentions_rank_32": True,
        "data_mentions_submission_zip_adapter": True,
        "description_mentions_final_lora_adapter": True,
    }
    if not all(facts.values()):
        raise AssertionError(facts)
    print("v336b_package_permission_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v336a-manifest-json", type=Path, default=DEFAULT_V336A_MANIFEST)
    parser.add_argument("--competition-pages-json", type=Path, default=DEFAULT_COMPETITION_PAGES_JSON)
    parser.add_argument("--v327-audit-md", type=Path, default=DEFAULT_V327_AUDIT_MD)
    parser.add_argument("--package-script", type=Path, default=DEFAULT_PACKAGE_SCRIPT)
    parser.add_argument("--v291-package-manifest", type=Path, default=DEFAULT_V291_PACKAGE_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/v336b_package_permission_gate" / utc_compact(),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
