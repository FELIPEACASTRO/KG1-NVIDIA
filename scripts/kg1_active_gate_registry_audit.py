#!/usr/bin/env python3
"""Audit the active KG1 gate registry.

This is a CPU-only registry audit. It does not train, package adapters, launch
paid jobs, call external APIs, or submit to Kaggle. Its job is to prove that
the active score/transfer path is still guarded by the expected gates before
any future training, paid GPU launch, adapter package, or score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "active_gate_registry_audit"

REQUIRED_SCRIPTS = {
    "notebook_release_gate": ROOT / "scripts" / "notebook_release_gate.py",
    "official_metric_prescore_gate": ROOT / "scripts" / "kg1_official_metric_prescore_gate.py",
    "solve_rate_gate": ROOT / "scripts" / "solve_rate_gate.py",
    "v1241_bit_equation_transfer_gate": ROOT / "scripts" / "kg1_v1241_bit_equation_transfer_gate.py",
    "v1241_full947_baseline_readiness_gate": ROOT / "scripts" / "kg1_v1241_full947_baseline_readiness_gate.py",
    "v1243_graft_trainer_contract_gate": ROOT / "scripts" / "kg1_v1243_graft_trainer_contract_gate.py",
    "score_path_operational_audit": ROOT / "scripts" / "kg1_score_path_operational_audit.py",
    "hf_job_train_v90": ROOT / "scripts" / "hf_job_train_v90.py",
    "evaluate_lora_adapter": ROOT / "scripts" / "evaluate_lora_adapter.py",
    "evaluate_lora_adapters_batch": ROOT / "scripts" / "evaluate_lora_adapters_batch.py",
}

REQUIRED_REPORTS = {
    "official_metric_prescore_gate": ROOT / "artifacts" / "official_metric_prescore_gate" / "report.json",
    "solve_rate_gate_self_test": ROOT / "artifacts" / "solve_rate_gate" / "self_test_report.json",
    "v1241_transfer_gate_selftest": (
        ROOT
        / "artifacts"
        / "v1241_bit_equation_transfer_gate_selftest"
        / "kg1_v1241_bit_equation_transfer_gate_selftest.json"
    ),
    "v1241_transfer_gate_legacy_alias_selftest": (
        ROOT
        / "artifacts"
        / "v1241_bit_equation_transfer_gate_selftest_legacy_alias"
        / "kg1_v1241_bit_equation_transfer_gate_selftest.json"
    ),
    "v1241_full947_baseline_readiness_selftest": (
        ROOT
        / "artifacts"
        / "v1241_full947_baseline_readiness_gate_selftest"
        / "kg1_v1241_full947_baseline_readiness_gate_selftest.json"
    ),
    "v1243_graft_trainer_contract_gate": (
        ROOT
        / "artifacts"
        / "v1243_solver_to_lora_graft_contract_gate"
        / "kg1_v1243_graft_trainer_contract_gate.json"
    ),
    "score_path_operational_audit": (
        ROOT
        / "artifacts"
        / "score_path_operational_audit"
        / "kg1_score_path_operational_audit.json"
    ),
}

V1243_MANIFEST = ROOT / "artifacts" / "v1243_solver_to_lora_graft" / "kg1_v1243_solver_to_lora_graft_manifest.json"
V1243_DRYRUN_REPORTS = {
    "bit": ROOT / "artifacts" / "v1243_solver_to_lora_graft_tokenize_dryrun" / "bit" / "dry_run_model_recipe_report.json",
    "equation": ROOT / "artifacts" / "v1243_solver_to_lora_graft_tokenize_dryrun" / "equation" / "dry_run_model_recipe_report.json",
    "micro_consolidation": (
        ROOT
        / "artifacts"
        / "v1243_solver_to_lora_graft_tokenize_dryrun"
        / "micro_consolidation"
        / "dry_run_model_recipe_report.json"
    ),
}
V1243_DATASETS = {
    "bit": ROOT / "artifacts" / "v1243_solver_to_lora_graft" / "v1243_bit_specialist_train.jsonl",
    "equation": ROOT / "artifacts" / "v1243_solver_to_lora_graft" / "v1243_equation_specialist_train.jsonl",
    "micro_consolidation": ROOT / "artifacts" / "v1243_solver_to_lora_graft" / "v1243_micro_consolidation_train.jsonl",
}
V1243_VAL = ROOT / "artifacts" / "v1243_solver_to_lora_graft" / "v1243_val170.jsonl"
FULL_ADAPTER_LORA_MODULES = [
    "down_proj",
    "in_proj",
    "k_proj",
    "lm_head",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]
SAFE_TRAINABLE_LORA_MODULES = [
    "down_proj",
    "in_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]
INIT_ADAPTER_REPO = "felipesp1983/kg1-recovered-v291-v290-checkpoint6-submit086"
INIT_ADAPTER_REVISION = "f4134a6d223249d27be2f1c5d94ed59d118d1ce5"
INIT_ADAPTER_CONFIG_SHA256 = "a3d74c5a52ce75f71a8406222d877b9760ea18a40a772bcf407686c8ea19f11d"
INIT_ADAPTER_WEIGHTS_SHA256 = "0a7b6144231d9358ae73a5e57d8778b32be1520fa47e3041414b3e025aaa1aa1"

ACTIVE_EXECUTION_SOURCES = [
    ROOT / "scripts" / "evaluate_lora_adapter.py",
    ROOT / "scripts" / "evaluate_lora_adapters_batch.py",
    ROOT / "scripts" / "hf_job_train_v90.py",
    ROOT / "scripts" / "kg1_v1241_bit_equation_transfer_gate.py",
    ROOT / "scripts" / "kg1_v1241_full947_baseline_readiness_gate.py",
    ROOT / "scripts" / "kg1_v1243_graft_trainer_contract_gate.py",
    ROOT / "scripts" / "kg1_v1243_solver_to_lora_graft_builder.py",
    ROOT / "scripts" / "solve_rate_gate.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def audit_scripts() -> dict[str, Any]:
    errors: list[str] = []
    scripts: dict[str, Any] = {}
    for name, path in REQUIRED_SCRIPTS.items():
        exists = path.exists()
        expect(errors, exists, f"required script missing: {path}")
        scripts[name] = {
            "path": str(path),
            "exists": exists,
            "sha256": sha256_file(path) if exists else "",
        }
    return {"errors": errors, "scripts": scripts}


def audit_reports() -> dict[str, Any]:
    errors: list[str] = []
    reports: dict[str, Any] = {}
    for name, path in REQUIRED_REPORTS.items():
        exists = path.exists()
        expect(errors, exists, f"required report missing: {path}")
        if not exists:
            reports[name] = {"path": str(path), "exists": False}
            continue
        payload = load_json(path)
        reports[name] = {"path": str(path), "exists": True, "sha256": sha256_file(path)}
        if name == "official_metric_prescore_gate":
            expect(errors, payload.get("decision") == "pass", "official metric prescore gate not pass")
            expect(errors, payload.get("blockers") == [], "official metric prescore gate has blockers")
        elif name == "solve_rate_gate_self_test":
            expect(errors, payload.get("prediction_only_blocked") is True, "solve_rate did not block prediction-only")
            expect(errors, payload.get("multi_box_blocked") is True, "solve_rate did not block multi-box")
            expect(errors, payload.get("truncation_blocked") is True, "solve_rate did not block truncation")
        elif name.startswith("v1241_transfer_gate"):
            expect(errors, payload.get("decision") == "pass_v1241_self_test", f"{name} not pass")
            expect(
                errors,
                all(case.get("expected_pass") == case.get("observed_pass") for case in payload.get("cases", [])),
                f"{name} case mismatch",
            )
            cases = {str(case.get("case")): case for case in payload.get("cases", [])}
            expect(
                errors,
                "pass_full947_clean_baseline_identity" in cases,
                f"{name} missing clean full947 baseline identity self-test",
            )
            dirty_identity = cases.get("fail_full947_dirty_baseline_identity", {})
            expect(errors, bool(dirty_identity), f"{name} missing dirty full947 baseline identity self-test")
            expect(
                errors,
                any(
                    "baseline_strict_clean_identity_unverified" in str(blocker)
                    for blocker in dirty_identity.get("blockers", [])
                ),
                f"{name} dirty baseline identity self-test did not block on strict-clean identity",
            )
            expect(
                errors,
                "probe_pass_full947_clean_baseline_identity" in cases,
                f"{name} missing clean baseline identity probe self-test",
            )
            probe_dirty_identity = cases.get("probe_fail_full947_dirty_baseline_identity", {})
            expect(errors, bool(probe_dirty_identity), f"{name} missing dirty baseline identity probe self-test")
            expect(
                errors,
                any(
                    "baseline_strict_clean_identity_not_pass" in str(blocker)
                    for blocker in probe_dirty_identity.get("blockers", [])
                ),
                f"{name} dirty baseline identity probe did not block on strict-clean identity",
            )
        elif name == "v1241_full947_baseline_readiness_selftest":
            expect(
                errors,
                payload.get("decision") == "pass_v1241_full947_baseline_readiness_self_test",
                "full947 baseline readiness self-test not pass",
            )
            expect(
                errors,
                all(case.get("expected_pass") == case.get("observed_pass") for case in payload.get("cases", [])),
                "full947 baseline readiness self-test case mismatch",
            )
            cases = {str(case.get("case")): case for case in payload.get("cases", [])}
            canonical_case = cases.get("pass_canonical_full947_solution_inventory", {})
            expect(errors, bool(canonical_case), "readiness self-test missing canonical full947 solution inventory case")
            expect(
                errors,
                canonical_case.get("observed_pass") is True,
                "canonical full947 solution inventory self-test did not pass",
            )
            expect(
                errors,
                canonical_case.get("rows") == 947,
                "canonical full947 solution inventory row count mismatch",
            )
            pass_case = cases.get("pass_dirty_natural_clean_answer_only", {})
            expect(errors, bool(pass_case), "readiness self-test missing dirty-natural/clean-answer-only pass case")
            expect(
                errors,
                pass_case.get("natural_probe_passed") is False,
                "readiness self-test should keep natural prompt as diagnostic, not required-pass",
            )
            expect(
                errors,
                pass_case.get("answer_only_probe_passed") is True,
                "readiness self-test answer-only probe did not pass",
            )
            missing_case = cases.get("fail_missing_answer_only_artifact", {})
            expect(errors, bool(missing_case), "readiness self-test missing answer-only artifact failure case")
            expect(
                errors,
                any("answer_only" in str(blocker) for blocker in missing_case.get("blockers", [])),
                "readiness self-test did not block missing answer-only artifact",
            )
        elif name == "v1243_graft_trainer_contract_gate":
            expect(
                errors,
                payload.get("decision") == "pass_v1243_graft_trainer_contract_no_gpu_no_submit",
                "V1243 contract gate not pass",
            )
            expect(errors, payload.get("errors") == [], "V1243 contract report has errors")
        elif name == "score_path_operational_audit":
            expect(
                errors,
                payload.get("schema_version") == "kg1_score_path_operational_audit_v2",
                "score path audit must be v2",
            )
            expect(
                errors,
                payload.get("decision") == "pass_score_path_operational_audit_no_gpu_no_submit",
                "score path audit not pass",
            )
            expect(errors, payload.get("errors") == [], "score path audit has errors")
            expect(errors, payload.get("warnings") == [], "score path audit has warnings")
    return {"errors": errors, "reports": reports}


def audit_active_artifacts() -> dict[str, Any]:
    errors: list[str] = []
    manifest = load_json(V1243_MANIFEST)
    not_authorized = set(manifest.get("algorithm", {}).get("not_authorized", []))
    for item in ("paid_gpu_launch", "adapter_package", "kaggle_submit", "score_claim"):
        expect(errors, item in not_authorized, f"V1243 manifest missing not_authorized={item}")
    expect(
        errors,
        manifest.get("decision") == "pass_v1243_cpu_dataset_graft_no_gpu_no_submit",
        "V1243 manifest decision mismatch",
    )
    adapter_contract = manifest.get("algorithm", {}).get("adapter_contract", {})
    expect(errors, adapter_contract.get("init_adapter_repo") == INIT_ADAPTER_REPO, "V1243 manifest init adapter repo mismatch")
    expect(
        errors,
        adapter_contract.get("init_adapter_revision") == INIT_ADAPTER_REVISION,
        "V1243 manifest init adapter revision mismatch",
    )
    expect(
        errors,
        adapter_contract.get("init_adapter_config_sha256") == INIT_ADAPTER_CONFIG_SHA256,
        "V1243 manifest init adapter config sha mismatch",
    )
    expect(
        errors,
        adapter_contract.get("init_adapter_weights_sha256") == INIT_ADAPTER_WEIGHTS_SHA256,
        "V1243 manifest init adapter weights sha mismatch",
    )

    dryruns: dict[str, Any] = {}
    for phase, path in V1243_DRYRUN_REPORTS.items():
        payload = load_json(path)
        data = payload.get("data", {})
        lora = payload.get("lora", {})
        trainable_filter = lora.get("trainable_lora_module_filter", {})
        dryruns[phase] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "decision": payload.get("decision", {}),
            "train_file_sha256": data.get("train_file_sha256"),
            "target_modules": lora.get("parsed_target_modules"),
            "trainable_modules": trainable_filter.get("requested"),
        }
        expect(
            errors,
            payload.get("decision", {}).get("tokenization_contract_passed") is True,
            f"{phase}: tokenization contract not passed",
        )
        expect(
            errors,
            payload.get("decision", {}).get("full_training_allowed") is False,
            f"{phase}: dry-run must not authorize full training",
        )
        expect(errors, data.get("train_file_sha256") == sha256_file(V1243_DATASETS[phase]), f"{phase}: stale dry-run train hash")
        expect(errors, data.get("validation_file_sha256") == sha256_file(V1243_VAL), f"{phase}: stale dry-run val hash")
        expect(errors, lora.get("r") == 32, f"{phase}: dry-run LoRA rank mismatch")
        expect(errors, lora.get("alpha") == 32, f"{phase}: dry-run LoRA alpha mismatch")
        expect(errors, lora.get("parsed_target_modules") == FULL_ADAPTER_LORA_MODULES, f"{phase}: dry-run active target modules mismatch")
        expect(errors, lora.get("init_adapter_repo") == INIT_ADAPTER_REPO, f"{phase}: dry-run init adapter repo mismatch")
        expect(errors, lora.get("init_adapter_revision") == INIT_ADAPTER_REVISION, f"{phase}: dry-run init adapter revision mismatch")
        expect(errors, trainable_filter.get("enabled") is True, f"{phase}: dry-run trainable filter disabled")
        expect(
            errors,
            trainable_filter.get("requested") == SAFE_TRAINABLE_LORA_MODULES,
            f"{phase}: dry-run trainable modules mismatch",
        )
    return {
        "errors": errors,
        "manifest_path": str(V1243_MANIFEST),
        "manifest_sha256": sha256_file(V1243_MANIFEST),
        "not_authorized": sorted(not_authorized),
        "dryruns": dryruns,
    }


def audit_forbidden_active_actions() -> dict[str, Any]:
    errors: list[str] = []
    findings: dict[str, list[str]] = {}
    forbidden_patterns = [
        "kaggle competitions submit",
        "subprocess.run(['kaggle', 'competitions', 'submit'",
        'subprocess.run(["kaggle", "competitions", "submit"',
    ]
    for path in ACTIVE_EXECUTION_SOURCES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [pattern for pattern in forbidden_patterns if pattern in text]
        if hits:
            findings[str(path.relative_to(ROOT))] = hits
            errors.append(f"forbidden active submit command in {path.relative_to(ROOT)}")
    return {"errors": errors, "findings": findings}


def audit_documentation_status() -> dict[str, Any]:
    errors: list[str] = []
    roadmap = ROOT / "docs" / "ROADMAP.md"
    score_audit_doc = ROOT / "docs" / "KG1_SCORE_DETERMINANTS_AUDIT_2026_06_13.md"
    expect(errors, roadmap.exists(), "ROADMAP.md missing")
    expect(errors, score_audit_doc.exists(), "score determinant audit doc missing")
    roadmap_text = roadmap.read_text(encoding="utf-8", errors="replace") if roadmap.exists() else ""
    expect(errors, "HISTORICO/FORENSE" in roadmap_text, "ROADMAP.md missing historical/superseded banner")
    expect(
        errors,
        "kg1_active_gate_registry_audit.py" in roadmap_text,
        "ROADMAP.md banner must point to active gate registry",
    )
    return {
        "errors": errors,
        "roadmap": str(roadmap),
        "score_audit_doc": str(score_audit_doc),
    }


def run(args: argparse.Namespace) -> int:
    print("[active-gate-registry] START", flush=True)
    checks = {
        "scripts": audit_scripts(),
        "reports": audit_reports(),
        "active_artifacts": audit_active_artifacts(),
        "forbidden_active_actions": audit_forbidden_active_actions(),
        "documentation_status": audit_documentation_status(),
    }
    errors: list[str] = []
    for check in checks.values():
        errors.extend(str(error) for error in check.get("errors", []))
    report = {
        "schema_version": "kg1_active_gate_registry_audit_v1",
        "generated_at_utc": utc_now(),
        "decision": "pass_active_gate_registry_no_train_no_paid_no_submit" if not errors else "fail",
        "active_train_path_authorized": False,
        "paid_gpu_launch_authorized": False,
        "adapter_package_authorized": False,
        "kaggle_submit_authorized": False,
        "score_claim_authorized": False,
        "required_before_any_future_paid_launch": [
            "workspace_clean_gate",
            "integrated_pretrain_gate",
            "objective_gate",
            "launcher_validate_only_gate",
            "pre_paid_launch_gate",
            "v276_canonical_full947_solution_hash_gate",
            "v1243_train_val_vs_full947_judge_leak_gate",
            "v291_086_full947_raw_output_readiness_gate",
            "v291_086_full947_natural_baseline_identity_probe",
            "v291_086_full947_answer_only_baseline_identity_probe",
            "score_path_operational_audit",
            "raw_output_transfer_gate",
        ],
        "checks": checks,
        "errors": errors,
    }
    report_path = args.output_dir.resolve() / "kg1_active_gate_registry_audit.json"
    write_json(report_path, report)
    print(f"[active-gate-registry] decision={report['decision']}", flush=True)
    print(f"[active-gate-registry] errors={len(errors)}", flush=True)
    print(f"[active-gate-registry] report={report_path}", flush=True)
    print("[active-gate-registry] END", flush=True)
    if errors:
        for error in errors[:50]:
            print(f"[active-gate-registry] ERROR {error}", flush=True)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
