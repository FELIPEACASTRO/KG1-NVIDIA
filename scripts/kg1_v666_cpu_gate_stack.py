#!/usr/bin/env python3
"""Aggregate the V666 CPU-only gates before any paid GPU work.

This script does not train or evaluate a model. It reads the concrete gate
artifacts that already exist and produces one auditable yes/no decision:
whether the current route is allowed to spend GPU. The default paths point to
the V666/V664 failure analysis artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_V478 = ROOT / "artifacts/v665_v664_failure_analysis/v666_v478_validation_row_weight_recheck_final.json"
DEFAULT_EOS = ROOT / "artifacts/v665_v664_failure_analysis/v666_v664_loss_mask_eos_contract_sample80_after_patch.json"
DEFAULT_V614 = ROOT / "artifacts/v665_v664_failure_analysis/v666_v664_v614_anti_runaway_gate.json"
DEFAULT_DRIFT = (
    ROOT
    / "artifacts/v665_v664_failure_analysis/v666_v664_label_free_drift_audit/"
    / "v666_v664_label_free_drift_summary.json"
)
DEFAULT_STATIC = ROOT / "artifacts/v665_v664_failure_analysis/v666_static_safety_gate_changed_files_final4.json"
DEFAULT_OPENROUTER_MANIFEST = (
    ROOT / "artifacts/openrouter/v666_post_train_rule_v664_executed/openrouter_manifest.json"
)
DEFAULT_OPENROUTER_CONSENSUS = (
    ROOT / "artifacts/openrouter/v666_post_train_rule_v664_executed/KG1_V666_POST_TRAIN_CONSENSUS.md"
)
DEFAULT_OUTPUT = ROOT / "artifacts/v665_v664_failure_analysis/v666_cpu_gate_stack.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def add_check(
    checks: list[dict[str, Any]],
    blockers: list[str],
    *,
    name: str,
    path: Path,
    ok: bool,
    details: dict[str, Any] | None = None,
    blocker: str | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "path": str(path),
            "ok": bool(ok),
            "details": details or {},
        }
    )
    if not ok:
        blockers.append(blocker or f"{name}_failed")


def check_v478(path: Path, checks: list[dict[str, Any]], blockers: list[str]) -> None:
    if not path.is_file():
        add_check(checks, blockers, name="v478_objective_alignment", path=path, ok=False, blocker="v478_missing")
        return
    payload = read_json(path)
    thresholds = payload.get("thresholds", {})
    train = payload.get("train", {})
    validation = payload.get("validation", {})
    train_rows = int(train.get("rows", 0) or 0)
    val_rows = int(validation.get("rows", 0) or 0)
    train_weight_rows = int(train.get("explicit_row_weight_rows", 0) or 0)
    val_weight_rows = int(validation.get("explicit_row_weight_rows", 0) or 0)
    train_share = train.get("effective_share_by_family", {})
    val_share = validation.get("effective_share_by_family", {})
    details = {
        "hf_gpu_allowed": bool(payload.get("hf_gpu_allowed", False)),
        "findings": payload.get("findings", []),
        "require_row_loss_weight": bool(thresholds.get("require_row_loss_weight", False)),
        "require_validation_row_loss_weight": bool(thresholds.get("require_validation_row_loss_weight", False)),
        "train_rows": train_rows,
        "train_explicit_row_weight_rows": train_weight_rows,
        "validation_rows": val_rows,
        "validation_explicit_row_weight_rows": val_weight_rows,
        "train_bit_effective_share": train_share.get("bit_manipulation", {}).get("share"),
        "train_equation_effective_share": train_share.get("equation_transform", {}).get("share"),
        "validation_bit_effective_share": val_share.get("bit_manipulation", {}).get("share"),
        "validation_equation_effective_share": val_share.get("equation_transform", {}).get("share"),
    }
    ok = (
        bool(payload.get("hf_gpu_allowed", False))
        and not payload.get("findings", [])
        and bool(thresholds.get("require_row_loss_weight", False))
        and bool(thresholds.get("require_validation_row_loss_weight", False))
        and train_rows > 0
        and val_rows > 0
        and train_weight_rows == train_rows
        and val_weight_rows == val_rows
    )
    add_check(
        checks,
        blockers,
        name="v478_objective_alignment",
        path=path,
        ok=ok,
        details=details,
        blocker="v478_row_loss_or_validation_weight_failed",
    )


def check_eos(path: Path, checks: list[dict[str, Any]], blockers: list[str]) -> None:
    if not path.is_file():
        add_check(checks, blockers, name="loss_mask_eos_contract", path=path, ok=False, blocker="eos_contract_missing")
        return
    payload = read_json(path)
    train = payload.get("train", {})
    validation = payload.get("validation", {})
    details = {
        "ok": bool(payload.get("ok", False)),
        "blockers": payload.get("blockers", []),
        "train_final_loss_eos_rate": train.get("final_loss_eos_rate"),
        "validation_final_loss_eos_rate": validation.get("final_loss_eos_rate"),
        "train_no_loss_rows": train.get("no_loss_rows"),
        "validation_no_loss_rows": validation.get("no_loss_rows"),
        "train_no_offset_rows": train.get("no_offset_rows"),
        "validation_no_offset_rows": validation.get("no_offset_rows"),
    }
    ok = (
        bool(payload.get("ok", False))
        and not payload.get("blockers", [])
        and float(train.get("final_loss_eos_rate", 0.0) or 0.0) >= 1.0
        and float(validation.get("final_loss_eos_rate", 0.0) or 0.0) >= 1.0
        and int(train.get("no_loss_rows", 0) or 0) == 0
        and int(validation.get("no_loss_rows", 0) or 0) == 0
        and int(train.get("no_offset_rows", 0) or 0) == 0
        and int(validation.get("no_offset_rows", 0) or 0) == 0
    )
    add_check(
        checks,
        blockers,
        name="loss_mask_eos_contract",
        path=path,
        ok=ok,
        details=details,
        blocker="loss_mask_eos_contract_failed",
    )


def check_v614(path: Path, checks: list[dict[str, Any]], blockers: list[str]) -> None:
    if not path.is_file():
        add_check(checks, blockers, name="v614_anti_runaway_promotion", path=path, ok=False, blocker="v614_missing")
        return
    payload = read_json(path)
    family_summary = payload.get("family_summary", {})
    details = {
        "ok": bool(payload.get("ok", False)),
        "decision": payload.get("decision"),
        "correct": payload.get("correct"),
        "rows": payload.get("rows"),
        "blockers": payload.get("blockers", []),
        "warnings": payload.get("warnings", []),
        "bit_p99_completion_tokens": family_summary.get("bit_manipulation", {}).get("p99_completion_tokens"),
        "equation_p99_completion_tokens": family_summary.get("equation_transform", {}).get("p99_completion_tokens"),
        "protected_results": payload.get("protected_results", []),
    }
    ok = bool(payload.get("ok", False)) and payload.get("decision") == "promotable"
    add_check(
        checks,
        blockers,
        name="v614_anti_runaway_promotion",
        path=path,
        ok=ok,
        details=details,
        blocker="v614_protected_or_length_or_score_failed",
    )


def check_label_free_drift(path: Path, checks: list[dict[str, Any]], blockers: list[str]) -> None:
    if not path.is_file():
        add_check(checks, blockers, name="label_free_drift_audit", path=path, ok=False, blocker="label_free_drift_missing")
        return
    payload = read_json(path)
    summary = payload.get("summary", {})
    details = {
        "schema_version": payload.get("schema_version"),
        "stored_vs_official_correctness_mismatch": summary.get("stored_vs_official_correctness_mismatch"),
        "first_boxed_vs_official_correctness_mismatch": summary.get("first_boxed_vs_official_correctness_mismatch"),
        "stored_vs_official_mismatch": summary.get("stored_vs_official_mismatch"),
        "first_boxed_vs_official_mismatch": summary.get("first_boxed_vs_official_mismatch"),
        "transition_counts": summary.get("transition_counts", {}),
        "conclusions": summary.get("conclusions", []),
    }
    stored_correctness_mismatch = int(summary.get("stored_vs_official_correctness_mismatch", -1))
    first_boxed_correctness_mismatch = int(summary.get("first_boxed_vs_official_correctness_mismatch", -1))
    ok = (
        payload.get("schema_version") == "kg1_v666_v664_label_free_drift_audit_v1"
        and stored_correctness_mismatch == 0
        and first_boxed_correctness_mismatch == 0
    )
    add_check(
        checks,
        blockers,
        name="label_free_drift_audit",
        path=path,
        ok=ok,
        details=details,
        blocker="label_free_drift_correctness_mismatch",
    )


def check_static(path: Path, checks: list[dict[str, Any]], blockers: list[str]) -> None:
    if not path.is_file():
        add_check(checks, blockers, name="static_safety_changed_files", path=path, ok=False, blocker="static_gate_missing")
        return
    payload = read_json(path)
    details = {
        "ok": bool(payload.get("ok", False)),
        "file_count": payload.get("file_count"),
        "findings": payload.get("findings", []),
    }
    ok = bool(payload.get("ok", False)) and not payload.get("findings", [])
    add_check(
        checks,
        blockers,
        name="static_safety_changed_files",
        path=path,
        ok=ok,
        details=details,
        blocker="static_safety_gate_failed",
    )


def check_openrouter(manifest_path: Path, consensus_path: Path, checks: list[dict[str, Any]], blockers: list[str]) -> None:
    if not manifest_path.is_file():
        add_check(
            checks,
            blockers,
            name="post_train_openrouter_manifest",
            path=manifest_path,
            ok=False,
            blocker="post_train_openrouter_manifest_missing",
        )
        return
    payload = read_json(manifest_path)
    consensus_exists = consensus_path.is_file() and consensus_path.stat().st_size > 0
    details = {
        "execute": bool(payload.get("execute", False)),
        "ok_count": int(payload.get("ok_count", 0) or 0),
        "api_key_present_without_value": bool(payload.get("api_key_present_without_value", False)),
        "models": payload.get("models", []),
        "prompt_path": payload.get("prompt_path"),
        "responses_path": payload.get("responses_path"),
        "consensus_path": str(consensus_path),
        "consensus_exists": consensus_exists,
    }
    ok = bool(payload.get("execute", False)) and int(payload.get("ok_count", 0) or 0) > 0 and consensus_exists
    add_check(
        checks,
        blockers,
        name="post_train_openrouter_rule",
        path=manifest_path,
        ok=ok,
        details=details,
        blocker="post_train_openrouter_rule_failed",
    )


def check_pycache(repo_root: Path, checks: list[dict[str, Any]], blockers: list[str]) -> None:
    pycache_dirs = [str(path.relative_to(repo_root)) for path in repo_root.rglob("__pycache__") if path.is_dir()]
    add_check(
        checks,
        blockers,
        name="workspace_no_pycache",
        path=repo_root,
        ok=not pycache_dirs,
        details={"pycache_dirs": pycache_dirs[:50], "pycache_dir_count": len(pycache_dirs)},
        blocker="workspace_pycache_present",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    check_v478(args.v478_report, checks, blockers)
    check_eos(args.eos_report, checks, blockers)
    check_v614(args.v614_report, checks, blockers)
    check_label_free_drift(args.drift_report, checks, blockers)
    check_static(args.static_report, checks, blockers)
    check_openrouter(args.openrouter_manifest, args.openrouter_consensus, checks, blockers)
    check_pycache(args.repo_root, checks, blockers)
    payload = {
        "schema_version": "kg1_v666_cpu_gate_stack_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(args.repo_root),
        "checks": checks,
        "finding_counts": {
            "blocker": len(set(blockers)),
            "failed_checks": sum(1 for check in checks if not check["ok"]),
            "passed_checks": sum(1 for check in checks if check["ok"]),
        },
        "blockers": sorted(set(blockers)),
        "ok": not blockers,
        "gpu_allowed": not blockers,
        "decision": "gpu_allowed" if not blockers else "gpu_blocked",
        "next_action": (
            "A100-large may be considered only after cost/preflight gates"
            if not blockers
            else "do not launch paid GPU; fix or replace the blocked route"
        ),
    }
    write_json(args.output_json, payload)
    return payload


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_v666_gate_") as tmp_name:
        tmp = Path(tmp_name)
        v478 = tmp / "v478.json"
        eos = tmp / "eos.json"
        v614 = tmp / "v614.json"
        drift = tmp / "drift.json"
        static = tmp / "static.json"
        manifest = tmp / "openrouter_manifest.json"
        consensus = tmp / "consensus.md"
        consensus.write_text("ok\n", encoding="utf-8")
        write_json(
            v478,
            {
                "hf_gpu_allowed": True,
                "findings": [],
                "thresholds": {"require_row_loss_weight": True, "require_validation_row_loss_weight": True},
                "train": {"rows": 2, "explicit_row_weight_rows": 2, "effective_share_by_family": {}},
                "validation": {"rows": 2, "explicit_row_weight_rows": 2, "effective_share_by_family": {}},
            },
        )
        write_json(
            eos,
            {
                "ok": True,
                "blockers": [],
                "train": {"final_loss_eos_rate": 1.0, "no_loss_rows": 0, "no_offset_rows": 0},
                "validation": {"final_loss_eos_rate": 1.0, "no_loss_rows": 0, "no_offset_rows": 0},
            },
        )
        write_json(v614, {"ok": True, "decision": "promotable", "family_summary": {}, "protected_results": []})
        write_json(
            drift,
            {
                "schema_version": "kg1_v666_v664_label_free_drift_audit_v1",
                "summary": {
                    "stored_vs_official_correctness_mismatch": 0,
                    "first_boxed_vs_official_correctness_mismatch": 0,
                    "transition_counts": {},
                    "conclusions": [],
                },
            },
        )
        write_json(static, {"ok": True, "findings": [], "file_count": 1})
        write_json(manifest, {"execute": True, "ok_count": 1, "api_key_present_without_value": True, "models": ["x"]})
        args = argparse.Namespace(
            v478_report=v478,
            eos_report=eos,
            v614_report=v614,
            drift_report=drift,
            static_report=static,
            openrouter_manifest=manifest,
            openrouter_consensus=consensus,
            output_json=tmp / "out.json",
            repo_root=tmp,
        )
        passed = run(args)
        assert passed["ok"], passed
        write_json(v614, {"ok": False, "decision": "blocked", "blockers": ["protected_failed_8740ed31"]})
        blocked = run(args)
        assert not blocked["ok"], blocked
        assert "v614_protected_or_length_or_score_failed" in blocked["blockers"], blocked
    print("kg1_v666_cpu_gate_stack_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v478-report", type=Path, default=DEFAULT_V478)
    parser.add_argument("--eos-report", type=Path, default=DEFAULT_EOS)
    parser.add_argument("--v614-report", type=Path, default=DEFAULT_V614)
    parser.add_argument("--drift-report", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--static-report", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--openrouter-manifest", type=Path, default=DEFAULT_OPENROUTER_MANIFEST)
    parser.add_argument("--openrouter-consensus", type=Path, default=DEFAULT_OPENROUTER_CONSENSUS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--allow-blocked-exit-zero",
        action="store_true",
        help="Write the report and exit zero even when the route is blocked.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    payload = run(args)
    print(
        "[V666] decision={decision} ok={ok} blockers={blockers}".format(
            decision=payload["decision"],
            ok=payload["ok"],
            blockers=",".join(payload["blockers"]) or "none",
        ),
        flush=True,
    )
    if payload["ok"] or args.allow_blocked_exit_zero:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
