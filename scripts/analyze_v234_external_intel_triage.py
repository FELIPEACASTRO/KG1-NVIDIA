#!/usr/bin/env python3
"""Build the V234 external-intelligence triage artifacts.

This script is CPU-only. It turns the roadmap's Kaggle/Hugging Face/OpenRouter
findings into explicit action matrices, validates roadmap coverage, checks local
metric parity for boxed-answer extraction, and emits the CSV/JSON artifacts that
the next solver/source-ingestion notebook must consume.

It does not train, run model generation, run scoring, download gated sources,
package artifacts, or submit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_boxed_answers, extract_final_answer  # noqa: E402


ROADMAP_REL = "artifacts/roadmaps/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md"
EXPECTED_REFS = [
    "metric/nvidia-nemotron-metric",
    "huikang/end-to-end-finetuning-for-lb-0-85",
    "huikang/tinker-submission-notebook",
    "mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers",
    "optiminist/equation-eda-operator-operation-84-solve-rate",
    "konbu17/bit-manipulation-solver-cot-generator",
    "johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster",
    "kalyankkr/all-6-puzzle-types-decoded-sft-training-data",
    "dgxchen/training-with-unsloth-to-achieve-0-85-lb",
    "hammadfarooq470/think-twice-self-correcting-reasoning",
    "anhtuan299/blackboard-expert-agent-assembly-solving-technique",
    "ryanholbrook/nvidia-nemotron-submission-demo",
    "dennisfong/nvidia-nemotron-sfttrainer-training",
    "kienngx/nvidia-nemotron-training-cot-labels",
    "kienngx/nvidia-nemotron-trained-models-submission",
    "asalhi/tinker-adapter-to-ready-to-submit-adapter",
    "huikang/adapter-validation-notebook",
    "kienngx/nvidia-nemotron-training-copy-run-instantly",
    "mayukh18/unsloth-sft-full-data-training",
    "llkh0a/nemotron-unsloth-sft-training-3-30-2",
    "newduck/nvidia-nemotron-soft-balanced-sampling-sft",
    "konbu17/nemotron-tong-style-cot-sft-updated-v2",
    "pearpn25/bit-cot-85-1364-sample",
    "kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset",
    "mohamedamr992/easy-loading-of-nemotron-3",
    "bloodymonday/eda-problem-families",
    "vickymaan/alice-puzzle-solver",
    "kishanvavdara/nemotron-reasoning-traj",
    "kienngx/nemotron-30b-competition-trainingdata-cot-labels",
    "konbu17/bit-manipulation-cot-dataset",
    "konbu17/bit-manipulation-synthetic-cot",
    "nctuan/nvidia-nemotron-reasoning-challenge",
    "mohammedtanvir/nemotron-reasoning-traces",
    "kevpan096/nemotron-reasoning-competition",
    "sebmontreal/nvidia-nemotron-model-reasoning-challenge",
    "harshmali0403/nvidia-nemotron-model-reasoning-challenge",
    "vsnihal/nvidia-nemotron-model-reasoning-challenge-01",
    "andy279/nemotron-reasoning-challenge",
    "andy279/nemotron-reasoning-challenge-raw-traces",
    "jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge",
    "justus27/reasoning-gym-bitwise-arithmetic",
    "nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2",
    "GaryNENE/nemotron-nano-8b-reasoning-lora",
    "AdaptKey/AdaptKey-Nemotron-30b",
    "Taurine511/nvidia-nemotron-model-reasoning-challenge",
    "kienngx/nemotron-nano-30b-trained",
    "atahalam/nvidia-nemotron-model-reasoning-30b-a3b-lora-0-80",
    "charancherrychowdary/nemotron-lora-adapter-v1",
    "sluitel/nemotron-70b-reasoning-lora",
    "nathangaskell/llama-3-1-nemotron-nano-8b",
    "metric/nemotron-3-nano-30b-a3b-bf16",
    "extract_boxed_answers",
    "external_metric_parity_report.json",
    "kaggle_kernel_triage.csv",
    "kaggle_dataset_triage.csv",
    "hf_dataset_triage.csv",
    "kaggle_model_triage.csv",
    "equation_numeric_operator_probe_results.csv",
    "bit_boolean_function_probe_results.csv",
    "external_adapter_registry_candidates.csv",
]


SOURCE_ROWS: list[dict[str, str]] = [
    {
        "ref": "metric/nvidia-nemotron-metric",
        "source_type": "kaggle_kernel",
        "status": "used_now",
        "priority": "P0",
        "family_focus": "metric_parity",
        "action_path": "validate extractor parity before any new family measurement",
        "required_output": "external_metric_parity_report.json",
        "gate": "metric_parity_failed",
    },
    {
        "ref": "huikang/end-to-end-finetuning-for-lb-0-85",
        "source_type": "kaggle_kernel",
        "status": "v234_required",
        "priority": "P0",
        "family_focus": "training_recipe",
        "action_path": "extract mask-loss and corpus criteria without copying training code",
        "required_output": "kaggle_kernel_triage.csv",
        "gate": "external_source_license_unknown",
    },
    {
        "ref": "huikang/tinker-submission-notebook",
        "source_type": "kaggle_kernel",
        "status": "v234_required",
        "priority": "P0",
        "family_focus": "adapter_registry",
        "action_path": "extract adapter registry and metric-parity checks",
        "required_output": "external_adapter_registry_candidates.csv",
        "gate": "external_adapter_requires_weak_check",
    },
    {
        "ref": "mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers",
        "source_type": "kaggle_kernel",
        "status": "v234_required",
        "priority": "P0",
        "family_focus": "taxonomy",
        "action_path": "compare family taxonomy and solver applicability against V230 misses",
        "required_output": "kaggle_kernel_triage.csv",
        "gate": "weak_miss_pack_overlap_missing",
    },
    {
        "ref": "optiminist/equation-eda-operator-operation-84-solve-rate",
        "source_type": "kaggle_kernel",
        "status": "v234_required",
        "priority": "P0",
        "family_focus": "equation_transform",
        "action_path": "probe Pre-Op Mid-Op Post-Op numeric equation routes",
        "required_output": "equation_numeric_operator_probe_results.csv",
        "gate": "equation_target_not_met",
    },
    {
        "ref": "konbu17/bit-manipulation-solver-cot-generator",
        "source_type": "kaggle_kernel",
        "status": "v234_required",
        "priority": "P0",
        "family_focus": "bit_manipulation",
        "action_path": "probe boolean functions by output bit including INHIB and IMPL variants",
        "required_output": "bit_boolean_function_probe_results.csv",
        "gate": "bit_guardrail_regression",
    },
    {
        "ref": "johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster",
        "source_type": "kaggle_kernel",
        "status": "future_triage",
        "priority": "P2",
        "family_focus": "verifier_training",
        "action_path": "review only after deterministic solvers exist",
        "required_output": "kaggle_kernel_triage.csv",
        "gate": "solver_verifier_missing",
    },
    {
        "ref": "kalyankkr/all-6-puzzle-types-decoded-sft-training-data",
        "source_type": "kaggle_kernel",
        "status": "future_triage",
        "priority": "P2",
        "family_focus": "taxonomy",
        "action_path": "use for format and taxonomy audit, not raw training",
        "required_output": "kaggle_kernel_triage.csv",
        "gate": "leakage_guard_missing",
    },
    {
        "ref": "dgxchen/training-with-unsloth-to-achieve-0-85-lb",
        "source_type": "kaggle_kernel",
        "status": "future_triage",
        "priority": "P2",
        "family_focus": "training_recipe",
        "action_path": "recipe only; local V221 adapter result was weaker",
        "required_output": "kaggle_kernel_triage.csv",
        "gate": "local_weak_check_required",
    },
    {
        "ref": "hammadfarooq470/think-twice-self-correcting-reasoning",
        "source_type": "kaggle_kernel",
        "status": "future_triage",
        "priority": "P3",
        "family_focus": "self_correction",
        "action_path": "use only if local gain is proven",
        "required_output": "kaggle_kernel_triage.csv",
        "gate": "local_gain_missing",
    },
    {
        "ref": "anhtuan299/blackboard-expert-agent-assembly-solving-technique",
        "source_type": "kaggle_kernel",
        "status": "future_triage",
        "priority": "P3",
        "family_focus": "tir_inspiration",
        "action_path": "method inspiration only until KG1 proof exists",
        "required_output": "kaggle_kernel_triage.csv",
        "gate": "not_actionable_without_probe",
    },
]


FUTURE_KERNELS = [
    "ryanholbrook/nvidia-nemotron-submission-demo",
    "dennisfong/nvidia-nemotron-sfttrainer-training",
    "kienngx/nvidia-nemotron-training-cot-labels",
    "kienngx/nvidia-nemotron-trained-models-submission",
    "asalhi/tinker-adapter-to-ready-to-submit-adapter",
    "huikang/adapter-validation-notebook",
    "kienngx/nvidia-nemotron-training-copy-run-instantly",
    "mayukh18/unsloth-sft-full-data-training",
    "llkh0a/nemotron-unsloth-sft-training-3-30-2",
    "newduck/nvidia-nemotron-soft-balanced-sampling-sft",
    "konbu17/nemotron-tong-style-cot-sft-updated-v2",
    "pearpn25/bit-cot-85-1364-sample",
    "kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset",
    "mohamedamr992/easy-loading-of-nemotron-3",
    "bloodymonday/eda-problem-families",
    "vickymaan/alice-puzzle-solver",
]


DATASETS = [
    ("kishanvavdara/nemotron-reasoning-traj", "kaggle_dataset", "v234_required", "P1", "reasoning_traces"),
    ("kienngx/nemotron-30b-competition-trainingdata-cot-labels", "kaggle_dataset", "v234_required", "P1", "cot_labels"),
    ("konbu17/bit-manipulation-cot-dataset", "kaggle_dataset", "v234_required", "P1", "bit_manipulation"),
    ("konbu17/bit-manipulation-synthetic-cot", "kaggle_dataset", "v234_required", "P1", "bit_manipulation"),
    ("nctuan/nvidia-nemotron-reasoning-challenge", "kaggle_dataset", "future_triage", "P2", "mirror_audit"),
    ("mohammedtanvir/nemotron-reasoning-traces", "kaggle_dataset", "future_triage", "P2", "reasoning_traces"),
    ("kevpan096/nemotron-reasoning-competition", "kaggle_dataset", "future_triage", "P3", "mirror_audit"),
    ("sebmontreal/nvidia-nemotron-model-reasoning-challenge", "kaggle_dataset", "future_triage", "P3", "mirror_audit"),
    ("harshmali0403/nvidia-nemotron-model-reasoning-challenge", "kaggle_dataset", "future_triage", "P3", "mirror_audit"),
    ("vsnihal/nvidia-nemotron-model-reasoning-challenge-01", "kaggle_dataset", "future_triage", "P3", "mirror_audit"),
    ("andy279/nemotron-reasoning-challenge", "hf_dataset", "v234_required", "P0", "official_gated_dataset"),
    ("andy279/nemotron-reasoning-challenge-raw-traces", "hf_dataset", "v234_required", "P0", "raw_teacher_traces"),
    ("jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge", "hf_dataset", "v234_required", "P1", "mirror_audit"),
    ("justus27/reasoning-gym-bitwise-arithmetic", "hf_dataset", "v234_required", "P1", "bit_manipulation"),
    ("nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2", "hf_dataset", "future_triage", "P3", "generic_kd_context"),
]


MODELS = [
    ("GaryNENE/nemotron-nano-8b-reasoning-lora", "hf_model", "reference_only", "P3", "8b_not_direct"),
    ("AdaptKey/AdaptKey-Nemotron-30b", "hf_model", "not_actionable", "P3", "telecom_not_kg1"),
    ("Taurine511/nvidia-nemotron-model-reasoning-challenge", "hf_model", "manual_verify", "P3", "api_not_found"),
    ("kienngx/nemotron-nano-30b-trained", "kaggle_model", "future_triage", "P1", "candidate_registry"),
    ("atahalam/nvidia-nemotron-model-reasoning-30b-a3b-lora-0-80", "kaggle_model", "future_triage", "P1", "candidate_registry"),
    ("charancherrychowdary/nemotron-lora-adapter-v1", "kaggle_model", "future_triage", "P2", "candidate_registry"),
    ("sluitel/nemotron-70b-reasoning-lora", "kaggle_model", "reference_only", "P3", "base_mismatch"),
    ("nathangaskell/llama-3-1-nemotron-nano-8b", "kaggle_model", "reference_only", "P3", "base_mismatch"),
    ("metric/nemotron-3-nano-30b-a3b-bf16", "kaggle_model", "reference_only", "P0", "base_metric_path"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def source_row(ref: str, source_type: str, status: str, priority: str, family_focus: str, action_path: str = "") -> dict[str, str]:
    return {
        "ref": ref,
        "source_type": source_type,
        "status": status,
        "priority": priority,
        "family_focus": family_focus,
        "action_path": action_path or "catalog and decide by hash/license/local weak evidence",
        "required_output": {
            "kaggle_kernel": "kaggle_kernel_triage.csv",
            "kaggle_dataset": "kaggle_dataset_triage.csv",
            "hf_dataset": "hf_dataset_triage.csv",
            "kaggle_model": "kaggle_model_triage.csv",
            "hf_model": "kaggle_model_triage.csv",
        }.get(source_type, "kaggle_kernel_triage.csv"),
        "gate": "hash_license_and_local_evidence_required",
    }


def build_registry_rows() -> list[dict[str, str]]:
    rows = list(SOURCE_ROWS)
    rows.extend(
        source_row(ref, "kaggle_kernel", "future_triage", "P2", "kernel_catalog")
        for ref in FUTURE_KERNELS
    )
    rows.extend(source_row(*item) for item in DATASETS)
    rows.extend(source_row(*item) for item in MODELS)
    rows.append(
        {
            "ref": "extract_boxed_answers",
            "source_type": "local_code",
            "status": "implemented_now",
            "priority": "P0",
            "family_focus": "metric_parity",
            "action_path": "local extractor fixed and tested",
            "required_output": "external_metric_parity_report.json",
            "gate": "metric_parity_failed",
        }
    )
    for artifact in [
        "external_metric_parity_report.json",
        "kaggle_kernel_triage.csv",
        "kaggle_dataset_triage.csv",
        "hf_dataset_triage.csv",
        "kaggle_model_triage.csv",
        "equation_numeric_operator_probe_results.csv",
        "bit_boolean_function_probe_results.csv",
        "external_adapter_registry_candidates.csv",
    ]:
        rows.append(
            {
                "ref": artifact,
                "source_type": "required_artifact",
                "status": "produced_by_v234",
                "priority": "P0",
                "family_focus": "audit_artifact",
                "action_path": "materialize artifact in V234 output manifest",
                "required_output": artifact,
                "gate": "artifact_missing",
            }
        )
    return rows


def metric_parity_report() -> dict[str, Any]:
    cases = [
        {"name": "plain", "text": r"\boxed{42}", "expected_boxes": ["42"], "expected_final": "42"},
        {"name": "last_boxed", "text": r"noise \boxed{1} more \boxed{2}", "expected_boxes": ["1", "2"], "expected_final": "2"},
        {"name": "nested_latex", "text": r"\boxed{\frac{1}{2}}", "expected_boxes": [r"\frac{1}{2}"], "expected_final": r"\frac{1}{2}"},
        {"name": "literal_brace_payload", "text": r"\boxed{}52}", "expected_boxes": ["}52"], "expected_final": "}52"},
        {"name": "unclosed", "text": r"\boxed{abc", "expected_boxes": ["abc"], "expected_final": "abc"},
        {"name": "fullwidth_colon", "text": "Final answer\uFF1A 99", "expected_boxes": [], "expected_final": "99"},
    ]
    results = []
    for case in cases:
        boxes = extract_boxed_answers(case["text"])
        final = extract_final_answer(case["text"])
        passed = boxes == case["expected_boxes"] and final == case["expected_final"]
        results.append({**case, "observed_boxes": boxes, "observed_final": final, "passed": passed})
    return {
        "schema_version": "kg1_v234_external_metric_parity_report_v1",
        "passed": all(row["passed"] for row in results),
        "cases": results,
        "source": "metric/nvidia-nemotron-metric behavior mirrored for boxed answer extraction",
    }


def roadmap_coverage(roadmap_path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    text = roadmap_path.read_text(encoding="utf-8")
    missing_refs = [ref for ref in EXPECTED_REFS if ref not in text]
    refs_without_action_path = sorted(row["ref"] for row in rows if not row.get("status") or not row.get("action_path"))
    duplicate_refs = sorted(ref for ref, count in Counter(row["ref"] for row in rows).items() if count > 1)
    return {
        "schema_version": "kg1_v234_roadmap_coverage_report_v1",
        "roadmap_path": str(roadmap_path),
        "roadmap_sha256": sha256_file(roadmap_path),
        "expected_ref_count": len(EXPECTED_REFS),
        "registry_ref_count": len(rows),
        "missing_refs": missing_refs,
        "refs_without_action_path": refs_without_action_path,
        "duplicate_refs": duplicate_refs,
        "passed": not missing_refs and not refs_without_action_path and not duplicate_refs,
    }


def equation_probe_rows() -> list[dict[str, Any]]:
    return [
        {
            "probe_name": "pre_op_mid_op_post_op_numeric_equation_probe",
            "source_ref": "optiminist/equation-eda-operator-operation-84-solve-rate",
            "family": "equation_transform",
            "status": "planned_v234_followup",
            "target_gain": 5,
            "guardrail": "must not lower bit_manipulation below 136",
            "required_input": "V230 equation miss pack plus source hash/license audit",
        },
        {
            "probe_name": "taxonomy_route_probe",
            "source_ref": "mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers",
            "family": "equation_transform",
            "status": "planned_v234_followup",
            "target_gain": 5,
            "guardrail": "abstain on ambiguous route",
            "required_input": "V230 equation miss pack",
        },
    ]


def bit_probe_rows() -> list[dict[str, Any]]:
    functions = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "INHIB", "REV_INHIB", "IMPL", "REV_IMPL"]
    return [
        {
            "probe_name": "boolean_function_by_output_bit",
            "source_ref": "konbu17/bit-manipulation-solver-cot-generator",
            "family": "bit_manipulation",
            "boolean_function": function,
            "status": "planned_v234_followup",
            "guardrail_min_correct": 136,
            "required_input": "V230 bit miss pack plus source hash/license audit",
        }
        for function in functions
    ]


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V234 EXTERNAL INTEL TRIAGE SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("roadmap_md =", args.roadmap_md, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    if not args.roadmap_md.exists():
        raise FileNotFoundError(args.roadmap_md)
    if not args.roadmap_md.is_file():
        raise IsADirectoryError(str(args.roadmap_md))

    rows = build_registry_rows()
    metric_report = metric_parity_report()
    coverage = roadmap_coverage(args.roadmap_md, rows)
    if not metric_report["passed"]:
        raise RuntimeError("metric parity failed")
    if not coverage["passed"]:
        raise RuntimeError("roadmap coverage failed: " + json.dumps(coverage, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "external_metric_parity_report_json": args.output_dir / "external_metric_parity_report.json",
        "roadmap_coverage_report_json": args.output_dir / f"{prefix}_roadmap_coverage_report.json",
        "kaggle_kernel_triage_csv": args.output_dir / "kaggle_kernel_triage.csv",
        "kaggle_dataset_triage_csv": args.output_dir / "kaggle_dataset_triage.csv",
        "hf_dataset_triage_csv": args.output_dir / "hf_dataset_triage.csv",
        "kaggle_model_triage_csv": args.output_dir / "kaggle_model_triage.csv",
        "equation_numeric_operator_probe_results_csv": args.output_dir / "equation_numeric_operator_probe_results.csv",
        "bit_boolean_function_probe_results_csv": args.output_dir / "bit_boolean_function_probe_results.csv",
        "external_adapter_registry_candidates_csv": args.output_dir / "external_adapter_registry_candidates.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    fieldnames = ["ref", "source_type", "status", "priority", "family_focus", "action_path", "required_output", "gate"]
    write_json(out_paths["external_metric_parity_report_json"], metric_report)
    write_json(out_paths["roadmap_coverage_report_json"], coverage)
    write_csv(out_paths["kaggle_kernel_triage_csv"], [row for row in rows if row["source_type"] == "kaggle_kernel"], fieldnames)
    write_csv(out_paths["kaggle_dataset_triage_csv"], [row for row in rows if row["source_type"] == "kaggle_dataset"], fieldnames)
    write_csv(out_paths["hf_dataset_triage_csv"], [row for row in rows if row["source_type"] == "hf_dataset"], fieldnames)
    write_csv(out_paths["kaggle_model_triage_csv"], [row for row in rows if row["source_type"] in {"kaggle_model", "hf_model"}], fieldnames)
    write_csv(
        out_paths["external_adapter_registry_candidates_csv"],
        [row for row in rows if row["family_focus"] == "candidate_registry" or "adapter" in row["family_focus"]],
        fieldnames,
    )
    write_csv(
        out_paths["equation_numeric_operator_probe_results_csv"],
        equation_probe_rows(),
        ["probe_name", "source_ref", "family", "status", "target_gain", "guardrail", "required_input"],
    )
    write_csv(
        out_paths["bit_boolean_function_probe_results_csv"],
        bit_probe_rows(),
        ["probe_name", "source_ref", "family", "boolean_function", "status", "guardrail_min_correct", "required_input"],
    )

    summary = {
        "source_type_counts": dict(sorted(Counter(row["source_type"] for row in rows).items())),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "priority_counts": dict(sorted(Counter(row["priority"] for row in rows).items())),
    }
    decision = {
        "decision": "external_intel_triage_ready_for_source_download",
        "reason": "roadmap references covered, action paths explicit, metric parity passed",
        "next_action": "Run source download/hash/license triage before implementing solver probes or evaluating external candidates.",
    }
    manifest = {
        "schema_version": "kg1_v234_external_intel_triage_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "roadmap_md": str(args.roadmap_md),
            "expected_ref_count": len(EXPECTED_REFS),
        },
        "input_artifact_hashes": {"roadmap_md": file_meta(args.roadmap_md)},
        "coverage": coverage,
        "metric_parity": {
            "passed": metric_report["passed"],
            "case_count": len(metric_report["cases"]),
        },
        "summary": summary,
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"},
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "scoring", "package", "kaggle_submit"],
    }
    write_json(out_paths["manifest_json"], manifest)

    print("coverage =", json.dumps(coverage, indent=2, sort_keys=True), flush=True)
    print("metric_parity =", json.dumps(manifest["metric_parity"], indent=2, sort_keys=True), flush=True)
    print("summary =", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V234 EXTERNAL INTEL TRIAGE SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap-md", type=Path, default=ROOT / ROADMAP_REL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v234_external_intel_triage")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        roadmap = root / "roadmap.md"
        roadmap.write_text("\n".join(EXPECTED_REFS), encoding="utf-8")
        args = argparse.Namespace(
            roadmap_md=roadmap,
            output_dir=root / "out",
            label="v234_external_intel_triage",
        )
        manifest = run_analysis(args)
        if manifest["coverage"]["missing_refs"]:
            raise AssertionError("expected no missing refs")
        if not manifest["metric_parity"]["passed"]:
            raise AssertionError("expected metric parity pass")
        for name in [
            "external_metric_parity_report_json",
            "kaggle_kernel_triage_csv",
            "kaggle_dataset_triage_csv",
            "hf_dataset_triage_csv",
            "kaggle_model_triage_csv",
            "equation_numeric_operator_probe_results_csv",
            "bit_boolean_function_probe_results_csv",
            "external_adapter_registry_candidates_csv",
        ]:
            path = Path(manifest["outputs"][name])
            if not path.exists() or path.stat().st_size <= 0:
                raise AssertionError(f"missing output {name}")
    print("v234_external_intel_triage_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
