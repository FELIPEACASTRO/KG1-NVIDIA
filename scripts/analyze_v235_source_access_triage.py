#!/usr/bin/env python3
"""Build V235 source-access, hash, and license triage artifacts.

This CPU-only script consumes the executed V234 manifest. It validates the V234
outputs, audits source access requirements, optionally checks Hugging Face
metadata through public HTTP APIs, and emits a controlled download/license plan.

It does not download source payloads, train, run model generation, score,
package artifacts, or submit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_V234_OUTPUTS = [
    "external_metric_parity_report_json",
    "kaggle_kernel_triage_csv",
    "kaggle_dataset_triage_csv",
    "hf_dataset_triage_csv",
    "kaggle_model_triage_csv",
    "equation_numeric_operator_probe_results_csv",
    "bit_boolean_function_probe_results_csv",
    "external_adapter_registry_candidates_csv",
]

INVENTORY_COLUMNS = [
    "ref",
    "source_type",
    "status",
    "priority",
    "family_focus",
    "action_path",
    "required_output",
    "gate",
    "source_url",
    "access_mode",
    "credential_requirement",
    "license_status",
    "hash_status",
    "download_allowed_now",
    "next_step",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def load_v234_paths(v234_manifest: dict[str, Any]) -> dict[str, Path]:
    outputs = v234_manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        raise RuntimeError("V234 manifest outputs must be an object")
    missing = sorted(name for name in REQUIRED_V234_OUTPUTS if not outputs.get(name))
    if missing:
        raise RuntimeError("V234 manifest missing outputs: " + json.dumps(missing))
    paths = {name: Path(str(outputs[name])) for name in REQUIRED_V234_OUTPUTS}
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"{name}: {path}")
    return paths


def validate_v234_manifest(v234_manifest: dict[str, Any]) -> None:
    coverage = v234_manifest.get("coverage", {})
    metric_parity = v234_manifest.get("metric_parity", {})
    if not isinstance(coverage, dict) or not coverage.get("passed"):
        raise RuntimeError("V234 coverage did not pass")
    if coverage.get("missing_refs"):
        raise RuntimeError("V234 has missing refs")
    if coverage.get("refs_without_action_path"):
        raise RuntimeError("V234 has refs without action path")
    if not isinstance(metric_parity, dict) or not metric_parity.get("passed"):
        raise RuntimeError("V234 metric parity did not pass")


def source_url(ref: str, source_type: str) -> str:
    if source_type == "kaggle_kernel":
        return "https://www.kaggle.com/code/" + ref
    if source_type == "kaggle_dataset":
        return "https://www.kaggle.com/datasets/" + ref
    if source_type == "kaggle_model":
        return "https://www.kaggle.com/models/" + ref
    if source_type == "hf_dataset":
        return "https://huggingface.co/datasets/" + ref
    if source_type == "hf_model":
        return "https://huggingface.co/" + ref
    return ""


def access_defaults(source_type: str) -> tuple[str, str, str, str]:
    if source_type.startswith("kaggle_"):
        return ("kaggle_cli_or_api_required", "kaggle_credentials_or_public_cli", "unknown_until_kaggle_metadata", "pending_download_hash")
    if source_type.startswith("hf_"):
        return ("huggingface_public_api_metadata", "none_or_hf_token_if_gated", "pending_hf_metadata", "pending_download_hash")
    if source_type == "local_code":
        return ("local_repo", "none", "project_license_context", "local_code_hash_available")
    if source_type == "required_artifact":
        return ("generated_artifact", "none", "not_applicable", "generated_hash_available")
    return ("manual_review", "manual_review", "unknown", "pending")


def credential_audit() -> dict[str, Any]:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return {
        "kaggle_cli_path": shutil.which("kaggle") or "",
        "kaggle_username_present": bool(os.environ.get("KAGGLE_USERNAME")),
        "kaggle_key_present": bool(os.environ.get("KAGGLE_KEY")),
        "kaggle_json_exists": kaggle_json.exists(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "openrouter_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


def query_hf_metadata(ref: str, source_type: str, timeout_s: int) -> dict[str, Any]:
    endpoint_type = "datasets" if source_type == "hf_dataset" else "models"
    url = f"https://huggingface.co/api/{endpoint_type}/{ref}"
    request = urllib.request.Request(url, headers={"User-Agent": "kg1-v235-source-access-triage"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        card_data = payload.get("cardData") if isinstance(payload.get("cardData"), dict) else {}
        tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
        license_values = []
        if card_data.get("license"):
            license_values.append(str(card_data["license"]))
        license_values.extend(str(tag).replace("license:", "") for tag in tags if str(tag).startswith("license:"))
        return {
            "ref": ref,
            "source_type": source_type,
            "api_url": url,
            "http_status": 200,
            "exists": True,
            "gated": str(payload.get("gated", "")),
            "private": str(payload.get("private", "")),
            "license": ";".join(sorted(set(license_values))),
            "sha": str(payload.get("sha", "")),
            "downloads": str(payload.get("downloads", "")),
            "likes": str(payload.get("likes", "")),
            "error": "",
        }
    except urllib.error.HTTPError as exc:
        return {
            "ref": ref,
            "source_type": source_type,
            "api_url": url,
            "http_status": int(exc.code),
            "exists": False,
            "gated": "",
            "private": "",
            "license": "",
            "sha": "",
            "downloads": "",
            "likes": "",
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ref": ref,
            "source_type": source_type,
            "api_url": url,
            "http_status": 0,
            "exists": False,
            "gated": "",
            "private": "",
            "license": "",
            "sha": "",
            "downloads": "",
            "likes": "",
            "error": repr(exc),
        }


def build_inventory(rows: list[dict[str, str]], hf_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = []
    for row in rows:
        ref = row.get("ref", "")
        source_type = row.get("source_type", "")
        access_mode, credential_requirement, license_status, hash_status = access_defaults(source_type)
        meta = hf_meta.get(ref, {})
        if meta:
            if meta.get("http_status") == 200:
                license_status = "known:" + str(meta.get("license", "") or "missing")
                if str(meta.get("gated", "")).lower() in {"true", "manual"}:
                    credential_requirement = "hf_token_required"
                access_mode = "hf_metadata_checked"
                hash_status = "repo_sha:" + str(meta.get("sha", "") or "missing")
            else:
                access_mode = "hf_metadata_error"
        status = row.get("status", "")
        allowed_now = (
            source_type in {"local_code", "required_artifact"}
            or (source_type.startswith("hf_") and meta.get("http_status") == 200 and bool(meta.get("license")))
        )
        next_step = "download_and_hash_with_license_guard" if allowed_now else "resolve_access_license_hash_before_use"
        inventory.append(
            {
                **row,
                "source_url": source_url(ref, source_type),
                "access_mode": access_mode,
                "credential_requirement": credential_requirement,
                "license_status": license_status,
                "hash_status": hash_status,
                "download_allowed_now": str(bool(allowed_now)).lower(),
                "next_step": next_step,
                "status": status,
            }
        )
    return inventory


def load_source_rows(paths: dict[str, Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in [
        "kaggle_kernel_triage_csv",
        "kaggle_dataset_triage_csv",
        "hf_dataset_triage_csv",
        "kaggle_model_triage_csv",
    ]:
        rows.extend(read_csv(paths[name]))
    return rows


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V235 SOURCE ACCESS TRIAGE SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v234-analysis-manifest-json =", args.v234_analysis_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("enable_network_metadata =", args.enable_network_metadata, flush=True)
    v234_manifest = read_json(args.v234_analysis_manifest_json)
    validate_v234_manifest(v234_manifest)
    paths = load_v234_paths(v234_manifest)
    source_rows = load_source_rows(paths)
    if not source_rows:
        raise RuntimeError("V235 requires non-empty source rows from V234")

    credential_report = credential_audit()
    hf_rows = [row for row in source_rows if row.get("source_type") in {"hf_dataset", "hf_model"}]
    hf_metadata_rows = []
    if args.enable_network_metadata:
        for row in hf_rows:
            hf_metadata_rows.append(query_hf_metadata(row.get("ref", ""), row.get("source_type", ""), args.network_timeout_s))
    else:
        hf_metadata_rows = [
            {
                "ref": row.get("ref", ""),
                "source_type": row.get("source_type", ""),
                "api_url": "",
                "http_status": 0,
                "exists": False,
                "gated": "",
                "private": "",
                "license": "",
                "sha": "",
                "downloads": "",
                "likes": "",
                "error": "network_metadata_disabled",
            }
            for row in hf_rows
        ]
    hf_meta = {row["ref"]: row for row in hf_metadata_rows}
    inventory = build_inventory(source_rows, hf_meta)
    kaggle_audit = [
        {
            "ref": row["ref"],
            "source_type": row["source_type"],
            "source_url": row["source_url"],
            "priority": row["priority"],
            "status": row["status"],
            "credential_requirement": row["credential_requirement"],
            "license_status": row["license_status"],
            "download_allowed_now": row["download_allowed_now"],
            "next_step": row["next_step"],
        }
        for row in inventory
        if row["source_type"].startswith("kaggle_")
    ]
    download_plan = [
        {
            "ref": row["ref"],
            "source_type": row["source_type"],
            "priority": row["priority"],
            "family_focus": row["family_focus"],
            "download_allowed_now": row["download_allowed_now"],
            "blocker": "" if row["download_allowed_now"] == "true" else row["gate"],
            "next_step": row["next_step"],
            "source_url": row["source_url"],
        }
        for row in inventory
        if row["status"] in {"used_now", "v234_required", "future_triage", "manual_verify"}
    ]
    license_gate = {
        "schema_version": "kg1_v235_license_gate_report_v1",
        "direct_ingestion_allowed": all(row["download_allowed_now"] == "true" for row in inventory if row["status"] in {"used_now", "v234_required"}),
        "required_not_allowed": [
            row["ref"]
            for row in inventory
            if row["status"] in {"used_now", "v234_required"} and row["download_allowed_now"] != "true"
        ],
        "credential_report": credential_report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "source_access_inventory_csv": args.output_dir / "source_access_inventory.csv",
        "hf_metadata_audit_csv": args.output_dir / "hf_metadata_audit.csv",
        "kaggle_access_audit_csv": args.output_dir / "kaggle_access_audit.csv",
        "source_download_plan_csv": args.output_dir / "source_download_plan.csv",
        "license_gate_report_json": args.output_dir / "license_gate_report.json",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    write_csv(out_paths["source_access_inventory_csv"], inventory, INVENTORY_COLUMNS)
    write_csv(
        out_paths["hf_metadata_audit_csv"],
        hf_metadata_rows,
        ["ref", "source_type", "api_url", "http_status", "exists", "gated", "private", "license", "sha", "downloads", "likes", "error"],
    )
    write_csv(
        out_paths["kaggle_access_audit_csv"],
        kaggle_audit,
        ["ref", "source_type", "source_url", "priority", "status", "credential_requirement", "license_status", "download_allowed_now", "next_step"],
    )
    write_csv(
        out_paths["source_download_plan_csv"],
        download_plan,
        ["ref", "source_type", "priority", "family_focus", "download_allowed_now", "blocker", "next_step", "source_url"],
    )
    write_json(out_paths["license_gate_report_json"], license_gate)

    summary = {
        "source_type_counts": dict(sorted(Counter(row["source_type"] for row in inventory).items())),
        "status_counts": dict(sorted(Counter(row["status"] for row in inventory).items())),
        "download_allowed_counts": dict(sorted(Counter(row["download_allowed_now"] for row in inventory).items())),
        "hf_metadata_http_status_counts": dict(sorted(Counter(str(row["http_status"]) for row in hf_metadata_rows).items())),
    }
    decision = {
        "decision": "source_access_plan_ready_needs_controlled_download",
        "reason": "V234 artifacts validated; source access inventory and license gate generated",
        "next_action": "Download only allowed sources with recorded hash/license, then map usable rows to V230 miss-pack before solver implementation.",
    }
    if license_gate["required_not_allowed"]:
        decision = {
            "decision": "manual_source_access_or_license_required_before_download",
            "reason": "required sources still need credentials, license metadata, or download hash: " + ",".join(license_gate["required_not_allowed"][:10]),
            "next_action": "Resolve Kaggle/HF credentials and license metadata for required sources before payload download.",
        }
    manifest = {
        "schema_version": "kg1_v235_source_access_triage_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v234_analysis_manifest_json": str(args.v234_analysis_manifest_json),
            "enable_network_metadata": bool(args.enable_network_metadata),
        },
        "input_artifact_hashes": {
            "v234_analysis_manifest_json": file_meta(args.v234_analysis_manifest_json),
            **{name: file_meta(path) for name, path in sorted(paths.items())},
        },
        "summary": summary,
        "license_gate": license_gate,
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"},
        "decision": decision,
        "blocked_actions": ["payload_download_without_license_hash", "train", "model_generation", "scoring", "package", "kaggle_submit"],
    }
    write_json(out_paths["manifest_json"], manifest)

    print("credential_report =", json.dumps(credential_report, indent=2, sort_keys=True), flush=True)
    print("summary =", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("license_gate =", json.dumps(license_gate, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V235 SOURCE ACCESS TRIAGE SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v234-analysis-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v235_source_access_triage")
    parser.add_argument("--enable-network-metadata", action="store_true")
    parser.add_argument("--network-timeout-s", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        out_dir = root / "out"
        v234_out = root / "v234"
        v234_out.mkdir()
        fieldnames = ["ref", "source_type", "status", "priority", "family_focus", "action_path", "required_output", "gate"]
        write_csv(
            v234_out / "kaggle_kernel_triage.csv",
            [
                {
                    "ref": "metric/nvidia-nemotron-metric",
                    "source_type": "kaggle_kernel",
                    "status": "used_now",
                    "priority": "P0",
                    "family_focus": "metric_parity",
                    "action_path": "validate extractor parity",
                    "required_output": "external_metric_parity_report.json",
                    "gate": "metric_parity_failed",
                }
            ],
            fieldnames,
        )
        write_csv(v234_out / "kaggle_dataset_triage.csv", [], fieldnames)
        write_csv(
            v234_out / "hf_dataset_triage.csv",
            [
                {
                    "ref": "jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge",
                    "source_type": "hf_dataset",
                    "status": "v234_required",
                    "priority": "P1",
                    "family_focus": "mirror_audit",
                    "action_path": "mirror audit",
                    "required_output": "hf_dataset_triage.csv",
                    "gate": "hash_license_and_local_evidence_required",
                }
            ],
            fieldnames,
        )
        write_csv(v234_out / "kaggle_model_triage.csv", [], fieldnames)
        for name in [
            "external_metric_parity_report_json",
            "equation_numeric_operator_probe_results_csv",
            "bit_boolean_function_probe_results_csv",
            "external_adapter_registry_candidates_csv",
        ]:
            path = v234_out / (name + (".json" if name.endswith("_json") else ".csv"))
            path.write_text("{}\n" if path.suffix == ".json" else "a\n", encoding="utf-8")
        v234_manifest = root / "v234_manifest.json"
        write_json(
            v234_manifest,
            {
                "coverage": {"passed": True, "missing_refs": [], "refs_without_action_path": []},
                "metric_parity": {"passed": True},
                "outputs": {
                    "external_metric_parity_report_json": str(v234_out / "external_metric_parity_report_json.json"),
                    "kaggle_kernel_triage_csv": str(v234_out / "kaggle_kernel_triage.csv"),
                    "kaggle_dataset_triage_csv": str(v234_out / "kaggle_dataset_triage.csv"),
                    "hf_dataset_triage_csv": str(v234_out / "hf_dataset_triage.csv"),
                    "kaggle_model_triage_csv": str(v234_out / "kaggle_model_triage.csv"),
                    "equation_numeric_operator_probe_results_csv": str(v234_out / "equation_numeric_operator_probe_results_csv.csv"),
                    "bit_boolean_function_probe_results_csv": str(v234_out / "bit_boolean_function_probe_results_csv.csv"),
                    "external_adapter_registry_candidates_csv": str(v234_out / "external_adapter_registry_candidates_csv.csv"),
                },
            },
        )
        args = argparse.Namespace(
            v234_analysis_manifest_json=v234_manifest,
            output_dir=out_dir,
            label="v235_source_access_triage",
            enable_network_metadata=False,
            network_timeout_s=1,
        )
        manifest = run_analysis(args)
        if not Path(manifest["outputs"]["source_access_inventory_csv"]).exists():
            raise AssertionError("source access inventory missing")
        if "source_access_plan" not in manifest["decision"]["decision"] and "manual_source_access" not in manifest["decision"]["decision"]:
            raise AssertionError("unexpected decision")
    print("v235_source_access_triage_self_test=ok", flush=True)
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
