#!/usr/bin/env python3
"""Deep V207B Colab notebook audit gate.

This gate is tailored to notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb.
It performs notebook, code-cell, embedded-script, baseline-artifact, public
adapter manifest, remote raw URL, and Kaggle model reference checks. It is
read-only except for writing audit reports under artifacts/.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb"
DEFAULT_BUILDER = ROOT / "scripts" / "build_v207b_external_adapter_triage_colab.py"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "v207b_deep_audit"
BRANCH = "v207b-external-triage"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb"
)
RAW_BASE = f"https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/{BRANCH}"


@dataclass
class Finding:
    severity: str
    code: str
    detail: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def add(findings: list[Finding], severity: str, code: str, detail: str) -> None:
    findings.append(Finding(severity, code, detail))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def source_to_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(x) for x in source)
    return str(source or "")


def load_notebook(path: Path, findings: list[Finding]) -> dict[str, Any]:
    if not path.exists():
        add(findings, "error", "notebook_missing", str(path))
        return {}
    try:
        payload = json.loads(read_text(path))
    except Exception as exc:
        add(findings, "error", "notebook_json_invalid", repr(exc))
        return {}
    if payload.get("nbformat") != 4:
        add(findings, "error", "notebook_nbformat", f"expected 4 got {payload.get('nbformat')}")
    if not isinstance(payload.get("cells"), list):
        add(findings, "error", "notebook_cells_missing", "cells is not a list")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def line_audit(nb: dict[str, Any], findings: list[Finding]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    cells = nb.get("cells", [])
    all_text_parts: list[str] = []
    cell_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        text = source_to_text(cell.get("source"))
        all_text_parts.append(text)
        lines = text.splitlines()
        start_count = text.count("START")
        end_count = text.count("END")
        ast_status = "not_code"
        ast_error = ""
        if cell_type == "code":
            try:
                ast.parse(text)
                ast_status = "ok"
            except SyntaxError as exc:
                ast_status = "error"
                ast_error = f"{exc.msg} line={exc.lineno} offset={exc.offset}"
                add(findings, "error", "code_cell_syntax", f"cell={idx} {ast_error}")
            if start_count < 1 or end_count < 1:
                add(findings, "error", "code_cell_missing_start_end", f"cell={idx}")
        cell_rows.append(
            {
                "cell_index": idx,
                "cell_type": cell_type,
                "line_count": len(lines),
                "char_count": len(text),
                "start_count": start_count,
                "end_count": end_count,
                "ast_status": ast_status,
                "ast_error": ast_error,
                "first_line": lines[0] if lines else "",
            }
        )
        for line_no, line in enumerate(lines, 1):
            lowered = line.lower()
            flags: list[str] = []
            if "competitions submit" in lowered or "competition_submit" in lowered:
                flags.append("submit_code")
            if "git clone" in lowered:
                flags.append("git_clone")
            if "userdata.get" in line:
                flags.append("colab_secret_read")
            if "run_cmd(" in line:
                flags.append("command_runner")
            if "raise RuntimeError" in line or "raise FileNotFoundError" in line:
                flags.append("hard_fail")
            if "print(" in line:
                flags.append("progress_log")
            line_rows.append(
                {
                    "cell_index": idx,
                    "cell_type": cell_type,
                    "line_no": line_no,
                    "flags": ";".join(flags),
                    "text": line,
                }
            )
    return cell_rows, line_rows, "\n".join(all_text_parts)


def check_required_contract(text: str, findings: list[Finding]) -> None:
    required = [
        "KG1 V207B External Adapter Triage Colab",
        COLAB_URL,
        "V207B DRIVE MOUNT START",
        "V207B CONFIG START",
        "V207B COLAB SECRETS BRIDGE START",
        "HF_TOKEN_ready",
        "KAGGLE_CREDENTIALS_READY",
        "LOG_POLICY",
        "DRIVE_ORGANIZED_OUTPUTS",
        "MANIFEST_DIR",
        "PRINT_COMMAND_OUTPUT",
        "command_output_suppressed_lines",
        "V207B HELPERS START",
        "V207B DEPENDENCY CHECK START",
        "kaggle==2.0.2",
        "vllm==0.20.1",
        "VLLM_WHEEL_URL",
        "VLLM_INSTALL_POLICY",
        "vllm-0.20.1%2Bcu129",
        "https://download.pytorch.org/whl/cu129",
        "vllm_import_preflight.log",
        "vllm_import_preflight_ok",
        "libcudart.so.13",
        "KAGGLE_CMD_PREFIX",
        "V207B WORKSPACE SETUP START",
        "V207B SCRIPT BOOTSTRAP START",
        "py_compile.compile",
        "V207B V207A ARTIFACT CHECK START",
        "baseline_artifact_audit",
        "Expected 947 validation/baseline rows",
        "V207B PUBLIC KAGGLE ADAPTER DOWNLOAD START",
        "adapter_model_too_small",
        "V207B CANDIDATE DISCOVERY START",
        "V207B STRUCTURE AUDIT START",
        "V207B WEAK FAMILY SCREEN START",
        "V207B FULL GATE START",
        "V207B FINAL SUMMARY START",
        "ALLOW_KAGGLE_SUBMIT = False",
        "submit_disabled",
    ]
    for marker in required:
        if marker not in text:
            add(findings, "error", "required_marker_missing", marker)

    forbidden = [
        r"kaggle\s+competitions\s+submit",
        r"competition_submit\s*\(",
        r"KaggleApi\s*\(",
        r"ALLOW_KAGGLE_SUBMIT\s*=\s*True",
        r"git\s+clone",
        r"\bREPO_URL\b",
        r"\bREPO_BRANCH\b",
        r"python\s+-m\s+kaggle\s+--version",
    ]
    for pattern in forbidden:
        if re.search(pattern, text):
            add(findings, "error", "forbidden_pattern_present", pattern)

    secret_value_print_patterns = [
        r"print\s*\(\s*secret_key\b",
        r"print\s*\(\s*env_key\b",
        r"print\s*\(\s*os\.environ\[[\"']KAGGLE_KEY[\"']\]",
        r"print\s*\(\s*os\.environ\[[\"']HF_TOKEN[\"']\]",
        r"print\s*\(\s*os\.environ\[[\"']HF_KEY[\"']\]",
    ]
    for pattern in secret_value_print_patterns:
        if re.search(pattern, text):
            add(findings, "error", "secret_value_print_risk", pattern)

    log_noise_forbidden = [
        "print(per.to_string(index=False))",
        "print('candidate =', item['label'], item['path'])",
        "download_status =', json.dumps(row, indent=2",
        "audit_df[['label', 'ready_for_eval', 'r', 'tensor_count', 'model_bytes', 'reason', 'path']].to_string",
    ]
    for marker in log_noise_forbidden:
        if marker in text:
            add(findings, "error", "noisy_notebook_log_pattern", marker)
    if "PRINT_COMMAND_OUTPUT or is_essential_output_line(line)" not in text:
        add(findings, "error", "command_log_filter_missing", "run_cmd must suppress noisy stdout by default")
    if "full_command_log =" not in text:
        add(findings, "error", "drive_command_log_missing", "full subprocess logs must be persisted to Drive")

    if "userdata.get(name)" not in text:
        add(findings, "error", "colab_secret_reader_missing", "userdata.get(name)")
    for secret_name in ["HF_TOKEN", "HF_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"]:
        if secret_name not in text:
            add(findings, "error", "colab_secret_name_missing", secret_name)


def extract_embedded_files(nb: dict[str, Any], findings: list[Finding]) -> dict[str, str]:
    for cell in nb.get("cells", []):
        text = source_to_text(cell.get("source"))
        if "FILES = json.loads(" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            add(findings, "error", "embedded_cell_syntax", repr(exc))
            return {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "FILES" for target in node.targets):
                continue
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "loads":
                payload = ast.literal_eval(call.args[0])
                return json.loads(payload)
    add(findings, "error", "embedded_files_payload_missing", "FILES json payload not found")
    return {}


def check_embedded_sources(nb: dict[str, Any], findings: list[Finding]) -> dict[str, Any]:
    expected_files = {
        "src/__init__.py",
        "src/competition_utils.py",
        "scripts/evaluate_lora_adapter.py",
        "scripts/solve_rate_gate.py",
    }
    embedded = extract_embedded_files(nb, findings)
    summary: dict[str, Any] = {"files": {}}
    if set(embedded) != expected_files:
        add(findings, "error", "embedded_file_set_mismatch", f"got={sorted(embedded)}")
    for rel, content in embedded.items():
        local_path = ROOT / rel
        item = {
            "embedded_bytes": len(content.encode("utf-8")),
            "embedded_sha256": sha256_bytes(content.encode("utf-8")),
            "local_exists": local_path.exists(),
            "local_bytes": local_path.stat().st_size if local_path.exists() else 0,
            "local_sha256": sha256_file(local_path) if local_path.exists() else "",
            "matches_local": False,
            "py_compile_ok": None,
        }
        if not local_path.exists():
            add(findings, "error", "embedded_local_missing", rel)
        else:
            local_text = read_text(local_path)
            item["matches_local"] = local_text == content
            if not item["matches_local"]:
                add(findings, "error", "embedded_source_mismatch", rel)
            if rel.endswith(".py"):
                proc = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(local_path)],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                item["py_compile_ok"] = proc.returncode == 0
                if proc.returncode != 0:
                    add(findings, "error", "embedded_source_py_compile_failed", f"{rel}: {proc.stdout[:500]}")
        summary["files"][rel] = item
    return summary


def check_script_contracts(findings: list[Finding]) -> dict[str, Any]:
    eval_path = ROOT / "scripts" / "evaluate_lora_adapter.py"
    gate_path = ROOT / "scripts" / "solve_rate_gate.py"
    comp_path = ROOT / "src" / "competition_utils.py"
    scripts = {
        "evaluate_lora_adapter.py": eval_path,
        "solve_rate_gate.py": gate_path,
        "competition_utils.py": comp_path,
    }
    summary: dict[str, Any] = {}
    for name, path in scripts.items():
        summary[name] = {"exists": path.exists(), "sha256": sha256_file(path) if path.exists() else ""}
        if not path.exists():
            add(findings, "error", "script_missing", str(path))
            continue
        text = read_text(path)
        try:
            ast.parse(text)
        except SyntaxError as exc:
            add(findings, "error", "script_syntax_error", f"{name}: {exc}")
        for forbidden in ["competitions submit", "competition_submit(", "KaggleApi("]:
            if forbidden in text:
                add(findings, "error", "script_submit_forbidden", f"{name}: {forbidden}")

    eval_text = read_text(eval_path) if eval_path.exists() else ""
    for marker in [
        "VLLM_USE_DEEP_GEMM",
        "VLLM_MOE_USE_DEEP_GEMM",
        "VLLM_DEEP_GEMM_WARMUP",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        "prepare_merged_predictions",
        "raw_predictions_pre_score_csv",
    ]:
        if marker not in eval_text:
            add(findings, "error", "evaluate_contract_missing", marker)

    gate_text = read_text(gate_path) if gate_path.exists() else ""
    for marker in ["family_regression_tolerance", "min_net_gain", "min_boxed_rate", "solve_rate_gate_report.json"]:
        if marker not in gate_text:
            add(findings, "error", "gate_contract_missing", marker)

    comp_text = read_text(comp_path) if comp_path.exists() else ""
    for marker in ["verify_answer", "extract_final_answer", "MODEL_REVISION", "cbd3fa9f933d55ef16a84236559f4ee2a0526848"]:
        if marker not in comp_text:
            add(findings, "error", "competition_utils_contract_missing", marker)

    return summary


def check_baseline_artifacts(findings: list[Finding]) -> dict[str, Any]:
    import pandas as pd

    pred_path = ROOT / "artifacts" / "drive_exports" / "v194_baseline_predictions.csv"
    per_path = ROOT / "artifacts" / "drive_exports" / "v194_baseline_per_task.csv"
    report_path = ROOT / "artifacts" / "drive_exports" / "v194_baseline_eval_report.json"
    summary: dict[str, Any] = {}
    for path in [pred_path, per_path, report_path]:
        summary[str(path.relative_to(ROOT))] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else "",
        }
        if not path.exists():
            add(findings, "error", "baseline_artifact_missing", str(path))
    if not pred_path.exists() or not per_path.exists() or not report_path.exists():
        return summary

    pred = pd.read_csv(pred_path)
    per = pd.read_csv(per_path)
    report = json.loads(read_text(report_path))
    required_pred = {"id", "answer", "prediction", "raw_output", "correct", "truncated"}
    required_per = {"task_type", "total", "correct", "accuracy", "truncated", "truncation_rate"}
    missing_pred = sorted(required_pred - set(pred.columns))
    missing_per = sorted(required_per - set(per.columns))
    if missing_pred:
        add(findings, "error", "baseline_predictions_schema", str(missing_pred))
    if missing_per:
        add(findings, "error", "baseline_per_task_schema", str(missing_per))
    correct = int(pred["correct"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    truncated = int(pred["truncated"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    summary["baseline_metrics"] = {
        "rows": int(len(pred)),
        "unique_ids": bool(pred["id"].astype(str).is_unique),
        "correct": correct,
        "truncated": truncated,
        "report_rows": int(report.get("rows", -1)),
        "report_correct": int(report.get("correct", -1)),
        "report_truncated": int(report.get("truncated", -1)),
        "report_accuracy": float(report.get("accuracy", 0.0)),
    }
    if len(pred) != 947 or not pred["id"].astype(str).is_unique:
        add(findings, "error", "baseline_predictions_rows_or_ids", json.dumps(summary["baseline_metrics"]))
    if report.get("rows") != len(pred) or report.get("correct") != correct or report.get("truncated") != truncated:
        add(findings, "error", "baseline_report_mismatch", json.dumps(summary["baseline_metrics"]))
    overall = per[per["task_type"].astype(str).eq("OVERALL")]
    if len(overall) != 1:
        add(findings, "error", "baseline_overall_row_count", str(len(overall)))
    else:
        row = overall.iloc[0]
        if int(row["total"]) != len(pred) or int(row["correct"]) != correct:
            add(findings, "error", "baseline_overall_mismatch", row.to_json())
    weak = per[per["task_type"].isin(["bit_manipulation", "equation_transform"])]
    summary["weak_baseline"] = {
        "correct": int(weak["correct"].sum()),
        "total": int(weak["total"].sum()),
        "truncated": int(weak["truncated"].sum()),
    }
    if summary["weak_baseline"] != {"correct": 190, "total": 315, "truncated": 1}:
        add(findings, "error", "weak_baseline_unexpected", json.dumps(summary["weak_baseline"]))
    return summary


def check_candidate_manifests(builder_text: str, findings: list[Finding]) -> dict[str, Any]:
    csv_path = ROOT / "artifacts" / "live_intel_20260507" / "public_adapter_triage_candidates_2026-05-07.csv"
    json_path = ROOT / "artifacts" / "live_intel_20260507" / "public_adapter_triage_candidates_2026-05-07.json"
    summary: dict[str, Any] = {}
    if not csv_path.exists() or not json_path.exists():
        add(findings, "error", "candidate_manifest_missing", f"{csv_path} {json_path}")
        return summary
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = json.loads(read_text(json_path))
    normalized_csv_rows: list[dict[str, Any]] = []
    for row in csv_rows:
        normalized = dict(row)
        normalized["priority"] = int(normalized["priority"])
        normalized["expected_size_bytes"] = int(normalized["expected_size_bytes"])
        normalized["default_download"] = str(normalized["default_download"]).lower() == "true"
        normalized_csv_rows.append(normalized)
    if normalized_csv_rows != json_rows:
        add(findings, "error", "candidate_csv_json_mismatch", "CSV and JSON candidate manifests differ")
    summary = {
        "candidate_rows": len(json_rows),
        "default_download_rows": sum(1 for row in json_rows if row.get("default_download")),
        "default_download_gib": round(
            sum(int(row.get("expected_size_bytes") or 0) for row in json_rows if row.get("default_download"))
            / (1024**3),
            4,
        ),
        "total_expected_gib": round(
            sum(int(row.get("expected_size_bytes") or 0) for row in json_rows) / (1024**3),
            4,
        ),
    }
    if summary["candidate_rows"] != 16 or summary["default_download_rows"] != 13:
        add(findings, "error", "candidate_count_unexpected", json.dumps(summary))
    for row in json_rows:
        for required in ["label", "kaggle_ref", "drive_adapter_dir", "expected_primary_file", "expected_size_bytes"]:
            if required not in row:
                add(findings, "error", "candidate_field_missing", f"{row.get('label')}: {required}")
        if row.get("label") and row.get("label") not in builder_text:
            add(findings, "error", "candidate_missing_from_builder", str(row.get("label")))
        if row.get("expected_size_bytes") and str(row.get("expected_size_bytes")) not in builder_text:
            add(findings, "error", "candidate_size_missing_from_builder", str(row.get("label")))
    return summary


def check_remote_raw(findings: list[Finding]) -> dict[str, Any]:
    urls = {
        "notebook": f"{RAW_BASE}/notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb",
        "builder": f"{RAW_BASE}/scripts/build_v207b_external_adapter_triage_colab.py",
        "baseline_predictions": f"{RAW_BASE}/artifacts/drive_exports/v194_baseline_predictions.csv",
        "baseline_per_task": f"{RAW_BASE}/artifacts/drive_exports/v194_baseline_per_task.csv",
        "baseline_report": f"{RAW_BASE}/artifacts/drive_exports/v194_baseline_eval_report.json",
    }
    summary: dict[str, Any] = {}
    for name, url in urls.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kg1-v207b-audit"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
                status = getattr(response, "status", None)
        except Exception as exc:
            add(findings, "error", "remote_raw_fetch_failed", f"{name}: {exc}")
            summary[name] = {"url": url, "ok": False}
            continue
        summary[name] = {"url": url, "ok": status == 200, "bytes": len(data), "sha256": sha256_bytes(data)}
        if status != 200:
            add(findings, "error", "remote_raw_non_200", f"{name}: {status}")
        if name == "notebook":
            text = data.decode("utf-8", errors="replace")
            for marker in [
                "V207B COLAB SECRETS BRIDGE START",
                "vllm==0.20.1",
                "vllm-0.20.1%2Bcu129",
                "vllm_import_preflight_ok",
                "baseline_artifact_audit",
                "LOG_POLICY",
                "DRIVE_ORGANIZED_OUTPUTS",
                "command_output_suppressed_lines",
            ]:
                if marker not in text:
                    add(findings, "error", "remote_notebook_marker_missing", marker)
    return summary


def check_kaggle_refs(findings: list[Finding]) -> dict[str, Any]:
    csv_path = ROOT / "artifacts" / "live_intel_20260507" / "public_adapter_triage_candidates_2026-05-07.csv"
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8"))) if csv_path.exists() else []
    summary = {"checked": 0, "failures": []}
    for row in rows:
        ref = row["kaggle_ref"]
        cmd = ["kaggle", "models", "instances", "versions", "files", ref, "--csv"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=75,
            )
        except Exception as exc:
            add(findings, "error", "kaggle_ref_check_exception", f"{row.get('label')}: {exc}")
            summary["failures"].append(row.get("label"))
            continue
        out = proc.stdout or ""
        summary["checked"] += 1
        has_config = "adapter_config.json" in out
        has_model = row.get("expected_primary_file", "adapter_model.safetensors") in out
        if proc.returncode != 0 or not has_config or not has_model:
            add(
                findings,
                "error",
                "kaggle_ref_check_failed",
                f"{row.get('label')}: rc={proc.returncode} config={has_config} model={has_model} out={out[:300]}",
            )
            summary["failures"].append(row.get("label"))
    return summary


def run_help_gates(findings: list[Finding]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    commands = {
        "evaluate_help": [sys.executable, "scripts/evaluate_lora_adapter.py", "--help"],
        "solve_gate_help": [sys.executable, "scripts/solve_rate_gate.py", "--help"],
        "builder_compile": [sys.executable, "-m", "py_compile", "scripts/build_v207b_external_adapter_triage_colab.py"],
        "evaluate_compile": [sys.executable, "-m", "py_compile", "scripts/evaluate_lora_adapter.py"],
        "solve_gate_compile": [sys.executable, "-m", "py_compile", "scripts/solve_rate_gate.py"],
        "competition_utils_compile": [sys.executable, "-m", "py_compile", "src/competition_utils.py"],
    }
    for name, cmd in commands.items():
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        summary[name] = {"returncode": proc.returncode, "output_sample": (proc.stdout or "")[:500]}
        if proc.returncode != 0:
            add(findings, "error", "help_or_compile_gate_failed", f"{name}: {proc.stdout[:500]}")
    return summary


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    lines = [
        "# V207B Deep Notebook Audit",
        "",
        f"Generated UTC: {report['generated_at_utc']}",
        f"Notebook: `{report['notebook']}`",
        f"Colab URL: {COLAB_URL}",
        f"Status: **{report['status']}**",
        "",
        "## Summary",
        "",
        f"- Findings: {len(findings)}",
        f"- Errors: {sum(1 for f in findings if f['severity'] == 'error')}",
        f"- Warnings: {sum(1 for f in findings if f['severity'] == 'warning')}",
        f"- Cells: {report['notebook_summary'].get('cell_count')}",
        f"- Code cells: {report['notebook_summary'].get('code_cell_count')}",
        "",
        "## Findings",
        "",
    ]
    if findings:
        for finding in findings:
            lines.append(f"- `{finding['severity']}` `{finding['code']}`: {finding['detail']}")
    else:
        lines.append("- No findings.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Cell audit CSV: `{report['outputs']['cell_audit_csv']}`",
            f"- Line audit CSV: `{report['outputs']['line_audit_csv']}`",
            f"- JSON report: `{report['outputs']['json_report']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-remote", action="store_true")
    parser.add_argument("--check-kaggle-refs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    findings: list[Finding] = []

    notebook = load_notebook(args.notebook, findings)
    cell_rows, line_rows, all_text = line_audit(notebook, findings) if notebook else ([], [], "")
    check_required_contract(all_text, findings)
    embedded_summary = check_embedded_sources(notebook, findings) if notebook else {}
    script_summary = check_script_contracts(findings)
    baseline_summary = check_baseline_artifacts(findings)
    builder_text = read_text(args.builder) if args.builder.exists() else ""
    if not args.builder.exists():
        add(findings, "error", "builder_missing", str(args.builder))
    candidate_summary = check_candidate_manifests(builder_text, findings)
    help_gate_summary = run_help_gates(findings)
    remote_summary = check_remote_raw(findings) if args.check_remote else {}
    kaggle_ref_summary = check_kaggle_refs(findings) if args.check_kaggle_refs else {}

    cell_csv = args.output_dir / "v207b_cell_audit.csv"
    line_csv = args.output_dir / "v207b_line_audit.csv"
    json_report = args.output_dir / "v207b_deep_audit_report.json"
    md_report = args.output_dir / "V207B_DEEP_NOTEBOOK_AUDIT.md"
    write_csv(
        cell_csv,
        cell_rows,
        ["cell_index", "cell_type", "line_count", "char_count", "start_count", "end_count", "ast_status", "ast_error", "first_line"],
    )
    write_csv(line_csv, line_rows, ["cell_index", "cell_type", "line_no", "flags", "text"])

    errors = [finding for finding in findings if finding.severity == "error"]
    report = {
        "schema_version": "v207b_deep_audit_v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if not errors else "fail",
        "notebook": str(args.notebook),
        "builder": str(args.builder),
        "notebook_sha256": sha256_file(args.notebook) if args.notebook.exists() else "",
        "notebook_summary": {
            "cell_count": len(notebook.get("cells", [])) if notebook else 0,
            "code_cell_count": sum(1 for cell in notebook.get("cells", []) if cell.get("cell_type") == "code") if notebook else 0,
        },
        "embedded_sources": embedded_summary,
        "scripts": script_summary,
        "baseline_artifacts": baseline_summary,
        "candidate_manifests": candidate_summary,
        "help_gates": help_gate_summary,
        "remote_raw": remote_summary,
        "kaggle_refs": kaggle_ref_summary,
        "findings": [asdict(finding) for finding in findings],
        "outputs": {
            "cell_audit_csv": str(cell_csv),
            "line_audit_csv": str(line_csv),
            "json_report": str(json_report),
            "markdown_report": str(md_report),
        },
    }
    json_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(md_report, report)
    print(json.dumps({"status": report["status"], "errors": len(errors), "findings": len(findings), "report": str(json_report)}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
