#!/usr/bin/env python3
"""Static safety gate for KG1 scripts, HF job launchers, and notebooks.

This gate catches repository-level regressions that are cheaper to block before
running Colab, HF Jobs, or paid GPU work. It is intentionally conservative for
training/preference files: format-only negatives are allowed only in diagnostic
builders/gates, never in active HF jobs or notebooks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OLD_MIXED_V435E_PATH = "data/v435e_adapter_probe_preference/20260515T_v435e_from_h200_probe"
OLD_MIXED_V435E_TRAIN_SHA = "7f5e11770ac09c15e695cf4690df2fe7b5985b4a8b5bf5f8a201e8b71fe8be81"
OLD_MIXED_V435E_VAL_SHA = "d66752ab8470e145744a8bf80bc9b8beab7a4a3479d9161d03d6cc61c8ff9d92"

CRITICAL_SNIPPETS = {
    "scripts/build_v435e_adapter_probe_preference_dataset.py": {
        "correct rows excluded by default": "Correct adapter rows are not included by default",
        "diagnostic flag only": "--include-format-negatives",
        "include flag manifest": "\"include_format_negatives\": args.include_format_negatives",
        "format diagnostic warning": "format-only negatives are useful for a format audit",
    },
    "scripts/run_v435f_adapter_probe_preference_gate.py": {
        "format absence condition": "format_negatives_absent_for_preference",
        "allow flag": "--allow-format-negatives",
        "format row count": "format_negative_rows",
        "default hard-only path": "20260515T_v435e_hardneg_only",
    },
    "scripts/hf_job_train_v315_preference.py": {
        "format default false": "ALLOW_FORMAT_NEGATIVES = env_bool(\"ALLOW_FORMAT_NEGATIVES\", False)",
        "format rows blocked": "format_negative_blocked",
        "negative type accuracy": "negative_type_accuracy",
        "negative type from tokenized pair": "pair.get(\"negative_type\")",
    },
    "scripts/kg1_pre_paid_job_integration_gate.py": {
        "dataset content audit": "audit_dataset_file",
        "target template check": "Final answer: \\\\boxed{",
        "audit manifest gate": "hf_gpu_allowed_for_same_objective",
        "h200 timeout gate": "launcher_timeout_not_one_hour",
        "first checkpoint eval gate": "launcher_missing_first_checkpoint_eval",
        "format negatives blocked": "launcher_allows_format_negatives",
    },
}

TRUE_FORMAT_NEGATIVE_RE = re.compile(
    r"ALLOW_FORMAT_NEGATIVES\s*(?:=|:)\s*['\"]?(?:1|true|yes|on)['\"]?",
    re.IGNORECASE,
)
CLI_FORMAT_NEGATIVE_RE = re.compile(r"--(?:include|allow)-format-negatives\b")


@dataclass
class Finding:
    path: str
    level: str
    code: str
    detail: str


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def run_git(args: list[str], check: bool = False) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stdout)
    return completed.stdout


def read_path_text(path: Path) -> str:
    if path.suffix.lower() != ".ipynb":
        return path.read_text(encoding="utf-8", errors="replace")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.append("".join(str(item) for item in source))
        else:
            chunks.append(str(source))
    return "\n".join(chunks)


def is_scannable(path: Path) -> bool:
    rel = repo_rel(path)
    suffix = path.suffix.lower()
    if suffix not in {".py", ".ipynb", ".sh", ".yml", ".yaml"}:
        return False
    return (
        rel.startswith("scripts/")
        or rel.startswith("notebooks/")
        or rel.startswith(".github/workflows/")
        or rel.startswith("artifacts/")
    )


def is_hf_job_or_notebook(path: Path, text: str) -> bool:
    rel = repo_rel(path)
    name = path.name.lower()
    if rel == "scripts/kg1_static_safety_gate.py":
        return False
    if path.suffix.lower() == ".ipynb":
        return True
    if "api.run_job(" in text or "HfApi(" in text or "huggingface_hub" in text:
        return True
    if rel.startswith("scripts/hf_job_"):
        return True
    if name.startswith("launch_") and "hf" in rel.lower():
        return True
    return False


def is_archived_fail_closed(text: str) -> bool:
    required = [
        "Archived V436 launcher",
        "raise RuntimeError(",
        "format-only negatives",
        "hard-negative-only V435E",
    ]
    return all(snippet in text for snippet in required)


def audit_text(path: Path, text: str) -> list[Finding]:
    rel = repo_rel(path)
    findings: list[Finding] = []
    job_or_notebook = is_hf_job_or_notebook(path, text)

    old_markers = [OLD_MIXED_V435E_PATH, OLD_MIXED_V435E_TRAIN_SHA, OLD_MIXED_V435E_VAL_SHA]
    if job_or_notebook and any(marker in text for marker in old_markers) and not is_archived_fail_closed(text):
        findings.append(
            Finding(
                rel,
                "error",
                "old_mixed_v435e_dataset_referenced",
                "Active job/notebook references archived V435E mixed preference data.",
            )
        )

    if job_or_notebook and TRUE_FORMAT_NEGATIVE_RE.search(text):
        findings.append(
            Finding(
                rel,
                "error",
                "allow_format_negatives_enabled",
                "Active job/notebook must not enable ALLOW_FORMAT_NEGATIVES.",
            )
        )

    if job_or_notebook and CLI_FORMAT_NEGATIVE_RE.search(text):
        findings.append(
            Finding(
                rel,
                "error",
                "format_negative_cli_in_active_job",
                "Active job/notebook must not pass --include-format-negatives or --allow-format-negatives.",
            )
        )

    for critical_rel, snippets in CRITICAL_SNIPPETS.items():
        if rel != critical_rel:
            continue
        for name, snippet in snippets.items():
            if snippet not in text:
                findings.append(Finding(rel, "error", "critical_safety_snippet_missing", name))
    return findings


def discover_changed_paths(from_ref: str | None, to_ref: str) -> list[Path]:
    if from_ref:
        output = run_git(["diff", "--name-only", "--diff-filter=ACMRT", from_ref, to_ref], check=False)
        raw = [line.strip() for line in output.splitlines() if line.strip()]
    else:
        output = run_git(["status", "--short"], check=False)
        raw = []
        for line in output.splitlines():
            if not line.strip():
                continue
            raw.append(line[3:].strip())
    return sorted({ROOT / item for item in raw if (ROOT / item).exists()})


def load_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.paths_file:
        for line in args.paths_file.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item:
                paths.append(ROOT / item if not Path(item).is_absolute() else Path(item))
    if args.paths:
        paths.extend(path if path.is_absolute() else ROOT / path for path in args.paths)
    if not paths:
        paths = discover_changed_paths(args.changed_from or None, args.changed_to)
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(item for item in path.rglob("*") if item.is_file())
        else:
            expanded.append(path)
    return sorted({path for path in expanded if path.exists() and is_scannable(path)})


def audit_paths(paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        try:
            findings.extend(audit_text(path, read_path_text(path)))
        except Exception as exc:
            findings.append(Finding(repo_rel(path), "error", "static_safety_read_failed", repr(exc)))
    for critical_rel, snippets in CRITICAL_SNIPPETS.items():
        critical_path = ROOT / critical_rel
        if critical_path.exists() and critical_path not in paths:
            text = critical_path.read_text(encoding="utf-8", errors="replace")
            for name, snippet in snippets.items():
                if snippet not in text:
                    findings.append(Finding(critical_rel, "error", "critical_safety_snippet_missing", name))
    return findings


def run_self_test() -> int:
    print("=== KG1 STATIC SAFETY GATE SELF TEST START ===", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        bad = tmp / "launch_bad_hf.py"
        bad.write_text(
            "from huggingface_hub import HfApi\n"
            f"DATA_ROOT='{OLD_MIXED_V435E_PATH}'\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        archived = tmp / "launch_archived_hf.py"
        archived.write_text(
            '"""Archived V436 launcher with format-only negatives and hard-negative-only V435E."""\n'
            f"DATA_ROOT='{OLD_MIXED_V435E_PATH}'\n"
            "def main():\n    raise RuntimeError('Archived launcher: hard-negative-only V435E required')\n",
            encoding="utf-8",
        )
        enabled = tmp / "job_enabled.py"
        enabled.write_text("from huggingface_hub import HfApi\nALLOW_FORMAT_NEGATIVES=1\n", encoding="utf-8")

        bad_findings = audit_text(bad, bad.read_text(encoding="utf-8"))
        if "old_mixed_v435e_dataset_referenced" not in {item.code for item in bad_findings}:
            print("missing old mixed dataset self-test finding", flush=True)
            return 1
        archived_findings = audit_text(archived, archived.read_text(encoding="utf-8"))
        if archived_findings:
            print(json.dumps([item.__dict__ for item in archived_findings], indent=2), flush=True)
            return 1
        enabled_findings = audit_text(enabled, enabled.read_text(encoding="utf-8"))
        if "allow_format_negatives_enabled" not in {item.code for item in enabled_findings}:
            print("missing ALLOW_FORMAT_NEGATIVES self-test finding", flush=True)
            return 1
    print("kg1_static_safety_gate_self_test=ok", flush=True)
    print("=== KG1 STATIC SAFETY GATE SELF TEST END ===", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Files to audit.")
    parser.add_argument("--paths-file", type=Path, default=None, help="File containing repo-relative paths to audit.")
    parser.add_argument("--changed-from", default="", help="Git ref/sha to diff from.")
    parser.add_argument("--changed-to", default="HEAD", help="Git ref/sha to diff to.")
    parser.add_argument("--allow-empty", action="store_true", help="Return success when no scannable files are selected.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    paths = load_paths(args)
    findings = audit_paths(paths)
    if not paths and not args.allow_empty:
        findings.append(
            Finding(
                "",
                "error",
                "no_scannable_files_selected",
                "Pass files, --paths-file, --changed-from, or --allow-empty explicitly.",
            )
        )
    report: dict[str, Any] = {
        "schema_version": "kg1_static_safety_gate_v1",
        "ok": not any(item.level == "error" for item in findings),
        "file_count": len(paths),
        "files": [repo_rel(path) for path in paths],
        "findings": [item.__dict__ for item in findings],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
