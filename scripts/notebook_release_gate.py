#!/usr/bin/env python3
"""Release gate for KG1 Colab/Jupyter notebooks.

The gate is intentionally static and cheap. It validates every notebook path
passed on the command line, or every changed notebook discovered from git.

It enforces the project rules that must be true before a notebook is pushed or
handed to the user:

- valid ipynb JSON / nbformat 4;
- every code cell parses as Python;
- every code cell has explicit START/END progress markers;
- every code cell starts with a ``# CELL:`` header;
- notebook contains an exact Colab URL when it is under notebooks/;
- long-running commands go through the logging wrapper pattern;
- training/eval notebooks include safety gates for data hashes, Drive adapter,
  GPU/runtime, dependency ordering, weak/full gates, and no auto-submit.

For old historical notebooks that do not meet the modern contract, run this
gate only on changed notebooks. New or modified notebooks must pass.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_RE = re.compile(r"\.ipynb$", re.IGNORECASE)
COLAB_RE = re.compile(r"https://colab\.research\.google\.com/github/[^\s`)]+\.ipynb")
GITHUB_BLOB_RE = re.compile(r"https://github\.com/[^\s`)]+/blob/[^\s`)]+\.ipynb")

GENERIC_REQUIRED_SNIPPETS = [
    "repo_commit",
    "py_compile",
]

GIT_CLONE_SNIPPETS = [
    "git clone",
    "'git', 'clone'",
    '"git", "clone"',
]

DATA_GATE_SNIPPETS = [
    "sha256_file",
    "EXPECTED_TRAIN_SHA256",
    "EXPECTED_VAL_SHA256",
    "MIN_TRAIN_EXAMPLES",
    "MIN_VAL_EXAMPLES",
]

TRAIN_GATE_SNIPPETS = [
    "TOKENIZE_ONLY_DRY_RUN",
    "MAX_PROMPT_TRUNCATION_RATE",
    "REQUIRE_OFFSET_MASK",
    "INIT_ADAPTER_DIR",
    "RUN_TRAIN",
]

RUNTIME_GATE_SNIPPETS = [
    "cuda_available",
    "gpu_total_gib",
    "content_free_gib",
    "causal_conv1d",
    "mamba_ssm",
]

V194_GATE_SNIPPETS = [
    "V194_ADAPTER",
    "adapter_config.json",
    "adapter_model.safetensors",
    "target_modules",
    "target_parameters",
]

EVAL_GATE_SNIPPETS = [
    "weak_gate_pass_for_full",
    "WEAK_MIN_FOR_FULL",
    "WEAK_EQ_MIN_FOR_FULL",
    "WEAK_BIT_MIN_FOR_FULL",
    "WEAK_MAX_TRUNC_FOR_FULL",
    "FULL_MIN_CANDIDATE",
    "FULL_MAX_TRUNC",
]

BANNED_SNIPPETS = [
    "ALLOW_KAGGLE_SUBMIT = True",
    "kaggle competitions submit",
    "v217_shortans_lr3e8_s24",
    "v217_score_push_train_no_prompt_trunc",
]


@dataclass
class Finding:
    level: str
    code: str
    detail: str


@dataclass
class NotebookAudit:
    path: str
    sha256: str
    code_cells: int
    markdown_cells: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(item.level == "error" for item in self.findings)


def run_git(args: list[str], check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(item) for item in source)
    return str(source)


def add(findings: list[Finding], level: str, code: str, detail: str) -> None:
    findings.append(Finding(level=level, code=code, detail=detail))


def contains_any(text: str, snippets: list[str]) -> bool:
    return any(snippet in text for snippet in snippets)


def missing_snippets(text: str, snippets: list[str]) -> list[str]:
    return [snippet for snippet in snippets if snippet not in text]


def is_training_or_eval_notebook(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "hf_job_train",
        "evaluate_lora_adapter",
        "final_adapter",
        "weak eval",
        "vllm",
        "lora",
    ]
    return any(marker in lowered for marker in markers)


def is_generated_colab(path: Path, notebook: dict[str, Any], text: str) -> bool:
    rel = repo_rel(path)
    return path.suffix.lower() == ".ipynb" and (
        "colab.research.google.com" in text
        or rel.startswith("notebooks/")
        or rel.startswith("competent-shamir/notebooks/")
    )


def audit_cells(notebook: dict[str, Any], findings: list[Finding]) -> tuple[list[str], list[str]]:
    code_cells: list[str] = []
    markdown_cells: list[str] = []
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        add(findings, "error", "notebook_cells_not_list", "notebook.cells must be a list")
        return code_cells, markdown_cells
    for idx, cell in enumerate(cells, start=1):
        ctype = cell.get("cell_type")
        source = cell_source(cell)
        if ctype == "code":
            code_cells.append(source)
            first_line = source.splitlines()[0].strip() if source.splitlines() else ""
            if not first_line.startswith("# CELL:"):
                add(findings, "error", "code_cell_missing_cell_header", f"cell {idx}: first line={first_line!r}")
            if " START ===" not in source or " END ===" not in source:
                add(findings, "error", "code_cell_missing_start_end_markers", f"cell {idx}: missing START/END")
            try:
                ast.parse(source)
            except SyntaxError as exc:
                add(findings, "error", "code_cell_syntax_error", f"cell {idx}: {exc}")
        elif ctype == "markdown":
            markdown_cells.append(source)
        else:
            add(findings, "warning", "unknown_cell_type", f"cell {idx}: {ctype!r}")
    return code_cells, markdown_cells


def audit_colab_urls(path: Path, text: str, findings: list[Finding]) -> None:
    rel = repo_rel(path)
    if not (rel.startswith("notebooks/") or rel.startswith("competent-shamir/notebooks/")):
        return
    colab_urls = COLAB_RE.findall(text)
    if not colab_urls:
        add(findings, "error", "missing_colab_url", "notebook under notebooks/ must include exact Colab URL")
        return
    if not any(url.endswith(rel) for url in colab_urls):
        add(findings, "error", "colab_url_path_mismatch", f"expected a Colab URL ending in {rel}")
    github_urls = GITHUB_BLOB_RE.findall(text)
    if github_urls and not any(url.endswith(rel) for url in github_urls):
        add(findings, "warning", "github_url_path_mismatch", f"expected a GitHub URL ending in {rel}")


def audit_logging_and_commands(code_text: str, findings: list[Finding]) -> None:
    long_running_markers = [
        "pip install",
        "git clone",
        "hf_job_train",
        "evaluate_lora_adapter",
        "AutoModelForCausalLM",
        "LLM(",
    ]
    if contains_any(code_text, long_running_markers):
        if "run_cmd(" not in code_text:
            add(findings, "error", "long_running_without_run_cmd", "long-running notebook must use run_cmd logging wrapper")
        for required in ["log_path", "returncode", "COMMAND START", "COMMAND END"]:
            if required not in code_text:
                add(findings, "error", "run_cmd_contract_missing", required)
    if "subprocess.run" in code_text and "run_cmd(" not in code_text:
        add(findings, "error", "raw_subprocess_without_wrapper", "use run_cmd so command, log, rc, and heartbeat are visible")


def audit_training_eval_contract(text: str, findings: list[Finding]) -> None:
    if not is_training_or_eval_notebook(text):
        return
    for group_name, snippets in [
        ("data_gate", DATA_GATE_SNIPPETS),
        ("train_gate", TRAIN_GATE_SNIPPETS),
        ("runtime_gate", RUNTIME_GATE_SNIPPETS),
        ("v194_gate", V194_GATE_SNIPPETS),
        ("eval_gate", EVAL_GATE_SNIPPETS),
    ]:
        for snippet in missing_snippets(text, snippets):
            add(findings, "error", f"{group_name}_snippet_missing", snippet)
    if "TOKENIZE_ONLY_DRY_RUN" in text and "KG1_" in text:
        if "RUN_TRAIN', '0'" not in text and 'RUN_TRAIN", "0"' not in text:
            add(findings, "error", "train_default_not_off", "training notebooks must default RUN_TRAIN to 0/off")
    if "ensure_vllm_for_eval" in text:
        train_pos = text.find("TRAIN START")
        call_positions = [
            match.start()
            for match in re.finditer(r"(?m)^\s*ensure_vllm_for_eval\(\)", text)
        ]
        if train_pos >= 0 and call_positions and min(call_positions) < train_pos:
            add(findings, "error", "vllm_eval_before_train", "vLLM install/import must happen after training")
    for banned in BANNED_SNIPPETS:
        if banned in text:
            add(findings, "error", "banned_snippet_present", banned)
    if "ALLOW_KAGGLE_SUBMIT" not in text:
        add(findings, "error", "hard_submit_lock_missing", "missing ALLOW_KAGGLE_SUBMIT hard lock")
    elif "ALLOW_KAGGLE_SUBMIT = False" not in text and "ALLOW_KAGGLE_SUBMIT=False" not in text:
        add(findings, "error", "hard_submit_lock_not_false", "ALLOW_KAGGLE_SUBMIT must be False")


def audit_notebook(path: Path) -> NotebookAudit:
    findings: list[Finding] = []
    rel = repo_rel(path)
    if not path.exists():
        add(findings, "error", "notebook_missing", rel)
        return NotebookAudit(path=rel, sha256="", code_cells=0, markdown_cells=0, findings=findings)
    try:
        notebook = load_notebook(path)
    except Exception as exc:
        add(findings, "error", "notebook_json_invalid", repr(exc))
        return NotebookAudit(path=rel, sha256=sha256_file(path), code_cells=0, markdown_cells=0, findings=findings)
    if notebook.get("nbformat") != 4:
        add(findings, "error", "nbformat_not_4", f"got {notebook.get('nbformat')!r}")
    code_cells, markdown_cells = audit_cells(notebook, findings)
    text = "\n".join(markdown_cells + code_cells)
    if is_generated_colab(path, notebook, text):
        audit_colab_urls(path, text, findings)
        audit_logging_and_commands("\n".join(code_cells), findings)
        audit_training_eval_contract(text, findings)
    for snippet in GENERIC_REQUIRED_SNIPPETS:
        if is_training_or_eval_notebook(text) and snippet not in text:
            add(findings, "error", "generic_training_snippet_missing", snippet)
    if is_training_or_eval_notebook(text) and not contains_any(text, GIT_CLONE_SNIPPETS):
        add(findings, "error", "generic_training_snippet_missing", "git clone")
    return NotebookAudit(
        path=rel,
        sha256=sha256_file(path),
        code_cells=len(code_cells),
        markdown_cells=len(markdown_cells),
        findings=findings,
    )


def changed_notebooks(from_ref: str | None, to_ref: str) -> list[Path]:
    if from_ref:
        output = run_git(["diff", "--name-only", "--diff-filter=ACMRT", from_ref, to_ref], check=False)
    else:
        output = run_git(["status", "--short"], check=False)
        paths: list[str] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            paths.append(line[3:].strip())
        return sorted({ROOT / path for path in paths if NOTEBOOK_RE.search(path)})
    return sorted({ROOT / line.strip() for line in output.splitlines() if NOTEBOOK_RE.search(line.strip())})


def all_notebooks() -> list[Path]:
    roots = [ROOT / "notebooks", ROOT / "competent-shamir" / "notebooks"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("*.ipynb")))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Notebook paths to audit.")
    parser.add_argument("--all", action="store_true", help="Audit all notebooks under notebooks/ and competent-shamir/notebooks/.")
    parser.add_argument("--changed-from", default="", help="Git ref/sha to diff from.")
    parser.add_argument("--changed-to", default="HEAD", help="Git ref/sha to diff to.")
    parser.add_argument("--output-json", type=Path, default=ROOT / "artifacts" / "notebook_release_gate" / "report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        paths = all_notebooks()
    elif args.paths:
        paths = [path if path.is_absolute() else ROOT / path for path in args.paths]
    else:
        paths = changed_notebooks(args.changed_from or None, args.changed_to)
    paths = [path for path in paths if NOTEBOOK_RE.search(str(path))]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    audits = [audit_notebook(path) for path in paths]
    report = {
        "schema_version": "kg1_notebook_release_gate_v1",
        "ok": all(audit.ok for audit in audits),
        "notebook_count": len(audits),
        "notebooks": [
            {
                "path": audit.path,
                "sha256": audit.sha256,
                "code_cells": audit.code_cells,
                "markdown_cells": audit.markdown_cells,
                "ok": audit.ok,
                "findings": [finding.__dict__ for finding in audit.findings],
            }
            for audit in audits
        ],
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
