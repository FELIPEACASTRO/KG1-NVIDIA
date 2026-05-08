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

V218_NOTEBOOK_REL = "notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb"
V218_TRAIN_REL = "data/v217/v217_short_answer_train.jsonl"
V218_VAL_REL = "data/v217/v217_short_answer_val.jsonl"
V218_TRAIN_SHA256 = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
V218_VAL_SHA256 = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"
V218_TRAIN_ROWS = 10206
V218_VAL_ROWS = 681

V218_REQUIRED_FILES = [
    ".gitattributes",
    V218_TRAIN_REL,
    V218_VAL_REL,
    "data/v217/v217_short_answer_manifest.json",
    "scripts/analyze_eval_predictions.py",
    "scripts/build_v218_decode_rescue_colab.py",
    "scripts/evaluate_lora_adapter.py",
    "scripts/notebook_release_gate.py",
    "src/__init__.py",
    "src/competition_utils.py",
]

V218_REQUIRED_SNIPPETS = {
    "repo clone branch": "'git', 'clone', '--depth', '1', '--branch', REPO_BRANCH",
    "repo sys.path insert": "sys.path.insert(0, str(ROOT))",
    "repo sys.path log": "repo_root_on_sys_path",
    "train sha constant": V218_TRAIN_SHA256,
    "val sha constant": V218_VAL_SHA256,
    "train sha gate": "observed_train_sha256 != EXPECTED_TRAIN_SHA256",
    "val sha gate": "observed_val_sha256 != EXPECTED_VAL_SHA256",
    "safetensors fallback": "pip_install_safetensors.log",
    "adapter tensor count": "tensor_count = len(handle.keys())",
    "v194 tensor gate": "V194 adapter tensor count mismatch",
    "v217 size gate": "V217 final adapter size mismatch",
    "v217 tensor floor gate": "V217 final adapter tensor count below expected floor",
    "target modules gate": "target_modules mismatch",
    "target parameters gate": "target_parameters mismatch",
    "submit lock false": "ALLOW_KAGGLE_SUBMIT = False",
    "submit lock guard": "Kaggle submission is disabled",
    "weak total gate": "WEAK_MIN_FOR_FULL = 193",
    "weak eq gate": "WEAK_EQ_MIN_FOR_FULL = 60",
    "weak bit gate": "WEAK_BIT_MIN_FOR_FULL = 133",
    "weak trunc gate": "WEAK_MAX_TRUNC_FOR_FULL = 3",
    "decode max tokens arg": "--max-tokens",
    "decode prompt suffix arg": "--prompt-suffix",
    "decode disable thinking arg": "--disable-thinking",
    "full eval blocked": "Full eval is intentionally not automatic in V218 diagnostic notebook",
}

V219_NOTEBOOK_REL = "notebooks/KG1_V219_WEAK_DECODE_AB_COLAB.ipynb"
V219_REQUIRED_FILES = [
    ".gitattributes",
    V218_TRAIN_REL,
    V218_VAL_REL,
    "data/v217/v217_short_answer_manifest.json",
    "scripts/analyze_eval_predictions.py",
    "scripts/build_v219_weak_decode_ab_colab.py",
    "scripts/evaluate_lora_adapter.py",
    "scripts/notebook_release_gate.py",
    "src/__init__.py",
    "src/competition_utils.py",
]

V219_REQUIRED_SNIPPETS = {
    "repo clone branch": "'git', 'clone', '--depth', '1', '--branch', REPO_BRANCH",
    "repo sys.path insert": "sys.path.insert(0, str(ROOT))",
    "repo sys.path log": "repo_root_on_sys_path",
    "train sha constant": V218_TRAIN_SHA256,
    "val sha constant": V218_VAL_SHA256,
    "v217 candidate": "v217_think1_mtok3584",
    "v194 candidate": "v194_think1_mtok3584",
    "thinking default false disable flag": "V219_DISABLE_THINKING_DEFAULT = False",
    "thinking default guard": "V219 must keep thinking enabled by default",
    "run train hard guard": "V219 is decode A/B only; RUN_TRAIN must stay false.",
    "max model len arg": "--max-model-len",
    "warmup rows arg": "--warmup-rows",
    "weak gate": "weak_gate_pass_for_full",
    "full eval opt in": "RUN_FULL_IF_GATE",
    "full eval blocked by default": "Full eval is blocked by default to avoid accidental GPU spend",
    "v220 roadmap": "Build V220 solver-trace training data for bit/equation families",
    "submit lock false": "ALLOW_KAGGLE_SUBMIT = False",
    "submit lock guard": "Kaggle submission is disabled",
}

V220_NOTEBOOK_REL = "notebooks/KG1_V220_PUBLIC_ADAPTER_PROBE_COLAB.ipynb"
V220_REQUIRED_FILES = [
    ".gitattributes",
    V218_TRAIN_REL,
    V218_VAL_REL,
    "data/v217/v217_short_answer_manifest.json",
    "scripts/analyze_eval_predictions.py",
    "scripts/build_v220_public_adapter_probe_colab.py",
    "scripts/evaluate_lora_adapter.py",
    "scripts/notebook_release_gate.py",
    "src/__init__.py",
    "src/competition_utils.py",
]

V220_REQUIRED_SNIPPETS = {
    "repo clone branch": "'git', 'clone', '--depth', '1', '--branch', REPO_BRANCH",
    "repo sys.path insert": "sys.path.insert(0, str(ROOT))",
    "repo sys.path log": "repo_root_on_sys_path",
    "public adapter repo": "NARIBOW_ADAPTER_REPO = os.environ.get('KG1_V220_PUBLIC_ADAPTER_REPO', 'Naribow/nemotron-sft-lora')",
    "public adapter local path": "NARIBOW_ADAPTER = OUT_ROOT / 'hf_adapters'",
    "hf snapshot download": "snapshot_download(",
    "hf adapter allow patterns": "allow_patterns=['adapter_config.json', 'adapter_model.safetensors', 'README.md']",
    "public adapter completeness": "NARIBOW_ADAPTER complete",
    "public adapter candidate": "naribow_public_think1_mtok3584",
    "init adapter public": "INIT_ADAPTER_DIR = NARIBOW_ADAPTER",
    "thinking default false disable flag": "V220_DISABLE_THINKING_DEFAULT = False",
    "thinking default guard": "V220 must keep thinking enabled by default",
    "run train hard guard": "V220 is public adapter probe only; RUN_TRAIN must stay false.",
    "max model len arg": "--max-model-len",
    "warmup rows arg": "--warmup-rows",
    "weak gate": "weak_gate_pass_for_full",
    "full eval opt in": "RUN_FULL_IF_GATE",
    "submit lock false": "ALLOW_KAGGLE_SUBMIT = False",
    "submit lock guard": "Kaggle submission is disabled",
    "manual review roadmap": "Manual review required before packaging; notebook still has hard submit lock.",
}

V221_NOTEBOOK_REL = "notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb"
V221_REQUIRED_FILES = [
    ".gitattributes",
    V218_TRAIN_REL,
    V218_VAL_REL,
    "data/v217/v217_short_answer_manifest.json",
    "scripts/analyze_eval_predictions.py",
    "scripts/build_v221_candidate_registry_weak_ab_colab.py",
    "scripts/evaluate_lora_adapter.py",
    "scripts/evaluate_lora_adapters_batch.py",
    "scripts/notebook_release_gate.py",
    "src/__init__.py",
    "src/competition_utils.py",
]

V221_REQUIRED_SNIPPETS = {
    "repo clone branch": "'git', 'clone', '--depth', '1', '--branch', REPO_BRANCH",
    "repo sys.path insert": "sys.path.insert(0, str(ROOT))",
    "repo sys.path log": "repo_root_on_sys_path",
    "train sha constant": V218_TRAIN_SHA256,
    "val sha constant": V218_VAL_SHA256,
    "registry object": "CANDIDATE_REGISTRY",
    "registry ready candidates": "v221_ready_candidates.json",
    "batch evaluator script": "scripts/evaluate_lora_adapters_batch.py",
    "batch candidates arg": "--candidates-json",
    "batch summary": "batch_candidate_summary.json",
    "hf candidate kind": "hf_model_adapter",
    "kaggle dataset candidate kind": "kaggle_dataset_adapter",
    "kaggle model candidate kind": "kaggle_model_adapter",
    "naribow candidate": "Naribow/nemotron-sft-lora",
    "dgxchen candidate": "dgxchen/trained-adapter",
    "konbu candidate": "konbu17/exp026-s012-lora",
    "kienngx candidate": "kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1",
    "thinking default false disable flag": "V221_DISABLE_THINKING_DEFAULT = False",
    "thinking default guard": "V221 must keep thinking enabled by default",
    "run train hard guard": "V221 is candidate weak A/B only; RUN_TRAIN must stay false.",
    "max candidates control": "V221_MAX_CANDIDATES",
    "weak gate": "weak_gate_pass_for_full",
    "full eval opt in": "RUN_FULL_IF_GATE",
    "full eval blocked by default": "Full eval is intentionally not automatic in V221 candidate registry notebook",
    "submit lock false": "ALLOW_KAGGLE_SUBMIT = False",
    "submit lock guard": "Kaggle submission is disabled",
}


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


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


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


def read_repo_text(rel_path: str) -> str:
    path = ROOT / rel_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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


def audit_v218_decode_rescue_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict one-file gate for V218 regressions already seen in Colab.

    This is intentionally inside the central release gate. When a V218 notebook
    error is found, add the regression check here so future notebook edits fail
    before GPU time is spent.
    """

    if repo_rel(path) != V218_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    outputs_total = sum(len(cell.get("outputs", [])) for cell in code_cells)
    if len(code_cells) != 9:
        add(findings, "error", "v218_code_cell_count", f"expected 9 code cells, found {len(code_cells)}")
    if outputs_total:
        add(findings, "error", "v218_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    for name, snippet in V218_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v218_required_snippet_missing", name)

    for rel_path in V218_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v218_required_file_missing", rel_path)

    gitattributes = read_repo_text(".gitattributes")
    for line in ["*.ipynb text eol=lf", "*.jsonl text eol=lf", "*.py text eol=lf"]:
        if line not in gitattributes:
            add(findings, "error", "v218_gitattributes_missing_lf_rule", line)

    train_path = ROOT / V218_TRAIN_REL
    val_path = ROOT / V218_VAL_REL
    if train_path.exists():
        observed = sha256_file(train_path)
        if observed != V218_TRAIN_SHA256:
            add(findings, "error", "v218_train_sha_mismatch", observed)
        rows = count_lines(train_path)
        if rows != V218_TRAIN_ROWS:
            add(findings, "error", "v218_train_row_count_mismatch", str(rows))
    if val_path.exists():
        observed = sha256_file(val_path)
        if observed != V218_VAL_SHA256:
            add(findings, "error", "v218_val_sha_mismatch", observed)
        rows = count_lines(val_path)
        if rows != V218_VAL_ROWS:
            add(findings, "error", "v218_val_row_count_mismatch", str(rows))

    analyzer_text = read_repo_text("scripts/analyze_eval_predictions.py")
    evaluator_text = read_repo_text("scripts/evaluate_lora_adapter.py")
    builder_text = read_repo_text("scripts/build_v218_decode_rescue_colab.py")
    competition_text = read_repo_text("src/competition_utils.py")

    for label, script_text in [
        ("analyze_eval_predictions", analyzer_text),
        ("evaluate_lora_adapter", evaluator_text),
    ]:
        if "ROOT = Path(__file__).resolve().parents[1]" not in script_text or "sys.path.insert(0, str(ROOT))" not in script_text:
            add(findings, "error", "v218_script_missing_repo_sys_path", label)

    for option in ["--max-tokens", "--max-num-seqs", "--disable-thinking", "--prompt-suffix"]:
        if option not in evaluator_text:
            add(findings, "error", "v218_evaluator_cli_option_missing", option)

    for snippet in V218_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v218_builder_required_snippet_missing", snippet)

    for snippet in [
        "MODEL_NAME = \"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\"",
        "MODEL_REVISION = \"cbd3fa9f933d55ef16a84236559f4ee2a0526848\"",
        "\"max_lora_rank\": 32",
        "\"max_model_len\": 8192",
        "\"max_tokens\": 7680",
        "def extract_final_answer",
        "def answers_equivalent",
    ]:
        if snippet not in competition_text:
            add(findings, "error", "v218_competition_utils_contract_missing", snippet)


def audit_v219_weak_decode_ab_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict one-file gate for the V219 weak-only decode A/B notebook."""

    if repo_rel(path) != V219_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    outputs_total = sum(len(cell.get("outputs", [])) for cell in code_cells)
    if len(code_cells) != 8:
        add(findings, "error", "v219_code_cell_count", f"expected 8 code cells, found {len(code_cells)}")
    if outputs_total:
        add(findings, "error", "v219_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    for name, snippet in V219_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v219_required_snippet_missing", name)

    if "--disable-thinking" in text:
        add(findings, "error", "v219_disable_thinking_banned", "V219 must not run the V218 failed no-thinking decode path")

    for rel_path in V219_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v219_required_file_missing", rel_path)

    train_path = ROOT / V218_TRAIN_REL
    val_path = ROOT / V218_VAL_REL
    if train_path.exists():
        observed = sha256_file(train_path)
        if observed != V218_TRAIN_SHA256:
            add(findings, "error", "v219_train_sha_mismatch", observed)
        rows = count_lines(train_path)
        if rows != V218_TRAIN_ROWS:
            add(findings, "error", "v219_train_row_count_mismatch", str(rows))
    if val_path.exists():
        observed = sha256_file(val_path)
        if observed != V218_VAL_SHA256:
            add(findings, "error", "v219_val_sha_mismatch", observed)
        rows = count_lines(val_path)
        if rows != V218_VAL_ROWS:
            add(findings, "error", "v219_val_row_count_mismatch", str(rows))

    evaluator_text = read_repo_text("scripts/evaluate_lora_adapter.py")
    builder_text = read_repo_text("scripts/build_v219_weak_decode_ab_colab.py")
    for option in ["--max-tokens", "--max-model-len", "--max-num-seqs", "--warmup-rows", "--prompt-suffix"]:
        if option not in evaluator_text:
            add(findings, "error", "v219_evaluator_cli_option_missing", option)
    for snippet in [
        "warmup_rows = int(config.get(\"warmup_rows\", 4))",
        "eval_config[\"max_model_len\"] = int(args.max_model_len)",
        "eval_config[\"warmup_rows\"] = max(0, int(args.warmup_rows))",
    ]:
        if snippet not in evaluator_text:
            add(findings, "error", "v219_evaluator_warmup_model_len_contract_missing", snippet)
    for snippet in V219_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v219_builder_required_snippet_missing", snippet)


def audit_v220_public_adapter_probe_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict one-file gate for the V220 public-adapter probe notebook."""

    if repo_rel(path) != V220_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    outputs_total = sum(len(cell.get("outputs", [])) for cell in code_cells)
    if len(code_cells) != 8:
        add(findings, "error", "v220_code_cell_count", f"expected 8 code cells, found {len(code_cells)}")
    if outputs_total:
        add(findings, "error", "v220_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    for name, snippet in V220_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v220_required_snippet_missing", name)

    if "--disable-thinking" in text:
        add(findings, "error", "v220_disable_thinking_banned", "V220 must not run the V218 failed no-thinking decode path")
    if "RUN_FULL_IF_GATE = os.environ.get('KG1_V220_RUN_FULL_IF_GATE', '0')" not in text:
        add(findings, "error", "v220_full_eval_default_not_blocked", "full eval must default off")
    if "NARIBOW_ADAPTER" not in text or "V217_FINAL_ADAPTER" in text and "'adapter': V217_FINAL_ADAPTER" in text:
        add(findings, "error", "v220_wrong_candidate_adapter", "V220 must evaluate the public Naribow adapter, not V217")

    for rel_path in V220_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v220_required_file_missing", rel_path)

    evaluator_text = read_repo_text("scripts/evaluate_lora_adapter.py")
    builder_text = read_repo_text("scripts/build_v220_public_adapter_probe_colab.py")
    for option in ["--max-tokens", "--max-model-len", "--max-num-seqs", "--warmup-rows", "--prompt-suffix"]:
        if option not in evaluator_text:
            add(findings, "error", "v220_evaluator_cli_option_missing", option)
    for snippet in V220_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v220_builder_required_snippet_missing", snippet)


def audit_v221_candidate_registry_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict one-file gate for the V221 candidate-registry weak A/B notebook."""

    if repo_rel(path) != V221_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    outputs_total = sum(len(cell.get("outputs", [])) for cell in code_cells)
    if len(code_cells) != 8:
        add(findings, "error", "v221_code_cell_count", f"expected 8 code cells, found {len(code_cells)}")
    if outputs_total:
        add(findings, "error", "v221_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    for name, snippet in V221_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v221_required_snippet_missing", name)

    if "--disable-thinking" in text:
        add(findings, "error", "v221_disable_thinking_banned", "V221 must keep thinking enabled; no no-thinking decode path in notebook")
    if "KG1_V221_RUN_FULL_IF_GATE', '0'" not in text:
        add(findings, "error", "v221_full_eval_default_not_blocked", "full eval must default off")
    if "RUN_TRAIN = os.environ.get('KG1_V221_RUN_TRAIN', '0')" not in text:
        add(findings, "error", "v221_train_default_not_blocked", "training must default off and be hard blocked")
    if "kaggle competitions submit" in text:
        add(findings, "error", "v221_submit_command_banned", "candidate probe notebook must not contain Kaggle submit command")
    for bad_literal in ['"required": true', '"required": false', '"required": null']:
        if bad_literal in text:
            add(
                findings,
                "error",
                "v221_json_boolean_literal_in_python_cell",
                f"embedded registry must be valid Python at runtime; found {bad_literal}",
            )

    for rel_path in V221_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v221_required_file_missing", rel_path)

    train_path = ROOT / V218_TRAIN_REL
    val_path = ROOT / V218_VAL_REL
    if train_path.exists():
        observed = sha256_file(train_path)
        if observed != V218_TRAIN_SHA256:
            add(findings, "error", "v221_train_sha_mismatch", observed)
        rows = count_lines(train_path)
        if rows != V218_TRAIN_ROWS:
            add(findings, "error", "v221_train_row_count_mismatch", str(rows))
    if val_path.exists():
        observed = sha256_file(val_path)
        if observed != V218_VAL_SHA256:
            add(findings, "error", "v221_val_sha_mismatch", observed)
        rows = count_lines(val_path)
        if rows != V218_VAL_ROWS:
            add(findings, "error", "v221_val_row_count_mismatch", str(rows))

    batch_text = read_repo_text("scripts/evaluate_lora_adapters_batch.py")
    builder_text = read_repo_text("scripts/build_v221_candidate_registry_weak_ab_colab.py")

    for option in [
        "--candidates-json",
        "--max-tokens",
        "--max-model-len",
        "--max-num-seqs",
        "--warmup-rows",
        "--prompt-suffix",
        "--continue-on-error",
    ]:
        if option not in batch_text:
            add(findings, "error", "v221_batch_evaluator_cli_option_missing", option)

    for snippet in [
        "from vllm import LLM",
        "from vllm.lora.request import LoRARequest",
        "llm = LLM(**llm_kwargs)",
        "render_prompts(tokenizer, questions, config)",
        "validate_adapter_dir",
        "for lora_id, candidate in enumerate(valid_candidates",
        "LoRARequest(candidate[\"name\"], lora_id, str(candidate[\"adapter\"]))",
        "batch_candidate_summary.csv",
        "batch_candidate_summary.json",
    ]:
        if snippet not in batch_text:
            add(findings, "error", "v221_batch_evaluator_contract_missing", snippet)

    for snippet in [
        "snapshot_download(",
        "allow_patterns=['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin', 'README.md']",
        "cmd = ['kaggle', 'datasets', 'download'",
        "cmd = ['kaggle', 'models', 'instances', 'versions', 'download'",
        "has_kaggle_credentials",
        "v221_ready_candidates.json",
        "v221_candidate_resolution.csv",
        "batch_candidate_summary.json",
    ]:
        if snippet not in builder_text:
            add(findings, "error", "v221_builder_candidate_resolution_contract_missing", snippet)

    for snippet in V221_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v221_builder_required_snippet_missing", snippet)


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
        audit_v218_decode_rescue_contract(path, notebook, text, findings)
        audit_v219_weak_decode_ab_contract(path, notebook, text, findings)
        audit_v220_public_adapter_probe_contract(path, notebook, text, findings)
        audit_v221_candidate_registry_contract(path, notebook, text, findings)
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
