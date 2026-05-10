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
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_RE = re.compile(r"\.ipynb$", re.IGNORECASE)
COLAB_RE = re.compile(r"https://colab\.research\.google\.com/github/[^\s`)]+\.ipynb")
GITHUB_BLOB_RE = re.compile(r"https://github\.com/[^\s`)]+/blob/[^\s`)]+\.ipynb")
LOCAL_REPO_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"("
    r"(?:scripts|src|data|notebooks|artifacts/notebook_release_gate)/[A-Za-z0-9_./-]+"
    r"|\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml"
    r"|\.gitattributes"
    r"|AGENTS\.md"
    r")"
)
SYNTHESIS_DEFAULT_ZERO_RE = re.compile(
    r"ALLOW_[A-Z0-9_]*SYNTHESIS\s*=\s*os\.environ\.get\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]0['\"]",
    re.MULTILINE,
)
SYNTHESIS_FLAG_RE = re.compile(r"\bALLOW_[A-Z0-9_]*SYNTHESIS\b")

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
    "SAMPLING_MODE='weighted'",
    'SAMPLING_MODE="weighted"',
    "SAMPLING_MODE = 'weighted'",
    'SAMPLING_MODE = "weighted"',
    "'SAMPLING_MODE': 'weighted'",
    '"SAMPLING_MODE": "weighted"',
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

V230_NOTEBOOK_REL = "notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb"
V230_COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb"
)
V230_GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb"
)
V230_REQUIRED_FILES = [
    V218_TRAIN_REL,
    V218_VAL_REL,
    "data/v217/v217_short_answer_manifest.json",
    "scripts/analyze_v230_v226_complementarity.py",
    "scripts/build_v230_v226_complementarity_colab.py",
    "scripts/notebook_release_gate.py",
    "src/competition_utils.py",
]
V230_REQUIRED_SNIPPETS = {
    "cpu-only purpose": "V230 CPU-only path does not install vLLM",
    "diagnostic mode explicit": "V230_DIAGNOSTIC_MODE",
    "diagnostic release block": "V230 diagnostic mode is not a release or submission authorization",
    "v226 diagnostic synthesis default": "'1' if V230_DIAGNOSTIC_MODE else '0'",
    "v226 preferred baseline": "v226__v226_best_checkpoint1_observed_191",
    "v226 canonical baseline alias": "v226__v226_checkpoint_1",
    "v226 preflight alias set": "V226_BASELINE_NAME_ALIASES",
    "v226 baseline adapter check": "--expected-baseline-adapter",
    "v226 known weak total": "KNOWN_V226_WEAK_TOTAL = 191",
    "mandatory analysis": "V230 complementarity analysis is mandatory",
    "analysis complete": "'analysis_complete': True",
    "allowed actions": "'allowed_actions':",
    "run id output": "RUN_ID =",
    "expected shared row contract env": "KG1_V230_EXPECTED_SHARED_ROW_CONTRACT_SHA256",
    "runtime analyzer self test": "v230_analyzer_self_test.log",
    "stale manifest removal": "removed_stale_analysis_manifest",
    "v221 nonempty gate": "V221 batch summary has no ok candidates.",
    "v229 required": "raise FileNotFoundError(V229_ANALYSIS_MANIFEST_JSON)",
    "v226 baseline row gate": "Required V226 baseline row missing:",
    "jsonl semantic audit": "inspect_short_answer_jsonl",
    "jsonl assistant audit": "assistant_answer_mismatch",
    "train family counts": "EXPECTED_TRAIN_FAMILY_COUNTS",
    "silent subprocess queue": "output_queue = queue.Queue()",
    "silent subprocess reader": "threading.Thread",
    "strict prediction fallback": "ambiguous prediction CSV fallback",
    "v226 weak315 materialization": "materialize_weak315_report",
    "v221 reference id contract": "load_reference_weak_ids_from_v221_batch_summary",
    "canonical v226 report": "canonical_candidate_name",
    "v226 source report provenance": "source_report_json",
    "v226 source report fingerprint": "source_fingerprint",
    "v226 synthetic rows contract": "synthetic_report['rows'] = 315",
    "prediction sha log": "prediction_sha256",
    "hard submit false": "ALLOW_KAGGLE_SUBMIT = False",
    "no package submit": "No package and no Kaggle submit can be created in V230.",
}

V231_NOTEBOOK_REL = "notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb"
V231_COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb"
)
V231_GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb"
)
V231_REQUIRED_FILES = [
    "scripts/analyze_v231_miss_packs.py",
    "scripts/build_v231_miss_pack_mining_colab.py",
    "scripts/notebook_release_gate.py",
]
V231_REQUIRED_SNIPPETS = {
    "cpu-only purpose": "V231 is CPU-only miss-pack mining",
    "known row contract": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
    "expected row contract env": "KG1_V231_EXPECTED_SHARED_ROW_CONTRACT_SHA256",
    "explicit manifest text guard": "V230_ANALYSIS_MANIFEST_JSON_TEXT",
    "explicit manifest file guard": "is_file()",
    "v230 manifest resolver": "resolve_latest_v230_manifest",
    "mining manifest file guard": "mining_v230_manifest_is_file",
    "v230 artifact metadata logs": "v230_output_artifact_meta",
    "miner script": "scripts/analyze_v231_miss_packs.py",
    "miner self test": "v231_miss_pack_mining_self_test.log",
    "equation taxonomy output": "equation_miss_taxonomy_csv",
    "equation rules output": "equation_solver_candidate_rules_json",
    "bit guardrail output": "bit_guardrail_candidates_json",
    "hard train false": "RUN_TRAIN = False",
    "hard full false": "RUN_FULL_IF_GATE = False",
    "hard submit false": "ALLOW_KAGGLE_SUBMIT = False",
    "no package submit": "No package and no Kaggle submit can be created in V231.",
}

V232_NOTEBOOK_REL = "notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb"
V232_COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb"
)
V232_GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb"
)
V232_REQUIRED_FILES = [
    "scripts/analyze_v232_verified_solver_workbench.py",
    "scripts/build_v232_verified_solver_workbench_colab.py",
    "scripts/notebook_release_gate.py",
]
V232_REQUIRED_SNIPPETS = {
    "cpu-only purpose": "V232 is CPU-only solver workbench",
    "known row contract": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
    "expected row contract env": "KG1_V232_EXPECTED_SHARED_ROW_CONTRACT_SHA256",
    "explicit v231 manifest text guard": "V231_ANALYSIS_MANIFEST_JSON_TEXT",
    "v231 manifest resolver": "resolve_latest_v231_manifest",
    "v231 nested row contract": "observed_shared_row_contract_sha256",
    "v231 artifact metadata logs": "v231_output_artifact_meta",
    "workbench script": "scripts/analyze_v232_verified_solver_workbench.py",
    "workbench self test": "v232_verified_solver_workbench_self_test.log",
    "equation workitems output": "equation_solver_workitems_jsonl",
    "bit workitems output": "bit_guardrail_workitems_jsonl",
    "acceptance matrix output": "acceptance_matrix_csv",
    "solver contracts output": "solver_contracts_json",
    "hard train false": "RUN_TRAIN = False",
    "hard full false": "RUN_FULL_IF_GATE = False",
    "hard submit false": "ALLOW_KAGGLE_SUBMIT = False",
    "no package submit": "No package and no Kaggle submit can be created in V232.",
}

V233_NOTEBOOK_REL = "notebooks/KG1_V233_VERIFIED_EQUATION_SOLVER_PROBES_COLAB.ipynb"
V233_COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V233_VERIFIED_EQUATION_SOLVER_PROBES_COLAB.ipynb"
)
V233_GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V233_VERIFIED_EQUATION_SOLVER_PROBES_COLAB.ipynb"
)
V233_REQUIRED_FILES = [
    "scripts/analyze_v233_verified_equation_solver_probes.py",
    "scripts/build_v233_verified_equation_solver_probes_colab.py",
    "scripts/notebook_release_gate.py",
]
V233_REQUIRED_SNIPPETS = {
    "cpu-only purpose": "V233 is CPU-only equation solver probes",
    "known row contract": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
    "expected row contract env": "KG1_V233_EXPECTED_SHARED_ROW_CONTRACT_SHA256",
    "explicit v232 manifest text guard": "V232_ANALYSIS_MANIFEST_JSON_TEXT",
    "directory placeholder guard": "ignored_directory_placeholder",
    "v232 manifest resolver": "resolve_latest_v232_manifest",
    "stale runtime manifest self-heal": "resolved_candidate_text",
    "v232 nested row contract": "observed_shared_row_contract_sha256",
    "v232 artifact metadata logs": "v232_output_artifact_meta",
    "probe script": "scripts/analyze_v233_verified_equation_solver_probes.py",
    "probe self test": "v233_verified_equation_solver_probes_self_test.log",
    "probe results output": "equation_probe_results_jsonl",
    "probe summary output": "equation_probe_summary_csv",
    "verified overrides output": "equation_verified_overrides_csv",
    "oracle evidence output": "equation_oracle_evidence_csv",
    "hard train false": "RUN_TRAIN = False",
    "hard full false": "RUN_FULL_IF_GATE = False",
    "hard submit false": "ALLOW_KAGGLE_SUBMIT = False",
    "no package submit": "No package and no Kaggle submit can be created in V233.",
}

V234_NOTEBOOK_REL = "notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb"
V234_COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb"
)
V234_GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb"
)
V234_REQUIRED_FILES = [
    "artifacts/roadmaps/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md",
    "scripts/analyze_v234_external_intel_triage.py",
    "scripts/build_v234_external_intel_triage_colab.py",
    "scripts/notebook_release_gate.py",
]
V234_REQUIRED_SNIPPETS = {
    "cpu-only purpose": "V234 is CPU-only external intel triage",
    "roadmap path": "KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md",
    "roadmap resolver": "resolve_roadmap_md",
    "roadmap marker preflight": "required_roadmap_markers",
    "triage script": "scripts/analyze_v234_external_intel_triage.py",
    "triage self test": "v234_external_intel_triage_self_test.log",
    "metric parity output": "external_metric_parity_report.json",
    "kernel triage output": "kaggle_kernel_triage.csv",
    "dataset triage output": "kaggle_dataset_triage.csv",
    "hf dataset triage output": "hf_dataset_triage.csv",
    "model triage output": "kaggle_model_triage.csv",
    "equation probe output": "equation_numeric_operator_probe_results.csv",
    "bit probe output": "bit_boolean_function_probe_results.csv",
    "external registry output": "external_adapter_registry_candidates.csv",
    "hard train false": "RUN_TRAIN = False",
    "hard full false": "RUN_FULL_IF_GATE = False",
    "hard submit false": "ALLOW_KAGGLE_SUBMIT = False",
    "hard model generation false": "ALLOW_MODEL_GENERATION = False",
    "no package submit": "No package and no Kaggle submit can be created in V234.",
}

V235_NOTEBOOK_REL = "notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb"
V235_COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb"
)
V235_GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v230-v226-complementarity/notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb"
)
V235_REQUIRED_FILES = [
    "scripts/analyze_v235_source_access_triage.py",
    "scripts/build_v235_source_access_triage_colab.py",
    "scripts/notebook_release_gate.py",
]
V235_REQUIRED_SNIPPETS = {
    "cpu-only purpose": "V235 is CPU-only source access triage",
    "v234 manifest resolver": "resolve_latest_v234_manifest",
    "v234 output preflight": "required_outputs = [",
    "source access script": "scripts/analyze_v235_source_access_triage.py",
    "source access self test": "v235_source_access_triage_self_test.log",
    "source inventory output": "source_access_inventory.csv",
    "hf metadata output": "hf_metadata_audit.csv",
    "kaggle audit output": "kaggle_access_audit.csv",
    "download plan output": "source_download_plan.csv",
    "license gate output": "license_gate_report.json",
    "hard source download false": "ALLOW_SOURCE_PAYLOAD_DOWNLOAD = False",
    "hard train false": "RUN_TRAIN = False",
    "hard full false": "RUN_FULL_IF_GATE = False",
    "hard submit false": "ALLOW_KAGGLE_SUBMIT = False",
    "hard model generation false": "ALLOW_MODEL_GENERATION = False",
    "no package submit": "No package and no Kaggle submit can be created in V235.",
}

RELEASE_NOTEBOOK_RELS = [
    V218_NOTEBOOK_REL,
    V219_NOTEBOOK_REL,
    V220_NOTEBOOK_REL,
    V221_NOTEBOOK_REL,
    V230_NOTEBOOK_REL,
    V231_NOTEBOOK_REL,
    V232_NOTEBOOK_REL,
    V233_NOTEBOOK_REL,
    V234_NOTEBOOK_REL,
    V235_NOTEBOOK_REL,
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


def audit_clean_notebook_state(notebook: dict[str, Any], findings: list[Finding]) -> None:
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    outputs_total = sum(len(cell.get("outputs", []) or []) for cell in code_cells)
    executed_total = sum(1 for cell in code_cells if cell.get("execution_count") is not None)
    if outputs_total:
        add(findings, "error", "notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")
    if executed_total:
        add(
            findings,
            "error",
            "notebook_has_execution_counts",
            f"notebook must be committed clean; non-null execution_count cells={executed_total}",
        )


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


def audit_local_repo_references(text: str, findings: list[Finding]) -> None:
    """Verify literal repo-relative files referenced by notebooks exist now.

    Colab notebooks clone this repo before execution, so a typo in a literal
    `scripts/...`, `src/...`, or `data/...` reference is statically detectable.
    Runtime output paths are ignored by extension/name to avoid blocking logs
    and generated reports.
    """

    ignored_suffixes = {
        ".log",
        ".jsonl",
        ".csv",
        ".zip",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
    }
    ignored_parts = (
        "output_",
        "analysis_",
        "eval_",
        "train_",
        "checkpoint-",
        "batch_candidate_summary",
        "candidate_summary",
        "predictions",
        "manifest",
        "report",
    )
    seen: set[str] = set()
    for match in LOCAL_REPO_PATH_RE.finditer(text.replace("\\/", "/")):
        rel = match.group(1).strip("'\"`),]")
        if rel in seen:
            continue
        seen.add(rel)
        if "{" in rel or "}" in rel or "*" in rel or "$" in rel:
            continue
        path = Path(rel)
        if path.suffix.lower() in ignored_suffixes and any(part in rel for part in ignored_parts):
            continue
        if rel.startswith("artifacts/notebook_release_gate/") and not (ROOT / rel).exists():
            continue
        if rel.startswith("notebooks/") and not rel.endswith(".ipynb"):
            continue
        if not (ROOT / rel).exists():
            add(findings, "error", "referenced_repo_file_missing", rel)


def audit_artifact_dependency_contract(text: str, findings: list[Finding]) -> None:
    """Catch the class of failures where a runtime artifact is missing but the
    notebook has no diagnostic fallback/resolver.
    """

    if SYNTHESIS_DEFAULT_ZERO_RE.search(text) and "DIAGNOSTIC_MODE" in text:
        add(
            findings,
            "error",
            "diagnostic_synthesis_default_disabled",
            "diagnostic notebooks must default synthesis/fallback to diagnostic mode, e.g. '1' if DIAGNOSTIC_MODE else '0'",
        )

    synthesis_flags = sorted(set(SYNTHESIS_FLAG_RE.findall(text)))
    for flag in synthesis_flags:
        release_guard_re = re.compile(rf"if\s+not\s+[A-Z0-9_]*DIAGNOSTIC_MODE\s+and\s+{re.escape(flag)}")
        if "DIAGNOSTIC_MODE" in text and not release_guard_re.search(text):
            add(
                findings,
                "error",
                "synthesis_release_guard_missing",
                f"{flag} must be hard-blocked when diagnostic mode is false",
            )

    if "resolve_existing_or_synthesize" in text:
        required = {
            "candidate existence logs": "exists =",
            "candidate path probe": "_candidate",
            "synthesis function": "synthesize_",
            "clear missing-artifact error": "FileNotFoundError",
        }
        for name, snippet in required.items():
            if snippet not in text:
                add(findings, "error", "artifact_resolver_contract_missing", name)

    has_external_summary_input = bool(re.search(r"\b[A-Z0-9_]*BATCH_SUMMARY_JSON\b", text)) and "/content/drive/" in text
    if has_external_summary_input and "resolve_existing_or_synthesize" not in text:
        for snippet in ["exists =", "FileNotFoundError"]:
            if snippet not in text:
                add(
                    findings,
                    "error",
                    "external_batch_summary_without_runtime_check",
                    f"external Drive batch summary inputs need explicit existence logging and failure context: {snippet}",
                )


def audit_submit_lock(text: str, findings: list[Finding]) -> None:
    for banned in BANNED_SNIPPETS:
        if banned in text:
            add(findings, "error", "banned_snippet_present", banned)
    if "kaggle competitions submit" in text:
        add(findings, "error", "banned_submit_command_present", "kaggle competitions submit")
    if "ALLOW_KAGGLE_SUBMIT" not in text:
        add(findings, "error", "hard_submit_lock_missing", "missing ALLOW_KAGGLE_SUBMIT hard lock")
    elif "ALLOW_KAGGLE_SUBMIT = False" not in text and "ALLOW_KAGGLE_SUBMIT=False" not in text:
        add(findings, "error", "hard_submit_lock_not_false", "ALLOW_KAGGLE_SUBMIT must be False")


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


def is_hf_jobs_notebook(text: str) -> bool:
    markers = [
        "HfApi().run_job",
        "api.run_job(",
        "list_jobs_hardware",
        "Hugging Face Jobs",
        "HF Jobs",
        "huggingface.co/jobs",
    ]
    return contains_any(text, markers)


def audit_hf_jobs_contract(text: str, findings: list[Finding]) -> None:
    """Static guard for notebooks that launch paid HF Jobs.

    The checks intentionally focus on failures already observed in V243:
    invalid sampling mode, silent torch replacement by mamba/causal-conv pip
    resolution, missing H200/A100 cost logging, and launching without cheap
    preflight gates.
    """

    if not is_hf_jobs_notebook(text):
        return

    required = {
        "hardware listing": "list_jobs_hardware",
        "h200 flavor": "h200",
        "a100 flavor": "a100",
        "cost logging": "unit_cost",
        "hf job url": "https://huggingface.co/jobs",
        "commit gate": "KG1_EXPECTED_COMMIT",
        "py compile": "py_compile",
        "torch before": "torch_before",
        "torch after": "torch_after",
        "torch unchanged gate": "torch changed unexpectedly",
        "causal conv import": "causal_conv1d",
        "mamba import": "mamba_ssm",
        "mamba kernel import": "mamba_ssm.ops.triton.layernorm_gated",
        "selective scan import": "mamba_ssm.ops.selective_scan_interface",
        "no deps extension install": "--no-deps",
        "no build isolation": "--no-build-isolation",
        "dataset train sha": "KG1_TRAIN_SHA",
        "dataset val sha": "KG1_VAL_SHA",
        "sampling mode gate": "weighted_replacement",
        "smoke train cap": "MAX_STEPS=4",
        "run train default off": "RUN_TRAIN",
        "secret token injection": "secrets={'HF_TOKEN'",
        "job cancel path": "cancel_job",
    }
    for name, snippet in required.items():
        if snippet not in text:
            add(findings, "error", "hf_jobs_contract_missing", name)

    if "SAMPLING_MODE='weighted'" in text or 'SAMPLING_MODE="weighted"' in text:
        add(
            findings,
            "error",
            "hf_jobs_invalid_sampling_mode",
            "SAMPLING_MODE must be 'shuffle' or 'weighted_replacement', never 'weighted'",
        )


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


def audit_v230_v226_complementarity_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict release gate for the V230 CPU-only complementarity notebook."""

    if repo_rel(path) != V230_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 8:
        add(findings, "error", "v230_code_cell_count", f"expected 8 code cells, found {len(code_cells)}")
    outputs_total = sum(len(cell.get("outputs", []) or []) for cell in code_cells)
    if outputs_total:
        add(findings, "error", "v230_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    if V230_COLAB_URL not in text:
        add(findings, "error", "v230_colab_url_mismatch", V230_COLAB_URL)
    if V230_GITHUB_URL not in text:
        add(findings, "error", "v230_github_url_mismatch", V230_GITHUB_URL)

    for name, snippet in V230_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v230_required_snippet_missing", name)

    for rel_path in V230_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v230_required_file_missing", rel_path)

    analyzer = ROOT / "scripts" / "analyze_v230_v226_complementarity.py"
    analyzer_text = analyzer.read_text(encoding="utf-8") if analyzer.exists() else ""
    analyzer_snippets = {
        "raw output required": '{"id", "prompt", "answer", "prediction", "raw_output"}',
        "extractor drift gate": "raw_output extraction differs from prediction",
        "csv correct drift gate": "CSV correct disagrees with current verifier",
        "row contract function": "validate_shared_row_contract",
        "row contract hash": "shared_row_contract_sha256",
        "expected row contract cli": "--expected-shared-row-contract-sha256",
        "required row contract cli": "--require-shared-row-contract-sha256",
        "required row contract guard": "required_shared_row_contract_sha256",
        "prediction csv string dtype": "pd.read_csv(path, dtype=str, keep_default_na=False)",
        "gate normalized gap": "gate_normalized_gap",
        "family calibration": "family_calibration_summary",
        "analyzer self test": "v230_analyzer_self_test=ok",
        "declared family prompt crosscheck": "declared family disagrees with prompt classifier",
        "relative prediction path": "report_path.parent / path",
        "duplicate candidate names": "duplicate candidate names after normalization",
        "canonical family": "canonical_family",
        "strict report predictions": "report predictions_csv does not exist",
        "strict ambiguous fallback": "ambiguous prediction CSV fallback",
        "required baseline missing": "required preferred baseline was not found",
        "baseline correct expected": "--expected-baseline-correct",
        "baseline adapter expected": "--expected-baseline-adapter",
        "baseline fallback opt in": "--allow-baseline-fallback",
        "rescore mismatch opt in": "--allow-rescore-mismatch",
        "manifest load meta": '"load_meta": load_meta',
    }
    for name, snippet in analyzer_snippets.items():
        if snippet not in analyzer_text:
            add(findings, "error", "v230_analyzer_contract_missing", name)
    if analyzer.exists():
        completed = subprocess.run(
            [sys.executable, str(analyzer), "--self-test"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            add(
                findings,
                "error",
                "v230_analyzer_self_test_failed",
                f"returncode={completed.returncode}; tail={completed.stdout[-4000:]}",
            )
        elif "v230_analyzer_self_test=ok" not in completed.stdout:
            add(findings, "error", "v230_analyzer_self_test_missing_ok", completed.stdout[-4000:])

    builder = ROOT / "scripts" / "build_v230_v226_complementarity_colab.py"
    builder_text = builder.read_text(encoding="utf-8") if builder.exists() else ""
    for snippet in V230_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v230_builder_required_snippet_missing", snippet)
    builder_snippets = {
        "idempotent cell ids": "_CELL_COUNTER = 0",
        "safetensors only adapter completeness": "adapter_model.safetensors').exists()",
        "v226 tensor floor": "MIN_V226_CHECKPOINT_TENSOR_COUNT",
        "v226 size floor": "MIN_V226_CHECKPOINT_BYTES",
        "v221 nonempty gate": "V221 batch summary has no ok candidates.",
        "preferred v226 correct gate": "Required V226 baseline correct count mismatch",
        "expected row contract cli": "--expected-shared-row-contract-sha256",
        "baseline adapter cli": "--expected-baseline-adapter",
        "required row contract release": "--require-shared-row-contract-sha256",
        "diagnostic release block": "V230 diagnostic mode is not a release or submission authorization",
        "diagnostic synthesis default": "'1' if V230_DIAGNOSTIC_MODE else '0'",
        "v226 preflight alias set": "V226_BASELINE_NAME_ALIASES",
        "runtime analyzer self test": "v230_analyzer_self_test.log",
        "v229 required": "raise FileNotFoundError(V229_ANALYSIS_MANIFEST_JSON)",
        "relative prediction path": "direct_path = report_json.parent / direct_path",
        "v226 weak315 materialization": "materialize_weak315_report",
        "v221 reference id contract": "load_reference_weak_ids_from_v221_batch_summary",
        "canonical v226 report": "canonical_candidate_name",
        "v226 source report provenance": "source_report_json",
        "v226 source report fingerprint": "source_fingerprint",
        "v226 synthetic rows contract": "synthetic_report['rows'] = 315",
    }
    for name, snippet in builder_snippets.items():
        if snippet not in builder_text:
            add(findings, "error", "v230_builder_contract_missing", name)
    if builder.exists():
        try:
            spec = importlib.util.spec_from_file_location("kg1_v230_builder_gate", builder)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load builder module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            generated = module.build_notebook()
            if generated != notebook:
                add(findings, "error", "v230_builder_notebook_mismatch", "builder output differs from committed notebook")
        except Exception as exc:
            add(findings, "error", "v230_builder_notebook_compare_failed", repr(exc))


def audit_v231_miss_pack_mining_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict release gate for the V231 CPU-only miss-pack mining notebook."""

    if repo_rel(path) != V231_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 7:
        add(findings, "error", "v231_code_cell_count", f"expected 7 code cells, found {len(code_cells)}")
    outputs_total = sum(len(cell.get("outputs", []) or []) for cell in code_cells)
    if outputs_total:
        add(findings, "error", "v231_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    if V231_COLAB_URL not in text:
        add(findings, "error", "v231_colab_url_mismatch", V231_COLAB_URL)
    if V231_GITHUB_URL not in text:
        add(findings, "error", "v231_github_url_mismatch", V231_GITHUB_URL)

    for name, snippet in V231_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v231_required_snippet_missing", name)

    for rel_path in V231_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v231_required_file_missing", rel_path)

    analyzer = ROOT / "scripts" / "analyze_v231_miss_packs.py"
    analyzer_text = analyzer.read_text(encoding="utf-8") if analyzer.exists() else ""
    analyzer_snippets = {
        "self test ok": "v231_miss_pack_mining_self_test=ok",
        "equation route classifier": "classify_equation_route",
        "bit route classifier": "classify_bit_route",
        "pairwise gain aggregation": "aggregate_pairwise_gains",
        "miss pack validation": "validate_miss_pack",
        "equation rules schema": "kg1_v231_equation_solver_candidate_rules_v1",
        "bit guardrail schema": "kg1_v231_bit_guardrail_candidates_v1",
        "manifest schema": "kg1_v231_miss_pack_mining_manifest_v1",
        "no override acceptance": "No override without a local proof/verifier.",
        "keep v226 fallback": "When parser/verifier is ambiguous, abstain and keep V226 baseline.",
        "shared row contract arg": "--expected-shared-row-contract-sha256",
        "required v230 outputs": "baseline_miss_hits_csv",
    }
    for name, snippet in analyzer_snippets.items():
        if snippet not in analyzer_text:
            add(findings, "error", "v231_analyzer_contract_missing", name)
    if analyzer.exists():
        completed = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--self-test",
                "--v230-analysis-manifest-json",
                "dummy",
                "--output-dir",
                "dummy",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            add(
                findings,
                "error",
                "v231_analyzer_self_test_failed",
                f"returncode={completed.returncode}; tail={completed.stdout[-4000:]}",
            )
        elif "v231_miss_pack_mining_self_test=ok" not in completed.stdout:
            add(findings, "error", "v231_analyzer_self_test_missing_ok", completed.stdout[-4000:])

    builder = ROOT / "scripts" / "build_v231_miss_pack_mining_colab.py"
    builder_text = builder.read_text(encoding="utf-8") if builder.exists() else ""
    for snippet in V231_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v231_builder_required_snippet_missing", snippet)
    builder_snippets = {
        "idempotent cell ids": "_CELL_COUNTER = 0",
        "known row contract constant": "EXPECTED_SHARED_ROW_CONTRACT_SHA256",
        "latest v230 resolver": "resolve_latest_v230_manifest",
        "v230 output preflight": "required_outputs = [",
        "miner command": "analyze_v231_miss_packs.py",
        "submission artifact scan": "blocked_artifacts",
        "final manifest": "v231_miss_pack_mining_final_manifest.json",
    }
    for name, snippet in builder_snippets.items():
        if snippet not in builder_text:
            add(findings, "error", "v231_builder_contract_missing", name)
    if builder.exists():
        try:
            spec = importlib.util.spec_from_file_location("kg1_v231_builder_gate", builder)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load builder module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            generated = module.build_notebook()
            if generated != notebook:
                add(findings, "error", "v231_builder_notebook_mismatch", "builder output differs from committed notebook")
        except Exception as exc:
            add(findings, "error", "v231_builder_notebook_compare_failed", repr(exc))


def audit_v232_verified_solver_workbench_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict release gate for the V232 CPU-only verified solver workbench."""

    if repo_rel(path) != V232_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 7:
        add(findings, "error", "v232_code_cell_count", f"expected 7 code cells, found {len(code_cells)}")
    outputs_total = sum(len(cell.get("outputs", []) or []) for cell in code_cells)
    if outputs_total:
        add(findings, "error", "v232_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    if V232_COLAB_URL not in text:
        add(findings, "error", "v232_colab_url_mismatch", V232_COLAB_URL)
    if V232_GITHUB_URL not in text:
        add(findings, "error", "v232_github_url_mismatch", V232_GITHUB_URL)

    for name, snippet in V232_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v232_required_snippet_missing", name)

    for rel_path in V232_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v232_required_file_missing", rel_path)

    analyzer = ROOT / "scripts" / "analyze_v232_verified_solver_workbench.py"
    analyzer_text = analyzer.read_text(encoding="utf-8") if analyzer.exists() else ""
    analyzer_snippets = {
        "self test ok": "v232_verified_solver_workbench_self_test=ok",
        "workitem schema": "kg1_v232_solver_workitem_v1",
        "contracts schema": "kg1_v232_solver_contracts_v1",
        "manifest schema": "kg1_v232_verified_solver_workbench_manifest_v1",
        "v231 output loader": "load_required_v231_outputs",
        "v231 row contract reader": "shared_row_contract_from_v231_manifest",
        "v230 miss pack loader": "load_v230_miss_pack_paths",
        "workitem builder": "build_workitems",
        "acceptance matrix": "acceptance_matrix_rows",
        "no unverified override": "override_allowed_before_verifier",
        "next action": "build_v233_verified_equation_solver_probes",
    }
    for name, snippet in analyzer_snippets.items():
        if snippet not in analyzer_text:
            add(findings, "error", "v232_analyzer_contract_missing", name)
    if analyzer.exists():
        completed = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--self-test",
                "--v231-analysis-manifest-json",
                "dummy",
                "--output-dir",
                "dummy",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            add(
                findings,
                "error",
                "v232_analyzer_self_test_failed",
                f"returncode={completed.returncode}; tail={completed.stdout[-4000:]}",
            )
        elif "v232_verified_solver_workbench_self_test=ok" not in completed.stdout:
            add(findings, "error", "v232_analyzer_self_test_missing_ok", completed.stdout[-4000:])

    builder = ROOT / "scripts" / "build_v232_verified_solver_workbench_colab.py"
    builder_text = builder.read_text(encoding="utf-8") if builder.exists() else ""
    for snippet in V232_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v232_builder_required_snippet_missing", snippet)
    builder_snippets = {
        "idempotent cell ids": "_CELL_COUNTER = 0",
        "known row contract constant": "EXPECTED_SHARED_ROW_CONTRACT_SHA256",
        "latest v231 resolver": "resolve_latest_v231_manifest",
        "v231 output preflight": "required_outputs = [",
        "workbench command": "analyze_v232_verified_solver_workbench.py",
        "submission artifact scan": "blocked_artifacts",
        "final manifest": "v232_verified_solver_workbench_final_manifest.json",
    }
    for name, snippet in builder_snippets.items():
        if snippet not in builder_text:
            add(findings, "error", "v232_builder_contract_missing", name)
    if builder.exists():
        try:
            spec = importlib.util.spec_from_file_location("kg1_v232_builder_gate", builder)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load builder module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            generated = module.build_notebook()
            if generated != notebook:
                add(findings, "error", "v232_builder_notebook_mismatch", "builder output differs from committed notebook")
        except Exception as exc:
            add(findings, "error", "v232_builder_notebook_compare_failed", repr(exc))


def audit_v233_verified_equation_solver_probes_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict release gate for the V233 CPU-only verified equation probes."""

    if repo_rel(path) != V233_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 7:
        add(findings, "error", "v233_code_cell_count", f"expected 7 code cells, found {len(code_cells)}")
    outputs_total = sum(len(cell.get("outputs", []) or []) for cell in code_cells)
    if outputs_total:
        add(findings, "error", "v233_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    if V233_COLAB_URL not in text:
        add(findings, "error", "v233_colab_url_mismatch", V233_COLAB_URL)
    if V233_GITHUB_URL not in text:
        add(findings, "error", "v233_github_url_mismatch", V233_GITHUB_URL)

    for name, snippet in V233_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v233_required_snippet_missing", name)

    for rel_path in V233_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v233_required_file_missing", rel_path)

    analyzer = ROOT / "scripts" / "analyze_v233_verified_equation_solver_probes.py"
    analyzer_text = analyzer.read_text(encoding="utf-8") if analyzer.exists() else ""
    analyzer_snippets = {
        "self test ok": "v233_verified_equation_solver_probes_self_test=ok",
        "result schema": "kg1_v233_equation_probe_result_v1",
        "manifest schema": "kg1_v233_verified_equation_solver_probes_manifest_v1",
        "v232 row contract reader": "shared_row_contract_from_v232_manifest",
        "single equation probe": "sympy_single_equation_probe",
        "oracle nondeployable": "oracle_alternative_candidate_probe",
        "stable result csv schema": "PROBE_RESULT_COLUMNS",
        "no model generation": "does not train, run model generation",
        "verified override output": "equation_verified_overrides_csv",
        "oracle evidence output": "equation_oracle_evidence_csv",
        "decision prepare eval": "prepare_gated_solver_rescue_eval",
        "decision improve parsers": "improve_solver_parsers_before_eval",
    }
    for name, snippet in analyzer_snippets.items():
        if snippet not in analyzer_text:
            add(findings, "error", "v233_analyzer_contract_missing", name)
    forbidden_analyzer_snippets = {
        "unstable empty verified override csv": "pd.DataFrame(deployable_verified).to_csv",
        "unstable empty oracle evidence csv": "pd.DataFrame(nondeployable_verified).to_csv",
    }
    for name, snippet in forbidden_analyzer_snippets.items():
        if snippet in analyzer_text:
            add(findings, "error", "v233_analyzer_forbidden_pattern", name)
    if analyzer.exists():
        completed = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--self-test",
                "--v232-analysis-manifest-json",
                "dummy",
                "--output-dir",
                "dummy",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            add(
                findings,
                "error",
                "v233_analyzer_self_test_failed",
                f"returncode={completed.returncode}; tail={completed.stdout[-4000:]}",
            )
        elif "v233_verified_equation_solver_probes_self_test=ok" not in completed.stdout:
            add(findings, "error", "v233_analyzer_self_test_missing_ok", completed.stdout[-4000:])

    builder = ROOT / "scripts" / "build_v233_verified_equation_solver_probes_colab.py"
    builder_text = builder.read_text(encoding="utf-8") if builder.exists() else ""
    for snippet in V233_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v233_builder_required_snippet_missing", snippet)
    builder_snippets = {
        "idempotent cell ids": "_CELL_COUNTER = 0",
        "known row contract constant": "EXPECTED_SHARED_ROW_CONTRACT_SHA256",
        "latest v232 resolver": "resolve_latest_v232_manifest",
        "v232 output preflight": "required_outputs = [",
        "probe command": "analyze_v233_verified_equation_solver_probes.py",
        "submission artifact scan": "blocked_artifacts",
        "final manifest": "v233_verified_equation_solver_probes_final_manifest.json",
    }
    for name, snippet in builder_snippets.items():
        if snippet not in builder_text:
            add(findings, "error", "v233_builder_contract_missing", name)
    if builder.exists():
        try:
            spec = importlib.util.spec_from_file_location("kg1_v233_builder_gate", builder)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load builder module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            generated = module.build_notebook()
            if generated != notebook:
                add(findings, "error", "v233_builder_notebook_mismatch", "builder output differs from committed notebook")
        except Exception as exc:
            add(findings, "error", "v233_builder_notebook_compare_failed", repr(exc))


def audit_v234_external_intel_triage_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict release gate for the V234 CPU-only external intelligence triage."""

    if repo_rel(path) != V234_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 7:
        add(findings, "error", "v234_code_cell_count", f"expected 7 code cells, found {len(code_cells)}")
    outputs_total = sum(len(cell.get("outputs", []) or []) for cell in code_cells)
    if outputs_total:
        add(findings, "error", "v234_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    if V234_COLAB_URL not in text:
        add(findings, "error", "v234_colab_url_mismatch", V234_COLAB_URL)
    if V234_GITHUB_URL not in text:
        add(findings, "error", "v234_github_url_mismatch", V234_GITHUB_URL)

    for name, snippet in V234_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v234_required_snippet_missing", name)

    for rel_path in V234_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v234_required_file_missing", rel_path)

    analyzer = ROOT / "scripts" / "analyze_v234_external_intel_triage.py"
    analyzer_text = analyzer.read_text(encoding="utf-8") if analyzer.exists() else ""
    analyzer_snippets = {
        "self test ok": "v234_external_intel_triage_self_test=ok",
        "manifest schema": "kg1_v234_external_intel_triage_manifest_v1",
        "coverage schema": "kg1_v234_roadmap_coverage_report_v1",
        "metric schema": "kg1_v234_external_metric_parity_report_v1",
        "expected refs": "EXPECTED_REFS",
        "coverage function": "roadmap_coverage",
        "metric parity function": "metric_parity_report",
        "equation probe rows": "equation_probe_rows",
        "bit probe rows": "bit_probe_rows",
        "decision": "external_intel_triage_ready_for_source_download",
        "blocked actions": "model_generation",
    }
    for name, snippet in analyzer_snippets.items():
        if snippet not in analyzer_text:
            add(findings, "error", "v234_analyzer_contract_missing", name)
    if analyzer.exists():
        completed = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--self-test",
                "--output-dir",
                "dummy",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            add(
                findings,
                "error",
                "v234_analyzer_self_test_failed",
                f"returncode={completed.returncode}; tail={completed.stdout[-4000:]}",
            )
        elif "v234_external_intel_triage_self_test=ok" not in completed.stdout:
            add(findings, "error", "v234_analyzer_self_test_missing_ok", completed.stdout[-4000:])

    builder = ROOT / "scripts" / "build_v234_external_intel_triage_colab.py"
    builder_text = builder.read_text(encoding="utf-8") if builder.exists() else ""
    for snippet in V234_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v234_builder_required_snippet_missing", snippet)
    builder_snippets = {
        "idempotent cell ids": "_CELL_COUNTER = 0",
        "roadmap rel constant": "ROADMAP_REL",
        "roadmap preflight": "required_roadmap_markers",
        "triage command": "analyze_v234_external_intel_triage.py",
        "artifact metadata logs": "v234_output_artifact_meta",
        "submission artifact scan": "blocked_artifacts",
        "final manifest": "v234_external_intel_triage_final_manifest.json",
    }
    for name, snippet in builder_snippets.items():
        if snippet not in builder_text:
            add(findings, "error", "v234_builder_contract_missing", name)
    if builder.exists():
        try:
            spec = importlib.util.spec_from_file_location("kg1_v234_builder_gate", builder)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load builder module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            generated = module.build_notebook()
            if generated != notebook:
                add(findings, "error", "v234_builder_notebook_mismatch", "builder output differs from committed notebook")
        except Exception as exc:
            add(findings, "error", "v234_builder_notebook_compare_failed", repr(exc))


def audit_v235_source_access_triage_contract(path: Path, notebook: dict[str, Any], text: str, findings: list[Finding]) -> None:
    """Strict release gate for the V235 CPU-only source access triage."""

    if repo_rel(path) != V235_NOTEBOOK_REL:
        return

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 7:
        add(findings, "error", "v235_code_cell_count", f"expected 7 code cells, found {len(code_cells)}")
    outputs_total = sum(len(cell.get("outputs", []) or []) for cell in code_cells)
    if outputs_total:
        add(findings, "error", "v235_notebook_has_outputs", f"notebook must be committed clean; outputs={outputs_total}")

    if V235_COLAB_URL not in text:
        add(findings, "error", "v235_colab_url_mismatch", V235_COLAB_URL)
    if V235_GITHUB_URL not in text:
        add(findings, "error", "v235_github_url_mismatch", V235_GITHUB_URL)

    for name, snippet in V235_REQUIRED_SNIPPETS.items():
        if snippet not in text:
            add(findings, "error", "v235_required_snippet_missing", name)

    for rel_path in V235_REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add(findings, "error", "v235_required_file_missing", rel_path)

    analyzer = ROOT / "scripts" / "analyze_v235_source_access_triage.py"
    analyzer_text = analyzer.read_text(encoding="utf-8") if analyzer.exists() else ""
    analyzer_snippets = {
        "self test ok": "v235_source_access_triage_self_test=ok",
        "manifest schema": "kg1_v235_source_access_triage_manifest_v1",
        "license schema": "kg1_v235_license_gate_report_v1",
        "credential audit": "credential_audit",
        "hf metadata": "query_hf_metadata",
        "inventory builder": "build_inventory",
        "v234 validator": "validate_v234_manifest",
        "download blocked action": "payload_download_without_license_hash",
        "decision manual": "manual_source_access_or_license_required_before_download",
        "decision plan ready": "source_access_plan_ready_needs_controlled_download",
    }
    for name, snippet in analyzer_snippets.items():
        if snippet not in analyzer_text:
            add(findings, "error", "v235_analyzer_contract_missing", name)
    if analyzer.exists():
        completed = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--self-test",
                "--v234-analysis-manifest-json",
                "dummy",
                "--output-dir",
                "dummy",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            add(
                findings,
                "error",
                "v235_analyzer_self_test_failed",
                f"returncode={completed.returncode}; tail={completed.stdout[-4000:]}",
            )
        elif "v235_source_access_triage_self_test=ok" not in completed.stdout:
            add(findings, "error", "v235_analyzer_self_test_missing_ok", completed.stdout[-4000:])

    builder = ROOT / "scripts" / "build_v235_source_access_triage_colab.py"
    builder_text = builder.read_text(encoding="utf-8") if builder.exists() else ""
    for snippet in V235_REQUIRED_SNIPPETS.values():
        if snippet not in builder_text:
            add(findings, "error", "v235_builder_required_snippet_missing", snippet)
    builder_snippets = {
        "idempotent cell ids": "_CELL_COUNTER = 0",
        "latest v234 resolver": "resolve_latest_v234_manifest",
        "v234 output preflight": "required_outputs = [",
        "source command": "analyze_v235_source_access_triage.py",
        "artifact metadata logs": "v235_output_artifact_meta",
        "submission artifact scan": "blocked_artifacts",
        "final manifest": "v235_source_access_triage_final_manifest.json",
    }
    for name, snippet in builder_snippets.items():
        if snippet not in builder_text:
            add(findings, "error", "v235_builder_contract_missing", name)
    if builder.exists():
        try:
            spec = importlib.util.spec_from_file_location("kg1_v235_builder_gate", builder)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load builder module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            generated = module.build_notebook()
            if generated != notebook:
                add(findings, "error", "v235_builder_notebook_mismatch", "builder output differs from committed notebook")
        except Exception as exc:
            add(findings, "error", "v235_builder_notebook_compare_failed", repr(exc))


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
        audit_clean_notebook_state(notebook, findings)
        audit_colab_urls(path, text, findings)
        audit_submit_lock(text, findings)
        audit_logging_and_commands("\n".join(code_cells), findings)
        audit_local_repo_references(text, findings)
        audit_artifact_dependency_contract(text, findings)
        audit_training_eval_contract(text, findings)
        audit_hf_jobs_contract(text, findings)
        audit_v218_decode_rescue_contract(path, notebook, text, findings)
        audit_v219_weak_decode_ab_contract(path, notebook, text, findings)
        audit_v220_public_adapter_probe_contract(path, notebook, text, findings)
        audit_v221_candidate_registry_contract(path, notebook, text, findings)
        audit_v230_v226_complementarity_contract(path, notebook, text, findings)
        audit_v231_miss_pack_mining_contract(path, notebook, text, findings)
        audit_v232_verified_solver_workbench_contract(path, notebook, text, findings)
        audit_v233_verified_equation_solver_probes_contract(path, notebook, text, findings)
        audit_v234_external_intel_triage_contract(path, notebook, text, findings)
        audit_v235_source_access_triage_contract(path, notebook, text, findings)
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


def synthetic_notebook(source: str, outputs: list[dict[str, Any]] | None = None, execution_count: int | None = None) -> dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"colab": {"name": "negative.ipynb"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
                    "test/notebooks/negative.ipynb"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": execution_count,
                "outputs": outputs or [],
                "source": source.splitlines(True),
            },
        ],
    }


def run_self_test() -> int:
    base_source = """# CELL: synthetic negative gate fixture.
print('=== SYN START ===')
ALLOW_KAGGLE_SUBMIT = False
print('=== SYN END ===')
"""
    fixtures = {
        "outputs": (
            base_source,
            [{"output_type": "stream", "name": "stdout", "text": ["bad"]}],
            None,
            "notebook_has_outputs",
        ),
        "execution_count": (base_source, [], 7, "notebook_has_execution_counts"),
        "missing_local_ref": (
            base_source + "print('scripts/does_not_exist_for_gate.py')\n",
            [],
            None,
            "referenced_repo_file_missing",
        ),
        "synthesis_default_zero": (
            base_source
            + "import os\n"
            + "V230_DIAGNOSTIC_MODE = True\n"
            + "ALLOW_FAKE_SYNTHESIS = os.environ.get('KG1_FAKE', '0').strip().lower() in {'1'}\n",
            [],
            None,
            "diagnostic_synthesis_default_disabled",
        ),
        "resolver_contract": (
            base_source + "def resolve_existing_or_synthesize_fake(path):\n    return path\n",
            [],
            None,
            "artifact_resolver_contract_missing",
        ),
    }
    print("=== NOTEBOOK RELEASE GATE SELF TEST START ===", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for name, (source, outputs, execution_count, expected_code) in fixtures.items():
            path = tmp / f"{name}.ipynb"
            path.write_text(
                json.dumps(synthetic_notebook(source, outputs, execution_count), indent=2),
                encoding="utf-8",
                newline="\n",
            )
            audit = audit_notebook(path)
            codes = {finding.code for finding in audit.findings}
            if expected_code not in codes:
                print(
                    json.dumps(
                        {
                            "fixture": name,
                            "expected_code": expected_code,
                            "observed_codes": sorted(codes),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return 1
            print(f"negative_gate_fixture_{name}=ok", flush=True)
    print("notebook_release_gate_self_test=ok", flush=True)
    print("=== NOTEBOOK RELEASE GATE SELF TEST END ===", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Notebook paths to audit.")
    parser.add_argument("--all", action="store_true", help="Audit all notebooks under notebooks/ and competent-shamir/notebooks/.")
    parser.add_argument("--changed-from", default="", help="Git ref/sha to diff from.")
    parser.add_argument("--changed-to", default="HEAD", help="Git ref/sha to diff to.")
    parser.add_argument("--allow-empty", action="store_true", help="Return success when notebook discovery finds no notebooks.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true", help="Run gate regression tests for generic notebook validation rules.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.all:
        paths = all_notebooks()
    elif args.paths:
        paths = [path if path.is_absolute() else ROOT / path for path in args.paths]
    else:
        paths = changed_notebooks(args.changed_from or None, args.changed_to)
    paths = [path for path in paths if NOTEBOOK_RE.search(str(path))]
    audits = [audit_notebook(path) for path in paths]
    top_findings: list[Finding] = []
    if not audits and not args.allow_empty:
        add(
            top_findings,
            "error",
            "no_notebooks_selected",
            "No notebooks were selected for audit. Pass notebook paths, --all, --changed-from, or --allow-empty explicitly.",
        )
    report = {
        "schema_version": "kg1_notebook_release_gate_v1",
        "ok": (not top_findings) and all(audit.ok for audit in audits),
        "notebook_count": len(audits),
        "findings": [finding.__dict__ for finding in top_findings],
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
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
