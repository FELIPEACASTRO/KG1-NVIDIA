#!/usr/bin/env python3
"""Strict execution gate for KG1 Colab notebooks.

The gate is intentionally conservative. It validates notebook code, local and
remote pack hashes, expected train/validation hashes, critical environment
variables, dependency pins, ZIP contents, and known failure modes seen in prior
KG1 Colab runs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_PINS = {
    "transformers": "5.7.0",
    "accelerate": "1.13.0",
    "peft": "0.19.1",
    "datasets": "4.8.5",
    "safetensors": "0.7.0",
    "huggingface_hub": "1.13.0",
    "sentencepiece": "0.2.1",
    "protobuf": "7.34.1",
    "ninja": "1.13.0",
    "causal-conv1d": "1.6.1",
    "mamba-ssm": "2.3.1",
}

APPROVED_PACK_SHA256 = {
    # Master notebook pack, commit 49c2c2fd. Data hashes match the V198
    # micro-distill dataset, but the embedded pack manifest predates the final
    # pack file SHA.
    "7e3e41b55bb6f5736c3d5325c7b481f3b52ac918eb13c311e9a343f43f6dedca",
    # Current repaired pack, commit 31d439bc.
    "e61908c0f75018b0d265c3668600170f6fa99a1a4d559508f489cba9cd6b7c93",
}

APPROVED_FIXED_TRAIN_SCRIPT_REVISIONS = {
    "31d439bc4a9b33b7b3c772d3526149847103a9b1",
}

APPROVED_POSTTRAIN_REVISIONS = {
    "cee9825b0edd6ea2e829c94bdd7b1ff9410b30f3",
}

APPROVED_PREFLIGHT_REVISIONS = {
    "8a1c0b934acde30c81237d33a74a18d11f6d5141",
}

APPROVED_FINAL_DOUBLECHECK_REVISIONS = {
    "b2209b4daefb85e9d3c9e4dc5e26b3ba54dbcbd2",
}

APPROVED_SAFE_SUBMIT_REVISIONS = {
    # First helper revision that stages the adapter as exactly submission.zip.
    "cee234f4bac2abcff9f1452a440f0f7576e62eef",
}

V198_FINAL_SUBMIT_ZIP_SHA256 = "52c585c7f075a1a9735d23c16905e535d1ebbf51246b03a50ac3d07c3768a3a9"

REQUIRED_ZIP_ENTRIES = {
    "data/v198/v198_micro_train.strict.jsonl",
    "data/v198/v198_micro_val.strict.jsonl",
    "data/v198/v198_micro_manifest.json",
    "scripts/hf_job_train_v90.py",
    "scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py",
    "scripts/kg1_training_data_gate.py",
    "scripts/kg1_sft_format_validator.py",
    "notebooks/KG1_V198_MICRO_DISTILL_COLAB_PRO.ipynb",
}

EXPECTED_ENV = {
    "UPLOAD_TO_HF": "0",
    "MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "DATA_FILE": "/content/kg1_v198/data/v198/v198_micro_train.strict.jsonl",
    "VAL_FILE": "/content/kg1_v198/data/v198/v198_micro_val.strict.jsonl",
    "INIT_ADAPTER_LOAD_MODE": "manual",
    "PEFT_MANUAL_LOAD_METHOD": "direct",
    "RUN_ID": "v198-micro-distill-v197-gates",
    "MAX_LENGTH": "2048",
    "BATCH_SIZE": "16",
    "MICRO_BATCH_SIZE": "1",
    "GRADIENT_CHECKPOINTING": "1",
    "MAX_STEPS": "45",
    "SAVE_EVERY_STEPS": "15",
    "EVAL_EVERY_STEPS": "15",
    "EVAL_MAX_EXAMPLES": "240",
    "LEARNING_RATE": "1e-5",
    "FINAL_LEARNING_RATE": "3e-6",
    "MIN_TRAIN_EXAMPLES": "1875",
    "MIN_TOKENIZED_TRAIN_EXAMPLES": "1600",
    "MIN_VAL_EXAMPLES": "720",
    "MIN_TOKENIZED_VAL_EXAMPLES": "700",
    "TRAINABLE_LORA_MODULES": "in_proj,out_proj,q_proj,k_proj,v_proj,o_proj",
    "MAX_TRAINABLE_PARAM_RATIO": "0.035",
}


def github_raw_revision(url: str) -> str:
    if "/KG1-NVIDIA/" not in url:
        return ""
    return url.split("/KG1-NVIDIA/", 1)[1].split("/", 1)[0]


@dataclass
class Finding:
    severity: str
    code: str
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            json.loads(line)
            count += 1
    return count


def add(findings: list[Finding], severity: str, code: str, detail: str) -> None:
    findings.append(Finding(severity=severity, code=code, detail=detail))


def regex_value(source: str, name: str) -> str | None:
    patterns = [
        rf"{re.escape(name)}\s*=\s*'([^']*)'",
        rf'{re.escape(name)}\s*=\s*"([^"]*)"',
        rf"os\.environ\[['\"]{re.escape(name)}['\"]\]\s*=\s*'([^']*)'",
        rf'os\.environ\[[\'"]{re.escape(name)}[\'"]\]\s*=\s*"([^"]*)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return match.group(1)
    return None


def regex_values(source: str, name: str) -> list[str]:
    patterns = [
        rf"{re.escape(name)}\s*=\s*'([^']*)'",
        rf'{re.escape(name)}\s*=\s*"([^"]*)"',
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(1) for match in re.finditer(pattern, source))
    return values


def extract_env_assignments(source: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for match in re.finditer(r"os\.environ\[['\"]([^'\"]+)['\"]\]\s*=\s*'([^']*)'", source):
        env[match.group(1)] = match.group(2)
    for match in re.finditer(r'os\.environ\[[\'"]([^\'"]+)[\'"]\]\s*=\s*"([^"]*)"', source):
        env[match.group(1)] = match.group(2)
    return env


def extract_pins(source: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for package, version in re.findall(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+-]+)", source):
        pins[package] = version
    return pins


def notebook_source(nb: dict[str, Any]) -> tuple[str, list[str]]:
    cells = nb.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook has no cells list")
    code_cells: list[str] = []
    all_source: list[str] = []
    for cell in cells:
        source = "".join(cell.get("source") or [])
        all_source.append(source)
        if cell.get("cell_type") == "code":
            code_cells.append(source)
    return "\n".join(all_source), code_cells


def compile_python_cells(code_cells: list[str], findings: list[Finding]) -> None:
    for idx, source in enumerate(code_cells):
        stripped_lines: list[str] = []
        shell_magic_seen = False
        for raw_line in source.splitlines():
            line = raw_line.rstrip()
            if line.startswith("%"):
                continue
            if line.startswith("!"):
                shell_magic_seen = True
                continue
            if "!" in line and line.lstrip().startswith("!"):
                shell_magic_seen = True
                continue
            stripped_lines.append(line)
        candidate = "\n".join(stripped_lines).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            # Cells with shell magics plus line continuations are intentionally
            # not pure Python. Anything else is a hard failure.
            if shell_magic_seen:
                continue
            add(findings, "error", "notebook_python_syntax", f"code cell {idx}: {exc}")


def check_required_strings(source: str, findings: list[Finding]) -> None:
    required = {
        "drive_mount": "drive.mount('/content/drive')",
        "root_path": "ROOT = pathlib.Path('/content/kg1_v198')",
        "drive_root": "DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V198')",
        "pack_name": "kg1_v198_colab_pack.zip",
        "pack_hash_assert": "assert pack_hash == PACK_SHA256",
        "stale_pack_delete": "PACK.unlink()",
        "baseline_model_hash": "BASE_ADAPTER_MODEL_SHA256",
        "baseline_config_hash": "BASE_ADAPTER_CONFIG_SHA256",
        "adapter_candidates": "V195_OUT / 'checkpoint-55'",
        "torchao_uninstall": "pip_uninstall('torchao')",
        "torchao_assert_absent": "assert importlib.util.find_spec('torchao') is None",
        "mamba_import": "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn",
        "output_cleanup": "shutil.rmtree(OUT, ignore_errors=True)",
        "stale_script_repair": "Runtime has stale hf_job_train_v90.py; downloading PEFT direct-load fixed script",
        "fixed_train_script_url": "FIXED_TRAIN_SCRIPT_URL",
        "fixed_train_script_download": "urllib.request.urlretrieve(FIXED_TRAIN_SCRIPT_URL, TRAIN_SCRIPT)",
        "fixed_train_script_patch_assert": "assert 'load_peft_weights_with_direct_fallback' in script_text",
        "fixed_train_script_env_assert": "assert 'PEFT_MANUAL_LOAD_METHOD' in script_text",
        "train_script": "!python scripts/hf_job_train_v90.py",
        "posttrain_gate_script": "kg1_v198_posttrain_gate.py",
        "posttrain_gate_fail_on_block": "--fail-on-block",
        "preflight_script": "nemotron_submission_preflight.py",
        "submission_gate_script": "kg1_submission_gate.py",
        "final_submit_doublecheck": "kg1_v198_final_submit_doublecheck.py",
    }
    for code, needle in required.items():
        if needle not in source:
            add(findings, "error", code, f"missing required notebook fragment: {needle}")
    has_legacy_converter = "!python scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py" in source
    has_posttrain_gate = "!python scripts/kg1_v198_posttrain_gate.py" in source
    if not has_legacy_converter and not has_posttrain_gate:
        add(
            findings,
            "error",
            "converter_or_posttrain_gate",
            "notebook must convert the trained adapter or run kg1_v198_posttrain_gate.py",
        )
    if has_posttrain_gate and "POSTTRAIN_SCRIPT_URL" not in source:
        add(
            findings,
            "error",
            "posttrain_gate_download_guard",
            "posttrain gate notebook cell must define POSTTRAIN_SCRIPT_URL for stale pack/runtime repair",
        )
    if "final_preflight.json" not in source:
        add(findings, "error", "final_preflight_missing", "notebook must run final ZIP preflight")
    if "checkpoint30_preflight.json" not in source:
        add(findings, "warning", "checkpoint30_preflight_missing", "checkpoint-30 preflight is recommended for fallback selection")
    if "final_submit_doublecheck.json" not in source:
        add(findings, "error", "final_doublecheck_missing", "notebook must write final_submit_doublecheck.json before any submit")


def check_paths_and_env(source: str, manifest: dict[str, Any], findings: list[Finding]) -> None:
    pack_url = regex_value(source, "PACK_URL")
    pack_sha = regex_value(source, "PACK_SHA256")
    fixed_train_script_url = regex_value(source, "FIXED_TRAIN_SCRIPT_URL")
    if not pack_url:
        add(findings, "error", "missing_pack_url", "PACK_URL not found")
    else:
        if "raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/" not in pack_url:
            add(findings, "error", "pack_url_wrong_repo", pack_url)
        revision = github_raw_revision(pack_url)
        if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
            add(findings, "error", "pack_url_not_commit_pinned", f"revision={revision!r}")
        if "kg1_v198_colab_pack.zip" not in pack_url:
            add(findings, "error", "pack_url_wrong_file", pack_url)
    if not fixed_train_script_url:
        add(findings, "error", "missing_fixed_train_script_url", "FIXED_TRAIN_SCRIPT_URL not found")
    else:
        if "raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/" not in fixed_train_script_url:
            add(findings, "error", "fixed_train_script_url_wrong_repo", fixed_train_script_url)
        fixed_revision = github_raw_revision(fixed_train_script_url)
        if not re.fullmatch(r"[0-9a-f]{7,40}", fixed_revision):
            add(findings, "error", "fixed_train_script_url_not_commit_pinned", f"revision={fixed_revision!r}")
        elif fixed_revision not in APPROVED_FIXED_TRAIN_SCRIPT_REVISIONS:
            add(findings, "error", "fixed_train_script_revision_not_approved", fixed_revision)
        if not fixed_train_script_url.endswith("/scripts/hf_job_train_v90.py"):
            add(findings, "error", "fixed_train_script_url_wrong_file", fixed_train_script_url)
    if not pack_sha:
        add(findings, "error", "missing_pack_sha", "PACK_SHA256 not found")
    elif pack_sha not in APPROVED_PACK_SHA256:
        add(findings, "error", "pack_sha_not_approved", pack_sha)
    elif manifest.get("pack_sha256") and pack_sha != manifest.get("pack_sha256"):
        add(
            findings,
            "warning",
            "pack_sha_manifest_mismatch",
            f"notebook={pack_sha} manifest={manifest.get('pack_sha256')}",
        )

    env = extract_env_assignments(source)
    for name, expected in EXPECTED_ENV.items():
        observed = env.get(name)
        if observed != expected:
            add(findings, "error", f"env_{name.lower()}_mismatch", f"observed={observed!r} expected={expected!r}")
    if env.get("EXPECTED_TRAIN_SHA256") != manifest.get("strict_train_sha256"):
        add(findings, "error", "expected_train_sha_mismatch", "notebook EXPECTED_TRAIN_SHA256 != manifest strict_train_sha256")
    if env.get("EXPECTED_VAL_SHA256") != manifest.get("val_sha256"):
        add(findings, "error", "expected_val_sha_mismatch", "notebook EXPECTED_VAL_SHA256 != manifest val_sha256")
    if "os.environ['INIT_ADAPTER_DIR'] = str(INIT_ADAPTER)" not in source:
        add(findings, "error", "init_adapter_not_dynamic", "INIT_ADAPTER_DIR must be set from selected INIT_ADAPTER")


def check_dependencies(source: str, findings: list[Finding]) -> None:
    pins = extract_pins(source)
    for package, expected in EXPECTED_PINS.items():
        observed = pins.get(package)
        if observed != expected:
            add(findings, "error", f"dependency_{package}_pin", f"observed={observed!r} expected={expected!r}")
    if re.search(r"pip_install\(\[[^\]]*'transformers'(?:,|\])", source):
        add(findings, "error", "unversioned_transformers_install", "transformers must be version-pinned")
    if "--no-build-isolation" not in source:
        add(findings, "error", "missing_no_build_isolation", "mamba/causal_conv installs should use --no-build-isolation")


def check_script_lineage_and_submission_cells(source: str, code_cells: list[str], findings: list[Finding]) -> None:
    posttrain_url = regex_value(source, "POSTTRAIN_SCRIPT_URL") or ""
    if not posttrain_url:
        add(findings, "error", "missing_posttrain_script_url", "POSTTRAIN_SCRIPT_URL not found")
    else:
        revision = github_raw_revision(posttrain_url)
        if revision not in APPROVED_POSTTRAIN_REVISIONS:
            add(findings, "error", "posttrain_script_revision_not_approved", revision or posttrain_url)

    base_urls = regex_values(source, "BASE")
    preflight_bases = [url for url in base_urls if "raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/" in url]
    if preflight_bases:
        latest_preflight_revision = github_raw_revision(preflight_bases[-1])
        if latest_preflight_revision not in APPROVED_PREFLIGHT_REVISIONS:
            add(findings, "error", "preflight_base_revision_not_approved", latest_preflight_revision)
    else:
        add(findings, "error", "missing_preflight_base_url", "preflight script download BASE URL not found")

    doublecheck_urls = [
        url for url in regex_values(source, "URL")
        if url.endswith("/scripts/kg1_v198_final_submit_doublecheck.py")
    ]
    if not doublecheck_urls:
        add(findings, "error", "missing_final_doublecheck_url", "kg1_v198_final_submit_doublecheck.py URL not found")
    else:
        revision = github_raw_revision(doublecheck_urls[-1])
        if revision not in APPROVED_FINAL_DOUBLECHECK_REVISIONS:
            add(findings, "error", "final_doublecheck_revision_not_approved", revision)

    raw_submit_cells: list[int] = []
    for idx, cell_source in enumerate(code_cells):
        for line in cell_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("!kaggle competitions submit"):
                raw_submit_cells.append(idx)
                break
    if raw_submit_cells:
        add(
            findings,
            "error",
            "active_raw_kaggle_submit_cells",
            f"raw kaggle CLI submit cells must be removed or commented before Run all: cells={raw_submit_cells}",
        )

    safe_submit_present = "kg1_v198_safe_kaggle_submit.py" in source
    if safe_submit_present:
        safe_urls = [
            url for url in regex_values(source, "URL")
            if url.endswith("/scripts/kg1_v198_safe_kaggle_submit.py")
        ]
        if not safe_urls:
            add(findings, "error", "safe_submit_helper_url_missing", "safe submit helper must be downloaded from a pinned URL")
        else:
            revision = github_raw_revision(safe_urls[-1])
            if revision not in APPROVED_SAFE_SUBMIT_REVISIONS:
                add(findings, "error", "safe_submit_revision_without_submission_zip_fix", revision)
        for fragment in [
            "--candidate-zip",
            "--doublecheck-json",
            "--expected-sha256",
            V198_FINAL_SUBMIT_ZIP_SHA256,
            "--no-raise-on-submit-error",
            "final_safe_submit_report_submission_name_fixed.json",
        ]:
            if fragment not in source:
                add(findings, "error", "safe_submit_missing_guard", fragment)

    if "files.upload()" in source:
        add(
            findings,
            "warning",
            "manual_kaggle_json_upload_cell_present",
            "manual upload of kaggle.json is present; Colab Secrets userdata path is preferred",
        )


def check_peft_direct_load_script_source(source: str, label: str, findings: list[Finding]) -> None:
    """Block PEFT/Transformers adapter-load regressions before Colab runtime."""

    required = [
        "PEFT_MANUAL_LOAD_METHOD = env_str",
        "def remap_peft_state_dict_for_direct_load",
        "def load_peft_weights_with_direct_fallback",
        "load_peft_weights_with_direct_fallback(loaded_model, weights, adapter_name=\"default\")",
        "set_peft_model_state_dict failed; falling back to direct PEFT state_dict load",
        "Direct PEFT adapter load mapping:",
        "coverage < 0.98",
        "missing_lora={len(missing_lora)}",
    ]
    for fragment in required:
        if fragment not in source:
            add(findings, "error", "peft_direct_load_guard_missing", f"{label}: missing {fragment!r}")

    forbidden_old_blocks = [
        "print(f\"Manual local adapter load: tensors={len(weights)}\")\n            set_peft_model_state_dict(",
        "print(f\"Manual HF adapter load: tensors={len(weights)}\")\n            set_peft_model_state_dict(",
    ]
    for fragment in forbidden_old_blocks:
        if fragment in source:
            add(findings, "error", "peft_stale_direct_setter_block", f"{label}: stale direct set_peft_model_state_dict block remains")

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        add(findings, "error", "peft_script_ast_parse_failed", f"{label}: {exc}")
        return

    def call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if function.name == "load_peft_weights_with_direct_fallback":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and call_name(node.func) == "set_peft_model_state_dict":
                add(
                    findings,
                    "error",
                    "peft_direct_setter_outside_fallback",
                    f"{label}: {function.name} still calls set_peft_model_state_dict directly",
                )


def check_pack(
    pack_path: Path,
    manifest: dict[str, Any],
    train_path: Path,
    val_path: Path,
    findings: list[Finding],
    expected_pack_sha: str = "",
) -> None:
    if not pack_path.exists():
        add(findings, "error", "pack_missing", str(pack_path))
        return
    local_pack_sha = sha256_file(pack_path)
    expected = expected_pack_sha or str(manifest.get("pack_sha256") or "")
    if expected and local_pack_sha != expected:
        add(findings, "error", "local_pack_sha_mismatch", f"{local_pack_sha} != {expected}")
    if local_pack_sha not in APPROVED_PACK_SHA256:
        add(findings, "error", "local_pack_sha_not_approved", local_pack_sha)
    if manifest.get("pack_sha256") and local_pack_sha != manifest.get("pack_sha256"):
        add(
            findings,
            "warning",
            "local_pack_sha_manifest_mismatch",
            f"{local_pack_sha} != {manifest.get('pack_sha256')}",
        )
    try:
        with zipfile.ZipFile(pack_path) as archive:
            bad = archive.testzip()
            if bad:
                add(findings, "error", "zip_crc_failure", bad)
            entries = set(archive.namelist())
            missing = sorted(REQUIRED_ZIP_ENTRIES - entries)
            if missing:
                add(findings, "error", "zip_missing_entries", ", ".join(missing))
            train_bytes = archive.read("data/v198/v198_micro_train.strict.jsonl")
            val_bytes = archive.read("data/v198/v198_micro_val.strict.jsonl")
            if hashlib.sha256(train_bytes).hexdigest() != sha256_file(train_path):
                add(findings, "error", "zip_train_hash_mismatch", "ZIP train JSONL differs from local train JSONL")
            if hashlib.sha256(val_bytes).hexdigest() != sha256_file(val_path):
                add(findings, "error", "zip_val_hash_mismatch", "ZIP val JSONL differs from local val JSONL")
            for script in ["scripts/hf_job_train_v90.py", "scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py"]:
                local = Path(script)
                if local.exists() and hashlib.sha256(archive.read(script)).hexdigest() != sha256_file(local):
                    add(findings, "error", "zip_script_hash_mismatch", script)
            train_script_source = archive.read("scripts/hf_job_train_v90.py").decode("utf-8")
            check_peft_direct_load_script_source(train_script_source, "zip:scripts/hf_job_train_v90.py", findings)
    except Exception as exc:
        add(findings, "error", "zip_read_failed", str(exc))


def check_data_files(train_path: Path, val_path: Path, manifest: dict[str, Any], findings: list[Finding]) -> None:
    for path, key in [(train_path, "strict_train_sha256"), (val_path, "val_sha256")]:
        if not path.exists():
            add(findings, "error", "data_file_missing", str(path))
            continue
        observed = sha256_file(path)
        expected = manifest.get(key)
        if observed != expected:
            add(findings, "error", "data_file_sha_mismatch", f"{path}: {observed} != {expected}")
    if train_path.exists() and read_jsonl_count(train_path) != int(manifest.get("train_rows", -1)):
        add(findings, "error", "train_row_count_mismatch", f"{read_jsonl_count(train_path)} != {manifest.get('train_rows')}")
    if val_path.exists() and read_jsonl_count(val_path) != int(manifest.get("val_rows", -1)):
        add(findings, "error", "val_row_count_mismatch", f"{read_jsonl_count(val_path)} != {manifest.get('val_rows')}")


def check_training_script(findings: list[Finding]) -> None:
    scripts = [Path("scripts/hf_job_train_v90.py"), Path("scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py")]
    for script in scripts:
        if not script.exists():
            add(findings, "error", "script_missing", str(script))
            continue
        try:
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
        except SyntaxError as exc:
            add(findings, "error", "script_syntax_error", f"{script}: {exc}")
    train_src = Path("scripts/hf_job_train_v90.py").read_text(encoding="utf-8")
    check_peft_direct_load_script_source(train_src, "local:scripts/hf_job_train_v90.py", findings)
    required_train_fragments = [
        "assert_file_sha256(train_path, EXPECTED_TRAIN_SHA256",
        "assert_file_sha256(val_path, EXPECTED_VAL_SHA256",
        "disable_peft_torchao_dispatch_if_incompatible()",
        "load_peft_weights_with_direct_fallback",
        "PEFT_MANUAL_LOAD_METHOD",
        "TRAINABLE_LORA_MODULES matched no LoRA parameters",
        "MAX_TRAINABLE_PARAM_RATIO",
        "bitsandbytes optimizer unavailable",
        "final_adapter",
    ]
    for fragment in required_train_fragments:
        if fragment not in train_src:
            add(findings, "error", "train_script_missing_guard", fragment)


def network_check(pack_url: str, pack_sha: str, findings: list[Finding]) -> None:
    if not pack_url:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as handle:
        tmp_path = Path(handle.name)
    try:
        with urllib.request.urlopen(pack_url, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                add(findings, "error", "pack_url_status", f"{status}")
            tmp_path.write_bytes(response.read())
        observed = sha256_file(tmp_path)
        if observed != pack_sha:
            add(findings, "error", "remote_pack_sha_mismatch", f"{observed} != {pack_sha}")
    except urllib.error.HTTPError as exc:
        add(findings, "error", "pack_url_http_error", f"{exc.code}: {exc.reason}")
    except Exception as exc:
        add(findings, "error", "pack_url_download_failed", str(exc))
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def network_check_fixed_train_script(fixed_train_script_url: str, findings: list[Finding]) -> None:
    if not fixed_train_script_url:
        return
    try:
        with urllib.request.urlopen(fixed_train_script_url, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                add(findings, "error", "fixed_train_script_url_status", f"{status}")
            data = response.read()
        source = data.decode("utf-8")
        check_peft_direct_load_script_source(source, "remote:FIXED_TRAIN_SCRIPT_URL", findings)
        local_script = Path("scripts/hf_job_train_v90.py")
        if local_script.exists():
            remote_sha = hashlib.sha256(data).hexdigest()
            local_sha = sha256_file(local_script)
            if remote_sha != local_sha:
                add(
                    findings,
                    "error",
                    "fixed_train_script_remote_local_sha_mismatch",
                    f"remote={remote_sha} local={local_sha}",
                )
    except urllib.error.HTTPError as exc:
        add(findings, "error", "fixed_train_script_url_http_error", f"{exc.code}: {exc.reason}")
    except Exception as exc:
        add(findings, "error", "fixed_train_script_download_failed", str(exc))


def pypi_check(findings: list[Finding]) -> None:
    for package, version in EXPECTED_PINS.items():
        url = f"https://pypi.org/pypi/{package}/{version}/json"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                if getattr(response, "status", 200) != 200:
                    add(findings, "error", "pypi_version_status", f"{package}=={version}: {getattr(response, 'status', None)}")
        except Exception as exc:
            add(findings, "error", "pypi_version_unavailable", f"{package}=={version}: {exc}")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    findings = payload["findings"]
    lines = [
        "# KG1 Colab IPYNB Execution Gate",
        "",
        f"- decision: `{payload['decision']}`",
        f"- errors: `{payload['error_count']}`",
        f"- warnings: `{payload['warning_count']}`",
        f"- notebook: `{payload['notebook']}`",
        f"- pack: `{payload['pack']}`",
        f"- pack_sha256: `{payload['pack_sha256']}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in payload["checks"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Findings", ""])
    if not findings:
        lines.append("- none")
    else:
        for item in findings:
            lines.append(f"- `{item['severity']}` `{item['code']}`: {item['detail']}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--notebook-url", help="Optional raw GitHub notebook URL to download before validation.")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/v198/v198_micro_manifest.json"))
    parser.add_argument("--train", type=Path, default=Path("data/v198/v198_micro_train.strict.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/v198/v198_micro_val.strict.jsonl"))
    parser.add_argument("--network", action="store_true", help="Download PACK_URL and query PyPI version JSON endpoints.")
    parser.add_argument("--output-json", type=Path, default=Path("runs/v198_micro_distill_colab_pack_20260503/v198_colab_ipynb_execution_gate.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runs/v198_micro_distill_colab_pack_20260503/V198_COLAB_IPYNB_EXECUTION_GATE.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []

    if args.notebook_url:
        try:
            args.notebook.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(args.notebook_url, timeout=60) as response:
                args.notebook.write_bytes(response.read())
        except Exception as exc:
            add(findings, "error", "notebook_url_download_failed", str(exc))

    if not args.notebook.exists():
        add(findings, "error", "notebook_missing", str(args.notebook))
        nb = {"cells": []}
        all_source = ""
        code_cells: list[str] = []
    else:
        try:
            nb = read_json(args.notebook)
            all_source, code_cells = notebook_source(nb)
            if nb.get("nbformat") != 4:
                add(findings, "error", "notebook_nbformat", f"observed={nb.get('nbformat')}")
            compile_python_cells(code_cells, findings)
        except Exception as exc:
            add(findings, "error", "notebook_parse_failed", str(exc))
            nb = {"cells": []}
            all_source = ""
            code_cells = []

    manifest = read_json(args.manifest) if args.manifest.exists() else {}
    if not manifest:
        add(findings, "error", "manifest_missing", str(args.manifest))

    check_required_strings(all_source, findings)
    check_paths_and_env(all_source, manifest, findings)
    check_dependencies(all_source, findings)
    check_script_lineage_and_submission_cells(all_source, code_cells, findings)
    check_data_files(args.train, args.val, manifest, findings)

    pack_url = regex_value(all_source, "PACK_URL") or ""
    pack_sha = regex_value(all_source, "PACK_SHA256") or ""
    fixed_train_script_url = regex_value(all_source, "FIXED_TRAIN_SCRIPT_URL") or ""
    check_pack(args.pack, manifest, args.train, args.val, findings, expected_pack_sha=pack_sha)
    check_training_script(findings)
    if args.network:
        network_check(pack_url, pack_sha, findings)
        network_check_fixed_train_script(fixed_train_script_url, findings)
        pypi_check(findings)

    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    decision = "PASS" if not errors else "FAIL"
    payload = {
        "decision": decision,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "notebook": str(args.notebook),
        "pack": str(args.pack),
        "pack_url": pack_url,
        "fixed_train_script_url": fixed_train_script_url,
        "pack_sha256": pack_sha or manifest.get("pack_sha256", ""),
        "checks": {
            "notebook_cells": len(nb.get("cells", [])),
            "code_cells": len(code_cells),
            "train_rows": manifest.get("train_rows"),
            "val_rows": manifest.get("val_rows"),
            "network": bool(args.network),
        },
        "findings": [finding.__dict__ for finding in findings],
    }
    write_report(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
