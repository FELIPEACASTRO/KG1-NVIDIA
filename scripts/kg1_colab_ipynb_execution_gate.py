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
        "train_script": "!python scripts/hf_job_train_v90.py",
        "converter_script": "!python scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py",
    }
    for code, needle in required.items():
        if needle not in source:
            add(findings, "error", code, f"missing required notebook fragment: {needle}")


def check_paths_and_env(source: str, manifest: dict[str, Any], findings: list[Finding]) -> None:
    pack_url = regex_value(source, "PACK_URL")
    pack_sha = regex_value(source, "PACK_SHA256")
    if not pack_url:
        add(findings, "error", "missing_pack_url", "PACK_URL not found")
    else:
        if "raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/" not in pack_url:
            add(findings, "error", "pack_url_wrong_repo", pack_url)
        revision = pack_url.split("/KG1-NVIDIA/", 1)[1].split("/", 1)[0] if "/KG1-NVIDIA/" in pack_url else ""
        if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
            add(findings, "error", "pack_url_not_commit_pinned", f"revision={revision!r}")
        if "kg1_v198_colab_pack.zip" not in pack_url:
            add(findings, "error", "pack_url_wrong_file", pack_url)
    if not pack_sha:
        add(findings, "error", "missing_pack_sha", "PACK_SHA256 not found")
    elif pack_sha != manifest.get("pack_sha256"):
        add(
            findings,
            "error",
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


def check_pack(
    pack_path: Path,
    manifest: dict[str, Any],
    train_path: Path,
    val_path: Path,
    findings: list[Finding],
) -> None:
    if not pack_path.exists():
        add(findings, "error", "pack_missing", str(pack_path))
        return
    local_pack_sha = sha256_file(pack_path)
    if local_pack_sha != manifest.get("pack_sha256"):
        add(findings, "error", "local_pack_sha_mismatch", f"{local_pack_sha} != {manifest.get('pack_sha256')}")
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
    check_data_files(args.train, args.val, manifest, findings)
    check_pack(args.pack, manifest, args.train, args.val, findings)
    check_training_script(findings)

    pack_url = regex_value(all_source, "PACK_URL") or ""
    pack_sha = regex_value(all_source, "PACK_SHA256") or ""
    if args.network:
        network_check(pack_url, pack_sha, findings)
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
