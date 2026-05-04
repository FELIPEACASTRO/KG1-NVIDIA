#!/usr/bin/env python3
"""Surgical triple-check for the V201C H100/A100 multi-candidate notebook.

This checker validates the notebook as an executable contract, not just as text:
cell structure, exact baseline lineage, candidate hyperparameters, package
inputs, raw dependency URLs, local script syntax, and deterministic builder
output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import py_compile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NOTEBOOK = Path("notebooks/KG1_V201C_H100_A100_MULTI_CANDIDATE_MICRO_COLAB_PRO.ipynb")
BRANCH_BASE = "https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir"
PACK_URL = (
    "https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/"
    "31d439bc4a9b33b7b3c772d3526149847103a9b1/"
    "runs/v198_micro_distill_colab_pack_20260503/kg1_v198_colab_pack.zip"
)

EXPECTED = {
    "v194_zip_sha256": "49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8",
    "v194_adapter_model_sha256": "01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f",
    "v194_adapter_config_sha256": "e5499f128fde60d32d0595d427e4fe84d8abe6dbde1d80886c970e8184e4b743",
    "model_revision": "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
    "train_sha256": "6d2742616300818eb50c54d36019551b24f5b71c607a2b28feda7461a709def0",
    "val_sha256": "e59c907c6545e5e587097a64762e3e874508e8cd74d85d5c7c79354ebe56e73c",
}
APPROVED_PACK_SHA256 = {
    "e61908c0f75018b0d265c3668600170f6fa99a1a4d559508f489cba9cd6b7c93",
    "7e3e41b55bb6f5736c3d5325c7b481f3b52ac918eb13c311e9a343f43f6dedca",
}
REQUIRED_PACK_MEMBERS = {
    "data/v198/v198_micro_train.strict.jsonl": EXPECTED["train_sha256"],
    "data/v198/v198_micro_val.strict.jsonl": EXPECTED["val_sha256"],
    "scripts/hf_job_train_v90.py": None,
    "scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py": None,
}
REQUIRED_RAW_SCRIPTS = [
    "scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py",
    "scripts/hf_job_train_v90.py",
    "scripts/kg1_v198_posttrain_gate.py",
    "scripts/kg1_v201c_posttrain_gate.py",
    "scripts/nemotron_submission_preflight.py",
    "scripts/kg1_submission_gate.py",
    "scripts/kg1_v198_final_submit_doublecheck.py",
]
LOCAL_SCRIPTS = REQUIRED_RAW_SCRIPTS + [
    "scripts/build_v201c_multicandidate_notebook.py",
    "scripts/kg1_v201c_multi_notebook_doublecheck.py",
    "scripts/kg1_v201c_static_doublecheck.py",
]
FORBIDDEN_NOTEBOOK_FRAGMENTS = [
    "files.upload",
    "kaggle competitions submit",
    "KaggleApi",
    "ALLOW_V194_REBUILD_FALLBACK=1",
    "V201C fallback rebuild requested",
    "Rebuilding exact V194 rank-19 adapter",
    "bit_manipulation=2.5",
    "v198_v196_wrong_anti_regression=2.0",
    "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5",
    "output_v201a_h100_solver_verified_micro_5",
    "output_v201b_h100_baseline_neutral_micro_3",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def clean_code_for_ast(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))


def literal_dict_assign(module: ast.Module, name: str) -> dict[str, Any]:
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Dict):
                result: dict[str, Any] = {}
                for key, value in zip(node.value.keys, node.value.values):
                    if key is None:
                        continue
                    key_value = ast.literal_eval(key)
                    try:
                        result[key_value] = ast.literal_eval(value)
                    except Exception:
                        result[key_value] = "<dynamic>"
                return result
    raise KeyError(name)


def literal_assign(module: ast.Module, name: str) -> Any:
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def parse_weight_map(value: str) -> dict[str, float]:
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(f"weight entry must use key=value: {item}")
        key, raw_value = item.split("=", 1)
        result[key] = float(raw_value)
    return result


def add(findings: list[dict[str, Any]], severity: str, code: str, detail: Any) -> None:
    findings.append({"severity": severity, "code": code, "detail": detail})


def load_notebook(path: Path) -> tuple[dict[str, Any], list[str], str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source") or []) for cell in notebook.get("cells", [])]
    return notebook, cells, "\n".join(cells)


def check_notebook_contract(cells: list[str], source: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cells) != 9:
        add(findings, "error", "unexpected_cell_count", len(cells))

    for fragment in FORBIDDEN_NOTEBOOK_FRAGMENTS:
        if fragment in source:
            add(findings, "error", "forbidden_notebook_fragment", fragment)

    required_fragments = [
        "KG1 V201C H100/A100 three-candidate micro-train",
        f"V194_RANK19_ZIP_SHA256 = '{EXPECTED['v194_zip_sha256']}'",
        f"V194_RANK19_ADAPTER_MODEL_SHA256 = '{EXPECTED['v194_adapter_model_sha256']}'",
        f"V194_RANK19_ADAPTER_CONFIG_SHA256 = '{EXPECTED['v194_adapter_config_sha256']}'",
        "ALLOW_V194_REBUILD_FALLBACK = False",
        "raise RuntimeError(missing_v194_zip_message())",
        "TRAIN_SCRIPT.parent.mkdir(parents=True, exist_ok=True)",
        "dst.parent.mkdir(parents=True, exist_ok=True)",
        "kg1_convert_local_training_adapter_to_kaggle_zip.py",
        "No Kaggle submit was performed.",
    ]
    for fragment in required_fragments:
        if fragment not in source:
            add(findings, "error", "missing_notebook_fragment", fragment)

    ensure_start = source.find("def ensure_rank19_v194_adapter():")
    ensure_end = source.find("DRIVE_ROOT.mkdir", ensure_start)
    ensure_block = source[ensure_start:ensure_end] if ensure_start >= 0 and ensure_end > ensure_start else ""
    if "ensure_aaitdads_component()" in ensure_block or "ensure_lineage_component()" in ensure_block:
        add(findings, "error", "init_adapter_has_rebuild_path", ensure_block)

    return {"cell_count": len(cells), "ensure_rank19_block_bytes": len(ensure_block)}


def check_training_cell(cells: list[str], findings: list[dict[str, Any]]) -> dict[str, Any]:
    module = ast.parse(clean_code_for_ast(cells[6]))
    base_env = literal_dict_assign(module, "BASE_ENV")
    candidates = literal_assign(module, "CANDIDATES")

    expected_base = {
        "UPLOAD_TO_HF": "0",
        "MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "MODEL_REVISION": EXPECTED["model_revision"],
        "DATA_FILE": "/content/kg1_v199/data/v198/v198_micro_train.strict.jsonl",
        "VAL_FILE": "/content/kg1_v199/data/v198/v198_micro_val.strict.jsonl",
        "INIT_ADAPTER_LOAD_MODE": "manual",
        "PEFT_MANUAL_LOAD_METHOD": "direct",
        "MAX_LENGTH": "2048",
        "BATCH_SIZE": "16",
        "MICRO_BATCH_SIZE": "1",
        "ABORT_EVAL_LOSS_GT": "0",
        "BASELINE_EVAL_BEFORE_TRAIN": "1",
        "REQUIRE_FINAL_EVAL_LTE_BASELINE": "1",
        "MAX_FINAL_EVAL_REGRESSION": "0.0",
        "EXPECTED_TRAIN_SHA256": EXPECTED["train_sha256"],
        "EXPECTED_VAL_SHA256": EXPECTED["val_sha256"],
        "TRAINABLE_LORA_MODULES": "in_proj,out_proj,q_proj,k_proj,v_proj,o_proj",
        "MAX_TRAINABLE_PARAM_RATIO": "0.035",
    }
    for key, value in expected_base.items():
        if base_env.get(key) != value:
            add(findings, "error", "base_env_mismatch", {"key": key, "expected": value, "actual": base_env.get(key)})

    expected_candidates = {
        "A_neutral_shuffle_3s": {"max_steps": "3", "learning_rate": "2e-7", "final_learning_rate": "1e-7", "sampling_mode": "shuffle"},
        "B_equation_crypt_low_2s": {"max_steps": "2", "learning_rate": "1e-7", "final_learning_rate": "5e-8", "sampling_mode": "weighted_replacement"},
        "C_bit_cipher_low_2s": {"max_steps": "2", "learning_rate": "1e-7", "final_learning_rate": "5e-8", "sampling_mode": "weighted_replacement"},
    }
    labels = [candidate.get("label") for candidate in candidates]
    if labels != list(expected_candidates):
        add(findings, "error", "candidate_label_order_mismatch", labels)
    if len(labels) != len(set(labels)):
        add(findings, "error", "duplicate_candidate_labels", labels)

    for candidate in candidates:
        label = candidate["label"]
        for key, value in expected_candidates.get(label, {}).items():
            if candidate.get(key) != value:
                add(findings, "error", "candidate_config_mismatch", {"label": label, "key": key, "expected": value, "actual": candidate.get(key)})
        for weight_field in ("subcategory_weights", "source_weights"):
            try:
                weights = parse_weight_map(candidate.get(weight_field, ""))
            except Exception as exc:
                add(findings, "error", "bad_weight_map", {"label": label, "field": weight_field, "error": repr(exc)})
                continue
            if any(value > 1.25 for value in weights.values()):
                add(findings, "error", "candidate_weight_too_aggressive", {"label": label, "field": weight_field, "weights": weights})
        if float(candidate["learning_rate"]) > 2e-7:
            add(findings, "error", "candidate_lr_too_high", candidate)
        if float(candidate["abort_relative_delta"]) > 0.003:
            add(findings, "error", "candidate_abort_delta_too_loose", candidate)

    return {"candidate_labels": labels, "base_env_keys": sorted(base_env)}


def check_scripts(findings: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for script in LOCAL_SCRIPTS:
        path = Path(script)
        item = {"path": script, "exists": path.exists(), "compile_ok": False}
        if path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
                item["compile_ok"] = True
            except Exception as exc:
                item["error"] = repr(exc)
        if not item["exists"] or not item["compile_ok"]:
            add(findings, "error", "local_script_invalid", item)
        items.append(item)
    return {"items": items}


def check_raw_scripts(findings: list[dict[str, Any]], skip_network: bool) -> dict[str, Any]:
    if skip_network:
        return {"skipped": True}
    items = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for script in REQUIRED_RAW_SCRIPTS:
            url = f"{BRANCH_BASE}/{script}"
            item: dict[str, Any] = {"path": script, "url": url}
            try:
                data = read_url(url)
                item["bytes"] = len(data)
                item["sha256"] = sha256_bytes(data)
                raw_path = tmp / Path(script).name
                raw_path.write_bytes(data)
                py_compile.compile(str(raw_path), doraise=True)
                item["compile_ok"] = True
            except Exception as exc:
                item["compile_ok"] = False
                item["error"] = repr(exc)
                add(findings, "error", "raw_script_invalid", item)
            items.append(item)
    return {"items": items}


def check_pack(findings: list[dict[str, Any]], skip_network: bool) -> dict[str, Any]:
    if skip_network:
        return {"skipped": True}
    data = read_url(PACK_URL)
    pack_sha = sha256_bytes(data)
    result: dict[str, Any] = {"url": PACK_URL, "bytes": len(data), "sha256": pack_sha, "member_checks": {}}
    if pack_sha not in APPROVED_PACK_SHA256:
        add(findings, "error", "unapproved_pack_sha", result)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pack.zip"
        path.write_bytes(data)
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for member, expected_sha in REQUIRED_PACK_MEMBERS.items():
                item = {"exists": member in names}
                if member in names and expected_sha:
                    item["sha256"] = sha256_bytes(archive.read(member))
                    item["expected_sha256"] = expected_sha
                    item["sha_ok"] = item["sha256"] == expected_sha
                result["member_checks"][member] = item
                if not item.get("exists") or item.get("sha_ok") is False:
                    add(findings, "error", "pack_member_invalid", {"member": member, "check": item})
    return result


def check_builder_determinism(cells: list[str], findings: list[dict[str, Any]]) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("build_v201c_multicandidate_notebook", "scripts/build_v201c_multicandidate_notebook.py")
    if spec is None or spec.loader is None:
        add(findings, "error", "builder_import_failed", "missing spec")
        return {"ok": False}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = module.build_v201c_notebook()
    built_text = json.dumps(built, indent=2)
    current_text = NOTEBOOK.read_text(encoding="utf-8")
    ok = built_text == current_text
    if not ok:
        add(findings, "error", "builder_output_differs_from_notebook", "run scripts/build_v201c_multicandidate_notebook.py")
    return {"ok": ok, "built_bytes": len(built_text), "notebook_bytes": len(current_text)}


def check_submission_zip_warning(findings: list[dict[str, Any]]) -> dict[str, Any]:
    local_zip = Path("submission.zip")
    if not local_zip.exists():
        return {"present": False}
    digest = sha256_path(local_zip)
    result = {"present": True, "path": str(local_zip), "sha256": digest, "expected_v194": EXPECTED["v194_zip_sha256"]}
    if digest != EXPECTED["v194_zip_sha256"]:
        add(findings, "warning", "local_submission_not_v194_rank19", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--skip-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: list[dict[str, Any]] = []
    notebook, cells, source = load_notebook(NOTEBOOK)
    checks = {
        "notebook_contract": check_notebook_contract(cells, source, findings),
        "training_cell": check_training_cell(cells, findings),
        "local_scripts": check_scripts(findings),
        "raw_scripts": check_raw_scripts(findings, skip_network=args.skip_network),
        "v198_pack": check_pack(findings, skip_network=args.skip_network),
        "builder_determinism": check_builder_determinism(cells, findings),
        "local_submission_zip": check_submission_zip_warning(findings),
        "metadata": {
            "notebook_name": notebook.get("metadata", {}).get("colab", {}).get("name"),
            "kernelspec": notebook.get("metadata", {}).get("kernelspec"),
        },
    }
    decision = "PASS" if not any(item["severity"] == "error" for item in findings) else "FAIL"
    report = {
        "generated_at": utc_now(),
        "decision": decision,
        "notebook": str(NOTEBOOK),
        "checks": checks,
        "findings": findings,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "findings": findings, "output": str(args.output_json)}, indent=2, sort_keys=True))
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
