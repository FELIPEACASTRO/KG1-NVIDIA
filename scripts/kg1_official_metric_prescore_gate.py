#!/usr/bin/env python3
"""Gate the KG1 pre-score script against the official Kaggle metric notebook."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_EXTRACTED = ROOT / "artifacts" / "official_metric_audit_20260605" / "nvidia_nemotron_metric_extracted.py"
OFFICIAL_NOTEBOOK = ROOT / "artifacts" / "official_metric_audit_20260605" / "nvidia-nemotron-metric.ipynb"
EXPECTED_OFFICIAL_NOTEBOOK_SHA256 = "d62302e4318de1dc9cf1103577b42b2d7f6234a9629b5198fdad800cefce24ed"
PRESCORE_SCRIPT = ROOT / "scripts" / "kg1_official_metric_prescore.py"
DEFAULT_REPORT = ROOT / "artifacts" / "official_metric_prescore_gate" / "report.json"
EXPECTED_VISIBLE_DEFAULTS = {
    "max_lora_rank": 32,
    "max_tokens": 7680,
    "top_p": 1.0,
    "temperature": 0.0,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prescore() -> ModuleType:
    spec = importlib.util.spec_from_file_location("kg1_official_metric_prescore", PRESCORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {PRESCORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_local_utils() -> ModuleType:
    return load_module_from_path("kg1_competition_utils", ROOT / "src" / "competition_utils.py")


def load_official_functions() -> dict[str, Any]:
    source = OFFICIAL_EXTRACTED.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"extract_final_answer", "verify"}
    ]
    if len(selected) != 2:
        raise RuntimeError("official extract did not contain exactly extract_final_answer and verify")
    module_ast = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math")]),
            ast.Import(names=[ast.alias(name="re")]),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module_ast)
    namespace: dict[str, Any] = {}
    exec(compile(module_ast, str(OFFICIAL_EXTRACTED), "exec"), namespace)
    return namespace


def assert_equal(name: str, got: Any, expected: Any, failures: list[dict[str, Any]]) -> None:
    if got != expected:
        failures.append({"case": name, "got": repr(got), "expected": repr(expected)})


def function_parity_tests(prescore: ModuleType, official: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    extract_cases = [
        ("none", None),
        ("multi_box_last_nonempty", r"a \boxed{wrong} b \boxed{right}"),
        ("multi_box_skip_empty", r"a \boxed{first} b \boxed{}"),
        ("all_empty_box", r"a \boxed{} b \boxed{}"),
        ("nested_latex", r"answer \boxed{\frac{1}{2}}"),
        ("literal_close_brace_symbol", r"answer \boxed{}52}"),
        ("unclosed_box", r"answer \boxed{abc"),
        ("fullwidth_colon", "Final answer\uff1a 42"),
        ("phrase_last", "Final answer: 1\nFinal answer: 2"),
        ("last_negative_decimal", "try -1.25 then -3.50"),
        ("last_line_string", "reasoning\nXLVII"),
        ("empty_text", ""),
    ]
    for name, text in extract_cases:
        assert_equal(
            f"extract:{name}",
            prescore.extract_final_answer(text),
            official["extract_final_answer"](text),
            failures,
        )

    verify_cases = [
        ("binary_exact", "10011000", "10011000"),
        ("binary_wrong", "10011000", "10011001"),
        ("binary_leading_zero_fail", "00011011", "11011"),
        ("binary_decimal_fail", "10", "10.0"),
        ("numeric_leading_zero_pass", "0271", "271"),
        ("numeric_tolerance_pass", "24.64", "24.6401"),
        ("numeric_tolerance_boundary", "200", "202"),
        ("numeric_abs_tolerance", "0", "0.000001"),
        ("roman_case_pass", "XLVII", "xlvii"),
        ("string_unicode_not_normalized", "é", "e\u0301"),
    ]
    for name, expected, prediction in verify_cases:
        assert_equal(
            f"verify:{name}",
            prescore.verify(expected, prediction),
            official["verify"](expected, prediction),
            failures,
        )
    return len(extract_cases) + len(verify_cases), failures


def local_utils_parity_tests(local_utils: ModuleType, official: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    extract_cases = [
        ("trailing_text_close_brace", r"\boxed{42} next text }"),
        ("multi_box_last", r"a \boxed{wrong} b \boxed{right}"),
        ("nested_latex", r"answer \boxed{\frac{1}{2}}"),
        ("literal_close_brace_symbol", r"answer \boxed{}52}"),
        ("unclosed_box", r"answer \boxed{abc"),
        ("fullwidth_colon", "Final answer\uff1a 42"),
        ("last_number", "try 10 then 42"),
        ("last_line", "reasoning\nXLVII"),
    ]
    for name, text in extract_cases:
        assert_equal(
            f"local_extract:{name}",
            local_utils.extract_final_answer(text),
            official["extract_final_answer"](text),
            failures,
        )
    verify_cases = [
        ("binary_decimal_fail", "10", "10.0"),
        ("binary_leading_zero_fail", "00011011", "11011"),
        ("numeric_pass", "24.64", "24.6401"),
        ("string_case_pass", "XLVII", "xlvii"),
    ]
    for name, expected, prediction in verify_cases:
        assert_equal(
            f"local_verify:{name}",
            local_utils.verify_answer(expected, prediction),
            official["verify"](expected, prediction),
            failures,
        )
    return len(extract_cases) + len(verify_cases), failures


def local_config_blockers(local_utils: ModuleType) -> list[str]:
    blockers: list[str] = []
    config = getattr(local_utils, "OFFICIAL_INFERENCE_CONFIG", {})
    for key, expected in EXPECTED_VISIBLE_DEFAULTS.items():
        observed = config.get(key)
        if observed != expected:
            blockers.append(f"official_inference_config_mismatch:{key}:{observed!r}!={expected!r}")
    return blockers


def prescore_defaults_blockers(prescore: ModuleType) -> list[str]:
    blockers: list[str] = []
    defaults = getattr(prescore, "OFFICIAL_VISIBLE_SCORE_DEFAULTS", {})
    for key, expected in EXPECTED_VISIBLE_DEFAULTS.items():
        observed = defaults.get(key)
        if observed != expected:
            blockers.append(f"prescore_visible_default_mismatch:{key}:{observed!r}!={expected!r}")
    return blockers


def unicode_integrity_tests() -> list[str]:
    blockers: list[str] = []
    for path in [
        OFFICIAL_EXTRACTED,
        PRESCORE_SCRIPT,
        ROOT / "src" / "competition_utils.py",
        ROOT / "artifacts" / "v284_official_gate_worktree" / "src" / "competition_utils.py",
    ]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "\uff1a" not in text and "\\uFF1A" not in text and "\\uff1a" not in text:
            blockers.append(f"missing_fullwidth_colon_u_ff1a:{path}")
        if "ï¼š" in text:
            blockers.append(f"mojibake_fullwidth_colon:{path}")
    return blockers


def score_predictions_fixture(prescore: ModuleType) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        solution = tmp / "solution.csv"
        predictions = tmp / "predictions.csv"
        family = tmp / "family.csv"
        out = tmp / "out"
        with solution.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
            writer.writeheader()
            writer.writerows(
                [
                    {"id": "b1", "answer": "10"},
                    {"id": "n1", "answer": "24.64"},
                    {"id": "s1", "answer": "XLVII"},
                    {"id": "e1", "answer": "}52"},
                ]
            )
        with predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "raw_output"])
            writer.writeheader()
            writer.writerows(
                [
                    {"id": "b1", "raw_output": r"reason \boxed{10}"},
                    {"id": "n1", "raw_output": "The final answer is: 24.6401"},
                    {"id": "s1", "raw_output": "reason\nxlvii"},
                    {"id": "e1", "raw_output": r"symbol \boxed{}52}"},
                ]
            )
        with family.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "family", "status"])
            writer.writeheader()
            writer.writerows(
                [
                    {"id": "b1", "family": "bit_manipulation", "status": "fixture"},
                    {"id": "n1", "family": "gravity", "status": "fixture"},
                    {"id": "s1", "family": "numeral", "status": "fixture"},
                    {"id": "e1", "family": "equation_transform", "status": "fixture"},
                ]
            )
        args = SimpleNamespace(
            solution_csv=solution,
            predictions=predictions,
            family_map=family,
            adapter_artifact=None,
            strict_submission_artifact=False,
            expected_zip_sha256=None,
            expected_adapter_config_sha256=None,
            expected_adapter_model_sha256=None,
            output_dir=out,
            id_col="id",
            answer_col="answer",
            raw_output_col="raw_output",
            prediction_col=None,
            allow_auto_column_detect=False,
            strict_row_match=True,
            min_score=1.0,
            baseline_score=0.99,
            require_score_above_baseline=True,
            min_family_correct=[
                "bit_manipulation=1",
                "equation_transform=1",
                "gravity=1",
                "numeral=1",
            ],
            min_family_accuracy=[
                "bit_manipulation=1.0",
                "equation_transform=1.0",
                "gravity=1.0",
                "numeral=1.0",
            ],
        )
        return prescore.score_predictions(args)


def duplicate_id_fixture(prescore: ModuleType) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        solution = tmp / "solution.csv"
        predictions = tmp / "predictions.csv"
        out = tmp / "out"
        solution.write_text("id,answer\nx,1\n", encoding="utf-8")
        predictions.write_text("id,raw_output\nx,\\boxed{1}\nx,\\boxed{1}\n", encoding="utf-8")
        args = SimpleNamespace(
            solution_csv=solution,
            predictions=predictions,
            family_map=None,
            adapter_artifact=None,
            strict_submission_artifact=False,
            expected_zip_sha256=None,
            expected_adapter_config_sha256=None,
            expected_adapter_model_sha256=None,
            output_dir=out,
            id_col="id",
            answer_col="answer",
            raw_output_col="raw_output",
            prediction_col=None,
            allow_auto_column_detect=False,
            strict_row_match=True,
            min_score=1.0,
            baseline_score=None,
            require_score_above_baseline=False,
            min_family_correct=[],
            min_family_accuracy=[],
        )
        return prescore.score_predictions(args)["metric_blockers"]


def main() -> int:
    blockers: list[str] = []
    if not OFFICIAL_EXTRACTED.exists():
        blockers.append(f"official_extract_missing:{OFFICIAL_EXTRACTED}")
    if not OFFICIAL_NOTEBOOK.exists():
        blockers.append(f"official_notebook_missing:{OFFICIAL_NOTEBOOK}")
    if not PRESCORE_SCRIPT.exists():
        blockers.append(f"prescore_script_missing:{PRESCORE_SCRIPT}")
    if blockers:
        report = {"decision": "fail", "blockers": blockers}
        DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    official_notebook_sha256 = sha256_file(OFFICIAL_NOTEBOOK)
    if official_notebook_sha256 != EXPECTED_OFFICIAL_NOTEBOOK_SHA256:
        blockers.append(
            "official_notebook_sha256_mismatch:"
            f"{official_notebook_sha256}!={EXPECTED_OFFICIAL_NOTEBOOK_SHA256}"
        )

    prescore = load_prescore()
    local_utils = load_local_utils()
    v284_utils_path = ROOT / "artifacts" / "v284_official_gate_worktree" / "src" / "competition_utils.py"
    v284_utils = load_module_from_path("kg1_v284_competition_utils", v284_utils_path) if v284_utils_path.exists() else None
    official = load_official_functions()
    case_count, parity_failures = function_parity_tests(prescore, official)
    local_case_count, local_parity_failures = local_utils_parity_tests(local_utils, official)
    v284_case_count = 0
    v284_parity_failures: list[dict[str, Any]] = []
    if v284_utils is not None:
        v284_case_count, v284_parity_failures = local_utils_parity_tests(v284_utils, official)
    blockers.extend(unicode_integrity_tests())
    if parity_failures:
        blockers.append(f"function_parity_failures:{len(parity_failures)}")
    if local_parity_failures:
        blockers.append(f"local_utils_parity_failures:{len(local_parity_failures)}")
    blockers.extend(prescore_defaults_blockers(prescore))
    blockers.extend(local_config_blockers(local_utils))
    if v284_utils is not None:
        if v284_parity_failures:
            blockers.append(f"v284_utils_parity_failures:{len(v284_parity_failures)}")
        blockers.extend(f"v284_{item}" for item in local_config_blockers(v284_utils))

    fixture_summary = score_predictions_fixture(prescore)
    if fixture_summary["decision"] != "pass" or not math.isclose(fixture_summary["score"], 1.0):
        blockers.append("score_predictions_fixture_failed")

    duplicate_blockers = duplicate_id_fixture(prescore)
    if not any(item.startswith("duplicate_prediction_ids:") for item in duplicate_blockers):
        blockers.append("duplicate_prediction_ids_not_blocked")

    report = {
        "decision": "pass" if not blockers else "fail",
        "blockers": blockers,
        "official_notebook_sha256": official_notebook_sha256,
        "expected_official_notebook_sha256": EXPECTED_OFFICIAL_NOTEBOOK_SHA256,
        "prescore_script_sha256": sha256_file(PRESCORE_SCRIPT),
        "official_extracted_sha256": sha256_file(OFFICIAL_EXTRACTED),
        "function_parity_cases": case_count,
        "function_parity_failures": parity_failures,
        "local_utils_parity_cases": local_case_count,
        "local_utils_parity_failures": local_parity_failures,
        "v284_utils_path": str(v284_utils_path) if v284_utils_path.exists() else "",
        "v284_utils_parity_cases": v284_case_count,
        "v284_utils_parity_failures": v284_parity_failures,
        "expected_visible_defaults": EXPECTED_VISIBLE_DEFAULTS,
        "prescore_visible_score_defaults": getattr(prescore, "OFFICIAL_VISIBLE_SCORE_DEFAULTS", {}),
        "local_official_inference_config": {
            key: getattr(local_utils, "OFFICIAL_INFERENCE_CONFIG", {}).get(key)
            for key in EXPECTED_VISIBLE_DEFAULTS
        },
        "v284_official_inference_config": {
            key: getattr(v284_utils, "OFFICIAL_INFERENCE_CONFIG", {}).get(key)
            for key in EXPECTED_VISIBLE_DEFAULTS
        } if v284_utils is not None else {},
        "fixture_score": fixture_summary["score"],
        "fixture_decision": fixture_summary["decision"],
        "duplicate_id_blockers": duplicate_blockers,
    }
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
