#!/usr/bin/env python3
"""Validate KG1 Kaggle submission and adapter artifacts before upload.

This gate is deliberately conservative: it checks that submission IDs match the
Kaggle test set exactly, predictions are present, diagnostic CSVs are rejected,
and adapter ZIPs contain only deployable LoRA files rather than training state
or secrets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

BLOCKED_ZIP_NAMES = {
    "kaggle.json",
    ".env",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
}
BLOCKED_ZIP_PATTERNS = (
    re.compile(r"(^|/)(token|secret|credential|api[_-]?key)", re.IGNORECASE),
    re.compile(r"(^|/)checkpoint-\d+/", re.IGNORECASE),
)
REQUIRED_ADAPTER_CONFIG = "adapter_config.json"
REQUIRED_ADAPTER_WEIGHTS = {"adapter_model.safetensors", "adapter_model.bin"}
REQUIRED_NEMOTRON_ADAPTER_ROOT_ENTRIES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}
REQUIRED_NEMOTRON_TARGET_MODULES = {
    "down_proj",
    "in_proj",
    "k_proj",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
}
OPTIONAL_NEMOTRON_TARGET_MODULES = {"lm_head"}
EXPECTED_NEMOTRON_TARGET_MODULES = (
    REQUIRED_NEMOTRON_TARGET_MODULES | OPTIONAL_NEMOTRON_TARGET_MODULES
)
CANONICAL_086_ZIP_SHA256 = "a3b64b154a6690a58f2338ba1c405422eadc6e1c1357f662eecb187463dfdeee"
CANONICAL_086_ADAPTER_SHA256 = "559fd024f5ffcaff0caceddeaf25c3801009d6cabf247fc8dfccbfaf2addd916"
KNOWN_REGRESSION_ADAPTER_SHA256 = {
    # Drive folders labelled 086/086-A carried this weight hash, but audit
    # showed it is not the confirmed 0.86 Kaggle artifact.
    "e831c5263951c012328df14f940936809276462ffaa23a6733885945a47c7bd4",
    # V160: huikang/nemotron-adapter default/27 converted with the Asalhi path.
    # It passed structural checks but scored 0.53 on the public leaderboard.
    "b0eea0d4c9f14a1df51371284804563078cd038d1daa1d7de5e89ff365e0dd11",
    # V088: preflight passed but public score regressed from 0.86 to 0.84.
    "c8a4bc30c37227541cba946345a07547813265cb5cf4202315109396f7fd0236",
    # Huikang validation candidate: local 88.1%, preflight passed, public score 0.84.
    "83c83c865e5bb9cb153430e2bcf7e0f3fa4327cf0ab857060c7923156124d9a0",
    # V174: gate antigo aprovou, mas score publico foi 0.41. Causa auditada:
    # lm_head salvo em base_model.model.backbone.lm_head.*.
    "6946c653f7684b08a78042da6fa3a346212e7c5b4648496755399d061f98d563",
}
KNOWN_REGRESSION_ZIP_SHA256 = {
    "fe2d6c5a33445a09cca9628c36cb49d4e913d7dcf8921512cba76ef91c1a3c16",
    "ba2fb4eef9178083ef097dc692ebb7b89e6a6252a0c6e12dd2606950e4e21ba5",
    # V160 public score 0.53.
    "2197e123a50e78c9c2a28f718ba5f236ffe769d5b388f3a23d7efd529d02f9c0",
    # V088 public score 0.84 despite clean layout/preflight.
    "be106a972df530caeac6e241c6814a72b380b3ab8e471a228ba8922e81e89d92",
    # Huikang validation candidate public score 0.84 despite local 88.1%.
    "09af34c657b88e5c7f37ce412dc3340978eea8919e3460cc1cc82f8d16653ada",
    # V174 public score 0.41.
    "c9be81b56962131861f217045b0b8d83e1175800c9523571a5fbf6db7090fb44",
}
KNOWN_LM_HEAD_EXTRA_KEYS = {
    "base_model.model.lm_head.lora_A.weight",
    "base_model.model.lm_head.lora_B.weight",
}
CANONICAL_LM_HEAD_LORA_KEYS = {
    "base_model.model.lm_head.lora_A.weight",
    "base_model.model.lm_head.lora_B.weight",
}
FORBIDDEN_TRAINING_TENSOR_PREFIXES = (
    "base_model.model.backbone.lm_head.",
)
ALLOWED_KAGGLE_TENSOR_PREFIXES = (
    "base_model.model.model.",
    "base_model.model.backbone.layers.",
    "base_model.model.lm_head.",
)
ALLOWED_NON_LORA_TENSOR_KEYS = {
    "base_model.model.lm_head.base_layer.weight",
}
PREDICTION_COLUMNS = ("answer", "prediction")
DIAGNOSTIC_COLUMNS = {
    "prompt",
    "output",
    "predicted",
    "correct",
    "truth",
    "label",
    "score",
}
SUBMISSION_CSV_NAMES = {"submission.csv"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def classify_known_hash(zip_sha256: str, adapter_sha256: str | None = None) -> dict[str, Any]:
    reasons: list[str] = []
    labels: list[str] = []
    if zip_sha256 == CANONICAL_086_ZIP_SHA256:
        labels.append("canonical_086_zip")
    if zip_sha256 in KNOWN_REGRESSION_ZIP_SHA256:
        labels.append("known_regression_zip")
        reasons.append("known_regression_zip_hash")
    if adapter_sha256 == CANONICAL_086_ADAPTER_SHA256:
        labels.append("canonical_086_adapter")
    if adapter_sha256 in KNOWN_REGRESSION_ADAPTER_SHA256:
        labels.append("known_regression_adapter")
        reasons.append("known_regression_adapter_hash")
    if not labels:
        labels.append("unseen_or_unclassified")
    return {
        "labels": labels,
        "reasons": reasons,
        "zip_sha256": zip_sha256,
        "adapter_model_sha256": adapter_sha256,
    }


def read_csv_ids(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "id" not in fields:
        raise ValueError(f"{path} does not contain an 'id' column")
    ids = [str(row.get("id", "")).strip() for row in rows]
    return ids, rows, fields


def read_csv_text(label: str, text: str) -> tuple[list[str], list[dict[str, str]], list[str]]:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    fields = list(reader.fieldnames or [])
    if "id" not in fields:
        raise ValueError(f"{label} does not contain an 'id' column")
    ids = [str(row.get("id", "")).strip() for row in rows]
    return ids, rows, fields


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return [value for value, count in counts.items() if value and count > 1]


def is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def extract_boxed_payload(text: str | None) -> str | None:
    if not text:
        return None
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    brace_start = text.find("{", starts[-1])
    if brace_start < 0:
        return None
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char in "{}" and is_escaped(text, idx):
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:idx].strip()
    return None


def validate_submission_rows(
    test_csv: Path,
    sub_ids: list[str],
    sub_rows: list[dict[str, str]],
    sub_fields: list[str],
    source_label: str,
    source_sha256: str | None,
    require_boxed: bool,
) -> dict[str, Any]:
    test_ids, test_rows, _test_fields = read_csv_ids(test_csv)
    prediction_columns = [column for column in PREDICTION_COLUMNS if column in sub_fields]
    prediction_column = prediction_columns[0] if len(prediction_columns) == 1 else None

    reasons: list[str] = []
    if prediction_column is None:
        reasons.append("missing_prediction_or_answer_column")
    if len(prediction_columns) > 1:
        reasons.append("multiple_prediction_columns")

    diagnostic_columns = sorted(set(sub_fields) & DIAGNOSTIC_COLUMNS)
    if diagnostic_columns:
        reasons.append("diagnostic_columns_present")

    allowed_fields = {"id"}
    if prediction_column:
        allowed_fields.add(prediction_column)
    unexpected_fields = [field for field in sub_fields if field not in allowed_fields]
    if unexpected_fields:
        reasons.append("unexpected_submission_columns")

    test_id_set = set(test_ids)
    sub_id_set = set(sub_ids)
    missing_ids = sorted(test_id_set - sub_id_set)[:25]
    extra_ids = sorted(sub_id_set - test_id_set)[:25]
    test_duplicates = duplicate_values(test_ids)
    sub_duplicates = duplicate_values(sub_ids)
    blank_predictions = 0
    unboxed_predictions = 0
    malformed_boxed_predictions = 0

    if len(sub_rows) != len(test_rows):
        reasons.append("row_count_mismatch")
    if test_duplicates:
        reasons.append("duplicate_test_ids")
    if sub_duplicates:
        reasons.append("duplicate_submission_ids")
    if missing_ids:
        reasons.append("missing_submission_ids")
    if extra_ids:
        reasons.append("extra_submission_ids")

    if prediction_column:
        for row in sub_rows:
            prediction = str(row.get(prediction_column, "")).strip()
            if not prediction:
                blank_predictions += 1
            if require_boxed:
                if "\\boxed" not in prediction:
                    unboxed_predictions += 1
                elif extract_boxed_payload(prediction) is None:
                    malformed_boxed_predictions += 1
        if blank_predictions:
            reasons.append("blank_predictions")
        if require_boxed and unboxed_predictions:
            reasons.append("unboxed_predictions")
        if require_boxed and malformed_boxed_predictions:
            reasons.append("malformed_boxed_predictions")

    return {
        "path": source_label,
        "sha256": source_sha256,
        "rows": len(sub_rows),
        "test_rows": len(test_rows),
        "fields": sub_fields,
        "prediction_column": prediction_column,
        "diagnostic_columns": diagnostic_columns,
        "unexpected_columns": unexpected_fields,
        "blank_predictions": blank_predictions,
        "unboxed_predictions": unboxed_predictions,
        "malformed_boxed_predictions": malformed_boxed_predictions,
        "missing_ids_sample": missing_ids,
        "extra_ids_sample": extra_ids,
        "duplicate_submission_ids_sample": sub_duplicates[:25],
        "valid": not reasons,
        "reasons": reasons,
    }


def validate_submission_csv(test_csv: Path, submission_csv: Path, require_boxed: bool) -> dict[str, Any]:
    sub_ids, sub_rows, sub_fields = read_csv_ids(submission_csv)
    return validate_submission_rows(
        test_csv=test_csv,
        sub_ids=sub_ids,
        sub_rows=sub_rows,
        sub_fields=sub_fields,
        source_label=str(submission_csv),
        source_sha256=sha256_file(submission_csv),
        require_boxed=require_boxed,
    )


def zip_members_by_basename(names: list[str], basename: str) -> list[str]:
    return [name for name in names if Path(name).name == basename]


def suspicious_zip_path(name: str) -> bool:
    path = Path(name)
    return path.is_absolute() or ".." in path.parts


def blocked_zip_entries(names: list[str]) -> list[str]:
    blocked: list[str] = []
    for name in names:
        basename = Path(name).name
        if (
            suspicious_zip_path(name)
            or basename in BLOCKED_ZIP_NAMES
            or any(pattern.search(name) for pattern in BLOCKED_ZIP_PATTERNS)
        ):
            blocked.append(name)
    return blocked


def validate_submission_zip(test_csv: Path, submission_zip: Path, require_boxed: bool) -> dict[str, Any]:
    reasons: list[str] = []
    csv_report: dict[str, Any] | None = None
    try:
        with zipfile.ZipFile(submission_zip) as zf:
            names = [name.replace("\\", "/") for name in zf.namelist() if not name.endswith("/")]
            blocked_entries = blocked_zip_entries(names)
            csv_members = [name for name in names if Path(name).suffix.lower() == ".csv"]
            non_csv_entries = [name for name in names if Path(name).suffix.lower() != ".csv"]

            if blocked_entries:
                reasons.append("blocked_zip_entries")
            if not csv_members:
                reasons.append("missing_submission_csv")
            if len(csv_members) > 1:
                reasons.append("multiple_csv_files")
            if non_csv_entries:
                reasons.append("unexpected_non_csv_zip_entries")
            if csv_members and Path(csv_members[0]).name not in SUBMISSION_CSV_NAMES:
                reasons.append("unexpected_submission_csv_name")

            if len(csv_members) == 1:
                member = csv_members[0]
                payload = zf.read(member)
                member_sha256 = hashlib.sha256(payload).hexdigest()
                text = payload.decode("utf-8-sig")
                sub_ids, sub_rows, sub_fields = read_csv_text(f"{submission_zip}!{member}", text)
                csv_report = validate_submission_rows(
                    test_csv=test_csv,
                    sub_ids=sub_ids,
                    sub_rows=sub_rows,
                    sub_fields=sub_fields,
                    source_label=f"{submission_zip}!{member}",
                    source_sha256=member_sha256,
                    require_boxed=require_boxed,
                )
                reasons.extend(f"submission_csv:{reason}" for reason in csv_report["reasons"])
    except zipfile.BadZipFile:
        return {
            "path": str(submission_zip),
            "valid": False,
            "reasons": ["bad_zip_file"],
            "entries": [],
        }
    except (UnicodeDecodeError, ValueError, csv.Error) as exc:
        return {
            "path": str(submission_zip),
            "valid": False,
            "reasons": ["invalid_submission_csv"],
            "error": str(exc),
            "entries": [],
        }

    return {
        "path": str(submission_zip),
        "sha256": sha256_file(submission_zip),
        "entry_count": len(names),
        "entries_sample": names[:50],
        "blocked_entries_sample": blocked_entries[:25],
        "csv_report": csv_report,
        "valid": not reasons,
        "reasons": reasons,
    }


def load_safetensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("safetensors is required for adapter key validation") from exc

    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shapes[key] = tuple(int(value) for value in handle.get_tensor(key).shape)
    return shapes


def is_lora_tensor_name(key: str) -> bool:
    return (
        ".lora_A." in key
        or ".lora_B." in key
        or ".lora_embedding_A" in key
        or ".lora_embedding_B" in key
    )


def read_safetensor_header_from_zip(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    with zf.open(member) as raw:
        raw_len = raw.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{member} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_raw = raw.read(header_len)
    header = json.loads(header_raw.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def inspect_adapter_tensor_contract(
    header: dict[str, Any],
    config_report: dict[str, Any] | None,
) -> dict[str, Any]:
    keys = sorted(header)
    target_modules = set((config_report or {}).get("target_modules") or [])
    forbidden_training_keys = [
        key for key in keys
        if any(key.startswith(prefix) for prefix in FORBIDDEN_TRAINING_TENSOR_PREFIXES)
    ]
    bad_lm_head_namespace_keys = [
        key for key in keys
        if key.startswith("base_model.model.backbone.lm_head.")
    ]
    unexpected_prefix_keys = [
        key for key in keys
        if not any(key.startswith(prefix) for prefix in ALLOWED_KAGGLE_TENSOR_PREFIXES)
    ]
    non_lora_keys = [
        key for key in keys
        if not is_lora_tensor_name(key) and key not in ALLOWED_NON_LORA_TENSOR_KEYS
    ]
    missing_lm_head_lora_keys: list[str] = []
    if "lm_head" in target_modules:
        missing_lm_head_lora_keys = sorted(CANONICAL_LM_HEAD_LORA_KEYS - set(keys))

    reasons: list[str] = []
    if forbidden_training_keys:
        reasons.append("forbidden_training_backbone_namespace")
    if bad_lm_head_namespace_keys:
        reasons.append("bad_lm_head_namespace")
    if unexpected_prefix_keys:
        reasons.append("unexpected_tensor_prefix")
    if non_lora_keys:
        reasons.append("non_lora_tensors_present")
    if missing_lm_head_lora_keys:
        reasons.append("missing_canonical_lm_head_lora_keys")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "tensor_count": len(keys),
        "forbidden_training_key_count": len(forbidden_training_keys),
        "forbidden_training_key_sample": forbidden_training_keys[:25],
        "bad_lm_head_namespace_count": len(bad_lm_head_namespace_keys),
        "bad_lm_head_namespace_sample": bad_lm_head_namespace_keys[:25],
        "unexpected_prefix_count": len(unexpected_prefix_keys),
        "unexpected_prefix_sample": unexpected_prefix_keys[:25],
        "non_lora_tensor_count": len(non_lora_keys),
        "non_lora_tensor_sample": non_lora_keys[:25],
        "missing_lm_head_lora_keys": missing_lm_head_lora_keys,
    }


def compare_adapter_key_contract(
    adapter_tensor_path: Path,
    reference_adapter_dir: Path,
    strict_adapter_keys: bool,
) -> dict[str, Any]:
    reference_tensor_path = reference_adapter_dir / "adapter_model.safetensors"
    if not reference_tensor_path.exists():
        return {
            "valid": False,
            "reasons": ["missing_reference_adapter_model_safetensors"],
            "reference_adapter_dir": str(reference_adapter_dir),
        }

    adapter_shapes = load_safetensor_shapes(adapter_tensor_path)
    reference_shapes = load_safetensor_shapes(reference_tensor_path)
    adapter_keys = set(adapter_shapes)
    reference_keys = set(reference_shapes)
    allowed_extra = set() if strict_adapter_keys else KNOWN_LM_HEAD_EXTRA_KEYS

    missing_keys = sorted(reference_keys - adapter_keys)
    extra_keys = sorted(adapter_keys - reference_keys)
    disallowed_extra_keys = [key for key in extra_keys if key not in allowed_extra]
    shape_mismatches = [
        {
            "key": key,
            "adapter_shape": list(adapter_shapes[key]),
            "reference_shape": list(reference_shapes[key]),
        }
        for key in sorted(adapter_keys & reference_keys)
        if adapter_shapes[key] != reference_shapes[key]
    ]

    reasons: list[str] = []
    if missing_keys:
        reasons.append("missing_reference_keys")
    if disallowed_extra_keys:
        reasons.append("disallowed_extra_reference_keys")
    if shape_mismatches:
        reasons.append("shape_mismatches")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "reference_adapter_dir": str(reference_adapter_dir),
        "adapter_key_count": len(adapter_shapes),
        "reference_key_count": len(reference_shapes),
        "missing_keys_sample": missing_keys[:25],
        "extra_keys_sample": extra_keys[:25],
        "disallowed_extra_keys_sample": disallowed_extra_keys[:25],
        "shape_mismatches_sample": shape_mismatches[:25],
        "strict_adapter_keys": strict_adapter_keys,
    }


def validate_adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        reasons.append("missing_target_modules")
        target_modules = []
    else:
        target_modules = [str(item) for item in target_modules]
        if "gate_proj" in target_modules or "x_proj" in target_modules:
            reasons.append("unconverted_mamba_target_modules")
        if "in_proj" not in target_modules:
            reasons.append("missing_in_proj_target_module")
        missing_required = sorted(REQUIRED_NEMOTRON_TARGET_MODULES - set(target_modules))
        missing_optional = sorted(OPTIONAL_NEMOTRON_TARGET_MODULES - set(target_modules))
        if missing_required:
            reasons.append("missing_required_nemotron_target_modules")
        if missing_optional:
            warnings.append("missing_optional_nemotron_target_modules")

    rank = config.get("r")
    try:
        rank_value = int(rank)
    except (TypeError, ValueError):
        rank_value = None
        reasons.append("missing_or_invalid_lora_rank")
    if rank_value is not None and rank_value > 32:
        reasons.append("lora_rank_exceeds_vllm_limit")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "rank": rank_value,
        "target_modules": target_modules,
        "required_target_modules": sorted(REQUIRED_NEMOTRON_TARGET_MODULES),
        "optional_target_modules": sorted(OPTIONAL_NEMOTRON_TARGET_MODULES),
        "expected_target_modules": sorted(EXPECTED_NEMOTRON_TARGET_MODULES),
        "missing_required_target_modules": (
            sorted(REQUIRED_NEMOTRON_TARGET_MODULES - set(target_modules))
            if isinstance(target_modules, list)
            else sorted(REQUIRED_NEMOTRON_TARGET_MODULES)
        ),
        "missing_optional_target_modules": (
            sorted(OPTIONAL_NEMOTRON_TARGET_MODULES - set(target_modules))
            if isinstance(target_modules, list)
            else sorted(OPTIONAL_NEMOTRON_TARGET_MODULES)
        ),
    }


def validate_adapter_zip(
    adapter_zip: Path,
    reference_adapter_dir: Path | None = None,
    strict_adapter_keys: bool = False,
    adapter_layout_mode: str = "safe-root-only",
) -> dict[str, Any]:
    reasons: list[str] = []
    adapter_config_report: dict[str, Any] | None = None
    key_contract_report: dict[str, Any] | None = None
    tensor_contract_report: dict[str, Any] | None = None
    adapter_model_sha256: str | None = None
    try:
        with zipfile.ZipFile(adapter_zip) as zf:
            names = [name.replace("\\", "/") for name in zf.namelist() if not name.endswith("/")]
            root_only_expected_entries = set(names) == REQUIRED_NEMOTRON_ADAPTER_ROOT_ENTRIES

            config_members = zip_members_by_basename(names, REQUIRED_ADAPTER_CONFIG)
            if config_members:
                with zf.open(config_members[0]) as raw:
                    config = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
                adapter_config_report = validate_adapter_config(config)

            safetensor_members = zip_members_by_basename(names, "adapter_model.safetensors")
            if safetensor_members:
                with zf.open(safetensor_members[0]) as raw:
                    adapter_model_sha256 = sha256_stream(raw)
                header = read_safetensor_header_from_zip(zf, safetensor_members[0])
                tensor_contract_report = inspect_adapter_tensor_contract(
                    header,
                    adapter_config_report,
                )

            if reference_adapter_dir:
                if not safetensor_members:
                    key_contract_report = {
                        "valid": False,
                        "reasons": ["missing_safetensors_for_reference_validation"],
                        "reference_adapter_dir": str(reference_adapter_dir),
                    }
                else:
                    with tempfile.TemporaryDirectory(prefix="kg1_adapter_gate_") as tmp:
                        tmp_path = Path(tmp) / "adapter_model.safetensors"
                        with zf.open(safetensor_members[0]) as src, tmp_path.open("wb") as dst:
                            dst.write(src.read())
                        key_contract_report = compare_adapter_key_contract(
                            tmp_path,
                            reference_adapter_dir,
                            strict_adapter_keys,
                        )
    except zipfile.BadZipFile:
        return {
            "path": str(adapter_zip),
            "valid": False,
            "reasons": ["bad_zip_file"],
            "entries": [],
        }
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "path": str(adapter_zip),
            "valid": False,
            "reasons": ["invalid_adapter_config_json"],
            "entries": [],
        }
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        return {
            "path": str(adapter_zip),
            "valid": False,
            "reasons": ["adapter_key_validation_error"],
            "error": str(exc),
            "entries": [],
        }

    basenames = {Path(name).name for name in names}
    adapter_layout_mode = (adapter_layout_mode or "safe-root-only").strip().lower()
    if adapter_layout_mode not in {"safe-root-only", "kaggle-recursive"}:
        raise ValueError(f"Unsupported adapter_layout_mode: {adapter_layout_mode}")

    if adapter_layout_mode == "safe-root-only" and not root_only_expected_entries:
        reasons.append("adapter_zip_not_root_safetensors_only")
    if REQUIRED_ADAPTER_CONFIG not in basenames:
        reasons.append("missing_adapter_config")
    if not (REQUIRED_ADAPTER_WEIGHTS & basenames):
        reasons.append("missing_adapter_weights")

    blocked_entries = blocked_zip_entries(names)
    if blocked_entries:
        reasons.append("blocked_zip_entries")
    if adapter_config_report and not adapter_config_report["valid"]:
        reasons.extend(f"adapter_config:{reason}" for reason in adapter_config_report["reasons"])
    if tensor_contract_report and not tensor_contract_report["valid"]:
        reasons.extend(
            f"adapter_tensor_contract:{reason}"
            for reason in tensor_contract_report["reasons"]
        )
    if key_contract_report and not key_contract_report["valid"]:
        reasons.extend(f"adapter_key_contract:{reason}" for reason in key_contract_report["reasons"])
    zip_sha256 = sha256_file(adapter_zip)
    known_artifact = classify_known_hash(zip_sha256, adapter_model_sha256)
    reasons.extend(known_artifact["reasons"])

    return {
        "path": str(adapter_zip),
        "sha256": zip_sha256,
        "adapter_model_sha256": adapter_model_sha256,
        "known_artifact": known_artifact,
        "entry_count": len(names),
        "entries_sample": names[:50],
        "root_only_expected_entries": root_only_expected_entries,
        "adapter_layout_mode": adapter_layout_mode,
        "kaggle_recursive_compatible": REQUIRED_ADAPTER_CONFIG in basenames and bool(REQUIRED_ADAPTER_WEIGHTS & basenames),
        "expected_root_entries": sorted(REQUIRED_NEMOTRON_ADAPTER_ROOT_ENTRIES),
        "blocked_entries_sample": blocked_entries[:25],
        "adapter_config": adapter_config_report,
        "adapter_tensor_contract": tensor_contract_report,
        "adapter_key_contract": key_contract_report,
        "valid": not reasons,
        "reasons": reasons,
    }


def classify_zip_candidate(candidate: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(candidate) as zf:
            names = [name.replace("\\", "/") for name in zf.namelist() if not name.endswith("/")]
    except zipfile.BadZipFile:
        return {"artifact_type": "bad_zip", "reasons": ["bad_zip_file"], "entries_sample": []}

    basenames = {Path(name).name for name in names}
    has_adapter = REQUIRED_ADAPTER_CONFIG in basenames and bool(REQUIRED_ADAPTER_WEIGHTS & basenames)
    csv_members = [name for name in names if Path(name).suffix.lower() == ".csv"]
    reasons: list[str] = []
    artifact_type = "unknown_zip"
    if has_adapter and csv_members:
        artifact_type = "ambiguous_zip"
        reasons.append("zip_contains_adapter_and_csv")
    elif has_adapter:
        artifact_type = "adapter_zip"
    elif csv_members:
        artifact_type = "submission_zip"
    else:
        reasons.append("zip_contains_no_adapter_or_csv")

    return {
        "artifact_type": artifact_type,
        "reasons": reasons,
        "entries_sample": names[:50],
        "csv_members": csv_members[:25],
        "has_adapter": has_adapter,
    }


def validate_candidate_artifact(
    candidate: Path,
    test_csv: Path,
    require_boxed: bool,
    reference_adapter_dir: Path | None = None,
    strict_adapter_keys: bool = False,
    adapter_layout_mode: str = "safe-root-only",
) -> dict[str, Any]:
    suffix = candidate.suffix.lower()
    if suffix == ".csv":
        csv_report = validate_submission_csv(test_csv, candidate, require_boxed)
        return {
            "artifact_type": "submission_csv",
            "path": str(candidate),
            "valid": csv_report["valid"],
            "reasons": csv_report["reasons"],
            "submission_csv": csv_report,
        }
    if suffix == ".zip":
        zip_kind = classify_zip_candidate(candidate)
        artifact_type = zip_kind["artifact_type"]
        if artifact_type == "submission_zip":
            zip_report = validate_submission_zip(test_csv, candidate, require_boxed)
            return {
                "artifact_type": artifact_type,
                "path": str(candidate),
                "valid": zip_report["valid"],
                "reasons": zip_report["reasons"],
                "zip_classification": zip_kind,
                "submission_zip": zip_report,
            }
        if artifact_type == "adapter_zip":
            adapter_report = validate_adapter_zip(
                candidate,
                reference_adapter_dir=reference_adapter_dir,
                strict_adapter_keys=strict_adapter_keys,
                adapter_layout_mode=adapter_layout_mode,
            )
            return {
                "artifact_type": artifact_type,
                "path": str(candidate),
                "valid": adapter_report["valid"],
                "reasons": adapter_report["reasons"],
                "zip_classification": zip_kind,
                "adapter_zip": adapter_report,
            }
        return {
            "artifact_type": artifact_type,
            "path": str(candidate),
            "valid": False,
            "reasons": zip_kind["reasons"],
            "zip_classification": zip_kind,
        }
    return {
        "artifact_type": "unsupported_file_type",
        "path": str(candidate),
        "valid": False,
        "reasons": ["unsupported_file_type"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, help="Auto-classify and validate a downloaded CSV or ZIP artifact.")
    parser.add_argument("--submission-csv", type=Path)
    parser.add_argument("--submission-zip", type=Path)
    parser.add_argument("--adapter-zip", type=Path)
    parser.add_argument("--reference-adapter-dir", type=Path)
    parser.add_argument("--strict-adapter-keys", action="store_true")
    parser.add_argument(
        "--adapter-layout-mode",
        choices=["safe-root-only", "kaggle-recursive"],
        default="safe-root-only",
        help=(
            "safe-root-only enforces our unambiguous submit policy; "
            "kaggle-recursive tests basename compatibility without rejecting nested adapters"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "runs" / "kg1_submission_gate.json")
    parser.add_argument("--require-boxed", action="store_true")
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reasons: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "test_csv": str(args.test_csv),
        "candidate": None,
        "submission_csv": None,
        "submission_zip": None,
        "adapter_zip": None,
    }
    if not args.test_csv.exists():
        print(f"[ERROR] test CSV not found: {args.test_csv}", file=sys.stderr)
        return 2

    if args.candidate:
        if not args.candidate.exists():
            print(f"[ERROR] candidate artifact not found: {args.candidate}", file=sys.stderr)
            return 2
        candidate_report = validate_candidate_artifact(
            args.candidate,
            test_csv=args.test_csv,
            require_boxed=args.require_boxed,
            reference_adapter_dir=args.reference_adapter_dir,
            strict_adapter_keys=args.strict_adapter_keys,
            adapter_layout_mode=args.adapter_layout_mode,
        )
        report["candidate"] = candidate_report
        reasons.extend(f"candidate:{reason}" for reason in candidate_report["reasons"])

    if args.submission_csv:
        if not args.submission_csv.exists():
            print(f"[ERROR] submission CSV not found: {args.submission_csv}", file=sys.stderr)
            return 2
        csv_report = validate_submission_csv(args.test_csv, args.submission_csv, args.require_boxed)
        report["submission_csv"] = csv_report
        reasons.extend(f"submission_csv:{reason}" for reason in csv_report["reasons"])

    if args.submission_zip:
        if not args.submission_zip.exists():
            print(f"[ERROR] submission ZIP not found: {args.submission_zip}", file=sys.stderr)
            return 2
        submission_zip_report = validate_submission_zip(args.test_csv, args.submission_zip, args.require_boxed)
        report["submission_zip"] = submission_zip_report
        reasons.extend(f"submission_zip:{reason}" for reason in submission_zip_report["reasons"])

    if args.adapter_zip:
        if not args.adapter_zip.exists():
            print(f"[ERROR] adapter ZIP not found: {args.adapter_zip}", file=sys.stderr)
            return 2
        zip_report = validate_adapter_zip(
            args.adapter_zip,
            reference_adapter_dir=args.reference_adapter_dir,
            strict_adapter_keys=args.strict_adapter_keys,
            adapter_layout_mode=args.adapter_layout_mode,
        )
        report["adapter_zip"] = zip_report
        reasons.extend(f"adapter_zip:{reason}" for reason in zip_report["reasons"])

    if not args.candidate and not args.submission_csv and not args.submission_zip and not args.adapter_zip:
        reasons.append("no_submission_or_adapter_checked")

    report["decision"] = {
        "promotion_ready": not reasons,
        "reasons": reasons,
        "require_boxed": args.require_boxed,
        "strict_adapter_keys": args.strict_adapter_keys,
        "adapter_layout_mode": args.adapter_layout_mode,
    }
    write_json(args.output_json, report)
    print(f"promotion_ready: {report['decision']['promotion_ready']}")
    if report.get("candidate"):
        print(f"candidate_type: {report['candidate']['artifact_type']}")
    print(f"report: {args.output_json}")
    if args.fail_on_block and reasons:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
