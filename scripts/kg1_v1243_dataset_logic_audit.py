#!/usr/bin/env python3
"""Cheap V1243 dataset/logic audit for Colab before GPU spend.

This script is intentionally stdlib-only. It proves that the launch pack is
using the exact bit/equation/replay datasets expected by the V1243 manifest,
and it emits friendly logs that can be read while the notebook is running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "v1243_solver_to_lora_graft"
DEFAULT_MANIFEST = DEFAULT_ARTIFACT_DIR / "kg1_v1243_solver_to_lora_graft_manifest.json"
DEFAULT_ENV_PREVIEW = DEFAULT_ARTIFACT_DIR / "v1243_hf_env_preview.json"
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
REQUIRED_FAMILIES = {
    "bit_manipulation",
    "equation_transform",
    "gravity_constant",
    "numeral_system",
    "text_encryption",
    "unit_conversion",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid jsonl: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path.name}:{line_no}: jsonl row is not an object")
            rows.append(item)
    return rows


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unescape_latex_braces(value: object) -> str:
    return str(value).replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")


def terminal_boxed_payload(text: object) -> str | None:
    value = str(text or "").strip()
    marker = "\\boxed{"
    start = value.rfind(marker)
    if start < 0 or not value.endswith("}"):
        return None
    return value[start + len(marker) : -1].strip()


def answer_matches_boxed(answer: object, payload: object) -> bool:
    expected = normalize_text(answer)
    observed = normalize_text(unescape_latex_braces(payload))
    if expected.lower() == observed.lower():
        return True
    try:
        return math.isclose(float(expected), float(observed), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return False


def last_message(messages: Any, role: str) -> dict[str, Any] | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            return message
    return None


def score_class_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    family = str(item.get("family") or item.get("category") or "unknown")
    if family == "bit_manipulation":
        rule = (
            metadata.get("v1240_solver_rule_kind")
            or metadata.get("v1240_requested_rule_kind")
            or item.get("subcategory")
            or "unknown"
        )
        return f"bit_rule:{rule}"
    if family == "equation_transform":
        op_counts = metadata.get("v1240_op_counts")
        if isinstance(op_counts, dict) and op_counts:
            active = "+".join(f"{key}x{value}" for key, value in sorted(op_counts.items()) if value)
            return "equation_ops:" + (active or "none")
        transform_counts = metadata.get("v1240_transform_counts")
        if isinstance(transform_counts, dict) and transform_counts:
            active = "+".join(
                f"{key}x{value}" for key, value in sorted(transform_counts.items()) if value
            )
            return "equation_transforms:" + (active or "none")
    subcategory = item.get("subcategory") or metadata.get("v1243_phase_role") or "unknown"
    return f"{family}:{subcategory}"


def prompt_fingerprint(item: dict[str, Any]) -> str:
    prompt = item.get("prompt")
    if not prompt:
        user_message = last_message(item.get("messages"), "user")
        prompt = user_message.get("content") if user_message else ""
    return hashlib.sha256(normalize_text(prompt).encode("utf-8")).hexdigest()


def expected_from_env_preview(env_preview: dict[str, Any], phase: str) -> dict[str, Any]:
    env = env_preview.get(phase)
    if not isinstance(env, dict):
        raise KeyError(f"env preview missing phase {phase}")
    data_file = Path(str(env.get("DATA_FILE", ""))).name
    val_file = Path(str(env.get("VAL_FILE", ""))).name
    return {
        "data_file": data_file,
        "val_file": val_file,
        "train_sha256": str(env.get("EXPECTED_TRAIN_SHA256", "")),
        "val_sha256": str(env.get("EXPECTED_VAL_SHA256", "")),
        "min_train_examples": int(env.get("MIN_TRAIN_EXAMPLES", "0")),
        "min_val_examples": int(env.get("MIN_VAL_EXAMPLES", "0")),
    }


def row_weight(item: dict[str, Any]) -> float:
    metadata = item.get("metadata") or {}
    for value in [
        item.get("row_loss_weight"),
        item.get("loss_weight"),
        metadata.get("v1243_sampling_weight"),
        metadata.get("row_loss_weight"),
    ]:
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return 1.0


def audit_rows(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    family_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_role_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    weighted_loss_by_family: defaultdict[str, float] = defaultdict(float)
    terminal_boxed_rows = 0
    prompt_suffix_rows = 0
    assistant_prefix_rows = 0
    train_gate_rows = 0
    gpu_authorized_rows = 0
    submit_authorized_rows = 0
    ids: set[str] = set()
    duplicate_ids: list[str] = []

    for index, item in enumerate(rows, start=1):
        row_id = str(item.get("id") or "")
        if not row_id:
            errors.append(f"{label}:{index}: missing id")
        elif row_id in ids:
            duplicate_ids.append(row_id)
        ids.add(row_id)

        family = str(item.get("family") or item.get("category") or "unknown")
        metadata = item.get("metadata") or {}
        messages = item.get("messages")
        user_message = last_message(messages, "user")
        assistant_message = last_message(messages, "assistant")
        prompt = item.get("prompt") or (user_message.get("content") if user_message else "")
        answer = item.get("answer")
        assistant_text = assistant_message.get("content") if assistant_message else ""
        payload = terminal_boxed_payload(assistant_text)

        family_counts[family] += 1
        class_counts[score_class_key(item)] += 1
        source_role_counts[str(item.get("source_role") or "unknown")] += 1
        subcategory_counts[str(item.get("subcategory") or "unknown")] += 1
        weighted_loss_by_family[family] += row_weight(item)

        if family not in REQUIRED_FAMILIES:
            errors.append(f"{label}:{row_id}: unknown family {family!r}")
        if metadata.get("v1243_family") and metadata.get("v1243_family") != family:
            errors.append(f"{label}:{row_id}: metadata family mismatch")
        if not isinstance(messages, list) or not user_message or not assistant_message:
            errors.append(f"{label}:{row_id}: missing user/assistant messages")
        if PROMPT_SUFFIX in str(prompt):
            prompt_suffix_rows += 1
        else:
            errors.append(f"{label}:{row_id}: missing official prompt suffix")
        if str(assistant_text).startswith("</think>\n"):
            assistant_prefix_rows += 1
        else:
            errors.append(f"{label}:{row_id}: assistant target does not start with </think>")
        if str(assistant_text).count("\\boxed{") != 1:
            errors.append(f"{label}:{row_id}: assistant target must contain exactly one boxed answer")
        if payload is None:
            errors.append(f"{label}:{row_id}: assistant target is not terminal boxed")
        elif answer_matches_boxed(answer, payload):
            terminal_boxed_rows += 1
        else:
            errors.append(f"{label}:{row_id}: boxed payload does not match top-level answer")
        weights = item.get("target_answer_char_loss_weights") or metadata.get("target_answer_char_loss_weights")
        if not isinstance(weights, list) or not weights:
            errors.append(f"{label}:{row_id}: missing target answer char weights")
        elif payload is not None and len(weights) != len(str(answer)):
            warnings.append(
                f"{label}:{row_id}: char weight count {len(weights)} != answer length {len(str(answer))}"
            )
        if metadata.get("gate_rows_used_for_training") or metadata.get("full_gate_rows_used_for_training"):
            train_gate_rows += 1
        if metadata.get("v1243_gpu_authorized"):
            gpu_authorized_rows += 1
        if metadata.get("v1243_submit_authorized"):
            submit_authorized_rows += 1

    if duplicate_ids:
        errors.append(f"{label}: duplicate ids found: {duplicate_ids[:10]}")
    if train_gate_rows:
        errors.append(f"{label}: {train_gate_rows} rows mark gate/full-gate usage for training")
    if gpu_authorized_rows or submit_authorized_rows:
        errors.append(
            f"{label}: unexpected authorization flags gpu={gpu_authorized_rows} submit={submit_authorized_rows}"
        )

    report = {
        "rows": len(rows),
        "families": dict(sorted(family_counts.items())),
        "classes": dict(sorted(class_counts.items())),
        "source_roles": dict(sorted(source_role_counts.items())),
        "subcategories": dict(sorted(subcategory_counts.items())),
        "weighted_loss_by_family": {
            key: round(value, 6) for key, value in sorted(weighted_loss_by_family.items())
        },
        "prompt_suffix_rows": prompt_suffix_rows,
        "assistant_prefix_rows": assistant_prefix_rows,
        "terminal_boxed_answer_rows": terminal_boxed_rows,
        "duplicate_ids": len(duplicate_ids),
        "gate_rows_used_for_training": train_gate_rows,
        "gpu_authorized_rows": gpu_authorized_rows,
        "submit_authorized_rows": submit_authorized_rows,
    }
    return report, errors, warnings


def teach_card(stage: str, status: str, *, what: str, why: str, watch: str, data: list[tuple[str, Any]], next_action: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[KG1-TEACH][{stamp}][{stage}][{status}]", flush=True)
    print(f"  O que e: {what}", flush=True)
    print(f"  Por que importa: {why}", flush=True)
    print(f"  Como ler: {watch}", flush=True)
    print("  Numeros-chave:", flush=True)
    for key, value in data:
        print(f"    - {key}: {value}", flush=True)
    print(f"  Proxima acao: {next_action}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--env-preview", type=Path, default=DEFAULT_ENV_PREVIEW)
    parser.add_argument("--phase", choices=["all", "bit_specialist", "equation_specialist"], default="all")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_dir = args.artifact_dir.resolve()
    manifest = load_json(args.manifest.resolve())
    env_preview = load_json(args.env_preview.resolve())
    expected_phases = (
        ["bit_specialist", "equation_specialist"] if args.phase == "all" else [args.phase]
    )
    errors: list[str] = []
    warnings: list[str] = []
    file_reports: dict[str, Any] = {}
    train_prompt_hashes: dict[str, set[str]] = {}
    val_prompt_hashes: set[str] = set()

    print("=== KG1 V1243 DATASET LOGIC AUDIT START ===", flush=True)
    print(f"audit_time_utc = {utc_now()}", flush=True)
    print(f"artifact_dir = {artifact_dir}", flush=True)
    print(f"manifest = {args.manifest.resolve()}", flush=True)
    print(f"env_preview = {args.env_preview.resolve()}", flush=True)

    val_file_name = expected_from_env_preview(env_preview, "bit_specialist")["val_file"]
    val_path = artifact_dir / val_file_name
    val_rows = load_jsonl(val_path)
    val_sha = sha256_file(val_path)
    val_expected = expected_from_env_preview(env_preview, "bit_specialist")
    if val_sha != val_expected["val_sha256"]:
        errors.append(f"{val_file_name}: sha mismatch {val_sha} != {val_expected['val_sha256']}")
    if len(val_rows) < val_expected["min_val_examples"]:
        errors.append(f"{val_file_name}: row count {len(val_rows)} < {val_expected['min_val_examples']}")
    report, row_errors, row_warnings = audit_rows(val_rows, val_file_name)
    errors.extend(row_errors)
    warnings.extend(row_warnings)
    report["sha256"] = val_sha
    file_reports[val_file_name] = report
    val_prompt_hashes = {prompt_fingerprint(item) for item in val_rows}

    for phase in expected_phases:
        expected = expected_from_env_preview(env_preview, phase)
        path = artifact_dir / expected["data_file"]
        rows = load_jsonl(path)
        observed_sha = sha256_file(path)
        if observed_sha != expected["train_sha256"]:
            errors.append(f"{path.name}: sha mismatch {observed_sha} != {expected['train_sha256']}")
        if len(rows) < expected["min_train_examples"]:
            errors.append(f"{path.name}: row count {len(rows)} < {expected['min_train_examples']}")
        report, row_errors, row_warnings = audit_rows(rows, path.name)
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        report["sha256"] = observed_sha
        report["phase"] = phase
        file_reports[path.name] = report
        train_hashes = {prompt_fingerprint(item) for item in rows}
        train_prompt_hashes[phase] = train_hashes
        overlap = sorted(train_hashes & val_prompt_hashes)
        if overlap:
            errors.append(f"{path.name}: train/val prompt overlap count={len(overlap)}")

    full947_judge = (manifest.get("audit") or {}).get("full947_judge") or {}
    canonical_rows = full947_judge.get("canonical_full947_solution_rows")
    if canonical_rows != 947:
        warnings.append(f"manifest canonical_full947_solution_rows unexpected: {canonical_rows}")
    if full947_judge.get("train_full947_prompt_overlap") != 0:
        errors.append("manifest train_full947_prompt_overlap is not zero")
    if full947_judge.get("val_full947_prompt_overlap") != 0:
        errors.append("manifest val_full947_prompt_overlap is not zero")

    status = "PASS" if not errors else "FAIL"
    summary = {
        "schema_version": "kg1_v1243_dataset_logic_audit_v1",
        "status": status,
        "artifact_dir": str(artifact_dir),
        "phases": expected_phases,
        "files": file_reports,
        "errors": errors,
        "warnings": warnings,
        "loss_is_not_score": True,
        "score_claim_gate": "V1241 full947 raw_output comparison",
    }
    print("KG1_V1243_DATASET_AUDIT_JSON_BEGIN", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("KG1_V1243_DATASET_AUDIT_JSON_END", flush=True)
    print(
        "KG1_V1243_DATASET_AUDIT_STATUS="
        f"{status} files={len(file_reports)} errors={len(errors)} warnings={len(warnings)}",
        flush=True,
    )
    teach_card(
        "DATASET_AUDIT",
        "OK" if status == "PASS" else "STOP",
        what="Prova barata de que o job esta usando os datasets V1243 esperados.",
        why="Se hash, familia, classe, replay protegido ou boxed target estiverem errados, loss bom nao vira score.",
        watch="PASS significa que linhas, hashes, familias/classes, answers boxed e separacao train/val batem.",
        data=[
            ("status", status),
            ("files", len(file_reports)),
            ("errors", len(errors)),
            ("warnings", len(warnings)),
            ("val_rows", file_reports.get(val_file_name, {}).get("rows")),
            ("val_families", file_reports.get(val_file_name, {}).get("families")),
        ],
        next_action=(
            "Seguir para tokenizacao/model dry-run." if status == "PASS" else "Parar e corrigir o pacote antes de GPU."
        ),
    )
    print("=== KG1 V1243 DATASET LOGIC AUDIT END ===", flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
