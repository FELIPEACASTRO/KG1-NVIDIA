#!/usr/bin/env python3
"""KG1 V1243 solver-to-LoRA GRAFT dataset builder.

This is a CPU-only preparation step. It does not train, launch paid GPU jobs,
package adapters, or submit to Kaggle.

GRAFT means:
  - Gate-verified rows only.
  - Replay protected skills.
  - Answer-payload focused targets.
  - Family-specialist transfer first.
  - Tight raw-output gates before promotion.

The script converts V1240 solver-verified weak-family rows into phase-specific
LoRA-ready JSONL packs and writes the exact validation/gate plan needed before
any GPU spend.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    extract_closed_boxed_answers,
    extract_final_answer,
    verify_answer,
)


DEFAULT_V1240_DIR = ROOT / "artifacts" / "v1240_solver_verified_weak_curriculum_gate"
DEFAULT_V1241_DIR = ROOT / "artifacts" / "v1241_bit_equation_real_transfer_gate"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "v1243_solver_to_lora_graft"
CANONICAL_FULL947_SOLUTION = (
    ROOT
    / "artifacts"
    / "v284_official_gate_worktree"
    / "artifacts"
    / "v1088_unicode_dataset_contract_audit"
    / "hf_cli_download"
    / "runtime_artifacts"
    / "v276_full_eval_bridge"
    / "v276-full947-bridge-20260511T1245Z"
    / "official_train_seed42_stratified10_val.csv"
)
CANONICAL_FULL947_SOLUTION_SHA256 = "84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935"

BIT_FAMILY = "bit_manipulation"
EQUATION_FAMILY = "equation_transform"
PROTECTED_FAMILIES = {
    "gravity_constant",
    "numeral_system",
    "text_encryption",
    "unit_conversion",
}
INIT_ADAPTER_REPO = "felipesp1983/kg1-recovered-v291-v290-checkpoint6-submit086"
INIT_ADAPTER_REVISION = "f4134a6d223249d27be2f1c5d94ed59d118d1ce5"
INIT_ADAPTER_CONFIG_SHA256 = "a3d74c5a52ce75f71a8406222d877b9760ea18a40a772bcf407686c8ea19f11d"
INIT_ADAPTER_WEIGHTS_SHA256 = "0a7b6144231d9358ae73a5e57d8778b32be1520fa47e3041414b3e025aaa1aa1"
FULL_ADAPTER_TARGET_MODULES = "down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj"
TRAINABLE_DELTA_MODULES = "down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj"
PROTECTED_REPLAY_WEIGHT = 2.0

EXPECTED_V1240_TRAIN_COUNTS = {
    BIT_FAMILY: 540,
    EQUATION_FAMILY: 360,
    "gravity_constant": 46,
    "numeral_system": 46,
    "text_encryption": 46,
    "unit_conversion": 46,
}
EXPECTED_V1240_VAL_COUNTS = {
    BIT_FAMILY: 90,
    EQUATION_FAMILY: 60,
    "gravity_constant": 5,
    "numeral_system": 5,
    "text_encryption": 5,
    "unit_conversion": 5,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ps_quote(value: str | Path) -> str:
    """Quote a path/argument for the PowerShell runbook commands."""
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def first_message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def assistant_content(row: dict[str, Any]) -> str:
    return first_message_content(row, "assistant")


def prompt_content(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or first_message_content(row, "user"))


def row_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or "").strip()


def rough_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = prompt_content(row)
    assistant = assistant_content(row)
    answer = row_answer(row)
    family = str(row.get("family") or "")
    closed = extract_closed_boxed_answers(assistant)
    extracted = extract_final_answer(assistant)
    prompt_suffix_count = prompt.count(PROMPT_SUFFIX.strip())
    return {
        "id": str(row.get("id") or ""),
        "family": family,
        "ok": bool(
            answer
            and prompt
            and assistant
            and prompt_suffix_count == 1
            and len(closed) == 1
            and assistant.count("\\boxed{") == 1
            and assistant.rstrip().endswith("}")
            and verify_answer(answer, extracted)
            and (family != BIT_FAMILY or bool(re.fullmatch(r"[01]{8}", answer)))
        ),
        "answer": answer,
        "extracted": extracted,
        "closed_boxed_count": len(closed),
        "boxed_marker_count": assistant.count("\\boxed{"),
        "prompt_suffix_count": prompt_suffix_count,
        "prompt_sha256": sha256_text(prompt.strip()),
        "assistant_sha256": sha256_text(assistant.strip()),
        "answer_len": len(answer),
        "prompt_rough_tokens": rough_token_count(prompt),
        "assistant_rough_tokens": rough_token_count(assistant),
        "has_control_chars": any((ord(char) < 32 and char not in "\n\r\t") for char in prompt + assistant),
    }


def assert_counts(name: str, rows: list[dict[str, Any]], expected: dict[str, int]) -> None:
    counts = Counter(str(row.get("family") or "") for row in rows)
    if dict(counts) != expected:
        raise ValueError(f"{name} family counts mismatch: expected={expected} observed={dict(counts)}")


def assert_all_rows_ok(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audits = [audit_row(row) for row in rows]
    bad = [item for item in audits if not item["ok"] or item["has_control_chars"]]
    if bad:
        raise ValueError(f"{name} has {len(bad)} invalid rows; sample={bad[:5]}")
    return audits


def assert_no_prompt_overlap(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_hashes = {sha256_text(prompt_content(row).strip()) for row in train_rows}
    val_hashes = {sha256_text(prompt_content(row).strip()) for row in val_rows}
    overlap = sorted(train_hashes & val_hashes)
    if overlap:
        raise ValueError(f"train/val prompt overlap detected: {overlap[:5]}")


def audit_canonical_full947_overlap(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not CANONICAL_FULL947_SOLUTION.exists():
        raise FileNotFoundError(f"canonical full947 solution missing: {CANONICAL_FULL947_SOLUTION}")
    actual_sha = sha256_file(CANONICAL_FULL947_SOLUTION)
    if actual_sha != CANONICAL_FULL947_SOLUTION_SHA256:
        raise ValueError(
            "canonical full947 solution sha256 mismatch: "
            f"{actual_sha} != {CANONICAL_FULL947_SOLUTION_SHA256}"
        )
    with CANONICAL_FULL947_SOLUTION.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"id", "prompt", "answer", "family"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"canonical full947 solution missing columns: {missing}")
        full947_rows = [dict(row) for row in reader]
    family_counts = Counter(str(row.get("family") or "") for row in full947_rows)
    full947_hashes = {sha256_text(str(row.get("prompt") or "").strip()) for row in full947_rows}
    train_hashes = {sha256_text(prompt_content(row).strip()) for row in train_rows}
    val_hashes = {sha256_text(prompt_content(row).strip()) for row in val_rows}
    train_overlap = sorted(train_hashes & full947_hashes)
    val_overlap = sorted(val_hashes & full947_hashes)
    if train_overlap or val_overlap:
        raise ValueError(
            "V1243 prompt overlap with canonical full947 judge: "
            f"train={len(train_overlap)} val={len(val_overlap)}"
        )
    return {
        "canonical_full947_solution_csv": str(CANONICAL_FULL947_SOLUTION),
        "canonical_full947_solution_sha256": actual_sha,
        "canonical_full947_solution_rows": len(full947_rows),
        "canonical_full947_solution_prompt_hashes": len(full947_hashes),
        "canonical_full947_solution_family_counts": dict(sorted(family_counts.items())),
        "train_full947_prompt_overlap": len(train_overlap),
        "val_full947_prompt_overlap": len(val_overlap),
        "leak_check_pass": True,
    }


def tag_row(row: dict[str, Any], *, phase: str, phase_role: str, sampling_weight: float) -> dict[str, Any]:
    tagged = copy.deepcopy(row)
    metadata = dict(tagged.get("metadata") or {})
    family = str(tagged.get("family") or "")
    metadata.update(
        {
            "kg1_objective_version": "v1243_graft",
            "v1243_algorithm": "GRAFT",
            "v1243_phase": phase,
            "v1243_phase_role": phase_role,
            "v1243_target_encoding": "short_close_think_terminal_boxed_answer",
            "v1243_loss_contract": "completion_only_with_boxed_payload_priority",
            "v1243_gpu_authorized": False,
            "v1243_submit_authorized": False,
            "v1243_expected_raw_output_gate": "V1241",
            "v1243_family": family,
            "v1243_sampling_weight": sampling_weight,
            "row_loss_weight": sampling_weight,
            "loss_weight": sampling_weight,
        }
    )
    tagged["metadata"] = metadata
    tagged["source_role"] = f"v1243_{phase_role}"
    tagged["subcategory"] = f"v1243_{phase_role}"
    tagged["row_loss_weight"] = sampling_weight
    tagged["loss_weight"] = sampling_weight
    return tagged


def select_protected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("family") or "") in PROTECTED_FAMILIES]


def phase_rows(train_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bit = [row for row in train_rows if str(row.get("family") or "") == BIT_FAMILY]
    equation = [row for row in train_rows if str(row.get("family") or "") == EQUATION_FAMILY]
    protected = select_protected(train_rows)

    bit_phase = [tag_row(row, phase="bit_specialist", phase_role="bit_solver_verified", sampling_weight=1.35) for row in bit]
    bit_phase += [
        tag_row(row, phase="bit_specialist", phase_role="protected_replay", sampling_weight=PROTECTED_REPLAY_WEIGHT)
        for row in protected
    ]

    equation_phase = [
        tag_row(row, phase="equation_specialist", phase_role="equation_solver_verified", sampling_weight=1.65)
        for row in equation
    ]
    equation_phase += [
        tag_row(row, phase="equation_specialist", phase_role="protected_replay", sampling_weight=PROTECTED_REPLAY_WEIGHT)
        for row in protected
    ]

    protected_phase = [
        tag_row(row, phase="protected_replay", phase_role="protected_anchor", sampling_weight=PROTECTED_REPLAY_WEIGHT)
        for row in protected
    ]

    consolidation = [
        tag_row(row, phase="micro_consolidation", phase_role="bit_solver_verified", sampling_weight=1.05)
        for row in bit
    ]
    consolidation += [
        tag_row(row, phase="micro_consolidation", phase_role="equation_solver_verified", sampling_weight=1.20)
        for row in equation
    ]
    consolidation += [
        tag_row(row, phase="micro_consolidation", phase_role="protected_replay", sampling_weight=PROTECTED_REPLAY_WEIGHT)
        for row in protected
    ]

    return {
        "bit_specialist": bit_phase,
        "equation_specialist": equation_phase,
        "protected_replay": protected_phase,
        "micro_consolidation": consolidation,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family") or "") for row in rows)
    role_counts = Counter(str(row.get("source_role") or "") for row in rows)
    weight_by_family: Counter[str] = Counter()
    for row in rows:
        raw_weight = row.get("row_loss_weight")
        if raw_weight in (None, ""):
            raw_weight = row.get("loss_weight")
        if raw_weight in (None, ""):
            raw_weight = 1.0
        weight_by_family[str(row.get("family") or "")] += float(raw_weight)
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_role_counts": dict(sorted(role_counts.items())),
        "weight_by_family": {key: round(value, 6) for key, value in sorted(weight_by_family.items())},
    }


def build_readme(manifest: dict[str, Any]) -> str:
    outputs = manifest["outputs"]
    gate_commands = manifest["gate_commands"]
    return "\n".join(
        [
            "# KG1 V1243 Solver-to-LoRA GRAFT",
            "",
            f"Generated UTC: `{manifest['generated_at_utc']}`",
            "",
            "## Verdict",
            "",
            f"- Decision: `{manifest['decision']}`",
            "- This is CPU-only data preparation. It does not authorize GPU, package, or Kaggle submission.",
            "",
            "## Algorithm",
            "",
            "`GRAFT` = Gate-verified Replay Answer-Focused Transfer.",
            "",
            "The method builds family-specialist LoRA training packs from V1240 solver-verified rows:",
            "",
            "- bit specialist: bit rows plus protected replay.",
            "- equation specialist: equation rows plus protected replay.",
            "- micro consolidation: bit + equation + protected replay, only after specialists pass gates.",
            "- validation: V1240 val170 held out for raw-output gate checks.",
            "",
            "Targets stay short and score-facing: one terminal boxed answer. The intended trainer",
            "contract is completion-only loss with priority on the final boxed payload. The generated",
            "HF preview locks `TOKENIZE_ONLY_DRY_RUN=1`, `REQUIRE_OFFSET_MASK=1`,",
            "`REQUIRE_BOXED_PAYLOAD_WEIGHT=1`, `BOXED_PAYLOAD_LOSS_WEIGHT=5.0`,",
            "the runtime score-contract indicator in hard-fail mode, and score-proxy",
            "evaluation logs for the boxed answer tail. It also enables `SCORE_TRAJECTORY_CHECK=1`,",
            "which emits `KG1_SCORE_TRAJECTORY_STATUS` so loss is never used as the only",
            "directional signal toward `>=0.89`. Human-friendly `KG1-TEACH` logs stay",
            "enabled for real-time job monitoring.",
            "",
            "## Outputs",
            "",
            f"- Bit specialist train: `{outputs['bit_specialist_train_jsonl']}`",
            f"- Equation specialist train: `{outputs['equation_specialist_train_jsonl']}`",
            f"- Protected replay train: `{outputs['protected_replay_train_jsonl']}`",
            f"- Micro consolidation train: `{outputs['micro_consolidation_train_jsonl']}`",
            f"- Val170 holdout: `{outputs['val170_jsonl']}`",
            f"- HF env preview: `{outputs['hf_env_preview_json']}`",
            "",
            "## Trainer Contract",
            "",
            "- The first validation run is tokenize-only and cannot upload to HF.",
            "- Dataset hashes and minimum row counts are pinned in the env preview.",
            f"- The full 086 adapter is loaded with `LORA_TARGET_MODULES={FULL_ADAPTER_TARGET_MODULES}`.",
            f"- The graft delta is restricted with `TRAINABLE_LORA_MODULES={TRAINABLE_DELTA_MODULES}`.",
            "- Row-level sampling weights must be preserved through tokenization before weighted replacement sampling.",
            "- The trainer must boost final boxed-payload tokens before any real GPU run.",
            "- Run `python scripts/kg1_v1243_graft_trainer_contract_gate.py` after regeneration.",
            "",
            "## Mandatory Gate Commands",
            "",
            "Run these only after real generation CSVs exist. They require raw_output columns.",
            f"The canonical full947 solution CSV is `{CANONICAL_FULL947_SOLUTION}`",
            f"with expected sha256 `{CANONICAL_FULL947_SOLUTION_SHA256}`.",
            "The builder hard-fails if V1243 train or val170 prompts overlap this full947 judge.",
            "Before comparing any candidate, generate two 086/V291 raw_output CSVs:",
            "one with the natural prompt and one with the same answer-only",
            "prompt family used by the candidate. The natural probe is diagnostic; the answer-only probe",
            "must pass strict-clean identity before any paid candidate train or full947 comparison.",
            "The objective remains `>=0.89`: full947_089 requires `843/947`, i.e. `+20` over the 086 baseline.",
            "",
            "```powershell",
            gate_commands["full947_086_baseline_readiness"],
            gate_commands["baseline_identity_probe_086_natural"],
            gate_commands["baseline_identity_probe_086_answer_only"],
            gate_commands["tiny"],
            gate_commands["val170"],
            gate_commands["full947_089"],
            "```",
            "",
            "## Do Not Do",
            "",
            "- Do not use this artifact as score proof.",
            "- Do not launch paid GPU until dry-run/tokenization/objective gates pass.",
            "- Do not submit unless V1241 full947 passes with raw outputs.",
            "- Do not train on FASE5/s140/fase5_mix.",
            "",
        ]
    )


def build_hf_env_preview(output_dir: Path) -> dict[str, Any]:
    bit_train = output_dir / "v1243_bit_specialist_train.jsonl"
    equation_train = output_dir / "v1243_equation_specialist_train.jsonl"
    micro_train = output_dir / "v1243_micro_consolidation_train.jsonl"
    val170 = output_dir / "v1243_val170.jsonl"

    common = {
        "DATA_REPO": "local-v1243-solver-to-lora-graft",
        "DRY_RUN_VALIDATE_ONLY": "1",
        "TOKENIZE_ONLY_DRY_RUN": "1",
        "UPLOAD_TO_HF": "0",
        "SAMPLING_MODE": "weighted_replacement",
        "BOXED_PAYLOAD_LOSS_WEIGHT": "5.0",
        "REQUIRE_BOXED_PAYLOAD_WEIGHT": "1",
        "REQUIRE_OFFSET_MASK": "1",
        "SCORE_CONTRACT_RUNTIME_CHECK": "1",
        "REQUIRE_SCORE_CONTRACT_PASS": "1",
        "SCORE_CONTRACT_TARGET_ACCURACY": "0.89",
        "SCORE_CONTRACT_FULL_ROWS": "947",
        "SCORE_CONTRACT_BASELINE_CORRECT": "823",
        "SCORE_CONTRACT_EXPECTED_TARGET_MODULES": FULL_ADAPTER_TARGET_MODULES,
        "SCORE_CONTRACT_EXPECTED_TRAINABLE_MODULES": TRAINABLE_DELTA_MODULES,
        "SCORE_CONTRACT_REQUIRE_TRAINABLE_FILTER": "1",
        "SCORE_CONTRACT_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "SCORE_PROXY_EVAL_CHECK": "1",
        "SCORE_PROXY_EVAL_MAX_EXAMPLES": "170",
        "SCORE_TRAJECTORY_CHECK": "1",
        "REQUIRE_SCORE_TRAJECTORY_PASS": "0",
        "REQUIRE_SCORE_TRAJECTORY_FINAL_ONLY": "0",
        "SCORE_TRAJECTORY_MIN_WEAK_EXACT_DELTA": "0.0",
        "SCORE_TRAJECTORY_MAX_PROTECTED_EXACT_DROP": "0.0",
        "SCORE_TRAJECTORY_MAX_OVERALL_EXACT_DROP": "0.0",
        "SCORE_TRAJECTORY_MAX_BOXED_LOSS_REGRESSION": "0.0",
        "BASELINE_EVAL_BEFORE_TRAIN": "1",
        "ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA": "-1.0",
        "REQUIRE_FINAL_EVAL_LTE_BASELINE": "1",
        "REQUIRE_FINAL_SCORE_PROXY_NON_REGRESSION": "1",
        "MAX_FINAL_EVAL_REGRESSION": "0.0",
        "MAX_FINAL_BOXED_TAIL_LOSS_REGRESSION": "0.0",
        "MAX_FINAL_BOXED_TAIL_TOKEN_ACCURACY_DROP": "0.0",
        "MAX_FINAL_BOXED_TAIL_EXACT_RATE_DROP": "0.0",
        "REQUIRE_INIT_ADAPTER": "1",
        "REQUIRE_INIT_ADAPTER_REVISION": "1",
        "INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "INIT_ADAPTER_REVISION": INIT_ADAPTER_REVISION,
        "EXPECTED_INIT_ADAPTER_CONFIG_SHA256": INIT_ADAPTER_CONFIG_SHA256,
        "EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256": INIT_ADAPTER_WEIGHTS_SHA256,
        "INIT_ADAPTER_LOAD_MODE": "manual",
        "PEFT_MANUAL_LOAD_METHOD": "direct",
        "EVAL_MAX_EXAMPLES": "170",
        "FRIENDLY_REALTIME_LOGS": "1",
        "FRIENDLY_LOG_SCORE_HINTS": "1",
        "MIN_VAL_EXAMPLES": "170",
        "MIN_TOKENIZED_VAL_EXAMPLES": "170",
        "EXPECTED_VAL_SHA256": sha256_file(val170),
        "VAL_FILE": val170.name,
        "LORA_R": "32",
        "LORA_ALPHA": "32",
        "LORA_DROPOUT": "0.0",
        "EVAL_EVERY_STEPS": "10",
        "SAVE_EVERY_STEPS": "10",
        "MAX_LENGTH": "2048",
        "LORA_TARGET_MODULES": FULL_ADAPTER_TARGET_MODULES,
        "TRAINABLE_LORA_MODULES": TRAINABLE_DELTA_MODULES,
    }
    return {
        "bit_specialist": {
            **common,
            "DATA_FILE": bit_train.name,
            "EXPECTED_TRAIN_SHA256": sha256_file(bit_train),
            "MIN_TRAIN_EXAMPLES": "724",
            "MIN_TOKENIZED_TRAIN_EXAMPLES": "724",
            "LEARNING_RATE": "0.000003",
            "FINAL_LEARNING_RATE": "0.000001",
            "MAX_STEPS": "60",
            "NOTES": "Tokenize-only validate first. Real train requires explicit gate update and user decision.",
        },
        "equation_specialist": {
            **common,
            "DATA_FILE": equation_train.name,
            "EXPECTED_TRAIN_SHA256": sha256_file(equation_train),
            "MIN_TRAIN_EXAMPLES": "544",
            "MIN_TOKENIZED_TRAIN_EXAMPLES": "544",
            "LEARNING_RATE": "0.000002",
            "FINAL_LEARNING_RATE": "0.0000008",
            "MAX_STEPS": "50",
            "NOTES": "Only run after bit specialist passes raw-output gates, or from baseline.",
        },
        "micro_consolidation": {
            **common,
            "DATA_FILE": micro_train.name,
            "EXPECTED_TRAIN_SHA256": sha256_file(micro_train),
            "MIN_TRAIN_EXAMPLES": "1084",
            "MIN_TOKENIZED_TRAIN_EXAMPLES": "1084",
            "LEARNING_RATE": "0.00000075",
            "FINAL_LEARNING_RATE": "0.00000020",
            "MAX_STEPS": "20",
            "EVAL_EVERY_STEPS": "2",
            "SAVE_EVERY_STEPS": "2",
            "LOG_EVERY_STEPS": "1",
            "NOTES": (
                "Final sprint micro consolidation: bit + equation + protected replay. "
                "Run model dry-run first, then real train with final-only score trajectory hard guard."
            ),
        },
    }


def build_manifest(
    *,
    output_dir: Path,
    v1240_dir: Path,
    v1241_dir: Path,
    train_path: Path,
    val_path: Path,
    v1240_manifest_path: Path,
    audits: dict[str, list[dict[str, Any]]],
    phases: dict[str, list[dict[str, Any]]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    outputs = {
        "bit_specialist_train_jsonl": str(output_dir / "v1243_bit_specialist_train.jsonl"),
        "equation_specialist_train_jsonl": str(output_dir / "v1243_equation_specialist_train.jsonl"),
        "protected_replay_train_jsonl": str(output_dir / "v1243_protected_replay_train.jsonl"),
        "micro_consolidation_train_jsonl": str(output_dir / "v1243_micro_consolidation_train.jsonl"),
        "val170_jsonl": str(output_dir / "v1243_val170.jsonl"),
        "hf_env_preview_json": str(output_dir / "v1243_hf_env_preview.json"),
        "readme": str(output_dir / "V1243_SOLVER_TO_LORA_GRAFT.md"),
    }
    gate = ROOT / "scripts" / "kg1_v1241_bit_equation_transfer_gate.py"
    readiness_gate = ROOT / "scripts" / "kg1_v1241_full947_baseline_readiness_gate.py"
    v1241_solution = v1241_dir / "v1241_v1240_val170_solution.csv"
    full947_solution = CANONICAL_FULL947_SOLUTION
    baseline_placeholder = "path\\to\\baseline_predictions_with_raw_output.csv"
    natural_086_placeholder = "path\\to\\086_natural_full947_raw_output.csv"
    answer_only_086_placeholder = "path\\to\\086_answer_only_full947_raw_output.csv"
    candidate_placeholder = "path\\to\\candidate_predictions_with_raw_output.csv"
    full947_leak_audit = audit_canonical_full947_overlap(phases["micro_consolidation"], val_rows)
    gate_commands = {
        "full947_086_baseline_readiness": (
            f"python {ps_quote(readiness_gate)} --solution-csv {ps_quote(full947_solution)} "
            f"--natural-baseline-predictions {ps_quote(natural_086_placeholder)} "
            f"--answer-only-baseline-predictions {ps_quote(answer_only_086_placeholder)}"
        ),
        "baseline_identity_probe_086_natural": (
            f"python {ps_quote(gate)} --baseline-identity-probe --profile full947_089 "
            f"--solution-csv {ps_quote(full947_solution)} "
            f"--baseline-predictions {ps_quote(natural_086_placeholder)}"
        ),
        "baseline_identity_probe_086_answer_only": (
            f"python {ps_quote(gate)} --baseline-identity-probe --profile full947_089 "
            f"--solution-csv {ps_quote(full947_solution)} "
            f"--baseline-predictions {ps_quote(answer_only_086_placeholder)}"
        ),
        "tiny": (
            f"python {ps_quote(gate)} --profile tiny --solution-csv "
            f"{ps_quote(v1241_dir / 'v1241_tiny_bit_equation_probe_solution.csv')} "
            f"--baseline-predictions {ps_quote(baseline_placeholder)} "
            f"--candidate-predictions {ps_quote(candidate_placeholder)}"
        ),
        "val170": (
            f"python {ps_quote(gate)} --profile val170 --solution-csv {ps_quote(v1241_solution)} "
            f"--baseline-predictions {ps_quote(baseline_placeholder)} "
            f"--candidate-predictions {ps_quote(candidate_placeholder)}"
        ),
        "full947_089": (
            f"python {ps_quote(gate)} --profile full947_089 --solution-csv {ps_quote(full947_solution)} "
            f"--baseline-predictions {ps_quote(answer_only_086_placeholder)} "
            f"--candidate-predictions {ps_quote(candidate_placeholder)}"
        ),
    }
    return {
        "schema_version": "kg1_v1243_solver_to_lora_graft_v1",
        "generated_at_utc": utc_now(),
        "decision": "pass_v1243_cpu_dataset_graft_no_gpu_no_submit",
        "algorithm": {
            "name": "GRAFT",
            "expanded_name": "Gate-verified Replay Answer-Focused Transfer",
            "purpose": "Convert solver-verified bit/equation rows into family-specialist LoRA-ready packs.",
            "objective": (
                "Minimize completion loss on short score-facing boxed targets while using protected replay "
                "to preserve solved families. Promotion is based only on raw-output V1241 gates."
            ),
            "adapter_contract": {
                "init_adapter_repo": INIT_ADAPTER_REPO,
                "init_adapter_revision": INIT_ADAPTER_REVISION,
                "init_adapter_config_sha256": INIT_ADAPTER_CONFIG_SHA256,
                "init_adapter_weights_sha256": INIT_ADAPTER_WEIGHTS_SHA256,
                "lora_r": 32,
                "lora_alpha": 32,
                "active_target_modules": FULL_ADAPTER_TARGET_MODULES,
                "trainable_delta_modules": TRAINABLE_DELTA_MODULES,
                "protected_replay_weight": PROTECTED_REPLAY_WEIGHT,
                "rationale": (
                    "Load the full submit086 adapter shape, then freeze non-delta LoRA modules so bit/equation "
                    "GRAFT training does not silently drop lm_head/out_proj adapter tensors."
                ),
            },
            "not_authorized": ["paid_gpu_launch", "adapter_package", "kaggle_submit", "score_claim"],
        },
        "inputs": {
            "v1240_dir": str(v1240_dir),
            "v1241_dir": str(v1241_dir),
            "v1240_train_jsonl": str(train_path),
            "v1240_train_sha256": sha256_file(train_path),
            "v1240_val_jsonl": str(val_path),
            "v1240_val_sha256": sha256_file(val_path),
            "v1240_manifest": str(v1240_manifest_path),
            "v1240_manifest_sha256": sha256_file(v1240_manifest_path),
            "canonical_full947_solution_csv": str(CANONICAL_FULL947_SOLUTION),
            "canonical_full947_solution_sha256": (
                sha256_file(CANONICAL_FULL947_SOLUTION) if CANONICAL_FULL947_SOLUTION.exists() else ""
            ),
            "expected_canonical_full947_solution_sha256": CANONICAL_FULL947_SOLUTION_SHA256,
        },
        "outputs": outputs,
        "summaries": {
            "bit_specialist": summarize_rows(phases["bit_specialist"]),
            "equation_specialist": summarize_rows(phases["equation_specialist"]),
            "protected_replay": summarize_rows(phases["protected_replay"]),
            "micro_consolidation": summarize_rows(phases["micro_consolidation"]),
            "val170": summarize_rows(val_rows),
        },
        "audit": {
            "train_rows_ok": len(audits["train"]),
            "val_rows_ok": len(audits["val"]),
            "train_prompt_hashes": len({item["prompt_sha256"] for item in audits["train"]}),
            "val_prompt_hashes": len({item["prompt_sha256"] for item in audits["val"]}),
            "train_val_prompt_overlap": 0,
            "full947_judge": full947_leak_audit,
        },
        "hf_env_preview": build_hf_env_preview(output_dir),
        "gate_commands": gate_commands,
        "promotion_rules": {
            "first_probe": "V1241 tiny then val170 with raw_output; no regressions.",
            "score_089": "V1241 full947_089 requires >=843/947, +20 total, +1 bit, +1 equation, zero regressions.",
            "score_090": "V1241 full947_090 requires >=853/947, +30 total, +1 bit, +1 equation, zero regressions.",
        },
        "quarantine": [
            "FASE5/s140/fase5_mix",
            "rows missing top-level answer",
            "rows without exactly one closed boxed answer",
            "public-metric-only gains",
            "symbolic equation brace/backslash rows unless strict extraction proof passes",
            "any candidate without raw_output",
        ],
    }


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    v1240_dir = args.v1240_dir.resolve()
    v1241_dir = args.v1241_dir.resolve()
    train_path = v1240_dir / "v1240_solver_verified_train.jsonl"
    val_path = v1240_dir / "v1240_solver_verified_val.jsonl"
    v1240_manifest_path = v1240_dir / "kg1_v1240_solver_verified_weak_curriculum_gate.json"

    for path in (train_path, val_path, v1240_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not v1241_dir.exists():
        raise FileNotFoundError(v1241_dir)

    print("[v1243] START", flush=True)
    print(f"[v1243] output_dir={output_dir}", flush=True)
    print(f"[v1243] v1240_dir={v1240_dir}", flush=True)
    print(f"[v1243] v1241_dir={v1241_dir}", flush=True)

    v1240_manifest = load_json(v1240_manifest_path)
    if v1240_manifest.get("decision") != "pass_v1240_solver_verified_weak_curriculum_no_gpu_no_submit":
        raise ValueError(f"unexpected V1240 decision: {v1240_manifest.get('decision')}")

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    assert_counts("v1240_train", train_rows, EXPECTED_V1240_TRAIN_COUNTS)
    assert_counts("v1240_val", val_rows, EXPECTED_V1240_VAL_COUNTS)
    assert_no_prompt_overlap(train_rows, val_rows)

    train_audit = assert_all_rows_ok("v1240_train", train_rows)
    val_audit = assert_all_rows_ok("v1240_val", val_rows)
    phases = phase_rows(train_rows)

    for name, rows in phases.items():
        assert_all_rows_ok(name, rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "v1243_bit_specialist_train.jsonl", phases["bit_specialist"])
    write_jsonl(output_dir / "v1243_equation_specialist_train.jsonl", phases["equation_specialist"])
    write_jsonl(output_dir / "v1243_protected_replay_train.jsonl", phases["protected_replay"])
    write_jsonl(output_dir / "v1243_micro_consolidation_train.jsonl", phases["micro_consolidation"])
    val_holdout_rows = [
        tag_row(row, phase="val170", phase_role="holdout", sampling_weight=0.0)
        for row in val_rows
    ]
    write_jsonl(output_dir / "v1243_val170.jsonl", val_holdout_rows)

    manifest = build_manifest(
        output_dir=output_dir,
        v1240_dir=v1240_dir,
        v1241_dir=v1241_dir,
        train_path=train_path,
        val_path=val_path,
        v1240_manifest_path=v1240_manifest_path,
        audits={"train": train_audit, "val": val_audit},
        phases=phases,
        val_rows=val_holdout_rows,
    )
    write_json(output_dir / "v1243_hf_env_preview.json", manifest["hf_env_preview"])
    write_json(output_dir / "kg1_v1243_solver_to_lora_graft_manifest.json", manifest)
    (output_dir / "V1243_SOLVER_TO_LORA_GRAFT.md").write_text(build_readme(manifest), encoding="utf-8")

    print("[v1243] summaries=", json.dumps(manifest["summaries"], sort_keys=True), flush=True)
    print(f"[v1243] wrote {output_dir}", flush=True)
    print("[v1243] END", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1240-dir", type=Path, default=DEFAULT_V1240_DIR)
    parser.add_argument("--v1241-dir", type=Path, default=DEFAULT_V1241_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
