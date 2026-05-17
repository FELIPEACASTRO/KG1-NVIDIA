#!/usr/bin/env python3
"""V526 CPU objective dry-run for V523/V525 style SFT datasets.

This gate is intentionally model-free. It validates the objective mechanics
that can be checked before spending GPU:

- examples contain literal ``\\boxed{...}`` targets, not control characters;
- weak/full training flags are false;
- V286 offset-mask/tokenization gate already passed;
- V524 token-mass bias is removed structurally by ``example_mean``;
- the resulting row-normalized family mix is close to the reference no-loss
  gain mix enough to allow only a short smoke, not a long run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


BOXED_LITERAL_RE = re.compile(r"\\boxed\{")
CONTROL_ALLOWED = {9, 10, 13}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return str(row.get("completion", ""))


def row_has_forbidden_training_flag(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    for key in (
        "gate_rows_used_for_training",
        "weak_gate_rows_used_for_training",
        "full_gate_rows_used_for_training",
    ):
        if row.get(key) is True or metadata.get(key) is True:
            return True
    return False


def inspect_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    boxed_missing: list[str] = []
    answer_mismatch: list[str] = []
    control_char_rows: list[str] = []
    forbidden_training_flags: list[str] = []
    assistant_chars_by_family: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        row_id = str(row.get("id", f"row_{idx}"))
        family = str(row.get("family") or row.get("metadata", {}).get("family") or "")
        family_counts[family] += 1
        text = assistant_text(row)
        assistant_chars_by_family[family] += len(text)

        if "\b" in text:
            control_char_rows.append(row_id)
        else:
            for char in text:
                if ord(char) < 32 and ord(char) not in CONTROL_ALLOWED:
                    control_char_rows.append(row_id)
                    break

        if not BOXED_LITERAL_RE.search(text):
            boxed_missing.append(row_id)
        else:
            answer = str(row.get("answer", "")).strip()
            extracted = extract_final_answer(text)
            if answer and not verify_answer(answer, extracted):
                answer_mismatch.append(row_id)

        if row_has_forbidden_training_flag(row):
            forbidden_training_flags.append(row_id)

    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "assistant_chars_by_family": dict(sorted(assistant_chars_by_family.items())),
        "boxed_missing_count": len(boxed_missing),
        "boxed_missing_first10": boxed_missing[:10],
        "answer_mismatch_count": len(answer_mismatch),
        "answer_mismatch_first10": answer_mismatch[:10],
        "control_char_count": len(control_char_rows),
        "control_char_first10": control_char_rows[:10],
        "forbidden_training_flag_count": len(forbidden_training_flags),
        "forbidden_training_flag_first10": forbidden_training_flags[:10],
    }


def parse_row_loss_weight(row: dict[str, Any], *, require: bool) -> tuple[float, bool]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    raw_weight = metadata.get("loss_weight", row.get("loss_weight"))
    if raw_weight is None:
        if require:
            raise ValueError(f"row {row.get('id', '<missing>')} is missing metadata.loss_weight")
        return 1.0, False
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row.get('id', '<missing>')} has non-numeric metadata.loss_weight={raw_weight!r}"
        ) from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            f"row {row.get('id', '<missing>')} has invalid metadata.loss_weight={raw_weight!r}"
        )
    return weight, True


def weighted_family_counts(
    rows: list[dict[str, Any]],
    *,
    use_row_loss_weight: bool,
    require_row_loss_weight: bool,
) -> tuple[dict[str, float], dict[str, int]]:
    weighted: Counter[str] = Counter()
    explicit: Counter[str] = Counter()
    for row in rows:
        family = str(row.get("family") or row.get("metadata", {}).get("family") or "")
        weight = 1.0
        has_explicit_weight = False
        if use_row_loss_weight:
            weight, has_explicit_weight = parse_row_loss_weight(row, require=require_row_loss_weight)
        weighted[family] += weight
        if has_explicit_weight:
            explicit[family] += 1
    return {key: float(value) for key, value in weighted.items()}, dict(explicit)


def compute_objective_mix(
    v524_manifest: dict[str, Any],
    train_rows: list[dict[str, Any]],
    max_example_mean_target_delta: float,
    *,
    use_row_loss_weight: bool,
    require_row_loss_weight: bool,
) -> dict[str, Any]:
    mix = v524_manifest["v523_train_mix"]
    counts = mix["row_family_counts"]
    loss_mass = mix["loss_token_mass"]
    weighted_counts, explicit_weight_counts = weighted_family_counts(
        train_rows,
        use_row_loss_weight=use_row_loss_weight,
        require_row_loss_weight=require_row_loss_weight,
    )
    total_rows = sum(counts.values())
    total_weight = sum(weighted_counts.values())
    total_tokens = sum(loss_mass.values())
    bit_rows = counts.get("bit_manipulation", 0)
    eq_rows = counts.get("equation_transform", 0)
    bit_weight = weighted_counts.get("bit_manipulation", 0.0)
    eq_weight = weighted_counts.get("equation_transform", 0.0)
    bit_tokens = loss_mass.get("bit_manipulation", 0)
    eq_tokens = loss_mass.get("equation_transform", 0)

    reference = v524_manifest["reference_gain_mix"]["target_bit_share"]
    token_mean_bit_share = bit_tokens / total_tokens if total_tokens else math.nan
    physical_example_mean_bit_share = bit_rows / total_rows if total_rows else math.nan
    weighted_example_mean_bit_share = bit_weight / total_weight if total_weight else math.nan
    example_mean_bit_share = (
        weighted_example_mean_bit_share if use_row_loss_weight else physical_example_mean_bit_share
    )
    token_mean_bit_to_eq = bit_tokens / eq_tokens if eq_tokens else math.inf
    physical_example_mean_bit_to_eq = bit_rows / eq_rows if eq_rows else math.inf
    weighted_example_mean_bit_to_eq = bit_weight / eq_weight if eq_weight else math.inf
    example_mean_bit_to_eq = (
        weighted_example_mean_bit_to_eq if use_row_loss_weight else physical_example_mean_bit_to_eq
    )
    reference_bit_to_eq = reference / (1.0 - reference)
    example_delta = abs(example_mean_bit_share - reference)

    return {
        "use_row_loss_weight": bool(use_row_loss_weight),
        "require_row_loss_weight": bool(require_row_loss_weight),
        "reference_bit_share": round(reference, 6),
        "token_mean_bit_share": round(token_mean_bit_share, 6),
        "physical_example_mean_bit_share": round(physical_example_mean_bit_share, 6),
        "weighted_example_mean_bit_share": round(weighted_example_mean_bit_share, 6),
        "example_mean_bit_share": round(example_mean_bit_share, 6),
        "token_mean_bit_to_equation_ratio": round(token_mean_bit_to_eq, 6),
        "physical_example_mean_bit_to_equation_ratio": round(physical_example_mean_bit_to_eq, 6),
        "weighted_example_mean_bit_to_equation_ratio": round(weighted_example_mean_bit_to_eq, 6),
        "example_mean_bit_to_equation_ratio": round(example_mean_bit_to_eq, 6),
        "reference_bit_to_equation_ratio": round(reference_bit_to_eq, 6),
        "example_mean_delta_from_reference": round(example_delta, 6),
        "max_example_mean_target_delta": max_example_mean_target_delta,
        "example_mean_close_to_reference": example_delta <= max_example_mean_target_delta,
        "token_mean_is_dominated_by_bit": token_mean_bit_share >= 0.85,
        "row_family_counts": dict(sorted(counts.items())),
        "weighted_family_counts": {key: round(value, 6) for key, value in sorted(weighted_counts.items())},
        "explicit_weight_counts": dict(sorted(explicit_weight_counts.items())),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_manifest = read_json(args.dataset_manifest_json)
    tokenization_manifest = read_json(args.tokenization_manifest_json)
    quota_manifest = read_json(args.quota_manifest_json)

    outputs = dataset_manifest.get("outputs", {})
    train_path = Path(outputs["train_jsonl"])
    val_path = Path(outputs["val_jsonl"])
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)

    train_inspection = inspect_rows(train_rows)
    val_inspection = inspect_rows(val_rows)
    try:
        objective_mix = compute_objective_mix(
            quota_manifest,
            train_rows,
            max_example_mean_target_delta=args.max_example_mean_target_delta,
            use_row_loss_weight=args.use_row_loss_weight,
            require_row_loss_weight=args.require_row_loss_weight,
        )
        objective_mix_error = ""
    except ValueError as exc:
        objective_mix = {
            "use_row_loss_weight": bool(args.use_row_loss_weight),
            "require_row_loss_weight": bool(args.require_row_loss_weight),
            "example_mean_close_to_reference": False,
            "example_mean_delta_from_reference": math.inf,
            "error": str(exc),
        }
        objective_mix_error = str(exc)

    tokenization_status = tokenization_manifest.get("decision", {}).get("status")
    tokenization = tokenization_manifest.get("tokenization", {})
    train_tok = tokenization.get("train", {})
    val_tok = tokenization.get("validation", {})

    blockers: list[dict[str, Any]] = []

    def add_blocker(code: str, detail: str) -> None:
        blockers.append({"code": code, "detail": detail, "severity": "blocker"})

    if tokenization_status != "tokenization_gate_passed":
        add_blocker("tokenization_gate_not_passed", f"status={tokenization_status!r}")
    for split_name, stats in (("train", train_tok), ("validation", val_tok)):
        if stats.get("completion_tokens_dropped", 0) != 0:
            add_blocker(f"{split_name}_completion_tokens_dropped", str(stats.get("completion_tokens_dropped")))
        if stats.get("offset_masks", 0) != stats.get("rows", -1):
            add_blocker(f"{split_name}_offset_masks_incomplete", f"{stats.get('offset_masks')} != {stats.get('rows')}")
        if stats.get("prompt_truncated", 0) != 0:
            add_blocker(f"{split_name}_prompt_truncated", str(stats.get("prompt_truncated")))

    for split_name, inspection in (("train", train_inspection), ("validation", val_inspection)):
        if inspection["boxed_missing_count"]:
            add_blocker(f"{split_name}_boxed_missing", str(inspection["boxed_missing_first10"]))
        if inspection["answer_mismatch_count"]:
            add_blocker(f"{split_name}_answer_mismatch", str(inspection["answer_mismatch_first10"]))
        if inspection["control_char_count"]:
            add_blocker(f"{split_name}_control_chars", str(inspection["control_char_first10"]))
        if inspection["forbidden_training_flag_count"]:
            add_blocker(f"{split_name}_forbidden_training_flags", str(inspection["forbidden_training_flag_first10"]))

    if objective_mix_error:
        add_blocker("row_loss_weight_invalid", objective_mix_error)
    if not objective_mix["example_mean_close_to_reference"]:
        add_blocker(
            "example_mean_mix_far_from_reference",
            f"delta={objective_mix['example_mean_delta_from_reference']}",
        )

    decision = {
        "status": "example_mean_dry_run_passed" if not blockers else "blocked",
        "gpu_allowed": not blockers,
        "gpu_scope": "one_short_h200_smoke_only" if not blockers else "none",
        "reason": (
            "example_mean keeps the row-normalized family mix close to the V522 reference"
            if not blockers
            else "one or more pre-GPU objective/data checks failed"
        ),
        "next_action": (
            "Create a V523 example_mean smoke launcher with first-checkpoint ACC kill-switch"
            if not blockers
            else "Build V525 with shorter bit traces and rerun V286/V513/V524/V526"
        ),
    }

    return {
        "version": "V526",
        "schema_version": "kg1_v526_example_mean_objective_dry_run_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_manifest_json": str(args.dataset_manifest_json),
            "tokenization_manifest_json": str(args.tokenization_manifest_json),
            "quota_manifest_json": str(args.quota_manifest_json),
        },
        "config": {
            "loss_normalization_mode_required": "example_mean",
            "max_example_mean_target_delta": args.max_example_mean_target_delta,
            "use_row_loss_weight": bool(args.use_row_loss_weight),
            "require_row_loss_weight": bool(args.require_row_loss_weight),
        },
        "train_inspection": train_inspection,
        "validation_inspection": val_inspection,
        "objective_mix": objective_mix,
        "blockers": blockers,
        "decision": decision,
    }


def write_report(manifest: dict[str, Any], output_dir: Path) -> Path:
    report = output_dir / "KG1_V526_EXAMPLE_MEAN_OBJECTIVE_DRY_RUN.md"
    mix = manifest["objective_mix"]
    lines = [
        "# KG1 V526 Example Mean Objective Dry Run",
        "",
        f"generated_at_utc: {manifest['generated_at_utc']}",
        "",
        "## Decision",
        "",
        f"- status: `{manifest['decision']['status']}`",
        f"- gpu_allowed: `{manifest['decision']['gpu_allowed']}`",
        f"- scope: `{manifest['decision']['gpu_scope']}`",
        f"- reason: {manifest['decision']['reason']}",
        f"- next_action: {manifest['decision']['next_action']}",
        "",
        "## Objective Mix",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| use row loss weight | {mix.get('use_row_loss_weight')} |",
        f"| require row loss weight | {mix.get('require_row_loss_weight')} |",
        f"| reference bit share | {mix.get('reference_bit_share')} |",
        f"| token_mean bit share | {mix.get('token_mean_bit_share')} |",
        f"| physical example_mean bit share | {mix.get('physical_example_mean_bit_share')} |",
        f"| weighted example_mean bit share | {mix.get('weighted_example_mean_bit_share')} |",
        f"| selected example_mean bit share | {mix.get('example_mean_bit_share')} |",
        f"| token_mean bit/equation ratio | {mix.get('token_mean_bit_to_equation_ratio')} |",
        f"| physical example_mean bit/equation ratio | {mix.get('physical_example_mean_bit_to_equation_ratio')} |",
        f"| weighted example_mean bit/equation ratio | {mix.get('weighted_example_mean_bit_to_equation_ratio')} |",
        f"| selected example_mean bit/equation ratio | {mix.get('example_mean_bit_to_equation_ratio')} |",
        f"| reference bit/equation ratio | {mix.get('reference_bit_to_equation_ratio')} |",
        f"| example_mean delta from reference | {mix.get('example_mean_delta_from_reference')} |",
        "",
        "## Dataset Checks",
        "",
        f"- train boxed missing: `{manifest['train_inspection']['boxed_missing_count']}`",
        f"- train answer mismatch: `{manifest['train_inspection']['answer_mismatch_count']}`",
        f"- train control chars: `{manifest['train_inspection']['control_char_count']}`",
        f"- train forbidden training flags: `{manifest['train_inspection']['forbidden_training_flag_count']}`",
        f"- validation boxed missing: `{manifest['validation_inspection']['boxed_missing_count']}`",
        f"- validation answer mismatch: `{manifest['validation_inspection']['answer_mismatch_count']}`",
        f"- validation control chars: `{manifest['validation_inspection']['control_char_count']}`",
        f"- validation forbidden training flags: `{manifest['validation_inspection']['forbidden_training_flag_count']}`",
        "",
        "## Blockers",
        "",
    ]
    if manifest["blockers"]:
        for blocker in manifest["blockers"]:
            lines.append(f"- `{blocker['code']}`: {blocker['detail']}")
    else:
        lines.append("- none")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def self_test() -> None:
    sample = {
        "id": "x",
        "answer": "101",
        "family": "bit_manipulation",
        "messages": [{"role": "assistant", "content": "Final answer: \\boxed{101}"}],
        "metadata": {"gate_rows_used_for_training": False},
    }
    result = inspect_rows([sample])
    assert result["boxed_missing_count"] == 0
    assert result["answer_mismatch_count"] == 0
    assert result["control_char_count"] == 0
    assert result["forbidden_training_flag_count"] == 0
    assert parse_row_loss_weight({"metadata": {"loss_weight": 1.25}}, require=True) == (1.25, True)
    assert parse_row_loss_weight({}, require=False) == (1.0, False)
    for bad in (
        {"metadata": {"loss_weight": 0}},
        {"metadata": {"loss_weight": -1}},
        {"metadata": {"loss_weight": "nan"}},
        {"metadata": {"loss_weight": "bad"}},
    ):
        try:
            parse_row_loss_weight(bad, require=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_row_loss_weight should reject {bad!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest-json", type=Path)
    parser.add_argument("--tokenization-manifest-json", type=Path)
    parser.add_argument("--quota-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-example-mean-target-delta", type=float, default=0.08)
    parser.add_argument("--use-row-loss-weight", action="store_true")
    parser.add_argument("--require-row-loss-weight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("self_test_ok")
        return 0

    required = [
        args.dataset_manifest_json,
        args.tokenization_manifest_json,
        args.quota_manifest_json,
        args.output_dir,
    ]
    if any(value is None for value in required):
        parser.error("manifest paths and --output-dir are required unless --self-test is used")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_audit(args)
    report_path = write_report(manifest, args.output_dir)
    manifest["outputs"] = {
        "manifest_json": str(args.output_dir / "v526_example_mean_objective_dry_run_manifest.json"),
        "report_md": str(report_path),
    }
    Path(manifest["outputs"]["manifest_json"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("status =", manifest["decision"]["status"])
    print("gpu_allowed =", manifest["decision"]["gpu_allowed"])
    print("report_md =", report_path)
    print("manifest_json =", manifest["outputs"]["manifest_json"])
    return 0 if not manifest["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
