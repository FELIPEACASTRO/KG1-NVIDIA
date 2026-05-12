#!/usr/bin/env python3
"""Build V311 verifier-distillation seed preferences from V306 gains.

This is not a direct training dataset. It converts the V306 no-loss
postprocessor/verifier gains into an audited seed pack for the next synthetic
data builder. The 15 full-gate rows are useful as rule exemplars, but they must
not be blindly reused as train rows for a submission candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_CSV = REPO_ROOT / "artifacts/v306_solver_promotion_gate/20260512T1530Z/v306_v291_full_v306_solver_promotion_audit.csv"
DEFAULT_PREDICTIONS_CSV = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v311_verifier_distillation_preference_pack"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def box(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return "\\boxed{" + text + "}"


def canonical_completion(answer: str, proof: str) -> str:
    lines = [
        "Verification:",
        proof.strip() or "The verifier-selected rule is applied exactly.",
        "Final answer: " + box(answer),
    ]
    return "\n".join(lines)


def rejected_completion(answer: str, reason: str) -> str:
    return "\n".join(
        [
            "Rejected candidate:",
            reason.strip() or "This candidate failed the verifier.",
            "Final answer: " + box(answer),
        ]
    )


def format_negative_completion(answer: str, mode: str) -> str:
    if mode == "no_box":
        return "Final answer: " + str(answer)
    if mode == "multiple_boxes":
        return "First guess: " + box(answer) + "\nActually maybe " + box(str(answer)[::-1])
    if mode == "trailing_text":
        return "Final answer: " + box(answer) + "\nThis is the answer."
    raise ValueError("unknown negative mode: " + mode)


def build_rule_proof(audit_row: dict[str, str]) -> str:
    family = audit_row.get("family", "")
    if family == "bit_manipulation":
        return (
            "Use the verified bit postprocessor rule "
            f"{audit_row.get('v300_bit_rule', '')}. "
            f"The baseline candidate {audit_row.get('baseline_prediction', '')!r} failed, "
            f"and the verifier-selected full-byte candidate is {audit_row.get('combined_prediction', '')!r}."
        )
    if family == "equation_transform":
        return (
            "Use the verified numeric equation postprocessor rule "
            f"{audit_row.get('v274_eq_rule', '')}. "
            f"The baseline candidate {audit_row.get('baseline_prediction', '')!r} failed, "
            f"and the verifier-selected candidate is {audit_row.get('combined_prediction', '')!r}."
        )
    return (
        f"The baseline candidate {audit_row.get('baseline_prediction', '')!r} failed, "
        f"and the verifier-selected candidate is {audit_row.get('combined_prediction', '')!r}."
    )


def row_key(row: dict[str, str]) -> str:
    return str(row.get("id", "")).strip()


def validate_inputs(audit_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]]) -> dict[str, Any]:
    ids = [row_key(row) for row in audit_rows]
    pred_ids = [row_key(row) for row in prediction_rows]
    duplicate_audit = sorted({item for item in ids if ids.count(item) > 1})
    duplicate_predictions = sorted({item for item in pred_ids if pred_ids.count(item) > 1})
    if duplicate_audit:
        raise RuntimeError("duplicate ids in audit csv: " + json.dumps(duplicate_audit[:10]))
    if duplicate_predictions:
        raise RuntimeError("duplicate ids in predictions csv: " + json.dumps(duplicate_predictions[:10]))
    pred_by_id = {row_key(row): row for row in prediction_rows}
    missing = [item for item in ids if item not in pred_by_id]
    if missing:
        raise RuntimeError("audit ids missing from predictions csv: " + json.dumps(missing[:10]))
    return {
        "audit_rows": len(audit_rows),
        "prediction_rows": len(prediction_rows),
        "duplicate_audit_ids": 0,
        "duplicate_prediction_ids": 0,
    }


def build_pack(audit_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pred_by_id = {row_key(row): row for row in prediction_rows}
    preferences: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    blocked_losses = 0

    for audit in audit_rows:
        if audit.get("combined_loss") == "True":
            blocked_losses += 1
            continue
        if audit.get("combined_gain") != "True":
            continue
        if audit.get("baseline_correct") != "False" or audit.get("combined_correct") != "True":
            raise RuntimeError("gain row has inconsistent correctness flags: " + json.dumps(audit, ensure_ascii=False))

        item_id = row_key(audit)
        pred = pred_by_id[item_id]
        prompt = str(pred.get("prompt", "")).strip()
        if not prompt:
            raise RuntimeError("gain row missing prompt: " + item_id)
        family = audit.get("family", "")
        answer = audit.get("combined_prediction", "")
        baseline = audit.get("baseline_prediction", "")
        rule = audit.get("v300_bit_rule") if family == "bit_manipulation" else audit.get("v274_eq_rule", "")
        proof = build_rule_proof(audit)
        chosen = canonical_completion(answer, proof)
        rejected = rejected_completion(baseline, "The baseline candidate did not pass the verifier.")
        metadata = {
            "schema_version": "kg1_v311_verifier_distillation_seed_v1",
            "source": "v306_solver_promotion_gate",
            "source_id": item_id,
            "family": family,
            "rule": rule,
            "baseline_prediction": baseline,
            "combined_prediction": answer,
            "training_authorization": "blocked_seed_only_until_synthetic_out_of_gate_variants",
            "gate_rows_used_for_training": False,
            "weak_gate_rows_used_for_training": False,
        }
        preferences.append(
            {
                "id": "v311_pref_" + item_id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": metadata,
            }
        )
        for mode in ("no_box", "multiple_boxes", "trailing_text"):
            preferences.append(
                {
                    "id": f"v311_pref_{item_id}_{mode}",
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": format_negative_completion(answer, mode),
                    "metadata": {**metadata, "negative_type": mode},
                }
            )
        sft_rows.append(
            {
                "id": "v311_sft_" + item_id,
                "prompt": prompt,
                "answer": answer,
                "family": family,
                "messages": [
                    {
                        "role": "system",
                        "content": "Solve the KG1 puzzle. End with exactly one final answer in \\boxed{}.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                "metadata": metadata,
            }
        )
        family_counts[family] = family_counts.get(family, 0) + 1
        rule_counts[str(rule)] = rule_counts.get(str(rule), 0) + 1

    manifest = {
        "schema_version": "kg1_v311_verifier_distillation_preference_pack_v1",
        "seed_gain_rows": len(sft_rows),
        "preference_rows": len(preferences),
        "blocked_losses": blocked_losses,
        "family_counts": family_counts,
        "rule_counts": rule_counts,
        "training_authorization": "blocked_seed_only_until_synthetic_out_of_gate_variants",
        "required_next_gate": [
            "generate_synthetic_out_of_gate_variants",
            "run_real_tokenization_gate",
            "prove_no_weak_or_full_gate_rows_used_as_train",
            "weak_eval_before_any_full_eval_or_submit",
        ],
    }
    if len(sft_rows) != 15:
        raise RuntimeError(f"expected 15 V306 gain seed rows, got {len(sft_rows)}")
    return preferences, sft_rows, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v311_verifier_distillation_seed")
    args = parser.parse_args()

    print("=== V311 VERIFIER DISTILLATION PREFERENCE PACK START ===", flush=True)
    print("audit_csv =", args.audit_csv, flush=True)
    print("predictions_csv =", args.predictions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    if not args.audit_csv.exists():
        raise FileNotFoundError(args.audit_csv)
    if not args.predictions_csv.exists():
        raise FileNotFoundError(args.predictions_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = read_csv_rows(args.audit_csv)
    prediction_rows = read_csv_rows(args.predictions_csv)
    input_summary = validate_inputs(audit_rows, prediction_rows)
    preferences, sft_rows, manifest = build_pack(audit_rows, prediction_rows)

    preference_path = args.output_dir / f"{args.label}_preferences.jsonl"
    sft_path = args.output_dir / f"{args.label}_sft_seed.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_jsonl(preference_path, preferences)
    write_jsonl(sft_path, sft_rows)
    manifest.update(
        {
            "audit_csv": str(args.audit_csv),
            "audit_csv_sha256": sha256_file(args.audit_csv),
            "predictions_csv": str(args.predictions_csv),
            "predictions_csv_sha256": sha256_file(args.predictions_csv),
            "input_summary": input_summary,
            "outputs": {
                "preferences_jsonl": str(preference_path),
                "sft_seed_jsonl": str(sft_path),
                "manifest_json": str(manifest_path),
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("manifest =", json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V311 VERIFIER DISTILLATION PREFERENCE PACK END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
