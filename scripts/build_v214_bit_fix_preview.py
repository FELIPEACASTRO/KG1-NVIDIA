#!/usr/bin/env python
"""Build V214 bit-manipulation fix preview and replay manifests.

The 24 V194 validation errors corrected by the deterministic bit solver are
kept as forensic references only. They must not be used directly for training,
otherwise the local 947-row gate is contaminated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition_utils import extract_boxed_answers, extract_final_answer, verify_answer  # noqa: E402

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Use concise reasoning, keep the final answer clean, and end with exactly one final answer in \\boxed{...}."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_answer_signature(prompt: object, answer: object) -> str:
    payload = json.dumps(
        {"prompt": str(prompt or "").strip(), "answer": str(answer or "").strip()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_validation_signatures(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    signatures: set[str] = set()
    for _, row in df.iterrows():
        prompt = ""
        for col in ("prompt", "prompt_x", "prompt_y", "question"):
            if col in row and str(row[col]):
                prompt = str(row[col])
                break
        answer = str(row.get("answer", ""))
        if prompt and answer:
            signatures.add(prompt_answer_signature(prompt, answer))
    return signatures


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content") or "")
    return str(row.get("completion") or row.get("assistant") or row.get("cot") or "")


def validate_training_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or "").strip()
    family = str(row.get("family") or row.get("category") or "").strip()
    text = assistant_text(row)
    extracted = extract_final_answer(text)
    boxed = extract_boxed_answers(text)
    final_box_pos = text.rfind(r"\boxed{")
    trailing = text[final_box_pos:] if final_box_pos >= 0 else ""
    after_final_box = ""
    if trailing:
        close = trailing.rfind("}")
        after_final_box = trailing[close + 1 :].strip() if close >= 0 else ""
    return {
        "id": row.get("id", ""),
        "family": family,
        "answer": answer,
        "boxed_count": len(boxed),
        "extracted": extracted,
        "verified": bool(answer and verify_answer(answer, extracted)),
        "single_boxed": len(boxed) == 1,
        "bit_answer_shape": bool(family == "bit_manipulation" and len(answer) == 8 and set(answer) <= {"0", "1"}),
        "text_after_final_box": after_final_box,
        "assistant_word_count": len(text.split()),
    }


def build_forensic_references(gains_csv: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    gains = pd.read_csv(gains_csv, dtype=str, keep_default_na=False)
    bit = gains[gains["task_type"] == "bit_manipulation"].copy()
    bit["training_use"] = "forbidden_local_val_reference_only"
    bit["reason"] = "V194 local validation row; use for taxonomy/template reference, not train."

    references: list[dict[str, Any]] = []
    for _, row in bit.iterrows():
        answer = str(row["legacy_solver_prediction"]).strip()
        prompt = str(row["prompt"])
        references.append(
            {
                "id": f"v214_forensic_bit_fix_{row['id']}",
                "family": "bit_manipulation",
                "source": "v214_forensic_reference_not_train",
                "prompt": prompt,
                "answer": answer,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": f"The verified bit rule gives the target output.\n\nFinal answer: \\boxed{{{answer}}}",
                    },
                ],
                "metadata": {
                    "train_allowed": False,
                    "reason": "local_validation_row_do_not_train",
                    "v194_prediction": row["v194_prediction"],
                    "legacy_solver_prediction": row["legacy_solver_prediction"],
                    "origin_id": row["id"],
                },
            }
        )
    return bit, references


def build_replay_manifest(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family") or row.get("category") or "unknown") for row in source_rows)
    source_counts = Counter(str(row.get("source") or "unknown") for row in source_rows)
    verified_counts: Counter[str] = Counter()
    single_box_counts: Counter[str] = Counter()
    for row in source_rows:
        family = str(row.get("family") or row.get("category") or "unknown")
        audit = validate_training_row(row)
        verified_counts[family] += int(audit["verified"])
        single_box_counts[family] += int(audit["single_boxed"])
    return {
        "rows": len(source_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "verified_counts": dict(sorted(verified_counts.items())),
        "single_box_counts": dict(sorted(single_box_counts.items())),
    }


def select_bit_replay_preview(
    source_rows: list[dict[str, Any]],
    limit: int,
    validation_signatures: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    rejected: Counter[str] = Counter()
    for row in source_rows:
        family = str(row.get("family") or row.get("category") or "")
        if family != "bit_manipulation":
            rejected["not_bit"] += 1
            continue
        prompt = str(row.get("prompt") or "")
        if not prompt or prompt in seen_prompts:
            rejected["missing_or_duplicate_prompt"] += 1
            continue
        if prompt_answer_signature(prompt, row.get("answer", "")) in validation_signatures:
            rejected["validation_overlap"] += 1
            continue
        audit = validate_training_row(row)
        if not (audit["verified"] and audit["bit_answer_shape"]):
            rejected["validation_failed"] += 1
            continue
        copied = dict(row)
        metadata = dict(copied.get("metadata") or {})
        metadata.update(
            {
                "v214_role": "candidate_bit_replay_preview",
                "train_allowed": True,
                "selection_note": "existing curated train source, not V194 local validation row",
            }
        )
        copied["metadata"] = metadata
        selected.append(copied)
        seen_prompts.add(prompt)
        if len(selected) >= limit:
            break
    return selected, rejected


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    gains_csv = Path(args.gains_csv)
    source_jsonl = Path(args.source_jsonl)
    validation_csv = Path(args.validation_csv) if args.validation_csv else Path()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[build_v214_bit_fix_preview] START")
    print(f"[build_v214_bit_fix_preview] gains_csv={gains_csv}")
    print(f"[build_v214_bit_fix_preview] source_jsonl={source_jsonl}")
    print(f"[build_v214_bit_fix_preview] validation_csv={validation_csv}")
    print(f"[build_v214_bit_fix_preview] output_dir={output_dir}")

    candidates_csv = output_dir / "v214_bit_solver_fix_candidates.csv"
    forensic_jsonl = output_dir / "v214_bit_fix_forensic_not_train.jsonl"
    replay_jsonl = output_dir / "v214_bit_replay_preview_train_allowed.jsonl"
    replay_manifest_json = output_dir / "v214_replay_pool_manifest.json"
    validation_report_json = output_dir / "v214_dataset_validation_report.json"

    candidates, forensic_refs = build_forensic_references(gains_csv)
    candidates.to_csv(candidates_csv, index=False)
    write_jsonl(forensic_jsonl, forensic_refs)

    source_rows = load_jsonl(source_jsonl)
    validation_signatures = load_validation_signatures(validation_csv) if args.validation_csv else set()
    replay_preview, replay_rejected = select_bit_replay_preview(
        source_rows,
        args.replay_preview_limit,
        validation_signatures,
    )
    write_jsonl(replay_jsonl, replay_preview)

    forensic_audits = [validate_training_row(row) for row in forensic_refs]
    replay_audits = [validate_training_row(row) for row in replay_preview]
    report = {
        "generated_at_utc": utc_now(),
        "decision": "preview_only_no_training_yet",
        "critical_warning": "Do not train on v214_bit_fix_forensic_not_train.jsonl; those are V194 local validation rows.",
        "inputs": {
            "gains_csv": str(gains_csv),
            "gains_sha256": sha256_file(gains_csv),
            "source_jsonl": str(source_jsonl),
            "source_exists": source_jsonl.exists(),
            "source_sha256": sha256_file(source_jsonl) if source_jsonl.exists() else "",
            "validation_csv": str(validation_csv) if args.validation_csv else "",
            "validation_signature_count": len(validation_signatures),
        },
        "outputs": {
            "candidates_csv": str(candidates_csv),
            "forensic_not_train_jsonl": str(forensic_jsonl),
            "replay_preview_train_allowed_jsonl": str(replay_jsonl),
            "replay_manifest_json": str(replay_manifest_json),
            "validation_report_json": str(validation_report_json),
        },
        "forensic_reference_rows": len(forensic_refs),
        "forensic_reference_verified": sum(1 for row in forensic_audits if row["verified"]),
        "forensic_train_allowed": False,
        "replay_preview_rows": len(replay_preview),
        "replay_preview_verified": sum(1 for row in replay_audits if row["verified"]),
        "replay_preview_train_allowed": True,
        "replay_rejected_counts": dict(sorted(replay_rejected.items())),
        "forensic_audit_failures": [row for row in forensic_audits if not row["verified"]],
        "replay_audit_failures": [row for row in replay_audits if not row["verified"]],
        "next_gate": [
            "build synthetic or curated non-validation bit fixes using the solver",
            "keep V194 947-row validation untouched for promotion",
            "audit V194 adapter_config/tokenizer/template before any continuation training",
        ],
    }
    replay_manifest = {
        "generated_at_utc": report["generated_at_utc"],
        "source_jsonl": str(source_jsonl),
        "source_sha256": report["inputs"]["source_sha256"],
        "validation_csv": report["inputs"]["validation_csv"],
        "validation_signature_count": len(validation_signatures),
        "source_pool": build_replay_manifest(source_rows),
        "selected_bit_replay_preview": {
            "rows": len(replay_preview),
            "sha256": sha256_file(replay_jsonl),
            "rejected_counts": dict(sorted(replay_rejected.items())),
        },
        "forbidden_local_val_reference": {
            "rows": len(forensic_refs),
            "sha256": sha256_file(forensic_jsonl),
            "train_allowed": False,
        },
    }
    write_json(replay_manifest_json, replay_manifest)
    write_json(validation_report_json, report)

    print(f"[build_v214_bit_fix_preview] wrote {candidates_csv}")
    print(f"[build_v214_bit_fix_preview] wrote {forensic_jsonl}")
    print(f"[build_v214_bit_fix_preview] wrote {replay_jsonl}")
    print(f"[build_v214_bit_fix_preview] wrote {replay_manifest_json}")
    print(f"[build_v214_bit_fix_preview] wrote {validation_report_json}")
    print(
        "[build_v214_bit_fix_preview] RESULT "
        f"forensic_refs={len(forensic_refs)} replay_preview={len(replay_preview)} "
        f"forensic_verified={report['forensic_reference_verified']} "
        f"replay_verified={report['replay_preview_verified']}"
    )
    print("[build_v214_bit_fix_preview] END")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gains-csv",
        default="artifacts/legacy_solver_probe_v194_2026-05-06/legacy_solver_probe_v194_error_gains.csv",
    )
    parser.add_argument("--source-jsonl", default="data/v206/v206_curated_train.jsonl")
    parser.add_argument("--validation-csv", default="artifacts/drive_exports/v194_baseline_predictions.csv")
    parser.add_argument("--output-dir", default="artifacts/v214_bit_fix_preview_2026-05-06")
    parser.add_argument("--replay-preview-limit", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
