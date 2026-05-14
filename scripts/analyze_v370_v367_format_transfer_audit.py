"""Audit whether V367 boxed-only training changed V368 inference format.

V369 showed that V368 transferred 0/8 accepted V366 gains. This script checks
the next concrete failure mode: the V367 dataset taught boxed-only answers, but
V368 bit rows may still emit the long pre-existing bit-reasoning trace.

This is CPU-only and exists to decide whether another HF job is justified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assistant_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def row_contract(df: pd.DataFrame) -> str:
    if len(df) != 315:
        raise RuntimeError(f"expected 315 rows, got {len(df)}")
    payload = "\n".join(
        f"{row.id}\t{row.type}\t{row.answer}\t{hashlib.sha256(str(row.prompt).encode('utf-8')).hexdigest()}"
        for row in df.sort_values("id").itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_flags(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    return {
        "raw_chars": len(stripped),
        "exact_boxed_only": bool(re.fullmatch(r"\\boxed\{[^}]+\}", stripped)),
        "starts_boxed": stripped.startswith("\\boxed"),
        "contains_boxed": "\\boxed{" in stripped,
        "contains_deduce_trace": "We need to deduce" in stripped,
        "contains_output_bit_columns": "Output bit columns" in stripped,
        "contains_matching_output": "Matching output" in stripped,
        "contains_final_answer_phrase": "final answer" in stripped.lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v367-train-jsonl",
        type=Path,
        default=Path("artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_train.jsonl"),
    )
    parser.add_argument(
        "--v367-val-jsonl",
        type=Path,
        default=Path("artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_val.jsonl"),
    )
    parser.add_argument(
        "--v368-predictions-csv",
        type=Path,
        default=Path("artifacts/v368_hf_a100_v367_bit_ternary_launch/eval_checkpoint1/predictions.csv"),
    )
    parser.add_argument(
        "--v369-manifest-json",
        type=Path,
        default=Path("artifacts/v369_v368_transfer_failure_audit/20260514T_cpu_audit/v369_v368_transfer_failure_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v370_v367_format_transfer_audit/20260514T_cpu_audit"),
    )
    parser.add_argument(
        "--expected-shared-row-contract-sha256",
        default=EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    )
    args = parser.parse_args()

    print("=== V370 V367 FORMAT TRANSFER AUDIT START ===", flush=True)
    print("v367_train_jsonl =", args.v367_train_jsonl, flush=True)
    print("v367_val_jsonl =", args.v367_val_jsonl, flush=True)
    print("v368_predictions_csv =", args.v368_predictions_csv, flush=True)
    print("v369_manifest_json =", args.v369_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    for path in [args.v367_train_jsonl, args.v367_val_jsonl, args.v368_predictions_csv, args.v369_manifest_json]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_rows = load_jsonl(args.v367_train_jsonl)
    val_rows = load_jsonl(args.v367_val_jsonl)
    v368 = pd.read_csv(args.v368_predictions_csv)
    v369_manifest = json.loads(args.v369_manifest_json.read_text(encoding="utf-8"))

    observed_contract = row_contract(v368[["id", "prompt", "answer", "type"]])
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + str(args.expected_shared_row_contract_sha256)
            + ", got "
            + observed_contract
        )

    v368["correct_bool"] = bool_series(v368["correct"])
    v368["truncated_bool"] = bool_series(v368["truncated"])
    raw_flags = pd.DataFrame([format_flags(text) for text in v368["raw_output"].astype(str)])
    raw_flags.insert(0, "id", v368["id"].astype(str))
    raw_flags.insert(1, "family", v368["type"].astype(str))
    raw_flags.insert(2, "correct", v368["correct_bool"])
    raw_flags.insert(3, "completion_tokens", v368["completion_tokens"].astype(int))

    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_rule: dict[str, str] = {}
    assistant_format_counts = Counter()
    for split, rows in [("train", train_rows), ("validation", val_rows)]:
        for row in rows:
            meta = row.get("metadata", {})
            source_id = str(meta.get("source_id", ""))
            if source_id:
                source_counts[source_id][split] += 1
                source_rule[source_id] = str(meta.get("rule_class", ""))
            text = assistant_text(row)
            flags = format_flags(text)
            assistant_format_counts[(split, "exact_boxed_only" if flags["exact_boxed_only"] else "not_boxed_only")] += 1

    accepted_ids = [str(x) for x in v369_manifest.get("v368_unique_gain_ids", [])]
    loss_ids = [str(x) for x in v369_manifest.get("v368_loss_ids", [])]
    transfer_csv = Path(v369_manifest["outputs"]["v366_gain_transfer_csv"])
    transfer_df = pd.read_csv(transfer_csv)
    v366_accepted_ids = transfer_df["id"].astype(str).tolist()

    coverage_rows: list[dict[str, Any]] = []
    for source_id in v366_accepted_ids:
        row = v368[v368["id"].astype(str) == source_id].iloc[0]
        flags = format_flags(str(row["raw_output"]))
        coverage_rows.append(
            {
                "id": source_id,
                "source_rule": source_rule.get(source_id, ""),
                "v367_train_rows": int(source_counts[source_id]["train"]),
                "v367_val_rows": int(source_counts[source_id]["validation"]),
                "answer": row["answer"],
                "v368_prediction": row["prediction"],
                "v368_correct": bool(row["correct_bool"]),
                "completion_tokens": int(row["completion_tokens"]),
                **flags,
            }
        )

    example_ids = v366_accepted_ids + accepted_ids + loss_ids
    seen: set[str] = set()
    example_rows: list[dict[str, Any]] = []
    for row_id in example_ids:
        if row_id in seen:
            continue
        seen.add(row_id)
        row = v368[v368["id"].astype(str) == row_id].iloc[0]
        raw = str(row["raw_output"])
        example_rows.append(
            {
                "id": row_id,
                "family": row["type"],
                "answer": row["answer"],
                "prediction": row["prediction"],
                "correct": bool(row["correct_bool"]),
                "completion_tokens": int(row["completion_tokens"]),
                "raw_prefix_400": raw[:400].replace("\n", "\\n"),
            }
        )

    family_rows: list[dict[str, Any]] = []
    for fam, grp in raw_flags.groupby("family", sort=True):
        family_rows.append(
            {
                "family": fam,
                "rows": int(len(grp)),
                "correct": int(grp["correct"].sum()),
                "exact_boxed_only": int(grp["exact_boxed_only"].sum()),
                "starts_boxed": int(grp["starts_boxed"].sum()),
                "contains_deduce_trace": int(grp["contains_deduce_trace"].sum()),
                "contains_output_bit_columns": int(grp["contains_output_bit_columns"].sum()),
                "completion_tokens_mean": float(grp["completion_tokens"].mean()),
                "completion_tokens_median": float(grp["completion_tokens"].median()),
                "completion_tokens_min": int(grp["completion_tokens"].min()),
                "completion_tokens_max": int(grp["completion_tokens"].max()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_flags_out = args.output_dir / "v370_v368_raw_output_format_flags.csv"
    family_out = args.output_dir / "v370_family_format_summary.csv"
    coverage_out = args.output_dir / "v370_v366_gain_training_coverage.csv"
    examples_out = args.output_dir / "v370_raw_output_examples.csv"
    manifest_out = args.output_dir / "v370_v367_format_transfer_manifest.json"
    summary_out = args.output_dir.parent / "V370_RESULT_SUMMARY.md"

    raw_flags.to_csv(raw_flags_out, index=False)
    pd.DataFrame(family_rows).to_csv(family_out, index=False)
    write_csv(
        coverage_out,
        coverage_rows,
        [
            "id",
            "source_rule",
            "v367_train_rows",
            "v367_val_rows",
            "answer",
            "v368_prediction",
            "v368_correct",
            "completion_tokens",
            "raw_chars",
            "exact_boxed_only",
            "starts_boxed",
            "contains_boxed",
            "contains_deduce_trace",
            "contains_output_bit_columns",
            "contains_matching_output",
            "contains_final_answer_phrase",
        ],
    )
    write_csv(
        examples_out,
        example_rows,
        ["id", "family", "answer", "prediction", "correct", "completion_tokens", "raw_prefix_400"],
    )

    bit_flags = raw_flags[raw_flags["family"] == "bit_manipulation"]
    exact_boxed_total = int(raw_flags["exact_boxed_only"].sum())
    bit_deduce_total = int(bit_flags["contains_deduce_trace"].sum())
    transferred_rows = int(sum(1 for row in coverage_rows if row["v368_correct"]))
    decision = {
        "decision": "v367_boxed_only_format_not_transferred",
        "hf_gpu_allowed": False,
        "next_action": (
            "Do not train more on boxed-only V367. If any LoRA route continues, first build a CPU-gated "
            "dataset that matches the actual bit reasoning trace style, or return to equation DSL."
        ),
        "reason": (
            f"V367 assistant targets were boxed-only, but V368 exact_boxed_only={exact_boxed_total}/315; "
            f"bit rows with old deduce trace={bit_deduce_total}/160; "
            f"V366 accepted gains transferred={transferred_rows}/{len(coverage_rows)}."
        ),
    }
    manifest = {
        "schema_version": "kg1_v370_v367_format_transfer_audit_v1",
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "inputs": {
            "v367_train_jsonl": str(args.v367_train_jsonl),
            "v367_train_sha256": sha256_file(args.v367_train_jsonl),
            "v367_val_jsonl": str(args.v367_val_jsonl),
            "v367_val_sha256": sha256_file(args.v367_val_jsonl),
            "v368_predictions_csv": str(args.v368_predictions_csv),
            "v368_predictions_sha256": sha256_file(args.v368_predictions_csv),
            "v369_manifest_json": str(args.v369_manifest_json),
            "v369_manifest_sha256": sha256_file(args.v369_manifest_json),
        },
        "v367_assistant_format_counts": {
            f"{split}_{kind}": count for (split, kind), count in sorted(assistant_format_counts.items())
        },
        "v368_exact_boxed_only_rows": exact_boxed_total,
        "v368_starts_boxed_rows": int(raw_flags["starts_boxed"].sum()),
        "v368_contains_boxed_rows": int(raw_flags["contains_boxed"].sum()),
        "v368_bit_rows_with_deduce_trace": bit_deduce_total,
        "v368_bit_rows_with_output_bit_columns": int(bit_flags["contains_output_bit_columns"].sum()),
        "v366_accepted_gain_rows": int(len(coverage_rows)),
        "v366_accepted_gain_rows_transferred_to_v368": transferred_rows,
        "v366_accepted_gain_training_coverage_min_train": int(min(row["v367_train_rows"] for row in coverage_rows)),
        "v366_accepted_gain_training_coverage_min_val": int(min(row["v367_val_rows"] for row in coverage_rows)),
        "family_format_summary": family_rows,
        "decision": decision,
        "outputs": {
            "raw_output_format_flags_csv": str(raw_flags_out),
            "family_format_summary_csv": str(family_out),
            "v366_gain_training_coverage_csv": str(coverage_out),
            "raw_output_examples_csv": str(examples_out),
            "manifest_json": str(manifest_out),
            "summary_md": str(summary_out),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out.write_text(
        "\n".join(
            [
                "# V370 V367 format transfer audit",
                "",
                "Generated: 2026-05-14",
                "",
                "## Result",
                "",
                f"- V367 train assistant targets boxed-only: `{assistant_format_counts[('train', 'exact_boxed_only')]}/{len(train_rows)}`.",
                f"- V367 validation assistant targets boxed-only: `{assistant_format_counts[('validation', 'exact_boxed_only')]}/{len(val_rows)}`.",
                f"- V368 raw outputs exact boxed-only: `{exact_boxed_total}/315`.",
                f"- V368 bit rows containing old `We need to deduce` trace: `{bit_deduce_total}/160`.",
                f"- V368 bit rows containing `Output bit columns`: `{int(bit_flags['contains_output_bit_columns'].sum())}/160`.",
                f"- V366 accepted gains had at least `{manifest['v366_accepted_gain_training_coverage_min_train']}` train and `{manifest['v366_accepted_gain_training_coverage_min_val']}` validation rows each in V367.",
                f"- V366 accepted gains transferred to V368: `{transferred_rows}/{len(coverage_rows)}`.",
                "",
                "## Decision",
                "",
                "Blocked. The boxed-only objective did not control the actual bit inference format. V368 continued producing the old long bit-reasoning trace on every bit row.",
                "",
                "Next action: do not spend HF on V367/V368. A future LoRA attempt must first be CPU-gated with targets that match the actual bit trace format, or the roadmap should return to equation DSL.",
                "",
                "## Local artifacts",
                "",
                f"- Manifest: `{manifest_out}`",
                f"- Family format summary: `{family_out}`",
                f"- V366 coverage detail: `{coverage_out}`",
                f"- Raw examples: `{examples_out}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("outputs =", json.dumps(manifest["outputs"], sort_keys=True), flush=True)
    print("=== V370 V367 FORMAT TRANSFER AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
