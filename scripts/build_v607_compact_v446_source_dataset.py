#!/usr/bin/env python3
"""Build V607 compact source-only dataset from V606-clean V446 rows.

V606 found a large pool of target-family V446 rows that are source-only,
answer-verified, unused by the quarantined V573/V579/V591/V596 routes, but too
long and multi-boxed to train verbatim. V607 keeps the verified source signal
while removing the two failure modes:

* no raw long CoT is copied into the assistant target;
* each target ends with exactly one label-free ``\boxed{}`` final answer.

This builder is CPU-only. It does not authorize GPU, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import PROMPT_SUFFIX, extract_final_answer, verify_answer  # noqa: E402


DEFAULT_V606_MANIFEST = ROOT / "artifacts/v606_unused_v446_source_pool_audit/v606_unused_v446_source_pool_manifest.json"
DEFAULT_V606_CLEAN_PREVIEW = ROOT / "artifacts/v606_unused_v446_source_pool_audit/v606_clean_unused_v446_rows_preview.csv"
DEFAULT_V446_AUDIT_CSV = (
    ROOT
    / "artifacts/v446_tong_source_target_alignment_gate/20260515T_v446_cpu_gate/"
    / "v446_tong_source_target_alignment_gate_candidate_audit.csv"
)
DEFAULT_SFT_JSONL = Path(r"C:\Users\davis\Downloads\sft_reconstructed.jsonl")
DEFAULT_COMPETITION_TRAIN_CSV = Path(r"C:\Users\davis\Downloads\competition_train.csv")
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v607_compact_v446_source_dataset/20260518T_v607_cpu_gate"

TARGET_FAMILIES = {"bit_manipulation", "equation_transform"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def ascii_clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u3010": "[",
        "\u3011": "]",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_boxed_payloads(text: str) -> str:
    return re.sub(r"\\boxed\{[^{}]*\}", "[boxed removed]", text)


def strip_return_boilerplate(text: str) -> str:
    text = re.split(r"\nI will now return|\nThe answer in|\n</think>", text, maxsplit=1)[0]
    lines = []
    for line in text.splitlines():
        if "boxed removed" in line or "I will now return" in line or "The answer in" in line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def compact_bit_trace(assistant: str, answer: str, max_body_chars: int) -> tuple[str, str]:
    text = assistant.replace("<think>", "").replace("</think>", "")
    marker = "\nSelected\n"
    apply_match = list(re.finditer(r"\nApplying to [^\n]+", text))
    selected = ""
    if marker in text:
        tail = text.rsplit(marker, 1)[1]
        selected = tail.split("\n\nApplying to ", 1)[0].strip()
        selected_lines = [line for line in selected.splitlines() if line.strip()]
        selected = "\n".join(selected_lines[:20])
    applying = ""
    if apply_match:
        start = apply_match[-1].start() + 1
        applying = text[start:]
        applying = strip_return_boilerplate(applying)
    body = "\n\n".join(part for part in (f"Selected rules:\n{selected}" if selected else "", applying) if part)
    if not body:
        body = text[-max_body_chars:]
    body = ascii_clean(strip_boxed_payloads(body))
    if len(body) > max_body_chars:
        body = body[-max_body_chars:].lstrip()
    return body, "compact_bit_selected_apply"


def compact_equation_trace(assistant: str, answer: str, max_body_chars: int) -> tuple[str, str]:
    text = assistant.replace("<think>", "").replace("</think>", "")
    matches = list(re.finditer(r"\nApplying to [^\n]+", text))
    if matches:
        body = text[matches[-1].start() + 1 :]
        body = strip_return_boilerplate(body)
    else:
        body = text[-max_body_chars:]
    body = ascii_clean(strip_boxed_payloads(body))
    if len(body) > max_body_chars:
        body = body[-max_body_chars:].lstrip()
    return body, "compact_equation_apply_only"


def final_boxed(answer: str) -> str:
    value = str(answer).strip()
    boxed = f"\\boxed{{{value}}}"
    if not verify_answer(value, extract_final_answer(f"Final answer: {boxed}")):
        raise ValueError(f"answer cannot round-trip through boxed final answer: {value!r}")
    return boxed


def accepted_v446_rows(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in read_csv(path):
        if str(row.get("accepted", "")).strip().lower() != "true":
            continue
        if str(row.get("family", "")).strip() not in TARGET_FAMILIES:
            continue
        out.append(row)
    return out


def competition_index(path: Path) -> dict[str, dict[str, str]]:
    return {str(row.get("id", "")): row for row in read_csv(path) if str(row.get("id", ""))}


def deterministic_split_key(row_id: str) -> int:
    return int(hashlib.sha256(row_id.encode("utf-8", errors="replace")).hexdigest()[:8], 16)


def split_rows(rows: list[dict[str, Any]], val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = sorted(
            [row for row in rows if str(row["family"]) == family],
            key=lambda item: deterministic_split_key(str(item["metadata"]["source_row_id"])),
        )
        val_count = max(1, round(len(family_rows) * val_fraction))
        val_ids = {str(row["id"]) for row in family_rows[:val_count]}
        for row in family_rows:
            if str(row["id"]) in val_ids:
                row["metadata"]["split"] = "validation"
                val.append(row)
            else:
                row["metadata"]["split"] = "train"
                train.append(row)
    return train, val


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = Counter(str(row.get("family", "")) for row in rows)
    subcategory = Counter(str(row.get("subcategory", "")) for row in rows)
    weights: Counter[str] = Counter()
    assistant_chars: list[int] = []
    boxed_count = 0
    answer_mismatch = 0
    non_ascii_rows = 0
    for row in rows:
        assistant = message_content(row, "assistant")
        answer = str(row.get("answer", ""))
        assistant_chars.append(len(assistant))
        boxed_count += assistant.count("\\boxed{")
        weights[str(row.get("metadata", {}).get("loss_weight", ""))] += 1
        if not verify_answer(answer, extract_final_answer(assistant)):
            answer_mismatch += 1
        if any(ord(ch) > 127 for ch in assistant + str(row.get("prompt", ""))):
            non_ascii_rows += 1
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family.items())),
        "subcategory_counts": dict(sorted(subcategory.items())),
        "loss_weight_counts": dict(sorted(weights.items())),
        "assistant_chars_max": max(assistant_chars) if assistant_chars else 0,
        "assistant_chars_p50": statistics.median(assistant_chars) if assistant_chars else 0,
        "assistant_boxed_total": boxed_count,
        "assistant_answer_mismatch_rows": answer_mismatch,
        "non_ascii_rows": non_ascii_rows,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v606 = read_json(args.v606_manifest)
    sft_rows = read_jsonl(args.sft_jsonl)
    competition = competition_index(args.competition_train_csv)
    accepted = accepted_v446_rows(args.v446_audit_csv)
    dirty_ids = {
        str(row.get("id", ""))
        for row in read_csv(args.v606_dirty_csv)
        if str(row.get("dirty_reason", "")).strip()
    }

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for audit_row in accepted:
        rid = str(audit_row.get("id", "")).strip()
        family = str(audit_row.get("family", "")).strip()
        row_no = int(str(audit_row.get("row_no", "0")))
        if rid in dirty_ids:
            continue
        if family not in TARGET_FAMILIES or row_no <= 0 or row_no > len(sft_rows):
            continue
        source = sft_rows[row_no - 1]
        source_comp = competition.get(rid, {})
        prompt = str(source_comp.get("prompt", "")).strip()
        answer = str(source_comp.get("answer", "")).strip()
        assistant_raw = message_content(source, "assistant")
        if not prompt or not answer or not verify_answer(answer, extract_final_answer(assistant_raw)):
            excluded.append({"id": rid, "family": family, "reason": "source_answer_not_verified"})
            continue
        try:
            boxed = final_boxed(answer)
        except ValueError:
            excluded.append({"id": rid, "family": family, "reason": "boxed_roundtrip_failed", "answer": answer})
            continue
        if family == "bit_manipulation":
            body, trace_style = compact_bit_trace(assistant_raw, answer, args.max_body_chars)
            loss_weight = args.bit_loss_weight
            subcategory = "v607_v446_bit_compact_source"
        else:
            body, trace_style = compact_equation_trace(assistant_raw, answer, args.max_body_chars)
            loss_weight = args.equation_loss_weight
            subcategory = "v607_v446_equation_compact_source"
        assistant = (
            "Verified compact source trace.\n"
            f"{body}\n\n"
            f"Final answer: {boxed}"
        )
        assistant = ascii_clean(assistant)
        if assistant.count("\\boxed{") != 1 or not verify_answer(answer, extract_final_answer(assistant)):
            excluded.append({"id": rid, "family": family, "reason": "compact_target_not_label_free", "answer": answer})
            continue
        prompt_for_training = prompt
        row = {
            "id": f"v607_{family}_{rid}",
            "prompt": prompt_for_training,
            "answer": answer,
            "family": family,
            "subcategory": subcategory,
            "source": "v607_compact_v446_source_dataset",
            "source_dataset": "v607_compact_v446_source_dataset",
            "messages": [
                {"role": "user", "content": prompt_for_training + PROMPT_SUFFIX},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {
                "schema_version": "kg1_v607_compact_v446_source_dataset_v1",
                "source": "v607_compact_v446_source_dataset",
                "source_dataset": "v607_compact_v446_source_dataset",
                "source_only": True,
                "source_row_id": rid,
                "v446_row_no": row_no,
                "v446_status": audit_row.get("status", ""),
                "v606_clean_source": True,
                "trace_style": trace_style,
                "prompt_contract": "official_like",
                "prompt_suffix": PROMPT_SUFFIX,
                "loss_weight": loss_weight,
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
                "gate_rows_used_for_training": False,
                "raw_output_required_for_promotion": True,
                "final_answer_format": "final_answer_boxed_label_free",
            },
        }
        rows.append(row)

    train, val = split_rows(rows, args.val_fraction)
    train_path = args.output_dir / "v607_compact_v446_source_train.jsonl"
    val_path = args.output_dir / "v607_compact_v446_source_val.jsonl"
    manifest_path = args.output_dir / "v607_compact_v446_source_dataset_manifest.json"
    report_path = args.output_dir / "KG1_V607_COMPACT_V446_SOURCE_DATASET.md"
    excluded_path = args.output_dir / "v607_excluded_rows.csv"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_csv(excluded_path, excluded, ["id", "family", "reason", "answer"])
    train_summary = summarize(train)
    val_summary = summarize(val)
    weight_sum = Counter()
    for row in train:
        weight_sum[str(row["family"])] += float(row["metadata"]["loss_weight"])
    total_weight = sum(weight_sum.values()) or 1.0
    manifest = {
        "schema_version": "kg1_v607_compact_v446_source_dataset_v1",
        "version": "V607",
        "generated_at_utc": utc_now(),
        "decision": {
            "status": "dataset_ready_for_cpu_gates",
            "gpu_allowed": False,
            "submit_allowed": False,
            "reason": "Compact source-only dataset built from V606-clean V446 rows; CPU gates still required.",
            "next_action": "Run V509 integrity, V286 real tokenization, V513 learnability, V524/V575 objective-contract gates before any paid GPU.",
        },
        "inputs": {
            "v606_manifest": str(args.v606_manifest),
            "v606_manifest_sha256": sha256_file(args.v606_manifest),
            "v606_decision": v606.get("decision"),
            "v446_audit_csv": str(args.v446_audit_csv),
            "v446_audit_sha256": sha256_file(args.v446_audit_csv),
            "sft_jsonl": str(args.sft_jsonl),
            "sft_jsonl_sha256": sha256_file(args.sft_jsonl),
            "competition_train_csv": str(args.competition_train_csv),
            "competition_train_sha256": sha256_file(args.competition_train_csv),
        },
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
            "summary_md": str(report_path),
            "excluded_rows_csv": str(excluded_path),
        },
        "weights": {
            "bit_loss_weight": args.bit_loss_weight,
            "equation_loss_weight": args.equation_loss_weight,
            "loss_weight_sum_by_family": dict(sorted(weight_sum.items())),
            "loss_weight_share_by_family": {
                family: round(value / total_weight, 6) for family, value in sorted(weight_sum.items())
            },
            "required_train_env": {
                "LOSS_NORMALIZATION_MODE": "example_mean",
                "USE_ROW_LOSS_WEIGHT": "1",
                "REQUIRE_ROW_LOSS_WEIGHT": "1",
            },
        },
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "excluded_rows": len(excluded),
        "blocked_actions": ["train_gpu", "full_eval", "package", "kaggle_submit"],
    }
    write_json(manifest_path, manifest)
    write_report(report_path, manifest)
    return manifest


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# KG1 V607 Compact V446 Source Dataset",
        "",
        f"Decision: `{manifest['decision']['status']}`",
        f"GPU allowed: `{manifest['decision']['gpu_allowed']}`",
        "",
        "## Dataset",
        "",
        f"- train rows: `{manifest['train_summary']['rows']}`",
        f"- validation rows: `{manifest['validation_summary']['rows']}`",
        f"- train family counts: `{json.dumps(manifest['train_summary']['family_counts'], sort_keys=True)}`",
        f"- validation family counts: `{json.dumps(manifest['validation_summary']['family_counts'], sort_keys=True)}`",
        f"- train assistant chars p50/max: `{manifest['train_summary']['assistant_chars_p50']}` / `{manifest['train_summary']['assistant_chars_max']}`",
        f"- validation assistant chars p50/max: `{manifest['validation_summary']['assistant_chars_p50']}` / `{manifest['validation_summary']['assistant_chars_max']}`",
        f"- assistant mismatch rows: train `{manifest['train_summary']['assistant_answer_mismatch_rows']}`, val `{manifest['validation_summary']['assistant_answer_mismatch_rows']}`",
        f"- non-ASCII rows: train `{manifest['train_summary']['non_ascii_rows']}`, val `{manifest['validation_summary']['non_ascii_rows']}`",
        f"- loss weight share train: `{json.dumps(manifest['weights']['loss_weight_share_by_family'], sort_keys=True)}`",
        "",
        "## Rule",
        "",
        (
            "This dataset is not submit-safe evidence. It is a source-only training candidate. "
            "The next step is CPU gates only; H200 remains blocked until the gate chain passes."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v606-manifest", type=Path, default=DEFAULT_V606_MANIFEST)
    parser.add_argument("--v606-dirty-csv", type=Path, default=DEFAULT_V606_CLEAN_PREVIEW.parent / "v606_dirty_v446_rows.csv")
    parser.add_argument("--v446-audit-csv", type=Path, default=DEFAULT_V446_AUDIT_CSV)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--competition-train-csv", type=Path, default=DEFAULT_COMPETITION_TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-body-chars", type=int, default=1800)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--bit-loss-weight", type=float, default=1.1)
    parser.add_argument("--equation-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    print("=== V607 COMPACT V446 SOURCE DATASET BUILD START ===", flush=True)
    print(f"v606_manifest = {args.v606_manifest}", flush=True)
    print(f"v446_audit_csv = {args.v446_audit_csv}", flush=True)
    print(f"output_dir = {args.output_dir}", flush=True)
    manifest = build(args)
    print("train_summary =", json.dumps(manifest["train_summary"], sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(manifest["validation_summary"], sort_keys=True), flush=True)
    print("loss_weight_share_by_family =", json.dumps(manifest["weights"]["loss_weight_share_by_family"], sort_keys=True), flush=True)
    print("gpu_allowed =", manifest["decision"]["gpu_allowed"], flush=True)
    print("=== V607 COMPACT V446 SOURCE DATASET BUILD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
