#!/usr/bin/env python3
"""Build V461 synthetic numeric prompt-only probe pack.

V461 is a CPU-only preflight. It creates synthetic Alice equation prompts for
numeric V274 rule classes that have weak diagnostic signal but insufficient
adapter-level hard-negative evidence. The prompt pack intentionally omits
answers; labels stay only in the local audit so a later inference-only HF job
can collect raw adapter outputs without seeing labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
for item in (ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import classify_puzzle, verify_answer  # noqa: E402
from kg1_v274_numeric_postprocessor import (  # noqa: E402
    choose_guarded_numeric_override,
    normalize_payload,
    parse_alice_prompt,
)


VERSION = "v461_synthetic_numeric_probe_pack"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "v461_synthetic_numeric_probe_pack" / "20260515T_cpu_gate"

TARGET_RULES = (
    "v274_guarded_numeric_add_direct_over_model_add_variant",
    "v274_guarded_numeric_colon_absdiff_restore_trailing_zero",
    "v274_guarded_numeric_minus_direct_negative_restore_sign",
    "v274_guarded_numeric_minus_signed_opposite_sign_guarded",
)

PROMPT_COLUMNS = ["id", "family", "target_rule_class", "prompt_sha256", "prompt_normalized_sha256", "prompt"]
AUDIT_COLUMNS = [
    "id",
    "family",
    "target_rule_class",
    "query",
    "answer",
    "simulated_wrong_prediction",
    "postprocessor_prediction",
    "postprocessor_rule",
    "postprocessor_proof",
    "prompt_sha256",
    "prompt_normalized_sha256",
    "selected_for_prompt_pack",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


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


def make_prompt(examples: list[tuple[str, str]], query: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"
    ]
    lines.extend(f"{lhs} = {rhs}" for lhs, rhs in examples)
    lines.append(f"Now, determine the result for: {query}")
    return "\n".join(lines)


def two_digit(value: int) -> str:
    return f"{value % 100:02d}"


def reverse_int_text(text: str) -> int:
    return int(str(text)[::-1])


def signed_reverse_keep_sign(value: int) -> str:
    text = str(value)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def add_variant_rows(max_per_rule: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    # Symmetric examples make direct add and reversed-operand add both plausible
    # for the examples; the target then exposes the model's add-variant error.
    example_sets = [
        [("12)21", "33"), ("34)43", "77")],
        [("16)61", "77"), ("25)52", "77")],
        [("13)31", "44"), ("46)64", "110")],
        [("18)81", "99"), ("23)32", "55")],
    ]
    targets = [("94", "40"), ("83", "50"), ("71", "60"), ("82", "41"), ("95", "31"), ("62", "70"), ("73", "52"), ("86", "20")]
    idx = 0
    for examples in example_sets:
        for left, right in targets:
            direct = int(left) + int(right)
            wrong = signed_reverse_keep_sign(reverse_int_text(left) + reverse_int_text(right))
            if normalize_payload(wrong) == normalize_payload(str(direct)):
                continue
            rows.append(
                {
                    "id": f"v461_syn_add_{idx:04d}",
                    "target_rule_class": "v274_guarded_numeric_add_direct_over_model_add_variant",
                    "query": f"{left}){right}",
                    "answer": str(direct),
                    "simulated_wrong_prediction": wrong,
                    "prompt": make_prompt(examples, f"{left}){right}"),
                }
            )
            idx += 1
            if len(rows) >= max_per_rule:
                return rows
    return rows


def colon_trailing_zero_rows(max_per_rule: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    examples_list = [
        [("80:20", "60"), ("91:31", "60"), ("77:37", "40")],
        [("95:15", "80"), ("82:12", "70"), ("64:24", "40")],
        [("70:10", "60"), ("83:23", "60"), ("58:18", "40")],
    ]
    targets = [("37", "67"), ("14", "54"), ("29", "89"), ("91", "21"), ("86", "36"), ("75", "15"), ("63", "03"), ("98", "18")]
    idx = 0
    for examples in examples_list:
        for left, right in targets:
            answer = str(abs(int(left) - int(right)))
            if not answer.endswith("0"):
                continue
            wrong = answer.rstrip("0") or "0"
            rows.append(
                {
                    "id": f"v461_syn_colon_{idx:04d}",
                    "target_rule_class": "v274_guarded_numeric_colon_absdiff_restore_trailing_zero",
                    "query": f"{left}:{right}",
                    "answer": answer,
                    "simulated_wrong_prediction": wrong,
                    "prompt": make_prompt(examples, f"{left}:{right}"),
                }
            )
            idx += 1
            if len(rows) >= max_per_rule:
                return rows
    return rows


def minus_direct_negative_rows(max_per_rule: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    examples = [("90-10", "80"), ("70-30", "40"), ("55-20", "35")]
    targets = [("11", "50"), ("08", "29"), ("35", "72"), ("44", "91"), ("12", "73"), ("04", "55"), ("19", "68"), ("22", "80")]
    for idx, (left, right) in enumerate(targets[:max_per_rule]):
        answer = str(int(left) - int(right))
        rows.append(
            {
                "id": f"v461_syn_minus_direct_{idx:04d}",
                "target_rule_class": "v274_guarded_numeric_minus_direct_negative_restore_sign",
                "query": f"{left}-{right}",
                "answer": answer,
                "simulated_wrong_prediction": answer[1:],
                "prompt": make_prompt(examples, f"{left}-{right}"),
            }
        )
    return rows


def minus_signed_opposite_rows(max_per_rule: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    example_sets = [
        [("06-63", "42"), ("96-32", "64"), ("87-15", "72"), ("58-64", "93"), ("87-63", "24")],
        [("33-39", "-06"), ("76-43", "33"), ("42-25", "-82")],
        [("22-53", "-31"), ("17-33", "83"), ("71-11", "6"), ("19-16", "03")],
    ]
    targets = [("85", "92", "92"), ("66", "67", "-01"), ("46", "83", "62"), ("12", "64", "-52"), ("08", "53", "54"), ("73", "21", "52"), ("97", "07", "9")]
    idx = 0
    for examples in example_sets:
        for left, right, answer in targets:
            wrong = answer[1:] if answer.startswith("-") else "-" + answer
            direct_difference = int(left) - int(right)
            if answer.startswith("-") and str(direct_difference) == answer:
                # This is the simpler direct-negative sign restore class, not
                # the guarded signed-opposite class. Keep V461 classes disjoint.
                continue
            rows.append(
                {
                    "id": f"v461_syn_minus_signed_{idx:04d}",
                    "target_rule_class": "v274_guarded_numeric_minus_signed_opposite_sign_guarded",
                    "query": f"{left}-{right}",
                    "answer": answer,
                    "simulated_wrong_prediction": wrong,
                    "prompt": make_prompt(examples, f"{left}-{right}"),
                }
            )
            idx += 1
            if len(rows) >= max_per_rule:
                return rows
    return rows


def validate_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = row["prompt"]
    examples, query, parse_status = parse_alice_prompt(prompt)
    if parse_status != "ok":
        raise RuntimeError(f"{row['id']}: parse_status={parse_status}")
    replacement, rule, proof = choose_guarded_numeric_override(examples, query, row["simulated_wrong_prediction"])
    rule_class = "v274_guarded_numeric_" + str(rule)
    if rule_class != row["target_rule_class"]:
        raise RuntimeError(f"{row['id']}: expected {row['target_rule_class']} got {rule_class}; proof={proof}")
    if not replacement or not verify_answer(row["answer"], replacement):
        raise RuntimeError(f"{row['id']}: replacement {replacement!r} does not verify against {row['answer']!r}")
    family = classify_puzzle(prompt)
    if family != "equation_transform":
        raise RuntimeError(f"{row['id']}: family={family}")
    prompt_sha = sha256_text(prompt.replace("\r\n", "\n"))
    prompt_norm_sha = sha256_text(normalize_prompt(prompt))
    return {
        **row,
        "family": family,
        "postprocessor_prediction": replacement,
        "postprocessor_rule": rule,
        "postprocessor_proof": proof,
        "prompt_sha256": prompt_sha,
        "prompt_normalized_sha256": prompt_norm_sha,
        "selected_for_prompt_pack": True,
    }


def build_rows(max_per_rule: int) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, str]] = []
    raw_rows.extend(add_variant_rows(max_per_rule))
    raw_rows.extend(colon_trailing_zero_rows(max_per_rule))
    raw_rows.extend(minus_direct_negative_rows(max_per_rule))
    raw_rows.extend(minus_signed_opposite_rows(max_per_rule))
    rows = [validate_row(row) for row in raw_rows]
    counts = Counter(str(row["target_rule_class"]) for row in rows)
    missing = [rule for rule in TARGET_RULES if counts[rule] == 0]
    if missing:
        raise RuntimeError("missing synthetic rule classes: " + ", ".join(missing))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V461 SYNTHETIC NUMERIC PROBE PACK START ===", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("max_per_rule =", args.max_per_rule, flush=True)
    rows = build_rows(args.max_per_rule)
    prompt_rows = [
        {
            "id": row["id"],
            "family": row["family"],
            "target_rule_class": row["target_rule_class"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_normalized_sha256": row["prompt_normalized_sha256"],
            "prompt": row["prompt"],
        }
        for row in rows
    ]
    forbidden = {"answer", "label", "target", "correct", "solution", "postprocessor_prediction", "simulated_wrong_prediction"}
    offenders = [row["id"] for row in prompt_rows if forbidden.intersection(row.keys())]
    if offenders:
        raise RuntimeError("prompt rows contain forbidden label-like keys: " + ",".join(offenders[:10]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.output_dir / f"{VERSION}_audit.csv"
    prompts_csv = args.output_dir / f"{VERSION}_prompts.csv"
    prompts_jsonl = args.output_dir / f"{VERSION}_prompts.jsonl"
    manifest_json = args.output_dir / f"{VERSION}_manifest.json"
    report_md = args.output_dir / "V461_SYNTHETIC_NUMERIC_PROBE_PACK.md"

    write_csv(audit_csv, rows, AUDIT_COLUMNS)
    write_csv(prompts_csv, prompt_rows, PROMPT_COLUMNS)
    write_jsonl(prompts_jsonl, prompt_rows)

    counts = Counter(str(row["target_rule_class"]) for row in rows)
    manifest = {
        "schema_version": "kg1_v461_synthetic_numeric_probe_pack_v1",
        "generated_at_utc": utc_now(),
        "version": VERSION,
        "source_policy": {
            "synthetic_prompts": True,
            "answers_in_prompt_pack": False,
            "weak_or_full_rows_used": False,
            "training": False,
            "inference": False,
            "submission": False,
            "purpose": "Create prompt-only synthetic probes to collect adapter raw outputs for multi-rule hard-negative mining.",
        },
        "summary": {
            "rows": len(rows),
            "rule_class_counts": dict(sorted(counts.items())),
            "hf_raw_probe_allowed": len(counts) >= 4 and all(counts[rule] >= min(4, args.max_per_rule) for rule in TARGET_RULES),
            "hf_gpu_train_allowed": False,
        },
        "outputs": {
            "audit_csv": str(audit_csv),
            "audit_csv_sha256": sha256_file(audit_csv),
            "prompts_csv": str(prompts_csv),
            "prompts_csv_sha256": sha256_file(prompts_csv),
            "prompts_jsonl": str(prompts_jsonl),
            "prompts_jsonl_sha256": sha256_file(prompts_jsonl),
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
        },
        "decision": {
            "decision": "v461_prompt_pack_ready_for_inference_only_raw_probe",
            "hf_raw_probe_allowed": True,
            "hf_gpu_train_allowed": False,
            "next_action": "Run inference-only adapter raw-output probe, then join labels locally and require multi-rule real hard negatives before any training.",
        },
    }
    write_json(manifest_json, manifest)

    lines = [
        "# V461 Synthetic Numeric Probe Pack",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Prompt rows | `{len(rows)}` |",
        f"| Rule classes | `{len(counts)}` |",
        f"| `hf_raw_probe_allowed` | `{manifest['summary']['hf_raw_probe_allowed']}` |",
        f"| `hf_gpu_train_allowed` | `{manifest['summary']['hf_gpu_train_allowed']}` |",
        "",
        "## Rule Counts",
        "",
        "| Rule class | Rows |",
        "|---|---:|",
    ]
    for rule, count in sorted(counts.items()):
        lines.append(f"| `{rule}` | `{count}` |")
    lines.extend(
        [
            "",
            "Answers are present only in the local audit CSV. The prompt JSONL used by HF raw inference has no label-like fields.",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print("audit_csv =", audit_csv, flush=True)
    print("prompts_jsonl =", prompts_jsonl, flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("rule_class_counts =", json.dumps(dict(sorted(counts.items())), sort_keys=True), flush=True)
    print("=== V461 SYNTHETIC NUMERIC PROBE PACK END ===", flush=True)
    return manifest


def self_test() -> None:
    rows = build_rows(4)
    counts = Counter(str(row["target_rule_class"]) for row in rows)
    assert set(counts) == set(TARGET_RULES), counts
    assert all(count >= 4 for count in counts.values()), counts
    for row in rows:
        assert verify_answer(row["answer"], row["postprocessor_prediction"])
        assert row["simulated_wrong_prediction"] != row["postprocessor_prediction"]
    print("v461_synthetic_numeric_probe_pack_self_test=ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-rule", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
