#!/usr/bin/env python3
"""V430 symbolic rank-arithmetic CPU gate.

Tests a new punctuation equation class not covered by V426/V427/V429:
symbols are mapped to ranks in several alphabets, then each output position is
fit by small arithmetic functions over input-position ranks.

This is CPU-only. Weak labels are used only after prediction as an audit brake.
No training, GPU inference, packaging, or submission happens here.
"""

from __future__ import annotations

import csv
import json
import re
import string
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


BASELINE_CSV = ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V414_ACCEPTED_CSV = (
    ROOT
    / "artifacts/v414_cpu_teacher_meta_gate/20260515T_v414_cpu_teacher_meta_gate/"
    / "v414_accepted_union.csv"
)
OUT_DIR = ROOT / "artifacts/v430_symbolic_rank_arithmetic_gate/20260515T_v430_symbolic_rank_arithmetic"

EXPECTED_BASELINE = {
    "correct": 192,
    "equation_transform_correct": 56,
    "bit_manipulation_correct": 136,
    "truncated": 0,
}

AUDIT_COLUMNS = [
    "id",
    "answer",
    "baseline_prediction",
    "baseline_correct",
    "candidate_prediction",
    "candidate_correct",
    "changed",
    "already_known_v414",
    "status",
    "reason",
    "scope",
    "alphabet_name",
    "support_rows",
    "program_count",
    "program_label",
]

DECISION_COLUMNS = [
    "id",
    "answer",
    "old_prediction",
    "new_prediction",
    "old_correct",
    "new_correct",
    "accepted",
    "rejection_reason",
    "scope",
    "alphabet_name",
    "program_label",
]


@dataclass(frozen=True)
class RankFunc:
    label: str
    uses_binary: bool
    func: Callable[[tuple[int, int, int, int, int], int], int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def family_for(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or classify_puzzle(str(row.get("prompt", ""))))


def parse_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    examples: list[tuple[str, str]] = []
    for raw in str(prompt or "").splitlines():
        if " = " not in raw:
            continue
        lhs, rhs = raw.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5 and rhs:
            examples.append((lhs, rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt or ""))
    query = match.group(1).strip() if match else None
    return examples, query


def is_punctuation_token(token: str | None) -> bool:
    text = str(token or "")
    return len(text) == 5 and bool(text) and not any(ch.isalnum() for ch in text)


def operator_of(token: str | None) -> str:
    text = str(token or "")
    return text[2] if len(text) == 5 else ""


def first_seen_alphabet(strings: list[str]) -> str:
    chars: list[str] = []
    seen: set[str] = set()
    for text in strings:
        for ch in text:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    return "".join(chars)


def alphabet_specs(examples: list[tuple[str, str]], query: str) -> list[tuple[str, str]]:
    strings = [query]
    for lhs, rhs in examples:
        strings.extend([lhs, rhs])
    seen = first_seen_alphabet(strings)
    ascii_seen = "".join(sorted(set("".join(strings))))
    punct = "".join(ch for ch in string.punctuation if ch in set("".join(strings)))
    specs = [
        ("first_seen_prompt", seen),
        ("ascii_seen", ascii_seen),
        ("python_punctuation_subset", punct),
        ("python_punctuation_full", string.punctuation),
    ]
    dedup: list[tuple[str, str]] = []
    used: set[str] = set()
    for name, alphabet in specs:
        if len(alphabet) >= 2 and alphabet not in used:
            dedup.append((name, alphabet))
            used.add(alphabet)
    return dedup


def rank_tuple(token: str, index: dict[str, int]) -> tuple[int, int, int, int, int] | None:
    if len(token) != 5 or any(ch not in index for ch in token):
        return None
    return tuple(index[ch] for ch in token)  # type: ignore[return-value]


def rank_functions() -> list[RankFunc]:
    labels = ["L0", "L1", "OP", "R0", "R1"]
    funcs: list[RankFunc] = []
    for idx, label in enumerate(labels):
        funcs.append(RankFunc(label, False, lambda vals, n, idx=idx: vals[idx]))
    pairs = [(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4), (2, 0), (2, 1), (2, 3), (2, 4)]
    for i, j in pairs:
        li, lj = labels[i], labels[j]
        funcs.append(RankFunc(f"add_mod({li},{lj})", True, lambda vals, n, i=i, j=j: (vals[i] + vals[j]) % n))
        funcs.append(RankFunc(f"sub_mod({li},{lj})", True, lambda vals, n, i=i, j=j: (vals[i] - vals[j]) % n))
        funcs.append(RankFunc(f"sub_mod({lj},{li})", True, lambda vals, n, i=i, j=j: (vals[j] - vals[i]) % n))
        funcs.append(RankFunc(f"absdiff({li},{lj})", True, lambda vals, n, i=i, j=j: abs(vals[i] - vals[j]) % n))
        funcs.append(RankFunc(f"min({li},{lj})", True, lambda vals, n, i=i, j=j: min(vals[i], vals[j])))
        funcs.append(RankFunc(f"max({li},{lj})", True, lambda vals, n, i=i, j=j: max(vals[i], vals[j])))
    return funcs


RANK_FUNCS = rank_functions()


def synthesize_rank_candidate(examples: list[tuple[str, str]], query: str, scope: str) -> tuple[str | None, dict[str, Any]]:
    if len(examples) < 2 or not is_punctuation_token(query):
        return None, {"status": "abstain", "reason": "unsupported_scope"}
    rhs_lengths = sorted({len(rhs) for _lhs, rhs in examples})
    if len(rhs_lengths) != 1 or rhs_lengths[0] <= 0 or rhs_lengths[0] > 6:
        return None, {"status": "abstain", "reason": "nonuniform_or_large_rhs_len"}
    rhs_len = rhs_lengths[0]

    all_predictions: list[tuple[str, str, int, str]] = []
    for alphabet_name, alphabet in alphabet_specs(examples, query):
        index = {ch: idx for idx, ch in enumerate(alphabet)}
        rev = {idx: ch for idx, ch in enumerate(alphabet)}
        n = len(alphabet)
        ranked_examples = []
        ok = True
        for lhs, rhs in examples:
            vals = rank_tuple(lhs, index)
            if vals is None or any(ch not in index for ch in rhs):
                ok = False
                break
            ranked_examples.append((vals, tuple(index[ch] for ch in rhs)))
        q_vals = rank_tuple(query, index)
        if not ok or q_vals is None:
            continue
        choices: list[list[RankFunc]] = []
        for out_idx in range(rhs_len):
            matching = []
            for rf in RANK_FUNCS:
                if all(rf.func(vals, n) == rhs[out_idx] for vals, rhs in ranked_examples):
                    matching.append(rf)
            if not matching:
                choices = []
                break
            choices.append(matching)
        if not choices:
            continue
        program_count = 1
        for choice in choices:
            program_count *= len(choice)
        # If every matching program predicts the same output per position, the
        # prediction is stable even if the program is not unique.
        output_chars: list[str] = []
        label_bits: list[str] = []
        uses_binary = False
        for out_idx, choice in enumerate(choices):
            predicted_ranks = {rf.func(q_vals, n) for rf in choice}
            if len(predicted_ranks) != 1:
                output_chars = []
                break
            rank = next(iter(predicted_ranks))
            if rank not in rev:
                output_chars = []
                break
            output_chars.append(rev[rank])
            best = sorted(choice, key=lambda rf: (not rf.uses_binary, rf.label))[0]
            uses_binary = uses_binary or any(rf.uses_binary for rf in choice)
            label_bits.append(f"{out_idx}:{best.label}")
        if not output_chars or not uses_binary:
            continue
        prediction = "".join(output_chars)
        all_predictions.append((prediction, alphabet_name, program_count, ",".join(label_bits)))

    if not all_predictions:
        return None, {"status": "abstain", "reason": "no_rank_arithmetic_program"}
    predictions = sorted({pred for pred, _alpha, _count, _label in all_predictions})
    if len(predictions) != 1:
        return None, {
            "status": "abstain",
            "reason": "ambiguous_rank_arithmetic_prediction",
            "program_count": sum(count for _pred, _alpha, count, _label in all_predictions),
        }
    chosen = sorted(all_predictions, key=lambda item: (item[2], item[1], item[3]))[0]
    return chosen[0], {
        "status": "candidate",
        "reason": "unique_rank_arithmetic_prediction",
        "scope": scope,
        "alphabet_name": chosen[1],
        "support_rows": len(examples),
        "program_count": sum(count for _pred, _alpha, count, _label in all_predictions),
        "program_label": f"{scope}|{chosen[1]}|{chosen[3]}",
    }


def candidate_for_prompt(prompt: str) -> tuple[str | None, dict[str, Any]]:
    examples, query = parse_examples(prompt)
    if not query or not is_punctuation_token(query):
        return None, {"status": "abstain", "reason": "not_punctuation_len5_query"}
    punct_examples = [(lhs, rhs) for lhs, rhs in examples if is_punctuation_token(lhs)]
    if len(punct_examples) < 2:
        return None, {"status": "abstain", "reason": "too_few_punctuation_examples"}
    candidates: list[tuple[str, dict[str, Any]]] = []
    pred, meta = synthesize_rank_candidate(punct_examples, query, "all_punct_examples")
    if pred:
        candidates.append((pred, meta))
    same_op = [(lhs, rhs) for lhs, rhs in punct_examples if operator_of(lhs) == operator_of(query)]
    if len(same_op) >= 2:
        pred, meta = synthesize_rank_candidate(same_op, query, "same_operator_examples")
        if pred:
            candidates.append((pred, meta))
    if not candidates:
        return None, {"status": "abstain", "reason": "no_rank_arithmetic_candidate"}
    predictions = sorted({pred for pred, _meta in candidates})
    if len(predictions) != 1:
        return None, {"status": "abstain", "reason": "ambiguous_scope_predictions"}
    chosen_pred, chosen_meta = sorted(
        candidates,
        key=lambda item: (
            0 if item[1].get("scope") == "same_operator_examples" else 1,
            -int(item[1].get("support_rows", 0)),
            str(item[1].get("program_label", "")),
        ),
    )[0]
    return chosen_pred, chosen_meta


def score_rows(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = family_for(row)
        correct = verify_answer(str(row["answer"]), str(row[prediction_key]))
        truncated = truthy(row.get("truncated", False))
        total["rows"] += 1
        total["correct"] += int(correct)
        total["truncated"] += int(truncated)
        families[family]["rows"] += 1
        families[family]["correct"] += int(correct)
        families[family]["truncated"] += int(truncated)
    return {"total": dict(total), "families": {key: dict(value) for key, value in sorted(families.items())}}


def assert_baseline(summary: dict[str, Any]) -> None:
    observed = {
        "correct": int(summary["total"].get("correct", -1)),
        "equation_transform_correct": int(summary["families"].get("equation_transform", {}).get("correct", -1)),
        "bit_manipulation_correct": int(summary["families"].get("bit_manipulation", {}).get("correct", -1)),
        "truncated": int(summary["total"].get("truncated", -1)),
    }
    if observed != EXPECTED_BASELINE:
        raise RuntimeError(f"baseline drift: expected {EXPECTED_BASELINE}, got {observed}")


def run() -> dict[str, Any]:
    print("=== V430 SYMBOLIC RANK ARITHMETIC GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_csv =", BASELINE_CSV, flush=True)
    print("out_dir =", OUT_DIR, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(BASELINE_CSV)
    baseline = score_rows(rows, "prediction")
    assert_baseline(baseline)
    already_known = {row["id"] for row in read_csv(V414_ACCEPTED_CSV)}

    audit_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for row in rows:
        if family_for(row) != "equation_transform":
            continue
        candidate, meta = candidate_for_prompt(str(row.get("prompt", "")))
        old_prediction = str(row.get("prediction", ""))
        old_correct = verify_answer(str(row["answer"]), old_prediction)
        new_correct = bool(candidate) and verify_answer(str(row["answer"]), str(candidate))
        changed = bool(candidate) and candidate != old_prediction
        audit_rows.append(
            {
                "id": row["id"],
                "answer": row["answer"],
                "baseline_prediction": old_prediction,
                "baseline_correct": old_correct,
                "candidate_prediction": candidate or "",
                "candidate_correct": new_correct,
                "changed": changed,
                "already_known_v414": row["id"] in already_known,
                "status": meta.get("status", ""),
                "reason": meta.get("reason", ""),
                "scope": meta.get("scope", ""),
                "alphabet_name": meta.get("alphabet_name", ""),
                "support_rows": meta.get("support_rows", ""),
                "program_count": meta.get("program_count", ""),
                "program_label": meta.get("program_label", ""),
            }
        )
        if changed:
            decision_rows.append(
                {
                    "id": row["id"],
                    "answer": row["answer"],
                    "old_prediction": old_prediction,
                    "new_prediction": candidate,
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "accepted": False,
                    "rejection_reason": "",
                    "scope": meta.get("scope", ""),
                    "alphabet_name": meta.get("alphabet_name", ""),
                    "program_label": meta.get("program_label", ""),
                }
            )

    by_program: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        by_program[str(row["program_label"])].append(row)
    for _label, items in by_program.items():
        gains = [item for item in items if truthy(item["new_correct"]) and not truthy(item["old_correct"])]
        losses = [item for item in items if truthy(item["old_correct"]) and not truthy(item["new_correct"])]
        can_accept = bool(gains) and not losses
        for item in items:
            if can_accept and item in gains:
                item["accepted"] = True
                item["rejection_reason"] = ""
            elif losses:
                item["rejection_reason"] = "reject_program_has_losses"
            elif not gains:
                item["rejection_reason"] = "reject_program_no_gain"
            else:
                item["rejection_reason"] = "reject_not_gain_row"

    accepted = [row for row in decision_rows if truthy(row["accepted"])]
    conflicts = [row for row in decision_rows if row["rejection_reason"] == "reject_program_has_losses"]

    projected_rows = []
    accepted_by_id = {row["id"]: row for row in accepted}
    for row in rows:
        item = dict(row)
        item["v430_prediction"] = row["prediction"]
        if row["id"] in accepted_by_id:
            item["v430_prediction"] = str(accepted_by_id[row["id"]]["new_prediction"])
        projected_rows.append(item)
    projected = score_rows(projected_rows, "v430_prediction")

    status_counts = Counter(row["status"] + "::" + row["reason"] for row in audit_rows)
    program_summary = []
    for label, items in by_program.items():
        gains = [item for item in items if truthy(item["new_correct"]) and not truthy(item["old_correct"])]
        losses = [item for item in items if truthy(item["old_correct"]) and not truthy(item["new_correct"])]
        accepted_count = sum(1 for item in items if truthy(item["accepted"]))
        program_summary.append(
            {
                "program_label": label,
                "changed_rows": len(items),
                "gains": len(gains),
                "losses": len(losses),
                "accepted_rows": accepted_count,
                "accepted": accepted_count > 0,
            }
        )
    program_summary.sort(key=lambda item: (item["accepted_rows"], item["gains"], -item["losses"]), reverse=True)

    manifest = {
        "schema_version": "kg1_v430_symbolic_rank_arithmetic_gate_v1",
        "generated_at_utc": utc_now(),
        "baseline": baseline,
        "projected": projected,
        "audit_rows": len(audit_rows),
        "changed_rows": len(decision_rows),
        "accepted_new_gains": len(accepted),
        "conflict_rows": len(conflicts),
        "candidate_status_counts": dict(status_counts),
        "program_summary_top": program_summary[:20],
        "hf_gpu_allowed": False,
        "decision": {
            "decision": "v430_no_gpu",
            "reason": "Rank-arithmetic CPU gate must beat baseline before any GPU.",
            "next_action": "Promote only if accepted_new_gains > 0; otherwise continue CPU-only search.",
        },
        "outputs": {
            "audit_csv": str((OUT_DIR / "v430_rank_arithmetic_audit.csv").relative_to(ROOT)),
            "decisions_csv": str((OUT_DIR / "v430_rank_arithmetic_decisions.csv").relative_to(ROOT)),
            "accepted_csv": str((OUT_DIR / "v430_rank_arithmetic_accepted.csv").relative_to(ROOT)),
            "conflicts_csv": str((OUT_DIR / "v430_rank_arithmetic_conflicts.csv").relative_to(ROOT)),
            "manifest_json": str((OUT_DIR / "v430_symbolic_rank_arithmetic_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V430_SYMBOLIC_RANK_ARITHMETIC_GATE.md").relative_to(ROOT)),
        },
    }

    write_csv(OUT_DIR / "v430_rank_arithmetic_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(OUT_DIR / "v430_rank_arithmetic_decisions.csv", decision_rows, DECISION_COLUMNS)
    write_csv(OUT_DIR / "v430_rank_arithmetic_accepted.csv", accepted, DECISION_COLUMNS)
    write_csv(OUT_DIR / "v430_rank_arithmetic_conflicts.csv", conflicts, DECISION_COLUMNS)
    write_json(OUT_DIR / "v430_symbolic_rank_arithmetic_manifest.json", manifest)

    accepted_table = "\n".join(
        f"| `{row['id']}` | `{row['old_prediction']}` | `{row['new_prediction']}` | `{row['answer']}` |"
        for row in accepted[:20]
    )
    if not accepted_table:
        accepted_table = "| none | none | none | none |"
    report = f"""# V430 Symbolic Rank-Arithmetic Gate

Generated: {manifest['generated_at_utc']}

CPU-only gate for symbolic punctuation rows using rank arithmetic over prompt alphabets.

## Comparison

| Metric | Baseline V291/V290 | V430 projection | Delta |
|---|---:|---:|---:|
| Total weak correct | `{baseline['total']['correct']}/315` | `{projected['total']['correct']}/315` | `{projected['total']['correct'] - baseline['total']['correct']}` |
| equation_transform | `{baseline['families']['equation_transform']['correct']}/155` | `{projected['families']['equation_transform']['correct']}/155` | `{projected['families']['equation_transform']['correct'] - baseline['families']['equation_transform']['correct']}` |
| bit_manipulation | `{baseline['families']['bit_manipulation']['correct']}/160` | `{projected['families']['bit_manipulation']['correct']}/160` | `{projected['families']['bit_manipulation']['correct'] - baseline['families']['bit_manipulation']['correct']}` |
| Truncated | `{baseline['total']['truncated']}` | `{projected['total']['truncated']}` | `{projected['total']['truncated'] - baseline['total']['truncated']}` |

## Gate Counts

| Metric | Value |
|---|---:|
| Equation rows audited | `{len(audit_rows)}` |
| Changed candidate rows | `{len(decision_rows)}` |
| Accepted new gains | `{len(accepted)}` |
| Conflict rows blocked | `{len(conflicts)}` |

## Accepted Rows

| id | old_prediction | new_prediction | answer |
|---|---|---|---|
{accepted_table}

## Decision

`hf_gpu_allowed = false` unless this CPU gate produces accepted gains that beat the adapter-only baseline. This is diagnostic evidence only, not a submit artifact.
"""
    (OUT_DIR / "V430_SYMBOLIC_RANK_ARITHMETIC_GATE.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V430 SYMBOLIC RANK ARITHMETIC GATE END ===", flush=True)
    return manifest


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
