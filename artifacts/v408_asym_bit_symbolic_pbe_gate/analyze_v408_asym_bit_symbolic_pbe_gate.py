#!/usr/bin/env python3
"""V408 CPU gate for asymmetric bit rules and symbolic equation PBE.

This script is diagnostic-only. It reads the locked V290 checkpoint-6 weak
predictions, generates abstaining solver candidates, and accepts only candidates
that are verified by weak labels without regressing an already-correct row.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402
from src.solvers.bit_manipulation_solver import parse_bit_problem  # noqa: E402
from scripts.run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    build_audit_row,
    classify_subtype,
    numeric_candidate,
    parse_alice_prompt,
    symbolic_candidates,
)


BASELINE_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = REPO_ROOT / "artifacts/v408_asym_bit_symbolic_pbe_gate/20260514T_v408_cpu_gate"


BitFunc = Callable[..., int]


BIT_UNARY: list[tuple[str, BitFunc]] = [
    ("ID", lambda a: a),
    ("NOT", lambda a: 1 - a),
]

BIT_BINARY: list[tuple[str, BitFunc]] = [
    ("AND", lambda a, b: a & b),
    ("OR", lambda a, b: a | b),
    ("XOR", lambda a, b: a ^ b),
    ("XNOR", lambda a, b: 1 - (a ^ b)),
    ("NAND", lambda a, b: 1 - (a & b)),
    ("NOR", lambda a, b: 1 - (a | b)),
    ("INHIB", lambda a, b: a & (1 - b)),
    ("RINHIB", lambda a, b: (1 - a) & b),
    ("IMPL", lambda a, b: (1 - a) | b),
    ("RIMPL", lambda a, b: a | (1 - b)),
]

BIT_TERNARY: list[tuple[str, BitFunc]] = [
    ("MAJ", lambda a, b, c: 1 if a + b + c >= 2 else 0),
    ("CH", lambda a, b, c: (a & b) | ((1 - a) & c)),
    ("XOR3", lambda a, b, c: a ^ b ^ c),
]


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


def bits(text: str) -> list[int]:
    return [1 if ch == "1" else 0 for ch in str(text).strip()]


def bit_candidates_for_position(
    inputs: list[list[int]],
    outputs: list[list[int]],
    query_bits: list[int],
    out_pos: int,
    *,
    allow_ternary: bool,
) -> list[dict[str, Any]]:
    expected = [row[out_pos] for row in outputs]
    candidates: list[dict[str, Any]] = []
    if all(value == 0 for value in expected):
        candidates.append({"rule": "C0", "prediction_bit": 0, "arity": 0})
    if all(value == 1 for value in expected):
        candidates.append({"rule": "C1", "prediction_bit": 1, "arity": 0})
    for i in range(8):
        for name, func in BIT_UNARY:
            if all(func(row[i]) == expected[idx] for idx, row in enumerate(inputs)):
                candidates.append({"rule": f"{name}({i})", "prediction_bit": func(query_bits[i]), "arity": 1})
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            for name, func in BIT_BINARY:
                if all(func(row[i], row[j]) == expected[idx] for idx, row in enumerate(inputs)):
                    candidates.append(
                        {"rule": f"{name}({i},{j})", "prediction_bit": func(query_bits[i], query_bits[j]), "arity": 2}
                    )
    if allow_ternary:
        for i, j, k in itertools.permutations(range(8), 3):
            for name, func in BIT_TERNARY:
                if all(func(row[i], row[j], row[k]) == expected[idx] for idx, row in enumerate(inputs)):
                    candidates.append(
                        {
                            "rule": f"{name}({i},{j},{k})",
                            "prediction_bit": func(query_bits[i], query_bits[j], query_bits[k]),
                            "arity": 3,
                        }
                    )
    return candidates


def solve_bit_unique_value(prompt: str, *, allow_ternary: bool, max_candidates_per_bit: int) -> tuple[str | None, dict[str, Any]]:
    examples, query = parse_bit_problem(prompt)
    if not examples or not query:
        return None, {"status": "abstain", "reason": "parse_failed"}
    inputs = [bits(lhs) for lhs, _ in examples]
    outputs = [bits(rhs) for _, rhs in examples]
    query_bits = bits(query)
    result_bits: list[str] = []
    proofs: list[str] = []
    candidate_counts: list[int] = []
    max_arity = 0
    for out_pos in range(8):
        candidates = bit_candidates_for_position(
            inputs,
            outputs,
            query_bits,
            out_pos,
            allow_ternary=allow_ternary,
        )
        candidate_counts.append(len(candidates))
        if not candidates:
            return None, {
                "status": "abstain",
                "reason": f"no_candidates_bit_{out_pos}",
                "candidate_counts": candidate_counts,
            }
        if len(candidates) > max_candidates_per_bit:
            return None, {
                "status": "abstain",
                "reason": f"too_many_candidates_bit_{out_pos}",
                "candidate_counts": candidate_counts,
            }
        predicted_values = sorted({int(item["prediction_bit"]) for item in candidates})
        if len(predicted_values) != 1:
            return None, {
                "status": "abstain",
                "reason": f"ambiguous_prediction_bit_{out_pos}",
                "candidate_counts": candidate_counts,
            }
        max_arity = max(max_arity, max(int(item["arity"]) for item in candidates))
        result_bits.append(str(predicted_values[0]))
        proofs.append(f"b{out_pos}:{'|'.join(item['rule'] for item in candidates[:6])}")
    return "".join(result_bits), {
        "status": "candidate",
        "reason": "unique_value_per_bit",
        "candidate_counts": candidate_counts,
        "max_arity": max_arity,
        "proof": "; ".join(proofs),
    }


def equation_candidates_for_row(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    examples, query, parse_status = parse_alice_prompt(str(row["prompt"]))
    if parse_status != "ok":
        return [
            build_audit_row(
                row,
                {"rule_class": "alice_parse_gate", "status": "abstain", "prediction": "", "proof": parse_status},
                examples,
                query,
            )
        ]
    helper_args = SimpleNamespace(
        pair_mapping_cap=args.pair_mapping_cap,
        global_mapping_cap=args.global_mapping_cap,
        max_char_subset_size=args.max_char_subset_size,
        max_position_sources=args.max_position_sources,
        min_same_operator_examples=args.min_same_operator_examples,
    )
    if classify_subtype(examples, query) == "equation_numeric_operator":
        results = [numeric_candidate(examples, query, args.min_same_operator_examples)]
    else:
        results = symbolic_candidates(examples, query, helper_args)
    return [build_audit_row(row, result, examples, query) for result in results]


def score(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = classify_puzzle(row["prompt"])
        correct = verify_answer(row["answer"], row[key])
        total["rows"] += 1
        total["correct"] += int(correct)
        by_family[family]["rows"] += 1
        by_family[family]["correct"] += int(correct)
    return {
        "total": dict(total),
        "families": {family: dict(counter) for family, counter in sorted(by_family.items())},
    }


def main() -> int:
    args = parse_args()
    rows = read_csv(args.baseline_csv)
    for row in rows:
        row["family"] = classify_puzzle(row["prompt"])
        row["baseline_correct"] = verify_answer(row["answer"], row["prediction"])
        row["v408_prediction"] = row["prediction"]

    candidate_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for row in rows:
        if row["family"] == "bit_manipulation":
            candidate, meta = solve_bit_unique_value(
                row["prompt"],
                allow_ternary=args.allow_ternary_bit,
                max_candidates_per_bit=args.max_candidates_per_bit,
            )
            if candidate is None:
                candidate_rows.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "source": "v408_bit_unique_value",
                        "status": "abstain",
                        "prediction": "",
                        "answer": row["answer"],
                        "baseline_prediction": row["prediction"],
                        "baseline_correct": row["baseline_correct"],
                        "candidate_correct": False,
                        "accepted": False,
                        "reason": meta.get("reason", ""),
                        "proof": json.dumps(meta, sort_keys=True)[:700],
                    }
                )
                continue
            candidate_correct = verify_answer(row["answer"], candidate)
            accepted = (not row["baseline_correct"]) and candidate_correct
            if row["baseline_correct"] and not candidate_correct:
                conflicts.append({"id": row["id"], "family": row["family"], "reason": "would_regress", "prediction": candidate})
                accepted = False
            if accepted:
                row["v408_prediction"] = candidate
            candidate_rows.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "source": "v408_bit_unique_value",
                    "status": "candidate",
                    "prediction": candidate,
                    "answer": row["answer"],
                    "baseline_prediction": row["prediction"],
                    "baseline_correct": row["baseline_correct"],
                    "candidate_correct": candidate_correct,
                    "accepted": accepted,
                    "reason": meta.get("reason", ""),
                    "proof": meta.get("proof", ""),
                }
            )
            if accepted:
                accepted_rows.append(candidate_rows[-1])
        elif row["family"] == "equation_transform":
            eq_candidates = equation_candidates_for_row(row, args)
            by_prediction: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in eq_candidates:
                prediction = str(item.get("prediction", "")).strip()
                if item.get("status") == "candidate" and prediction:
                    by_prediction[prediction].append(item)
                candidate_rows.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "source": "v408_equation_symbolic_pbe",
                        "status": item.get("status", ""),
                        "prediction": prediction,
                        "answer": row["answer"],
                        "baseline_prediction": row["prediction"],
                        "baseline_correct": row["baseline_correct"],
                        "candidate_correct": verify_answer(row["answer"], prediction) if prediction else False,
                        "accepted": False,
                        "reason": item.get("rule_class", ""),
                        "proof": item.get("proof", ""),
                    }
                )
            valid_predictions = sorted(by_prediction)
            if len(valid_predictions) != 1:
                continue
            candidate = valid_predictions[0]
            candidate_correct = verify_answer(row["answer"], candidate)
            if row["baseline_correct"] and not candidate_correct:
                conflicts.append({"id": row["id"], "family": row["family"], "reason": "would_regress", "prediction": candidate})
                continue
            if (not row["baseline_correct"]) and candidate_correct:
                row["v408_prediction"] = candidate
                accepted = {
                    "id": row["id"],
                    "family": row["family"],
                    "source": "v408_equation_symbolic_pbe",
                    "status": "candidate",
                    "prediction": candidate,
                    "answer": row["answer"],
                    "baseline_prediction": row["prediction"],
                    "baseline_correct": row["baseline_correct"],
                    "candidate_correct": True,
                    "accepted": True,
                    "reason": ";".join(sorted({str(item.get("rule_class", "")) for item in by_prediction[candidate]})),
                    "proof": " | ".join(str(item.get("proof", "")) for item in by_prediction[candidate])[:700],
                }
                accepted_rows.append(accepted)

    baseline_score = score(rows, "prediction")
    v408_score = score(rows, "v408_prediction")
    out = args.output_dir
    columns = [
        "id",
        "family",
        "source",
        "status",
        "prediction",
        "answer",
        "baseline_prediction",
        "baseline_correct",
        "candidate_correct",
        "accepted",
        "reason",
        "proof",
    ]
    write_csv(out / "v408_asym_bit_symbolic_pbe_candidates.csv", candidate_rows, columns)
    write_csv(out / "v408_asym_bit_symbolic_pbe_accepted.csv", accepted_rows, columns)
    write_csv(out / "v408_asym_bit_symbolic_pbe_conflicts.csv", conflicts, ["id", "family", "reason", "prediction"])

    manifest = {
        "schema_version": "kg1_v408_asym_bit_symbolic_pbe_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"baseline_csv": str(args.baseline_csv)},
        "config": {
            "allow_ternary_bit": args.allow_ternary_bit,
            "max_candidates_per_bit": args.max_candidates_per_bit,
            "pair_mapping_cap": args.pair_mapping_cap,
            "global_mapping_cap": args.global_mapping_cap,
            "max_char_subset_size": args.max_char_subset_size,
            "max_position_sources": args.max_position_sources,
            "min_same_operator_examples": args.min_same_operator_examples,
        },
        "baseline_score": baseline_score,
        "v408_score": v408_score,
        "accepted_gain_count": len(accepted_rows),
        "conflict_count": len(conflicts),
        "accepted_by_source": dict(Counter(row["source"] for row in accepted_rows)),
        "decision": (
            "v408_cpu_signal_found_conflicts_blocked_not_adapter_submit_safe"
            if accepted_rows
            else "v408_no_new_safe_cpu_signal"
        ),
        "outputs": {
            "candidates_csv": str(out / "v408_asym_bit_symbolic_pbe_candidates.csv"),
            "accepted_csv": str(out / "v408_asym_bit_symbolic_pbe_accepted.csv"),
            "conflicts_csv": str(out / "v408_asym_bit_symbolic_pbe_conflicts.csv"),
            "manifest_json": str(out / "v408_asym_bit_symbolic_pbe_manifest.json"),
        },
    }
    write_json(out / "v408_asym_bit_symbolic_pbe_manifest.json", manifest)

    fam_base = baseline_score["families"]
    fam_new = v408_score["families"]
    report = [
        "# V408 Asymmetric Bit and Symbolic PBE CPU Gate",
        "",
        "| Metric | Baseline | V408 projection | Delta |",
        "|---|---:|---:|---:|",
        f"| Weak total | `{baseline_score['total']['correct']}/315` | `{v408_score['total']['correct']}/315` | `+{v408_score['total']['correct'] - baseline_score['total']['correct']}` |",
        f"| equation_transform | `{fam_base['equation_transform']['correct']}/155` | `{fam_new['equation_transform']['correct']}/155` | `+{fam_new['equation_transform']['correct'] - fam_base['equation_transform']['correct']}` |",
        f"| bit_manipulation | `{fam_base['bit_manipulation']['correct']}/160` | `{fam_new['bit_manipulation']['correct']}/160` | `+{fam_new['bit_manipulation']['correct'] - fam_base['bit_manipulation']['correct']}` |",
        "",
        f"- Accepted gains: `{len(accepted_rows)}`.",
        f"- Conflicts/losses blocked: `{len(conflicts)}`.",
        "",
        "This is a CPU solver/verifier projection only. It is not an adapter-only Kaggle submission.",
        "",
        "## Accepted Gains",
        "",
    ]
    for item in accepted_rows:
        report.append(
            f"- `{item['id']}` `{item['family']}`: `{item['baseline_prediction']}` -> `{item['prediction']}` via `{item['source']}` / `{item['reason']}`"
        )
    (out / "V408_ASYM_BIT_SYMBOLIC_PBE_GATE.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--allow-ternary-bit", action="store_true")
    parser.add_argument("--max-candidates-per-bit", type=int, default=64)
    parser.add_argument("--pair-mapping-cap", type=int, default=3000)
    parser.add_argument("--global-mapping-cap", type=int, default=12000)
    parser.add_argument("--max-char-subset-size", type=int, default=4)
    parser.add_argument("--max-position-sources", type=int, default=7)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
