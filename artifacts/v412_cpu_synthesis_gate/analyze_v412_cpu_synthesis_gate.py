#!/usr/bin/env python3
"""V412 CPU synthesis gate for bit_manipulation and equation_transform.

This gate implements the next roadmap step after V411B/V411C:

* bit_manipulation: per-output-bit LUT search for k=1/2/3 input bits,
  with abstention unless all accepted LUT candidates agree on the query bit.
* equation_transform: FlashFill/VSA-style program ranking over small symbolic
  and numeric DSL programs, with undefined/OOD-format penalties and abstention
  on ambiguous top predictions.

The script is CPU-only and diagnostic. Weak labels are used only to audit
candidate gains/losses and to decide whether another GPU transfer run is worth
the cost. It does not package or submit anything.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402
from src.solvers.bit_manipulation_solver import parse_bit_problem  # noqa: E402
from scripts.run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    parse_alice_prompt,
    parse_numeric_token,
)
from scripts.analyze_v241_abstain_rule_candidate_audit import (  # noqa: E402
    merge_mapping_sets,
    numeric_rule_functions as v241_numeric_rule_functions,
    transducer_mappings_for_pair,
)


BASELINE_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V409_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v409_integrated_solver_projection_v2/20260514T_v409_integrated_projection_v2/"
    / "v409_integrated_solver_accepted.csv"
)
OUT_DIR = REPO_ROOT / "artifacts/v412_cpu_synthesis_gate/20260514T_v412_cpu_gate"


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


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def score(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        correct = verify_answer(row["answer"], row[prediction_key])
        total["rows"] += 1
        total["correct"] += int(correct)
        by_family[family]["rows"] += 1
        by_family[family]["correct"] += int(correct)
    return {
        "total": dict(total),
        "families": {family: dict(counter) for family, counter in sorted(by_family.items())},
    }


def bits(text: str) -> list[int]:
    return [1 if ch == "1" else 0 for ch in str(text).strip()]


def solve_bit_lut_unique(
    prompt: str,
    *,
    max_k: int,
    max_total_candidates_per_bit: int,
    max_accepted_candidates_per_bit: int,
) -> tuple[str | None, dict[str, Any]]:
    examples, query = parse_bit_problem(prompt)
    if not examples or not query:
        return None, {"status": "abstain", "reason": "parse_failed"}
    input_bits = [bits(lhs) for lhs, _ in examples]
    output_bits = [bits(rhs) for _, rhs in examples]
    query_bits = bits(query)
    result_bits: list[str] = []
    proofs: list[str] = []
    counts: list[int] = []
    accepted_counts: list[int] = []

    for out_pos in range(8):
        candidates: list[dict[str, Any]] = []
        total_consistent = 0
        for k in range(1, max_k + 1):
            for positions in itertools.combinations(range(8), k):
                table: dict[tuple[int, ...], int] = {}
                consistent = True
                for row_idx, row_bits in enumerate(input_bits):
                    key = tuple(row_bits[pos] for pos in positions)
                    expected = output_bits[row_idx][out_pos]
                    if key in table and table[key] != expected:
                        consistent = False
                        break
                    table[key] = expected
                if not consistent:
                    continue
                total_consistent += 1
                query_key = tuple(query_bits[pos] for pos in positions)
                if query_key not in table:
                    continue
                pred = table[query_key]
                # Leave-one-out stability: if an omitted example key is still
                # present in the remaining table, the prediction must match it.
                loo_checks = 0
                loo_pass = 0
                for omit_idx, row_bits in enumerate(input_bits):
                    train_table: dict[tuple[int, ...], int] = {}
                    ok = True
                    for row_idx, train_bits in enumerate(input_bits):
                        if row_idx == omit_idx:
                            continue
                        key = tuple(train_bits[pos] for pos in positions)
                        expected = output_bits[row_idx][out_pos]
                        if key in train_table and train_table[key] != expected:
                            ok = False
                            break
                        train_table[key] = expected
                    if not ok:
                        continue
                    held_key = tuple(row_bits[pos] for pos in positions)
                    if held_key in train_table:
                        loo_checks += 1
                        loo_pass += int(train_table[held_key] == output_bits[omit_idx][out_pos])
                score_value = 10 * k - 2 * loo_pass + (0 if loo_checks else 3)
                candidates.append(
                    {
                        "positions": positions,
                        "prediction_bit": pred,
                        "score": score_value,
                        "loo_checks": loo_checks,
                        "loo_pass": loo_pass,
                    }
                )
        counts.append(total_consistent)
        accepted_counts.append(len(candidates))
        if total_consistent > max_total_candidates_per_bit:
            return None, {
                "status": "abstain",
                "reason": f"too_many_total_lut_candidates_bit_{out_pos}",
                "consistent_counts": counts,
                "accepted_counts": accepted_counts,
            }
        if not candidates:
            return None, {
                "status": "abstain",
                "reason": f"no_query_seen_lut_candidate_bit_{out_pos}",
                "consistent_counts": counts,
                "accepted_counts": accepted_counts,
            }
        candidates.sort(key=lambda item: (item["score"], len(item["positions"]), item["positions"]))
        top_score = candidates[0]["score"]
        top = [item for item in candidates if item["score"] == top_score]
        if len(candidates) > max_accepted_candidates_per_bit:
            # Ranking can still pass if all best-scoring candidates agree.
            ranked_pool = top
        else:
            ranked_pool = candidates
        predicted_values = sorted({int(item["prediction_bit"]) for item in ranked_pool})
        if len(predicted_values) != 1:
            return None, {
                "status": "abstain",
                "reason": f"ambiguous_lut_prediction_bit_{out_pos}",
                "consistent_counts": counts,
                "accepted_counts": accepted_counts,
                "top_score": top_score,
                "top_predictions": predicted_values,
            }
        result_bits.append(str(predicted_values[0]))
        proofs.append(
            "b"
            + str(out_pos)
            + ":"
            + "|".join("LUT" + str(len(item["positions"])) + str(tuple(item["positions"])) for item in top[:5])
        )
    return "".join(result_bits), {
        "status": "candidate",
        "reason": "v412_lut_unique_or_top_consensus",
        "consistent_counts": counts,
        "accepted_counts": accepted_counts,
        "proof": "; ".join(proofs),
    }


def split_symbolic_token(token: str) -> list[tuple[str, str, str]]:
    text = str(token or "")
    splits: list[tuple[str, str, str]] = []
    for index in range(1, len(text) - 1):
        op = text[index]
        if op and not op.isalnum():
            splits.append((text[:index], op, text[index + 1 :]))
    return splits


@dataclass(frozen=True)
class Program:
    rule_class: str
    name: str
    apply: Callable[[str], str | None]
    depth: int
    nodes: int
    literal_count: int


def delete_char_programs(examples: list[tuple[str, str]], max_subset_size: int) -> list[Program]:
    alphabet = sorted(set("".join(lhs for lhs, _ in examples)))
    programs: list[Program] = []
    for size in range(1, min(max_subset_size, len(alphabet)) + 1):
        for chars in itertools.combinations(alphabet, size):
            remove = set(chars)
            if all("".join(ch for ch in lhs if ch not in remove) == rhs for lhs, rhs in examples):
                label = "".join(chars)
                programs.append(
                    Program(
                        "v412_symbolic_delete_chars",
                        "delete:" + label,
                        lambda text, remove=remove: "".join(ch for ch in text if ch not in remove),
                        1,
                        1,
                        len(remove),
                    )
                )
    return programs


def keep_char_programs(examples: list[tuple[str, str]], max_subset_size: int) -> list[Program]:
    alphabet = sorted(set("".join(lhs for lhs, _ in examples)))
    programs: list[Program] = []
    for size in range(1, min(max_subset_size, len(alphabet)) + 1):
        for chars in itertools.combinations(alphabet, size):
            keep = set(chars)
            if all("".join(ch for ch in lhs if ch in keep) == rhs for lhs, rhs in examples):
                label = "".join(chars)
                programs.append(
                    Program(
                        "v412_symbolic_keep_chars",
                        "keep:" + label,
                        lambda text, keep=keep: "".join(ch for ch in text if ch in keep),
                        1,
                        1,
                        len(keep),
                    )
                )
    return programs


def marker_programs(examples: list[tuple[str, str]]) -> list[Program]:
    alphabet = sorted(set("".join(lhs for lhs, _ in examples)))
    specs: list[tuple[str, Callable[[str, str], str | None]]] = [
        ("prefix_first", lambda text, marker: text.split(marker, 1)[0] if marker in text else None),
        ("suffix_first", lambda text, marker: text.split(marker, 1)[1] if marker in text else None),
        ("prefix_last", lambda text, marker: text.rsplit(marker, 1)[0] if marker in text else None),
        ("suffix_last", lambda text, marker: text.rsplit(marker, 1)[1] if marker in text else None),
        ("remove_first", lambda text, marker: text.replace(marker, "", 1) if marker in text else None),
    ]
    programs: list[Program] = []
    for spec_name, func in specs:
        for marker in alphabet:
            if all(func(lhs, marker) == rhs for lhs, rhs in examples):
                programs.append(
                    Program(
                        "v412_symbolic_marker",
                        f"{spec_name}:{marker}",
                        lambda text, marker=marker, func=func: func(text, marker),
                        1,
                        2,
                        1,
                    )
                )
    return programs


def op_split_programs(examples: list[tuple[str, str]]) -> list[Program]:
    transforms: dict[str, Callable[[str, str, str], str]] = {
        "drop_operator": lambda left, _op, right: left + right,
        "reverse_drop_operator": lambda left, _op, right: (left + right)[::-1],
        "left_only": lambda left, _op, _right: left,
        "right_only": lambda _left, _op, right: right,
        "right_left": lambda left, _op, right: right + left,
        "left_reverse_right": lambda left, _op, right: left + right[::-1],
        "reverse_left_right": lambda left, _op, right: left[::-1] + right,
        "right_reverse_left": lambda left, _op, right: right + left[::-1],
        "reverse_right_left": lambda left, _op, right: right[::-1] + left,
        "operator_between_reversed": lambda left, op, right: right + op + left,
    }
    programs: list[Program] = []
    for name, transform in transforms.items():
        possible = True
        for lhs, rhs in examples:
            if not any(transform(left, op, right) == rhs for left, op, right in split_symbolic_token(lhs)):
                possible = False
                break
        if not possible:
            continue

        def apply(text: str, transform: Callable[[str, str, str], str] = transform) -> str | None:
            preds = sorted({transform(left, op, right) for left, op, right in split_symbolic_token(text)})
            return preds[0] if len(preds) == 1 else None

        programs.append(Program("v412_symbolic_op_split", name, apply, 2, 3, 0))
    return programs


def position_template_programs(
    examples: list[tuple[str, str]],
    query: str,
    *,
    max_position_sources: int,
    max_program_count: int,
) -> list[Program]:
    rhs_lengths = {len(rhs) for _, rhs in examples}
    if len(rhs_lengths) != 1:
        return []
    output_len = next(iter(rhs_lengths))
    if output_len <= 0 or output_len > 7:
        return []
    max_len = min(max_position_sources, max(len(lhs) for lhs, _ in examples + [(query, "")]))

    def sources_for_index(out_index: int) -> list[tuple[str, Callable[[str], str | None], int]]:
        choices: list[tuple[str, Callable[[str], str | None], int]] = []
        literal_value = examples[0][1][out_index]
        if all(rhs[out_index] == literal_value for _, rhs in examples):
            choices.append((f"lit:{literal_value}", lambda _text, value=literal_value: value, 1))
        for idx in range(max_len):
            funcs: list[tuple[str, Callable[[str], str | None]]] = [
                (f"start{idx}", lambda text, idx=idx: text[idx] if idx < len(text) else None),
                (f"end{idx}", lambda text, idx=idx: text[-idx - 1] if idx < len(text) else None),
            ]
            for name, func in funcs:
                values = [func(lhs) for lhs, _ in examples]
                if all(value is not None for value in values) and all(
                    str(value) == rhs[out_index] for value, (_, rhs) in zip(values, examples)
                ):
                    if func(query) is not None:
                        choices.append((name, func, 0))
        return choices

    choices = [sources_for_index(out_idx) for out_idx in range(output_len)]
    if any(not item for item in choices):
        return []
    program_count = math.prod(len(item) for item in choices)
    if program_count > max_program_count:
        return []
    programs: list[Program] = []
    for combo in itertools.product(*choices):
        names = [name for name, _func, _literal in combo]
        if all(name.startswith("lit:") for name in names):
            continue

        def apply(text: str, combo: tuple[tuple[str, Callable[[str], str | None], int], ...] = combo) -> str | None:
            pieces: list[str] = []
            for _name, func, _literal in combo:
                value = func(text)
                if value is None:
                    return None
                pieces.append(str(value))
            return "".join(pieces)

        literal_count = sum(literal for _name, _func, literal in combo)
        programs.append(
            Program(
                "v412_symbolic_position_template",
                "|".join(names),
                apply,
                2,
                len(combo),
                literal_count,
            )
        )
    return programs


def transducer_programs(examples: list[tuple[str, str]], *, pair_cap: int, global_cap: int) -> list[Program]:
    mappings: list[dict[str, str]] = [{}]
    for lhs, rhs in examples:
        pair_mappings = transducer_mappings_for_pair(lhs, rhs, pair_cap)
        if not pair_mappings:
            return []
        mappings = merge_mapping_sets(mappings, pair_mappings, global_cap)
        if not mappings:
            return []
    programs: list[Program] = []
    for mapping in mappings[:global_cap]:
        def apply(text: str, mapping: dict[str, str] = mapping) -> str | None:
            try:
                return "".join(mapping.get(ch, ch) for ch in text)
            except Exception:
                return None

        literal_count = sum(1 for value in mapping.values() if value)
        marker = ",".join(f"{key}->{value}" for key, value in sorted(mapping.items()))
        programs.append(Program("v412_symbolic_transducer", marker, apply, 3, len(mapping), literal_count))
    return programs


def reverse_text(value: str) -> str:
    text = str(value)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def digits(value: int) -> list[int]:
    return [int(ch) for ch in str(abs(int(value)))]


def digits2(value: int) -> tuple[int, int]:
    text = f"{abs(int(value)):02d}"[-2:]
    return int(text[0]), int(text[1])


def extra_numeric_functions() -> dict[str, Callable[[int, int], str]]:
    funcs = dict(v241_numeric_rule_functions())
    funcs.update(
        {
            "max": lambda a, b: str(max(a, b)),
            "min": lambda a, b: str(min(a, b)),
            "div_ab": lambda a, b: str(a // b) if b else "",
            "div_ba": lambda a, b: str(b // a) if a else "",
            "mod_ab": lambda a, b: str(a % b) if b else "",
            "mod_ba": lambda a, b: str(b % a) if a else "",
            "gcd": lambda a, b: str(math.gcd(a, b)),
            "lcm": lambda a, b: str(abs(a * b) // math.gcd(a, b)) if a and b else "0",
            "square_sum": lambda a, b: str(a * a + b * b),
            "square_diff_abs": lambda a, b: str(abs(a * a - b * b)),
            "cross_tens_mul_ones_mul": lambda a, b: str(digits2(a)[0] * digits2(b)[0]) + str(digits2(a)[1] * digits2(b)[1]),
            "cross_tens_add_ones_mul": lambda a, b: str(digits2(a)[0] + digits2(b)[0]) + str(digits2(a)[1] * digits2(b)[1]),
            "determinant_digits": lambda a, b: str(digits2(a)[0] * digits2(b)[1] - digits2(a)[1] * digits2(b)[0]),
            "abs_determinant_digits": lambda a, b: str(abs(digits2(a)[0] * digits2(b)[1] - digits2(a)[1] * digits2(b)[0])),
        }
    )
    return funcs


def numeric_programs(examples: list[tuple[str, str]], query: str, *, min_same_operator_examples: int) -> list[Program]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return []
    q_left, q_op, q_right = parsed_query
    groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    all_numeric: list[tuple[int, int, str]] = []
    for lhs, rhs in examples:
        parsed = parse_numeric_token(lhs)
        if not parsed:
            return []
        left, op, right = parsed
        try:
            item = (int(left), int(right), str(rhs))
        except ValueError:
            return []
        groups[op].append(item)
        all_numeric.append(item)
    funcs = extra_numeric_functions()
    train_sets: list[tuple[str, list[tuple[int, int, str]]]] = []
    if len(groups.get(q_op, [])) >= min_same_operator_examples:
        train_sets.append(("same_op", groups[q_op]))
    train_sets.append(("all_ops", all_numeric))
    programs: list[Program] = []
    for train_name, train_rows in train_sets:
        for func_name, func in funcs.items():
            for reverse_operands in (False, True):
                for reverse_result in (False, True):
                    ok = True
                    for left, right, expected in train_rows:
                        a = int(str(left)[::-1]) if reverse_operands else left
                        b = int(str(right)[::-1]) if reverse_operands else right
                        try:
                            raw = func(a, b)
                        except Exception:
                            ok = False
                            break
                        if raw == "":
                            ok = False
                            break
                        pred = reverse_text(str(raw)) if reverse_result else str(raw)
                        if pred != expected:
                            ok = False
                            break
                    if not ok:
                        continue

                    def apply(
                        text: str,
                        func: Callable[[int, int], str] = func,
                        reverse_operands: bool = reverse_operands,
                        reverse_result: bool = reverse_result,
                    ) -> str | None:
                        parsed = parse_numeric_token(text)
                        if not parsed:
                            return None
                        left_raw, _op, right_raw = parsed
                        left_text = str(left_raw)
                        right_text = str(right_raw)
                        a = int(left_text[::-1]) if reverse_operands else int(left_text)
                        b = int(right_text[::-1]) if reverse_operands else int(right_text)
                        try:
                            raw = func(a, b)
                        except Exception:
                            return None
                        if raw == "":
                            return None
                        return reverse_text(str(raw)) if reverse_result else str(raw)

                    programs.append(
                        Program(
                            "v412_numeric_vsa",
                            f"{train_name}:{func_name}:revop={int(reverse_operands)}:revres={int(reverse_result)}",
                            apply,
                            2 if train_name == "same_op" else 3,
                            2,
                            0,
                        )
                    )
    return programs


def synthetic_probes(query: str, examples: list[tuple[str, str]]) -> list[str]:
    probes = {query}
    if query:
        probes.add(query[::-1])
    operators = sorted({parsed[1] for lhs, _ in examples if (parsed := parse_numeric_token(lhs))})
    q_parsed = parse_numeric_token(query)
    if q_parsed:
        left_raw, op, right_raw = q_parsed
        left = str(left_raw)
        right = str(right_raw)
        for delta in (-1, 1):
            probes.add(f"{max(0, int(left) + delta):0{len(left)}d}{op}{right}")
            probes.add(f"{left}{op}{max(0, int(right) + delta):0{len(right)}d}")
        probes.add(f"{left.zfill(len(left)+1)}{op}{right}")
        for seen_op in operators:
            probes.add(f"{left}{seen_op}{right}")
    else:
        split_ops = sorted({op for lhs, _ in examples for _left, op, _right in split_symbolic_token(lhs)})
        for left, op, right in split_symbolic_token(query):
            for seen_op in split_ops[:6]:
                probes.add(left + seen_op + right)
            if left and right:
                probes.add(left[::-1] + op + right)
                probes.add(left + op + right[::-1])
    return sorted(item for item in probes if item)


def program_score(program: Program, prediction: str, examples: list[tuple[str, str]], query: str) -> tuple[int, dict[str, Any]]:
    rhs_lengths = {len(rhs) for _, rhs in examples}
    rhs_chars = set("".join(rhs for _, rhs in examples))
    query_chars = set(query)
    probes = synthetic_probes(query, examples)
    undefined = 0
    ood = 0
    for probe in probes:
        value = program.apply(probe)
        if value is None or value == "":
            undefined += 1
            continue
        if rhs_lengths and len(value) not in rhs_lengths and probe == query:
            ood += 2
        elif rhs_lengths and len(value) not in rhs_lengths:
            ood += 1
        if rhs_chars and any(ch not in rhs_chars and ch not in query_chars for ch in value):
            ood += 1
    score_value = (
        5 * program.depth
        + 2 * program.nodes
        + 4 * program.literal_count
        + 10 * undefined
        + 6 * ood
        + max(0, len(prediction) - max(rhs_lengths or {len(prediction)}))
    )
    return score_value, {"undefined": undefined, "ood": ood, "probe_count": len(probes)}


def vsa_candidate_for_equation(
    examples: list[tuple[str, str]],
    query: str,
    *,
    max_char_subset_size: int,
    max_position_sources: int,
    max_position_programs: int,
    pair_mapping_cap: int,
    global_mapping_cap: int,
    min_same_operator_examples: int,
    max_programs_before_rank: int,
) -> tuple[str | None, dict[str, Any]]:
    programs: list[Program] = []
    if parse_numeric_token(query) and all(parse_numeric_token(lhs) for lhs, _ in examples):
        programs.extend(numeric_programs(examples, query, min_same_operator_examples=min_same_operator_examples))
    programs.extend(delete_char_programs(examples, max_char_subset_size))
    programs.extend(keep_char_programs(examples, max_char_subset_size))
    programs.extend(marker_programs(examples))
    programs.extend(op_split_programs(examples))
    programs.extend(
        position_template_programs(
            examples,
            query,
            max_position_sources=max_position_sources,
            max_program_count=max_position_programs,
        )
    )
    programs.extend(transducer_programs(examples, pair_cap=pair_mapping_cap, global_cap=global_mapping_cap))
    if len(programs) > max_programs_before_rank:
        # Keep deterministic low-complexity programs first. This is a CPU/FinOps
        # guard, not a ranking decision.
        programs = sorted(programs, key=lambda item: (item.depth, item.nodes, item.literal_count, item.rule_class, item.name))[
            :max_programs_before_rank
        ]
    ranked: list[dict[str, Any]] = []
    for program in programs:
        prediction = program.apply(query)
        if prediction is None or prediction == "":
            continue
        if not all(program.apply(lhs) == rhs for lhs, rhs in examples):
            continue
        score_value, score_meta = program_score(program, str(prediction), examples, query)
        ranked.append(
            {
                "rule_class": program.rule_class,
                "program_name": program.name,
                "prediction": str(prediction),
                "score": score_value,
                **score_meta,
            }
        )
    if not ranked:
        return None, {"status": "abstain", "reason": "no_consistent_vsa_program", "program_count": len(programs)}
    ranked.sort(key=lambda item: (int(item["score"]), str(item["rule_class"]), str(item["program_name"])))
    top_score = int(ranked[0]["score"])
    near = [item for item in ranked if int(item["score"]) <= top_score + 1]
    near_predictions = sorted({str(item["prediction"]) for item in near})
    if len(near_predictions) != 1:
        return None, {
            "status": "abstain",
            "reason": "ambiguous_near_top_vsa_predictions",
            "program_count": len(programs),
            "ranked_count": len(ranked),
            "top_score": top_score,
            "near_predictions": near_predictions,
            "top_programs": ranked[:10],
        }
    return near_predictions[0], {
        "status": "candidate",
        "reason": "v412_vsa_ranked_unique_prediction",
        "program_count": len(programs),
        "ranked_count": len(ranked),
        "top_score": top_score,
        "top_programs": ranked[:10],
        "proof": "; ".join(
            f"{item['rule_class']}:{item['program_name']}=>{item['prediction']}@{item['score']}" for item in ranked[:5]
        ),
    }


def load_v409_candidates(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    out: list[dict[str, str]] = []
    for row in read_csv(path):
        out.append(
            {
                "id": row["id"],
                "family": row["family"],
                "prediction": row["new_prediction"],
                "source": "v409_existing_projection",
                "reason": row.get("reasons", ""),
                "proof": row.get("sources", ""),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument("--v409-accepted-csv", type=Path, default=V409_ACCEPTED_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bit-max-k", type=int, default=3)
    parser.add_argument("--bit-max-total-candidates-per-bit", type=int, default=5000)
    parser.add_argument("--bit-max-accepted-candidates-per-bit", type=int, default=256)
    parser.add_argument("--max-char-subset-size", type=int, default=4)
    parser.add_argument("--max-position-sources", type=int, default=7)
    parser.add_argument("--max-position-programs", type=int, default=20000)
    parser.add_argument("--pair-mapping-cap", type=int, default=3000)
    parser.add_argument("--global-mapping-cap", type=int, default=12000)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    parser.add_argument("--max-programs-before-rank", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=== V412 CPU SYNTHESIS GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_csv =", args.baseline_csv, flush=True)
    print("v409_accepted_csv =", args.v409_accepted_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.baseline_csv)
    for row in rows:
        row["family"] = classify_puzzle(row["prompt"])
        row["baseline_correct"] = verify_answer(row["answer"], row["prediction"])
        row["truncated_bool"] = boolish(row.get("truncated", ""))
        row["v409_prediction"] = row["prediction"]
        row["v412_prediction"] = row["prediction"]

    by_id = {row["id"]: row for row in rows}
    candidate_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    v409_candidates = load_v409_candidates(args.v409_accepted_csv)
    v409_candidate_ids = {item["id"] for item in v409_candidates}
    all_candidates: list[dict[str, str]] = []
    print("loaded_v409_candidates =", len(v409_candidates), flush=True)

    accepted_rows: list[dict[str, Any]] = []
    for item in v409_candidates:
        row = by_id.get(item["id"])
        if row is not None:
            row["v409_prediction"] = item["prediction"]
            row["v412_prediction"] = item["prediction"]
            accepted_rows.append(
                {
                    "id": item["id"],
                    "family": row["family"],
                    "old_prediction": row["prediction"],
                    "new_prediction": item["prediction"],
                    "answer": row["answer"],
                    "old_correct": row["baseline_correct"],
                    "new_correct": verify_answer(row["answer"], item["prediction"]),
                    "sources": item.get("source", "v409_integrated_solver"),
                    "reasons": item.get("reason", "v409_preaccepted_projection"),
                }
            )

    target_rows = [row for row in rows if row["family"] in {"bit_manipulation", "equation_transform"}]
    for idx, row in enumerate(target_rows, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(target_rows):
            print(f"v412_candidate_generation_progress = {idx}/{len(target_rows)}", flush=True)
        if row["family"] == "bit_manipulation":
            prediction, meta = solve_bit_lut_unique(
                row["prompt"],
                max_k=args.bit_max_k,
                max_total_candidates_per_bit=args.bit_max_total_candidates_per_bit,
                max_accepted_candidates_per_bit=args.bit_max_accepted_candidates_per_bit,
            )
            if prediction:
                all_candidates.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "prediction": prediction,
                        "source": "v412_bit_lut_k123",
                        "reason": meta.get("reason", ""),
                        "proof": meta.get("proof", ""),
                    }
                )
                candidate_rows.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "source": "v412_bit_lut_k123",
                        "status": "candidate",
                        "prediction": prediction,
                        "answer": row["answer"],
                        "baseline_prediction": row["prediction"],
                        "baseline_correct": row["baseline_correct"],
                        "candidate_correct": verify_answer(row["answer"], prediction),
                        "reason": meta.get("reason", ""),
                        "proof": meta.get("proof", ""),
                    }
                )
            else:
                candidate_rows.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "source": "v412_bit_lut_k123",
                        "status": "abstain",
                        "prediction": "",
                        "answer": row["answer"],
                        "baseline_prediction": row["prediction"],
                        "baseline_correct": row["baseline_correct"],
                        "candidate_correct": False,
                        "reason": meta.get("reason", ""),
                        "proof": json.dumps(meta, sort_keys=True)[:700],
                    }
                )
        elif row["family"] == "equation_transform":
            examples, query, parse_status = parse_alice_prompt(row["prompt"])
            if parse_status == "ok":
                prediction, meta = vsa_candidate_for_equation(
                    examples,
                    query,
                    max_char_subset_size=args.max_char_subset_size,
                    max_position_sources=args.max_position_sources,
                    max_position_programs=args.max_position_programs,
                    pair_mapping_cap=args.pair_mapping_cap,
                    global_mapping_cap=args.global_mapping_cap,
                    min_same_operator_examples=args.min_same_operator_examples,
                    max_programs_before_rank=args.max_programs_before_rank,
                )
            else:
                prediction, meta = None, {"status": "abstain", "reason": parse_status}
            if prediction:
                all_candidates.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "prediction": prediction,
                        "source": "v412_equation_vsa_ranked",
                        "reason": meta.get("reason", ""),
                        "proof": meta.get("proof", ""),
                    }
                )
                candidate_rows.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "source": "v412_equation_vsa_ranked",
                        "status": "candidate",
                        "prediction": prediction,
                        "answer": row["answer"],
                        "baseline_prediction": row["prediction"],
                        "baseline_correct": row["baseline_correct"],
                        "candidate_correct": verify_answer(row["answer"], prediction),
                        "reason": meta.get("reason", ""),
                        "proof": meta.get("proof", ""),
                    }
                )
            else:
                candidate_rows.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "source": "v412_equation_vsa_ranked",
                        "status": "abstain",
                        "prediction": "",
                        "answer": row["answer"],
                        "baseline_prediction": row["prediction"],
                        "baseline_correct": row["baseline_correct"],
                        "candidate_correct": False,
                        "reason": meta.get("reason", ""),
                        "proof": json.dumps(meta, sort_keys=True)[:700],
                    }
                )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in all_candidates:
        grouped[item["id"]].append(item)

    new_v412_rows: list[dict[str, Any]] = []
    for row_id, items in sorted(grouped.items()):
        row = by_id.get(row_id)
        if row is None:
            continue
        if row_id in v409_candidate_ids:
            continue
        predictions = sorted({item["prediction"] for item in items if item.get("prediction")})
        if len(predictions) != 1:
            conflicts.append(
                {
                    "id": row_id,
                    "family": row["family"],
                    "reason": "prediction_conflict",
                    "predictions": "|".join(predictions),
                    "sources": ";".join(sorted({item["source"] for item in items})),
                }
            )
            continue
        prediction = predictions[0]
        candidate_correct = verify_answer(row["answer"], prediction)
        if row["baseline_correct"] and not candidate_correct:
            conflicts.append(
                {
                    "id": row_id,
                    "family": row["family"],
                    "reason": "would_regress_baseline_correct",
                    "predictions": prediction,
                    "sources": ";".join(sorted({item["source"] for item in items})),
                }
            )
            continue
        if (not row["baseline_correct"]) and not candidate_correct:
            false_positive_rows.append(
                {
                    "id": row_id,
                    "family": row["family"],
                    "prediction": prediction,
                    "answer": row["answer"],
                    "baseline_prediction": row["prediction"],
                    "sources": ";".join(sorted({item["source"] for item in items})),
                }
            )
            continue
        if (not row["baseline_correct"]) and candidate_correct:
            row["v412_prediction"] = prediction
            accepted = {
                "id": row_id,
                "family": row["family"],
                "old_prediction": row["prediction"],
                "new_prediction": prediction,
                "answer": row["answer"],
                "old_correct": row["baseline_correct"],
                "new_correct": candidate_correct,
                "sources": ";".join(sorted({item["source"] for item in items})),
                "reasons": ";".join(sorted({item.get("reason", "") for item in items})),
            }
            accepted_rows.append(accepted)
            if "v412" in accepted["sources"]:
                new_v412_rows.append(accepted)

    baseline_score = score(rows, "prediction")
    v409_score = score(rows, "v409_prediction")
    v412_score = score(rows, "v412_prediction")

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
        "reason",
        "proof",
    ]
    write_csv(args.output_dir / "v412_candidate_audit.csv", candidate_rows, columns)
    write_csv(
        args.output_dir / "v412_integrated_accepted.csv",
        accepted_rows,
        ["id", "family", "old_prediction", "new_prediction", "answer", "old_correct", "new_correct", "sources", "reasons"],
    )
    write_csv(
        args.output_dir / "v412_new_accepted.csv",
        new_v412_rows,
        ["id", "family", "old_prediction", "new_prediction", "answer", "old_correct", "new_correct", "sources", "reasons"],
    )
    write_csv(args.output_dir / "v412_false_positive_candidates.csv", false_positive_rows, ["id", "family", "prediction", "answer", "baseline_prediction", "sources"])
    write_csv(args.output_dir / "v412_conflicts.csv", conflicts, ["id", "family", "reason", "predictions", "sources"])

    fam_base = baseline_score["families"]
    fam_v409 = v409_score["families"]
    fam_v412 = v412_score["families"]
    decision = (
        "v412_new_cpu_signal_found_not_adapter_submit_safe"
        if new_v412_rows
        else "v412_no_new_safe_cpu_signal_beyond_v409"
    )
    manifest = {
        "schema_version": "kg1_v412_cpu_synthesis_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "baseline_csv": str(args.baseline_csv),
            "v409_accepted_csv": str(args.v409_accepted_csv),
        },
        "config": vars(args) | {"baseline_csv": str(args.baseline_csv), "v409_accepted_csv": str(args.v409_accepted_csv), "output_dir": str(args.output_dir)},
        "baseline_score": baseline_score,
        "v409_score": v409_score,
        "v412_score": v412_score,
        "accepted_gain_count": len(accepted_rows),
        "new_v412_gain_count": len(new_v412_rows),
        "new_v412_by_family": dict(Counter(row["family"] for row in new_v412_rows)),
        "false_positive_count": len(false_positive_rows),
        "conflict_count": len(conflicts),
        "decision": {
            "decision": decision,
            "reason": (
                f"baseline={baseline_score['total']['correct']}; v409={v409_score['total']['correct']}; "
                f"v412={v412_score['total']['correct']}; new_v412={len(new_v412_rows)}; "
                f"false_positive={len(false_positive_rows)}; conflicts={len(conflicts)}"
            ),
            "gpu_authorized": bool(new_v412_rows and not conflicts and not false_positive_rows),
            "submit_authorized": False,
        },
        "outputs": {
            "candidate_audit_csv": str(args.output_dir / "v412_candidate_audit.csv"),
            "accepted_csv": str(args.output_dir / "v412_integrated_accepted.csv"),
            "new_accepted_csv": str(args.output_dir / "v412_new_accepted.csv"),
            "false_positive_csv": str(args.output_dir / "v412_false_positive_candidates.csv"),
            "conflicts_csv": str(args.output_dir / "v412_conflicts.csv"),
            "manifest_json": str(args.output_dir / "v412_cpu_synthesis_gate_manifest.json"),
            "report_md": str(args.output_dir / "V412_CPU_SYNTHESIS_GATE.md"),
        },
    }
    write_json(args.output_dir / "v412_cpu_synthesis_gate_manifest.json", manifest)

    report = [
        "# V412 CPU Synthesis Gate",
        "",
        "| Metric | Baseline V291/V290 | V409 CPU projection | V412 CPU projection | Delta vs V409 |",
        "|---|---:|---:|---:|---:|",
        f"| Weak total | `{baseline_score['total']['correct']}/315` | `{v409_score['total']['correct']}/315` | `{v412_score['total']['correct']}/315` | `{v412_score['total']['correct'] - v409_score['total']['correct']:+d}` |",
        f"| equation_transform | `{fam_base['equation_transform']['correct']}/155` | `{fam_v409['equation_transform']['correct']}/155` | `{fam_v412['equation_transform']['correct']}/155` | `{fam_v412['equation_transform']['correct'] - fam_v409['equation_transform']['correct']:+d}` |",
        f"| bit_manipulation | `{fam_base['bit_manipulation']['correct']}/160` | `{fam_v409['bit_manipulation']['correct']}/160` | `{fam_v412['bit_manipulation']['correct']}/160` | `{fam_v412['bit_manipulation']['correct'] - fam_v409['bit_manipulation']['correct']:+d}` |",
        "",
        f"- New V412 accepted gains beyond V409: `{len(new_v412_rows)}`.",
        f"- False-positive candidates blocked by weak labels: `{len(false_positive_rows)}`.",
        f"- Conflicts/losses blocked: `{len(conflicts)}`.",
        "",
        "CPU solver/verifier projection only. Not adapter-only and not Kaggle-submitable as-is.",
        "",
        "## New V412 Gains",
        "",
    ]
    if new_v412_rows:
        for row in new_v412_rows:
            report.append(
                f"- `{row['id']}` `{row['family']}`: `{row['old_prediction']}` -> `{row['new_prediction']}` via `{row['sources']}` / `{row['reasons']}`"
            )
    else:
        report.append("- None.")
    report.extend(["", "## Decision", "", manifest["decision"]["decision"], ""])
    (args.output_dir / "V412_CPU_SYNTHESIS_GATE.md").write_text("\n".join(report), encoding="utf-8")

    print("baseline_score =", json.dumps(baseline_score, sort_keys=True), flush=True)
    print("v409_score =", json.dumps(v409_score, sort_keys=True), flush=True)
    print("v412_score =", json.dumps(v412_score, sort_keys=True), flush=True)
    print("new_v412_gain_count =", len(new_v412_rows), flush=True)
    print("false_positive_count =", len(false_positive_rows), flush=True)
    print("conflict_count =", len(conflicts), flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("manifest_json =", args.output_dir / "v412_cpu_synthesis_gate_manifest.json", flush=True)
    print("=== V412 CPU SYNTHESIS GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
