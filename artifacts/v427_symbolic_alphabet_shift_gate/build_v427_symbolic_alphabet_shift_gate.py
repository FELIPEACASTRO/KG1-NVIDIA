#!/usr/bin/env python3
"""V427 symbolic alphabet/ASCII shift gate.

This CPU-only gate tests a structural class not covered by V421-V426:
per-output-position programs that copy an input position after an ASCII or
prompt-alphabet shift. This can generalize punctuation mappings to query
symbols not seen in the examples, while still abstaining on ambiguity.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src", ROOT / "artifacts/v426_symbolic_structural_template_gate"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402
from build_v426_symbolic_structural_template_gate import (  # noqa: E402
    BASELINE_CSV,
    EXPECTED_BASELINE,
    V414_ACCEPTED_CSV,
    boolish,
    parse_examples,
    read_csv,
    write_csv,
    write_json,
)


OUT_DIR = ROOT / "artifacts/v427_symbolic_alphabet_shift_gate/20260515T_v427_symbolic_alphabet_shift"
PRINTABLE_MIN = 32
PRINTABLE_MAX = 126
MAX_ASCII_SHIFT = 20


@dataclass(frozen=True)
class Term:
    kind: str
    pos: int | None
    value: str

    def apply(self, token: str, alphabets: dict[str, str]) -> str | None:
        if self.kind == "const":
            return self.value
        if self.pos is None or self.pos >= len(token):
            return None
        ch = token[self.pos]
        if self.kind == "pos":
            return ch
        shift = int(self.value)
        if self.kind == "ascii_shift":
            code = ord(ch) + shift
            if PRINTABLE_MIN <= code <= PRINTABLE_MAX:
                return chr(code)
            return None
        alphabet = alphabets.get(self.kind, "")
        if ch not in alphabet:
            return None
        idx = alphabet.index(ch)
        return alphabet[(idx + shift) % len(alphabet)]

    def label(self) -> str:
        return f"{self.kind}:{'' if self.pos is None else self.pos}:{self.value}"


@dataclass(frozen=True)
class Program:
    scope: str
    terms: tuple[Term, ...]
    prediction: str
    support_rows: int
    loo_checked: int
    loo_passed: int
    score: tuple[int, int, int, int, str]

    def label(self) -> str:
        return self.scope + "::" + ",".join(term.label() for term in self.terms)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def family_for(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or row.get("task_type") or classify_puzzle(str(row.get("prompt", ""))))


def ordered_unique(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for ch in text:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return "".join(out)


def alphabets_for(examples: list[tuple[str, str]], query: str) -> dict[str, str]:
    text = "".join(lhs + rhs for lhs, rhs in examples) + query
    sorted_alpha = "".join(sorted(set(text)))
    seen_alpha = ordered_unique(text)
    return {
        "sorted_shift": sorted_alpha,
        "seen_shift": seen_alpha,
    }


def apply_terms(token: str, terms: tuple[Term, ...], alphabets: dict[str, str]) -> str | None:
    out: list[str] = []
    for term in terms:
        val = term.apply(token, alphabets)
        if val is None:
            return None
        out.append(val)
    return "".join(out)


def same_mod_shift(values: list[tuple[str, str]], alphabet: str) -> int | None:
    if not alphabet:
        return None
    shifts: set[int] = set()
    for source, target in values:
        if source not in alphabet or target not in alphabet:
            return None
        shifts.add((alphabet.index(target) - alphabet.index(source)) % len(alphabet))
    return shifts.pop() if len(shifts) == 1 else None


def possible_terms(
    examples: list[tuple[str, str]],
    out_idx: int,
    alphabets: dict[str, str],
) -> list[Term]:
    terms: list[Term] = []
    for pos in range(5):
        pairs = [(lhs[pos], rhs[out_idx]) for lhs, rhs in examples if len(rhs) > out_idx and len(lhs) == 5]
        if len(pairs) != len(examples):
            continue
        if all(src == dst for src, dst in pairs):
            terms.append(Term("pos", pos, "0"))
        ascii_shifts = {ord(dst) - ord(src) for src, dst in pairs}
        if len(ascii_shifts) == 1:
            shift = ascii_shifts.pop()
            if 0 < abs(shift) <= MAX_ASCII_SHIFT:
                terms.append(Term("ascii_shift", pos, str(shift)))
        for kind in ("sorted_shift", "seen_shift"):
            shift = same_mod_shift(pairs, alphabets[kind])
            if shift is not None and shift != 0:
                # Keep shortest equivalent signed representation for scoring.
                n = len(alphabets[kind])
                signed = shift if shift <= n // 2 else shift - n
                terms.append(Term(kind, pos, str(signed)))
    constants = sorted({rhs[out_idx] for _lhs, rhs in examples if len(rhs) > out_idx})
    for ch in constants:
        if all(len(rhs) > out_idx and rhs[out_idx] == ch for _lhs, rhs in examples):
            terms.append(Term("const", None, ch))
    # Stable de-duplication by label.
    uniq = {term.label(): term for term in terms}
    return [uniq[key] for key in sorted(uniq)]


def synthesize(
    examples: list[tuple[str, str]],
    query: str,
    *,
    scope: str,
    max_programs_per_len: int = 6000,
) -> list[Program]:
    programs: list[Program] = []
    if len(query) != 5 or len(examples) < 2:
        return programs
    alphabets = alphabets_for(examples, query)
    by_len: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for lhs, rhs in examples:
        if len(lhs) == 5 and 0 < len(rhs) <= 6:
            by_len[len(rhs)].append((lhs, rhs))
    for out_len, subset in sorted(by_len.items()):
        if len(subset) < 2:
            continue
        per_pos = [possible_terms(subset, idx, alphabets) for idx in range(out_len)]
        if any(not terms for terms in per_pos):
            continue
        total = 1
        for terms in per_pos:
            total *= len(terms)
        if total > max_programs_per_len:
            continue
        for terms_tuple in itertools.product(*per_pos):
            terms_tuple = tuple(terms_tuple)
            if not all(apply_terms(lhs, terms_tuple, alphabets) == rhs for lhs, rhs in subset):
                continue
            pred = apply_terms(query, terms_tuple, alphabets)
            if pred is None:
                continue
            loo_checked = 0
            loo_passed = 0
            for omit_idx, (held_lhs, held_rhs) in enumerate(subset):
                train = [item for idx, item in enumerate(subset) if idx != omit_idx]
                if len(train) < 2:
                    continue
                train_alphabets = alphabets_for(train, query)
                train_terms = [possible_terms(train, idx, train_alphabets) for idx in range(out_len)]
                if all(terms_tuple[idx] in train_terms[idx] for idx in range(out_len)):
                    held_pred = apply_terms(held_lhs, terms_tuple, train_alphabets)
                    if held_pred is not None:
                        loo_checked += 1
                        loo_passed += int(held_pred == held_rhs)
            const_count = sum(1 for term in terms_tuple if term.kind == "const")
            shift_penalty = sum(abs(int(term.value)) for term in terms_tuple if "shift" in term.kind)
            op_pos_count = sum(1 for term in terms_tuple if term.pos == 2)
            score = (-loo_passed, const_count, shift_penalty, op_pos_count, ",".join(t.label() for t in terms_tuple))
            programs.append(Program(scope, terms_tuple, pred, len(subset), loo_checked, loo_passed, score))
    return programs


def operator_of(token: str | None) -> str:
    text = str(token or "")
    return text[2] if len(text) == 5 else ""


def predict(prompt: str) -> tuple[str | None, dict[str, Any]]:
    examples, query = parse_examples(prompt)
    if not query or len(query) != 5:
        return None, {"status": "abstain", "reason": "query_not_len5_or_missing"}
    programs = synthesize(examples, query, scope="all_examples")
    same_op = [(lhs, rhs) for lhs, rhs in examples if operator_of(lhs) == operator_of(query)]
    if len(same_op) >= 2:
        programs.extend(synthesize(same_op, query, scope="same_operator_examples"))
    if not programs:
        return None, {"status": "abstain", "reason": "no_alphabet_shift_program"}
    programs.sort(key=lambda item: item.score)
    top_key = programs[0].score[:4]
    top = [item for item in programs if item.score[:4] == top_key]
    predictions = sorted({item.prediction for item in top})
    if len(predictions) != 1:
        return None, {
            "status": "abstain",
            "reason": "ambiguous_top_alphabet_shift",
            "top_predictions": predictions[:20],
            "top_programs": [item.label() for item in top[:20]],
        }
    return predictions[0], {
        "status": "candidate",
        "reason": "unique_alphabet_shift_program",
        "top_score": list(top_key),
        "top_programs": [item.label() for item in top[:10]],
        "support_rows": max(item.support_rows for item in top),
        "loo_checked": max(item.loo_checked for item in top),
        "loo_passed": max(item.loo_passed for item in top),
    }


def score_rows(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = family_for(row)
        correct = verify_answer(row["answer"], row[prediction_key])
        truncated = boolish(row.get("truncated", False))
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


def main() -> int:
    rows = read_csv(BASELINE_CSV)
    baseline = score_rows(rows, "prediction")
    assert_baseline(baseline)
    already_known = {row["id"] for row in read_csv(V414_ACCEPTED_CSV)}
    audit: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    integrated: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["v427_prediction"] = row["prediction"]
        out["v427_applied"] = False
        if family_for(row) != "equation_transform":
            integrated.append(out)
            continue
        cand, meta = predict(row.get("prompt", ""))
        base_correct = verify_answer(row["answer"], row["prediction"])
        cand_correct = bool(cand) and verify_answer(row["answer"], cand)
        audit_row = {
            "id": row["id"],
            "answer": row["answer"],
            "baseline_prediction": row["prediction"],
            "baseline_correct": base_correct,
            "candidate_prediction": cand or "",
            "candidate_correct": cand_correct,
            "already_known_v414": row["id"] in already_known,
            "status": meta.get("status", ""),
            "reason": meta.get("reason", ""),
            "proof": json.dumps(meta, sort_keys=True),
        }
        audit.append(audit_row)
        if cand:
            if base_correct and not cand_correct:
                conflicts.append(audit_row)
            elif (not base_correct) and cand_correct and row["id"] not in already_known:
                accepted.append(audit_row)
                out["v427_prediction"] = cand
                out["v427_applied"] = True
        integrated.append(out)

    projected = score_rows(integrated, "v427_prediction")
    candidate_count = sum(1 for row in audit if row["status"] == "candidate")
    projected_total = int(projected["total"]["correct"])
    projected_eq = int(projected["families"].get("equation_transform", {}).get("correct", 0))
    projected_bit = int(projected["families"].get("bit_manipulation", {}).get("correct", 0))
    decision = (
        "v427_safe_gain_found"
        if accepted and not conflicts and projected_total > EXPECTED_BASELINE["correct"] and projected_eq > EXPECTED_BASELINE["equation_transform_correct"]
        else "v427_no_safe_new_gain"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "answer",
        "baseline_prediction",
        "baseline_correct",
        "candidate_prediction",
        "candidate_correct",
        "already_known_v414",
        "status",
        "reason",
        "proof",
    ]
    write_csv(OUT_DIR / "v427_symbolic_alphabet_shift_audit.csv", audit, cols)
    write_csv(OUT_DIR / "v427_symbolic_alphabet_shift_accepted.csv", accepted, cols)
    write_csv(OUT_DIR / "v427_symbolic_alphabet_shift_conflicts.csv", conflicts, cols)
    manifest = {
        "schema_version": "kg1_v427_symbolic_alphabet_shift_gate_v1",
        "generated_at_utc": utc_now(),
        "baseline": EXPECTED_BASELINE,
        "candidate_count": candidate_count,
        "accepted_new_gain_count": len(accepted),
        "conflict_count": len(conflicts),
        "projected": {
            "correct": projected_total,
            "equation_transform_correct": projected_eq,
            "bit_manipulation_correct": projected_bit,
            "truncated": int(projected["total"].get("truncated", 0)),
        },
        "decision": {
            "decision": decision,
            "hf_gpu_allowed": False,
            "reason": f"accepted_new_gain={len(accepted)}; conflicts={len(conflicts)}; projected={projected_total}/315 eq={projected_eq}/155 bit={projected_bit}/160",
            "next_action": (
                "Use accepted rows as CPU teacher only; adapter/package gate still required."
                if decision == "v427_safe_gain_found"
                else "Reject alphabet/ASCII shift DSL for training; continue with another class or stop GPU spending."
            ),
        },
        "outputs": {
            "manifest_json": str((OUT_DIR / "v427_symbolic_alphabet_shift_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V427_SYMBOLIC_ALPHABET_SHIFT_GATE.md").relative_to(ROOT)),
            "audit_csv": str((OUT_DIR / "v427_symbolic_alphabet_shift_audit.csv").relative_to(ROOT)),
            "accepted_csv": str((OUT_DIR / "v427_symbolic_alphabet_shift_accepted.csv").relative_to(ROOT)),
            "conflicts_csv": str((OUT_DIR / "v427_symbolic_alphabet_shift_conflicts.csv").relative_to(ROOT)),
        },
    }
    write_json(OUT_DIR / "v427_symbolic_alphabet_shift_manifest.json", manifest)
    report = [
        "# V427 Symbolic Alphabet Shift Gate",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "| State | Total | equation_transform | bit_manipulation | Truncated |",
        "|---|---:|---:|---:|---:|",
        f"| V291/V290 baseline | `{EXPECTED_BASELINE['correct']}/315` | `{EXPECTED_BASELINE['equation_transform_correct']}/155` | `{EXPECTED_BASELINE['bit_manipulation_correct']}/160` | `0` |",
        f"| V427 projection | `{projected_total}/315` | `{projected_eq}/155` | `{projected_bit}/160` | `{manifest['projected']['truncated']}` |",
        "",
        f"- Candidate rows: `{candidate_count}`.",
        f"- Accepted new gains outside V414: `{len(accepted)}`.",
        f"- Conflicts/losses: `{len(conflicts)}`.",
        "",
        f"Decision: `{decision}`. {manifest['decision']['reason']}",
    ]
    (OUT_DIR / "V427_SYMBOLIC_ALPHABET_SHIFT_GATE.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
