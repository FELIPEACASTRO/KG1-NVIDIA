#!/usr/bin/env python3
"""Audit a cryptarithm-style solver on current equation misses.

V273 is CPU-only. It adapts the public tonghuikang/nemotron cryptarithm insight:
for prompts of the form AB op CD = result, infer a symbol-to-digit mapping and
an operator-to-arithmetic mapping over add, absolute difference, multiplication,
concat, and reverse concat. Weak labels are used only to audit the solver class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402
from analyze_v238_alice_parser_probes import answers_equal, parse_alice_prompt  # noqa: E402


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_BASELINE_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_BASELINE_FILENAME = (
    "evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/"
    "v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv"
)

OPS = {
    "add": lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
    "concat": lambda a, b: a * 100 + b,
    "rev_concat": lambda a, b: b * 100 + a,
}

AUDIT_COLUMNS = [
    "id",
    "status",
    "prediction",
    "answer",
    "baseline_prediction",
    "verified_by_weak_label",
    "incorrect_by_weak_label",
    "promotable_after_class_gate",
    "solver_mode",
    "query",
    "example_count",
    "answer_vote_count",
    "total_solution_votes",
    "mapping",
    "operator_mapping",
    "proof",
]

SUMMARY_COLUMNS = [
    "solver_mode",
    "rows",
    "candidate_rows",
    "verified_candidates",
    "incorrect_candidates",
    "abstain_rows",
    "promotable_after_class_gate",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def download_file(repo_id: str, filename: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V273") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename.strip("/"),
            local_dir=str(local_dir),
            token=token,
        )
    )


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", "")).strip()
    prediction = str(row.get("prediction", "")).strip()
    family = canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt))
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": answer,
        "prediction": prediction,
        "family": family,
        "prompt_sha256": sha256_text(prompt),
        "correct_bool": verify_answer(answer, prediction),
        "truncated_bool": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
    }


def row_contract(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 315:
        raise RuntimeError(f"expected 315 rows, got {len(rows)}")
    payload = "\n".join(
        f"{row['id']}\t{row['family']}\t{row['answer']}\t{row['prompt_sha256']}"
        for row in sorted(rows, key=lambda item: item["id"])
    )
    return sha256_text(payload)


def digits_for_number(value: int, *, width4: bool = False) -> tuple[int, ...]:
    if width4:
        if value < 0 or value >= 10000:
            return ()
        return (value // 1000, (value // 100) % 10, (value // 10) % 10, value % 10)
    if value == 0:
        return (0,)
    digits: list[int] = []
    while value > 0:
        digits.append(value % 10)
        value //= 10
    return tuple(reversed(digits))


def is_concat_example(example: tuple[str, str, str, str, str, tuple[str, ...]]) -> bool:
    s0, s1, _, s3, s4, out = example
    return out == (s0, s1, s3, s4) or out == (s3, s4, s0, s1)


def parse_cryptarithm(examples: list[tuple[str, str]], query: str) -> tuple[list[tuple[str, str, str, str, str, tuple[str, ...]]], tuple[str, str, str, str, str] | None, str]:
    parsed_examples = []
    for lhs, rhs in examples:
        if len(lhs) != 5:
            return [], None, "example_lhs_not_len5"
        parsed_examples.append((lhs[0], lhs[1], lhs[2], lhs[3], lhs[4], tuple(rhs)))
    if len(query) != 5:
        return [], None, "query_not_len5"
    return parsed_examples, (query[0], query[1], query[2], query[3], query[4]), "ok"


class CryptarithmSolver:
    def __init__(
        self,
        examples: list[tuple[str, str, str, str, str, tuple[str, ...]]],
        query: tuple[str, str, str, str, str],
        *,
        unique_digits: bool,
        max_solution_votes: int,
    ) -> None:
        self.examples = examples
        self.query = query
        self.unique_digits = unique_digits
        self.max_solution_votes = max_solution_votes
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.operator_mapping: dict[str, str] = {}
        self.answers: Counter[str] = Counter()
        self.answer_info: dict[str, tuple[dict[str, int], dict[str, str]]] = {}

    def solve(self) -> dict[str, Any]:
        self._walk(0)
        if not self.answers:
            return {"status": "abstain", "prediction": "", "proof": "no_solution"}
        prediction, votes = self.answers.most_common(1)[0]
        total = sum(self.answers.values())
        if len(self.answers) > 1 and votes < total * 0.5:
            return {
                "status": "abstain",
                "prediction": "",
                "proof": f"weak_consensus unique_answers={len(self.answers)} top_votes={votes} total={total}",
                "answer_vote_count": votes,
                "total_solution_votes": total,
            }
        mapping, opmap = self.answer_info.get(prediction, ({}, {}))
        return {
            "status": "candidate",
            "prediction": prediction,
            "proof": f"unique_digits={self.unique_digits} unique_answers={len(self.answers)} top_votes={votes} total={total}",
            "answer_vote_count": votes,
            "total_solution_votes": total,
            "mapping": mapping,
            "operator_mapping": opmap,
        }

    def _values(self, symbol: str) -> range | tuple[int, ...]:
        if symbol in self.mapping:
            return (self.mapping[symbol],)
        if self.unique_digits:
            return tuple(digit for digit in range(10) if digit not in self.used)
        return range(10)

    def _assign(self, symbol: str, digit: int) -> bool | None:
        if symbol in self.mapping:
            return False if self.mapping[symbol] == digit else None
        if self.unique_digits and digit in self.used:
            return None
        self.mapping[symbol] = digit
        if self.unique_digits:
            self.used.add(digit)
        return True

    def _undo(self, symbol: str, was_new: bool | None) -> None:
        if was_new is True:
            if self.unique_digits:
                self.used.discard(self.mapping[symbol])
            del self.mapping[symbol]

    def _candidate_ops(self, op_symbol: str, out_len: int) -> list[str]:
        if op_symbol in self.operator_mapping:
            return [self.operator_mapping[op_symbol]]
        ops = []
        if out_len <= 3:
            ops.append("add")
        if out_len <= 2:
            ops.append("abs_diff")
        if out_len <= 4:
            ops.append("mul")
        if out_len == 4:
            ops.extend(["concat", "rev_concat"])
        return ops

    def _walk(self, index: int) -> None:
        if sum(self.answers.values()) >= self.max_solution_votes:
            return
        if index == len(self.examples):
            self._compute_query()
            return
        s0, s1, op, s3, s4, out_symbols = self.examples[index]
        for d0 in self._values(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._values(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                left = d0 * 10 + d1
                for d3 in self._values(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._values(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        right = d3 * 10 + d4
                        for op_name in self._candidate_ops(op, len(out_symbols)):
                            result_value = OPS[op_name](left, right)
                            result_digits = digits_for_number(result_value, width4=op_name in {"concat", "rev_concat"})
                            if len(result_digits) != len(out_symbols):
                                continue
                            new_assigns: list[tuple[str, bool | None]] = []
                            ok = True
                            for symbol, digit in zip(out_symbols, result_digits):
                                was_new = self._assign(symbol, digit)
                                if was_new is None:
                                    ok = False
                                    break
                                new_assigns.append((symbol, was_new))
                            if ok:
                                op_new = op not in self.operator_mapping
                                if op_new:
                                    self.operator_mapping[op] = op_name
                                self._walk(index + 1)
                                if op_new:
                                    del self.operator_mapping[op]
                            for symbol, was_new in reversed(new_assigns):
                                self._undo(symbol, was_new)
                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _compute_query(self) -> None:
        s0, s1, op, s3, s4 = self.query
        if any(symbol not in self.mapping for symbol in (s0, s1, s3, s4)):
            return
        left = self.mapping[s0] * 10 + self.mapping[s1]
        right = self.mapping[s3] * 10 + self.mapping[s4]
        digit_to_symbol: dict[int, str] = {}
        for symbol, digit in self.mapping.items():
            digit_to_symbol.setdefault(digit, symbol)
        op_candidates = [self.operator_mapping[op]] if op in self.operator_mapping else list(OPS)
        for op_name in op_candidates:
            result_value = OPS[op_name](left, right)
            result_digits = digits_for_number(result_value, width4=op_name in {"concat", "rev_concat"})
            if not result_digits:
                continue
            output: list[str] = []
            for digit in result_digits:
                if digit not in digit_to_symbol:
                    output = []
                    break
                output.append(digit_to_symbol[digit])
            if not output:
                continue
            answer = "".join(output)
            self.answers[answer] += 1
            self.answer_info.setdefault(answer, (dict(self.mapping), {**self.operator_mapping, op: op_name}))


def solve_cryptarithm(examples: list[tuple[str, str]], query: str, max_solution_votes: int) -> dict[str, Any]:
    parsed_examples, parsed_query, status = parse_cryptarithm(examples, query)
    if status != "ok" or parsed_query is None:
        return {"status": "abstain", "prediction": "", "solver_mode": "parse_gate", "proof": status}
    if not parsed_examples:
        return {"status": "abstain", "prediction": "", "solver_mode": "constraint_gate", "proof": "no_examples"}
    concat_ops = {ex[2] for ex in parsed_examples if is_concat_example(ex)}
    nonconcat_ops = {ex[2] for ex in parsed_examples if not is_concat_example(ex)}
    qop = parsed_query[2]
    if qop in concat_ops and qop not in nonconcat_ops:
        for ex in parsed_examples:
            if ex[2] != qop or not is_concat_example(ex):
                continue
            if ex[5] == (ex[0], ex[1], ex[3], ex[4]):
                return {"status": "candidate", "prediction": parsed_query[0] + parsed_query[1] + parsed_query[3] + parsed_query[4], "solver_mode": "concat_shortcut", "proof": "query_operator_observed_concat"}
            return {"status": "candidate", "prediction": parsed_query[3] + parsed_query[4] + parsed_query[0] + parsed_query[1], "solver_mode": "concat_shortcut", "proof": "query_operator_observed_rev_concat"}
    arithmetic_examples = [ex for ex in parsed_examples if not is_concat_example(ex)]
    if not arithmetic_examples:
        return {"status": "abstain", "prediction": "", "solver_mode": "constraint_gate", "proof": "no_arithmetic_examples_for_query_operator"}
    observed_input_symbols = {
        symbol
        for ex in arithmetic_examples
        for symbol in (ex[0], ex[1], ex[3], ex[4])
    }
    missing_query_symbols = sorted(set((parsed_query[0], parsed_query[1], parsed_query[3], parsed_query[4])) - observed_input_symbols)
    if missing_query_symbols:
        return {
            "status": "abstain",
            "prediction": "",
            "solver_mode": "constraint_gate",
            "proof": "query_symbols_not_observed_in_arithmetic_examples=" + "".join(missing_query_symbols),
        }
    for mode, unique in (("cryptarithm_unique_digit", True), ("cryptarithm_nonunique_digit", False)):
        result = CryptarithmSolver(arithmetic_examples, parsed_query, unique_digits=unique, max_solution_votes=max_solution_votes).solve()
        result["solver_mode"] = mode
        if result.get("status") == "candidate":
            return result
    return {"status": "abstain", "prediction": "", "solver_mode": "cryptarithm_solver", "proof": "no_candidate"}


def audit_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    examples, query, parse_status = parse_alice_prompt(row["prompt"])
    if parse_status != "ok":
        result = {"status": "abstain", "prediction": "", "solver_mode": "alice_parse_gate", "proof": parse_status}
    else:
        result = solve_cryptarithm(examples, query, args.max_solution_votes)
    status = str(result.get("status", "abstain"))
    prediction = str(result.get("prediction", ""))
    verified = status == "candidate" and answers_equal(prediction, row["answer"])
    incorrect = status == "candidate" and not verified
    return {
        "id": row["id"],
        "status": status,
        "prediction": prediction,
        "answer": row["answer"],
        "baseline_prediction": row["prediction"],
        "verified_by_weak_label": verified,
        "incorrect_by_weak_label": incorrect,
        "promotable_after_class_gate": False,
        "solver_mode": result.get("solver_mode", ""),
        "query": query,
        "example_count": len(examples),
        "answer_vote_count": result.get("answer_vote_count", ""),
        "total_solution_votes": result.get("total_solution_votes", ""),
        "mapping": json.dumps(result.get("mapping", {}), sort_keys=True),
        "operator_mapping": json.dumps(result.get("operator_mapping", {}), sort_keys=True),
        "proof": str(result.get("proof", ""))[:500],
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["solver_mode"]), []).append(row)
    summary = []
    for mode, items in sorted(by_mode.items()):
        candidates = [row for row in items if row["status"] == "candidate"]
        verified = [row for row in candidates if row["verified_by_weak_label"]]
        incorrect = [row for row in candidates if row["incorrect_by_weak_label"]]
        promotable = bool(candidates) and not incorrect
        for row in items:
            if row["status"] == "candidate":
                row["promotable_after_class_gate"] = promotable
        summary.append(
            {
                "solver_mode": mode,
                "rows": len(items),
                "candidate_rows": len(candidates),
                "verified_candidates": len(verified),
                "incorrect_candidates": len(incorrect),
                "abstain_rows": len(items) - len(candidates),
                "promotable_after_class_gate": promotable,
            }
        )
    summary.sort(key=lambda row: (int(row["verified_candidates"]), -int(row["incorrect_candidates"]), str(row["solver_mode"])), reverse=True)
    return summary


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V273 CRYPTARITHM SOLVER AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("target_verified_gain =", args.target_verified_gain, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v273_") as temp_name:
        path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(path)]
        input_meta = {
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": path.name,
            "sha256": sha256_file(path),
            "rows": len(rows),
        }

    observed = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed, flush=True)
    if observed != args.expected_shared_row_contract_sha256:
        raise RuntimeError(f"row contract mismatch: expected {args.expected_shared_row_contract_sha256}, got {observed}")

    equation_misses = [
        row
        for row in rows
        if row["family"] == "equation_transform" and not row["correct_bool"] and not row["truncated_bool"]
    ]
    audit_rows = [audit_row(row, args) for row in equation_misses]
    summary_rows = summarize(audit_rows)
    verified_promotable = [
        row for row in audit_rows if row["verified_by_weak_label"] and row["promotable_after_class_gate"]
    ]
    incorrect_promotable = [row for row in audit_rows if row["incorrect_by_weak_label"] and row["promotable_after_class_gate"]]

    outputs = {
        "audit_csv": args.output_dir / "v273_cryptarithm_solver_candidate_audit.csv",
        "summary_csv": args.output_dir / "v273_cryptarithm_solver_summary.csv",
        "verified_promotable_csv": args.output_dir / "v273_verified_promotable_candidates.csv",
        "manifest_json": args.output_dir / "v273_cryptarithm_solver_audit_manifest.json",
    }
    write_csv(outputs["audit_csv"], audit_rows, AUDIT_COLUMNS)
    write_csv(outputs["summary_csv"], summary_rows, SUMMARY_COLUMNS)
    write_csv(outputs["verified_promotable_csv"], verified_promotable, AUDIT_COLUMNS)

    if len(verified_promotable) >= args.target_verified_gain and not incorrect_promotable:
        decision = "cryptarithm_solver_ready_for_guarded_override_eval"
        next_action = "Build a guarded weak eval candidate applying only promotable cryptarithm solver classes."
    elif verified_promotable:
        decision = "partial_cryptarithm_solver_signal"
        next_action = "Review promotable classes and extend solver coverage before eval."
    else:
        decision = "no_cryptarithm_solver_signal"
        next_action = "Use gated solver traces or add richer symbolic operation families."

    manifest = {
        "schema_version": "kg1_v273_cryptarithm_solver_audit_v1",
        "generated_at_utc": utc_now(),
        "input": input_meta,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed,
        "equation_miss_rows": len(equation_misses),
        "summary": summary_rows,
        "verified_promotable_candidates": len(verified_promotable),
        "incorrect_promotable_candidates": len(incorrect_promotable),
        "external_basis": {
            "repo": "https://github.com/tonghuikang/nemotron",
            "file": "investigators/cryptarithm_deduce.py",
            "note": "Public progress-prize repository models AB op CD symbolic equations as digit/operator cryptarithms.",
        },
        "decision": {
            "decision": decision,
            "reason": (
                f"equation_misses={len(equation_misses)}; "
                f"verified_promotable={len(verified_promotable)}; "
                f"incorrect_promotable={len(incorrect_promotable)}"
            ),
            "next_action": next_action,
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("summary =", json.dumps(summary_rows, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V273 CRYPTARITHM SOLVER AUDIT END ===", flush=True)
    return manifest


def run_self_test() -> None:
    examples = [("ab+cd", "abcd"), ("ef+gh", "efgh")]
    result = solve_cryptarithm(examples, "ij+kl", 100)
    if result["status"] != "candidate" or result["prediction"] != "ijkl":
        raise AssertionError(result)
    print("v273_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v273_cryptarithm_solver_audit"))
    parser.add_argument("--run-id", default=f"v273-hf-cpu-cryptarithm-solver-audit-{utc_compact()}")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-verified-gain", type=int, default=4)
    parser.add_argument("--max-solution-votes", type=int, default=500)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
