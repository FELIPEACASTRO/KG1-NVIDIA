#!/usr/bin/env python3
"""Build V443 CPU-certified equation preference pairs.

The builder uses only prompt examples to infer a small symbolic rule. The public
train answer is used only after the rule is frozen, for audit. Weak/full rows are
not used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V439_MANIFEST = ROOT / "artifacts/v439_final_answer_only_pairs/20260515T_v439_final_answer_only/v439_final_answer_only_pairs_manifest.json"


@dataclass(frozen=True)
class FrozenRule:
    rule_class: str
    scope: str
    program_label: str
    prediction: str
    support_rows: int
    candidate_count: int
    loo_checked: int
    loo_passed: int
    renaming_checked: int
    renaming_passed: int
    mdl_score: int


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def to_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_prompt(prompt: str) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str] | None]:
    examples: list[tuple[str, str, str, str]] = []
    for raw in str(prompt or "").splitlines():
        if " = " not in raw:
            continue
        lhs, rhs = raw.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5 and rhs:
            examples.append((lhs[:2], lhs[2], lhs[3:], rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt or ""))
    query_text = match.group(1).strip() if match else ""
    query = (query_text[:2], query_text[2], query_text[3:]) if len(query_text) == 5 else None
    return examples, query


def unique_order(text: str) -> str:
    out: list[str] = []
    for char in text:
        if char not in out:
            out.append(char)
    return "".join(out)


def ordered_intersection(left: str, right: str) -> str:
    return "".join(char for char in left if char in right)


def ordered_difference(left: str, right: str) -> str:
    return "".join(char for char in left if char not in right)


def counted_intersection(left: str, right: str) -> str:
    remaining = Counter(right)
    out: list[str] = []
    for char in left:
        if remaining[char] > 0:
            out.append(char)
            remaining[char] -= 1
    return "".join(out)


def prefix(text: str, size: int) -> str:
    if not text:
        return ""
    return (text * size)[:size]


def transform_values(left: str, op: str, right: str) -> dict[str, str]:
    token = left + op + right
    base: dict[str, str] = {
        "left": left,
        "right": right,
        "op": op,
        "token": token,
        "concat_lr": left + right,
        "concat_rl": right + left,
        "reverse_concat_lr": (left + right)[::-1],
        "reverse_concat_rl": (right + left)[::-1],
        "reverse_token": token[::-1],
        "reverse_left": left[::-1],
        "reverse_right": right[::-1],
        "intersection_lr": ordered_intersection(left, right),
        "intersection_rl": ordered_intersection(right, left),
        "difference_lr": ordered_difference(left, right),
        "difference_rl": ordered_difference(right, left),
        "symdiff_lr": ordered_difference(left, right) + ordered_difference(right, left),
        "symdiff_rl": ordered_difference(right, left) + ordered_difference(left, right),
        "union_lr": unique_order(left + right),
        "union_rl": unique_order(right + left),
        "counted_intersection_lr": counted_intersection(left, right),
        "counted_intersection_rl": counted_intersection(right, left),
        "left_first": left[:1],
        "left_second": left[1:],
        "right_first": right[:1],
        "right_second": right[1:],
        "outer": left[:1] + right[1:],
        "inner": left[1:] + right[:1],
        "swap_outer": right[1:] + left[:1],
        "swap_inner": right[:1] + left[1:],
        "left_op": left + op,
        "op_right": op + right,
        "op_left": op + left,
        "right_op": right + op,
    }
    values = dict(base)
    for name, value in list(base.items()):
        if value:
            values[f"{name}_x2"] = value * 2
            for size in (1, 2, 3, 4, 5):
                values[f"{name}_prefix{size}"] = prefix(value, size)
                values[f"{name}_suffix{size}"] = (value * size)[-size:]
    return values


def token_text(left: str, op: str, right: str) -> str:
    return left + op + right


def infer_direct_transform(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
) -> list[FrozenRule]:
    q_left, q_op, q_right = query
    scoped = [item for item in examples if scope == "global" or item[1] == q_op]
    if len(scoped) < 2:
        return []
    candidates: dict[str, list[str]] = {}
    for name in sorted(transform_values(scoped[0][0], scoped[0][1], scoped[0][2])):
        if all(transform_values(left, op, right).get(name) == rhs for left, op, right, rhs in scoped):
            pred = transform_values(q_left, q_op, q_right).get(name)
            if pred:
                candidates.setdefault(pred, []).append(name)
    return freeze_prediction_candidates(
        examples,
        query,
        scope=scope,
        rule_class="direct_string_transform",
        candidates=candidates,
    )


def infer_slot_substitution(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
    global_mapping: bool,
) -> list[FrozenRule]:
    q_left, q_op, q_right = query
    scoped = [item for item in examples if scope == "global" or item[1] == q_op]
    if len(scoped) < 3:
        return []
    rhs_lengths = sorted({len(rhs) for *_prefix, rhs in scoped if 1 <= len(rhs) <= 5})
    candidates: dict[str, list[str]] = {}
    for out_len in rhs_lengths:
        length_rows = [item for item in scoped if len(item[3]) == out_len]
        if len(length_rows) < 3:
            continue
        for positions in itertools.product(range(5), repeat=out_len):
            maps: list[dict[str, str]] = [dict() for _ in range(out_len if not global_mapping else 1)]
            ok = True
            for left, op, right, rhs in length_rows:
                source = token_text(left, op, right)
                for idx, pos in enumerate(positions):
                    key = source[pos]
                    value = rhs[idx]
                    mapping = maps[0] if global_mapping else maps[idx]
                    if key in mapping and mapping[key] != value:
                        ok = False
                        break
                    mapping[key] = value
                if not ok:
                    break
            if not ok:
                continue
            query_source = token_text(q_left, q_op, q_right)
            out: list[str] = []
            for idx, pos in enumerate(positions):
                mapping = maps[0] if global_mapping else maps[idx]
                key = query_source[pos]
                if key not in mapping:
                    ok = False
                    break
                out.append(mapping[key])
            if not ok:
                continue
            label = (
                f"{'global' if global_mapping else 'slot'}_map:"
                + ",".join(str(pos) for pos in positions)
            )
            candidates.setdefault("".join(out), []).append(label)
    return freeze_prediction_candidates(
        examples,
        query,
        scope=scope,
        rule_class="global_substitution" if global_mapping else "slot_substitution",
        candidates=candidates,
    )


def infer_candidate_predictions(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
) -> list[FrozenRule]:
    rules: list[FrozenRule] = []
    for scope in ("same_operator", "global"):
        rules.extend(infer_direct_transform(examples, query, scope=scope))
        rules.extend(infer_slot_substitution(examples, query, scope=scope, global_mapping=False))
        rules.extend(infer_slot_substitution(examples, query, scope=scope, global_mapping=True))
    return sorted(rules, key=lambda item: (item.mdl_score, -item.support_rows, item.rule_class, item.program_label))


def freeze_prediction_candidates(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
    rule_class: str,
    candidates: dict[str, list[str]],
) -> list[FrozenRule]:
    unique_predictions = {pred: labels for pred, labels in candidates.items() if pred}
    if len(unique_predictions) != 1:
        return []
    prediction, labels = next(iter(unique_predictions.items()))
    label = sorted(labels, key=lambda item: (len(item), item))[0]
    support_rows = len([item for item in examples if scope == "global" or item[1] == query[1]])
    candidate_count = len(unique_predictions)
    loo_checked, loo_passed = loo_score(examples, query, scope=scope, rule_class=rule_class, selected_label=label)
    renaming_checked, renaming_passed = renaming_score(examples, query, scope=scope, rule_class=rule_class, expected_prediction=prediction)
    mdl = len(label) + (0 if scope == "global" else 3) + (2 if rule_class == "direct_string_transform" else 8)
    if loo_checked < max(2, min(3, support_rows - 1)) or loo_checked != loo_passed:
        return []
    if renaming_checked == 0 or renaming_checked != renaming_passed:
        return []
    return [
        FrozenRule(
            rule_class=rule_class,
            scope=scope,
            program_label=label,
            prediction=prediction,
            support_rows=support_rows,
            candidate_count=candidate_count,
            loo_checked=loo_checked,
            loo_passed=loo_passed,
            renaming_checked=renaming_checked,
            renaming_passed=renaming_passed,
            mdl_score=mdl,
        )
    ]


def infer_with_restricted_rule(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
    rule_class: str,
    selected_label: str | None = None,
) -> str | None:
    rules = []
    if rule_class == "direct_string_transform":
        rules = infer_direct_transform_no_checks(examples, query, scope=scope)
    elif rule_class == "slot_substitution":
        rules = infer_slot_substitution_no_checks(examples, query, scope=scope, global_mapping=False)
    elif rule_class == "global_substitution":
        rules = infer_slot_substitution_no_checks(examples, query, scope=scope, global_mapping=True)
    for pred, labels in rules.items():
        if selected_label is None or selected_label in labels:
            return pred
    return None


def infer_direct_transform_no_checks(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
) -> dict[str, list[str]]:
    q_left, q_op, q_right = query
    scoped = [item for item in examples if scope == "global" or item[1] == q_op]
    if len(scoped) < 1:
        return {}
    candidates: dict[str, list[str]] = {}
    for name in sorted(transform_values(scoped[0][0], scoped[0][1], scoped[0][2])):
        if all(transform_values(left, op, right).get(name) == rhs for left, op, right, rhs in scoped):
            pred = transform_values(q_left, q_op, q_right).get(name)
            if pred:
                candidates.setdefault(pred, []).append(name)
    return candidates


def infer_slot_substitution_no_checks(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
    global_mapping: bool,
) -> dict[str, list[str]]:
    q_left, q_op, q_right = query
    scoped = [item for item in examples if scope == "global" or item[1] == q_op]
    if len(scoped) < 1:
        return {}
    candidates: dict[str, list[str]] = {}
    rhs_lengths = sorted({len(rhs) for *_prefix, rhs in scoped if 1 <= len(rhs) <= 5})
    for out_len in rhs_lengths:
        length_rows = [item for item in scoped if len(item[3]) == out_len]
        if not length_rows:
            continue
        for positions in itertools.product(range(5), repeat=out_len):
            maps: list[dict[str, str]] = [dict() for _ in range(out_len if not global_mapping else 1)]
            ok = True
            for left, op, right, rhs in length_rows:
                source = token_text(left, op, right)
                for idx, pos in enumerate(positions):
                    key = source[pos]
                    value = rhs[idx]
                    mapping = maps[0] if global_mapping else maps[idx]
                    if key in mapping and mapping[key] != value:
                        ok = False
                        break
                    mapping[key] = value
                if not ok:
                    break
            if not ok:
                continue
            query_source = token_text(q_left, q_op, q_right)
            out: list[str] = []
            for idx, pos in enumerate(positions):
                mapping = maps[0] if global_mapping else maps[idx]
                key = query_source[pos]
                if key not in mapping:
                    ok = False
                    break
                out.append(mapping[key])
            if not ok:
                continue
            label = (
                f"{'global' if global_mapping else 'slot'}_map:"
                + ",".join(str(pos) for pos in positions)
            )
            candidates.setdefault("".join(out), []).append(label)
    return candidates


def loo_score(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
    rule_class: str,
    selected_label: str,
) -> tuple[int, int]:
    checked = 0
    passed = 0
    for idx, (left, op, right, rhs) in enumerate(examples):
        if scope != "global" and op != query[1]:
            continue
        train = [row for row_idx, row in enumerate(examples) if row_idx != idx]
        pred = infer_with_restricted_rule(
            train,
            (left, op, right),
            scope=scope,
            rule_class=rule_class,
            selected_label=selected_label,
        )
        if pred is None:
            continue
        checked += 1
        if pred == rhs:
            passed += 1
    return checked, passed


def rename_text(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(char, char) for char in text)


def rename_examples(
    examples: list[tuple[str, str, str, str]],
    mapping: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    renamed: list[tuple[str, str, str, str]] = []
    for left, op, right, rhs in examples:
        renamed.append((rename_text(left, mapping), rename_text(op, mapping), rename_text(right, mapping), rename_text(rhs, mapping)))
    return renamed


def make_renaming(chars: list[str], salt: int) -> dict[str, str]:
    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()[]{}<>?/|\\:;`~+-_=,\"'")
    pool = [char for char in pool if char not in chars]
    rotated = pool[salt:] + pool[:salt]
    return {char: rotated[idx] for idx, char in enumerate(chars)}


def renaming_score(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
    *,
    scope: str,
    rule_class: str,
    expected_prediction: str,
) -> tuple[int, int]:
    chars = sorted(set("".join(left + op + right + rhs for left, op, right, rhs in examples) + "".join(query)))
    if len(chars) > 32:
        return 0, 0
    checked = 0
    passed = 0
    for salt in (0, 7, 19):
        mapping = make_renaming(chars, salt)
        inverse = {value: key for key, value in mapping.items()}
        renamed_examples = rename_examples(examples, mapping)
        q_left, q_op, q_right = query
        renamed_query = (
            rename_text(q_left, mapping),
            rename_text(q_op, mapping),
            rename_text(q_right, mapping),
        )
        rules = []
        if rule_class == "direct_string_transform":
            rules = infer_direct_transform_no_checks(renamed_examples, renamed_query, scope=scope)
        elif rule_class == "slot_substitution":
            rules = infer_slot_substitution_no_checks(renamed_examples, renamed_query, scope=scope, global_mapping=False)
        elif rule_class == "global_substitution":
            rules = infer_slot_substitution_no_checks(renamed_examples, renamed_query, scope=scope, global_mapping=True)
        if len(rules) != 1:
            continue
        renamed_pred = next(iter(rules))
        checked += 1
        if rename_text(renamed_pred, inverse) == expected_prediction:
            passed += 1
    return checked, passed


def boxed(answer: str) -> str:
    return f"Final answer: \\boxed{{{answer}}}"


def build_pair(row: dict[str, Any], frozen: FrozenRule, split: str) -> dict[str, Any]:
    meta = row.get("metadata", {})
    prompt = row.get("prompt", "")
    rejected_answer = str(meta.get("adapter_prediction", ""))
    return {
        "id": "v443_" + str(row.get("id", "")),
        "source": "v443_certified_equation_pair_builder",
        "family": "equation_transform",
        "subcategory": f"v443_{frozen.scope}_{frozen.rule_class}",
        "prompt": prompt,
        "chosen": boxed(frozen.prediction),
        "rejected": boxed(rejected_answer),
        "messages": [
            {"role": "system", "content": "Solve the KG1 puzzle. End with exactly one final answer in \\boxed{}."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": boxed(frozen.prediction)},
        ],
        "metadata": {
            "schema_version": "kg1_v443_certified_equation_pair_v1",
            "source_id": meta.get("source_id", ""),
            "source_pair_id": row.get("id", ""),
            "split": split,
            "adapter_prediction": rejected_answer,
            "adapter_exact_wrong_certificate": bool(meta.get("adapter_exact_wrong_certificate")),
            "adapter_repo": meta.get("adapter_repo", ""),
            "adapter_subfolder": meta.get("adapter_subfolder", ""),
            "prompt_sha256": meta.get("prompt_sha256", ""),
            "prompt_normalized_sha256": meta.get("prompt_normalized_sha256", ""),
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "labels_joined_after_collection_from_public_train": True,
            "rule_frozen_before_answer": True,
            "rule_unique_label_free": True,
            "rule_candidate_count": frozen.candidate_count,
            "program_or_rule": frozen.program_label,
            "rule_class": frozen.rule_class,
            "scope": frozen.scope,
            "support_rows": frozen.support_rows,
            "mdl_score": frozen.mdl_score,
            "leave_one_out_checked": frozen.loo_checked,
            "leave_one_out_pass": frozen.loo_passed == frozen.loo_checked,
            "renaming_stability_checked": frozen.renaming_checked,
            "renaming_stability_pass": frozen.renaming_passed == frozen.renaming_checked,
            "slot_alignment_stats": {
                "scope": frozen.scope,
                "program_label": frozen.program_label,
                "support_rows": frozen.support_rows,
            },
            "answer_audit": meta.get("answer", ""),
            "audit_correct_after_freeze": frozen.prediction == str(meta.get("answer", "")),
        },
    }


def audit_row(row: dict[str, Any], split: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    meta = row.get("metadata", {})
    prompt = str(row.get("prompt", ""))
    examples, query = parse_prompt(prompt)
    answer = str(meta.get("answer", ""))
    adapter_prediction = str(meta.get("adapter_prediction", ""))
    audit: dict[str, Any] = {
        "id": row.get("id", ""),
        "source_id": meta.get("source_id", ""),
        "split": split,
        "family": row.get("family", meta.get("family", "")),
        "source_rule_class": meta.get("rule_class", row.get("subcategory", "")),
        "answer": answer,
        "adapter_prediction": adapter_prediction,
        "examples": len(examples),
        "status": "abstain",
        "reason": "",
    }
    if row.get("family") != "equation_transform":
        audit["reason"] = "not_equation"
        return audit, None
    if query is None:
        audit["reason"] = "parse_query_failed"
        return audit, None
    if len(examples) < 3:
        audit["reason"] = "too_few_examples"
        return audit, None

    frozen_rules = infer_candidate_predictions(examples, query)
    if not frozen_rules:
        audit["reason"] = "no_unique_certified_rule"
        return audit, None
    frozen = frozen_rules[0]
    audit.update(
        {
            "status": "candidate",
            "reason": "certified_rule_found",
            "prediction": frozen.prediction,
            "candidate_correct": frozen.prediction == answer,
            "candidate_conflict": frozen.prediction != answer,
            "rule_class": frozen.rule_class,
            "scope": frozen.scope,
            "program_label": frozen.program_label,
            "support_rows": frozen.support_rows,
            "candidate_count": frozen.candidate_count,
            "mdl_score": frozen.mdl_score,
            "loo_checked": frozen.loo_checked,
            "loo_passed": frozen.loo_passed,
            "renaming_checked": frozen.renaming_checked,
            "renaming_passed": frozen.renaming_passed,
        }
    )
    if frozen.prediction != answer:
        return audit, None
    if frozen.prediction == adapter_prediction:
        audit["status"] = "abstain"
        audit["reason"] = "adapter_already_same_prediction"
        return audit, None
    return audit, build_pair(row, frozen, split)


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("metadata", {}).get("source_id", row.get("id", "")))
        bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest(), 16) % 5
        (val if bucket == 0 else train).append(row)
    return train, val


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    v439_manifest = read_json(args.v439_manifest_json)
    source_rows = read_jsonl(to_path(v439_manifest["train_jsonl"])) + read_jsonl(to_path(v439_manifest["val_jsonl"]))

    audit_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for row in source_rows:
        split = str(row.get("metadata", {}).get("split", ""))
        audit, pair = audit_row(row, split)
        audit_rows.append(audit)
        if pair:
            pairs.append(pair)

    train_pairs, val_pairs = split_rows(pairs)
    pair_train_jsonl = output_dir / f"{args.label}_preferences_train.jsonl"
    pair_val_jsonl = output_dir / f"{args.label}_preferences_val.jsonl"
    audit_csv = output_dir / f"{args.label}_audit.csv"
    manifest_json = output_dir / f"{args.label}_manifest.json"

    write_jsonl(pair_train_jsonl, train_pairs)
    write_jsonl(pair_val_jsonl, val_pairs)
    write_csv(audit_csv, audit_rows)

    status_counts = Counter(str(row.get("status", "")) for row in audit_rows)
    reason_counts = Counter(str(row.get("reason", "")) for row in audit_rows)
    pair_rule_counts = Counter(str(pair.get("subcategory", "")) for pair in pairs)
    independent_modes = len(pair_rule_counts)
    hf_gpu_allowed = (
        len(pairs) >= args.min_pairs
        and independent_modes >= args.min_independent_modes
        and len(val_pairs) >= args.min_val_pairs
    )
    decision = {
        "decision": "certified_pairs_ready_for_pre_paid_gate" if hf_gpu_allowed else "cpu_pairs_insufficient_no_gpu",
        "reason": (
            f"pairs={len(pairs)}, modes={independent_modes}, val_pairs={len(val_pairs)}"
        ),
        "next_action": (
            "Run integration/tokenization gate before any HF job."
            if hf_gpu_allowed
            else "Expand certified CPU builder; do not launch GPU from V443 yet."
        ),
    }

    manifest: dict[str, Any] = {
        "schema_version": "kg1_v443_certified_equation_pair_builder_v1",
        "label": args.label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v439_manifest_json": repo_rel(args.v439_manifest_json),
            "source_rows": len(source_rows),
        },
        "summary": {
            "audit_rows": len(audit_rows),
            "candidate_rows": status_counts.get("candidate", 0),
            "certified_pair_rows": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "status_counts": dict(status_counts),
            "reason_counts": dict(reason_counts),
            "pair_rule_counts": dict(pair_rule_counts),
            "independent_modes": independent_modes,
        },
        "hf_gpu_allowed": hf_gpu_allowed,
        "decision": decision,
        "outputs": {
            "audit_csv": repo_rel(audit_csv),
            "preferences_train_jsonl": repo_rel(pair_train_jsonl),
            "preferences_train_sha256": sha256_path(pair_train_jsonl),
            "preferences_val_jsonl": repo_rel(pair_val_jsonl),
            "preferences_val_sha256": sha256_path(pair_val_jsonl),
            "manifest_json": repo_rel(manifest_json),
        },
    }
    write_json(manifest_json, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v439-manifest-json", type=Path, default=DEFAULT_V439_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v443_certified_equation_pair_builder")
    parser.add_argument("--min-pairs", type=int, default=24)
    parser.add_argument("--min-val-pairs", type=int, default=4)
    parser.add_argument("--min-independent-modes", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=== V443 CERTIFIED EQUATION PAIR BUILDER START ===", flush=True)
    print("v439_manifest_json =", args.v439_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    manifest = run(args)
    print("summary =", json.dumps(manifest["summary"], sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("hf_gpu_allowed =", manifest["hf_gpu_allowed"], flush=True)
    print("manifest_json =", manifest["outputs"]["manifest_json"], flush=True)
    print("=== V443 CERTIFIED EQUATION PAIR BUILDER END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
