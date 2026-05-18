#!/usr/bin/env python3
"""CPU-only learnability gate for KG1 trace training datasets.

V513 audits the active V510 training pool before any paid GPU job. It checks
whether the data is structurally suitable for transfer, not only whether JSONL
and tokenization are valid. This gate is intentionally conservative after the
V511 result: loss moved but weak ACC did not improve.

The script never trains, evaluates a model, packages artifacts, or submits to
Kaggle. It writes a manifest plus compact CSV/Markdown diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import PROMPT_SUFFIX, extract_final_answer, verify_answer  # noqa: E402


DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "artifacts/v510_canonical_training_dataset/v510_canonical_active_training_pool"
)
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_ROOT / "v510_canonical_active_training_pool_train.jsonl"
DEFAULT_VAL_JSONL = DEFAULT_DATA_ROOT / "v510_canonical_active_training_pool_val.jsonl"
DEFAULT_TOKENIZATION_MANIFEST = (
    DEFAULT_DATA_ROOT / "tokenization_gate_real_local/v286_generic_tokenization_gate_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v513_trace_learnability_gate"

ANTI_LEAK_FLAGS = (
    "weak_gate_rows_used_for_training",
    "gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
)
EXPECTED_ROLE_SEQUENCE = ("system", "user", "assistant")
OFFICIAL_LIKE_ROLE_SEQUENCE = ("user", "assistant")
MAX_ASSISTANT_WORDS_WARN = 1300
MAX_LOSS_TOKENS_WARN = 1300
MAX_PROMPT_TRUNCATION_RATE = 0.0
MAX_COMPLETION_DROPPED = 0
MAX_FALLBACK_MASKS = 0
MAX_BIT_ANSWER_ONLY_SHARE_FOR_GPU = 0.05
MIN_BIT_TRACE_ROWS_FOR_GPU = 32
MIN_EQUATION_TRACE_WORDS_P50_FOR_GPU = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def quantile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    index = max(0, min(index, len(ordered) - 1))
    return int(ordered[index])


def compact_counter(counter: Counter[str], limit: int = 20) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def load_jsonl(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            row["_split"] = split
            row["_path"] = str(path)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def row_messages(row: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", "", "", tuple()
    roles: list[str] = []
    content_by_role: dict[str, str] = {}
    for item in messages:
        if not isinstance(item, dict):
            roles.append("<non-dict>")
            continue
        role = str(item.get("role", ""))
        roles.append(role)
        content_by_role.setdefault(role, str(item.get("content", "")))
    return (
        content_by_role.get("system", ""),
        content_by_role.get("user", ""),
        content_by_role.get("assistant", ""),
        tuple(roles),
    )


def family_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("family") or row.get("task_type") or metadata.get("family") or "").strip()


def source_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("source") or metadata.get("source") or "").strip()


def subcategory_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("subcategory") or metadata.get("subcategory") or metadata.get("subtype") or "").strip()


def answer_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("answer") or metadata.get("answer") or "").strip()


def assistant_style(assistant: str, family: str) -> str:
    text = assistant.strip()
    lowered = text.lower()
    if not text:
        return "missing"
    line_count = len([line for line in text.splitlines() if line.strip()])
    has_box = "\\boxed{" in text
    has_rule = "rule:" in lowered or "reject " in lowered or "query " in lowered
    has_bit_terms = any(token in lowered for token in ("bit", "xor", "and", "or", "rot", "shl", "shr", "stride", "bitsum"))
    if family == "bit_manipulation" and has_box and line_count <= 1 and not has_bit_terms:
        return "bit_answer_only_boxed"
    if family == "bit_manipulation" and has_bit_terms:
        return "bit_trace_with_rule_terms"
    if family == "equation_transform" and has_rule and has_box:
        return "equation_short_rule_reject_boxed"
    if has_box:
        return "boxed_other"
    if "final answer:" in lowered:
        return "final_answer_unboxed"
    return "other"


BOX_RE = re.compile(r"\\boxed\{[^{}]*\}")
BIN_RE = re.compile(r"\b[01]{8}\b")
NUM_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?![A-Za-z_])")
SPACE_RE = re.compile(r"\s+")


def normalized_assistant_template(assistant: str) -> str:
    text = BOX_RE.sub(r"\\boxed{<ANS>}", assistant)
    text = BIN_RE.sub("<BIN8>", text)
    text = NUM_RE.sub("<NUM>", text)
    text = SPACE_RE.sub(" ", text).strip().lower()
    return text


def row_projection(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    system, user, assistant, roles = row_messages(row)
    prompt_contract = str(metadata.get("prompt_contract", "legacy_system"))
    family = family_of(row)
    answer = answer_of(row)
    source = source_of(row)
    subcategory = subcategory_of(row)
    row_id = str(row.get("id", "")).strip()
    prompt = str(row.get("prompt", "")).strip()
    extracted = extract_final_answer(assistant)
    style = assistant_style(assistant, family)
    words = len(assistant.split())
    lines = len([line for line in assistant.splitlines() if line.strip()])
    template = normalized_assistant_template(assistant)
    if prompt_contract == "official_like":
        expected_roles = OFFICIAL_LIKE_ROLE_SEQUENCE
        expected_user = prompt.strip() + str(metadata.get("prompt_suffix", PROMPT_SUFFIX)).rstrip()
        prompt_user_matches = bool(prompt and user.strip() == expected_user)
    else:
        expected_roles = EXPECTED_ROLE_SEQUENCE
        prompt_user_matches = bool(prompt and user.strip() == prompt.strip())
    return {
        "split": row["_split"],
        "path": row["_path"],
        "line_no": row["_line_no"],
        "id": row_id,
        "family": family,
        "source": source,
        "subcategory": subcategory,
        "prompt_sha256": sha256_text(prompt),
        "prompt_answer_sha256": sha256_text(prompt + "\n===ANSWER===\n" + answer),
        "answer": answer,
        "assistant": assistant,
        "assistant_sha256": sha256_text(assistant),
        "assistant_template_sha256": sha256_text(template),
        "assistant_template_preview": template[:220],
        "assistant_words": words,
        "assistant_lines": lines,
        "assistant_style": style,
        "extracted": extracted,
        "assistant_answer_matches": bool(answer and verify_answer(answer, extracted)),
        "role_sequence": ",".join(roles),
        "expected_role_sequence": ",".join(expected_roles),
        "prompt_contract": prompt_contract,
        "prompt_user_matches": prompt_user_matches,
        "system": system,
        "metadata": metadata,
    }


def summarize_rows(projected: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    lengths_by_family: dict[str, list[int]] = defaultdict(list)
    lines_by_family: dict[str, list[int]] = defaultdict(list)
    for row in projected:
        family_counts[str(row["family"])] += 1
        source_counts[str(row["source"])] += 1
        subcategory_counts[str(row["subcategory"])] += 1
        style_counts[str(row["assistant_style"])] += 1
        lengths_by_family[str(row["family"])].append(int(row["assistant_words"]))
        lines_by_family[str(row["family"])].append(int(row["assistant_lines"]))
    length_summary: dict[str, Any] = {}
    for family, values in sorted(lengths_by_family.items()):
        line_values = lines_by_family.get(family, [])
        length_summary[family] = {
            "assistant_word_min": min(values) if values else 0,
            "assistant_word_p50": quantile(values, 0.50),
            "assistant_word_p90": quantile(values, 0.90),
            "assistant_word_p99": quantile(values, 0.99),
            "assistant_word_max": max(values) if values else 0,
            "assistant_line_p50": quantile(line_values, 0.50),
            "assistant_line_max": max(line_values) if line_values else 0,
        }
    return {
        "rows": len(projected),
        "family_counts": compact_counter(family_counts, 50),
        "source_counts": compact_counter(source_counts, 50),
        "subcategory_counts": compact_counter(subcategory_counts, 80),
        "assistant_style_counts": compact_counter(style_counts, 50),
        "assistant_length_by_family": length_summary,
    }


def duplicate_groups(projected: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in projected:
        groups[str(row[key])].append(row)
    results: list[dict[str, Any]] = []
    for value, rows in groups.items():
        if len(rows) <= 1:
            continue
        answers = sorted({str(row["answer"]) for row in rows})
        families = sorted({str(row["family"]) for row in rows})
        subcategories = sorted({str(row["subcategory"]) for row in rows})
        results.append(
            {
                "key": value,
                "count": len(rows),
                "answer_count": len(answers),
                "family_count": len(families),
                "subcategory_count": len(subcategories),
                "answers_preview": "|".join(answers[:10]),
                "families": "|".join(families[:10]),
                "subcategories": "|".join(subcategories[:10]),
                "ids_preview": "|".join(str(row["id"]) for row in rows[:10]),
                "template_preview": str(rows[0].get("assistant_template_preview", "")),
            }
        )
    return sorted(results, key=lambda item: (-int(item["count"]), str(item["key"])))


def tokenization_findings(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        return {}, [{"code": "tokenization_manifest_missing", "severity": "blocker", "detail": str(path)}]
    manifest = read_json(path)
    findings: list[dict[str, Any]] = []
    decision_status = str(manifest.get("decision", {}).get("status", ""))
    if decision_status != "tokenization_gate_passed":
        findings.append({"code": "tokenization_gate_not_passed", "severity": "blocker", "detail": decision_status})
    for split_key in ("train", "validation"):
        split = manifest.get("tokenization", {}).get(split_key, {})
        prompt_truncation_rate = float(split.get("prompt_truncation_rate", 1.0) or 0.0)
        dropped = int(split.get("completion_tokens_dropped", 0) or 0)
        fallback_masks = int(split.get("fallback_masks", 0) or 0)
        loss_token_max = int(split.get("loss_token_max", 0) or 0)
        if prompt_truncation_rate > MAX_PROMPT_TRUNCATION_RATE:
            findings.append(
                {
                    "code": f"{split_key}_prompt_truncation_nonzero",
                    "severity": "blocker",
                    "detail": prompt_truncation_rate,
                }
            )
        if dropped > MAX_COMPLETION_DROPPED:
            findings.append(
                {"code": f"{split_key}_completion_tokens_dropped", "severity": "blocker", "detail": dropped}
            )
        if fallback_masks > MAX_FALLBACK_MASKS:
            findings.append({"code": f"{split_key}_fallback_masks", "severity": "blocker", "detail": fallback_masks})
        if loss_token_max > MAX_LOSS_TOKENS_WARN:
            findings.append({"code": f"{split_key}_loss_tokens_too_long", "severity": "warning", "detail": loss_token_max})
    return manifest, findings


def build_findings(projected: list[dict[str, Any]], tokenization_manifest: dict[str, Any], token_findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = list(token_findings)
    by_split: Counter[str] = Counter(str(row["split"]) for row in projected)
    if by_split.get("train", 0) == 0 or by_split.get("validation", 0) == 0:
        findings.append({"code": "missing_train_or_validation_split", "severity": "blocker", "detail": dict(by_split)})

    prompt_hash_to_answers: dict[str, set[str]] = defaultdict(set)
    prompt_hash_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_answer_seen: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    missing_flags: Counter[str] = Counter()
    true_flags: Counter[str] = Counter()
    style_by_family: Counter[str] = Counter()
    bit_answer_only = 0
    bit_rows = 0
    bit_trace_rows = 0
    equation_words: list[int] = []

    for row in projected:
        ids[str(row["id"])] += 1
        prompt_hash_to_answers[str(row["prompt_sha256"])].add(str(row["answer"]))
        prompt_hash_to_rows[str(row["prompt_sha256"])].append(row)
        prompt_answer_seen[str(row["prompt_answer_sha256"])] += 1
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for flag in ANTI_LEAK_FLAGS:
            if flag not in metadata:
                missing_flags[flag] += 1
            elif bool(metadata.get(flag)):
                true_flags[flag] += 1
        if not row["assistant_answer_matches"]:
            findings.append(
                {
                    "code": "assistant_final_answer_mismatch",
                    "severity": "blocker",
                    "detail": f"{row['split']}:{row['id']} expected={row['answer']} extracted={row['extracted']}",
                }
            )
        if row["role_sequence"] != row.get("expected_role_sequence", ",".join(EXPECTED_ROLE_SEQUENCE)):
            findings.append(
                {
                    "code": "unexpected_role_sequence",
                    "severity": "blocker",
                    "detail": f"{row['split']}:{row['id']} roles={row['role_sequence']}",
                }
            )
        if not row["prompt_user_matches"]:
            findings.append({"code": "prompt_user_mismatch", "severity": "blocker", "detail": f"{row['split']}:{row['id']}"})
        if int(row["assistant_words"]) > MAX_ASSISTANT_WORDS_WARN:
            findings.append(
                {
                    "code": "assistant_trace_too_long",
                    "severity": "warning",
                    "detail": f"{row['split']}:{row['id']} words={row['assistant_words']}",
                }
            )

        family = str(row["family"])
        style = str(row["assistant_style"])
        style_by_family[f"{family}:{style}"] += 1
        if family == "bit_manipulation":
            bit_rows += 1
            if style == "bit_answer_only_boxed":
                bit_answer_only += 1
            if style == "bit_trace_with_rule_terms":
                bit_trace_rows += 1
        if family == "equation_transform":
            equation_words.append(int(row["assistant_words"]))

    duplicate_ids = [key for key, count in ids.items() if key and count > 1]
    if duplicate_ids:
        findings.append(
            {
                "code": "duplicate_ids",
                "severity": "blocker",
                "detail": json.dumps(duplicate_ids[:20], sort_keys=True),
            }
        )
    conflicting_prompt_answers = [
        (key, sorted(values), [row["id"] for row in prompt_hash_to_rows[key][:10]])
        for key, values in prompt_hash_to_answers.items()
        if len(values) > 1
    ]
    if conflicting_prompt_answers:
        findings.append(
            {
                "code": "same_prompt_multiple_answers",
                "severity": "blocker",
                "detail": json.dumps(conflicting_prompt_answers[:5], sort_keys=True),
            }
        )
    duplicate_prompt_answer = int(sum(count - 1 for count in prompt_answer_seen.values() if count > 1))
    if duplicate_prompt_answer:
        findings.append(
            {
                "code": "duplicate_prompt_answer_rows",
                "severity": "warning",
                "detail": duplicate_prompt_answer,
            }
        )
    if missing_flags:
        findings.append(
            {
                "code": "missing_anti_leak_flags",
                "severity": "blocker",
                "detail": json.dumps(compact_counter(missing_flags), sort_keys=True),
            }
        )
    if true_flags:
        findings.append(
            {
                "code": "true_anti_leak_flags",
                "severity": "blocker",
                "detail": json.dumps(compact_counter(true_flags), sort_keys=True),
            }
        )

    bit_answer_only_share = (bit_answer_only / bit_rows) if bit_rows else 0.0
    if bit_rows and bit_answer_only_share > MAX_BIT_ANSWER_ONLY_SHARE_FOR_GPU:
        findings.append(
            {
                "code": "bit_answer_only_trace_not_learnable_enough",
                "severity": "blocker",
                "detail": (
                    f"bit_answer_only={bit_answer_only}/{bit_rows} "
                    f"share={bit_answer_only_share:.4f}; V511 showed loss-only transfer is insufficient"
                ),
            }
        )
    if bit_rows and bit_trace_rows < MIN_BIT_TRACE_ROWS_FOR_GPU:
        findings.append(
            {
                "code": "bit_trace_rows_below_gpu_floor",
                "severity": "blocker",
                "detail": f"bit_trace_rows={bit_trace_rows}; required>={MIN_BIT_TRACE_ROWS_FOR_GPU}",
            }
        )
    eq_p50 = quantile(equation_words, 0.50)
    if equation_words and eq_p50 < MIN_EQUATION_TRACE_WORDS_P50_FOR_GPU:
        findings.append(
            {
                "code": "equation_trace_too_short",
                "severity": "warning",
                "detail": f"equation_assistant_word_p50={eq_p50}",
            }
        )
    if tokenization_manifest:
        train_val_prompt_overlap = int(tokenization_manifest.get("validation", {}).get("train_val_prompt_overlap", 0) or 0)
        if train_val_prompt_overlap:
            findings.append(
                {"code": "tokenization_manifest_train_val_prompt_overlap", "severity": "blocker", "detail": train_val_prompt_overlap}
            )

    findings.append(
        {
            "code": "style_by_family_snapshot",
            "severity": "info",
            "detail": json.dumps(compact_counter(style_by_family, 50), sort_keys=True),
        }
    )
    return findings


def severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(item.get("severity", "")) for item in findings)
    return {key: int(counts.get(key, 0)) for key in ("blocker", "warning", "info")}


def markdown_summary(manifest: dict[str, Any], top_templates: list[dict[str, Any]]) -> str:
    decision = manifest["decision"]
    counts = manifest["finding_counts"]
    train = manifest["dataset_summary"]["train"]
    validation = manifest["dataset_summary"]["validation"]
    lines = [
        "# V513 Trace Learnability Gate",
        "",
        f"- Generated UTC: `{manifest['generated_at_utc']}`",
        f"- Decision: `{decision['status']}`",
        f"- Reason: {decision['reason']}",
        f"- Findings: blockers `{counts['blocker']}`, warnings `{counts['warning']}`, info `{counts['info']}`",
        f"- Train rows: `{train['rows']}`; validation rows: `{validation['rows']}`",
        "",
        "## Family And Style",
        "",
        "| Split | Families | Assistant styles |",
        "|---|---:|---:|",
        f"| train | `{json.dumps(train['family_counts'], sort_keys=True)}` | `{json.dumps(train['assistant_style_counts'], sort_keys=True)}` |",
        f"| validation | `{json.dumps(validation['family_counts'], sort_keys=True)}` | `{json.dumps(validation['assistant_style_counts'], sort_keys=True)}` |",
        "",
        "## Lengths",
        "",
        "| Split | Length summary |",
        "|---|---|",
        f"| train | `{json.dumps(train['assistant_length_by_family'], sort_keys=True)}` |",
        f"| validation | `{json.dumps(validation['assistant_length_by_family'], sort_keys=True)}` |",
        "",
        "## Top Template Groups",
        "",
        "| Count | Answers | Families | Subcategories | Preview |",
        "|---:|---:|---|---|---|",
    ]
    for row in top_templates[:10]:
        preview = str(row.get("template_preview", "")).replace("|", "\\|")[:160]
        lines.append(
            f"| {row['count']} | {row['answer_count']} | `{row['families']}` | `{row['subcategories']}` | {preview} |"
        )
    lines.extend(
        [
            "",
            "## Gate Meaning",
            "",
            "- `blocked_no_gpu`: no HF GPU train should be launched from this dataset as-is.",
            "- `passed_cpu_structure_only`: this still is not submit permission; it only authorizes a tiny paid smoke if other gates pass.",
            "- V513 is CPU-only and does not package or submit.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V513 TRACE LEARNABILITY GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("train_jsonl =", args.train_jsonl, flush=True)
    print("val_jsonl =", args.val_jsonl, flush=True)
    print("tokenization_manifest =", args.tokenization_manifest, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_jsonl, "train")
    val_rows = load_jsonl(args.val_jsonl, "validation")
    print("train_rows =", len(train_rows), flush=True)
    print("validation_rows =", len(val_rows), flush=True)

    projected = [row_projection(row) for row in train_rows + val_rows]
    projected_train = [row for row in projected if row["split"] == "train"]
    projected_val = [row for row in projected if row["split"] == "validation"]
    print("projected_rows =", len(projected), flush=True)

    tokenization_manifest, token_findings = tokenization_findings(args.tokenization_manifest)
    findings = build_findings(projected, tokenization_manifest, token_findings)
    counts = severity_counts(findings)
    status = "blocked_no_gpu" if counts["blocker"] else "passed_cpu_structure_only"
    reason = (
        "blockers found; do not launch paid GPU from this dataset"
        if counts["blocker"]
        else "no structural blockers; still requires objective and FinOps gates before GPU"
    )
    print("v513_status =", status, flush=True)
    print("v513_finding_counts =", json.dumps(counts, sort_keys=True), flush=True)

    template_groups = duplicate_groups(projected, "assistant_template_sha256")
    assistant_groups = duplicate_groups(projected, "assistant_sha256")
    template_csv_rows = template_groups[:200]
    assistant_csv_rows = assistant_groups[:200]

    outputs = {
        "manifest_json": str(args.output_dir / "v513_trace_learnability_gate_manifest.json"),
        "summary_md": str(args.output_dir / "V513_TRACE_LEARNABILITY_GATE_SUMMARY.md"),
        "top_template_groups_csv": str(args.output_dir / "v513_top_template_groups.csv"),
        "top_assistant_exact_groups_csv": str(args.output_dir / "v513_top_assistant_exact_groups.csv"),
        "findings_csv": str(args.output_dir / "v513_findings.csv"),
    }
    manifest: dict[str, Any] = {
        "schema_version": "kg1_v513_trace_learnability_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "train_jsonl": str(args.train_jsonl),
            "train_sha256": sha256_path(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl),
            "val_sha256": sha256_path(args.val_jsonl),
            "tokenization_manifest": str(args.tokenization_manifest),
            "tokenization_manifest_sha256": sha256_path(args.tokenization_manifest) if args.tokenization_manifest.exists() else "",
        },
        "thresholds": {
            "max_bit_answer_only_share_for_gpu": MAX_BIT_ANSWER_ONLY_SHARE_FOR_GPU,
            "min_bit_trace_rows_for_gpu": MIN_BIT_TRACE_ROWS_FOR_GPU,
            "max_assistant_words_warning": MAX_ASSISTANT_WORDS_WARN,
            "max_loss_tokens_warning": MAX_LOSS_TOKENS_WARN,
            "max_prompt_truncation_rate": MAX_PROMPT_TRUNCATION_RATE,
        },
        "dataset_summary": {
            "train": summarize_rows(projected_train),
            "validation": summarize_rows(projected_val),
            "combined": summarize_rows(projected),
        },
        "tokenization_gate_status": tokenization_manifest.get("decision", {}).get("status", "") if tokenization_manifest else "",
        "finding_counts": counts,
        "findings": findings[:200],
        "duplicate_group_counts": {
            "assistant_template_groups": len(template_groups),
            "assistant_exact_groups": len(assistant_groups),
        },
        "outputs": outputs,
        "decision": {
            "status": status,
            "reason": reason,
            "hf_gpu_allowed": counts["blocker"] == 0,
            "full_eval_allowed": False,
            "package_allowed": False,
            "kaggle_submit_allowed": False,
            "next_action": (
                "Replace answer-only bit replay with deterministic bit-pair/bitsum/stride traces and rerun V513."
                if status == "blocked_no_gpu"
                else "Run only a tiny paid smoke after objective alignment and FinOps gates."
            ),
        },
    }

    write_csv(
        Path(outputs["top_template_groups_csv"]),
        template_csv_rows,
        ["key", "count", "answer_count", "family_count", "subcategory_count", "answers_preview", "families", "subcategories", "ids_preview", "template_preview"],
    )
    write_csv(
        Path(outputs["top_assistant_exact_groups_csv"]),
        assistant_csv_rows,
        ["key", "count", "answer_count", "family_count", "subcategory_count", "answers_preview", "families", "subcategories", "ids_preview", "template_preview"],
    )
    write_csv(Path(outputs["findings_csv"]), findings, ["severity", "code", "detail"])
    write_json(Path(outputs["manifest_json"]), manifest)
    Path(outputs["summary_md"]).write_text(markdown_summary(manifest, template_groups), encoding="utf-8")
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("summary_md =", outputs["summary_md"], flush=True)
    print("=== V513 TRACE LEARNABILITY GATE END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--tokenization-manifest", type=Path, default=DEFAULT_TOKENIZATION_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
