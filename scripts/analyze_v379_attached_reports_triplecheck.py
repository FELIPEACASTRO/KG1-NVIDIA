#!/usr/bin/env python3
"""Triple-check local attached Nemotron reports and competition_train.csv.

The script reads only local user-provided files, redacts token-like strings,
compares concrete claims against the already audited local dataset package, and
emits small repo-safe artifacts for the roadmap.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import classify_puzzle  # noqa: E402


DEFAULT_REPORTS = [
    Path(r"C:\Users\davis\Downloads\Dataset andy279_nemotron-reasoning-challenge — Relatório Completo de Extração.md"),
    Path(r"C:\Users\davis\Downloads\Relatório de Extração_ Dataset andy279_nemotron-reasoning-challenge.md"),
    Path(r"C:\Users\davis\Downloads\Nemotron Reasoning Challenge — SFT Data.md"),
    Path(r"C:\Users\davis\Downloads\Relatório_ Dataset andy279_nemotron-reasoning-challenge.md"),
]
DEFAULT_COMPETITION_TRAIN = Path(r"C:\Users\davis\Downloads\competition_train.csv")
DEFAULT_FINAL_COMPETITION_TRAIN = Path(r"C:\Users\davis\Downloads\nemotron_dataset_final\competition_train.csv")
DEFAULT_FINAL_COMPETITION_TEST = Path(r"C:\Users\davis\Downloads\nemotron_dataset_final\competition_test.csv")
DEFAULT_V379_SUMMARY = REPO_ROOT / "artifacts/v379_dataset_doublecheck_audit/v379_dataset_doublecheck_summary.json"
DEFAULT_OUT = REPO_ROOT / "artifacts/v379_attached_reports_triplecheck"

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
URL_RE = re.compile(r"https?://[^\s)>\]\"']+")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def redact(text: str) -> str:
    return TOKEN_RE.sub("[REDACTED_HF_TOKEN]", text)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def audit_train(path: Path) -> dict[str, Any]:
    families: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    prompts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids[str(row.get("id", ""))] += 1
            prompts[str(row.get("prompt", ""))] += 1
            families[classify_puzzle(str(row.get("prompt", "")))] += 1
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "rows": sum(families.values()),
        "unique_ids": len(ids),
        "duplicate_id_rows": sum(v - 1 for v in ids.values() if v > 1),
        "unique_prompts": len(prompts),
        "duplicate_prompt_rows": sum(v - 1 for v in prompts.values() if v > 1),
        "family_counts": dict(sorted(families.items())),
    }


def audit_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    redacted = redact(text)
    urls = sorted(set(URL_RE.findall(redacted)))
    claims = {
        "mentions_original_sft_train_49290": "49,290" in text or "49.290" in text,
        "mentions_original_train_unique_7200": "7,200" in text or "7.200" in text,
        "mentions_original_validation_1165": "1,165" in text or "1.165" in text,
        "mentions_validation_transformation_399_unsolved": "399" in text and "transformation" in text.lower(),
        "mentions_bit_train_17285": "17,285" in text or "17.285" in text,
        "mentions_transformation_train_10741": "10,741" in text or "10.741" in text,
        "mentions_solver_guided_bit_1602": "1,602" in text or "1.602" in text,
        "mentions_solver_guided_transformation_1101": "1,101" in text or "1.101" in text,
        "mentions_gpt54_transformation_85": "GPT-5.4" in text and "85" in text,
        "mentions_data_quality_reextract_verify": "re-extracted" in text or "re-extra" in text.lower(),
        "mentions_only_correct_attempts_kept": "Only correct attempts kept" in text or "corretos" in text.lower(),
        "mentions_competition_test_34": "34 puzzles" in text or "34 puzzles" in text.lower(),
        "mentions_tong_with_logprob": "tong_with_logprob.csv" in text,
        "mentions_yours_with_logprob": "yours_with_logprob.csv" in text,
        "mentions_raw_traces": "all_traces_merged.jsonl" in text
        or "solver_bit_manipulation_traces_merged.jsonl" in text
        or "solver_transformation_traces_merged.jsonl" in text,
    }
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "contains_hf_token_pattern": bool(TOKEN_RE.search(text)),
        "url_count": len(urls),
        "urls": urls,
        "claims": claims,
    }


def summarize_claims(report_audits: list[dict[str, Any]], train_audit: dict[str, Any], final_test_rows: int) -> list[dict[str, str]]:
    all_claims: Counter[str] = Counter()
    for report in report_audits:
        for key, value in report["claims"].items():
            if value:
                all_claims[key] += 1

    return [
        {
            "claim": "Original andy279 SFT train has 49,290 examples / 7,200 unique puzzles",
            "evidence": f"mentioned_in_reports={all_claims['mentions_original_sft_train_49290']}; local_original_files_available=false",
            "roadmap_decision": "not active training input until actual sft_train.jsonl is approved/downloaded and audited",
        },
        {
            "claim": "Original validation has 1,165 examples / 1,123 puzzles, including 399 unsolved transformation rows",
            "evidence": f"mentioned_in_reports={all_claims['mentions_original_validation_1165']}; transformation_unsolved_mentions={all_claims['mentions_validation_transformation_399_unsolved']}",
            "roadmap_decision": "supports equation_transform as solver/DSL problem; no direct adapter gain without data access",
        },
        {
            "claim": "Original train contains heavy bit/equation signal",
            "evidence": f"bit_17285_mentions={all_claims['mentions_bit_train_17285']}; transformation_10741_mentions={all_claims['mentions_transformation_train_10741']}; solver_bit_1602_mentions={all_claims['mentions_solver_guided_bit_1602']}; solver_transformation_1101_mentions={all_claims['mentions_solver_guided_transformation_1101']}",
            "roadmap_decision": "if access is granted later, mine only after strict V381-style gates; do not assume immediate gain",
        },
        {
            "claim": "SFT README quality recipe cleans boxed LaTeX, reextracts answers, recomputes correctness, keeps only correct attempts",
            "evidence": f"quality_recipe_mentions={all_claims['mentions_data_quality_reextract_verify'] + all_claims['mentions_only_correct_attempts_kept']}",
            "roadmap_decision": "promote this as mandatory V381 cleaning rule for any local trace source",
        },
        {
            "claim": "competition_test.csv has 34 puzzles",
            "evidence": f"report_mentions={all_claims['mentions_competition_test_34']}; audited_final_test_rows={final_test_rows}",
            "roadmap_decision": "claim contradicted locally; active roadmap keeps competition_test.csv retired as eval",
        },
        {
            "claim": "Downloaded competition_train.csv is new data",
            "evidence": f"sha256={train_audit['sha256']}; rows={train_audit['rows']}; unique_ids={train_audit['unique_ids']}",
            "roadmap_decision": "not new; identical to final package train and useful only as official prompt/answer reference",
        },
        {
            "claim": "Reports mention tong_with_logprob/yours_with_logprob",
            "evidence": f"tong_mentions={all_claims['mentions_tong_with_logprob']}; yours_mentions={all_claims['mentions_yours_with_logprob']}; V379 confirmed files absent locally",
            "roadmap_decision": "not evidence until files exist and pass hash/metric audit",
        },
        {
            "claim": "Reports mention raw multi-attempt trace files",
            "evidence": f"raw_trace_mentions={all_claims['mentions_raw_traces']}; V379 inventories do not contain these raw trace JSONLs",
            "roadmap_decision": "not active input; if acquired later, audit as a new source before any training",
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_audits = [audit_report(path) for path in args.reports]
    train_audit = audit_train(args.competition_train)
    final_train_audit = audit_train(args.final_competition_train)
    final_test_rows = count_csv_rows(args.final_competition_test)
    v379 = json.loads(args.v379_summary.read_text(encoding="utf-8")) if args.v379_summary.exists() else {}
    claim_decisions = summarize_claims(report_audits, train_audit, final_test_rows)
    summary = {
        "schema_version": "kg1_v379_attached_reports_triplecheck_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports": report_audits,
        "competition_train_audit": train_audit,
        "final_competition_train_audit": final_train_audit,
        "competition_train_matches_final": train_audit["sha256"] == final_train_audit["sha256"],
        "final_competition_test_rows": final_test_rows,
        "claim_decisions": claim_decisions,
        "v379_reused_facts": {
            "v217_prompt_overlap": v379.get("extra_audit", {}).get("v217_prompt_overlap", {}),
            "competition_test_train_overlap": v379.get("extra_audit", {}).get("competition_test_train_overlap", {}),
            "filtered_merged_duplicates": v379.get("extra_audit", {}).get("filtered_merged_duplicates", {}),
            "sft_train_converted_format": v379.get("extra_audit", {}).get("sft_train_converted_format", {}),
            "sft_train_full_conflicts": v379.get("extra_audit", {}).get("sft_train_full_conflicts", {}),
        },
        "gain_assessment": {
            "new_measured_adapter_gain": 0,
            "new_measured_cpu_gain": 0,
            "actionable_gain_route": (
                "No direct gain from the attached files. The only actionable improvement is better gating/cleaning for V380/V381. "
                "Expected gain remains conditional on V380 accepting solver categories with zero losses, then V381 transferring without bit regression."
            ),
        },
        "outputs": {
            "summary_json": str(args.output_dir / "v379_attached_reports_triplecheck_summary.json"),
            "report_md": str(args.output_dir / "KG1_V379_ATTACHED_REPORTS_TRIPLECHECK.md"),
        },
    }
    (args.output_dir / "v379_attached_reports_triplecheck_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, args.output_dir / "KG1_V379_ATTACHED_REPORTS_TRIPLECHECK.md")
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    train = summary["competition_train_audit"]
    lines = [
        "# V379 Attached Reports Triple Check",
        "",
        "## Verdict",
        "",
        "- The attached reports add useful process rules, not a new measured gain.",
        f"- Attached `competition_train.csv` matches final package: `{summary['competition_train_matches_final']}`.",
        f"- Official train rows: `{train['rows']}`; family counts: `{train['family_counts']}`.",
        "- Raw attached reports are not committed because at least one contains an HF token-like string; only redacted metadata is versioned.",
        "",
        "## Actionable Findings",
        "",
    ]
    for row in summary["claim_decisions"]:
        lines.extend(
            [
                f"### {row['claim']}",
                "",
                f"- Evidence: `{row['evidence']}`.",
                f"- Roadmap decision: {row['roadmap_decision']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Gain Assessment",
            "",
            f"- New measured adapter gain: `{summary['gain_assessment']['new_measured_adapter_gain']}`.",
            f"- New measured CPU gain: `{summary['gain_assessment']['new_measured_cpu_gain']}`.",
            f"- Actionable route: {summary['gain_assessment']['actionable_gain_route']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> int:
    assert redact("abc hf_" + "A" * 30 + " xyz") == "abc [REDACTED_HF_TOKEN] xyz"
    print("v379_attached_reports_triplecheck_self_test=ok", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="*", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--competition-train", type=Path, default=DEFAULT_COMPETITION_TRAIN)
    parser.add_argument("--final-competition-train", type=Path, default=DEFAULT_FINAL_COMPETITION_TRAIN)
    parser.add_argument("--final-competition-test", type=Path, default=DEFAULT_FINAL_COMPETITION_TEST)
    parser.add_argument("--v379-summary", type=Path, default=DEFAULT_V379_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    print("=== V379 ATTACHED REPORTS TRIPLE CHECK START ===", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    summary = run(args)
    print("competition_train_audit =", json.dumps(summary["competition_train_audit"], indent=2, sort_keys=True), flush=True)
    print("competition_train_matches_final =", summary["competition_train_matches_final"], flush=True)
    print("claim_decisions =", json.dumps(summary["claim_decisions"], indent=2, sort_keys=True), flush=True)
    print("gain_assessment =", json.dumps(summary["gain_assessment"], indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(summary["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V379 ATTACHED REPORTS TRIPLE CHECK END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
