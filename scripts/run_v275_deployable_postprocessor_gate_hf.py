#!/usr/bin/env python3
"""V275 gate for the V274 deployable numeric postprocessor.

This script separates the deployable path from the scoring path. The
postprocessor module receives only prompt/prediction/family/truncation fields;
this gate reads labels only after postprocessing to verify weak metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402
from kg1_v274_numeric_postprocessor import postprocess_rows, self_test as postprocessor_self_test  # noqa: E402


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_BASELINE_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_BASELINE_FILENAME = (
    "evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/"
    "v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv"
)


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
        raise RuntimeError("huggingface_hub is required for V275") from exc
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
    prediction = str(row.get("prediction", "")).strip()
    family = canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt))
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": str(row.get("answer", "")).strip(),
        "prediction": prediction,
        "family": family,
        "prompt_sha256": sha256_text(prompt),
        "truncated": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
    }


def row_contract(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 315:
        raise RuntimeError(f"expected 315 rows, got {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate ids in baseline predictions")
    payload = "\n".join(
        f"{row['id']}\t{row['family']}\t{row['answer']}\t{row['prompt_sha256']}"
        for row in sorted(rows, key=lambda item: item["id"])
    )
    return sha256_text(payload)


def source_guard() -> dict[str, Any]:
    module_path = SRC_ROOT / "kg1_v274_numeric_postprocessor.py"
    text = module_path.read_text(encoding="utf-8")
    lower = text.lower()
    forbidden = ["answer", "correct", "verify_answer", "solution"]
    hits = [token for token in forbidden if token in lower]
    if hits:
        raise RuntimeError(f"postprocessor module contains forbidden scoring terms: {hits}")
    return {
        "module_path": str(module_path),
        "sha256": sha256_file(module_path),
        "forbidden_terms": forbidden,
        "forbidden_hits": hits,
    }


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    family_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "correct": 0, "truncated": 0})
    total = 0
    correct = 0
    truncated = 0
    for row in rows:
        family = str(row["family"])
        total += 1
        family_rows[family]["rows"] += 1
        if truthy(row.get("truncated", False)):
            truncated += 1
            family_rows[family]["truncated"] += 1
        if verify_answer(row["answer"], row.get(prediction_key, "")):
            correct += 1
            family_rows[family]["correct"] += 1
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "truncated": truncated,
        "family": dict(sorted(family_rows.items())),
    }


def build_audit_rows(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for before, after in zip(before_rows, after_rows):
        baseline_ok = verify_answer(before["answer"], before["prediction"])
        post_ok = verify_answer(before["answer"], after["prediction"])
        out.append(
            {
                "id": before["id"],
                "family": before["family"],
                "baseline_prediction": before["prediction"],
                "postprocessed_prediction": after["prediction"],
                "postprocessor_applied": bool(after.get("postprocessor_applied", False)),
                "postprocessor_rule": after.get("postprocessor_rule", ""),
                "postprocessor_proof": after.get("postprocessor_proof", ""),
                "baseline_correct": baseline_ok,
                "postprocessed_correct": post_ok,
                "gain": (not baseline_ok) and post_ok,
                "loss": baseline_ok and (not post_ok),
                "wrong_on_baseline_miss": (not baseline_ok) and (not post_ok) and bool(after.get("postprocessor_applied", False)),
            }
        )
    return out


def summarize_rules(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit_rows:
        rule = str(row.get("postprocessor_rule", ""))
        grouped[rule]["rows"] += 1
        if bool(row.get("postprocessor_applied", False)):
            grouped[rule]["applied"] += 1
        if bool(row.get("gain", False)):
            grouped[rule]["gains"] += 1
        if bool(row.get("loss", False)):
            grouped[rule]["losses"] += 1
        if bool(row.get("wrong_on_baseline_miss", False)):
            grouped[rule]["wrong_on_baseline_miss"] += 1
    return [{"rule": rule, **dict(counts)} for rule, counts in sorted(grouped.items(), key=lambda item: (-item[1]["applied"], item[0]))]


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V275 DEPLOYABLE POSTPROCESSOR GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("expected_shared_row_contract_sha256 =", args.expected_shared_row_contract_sha256, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    postprocessor_self_test()
    guard = source_guard()
    print("postprocessor_source_guard =", json.dumps(guard, sort_keys=True), flush=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v275_") as temp_name:
        baseline_path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(baseline_path)]
        input_meta = {
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": baseline_path.name,
            "sha256": sha256_file(baseline_path),
            "rows": len(rows),
        }

    observed = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed, flush=True)
    if observed != args.expected_shared_row_contract_sha256:
        raise RuntimeError(f"row contract mismatch: expected {args.expected_shared_row_contract_sha256}, got {observed}")

    deployable_inputs = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "prediction": row["prediction"],
            "family": row["family"],
            "truncated": row["truncated"],
        }
        for row in rows
    ]
    postprocessed = postprocess_rows(deployable_inputs)
    by_id = {row["id"]: row for row in rows}
    merged_after = [{**by_id[row["id"]], **row} for row in postprocessed]

    baseline_summary = summarize(rows, "prediction")
    postprocessed_summary = summarize(merged_after, "prediction")
    audit_rows = build_audit_rows(rows, merged_after)
    rule_summary = summarize_rules(audit_rows)

    applied_rows = [row for row in audit_rows if bool(row["postprocessor_applied"])]
    gains = [row for row in audit_rows if bool(row["gain"])]
    losses = [row for row in audit_rows if bool(row["loss"])]
    wrong_on_misses = [row for row in audit_rows if bool(row["wrong_on_baseline_miss"])]
    eq_after = postprocessed_summary["family"].get("equation_transform", {})
    bit_after = postprocessed_summary["family"].get("bit_manipulation", {})
    weak_gate_pass = (
        int(postprocessed_summary["correct"]) >= args.weak_total_min
        and int(eq_after.get("correct", 0)) >= args.weak_eq_min
        and int(bit_after.get("correct", 0)) >= args.weak_bit_min
        and int(postprocessed_summary["truncated"]) <= args.weak_trunc_max
        and not losses
        and not wrong_on_misses
    )

    outputs = {
        "postprocessed_predictions_csv": args.output_dir / "v275_postprocessed_predictions.csv",
        "audit_csv": args.output_dir / "v275_postprocessor_audit.csv",
        "rule_summary_csv": args.output_dir / "v275_postprocessor_rule_summary.csv",
        "manifest_json": args.output_dir / "v275_deployable_postprocessor_gate_manifest.json",
    }
    prediction_columns = [
        "id",
        "prompt",
        "answer",
        "prediction",
        "family",
        "truncated",
        "baseline_prediction",
        "postprocessor",
        "postprocessor_applied",
        "postprocessor_rule",
        "postprocessor_proof",
    ]
    audit_columns = [
        "id",
        "family",
        "baseline_prediction",
        "postprocessed_prediction",
        "postprocessor_applied",
        "postprocessor_rule",
        "postprocessor_proof",
        "baseline_correct",
        "postprocessed_correct",
        "gain",
        "loss",
        "wrong_on_baseline_miss",
    ]
    write_csv(outputs["postprocessed_predictions_csv"], merged_after, prediction_columns)
    write_csv(outputs["audit_csv"], audit_rows, audit_columns)
    write_csv(
        outputs["rule_summary_csv"],
        rule_summary,
        ["rule", "rows", "applied", "gains", "losses", "wrong_on_baseline_miss"],
    )

    manifest = {
        "schema_version": "kg1_v275_deployable_postprocessor_gate_v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id or utc_compact(),
        "inputs": {
            "baseline": input_meta,
            "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
            "observed_shared_row_contract_sha256": observed,
        },
        "deployable_input_fields": ["id", "prompt", "prediction", "family", "truncated"],
        "postprocessor_source_guard": guard,
        "baseline_summary": baseline_summary,
        "postprocessed_summary": postprocessed_summary,
        "rule_summary": rule_summary,
        "applied_rows": len(applied_rows),
        "gains": len(gains),
        "losses": len(losses),
        "wrong_on_baseline_misses": len(wrong_on_misses),
        "weak_gate": {
            "pass": weak_gate_pass,
            "weak_total_min": args.weak_total_min,
            "weak_eq_min": args.weak_eq_min,
            "weak_bit_min": args.weak_bit_min,
            "weak_trunc_max": args.weak_trunc_max,
        },
        "decision": {
            "decision": "v275_postprocessor_ready_for_full_eval_gate" if weak_gate_pass else "v275_postprocessor_not_ready",
            "reason": (
                f"baseline={baseline_summary['correct']}; postprocessed={postprocessed_summary['correct']}; "
                f"eq={eq_after.get('correct', 0)}; bit={bit_after.get('correct', 0)}; "
                f"applied={len(applied_rows)}; gains={len(gains)}; losses={len(losses)}; "
                f"wrong_on_misses={len(wrong_on_misses)}"
            ),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("baseline_summary =", json.dumps(baseline_summary, sort_keys=True), flush=True)
    print("postprocessed_summary =", json.dumps(postprocessed_summary, sort_keys=True), flush=True)
    print("rule_summary =", json.dumps(rule_summary, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V275 DEPLOYABLE POSTPROCESSOR GATE END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf_cpu_runs/v275_deployable_postprocessor_gate"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=133)
    parser.add_argument("--weak-trunc-max", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        postprocessor_self_test()
        source_guard()
        print("v275_deployable_postprocessor_gate_self_test=ok", flush=True)
        return 0
    run_gate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
