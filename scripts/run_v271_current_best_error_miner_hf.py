#!/usr/bin/env python3
"""Mine current-best weak errors from HF artifacts without spending GPU.

V271 is a CPU-only diagnostic gate. It downloads the current best weak
prediction artifacts, validates the shared V221 row contract, compares the
V269 reasoning smoke against V259 checkpoint-4, and emits a concrete next-step
manifest before any additional H100/H200 spend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_HF_OUTPUT_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_OUTPUT_PREFIX = "runtime_artifacts/v271_current_best_error_miner"

DEFAULT_PREDICTION_SPECS = [
    {
        "name": "v259_checkpoint4_current_best",
        "repo_id": "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke",
        "repo_type": "model",
        "filename": (
            "evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/"
            "v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv"
        ),
        "role": "baseline",
    },
    {
        "name": "v269_checkpoint2_reasoning_smoke",
        "repo_id": "felipesp1983/kg1-nemotron-lora-v269-v268-reasoning-v259ckpt4-smoke",
        "repo_type": "model",
        "filename": (
            "evals/v270-h200-v221contract-v269-v268-reasoning-eval-20260511T0925Z/eval/"
            "v269_checkpoint_2_v221_contract/v270_hf_weak_v269_checkpoint_2_v221_contract_predictions.csv"
        ),
        "role": "candidate",
    },
    {
        "name": "v269_final_reasoning_smoke",
        "repo_id": "felipesp1983/kg1-nemotron-lora-v269-v268-reasoning-v259ckpt4-smoke",
        "repo_type": "model",
        "filename": (
            "evals/v270-h200-v221contract-v269-v268-reasoning-eval-20260511T0925Z/eval/"
            "v269_final_v221_contract/v270_hf_weak_v269_final_v221_contract_predictions.csv"
        ),
        "role": "candidate",
    },
]


FAMILY_COLUMNS = [
    "candidate",
    "rows",
    "correct",
    "accuracy",
    "equation_transform_rows",
    "equation_transform_correct",
    "bit_manipulation_rows",
    "bit_manipulation_correct",
    "truncated",
    "weak_total_gap",
    "weak_eq_gap",
    "weak_bit_gap",
    "bit_guardrail_gap",
]

EQUATION_MISS_COLUMNS = [
    "id",
    "subtype",
    "answer_kind",
    "answer",
    "baseline_prediction",
    "baseline_correct",
    "v269_checkpoint2_prediction",
    "v269_checkpoint2_correct",
    "v269_final_prediction",
    "v269_final_correct",
    "any_candidate_gain",
    "query",
    "example_count",
    "operator_set",
    "prompt_preview",
]

DIFF_COLUMNS = [
    "id",
    "family",
    "answer",
    "baseline_prediction",
    "candidate_prediction",
    "baseline_correct",
    "candidate_correct",
    "gain",
    "loss",
    "changed_prediction",
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


def download_file(repo_id: str, repo_type: str, filename: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V271") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename.strip("/"),
            local_dir=str(local_dir),
            token=token,
        )
    )


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V271 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V271 current-best error miner {path_in_repo.strip('/')}",
    )
    return str(info)


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", "")).strip()
    prediction = str(row.get("prediction", "")).strip()
    family = canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt))
    correct = verify_answer(answer, prediction)
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": answer,
        "prediction": prediction,
        "family": family,
        "prompt_sha256": sha256_text(prompt),
        "correct_bool": bool(correct),
        "truncated_bool": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
    }


def load_predictions(spec: dict[str, str], path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_csv(path)
    rows = [normalize_row(row) for row in raw_rows]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{spec['name']} has duplicate ids")
    meta = {
        "candidate": spec["name"],
        "repo_id": spec["repo_id"],
        "repo_type": spec["repo_type"],
        "filename": spec["filename"],
        "local_path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "family_counts": dict(Counter(row["family"] for row in rows)),
    }
    print("loaded_prediction_artifact =", json.dumps(meta, sort_keys=True), flush=True)
    return rows, meta


def shared_row_contract(rows_by_candidate: dict[str, list[dict[str, Any]]]) -> str:
    common_ids: set[str] | None = None
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in rows_by_candidate.items():
        lookup = {str(row["id"]): row for row in rows}
        by_candidate[name] = lookup
        ids = set(lookup)
        common_ids = ids if common_ids is None else common_ids & ids
    if common_ids is None:
        raise RuntimeError("no candidates loaded")
    if len(common_ids) != 315:
        raise RuntimeError(f"expected 315 shared rows, got {len(common_ids)}")
    reference_name = sorted(by_candidate)[0]
    reference = by_candidate[reference_name]
    for name, lookup in by_candidate.items():
        mismatches = []
        for row_id in sorted(common_ids):
            ref = reference[row_id]
            cur = lookup[row_id]
            ref_tuple = (ref["family"], ref["answer"], ref["prompt_sha256"])
            cur_tuple = (cur["family"], cur["answer"], cur["prompt_sha256"])
            if ref_tuple != cur_tuple:
                mismatches.append({"id": row_id, "candidate": name, "reference": ref_tuple, "observed": cur_tuple})
        if mismatches:
            raise RuntimeError(f"shared row contract mismatch for {name}: {mismatches[:3]}")
    payload = "\n".join(
        f"{row_id}\t{reference[row_id]['family']}\t{reference[row_id]['answer']}\t{reference[row_id]['prompt_sha256']}"
        for row_id in sorted(common_ids)
    )
    return sha256_text(payload)


def answer_kind(answer: str) -> str:
    text = answer.strip()
    if re.fullmatch(r"[01]+", text):
        return "binary"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return "numeric"
    if re.fullmatch(r"[A-Za-z]+", text):
        return "alpha"
    if re.fullmatch(r"[^\w\s]+", text):
        return "symbolic"
    return "mixed"


def extract_query(prompt: str) -> str:
    matches = re.findall(r"(?:Now,\s*)?determine the result for:\s*`?([^\n`]+)`?", prompt, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    tail = [line.strip() for line in prompt.splitlines() if line.strip()]
    return tail[-1] if tail else ""


def equation_examples(prompt: str) -> list[str]:
    examples: list[str] = []
    for line in prompt.splitlines():
        text = line.strip().strip("`")
        if not text or "determine the result" in text.lower():
            continue
        if "=" in text or re.search(r"\s[-+*/#@$%^&|<>:;~`'\"\\(){}\[\]]\s*", text):
            if any(ch.isdigit() or ch.isalpha() for ch in text):
                examples.append(text)
    return examples


def classify_equation_subtype(prompt: str, answer: str) -> dict[str, Any]:
    query = extract_query(prompt)
    examples = equation_examples(prompt)
    operator_set = "".join(sorted(set(re.findall(r"[^A-Za-z0-9\s]", "\n".join(examples + [query])))))
    has_digit = bool(re.search(r"\d", query))
    has_alpha = bool(re.search(r"[A-Za-z]", query))
    has_symbol = bool(re.search(r"[^A-Za-z0-9\s]", query))
    if has_digit and has_symbol:
        subtype = "equation_numeric_operator"
    elif has_digit:
        subtype = "equation_numeric_plain"
    elif has_alpha and has_symbol:
        subtype = "equation_symbolic_mixed"
    elif has_alpha:
        subtype = "equation_symbolic_alpha"
    elif has_symbol:
        subtype = "equation_symbolic_punct"
    else:
        subtype = "equation_unknown"
    return {
        "subtype": subtype,
        "answer_kind": answer_kind(answer),
        "query": query,
        "example_count": len(examples),
        "operator_set": operator_set,
    }


def summarize_candidate(
    name: str,
    rows: list[dict[str, Any]],
    *,
    weak_total_min: int,
    weak_eq_min: int,
    weak_bit_min: int,
    bit_guardrail_min: int,
) -> dict[str, Any]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    total_correct = sum(1 for row in rows if row["correct_bool"])
    eq_correct = sum(1 for row in by_family["equation_transform"] if row["correct_bool"])
    bit_correct = sum(1 for row in by_family["bit_manipulation"] if row["correct_bool"])
    truncated = sum(1 for row in rows if row["truncated_bool"])
    return {
        "candidate": name,
        "rows": len(rows),
        "correct": total_correct,
        "accuracy": round(total_correct / len(rows), 9) if rows else 0.0,
        "equation_transform_rows": len(by_family["equation_transform"]),
        "equation_transform_correct": eq_correct,
        "bit_manipulation_rows": len(by_family["bit_manipulation"]),
        "bit_manipulation_correct": bit_correct,
        "truncated": truncated,
        "weak_total_gap": max(0, weak_total_min - total_correct),
        "weak_eq_gap": max(0, weak_eq_min - eq_correct),
        "weak_bit_gap": max(0, weak_bit_min - bit_correct),
        "bit_guardrail_gap": max(0, bit_guardrail_min - bit_correct),
    }


def compare_rows(
    baseline_name: str,
    candidate_name: str,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {row["id"]: row for row in baseline_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    diff_rows: list[dict[str, Any]] = []
    for row_id in sorted(set(baseline) & set(candidate)):
        base = baseline[row_id]
        cand = candidate[row_id]
        changed = base["prediction"] != cand["prediction"]
        gain = (not base["correct_bool"]) and cand["correct_bool"]
        loss = base["correct_bool"] and (not cand["correct_bool"])
        if changed or gain or loss:
            diff_rows.append(
                {
                    "id": row_id,
                    "family": base["family"],
                    "answer": base["answer"],
                    "baseline_prediction": base["prediction"],
                    "candidate_prediction": cand["prediction"],
                    "baseline_correct": base["correct_bool"],
                    "candidate_correct": cand["correct_bool"],
                    "gain": gain,
                    "loss": loss,
                    "changed_prediction": changed,
                }
            )
    print(
        "candidate_diff =",
        json.dumps(
            {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "changed_or_scored_rows": len(diff_rows),
                "gains": sum(1 for row in diff_rows if row["gain"]),
                "losses": sum(1 for row in diff_rows if row["loss"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return diff_rows


def equation_miss_rows(rows_by_candidate: dict[str, list[dict[str, Any]]], baseline_name: str) -> list[dict[str, Any]]:
    lookups = {name: {row["id"]: row for row in rows} for name, rows in rows_by_candidate.items()}
    baseline = lookups[baseline_name]
    candidate_names = [name for name in rows_by_candidate if name != baseline_name]
    output: list[dict[str, Any]] = []
    for row_id, base in sorted(baseline.items()):
        if base["family"] != "equation_transform" or base["correct_bool"]:
            continue
        extra = classify_equation_subtype(base["prompt"], base["answer"])
        item = {
            "id": row_id,
            **extra,
            "answer": base["answer"],
            "baseline_prediction": base["prediction"],
            "baseline_correct": base["correct_bool"],
            "prompt_preview": " ".join(base["prompt"].split())[:500],
        }
        any_gain = False
        for candidate_name in candidate_names:
            short = candidate_name.replace("v269_checkpoint2_reasoning_smoke", "v269_checkpoint2")
            short = short.replace("v269_final_reasoning_smoke", "v269_final")
            cand = lookups[candidate_name][row_id]
            item[f"{short}_prediction"] = cand["prediction"]
            item[f"{short}_correct"] = cand["correct_bool"]
            any_gain = any_gain or bool(cand["correct_bool"])
        item["any_candidate_gain"] = any_gain
        output.append(item)
    return output


def counter_to_rows(counter: Counter[tuple[str, ...]], columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, count in counter.most_common():
        row = {column: value for column, value in zip(columns, key)}
        row["rows"] = count
        rows.append(row)
    return rows


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V271 CURRENT BEST ERROR MINER START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("hf_output_repo =", args.hf_output_repo, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("weak_gate =", json.dumps({"total": args.weak_total_min, "equation": args.weak_eq_min, "bit": args.weak_bit_min}, sort_keys=True), flush=True)
    print("bit_guardrail_min =", args.bit_guardrail_min, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v271_") as temp_name:
        temp_dir = Path(temp_name)
        rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
        input_meta: list[dict[str, Any]] = []
        for spec in DEFAULT_PREDICTION_SPECS:
            path = download_file(spec["repo_id"], spec["repo_type"], spec["filename"], temp_dir, token)
            rows, meta = load_predictions(spec, path)
            rows_by_candidate[spec["name"]] = rows
            input_meta.append(meta)

    observed_contract = shared_row_contract(rows_by_candidate)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    baseline_name = "v259_checkpoint4_current_best"
    family_rows = [
        summarize_candidate(
            name,
            rows,
            weak_total_min=args.weak_total_min,
            weak_eq_min=args.weak_eq_min,
            weak_bit_min=args.weak_bit_min,
            bit_guardrail_min=args.bit_guardrail_min,
        )
        for name, rows in rows_by_candidate.items()
    ]
    family_rows.sort(key=lambda row: (int(row["correct"]), int(row["equation_transform_correct"]), int(row["bit_manipulation_correct"])), reverse=True)

    diff_paths: dict[str, str] = {}
    diff_summaries: dict[str, Any] = {}
    for candidate_name in rows_by_candidate:
        if candidate_name == baseline_name:
            continue
        diff = compare_rows(baseline_name, candidate_name, rows_by_candidate[baseline_name], rows_by_candidate[candidate_name])
        path = args.output_dir / f"v271_{candidate_name}_vs_current_best_diff.csv"
        write_csv(path, diff, DIFF_COLUMNS)
        diff_paths[candidate_name] = str(path)
        by_family: dict[str, Any] = {}
        for family in sorted({row["family"] for row in diff}):
            rows = [row for row in diff if row["family"] == family]
            by_family[family] = {
                "changed_predictions": sum(1 for row in rows if row["changed_prediction"]),
                "gains": sum(1 for row in rows if row["gain"]),
                "losses": sum(1 for row in rows if row["loss"]),
                "net": sum(1 for row in rows if row["gain"]) - sum(1 for row in rows if row["loss"]),
            }
        diff_summaries[candidate_name] = by_family

    eq_misses = equation_miss_rows(rows_by_candidate, baseline_name)
    subtype_counter = Counter((row["subtype"], row["answer_kind"]) for row in eq_misses)
    operator_counter = Counter((row["operator_set"] or "none", row["subtype"]) for row in eq_misses)

    output_paths = {
        "family_score_summary_csv": args.output_dir / "v271_family_score_summary.csv",
        "equation_current_best_misses_csv": args.output_dir / "v271_equation_current_best_misses.csv",
        "equation_miss_subtype_summary_csv": args.output_dir / "v271_equation_miss_subtype_summary.csv",
        "equation_operator_summary_csv": args.output_dir / "v271_equation_operator_summary.csv",
        "manifest_json": args.output_dir / "v271_current_best_error_miner_manifest.json",
    }
    write_csv(output_paths["family_score_summary_csv"], family_rows, FAMILY_COLUMNS)
    write_csv(output_paths["equation_current_best_misses_csv"], eq_misses, EQUATION_MISS_COLUMNS)
    write_csv(output_paths["equation_miss_subtype_summary_csv"], counter_to_rows(subtype_counter, ["subtype", "answer_kind"]), ["subtype", "answer_kind", "rows"])
    write_csv(output_paths["equation_operator_summary_csv"], counter_to_rows(operator_counter, ["operator_set", "subtype"]), ["operator_set", "subtype", "rows"])

    baseline_summary = next(row for row in family_rows if row["candidate"] == baseline_name)
    checkpoint2_summary = next(row for row in family_rows if row["candidate"] == "v269_checkpoint2_reasoning_smoke")
    equation_net = diff_summaries["v269_checkpoint2_reasoning_smoke"].get("equation_transform", {}).get("net", 0)
    bit_net = diff_summaries["v269_checkpoint2_reasoning_smoke"].get("bit_manipulation", {}).get("net", 0)
    if int(checkpoint2_summary["correct"]) > int(baseline_summary["correct"]):
        decision = "candidate_improves_current_best"
        next_action = "Run focused H200 confirmation only if equation and bit guardrails are both met."
    elif equation_net > 0:
        decision = "equation_signal_without_total_gain"
        next_action = "Mine changed equation rows for verified override rules before another train."
    else:
        decision = "no_new_gpu_training_signal"
        next_action = "Do CPU-only equation solver/verifier mining or unlock gated solver traces before spending H200 again."

    manifest = {
        "schema_version": "kg1_v271_current_best_error_miner_v1",
        "generated_at_utc": utc_now(),
        "inputs": input_meta,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed_contract,
        "baseline_candidate": baseline_name,
        "thresholds": {
            "weak_total_min": args.weak_total_min,
            "weak_eq_min": args.weak_eq_min,
            "weak_bit_min": args.weak_bit_min,
            "bit_guardrail_min": args.bit_guardrail_min,
        },
        "family_score_summary": family_rows,
        "diff_summaries": diff_summaries,
        "equation_miss_count": len(eq_misses),
        "equation_miss_subtype_summary": counter_to_rows(subtype_counter, ["subtype", "answer_kind"]),
        "equation_operator_summary": counter_to_rows(operator_counter, ["operator_set", "subtype"])[:25],
        "external_evidence_used": [
            {
                "source": "https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces",
                "finding": "Gated raw traces include solver-guided transformation and bit-manipulation files; access is required before use.",
            },
            {
                "source": "https://github.com/tonghuikang/nemotron",
                "finding": "Public progress-prize repo exposes problem, corpus, reasoning, training, and metrics artifacts; V268 ingestion already tested a leak-filtered subset.",
            },
            {
                "source": "https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/",
                "finding": "NVIDIA points to open training datasets, data processing pipelines, tokenizer configuration, long-context setup, and RL recipes as supported routes.",
            },
        ],
        "decision": {
            "decision": decision,
            "reason": (
                f"baseline={baseline_summary['correct']}/315 eq={baseline_summary['equation_transform_correct']} "
                f"bit={baseline_summary['bit_manipulation_correct']}; "
                f"v269_checkpoint2={checkpoint2_summary['correct']}/315 eq={checkpoint2_summary['equation_transform_correct']} "
                f"bit={checkpoint2_summary['bit_manipulation_correct']}; equation_net={equation_net}; bit_net={bit_net}"
            ),
            "next_action": next_action,
        },
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }
    write_json(output_paths["manifest_json"], manifest)
    print("family_score_summary =", json.dumps(family_rows, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)

    upload_commit = ""
    if args.upload:
        path_in_repo = f"{args.output_prefix.rstrip('/')}/{args.run_id}"
        upload_commit = upload_outputs(args.hf_output_repo, args.output_dir, path_in_repo, token)
        manifest["upload"] = {"repo_id": args.hf_output_repo, "path_in_repo": path_in_repo, "commit": upload_commit}
        write_json(output_paths["manifest_json"], manifest)
        print("hf_upload_commit =", upload_commit, flush=True)

    print("manifest_json =", output_paths["manifest_json"], flush=True)
    print("=== V271 CURRENT BEST ERROR MINER END ===", flush=True)
    return manifest


def run_self_test() -> None:
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations.\n"
        "72)27 = 99\n26#48 = 22\nNow, determine the result for: 94)40"
    )
    subtype = classify_equation_subtype(prompt, "54")
    if subtype["subtype"] != "equation_numeric_operator":
        raise AssertionError(subtype)
    if not verify_answer("30", "30"):
        raise AssertionError("verify_answer failed")
    print("v271_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v271_current_best_error_miner"))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--run-id", default=f"v271-hf-cpu-current-best-error-miner-{utc_compact()}")
    parser.add_argument("--hf-output-repo", default=DEFAULT_HF_OUTPUT_REPO)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=133)
    parser.add_argument("--bit-guardrail-min", type=int, default=136)
    parser.add_argument("--upload", action="store_true")
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
