#!/usr/bin/env python3
"""Build V435E adapter-exact-wrong preference data from V435D probes.

V435C collected V291/V290 adapter outputs without labels. V435D joined public
train labels only after collection and filtered weak/full reference overlap.
This builder converts those permitted, real adapter mistakes into short
chosen/rejected preference pairs. It does not train, submit, or use weak/full
labels as training data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V435D_DIR = REPO_ROOT / "artifacts/v435d_adapter_probe_output_analysis/20260515T141924Z_hf"
DEFAULT_V435D_MANIFEST = DEFAULT_V435D_DIR / "v435d_adapter_probe_output_analysis_manifest.json"
DEFAULT_DETAIL_CSV = DEFAULT_V435D_DIR / "v435d_adapter_probe_output_analysis_detail.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v435e_adapter_probe_preference_dataset"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def escape_boxed_answer(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def box_answer(value: object) -> str:
    return "\\boxed{" + escape_boxed_answer(value) + "}"


def infer_equation_rule_class(row: dict[str, str]) -> str:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    before_query = prompt.split("Now, determine", 1)[0]
    example_lines = "\n".join(line for line in before_query.splitlines() if "=" in line)
    has_digit = bool(re.search(r"\d", example_lines + answer))
    has_symbol = bool(re.sub(r"[A-Za-z0-9\s=]", "", example_lines + answer))
    answer_numeric = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", answer))
    if has_digit and not has_symbol:
        return "equation_numeric_arithmetic"
    if has_digit and has_symbol and answer_numeric:
        return "equation_numeric_operator_to_number"
    if has_digit and has_symbol and not answer_numeric:
        return "equation_numeric_operator_to_symbolic"
    if not has_digit and has_symbol and len(answer) <= 2:
        return "equation_symbolic_short"
    if not has_digit and has_symbol:
        return "equation_symbolic_sequence"
    return "equation_other"


def infer_rule_class(row: dict[str, str]) -> str:
    family = str(row.get("family", ""))
    if family == "bit_manipulation":
        return "bit_adapter_exact_wrong" if not truthy(row.get("correct")) else "bit_adapter_correct_replay"
    if family == "equation_transform":
        return infer_equation_rule_class(row)
    return family or "unknown"


def chosen_completion(row: dict[str, str], mode: str) -> str:
    answer = row["answer"]
    family = row["family"]
    prediction = row["prediction"]
    if mode == "hard_negative":
        if family == "bit_manipulation":
            proof = (
                "Check the output bits against every example. "
                f"The frozen adapter candidate {prediction!r} is rejected by the public-train label audit."
            )
        else:
            proof = (
                "Check the hidden transformation against every example. "
                f"The frozen adapter candidate {prediction!r} is rejected by the public-train label audit."
            )
    else:
        proof = "Replay guardrail: keep the verified final answer format unchanged."
    return "Verification:\n" + proof + "\nFinal answer: " + box_answer(answer)


def rejected_completion(row: dict[str, str], mode: str) -> str:
    if mode == "hard_negative":
        return (
            "Rejected adapter candidate:\n"
            "This is the exact final answer selected by the frozen adapter on the prompt-only probe.\n"
            "Final answer: "
            + box_answer(row["prediction"])
        )
    if mode == "format_no_box":
        return "Final answer: " + str(row["answer"])
    if mode == "format_extra_text":
        return "Final answer: " + box_answer(row["answer"]) + "\nThis answer is not final."
    raise ValueError("unknown rejected mode: " + mode)


def base_metadata(row: dict[str, str], negative_type: str) -> dict[str, Any]:
    return {
        "schema_version": "kg1_v435e_adapter_probe_preference_row_v1",
        "source": "v435d_adapter_probe_output_analysis",
        "source_id": row["id"],
        "family": row["family"],
        "rule_class": infer_rule_class(row),
        "negative_type": negative_type,
        "answer": row["answer"],
        "adapter_prediction": row["prediction"],
        "adapter_raw_output_sha256": sha256_text(row.get("raw_output", "")),
        "v291_raw_output": row.get("raw_output", ""),
        "v291_decode_config_sha256": row.get("decode_config_sha256", ""),
        "adapter_repo": row.get("adapter_repo", ""),
        "adapter_subfolder": row.get("adapter_subfolder", ""),
        "adapter_identity": f"{row.get('adapter_repo', '')}:{row.get('adapter_subfolder', '')}",
        "prompt_sha256": row.get("prompt_sha256", ""),
        "prompt_normalized_sha256": row.get("prompt_normalized_sha256", ""),
        "raw_output_collected_without_labels": True,
        "labels_joined_after_collection_from_public_train": True,
        "locked_before_answer_audit": True,
        "adapter_exact_wrong_certificate": negative_type.startswith("hard_negative"),
        "gate_rows_used_for_training": False,
        "weak_gate_rows_used_for_training": False,
        "full_gate_rows_used_for_training": False,
    }


def make_pair(row: dict[str, str], mode: str) -> dict[str, Any]:
    negative_type = (
        "hard_negative_adapter_exact_wrong"
        if mode == "hard_negative"
        else f"format_negative_{mode}"
    )
    metadata = base_metadata(row, negative_type)
    chosen = chosen_completion(row, "hard_negative" if mode == "hard_negative" else "replay")
    return {
        "id": f"v435e_{mode}_{row['id']}",
        "family": row["family"],
        "subcategory": metadata["rule_class"],
        "source": "v435e_adapter_probe_preference_dataset",
        "prompt": row["prompt"],
        "chosen": chosen,
        "rejected": rejected_completion(row, mode),
        "messages": [
            {
                "role": "system",
                "content": "Solve the KG1 puzzle. End with exactly one final answer in \\boxed{}.",
            },
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": chosen},
        ],
        "metadata": metadata,
    }


def split_rows(rows: list[dict[str, Any]], validation_mod: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item.get("metadata", {}).get("family", ""), item["id"])):
        bucket = int(sha256_text(row["id"])[:8], 16) % validation_mod
        (val if bucket == 0 else train).append(row)
    return train, val


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: Counter[str] = Counter()
    by_negative: Counter[str] = Counter()
    by_rule: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata", {})
        by_family[str(metadata.get("family", "unknown"))] += 1
        by_negative[str(metadata.get("negative_type", "unknown"))] += 1
        by_rule[str(metadata.get("rule_class", "unknown"))] += 1
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(by_family.items())),
        "negative_type_counts": dict(sorted(by_negative.items())),
        "rule_class_counts": dict(by_rule.most_common(40)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V435E ADAPTER PROBE PREFERENCE DATASET START ===", flush=True)
    print("v435d_manifest_json =", args.v435d_manifest_json, flush=True)
    print("v435d_detail_csv =", args.v435d_detail_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    if not args.v435d_manifest_json.is_file():
        raise FileNotFoundError(args.v435d_manifest_json)
    if not args.v435d_detail_csv.is_file():
        raise FileNotFoundError(args.v435d_detail_csv)
    manifest = read_json(args.v435d_manifest_json)
    expected_detail_sha = str(manifest.get("outputs", {}).get("detail_sha256", ""))
    observed_detail_sha = sha256_file(args.v435d_detail_csv)
    if expected_detail_sha and observed_detail_sha != expected_detail_sha:
        raise RuntimeError(f"V435D detail hash mismatch: expected {expected_detail_sha}, got {observed_detail_sha}")

    detail_rows = read_csv(args.v435d_detail_csv)
    hard_negative_rows = [row for row in detail_rows if not truthy(row.get("correct"))]
    replay_source_rows = [
        row
        for row in detail_rows
        if truthy(row.get("correct")) and row.get("family") == "bit_manipulation"
    ]
    pairs = [make_pair(row, "hard_negative") for row in hard_negative_rows]
    for row in replay_source_rows:
        pairs.append(make_pair(row, "format_no_box"))

    train_rows, val_rows = split_rows(pairs, max(2, args.validation_mod))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_preferences_train.jsonl"
    val_path = args.output_dir / f"{args.label}_preferences_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    summary_csv = args.output_dir / f"{args.label}_summary.csv"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    summary_rows = [
        {"split": "train", **summarize(train_rows)},
        {"split": "validation", **summarize(val_rows)},
        {"split": "all", **summarize(pairs)},
    ]
    write_csv(summary_csv, summary_rows, ["split", "rows", "family_counts", "negative_type_counts", "rule_class_counts"])

    hard_negative_counts = Counter(row["family"] for row in hard_negative_rows)
    replay_counts = Counter(row["family"] for row in replay_source_rows)
    out_manifest = {
        "schema_version": "kg1_v435e_adapter_probe_preference_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "raw_outputs_collected_without_labels": True,
            "labels_joined_after_collection_from_public_train": True,
            "weak_full_rows_used_for_training": False,
            "weak_full_used_only_as_reference_overlap_filter": True,
            "purpose": "Train only on public-train prompt probes where the frozen adapter made real mistakes.",
        },
        "inputs": {
            "v435d_manifest_json": str(args.v435d_manifest_json),
            "v435d_manifest_sha256": sha256_file(args.v435d_manifest_json),
            "v435d_detail_csv": str(args.v435d_detail_csv),
            "v435d_detail_sha256": observed_detail_sha,
        },
        "build_policy": {
            "hard_negative_source": "V435D rows with correct=false",
            "chosen": "short deterministic completion ending with escaped boxed public-train answer",
            "rejected": "short exact-wrong completion ending with escaped boxed adapter prediction",
            "bit_guardrail": "V435D bit rows with correct=true become format no-box replay pairs",
            "validation_mod": args.validation_mod,
        },
        "source_counts": {
            "detail_rows": len(detail_rows),
            "hard_negative_rows": len(hard_negative_rows),
            "hard_negative_family_counts": dict(sorted(hard_negative_counts.items())),
            "bit_replay_source_rows": len(replay_source_rows),
            "bit_replay_family_counts": dict(sorted(replay_counts.items())),
        },
        "summary": {"train": summarize(train_rows), "validation": summarize(val_rows), "all": summarize(pairs)},
        "outputs": {
            "preferences_train_jsonl": str(train_path),
            "preferences_train_sha256": sha256_file(train_path),
            "preferences_val_jsonl": str(val_path),
            "preferences_val_sha256": sha256_file(val_path),
            "summary_csv": str(summary_csv),
            "summary_sha256": sha256_file(summary_csv),
            "manifest_json": str(manifest_path),
        },
        "next_gate": "Run V435F adapter-probe preference gate before any HF GPU job.",
    }
    write_json(manifest_path, out_manifest)
    print("source_counts =", json.dumps(out_manifest["source_counts"], sort_keys=True), flush=True)
    print("summary =", json.dumps(out_manifest["summary"], sort_keys=True), flush=True)
    print("preferences_train_sha256 =", out_manifest["outputs"]["preferences_train_sha256"], flush=True)
    print("preferences_val_sha256 =", out_manifest["outputs"]["preferences_val_sha256"], flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V435E ADAPTER PROBE PREFERENCE DATASET END ===", flush=True)
    return out_manifest


def self_test() -> None:
    row = {
        "id": "x",
        "family": "equation_transform",
        "answer": "|@{",
        "prediction": "bad}",
        "prompt": "In Alice's Wonderland...\n1+1 = 2\nNow, determine the result for: x",
        "correct": "False",
        "raw_output": "Final answer: bad",
        "decode_config_sha256": "abc",
        "adapter_repo": "repo",
        "adapter_subfolder": "checkpoint",
        "prompt_sha256": "p",
        "prompt_normalized_sha256": "pn",
    }
    pair = make_pair(row, "hard_negative")
    assert "\\boxed{|@\\{}" in pair["chosen"]
    assert "\\boxed{bad\\}}" in pair["rejected"]
    assert infer_rule_class(row) == "equation_numeric_operator_to_symbolic"
    print("v435e_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v435d-manifest-json", type=Path, default=DEFAULT_V435D_MANIFEST)
    parser.add_argument("--v435d-detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v435e_adapter_probe_preference_dataset")
    parser.add_argument("--validation-mod", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
