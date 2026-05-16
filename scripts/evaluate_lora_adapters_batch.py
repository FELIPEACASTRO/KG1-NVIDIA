#!/usr/bin/env python3
"""Batch weak-evaluate multiple LoRA adapters with one vLLM model load.

This is evaluation-only. It exists for candidate triage: load the Nemotron base
model once, iterate adapters with LoRARequest, and write the same report shape
used by ``evaluate_lora_adapter.py`` for each candidate plus an aggregate CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_lora_adapter import (  # noqa: E402
    _sampling_params,
    apply_prediction_postprocessor,
    apply_vllm_runtime_safety_settings,
    normalize_questions,
    prepare_merged_predictions,
    render_prompts,
    resolve_base_model_path,
    resolve_model_revision,
    row_id_column,
    summarize_per_task,
    validate_adapter_dir,
)
from src.competition_utils import (  # noqa: E402
    OFFICIAL_INFERENCE_CONFIG,
    classify_puzzle,
    extract_final_answer,
    verify_answer,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_candidates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("candidates", [])
    if not isinstance(data, list):
        raise ValueError("candidates JSON must be a list or {'candidates': [...]}")
    candidates: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {idx} is not an object")
        name = str(item.get("name") or item.get("id") or f"candidate_{idx + 1}").strip()
        adapter = str(item.get("adapter") or item.get("adapter_path") or "").strip()
        if not name:
            raise ValueError(f"candidate {idx} missing name")
        if not adapter:
            raise ValueError(f"candidate {name} missing adapter")
        candidates.append({**item, "name": name, "adapter": adapter})
    return candidates


def safe_label(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def build_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(OFFICIAL_INFERENCE_CONFIG)
    if args.max_tokens > 0:
        config["max_tokens"] = int(args.max_tokens)
    if args.max_model_len > 0:
        config["max_model_len"] = int(args.max_model_len)
    if args.max_num_seqs > 0:
        config["max_num_seqs"] = int(args.max_num_seqs)
    if args.gpu_memory_utilization > 0:
        config["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    config["warmup_rows"] = max(0, int(args.warmup_rows))
    if args.disable_thinking:
        config["enable_thinking"] = False
    if args.no_prompt_suffix:
        config["prompt_suffix"] = ""
    elif args.prompt_suffix:
        config["prompt_suffix"] = args.prompt_suffix
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-csv", type=Path, required=True)
    parser.add_argument("--questions-csv", type=Path, default=None)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--label-prefix", default="batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.0)
    parser.add_argument("--warmup-rows", type=int, default=4)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--no-prompt-suffix", action="store_true")
    parser.add_argument(
        "--prediction-postprocessor",
        default="none",
        choices=["none", "v274_numeric_operator_overrides"],
        help="Optional label-free postprocessor applied after generation and before scoring.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_eval_config(args)
    base_model_path = resolve_base_model_path(args.base_model_path)
    candidates = load_candidates(args.candidates_json)

    print("========================================================================")
    print("KG1 batch LoRA adapter evaluation")
    print("========================================================================")
    print("generated_at_utc =", utc_now())
    print("base_model_path =", base_model_path)
    print("candidates_json =", args.candidates_json)
    print("candidate_count =", len(candidates))
    print("seed =", args.seed)
    print("limit =", args.limit)
    print("output_dir =", args.output_dir)
    print("config =", json.dumps(config, indent=2, sort_keys=True))
    print(
        "vllm_runtime_safety_settings =",
        json.dumps(apply_vllm_runtime_safety_settings(), indent=2, sort_keys=True),
    )

    solution = pd.read_csv(args.solution_csv)
    if args.limit > 0:
        solution = solution.head(args.limit).copy()
    if "answer" not in solution.columns:
        raise RuntimeError(
            "solution CSV is missing the answer column; refusing to emit an all-false accuracy report."
        )
    questions = pd.read_csv(args.questions_csv or args.solution_csv)
    if args.limit > 0:
        ids = set(solution[row_id_column(solution)].astype(str))
        q_id = row_id_column(questions)
        questions = questions[questions[q_id].astype(str).isin(ids)].copy()
    questions = normalize_questions(solution, questions, limit=0)
    solution = solution.copy()
    id_col = row_id_column(solution)
    if id_col != "id":
        solution = solution.rename(columns={id_col: "id"})
    solution["id"] = solution["id"].astype(str)
    print("rows =", len(questions))

    valid_candidates: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    for item in candidates:
        row = {"name": item["name"], "adapter": item["adapter"], "preflight_ok": False, "error": ""}
        try:
            adapter_dir = validate_adapter_dir(item["adapter"])
            row["adapter"] = str(adapter_dir)
            row["preflight_ok"] = True
            valid_candidates.append({**item, "adapter": str(adapter_dir)})
        except Exception as exc:
            row["error"] = repr(exc)
            print("candidate_preflight_failed =", json.dumps(row, sort_keys=True), flush=True)
            if not args.continue_on_error:
                raise
        preflight_rows.append(row)
    pd.DataFrame(preflight_rows).to_csv(args.output_dir / "candidate_preflight.csv", index=False)
    print("valid_candidate_count =", len(valid_candidates))
    if not valid_candidates:
        raise RuntimeError("no valid candidates to evaluate")

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    llm_kwargs = {
        "model": str(base_model_path),
        "tensor_parallel_size": int(config.get("tensor_parallel_size", 1)),
        "max_num_seqs": int(config.get("max_num_seqs", 64)),
        "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.85)),
        "dtype": config.get("dtype", "auto"),
        "max_model_len": int(config.get("max_model_len", 8192)),
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        "enable_lora": True,
        "max_lora_rank": int(config.get("max_lora_rank", 32)),
        "enable_prefix_caching": bool(config.get("enable_prefix_caching", True)),
        "enable_chunked_prefill": bool(config.get("enable_chunked_prefill", True)),
    }
    model_revision = resolve_model_revision(str(base_model_path), config)
    if model_revision:
        llm_kwargs["revision"] = model_revision
        llm_kwargs["tokenizer_revision"] = model_revision
    print("llm_revision =", llm_kwargs.get("revision", "local_path_or_default"))
    if config.get("enforce_eager") is not None:
        llm_kwargs["enforce_eager"] = bool(config["enforce_eager"])

    load_start = time.time()
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"vLLM loaded in {time.time() - load_start:.1f}s")
    rendered = render_prompts(tokenizer, questions, config)
    sampling_params = _sampling_params(config, int(args.seed))

    aggregate_rows: list[dict[str, Any]] = []
    for lora_id, candidate in enumerate(valid_candidates, start=1):
        label = safe_label(f"{args.label_prefix}_{candidate['name']}")
        candidate_dir = args.output_dir / safe_label(candidate["name"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        print("------------------------------------------------------------------------")
        print("candidate_start =", json.dumps(candidate, sort_keys=True))
        cand_start = time.time()
        try:
            lora_request = LoRARequest(candidate["name"], lora_id, str(candidate["adapter"]))
            warmup_rows = int(config.get("warmup_rows", 4))
            if rendered and warmup_rows > 0:
                warmup_n = min(warmup_rows, len(rendered))
                print(f"warmup_rows = {warmup_n}")
                warmup_start = time.time()
                _ = llm.generate(rendered[:warmup_n], sampling_params=sampling_params, lora_request=lora_request)
                print(f"warmup_elapsed_s = {time.time() - warmup_start:.1f}")
            else:
                print("warmup_rows = 0")

            gen_start = time.time()
            outputs = llm.generate(rendered, sampling_params=sampling_params, lora_request=lora_request)
            gen_elapsed = time.time() - gen_start
            if len(outputs) != len(rendered):
                raise RuntimeError(f"vLLM output count mismatch: outputs={len(outputs)} prompts={len(rendered)}")
            print(f"generation_elapsed_s = {gen_elapsed:.1f}")

            rows: list[dict[str, Any]] = []
            for row, output in zip(questions.itertuples(index=False), outputs):
                completion = output.outputs[0]
                raw_output = completion.text
                prompt = str(getattr(row, "prompt"))
                rows.append(
                    {
                        "id": str(getattr(row, "id")),
                        "prompt": prompt,
                        "raw_output": raw_output,
                        "prediction": extract_final_answer(raw_output),
                        "prompt_tokens": len(getattr(output, "prompt_token_ids", []) or []),
                        "completion_tokens": len(getattr(completion, "token_ids", []) or []),
                        "finish_reason": completion.finish_reason or "",
                        "type": classify_puzzle(prompt),
                    }
                )

            pred = pd.DataFrame(rows)
            raw_predictions_path = candidate_dir / f"{label}_raw_predictions_pre_score.csv"
            pred.to_csv(raw_predictions_path, index=False)
            pred = apply_prediction_postprocessor(pred, args.prediction_postprocessor)
            postprocessed_raw_predictions_path = ""
            if args.prediction_postprocessor != "none":
                post_path = candidate_dir / f"{label}_postprocessed_raw_predictions_pre_score.csv"
                pred.to_csv(post_path, index=False)
                postprocessed_raw_predictions_path = str(post_path)
            merged = prepare_merged_predictions(solution, pred)
            if "answer" not in merged.columns:
                raise RuntimeError("merged predictions are missing answer column after join")
            merged["correct"] = merged.apply(lambda r: verify_answer(r["answer"], r["prediction"]), axis=1)
            merged["truncated"] = merged["finish_reason"].fillna("").astype(str).eq("length")

            predictions_path = candidate_dir / f"{label}_predictions.csv"
            per_task_path = candidate_dir / f"{label}_per_task.csv"
            report_path = candidate_dir / f"{label}_eval_report.json"
            merged.to_csv(predictions_path, index=False)
            per_task = summarize_per_task(merged)
            per_task.to_csv(per_task_path, index=False)
            total_tokens = int(merged["completion_tokens"].fillna(0).sum())
            report = {
                "generated_at_utc": utc_now(),
                "label": label,
                "candidate_name": candidate["name"],
                "base_model_path": str(base_model_path),
                "adapter_dir": str(candidate["adapter"]),
                "rows": int(len(merged)),
                "correct": int(merged["correct"].sum()),
                "accuracy": float(merged["correct"].mean()) if len(merged) else 0.0,
                "truncated": int(merged["truncated"].sum()),
                "truncation_rate": float(merged["truncated"].mean()) if len(merged) else 0.0,
                "completion_tokens": total_tokens,
                "generation_elapsed_s": gen_elapsed,
                "tokens_per_second": float(total_tokens / gen_elapsed) if gen_elapsed > 0 else 0.0,
                "candidate_elapsed_s": time.time() - cand_start,
                "seed": int(args.seed),
                "config": config,
                "prediction_postprocessor": str(args.prediction_postprocessor or "none"),
                "outputs": {
                    "raw_predictions_pre_score_csv": str(raw_predictions_path),
                    "postprocessed_raw_predictions_pre_score_csv": postprocessed_raw_predictions_path,
                    "predictions_csv": str(predictions_path),
                    "per_task_csv": str(per_task_path),
                    "report_json": str(report_path),
                },
            }
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            by_task = {str(r["task_type"]): r for _, r in per_task.iterrows()}
            aggregate = {
                "name": candidate["name"],
                "adapter": str(candidate["adapter"]),
                "status": "ok",
                "correct": report["correct"],
                "accuracy": report["accuracy"],
                "truncated": report["truncated"],
                "truncation_rate": report["truncation_rate"],
                "equation_transform_correct": int(by_task.get("equation_transform", {}).get("correct", 0)),
                "bit_manipulation_correct": int(by_task.get("bit_manipulation", {}).get("correct", 0)),
                "completion_tokens": report["completion_tokens"],
                "tokens_per_second": report["tokens_per_second"],
                "report_json": str(report_path),
                "error": "",
            }
            aggregate_rows.append(aggregate)
            print("candidate_summary =", json.dumps(aggregate, indent=2, sort_keys=True))
            print("candidate_per_task =")
            print(per_task.to_string(index=False))
        except Exception as exc:
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print("candidate_failed =", candidate["name"], error_text, flush=True)
            aggregate_rows.append(
                {
                    "name": candidate["name"],
                    "adapter": str(candidate["adapter"]),
                    "status": "failed",
                    "correct": 0,
                    "accuracy": 0.0,
                    "truncated": 999999,
                    "truncation_rate": 1.0,
                    "equation_transform_correct": 0,
                    "bit_manipulation_correct": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0.0,
                    "report_json": "",
                    "error": error_text,
                }
            )
            if not args.continue_on_error:
                raise

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_csv = args.output_dir / "batch_candidate_summary.csv"
    aggregate_json = args.output_dir / "batch_candidate_summary.json"
    aggregate.to_csv(aggregate_csv, index=False)
    aggregate_json.write_text(
        json.dumps(
            {
                "generated_at_utc": utc_now(),
                "elapsed_s": time.time() - started_at,
                "base_model_path": str(base_model_path),
                "solution_csv": str(args.solution_csv),
                "questions_csv": str(args.questions_csv or args.solution_csv),
                "candidates_json": str(args.candidates_json),
                "config": config,
                "rows": aggregate_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("aggregate_csv =", aggregate_csv)
    print("aggregate_json =", aggregate_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
