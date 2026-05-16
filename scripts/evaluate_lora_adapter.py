#!/usr/bin/env python3
"""Official-like vLLM evaluator for Nemotron LoRA adapters.

This script is intentionally evaluation-only. It does not train, package, or
submit. By default it generates answers with the same scoring-facing settings
used by the public local-CV notebooks: LoRA enabled, max rank 32, 8192 context,
7680 output tokens, deterministic sampling, boxed-answer extraction, and
per-family ACC. CLI overrides are diagnostic unless they match the official
defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    MODEL_NAME,
    OFFICIAL_INFERENCE_CONFIG,
    PROMPT_SUFFIX,
    classify_puzzle,
    extract_final_answer,
    verify_answer,
)
from src.kg1_v274_numeric_postprocessor import postprocess_rows as v274_postprocess_rows  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seeds(raw: str | int | None) -> list[int]:
    if raw is None or raw == "":
        return [42]
    if isinstance(raw, int):
        return [raw]
    seeds: list[int] = []
    for chunk in str(raw).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            seeds.append(int(chunk))
    return seeds or [42]


def resolve_base_model_path(base_model_path: str = "") -> str:
    """Resolve the base model path for Colab, Kaggle, or local H100 runs."""

    if base_model_path:
        return base_model_path
    env_path = os.environ.get("KG1_BASE_MODEL_PATH") or os.environ.get("BASE_MODEL_PATH")
    if env_path:
        return env_path

    kaggle_candidates = [
        "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
        "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    ]
    for candidate in kaggle_candidates:
        if Path(candidate).exists():
            return candidate
    return MODEL_NAME


def resolve_model_revision(base_model_path: str, config: dict[str, Any]) -> str | None:
    """Use the pinned HF revision for repo IDs, but not for local model paths."""

    revision = str(config.get("model_revision") or "").strip()
    if not revision:
        return None
    model_text = str(base_model_path)
    if Path(model_text).exists() or os.path.isabs(model_text):
        return None
    return revision


def row_id_column(frame: pd.DataFrame) -> str:
    for candidate in ("id", "row_id"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns.to_list()[0])


def normalize_questions(solution: pd.DataFrame, questions: pd.DataFrame, limit: int = 0) -> pd.DataFrame:
    solution = solution.copy()
    questions = questions.copy()
    sol_id = row_id_column(solution)
    q_id = row_id_column(questions)
    if sol_id != "id":
        solution = solution.rename(columns={sol_id: "id"})
    if q_id != "id":
        questions = questions.rename(columns={q_id: "id"})
    solution["id"] = solution["id"].astype(str)
    questions["id"] = questions["id"].astype(str)
    if "prompt" not in questions.columns:
        if "prompt" not in solution.columns:
            raise ValueError("questions or solution must contain a prompt column")
        questions = solution[["id", "prompt"]].copy()
    ordered = solution[["id"]].merge(questions, on="id", how="left", validate="one_to_one")
    missing_prompt = ordered["prompt"].isna().sum()
    if missing_prompt:
        raise ValueError(f"questions missing prompts for {missing_prompt} solution rows")
    if limit > 0:
        ordered = ordered.head(limit).copy()
    return ordered


def validate_adapter_dir(adapter_dir: str | Path) -> Path:
    path = Path(adapter_dir)
    if not path.exists():
        raise FileNotFoundError(f"adapter path does not exist: {path}")
    if path.is_file() and path.suffix == ".zip":
        raise ValueError("adapter zip must be extracted before vLLM evaluation")
    config = path / "adapter_config.json"
    if not config.exists():
        raise FileNotFoundError(f"missing adapter_config.json: {config}")
    model_files = list(path.glob("adapter_model.safetensors")) + list(path.glob("adapter_model.bin"))
    if not model_files:
        raise FileNotFoundError(f"missing adapter_model.safetensors or adapter_model.bin in {path}")
    return path


def render_prompts(tokenizer: Any, questions: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    prompt_suffix = str(config.get("prompt_suffix", PROMPT_SUFFIX))
    enable_thinking = bool(config.get("enable_thinking", True))
    for row in questions.itertuples(index=False):
        user_content = str(getattr(row, "prompt")) + prompt_suffix
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)
    return prompts


def _sampling_params(config: dict[str, Any], seed: int):
    from vllm import SamplingParams

    kwargs = {
        "temperature": float(config.get("temperature", 0.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "max_tokens": int(config.get("max_tokens", 7680)),
    }
    try:
        return SamplingParams(**kwargs, seed=int(seed))
    except TypeError:
        return SamplingParams(**kwargs)


def apply_vllm_runtime_safety_settings() -> dict[str, str]:
    """Apply Colab-safe vLLM settings before importing/initializing vLLM.

    Colab H100 runtimes may install vLLM builds where DeepGEMM is enabled by
    default, but the `deep_gemm` backend package is absent or too old. In that
    state vLLM can fail during engine warmup before any generation starts. The
    challenge evaluation does not require DeepGEMM specifically, so disable it
    unless the caller explicitly opts back in.
    """

    allow_deep_gemm = os.environ.get("KG1_ALLOW_VLLM_DEEP_GEMM", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow_deep_gemm:
        os.environ["VLLM_USE_DEEP_GEMM"] = "0"
        os.environ["VLLM_MOE_USE_DEEP_GEMM"] = "0"
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"
        os.environ["VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES"] = "0"
        os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"
    os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
    keys = [
        "KG1_ALLOW_VLLM_DEEP_GEMM",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_MOE_USE_DEEP_GEMM",
        "VLLM_USE_DEEP_GEMM_E8M0",
        "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES",
        "VLLM_DEEP_GEMM_WARMUP",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    ]
    return {key: os.environ.get(key, "") for key in keys}


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def prepare_merged_predictions(solution: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Merge generated rows with labels without relying on pandas suffix names."""

    pred_for_merge = pred.rename(columns={"prompt": "generated_prompt", "type": "pred_type"}).copy()
    merged = solution.merge(pred_for_merge, on="id", how="left", validate="one_to_one")

    prompt_col = first_existing_column(merged, ["prompt", "generated_prompt", "prompt_x", "prompt_y"])
    if prompt_col is None:
        merged["prompt"] = ""
    elif prompt_col != "prompt":
        merged["prompt"] = merged[prompt_col].fillna("").astype(str)

    if "prediction" not in merged.columns:
        merged["prediction"] = ""
    if "finish_reason" not in merged.columns:
        merged["finish_reason"] = ""
    if "completion_tokens" not in merged.columns:
        merged["completion_tokens"] = 0

    type_col = first_existing_column(merged, ["type", "task_type", "family", "type_x", "pred_type", "type_y"])
    if type_col is None:
        merged["type"] = merged["prompt"].map(classify_puzzle)
    elif type_col != "type":
        merged["type"] = merged[type_col].fillna("").astype(str)
    missing_type = merged["type"].fillna("").astype(str).eq("")
    if missing_type.any():
        merged.loc[missing_type, "type"] = merged.loc[missing_type, "prompt"].map(classify_puzzle)

    return merged


def apply_prediction_postprocessor(pred: pd.DataFrame, name: str) -> pd.DataFrame:
    mode = str(name or "none").strip()
    if mode in {"", "none"}:
        return pred
    if mode != "v274_numeric_operator_overrides":
        raise ValueError(f"unknown prediction postprocessor: {mode}")
    rows = v274_postprocess_rows(pred.to_dict(orient="records"))
    out = pd.DataFrame(rows)
    applied = int(out.get("postprocessor_applied", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    print("prediction_postprocessor =", mode, flush=True)
    print("prediction_postprocessor_applied_rows =", applied, flush=True)
    return out


def evaluate_adapter(
    solution: pd.DataFrame,
    questions: pd.DataFrame,
    *,
    lora_path: str,
    base_model_path: str,
    config: dict[str, Any] | None = None,
    seed: int = 42,
    raw_predictions_path: str | Path | None = None,
    prediction_postprocessor: str = "none",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run vLLM adapter inference and return a summary plus row predictions."""

    config = {**OFFICIAL_INFERENCE_CONFIG, **(config or {})}
    adapter_dir = validate_adapter_dir(lora_path)
    questions = normalize_questions(solution, questions, limit=0)
    solution = solution.copy()
    id_col = row_id_column(solution)
    if id_col != "id":
        solution = solution.rename(columns={id_col: "id"})
    solution["id"] = solution["id"].astype(str)
    if "answer" not in solution.columns:
        raise RuntimeError(
            "solution CSV is missing the answer column; refusing to emit an all-false accuracy report."
        )

    print("========================================================================")
    print("KG1 official-like adapter evaluation")
    print("========================================================================")
    print("generated_at_utc =", utc_now())
    print("base_model_path =", base_model_path)
    print("adapter_dir =", adapter_dir)
    print("rows =", len(questions))
    print("seed =", seed)
    print("config =", json.dumps(config, indent=2, sort_keys=True))
    print(
        "vllm_runtime_safety_settings =",
        json.dumps(apply_vllm_runtime_safety_settings(), indent=2, sort_keys=True),
    )

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

    start = time.time()
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"vLLM loaded in {time.time() - start:.1f}s")

    rendered = render_prompts(tokenizer, questions, config)
    sampling_params = _sampling_params(config, seed)
    lora_request = LoRARequest("adapter", 1, str(adapter_dir))

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
        prediction = extract_final_answer(raw_output)
        row_id = str(getattr(row, "id"))
        prompt = str(getattr(row, "prompt"))
        rows.append(
            {
                "id": row_id,
                "prompt": prompt,
                "raw_output": raw_output,
                "prediction": prediction,
                "prompt_tokens": len(getattr(output, "prompt_token_ids", []) or []),
                "completion_tokens": len(getattr(completion, "token_ids", []) or []),
                "finish_reason": completion.finish_reason or "",
                "type": classify_puzzle(prompt),
            }
        )

    pred = pd.DataFrame(rows)
    if raw_predictions_path is not None:
        raw_path = Path(raw_predictions_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(raw_path, index=False)
        print("raw_predictions_pre_score_csv =", raw_path)
        print("raw_predictions_pre_score_rows =", len(pred))

    pred = apply_prediction_postprocessor(pred, prediction_postprocessor)

    merged = prepare_merged_predictions(solution, pred)
    if "answer" not in merged.columns:
        raise RuntimeError("merged predictions are missing answer column after join")
    merged["correct"] = merged.apply(lambda r: verify_answer(r["answer"], r["prediction"]), axis=1)
    merged["truncated"] = merged["finish_reason"].fillna("").astype(str).eq("length")

    total_tokens = int(merged["completion_tokens"].fillna(0).sum())
    summary = {
        "generated_at_utc": utc_now(),
        "base_model_path": str(base_model_path),
        "adapter_dir": str(adapter_dir),
        "rows": int(len(merged)),
        "correct": int(merged["correct"].sum()),
        "accuracy": float(merged["correct"].mean()) if len(merged) else 0.0,
        "truncated": int(merged["truncated"].sum()),
        "truncation_rate": float(merged["truncated"].mean()) if len(merged) else 0.0,
        "completion_tokens": total_tokens,
        "generation_elapsed_s": gen_elapsed,
        "tokens_per_second": float(total_tokens / gen_elapsed) if gen_elapsed > 0 else 0.0,
        "seed": int(seed),
        "config": config,
        "prediction_postprocessor": str(prediction_postprocessor or "none"),
    }
    print("summary =", json.dumps(summary, indent=2, sort_keys=True))
    return summary, merged


def summarize_per_task(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby("type", dropna=False)
    for family, group in grouped:
        total = int(len(group))
        correct = int(group["correct"].sum())
        truncated = int(group["truncated"].sum()) if "truncated" in group else 0
        rows.append(
            {
                "task_type": str(family),
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "truncated": truncated,
                "truncation_rate": truncated / total if total else 0.0,
            }
        )
    total = int(len(frame))
    correct = int(frame["correct"].sum()) if "correct" in frame else 0
    truncated = int(frame["truncated"].sum()) if "truncated" in frame else 0
    rows.append(
        {
            "task_type": "OVERALL",
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "truncated": truncated,
            "truncation_rate": truncated / total if total else 0.0,
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-csv", type=Path, required=True)
    parser.add_argument("--questions-csv", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--label", default="adapter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=0, help="Diagnostic override for generation max_tokens.")
    parser.add_argument("--max-model-len", type=int, default=0, help="Diagnostic override for vLLM max_model_len.")
    parser.add_argument("--max-num-seqs", type=int, default=0, help="Diagnostic override for vLLM max_num_seqs.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.0, help="Diagnostic override for vLLM memory fraction.")
    parser.add_argument("--warmup-rows", type=int, default=4, help="Number of explicit warmup rows before measured generation.")
    parser.add_argument("--disable-thinking", action="store_true", help="Diagnostic prompt rendering with enable_thinking=False.")
    parser.add_argument("--prompt-suffix", default="", help="Diagnostic override for the prompt suffix.")
    parser.add_argument("--no-prompt-suffix", action="store_true", help="Diagnostic override: render prompts without the default suffix.")
    parser.add_argument(
        "--prediction-postprocessor",
        default="none",
        choices=["none", "v274_numeric_operator_overrides"],
        help="Optional label-free postprocessor applied after generation and before scoring.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    label = args.label.replace("/", "_").replace("\\", "_")
    raw_predictions_path = args.output_dir / f"{label}_raw_predictions_pre_score.csv"
    eval_config = dict(OFFICIAL_INFERENCE_CONFIG)
    if args.max_tokens > 0:
        eval_config["max_tokens"] = int(args.max_tokens)
    if args.max_model_len > 0:
        eval_config["max_model_len"] = int(args.max_model_len)
    if args.max_num_seqs > 0:
        eval_config["max_num_seqs"] = int(args.max_num_seqs)
    if args.gpu_memory_utilization > 0:
        eval_config["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    eval_config["warmup_rows"] = max(0, int(args.warmup_rows))
    if args.disable_thinking:
        eval_config["enable_thinking"] = False
    if args.no_prompt_suffix:
        eval_config["prompt_suffix"] = ""
    elif args.prompt_suffix:
        eval_config["prompt_suffix"] = args.prompt_suffix

    summary, predictions = evaluate_adapter(
        solution,
        questions,
        lora_path=str(args.adapter),
        base_model_path=resolve_base_model_path(args.base_model_path),
        config=eval_config,
        seed=args.seed,
        raw_predictions_path=raw_predictions_path,
        prediction_postprocessor=args.prediction_postprocessor,
    )
    predictions_path = args.output_dir / f"{label}_predictions.csv"
    per_task_path = args.output_dir / f"{label}_per_task.csv"
    report_path = args.output_dir / f"{label}_eval_report.json"
    predictions.to_csv(predictions_path, index=False)
    summarize_per_task(predictions).to_csv(per_task_path, index=False)
    report = {
        **summary,
        "label": args.label,
        "inputs": {
            "solution_csv": str(args.solution_csv),
            "questions_csv": str(args.questions_csv or args.solution_csv),
            "adapter": str(args.adapter),
            "limit": args.limit,
        },
        "outputs": {
            "raw_predictions_pre_score_csv": str(raw_predictions_path),
            "predictions_csv": str(predictions_path),
            "per_task_csv": str(per_task_path),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("predictions_csv =", predictions_path)
    print("per_task_csv =", per_task_path)
    print("report_json =", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
