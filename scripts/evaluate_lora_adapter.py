#!/usr/bin/env python3
"""Official-like vLLM evaluator for Nemotron LoRA adapters.

This script is intentionally evaluation-only. It does not train, package, or
submit. It generates answers with the same scoring-facing settings used by the
public local-CV notebooks: LoRA enabled, max rank 32, 8192 context, 7680 output
tokens, deterministic sampling, boxed-answer extraction, and per-family ACC.
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


def render_prompts(tokenizer: Any, questions: pd.DataFrame) -> list[str]:
    prompts: list[str] = []
    for row in questions.itertuples(index=False):
        user_content = str(getattr(row, "prompt")) + PROMPT_SUFFIX
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
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
    os.environ.setdefault("VLLM_DISABLE_LOG_STATS", "1")
    keys = [
        "KG1_ALLOW_VLLM_DEEP_GEMM",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_MOE_USE_DEEP_GEMM",
        "VLLM_USE_DEEP_GEMM_E8M0",
        "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES",
        "VLLM_DEEP_GEMM_WARMUP",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        "VLLM_DISABLE_LOG_STATS",
    ]
    return {key: os.environ.get(key, "") for key in keys}


def evaluate_adapter(
    solution: pd.DataFrame,
    questions: pd.DataFrame,
    *,
    lora_path: str,
    base_model_path: str,
    config: dict[str, Any] | None = None,
    seed: int = 42,
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

    rendered = render_prompts(tokenizer, questions)
    sampling_params = _sampling_params(config, seed)
    lora_request = LoRARequest("adapter", 1, str(adapter_dir))

    if rendered:
        warmup_n = min(4, len(rendered))
        print(f"warmup_rows = {warmup_n}")
        warmup_start = time.time()
        _ = llm.generate(rendered[:warmup_n], sampling_params=sampling_params, lora_request=lora_request)
        print(f"warmup_elapsed_s = {time.time() - warmup_start:.1f}")

    gen_start = time.time()
    outputs = llm.generate(rendered, sampling_params=sampling_params, lora_request=lora_request)
    gen_elapsed = time.time() - gen_start
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
    merged = solution.merge(pred, on="id", how="left", validate="one_to_one")
    if "answer" in merged.columns:
        merged["correct"] = merged.apply(lambda r: verify_answer(r["answer"], r["prediction"]), axis=1)
    else:
        merged["correct"] = False
    if "type" not in merged.columns:
        merged["type"] = merged["prompt"].map(classify_puzzle)
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solution = pd.read_csv(args.solution_csv)
    if args.limit > 0:
        solution = solution.head(args.limit).copy()
    questions = pd.read_csv(args.questions_csv or args.solution_csv)
    if args.limit > 0:
        ids = set(solution[row_id_column(solution)].astype(str))
        q_id = row_id_column(questions)
        questions = questions[questions[q_id].astype(str).isin(ids)].copy()

    summary, predictions = evaluate_adapter(
        solution,
        questions,
        lora_path=str(args.adapter),
        base_model_path=resolve_base_model_path(args.base_model_path),
        config=OFFICIAL_INFERENCE_CONFIG,
        seed=args.seed,
    )
    label = args.label.replace("/", "_").replace("\\", "_")
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
