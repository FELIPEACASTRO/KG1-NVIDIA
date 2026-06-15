#!/usr/bin/env python3
"""Official-like vLLM evaluator for Nemotron LoRA adapters.

This script is intentionally evaluation-only. It does not train, package, or
submit. By default it generates answers with the scoring-facing settings
visible in the downloaded public Kaggle metric notebook: LoRA enabled, max
rank 32, 8192 context, 7680 output tokens, temperature 0.0/top_p 1.0,
official final-answer extraction, and per-family ACC.
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
    canonical_answer,
    classify_puzzle,
    ensure_prompt_suffix,
    extract_boxed_answers,
    extract_final_answer,
    extract_final_boxed_answer,
    has_closed_boxed_answer,
    verify_answer,
    verify_strict_boxed_answer,
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
    raise ValueError("CSV must contain an id or row_id column")


def read_csv_str(path: str | Path) -> pd.DataFrame:
    """Read challenge CSVs without numeric coercion or NA rewriting."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    empty_columns = [column for column in frame.columns if not str(column).strip()]
    if empty_columns:
        raise ValueError(f"{path} contains empty CSV column names")
    return frame


def exact_answer_match(expected: object, observed: object) -> bool:
    expected_text = canonical_answer(expected).lower()
    observed_text = canonical_answer(observed).lower()
    return bool(expected_text and observed_text and observed_text != "not_found" and expected_text == observed_text)


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
        user_content = ensure_prompt_suffix(getattr(row, "prompt"), prompt_suffix)
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


TRUNCATED_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "token_limit",
    "truncated",
}


def is_truncated_completion(
    finish_reason: object,
    completion_tokens: object,
    max_tokens: int,
) -> bool:
    """Detect generation truncation across vLLM/HF reason variants."""

    normalized = str(finish_reason or "").strip().lower()
    if normalized in TRUNCATED_FINISH_REASONS:
        return True
    try:
        token_count = int(float(str(completion_tokens).strip()))
    except (TypeError, ValueError):
        return False
    return token_count >= int(max_tokens)


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


def evaluate_adapter(
    solution: pd.DataFrame,
    questions: pd.DataFrame,
    *,
    lora_path: str,
    base_model_path: str,
    config: dict[str, Any] | None = None,
    seed: int = 42,
    raw_predictions_path: str | Path | None = None,
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
    print(f"generation_elapsed_s = {gen_elapsed:.1f}")

    rows: list[dict[str, Any]] = []
    for row, output in zip(questions.itertuples(index=False), outputs):
        completion = output.outputs[0]
        raw_output = completion.text
        prediction = extract_final_answer(raw_output)
        boxed_answers = [item.strip() for item in extract_boxed_answers(raw_output)]
        strict_boxed_prediction = extract_final_boxed_answer(raw_output)
        well_formed_boxed = has_closed_boxed_answer(raw_output)
        row_id = str(getattr(row, "id"))
        prompt = str(getattr(row, "prompt"))
        expected = getattr(row, "answer", None)
        rows.append(
            {
                "id": row_id,
                "prompt": prompt,
                "raw_output": raw_output,
                "prediction": prediction,
                "strict_boxed_prediction": strict_boxed_prediction,
                "strict_boxed_correct": (
                    verify_strict_boxed_answer(expected, raw_output)
                    if expected is not None
                    else False
                ),
                "exact_prediction_correct": (
                    exact_answer_match(expected, prediction)
                    if expected is not None
                    else False
                ),
                "boxed_count": len(boxed_answers),
                "first_boxed_prediction": boxed_answers[0] if boxed_answers else "",
                "well_formed_boxed": well_formed_boxed,
                "malformed_boxed": len(boxed_answers) > 0 and not well_formed_boxed,
                "no_box_fallback_used": len(boxed_answers) == 0,
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

    merged = prepare_merged_predictions(solution, pred)
    if "answer" not in merged.columns:
        raise RuntimeError("merged predictions are missing answer column after join")
    merged["correct"] = merged.apply(lambda r: verify_answer(r["answer"], r["prediction"]), axis=1)
    if "first_boxed_prediction" in merged.columns:
        merged["first_boxed_correct"] = merged.apply(
            lambda r: verify_answer(r["answer"], r["first_boxed_prediction"]),
            axis=1,
        )
    if "strict_boxed_prediction" in merged.columns:
        merged["strict_boxed_correct"] = merged.apply(
            lambda r: exact_answer_match(r["answer"], r["strict_boxed_prediction"]),
            axis=1,
        )
    merged["exact_prediction_correct"] = merged.apply(
        lambda r: exact_answer_match(r["answer"], r["prediction"]),
        axis=1,
    )
    if "strict_boxed_correct" not in merged.columns:
        raise RuntimeError("merged predictions missing strict_boxed_correct")
    merged["strict_boxed_correct"] = merged["strict_boxed_correct"].fillna(False).astype(bool)
    merged["exact_prediction_correct"] = merged["exact_prediction_correct"].fillna(False).astype(bool)
    merged["public_metric_only_false_gain"] = merged["correct"] & ~merged["strict_boxed_correct"]
    max_tokens = int(config.get("max_tokens", 7680))
    merged["truncated"] = merged.apply(
        lambda r: is_truncated_completion(r.get("finish_reason", ""), r.get("completion_tokens", ""), max_tokens),
        axis=1,
    )

    # ---- JOB B: PAINEL DIDATICO (sempre) + TRACE por-item (KG1_EVAL_TRACE=1) ----
    # Reusa a metrica oficial via kg1_score_live. try/except: o juiz e o report do eval;
    # isto e camada de leitura humana e NUNCA pode quebrar o eval.
    try:
        import sys as _sys
        _sd = os.path.dirname(os.path.abspath(__file__))
        if _sd not in _sys.path:
            _sys.path.insert(0, _sd)
        from kg1_score_live import compute_score_telemetry, render_dashboard

        _fam_col = "family" if "family" in merged.columns else ("type" if "type" in merged.columns else None)
        _recs = merged.to_dict("records")
        _tel_rows = [
            {
                "id": str(r.get("id", "")),
                "raw_output": r.get("raw_output", "") or "",
                "answer": r.get("answer", ""),
                "family": (str(r.get(_fam_col, "unknown")) if _fam_col else "unknown"),
                "finish_reason": r.get("finish_reason", ""),
                "completion_tokens": r.get("completion_tokens", 0),
            }
            for r in _recs
        ]
        _tel = compute_score_telemetry(_tel_rows, step=None, max_step=None)
        _title = "JOB B - EVAL full947 [" + os.path.basename(str(adapter_dir).rstrip("/")) + "]"
        print(render_dashboard(_tel, title=_title, source="vLLM greedy full947 (JUIZ OFICIAL)"), flush=True)
        if os.environ.get("KG1_EVAL_TRACE", "0") == "1":
            print(
                f"[KG1-EVAL-TRACE] por-item ({len(_recs)} itens) -- familia | id | correct | trunc | pred -> gold",
                flush=True,
            )
            for r in _recs:
                _fam = (str(r.get(_fam_col, "?")) if _fam_col else "?")
                print(
                    f"[TRACE] {_fam:<18} {str(r.get('id', ''))[:10]:<10} "
                    f"correct={'Y' if bool(r.get('correct')) else 'N'} "
                    f"trunc={'Y' if bool(r.get('truncated')) else 'N'} "
                    f"pred={str(r.get('prediction', ''))[:40]!r} gold={str(r.get('answer', ''))[:40]!r}",
                    flush=True,
                )
    except Exception as _exc:  # diagnostico NUNCA quebra o eval
        print(f"[KG1-EVAL-PANEL] falhou (ignorado, eval segue): {type(_exc).__name__}: {_exc}", flush=True)

    total_tokens = int(merged["completion_tokens"].fillna(0).sum())
    boxed_rows = int(merged["boxed_count"].fillna(0).astype(int).gt(0).sum()) if "boxed_count" in merged else 0
    malformed_boxed_rows = (
        int(merged["malformed_boxed"].fillna(False).astype(bool).sum())
        if "malformed_boxed" in merged
        else 0
    )
    no_box_fallback_rows = (
        int(merged["no_box_fallback_used"].fillna(False).astype(bool).sum())
        if "no_box_fallback_used" in merged
        else int(len(merged))
    )
    first_boxed_correct = (
        int(merged["first_boxed_correct"].fillna(False).astype(bool).sum())
        if "first_boxed_correct" in merged
        else 0
    )
    strict_boxed_correct = int(merged["strict_boxed_correct"].sum())
    exact_prediction_correct = int(merged["exact_prediction_correct"].sum())
    public_metric_only_false_gain_rows = int(merged["public_metric_only_false_gain"].sum())
    summary = {
        "generated_at_utc": utc_now(),
        "base_model_path": str(base_model_path),
        "adapter_dir": str(adapter_dir),
        "rows": int(len(merged)),
        "correct": int(merged["correct"].sum()),
        "accuracy": float(merged["correct"].mean()) if len(merged) else 0.0,
        "truncated": int(merged["truncated"].sum()),
        "truncation_rate": float(merged["truncated"].mean()) if len(merged) else 0.0,
        "boxed_rows": boxed_rows,
        "boxed_rate": float(boxed_rows / len(merged)) if len(merged) else 0.0,
        "malformed_boxed_rows": malformed_boxed_rows,
        "malformed_boxed_rate": float(malformed_boxed_rows / len(merged)) if len(merged) else 0.0,
        "no_box_fallback_rows": no_box_fallback_rows,
        "no_box_fallback_rate": float(no_box_fallback_rows / len(merged)) if len(merged) else 0.0,
        "first_boxed_correct": first_boxed_correct,
        "first_boxed_accuracy": float(first_boxed_correct / len(merged)) if len(merged) else 0.0,
        "strict_boxed_correct": strict_boxed_correct,
        "strict_boxed_accuracy": float(strict_boxed_correct / len(merged)) if len(merged) else 0.0,
        "exact_prediction_correct": exact_prediction_correct,
        "exact_prediction_accuracy": float(exact_prediction_correct / len(merged)) if len(merged) else 0.0,
        "public_metric_only_false_gain_rows": public_metric_only_false_gain_rows,
        "public_metric_only_false_gain_rate": (
            float(public_metric_only_false_gain_rows / len(merged)) if len(merged) else 0.0
        ),
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
    parser.add_argument("--max-tokens", type=int, default=0, help="Diagnostic override for generation max_tokens.")
    parser.add_argument("--max-model-len", type=int, default=0, help="Diagnostic override for vLLM max_model_len.")
    parser.add_argument("--max-num-seqs", type=int, default=0, help="Diagnostic override for vLLM max_num_seqs.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.0, help="Diagnostic override for vLLM memory fraction.")
    parser.add_argument("--warmup-rows", type=int, default=4, help="Number of explicit warmup rows before measured generation.")
    parser.add_argument("--disable-thinking", action="store_true", help="Diagnostic prompt rendering with enable_thinking=False.")
    parser.add_argument("--prompt-suffix", default="", help="Diagnostic override for the prompt suffix.")
    parser.add_argument("--no-prompt-suffix", action="store_true", help="Diagnostic override: render prompts without the default suffix.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solution = read_csv_str(args.solution_csv)
    if args.limit > 0:
        solution = solution.head(args.limit).copy()
    questions = read_csv_str(args.questions_csv or args.solution_csv)
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
