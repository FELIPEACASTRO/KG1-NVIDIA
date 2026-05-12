#!/usr/bin/env python3
"""HF Jobs preference trainer V315.

This is a lightweight verifier-distillation trainer for V312 chosen/rejected
pairs. It intentionally avoids full DPO with a duplicated reference model: a
30B BF16 base plus a second reference copy is too expensive and fragile for the
current budget. Instead it uses one policy model with a contrastive preference
loss over completion log-probabilities plus a small chosen-completion CE term.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import hf_job_train_v90 as base


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


PREF_TRAIN_FILE = env_str("PREF_TRAIN_FILE", "")
PREF_VAL_FILE = env_str("PREF_VAL_FILE", "")
EXPECTED_PREF_TRAIN_SHA256 = env_str("EXPECTED_PREF_TRAIN_SHA256", "")
EXPECTED_PREF_VAL_SHA256 = env_str("EXPECTED_PREF_VAL_SHA256", "")
MIN_PREF_TRAIN_EXAMPLES = env_int("MIN_PREF_TRAIN_EXAMPLES", 1)
MIN_PREF_VAL_EXAMPLES = env_int("MIN_PREF_VAL_EXAMPLES", 1)

PREF_BETA = env_float("PREF_BETA", 0.10)
PREF_MARGIN = env_float("PREF_MARGIN", 0.0)
PREF_LOSS_WEIGHT = env_float("PREF_LOSS_WEIGHT", 1.0)
CHOSEN_CE_WEIGHT = env_float("CHOSEN_CE_WEIGHT", 0.15)
REJECTED_CE_WEIGHT = env_float("REJECTED_CE_WEIGHT", 0.0)
PAIR_SCORE_MODE = env_str("PAIR_SCORE_MODE", "mean_nll")
if PAIR_SCORE_MODE not in {"mean_nll", "sum_nll"}:
    raise ValueError("PAIR_SCORE_MODE must be mean_nll or sum_nll")

SYSTEM_PROMPT = env_str(
    "PREFERENCE_SYSTEM_PROMPT",
    (
        "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
        "Infer the hidden rule from the examples, verify the candidate briefly, "
        "then end with exactly one final answer in \\boxed{}."
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def validate_preference_rows(rows: list[dict[str, Any]], label: str, min_rows: int) -> dict[str, Any]:
    if len(rows) < min_rows:
        raise RuntimeError(f"{label} preference rows below floor: {len(rows)} < {min_rows}")
    bad: list[str] = []
    family_counts: dict[str, int] = {}
    negative_counts: dict[str, int] = {}
    ids: set[str] = set()
    for idx, row in enumerate(rows):
        row_id = str(row.get("id", ""))
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        metadata = row.get("metadata") or {}
        family = str(metadata.get("family") or row.get("family") or "unknown")
        negative_type = str(metadata.get("negative_type") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        negative_counts[negative_type] = negative_counts.get(negative_type, 0) + 1
        if not row_id or not prompt or not chosen or not rejected:
            bad.append(f"{idx}:missing_required_field")
        if row_id in ids:
            bad.append(f"{row_id}:duplicate_id")
        ids.add(row_id)
        if chosen == rejected:
            bad.append(f"{row_id}:chosen_equals_rejected")
        if chosen.count("\\boxed{") != 1:
            bad.append(f"{row_id}:chosen_box_count")
        for flag in (
            "gate_rows_used_for_training",
            "weak_gate_rows_used_for_training",
            "full_gate_rows_used_for_training",
        ):
            if metadata.get(flag) is not False:
                bad.append(f"{row_id}:{flag}_not_false")
        if len(bad) >= 20:
            break
    if bad:
        raise RuntimeError(f"{label} preference validation failed: " + json.dumps(bad, sort_keys=True))
    summary = {
        "rows": len(rows),
        "unique_ids": len(ids),
        "family_counts": family_counts,
        "negative_type_counts": negative_counts,
    }
    print(f"{label}_preference_validation = {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def make_example(row: dict[str, Any], completion_key: str, suffix: str) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "id": f"{row['id']}::{suffix}",
        "family": metadata.get("family") or row.get("family") or "unknown",
        "source": metadata.get("source") or "v312_preference",
        "metadata": metadata,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row[completion_key])},
        ],
    }


def tokenize_preference_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    label: str,
) -> list[dict[str, Any]]:
    chosen_examples = [make_example(row, "chosen", "chosen") for row in rows]
    rejected_examples = [make_example(row, "rejected", "rejected") for row in rows]
    chosen = base.tokenize_examples(chosen_examples, tokenizer, f"{label}_chosen")
    rejected = base.tokenize_examples(rejected_examples, tokenizer, f"{label}_rejected")
    if len(chosen) != len(rows) or len(rejected) != len(rows):
        raise RuntimeError(
            f"{label} tokenization row loss: raw={len(rows)} "
            f"chosen={len(chosen)} rejected={len(rejected)}"
        )
    pairs: list[dict[str, Any]] = []
    for raw, chosen_item, rejected_item in zip(rows, chosen, rejected):
        raw_id = str(raw["id"])
        if not str(chosen_item["id"]).startswith(raw_id) or not str(rejected_item["id"]).startswith(raw_id):
            raise RuntimeError(f"{label} tokenization id drift around {raw_id}")
        metadata = raw.get("metadata") or {}
        pairs.append(
            {
                "id": raw_id,
                "chosen": chosen_item,
                "rejected": rejected_item,
                "family": metadata.get("family") or raw.get("family") or "unknown",
                "negative_type": metadata.get("negative_type") or "unknown",
                "subcategory": metadata.get("subcategory") or "unknown",
            }
        )
    print(f"{label}_preference_tokenized_pairs = {len(pairs)}", flush=True)
    return pairs


def build_epoch_pairs(pairs: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    epoch = list(pairs)
    rng.shuffle(epoch)
    return epoch


def pad_batch(items: list[dict[str, Any]], pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in items)
    batch_input_ids: list[list[int]] = []
    batch_attention: list[list[int]] = []
    batch_loss_mask: list[list[int]] = []
    for item in items:
        ids = list(item["input_ids"])
        mask = list(item["loss_mask"])
        pad_len = max_len - len(ids)
        batch_input_ids.append(ids + [pad_token_id] * pad_len)
        batch_attention.append([1] * len(ids) + [0] * pad_len)
        batch_loss_mask.append(mask + [0] * pad_len)
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_attention, dtype=torch.long, device=device),
        "loss_mask": torch.tensor(batch_loss_mask, dtype=torch.float32, device=device),
    }


def sequence_nll(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    batch = pad_batch(items, pad_token_id, device)
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = batch["input_ids"][:, 1:].contiguous()
    loss_mask = batch["loss_mask"][:, 1:].contiguous()
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view(labels.shape)
    token_loss = token_loss * loss_mask
    token_counts = loss_mask.sum(dim=1).clamp_min(1.0)
    sum_nll = token_loss.sum(dim=1)
    mean_nll = sum_nll / token_counts
    return mean_nll, sum_nll, token_counts


@torch.no_grad()
def evaluate_preferences(
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    pad_token_id: int,
    max_examples: int,
) -> dict[str, Any]:
    model.eval()
    sample = pairs[: max(1, min(max_examples, len(pairs)))]
    total = 0
    correct = 0
    chosen_losses: list[float] = []
    rejected_losses: list[float] = []
    family_totals: dict[str, int] = {}
    family_correct: dict[str, int] = {}
    micro = max(1, base.MICRO_BATCH_SIZE)
    for start in range(0, len(sample), micro):
        batch_pairs = sample[start : start + micro]
        chosen_items = [item["chosen"] for item in batch_pairs]
        rejected_items = [item["rejected"] for item in batch_pairs]
        chosen_mean, chosen_sum, _ = sequence_nll(model, chosen_items, pad_token_id)
        rejected_mean, rejected_sum, _ = sequence_nll(model, rejected_items, pad_token_id)
        chosen_score = -chosen_mean if PAIR_SCORE_MODE == "mean_nll" else -chosen_sum
        rejected_score = -rejected_mean if PAIR_SCORE_MODE == "mean_nll" else -rejected_sum
        wins = (chosen_score > rejected_score).detach().cpu().tolist()
        for pair, win, c_loss, r_loss in zip(
            batch_pairs,
            wins,
            chosen_mean.detach().cpu().tolist(),
            rejected_mean.detach().cpu().tolist(),
        ):
            total += 1
            correct += int(bool(win))
            family = str(pair.get("family", "unknown"))
            family_totals[family] = family_totals.get(family, 0) + 1
            family_correct[family] = family_correct.get(family, 0) + int(bool(win))
            chosen_losses.append(float(c_loss))
            rejected_losses.append(float(r_loss))
    model.train()
    return {
        "rows": total,
        "preference_accuracy": correct / max(1, total),
        "preference_correct": correct,
        "chosen_mean_nll": sum(chosen_losses) / max(1, len(chosen_losses)),
        "rejected_mean_nll": sum(rejected_losses) / max(1, len(rejected_losses)),
        "family_totals": family_totals,
        "family_correct": family_correct,
        "family_accuracy": {
            key: family_correct.get(key, 0) / max(1, value)
            for key, value in sorted(family_totals.items())
        },
    }


def save_training_manifest(
    final_dir: Path,
    train_path: Path,
    val_path: Path,
    train_summary: dict[str, Any],
    val_summary: dict[str, Any],
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    final_step: int,
    best_eval_accuracy: float,
    final_eval: dict[str, Any],
    elapsed_s: float,
) -> None:
    manifest = {
        "schema_version": "kg1_v315_preference_training_manifest_v1",
        "run_id": base.RUN_ID,
        "data": {
            "data_repo": base.DATA_REPO,
            "preference_train_file": PREF_TRAIN_FILE,
            "preference_val_file": PREF_VAL_FILE,
            "preference_train_sha256": base.file_sha256(train_path),
            "preference_val_sha256": base.file_sha256(val_path),
            "train_summary": train_summary,
            "validation_summary": val_summary,
            "tokenized_train_pairs": len(train_pairs),
            "tokenized_validation_pairs": len(val_pairs),
        },
        "preference_objective": {
            "beta": PREF_BETA,
            "margin": PREF_MARGIN,
            "preference_loss_weight": PREF_LOSS_WEIGHT,
            "chosen_ce_weight": CHOSEN_CE_WEIGHT,
            "rejected_ce_weight": REJECTED_CE_WEIGHT,
            "pair_score_mode": PAIR_SCORE_MODE,
        },
        "lora": {
            "r": base.LORA_R,
            "alpha": base.LORA_ALPHA,
            "dropout": base.LORA_DROPOUT,
            "target_modules": base.LORA_TARGET_MODULES,
            "target_parameters": base.LORA_TARGET_PARAMETERS,
            "init_adapter_repo": base.INIT_ADAPTER_REPO,
            "init_adapter_subfolder": base.INIT_ADAPTER_SUBFOLDER,
            "trainable_lora_modules": base.TRAINABLE_LORA_MODULES,
        },
        "training": {
            "max_steps": base.MAX_STEPS,
            "final_step": final_step,
            "batch_size": base.BATCH_SIZE,
            "micro_batch_size": base.MICRO_BATCH_SIZE,
            "gradient_accumulation": base.GRADIENT_ACCUMULATION,
            "learning_rate": base.LEARNING_RATE,
            "final_learning_rate": base.FINAL_LEARNING_RATE,
            "best_eval_preference_accuracy": best_eval_accuracy,
            "final_eval": final_eval,
            "elapsed_s": elapsed_s,
        },
        "runtime": base.cuda_runtime_report(),
        "decision": {
            "full_eval_allowed_by_this_manifest": False,
            "next_gate": (
                "Run weak V221 evaluation and promote only if family ACC beats "
                "V290 checkpoint-6 without bit regression."
            ),
        },
    }
    path = final_dir / "v315_preference_training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Preference training manifest saved: {path}", flush=True)


def train() -> None:
    if not PREF_TRAIN_FILE or not PREF_VAL_FILE:
        raise RuntimeError("PREF_TRAIN_FILE and PREF_VAL_FILE are required")
    started = time.time()
    rng = random.Random(base.SEED)
    base.apply_runtime_performance_settings()
    if env_bool("KG1_REQUIRE_MAMBA_IMPORTS", True):
        base.verify_model_runtime_dependencies()
    else:
        print(
            "KG1_REQUIRE_MAMBA_IMPORTS=0; skipping V90 Mamba/causal-conv1d runtime import check "
            "for V315 single-policy preference trainer.",
            flush=True,
        )
    print("=== V315 PREFERENCE TRAIN START ===", flush=True)
    print(f"run_id={base.RUN_ID}", flush=True)
    print(f"preference_train_file={PREF_TRAIN_FILE}", flush=True)
    print(f"preference_val_file={PREF_VAL_FILE}", flush=True)
    print(f"objective beta={PREF_BETA} margin={PREF_MARGIN} pref_weight={PREF_LOSS_WEIGHT}", flush=True)

    train_path = base.resolve_data_file(PREF_TRAIN_FILE)
    val_path = base.resolve_data_file(PREF_VAL_FILE)
    if EXPECTED_PREF_TRAIN_SHA256:
        base.assert_file_sha256(train_path, EXPECTED_PREF_TRAIN_SHA256, "preference train dataset")
    if EXPECTED_PREF_VAL_SHA256:
        base.assert_file_sha256(val_path, EXPECTED_PREF_VAL_SHA256, "preference validation dataset")
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    train_summary = validate_preference_rows(train_rows, "train", MIN_PREF_TRAIN_EXAMPLES)
    val_summary = validate_preference_rows(val_rows, "validation", MIN_PREF_VAL_EXAMPLES)

    tokenizer = base.AutoTokenizer.from_pretrained(
        base.MODEL_NAME,
        revision=base.MODEL_REVISION or None,
        trust_remote_code=True,
        token=base.HF_TOKEN or None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)
    train_pairs = tokenize_preference_rows(train_rows, tokenizer, "train")
    val_pairs = tokenize_preference_rows(val_rows, tokenizer, "validation")
    if len(train_pairs) < base.MIN_TOKENIZED_TRAIN_EXAMPLES:
        raise RuntimeError("tokenized train preference pairs below floor")
    if len(val_pairs) < base.MIN_TOKENIZED_VAL_EXAMPLES:
        raise RuntimeError("tokenized validation preference pairs below floor")
    if base.TOKENIZE_ONLY_DRY_RUN:
        report_path = Path(base.OUTPUT_DIR) / "v315_preference_tokenize_only_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "train_summary": train_summary,
            "validation_summary": val_summary,
            "tokenized_train_pairs": len(train_pairs),
            "tokenized_validation_pairs": len(val_pairs),
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"TOKENIZE_ONLY_DRY_RUN=1; wrote {report_path}", flush=True)
        base.upload_dry_run_report(report_path)
        return

    model_device_map = base.parse_model_device_map(base.MODEL_DEVICE_MAP)
    print(
        f"Loading model {base.MODEL_NAME} in BF16 with device_map={model_device_map} "
        f"attn_implementation={base.ATTN_IMPLEMENTATION or 'transformers-default'}...",
        flush=True,
    )
    model_kwargs = {
        "pretrained_model_name_or_path": base.MODEL_NAME,
        "revision": base.MODEL_REVISION or None,
        "dtype": torch.bfloat16,
        "device_map": model_device_map,
        "trust_remote_code": True,
        "token": base.HF_TOKEN or None,
    }
    if base.ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = base.ATTN_IMPLEMENTATION
    model = base.AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    post_load_device = base.model_post_load_device(base.MODEL_DEVICE_MAP)
    if post_load_device:
        model.to(post_load_device)

    print("Applying/loading trainable LoRA adapter for preference training...", flush=True)
    model = base.load_trainable_adapter_or_create(model)
    lora_filter_report = base.apply_trainable_lora_module_filter(model)
    print("lora_filter_report =", json.dumps(lora_filter_report, indent=2, sort_keys=True), flush=True)
    model.enable_input_require_grads()
    if base.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.print_trainable_parameters()
    trainable_report = base.trainable_parameter_report(model)
    print("trainable_report =", json.dumps(trainable_report, indent=2, sort_keys=True), flush=True)
    if float(trainable_report["ratio"]) > base.MAX_TRAINABLE_PARAM_RATIO:
        raise RuntimeError("trainable parameter ratio exceeds guard")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=base.LEARNING_RATE,
        betas=(base.ADAM_BETA1, base.ADAM_BETA2),
        eps=base.ADAM_EPS,
        weight_decay=base.WEIGHT_DECAY,
    )
    model.train()
    best_eval_accuracy = -1.0
    global_step = 0
    epoch_pairs: list[dict[str, Any]] = []
    epoch_index = 0
    grad_accum = max(1, base.GRADIENT_ACCUMULATION)
    micro = max(1, base.MICRO_BATCH_SIZE)

    baseline_eval = evaluate_preferences(model, val_pairs, pad_token_id, base.EVAL_MAX_EXAMPLES)
    print("baseline_preference_eval =", json.dumps(baseline_eval, sort_keys=True), flush=True)
    best_eval_accuracy = max(best_eval_accuracy, float(baseline_eval["preference_accuracy"]))

    while global_step < base.MAX_STEPS:
        if len(epoch_pairs) < micro * grad_accum:
            epoch_pairs.extend(build_epoch_pairs(train_pairs, rng))
            epoch_index += 1
            print(f"preference_epoch_buffer_refill epoch={epoch_index} buffer={len(epoch_pairs)}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        step_loss_value = 0.0
        step_pref_value = 0.0
        step_chosen_value = 0.0
        step_rejected_value = 0.0
        for _ in range(grad_accum):
            batch_pairs = [epoch_pairs.pop() for _ in range(micro)]
            chosen_items = [item["chosen"] for item in batch_pairs]
            rejected_items = [item["rejected"] for item in batch_pairs]
            chosen_mean, chosen_sum, _ = sequence_nll(model, chosen_items, pad_token_id)
            rejected_mean, rejected_sum, _ = sequence_nll(model, rejected_items, pad_token_id)
            chosen_score = -chosen_mean if PAIR_SCORE_MODE == "mean_nll" else -chosen_sum
            rejected_score = -rejected_mean if PAIR_SCORE_MODE == "mean_nll" else -rejected_sum
            pref_logits = PREF_BETA * (chosen_score - rejected_score - PREF_MARGIN)
            preference_loss = F.softplus(-pref_logits).mean()
            chosen_ce = chosen_mean.mean()
            rejected_ce = rejected_mean.mean()
            loss = (
                PREF_LOSS_WEIGHT * preference_loss
                + CHOSEN_CE_WEIGHT * chosen_ce
                - REJECTED_CE_WEIGHT * rejected_ce
            ) / grad_accum
            loss.backward()
            step_loss_value += float(loss.detach().cpu()) * grad_accum
            step_pref_value += float(preference_loss.detach().cpu())
            step_chosen_value += float(chosen_ce.detach().cpu())
            step_rejected_value += float(rejected_ce.detach().cpu())
            del chosen_mean, chosen_sum, rejected_mean, rejected_sum, loss, preference_loss
        global_step += 1
        lr = base.get_lr(global_step, base.MAX_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if base.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, base.GRAD_CLIP_NORM)
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            "preference_step "
            f"step={global_step}/{base.MAX_STEPS} lr={lr:.3e} "
            f"loss={step_loss_value / grad_accum:.4f} "
            f"pref={step_pref_value / grad_accum:.4f} "
            f"chosen_nll={step_chosen_value / grad_accum:.4f} "
            f"rejected_nll={step_rejected_value / grad_accum:.4f} "
            f"{base.cuda_memory_line()}",
            flush=True,
        )
        if global_step % base.EVAL_EVERY_STEPS == 0 or global_step == base.MAX_STEPS:
            eval_report = evaluate_preferences(model, val_pairs, pad_token_id, base.EVAL_MAX_EXAMPLES)
            best_eval_accuracy = max(best_eval_accuracy, float(eval_report["preference_accuracy"]))
            print(f"preference_eval_step_{global_step} = {json.dumps(eval_report, sort_keys=True)}", flush=True)
        if global_step % base.SAVE_EVERY_STEPS == 0 or global_step == base.MAX_STEPS:
            checkpoint_dir = Path(base.OUTPUT_DIR) / f"checkpoint-{global_step}"
            model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            print(f"Preference checkpoint saved: {checkpoint_dir}", flush=True)
            base.upload_checkpoint_during_training(checkpoint_dir)
        if base.ABORT_MAX_RESERVED_GIB > 0 and base.cuda_reserved_gib() > base.ABORT_MAX_RESERVED_GIB:
            raise RuntimeError("abort_cuda_reserved_guard_exceeded")

    final_eval = evaluate_preferences(model, val_pairs, pad_token_id, base.EVAL_MAX_EXAMPLES)
    final_dir = Path(base.OUTPUT_DIR) / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    elapsed = time.time() - started
    save_training_manifest(
        final_dir=final_dir,
        train_path=train_path,
        val_path=val_path,
        train_summary=train_summary,
        val_summary=val_summary,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        final_step=global_step,
        best_eval_accuracy=best_eval_accuracy,
        final_eval=final_eval,
        elapsed_s=elapsed,
    )
    print("final_preference_eval =", json.dumps(final_eval, sort_keys=True), flush=True)
    print(f"Final preference adapter saved: {final_dir}", flush=True)
    base.upload_outputs(final_dir)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("=== V315 PREFERENCE TRAIN END ===", flush=True)


if __name__ == "__main__":
    train()
