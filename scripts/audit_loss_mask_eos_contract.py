#!/usr/bin/env python3
"""Audit whether assistant loss masks include the chat EOS token.

This CPU-only audit mirrors the offset-mask logic used by hf_job_train_v90.py.
It loads only the tokenizer, never the model. The goal is to prevent a silent
loss/ACC gap where the model learns the boxed answer text but not the stop token,
causing long generations despite lower training loss.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import get_token
from transformers import AutoTokenizer


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if limit and len(rows) >= limit:
                break
    return rows


def render_chat(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_loss_mask(tokenizer: Any, full_text: str, messages: list[dict[str, Any]]) -> tuple[list[int], list[float], bool]:
    assistant_text = ""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            assistant_text = str(message.get("content", ""))
            break
    if not assistant_text:
        return [], [], False
    assistant_start = full_text.rfind(assistant_text)
    if assistant_start < 0:
        return [], [], False
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if not offsets or len(offsets) != len(input_ids):
        return input_ids, [], False
    loss_mask = [1.0 if int(end) > assistant_start else 0.0 for _, end in offsets]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        for idx, token_id in enumerate(input_ids):
            if loss_mask[idx] and int(token_id) == int(eos_id):
                for after_idx in range(idx + 1, len(loss_mask)):
                    loss_mask[after_idx] = 0.0
                break
    return input_ids, loss_mask, True


def audit_split(tokenizer: Any, path: Path, limit: int, examples: int) -> dict[str, Any]:
    rows = read_jsonl(path, limit)
    eos_id = tokenizer.eos_token_id
    final_loss_eos_rows = 0
    loss_contains_eos_rows = 0
    no_loss_rows = 0
    no_offset_rows = 0
    samples: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages") if isinstance(row.get("messages"), list) else []
        full_text = render_chat(tokenizer, messages)
        input_ids, loss_mask, used_offsets = build_loss_mask(tokenizer, full_text, messages)
        if not used_offsets:
            no_offset_rows += 1
            continue
        loss_indices = [idx for idx, value in enumerate(loss_mask) if value]
        if not loss_indices:
            no_loss_rows += 1
            continue
        final_idx = loss_indices[-1]
        final_id = int(input_ids[final_idx])
        final_is_eos = final_id == eos_id
        loss_contains_eos = any(int(input_ids[idx]) == eos_id for idx in loss_indices) if eos_id is not None else False
        if final_is_eos:
            final_loss_eos_rows += 1
        if loss_contains_eos:
            loss_contains_eos_rows += 1
        if len(samples) < examples:
            start = max(0, final_idx - 4)
            final_window = input_ids[start : final_idx + 1]
            samples.append(
                {
                    "id": row.get("id", ""),
                    "family": row.get("family", ""),
                    "loss_token_count": len(loss_indices),
                    "final_loss_token_id": final_id,
                    "final_loss_token_text": tokenizer.decode([final_id]),
                    "final_loss_token_is_eos": final_is_eos,
                    "loss_contains_eos": loss_contains_eos,
                    "final_loss_window_text": tokenizer.decode(final_window),
                }
            )
    return {
        "path": str(path),
        "rows_checked": len(rows),
        "offset_mask_rows": len(rows) - no_offset_rows,
        "no_offset_rows": no_offset_rows,
        "no_loss_rows": no_loss_rows,
        "loss_contains_eos_rows": loss_contains_eos_rows,
        "final_loss_eos_rows": final_loss_eos_rows,
        "loss_contains_eos_rate": round(loss_contains_eos_rows / len(rows), 6) if rows else 0.0,
        "final_loss_eos_rate": round(final_loss_eos_rows / len(rows), 6) if rows else 0.0,
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-examples", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        token=get_token(),
        trust_remote_code=True,
    )
    report = {
        "schema_version": "kg1_loss_mask_eos_contract_v1",
        "generated_at_utc": utc_now(),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "tokenizer_info": {
            "class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        },
        "train": audit_split(tokenizer, args.train_jsonl, args.limit, args.sample_examples),
        "validation": audit_split(tokenizer, args.val_jsonl, args.limit, args.sample_examples),
    }
    blockers: list[str] = []
    for split in ("train", "validation"):
        payload = report[split]
        if payload["no_offset_rows"]:
            blockers.append(f"{split}_offset_masks_missing")
        if payload["no_loss_rows"]:
            blockers.append(f"{split}_no_loss_rows")
        if payload["loss_contains_eos_rows"] != payload["rows_checked"]:
            blockers.append(f"{split}_loss_mask_missing_eos")
        if payload["final_loss_eos_rows"] != payload["rows_checked"]:
            blockers.append(f"{split}_final_loss_token_not_eos")
    report["blockers"] = blockers
    report["ok"] = not blockers
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 1 if args.enforce and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
