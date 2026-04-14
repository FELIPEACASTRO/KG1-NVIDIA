#!/usr/bin/env python3
"""
HF Jobs training script V71 - applies huikang's Tinker recipe FIXES on top of v70.

KEY DIFFERENCES vs v70 (same dataset, same loss masking, same architecture):
  1. LEARNING_RATE: 2e-4 → 2e-5 (10x lower, matches huikang's tinker default)
  2. LR schedule: linear-decay-to-0 → LinearDecay 2e-5 → 1e-5 (matches huikang
     LinearDecayLRSchedule in lr_schedule.py)
  3. MAX_LENGTH: 2048 → 4096 (covers tail of equation/cipher CoTs)
  4. NUM_EPOCHS: 1 → 2 (huikang trains longer)
  5. Output repo: kg1-nemotron-lora-v71-tinker

Everything else stays IDENTICAL to v70 (loss masking, BF16, LoRA r=32 alpha=32
all-linear, Adam beta2=0.95, weight_decay=0, no grad clip, batch=64).

Usage (HF Jobs):
  python hf_job_train_v71.py

Hardware: A100 80GB (single GPU)
Expected time: ~2-3 hours (with 4096 + 2 epochs)
Expected cost: ~$10-15
Expected delta vs v70: +0.01 to +0.03 (huikang's actual recipe ends up at 0.85)
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG V71 - huikang Tinker recipe (LR + max_length + epochs)
# ============================================================
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATA_FILE = "data/sft_v70_huikang_full.jsonl"

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# *** V71 CHANGE 1: MAX_LENGTH 2048 → 8192 ***
# Real distribution analysis (16365 examples, no trunc):
#   bit_manipulation: p99=7703, max=7987   → needs 8192
#   cipher: p99=6502, max=7547             → needs 8192
#   equation_numeric_*: p99=6655-6829      → needs 8192
#   spelling: p99=4989                     → needs >5120
#   gravity: p99=5626                      → needs >6144
# Setting 8192 covers 100% of dataset with 0 truncation.
# A100 80GB tested: BF16 30B + 884M LoRA + 8bit Adam + grad_ckpt fits at batch=1.
# If OOM mid-training: fall back to 4096 (will truncate ~30%).
MAX_LENGTH = 8192
BATCH_SIZE = 64          # effective batch (same as v70)
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = BATCH_SIZE // MICRO_BATCH_SIZE  # 64

# *** V71 CHANGE 2: LR 2e-4 → 2e-5 (matches huikang LRSchedule default) ***
LEARNING_RATE = 2e-5
# *** V71 CHANGE 3: LinearDecay LR 2e-5 → 1e-5 (matches huikang LinearDecayLRSchedule) ***
FINAL_LEARNING_RATE = 1e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1e9     # effectively no clipping (same as v70)

# *** V71 CHANGE 4: NUM_EPOCHS 1 → 2 (huikang trains longer) ***
NUM_EPOCHS = 2
SEED = 42

OUTPUT_DIR = "/tmp/kg1_v71_output"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v71-tinker"

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def setup_causal_conv1d_stub():
    """Inject stub if causal_conv1d not available."""
    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        import importlib.machinery
        import types
        stub = types.ModuleType("causal_conv1d")
        stub.causal_conv1d_fn = None
        stub.causal_conv1d_update = None
        stub.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", loader=None)
        sys.modules["causal_conv1d"] = stub
        print("Injected causal_conv1d stub")


def load_and_tokenize_data(tokenizer):
    """Load dataset and tokenize with loss masking (only completion tokens).

    Identical to v70 except MAX_LENGTH bound.
    """
    print(f"Loading dataset from {DATA_REPO}/{DATA_FILE}...")
    data_path = hf_hub_download(
        repo_id=DATA_REPO,
        filename=DATA_FILE,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    tokenized = []
    for ex in examples:
        msgs = ex.get("messages", [])
        if not msgs:
            continue

        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(full_ids) > MAX_LENGTH:
            full_ids = full_ids[:MAX_LENGTH]

        prompt_len = min(len(prompt_ids), len(full_ids))
        loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

        if sum(loss_mask) == 0:
            continue

        tokenized.append({
            "input_ids": full_ids,
            "loss_mask": loss_mask,
            "category": ex.get("family", ex.get("category", "unknown")),
        })

    print(f"Tokenized: {len(tokenized)} examples")
    total_tokens = sum(len(t["input_ids"]) for t in tokenized)
    unmasked_tokens = sum(sum(t["loss_mask"]) for t in tokenized)
    print(f"Total tokens: {total_tokens:,}, Unmasked (trained): {unmasked_tokens:,}")

    # Per-category length distribution (helpful for MAX_LENGTH tuning)
    cat_lens = {}
    for t in tokenized:
        c = t["category"]
        cat_lens.setdefault(c, []).append(len(t["input_ids"]))
    print("\nPer-category token length stats (after truncation):")
    for c, lens in sorted(cat_lens.items()):
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        p50 = lens_sorted[n // 2]
        p90 = lens_sorted[int(n * 0.9)]
        p99 = lens_sorted[int(n * 0.99)] if n >= 100 else lens_sorted[-1]
        truncated = sum(1 for x in lens if x == MAX_LENGTH)
        print(
            f"  {c}: n={n} p50={p50} p90={p90} p99={p99} "
            f"truncated={truncated}/{n}"
        )

    return tokenized


def masked_cross_entropy_loss(logits, input_ids, loss_mask):
    """Identical to v70 - cross-entropy ONLY on unmasked (completion) tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous().float()

    B, T, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1)

    per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")

    masked_loss = per_token_loss * flat_mask
    num_unmasked = flat_mask.sum()

    if num_unmasked == 0:
        return torch.tensor(0.0, device=logits.device)

    return masked_loss.sum() / num_unmasked


def get_lr(global_step, total_steps):
    """huikang LinearDecayLRSchedule: linear decay LR_INITIAL → LR_FINAL.

    Note: huikang uses epoch-based decay; here we use step-based for finer control.
    Equivalent for single-epoch runs; minor diff for multi-epoch.
    """
    if total_steps <= 1:
        return LEARNING_RATE
    progress = min(1.0, max(0.0, global_step / max(1, total_steps - 1)))
    return FINAL_LEARNING_RATE + (LEARNING_RATE - FINAL_LEARNING_RATE) * (1.0 - progress)


def train():
    """Main training loop - V71 with Tinker recipe fixes."""
    print("=" * 60)
    print("KG1 v71 - Huikang Tinker Recipe")
    print("=" * 60)
    print(f"  LR: {LEARNING_RATE} -> {FINAL_LEARNING_RATE} (LinearDecay)")
    print(f"  MAX_LENGTH: {MAX_LENGTH}")
    print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  Output: {OUTPUT_REPO}")
    print()

    setup_causal_conv1d_stub()
    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_and_tokenize_data(tokenizer)

    print(f"\nLoading model {MODEL_NAME} in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="eager",
    )

    print("Applying LoRA (r=32, alpha=32, all-linear)...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdam8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
        print("Optimizer: PagedAdam8bit")
    except Exception as e:
        print(f"bnb failed ({e}), fallback to torch Adam")
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )

    total_steps = math.ceil(len(data) / BATCH_SIZE) * NUM_EPOCHS
    print(f"\nTraining: {len(data)} examples, batch={BATCH_SIZE}, steps={total_steps}")
    print(f"LR schedule: {LEARNING_RATE} -> {FINAL_LEARNING_RATE} (LinearDecay)")

    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch} ---")
        random.shuffle(data)

        for i in range(0, len(data), MICRO_BATCH_SIZE):
            batch = data[i:i + MICRO_BATCH_SIZE]
            if not batch:
                continue

            max_len = max(len(ex["input_ids"]) for ex in batch)
            input_ids_batch = []
            loss_mask_batch = []
            for ex in batch:
                pad_len = max_len - len(ex["input_ids"])
                input_ids_batch.append(ex["input_ids"] + [tokenizer.pad_token_id] * pad_len)
                loss_mask_batch.append(ex["loss_mask"] + [0] * pad_len)

            input_ids = torch.tensor(input_ids_batch, dtype=torch.long, device="cuda")
            loss_mask = torch.tensor(loss_mask_batch, dtype=torch.long, device="cuda")

            outputs = model(input_ids=input_ids)
            loss = masked_cross_entropy_loss(outputs.logits, input_ids, loss_mask)
            loss = loss / GRADIENT_ACCUMULATION

            loss.backward()

            accum_loss += loss.item() * GRADIENT_ACCUMULATION
            accum_count += 1

            if accum_count % GRADIENT_ACCUMULATION == 0:
                lr = get_lr(global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if GRAD_CLIP_NORM < 1e8:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / GRADIENT_ACCUMULATION
                elapsed = time.time() - start_time

                if global_step % 5 == 0:
                    print(
                        f"step={global_step}/{total_steps} "
                        f"lr={lr:.2e} loss={avg_loss:.4f} "
                        f"time={elapsed:.0f}s"
                    )

                if global_step > 0 and global_step % 50 == 0:
                    ckpt_dir = Path(OUTPUT_DIR) / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    print(f"Checkpoint saved: {ckpt_dir}")

                accum_loss = 0.0
                global_step += 1

    elapsed = time.time() - start_time
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Time: {elapsed / 3600:.2f}h")
    print(f"Final step: {global_step}")

    final_dir = Path(OUTPUT_DIR) / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final adapter saved: {final_dir}")

    if HF_TOKEN:
        print(f"\nUploading to {OUTPUT_REPO}...")
        api = HfApi(token=HF_TOKEN)
        api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(final_dir),
            path_in_repo="final",
            repo_id=OUTPUT_REPO,
            commit_message=f"v71 tinker recipe - step {global_step}",
            token=HF_TOKEN,
        )
        # Also upload latest checkpoint dirs
        for ck in sorted(Path(OUTPUT_DIR).glob("checkpoint-*")):
            api.upload_folder(
                folder_path=str(ck),
                path_in_repo=ck.name,
                repo_id=OUTPUT_REPO,
                commit_message=f"v71 tinker - {ck.name}",
                token=HF_TOKEN,
            )
        print(f"Upload complete: {OUTPUT_REPO}")


if __name__ == "__main__":
    train()
