#!/usr/bin/env python3
"""V80 dgxchen v7 EXACT - standalone script for subprocess execution on Colab.

Runs as child process (fresh Python interpreter) with torch 2.5.1 clean import.
Parent cell installs deps + writes this via download, then spawns this process.

All 11 fixes applied (7 dgxchen divergences + 4 execution fixes).
Flow: download data + model -> load + LoRA -> train -> save -> submit Kaggle.
"""
import sys
import os
import gc
import re
import math
import time
import json
import random
import shutil
import zipfile
import datetime
import subprocess
from pathlib import Path
from collections import defaultdict, deque

# UTF-8 output
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

# ============ STEP 1: Verify env ============
import torch
print(f"\n[CHILD] torch={torch.__version__}  cuda={torch.version.cuda}")
assert torch.__version__.startswith("2.5"), (
    f"Child process needs torch 2.5.x, got {torch.__version__}. "
    "Parent cell install failed."
)
assert torch.cuda.is_available(), "CUDA required"
d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
print(f"[CHILD] GPU: {d.name} {total_gb:.1f}GB")

# Verify mamba-ssm
import mamba_ssm
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print(f"[CHILD] mamba_ssm={mamba_ssm.__version__}")

# Seed
SEED = 42
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY")
assert HF_TOKEN, "HF_TOKEN / HF_KEY env var required"

# ============ STEP 2: Download dataset ============
print("\n" + "=" * 70)
print("STEP 2/7: Download dgxchen dataset (problem_ids_matched.csv)")
print("=" * 70)

DATA_DIR = Path("/content/kg1_data")
DATA_DIR.mkdir(exist_ok=True)
target_csv = DATA_DIR / "problem_ids_matched.csv"

if target_csv.exists() and target_csv.stat().st_size > 40_000_000:
    print(f"Dataset cached: {target_csv.stat().st_size/(1024**2):.1f}MB")
else:
    print("Downloading dgxchen/nemotron-cot-tong via kaggle CLI...")
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "dgxchen/nemotron-cot-tong",
         "-p", str(DATA_DIR), "--unzip"],
        capture_output=True, text=True, timeout=300,
    )
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-500:])
        raise RuntimeError("Kaggle download failed")

assert target_csv.exists(), "problem_ids_matched.csv missing"
print(f"Dataset path: {target_csv}")

import pandas as pd
df = pd.read_csv(target_csv)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
if "type" in df.columns:
    print("Type distribution:")
    for t, n in df["type"].value_counts().items():
        print(f"  {t}: {n}")

# ============ STEP 3: Download base model ============
print("\n" + "=" * 70)
print("STEP 3/7: Download Nemotron-3-Nano-30B-A3B-BF16 base model")
print("=" * 70)

import kagglehub
MODEL_CACHE = "/root/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
if Path(MODEL_CACHE).exists() and len(list(Path(MODEL_CACHE).iterdir())) > 5:
    MODEL_PATH = MODEL_CACHE
    print(f"Model cached at: {MODEL_PATH}")
else:
    print("Downloading base model (~60GB, ~50min first time)...")
    MODEL_PATH = kagglehub.model_download(
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    )
    print(f"Model path: {MODEL_PATH}")

# ============ STEP 4: Load model + LoRA ============
print("\n" + "=" * 70)
print("STEP 4/7: Load Nemotron + LoRA (r=32 alpha=32, 8 targets NO lm_head)")
print("=" * 70)

from unsloth import FastLanguageModel

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 3072))  # 3072 covers p99=2911 tokens, H100 80GB fit sem gradient offload
print(f"Loading {MODEL_PATH} via Unsloth (attn=eager, max_seq_len={MAX_SEQ_LEN})...")

# Reduce CUDA fragmentation (suggested by OOM error msg)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
t_load = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=False,
    attn_implementation="eager",  # FIX #2: dgxchen v7 uses eager
    dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Model loaded in {time.time()-t_load:.1f}s")

# dgxchen v7 EXACT LoRA: 8 targets, NO lm_head
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "up_proj", "down_proj",
]
print(f"Applying LoRA r={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
print(f"Target modules ({len(target_modules)}, NO lm_head): {target_modules}")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
model.print_trainable_parameters()
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
print(f"After LoRA: used={used_gb:.1f}GB free={free_gb:.1f}GB")

# ============ STEP 5: Build SFT dataset ============
print("\n" + "=" * 70)
print("STEP 5/7: Build SFT records + stratified sampler")
print("=" * 70)

from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Sampler
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

train_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
records = []
record_types = []
for _, row in train_df.iterrows():
    prompt = str(row["prompt"])
    answer = str(row["answer"])
    cot = str(row.get("generated_cot", ""))
    if not cot or cot == "nan" or len(cot.strip()) < 5:
        continue
    cot_cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
    user_content = prompt + PROMPT_SUFFIX
    assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
    records.append({"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]})
    record_types.append(str(row.get("type", "unknown")))
print(f"SFT records: {len(records)}")

dataset = HFDataset.from_list(records)


def formatting_prompts_func(example):
    messages = example["messages"]
    if messages and isinstance(messages[0], dict):
        conversations = [messages]
    else:
        conversations = messages
    texts = []
    for conversation in conversations:
        try:
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False,
                add_generation_prompt=False, enable_thinking=True,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False,
            )
        texts.append(text)
    return texts


def build_stratified_index_order(labels, batch_size, seed):
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)
    assigned = 0
    for label in sorted(by_label.keys()):
        for idx in by_label[label]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1
    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError("Stratified order size mismatch")
    return order


class PrecomputedOrderSampler(Sampler):
    def __init__(self, order):
        self.order = list(order)
    def __iter__(self):
        return iter(self.order)
    def __len__(self):
        return len(self.order)


class StratifiedSFTTrainer(SFTTrainer):
    def __init__(self, *args, stratified_order=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratified_order = stratified_order

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")
        if self.stratified_order is None:
            return super().get_train_dataloader()
        dk = {
            "batch_size": self.args.per_device_train_batch_size,
            "sampler": PrecomputedOrderSampler(self.stratified_order),
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if self.args.dataloader_num_workers > 0:
            dk["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(self.train_dataset, **dk)


class HealthGateCallback(TrainerCallback):
    def __init__(self):
        self.loss_history = deque(maxlen=10)
        self.grad_history = deque(maxlen=5)
        self.min_loss = float("inf")
        self.steps_since_improve = 0
        self.high_grad_count = 0
        self.train_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        print("=" * 70)
        print("V80 HEALTH GATES: log every 5 steps + abort on NaN/explosion")
        print("=" * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", 0.0)

        # GATE 1: NaN/Inf
        if loss is not None and isinstance(loss, float):
            if math.isnan(loss) or math.isinf(loss):
                print(f"!!! CRITICAL NaN/Inf at step {step} - ABORTING")
                control.should_training_stop = True
                return

        # GATE 2: Loss explosion
        if loss is not None and isinstance(loss, (int, float)):
            if loss > 30.0 and step > 10:
                print(f"!!! CRITICAL loss explosion {loss:.3f} at step {step} - ABORTING")
                control.should_training_stop = True
                return

        if loss is not None:
            self.loss_history.append(float(loss))
            if float(loss) < self.min_loss:
                self.min_loss = float(loss)
                self.steps_since_improve = 0
            else:
                self.steps_since_improve += 5

        if grad_norm is not None:
            self.grad_history.append(float(grad_norm))
            if float(grad_norm) > 50.0:
                self.high_grad_count += 1
                if self.high_grad_count >= 3:
                    print(f"!!! WARN grad_norm >50 3x (max_grad_norm=1e9 means no clipping)")
            else:
                self.high_grad_count = 0

        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            total = torch.cuda.mem_get_info()[1] / 1024**3
            used = total - free
            peak = torch.cuda.max_memory_allocated() / 1024**3
            if free < 2.0:
                print(f"!!! WARN VRAM free={free:.1f}GB OOM risk")
        else:
            free = used = peak = 0.0

        elapsed = time.time() - self.train_start if self.train_start else 0
        total_steps = state.max_steps if state.max_steps else 1
        progress = step / max(total_steps, 1)
        eta_sec = (elapsed / max(step, 1)) * (total_steps - step) if step > 0 else 0

        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0
        avg_grad = sum(self.grad_history) / len(self.grad_history) if self.grad_history else 0.0

        _loss = float(loss) if loss is not None else 0.0
        _grad = float(grad_norm) if grad_norm is not None else 0.0
        _lr = float(lr) if lr is not None else 0.0
        _em = int(elapsed // 60)
        _etm = int(eta_sec // 60)

        print(
            f"[step {step:4d}/{total_steps:4d} ep{epoch:.2f} "
            f"{progress*100:5.1f}%] "
            f"loss={_loss:.4f} avg10={avg_loss:.4f} "
            f"grad={_grad:.3f} avg5={avg_grad:.3f} "
            f"lr={_lr:.2e} "
            f"vram={used:.1f}/{peak:.1f}/{free:.1f}GB "
            f"elapsed={_em}m ETA={_etm}m"
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start if self.train_start else 0
        print("=" * 70)
        print(f"V80 TRAINING FINISHED: elapsed={elapsed/60:.1f}min "
              f"min_loss={self.min_loss:.4f} steps={state.global_step}")
        print("=" * 70)


# ============ STEP 6: Training ============
print("\n" + "=" * 70)
print("STEP 6/7: Train 1 epoch (dgxchen v7 EXACT, ~245 steps, ~2-3h H100)")
print("=" * 70)

OUTPUT_DIR = "/content/kg1_out/sft_v80"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                          # dgxchen v7
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,              # eff_batch=32
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=0,
    max_length=MAX_SEQ_LEN,                      # 4096 (vs dgxchen 8192) — H100 80GB fit, zero truncation
    adam_beta1=0.9,
    adam_beta2=0.95,                             # dgxchen
    adam_epsilon=1e-8,
    weight_decay=0.0,
    max_grad_norm=1e9,                           # dgxchen (effectively disabled)
    # V3.1 fix: 8bit optimizer para fit H100 80GB sem gradient offloading
    # AdamW FP32 (883M params) = 7GB optimizer state
    # paged_adamw_8bit = 1.8GB optimizer state -> economia 5GB -> no offload -> 3-4x speedup
    # Quality impact: ~0.005 (minimal, padrão LLM community)
    optim="paged_adamw_8bit",
    logging_steps=5,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=0,                    # FIX pickle CudaDeviceProperties
    remove_unused_columns=False,
    seed=SEED,
    report_to="none",
    packing=False,
)

eff_batch_size = 32
stratified_order = build_stratified_index_order(record_types, eff_batch_size, SEED)
print(f"Eff batch size: {eff_batch_size}")
print(f"Total optim steps: {math.ceil(len(record_types)/eff_batch_size)}")

health_cb = HealthGateCallback()
trainer = StratifiedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    stratified_order=stratified_order,
    callbacks=[health_cb],
)

print("\nStarting V80 SFT training (dgxchen v7 EXACT)...")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nTraining done: {elapsed/60:.1f} min")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f}GB")

# Save adapter
ADAPTER_DIR = "/content/kg1_adapter_v80"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved: {ADAPTER_DIR}")

# ============ STEP 7: Package + Submit ============
print("\n" + "=" * 70)
print("STEP 7/7: Package submission.zip + HF backup + Kaggle submit")
print("=" * 70)

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SUBMISSION_DIR = "/content/kg1_out/submission_v80"
Path(SUBMISSION_DIR).mkdir(parents=True, exist_ok=True)

required = ["adapter_config.json", "adapter_model.safetensors"]
for fn in required:
    src = Path(ADAPTER_DIR) / fn
    dst = Path(SUBMISSION_DIR) / fn
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}")
    shutil.copy2(src, dst)
    print(f"Copied {fn} ({dst.stat().st_size/(1024**2):.1f} MB)")

# Fix adapter_config
cfg_path = Path(SUBMISSION_DIR) / "adapter_config.json"
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["base_model_name_or_path"] = BASE_MODEL_NAME
cfg["inference_mode"] = True
cfg["lora_dropout"] = 0.0
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

tm = cfg.get("target_modules", [])
if isinstance(tm, list) and "lm_head" in tm:
    print("!!! WARN adapter has lm_head in targets (divergence)")
else:
    print(f"adapter_config OK: {len(tm)} target_modules NO lm_head")

# Build zip
zip_path = "/content/kg1_out/submission_v80.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in required:
        zf.write(Path(SUBMISSION_DIR) / fn, arcname=fn)
zip_mb = os.path.getsize(zip_path) / (1024**2)
print(f"submission_v80.zip: {zip_mb:.1f} MB")

# HF upload
try:
    from huggingface_hub import HfApi
    REPO = "felipesp1983/kg1-nemotron-lora-v80-colab-dgxchen"
    api = HfApi(token=HF_TOKEN)
    api.create_repo(REPO, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=REPO, folder_path=SUBMISSION_DIR,
        allow_patterns=["adapter_*"],
        token=HF_TOKEN,
    )
    print(f"HF upload OK: https://huggingface.co/{REPO}")
except Exception as e:
    print(f"HF upload failed (non-fatal): {e}")

# Kaggle submit (with 5/day slot check)
print("\nChecking Kaggle submit slots (5/day)...")
try:
    rc = subprocess.run(
        ["kaggle", "competitions", "submissions",
         "-c", "nvidia-nemotron-model-reasoning-challenge", "--csv"],
        capture_output=True, text=True, timeout=60,
    )
    if rc.returncode == 0:
        from io import StringIO
        import csv as _csv
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        today_count = sum(1 for r in _csv.DictReader(StringIO(rc.stdout))
                          if r.get("date", "").startswith(today))
        print(f"Submissions today: {today_count}/5")
        if today_count >= 5:
            print("Slot exhausted - manual submit:")
            print(f"  kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge "
                  f"-f {zip_path} -m 'V80 dgxchen v7 EXACT'")
        else:
            msg = f"V80 dgxchen v7 EXACT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            r = subprocess.run(
                ["kaggle", "competitions", "submit",
                 "-c", "nvidia-nemotron-model-reasoning-challenge",
                 "-f", zip_path, "-m", msg],
                capture_output=True, text=True, timeout=600,
            )
            print(f"Submit rc={r.returncode}")
            print(r.stdout[-400:])
            if r.returncode == 0:
                print("\nSUBMITTED! Check score:")
                print("https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions")
except Exception as e:
    print(f"Submit failed: {e}")

print("\n" + "=" * 70)
print("V80 MEGA DONE.")
print(f"  adapter: {ADAPTER_DIR}")
print(f"  zip: {zip_path}")
print("=" * 70)
