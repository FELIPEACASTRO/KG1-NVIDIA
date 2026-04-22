#!/usr/bin/env python3
"""
HF Jobs training script V80 - dgxchen v7 EXACT replica on remote H100/A100.

PARITY WITH COLAB NOTEBOOK KG1_V80_DGXCHEN_EXACT.ipynb:
  Same 7 dgxchen v7 params -> 0.85 LB target:
    1. Dataset: problem_ids_matched.csv (7830 rows, 9 types)
    2. attn_implementation='eager'
    3. LoRA r=32 alpha=32 dropout=0, targets 8 items (NO lm_head):
       [q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, up_proj, down_proj]
    4. num_train_epochs=1
    5. max_grad_norm=1e9 (effectively disabled)
    6. gradient_checkpointing=True + use_reentrant=False
    7. formatting_func with try/except enable_thinking fallback

DIFF vs Colab notebook:
  - Env vars instead of Colab secrets
  - Dataset from HF Hub (felipesp1983/kg1-nemotron-training/data/dgxchen_v80/problem_ids_matched.csv)
    instead of kagglehub
  - Base model from HF Hub (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
    instead of kaggle metric path
  - Upload to felipesp1983/kg1-nemotron-lora-v80-hf-dgxchen

Required env:
  HF_TOKEN                 - HuggingFace write token
  KAGGLE_USERNAME (opt)    - for Kaggle submit step (skipped if absent)
  KAGGLE_KEY (opt)         - same

Optional env:
  MODEL_NAME, DATA_REPO, DATA_FILE, OUTPUT_REPO, OUTPUT_DIR
  MAX_LENGTH (8192), BATCH_SIZE (32 effective), MICRO_BATCH_SIZE (1)
  LEARNING_RATE (2e-4), NUM_EPOCHS (1), MAX_STEPS (-1 = 1 epoch)
  LORA_RANK (32), LORA_ALPHA (32)
  SEED (42), RUN_ID, DRY_RUN_VALIDATE_ONLY (0)
"""
from __future__ import annotations

import os
import sys
import gc
import re
import math
import time
import json
import random
import subprocess
from pathlib import Path
from collections import defaultdict, deque

# Ensure UTF-8 output
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


def env_int(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else int(v)


def env_float(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else float(v)


def env_str(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v


def env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() not in {"0", "false", "no", "off"}


# ============ CONFIG (dgxchen v7 EXACT) ============
MODEL_NAME = env_str("MODEL_NAME", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
DATA_REPO = env_str("DATA_REPO", "felipesp1983/kg1-nemotron-training")
DATA_FILE = env_str("DATA_FILE", "data/dgxchen_v80/problem_ids_matched.csv")
OUTPUT_REPO = env_str("OUTPUT_REPO", "felipesp1983/kg1-nemotron-lora-v80-hf-dgxchen")
OUTPUT_DIR = env_str("OUTPUT_DIR", "/tmp/kg1_v80_out")
ADAPTER_DIR = env_str("ADAPTER_DIR", "/tmp/kg1_v80_adapter")
RUN_ID = env_str("RUN_ID", f"v80-dgxchen-exact-{int(time.time())}")

MAX_LENGTH = env_int("MAX_LENGTH", 8192)
BATCH_SIZE = env_int("BATCH_SIZE", 32)
MICRO_BATCH_SIZE = env_int("MICRO_BATCH_SIZE", 1)
GRAD_ACCUM = max(1, BATCH_SIZE // MICRO_BATCH_SIZE)
LEARNING_RATE = env_float("LEARNING_RATE", 2e-4)
NUM_EPOCHS = env_int("NUM_EPOCHS", 1)          # dgxchen v7 = 1 epoch
MAX_STEPS = env_int("MAX_STEPS", -1)           # -1 = full 1 epoch

LORA_RANK = env_int("LORA_RANK", 32)
LORA_ALPHA = env_int("LORA_ALPHA", 32)

SEED = env_int("SEED", 42)
LOG_EVERY_STEPS = env_int("LOG_EVERY_STEPS", 5)
SAVE_EVERY_STEPS = env_int("SAVE_EVERY_STEPS", 50)
DRY_RUN = env_bool("DRY_RUN_VALIDATE_ONLY", False)

# Disable cache during training
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 70)
print(f"V80 HF Jobs - dgxchen v7 EXACT replica (RUN_ID={RUN_ID})")
print("=" * 70)
print(f"Model:        {MODEL_NAME}")
print(f"Data repo:    {DATA_REPO}")
print(f"Data file:    {DATA_FILE}")
print(f"Output repo:  {OUTPUT_REPO}")
print(f"Output dir:   {OUTPUT_DIR}")
print(f"Max length:   {MAX_LENGTH}")
print(f"Batch:        eff={BATCH_SIZE} micro={MICRO_BATCH_SIZE} grad_accum={GRAD_ACCUM}")
print(f"LR:           {LEARNING_RATE} linear, warmup=0")
print(f"Epochs:       {NUM_EPOCHS}")
print(f"LoRA:         r={LORA_RANK} alpha={LORA_ALPHA} dropout=0 NO lm_head")
print(f"Seed:         {SEED}")
print(f"Dry-run:      {DRY_RUN}")
print("=" * 70)


# ============ GPU check ============
import torch
assert torch.cuda.is_available(), "CUDA required"
d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f"GPU: {d.name}")
print(f"VRAM: total={total_gb:.1f}GB free={free_gb:.1f}GB")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
assert total_gb >= 38, f"need >=40GB GPU, got {total_gb:.1f}GB"


# ============ Install dependencies at runtime if needed ============
# Unsloth + minimal deps for dgxchen recipe
def ensure_installed(module, pip_name=None):
    pip_name = pip_name or module
    try:
        __import__(module)
        print(f"OK {module}")
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        __import__(module)
        print(f"OK {module} (installed)")


print("\nVerifying deps...")
ensure_installed("transformers")
ensure_installed("peft")
ensure_installed("trl")
ensure_installed("accelerate")
ensure_installed("datasets")
ensure_installed("huggingface_hub")
ensure_installed("pandas")

# Unsloth: try to install, fall back to pure transformers+peft if fails
USE_UNSLOTH = env_bool("USE_UNSLOTH", True)
if USE_UNSLOTH:
    try:
        import unsloth
        print(f"OK unsloth {unsloth.__version__}")
    except ImportError:
        print("Installing unsloth + unsloth_zoo...")
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q",
                              "unsloth", "unsloth_zoo"])
        if rc != 0:
            print("unsloth install FAILED - falling back to pure transformers+peft")
            USE_UNSLOTH = False
        else:
            try:
                import unsloth
                print(f"OK unsloth {unsloth.__version__} (installed)")
            except Exception as e:
                print(f"unsloth import FAILED after install: {e}")
                USE_UNSLOTH = False

if USE_UNSLOTH:
    ensure_installed("bitsandbytes")
    ensure_installed("xformers")

import transformers
import peft
import trl
import accelerate
import datasets
print(f"  transformers={transformers.__version__}")
print(f"  peft={peft.__version__}")
print(f"  trl={trl.__version__}")
print(f"  accelerate={accelerate.__version__}")
print(f"  datasets={datasets.__version__}")


# ============ Seed everything ============
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============ Download dataset ============
from huggingface_hub import HfApi, get_token, hf_hub_download
HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()
assert HF_TOKEN, "HF_TOKEN required"

print(f"\nDownloading dataset from {DATA_REPO}/{DATA_FILE}...")
try:
    local_csv = hf_hub_download(
        repo_id=DATA_REPO,
        filename=DATA_FILE,
        repo_type="dataset",
        token=HF_TOKEN,
    )
except Exception as e:
    print(f"HF dataset download FAILED: {e}")
    print("Fallback: try Kaggle CLI")
    # Fallback: kaggle download if credentials present
    ku, kk = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if ku and kk:
        os.environ["KAGGLE_USERNAME"] = ku
        os.environ["KAGGLE_KEY"] = kk
        kpath = Path.home() / ".kaggle" / "kaggle.json"
        kpath.parent.mkdir(parents=True, exist_ok=True)
        kpath.write_text(json.dumps({"username": ku, "key": kk}))
        kpath.chmod(0o600)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        dl_dir = Path("/tmp/dgxchen_v80")
        dl_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([
            "kaggle", "datasets", "download", "-d", "dgxchen/nemotron-cot-tong",
            "-f", "problem_ids_matched.csv", "-p", str(dl_dir), "--unzip",
        ])
        local_csv = str(dl_dir / "problem_ids_matched.csv")
    else:
        raise RuntimeError(f"Cannot get dataset - no HF repo data + no Kaggle creds: {e}")

import pandas as pd
df = pd.read_csv(local_csv)
print(f"Loaded: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
assert all(c in df.columns for c in ["prompt", "answer", "generated_cot", "type"]), (
    f"Missing required columns. Got: {list(df.columns)}"
)
if "type" in df.columns:
    print(f"Type distribution:")
    for t, n in df["type"].value_counts().items():
        print(f"  {t}: {n}")


# ============ Build dataset (dgxchen v7 EXACT prompt format) ============
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Sampler

PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

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
print(f"\nSFT records: {len(records)}")

dataset = HFDataset.from_list(records)


# ============ Load model + LoRA ============
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainerCallback)
from peft import LoraConfig, get_peft_model

print(f"\nLoading base model {MODEL_NAME}...")
t_load = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if USE_UNSLOTH:
    from unsloth import FastLanguageModel
    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",       # dgxchen v7 exact
        dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
else:
    # Fallback: pure transformers+peft (if Unsloth unavailable)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map={"": 0},
        token=HF_TOKEN,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print(f"Base model loaded in {time.time()-t_load:.1f}s")

# dgxchen v7 LoRA config - 8 items NO lm_head
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "up_proj", "down_proj",
]

if USE_UNSLOTH:
    from unsloth import FastLanguageModel
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
else:
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
print(f"After LoRA: used={used_gb:.1f}GB free={free_gb:.1f}GB")


# ============ formatting_func (dgxchen v7 EXACT) ============
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


# ============ Stratified batching (dgxchen v7 uses this) ============
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


from trl import SFTTrainer, SFTConfig


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


# ============ HealthGateCallback (identical to V80 notebook) ============
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
        print("V80 HF HEALTH GATES ATIVOS - log every 5 steps + abort on NaN/explosion")
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

        # GATE 3: Overfit warning
        if loss is not None and isinstance(loss, (int, float)):
            if loss < 0.01 and step < 50:
                print(f"!!! WARN severe overfit: loss={loss:.4f} at step {step}")

        if loss is not None:
            self.loss_history.append(float(loss))
            if float(loss) < self.min_loss:
                self.min_loss = float(loss)
                self.steps_since_improve = 0
            else:
                self.steps_since_improve += LOG_EVERY_STEPS

        # GATE 4: grad_norm watch (V80 has max_grad_norm=1e9 = no clipping)
        if grad_norm is not None:
            self.grad_history.append(float(grad_norm))
            if float(grad_norm) > 50.0:
                self.high_grad_count += 1
                if self.high_grad_count >= 3:
                    print(f"!!! WARN grad_norm >50 3x (last={grad_norm:.2f}) - no clipping!")
            else:
                self.high_grad_count = 0

        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            total = torch.cuda.mem_get_info()[1] / 1024**3
            used = total - free
            peak = torch.cuda.max_memory_allocated() / 1024**3
            if free < 2.0:
                print(f"!!! WARN VRAM free={free:.1f}GB - OOM risk")
        else:
            free = used = peak = 0.0

        elapsed = time.time() - self.train_start if self.train_start else 0
        total_steps = state.max_steps if state.max_steps else 1
        progress = step / max(total_steps, 1)
        eta_sec = (elapsed / max(step, 1)) * (total_steps - step) if step > 0 else 0

        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0
        avg_grad = sum(self.grad_history) / len(self.grad_history) if self.grad_history else 0.0

        plateau_warn = ""
        if self.steps_since_improve >= 100:
            plateau_warn = f" [PLATEAU {self.steps_since_improve}s]"

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
            f"{plateau_warn}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start if self.train_start else 0
        print("=" * 70)
        print(f"V80 HF TRAINING FINISHED - elapsed={elapsed/60:.1f}min "
              f"min_loss={self.min_loss:.4f} steps={state.global_step}")
        print("=" * 70)


# ============ SFTConfig (dgxchen v7 EXACT) ============
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="linear",
    warmup_steps=0,
    max_length=MAX_LENGTH,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=0.0,
    max_grad_norm=1e9,                          # dgxchen v7 exact (no clipping)
    logging_steps=LOG_EVERY_STEPS,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=SAVE_EVERY_STEPS,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,                # dgxchen v7 exact
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=0,                   # HF Jobs fix: num_workers>=1 fails
                                                # pickling CudaDeviceProperties in formatting_func closure
    remove_unused_columns=False,
    seed=SEED,
    report_to="none",
    packing=False,
)


eff_batch_size = MICRO_BATCH_SIZE * GRAD_ACCUM
stratified_order = build_stratified_index_order(record_types, eff_batch_size, SEED)

health_cb = HealthGateCallback()

if DRY_RUN:
    print("\n[DRY RUN] Skipping training. Config validated.")
    sys.exit(0)

# Build trainer
trainer = StratifiedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,    # dgxchen v7 exact
    stratified_order=stratified_order,
    callbacks=[health_cb],
)

print("\nStarting V80 HF SFT (dgxchen v7 EXACT)...")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nTraining done: {elapsed/60:.1f} min")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f}GB")


# ============ Save + Upload adapter ============
Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
print(f"\nSaving adapter to {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# Fix adapter_config for Kaggle inference
cfg_path = Path(ADAPTER_DIR) / "adapter_config.json"
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = MODEL_NAME
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    # Verify no lm_head
    tm = cfg.get("target_modules", [])
    if isinstance(tm, list) and "lm_head" in tm:
        print(f"WARN adapter_config has lm_head in target_modules! Removing.")
        tm = [m for m in tm if m != "lm_head"]
        cfg["target_modules"] = tm
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"adapter_config: base={cfg['base_model_name_or_path']} targets={len(tm)} items")

# Upload to HF
print(f"\nUploading to HF {OUTPUT_REPO}...")
try:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=OUTPUT_REPO,
        folder_path=ADAPTER_DIR,
        allow_patterns=["adapter_*", "tokenizer*", "special_tokens_map.json"],
        token=HF_TOKEN,
        commit_message=f"V80 dgxchen v7 EXACT {RUN_ID}",
    )
    print(f"HF upload OK: https://huggingface.co/{OUTPUT_REPO}")
except Exception as e:
    print(f"HF upload FAILED: {e}")
    print(f"Adapter saved locally at: {ADAPTER_DIR}")

print(f"\n{'=' * 70}")
print(f"V80 HF JOB DONE - RUN_ID={RUN_ID}")
print(f"Adapter: {ADAPTER_DIR}")
print(f"HF repo: {OUTPUT_REPO}")
print(f"{'=' * 70}")
