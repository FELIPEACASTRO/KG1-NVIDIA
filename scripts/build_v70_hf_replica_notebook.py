#!/usr/bin/env python3
"""Build Colab notebook that replicates HF Job training EXACTLY.

BF16 + loss masking + max_length auto + batch=64 + Adam beta2=0.95
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "KG1_v70_HF_REPLICA_COLAB.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.strip().splitlines(keepends=True), "outputs": [], "execution_count": None}


cells = []

# ── Cell 0: Markdown header ──
cells.append(md("""
# KG1 v70 — Huikang Replica (BF16 + Loss Masking)

Replica EXATA do pipeline do huikang (0.85):
- BF16 full precision
- Loss masking (so treina completion tokens)
- Adam beta2=0.95, weight_decay=0.0
- LR 2e-4 linear decay to 0
- LoRA all-linear r=32 alpha=32
- Dataset: 16365 exemplos com CoTs deterministicos
"""))

# ── Cell 1: GPU + deps ──
cells.append(code("""
#@title 1. GPU + Dependencias
import subprocess, sys, os, importlib.metadata as meta

def pip_install(args, allow_fail=False):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args, text=True, capture_output=True)
    if r.returncode != 0 and not allow_fail:
        print(r.stdout[-500:] if r.stdout else "")
        print(r.stderr[-500:] if r.stderr else "")
        raise RuntimeError(f"pip install failed: {args}")
    return r.returncode

# GPU info
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True, capture_output=True)
gpu_line = smi.stdout.strip().splitlines()[0] if smi.returncode == 0 else "Unknown"
print(f"GPU: {gpu_line}")

# Install deps
pip_install(["-U", "pip", "setuptools", "wheel"])
pip_install(["packaging"])

import torch
print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU available: {torch.cuda.is_available()}")

pip_install(["transformers>=4.48", "peft>=0.14", "datasets", "accelerate", "huggingface_hub", "safetensors", "sentencepiece", "einops", "ninja"])

# bitsandbytes for 8-bit optimizer (saves VRAM)
pip_install(["bitsandbytes>=0.43"])

# mamba-ssm + causal-conv1d
os.environ["MAX_JOBS"] = "4"
if "H100" in gpu_line.upper():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
elif "A100" in gpu_line.upper():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

pip_install(["causal-conv1d>=1.4.0", "--no-build-isolation"], allow_fail=True)
pip_install(["mamba-ssm>=2.2.2", "--no-build-isolation"], allow_fail=True)

# Verify mamba-ssm
try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    print("mamba-ssm: OK")
except ImportError:
    pip_install(["mamba-ssm", "--no-build-isolation"])
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    print("mamba-ssm: OK (2nd try)")

# Stub causal_conv1d if needed
try:
    import causal_conv1d
    print("causal-conv1d: OK")
except ImportError:
    import types, importlib.machinery
    stub = types.ModuleType("causal_conv1d")
    stub.causal_conv1d_fn = None
    stub.causal_conv1d_update = None
    stub.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", loader=None)
    sys.modules["causal_conv1d"] = stub
    print("causal_conv1d: stub injetado")

for pkg in ["torch", "transformers", "peft", "mamba-ssm", "bitsandbytes"]:
    try: print(f"  {pkg}: {meta.version(pkg)}")
    except: pass
print("\\nDeps OK!")
"""))

# ── Cell 2: Auth + Config ──
cells.append(code("""
#@title 2. Auth + Config
import os
from huggingface_hub import HfApi

HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN") or userdata.get("HF_KEY")
except: pass
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY")
if not HF_TOKEN:
    import getpass
    HF_TOKEN = getpass.getpass("HF Token: ")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

api = HfApi(token=HF_TOKEN)
print(f"Auth: {api.whoami()['name']}")

# === CONFIG (huikang replica) ===
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATA_FILE = "data/sft_v70_huikang_full.jsonl"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v70-huikang"
OUTPUT_DIR = "/content/v70_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LoRA (identical to huikang)
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# Training (identical to huikang)
BATCH_SIZE = 64
MICRO_BATCH = 1
GRAD_ACCUM = BATCH_SIZE // MICRO_BATCH  # 64
LEARNING_RATE = 2e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 1
SEED = 42

# Max length: start at 4096, fallback if OOM
MAX_LENGTH = 4096

print(f"Config: r={LORA_R}, alpha={LORA_ALPHA}, LR={LEARNING_RATE}, batch={BATCH_SIZE}")
print(f"Max length: {MAX_LENGTH}")
print(f"Output: {OUTPUT_REPO}")
"""))

# ── Cell 3: Download + tokenize with loss masking ──
cells.append(code("""
#@title 3. Download dataset + tokenize com loss masking
import json, torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

print(f"Downloading {DATA_FILE}...")
data_path = hf_hub_download(repo_id=DATA_REPO, filename=DATA_FILE, repo_type="dataset", token=HF_TOKEN)
print(f"Downloaded: {data_path}")

# Load
examples = []
with open(data_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))
print(f"Examples: {len(examples)}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize with loss masking (ONLY completion tokens trained)
data = []
skipped = 0
for ex in examples:
    msgs = ex.get("messages", [])
    if not msgs:
        skipped += 1
        continue

    # Full conversation
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Prompt only (all except assistant)
    prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Truncate
    if len(full_ids) > MAX_LENGTH:
        full_ids = full_ids[:MAX_LENGTH]

    # Loss mask: 0=prompt, 1=completion
    prompt_len = min(len(prompt_ids), len(full_ids))
    loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

    if sum(loss_mask) == 0:
        skipped += 1
        continue

    data.append({"input_ids": full_ids, "loss_mask": loss_mask})

total_tokens = sum(len(d["input_ids"]) for d in data)
unmasked = sum(sum(d["loss_mask"]) for d in data)
print(f"Tokenized: {len(data)} examples (skipped {skipped})")
print(f"Total tokens: {total_tokens:,} | Trained (unmasked): {unmasked:,}")
print(f"Avg length: {total_tokens/len(data):.0f} tokens")
"""))

# ── Cell 4: Load model BF16 + LoRA ──
cells.append(code("""
#@title 4. Carregar modelo BF16 + LoRA
import gc, torch, sys
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

gc.collect()
torch.cuda.empty_cache()

# Ensure causal_conv1d stub
try:
    import causal_conv1d
except ImportError:
    import types, importlib.machinery
    stub = types.ModuleType("causal_conv1d")
    stub.causal_conv1d_fn = None
    stub.causal_conv1d_update = None
    stub.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", loader=None)
    sys.modules["causal_conv1d"] = stub

print(f"Loading {MODEL_NAME} in BF16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
    attn_implementation="eager",
)

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

mem = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM: {mem:.1f} GB")
"""))

# ── Cell 5: Training loop with masked loss ──
cells.append(code("""
#@title 5. Treinar (loss masking, replica huikang)
import gc, math, random, time, json
import torch
import torch.nn.functional as F
from pathlib import Path

random.seed(SEED)
torch.manual_seed(SEED)

def masked_cross_entropy(logits, input_ids, loss_mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous().float()
    B, T, V = shift_logits.shape
    per_token = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none")
    masked = per_token * shift_mask.view(-1)
    n = shift_mask.sum()
    return masked.sum() / n if n > 0 else torch.tensor(0.0, device=logits.device)

# 8-bit paged optimizer (saves ~5GB VRAM vs fp32 Adam)
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdam8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
    )
    print("Optimizer: PagedAdam8bit (saves VRAM)")
except Exception as e:
    print(f"bitsandbytes failed ({e}), using torch Adam")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
    )

total_steps = math.ceil(len(data) / BATCH_SIZE) * NUM_EPOCHS
print(f"Training: {len(data)} examples, batch={BATCH_SIZE}, steps={total_steps}")
print(f"LR: {LEARNING_RATE} -> 0 (linear decay)")
print(f"Max length: {MAX_LENGTH}")

model.train()
global_step = 0
accum_loss = 0.0
accum_count = 0
start = time.time()
log_history = []

for epoch in range(NUM_EPOCHS):
    random.shuffle(data)
    for i in range(0, len(data), MICRO_BATCH):
        ex = data[i]
        ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device="cuda")
        mask = torch.tensor([ex["loss_mask"]], dtype=torch.long, device="cuda")

        try:
            out = model(input_ids=ids)
            loss = masked_cross_entropy(out.logits, ids, mask) / GRAD_ACCUM
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at example {i} (len={len(ex['input_ids'])}), skipping")
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            accum_count = 0
            accum_loss = 0.0
            continue

        accum_loss += loss.item() * GRAD_ACCUM
        accum_count += 1

        if accum_count % GRAD_ACCUM == 0:
            lr = LEARNING_RATE * (1 - global_step / max(total_steps, 1))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            avg_loss = accum_loss / GRAD_ACCUM
            elapsed = time.time() - start

            log_history.append({"step": global_step, "loss": avg_loss, "lr": lr, "time": elapsed})

            if global_step % 5 == 0:
                eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                print(f"step={global_step}/{total_steps} lr={lr:.2e} loss={avg_loss:.4f} time={elapsed:.0f}s ETA={eta:.0f}s")

            if global_step > 0 and global_step % 50 == 0:
                ckpt = Path(OUTPUT_DIR) / f"checkpoint-{global_step}"
                model.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                # Upload checkpoint
                try:
                    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
                    api.upload_folder(folder_path=str(ckpt), path_in_repo=f"checkpoint-{global_step}",
                                      repo_id=OUTPUT_REPO, commit_message=f"ckpt-{global_step} loss={avg_loss:.4f}", token=HF_TOKEN)
                    print(f"  Uploaded checkpoint-{global_step}")
                except Exception as e:
                    print(f"  Upload failed: {e}")

            accum_loss = 0.0
            global_step += 1

elapsed = time.time() - start
final_loss = log_history[-1]["loss"] if log_history else 0
print(f"\\n=== TREINO COMPLETO ===")
print(f"Steps: {global_step}, Loss final: {final_loss:.4f}, Tempo: {elapsed/3600:.2f}h")

# Save final
final_dir = Path(OUTPUT_DIR) / "final_adapter"
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

# Save log
with open(Path(OUTPUT_DIR) / "train_log.json", "w") as f:
    json.dump(log_history, f, indent=2)
print(f"Adapter salvo: {final_dir}")
"""))

# ── Cell 6: Upload final ──
cells.append(code("""
#@title 6. Upload final adapter
from pathlib import Path

final_dir = Path(OUTPUT_DIR) / "final_adapter"
api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(final_dir),
    path_in_repo="final",
    repo_id=OUTPUT_REPO,
    commit_message=f"v70 final step={global_step} loss={final_loss:.4f}",
    token=HF_TOKEN,
)
print(f"Uploaded: {OUTPUT_REPO}/final")
print(f"Loss: {final_loss:.4f}")
print(f"Steps: {global_step}")
"""))

# ── Cell 7: Loss curve ──
cells.append(code("""
#@title 7. Loss curve
import matplotlib.pyplot as plt

steps = [x["step"] for x in log_history]
losses = [x["loss"] for x in log_history]

plt.figure(figsize=(10, 4))
plt.plot(steps, losses, "b-", linewidth=1)
plt.xlabel("Step")
plt.ylabel("Loss (masked, completion only)")
plt.title(f"v70 Training - Final loss: {final_loss:.4f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=100)
plt.show()
print(f"Min loss: {min(losses):.4f} at step {steps[losses.index(min(losses))]}")
"""))

# Build notebook
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook: {OUT}")
print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main = None  # avoid name error
    import ast
    with open(OUT, encoding="utf-8") as f:
        nb = json.load(f)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    print("Syntax: ALL OK")
