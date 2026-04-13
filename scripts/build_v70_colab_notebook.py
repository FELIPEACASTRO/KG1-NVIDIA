#!/usr/bin/env python3
"""Build KG1 v70 Colab notebook — SFT with code-generated CoTs + adapter converter.

Based on analysis of huikang's 0.85 approach:
- r=32, alpha=32, all-linear, maxlen=4096
- Code-generated deterministic CoTs (no LLM distillation)
- Post-training adapter conversion (unfuse experts + SVD merge mamba)
- Loss target ~6.5-7.0 (NOT below 6.0 to avoid overfitting)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "KG1_v70_SFT_CODEGEN_COLAB.ipynb"


def src(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id},
        "outputs": [],
        "source": src(text),
    }


cells: list[dict] = []

# ─── Cell 0: Header ─────────────────────────────────────────────────
cells.append(md(r"""
# KG1 v70 — SFT with Code-Generated CoTs + Adapter Converter

**Objetivo**: Treinar adapter LoRA all-linear com dados code-generated e converter para formato vLLM.

**Baseado em**: Analise do huikang (score 0.85, #10 leaderboard)

**Config**:
- Base: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- LoRA: r=32, alpha=32, dropout=0, all-linear
- Dataset: ~4400 exemplos code-generated (0 traces fracos)
- MaxLen: 4096 (A100 80GB) ou fallback 2048 (menor GPU)
- Loss target: 6.5-7.0 (NAO abaixo de 6.0)
- Converter: expert unfuse + SVD merge + key rename
"""))

# ─── Cell 1: GPU Preflight ──────────────────────────────────────────
cells.append(code(r"""
#@title 1. GPU Preflight
import subprocess, os, sys

def get_gpu_info():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total",
                                        "--format=csv,noheader"], text=True)
        name, mem = out.strip().split(",")
        mem_gb = int(mem.strip().split()[0]) / 1024
        return name.strip(), mem_gb
    except Exception:
        return "Unknown", 0

gpu_name, gpu_mem = get_gpu_info()
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

if gpu_mem < 30:
    print("AVISO: GPU com menos de 30GB. Considere usar Colab Pro com A100.")
    USE_QLORA = True
    MAX_LENGTH = 2048
elif gpu_mem < 60:
    USE_QLORA = False
    MAX_LENGTH = 2048
    print("GPU media. Usando BF16 com maxlen=2048.")
else:
    USE_QLORA = False
    MAX_LENGTH = 4096
    print("GPU A100/H100. Usando BF16 com maxlen=4096.")
""", "gpu_preflight"))

# ─── Cell 2: Install dependencies ───────────────────────────────────
cells.append(code(r"""
#@title 2. Instalar dependencias (Nemotron-H requer mamba-ssm)
import importlib.metadata as metadata
import os, shutil, subprocess, sys

def pip_install(args, allow_fail=False):
    print("INSTALL:", " ".join(args))
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args, text=True, capture_output=True)
    if result.returncode != 0:
        print("INSTALL FALHOU:", " ".join(args))
        tail = (result.stdout + "\n" + result.stderr).strip().splitlines()[-20:]
        print("\n".join(tail))
        if not allow_fail:
            raise RuntimeError("pip install failed: " + " ".join(args))
    return result.returncode

pip_install(["-U", "pip", "setuptools", "wheel"])
pip_install(["packaging"])

# Detectar torch pre-instalado (Colab ja vem com CUDA)
try:
    import torch
    print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | available: {torch.cuda.is_available()}")
except Exception:
    pip_install(["torch"])

pip_install([
    "transformers>=4.48",
    "peft>=0.14",
    "trl>=0.13",
    "datasets>=3.2",
    "accelerate>=1.2",
    "huggingface_hub",
    "safetensors",
    "pandas",
    "sentencepiece",
    "einops",
    "ninja",
])

# mamba-ssm e OBRIGATORIO para Nemotron-H (arquitetura hibrida Mamba+Attention)
# causal-conv1d e opcional (fast-path), mas pode falhar no build CUDA
os.environ.setdefault("MAX_JOBS", "4")

# Detectar arch CUDA
if "H100" in gpu_name.upper() or "HOPPER" in gpu_name.upper():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
elif "A100" in gpu_name.upper():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

# Tentar causal-conv1d (opcional, pode falhar)
pip_install(["causal-conv1d>=1.4.0", "--no-build-isolation"], allow_fail=True)

# mamba-ssm (OBRIGATORIO) - tentar varias estrategias
mamba_ok = pip_install(["mamba-ssm>=2.2.2", "--no-build-isolation"], allow_fail=True)
if mamba_ok != 0:
    print("Tentando mamba-ssm sem --no-build-isolation...")
    mamba_ok = pip_install(["mamba-ssm>=2.2.2"], allow_fail=True)
if mamba_ok != 0:
    print("Tentando mamba-ssm via pip direto...")
    mamba_ok = pip_install(["mamba-ssm"], allow_fail=True)

# Verificar imports
def check_import(label, code, required=False):
    try:
        exec(code, {})
        print(f"{label}: OK")
        return True
    except Exception as exc:
        print(f"{label}: FALHOU ({type(exc).__name__}: {str(exc)[:200]})")
        if required:
            raise RuntimeError(f"Dependencia obrigatoria: {label}") from exc
        return False

CAUSAL_CONV1D_OK = check_import("causal_conv1d", "import causal_conv1d")
MAMBA_SSM_OK = check_import("mamba_ssm", "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn", required=True)

if not CAUSAL_CONV1D_OK:
    print("AVISO: causal_conv1d ausente. Sera usado fallback sem fast-path.")

# Print versions
for pkg in ["torch", "transformers", "peft", "trl", "accelerate", "mamba-ssm", "causal-conv1d"]:
    try:
        print(f"  {pkg}: {metadata.version(pkg)}")
    except: pass
print("Dependencias OK.")
""", "install_deps"))

# ─── Cell 3: Auth + Config ──────────────────────────────────────────
cells.append(code(r"""
#@title 3. Autenticacao e Configuracao
import os
from pathlib import Path
from huggingface_hub import HfApi

# === AUTH ===
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN") or userdata.get("HF_KEY")
except Exception:
    pass
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    import getpass
    HF_TOKEN = getpass.getpass("HF Token: ")
os.environ["HF_TOKEN"] = HF_TOKEN

api = HfApi(token=HF_TOKEN)
whoami = api.whoami()
print(f"Autenticado como: {whoami['name']}")

# === CONFIG ===
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATA_FILENAME = "data/sft_v70_huikang_full.jsonl"

RUN_ID = "v70-sft-huikang-m4096-lr2e4-a32-d0-b64"
OUTPUT_REPO = f"felipesp1983/kg1-nemotron-lora-v70-huikang"
OUTPUT_DIR = Path("/content/kg1_v70_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR = Path("/content/kg1_v70_converted")

# LoRA config (huikang-validated)
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# Training config
# Hyperparams matched to huikang's 0.85 config
LEARNING_RATE = 2e-4  # huikang uses 2e-4 (4x our previous 5e-5)
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 64  # effective batch = 64 (huikang: micro=16, GA=4)
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0  # huikang: 0.0
MAX_GRAD_NORM = 1e9  # huikang: effectively no clipping
MAX_STEPS = -1  # use epochs instead
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 10
NUM_TRAIN_EPOCHS = 1  # huikang: 1 epoch
SEED = 42

# Adam betas matched to huikang
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95  # huikang: 0.95 (not default 0.999)

# Loss gates
LOSS_KILL_THRESHOLD = 10.0
LOSS_SWEET_SPOT = (5.0, 8.0)

print(f"Config: r={LORA_R}, alpha={LORA_ALPHA}, maxlen={MAX_LENGTH}, lr={LEARNING_RATE}")
print(f"Output: {OUTPUT_REPO}")
""", "auth_config"))

# ─── Cell 4: Download dataset ────────────────────────────────────────
cells.append(code(r"""
#@title 4. Baixar dataset
import json, hashlib
from pathlib import Path
from collections import Counter
from huggingface_hub import hf_hub_download

ALLOW_UPLOAD_LOCAL = True

try:
    dataset_path = Path(hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=DATA_FILENAME,
        token=HF_TOKEN,
        local_dir="/content/kg1_data",
    ))
    print(f"Dataset baixado: {dataset_path}")
except Exception as exc:
    print(f"HF download falhou: {exc}")
    if ALLOW_UPLOAD_LOCAL:
        from google.colab import files
        print("Selecione o arquivo sft_v70_final.jsonl:")
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("Nenhum arquivo enviado.")
        dataset_path = Path("/content") / next(iter(uploaded.keys()))
    else:
        raise

# Validate
examples = []
with open(dataset_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

cats = Counter(e.get("family", "unknown") for e in examples)
print(f"\nDataset: {len(examples)} exemplos")
for fam, n in sorted(cats.items()):
    print(f"  {fam}: {n} ({n/len(examples)*100:.1f}%)")

# Check for weak traces
weak = sum(1 for e in examples if e.get("trace_quality") == "weak")
if weak > 0:
    print(f"\nAVISO: {weak} traces fracos encontrados!")
else:
    print(f"\n0 traces fracos. Dataset limpo.")
""", "download_data"))

# ─── Cell 5: Load model + LoRA ───────────────────────────────────────
cells.append(code(r"""
#@title 5. Carregar modelo + LoRA
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Patch: se causal_conv1d nao instalou, injetar stub para evitar ImportError no load
try:
    import causal_conv1d
except ImportError:
    import types, sys as _sys
    stub = types.ModuleType("causal_conv1d")
    stub.__version__ = "0.0.0"
    stub.causal_conv1d_fn = None
    stub.causal_conv1d_update = None
    _sys.modules["causal_conv1d"] = stub
    print("Injetado stub causal_conv1d (fallback sem fast-path)")

print(f"Carregando {BASE_MODEL}...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model em BF16 (H100/A100 80GB tem VRAM suficiente)
print("Usando BF16 full precision...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True, token=HF_TOKEN,
    attn_implementation="eager",  # evita flash_attn issues
)

# LoRA config - all-linear (huikang approach)
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules="all-linear",
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nParametros treinaveis: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, target=all-linear")
""", "load_model"))

# ─── Cell 6: Prepare dataset ─────────────────────────────────────────
cells.append(code(r"""
#@title 6. Preparar dataset para treino
from datasets import Dataset

def format_chat(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return {"text": text}

# Load and format
raw_data = []
with open(dataset_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_data.append(json.loads(line))

ds = Dataset.from_list(raw_data)
ds = ds.map(format_chat, remove_columns=[c for c in ds.column_names if c != "text"])

# Tokenize and filter by length
def tokenize_and_filter(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH)
    example["input_ids"] = tokens["input_ids"]
    example["attention_mask"] = tokens["attention_mask"]
    example["token_count"] = len(tokens["input_ids"])
    return example

ds = ds.map(tokenize_and_filter)

# Filter out examples longer than MAX_LENGTH
original_len = len(ds)
ds = ds.filter(lambda x: x["token_count"] <= MAX_LENGTH)
filtered_len = len(ds)
print(f"Dataset: {original_len} -> {filtered_len} (removed {original_len - filtered_len} too-long)")

# Stats
token_counts = ds["token_count"]
print(f"Token counts: min={min(token_counts)}, max={max(token_counts)}, mean={sum(token_counts)/len(token_counts):.0f}")
print(f"Total tokens: {sum(token_counts):,}")
""", "prepare_data"))

# ─── Cell 7: Train ───────────────────────────────────────────────────
cells.append(code(r"""
#@title 7. Treinar SFT
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    adam_beta1=ADAM_BETA1,
    adam_beta2=ADAM_BETA2,
    lr_scheduler_type="linear",  # huikang: linear decay to 0
    bf16=True,
    logging_steps=5,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    seed=SEED,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
    dataset_text_field="text",
    max_seq_length=MAX_LENGTH,
    packing=False,
)

print(f"Iniciando treino: {MAX_STEPS} steps, batch={PER_DEVICE_TRAIN_BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}")
print(f"Dataset: {len(ds)} exemplos, MaxLen: {MAX_LENGTH}")

train_result = trainer.train()

# Print final metrics
metrics = train_result.metrics
final_loss = metrics.get("train_loss", 0)
print(f"\n=== TREINO COMPLETO ===")
print(f"Loss final: {final_loss:.4f}")
print(f"Sweet spot: {LOSS_SWEET_SPOT}")
if LOSS_SWEET_SPOT[0] <= final_loss <= LOSS_SWEET_SPOT[1]:
    print("LOSS NO SWEET SPOT!")
elif final_loss < LOSS_SWEET_SPOT[0]:
    print("AVISO: Loss muito baixo - possivel overfitting! Considere usar checkpoint anterior.")
else:
    print("AVISO: Loss alto - treino insuficiente. Considere mais steps.")

# Save raw adapter
raw_adapter_dir = OUTPUT_DIR / "raw_adapter"
trainer.model.save_pretrained(str(raw_adapter_dir))
tokenizer.save_pretrained(str(raw_adapter_dir))
print(f"\nAdapter salvo em: {raw_adapter_dir}")
""", "train"))

# ─── Cell 8: Convert adapter ─────────────────────────────────────────
cells.append(code(r"""
#@title 8. Converter adapter (unfuse experts + SVD merge)
import re, math, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

raw_adapter_dir = OUTPUT_DIR / "raw_adapter"
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

LORA_SUFFIX_RE = re.compile(r"\.lora_([AB])\.weight$")
DEFAULT_TARGET_MODULES = ["k_proj","o_proj","in_proj","q_proj","up_proj","v_proj","down_proj","out_proj","lm_head"]

def rename_key(base: str) -> str:
    return base.replace("base_model.model.model", "base_model.model.backbone", 1)

# Load tensors
print("Carregando adapter tensors...")
input_path = raw_adapter_dir / "adapter_model.safetensors"
adapter_tensors = {}
with safe_open(str(input_path), framework="pt", device="cpu") as f:
    for key in f.keys():
        adapter_tensors[key] = f.get_tensor(key)
print(f"Input: {len(adapter_tensors)} tensors")

# Collect LoRA bases
bases = {}
for key in adapter_tensors:
    m = LORA_SUFFIX_RE.search(key)
    if m:
        base = key[:m.start()]
        bases.setdefault(base, {})[m.group(1)] = key

# Identify Mamba merge layers
mamba_layers = {}
for base in bases:
    for proj in ("gate_proj", "x_proj"):
        if f".{proj}" in base:
            layer_path = base.rsplit(f".{proj}", 1)[0]
            mamba_layers.setdefault(layer_path, {})[proj] = base

complete_mamba = {k: v for k, v in mamba_layers.items() if {"gate_proj", "x_proj"}.issubset(v)}
mamba_merge_bases = {b for projs in complete_mamba.values() for b in projs.values()}

# Build output tensors
out_tensors = {}
stats = {"direct": 0, "expert_split": 0, "mamba_merged": 0, "skipped_w3": 0}

for base, keys in sorted(bases.items()):
    if set(keys) != {"A", "B"}:
        continue
    lora_a = adapter_tensors[keys["A"]]
    lora_b = adapter_tensors[keys["B"]]
    renamed = rename_key(base)

    # Skip empty w3
    if ".experts.w3" in renamed and lora_a.numel() == 0:
        stats["skipped_w3"] += 1
        continue

    # Skip mamba merge bases
    if base in mamba_merge_bases:
        continue

    # Expert unfusing: w1 -> up_proj, w2 -> down_proj
    if ".experts.w1" in renamed or ".experts.w2" in renamed:
        proj_name = "up_proj" if ".w1" in renamed else "down_proj"
        if lora_a.ndim == 3:
            n_experts = max(lora_a.shape[0], lora_b.shape[0])
            if lora_a.shape[0] == 1:
                lora_a = lora_a.expand(n_experts, -1, -1).contiguous()
            if lora_b.shape[0] == 1:
                lora_b = lora_b.expand(n_experts, -1, -1).contiguous()
            for i in range(n_experts):
                exp_name = re.sub(r"\.experts\.w[12]", f".experts.{i}.{proj_name}", renamed)
                out_tensors[f"{exp_name}.lora_A.weight"] = lora_a[i].contiguous()
                out_tensors[f"{exp_name}.lora_B.weight"] = lora_b[i].contiguous()
            stats["expert_split"] += 1
            continue

    # Direct rename
    out_tensors[f"{renamed}.lora_A.weight"] = lora_a.contiguous()
    out_tensors[f"{renamed}.lora_B.weight"] = lora_b.contiguous()
    stats["direct"] += 1

# Mamba SVD merge: gate_proj + x_proj -> in_proj
for layer_path, projs in sorted(complete_mamba.items()):
    renamed_layer = rename_key(layer_path)
    gate_a = adapter_tensors[bases[projs["gate_proj"]]["A"]].float()
    gate_b = adapter_tensors[bases[projs["gate_proj"]]["B"]].float()
    x_a = adapter_tensors[bases[projs["x_proj"]]["A"]].float()
    x_b = adapter_tensors[bases[projs["x_proj"]]["B"]].float()
    rank = gate_a.shape[0]

    # QR + SVD compression: rank-64 -> rank-32
    a_cat = torch.cat([gate_a, x_a], dim=0)
    in_proj_dim = gate_b.shape[0] + x_b.shape[0]
    b_block = torch.zeros(in_proj_dim, 2 * rank)
    b_block[:gate_b.shape[0], :rank] = gate_b
    b_block[gate_b.shape[0]:gate_b.shape[0]+x_b.shape[0], rank:] = x_b

    q_b, r_b = torch.linalg.qr(b_block)
    q_a, r_a = torch.linalg.qr(a_cat.T)
    core = r_b @ r_a.T
    u, s, vh = torch.linalg.svd(core, full_matrices=False)

    sqrt_s = torch.sqrt(torch.clamp(s[:rank], min=0))
    new_b = (q_b @ u[:, :rank]) * sqrt_s.unsqueeze(0)
    new_a = (sqrt_s.unsqueeze(1) * vh[:rank, :]) @ q_a.T

    kept = s[:rank].sum().item()
    total_s = s.sum().item()
    pct = kept / total_s * 100 if total_s > 0 else 0
    print(f"  {layer_path}: SVD kept {pct:.1f}% of singular value mass")

    out_tensors[f"{renamed_layer}.in_proj.lora_A.weight"] = new_a
    out_tensors[f"{renamed_layer}.in_proj.lora_B.weight"] = new_b
    stats["mamba_merged"] += 1

# Save converted adapter
save_file(out_tensors, str(CONVERTED_DIR / "adapter_model.safetensors"))
print(f"\nConversao: {len(adapter_tensors)} -> {len(out_tensors)} tensors")
print(f"Stats: {stats}")

# Write converted adapter_config
import json
config_path = raw_adapter_dir / "adapter_config.json"
with open(config_path) as f:
    config = json.load(f)
config["target_modules"] = DEFAULT_TARGET_MODULES
config["inference_mode"] = True
with open(CONVERTED_DIR / "adapter_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Copy optional files
for fname in ["README.md", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    src = raw_adapter_dir / fname
    if src.exists():
        shutil.copy2(src, CONVERTED_DIR / fname)

# Write checkpoint_complete marker
(CONVERTED_DIR / "checkpoint_complete").touch()

print(f"\nAdapter convertido salvo em: {CONVERTED_DIR}")
print(f"Tamanho: {sum(f.stat().st_size for f in CONVERTED_DIR.iterdir()) / 1024**3:.2f} GB")
""", "convert_adapter"))

# ─── Cell 9: Package submission ──────────────────────────────────────
cells.append(code(r"""
#@title 9. Empacotar submission.zip
import zipfile

submission_path = OUTPUT_DIR / "submission.zip"
with zipfile.ZipFile(submission_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in CONVERTED_DIR.iterdir():
        if f.is_file():
            zf.write(f, f.name)
            print(f"  Added: {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

print(f"\nsubmission.zip: {submission_path.stat().st_size / 1024**3:.2f} GB")
""", "package"))

# ─── Cell 10: Upload to HF ──────────────────────────────────────────
cells.append(code(r"""
#@title 10. Upload adapter para Hugging Face
from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)

# Create repo if needed
try:
    api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", private=True, exist_ok=True)
except Exception as e:
    print(f"Repo creation: {e}")

# Upload converted adapter
api.upload_folder(
    folder_path=str(CONVERTED_DIR),
    repo_id=OUTPUT_REPO,
    repo_type="model",
    commit_message=f"{RUN_ID} converted adapter (loss={final_loss:.4f})",
)
print(f"Adapter uploaded: https://huggingface.co/{OUTPUT_REPO}")

# Upload submission.zip
api.upload_file(
    path_or_fileobj=str(submission_path),
    path_in_repo="submission.zip",
    repo_id=OUTPUT_REPO,
    repo_type="model",
    commit_message=f"{RUN_ID} submission.zip",
)
print(f"submission.zip uploaded.")

# Upload raw adapter too (for debugging)
api.upload_folder(
    folder_path=str(raw_adapter_dir),
    path_in_repo="raw_adapter",
    repo_id=OUTPUT_REPO,
    repo_type="model",
    commit_message=f"{RUN_ID} raw adapter (pre-conversion)",
)
print(f"Raw adapter uploaded.")
""", "upload"))

# ─── Cell 11: Summary ───────────────────────────────────────────────
cells.append(code(r"""
#@title 11. Resumo
print("=" * 60)
print(f"  KG1 v70 TREINO COMPLETO")
print("=" * 60)
print(f"  Loss final: {final_loss:.4f}")
print(f"  Dataset: {len(ds)} exemplos")
print(f"  MaxLen: {MAX_LENGTH}")
print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"  Adapter convertido: {len(out_tensors)} tensors")
print(f"  submission.zip: {submission_path.stat().st_size / 1024**3:.2f} GB")
print(f"  HF repo: {OUTPUT_REPO}")
print("=" * 60)
print()
print("PROXIMO PASSO:")
print("  1. Baixe submission.zip do HF ou do Colab")
print("  2. Submeta no Kaggle:")
print(f"     kaggle competitions submit nvidia-nemotron-model-reasoning-challenge -f submission.zip -m '{RUN_ID}'")
""", "summary"))

# ─── Build notebook ──────────────────────────────────────────────────
def main():
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook salvo: {OUT}")
    print(f"  Celulas: {len(cells)}")


if __name__ == "__main__":
    main()
