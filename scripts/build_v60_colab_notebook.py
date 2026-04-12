#!/usr/bin/env python3
"""Build the KG1 v61 Colab Pro A100/H100 training notebook.

The notebook is intentionally self-contained: it installs dependencies,
authenticates with Hugging Face, validates the boxed dataset, trains LoRA/QLoRA,
and uploads an adapter-only final artifact.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "KG1_v61_COLAB_PRO_A100_H100_TRAIN.ipynb"


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

cells.append(md(r"""
# KG1 v61 Colab Pro A100/H100 Training

Notebook completo para treinar o adapter `v61` no Colab Pro/Pro+ com GPU A100 ou H100.

Config padrao desta run:

- Base: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Dataset boxed deduplicado: `data/v60_selective_train_boxed_dedup.jsonl`
- Dataset hash esperado: `bfe2421b917d761c6528c4aa7bcecc105e8067381fd13a08e066a75cb7b8ad8f`
- Output repo: `felipesp1983/kg1-nemotron-lora-v61-dedup-20260412-001`
- Run: `v61-sft-dedup-weakfocus-m2500-lr5e5-a32-d0-20260412-001`
- LoRA: rank `32`, alpha `32`, dropout `0.0`
- Gate de loss: promover se `< 6.87`; matar se `> 10` depois de 400 steps

Nao cole tokens no notebook. Use Colab Secrets com `HF_TOKEN`, ou digite quando a celula pedir. O token nao sera impresso.
"""))

cells.append(md(r"""
## Como executar

1. No Colab, selecione `Runtime > Change runtime type > GPU`.
2. Para treino serio, prefira H100 ou A100 80GB. A100 40GB usa QLoRA 4-bit automaticamente.
3. Em `Secrets`, crie `HF_TOKEN` com leitura do modelo/dataset e escrita no output repo.
4. Rode as celulas em ordem.
5. Nao submeta ao Kaggle antes do metric gate local contra `v30`, `v34` e `v50c`.
"""))

cells.append(code(r"""
#@title 0. Preflight: GPU e ambiente
import os, re, json, time, random, hashlib, shutil, subprocess, getpass, gc
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

ARTIFACT_DIR = Path("/content/kg1_v61_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def sh(cmd, check=False):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, check=check)

print("Python:", os.sys.version.split()[0])
smi = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
if smi.returncode != 0:
    raise SystemExit("Nenhuma GPU detectada. Ative runtime GPU no Colab.")

gpu_line = smi.stdout.strip().splitlines()[0]
print("GPU:", gpu_line)
m = re.search(r"(.+),\s*(\d+)\s*MiB", gpu_line)
GPU_NAME = m.group(1).strip() if m else gpu_line
GPU_MEM_GB = int(m.group(2)) / 1024 if m else 0
if "H100" in GPU_NAME.upper():
    GPU_TIER = "h100"
elif "A100" in GPU_NAME.upper() and GPU_MEM_GB >= 70:
    GPU_TIER = "a100_80gb"
elif "A100" in GPU_NAME.upper():
    GPU_TIER = "a100_40gb"
else:
    GPU_TIER = "unsupported_or_smoke_only"
print("GPU_TIER:", GPU_TIER, "| VRAM GB:", round(GPU_MEM_GB, 1))
if GPU_TIER == "unsupported_or_smoke_only":
    print("AVISO: notebook desenhado para A100/H100. Use SMOKE_TEST=True se continuar.")
""", "preflight"))

cells.append(code(r"""
#@title 1. Instalar dependencias
import importlib.metadata as metadata
import os, shutil, subprocess, sys

def pip_install(args, allow_fail=False):
    print("INSTALL:", " ".join(args))
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args, text=True, capture_output=True)
    if result.returncode != 0:
        print("INSTALL FALHOU:", " ".join(args))
        tail = (result.stdout + "\n" + result.stderr).strip().splitlines()[-30:]
        print("\n".join(tail))
        if not allow_fail:
            raise SystemExit("pip install failed: " + " ".join(args))
    return result.returncode

pip_install(["-U", "pip", "setuptools", "wheel"])
pip_install(["packaging"])

# Em Colab A100/H100, nao reinstale torch se ele ja veio com CUDA funcionando.
# Reinstalar torch via PyPI pode trocar a build CUDA e quebrar o runtime.
try:
    import torch
    print("Torch preinstalado:", torch.__version__, "| CUDA:", torch.version.cuda, "| available:", torch.cuda.is_available())
except Exception:
    pip_install(["torch"])

pip_install([
    "transformers==4.48.0",
    "peft==0.14.0",
    "trl==0.13.0",
    "datasets==3.2.0",
    "accelerate==1.2.1",
    "huggingface_hub==0.27.1",
    "safetensors==0.4.5",
    "pandas",
    "matplotlib",
    "sentencepiece",
    "einops",
    "ninja",
])

# bitsandbytes e necessario apenas para QLoRA. No Colab atual com torch 2.10/cu128,
# bnb 0.45 pode instalar sem binario CUDA e quebrar o PEFT mesmo no modo BF16
# com "No module named 'triton.ops'". Removemos qualquer resto aqui; QLoRA instala
# bnb sob demanda no bloco de load.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"], text=True, capture_output=True)
print("CLEAN: bitsandbytes removido do caminho BF16; sera instalado apenas se LOAD_MODE=qlora_4bit.")

# Nemotron remote code precisa de mamba-ssm e pode usar causal-conv1d.
# causal-conv1d e apenas fast-path opcional; no Colab com torch/CUDA novos ele
# pode ficar compilando por muitos minutos. Pulamos esse build e usamos o
# fallback injetado antes do load do modelo.
os.environ.setdefault("MAX_JOBS", "4")
if GPU_TIER.startswith("a100"):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
elif GPU_TIER == "h100":
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
if shutil.which("nvcc"):
    print("NVCC:", subprocess.run(["nvcc", "--version"], text=True, capture_output=True).stdout.splitlines()[-1])
else:
    print("AVISO: nvcc nao encontrado. causal-conv1d pode falhar e o notebook usara fallback lento.")

SKIP_CAUSAL_CONV1D_BUILD = True
if SKIP_CAUSAL_CONV1D_BUILD:
    print("SKIP: causal-conv1d opcional; evitando build CUDA longo. Sera usado fallback sem fast-path.")
else:
    pip_install(["causal-conv1d", "--no-build-isolation"], allow_fail=True)
pip_install(["mamba-ssm", "--no-build-isolation"], allow_fail=True)

def check_import(label, code, required=False):
    try:
        exec(code, {})
        print(f"{label}: import OK")
        return True
    except Exception as exc:
        print(f"{label}: import FALHOU ({type(exc).__name__}: {str(exc)[:240]})")
        if required:
            raise RuntimeError(f"Dependencia obrigatoria ausente: {label}") from exc
        return False

CAUSAL_CONV1D_READY = check_import("causal_conv1d", "import causal_conv1d", required=False)
MAMBA_SSM_READY = check_import(
    "mamba_ssm.rmsnorm_fn",
    "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn",
    required=True,
)
if not CAUSAL_CONV1D_READY:
    print("AVISO: causal_conv1d ausente. O load do Nemotron sera liberado com fallback sem fast-path.")

for pkg in ["torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes", "mamba-ssm", "causal-conv1d"]:
    try:
        print(f"{pkg}:", metadata.version(pkg))
    except Exception as exc:
        print(f"{pkg}: versao indisponivel ({exc})")
print("Dependencias instaladas.")
""", "install_deps"))

cells.append(code(r"""
#@title 2. Autenticacao HF e configuracao da run
import os, getpass
from pathlib import Path
from huggingface_hub import login, HfApi

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None

HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = getpass.getpass("Digite HF_TOKEN com permissao read/write no Hugging Face: ")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi(token=HF_TOKEN)
print("HF auth OK. Token carregado, nao impresso.")

HF_OWNER = "felipesp1983"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATA_FILENAME = "data/v60_selective_train_boxed_dedup.jsonl"
EXPECTED_DATA_SHA256 = "bfe2421b917d761c6528c4aa7bcecc105e8067381fd13a08e066a75cb7b8ad8f"

RUN_ID = "v61-sft-dedup-weakfocus-m2500-lr5e5-a32-d0-20260412-001"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v61-dedup-20260412-001"
OUTPUT_DIR = Path("/content/kg1_v61_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOAD_MODE = "auto"  # "auto", "bf16", "qlora_4bit"
SMOKE_TEST = False
SMOKE_N = 64

MAX_LENGTH = 2500
MAX_STEPS = 800
SAVE_STEPS = 100
NUM_TRAIN_EPOCHS = 1.0
SEED = 42

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SAVE_TOTAL_LIMIT = 5

print("RUN_ID:", RUN_ID)
print("OUTPUT_REPO:", OUTPUT_REPO)
""", "auth_config"))

cells.append(code(r"""
#@title 3. Baixar ou enviar dataset boxed
import json, hashlib
from pathlib import Path
from collections import Counter
from huggingface_hub import hf_hub_download

ALLOW_UPLOAD_LOCAL_DATASET = False  # use True se o arquivo ainda nao estiver no DATA_REPO

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

try:
    dataset_path = Path(hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=DATA_FILENAME,
        token=HF_TOKEN,
        local_dir="/content/kg1_data",
    ))
    print("Dataset baixado do HF:", dataset_path)
except Exception as exc:
    print("Falha ao baixar dataset do HF:", type(exc).__name__, str(exc)[:300])
    if not ALLOW_UPLOAD_LOCAL_DATASET:
        raise SystemExit("Ative ALLOW_UPLOAD_LOCAL_DATASET=True para enviar o JSONL manualmente.")
    from google.colab import files
    uploaded = files.upload()
    if not uploaded:
        raise SystemExit("Nenhum arquivo enviado.")
    dataset_path = Path("/content") / next(iter(uploaded.keys()))
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo=DATA_FILENAME,
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message=f"{RUN_ID} upload boxed dataset",
        token=HF_TOKEN,
    )
    print("Dataset publicado no HF dataset repo.")

observed_sha = sha256_file(dataset_path)
print("Dataset SHA256:", observed_sha)
if observed_sha != EXPECTED_DATA_SHA256:
    raise SystemExit(f"Dataset hash mismatch. Esperado {EXPECTED_DATA_SHA256}, observado {observed_sha}")

examples = []
with dataset_path.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        if line.strip():
            item = json.loads(line)
            item["_line_no"] = line_no
            examples.append(item)

print("Exemplos:", len(examples))
print("Familias:", dict(Counter(ex.get("family", "unknown") for ex in examples)))
print("Fontes:", dict(Counter(ex.get("source", "unknown") for ex in examples)))
""", "dataset_download"))

cells.append(code(r"""
#@title 4. Gate de dataset e parser boxed
import re, json
from collections import Counter

FAMILIES = {"gravity_constant", "unit_conversion", "numeral_system", "text_encryption", "bit_manipulation", "equation_transform"}

def is_escaped(text: str, idx: int) -> bool:
    n = 0
    j = idx - 1
    while j >= 0 and text[j] == "\\":
        n += 1
        j -= 1
    return n % 2 == 1

def extract_boxed(text: str | None) -> str | None:
    if not text:
        return None
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace_start = text.find("{", start)
    depth = 0
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if ch in "{}" and is_escaped(text, idx):
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:idx]
    return None

def unescape_latex(value: str) -> str:
    return value.strip().replace(r"\{", "{").replace(r"\}", "}").replace(r"\\", "\\")

def answers_match(expected: str, observed: str, rel_tol: float = 1e-2) -> bool:
    exp, obs = unescape_latex(str(expected)), unescape_latex(str(observed))
    if exp == obs:
        return True
    try:
        exp_f, obs_f = float(exp), float(obs)
    except Exception:
        return False
    return abs(exp_f - obs_f) / max(abs(exp_f), 1e-12) <= rel_tol

parser_cases = [
    ("basic_numeric", r"\boxed{123}", "123", True),
    ("escaped_closing_brace", r"\boxed{\}}", "}", True),
    ("starts_with_closing_brace", r"\boxed{\}abc}", "}abc", True),
    ("nested_unescaped_braces", r"\boxed{a{b}c}", "a{b}c", True),
]
for name, text, expected, should_match in parser_cases:
    obs = extract_boxed(text)
    ok = obs is not None and answers_match(expected, obs)
    print(name, "observed=", obs, "ok=", ok)
    if should_match and not ok:
        raise SystemExit(f"Parser robusto falhou no caso {name}")

def assistant_text(ex: dict) -> str:
    if ex.get("text"):
        return str(ex["text"])
    for msg in reversed(ex.get("messages", [])):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""

bad = []
ids = Counter()
by_family = Counter()
by_source = Counter()
boxed = 0
correctness_false = 0
for i, ex in enumerate(examples, start=1):
    ex_id = str(ex.get("id", f"line-{i}"))
    ids[ex_id] += 1
    fam = ex.get("family", "unknown")
    by_family[fam] += 1
    by_source[ex.get("source", "unknown")] += 1
    if fam not in FAMILIES:
        bad.append({"line": i, "id": ex_id, "reason": "unknown_family"})
    if str(ex.get("correctness", "")).lower() == "false":
        correctness_false += 1
        bad.append({"line": i, "id": ex_id, "reason": "correctness_false"})
    if extract_boxed(assistant_text(ex)) is None:
        bad.append({"line": i, "id": ex_id, "reason": "missing_or_unparseable_boxed"})
    else:
        boxed += 1

duplicate_ids = [k for k, v in ids.items() if v > 1]
gate = {
    "dataset_path": str(dataset_path),
    "sha256": observed_sha,
    "total": len(examples),
    "boxed_rate": boxed / max(len(examples), 1),
    "parse_errors": sum(1 for x in bad if x["reason"] == "missing_or_unparseable_boxed"),
    "correctness_false": correctness_false,
    "duplicate_id_count": len(duplicate_ids),
    "by_family": dict(sorted(by_family.items())),
    "by_source": dict(sorted(by_source.items())),
    "bad_sample": bad[:20],
}
gate_path = ARTIFACT_DIR / "v61_colab_dataset_parser_gate.json"
gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
print(json.dumps(gate, indent=2)[:4000])
if gate["boxed_rate"] != 1.0 or gate["parse_errors"] != 0 or gate["correctness_false"] != 0:
    raise SystemExit(f"Dataset/parser gate failed. Ver {gate_path}")
print("Gate OK:", gate_path)
""", "dataset_parser_gate"))

cells.append(code(r"""
#@title 5. Preparar tokenizer e Dataset
import inspect, torch
from datasets import Dataset
from transformers import AutoTokenizer

random.seed(SEED)
torch.manual_seed(SEED)

if SMOKE_TEST:
    random.shuffle(examples)
    examples_for_training = examples[:SMOKE_N]
    effective_max_steps = min(MAX_STEPS, 10)
    print("SMOKE_TEST ativo:", len(examples_for_training), "exemplos,", effective_max_steps, "steps")
else:
    examples_for_training = examples
    effective_max_steps = MAX_STEPS

try:
    info = api.model_info(BASE_MODEL)
    MODEL_REVISION = info.sha or None
except Exception as exc:
    print("AVISO: nao consegui resolver SHA do modelo:", type(exc).__name__, str(exc)[:200])
    MODEL_REVISION = None
print("MODEL_REVISION:", MODEL_REVISION)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, trust_remote_code=True, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

texts = []
for ex in examples_for_training:
    if ex.get("text"):
        texts.append(ex["text"])
    else:
        texts.append(tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False))

train_ds = Dataset.from_dict({"text": texts})
print(train_ds)
print("Primeiro exemplo:", texts[0][:300].replace("\n", " "))
""", "prepare_dataset"))

cells.append(code(r"""
#@title 6. Carregar modelo e aplicar LoRA/QLoRA
import torch, gc, sys, types, importlib.machinery, subprocess, time
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if LOAD_MODE == "auto":
    effective_load_mode = "bf16" if GPU_MEM_GB >= 70 else "qlora_4bit"
else:
    effective_load_mode = LOAD_MODE
print("LOAD_MODE efetivo:", effective_load_mode)

gc.collect()
torch.cuda.empty_cache()

model_kwargs = {
    "revision": MODEL_REVISION,
    "trust_remote_code": True,
    "token": HF_TOKEN,
    "torch_dtype": torch.bfloat16,
}
if effective_load_mode == "qlora_4bit":
    print("QLoRA ativo: instalando bitsandbytes sob demanda.")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], text=True, capture_output=True)
    if result.returncode != 0:
        print("\n".join((result.stdout + "\n" + result.stderr).strip().splitlines()[-30:]))
        raise RuntimeError("Falha ao instalar bitsandbytes para QLoRA.")
    try:
        import bitsandbytes as bnb  # noqa: F401
        print("bitsandbytes import OK para QLoRA.")
    except Exception as exc:
        raise RuntimeError("bitsandbytes nao importou; use A100 80GB/H100 em BF16 ou ajuste versao bnb.") from exc
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs.update({"device_map": "auto", "quantization_config": bnb_config})
else:
    model_kwargs.update({"device_map": {"": 0}})
    try:
        import peft.tuners.lora.model as peft_lora_model
        peft_lora_model.is_bnb_available = lambda: False
        print("PEFT: bitsandbytes desativado para BF16; usando Linear/AdamW torch.")
    except Exception as exc:
        print("AVISO: nao consegui desativar bnb no PEFT:", type(exc).__name__, str(exc)[:160])

try:
    import causal_conv1d  # noqa: F401
    CAUSAL_CONV1D_READY = True
except Exception:
    CAUSAL_CONV1D_READY = False

if not CAUSAL_CONV1D_READY:
    # O arquivo remoto do Nemotron tem import de causal_conv1d protegido por
    # is_causal_conv1d_available(), mas o validador dinamico do Transformers
    # checa imports de forma textual e bloqueia antes do fallback lento.
    fake_causal_conv1d = types.ModuleType("causal_conv1d")
    fake_causal_conv1d.causal_conv1d_fn = None
    fake_causal_conv1d.causal_conv1d_update = None
    fake_causal_conv1d.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", loader=None)
    sys.modules.setdefault("causal_conv1d", fake_causal_conv1d)
    print("AVISO: causal_conv1d nao importou; liberando load do Nemotron com fallback sem fast-path.")

TRANSIENT_DOWNLOAD_MARKERS = (
    "IncompleteRead",
    "ChunkedEncodingError",
    "ProtocolError",
    "Connection broken",
    "ConnectionResetError",
    "Read timed out",
    "RemoteDisconnected",
    "Temporary failure",
)

def is_transient_download_error(exc: Exception) -> bool:
    message = repr(exc)
    return any(marker in message for marker in TRANSIENT_DOWNLOAD_MARKERS)

def load_model_with_retries(max_attempts: int = 5):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"MODEL LOAD attempt {attempt}/{max_attempts}")
            return AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_transient_download_error(exc) or attempt == max_attempts:
                raise
            wait_seconds = min(90, 15 * attempt)
            print("AVISO: falha transiente no download/load do modelo:", type(exc).__name__, str(exc)[:500])
            print(f"Retry em {wait_seconds}s. O cache parcial do Hugging Face sera reutilizado.")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(wait_seconds)
    raise last_exc

model = load_model_with_retries()

DISABLE_NEMOTRON_FAST_PATH = False
if DISABLE_NEMOTRON_FAST_PATH:
    for module in model.modules():
        if hasattr(module, "is_fast_path_available"):
            module.is_fast_path_available = False
    print("Nemotron fast path desabilitado.")

if effective_load_mode == "qlora_4bit":
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model.enable_input_require_grads()

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
""", "load_model_lora"))

cells.append(code(r"""
#@title 7. Manifesto, Trainer e callback de upload adapter-only
import glob, json, time, shutil, inspect
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

MANIFEST_NAME = "run_manifest.json"
UPLOAD_ALLOW_FILES = {"adapter_config.json", "adapter_model.safetensors", "README.md", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", MANIFEST_NAME}

def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

dataset_meta = {
    "path": str(dataset_path),
    "hf_repo": DATA_REPO,
    "hf_filename": DATA_FILENAME,
    "sha256": observed_sha,
    "n_examples": len(examples),
    "by_family": gate["by_family"],
    "by_source": gate["by_source"],
}
training_params = {
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit" if effective_load_mode == "qlora_4bit" else "adamw_torch",
    "bf16": True,
    "logging_steps": 5,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "max_grad_norm": MAX_GRAD_NORM,
}
lora_meta = {"r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT, "target_modules": "all-linear", "bias": "none", "task_type": "CAUSAL_LM", "load_mode": effective_load_mode}

def latest_logged_loss(log_history):
    for item in reversed(log_history or []):
        if "loss" in item:
            return item["loss"]
    return None

def build_manifest(stage, checkpoint_step=None, latest_loss=None, elapsed_seconds=None):
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": stage,
        "created_at": utc_now(),
        "output_repo": OUTPUT_REPO,
        "artifact": {"checkpoint_step": checkpoint_step, "path_in_repo": f"checkpoint-{checkpoint_step}" if checkpoint_step is not None else "final"},
        "base_model": {"repo_id": BASE_MODEL, "revision": MODEL_REVISION or "unresolved", "trust_remote_code": True},
        "dataset": dataset_meta,
        "parser_gate": {"boxed_rate": gate["boxed_rate"], "parse_errors": gate["parse_errors"], "robust_gate_passed": True},
        "lora": lora_meta,
        "training": {**training_params, "max_steps": effective_max_steps, "num_train_epochs": NUM_TRAIN_EPOCHS, "max_length": MAX_LENGTH, "seed": SEED, "latest_loss": latest_loss, "elapsed_seconds": elapsed_seconds},
        "promotion_gates": {"baseline_v30_loss": 6.87, "promote_if_final_loss_below": 6.87, "kill_if_loss_above_after_step_400": 10.0, "minimum_local_metric_accuracy": 0.69},
        "notes": ["Adapter-only upload. Do not submit mutable repo HEAD.", "No secret is embedded in this manifest."],
    }

def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def validate_adapter_dir(path: Path):
    missing = [name for name in ("adapter_config.json", "adapter_model.safetensors") if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} missing adapter files: {missing}")
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if not isinstance(cfg.get("r"), int) or cfg["r"] > 32:
        raise ValueError(f"LoRA rank must be <=32; observed r={cfg.get('r')}")

def stage_adapter_only(source_dir: Path, stage_dir: Path, manifest: dict) -> Path:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for file in source_dir.iterdir():
        if file.is_file() and file.name in UPLOAD_ALLOW_FILES:
            shutil.copy2(file, stage_dir / file.name)
    write_json(stage_dir / MANIFEST_NAME, manifest)
    validate_adapter_dir(stage_dir)
    return stage_dir

class AdapterUploadCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        loss = latest_logged_loss(state.log_history)
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.exists():
            candidates = sorted(glob.glob(f"{args.output_dir}/checkpoint-*"))
            if not candidates:
                return
            checkpoint_dir = Path(candidates[-1])
        manifest = build_manifest(stage="checkpoint", checkpoint_step=step, latest_loss=loss)
        stage_dir = stage_adapter_only(checkpoint_dir, Path(f"/content/kg1_upload_checkpoint_{step}"), manifest)
        api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(stage_dir), path_in_repo=f"checkpoint-{step}", repo_id=OUTPUT_REPO, commit_message=f"{RUN_ID} checkpoint-{step} loss={loss}", token=HF_TOKEN)
        print(f"UPLOAD OK: checkpoint-{step}, loss={loss}, repo={OUTPUT_REPO}")

write_json(OUTPUT_DIR / MANIFEST_NAME, build_manifest(stage="pretrain"))
sft_params = inspect.signature(SFTConfig).parameters
length_key = "max_seq_length" if "max_seq_length" in sft_params else "max_length"
trainer_params = inspect.signature(SFTTrainer).parameters
tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"

training_args = SFTConfig(**{
    "output_dir": str(OUTPUT_DIR),
    "dataset_text_field": "text",
    length_key: MAX_LENGTH,
    "packing": False,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "max_steps": effective_max_steps,
    "save_strategy": "steps",
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "report_to": "none",
    "dataloader_num_workers": 0,
    **training_params,
})
trainer = SFTTrainer(model=model, train_dataset=train_ds, **{tok_key: tokenizer}, args=training_args, callbacks=[AdapterUploadCallback()])
print("Trainer pronto. Manifesto pretrain:", OUTPUT_DIR / MANIFEST_NAME)
""", "trainer_setup"))

cells.append(code(r"""
#@title 8. Treinar
import time, pandas as pd

start = time.time()
train_result = trainer.train()
elapsed = time.time() - start
final_loss = latest_logged_loss(trainer.state.log_history)

print(f"Treino concluido em {elapsed/3600:.2f}h")
print("Final loss:", final_loss)
print("Global step:", trainer.state.global_step)

history_path = ARTIFACT_DIR / "v61_train_log_history.json"
write_json(history_path, trainer.state.log_history)
loss_rows = [x for x in trainer.state.log_history if "loss" in x]
loss_csv = ARTIFACT_DIR / "v61_loss_curve.csv"
pd.DataFrame(loss_rows).to_csv(loss_csv, index=False)
print("Log history:", history_path)
print("Loss CSV:", loss_csv)
""", "train"))

cells.append(code(r"""
#@title 9. Salvar final adapter-only e publicar no HF
import shutil
from pathlib import Path

final_dir = OUTPUT_DIR / "final_adapter"
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

final_manifest = build_manifest(stage="final", checkpoint_step=int(trainer.state.global_step), latest_loss=final_loss, elapsed_seconds=elapsed)
final_stage_dir = stage_adapter_only(final_dir, Path("/content/kg1_upload_final_adapter"), final_manifest)
api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(final_stage_dir),
    path_in_repo="final",
    repo_id=OUTPUT_REPO,
    commit_message=f"{RUN_ID} final step={trainer.state.global_step} loss={final_loss}",
    token=HF_TOKEN,
)

zip_path = shutil.make_archive("/content/kg1_v61_final_adapter_only", "zip", final_stage_dir)
print("FINAL UPLOAD OK")
print("HF repo:", OUTPUT_REPO)
print("HF path:", "final")
print("Adapter-only zip:", zip_path)
print("Manifest:", final_stage_dir / MANIFEST_NAME)
""", "save_upload_final"))

cells.append(code(r"""
#@title 10. Gates pos-treino e proxima decisao
import json

try:
    final_loss_value = float(final_loss)
except Exception:
    final_loss_value = None

decision = {
    "run_id": RUN_ID,
    "output_repo": OUTPUT_REPO,
    "final_step": int(trainer.state.global_step),
    "final_loss": final_loss,
    "loss_gate_promote_below_6_87": bool(final_loss_value is not None and final_loss_value < 6.87),
    "loss_gate_kill_if_after_400_above_10": bool(final_loss_value is not None and trainer.state.global_step >= 400 and final_loss_value > 10.0),
    "dataset_sha256": observed_sha,
    "boxed_rate": gate["boxed_rate"],
    "parse_errors": gate["parse_errors"],
    "next_required_before_kaggle": [
        "baixar checkpoint escolhido do HF",
        "rodar metric gate local contra v30/v34/v50c",
        "gerar tabela por familia",
        "validar zip de submissao",
        "pin checkpoint no notebook Kaggle",
    ],
}
decision_path = ARTIFACT_DIR / "v61_post_train_decision.json"
write_json(decision_path, decision)
print(json.dumps(decision, indent=2))

if decision["loss_gate_kill_if_after_400_above_10"]:
    print("DECISAO: nao promover. Loss acima de 10 depois de 400 steps.")
elif decision["loss_gate_promote_below_6_87"]:
    print("DECISAO: candidato tecnico. Ainda precisa metric gate local e tabela por familia antes de Kaggle.")
else:
    print("DECISAO: diagnostico. Pode exigir ablação ou justificativa por family accuracy antes de submissao.")
""", "post_train_decision"))

cells.append(md(r"""
## Saida esperada

Ao fim do notebook:

- Repo HF privado com checkpoints adapter-only em `checkpoint-100`, `checkpoint-200`, ... e `final`
- Manifesto `run_manifest.json` em cada checkpoint
- `v61_loss_curve.csv` em `/content/kg1_v61_artifacts`
- `v61_post_train_decision.json` em `/content/kg1_v61_artifacts`
- Zip adapter-only em `/content/kg1_v61_final_adapter_only.zip`

Nao submeta ao Kaggle se loss final ficou acima de `6.87` sem ganho local por familia, se loss ficou acima de `10` depois de 400 steps, ou se ainda nao houve comparacao contra `v30`, `v34` e `v50c`.
"""))

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


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
