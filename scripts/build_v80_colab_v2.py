"""Build KG1_V80_COLAB_V2.ipynb - V80 dgxchen v7 EXACT + ALL 11 fixes consolidated.

V80 v1 -> v2: adicionados fixes descobertos na execução de hoje (22/04/2026):
  FIX #8: mamba-ssm NÃO instalado pelo Unsloth (NemotronH exige, sem slow path fallback)
  FIX #9: torch 2.10 cu128 (Colab default) incompatível com mamba-ssm wheels (ABI break)
  FIX #10: dataloader_num_workers=2 falha pickle CudaDeviceProperties em alguns envs
  FIX #11: Unsloth --upgrade puxa torch >=2.5 mais novo -> força torch 2.10 -> quebra mamba-ssm

Solução arquitetural: 2-cell split com RESTART obrigatório no meio (torch só recarrega em processo novo).
  - Cell 1: GPU + secrets + downgrade torch 2.5.1 + mamba-ssm cp312 wheel + causal-conv1d wheel -> RESTART
  - Cell 2: verify torch 2.5 + install ML stack (unsloth --no-deps) SEM upgrade torch
  - Cell 3: download dgxchen dataset + Nemotron base model
  - Cell 4: load model (attn=eager) + LoRA (8 targets NO lm_head)
  - Cell 5: train (num_workers=0, formatting_func, 1 epoch, max_grad_norm=1e9, grad_checkpoint=True)
  - Cell 6: package submission.zip + Kaggle submit

Matriz completa de 11 fixes vs V78/V79:
  1. Dataset problem_ids_matched.csv (V79 usou less_cot.csv com 6014 rows vs 7830)
  2. attn_implementation='eager' (V79 'sdpa')
  3. LoRA 8 items SEM lm_head (V79 9 com lm_head)
  4. num_train_epochs=1 (V79 2)
  5. max_grad_norm=1e9 (V79 1.0)
  6. gradient_checkpointing=True + use_reentrant=False no SFTConfig (V79 removido)
  7. formatting_func no trainer com conversation wrap (V79 removido)
  8. mamba-ssm + causal-conv1d install explicit (V79 esqueceu - NemotronH crashava)
  9. torch 2.5.1 pin PRE-Unsloth (V79 tinha torch 2.10 que quebra mamba-ssm ABI)
 10. dataloader_num_workers=0 (prev pickle CudaDeviceProperties)
 11. Unsloth install com --no-deps (prev torch upgrade accidental)
"""
import json
import os

HEADER = r"""# KG1 V80 COLAB v2 - dgxchen v7 EXACT Replica + ALL FIXES

## Source (verified via Kaggle MCP 22/04/2026)
Kernel: `dgxchen/training-with-unsloth-to-achieve-0-85-lb` (**0.85 LB**, 185 votos)

## Execution flow (IMPORTANT)
1. **Cell 1**: setup + torch downgrade + mamba-ssm install. **PEDE RESTART** no final.
2. **[RESTART RUNTIME]** (Runtime > Restart runtime ou Ctrl+M .)
3. **Cell 2**: verify torch 2.5 + install ML stack (unsloth --no-deps, no torch upgrade)
4. **Cell 3**: download dgxchen dataset + Nemotron-30B base model (~60GB)
5. **Cell 4**: load model (Unsloth attn=eager) + LoRA (r=32 alpha=32 8 targets)
6. **Cell 5**: build SFT dataset + train (1 epoch, ~2-3h H100)
7. **Cell 6**: package submission.zip + Kaggle submit + HF backup

## Fixes consolidados (11 total)

### Fixes V79 -> V80 (divergences dgxchen v7 revertidas):
| # | Param | V79 (wrong) | V80 = dgxchen v7 |
|---|---|---|---|
| 1 | Dataset | less_cot.csv (6014 rows) | **problem_ids_matched.csv (7830 rows)** |
| 2 | attn_implementation | 'sdpa' | **'eager'** |
| 3 | LoRA targets | 9 (com lm_head) | **8 (SEM lm_head)** |
| 4 | num_train_epochs | 2 | **1** |
| 5 | max_grad_norm | 1.0 | **1e9** |
| 6 | grad_checkpoint SFTConfig | removido | **True + use_reentrant=False** |
| 7 | formatting_func | removido | **presente + conversation wrap** |

### Fixes V80 v1 -> v2 (descobertos em execução 22/04):
| # | Problema | Solução |
|---|---|---|
| 8 | mamba-ssm não instalado | wheel cp312+cu12+torch2.5 em Cell 1 |
| 9 | torch 2.10 quebra mamba-ssm ABI | downgrade torch 2.5.1 com `--force-reinstall` |
| 10 | pickle CudaDeviceProperties fail | `dataloader_num_workers=0` |
| 11 | Unsloth upgrade puxa torch | `--no-deps` no unsloth install |

## Target
- Colab Pro+ H100 80GB
- Budget: ~$30-50 (3-4h)
- Expected score: **0.84-0.85** (replica exata do dgxchen v7)

## Credenciais (Colab Secrets - chave do cadeado à esquerda)
- `HF_KEY` - HuggingFace token
- `KAGGLE_USERNAME` = felipe1983
- `KAGGLE_KEY` = (do kaggle.json)
"""

CELL_1_SETUP = r"""# CELL 1 - Setup + torch downgrade + mamba-ssm install -> REQUER RESTART
# ATENÇÃO: ao final dessa célula VOCÊ PRECISA CLICAR Runtime > Restart runtime
# (ou Ctrl+M .) para carregar torch 2.5.1. Depois rode a Cell 2.
import subprocess, sys, os, json, gc, time
from pathlib import Path

print('=' * 70)
print('V80 COLAB v2 CELL 1 - Setup + torch 2.5 pin + mamba-ssm + causal-conv1d')
print('=' * 70)

# Check if already on torch 2.5 (post-restart re-run safety)
import torch as _torch
if _torch.__version__.startswith('2.5'):
    print(f'\ntorch {_torch.__version__} already pinned. Cell 1 already ran.')
    print('Go directly to Cell 2.')
    # Don't raise - just exit soft
    import sys as _sys
    _sys.exit(0) if False else None  # keep flow for idempotency

# GPU check
import torch
assert torch.cuda.is_available(), 'CUDA required'
d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
print(f'GPU: {d.name}')
print(f'VRAM: {total_gb:.1f}GB total')
assert total_gb >= 75, f'Need H100 80GB, got {total_gb:.1f}GB'
print(f'torch current: {torch.__version__} (will downgrade to 2.5.1)')

# Colab secrets
try:
    from google.colab import userdata
    hf_key = userdata.get('HF_KEY')
    kaggle_user = userdata.get('KAGGLE_USERNAME')
    kaggle_key = userdata.get('KAGGLE_KEY')
except Exception:
    hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    kaggle_user = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

assert hf_key, 'HF_KEY missing - add in Colab Secrets'
assert kaggle_user and kaggle_key, 'KAGGLE_USERNAME/KAGGLE_KEY missing'

# Save kaggle.json
kpath = Path.home() / '.kaggle' / 'kaggle.json'
kpath.parent.mkdir(parents=True, exist_ok=True)
kpath.write_text(json.dumps({'username': kaggle_user, 'key': kaggle_key}))
kpath.chmod(0o600)

# Persist secrets for Cell 2+ (survives restart)
Path('/content/kg1_secrets.json').write_text(json.dumps({
    'hf_key': hf_key, 'kaggle_user': kaggle_user, 'kaggle_key': kaggle_key,
}))
print(f'HF user token: ...{hf_key[-8:]}')
print(f'Kaggle user: {kaggle_user}')
print(f'Secrets saved to /content/kg1_secrets.json (survives restart)')

# ============================================================
# Step 1: Downgrade torch to 2.5.1 (compatible with mamba-ssm wheels)
# ============================================================
print('\n[1/3] Downgrading torch -> 2.5.1 + cu124 (~3-5min)...')
r = subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
    'torch==2.5.1', 'torchvision==0.20.1', 'torchaudio==2.5.1',
    '--index-url', 'https://download.pytorch.org/whl/cu124',
], capture_output=True, text=True, timeout=900)
print(f'  rc={r.returncode}')
if r.returncode != 0:
    print(f'  stderr: {r.stderr[-500:]}')
    raise RuntimeError('torch downgrade failed')

# ============================================================
# Step 2: Install mamba-ssm wheel (cp312+cu12+torch2.5) — REQUIRED by NemotronH
# ============================================================
print('\n[2/3] Install mamba-ssm 2.2.4 cp312 wheel...')
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
    'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'],
    capture_output=True, text=True, timeout=300)
print(f'  rc={r.returncode}')
if r.returncode != 0:
    print(f'  stderr: {r.stderr[-500:]}')
    # Try v2.3.0 as fallback
    print('  Trying mamba-ssm v2.3.0 fallback...')
    r2 = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
        'https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'],
        capture_output=True, text=True, timeout=300)
    if r2.returncode != 0:
        raise RuntimeError(f'mamba-ssm install failed: {r2.stderr[-500:]}')

# ============================================================
# Step 3: Install causal-conv1d wheel
# ============================================================
print('\n[3/3] Install causal-conv1d 1.5.0.post8 cp312 wheel...')
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'],
    capture_output=True, text=True, timeout=300)
print(f'  rc={r.returncode}')
# Non-fatal - NemotronH has fallback without causal-conv1d

# Clean up
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# RESTART REQUIRED - torch cannot reload in same process
# ============================================================
print('\n' + '=' * 70)
print('CELL 1 DONE. *** RESTART RUNTIME NOW ***')
print('=' * 70)
print()
print('  AÇÃO OBRIGATÓRIA:')
print('  1. Menu: Runtime > Restart runtime   (ou Ctrl+M .)')
print('  2. Após restart, execute Cell 2')
print()
print('  (NÃO rode Cell 2 antes de restart - vai falhar com ABI mismatch)')
print('=' * 70)

# Programmatic restart attempt (Colab-compatible)
try:
    import IPython
    print('\n  Tentando restart programático em 5s...')
    time.sleep(5)
    IPython.Application.instance().kernel.do_shutdown(True)  # True = restart kernel
except Exception as e:
    print(f'\n  Restart programático falhou: {e}')
    print('  Por favor clique Runtime > Restart runtime manualmente.')
"""

CELL_2_VERIFY_AND_ML = r"""# CELL 2 - Post-restart: verify torch 2.5 + install ML stack (no torch upgrade)
import subprocess, sys, os, json, gc
from pathlib import Path

print('=' * 70)
print('V80 COLAB v2 CELL 2 - Verify + install ML stack (unsloth --no-deps)')
print('=' * 70)

# Load secrets saved by Cell 1
secrets_path = Path('/content/kg1_secrets.json')
if not secrets_path.exists():
    raise RuntimeError('Secrets file missing. Re-run Cell 1 first.')
secrets = json.loads(secrets_path.read_text())
os.environ.update({
    'HF_TOKEN': secrets['hf_key'],
    'HF_KEY': secrets['hf_key'],
    'KAGGLE_USERNAME': secrets['kaggle_user'],
    'KAGGLE_KEY': secrets['kaggle_key'],
})

# Verify torch 2.5
import torch
print(f'torch: {torch.__version__} cuda_avail={torch.cuda.is_available()}')
assert torch.__version__.startswith('2.5'), (
    f'Need torch 2.5, got {torch.__version__}. Did you restart runtime after Cell 1?'
)

# Verify mamba-ssm (installed in Cell 1)
print('\nVerifying mamba-ssm (NemotronH requirement)...')
import mamba_ssm
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print(f'  mamba_ssm={mamba_ssm.__version__} OK')
try:
    import causal_conv1d
    print(f'  causal_conv1d OK')
except ImportError:
    print(f'  causal_conv1d MISSING (non-fatal)')

# ============================================================
# Install ML stack (transformers, peft, trl, accelerate, datasets)
# Pinned to torch 2.5 compatible ranges - no torch upgrade
# ============================================================
print('\nInstalling ML stack (transformers/peft/trl/accelerate/datasets)...')
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.48,<4.58',
    'peft>=0.14,<0.18',
    'trl>=0.14,<0.26',
    'accelerate>=1.0,<2.0',
    'datasets>=3.2,<5',
    'bitsandbytes',
    'huggingface_hub', 'safetensors', 'einops', 'sentencepiece',
    'pandas', 'kagglehub',
], capture_output=True, text=True, timeout=600)
print(f'  rc={r.returncode}')
if r.returncode != 0:
    print(f'  stderr: {r.stderr[-500:]}')
    raise RuntimeError('ML stack install failed')

# ============================================================
# Install Unsloth with --no-deps (prevents torch upgrade)
# ============================================================
print('\nInstalling Unsloth + unsloth_zoo (--no-deps)...')
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    'unsloth', 'unsloth_zoo'],
    capture_output=True, text=True, timeout=300)
print(f'  rc={r.returncode}')
if r.returncode != 0:
    print(f'  WARN: {r.stderr[-400:]}')
    print('  (will try fallback pure transformers+peft in Cell 4 if unsloth import fails)')

# Unsloth deps that don't pull torch
print('\nInstalling unsloth helpers (tyro, hf_transfer)...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'tyro', 'hf_transfer'],
    capture_output=True, text=True, timeout=180)

# xformers may try to upgrade torch - install --no-deps
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'xformers'],
    capture_output=True, text=True, timeout=180)

# Safety check: torch still 2.5?
import importlib, torch
importlib.reload(torch)
if not torch.__version__.startswith('2.5'):
    print(f'\n  WARN: torch upgraded to {torch.__version__} - force re-pinning...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
        'torch==2.5.1', 'torchvision==0.20.1', 'torchaudio==2.5.1',
        '--index-url', 'https://download.pytorch.org/whl/cu124'], timeout=600)
    print('  torch re-pinned. RESTART RUNTIME AGAIN and re-run Cell 2.')
    raise RuntimeError('torch was upgraded - restart required')

# ============================================================
# Final verify: all critical imports work
# ============================================================
print('\nFinal verify - all critical imports:')
import transformers, peft, trl, accelerate, datasets
print(f'  transformers={transformers.__version__}')
print(f'  peft={peft.__version__}')
print(f'  trl={trl.__version__}')
print(f'  accelerate={accelerate.__version__}')
print(f'  datasets={datasets.__version__}')

try:
    from unsloth import FastLanguageModel
    print(f'  unsloth.FastLanguageModel OK')
    UNSLOTH_OK = True
except Exception as e:
    print(f'  WARN unsloth import failed: {e}')
    print('  Cell 4 will use fallback pure transformers+peft (slower but works)')
    UNSLOTH_OK = False

# Persist for Cell 4
Path('/content/kg1_state.json').write_text(json.dumps({'unsloth_ok': UNSLOTH_OK}))

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print('\n' + '=' * 70)
print('STACK READY - proceed to Cell 3')
print('=' * 70)
"""

CELL_3_DATA = r"""# CELL 3 - Download dgxchen dataset (problem_ids_matched.csv) + Nemotron-30B base model
import os, subprocess, sys, json
from pathlib import Path

print('=' * 70)
print('V80 CELL 3 - Download dgxchen/nemotron-cot-tong + Nemotron base')
print('Using problem_ids_matched.csv (7830 rows, dgxchen v7 EXACT)')
print('=' * 70)

# Ensure secrets
secrets = json.loads(Path('/content/kg1_secrets.json').read_text())
os.environ.update({
    'HF_TOKEN': secrets['hf_key'],
    'HF_KEY': secrets['hf_key'],
    'KAGGLE_USERNAME': secrets['kaggle_user'],
    'KAGGLE_KEY': secrets['kaggle_key'],
})
# Re-write kaggle.json (in case /root/.kaggle was wiped)
kpath = Path.home() / '.kaggle' / 'kaggle.json'
kpath.parent.mkdir(parents=True, exist_ok=True)
kpath.write_text(json.dumps({'username': secrets['kaggle_user'], 'key': secrets['kaggle_key']}))
kpath.chmod(0o600)

DATA_DIR = Path('/content/kg1_data')
DATA_DIR.mkdir(exist_ok=True)

# Download dgxchen dataset (if not already cached)
target_csv = DATA_DIR / 'problem_ids_matched.csv'
if not target_csv.exists() or target_csv.stat().st_size < 40_000_000:
    print('Downloading dgxchen/nemotron-cot-tong (~90MB)...')
    r = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', 'dgxchen/nemotron-cot-tong',
         '-p', str(DATA_DIR), '--unzip'],
        capture_output=True, text=True, timeout=300,
    )
    print(r.stdout)
    if r.returncode != 0:
        print('stderr:', r.stderr[-500:])
        raise RuntimeError('Kaggle download failed')
else:
    print(f'Dataset already cached: {target_csv.stat().st_size/(1024**2):.1f}MB')

# Verify problem_ids_matched.csv
assert target_csv.exists(), 'problem_ids_matched.csv missing after download'
print(f'V80 target path: {target_csv}')
print(f'Size: {target_csv.stat().st_size/(1024**2):.1f}MB')

import pandas as pd
df = pd.read_csv(target_csv)
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
if 'type' in df.columns:
    print(f'Type distribution:')
    for t, n in df['type'].value_counts().items():
        print(f'  {t}: {n}')

# Base model download via kagglehub
import kagglehub
MODEL_CACHE = '/root/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1'
if Path(MODEL_CACHE).exists() and len(list(Path(MODEL_CACHE).iterdir())) > 5:
    print(f'\nBase model already cached at: {MODEL_CACHE}')
    MODEL_PATH = MODEL_CACHE
else:
    print('\nDownloading base model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (~60GB, ~50min)...')
    MODEL_PATH = kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')
print(f'Model path: {MODEL_PATH}')

# Persist state for subsequent cells
state_path = Path('/content/kg1_state.json')
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state.update({'model_path': str(MODEL_PATH), 'target_csv': str(target_csv)})
state_path.write_text(json.dumps(state))

print('\nDATA READY - proceed to Cell 4')
"""

CELL_4_MODEL_LORA = r"""# CELL 4 - Load Nemotron-30B + LoRA (dgxchen v7 EXACT: attn=eager, 8 targets NO lm_head)
import json, torch, gc
from pathlib import Path

print('=' * 70)
print('V80 CELL 4 - Load model + LoRA (attn=eager, r=32 alpha=32, 8 targets NO lm_head)')
print('=' * 70)

state = json.loads(Path('/content/kg1_state.json').read_text())
MODEL_PATH = state['model_path']
UNSLOTH_OK = state.get('unsloth_ok', True)

MAX_SEQ_LEN = 8192
print(f'Loading {MODEL_PATH}...')
print(f'Using: {"Unsloth FastLanguageModel" if UNSLOTH_OK else "pure transformers (fallback)"}')

if UNSLOTH_OK:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation='eager',  # V80 FIX #2: dgxchen uses eager (not sdpa)
        dtype=torch.bfloat16,
    )
else:
    # Fallback: pure transformers+peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        trust_remote_code=True,
        device_map={'': 0},
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('Base model loaded.')

# dgxchen v7 EXACT LoRA config - 8 targets NO lm_head (V80 FIX #3)
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
target_modules = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'in_proj', 'out_proj', 'up_proj', 'down_proj',
    # lm_head REMOVED (V79 had it - divergence)
]

print(f'\nApplying LoRA r={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}')
print(f'Target modules ({len(target_modules)} items, NO lm_head): {target_modules}')

if UNSLOTH_OK:
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=42,
    )
else:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
print(f'\nAfter LoRA: used={used_gb:.1f}GB free={free_gb:.1f}GB')

# Store for Cell 5
import builtins
builtins._v80_model = model
builtins._v80_tokenizer = tokenizer
builtins._v80_unsloth = UNSLOTH_OK

gc.collect()
torch.cuda.empty_cache()
print('\nLORA APPLIED - proceed to Cell 5 (training)')
"""

CELL_5_TRAIN = r"""# CELL 5 - Build SFT dataset + Train (dgxchen v7 EXACT + all training fixes)
import builtins, pandas as pd, random, re, math, gc, time, json, torch
from collections import defaultdict, deque
from pathlib import Path
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Sampler
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

print('=' * 70)
print('V80 CELL 5 - Train (1 epoch, eff_batch=32, lr=2e-4, max_grad_norm=1e9)')
print('=' * 70)

model = builtins._v80_model
tokenizer = builtins._v80_tokenizer
UNSLOTH_OK = builtins._v80_unsloth

state = json.loads(Path('/content/kg1_state.json').read_text())
target_csv = state['target_csv']

SEED = 42
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
OUTPUT_DIR = '/content/kg1_out/sft_v80'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(target_csv)
print(f'Loaded: {len(df)} rows from problem_ids_matched.csv')
train_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

records, record_types = [], []
for _, row in train_df.iterrows():
    prompt = str(row['prompt'])
    answer = str(row['answer'])
    cot = str(row.get('generated_cot', ''))
    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        continue
    cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
    user_content = prompt + PROMPT_SUFFIX
    assistant_content = cot_cleaned + f'\n</think>\n\\boxed{{{answer}}}'
    records.append({'messages': [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content},
    ]})
    record_types.append(str(row.get('type', 'unknown')))
print(f'SFT records: {len(records)} (expected 7830 for problem_ids_matched.csv)')

dataset = HFDataset.from_list(records)


# V80 FIX #7: formatting_func with conversation wrap (dgxchen v7 EXACT)
def formatting_prompts_func(example):
    messages = example['messages']
    # Handle both flat dict list AND nested list-of-lists (batched)
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


# SFTConfig — dgxchen v7 EXACT (FIX #4, #5, #6 applied)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                          # FIX #4: dgxchen=1 (V79=2)
    per_device_train_batch_size=1,               # dgxchen
    gradient_accumulation_steps=32,              # dgxchen (eff_batch=32)
    learning_rate=2e-4,                          # dgxchen VERIFIED 0.85
    lr_scheduler_type='linear',                  # dgxchen
    warmup_steps=0,                              # dgxchen
    max_length=8192,                             # dgxchen
    adam_beta1=0.9,
    adam_beta2=0.95,                             # dgxchen
    adam_epsilon=1e-8,
    weight_decay=0.0,                            # dgxchen
    max_grad_norm=1e9,                           # FIX #5: dgxchen=1e9 (V79=1.0)
    logging_steps=5,
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    # FIX #6: V79 removed this, dgxchen v7 KEEPS it
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    # FIX #10: num_workers>=1 fails pickle CudaDeviceProperties in some envs
    dataloader_num_workers=0,
    remove_unused_columns=False,
    seed=SEED,
    report_to='none',
    packing=False,
)


# HealthGateCallback - 5-step logging + 6 automatic abort gates
class HealthGateCallback(TrainerCallback):
    def __init__(self):
        self.loss_history = deque(maxlen=10)
        self.grad_history = deque(maxlen=5)
        self.min_loss = float('inf')
        self.steps_since_improve = 0
        self.high_grad_count = 0
        self.train_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        print('=' * 70)
        print('V80 HEALTH GATES - log every 5 steps + abort on NaN/explosion')
        print('=' * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        loss = logs.get('loss')
        grad_norm = logs.get('grad_norm')
        lr = logs.get('learning_rate')
        epoch = logs.get('epoch', 0.0)

        # GATE 1: NaN/Inf
        if loss is not None and isinstance(loss, float):
            if math.isnan(loss) or math.isinf(loss):
                print(f'!!! CRITICAL NaN/Inf at step {step} - ABORTING')
                control.should_training_stop = True
                return

        # GATE 2: Explosion (V80 has max_grad_norm=1e9 = no clipping)
        if loss is not None and isinstance(loss, (int, float)):
            if loss > 30.0 and step > 10:
                print(f'!!! CRITICAL loss explosion {loss:.3f} at step {step} - ABORTING')
                control.should_training_stop = True
                return

        # GATE 3: Overfit warning
        if loss is not None and isinstance(loss, (int, float)):
            if loss < 0.01 and step < 50:
                print(f'!!! WARN severe overfit: loss={loss:.4f} at step {step}')

        if loss is not None:
            self.loss_history.append(float(loss))
            if float(loss) < self.min_loss:
                self.min_loss = float(loss)
                self.steps_since_improve = 0
            else:
                self.steps_since_improve += 5

        # GATE 4: High grad_norm (no clipping with 1e9!)
        if grad_norm is not None:
            self.grad_history.append(float(grad_norm))
            if float(grad_norm) > 50.0:
                self.high_grad_count += 1
                if self.high_grad_count >= 3:
                    print(f'!!! WARN grad_norm >50 3x (last={grad_norm:.2f}) no-clipping')
            else:
                self.high_grad_count = 0

        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            total = torch.cuda.mem_get_info()[1] / 1024**3
            used = total - free
            peak = torch.cuda.max_memory_allocated() / 1024**3
            if free < 2.0:
                print(f'!!! WARN VRAM free={free:.1f}GB OOM risk')
        else:
            free = used = peak = 0.0

        elapsed = time.time() - self.train_start if self.train_start else 0
        total_steps = state.max_steps if state.max_steps else 1
        progress = step / max(total_steps, 1)
        eta_sec = (elapsed / max(step, 1)) * (total_steps - step) if step > 0 else 0

        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0
        avg_grad = sum(self.grad_history) / len(self.grad_history) if self.grad_history else 0.0

        plateau_warn = ''
        if self.steps_since_improve >= 100:
            plateau_warn = f' [PLATEAU {self.steps_since_improve}s]'

        # Pre-compute safe f-string values
        _loss = float(loss) if loss is not None else 0.0
        _grad = float(grad_norm) if grad_norm is not None else 0.0
        _lr = float(lr) if lr is not None else 0.0
        _em = int(elapsed // 60)
        _etm = int(eta_sec // 60)

        print(
            f'[step {step:4d}/{total_steps:4d} ep{epoch:.2f} '
            f'{progress*100:5.1f}%] '
            f'loss={_loss:.4f} avg10={avg_loss:.4f} '
            f'grad={_grad:.3f} avg5={avg_grad:.3f} '
            f'lr={_lr:.2e} '
            f'vram={used:.1f}/{peak:.1f}/{free:.1f}GB '
            f'elapsed={_em}m ETA={_etm}m'
            f'{plateau_warn}'
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start if self.train_start else 0
        print('=' * 70)
        print(f'V80 TRAINING FINISHED - elapsed={elapsed/60:.1f}min '
              f'min_loss={self.min_loss:.4f} steps={state.global_step}')
        print('=' * 70)


# Stratified batching by type (dgxchen v7 uses this)
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
        raise ValueError('Stratified order size mismatch')
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
            raise ValueError('Trainer requires a train_dataset.')
        if self.stratified_order is None:
            return super().get_train_dataloader()
        dk = {
            'batch_size': self.args.per_device_train_batch_size,
            'sampler': PrecomputedOrderSampler(self.stratified_order),
            'collate_fn': self.data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'persistent_workers': self.args.dataloader_persistent_workers,
            'drop_last': self.args.dataloader_drop_last,
        }
        if self.args.dataloader_num_workers > 0:
            dk['prefetch_factor'] = self.args.dataloader_prefetch_factor
        return DataLoader(self.train_dataset, **dk)


eff_batch_size = 32
stratified_order = build_stratified_index_order(record_types, eff_batch_size, SEED)
print(f'\nEff batch size: {eff_batch_size}')
print(f'Total optim steps: ~{math.ceil(len(record_types)/eff_batch_size)} (1 epoch)')

health_cb = HealthGateCallback()

# V80 FIX #7: pass formatting_func to trainer (dgxchen v7 EXACT)
trainer = StratifiedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,  # FIX #7: present in dgxchen v7
    stratified_order=stratified_order,
    callbacks=[health_cb],
)

print('\nStarting V80 SFT training (dgxchen v7 EXACT replica)...')
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f'\nTraining done: {elapsed/60:.1f} min')
print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f}GB')

# Save adapter
ADAPTER_DIR = '/content/kg1_adapter_v80'
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f'\nAdapter saved: {ADAPTER_DIR}')

builtins._v80_adapter_dir = ADAPTER_DIR
print('PROCEED TO CELL 6 (package + submit)')
"""

CELL_6_SUBMIT = r"""# CELL 6 - Package submission.zip + Kaggle submit + HF backup
import os, json, shutil, zipfile, subprocess, datetime, builtins
from pathlib import Path

ADAPTER_DIR = getattr(builtins, '_v80_adapter_dir', '/content/kg1_adapter_v80')
BASE_MODEL_NAME = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
OUTPUT_DIR = '/content/kg1_out'
Path(OUTPUT_DIR).mkdir(exist_ok=True)
SUBMISSION_DIR = f'{OUTPUT_DIR}/submission_v80'
Path(SUBMISSION_DIR).mkdir(exist_ok=True)

print('=' * 70)
print('V80 CELL 6 - Package submission + Kaggle submit (dgxchen v7 EXACT)')
print('=' * 70)

# Copy required files
required = ['adapter_config.json', 'adapter_model.safetensors']
for fn in required:
    src = Path(ADAPTER_DIR) / fn
    dst = Path(SUBMISSION_DIR) / fn
    if not src.exists():
        raise FileNotFoundError(f'Missing: {src}')
    shutil.copy2(src, dst)
    print(f'Copied {fn} ({dst.stat().st_size/(1024**2):.1f} MB)')

# Fix adapter_config.json
cfg_path = Path(SUBMISSION_DIR) / 'adapter_config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['base_model_name_or_path'] = BASE_MODEL_NAME
cfg['inference_mode'] = True
cfg['lora_dropout'] = 0.0
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)

# Verify 8 targets NO lm_head (V80 FIX #3)
tm = cfg.get('target_modules', [])
if isinstance(tm, list):
    if 'lm_head' in tm:
        print(f'!!! WARN: adapter_config.json has lm_head in target_modules!')
    else:
        print(f'V80 adapter OK: {len(tm)} target_modules (NO lm_head): {tm}')

# Build zip
zip_path = f'{OUTPUT_DIR}/submission_v80.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in required:
        zf.write(Path(SUBMISSION_DIR) / fn, arcname=fn)
zip_mb = os.path.getsize(zip_path) / (1024**2)
print(f'\nsubmission_v80.zip: {zip_mb:.1f} MB')
assert zip_mb < 500, f'zip too big: {zip_mb}MB (Kaggle limit 500MB)'

# GDrive backup
try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive', force_remount=False)
    bkp_dir = Path('/content/drive/MyDrive/kg1_v80')
    bkp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(zip_path, bkp_dir / f'submission_{ts}.zip')
    shutil.copytree(ADAPTER_DIR, bkp_dir / f'adapter_{ts}', dirs_exist_ok=True)
    print(f'GDrive backup: {bkp_dir}')
except Exception as e:
    print(f'GDrive backup skipped: {e}')

# HF upload
try:
    from huggingface_hub import HfApi, upload_folder
    HF_TOKEN = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    REPO = 'felipesp1983/kg1-nemotron-lora-v80-colab-dgxchen'
    api = HfApi(token=HF_TOKEN)
    api.create_repo(REPO, private=True, exist_ok=True)
    upload_folder(
        repo_id=REPO, folder_path=SUBMISSION_DIR,
        allow_patterns=['adapter_*'],
        token=HF_TOKEN,
    )
    print(f'HF upload OK: https://huggingface.co/{REPO}')
except Exception as e:
    print(f'HF upload failed: {e}')

# Kaggle submit with slot check
print('\nChecking Kaggle slots (5/day hard limit)...')
try:
    rc = subprocess.run(
        ['kaggle', 'competitions', 'submissions',
         '-c', 'nvidia-nemotron-model-reasoning-challenge', '--csv'],
        capture_output=True, text=True, timeout=60,
    )
    if rc.returncode == 0:
        from io import StringIO
        import csv as _csv
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        today_count = sum(1 for r in _csv.DictReader(StringIO(rc.stdout))
                          if r.get('date', '').startswith(today))
        print(f'Submissions today: {today_count}/5')
        if today_count >= 5:
            print('Slot exhausted - manual submit later:')
            print(f'  kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge \\')
            print(f'    -f {zip_path} -m "V80 dgxchen v7 EXACT replica"')
        else:
            msg = f'V80 dgxchen v7 EXACT {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
            r = subprocess.run(
                ['kaggle', 'competitions', 'submit',
                 '-c', 'nvidia-nemotron-model-reasoning-challenge',
                 '-f', zip_path, '-m', msg],
                capture_output=True, text=True, timeout=600,
            )
            print(f'Submit rc={r.returncode}')
            print(r.stdout[-400:])
            if r.returncode == 0:
                print('\nSUBMITTED! Check score at:')
                print('https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions')
except Exception as e:
    print(f'Submit failed: {e}')
    print(f'\nManual submit:')
    print(f'  kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge \\')
    print(f'    -f {zip_path} -m "V80 dgxchen v7 EXACT"')

print('\nV80 DONE. Review:')
print(f'  adapter: {ADAPTER_DIR}')
print(f'  zip: {zip_path}')
"""


def cell_md(c, i):
    return {'cell_type': 'markdown', 'metadata': {'id': i},
            'source': c.splitlines(keepends=True)}


def cell_code(c, i):
    return {'cell_type': 'code', 'metadata': {'id': i},
            'execution_count': None, 'outputs': [],
            'source': c.splitlines(keepends=True)}


NB = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'colab': {'provenance': [], 'machine_shape': 'hm'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU',
    },
    'cells': [
        cell_md(HEADER, 'header'),
        cell_code(CELL_1_SETUP, 'c1'),
        cell_code(CELL_2_VERIFY_AND_ML, 'c2'),
        cell_code(CELL_3_DATA, 'c3'),
        cell_code(CELL_4_MODEL_LORA, 'c4'),
        cell_code(CELL_5_TRAIN, 'c5'),
        cell_code(CELL_6_SUBMIT, 'c6'),
    ],
}


OUT = 'notebooks/KG1_V80_COLAB_V2.ipynb'
os.makedirs('notebooks', exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1)

size = os.path.getsize(OUT)
print(f'Wrote {OUT} ({size} bytes)')

# py_compile each cell
import py_compile, tempfile
cells = [
    ('Cell 1 SETUP', CELL_1_SETUP),
    ('Cell 2 VERIFY+ML', CELL_2_VERIFY_AND_ML),
    ('Cell 3 DATA', CELL_3_DATA),
    ('Cell 4 MODEL+LORA', CELL_4_MODEL_LORA),
    ('Cell 5 TRAIN', CELL_5_TRAIN),
    ('Cell 6 SUBMIT', CELL_6_SUBMIT),
]
all_ok = True
for name, c in cells:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as t:
        t.write(c)
        path = t.name
    try:
        py_compile.compile(path, doraise=True)
        print(f'{name}: py_compile OK ({len(c.splitlines())} lines)')
    except py_compile.PyCompileError as e:
        print(f'{name}: FAIL -- {e}')
        all_ok = False
    finally:
        os.unlink(path)

if all_ok:
    print('\nV80 COLAB v2 notebook READY. All 11 fixes applied.')
    print('Execution: Cell 1 -> RESTART -> Cell 2 -> Cell 3 -> Cell 4 -> Cell 5 -> Cell 6')
else:
    print('\nV80 v2 FAILED py_compile')
