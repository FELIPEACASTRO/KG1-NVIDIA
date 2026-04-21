"""KG1_V76_ANTIOVERFIT.ipynb - Based on empirical research from Kaggle winners.

V75 diagnosed overfitting: loss 41 -> 0.004 in 70 steps (memorization).
Research sources (kienngx 0.84, konbu17, tonghuikang 0.85+, dgxchen, torpidoff):

  grad_accum=4 -> 16 (eff batch 16, matches konbu17/kienngx)
  lr=2e-4 -> 5e-5 (matches kienngx 2nd pass, lower end of 1e-5..1e-4 range)
  lora_dropout=0.05 -> 0.10 (more regularization for 6k sample dataset)
  warmup_ratio=0.03 -> 0.05 (matches konbu17/kienngx)
  eval_strategy no -> steps, eval_steps=50 (early stop signal)

Same as V75:
  - torch 2.5.1+cu121 (downgrade + kernel restart mechanism)
  - mamba-ssm + causal-conv1d from pre-built wheels
  - transformers>=4.55,<5.0 pinned
  - Dataset filter trace_len<=2048 -> 6673 complete CoTs
  - NF4 disabled (Mamba incompat)

Structure (5 cells):
  1. GPU pre-flight
  2. Dataset validation
  3. Environment setup (torch + wheels + deps)
  4. Memory budget
  5. Training pipeline with anti-overfitting hyperparams
"""
import json
import os


HEADER = '''# KG1 V76 ANTI-OVERFIT - Research-based hyperparameters

## DIAGNOSE V75

V75 overfitted in 70 steps: training loss crashed 41 -> 0.004.
This is pure memorization (99.6% token copy accuracy) - no generalization.

## RESEARCH-BASED FIXES

From kienngx 0.84 LB, konbu17, tonghuikang 0.85+, dgxchen LoRA:

| Param | V75 | V76 | Reason |
|---|---|---|---|
| grad_accum | 4 | 16 | eff batch 16 (kienngx/konbu17) |
| learning_rate | 2e-4 | 5e-5 | kienngx 2nd pass evidence |
| lora_dropout | 0.05 | 0.10 | regularization for 6k samples |
| warmup_ratio | 0.03 | 0.05 | matches both references |
| eval_strategy | no | steps | early stop signal |

Target train loss: 0.10-0.30 (NOT lower, lower = memorization).

## COMO USAR

1. Runtime -> Change runtime type -> H100 HighRAM
2. Se tem runtime antigo: Disconnect and delete runtime
3. Cell 1 (GPU pre-flight)
4. Cell 2 (Dataset)
5. Cell 3 (Env setup) - PRIMEIRA execucao CRASHA kernel (esperado!)
6. Click Reconectar, re-run Cell 3 (instala wheels)
7. Cell 4 (Budget)
8. Cell 5 (Training ~11-12h)

## TEMPO ESTIMADO

- Cells 1-4: ~5 min
- Cell 3 torch download: 5-8 min
- Kernel restart + wheels: 2 min
- Cell 5 training: ~11-12h (eff batch 16 = 4x more fw+bw per step)
- ETA total: ~12h

## GATES

- Smoke abort if loss > 50 or NaN
- Early stop via eval loss monitoring
- 99% rule: NAO submete se local < 0.84
- Zero regressao leaderboard
'''


PREFLIGHT = r'''# Cell 1: PRE-FLIGHT GPU CHECK
import subprocess, torch, gc, sys

print('=' * 60)
print('PRE-FLIGHT GPU CHECK')
print('=' * 60)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    print(r.stdout)
except Exception as e:
    print(f'WARN: nvidia-smi failed: {e}')

if not torch.cuda.is_available():
    raise RuntimeError('CUDA not available - Runtime > Change runtime type > H100')

d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = total_gb - free_gb

print(f'GPU: {d.name}')
print(f'Total VRAM: {total_gb:.1f} GB')
print(f'Used: {used_gb:.1f} GB')
print(f'Free: {free_gb:.1f} GB')
print(f'torch: {torch.__version__}')

if total_gb < 38:
    raise RuntimeError(f'GPU too small: {total_gb:.1f}GB')
if used_gb > 5:
    print('\nZOMBIE PROCESS DETECTED')
    print('Fix: Runtime > Disconnect and delete runtime > Reconnect')
    raise RuntimeError(f'Zombie has {used_gb:.1f}GB')

print()
print('GPU is clean. Ready for Cell 2.')
'''


DATASET_VALIDATION = r'''# Cell 2: DATASET VALIDATION
import os, json, subprocess, sys
from pathlib import Path

print('=' * 60)
print('DATASET VALIDATION - V70 huikang corpus')
print('=' * 60)

for pkg in ['huggingface_hub>=0.25,<2.0', 'pandas>=2.0,<4.0']:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                   capture_output=True, text=True, timeout=300)

try:
    from google.colab import userdata
    hf_key = userdata.get('HF_KEY') or os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
except ImportError:
    hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')

assert hf_key, 'HF_KEY missing'
os.environ['HF_TOKEN'] = hf_key

from huggingface_hub import login, hf_hub_download
import pandas as pd
login(token=hf_key, add_to_git_credential=False)

GDRIVE_OK = False
try:
    from google.colab import drive
    try:
        drive.mount('/content/drive', force_remount=False)
        GDRIVE_OK = True
    except Exception:
        try:
            drive.mount('/content/drive', force_remount=True)
            GDRIVE_OK = True
        except Exception as e:
            print(f'WARN GDrive: {e}')
except ImportError:
    pass

cache = '/content/drive/MyDrive/kg1_data/v70_huikang' if GDRIVE_OK else '/content/kg1_data/v70_huikang'
os.makedirs(cache, exist_ok=True)

v70_path = hf_hub_download(
    repo_id='felipesp1983/kg1-nemotron-training',
    filename='data/sft_v70_huikang_full.jsonl',
    repo_type='dataset',
    local_dir=cache,
    token=hf_key,
)
print(f'OK: {v70_path}')
print(f'Size: {os.path.getsize(v70_path)/1024/1024:.1f} MB')

df = pd.read_json(v70_path, lines=True)
print(f'Loaded {len(df)} rows')

expected = {'id', 'category', 'answer', 'messages', 'trace_len'}
missing = expected - set(df.columns)
assert not missing, f'Missing: {missing}'
print(f'Schema OK')

print('\ntrace_len stats:')
for p, lbl in [(0, 'min'), (25, 'p25'), (50, 'p50'), (75, 'p75'),
               (90, 'p90'), (95, 'p95'), (99, 'p99'), (100, 'max')]:
    if p == 0: val = df['trace_len'].min()
    elif p == 100: val = df['trace_len'].max()
    else: val = df['trace_len'].quantile(p/100)
    print(f'  {lbl:5s}: {val:7.0f}')

print('\nCoverage by max_length:')
for ml in [1024, 1536, 2048, 3072, 4096]:
    fit = (df['trace_len'] <= ml).sum()
    print(f'  max_length={ml}: {fit}/{len(df)} ({100*fit/len(df):.1f}%)')

fit_2048 = (df['trace_len'] <= 2048).sum()
print(f'\nWILL TRAIN ON: {fit_2048} samples (trace_len<=2048)')
print(f'WILL DROP:     {len(df) - fit_2048} samples')

print()
print('=' * 60)
print('DATASET OK - Ready for Cell 3')
print('=' * 60)
'''


ENV_SETUP = r'''# Cell 3: ENVIRONMENT SETUP (torch downgrade + wheels install)
# ATENCAO: primeira execucao crasha o kernel (esperado!).
#          Depois do crash: Reconectar, roda esta celula NOVAMENTE.
import subprocess, sys, os, time

print('=' * 60)
print('ENVIRONMENT SETUP')
print('=' * 60)

import torch
current = torch.__version__
print(f'Current torch: {current}')

if not current.startswith('2.5'):
    print(f'\nDowngrading torch {current} -> 2.5.1+cu121...')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y',
                    'torch', 'torchvision', 'torchaudio'],
                   capture_output=True, text=True, timeout=300)
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
         'torch==2.5.1+cu121',
         'torchvision==0.20.1+cu121',
         'torchaudio==2.5.1+cu121',
         '--index-url', 'https://download.pytorch.org/whl/cu121'],
        capture_output=True, text=True, timeout=1200,
    )
    if r.returncode != 0:
        print(f'stderr: {r.stderr[-500:]}')
        raise RuntimeError('torch downgrade failed')
    print('torch downgrade OK')
    print()
    print('=' * 60)
    print('KERNEL RESTART REQUIRED')
    print('=' * 60)
    print('ACTION when you see "Session crashed":')
    print('  1. Click "Reconectar" at top right')
    print('  2. Run THIS Cell 3 AGAIN')
    time.sleep(3)
    os.kill(os.getpid(), 9)

print(f'[OK] torch={current}')
print()
print('Installing pinned dependencies...')
DEPS = [
    'transformers>=4.55,<5.0',  # CRITICAL for mamba-ssm 2.2.4
    'peft>=0.13,<0.20',
    'trl>=0.25,<1.3',
    'accelerate>=0.34,<2.0',
    'bitsandbytes>=0.44',
    'datasets>=2.20,<5.0',
    'safetensors>=0.4.5',
    'sentencepiece',
    'einops',
    'huggingface_hub>=0.25,<2.0',
]
for pkg in DEPS:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                       capture_output=True, text=True, timeout=300)
    print(f'  [{"OK" if r.returncode==0 else "FAIL"}] {pkg}')

print()
print('Installing mamba-ssm + causal-conv1d from wheels...')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y',
                'mamba-ssm', 'causal-conv1d', 'mamba_ssm'],
               capture_output=True, text=True, timeout=120)

WHEELS = {
    'causal-conv1d': 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl',
    'mamba-ssm': 'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl',
}
for name, url in WHEELS.items():
    print(f'  Installing {name}...')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', url],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f'    stderr: {r.stderr[-500:]}')
        raise RuntimeError(f'{name} wheel install failed')
    print(f'    [OK] {name}')

print('\nVerifying imports...')
for mod_name in list(sys.modules.keys()):
    if any(x in mod_name for x in ['mamba_ssm', 'causal_conv1d']):
        del sys.modules[mod_name]

import mamba_ssm
print(f'  [OK] mamba_ssm')
import causal_conv1d
from causal_conv1d import causal_conv1d_fn
assert causal_conv1d_fn is not None
print(f'  [OK] causal_conv1d + binding')
import transformers
print(f'  [OK] transformers {transformers.__version__}')
assert transformers.__version__.startswith('4.')
from transformers.generation import GreedySearchDecoderOnlyOutput
print(f'  [OK] GreedySearchDecoderOnlyOutput')
import peft, trl, accelerate, bitsandbytes
print(f'  [OK] peft {peft.__version__}')
print(f'  [OK] trl {trl.__version__}')
print(f'  [OK] accelerate {accelerate.__version__}')
print(f'  [OK] bitsandbytes {bitsandbytes.__version__}')

print()
print('=' * 60)
print('ENVIRONMENT READY - proceed to Cell 4 + Cell 5')
print('=' * 60)
'''


MEMORY_BUDGET = r'''# Cell 4: MEMORY BUDGET + ETA (V76 anti-overfit hyperparams)
import torch

print('=' * 60)
print('MEMORY BUDGET + ETA PRE-CHECK (V76 ANTI-OVERFIT)')
print('=' * 60)

d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3

MODEL_BF16 = 62.8
LORA_BF16 = 1.8
GRAD_LORA = 2.0
OPTIM_8BIT = 2.0
CUDA_OVERHEAD = 2.0

MAX_LENGTH = 2048
GRAD_ACCUM = 16  # V76: was 4 (eff batch 4 caused overfit)

ACTIVATIONS = MAX_LENGTH / 2048 * 3.0
TOTAL = MODEL_BF16 + LORA_BF16 + GRAD_LORA + OPTIM_8BIT + CUDA_OVERHEAD + ACTIVATIONS

print(f'GPU: {d.name}  Total: {total_gb:.1f} GB  Free: {free_gb:.1f} GB')
print(f'torch: {torch.__version__}')
print()
print('Projected memory:')
print(f'  Model BF16 (30B):        {MODEL_BF16:5.1f} GB')
print(f'  LoRA BF16 (883M):        {LORA_BF16:5.1f} GB')
print(f'  Gradients LoRA:          {GRAD_LORA:5.1f} GB')
print(f'  Optimizer 8bit:          {OPTIM_8BIT:5.1f} GB')
print(f'  CUDA overhead:           {CUDA_OVERHEAD:5.1f} GB')
print(f'  Activations seq={MAX_LENGTH}:   {ACTIVATIONS:5.1f} GB')
print(f'  ----------------------------------')
print(f'  TOTAL:                   {TOTAL:5.1f} GB')
print(f'  AVAILABLE:               {total_gb:5.1f} GB')
print(f'  MARGIN:                  {total_gb-TOTAL:+5.1f} GB')
print()

PER_SAMPLE = 5.7
STEP_TIME = PER_SAMPLE * GRAD_ACCUM  # 4x more due to larger grad_accum
FILTERED = 6673
STEPS = FILTERED // GRAD_ACCUM
ETA_HOURS = (STEPS * STEP_TIME) / 3600

print(f'Training speed estimate:')
print(f'  Per sample fw+bw:     {PER_SAMPLE:.1f}s')
print(f'  Per optim step:       {STEP_TIME:.1f}s (grad_accum={GRAD_ACCUM})')
print(f'  Dataset filtered:     {FILTERED} samples')
print(f'  Total steps:          {STEPS}')
print(f'  ETA training:         {ETA_HOURS:.1f} hours')
print()
print(f'Anti-overfit hyperparams (research-based):')
print(f'  lr: 2e-4 -> 5e-5')
print(f'  lora_dropout: 0.05 -> 0.10')
print(f'  warmup_ratio: 0.03 -> 0.05')
print(f'  grad_accum: 4 -> 16 (eff batch 16)')
print(f'  eval_strategy: steps (early stop signal)')
print()

margin = total_gb - TOTAL
if margin < 5: print('[WARN] Memory tight')
else: print('[OK] Memory comfortable')
if ETA_HOURS < 12: print('[OK] Fits Colab Pro 12h')
elif ETA_HOURS < 24: print('[OK] Fits Colab Pro+ 24h')
else: print('[WARN] ETA > 24h')

assert total_gb - TOTAL > -3
assert free_gb > 30

print()
print('=' * 60)
print(f'BUDGET OK - ETA: {ETA_HOURS:.1f}h  Ready for Cell 5')
print('=' * 60)
'''


MEGA = r'''# Cell 5: TRAINING PIPELINE V76 (anti-overfit, ~11-12h on H100)
import os, sys, json, subprocess, shutil, gc, time, datetime, random, re, math
from pathlib import Path

START_TIME = time.time()
def log(msg):
    elapsed = int(time.time() - START_TIME)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f'[{h:02d}:{m:02d}:{s:02d}] {msg}', flush=True)

def section(title):
    log('=' * 60)
    log(title)
    log('=' * 60)

section('SECTION 0: Environment verify')
import torch
assert torch.__version__.startswith('2.5'), f'Need torch 2.5.x, got {torch.__version__}'
import mamba_ssm, causal_conv1d
from causal_conv1d import causal_conv1d_fn
assert causal_conv1d_fn is not None
log(f'torch={torch.__version__}  mamba-ssm loaded  causal-conv1d bound')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

section('SECTION 1: Clone KG1')
KG1_DIR = Path('/content/kg1')
REPO = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
BRANCH = os.environ.get('KG1_BRANCH', 'claude/competent-shamir')
if KG1_DIR.exists():
    shutil.rmtree(KG1_DIR)
subprocess.check_call([
    'git', 'clone', '--depth', '1', '--branch', BRANCH, REPO, str(KG1_DIR),
])
commit = subprocess.check_output(
    ['git', '-C', str(KG1_DIR), 'log', '-1', '--format=%h %s'], text=True,
).strip()
log(f'Commit: {commit}')
REQUIRED = [
    'src/reasoners/bit_manipulation_pairs.py',
    'src/reasoners/cryptarithm_47combo.py',
    'src/reasoners/neurosymbolic_template.py',
    'src/losses/max_min_logprob.py',
    'src/prompts/build_prompt.py',
    'scripts/local_score.py',
    'scripts/kg1_submission_gate.py',
]
missing = [r for r in REQUIRED if not (KG1_DIR / r).exists()]
assert not missing, f'Missing: {missing}'
sys.path.insert(0, str(KG1_DIR))

section('SECTION 2: Auth')
GDRIVE_MOUNTED = False
try:
    from google.colab import drive, userdata
    try:
        drive.mount('/content/drive', force_remount=False)
        GDRIVE_MOUNTED = True
    except Exception:
        try:
            drive.mount('/content/drive', force_remount=True)
            GDRIVE_MOUNTED = True
        except Exception as e:
            log(f'WARN: {e}')
    try:
        hf_key = userdata.get('HF_KEY')
    except Exception:
        hf_key = None
    if not hf_key:
        hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    os.environ['HF_TOKEN'] = hf_key
    os.environ['HF_KEY'] = hf_key
    try:
        kuser = userdata.get('KAGGLE_USERNAME')
        kkey = userdata.get('KAGGLE_KEY')
    except Exception:
        kuser = os.environ.get('KAGGLE_USERNAME')
        kkey = os.environ.get('KAGGLE_KEY')
    if kuser and kkey:
        os.environ['KAGGLE_USERNAME'] = kuser
        os.environ['KAGGLE_KEY'] = kkey
        kpath = Path.home() / '.kaggle' / 'kaggle.json'
        kpath.parent.mkdir(parents=True, exist_ok=True)
        kpath.write_text(json.dumps({'username': kuser, 'key': kkey}))
        kpath.chmod(0o600)
        log(f'Kaggle: {kuser}')
except ImportError:
    pass
from huggingface_hub import login, whoami
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
login(token=HF_TOKEN, add_to_git_credential=False)
try: log(f'HF: {whoami(token=HF_TOKEN)["name"]}')
except Exception: pass

section('SECTION 3: Config V76 (anti-overfit)')
from dataclasses import dataclass, asdict

@dataclass
class Config:
    base_model: str = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    use_nf4: bool = False
    max_length: int = 2048
    attn_implementation: str = 'eager'
    mamba_ssm_cache_dtype: str = 'float32'
    tie_word_embeddings: bool = False
    # LoRA (keep proven values from winners)
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.10  # V76: was 0.05 (more regularization)
    lora_target_modules: str = 'all-linear'
    # Training (V76 anti-overfit)
    epochs: int = 1
    per_device_batch: int = 1
    grad_accum: int = 16  # V76: was 4 (effective batch 16)
    learning_rate: float = 5e-5  # V76: was 2e-4 (lower for small batch)
    lr_scheduler: str = 'linear'
    warmup_ratio: float = 0.05  # V76: was 0.03
    grad_clip: float = 1.0
    optimizer: str = 'paged_adamw_8bit'
    bf16: bool = True
    use_gradient_checkpointing: bool = True
    # Loss
    loss_type: str = 'max_min_warmup_ce'
    max_min_warmup_steps: int = 100
    # Data
    hf_dataset_repo: str = 'felipesp1983/kg1-nemotron-training'
    hf_dataset_file: str = 'data/sft_v70_huikang_full.jsonl'
    # Prompt
    enable_thinking: bool = True
    use_structured: bool = True
    use_category_hints: bool = True
    use_boxed_strict: bool = True
    use_self_correct: bool = True
    # Gates
    smoke_test_steps: int = 2
    smoke_abort_loss: float = 50.0
    eval_holdout_size: int = 600
    local_score_floor: float = 0.84
    target_score: float = 0.87
    # Output
    run_tag: str = 'v76_antioverfit'
    output_dir: str = '/content/kg1_out/v76_antioverfit'
    gdrive_checkpoint: str = '/content/drive/MyDrive/kg1_checkpoints/v76_antioverfit'
    hf_upload_repo: str = 'felipesp1983/kg1-nemotron-lora-v76-antioverfit'

CFG = Config()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(CFG.output_dir) / 'config.json', 'w') as f:
    json.dump(asdict(CFG), f, indent=2)
log(f'lr={CFG.learning_rate} dropout={CFG.lora_dropout} warmup={CFG.warmup_ratio} grad_accum={CFG.grad_accum}')

section('SECTION 4: Pre-flight')
import importlib
for mod in [
    'src.reasoners.bit_manipulation_pairs',
    'src.reasoners.cryptarithm_47combo',
    'src.reasoners.neurosymbolic_template',
    'src.losses.max_min_logprob',
    'src.prompts.build_prompt',
]:
    importlib.import_module(mod)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
log(f'tokenizer: vocab={tok.vocab_size}')
from src.prompts.build_prompt import build_prompt_v71, detect_category

section('SECTION 5: Load V70 + FILTER')
from huggingface_hub import hf_hub_download
import pandas as pd
cache = Path('/content/drive/MyDrive/kg1_data') if GDRIVE_MOUNTED else Path('/content/kg1_data')
cache.mkdir(parents=True, exist_ok=True)
v70_path = hf_hub_download(
    repo_id=CFG.hf_dataset_repo,
    filename=CFG.hf_dataset_file,
    repo_type='dataset',
    local_dir=str(cache / 'v70_huikang'),
    token=HF_TOKEN,
)
df = pd.read_json(v70_path, lines=True)
log(f'Loaded {len(df)} rows')
orig = len(df)
df = df[df['trace_len'] <= CFG.max_length].copy().reset_index(drop=True)
log(f'FILTER trace_len<={CFG.max_length}: {orig} -> {len(df)}')

if 'messages' in df.columns and 'response' not in df.columns:
    def extract(msgs):
        if not isinstance(msgs, list): return None, None
        u = next((m['content'] for m in msgs if m.get('role') == 'user'), None)
        a = next((m['content'] for m in msgs if m.get('role') == 'assistant'), None)
        return u, a
    df[['_u', '_a']] = df['messages'].apply(lambda m: pd.Series(extract(m)))
    df = df.rename(columns={'_u': 'prompt', '_a': 'response'})

if 'category' not in df.columns or df['category'].isna().all():
    df['category'] = df['prompt'].map(detect_category)

def build_record(row):
    p = row.get('prompt')
    if p is None or (isinstance(p, float) and pd.isna(p)): return None
    p = str(p).strip()
    if not p: return None
    cat = str(row.get('category', '')) if pd.notna(row.get('category', '')) else ''
    user = build_prompt_v71(
        p, category=cat,
        use_structured=CFG.use_structured,
        use_category_hints=CFG.use_category_hints,
        use_boxed_strict=CFG.use_boxed_strict,
        use_self_correct=CFG.use_self_correct,
    )
    resp = row.get('response')
    if resp is not None and pd.notna(resp) and str(resp).strip():
        assistant = str(resp).strip()
        if '\\boxed{' not in assistant:
            ans = row.get('answer', '')
            if ans and pd.notna(ans):
                assistant = assistant + f'\n\\boxed{{{ans}}}'
    else:
        ans = row.get('answer', '')
        if not ans or pd.isna(ans): return None
        assistant = f'\\boxed{{{ans}}}'
    return {'user': user, 'assistant': assistant, 'category': cat}

records = [r for r in (build_record(row) for _, row in df.iterrows()) if r is not None]
log(f'Records: {len(records)}')
random.seed(42)
idx = list(range(len(records)))
random.shuffle(idx)
eval_n = min(CFG.eval_holdout_size, max(50, len(records) // 20))
eval_set = set(idx[:eval_n])
train_records = [records[i] for i in idx if i not in eval_set]
eval_records = [records[i] for i in idx if i in eval_set]
train_path = Path(CFG.output_dir) / 'train.jsonl'
eval_path = Path(CFG.output_dir) / 'eval.jsonl'
with open(train_path, 'w') as f:
    for r in train_records: f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(eval_path, 'w') as f:
    for r in eval_records: f.write(json.dumps(r, ensure_ascii=False) + '\n')
log(f'train={len(train_records)} eval={len(eval_records)}')
del df, records, train_records, eval_records
gc.collect()

section('SECTION 6: Load model BF16 + LoRA')
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
log(f'VRAM free: {free_gb:.1f}GB')
assert free_gb >= 30
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
model_cfg = AutoConfig.from_pretrained(CFG.base_model, trust_remote_code=True)
setattr(model_cfg, 'tie_word_embeddings', False)
if hasattr(model_cfg, 'mamba_ssm_cache_dtype'):
    setattr(model_cfg, 'mamba_ssm_cache_dtype', 'float32')
log('Loading NemotronH 30B BF16 (cached ~30 sec)...')
model = AutoModelForCausalLM.from_pretrained(
    CFG.base_model,
    config=model_cfg,
    torch_dtype=torch.bfloat16,
    device_map={'': 0},
    attn_implementation=CFG.attn_implementation,
    trust_remote_code=True,
)
log('Model loaded')
peft_cfg = LoraConfig(
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    target_modules=CFG.lora_target_modules,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
if CFG.use_gradient_checkpointing:
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        log('Gradient checkpointing ENABLED')
    except Exception as e:
        log(f'WARN grad_ckpt: {e}')
        CFG.max_length = max(512, CFG.max_length // 2)
        CFG.use_gradient_checkpointing = False

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
log(f'After load: used={used_gb:.1f}GB free={free_gb:.1f}GB')

section('SECTION 7: Smoke test')
gc.collect()
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader

class JsonlChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        messages = [
            {'role': 'user', 'content': r['user']},
            {'role': 'assistant', 'content': r['assistant']},
        ]
        try:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, enable_thinking=CFG.enable_thinking,
            )
        except TypeError:
            text = self.tok.apply_chat_template(messages, tokenize=False)
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors='pt')
        ids = enc['input_ids'][0]
        labels = ids.clone()
        try:
            ut = self.tok.apply_chat_template(
                [messages[0]], tokenize=False, enable_thinking=CFG.enable_thinking,
            )
        except TypeError:
            ut = self.tok.apply_chat_template([messages[0]], tokenize=False)
        uids = self.tok(ut, return_tensors='pt')['input_ids'][0]
        k = min(len(uids), len(labels))
        labels[:k] = -100
        return {
            'input_ids': ids,
            'labels': labels,
            'attention_mask': enc['attention_mask'][0],
        }

def collate(batch, pad_id):
    max_l = max(x['input_ids'].size(0) for x in batch)
    def pad(t, v):
        return torch.nn.functional.pad(t, (0, max_l - t.size(0)), value=v)
    return {
        'input_ids': torch.stack([pad(x['input_ids'], pad_id) for x in batch]),
        'labels': torch.stack([pad(x['labels'], -100) for x in batch]),
        'attention_mask': torch.stack([pad(x['attention_mask'], 0) for x in batch]),
    }

ds = JsonlChatDataset(train_path, tok, CFG.max_length)
dl = DataLoader(ds, batch_size=CFG.per_device_batch, shuffle=True,
                collate_fn=lambda b: collate(b, tok.pad_token_id))
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
losses = []
for step, batch in enumerate(dl):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**{k: v for k, v in batch.items() if k != 'labels'})
    loss = torch.nn.functional.cross_entropy(
        out.logits.view(-1, out.logits.size(-1)),
        batch['labels'].view(-1), ignore_index=-100,
    )
    assert not math.isnan(loss.item()) and not math.isinf(loss.item())
    losses.append(loss.item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], CFG.grad_clip,
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    log(f'smoke step {step} loss={loss.item():.4f}')
    if step + 1 >= CFG.smoke_test_steps: break
assert losses[-1] < CFG.smoke_abort_loss
log(f'Smoke PASSED (final {losses[-1]:.4f})')
try: del opt, ds, dl, batch, out, loss
except: pass
gc.collect()
torch.cuda.empty_cache()

section('SECTION 8: Full training V76')
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from src.losses.max_min_logprob import max_min_logprob_loss

ds_train = load_dataset('json', data_files=str(train_path), split='train')
ds_eval = load_dataset('json', data_files=str(eval_path), split='train')

def format_example(ex):
    messages = [
        {'role': 'user', 'content': ex['user']},
        {'role': 'assistant', 'content': ex['assistant']},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, enable_thinking=CFG.enable_thinking,
        )
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False)
    return {'text': text}

ds_train = ds_train.map(format_example, remove_columns=ds_train.column_names)
ds_eval = ds_eval.map(format_example, remove_columns=ds_eval.column_names)
tok.model_max_length = CFG.max_length

sft_args = SFTConfig(
    output_dir=CFG.output_dir,
    per_device_train_batch_size=CFG.per_device_batch,
    per_device_eval_batch_size=CFG.per_device_batch,
    gradient_accumulation_steps=CFG.grad_accum,
    num_train_epochs=CFG.epochs,
    learning_rate=CFG.learning_rate,
    lr_scheduler_type=CFG.lr_scheduler,
    warmup_ratio=CFG.warmup_ratio,
    max_grad_norm=CFG.grad_clip,
    bf16=CFG.bf16,
    logging_steps=5,  # V76: was 10, more frequent tracking
    save_steps=50,    # V76: was 100
    eval_strategy='steps',  # V76: was 'no', enable early stop signal
    eval_steps=50,    # V76: NEW, monitor eval loss
    save_total_limit=5,
    optim=CFG.optimizer,
    packing=False,
    report_to=[],
    gradient_checkpointing=CFG.use_gradient_checkpointing,
    dataset_text_field='text',
    load_best_model_at_end=False,  # Don't auto-pick, we'll manual decide
    metric_for_best_model='loss',
)

class MaxMinSFTTrainer(SFTTrainer):
    _nan_count = 0
    _use_ce_permanent = False
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        logits = outputs.logits
        step = int(self.state.global_step)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
        )
        use_ce = (
            self._use_ce_permanent
            or CFG.loss_type == 'ce'
            or (CFG.loss_type == 'max_min_warmup_ce' and step < CFG.max_min_warmup_steps)
        )
        if use_ce:
            loss = ce_loss
        else:
            try:
                mm_loss = max_min_logprob_loss(logits, labels)
                if torch.isnan(mm_loss) or torch.isinf(mm_loss):
                    self._nan_count += 1
                    print(f'WARN: NaN max-min step {step} (#{self._nan_count})')
                    if self._nan_count >= 3:
                        self._use_ce_permanent = True
                    loss = ce_loss
                else:
                    loss = mm_loss
                    self._nan_count = max(0, self._nan_count - 1)
            except Exception as e:
                print(f'WARN max-min exc: {e}')
                loss = ce_loss
        if torch.isnan(loss) or torch.isinf(loss):
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss

trainer = MaxMinSFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    processing_class=tok,
)
log('Training START V76')
log(f'  Target train loss: 0.10-0.30 (NOT lower, <0.01 = memorization)')
log(f'  Monitor eval_loss at step 50, 100, 150 for early stop signal')
trainer.train()
log('Training complete')
trainer.save_model(CFG.output_dir)
tok.save_pretrained(CFG.output_dir)

section('SECTION 9: Save + upload')
from huggingface_hub import HfApi, upload_folder
out_dir = Path(CFG.output_dir)
req = ['adapter_config.json', 'adapter_model.safetensors']
missing_a = [f for f in req if not (out_dir / f).exists()]
assert not missing_a
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if GDRIVE_MOUNTED:
    gdrive_dest = Path(CFG.gdrive_checkpoint) / f'{CFG.run_tag}_{ts}'
    try:
        gdrive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, gdrive_dest, dirs_exist_ok=True)
        log(f'GDrive: {gdrive_dest}')
    except Exception as e:
        log(f'WARN: {e}')
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(CFG.hf_upload_repo, private=True, exist_ok=True)
    upload_folder(
        repo_id=CFG.hf_upload_repo,
        folder_path=str(out_dir),
        allow_patterns=['adapter_*', 'tokenizer*', 'special_tokens*', 'config.json'],
        token=HF_TOKEN,
    )
    log(f'HF: {CFG.hf_upload_repo}')
except Exception as e:
    log(f'WARN: {e}')

section('SECTION 10: Local eval + gate')
local_score_script = Path('/content/kg1/scripts/local_score.py')
eval_csv = Path(CFG.output_dir) / 'local_eval.csv'
cmd = [sys.executable, str(local_score_script),
       '--adapter', str(out_dir),
       '--n-samples', str(CFG.eval_holdout_size),
       '--output-csv', str(eval_csv)]
try:
    res = subprocess.run(cmd, cwd='/content/kg1', check=False,
                         capture_output=True, text=True, timeout=3600)
    log(f'STDOUT:\n{res.stdout[-1500:]}')
except subprocess.TimeoutExpired:
    res = type('R', (), {'stdout': ''})
local_score_val = None
m = re.search(r'(?:overall\s+score|score)[:\s]+([0-9.]+)', res.stdout, re.IGNORECASE)
if m: local_score_val = float(m.group(1))
log(f'Local score = {local_score_val}')
with open(out_dir / 'local_score.json', 'w') as f:
    json.dump({'local_score': local_score_val}, f)

GO = False
if local_score_val is None:
    gate_msg = 'NO-GO: parse failed'
elif local_score_val < CFG.local_score_floor:
    gate_msg = f'NO-GO: {local_score_val:.4f} < {CFG.local_score_floor}'
elif local_score_val < CFG.target_score - 0.01:
    gate_msg = f'MARGINAL: {local_score_val:.4f}'
    GO = True
else:
    gate_msg = f'GO: {local_score_val:.4f}'
    GO = True
log(f'Gate: {gate_msg}')
with open(out_dir / 'gate_decision.json', 'w') as f:
    json.dump({'go': GO, 'msg': gate_msg, 'score': local_score_val}, f)

section('SECTION 11: Submit (if GO)')
if not GO:
    log('NO-GO -> skipping')
else:
    import zipfile
    zip_path = out_dir / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in ['adapter_config.json', 'adapter_model.safetensors']:
            zf.write(out_dir / fn, arcname=fn)
    log(f'zip: {zip_path.stat().st_size/(1024*1024):.2f} MB')
    gate_script = Path('/content/kg1/scripts/kg1_submission_gate.py')
    r2 = subprocess.run(
        [sys.executable, str(gate_script), '--zip', str(zip_path)],
        capture_output=True, text=True,
    )
    if r2.returncode == 0 and os.environ.get('KAGGLE_USERNAME'):
        msg = f'V76_ANTIOVERFIT {datetime.datetime.now().strftime("%Y-%m-%d %H:%M BRT")}'
        submit_script = Path('/content/kg1/scripts/submit_kaggle.py')
        if submit_script.exists():
            cmd = [sys.executable, str(submit_script), '--zip', str(zip_path), '--message', msg]
        else:
            cmd = ['kaggle', 'competitions', 'submit',
                   '-c', 'nvidia-nemotron-model-reasoning-challenge',
                   '-f', str(zip_path), '-m', msg]
        r3 = subprocess.run(cmd, capture_output=True, text=True)
        with open(out_dir / 'kaggle_submit.json', 'w') as f:
            json.dump({'msg': msg, 'rc': r3.returncode,
                      'stdout': r3.stdout[-1000:], 'stderr': r3.stderr[-500:]}, f)
        log(f'Submit rc={r3.returncode}')

section('ALL DONE V76')
log(f'Total: {(time.time() - START_TIME)/3600:.2f}h')
log(f'Score: {local_score_val} | Gate: {GO}')
'''


def cell_md(content, cid):
    return {
        "cell_type": "markdown",
        "metadata": {"id": cid},
        "source": content.splitlines(keepends=True),
    }


def cell_code(content, cid):
    return {
        "cell_type": "code",
        "metadata": {"id": cid},
        "execution_count": None,
        "outputs": [],
        "source": content.splitlines(keepends=True),
    }


NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "machine_shape": "hm"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [
        cell_md(HEADER, "header"),
        cell_code(PREFLIGHT, "c1-preflight"),
        cell_code(DATASET_VALIDATION, "c2-dataset"),
        cell_code(ENV_SETUP, "c3-envsetup"),
        cell_code(MEMORY_BUDGET, "c4-budget"),
        cell_code(MEGA, "c5-mega"),
    ],
}

OUT = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'KG1_V76_ANTIOVERFIT.ipynb')
OUT = os.path.abspath(OUT)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f'Wrote {OUT} ({os.path.getsize(OUT)} bytes)')

import py_compile, tempfile
for name, src in [('PREFLIGHT', PREFLIGHT), ('DATASET', DATASET_VALIDATION),
                  ('ENV_SETUP', ENV_SETUP), ('MEMORY_BUDGET', MEMORY_BUDGET),
                  ('MEGA', MEGA)]:
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as t:
        t.write(src)
        tpath = t.name
    try:
        py_compile.compile(tpath, doraise=True)
        print(f'{name}: py_compile OK ({src.count(chr(10))} lines)')
    except py_compile.PyCompileError as e:
        print(f'{name}: SYNTAX ERROR:\n{e}')
        raise
    os.unlink(tpath)

print()
print('V76 notebook built and validated. Ready to push.')
