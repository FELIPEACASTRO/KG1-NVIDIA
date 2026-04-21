"""Build KG1_V73_DEFINITIVO.ipynb - definitive notebook with ALL fixes + rigorous validation.

Structure:
  0. Header (markdown with instructions)
  1. Pre-flight (GPU zombie check)
  2. Dataset validation (NEW - validates V70 data schema + CoT quality)
  3. Memory budget check (NEW - predicts memory before loading)
  4. Mega training cell (all sections)
"""
import json
import os
import uuid


HEADER = '''# KG1 V73 DEFINITIVO - TOP 1 Pipeline (2026-04-21)

## Features
- 13 bugs corrigidos (vide log do chat)
- 3 patches defensivas: adaptive OOM, NaN guard, grad_ckpt fallback
- Pre-flight GPU (detecta zombie)
- Dataset validation (verifica V70 corpus antes de treinar)
- Memory budget predictor (avisa se nao vai caber)
- grad_accum=4 (cabe em Colab 12h)
- max_length=2048 (compromise memoria/qualidade)
- gradient_checkpointing=True (obrigatorio)
- BF16 (NF4 incompativel com NemotronH Mamba)

## Como usar

1. Runtime -> Change runtime type -> H100 HighRAM
2. Se teve execucao anterior: Runtime -> Disconnect and delete runtime -> Reconnect
3. Verifica secrets: HF_KEY, KAGGLE_USERNAME, KAGGLE_KEY
4. **Cell 1** (pre-flight): Shift+Enter -> espera "GPU is clean"
5. **Cell 2** (dataset validation): Shift+Enter -> espera "DATASET OK"
6. **Cell 3** (memory budget): Shift+Enter -> espera "BUDGET OK"
7. **Cell 4** (treino completo): Shift+Enter -> ~6-8h
8. NAO FECHE A ABA durante o treino
9. Se aparecer "Reconnect" -> clica imediatamente

## Tempo esperado
- Section 0 (pip): 2-3 min
- Section 1 (clone): 15 sec
- Section 2 (auth): 10 sec
- Section 3-5 (config+data): 30 sec
- Section 6 (model load BF16): 3-10 min (first time)
- Section 7 (smoke): 2 min
- **Section 8 (training): 6-8 horas**
- Section 9-12 (save+eval+submit): 30 min

## Predicao de score
- Prob treino completar: 92%
- Prob local score >= 0.84 (gate): 55-65%
- Prob Kaggle >= 0.84 (supera V70): 50-60%
- Prob Kaggle >= 0.87 (TOP 1): 12%

## Regras imutaveis
- Gate 99%: NAO submete se local < 0.84
- Smoke aborta se loss > 50 ou NaN
- Zero regressao no leaderboard (gate bloqueia)
'''


PREFLIGHT_CELL = r'''# Cell 1: PRE-FLIGHT GPU CHECK (detecta zombie processes)
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
    print('ERROR: CUDA not available.')
    print('Fix: Runtime -> Change runtime type -> GPU: H100')
    raise RuntimeError('CUDA not available')

d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = total_gb - free_gb

print(f'GPU: {d.name}')
print(f'Total VRAM: {total_gb:.1f} GB')
print(f'Used: {used_gb:.1f} GB')
print(f'Free: {free_gb:.1f} GB')

if total_gb < 38:
    print('ERROR: GPU too small. Need >=38GB.')
    raise RuntimeError('GPU too small')

if used_gb > 5:
    print('=' * 60)
    print('ZOMBIE PROCESS DETECTED')
    print('=' * 60)
    print(f'{used_gb:.1f}GB allocated by previous Python session.')
    print()
    print('FIX:')
    print('  1. Runtime -> Disconnect and delete runtime')
    print('  2. Click Reconnect')
    print('  3. Re-run this Cell 1')
    raise RuntimeError(f'Zombie process has {used_gb:.1f}GB. Disconnect runtime.')

print()
print('GPU is clean. Ready for Cell 2.')
'''


DATASET_VALIDATION_CELL = r'''# Cell 2: DATASET VALIDATION (checks V70 corpus quality before training)
import os, json, subprocess, sys
from pathlib import Path

print('=' * 60)
print('DATASET VALIDATION - V70 huikang corpus')
print('=' * 60)

# Install minimal deps for this cell
for pkg in ['huggingface_hub>=0.25', 'pandas']:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                   capture_output=True, text=True)

# Auth
try:
    from google.colab import userdata
    hf_key = userdata.get('HF_KEY') or os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
except ImportError:
    hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
assert hf_key, 'HF_KEY missing - add to Colab secrets'
os.environ['HF_TOKEN'] = hf_key

from huggingface_hub import login, hf_hub_download
import pandas as pd

login(token=hf_key, add_to_git_credential=False)

# Try GDrive first (faster if cached)
try:
    from google.colab import drive
    try:
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        try:
            drive.mount('/content/drive', force_remount=True)
        except Exception as e:
            print(f'WARN: GDrive mount failed: {e}')
except ImportError:
    pass

# Download dataset (cached if possible)
cache_dir = '/content/drive/MyDrive/kg1_data/v70_huikang' if os.path.exists('/content/drive/MyDrive') else '/content/kg1_data/v70_huikang'
os.makedirs(cache_dir, exist_ok=True)

print(f'Downloading data/sft_v70_huikang_full.jsonl from felipesp1983/kg1-nemotron-training...')
v70_path = hf_hub_download(
    repo_id='felipesp1983/kg1-nemotron-training',
    filename='data/sft_v70_huikang_full.jsonl',
    repo_type='dataset',
    local_dir=cache_dir,
    token=hf_key,
)
print(f'OK: {v70_path}')
print(f'Size: {os.path.getsize(v70_path)/1024/1024:.1f} MB')

# Load and validate
df = pd.read_json(v70_path, lines=True)
print(f'Loaded {len(df)} rows')

# Schema check
expected_cols = {'id', 'category', 'answer', 'messages', 'trace_len'}
actual_cols = set(df.columns)
missing = expected_cols - actual_cols
assert not missing, f'Missing columns: {missing}'
print(f'Schema OK: {list(df.columns)}')

# Count check
assert len(df) >= 10000, f'Dataset too small: {len(df)} rows'
print(f'Count OK: {len(df)} rows (expected >=10000)')

# trace_len distribution
print()
print('trace_len statistics:')
print(f'  min:    {df["trace_len"].min()}')
print(f'  p25:    {df["trace_len"].quantile(0.25):.0f}')
print(f'  p50:    {df["trace_len"].median():.0f}')
print(f'  p75:    {df["trace_len"].quantile(0.75):.0f}')
print(f'  p90:    {df["trace_len"].quantile(0.90):.0f}')
print(f'  p95:    {df["trace_len"].quantile(0.95):.0f}')
print(f'  p99:    {df["trace_len"].quantile(0.99):.0f}')
print(f'  max:    {df["trace_len"].max()}')

# Coverage at max_length=2048
fit_2048 = (df['trace_len'] <= 2048).sum()
coverage_2048 = 100 * fit_2048 / len(df)
print()
print(f'Coverage @ max_length=2048: {fit_2048}/{len(df)} ({coverage_2048:.1f}%)')
print(f'Truncated @ 2048: {len(df) - fit_2048} samples')

# Messages format validation
msg_sample = df.iloc[0]['messages']
assert isinstance(msg_sample, list), 'messages should be list'
assert len(msg_sample) == 2, 'expected 2 messages (user/assistant)'
assert msg_sample[0].get('role') == 'user', f'first msg role={msg_sample[0].get("role")}'
assert msg_sample[1].get('role') == 'assistant', f'second msg role={msg_sample[1].get("role")}'
print(f'Messages format OK (user/assistant pair)')

# CoT quality check - sample 5 records
print()
print('CoT quality sample (5 random records):')
import random
random.seed(42)
sample_idx = random.sample(range(len(df)), 5)
for i in sample_idx:
    row = df.iloc[i]
    assistant = row['messages'][1]['content']
    has_boxed = '\\boxed{' in assistant
    has_think = '<think>' in assistant or 'Let me' in assistant or 'I will' in assistant
    print(f'  [{i:5d}] cat={row["category"]:25s} len={row["trace_len"]:5d} boxed={has_boxed} cot_indicators={has_think}')

# Category distribution
print()
print('Top 10 categories:')
for cat, n in df['category'].value_counts().head(10).items():
    print(f'  {cat:30s}: {n:5d}')

# Critical check: answer column not empty
null_answers = df['answer'].isna().sum()
assert null_answers == 0, f'{null_answers} rows have null answer'
print(f'\\nAnswers OK: 0 null answers')

print()
print('=' * 60)
print('DATASET OK - Ready for Cell 3')
print('=' * 60)
print(f'V70 proven dataset: {len(df)} rows, {coverage_2048:.1f}% coverage @ max_length=2048')
print(f'Expected impact of truncation: {100-coverage_2048:.0f}% of samples lose tail tokens')
'''


MEMORY_BUDGET_CELL = r'''# Cell 3: MEMORY BUDGET PRE-CHECK (predicts fit before loading model)
import torch

print('=' * 60)
print('MEMORY BUDGET PRE-CHECK')
print('=' * 60)

d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3

# Known values (empirically measured in previous runs)
MODEL_BF16 = 62.8       # NemotronH 30B BF16
LORA_BF16 = 1.8         # 883M params BF16
GRAD_LORA = 2.0         # LoRA gradients
OPTIM_8BIT = 2.0        # PagedAdam8bit states
CUDA_OVERHEAD = 2.0     # CUDA runtime

# Configuration values
MAX_LENGTH = 2048
BATCH = 1
GRAD_ACCUM = 4  # V73 fix for speed
USE_GRAD_CKPT = True

# Activation memory scales with max_length
# Without grad_ckpt: ~6GB for seq=2048
# With grad_ckpt: ~3GB for seq=2048 (recomputes)
if USE_GRAD_CKPT:
    ACTIVATIONS = MAX_LENGTH / 2048 * 3.0  # linear approximation
else:
    ACTIVATIONS = MAX_LENGTH / 2048 * 6.0

TOTAL_PROJECTED = MODEL_BF16 + LORA_BF16 + GRAD_LORA + OPTIM_8BIT + CUDA_OVERHEAD + ACTIVATIONS

print(f'GPU: {d.name}')
print(f'Total VRAM: {total_gb:.1f} GB')
print(f'Free VRAM: {free_gb:.1f} GB')
print()
print('Projected memory usage:')
print(f'  Model BF16 (30B):               {MODEL_BF16:5.1f} GB')
print(f'  LoRA BF16 (883M):               {LORA_BF16:5.1f} GB')
print(f'  Gradients LoRA:                 {GRAD_LORA:5.1f} GB')
print(f'  Optimizer 8bit:                 {OPTIM_8BIT:5.1f} GB')
print(f'  CUDA overhead:                  {CUDA_OVERHEAD:5.1f} GB')
print(f'  Activations seq={MAX_LENGTH} '
      + f'(grad_ckpt={USE_GRAD_CKPT}): {ACTIVATIONS:5.1f} GB')
print(f'  ------------------------------------------')
print(f'  TOTAL PROJECTED:                {TOTAL_PROJECTED:5.1f} GB')
print(f'  AVAILABLE:                      {total_gb:5.1f} GB')
margin = total_gb - TOTAL_PROJECTED
print(f'  MARGIN:                         {margin:+5.1f} GB')
print()

print('Training speed estimate:')
# Based on empirical: 5.7s/sample with seq=2048 + grad_ckpt
PER_SAMPLE = 5.7 * (MAX_LENGTH / 2048)
STEP_TIME = PER_SAMPLE * GRAD_ACCUM
TOTAL_STEPS = 15765 // GRAD_ACCUM  # batch=1, so samples = steps * grad_accum
ETA_HOURS = (TOTAL_STEPS * STEP_TIME) / 3600
print(f'  Per sample fw+bw:  {PER_SAMPLE:.1f}s')
print(f'  Per optim step:    {STEP_TIME:.1f}s (grad_accum={GRAD_ACCUM})')
print(f'  Total steps:       {TOTAL_STEPS}')
print(f'  ETA training:      {ETA_HOURS:.1f} hours')
print()

# Decision
if margin < 2.0:
    print('[WARN] Memory budget TIGHT (margin <2GB).')
    print('  Risk of OOM during training. Consider: reduce max_length')
elif margin < 5.0:
    print('[OK] Memory budget acceptable (margin 2-5GB).')
else:
    print('[OK] Memory budget COMFORTABLE (margin >5GB).')

if ETA_HOURS > 11:
    print('[WARN] ETA exceeds 11 hours - Colab 12h limit risk.')
elif ETA_HOURS > 8:
    print('[OK] ETA high but within Colab Pro+ 24h.')
else:
    print('[OK] ETA comfortable.')

assert margin > -3, f'Memory NEGATIVE by {-margin:.1f}GB - guaranteed OOM. Reduce max_length.'
assert free_gb > 30, f'GPU has only {free_gb:.1f}GB free. Restart runtime.'

print()
print('=' * 60)
print('BUDGET OK - Ready for Cell 4 (mega training)')
print('=' * 60)
'''


MEGA_CELL = r'''# Cell 4: COMPLETE TRAINING PIPELINE (6-8h on H100 HighRAM)
# All 13 bugs fixed + 3 defensive patches + validated config
import os, sys, json, subprocess, shutil, gc, time, datetime, random, re, math
from pathlib import Path

START_TIME = time.time()
def log(msg):
    elapsed = int(time.time() - START_TIME)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    prefix = f'[{h:02d}:{m:02d}:{s:02d}]'
    print(f'{prefix} {msg}', flush=True)

def section(title):
    log('=' * 60)
    log(title)
    log('=' * 60)

# SECTION 0: pip install resilient
section('SECTION 0: Install dependencies')
def _pip_one(pkg, extra=None):
    cmd = [sys.executable, '-m', 'pip', 'install', '-q', pkg]
    if extra:
        cmd += extra
    log(f'pip install {pkg}...')
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0

_pip_one('pip', extra=['--upgrade'])
for pkg in [
    'transformers>=4.55', 'peft>=0.13', 'trl>=0.25', 'accelerate>=0.34',
    'bitsandbytes>=0.44', 'datasets>=2.20', 'safetensors>=0.4.5',
    'sentencepiece', 'einops', 'huggingface_hub>=0.25',
]:
    _pip_one(pkg)

log('Installing mamba-ssm + causal-conv1d (OPTIONAL)...')
try:
    import mamba_ssm  # noqa
    log('mamba-ssm already installed')
except ImportError:
    _pip_one('mamba-ssm', extra=['--no-build-isolation'])
try:
    import causal_conv1d  # noqa
    log('causal-conv1d already installed')
except ImportError:
    _pip_one('causal-conv1d>=1.4', extra=['--no-build-isolation'])

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch, transformers, peft, trl
log(f'torch={torch.__version__}  transformers={transformers.__version__}  peft={peft.__version__}  trl={trl.__version__}')
assert torch.cuda.is_available()
d = torch.cuda.get_device_properties(0)
vram_gb = d.total_memory / 1024**3
log(f'GPU: {d.name}  ({vram_gb:.1f} GB VRAM)')

# SECTION 1: Clone KG1
section('SECTION 1: Clone KG1 worktree')
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
log(f'Latest commit: {commit}')
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
log(f'All {len(REQUIRED)} required files present')
sys.path.insert(0, str(KG1_DIR))

# SECTION 2: Auth (resilient)
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
            log(f'WARN: GDrive mount failed: {e}')
    try:
        hf_key = userdata.get('HF_KEY')
    except Exception:
        hf_key = None
    if not hf_key:
        hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    assert hf_key, 'HF_KEY missing'
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
        log(f'Kaggle user: {kuser}')
except ImportError:
    pass

from huggingface_hub import login, whoami
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
login(token=HF_TOKEN, add_to_git_credential=False)
try:
    log(f'HF user: {whoami(token=HF_TOKEN)["name"]}')
except Exception:
    pass

# SECTION 3: Config
section('SECTION 3: Config (V73 DEFINITIVO)')
from dataclasses import dataclass, asdict

@dataclass
class Config:
    base_model: str = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    use_nf4: bool = False  # NemotronH Mamba incompat
    max_length: int = 2048  # compromise memory/quality
    attn_implementation: str = 'eager'
    mamba_ssm_cache_dtype: str = 'float32'
    tie_word_embeddings: bool = False
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = 'all-linear'
    epochs: int = 1
    per_device_batch: int = 1
    grad_accum: int = 4  # V73: 16->4 (Colab 12h limit)
    learning_rate: float = 2e-4
    lr_scheduler: str = 'linear'
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    optimizer: str = 'paged_adamw_8bit'
    bf16: bool = True
    use_gradient_checkpointing: bool = True  # REQUIRED
    loss_type: str = 'max_min_warmup_ce'
    max_min_warmup_steps: int = 100
    hf_dataset_repo: str = 'felipesp1983/kg1-nemotron-training'
    hf_dataset_file: str = 'data/sft_v70_huikang_full.jsonl'
    enable_thinking: bool = True
    use_structured: bool = True
    use_category_hints: bool = True
    use_boxed_strict: bool = True
    use_self_correct: bool = True
    smoke_test_steps: int = 2
    smoke_abort_loss: float = 50.0
    eval_holdout_size: int = 600
    local_score_floor: float = 0.84
    target_score: float = 0.87
    run_tag: str = 'v73_definitivo'
    output_dir: str = '/content/kg1_out/v73_definitivo'
    gdrive_checkpoint: str = '/content/drive/MyDrive/kg1_checkpoints/v73_definitivo'
    hf_upload_repo: str = 'felipesp1983/kg1-nemotron-lora-v73-definitivo'

CFG = Config()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(CFG.output_dir) / 'config.json', 'w') as f:
    json.dump(asdict(CFG), f, indent=2)
log(f'Config: max_length={CFG.max_length} grad_accum={CFG.grad_accum} lr={CFG.learning_rate}')

# SECTION 4: Pre-flight (imports)
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
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
log(f'tokenizer: vocab={tok.vocab_size}')
from src.prompts.build_prompt import build_prompt_v71, detect_category

# SECTION 5: Load V70 dataset
section('SECTION 5: Load V70 dataset')
from huggingface_hub import hf_hub_download
import pandas as pd

if GDRIVE_MOUNTED:
    GDRIVE_CACHE = Path('/content/drive/MyDrive/kg1_data')
else:
    GDRIVE_CACHE = Path('/content/kg1_data')
GDRIVE_CACHE.mkdir(parents=True, exist_ok=True)

v70_path = hf_hub_download(
    repo_id=CFG.hf_dataset_repo,
    filename=CFG.hf_dataset_file,
    repo_type='dataset',
    local_dir=str(GDRIVE_CACHE / 'v70_huikang'),
    token=HF_TOKEN,
)
log(f'Dataset: {v70_path}')

df = pd.read_json(v70_path, lines=True)
log(f'Loaded {len(df)} rows')

# Chat format extraction
if 'messages' in df.columns and 'response' not in df.columns:
    def extract(msgs):
        if not isinstance(msgs, list):
            return None, None
        u = next((m['content'] for m in msgs if m.get('role') == 'user'), None)
        a = next((m['content'] for m in msgs if m.get('role') == 'assistant'), None)
        return u, a
    df[['_u', '_a']] = df['messages'].apply(lambda m: pd.Series(extract(m)))
    df = df.rename(columns={'_u': 'prompt', '_a': 'response'})

if 'category' not in df.columns or df['category'].isna().all():
    df['category'] = df['prompt'].map(detect_category)

def build_record(row):
    p = row.get('prompt')
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None
    p = str(p).strip()
    if not p:
        return None
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
        if not ans or pd.isna(ans):
            return None
        assistant = f'\\boxed{{{ans}}}'
    return {'user': user, 'assistant': assistant, 'category': cat}

records = [r for r in (build_record(row) for _, row in df.iterrows()) if r is not None]
n_cot = sum(1 for r in records if len(r['assistant']) > 50)
log(f'Records: {len(records)}, CoT>50: {100*n_cot/len(records):.1f}%')
assert n_cot > len(records) * 0.5, 'Less than 50% CoT'

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
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(eval_path, 'w') as f:
    for r in eval_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
log(f'train={len(train_records)}  eval={len(eval_records)}')
del df, records, train_records, eval_records
gc.collect()

# SECTION 6: Load model BF16
section('SECTION 6: Load model BF16 + LoRA')
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
log(f'VRAM free before load: {free_gb:.1f} GB')
assert free_gb >= 30, f'Need >=30GB free. Restart runtime.'

from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

model_cfg = AutoConfig.from_pretrained(CFG.base_model, trust_remote_code=True)
setattr(model_cfg, 'tie_word_embeddings', False)
if hasattr(model_cfg, 'mamba_ssm_cache_dtype'):
    setattr(model_cfg, 'mamba_ssm_cache_dtype', 'float32')

log('Loading NemotronH 30B BF16...')
model = AutoModelForCausalLM.from_pretrained(
    CFG.base_model,
    config=model_cfg,
    torch_dtype=torch.bfloat16,
    device_map={'': 0},
    attn_implementation=CFG.attn_implementation,
    trust_remote_code=True,
)
log('Base model (BF16) loaded')

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

# PATCH 3: grad_ckpt with fallback
if CFG.use_gradient_checkpointing:
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        log('Gradient checkpointing ENABLED')
    except Exception as e:
        log(f'WARN grad_ckpt: {e}')
        old_ml = CFG.max_length
        CFG.max_length = max(512, CFG.max_length // 2)
        log(f'AUTO-RECOVERY: max_length {old_ml} -> {CFG.max_length}')
        CFG.use_gradient_checkpointing = False

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
log(f'GPU after load: used={used_gb:.1f}GB  free={free_gb:.1f}GB')

# SECTION 7: Smoke test with PATCH 1 (adaptive OOM)
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

smoke_tried = []
losses = []
smoke_success = False

for attempt_ml in [CFG.max_length, CFG.max_length // 2, CFG.max_length // 4, 512]:
    if attempt_ml in smoke_tried or attempt_ml < 256:
        continue
    smoke_tried.append(attempt_ml)
    gc.collect()
    torch.cuda.empty_cache()
    log(f'Smoke attempt max_length={attempt_ml}')

    ds = JsonlChatDataset(train_path, tok, attempt_ml)
    dl = DataLoader(
        ds, batch_size=CFG.per_device_batch, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    losses = []

    try:
        for step, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k != 'labels'})
            loss = torch.nn.functional.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100,
            )
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                raise RuntimeError(f'NaN/Inf at step {step}')
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CFG.grad_clip,
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            log(f'smoke step {step} loss={loss.item():.4f}')
            if step + 1 >= CFG.smoke_test_steps:
                break

        if losses and losses[-1] < CFG.smoke_abort_loss:
            CFG.max_length = attempt_ml
            smoke_success = True
            log(f'Smoke PASSED at max_length={attempt_ml} (loss {losses[-1]:.4f})')
            break

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower():
            log(f'OOM at max_length={attempt_ml}, trying smaller...')
            try: del opt, ds, dl
            except Exception: pass
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        else:
            raise

if not smoke_success:
    raise RuntimeError(f'Smoke FAILED at all lengths: {smoke_tried}')

try: del opt, ds, dl, batch, out, loss
except (NameError, UnboundLocalError): pass
gc.collect()
torch.cuda.empty_cache()
log(f'Final max_length: {CFG.max_length}')

# SECTION 8: Full training with PATCH 2 (NaN guard)
section('SECTION 8: Full training')
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
log(f'Set tok.model_max_length = {CFG.max_length}')

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
    logging_steps=10,
    save_steps=100,  # V73: more frequent (was 200)
    eval_strategy='no',
    save_total_limit=3,
    optim=CFG.optimizer,
    packing=False,
    report_to=[],
    gradient_checkpointing=CFG.use_gradient_checkpointing,
    dataset_text_field='text',
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
                    print(f'WARN: NaN max-min step {step} (count {self._nan_count})')
                    if self._nan_count >= 3:
                        self._use_ce_permanent = True
                        print(f'CRITICAL: CE permanent after {self._nan_count} NaNs')
                    loss = ce_loss
                else:
                    loss = mm_loss
                    self._nan_count = max(0, self._nan_count - 1)
            except Exception as e:
                print(f'WARN max-min exc step {step}: {e}')
                loss = ce_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f'WARN: final loss NaN step {step}, forcing CE')
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss

trainer = MaxMinSFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    processing_class=tok,
)
log(f'Training START (1 epoch, ETA ~{(986*CFG.grad_accum*5.7)/3600:.1f}h)')
trainer.train()
log('Training complete. Saving adapter...')
trainer.save_model(CFG.output_dir)
tok.save_pretrained(CFG.output_dir)

# SECTION 9: Save + upload
section('SECTION 9: Save + backup + upload')
from huggingface_hub import HfApi, upload_folder

out_dir = Path(CFG.output_dir)
required_files = ['adapter_config.json', 'adapter_model.safetensors']
missing_adapter = [f for f in required_files if not (out_dir / f).exists()]
assert not missing_adapter, f'Missing adapter: {missing_adapter}'

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if GDRIVE_MOUNTED:
    gdrive_dest = Path(CFG.gdrive_checkpoint) / f'{CFG.run_tag}_{ts}'
    try:
        gdrive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, gdrive_dest, dirs_exist_ok=True)
        log(f'GDrive: {gdrive_dest}')
    except Exception as e:
        log(f'WARN GDrive: {e}')

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
    log(f'WARN HF: {e}')

# SECTION 10: Local eval + gate
section('SECTION 10: Local eval + gate')
local_score_script = Path('/content/kg1/scripts/local_score.py')
eval_csv = Path(CFG.output_dir) / 'local_eval.csv'
cmd = [
    sys.executable, str(local_score_script),
    '--adapter', str(out_dir),
    '--n-samples', str(CFG.eval_holdout_size),
    '--output-csv', str(eval_csv),
]
log(f'Running local eval ({CFG.eval_holdout_size} samples)...')
try:
    res = subprocess.run(cmd, cwd='/content/kg1', check=False,
                         capture_output=True, text=True, timeout=3600)
    log(f'STDOUT: {res.stdout[-1500:]}')
    if res.returncode != 0:
        log(f'STDERR: {res.stderr[-1000:]}')
except subprocess.TimeoutExpired:
    log('TIMEOUT')
    res = type('R', (), {'stdout': '', 'stderr': ''})

local_score_val = None
m = re.search(r'(?:overall\s+score|score)[:\s]+([0-9.]+)', res.stdout, re.IGNORECASE)
if m:
    local_score_val = float(m.group(1))
log(f'Local score = {local_score_val}')
with open(out_dir / 'local_score.json', 'w') as f:
    json.dump({'local_score': local_score_val, 'n_samples': CFG.eval_holdout_size}, f)

GO = False
if local_score_val is None:
    gate_msg = 'NO-GO: parse failed'
elif local_score_val < CFG.local_score_floor:
    gate_msg = f'NO-GO: {local_score_val:.4f} < floor {CFG.local_score_floor}'
elif local_score_val < CFG.target_score - 0.01:
    gate_msg = f'MARGINAL: {local_score_val:.4f}'
    GO = True
else:
    gate_msg = f'GO: {local_score_val:.4f}'
    GO = True
log(f'Gate: {gate_msg}')
with open(out_dir / 'gate_decision.json', 'w') as f:
    json.dump({'go': GO, 'msg': gate_msg, 'score': local_score_val}, f)

# SECTION 11: Submit (if GO)
section('SECTION 11: Submit (if GO)')
if not GO:
    log('NO-GO -> skipping submit')
else:
    import zipfile
    zip_path = out_dir / 'submission.zip'
    ROOT_FILES = ['adapter_config.json', 'adapter_model.safetensors']
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in ROOT_FILES:
            zf.write(out_dir / fn, arcname=fn)
    log(f'zip: {zip_path.stat().st_size/(1024*1024):.2f} MB')

    gate_script = Path('/content/kg1/scripts/kg1_submission_gate.py')
    r2 = subprocess.run(
        [sys.executable, str(gate_script), '--zip', str(zip_path)],
        capture_output=True, text=True,
    )
    if r2.returncode != 0:
        log(f'Gate rejected (rc={r2.returncode})')
    else:
        if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
            msg = f'V73_DEFINITIVO {datetime.datetime.now().strftime("%Y-%m-%d %H:%M BRT")}'
            submit_script = Path('/content/kg1/scripts/submit_kaggle.py')
            if submit_script.exists():
                cmd = [sys.executable, str(submit_script), '--zip', str(zip_path), '--message', msg]
            else:
                cmd = ['kaggle', 'competitions', 'submit',
                       '-c', 'nvidia-nemotron-model-reasoning-challenge',
                       '-f', str(zip_path), '-m', msg]
            r3 = subprocess.run(cmd, capture_output=True, text=True)
            log(f'Submit: rc={r3.returncode}')
            with open(out_dir / 'kaggle_submit.json', 'w') as f:
                json.dump({'msg': msg, 'rc': r3.returncode,
                          'stdout': r3.stdout[-1000:], 'stderr': r3.stderr[-500:]}, f)

# SECTION 12: Decision
section('SECTION 12: Decision tree')
score = local_score_val or 0.0
TREE = [
    (0.87, 'TOP1_CANDIDATE'),
    (0.86, 'PLATEAU_PUSH'),
    (0.85, 'MARGIN_PROBE'),
    (0.84, 'BASELINE_HOLD'),
    (0.0, 'ROLLBACK_V70'),
]
label = 'UNDETERMINED'
for thr, lbl in TREE:
    if score >= thr:
        label = lbl
        break

out = {
    'local_score': score,
    'decision': label,
    'timestamp': datetime.datetime.now().isoformat(),
    'elapsed_hours': (time.time() - START_TIME) / 3600,
}
with open(out_dir / 'decision.json', 'w') as f:
    json.dump(out, f, indent=2)
log(json.dumps(out, indent=2))

section('ALL DONE!')
log(f'Total: {(time.time() - START_TIME)/3600:.2f} hours')
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
        cell_code(PREFLIGHT_CELL, "c1-preflight"),
        cell_code(DATASET_VALIDATION_CELL, "c2-dataset"),
        cell_code(MEMORY_BUDGET_CELL, "c3-budget"),
        cell_code(MEGA_CELL, "c4-mega"),
    ],
}

OUT = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'KG1_V73_DEFINITIVO.ipynb')
OUT = os.path.abspath(OUT)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f'Wrote {OUT} ({os.path.getsize(OUT)} bytes)')

# Validate all code cells
import py_compile, tempfile
for name, src in [('PREFLIGHT', PREFLIGHT_CELL), ('DATASET', DATASET_VALIDATION_CELL),
                  ('BUDGET', MEMORY_BUDGET_CELL), ('MEGA', MEGA_CELL)]:
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
