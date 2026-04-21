"""Build KG1_V72_CLEAN.ipynb - fresh notebook with ALL accumulated fixes.

Cells:
  0. Markdown header (instructions)
  1. Pre-flight GPU check (detects zombie processes, fail-fast)
  2. Mega-cell with complete training pipeline
"""
import json, os

HEADER = '''# KG1 V72 CLEAN - Training pipeline with ALL fixes (2026-04-21)

## USAGE

1. Runtime -> Change runtime type -> H100 + High-RAM
2. If you had previous failed execution, do: Runtime -> Disconnect and delete runtime
3. Verify Colab secrets are set: HF_KEY, KAGGLE_USERNAME, KAGGLE_KEY
4. Run Cell 1 first (GPU pre-flight check). If it FAILS, disconnect runtime.
5. Run Cell 2 (mega training cell). Takes 4-8 hours on H100.
6. Do NOT close the tab during training.

## FIXES APPLIED

- pip install resilient (no tight version pins)
- Force re-clone KG1 branch
- HF login() (not deprecated HfFolder)
- GDrive mount with fallback (force_remount -> skip)
- NF4 DISABLED (NemotronH Mamba custom layers break NF4 matmul)
- BF16 full loading (fits H100 80GB with margin)
- max_length=2048 (V70 compatible, avoids OOM)
- Relaxed VRAM assertions (warn instead of fail)
- Pre-smoke cleanup (gc + empty_cache)
- Dataset: felipesp1983/kg1-nemotron-training sft_v70_huikang_full.jsonl (100% CoT)
- max-min-warmup-CE loss (V5 Tong winner recipe)
- 99% rule gate before Kaggle submit
- Automatic submit if local_score >= 0.84

## EXPECTED BEHAVIOR

- Section 0 (pip): 2-3 min
- Section 1 (clone): 15 sec
- Section 2 (auth): 10 sec
- Section 3-5 (config+data): 30 sec
- Section 6 (model load BF16): 5-10 min (first time download)
- Section 7 (smoke test): 1-2 min
- Section 8 (training): 4-8 hours
- Section 9-12 (save+eval+submit): 30 min
'''


PREFLIGHT = r'''# Cell 1: Pre-flight GPU check
# Detects zombie Python processes holding GPU memory.
# If any found, instructs user to disconnect runtime (not just restart).
import subprocess, torch, gc, sys

print('=' * 60)
print('PRE-FLIGHT GPU CHECK')
print('=' * 60)

# Free any cached memory first
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Check GPU via nvidia-smi
try:
    r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    print(r.stdout)
except Exception as e:
    print(f'WARN: nvidia-smi failed: {e}')

if not torch.cuda.is_available():
    print('ERROR: CUDA not available.')
    print('Fix: Runtime -> Change runtime type -> GPU: H100')
    sys.exit(1)

d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = total_gb - free_gb

print(f'GPU: {d.name}')
print(f'Total VRAM: {total_gb:.1f} GB')
print(f'Used: {used_gb:.1f} GB')
print(f'Free: {free_gb:.1f} GB')

if total_gb < 38:
    print('\nERROR: GPU too small. Need >=38GB (A100 HighRAM or H100).')
    print('Fix: Runtime -> Change runtime type -> H100 + High-RAM')
    sys.exit(1)

if used_gb > 5:
    print('\n' + '=' * 60)
    print('ZOMBIE PROCESS DETECTED')
    print('=' * 60)
    print(f'{used_gb:.1f}GB are allocated by a previous Python session.')
    print()
    print('FIX (required):')
    print('  1. Runtime -> Disconnect and delete runtime')
    print('  2. Confirm disconnect')
    print('  3. Click Reconnect (top right)')
    print('  4. Verify runtime is H100 HighRAM')
    print('  5. Run this Cell 1 again')
    print()
    print('Why: Restart runtime does NOT kill zombie Python processes.')
    print('Only DELETE runtime fully releases GPU memory.')
    print()
    print('DO NOT PROCEED TO CELL 2 UNTIL GPU IS CLEAN.')
    raise RuntimeError(f'GPU has {used_gb:.1f}GB allocated by zombie process. Disconnect runtime.')

print('\nGPU is clean. Ready to proceed to Cell 2.')
print('Expected free: ~78GB before loading model')
'''


MEGA = r'''# Cell 2: Complete training pipeline (single cell, 4-8h on H100)
# All fixes applied: BF16 (no NF4), max_length=2048, resilient auth + data
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

# ============================================================
# SECTION 0: pip install deps (resilient)
# ============================================================
section('SECTION 0: Install dependencies')
def _pip_one(pkg, extra=None):
    cmd = [sys.executable, '-m', 'pip', 'install', '-q', pkg]
    if extra:
        cmd += extra
    log(f'pip install {pkg}...')
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log(f'  TIMEOUT')
        return False
    if r.returncode != 0:
        log(f'  FAIL: {r.stderr[-300:]}')
        return False
    return True

_pip_one('pip', extra=['--upgrade'])
for pkg in [
    'transformers>=4.55', 'peft>=0.13', 'trl>=0.25', 'accelerate>=0.34',
    'bitsandbytes>=0.44', 'datasets>=2.20', 'safetensors>=0.4.5',
    'sentencepiece', 'einops', 'huggingface_hub>=0.25',
]:
    _pip_one(pkg)

log('Installing mamba-ssm + causal-conv1d (OPTIONAL, slow-path fallback)...')
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
log('PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True')

import torch, transformers, peft, trl
log(f'torch={torch.__version__}  transformers={transformers.__version__}  peft={peft.__version__}  trl={trl.__version__}')
assert torch.cuda.is_available(), 'CUDA not available - switch to A100/H100'
d = torch.cuda.get_device_properties(0)
vram_gb = d.total_memory / 1024**3
log(f'GPU: {d.name}  ({vram_gb:.1f} GB VRAM)')
assert vram_gb >= 38, f'Need >=38GB VRAM, got {vram_gb:.1f}'

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
log(f'VRAM free at start: {free_gb:.1f}GB')
if free_gb < 70:
    log(f'WARN: Only {free_gb:.1f}GB free. Expected ~78GB on fresh runtime.')
    log('If this fails OOM in Section 6, disconnect and delete runtime first.')

# ============================================================
# SECTION 1: Clone KG1 worktree (force fresh)
# ============================================================
section('SECTION 1: Clone KG1 worktree')
KG1_DIR = Path('/content/kg1')
REPO = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
BRANCH = os.environ.get('KG1_BRANCH', 'claude/competent-shamir')
if KG1_DIR.exists():
    log(f'Removing stale {KG1_DIR}')
    shutil.rmtree(KG1_DIR)
log(f'Cloning {REPO} branch={BRANCH}')
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
log(f'All {len(REQUIRED)} required files present.')
sys.path.insert(0, str(KG1_DIR))

# ============================================================
# SECTION 2: Mount GDrive (resilient) + HF/Kaggle auth
# ============================================================
section('SECTION 2: Auth')
GDRIVE_MOUNTED = False
try:
    from google.colab import drive, userdata  # type: ignore
    try:
        drive.mount('/content/drive', force_remount=False)
        GDRIVE_MOUNTED = True
        log('GDrive mounted OK')
    except Exception as e1:
        log(f'Initial mount failed: {e1}. Trying force_remount=True...')
        try:
            drive.mount('/content/drive', force_remount=True)
            GDRIVE_MOUNTED = True
            log('GDrive mounted OK (force_remount)')
        except Exception as e2:
            log(f'WARN: GDrive mount failed: {e2}')
            log('Continuing WITHOUT GDrive. Checkpoints will only be in /content/')
    try:
        hf_key = userdata.get('HF_KEY')
    except Exception:
        hf_key = None
    if not hf_key:
        hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    assert hf_key, 'HF_KEY missing - add to Colab secrets (name: HF_KEY)'
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
    else:
        log('WARN: Kaggle creds not set (submit will be skipped)')
except ImportError:
    log('Not in Colab - using env vars')

from huggingface_hub import login, whoami
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
login(token=HF_TOKEN, add_to_git_credential=False)
try:
    log(f'HF user: {whoami(token=HF_TOKEN)["name"]}')
except Exception as e:
    log(f'WARN whoami: {e}')

# ============================================================
# SECTION 3: Config (BF16 + max_length=2048)
# ============================================================
section('SECTION 3: Config')
from dataclasses import dataclass, asdict

@dataclass
class Config:
    base_model: str = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    use_nf4: bool = False  # DISABLED: NemotronH custom Mamba/MoE breaks NF4 matmul
    max_length: int = 2048  # V70 compatible, fits H100 80GB
    attn_implementation: str = 'eager'
    mamba_ssm_cache_dtype: str = 'float32'
    tie_word_embeddings: bool = False
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = 'all-linear'
    epochs: int = 1
    per_device_batch: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = 'linear'
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    optimizer: str = 'paged_adamw_8bit'
    bf16: bool = True
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
    run_tag: str = 'v72_clean'
    output_dir: str = '/content/kg1_out/v72_clean'
    gdrive_checkpoint: str = '/content/drive/MyDrive/kg1_checkpoints/v72_clean'
    hf_upload_repo: str = 'felipesp1983/kg1-nemotron-lora-v72-clean'

CFG = Config()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(CFG.output_dir) / 'config.json', 'w') as f:
    json.dump(asdict(CFG), f, indent=2)
log(f'Config -> {CFG.output_dir}/config.json')
log(f'  use_nf4={CFG.use_nf4}  max_length={CFG.max_length}  lr={CFG.learning_rate}')

# ============================================================
# SECTION 4: Pre-flight (modules + tokenizer)
# ============================================================
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
    log(f'imported {mod}')

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
log(f'tokenizer: vocab={tok.vocab_size}  pad={tok.pad_token}')
try:
    _ = tok.apply_chat_template(
        [{'role': 'user', 'content': 'test'}],
        enable_thinking=True, tokenize=False,
    )
    log('enable_thinking=True SUPPORTED')
except TypeError as e:
    log(f'WARN enable_thinking: {e}')

from src.reasoners.bit_manipulation_pairs import generate_cot as gen_bit
pred, _ = gen_bit([('00000000', '10101010'), ('11111111', '01010101')], '10101010')
assert pred is not None, 'bit_manipulation self-test failed'
log(f'bit_manipulation self-test OK  pred={pred}')

from src.prompts.build_prompt import build_prompt_v71, detect_category

# ============================================================
# SECTION 5: Load V70 dataset
# ============================================================
section('SECTION 5: Load V70 dataset')
from huggingface_hub import hf_hub_download
import pandas as pd

if GDRIVE_MOUNTED:
    GDRIVE_CACHE = Path('/content/drive/MyDrive/kg1_data')
else:
    GDRIVE_CACHE = Path('/content/kg1_data')
GDRIVE_CACHE.mkdir(parents=True, exist_ok=True)
log(f'Using data cache: {GDRIVE_CACHE}')
log(f'Downloading {CFG.hf_dataset_file} from {CFG.hf_dataset_repo}...')
v70_path = hf_hub_download(
    repo_id=CFG.hf_dataset_repo,
    filename=CFG.hf_dataset_file,
    repo_type='dataset',
    local_dir=str(GDRIVE_CACHE / 'v70_huikang'),
    token=HF_TOKEN,
)
log(f'OK: {v70_path}  ({os.path.getsize(v70_path)/1024/1024:.1f} MB)')

df = pd.read_json(v70_path, lines=True)
log(f'Loaded {len(df)} rows  cols={list(df.columns)}')

if 'messages' in df.columns and 'response' not in df.columns:
    def extract(msgs):
        if not isinstance(msgs, list):
            return None, None
        u = next((m['content'] for m in msgs if m.get('role') == 'user'), None)
        a = next((m['content'] for m in msgs if m.get('role') == 'assistant'), None)
        return u, a
    df[['_u', '_a']] = df['messages'].apply(lambda m: pd.Series(extract(m)))
    df = df.rename(columns={'_u': 'prompt', '_a': 'response'})
    log('Extracted prompt/response from messages column')

if 'category' not in df.columns or df['category'].isna().all():
    df['category'] = df['prompt'].map(detect_category)

log(f'Category distribution (top 10):')
for cat, n in df['category'].value_counts().head(10).items():
    log(f'  {cat}: {n}')

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
log(f'Built {len(records)} records (dropped {len(df)-len(records)} null)')
n_cot = sum(1 for r in records if len(r['assistant']) > 50)
log(f'With CoT (>50 chars): {n_cot}/{len(records)}  ({100*n_cot/len(records):.1f}%)')
assert n_cot > len(records) * 0.5, 'Less than 50% records have CoT'

random.seed(42)
idx = list(range(len(records)))
random.shuffle(idx)
eval_n = min(CFG.eval_holdout_size, max(50, len(records) // 20))
eval_set = set(idx[:eval_n])
train_records = [records[i] for i in idx if i not in eval_set]
eval_records = [records[i] for i in idx if i in eval_set]
log(f'train={len(train_records)}  eval={len(eval_records)}')

train_path = Path(CFG.output_dir) / 'train.jsonl'
eval_path = Path(CFG.output_dir) / 'eval.jsonl'
with open(train_path, 'w') as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(eval_path, 'w') as f:
    for r in eval_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
log(f'wrote {train_path}  ({train_path.stat().st_size/1024/1024:.2f} MB)')
log(f'wrote {eval_path}  ({eval_path.stat().st_size/1024/1024:.2f} MB)')

del df, records, train_records, eval_records
gc.collect()

# ============================================================
# SECTION 6: Load model BF16 + LoRA
# ============================================================
section('SECTION 6: Load model BF16 + LoRA')
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
log(f'VRAM free before load: {free_gb:.1f} GB')
if free_gb < 30:
    log('=' * 60)
    log('CRITICAL: Insufficient VRAM. GPU has zombie process.')
    log('=' * 60)
    log('Fix: Runtime -> Disconnect and delete runtime -> Reconnect')
    log('Then re-run Cell 1 + Cell 2 from scratch.')
    raise RuntimeError(f'Only {free_gb:.1f}GB free - need >=30GB. Disconnect runtime.')

from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

model_cfg = AutoConfig.from_pretrained(CFG.base_model, trust_remote_code=True)
setattr(model_cfg, 'tie_word_embeddings', False)
if hasattr(model_cfg, 'mamba_ssm_cache_dtype'):
    setattr(model_cfg, 'mamba_ssm_cache_dtype', 'float32')

log('Loading NemotronH 30B BF16 (first time 5-10 min, cached 1-2 min)...')
model = AutoModelForCausalLM.from_pretrained(
    CFG.base_model,
    config=model_cfg,
    torch_dtype=torch.bfloat16,
    device_map={'': 0},
    attn_implementation=CFG.attn_implementation,
    trust_remote_code=True,
)
log('Base model (BF16) loaded.')

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

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
log(f'GPU after load: used={used_gb:.1f}GB  free={free_gb:.1f}GB')
if free_gb < 10:
    log(f'WARN: Only {free_gb:.1f}GB free - may OOM during training')
else:
    log(f'OK: {free_gb:.1f}GB free sufficient for seq=2048 training')

# ============================================================
# SECTION 7: Smoke test (2 steps, abort if loss>50)
# ============================================================
section('SECTION 7: Smoke test')
gc.collect()
torch.cuda.empty_cache()
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
log(f'VRAM free before smoke: {free_gb:.1f}GB')
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
dl = DataLoader(
    ds, batch_size=CFG.per_device_batch, shuffle=True,
    collate_fn=lambda b: collate(b, tok.pad_token_id),
)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
losses = []
for step, batch in enumerate(dl):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**{k: v for k, v in batch.items() if k != 'labels'})
    loss = torch.nn.functional.cross_entropy(
        out.logits.view(-1, out.logits.size(-1)),
        batch['labels'].view(-1),
        ignore_index=-100,
    )
    losses.append(loss.item())
    assert not math.isnan(loss.item()) and not math.isinf(loss.item()), \
        f'NaN/Inf loss at step {step}'
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], CFG.grad_clip,
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    log(f'smoke step {step}  loss={loss.item():.4f}')
    if step + 1 >= CFG.smoke_test_steps:
        break

assert losses[-1] < CFG.smoke_abort_loss, \
    f'ABORT smoke loss {losses[-1]:.2f} > threshold'
log(f'Smoke test PASSED (final loss {losses[-1]:.4f})')
del opt, batch, out, loss, losses, ds, dl
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# SECTION 8: Full training (max-min + CE warmup)
# ============================================================
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
    save_steps=200,
    eval_strategy='no',
    save_total_limit=2,
    optim=CFG.optimizer,
    max_seq_length=CFG.max_length,
    packing=False,
    report_to=[],
    gradient_checkpointing=False,
    dataset_text_field='text',
)

class MaxMinSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        logits = outputs.logits
        step = int(self.state.global_step)
        if CFG.loss_type == 'ce':
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )
        elif CFG.loss_type == 'max_min':
            loss = max_min_logprob_loss(logits, labels)
        else:
            if step < CFG.max_min_warmup_steps:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
                )
            else:
                loss = max_min_logprob_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = MaxMinSFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    processing_class=tok,
)
log('Training START (1 epoch, ~4-8h on H100 BF16)')
trainer.train()
log('Training complete. Saving adapter...')
trainer.save_model(CFG.output_dir)
tok.save_pretrained(CFG.output_dir)
log(f'Saved to {CFG.output_dir}')

# ============================================================
# SECTION 9: Save + GDrive backup + HF upload
# ============================================================
section('SECTION 9: Save + backup + upload')
from huggingface_hub import HfApi, upload_folder

out_dir = Path(CFG.output_dir)
required_files = ['adapter_config.json', 'adapter_model.safetensors']
missing_adapter = [f for f in required_files if not (out_dir / f).exists()]
assert not missing_adapter, f'Missing adapter files: {missing_adapter}'

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if GDRIVE_MOUNTED:
    gdrive_dest = Path(CFG.gdrive_checkpoint) / f'{CFG.run_tag}_{ts}'
    try:
        gdrive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, gdrive_dest, dirs_exist_ok=True)
        log(f'GDrive checkpoint -> {gdrive_dest}')
    except Exception as e:
        log(f'WARN GDrive save: {e}')
else:
    log('GDrive not mounted - skipping GDrive backup')

api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(CFG.hf_upload_repo, private=True, exist_ok=True)
    upload_folder(
        repo_id=CFG.hf_upload_repo,
        folder_path=str(out_dir),
        allow_patterns=['adapter_*', 'tokenizer*', 'special_tokens*', 'config.json'],
        token=HF_TOKEN,
    )
    log(f'Uploaded -> HF: {CFG.hf_upload_repo}')
except Exception as e:
    log(f'WARN HF upload: {e}')

# ============================================================
# SECTION 10: Local eval + 99% rule gate
# ============================================================
section('SECTION 10: Local eval + gate')
local_score_script = Path('/content/kg1/scripts/local_score.py')
eval_csv = Path(CFG.output_dir) / 'local_eval.csv'
cmd = [
    sys.executable, str(local_score_script),
    '--adapter', str(out_dir),
    '--n-samples', str(CFG.eval_holdout_size),
    '--output-csv', str(eval_csv),
]
log(f'Running: {" ".join(cmd)}')
try:
    res = subprocess.run(cmd, cwd='/content/kg1', check=False,
                         capture_output=True, text=True, timeout=3600)
    log(f'local_score STDOUT tail:\n{res.stdout[-1500:]}')
    if res.returncode != 0:
        log(f'local_score STDERR:\n{res.stderr[-1000:]}')
except subprocess.TimeoutExpired:
    log('local_score TIMEOUT')
    res = type('R', (), {'stdout': '', 'stderr': ''})

local_score_val = None
m = re.search(r'(?:overall\s+score|score)[:\s]+([0-9.]+)', res.stdout, re.IGNORECASE)
if m:
    local_score_val = float(m.group(1))
log(f'Parsed local score = {local_score_val}')
with open(out_dir / 'local_score.json', 'w') as f:
    json.dump({'local_score': local_score_val, 'n_samples': CFG.eval_holdout_size}, f)

GO = False
if local_score_val is None:
    gate_msg = 'NO-GO: parse failed (99% rule)'
elif local_score_val < CFG.local_score_floor:
    gate_msg = f'NO-GO: {local_score_val:.4f} < floor {CFG.local_score_floor}'
elif local_score_val < CFG.target_score - 0.01:
    gate_msg = f'MARGINAL: {local_score_val:.4f} ~ target {CFG.target_score}'
    GO = True
else:
    gate_msg = f'GO: {local_score_val:.4f} >= target {CFG.target_score}'
    GO = True
log(f'Gate decision: {gate_msg}')
with open(out_dir / 'gate_decision.json', 'w') as f:
    json.dump({'go': GO, 'msg': gate_msg, 'score': local_score_val}, f)

# ============================================================
# SECTION 11: Build ZIP + kg1_submission_gate + Kaggle submit (if GO)
# ============================================================
section('SECTION 11: Submit (if GO)')
if not GO:
    log('NO-GO -> skipping submit. Rollback to V70 or retry.')
else:
    import zipfile
    zip_path = out_dir / 'submission.zip'
    ROOT_FILES = ['adapter_config.json', 'adapter_model.safetensors']
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in ROOT_FILES:
            zf.write(out_dir / fn, arcname=fn)
    size_mb = zip_path.stat().st_size / (1024*1024)
    log(f'submission.zip: {size_mb:.2f} MB')

    gate_script = Path('/content/kg1/scripts/kg1_submission_gate.py')
    r2 = subprocess.run(
        [sys.executable, str(gate_script), '--zip', str(zip_path)],
        capture_output=True, text=True,
    )
    log(f'submission_gate STDOUT:\n{r2.stdout[-1000:]}')
    if r2.returncode != 0:
        log(f'submission_gate REJECTED (rc={r2.returncode})')
        log(f'STDERR:\n{r2.stderr[-500:]}')
    else:
        log('submission_gate PASSED.')
        if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
            msg = f'V72_CLEAN {datetime.datetime.now().strftime("%Y-%m-%d %H:%M BRT")}'
            submit_script = Path('/content/kg1/scripts/submit_kaggle.py')
            if submit_script.exists():
                submit_cmd = [sys.executable, str(submit_script),
                              '--zip', str(zip_path), '--message', msg]
            else:
                submit_cmd = [
                    'kaggle', 'competitions', 'submit',
                    '-c', 'nvidia-nemotron-model-reasoning-challenge',
                    '-f', str(zip_path), '-m', msg,
                ]
            log(f'Submitting: {" ".join(submit_cmd)}')
            r3 = subprocess.run(submit_cmd, capture_output=True, text=True)
            log(f'Submit STDOUT:\n{r3.stdout}')
            log(f'Submit STDERR:\n{r3.stderr}')
            with open(out_dir / 'kaggle_submit.json', 'w') as f:
                json.dump({
                    'msg': msg,
                    'returncode': r3.returncode,
                    'stdout_tail': r3.stdout[-1000:],
                    'stderr_tail': r3.stderr[-500:],
                }, f)
            log('Submit dispatched. Monitor at kaggle.com/competitions Submissions.')
        else:
            log('Kaggle creds missing - zip built but not submitted')

# ============================================================
# SECTION 12: Decision tree
# ============================================================
section('SECTION 12: Decision tree')
score = local_score_val or 0.0
DECISION_TREE = [
    (0.87, 'TOP1_CANDIDATE', 'Stage 2 (LoRA Soup DARE-TIES) + 3 seeds'),
    (0.86, 'PLATEAU_PUSH', 'Add programmatic per-family solvers inference'),
    (0.85, 'MARGIN_PROBE', 'Ablation V71b DoRA + V71c rank_pattern'),
    (0.84, 'BASELINE_HOLD', 'No regression. CoT quality (distill)'),
    (0.0, 'ROLLBACK_V70', 'Score < 0.84 - ROLLBACK. Audit before retry'),
]
label, plan = 'UNDETERMINED', 'check Kaggle LB manually'
for thr, lbl, p in DECISION_TREE:
    if score >= thr:
        label, plan = lbl, p
        break

out = {
    'local_score': score,
    'decision_label': label,
    'next_action': plan,
    'timestamp': datetime.datetime.now().isoformat(),
    'elapsed_hours': (time.time() - START_TIME) / 3600,
}
with open(out_dir / 'decision.json', 'w') as f:
    json.dump(out, f, indent=2)
log(json.dumps(out, indent=2))

section('ALL DONE!')
log(f'Total elapsed: {(time.time() - START_TIME)/3600:.2f} hours')
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
        cell_code(PREFLIGHT, "cell1-preflight"),
        cell_code(MEGA, "cell2-mega"),
    ],
}

OUT = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'KG1_V72_CLEAN.ipynb')
OUT = os.path.abspath(OUT)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f'Wrote {OUT} ({os.path.getsize(OUT)} bytes)')

# py_compile
import py_compile, tempfile
for name, src in [('PREFLIGHT', PREFLIGHT), ('MEGA', MEGA)]:
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as t:
        t.write(src)
        tpath = t.name
    try:
        py_compile.compile(tpath, doraise=True)
        print(f'{name} py_compile OK ({src.count(chr(10))} lines)')
    except py_compile.PyCompileError as e:
        print(f'{name} SYNTAX ERROR:\n{e}')
        raise
    os.unlink(tpath)
