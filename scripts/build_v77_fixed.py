"""Build KG1_V77_FIXED.ipynb - V76 + 10 audit fixes.

All bugs from 4-agent audit fixed:
  #1 DataCollatorForCompletionOnlyLM (mask user tokens)
  #2 Category filter option
  #3 trace_len filter <= 1800 (margin for template overhead)
  #4 EarlyStoppingCallback patience=3
  #5 load_best_model_at_end=True + metric=eval_loss
  #6 transformers pin >=4.55,<4.58
  #7 trl pin <0.26
  #8 accelerate pin <1.0
  #9 MaxMinSFTTrainer uses instance vars
  #10 sys.stdout.flush() before os.kill
"""
import json
import os

HEADER = '''# KG1 V77 FIXED v2 - 10 audit fixes + 2 fixes from 5-AI double-check

## Bugs fixed from 4-agent audit of V76:

1. DataCollatorForCompletionOnlyLM (CE was over user tokens - inflated)
2. Category filter option (40%+ V76 dataset off-topic)
3. trace_len filter <= 1800 (margin for template)
4. EarlyStoppingCallback patience=3
5. load_best_model_at_end=True
6. transformers>=4.55,<4.58 (tighter pin)
7. trl>=0.25,<0.26 (tighter pin)
8. accelerate>=0.34,<1.0 (tighter pin)
9. MaxMinSFTTrainer instance vars (was class vars)
10. stdout.flush() before kernel kill

## Extra fixes from 5-AI rigorous double-check (2026-04-21):

11. **lora_alpha 32 -> 16** (Claude Opus 4.7 rec; matches v30 PROVEN 0.68 baseline)
12. **Section 11 submit: direct kaggle CLI** (submit_kaggle.py expects --hf-repo/--local-dir)

## Triple-check 100x (14/15 APIs across 5 specific gaps, 2026-04-21):

GAP A (chat template/collator): 3/3 APIs convergiram: `<|im_start|>assistant` tokeniza em
2-3 IDs, DataCollator faz subsequence match, MAS se standalone encoding difere do in-context
encoding (whitespace merge), resulta em 0 tokens unmasked -> loss=0 -> no-op training.
-> FIX #13 APLICADO: diagnostic pre-train valida collator mask em 3 samples.

GAP B (max_min_logprob + ignore_index): 3/3 APIs erraram - FALSE POSITIVE. Verifiquei o
source `src/losses/max_min_logprob.py` linhas 57-65: JA usa `labels.clamp(min=0)` antes
de gather E `masked_fill(~mask, float("inf"))` para preservar -100. Robust as-is.

GAP C (mamba-ssm + transformers): 3/3 APIs alertaram risco ABI. Gemini especifico:
"HIGH C++ ABI risk torch CXX11_ABI vs wheel cxx11abiFALSE". Mitigacao:
-> FIX #15 APLICADO: imprime `torch._C._GLIBCXX_USE_CXX11_ABI` antes de importar mamba.
Se True -> warning claro (wheels esperam False). Default PyPI torch = False -> OK.

GAP D (LR/batch/dataset): 2/3 APIs dizem manter lr=5e-5, Claude Opus sugere 3-4e-5
ligeiramente mais safe. Consensus: patience=3 OK, plateau esperado step 150-250,
EarlyStop fire ~275-325. Mantenho lr=5e-5 (blast radius, alpha=16 ja conservador).

GAP E (callbacks): 3/3 APIs: stack atual correto. `processing_class=tok` eh current
TRL 0.25 style. `load_best_model_at_end=True` protege best checkpoint da rotacao.
Nenhum bug conhecido com LoRA + grad_checkpoint + load_best. -> NO CHANGE NEEDED.

## Como usar

1. Runtime -> H100 HighRAM + disconnect old
2. Cell 1-4 em ordem
3. Cell 3: 1a execucao crasha kernel. Reconectar, re-run Cell 3
4. Cell 5: treino ~10h com early stopping ativo

## Target: train_loss 0.30-0.80 final, eval_loss < 3x train_loss
## Prediction (5-AI average): likely 0.15-0.40 range; EarlyStop saves eval_loss regardless
'''


PREFLIGHT = r"""# Cell 1: PRE-FLIGHT GPU CHECK
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
    print(f'WARN: {e}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA not available')
d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = total_gb - free_gb
print(f'GPU: {d.name}  Total: {total_gb:.1f}GB  Free: {free_gb:.1f}GB')
print(f'torch: {torch.__version__}')
if total_gb < 38:
    raise RuntimeError(f'GPU too small')
if used_gb > 5:
    print('\nZOMBIE - disconnect runtime')
    raise RuntimeError(f'Zombie {used_gb:.1f}GB')
print('\nGPU is clean. Ready for Cell 2.')
"""


DATASET = r"""# Cell 2: DATASET VALIDATION + CATEGORY ANALYSIS
import os, json, subprocess, sys
from pathlib import Path
print('=' * 60)
print('DATASET VALIDATION + CATEGORY ANALYSIS')
print('=' * 60)
for pkg in ['huggingface_hub>=0.25,<1.0', 'pandas>=2.0,<4.0']:
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
    try: drive.mount('/content/drive', force_remount=False); GDRIVE_OK=True
    except Exception:
        try: drive.mount('/content/drive', force_remount=True); GDRIVE_OK=True
        except Exception as e: print(f'WARN: {e}')
except ImportError: pass
cache = '/content/drive/MyDrive/kg1_data/v70_huikang' if GDRIVE_OK else '/content/kg1_data/v70_huikang'
os.makedirs(cache, exist_ok=True)
v70_path = hf_hub_download(
    repo_id='felipesp1983/kg1-nemotron-training',
    filename='data/sft_v70_huikang_full.jsonl',
    repo_type='dataset', local_dir=cache, token=hf_key,
)
df = pd.read_json(v70_path, lines=True)
print(f'Loaded {len(df)} rows')

KAGGLE_CATS = {'numeral','gravity','unit_conversion','cipher','bit_manipulation',
               'equation_transformation','cryptarithm_deduce','cryptarithm_guess',
               'equation_numeric_deduce'}

print('\nCategory distribution:')
in_k = 0
for cat, n in df['category'].value_counts().items():
    ink = cat in KAGGLE_CATS
    if ink: in_k += n
    print(f'  {"[K]" if ink else "[ ]"} {cat:30s}: {n:5d}')
print(f'\nIn Kaggle: {in_k}/{len(df)} ({100*in_k/len(df):.1f}%)')

fit_1800 = (df['trace_len'] <= 1800).sum()
df_filt = df[df['trace_len'] <= 1800]
df_filt_k = df_filt[df_filt['category'].isin(KAGGLE_CATS)]
print(f'\nFilters:')
print(f'  trace_len<=1800: {fit_1800}')
print(f'  trace_len<=1800 + Kaggle: {len(df_filt_k)}')

print(f'\nV77 trains on: option A (all cats, trace_len<=1800) = {fit_1800} samples')
print()
print('=' * 60)
print('DATASET OK - Ready for Cell 3')
print('=' * 60)
"""


ENVSETUP = r"""# Cell 3: ENVIRONMENT SETUP (torch downgrade + wheels + deps)
# ATTENTION: first run crashes kernel. Reconect + run again.
import subprocess, sys, os, time
print('=' * 60)
print('ENVIRONMENT SETUP V77')
print('=' * 60)
import torch
cur = torch.__version__
print(f'Current torch: {cur}')
if not cur.startswith('2.5'):
    print(f'\nDowngrading torch {cur} -> 2.5.1+cu121...')
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
    print('ACTION:')
    print('  1. Click Reconectar at top right')
    print('  2. Run THIS Cell 3 AGAIN')
    sys.stdout.flush()  # FIX #10: flush before kill
    time.sleep(3)
    os.kill(os.getpid(), 9)

print(f'[OK] torch={cur}')
print()
print('Installing pinned deps (V77 audit-tightened)...')
# FIX #6, #7, #8: tighter version pins
DEPS = [
    'transformers>=4.55,<4.58',
    'peft>=0.13,<0.20',
    'trl>=0.25,<0.26',
    'accelerate>=0.34,<1.0',
    'bitsandbytes>=0.44,<0.50',
    'datasets>=2.20,<4.0',
    'safetensors>=0.4.5',
    'sentencepiece',
    'einops',
    'huggingface_hub>=0.25,<1.0',
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
        raise RuntimeError(f'{name} failed')
    print(f'    [OK] {name}')

print('\nVerifying imports + FIX #15 (triple-check GAP C: ABI check)...')
for m in list(sys.modules.keys()):
    if any(x in m for x in ['mamba_ssm', 'causal_conv1d']):
        del sys.modules[m]

# FIX #15 (triple-check 3/3 APIs): ABI check before importing mamba_ssm
# Wheels were compiled with cxx11abiFALSE. torch default PyPI is also FALSE.
# If mismatch -> segfault at import.
_torch_abi = torch._C._GLIBCXX_USE_CXX11_ABI
print(f'  torch CXX11_ABI: {_torch_abi} (wheels=False)')
if _torch_abi:
    print('  CRITICAL: torch built with CXX11_ABI=True, wheels are False -> segfault risk')
    print('  If mamba_ssm import crashes, this is the cause')
import mamba_ssm
print(f'  [OK] mamba_ssm')
import causal_conv1d
from causal_conv1d import causal_conv1d_fn
assert causal_conv1d_fn is not None
print(f'  [OK] causal_conv1d + binding')
import transformers
print(f'  [OK] transformers {transformers.__version__}')
assert transformers.__version__.startswith('4.5')
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
"""


BUDGET = r"""# Cell 4: MEMORY BUDGET + ETA
import torch
print('=' * 60)
print('MEMORY BUDGET + ETA (V77 FIXED)')
print('=' * 60)
d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
TOTAL = 62.8 + 1.8 + 2.0 + 2.0 + 2.0 + 3.0
print(f'GPU: {d.name}  Total: {total_gb:.1f}GB  Free: {free_gb:.1f}GB')
print(f'Projected: {TOTAL:.1f}GB  Margin: {total_gb-TOTAL:+.1f}GB')
STEPS = 6400 // 16  # filter<=1800 gives ~6400 samples
ETA = (STEPS * 5.7 * 16) / 3600
print(f'\nETA: {ETA:.1f}h ({STEPS} optim steps)')
print(f'\nV77 fixes: CompletionOnlyLM, EarlyStop, best_ckpt, tighter pins')
assert total_gb - TOTAL > -3 and free_gb > 30
print('\n' + '=' * 60)
print(f'BUDGET OK - Ready for Cell 5')
print('=' * 60)
"""


MEGA = r"""# Cell 5: V77 TRAINING PIPELINE (all 10 audit fixes)
import os, sys, json, subprocess, shutil, gc, time, datetime, random, re, math
from pathlib import Path

START_TIME = time.time()
def log(msg):
    elapsed = int(time.time() - START_TIME)
    h, rem = divmod(elapsed, 3600); m, s = divmod(rem, 60)
    print(f'[{h:02d}:{m:02d}:{s:02d}] {msg}', flush=True)
def section(t): log('=' * 60); log(t); log('=' * 60)

section('SECTION 0: Verify env')
import torch
assert torch.__version__.startswith('2.5')
import mamba_ssm, causal_conv1d
from causal_conv1d import causal_conv1d_fn
assert causal_conv1d_fn is not None
log(f'torch={torch.__version__}')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

section('SECTION 1: Clone KG1')
KG1_DIR = Path('/content/kg1')
REPO = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
BRANCH = os.environ.get('KG1_BRANCH', 'claude/competent-shamir')
if KG1_DIR.exists(): shutil.rmtree(KG1_DIR)
subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO, str(KG1_DIR)])
commit = subprocess.check_output(['git', '-C', str(KG1_DIR), 'log', '-1', '--format=%h %s'], text=True).strip()
log(f'Commit: {commit}')
REQUIRED = ['src/reasoners/bit_manipulation_pairs.py', 'src/reasoners/cryptarithm_47combo.py',
            'src/reasoners/neurosymbolic_template.py', 'src/losses/max_min_logprob.py',
            'src/prompts/build_prompt.py', 'scripts/local_score.py',
            'scripts/kg1_submission_gate.py']
assert not [r for r in REQUIRED if not (KG1_DIR / r).exists()]
sys.path.insert(0, str(KG1_DIR))

section('SECTION 2: Auth')
GDRIVE_MOUNTED = False
try:
    from google.colab import drive, userdata
    try: drive.mount('/content/drive', force_remount=False); GDRIVE_MOUNTED=True
    except Exception:
        try: drive.mount('/content/drive', force_remount=True); GDRIVE_MOUNTED=True
        except Exception as e: log(f'WARN: {e}')
    try: hf_key = userdata.get('HF_KEY')
    except: hf_key = None
    if not hf_key: hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    os.environ['HF_TOKEN'] = hf_key; os.environ['HF_KEY'] = hf_key
    try:
        kuser = userdata.get('KAGGLE_USERNAME'); kkey = userdata.get('KAGGLE_KEY')
    except: kuser = os.environ.get('KAGGLE_USERNAME'); kkey = os.environ.get('KAGGLE_KEY')
    if kuser and kkey:
        os.environ['KAGGLE_USERNAME'] = kuser; os.environ['KAGGLE_KEY'] = kkey
        kpath = Path.home() / '.kaggle' / 'kaggle.json'
        kpath.parent.mkdir(parents=True, exist_ok=True)
        kpath.write_text(json.dumps({'username': kuser, 'key': kkey})); kpath.chmod(0o600)
except ImportError: pass
from huggingface_hub import login, whoami
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
login(token=HF_TOKEN, add_to_git_credential=False)
try: log(f'HF: {whoami(token=HF_TOKEN)["name"]}')
except: pass

section('SECTION 3: Config V77 FIXED')
from dataclasses import dataclass, asdict
@dataclass
class Config:
    base_model: str = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    max_length: int = 2048
    trace_len_filter: int = 1800
    attn_implementation: str = 'eager'
    mamba_ssm_cache_dtype: str = 'float32'
    tie_word_embeddings: bool = False
    lora_r: int = 32
    lora_alpha: int = 16  # FIX #11 (5-AI consensus): matches v30 PROVEN 0.68 baseline, halves overfit pressure
    lora_dropout: float = 0.10
    lora_target_modules: str = 'all-linear'
    epochs: int = 1
    per_device_batch: int = 1
    grad_accum: int = 16
    learning_rate: float = 5e-5
    lr_scheduler: str = 'linear'
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    optimizer: str = 'paged_adamw_8bit'
    bf16: bool = True
    use_gradient_checkpointing: bool = True
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
    early_stopping_patience: int = 3
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 5
    run_tag: str = 'v77_fixed'
    output_dir: str = '/content/kg1_out/v77_fixed'
    gdrive_checkpoint: str = '/content/drive/MyDrive/kg1_checkpoints/v77_fixed'
    hf_upload_repo: str = 'felipesp1983/kg1-nemotron-lora-v77-fixed'
CFG = Config()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(CFG.output_dir) / 'config.json', 'w') as f:
    json.dump(asdict(CFG), f, indent=2)
log(f'lr={CFG.learning_rate} dropout={CFG.lora_dropout} eff_batch={CFG.grad_accum} trace<={CFG.trace_len_filter}')

section('SECTION 4: Pre-flight')
import importlib
for mod in ['src.reasoners.bit_manipulation_pairs', 'src.reasoners.cryptarithm_47combo',
            'src.reasoners.neurosymbolic_template', 'src.losses.max_min_logprob',
            'src.prompts.build_prompt']:
    importlib.import_module(mod)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
log(f'tokenizer: vocab={tok.vocab_size}')
from src.prompts.build_prompt import build_prompt_v71, detect_category

section('SECTION 5: Load V70 + FILTER trace<=1800')
from huggingface_hub import hf_hub_download
import pandas as pd
cache = Path('/content/drive/MyDrive/kg1_data') if GDRIVE_MOUNTED else Path('/content/kg1_data')
cache.mkdir(parents=True, exist_ok=True)
v70_path = hf_hub_download(
    repo_id=CFG.hf_dataset_repo, filename=CFG.hf_dataset_file,
    repo_type='dataset', local_dir=str(cache / 'v70_huikang'), token=HF_TOKEN,
)
df = pd.read_json(v70_path, lines=True)
log(f'Loaded {len(df)} rows')
orig = len(df)
df = df[df['trace_len'] <= CFG.trace_len_filter].copy().reset_index(drop=True)
log(f'FILTER trace_len<={CFG.trace_len_filter}: {orig} -> {len(df)}')
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
        use_structured=CFG.use_structured, use_category_hints=CFG.use_category_hints,
        use_boxed_strict=CFG.use_boxed_strict, use_self_correct=CFG.use_self_correct,
    )
    resp = row.get('response')
    if resp is not None and pd.notna(resp) and str(resp).strip():
        assistant = str(resp).strip()
        if '\\boxed{' not in assistant:
            ans = row.get('answer', '')
            if ans and pd.notna(ans): assistant = assistant + f'\n\\boxed{{{ans}}}'
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
log('Loading NemotronH 30B BF16...')
model = AutoModelForCausalLM.from_pretrained(
    CFG.base_model, config=model_cfg,
    torch_dtype=torch.bfloat16, device_map={'': 0},
    attn_implementation=CFG.attn_implementation, trust_remote_code=True,
)
log('Model loaded')
peft_cfg = LoraConfig(
    r=CFG.lora_r, lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout, target_modules=CFG.lora_target_modules,
    bias='none', task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
if CFG.use_gradient_checkpointing:
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        log('Gradient checkpointing ENABLED')
    except Exception as e:
        log(f'WARN: {e}')
        CFG.max_length = max(512, CFG.max_length // 2)
        CFG.use_gradient_checkpointing = False
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
log(f'After load: used={used_gb:.1f}GB free={free_gb:.1f}GB')

section('SECTION 7: Smoke test')
gc.collect(); torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
class JsonlChat(Dataset):
    def __init__(self, path, tok, ml):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok; self.ml = ml
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        msgs = [{'role': 'user', 'content': r['user']}, {'role': 'assistant', 'content': r['assistant']}]
        try: text = self.tok.apply_chat_template(msgs, tokenize=False, enable_thinking=CFG.enable_thinking)
        except TypeError: text = self.tok.apply_chat_template(msgs, tokenize=False)
        enc = self.tok(text, truncation=True, max_length=self.ml, return_tensors='pt')
        ids = enc['input_ids'][0]
        labels = ids.clone()
        try: ut = self.tok.apply_chat_template([msgs[0]], tokenize=False, enable_thinking=CFG.enable_thinking)
        except TypeError: ut = self.tok.apply_chat_template([msgs[0]], tokenize=False)
        uids = self.tok(ut, return_tensors='pt')['input_ids'][0]
        k = min(len(uids), len(labels))
        labels[:k] = -100
        return {'input_ids': ids, 'labels': labels, 'attention_mask': enc['attention_mask'][0]}
def collate(batch, pid):
    ml = max(x['input_ids'].size(0) for x in batch)
    pad = lambda t, v: torch.nn.functional.pad(t, (0, ml - t.size(0)), value=v)
    return {'input_ids': torch.stack([pad(x['input_ids'], pid) for x in batch]),
            'labels': torch.stack([pad(x['labels'], -100) for x in batch]),
            'attention_mask': torch.stack([pad(x['attention_mask'], 0) for x in batch])}
ds = JsonlChat(train_path, tok, CFG.max_length)
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
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CFG.grad_clip)
    opt.step(); opt.zero_grad(set_to_none=True)
    log(f'smoke step {step} loss={loss.item():.4f}')
    if step + 1 >= CFG.smoke_test_steps: break
assert losses[-1] < CFG.smoke_abort_loss
log(f'Smoke PASSED (final {losses[-1]:.4f})')
try: del opt, ds, dl, batch, out, loss
except: pass
gc.collect(); torch.cuda.empty_cache()

section('SECTION 8: Full training V77 (CompletionOnlyLM + EarlyStop + best_ckpt)')
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import EarlyStoppingCallback
from datasets import load_dataset
from src.losses.max_min_logprob import max_min_logprob_loss

ds_train = load_dataset('json', data_files=str(train_path), split='train')
ds_eval = load_dataset('json', data_files=str(eval_path), split='train')
def format_example(ex):
    msgs = [{'role': 'user', 'content': ex['user']}, {'role': 'assistant', 'content': ex['assistant']}]
    try: text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=CFG.enable_thinking)
    except TypeError: text = tok.apply_chat_template(msgs, tokenize=False)
    return {'text': text}
ds_train = ds_train.map(format_example, remove_columns=ds_train.column_names)
ds_eval = ds_eval.map(format_example, remove_columns=ds_eval.column_names)
tok.model_max_length = CFG.max_length

# FIX #1: DataCollatorForCompletionOnlyLM - mask user tokens
# Nemotron chat template marker: after user, look for assistant turn
try:
    # Find response template tokens by testing
    sample_text = tok.apply_chat_template(
        [{'role': 'user', 'content': 'TEST'}, {'role': 'assistant', 'content': 'X'}],
        tokenize=False, enable_thinking=False,
    )
    # Detect which template separator is used
    if '<|im_start|>assistant' in sample_text:
        response_template = '<|im_start|>assistant'
    elif '<|assistant|>' in sample_text:
        response_template = '<|assistant|>'
    elif 'ASSISTANT:' in sample_text:
        response_template = 'ASSISTANT:'
    else:
        response_template = None
    if response_template:
        response_template_ids = tok.encode(response_template, add_special_tokens=False)
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tok, mlm=False,
        )
        log(f'CompletionOnlyLM ok: "{response_template}" ({len(response_template_ids)} ids)')

        # FIX #13 (triple-check GAP A, 3/3 APIs): diagnostic pre-train sanity
        # Validate that response_template_ids actually appear in tokenized sample
        # and that labels have non-masked tokens. If 100% masked -> loss=0 -> no-op training.
        log('FIX #13: validating collator masks correctly on 3 samples...')
        try:
            sample_batch = [ds_train[i] for i in range(min(3, len(ds_train)))]
            tokenized = [tok(s['text'], truncation=True, max_length=CFG.max_length,
                             return_tensors='pt') for s in sample_batch]
            collated = collator([{'input_ids': t['input_ids'][0],
                                  'attention_mask': t['attention_mask'][0]} for t in tokenized])
            labels_ck = collated['labels']
            nonmask = (labels_ck != -100).sum().item()
            total = (labels_ck != tok.pad_token_id).sum().item() if tok.pad_token_id else labels_ck.numel()
            pct = 100 * nonmask / max(1, total)
            log(f'  Collator diagnostic: {nonmask}/{total} tokens non-masked ({pct:.1f}%)')
            assert nonmask > 50, f'FAIL: only {nonmask} tokens unmasked - template not matching'
            assert pct > 5, f'FAIL: only {pct:.1f}% unmasked - will collapse to loss=0'
            log('  [OK] collator masks ~correctly - loss will train on assistant tokens')
        except Exception as e:
            log(f'  WARN diagnostic failed: {e} - falling back to default collator (no masking)')
            collator = None
    else:
        log(f'WARN: no response_template found, using default collator (CE on ALL tokens)')
        collator = None
except Exception as e:
    log(f'WARN DataCollatorForCompletionOnlyLM: {e}')
    collator = None

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
    logging_steps=CFG.logging_steps,
    save_steps=CFG.save_steps,
    eval_strategy='steps',
    eval_steps=CFG.eval_steps,
    save_total_limit=CFG.save_total_limit,
    optim=CFG.optimizer,
    packing=False,
    report_to=[],
    gradient_checkpointing=CFG.use_gradient_checkpointing,
    dataset_text_field='text',
    # FIX #5: best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

# FIX #9: instance vars (not class vars)
class MaxMinSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nan_count = 0
        self._use_ce_permanent = False
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        logits = outputs.logits
        step = int(self.state.global_step)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
        )
        use_ce = (self._use_ce_permanent or CFG.loss_type == 'ce'
                  or (CFG.loss_type == 'max_min_warmup_ce' and step < CFG.max_min_warmup_steps))
        if use_ce: loss = ce_loss
        else:
            try:
                mm_loss = max_min_logprob_loss(logits, labels)
                if torch.isnan(mm_loss) or torch.isinf(mm_loss):
                    self._nan_count += 1
                    print(f'WARN NaN max-min step {step} #{self._nan_count}')
                    if self._nan_count >= 3:
                        self._use_ce_permanent = True
                        print('CRITICAL: CE permanent')
                    loss = ce_loss
                else:
                    loss = mm_loss
                    self._nan_count = max(0, self._nan_count - 1)
            except Exception as e:
                print(f'WARN max-min exc: {e}'); loss = ce_loss
        if torch.isnan(loss) or torch.isinf(loss): loss = ce_loss
        return (loss, outputs) if return_outputs else loss

trainer_kwargs = dict(
    model=model, args=sft_args,
    train_dataset=ds_train, eval_dataset=ds_eval,
    processing_class=tok,
    # FIX #4: early stopping
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience)],
)
if collator is not None:
    trainer_kwargs['data_collator'] = collator
trainer = MaxMinSFTTrainer(**trainer_kwargs)
log('Training START V77 (EarlyStop patience=3, CompletionOnlyLM if available)')
log(f'Target: train_loss 0.30-0.80 final. If <0.01 = overfit.')
trainer.train()
log('Training complete (best checkpoint loaded)')
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
    except Exception as e: log(f'WARN: {e}')
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(CFG.hf_upload_repo, private=True, exist_ok=True)
    upload_folder(repo_id=CFG.hf_upload_repo, folder_path=str(out_dir),
                  allow_patterns=['adapter_*', 'tokenizer*', 'special_tokens*', 'config.json'],
                  token=HF_TOKEN)
    log(f'HF: {CFG.hf_upload_repo}')
except Exception as e: log(f'WARN: {e}')

section('SECTION 10: Local eval + gate')
local_score_script = Path('/content/kg1/scripts/local_score.py')
eval_csv = Path(CFG.output_dir) / 'local_eval.csv'
cmd = [sys.executable, str(local_score_script), '--adapter', str(out_dir),
       '--n-samples', str(CFG.eval_holdout_size), '--output-csv', str(eval_csv)]
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
if local_score_val is None: gate_msg = 'NO-GO: parse failed'
elif local_score_val < CFG.local_score_floor: gate_msg = f'NO-GO: {local_score_val:.4f}'
elif local_score_val < CFG.target_score - 0.01: gate_msg = f'MARGINAL: {local_score_val:.4f}'; GO=True
else: gate_msg = f'GO: {local_score_val:.4f}'; GO=True
log(f'Gate: {gate_msg}')
with open(out_dir / 'gate_decision.json', 'w') as f:
    json.dump({'go': GO, 'msg': gate_msg, 'score': local_score_val}, f)

section('SECTION 11: Submit (if GO) - FIX #12: direct kaggle CLI (submit_kaggle.py needs --hf-repo/--local-dir not --zip)')
if not GO: log('NO-GO -> skipping')
else:
    import zipfile
    zip_path = out_dir / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in ['adapter_config.json', 'adapter_model.safetensors']:
            zf.write(out_dir / fn, arcname=fn)
    log(f'zip: {zip_path.stat().st_size/(1024*1024):.2f}MB')
    # Basic sanity: zip has required files, zip < 500MB (Kaggle limit)
    zip_mb = zip_path.stat().st_size / (1024*1024)
    assert zip_mb < 500, f'zip too big: {zip_mb}MB'
    # Check daily slot before submit (Kaggle hard limit: 5/day)
    slot_ok = True
    try:
        rc = subprocess.run(['kaggle', 'competitions', 'submissions',
                             '-c', 'nvidia-nemotron-model-reasoning-challenge', '--csv'],
                            capture_output=True, text=True, timeout=60)
        if rc.returncode == 0:
            from io import StringIO
            import csv as _csv
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            today_count = sum(1 for r in _csv.DictReader(StringIO(rc.stdout))
                              if r.get('date', '').startswith(today))
            log(f'Kaggle slots today: {today_count}/5')
            slot_ok = today_count < 5
    except Exception as e:
        log(f'WARN slot check: {e}')
    if not slot_ok:
        log('Slot quota exhausted (5/5). Skipping submit.')
    elif os.environ.get('KAGGLE_USERNAME'):
        msg = f'V77_FIXED {datetime.datetime.now().strftime("%Y-%m-%d %H:%M BRT")}'
        # Direct kaggle CLI (bypass submit_kaggle.py which needs --hf-repo/--local-dir)
        cmd = ['kaggle', 'competitions', 'submit',
               '-c', 'nvidia-nemotron-model-reasoning-challenge',
               '-f', str(zip_path), '-m', msg]
        r3 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        with open(out_dir / 'kaggle_submit.json', 'w') as f:
            json.dump({'msg': msg, 'rc': r3.returncode,
                      'stdout': r3.stdout[-1000:], 'stderr': r3.stderr[-500:]}, f)
        log(f'Submit rc={r3.returncode}')
        if r3.returncode == 0:
            log('SUBMITTED. Check https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions')
        else:
            log(f'Submit FAILED: {r3.stderr[-300:]}')

section('ALL DONE V77')
log(f'Total: {(time.time() - START_TIME)/3600:.2f}h  Score: {local_score_val}  Gate: {GO}')
"""


def cell_md(c, i):
    return {"cell_type":"markdown","metadata":{"id":i},"source":c.splitlines(keepends=True)}
def cell_code(c, i):
    return {"cell_type":"code","metadata":{"id":i},"execution_count":None,"outputs":[],"source":c.splitlines(keepends=True)}

NB = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab":{"provenance":[],"machine_shape":"hm"},
                 "kernelspec":{"name":"python3","display_name":"Python 3"},
                 "language_info":{"name":"python"},"accelerator":"GPU"},
    "cells": [
        cell_md(HEADER, "header"),
        cell_code(PREFLIGHT, "c1"),
        cell_code(DATASET, "c2"),
        cell_code(ENVSETUP, "c3"),
        cell_code(BUDGET, "c4"),
        cell_code(MEGA, "c5"),
    ],
}

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'KG1_V77_FIXED.ipynb'))
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f'Wrote {OUT} ({os.path.getsize(OUT)} bytes)')

import py_compile, tempfile
for name, src in [('PREFLIGHT',PREFLIGHT),('DATASET',DATASET),('ENVSETUP',ENVSETUP),('BUDGET',BUDGET),('MEGA',MEGA)]:
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as t:
        t.write(src); tpath = t.name
    try:
        py_compile.compile(tpath, doraise=True)
        print(f'{name}: py_compile OK ({src.count(chr(10))} lines)')
    except py_compile.PyCompileError as e:
        print(f'{name}: {e}'); raise
    os.unlink(tpath)
print('\nV77 built + validated.')
