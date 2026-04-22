"""Build KG1_V80_DGXCHEN_EXACT.ipynb — Colab H100 EXACT replica of dgxchen v7 (0.85 LB, 22/04/2026).

SOURCE VERIFIED: kernel https://www.kaggle.com/code/dgxchen/training-with-unsloth-to-achieve-0-85-lb
  - Ran successfully 22/04/2026 01:57-08:50 UTC -> 0.85 LB
  - Fetched via Kaggle MCP get_notebook_info = fresh verified recipe

V79 -> V80: REVERT 7 divergences identified from dgxchen v7 real source.
  V79 applied 11-API "review fixes" that were actually divergences from the proven recipe.
  When APIs don't have the real source, their "fixes" introduce risk.

| # | Param                               | V79 (wrong)           | V80 = dgxchen v7 REAL              |
|---|-------------------------------------|-----------------------|-------------------------------------|
| 1 | Dataset file                        | less_cot.csv          | problem_ids_matched.csv             |
| 2 | attn_implementation                 | 'sdpa'                | 'eager'                             |
| 3 | target_modules (count)              | 9 (WITH lm_head)      | 8 (NO lm_head)                      |
| 4 | num_train_epochs                    | 2                     | 1                                   |
| 5 | max_grad_norm                       | 1.0                   | 1e9 (effectively disabled)          |
| 6 | gradient_checkpointing in SFTConfig | REMOVED               | True + {"use_reentrant": False}     |
| 7 | formatting_func in trainer          | REMOVED               | present + custom conversation wrap  |

KEEP from V79 (no regression, not divergence):
  - num_proc=1 (pickling safety, dgxchen doesn't care about num_proc)
  - HealthGateCallback (Felipe wants gates)
  - _grad/_lr pre-computed f-string vars (fixes SyntaxError)
  - 5-step logging
  - Stratified batching (dgxchen uses this)
"""
import json
import os

HEADER = r"""# KG1 V80 — dgxchen v7 EXACT Replica (0.85 LB verified 22/04/2026)

## Source (verified via Kaggle MCP)
Kernel: `dgxchen/training-with-unsloth-to-achieve-0-85-lb`
Ran: 22/04/2026 01:57-08:50 UTC -> **0.85 LB** on Kaggle leaderboard

## V79 -> V80 changes: REVERT 7 divergences

| # | Param                               | V79 (wrong)      | V80 = dgxchen v7 REAL       |
|---|-------------------------------------|------------------|------------------------------|
| 1 | Dataset                             | less_cot.csv     | problem_ids_matched.csv      |
| 2 | attn_implementation                 | sdpa             | eager                        |
| 3 | LoRA targets (count)                | 9 (lm_head)      | 8 (NO lm_head)               |
| 4 | num_train_epochs                    | 2                | 1                            |
| 5 | max_grad_norm                       | 1.0              | 1e9                          |
| 6 | gradient_checkpointing (SFTConfig)  | removed          | True + use_reentrant=False   |
| 7 | formatting_func (trainer)           | removed          | present + conversation wrap  |

## Recipe dgxchen v7 EXACT (all 8 verified params)
- Dataset: `dgxchen/nemotron-cot-tong/problem_ids_matched.csv` (48MB)
- LoRA r=32 alpha=32 dropout=0.0 on [q/k/v/o_proj, in/out/up/down_proj] (8 items, NO lm_head)
- 1 epoch, eff_batch=32 (micro=1, grad_accum=32), lr=2e-4 linear, warmup=0
- max_length=8192, adam_beta2=0.95, weight_decay=0, max_grad_norm=1e9
- Unsloth FastLanguageModel, attn_implementation='eager'
- gradient_checkpointing=True + gradient_checkpointing_kwargs={use_reentrant: False}
- enable_thinking=True via chat template
- formatting_func: conversation wrap + try/except TypeError for enable_thinking fallback

## Why V80 is correct when V79 wasn't
V79 applied 4 "API-suggested fixes" that were actually divergences from the proven 0.85 recipe.
APIs can only review CODE, not cross-reference with a working kernel they haven't seen.
V80 reverts those fixes after fetching the actual dgxchen v7 source via Kaggle MCP.

## Target
- Input: H100 80GB Colab Pro+
- Budget: ~$30-50 (1.5-2.5h on H100 at 1 epoch)
- Output: `submission.zip` -> Kaggle + adapter HF

## Credenciais (Colab Secrets)
- `HF_KEY` — HuggingFace Pro token
- `KAGGLE_USERNAME` = felipe1983
- `KAGGLE_KEY` = (do kaggle.json)
"""

CELL_1_CHECK = r"""# CELL 1: GPU + Secrets check (identical V79 - no divergence)
import subprocess, torch, os, gc

print('=' * 60)
print('PRE-FLIGHT CHECK V80 (dgxchen v7 EXACT replica)')
print('=' * 60)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                        '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    print('GPU:', r.stdout.strip())
except Exception as e:
    print('WARN nvidia-smi:', e)

assert torch.cuda.is_available(), 'CUDA not available'
d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f'VRAM: total={total_gb:.1f}GB free={free_gb:.1f}GB')
assert total_gb >= 75, f'need H100 80GB, got {total_gb:.1f}GB'
print(f'torch={torch.__version__}')

try:
    from google.colab import userdata
    hf_key = userdata.get('HF_KEY')
    kaggle_user = userdata.get('KAGGLE_USERNAME')
    kaggle_key = userdata.get('KAGGLE_KEY')
except Exception:
    hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    kaggle_user = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

assert hf_key, 'HF_KEY missing (add in Colab Secrets)'
assert kaggle_user and kaggle_key, 'KAGGLE_USERNAME/KAGGLE_KEY missing'

os.environ['HF_TOKEN'] = hf_key
os.environ['HF_KEY'] = hf_key
os.environ['KAGGLE_USERNAME'] = kaggle_user
os.environ['KAGGLE_KEY'] = kaggle_key

from pathlib import Path
kpath = Path.home() / '.kaggle' / 'kaggle.json'
kpath.parent.mkdir(parents=True, exist_ok=True)
import json
kpath.write_text(json.dumps({'username': kaggle_user, 'key': kaggle_key}))
kpath.chmod(0o600)

print(f'HF user token: ...{hf_key[-8:]}')
print(f'Kaggle user: {kaggle_user}')
print('READY for Cell 2')
"""

CELL_2_INSTALL = r"""# CELL 2: Install Unsloth + deps (identical V79)
import subprocess, sys, os

print('=' * 60)
print('INSTALLING Unsloth + deps (H100 compatible)')
print('=' * 60)

CMDS = [
    [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
     'unsloth', 'unsloth_zoo', 'trl', 'peft', 'transformers',
     'accelerate', 'bitsandbytes', 'datasets', 'xformers', 'huggingface_hub', 'kagglehub'],
]
for cmd in CMDS:
    print(f'> {" ".join(cmd[-10:])}')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print('stderr:', r.stderr[-500:])
        raise RuntimeError(f'install failed: {cmd}')

print()
print('Verifying imports...')
import torch
print(f'  torch={torch.__version__}  cuda={torch.version.cuda}')
import transformers; print(f'  transformers={transformers.__version__}')
import peft; print(f'  peft={peft.__version__}')
import trl; print(f'  trl={trl.__version__}')
import accelerate; print(f'  accelerate={accelerate.__version__}')
import datasets; print(f'  datasets={datasets.__version__}')
from unsloth import FastLanguageModel
print('  unsloth.FastLanguageModel imported')

import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print('\nINSTALL OK -> proceed to Cell 3')
"""

# FIX #1: Dataset file -> problem_ids_matched.csv (V79 used less_cot.csv)
CELL_3_DATA = r"""# CELL 3: Download dgxchen dataset + Nemotron base model
# FIX #1 V80: use problem_ids_matched.csv (dgxchen v7 REAL), NOT less_cot.csv (V79 divergence)
import os, subprocess, sys
from pathlib import Path

print('=' * 60)
print('DOWNLOAD dgxchen/nemotron-cot-tong + Nemotron base')
print('V80 FIX #1: problem_ids_matched.csv (48MB) - dgxchen v7 REAL dataset')
print('=' * 60)

DATA_DIR = Path('/content/kg1_data')
DATA_DIR.mkdir(exist_ok=True)

print('Downloading dgxchen/nemotron-cot-tong (~90MB total)...')
r = subprocess.run(
    ['kaggle', 'datasets', 'download', '-d', 'dgxchen/nemotron-cot-tong',
     '-p', str(DATA_DIR), '--unzip'],
    capture_output=True, text=True, timeout=300,
)
print(r.stdout)
if r.returncode != 0:
    print('stderr:', r.stderr[-500:])
    raise RuntimeError('Kaggle dataset download failed')

files = list(DATA_DIR.rglob('*.csv'))
print(f'Files: {[f.name for f in files]}')

# V80 uses problem_ids_matched.csv (dgxchen v7 REAL), NOT less_cot.csv
assert any(f.name == 'problem_ids_matched.csv' for f in files), 'problem_ids_matched.csv not found'
target_csv = next(f for f in files if f.name == 'problem_ids_matched.csv')
print(f'V80 target path: {target_csv}  size={target_csv.stat().st_size/(1024**2):.1f}MB')

import pandas as pd
df = pd.read_csv(target_csv)
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
if 'type' in df.columns:
    print(f'Type distribution:')
    print(df['type'].value_counts().head(20))

import kagglehub
print('\nDownloading base model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (~60GB)...')
MODEL_PATH = kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')
print(f'Model path: {MODEL_PATH}')

import json
with open('/content/kg1_state.json', 'w') as f:
    json.dump({'model_path': str(MODEL_PATH), 'target_csv': str(target_csv)}, f)

print('\nDATA READY -> proceed to Cell 4')
"""

# FIX #2: attn_implementation -> 'eager' (V79 used 'sdpa')
# FIX #3: target_modules = 8 items NO lm_head (V79 had 9)
CELL_4_MODEL_LORA = r"""# CELL 4: Load model + LoRA (dgxchen v7 EXACT)
# FIX #2 V80: attn_implementation='eager' (V79 used 'sdpa')
# FIX #3 V80: target_modules = 8 items, REMOVE lm_head (V79 had 9)
import json, torch
from pathlib import Path

with open('/content/kg1_state.json') as f:
    state = json.load(f)
MODEL_PATH = state['model_path']

print('=' * 60)
print('LOAD MODEL + LoRA V80 (dgxchen v7 r=32 alpha=32 all-linear NO lm_head)')
print('=' * 60)

from unsloth import FastLanguageModel

MAX_SEQ_LEN = 8192
print(f'Loading {MODEL_PATH} via Unsloth (attn=eager)...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=False,
    attn_implementation='eager',  # FIX #2: V79 used 'sdpa', dgxchen v7 uses 'eager'
    dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('Base model loaded.')

# FIX #3 V80: dgxchen v7 target_modules = 8 items (NO lm_head)
# V79 had 9 items WITH lm_head - divergence.
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
target_modules = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'in_proj', 'out_proj', 'up_proj', 'down_proj',
    # lm_head REMOVED (V79 divergence)
]

print(f'Applying LoRA r={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}')
print(f'Target modules ({len(target_modules)} items, NO lm_head): {target_modules}')
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
model.print_trainable_parameters()

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
print(f'\nAfter LoRA: used={used_gb:.1f}GB free={free_gb:.1f}GB')

import builtins
builtins._v80_model = model
builtins._v80_tokenizer = tokenizer

print('\nLORA APPLIED -> proceed to Cell 5')
"""

# FIX #4: num_train_epochs=1 (V79 had 2)
# FIX #5: max_grad_norm=1e9 (V79 had 1.0)
# FIX #6: gradient_checkpointing=True + kwargs in SFTConfig (V79 removed)
# FIX #7: formatting_func in trainer with conversation wrap (V79 removed)
CELL_5_TRAIN = r"""# CELL 5: Build dataset + Train (dgxchen v7 EXACT + 7 FIXES from V79)
# FIX #4 V80: num_train_epochs=1 (V79=2)
# FIX #5 V80: max_grad_norm=1e9 (V79=1.0)
# FIX #6 V80: gradient_checkpointing=True + use_reentrant=False (V79 removed)
# FIX #7 V80: formatting_func in trainer with conversation wrap (V79 removed)
import builtins, pandas as pd, random, re, math, gc, time, json, torch
from collections import defaultdict, deque
from pathlib import Path
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Sampler
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerControl, TrainerState

print('=' * 60)
print('BUILD DATASET + TRAINING V80 (1 epoch, eff_batch=32, lr=2e-4, grad_norm=1e9)')
print('=' * 60)

model = builtins._v80_model
tokenizer = builtins._v80_tokenizer

with open('/content/kg1_state.json') as f:
    state = json.load(f)
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
print(f'SFT records: {len(records)}')
print(f'Type distribution:')
_tc = pd.Series(record_types).value_counts()
for t, n in _tc.items():
    print(f'  {t}: {n}')

dataset = HFDataset.from_list(records)


# FIX #7 V80: formatting_prompts_func with conversation wrap (dgxchen v7 EXACT)
# V79 removed this and pre-formatted via dataset.map() -> divergence.
# dgxchen v7 passes this func to trainer; trainer handles formatting during tokenization.
def formatting_prompts_func(example):
    messages = example['messages']
    # Conversation wrap: handle both flat dict list AND nested list-of-lists (batched)
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
            # Some tokenizer configs don't support enable_thinking kwarg
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False,
            )
        texts.append(text)
    return texts


# V80 does NOT pre-format dataset via dataset.map() — trainer handles it via formatting_func.
# dgxchen v7 passes raw 'messages' dataset + formatting_func to trainer.

# SFTConfig — dgxchen v7 EXACT + 5-step logging + health gates
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                   # FIX #4: V79=2, dgxchen v7=1
    per_device_train_batch_size=1,        # dgxchen
    gradient_accumulation_steps=32,       # dgxchen (eff_batch=32)
    learning_rate=2e-4,                   # dgxchen VERIFIED 0.85
    lr_scheduler_type='linear',           # dgxchen
    warmup_steps=0,                       # dgxchen
    max_length=8192,                      # dgxchen
    adam_beta1=0.9,
    adam_beta2=0.95,                      # dgxchen
    adam_epsilon=1e-8,
    weight_decay=0.0,                     # dgxchen
    max_grad_norm=1e9,                    # FIX #5: V79=1.0, dgxchen v7=1e9 (disabled)
    logging_steps=5,                      # Felipe: log every 5 steps
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    # FIX #6: V79 removed gradient_checkpointing from SFTConfig -> dgxchen v7 KEEPS it
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    dataloader_num_workers=0,  # V80 fix: num_workers>=1 fails pickle CudaDeviceProperties
                                # in formatting_func closure on HF Jobs (and some Colab envs)
    remove_unused_columns=False,
    seed=SEED,
    report_to='none',
    packing=False,
)


# HealthGateCallback — 5-step logging + automatic abort gates (KEPT from V79)
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
        print('V80 HEALTH GATES ATIVOS - log every 5 steps + abort on NaN/explosion')
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

        # GATE 2: Loss explosion (note: V80 has max_grad_norm=1e9 so more important)
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

        # GATE 4: High grad_norm sustained
        # NOTE: V80 max_grad_norm=1e9 means grads are NOT clipped. Monitor carefully.
        if grad_norm is not None:
            self.grad_history.append(float(grad_norm))
            if float(grad_norm) > 50.0:
                self.high_grad_count += 1
                if self.high_grad_count >= 3:
                    print(f'!!! WARN grad_norm >50 3x consecutive (last={grad_norm:.2f})')
                    print(f'    V80 max_grad_norm=1e9 means NO clipping - watching for divergence')
            else:
                self.high_grad_count = 0

        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            total = torch.cuda.mem_get_info()[1] / 1024**3
            used = total - free
            peak = torch.cuda.max_memory_allocated() / 1024**3
            if free < 2.0:
                print(f'!!! WARN VRAM free={free:.1f}GB - OOM risk')
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

        # KEPT from V79 FIX #1: pre-compute safe f-string values (invalid syntax fix)
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


# Stratified batching by type (dgxchen v7 has this)
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


eff_batch_size = max(1, training_args.per_device_train_batch_size
                     * training_args.gradient_accumulation_steps)
stratified_order = build_stratified_index_order(record_types, eff_batch_size, SEED)
print(f'\nEff batch size: {eff_batch_size}')
print(f'Total batches: {math.ceil(len(record_types)/eff_batch_size)}')

health_cb = HealthGateCallback()

# FIX #7 V80: Pass formatting_func to trainer (dgxchen v7 EXACT)
# V79 removed formatting_func and used dataset_text_field='text' instead -> divergence.
# dgxchen v7 passes raw dataset with 'messages' + formatting_func.
trainer = StratifiedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,  # FIX #7: V79 removed
    stratified_order=stratified_order,
    callbacks=[health_cb],
)

print('\nStarting V80 SFT (dgxchen v7 EXACT replica)...')
print(f'Epochs=1, eff_batch=32, lr=2e-4, max_grad_norm=1e9')
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

CELL_6_SUBMIT = r"""# CELL 6: Package submission.zip + Kaggle submit (V80)
import os, json, shutil, zipfile, subprocess, sys, datetime, builtins
from pathlib import Path

ADAPTER_DIR = getattr(builtins, '_v80_adapter_dir', '/content/kg1_adapter_v80')
BASE_MODEL_NAME = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
OUTPUT_DIR = '/content/kg1_out'
Path(OUTPUT_DIR).mkdir(exist_ok=True)
SUBMISSION_DIR = f'{OUTPUT_DIR}/submission_adapter_v80'
Path(SUBMISSION_DIR).mkdir(exist_ok=True)

print('=' * 60)
print('V80 PACKAGE + KAGGLE SUBMIT (dgxchen v7 exact replica)')
print('=' * 60)

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

# Verify target_modules has 8 items (NO lm_head) - V80 fix #3
tm = cfg.get('target_modules', [])
if isinstance(tm, list):
    if 'lm_head' in tm:
        print(f'!!! WARN: adapter_config.json still has lm_head in target_modules!')
        print(f'    Expected 8 items, got {len(tm)} with lm_head')
    else:
        print(f'V80 adapter OK: {len(tm)} target_modules (NO lm_head): {tm}')

# Build zip
zip_path = f'{OUTPUT_DIR}/submission_v80.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in required:
        zf.write(Path(SUBMISSION_DIR) / fn, arcname=fn)
zip_mb = os.path.getsize(zip_path) / (1024**2)
print(f'\nsubmission_v80.zip: {zip_mb:.1f} MB')
assert zip_mb < 500, f'zip too big: {zip_mb}MB'

# Backup to GDrive
try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive', force_remount=False)
    bkp_dir = Path('/content/drive/MyDrive/kg1_v80')
    bkp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(zip_path, bkp_dir / f'submission_{ts}.zip')
    shutil.copytree(ADAPTER_DIR, bkp_dir / f'adapter_{ts}', dirs_exist_ok=True)
    print(f'Backup: {bkp_dir}')
except Exception as e:
    print(f'GDrive backup skipped: {e}')

# HF upload
try:
    from huggingface_hub import HfApi, upload_folder
    HF_TOKEN = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    REPO = 'felipesp1983/kg1-nemotron-lora-v80-dgxchen-exact'
    api = HfApi(token=HF_TOKEN)
    api.create_repo(REPO, private=True, exist_ok=True)
    upload_folder(
        repo_id=REPO, folder_path=SUBMISSION_DIR,
        allow_patterns=['adapter_*'],
        token=HF_TOKEN,
    )
    print(f'\nHF upload OK: {REPO}')
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
            print('Slot exhausted - will NOT submit. Manual submit later:')
            print(f'  kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge \\')
            print(f'    -f {zip_path} -m "V80 dgxchen v7 EXACT"')
        else:
            msg = f'V80 dgxchen v7 EXACT replica {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
            r = subprocess.run(
                ['kaggle', 'competitions', 'submit',
                 '-c', 'nvidia-nemotron-model-reasoning-challenge',
                 '-f', zip_path, '-m', msg],
                capture_output=True, text=True, timeout=600,
            )
            print(f'Submit rc={r.returncode}')
            print(r.stdout[-400:])
            if r.returncode == 0:
                print('\nSUBMITTED! Check score:')
                print('https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions')
except Exception as e:
    print(f'Submit attempt failed: {e}')
    print(f'\nManual submit later:')
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
        cell_code(CELL_1_CHECK, 'c1'),
        cell_code(CELL_2_INSTALL, 'c2'),
        cell_code(CELL_3_DATA, 'c3'),
        cell_code(CELL_4_MODEL_LORA, 'c4'),
        cell_code(CELL_5_TRAIN, 'c5'),
        cell_code(CELL_6_SUBMIT, 'c6'),
    ],
}


OUT = 'notebooks/KG1_V80_DGXCHEN_EXACT.ipynb'
os.makedirs('notebooks', exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1)

size = os.path.getsize(OUT)
print(f'Wrote {OUT} ({size} bytes)')

# py_compile each cell
import py_compile, tempfile
cells = [
    ('Cell 1 CHECK', CELL_1_CHECK),
    ('Cell 2 INSTALL', CELL_2_INSTALL),
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
    print('\nV80 notebook READY. All 7 divergences REVERTED to dgxchen v7 exact.')
    print('Dataset: problem_ids_matched.csv (V79: less_cot.csv)')
    print('Attn: eager (V79: sdpa)')
    print('LoRA targets: 8 items NO lm_head (V79: 9 with lm_head)')
    print('Epochs: 1 (V79: 2)')
    print('max_grad_norm: 1e9 (V79: 1.0)')
    print('gradient_checkpointing: True in SFTConfig + use_reentrant=False (V79: removed)')
    print('formatting_func: passed to trainer with conversation wrap (V79: removed)')
else:
    print('\nV80 FAILED py_compile - review errors above')
