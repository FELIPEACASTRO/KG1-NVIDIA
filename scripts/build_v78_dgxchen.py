"""Build KG1_V78_DGXCHEN_REPLICA.ipynb — Colab H100 replica of dgxchen 0.85 LB recipe.

CONSENSUS: 26 APIs (17 earlier + 9 surgical) converged on:
  - Replicate dgxchen recipe EXACTLY (S1=G, 55%)
  - Kaggle prompt suffix MUST match metric (S3=C, 78%)
  - Distill weak-cat later (Stage 2) — not this notebook

FIX from APIs (S5=B, 44%): max_grad_norm=1e9 (dgxchen default) identified as
risk of exploding gradients on outlier samples. Change to max_grad_norm=1.0
(standard safe clipping, no expected impact on 0.85 result).
"""
import json
import os

HEADER = r"""# KG1 V78 — dgxchen 0.85 LB Replica (Colab H100 Pro+)

## Fonte
Recipe replicada do kernel público **dgxchen/training-with-unsloth-to-achieve-0-85-lb** (185 votos).
Autor: dz. Baseline verificado: **0.85 LB** em RTX Pro 6000 (6h47min).

Consensus de 26 APIs (Claude Opus/Sonnet/Haiku, GPT-5.4/5.3-codex/mini, DeepSeek R1/Chat,
Gemini 2.5, Grok 4.20 reasoning, Cohere Command-A, gpt-oss-120b, Qwen3-Next, GLM-4.6,
Llama-3.3-70B, Nemotron-70B) convergiu em:

- Replicar recipe EXATO (sem inventar) — 55% S1=G, 89% S7=B
- Prompt suffix IGUAL ao metric oficial — 78% S3=C
- Distillation como Stage 2 (este notebook é Stage 1) — 89% S7=B

## Fix aplicado (vs dgxchen original)
- `max_grad_norm: 1e9 → 1.0` (APIs identificaram risco de explosão)

## Target
- Input: H100 High-RAM Colab Pro+
- Budget: ~$50-75 em créditos (3-5h esperado)
- Output: `submission.zip` pronto para Kaggle + adapter HF

## Credenciais (Colab Secrets)
- `HF_KEY` — HuggingFace Pro token
- `KAGGLE_USERNAME` = felipe1983
- `KAGGLE_KEY` = (do kaggle.json)
"""

CELL_1_CHECK = r"""# CELL 1: GPU + Secrets check
import subprocess, torch, os, gc

print('=' * 60)
print('PRE-FLIGHT CHECK')
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

# Secrets via Colab userdata
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

# write kaggle.json
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

CELL_2_INSTALL = r"""# CELL 2: Install Unsloth + deps (Colab H100)
import subprocess, sys, os

print('=' * 60)
print('INSTALLING Unsloth + deps (H100 compatible)')
print('=' * 60)

# Unsloth install — works on Colab with A100/H100
# Reference: https://docs.unsloth.ai/get-started/installation
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

# Verify
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

# Clear cache
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print('\nINSTALL OK — proceed to Cell 3')
"""

CELL_3_DATA = r"""# CELL 3: Download dgxchen dataset + Nemotron base model
import os, subprocess, sys
from pathlib import Path

print('=' * 60)
print('DOWNLOAD dgxchen/nemotron-cot-tong + Nemotron base')
print('=' * 60)

# Kaggle dataset download
DATA_DIR = Path('/content/kg1_data')
DATA_DIR.mkdir(exist_ok=True)

print('Downloading dgxchen/nemotron-cot-tong (~88MB)...')
r = subprocess.run(
    ['kaggle', 'datasets', 'download', '-d', 'dgxchen/nemotron-cot-tong',
     '-p', str(DATA_DIR), '--unzip'],
    capture_output=True, text=True, timeout=300,
)
print(r.stdout)
if r.returncode != 0:
    print('stderr:', r.stderr[-500:])
    raise RuntimeError('Kaggle dataset download failed')

# Verify
files = list(DATA_DIR.rglob('*.csv'))
print(f'Files: {[f.name for f in files]}')
assert any(f.name == 'less_cot.csv' for f in files), 'less_cot.csv not found'

less_cot = next(f for f in files if f.name == 'less_cot.csv')
print(f'less_cot path: {less_cot}  size={less_cot.stat().st_size/(1024**2):.1f}MB')

# Quick peek
import pandas as pd
df = pd.read_csv(less_cot)
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Sample type distribution:')
print(df['type'].value_counts().head(20) if 'type' in df.columns else 'no type col')

# Base model download via kagglehub (same source as dgxchen)
import kagglehub
print('\nDownloading base model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (~60GB)...')
MODEL_PATH = kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')
print(f'Model path: {MODEL_PATH}')

# Store globally for next cells
import json
with open('/content/kg1_state.json', 'w') as f:
    json.dump({'model_path': str(MODEL_PATH), 'less_cot': str(less_cot)}, f)

print('\nDATA READY — proceed to Cell 4')
"""

CELL_4_MODEL_LORA = r"""# CELL 4: Load model + apply LoRA (dgxchen EXACT config)
import json, torch
from pathlib import Path

with open('/content/kg1_state.json') as f:
    state = json.load(f)
MODEL_PATH = state['model_path']

print('=' * 60)
print('LOAD MODEL + LoRA (dgxchen r=32 alpha=32 all-linear + lm_head)')
print('=' * 60)

from unsloth import FastLanguageModel

MAX_SEQ_LEN = 8192
print(f'Loading {MODEL_PATH} via Unsloth...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=False,
    attn_implementation='sdpa',
    dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('Base model loaded.')

# dgxchen EXACT LoRA config
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
target_modules = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'in_proj', 'out_proj', 'up_proj', 'down_proj',
    'lm_head',
]

print(f'Applying LoRA r={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}')
print(f'Target modules: {target_modules}')
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

# Save for next cell (Python objects don't persist across restarts
# but they do across cells in same session)
import builtins
builtins._v78_model = model
builtins._v78_tokenizer = tokenizer

print('\nLORA APPLIED — proceed to Cell 5')
"""

CELL_5_TRAIN = r"""# CELL 5: Build dataset + Train (dgxchen EXACT + HEALTH GATES + FIX max_grad_norm=1.0)
import builtins, pandas as pd, random, re, math, gc, time, json, torch
from collections import defaultdict, deque
from pathlib import Path
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Sampler
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerControl, TrainerState

print('=' * 60)
print('BUILD DATASET + TRAINING (2 epochs, eff_batch=32, lr=2e-4)')
print('=' * 60)

model = builtins._v78_model
tokenizer = builtins._v78_tokenizer

with open('/content/kg1_state.json') as f:
    state = json.load(f)
less_cot = state['less_cot']

SEED = 42
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
OUTPUT_DIR = '/content/kg1_out/sft_v78'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load + clean
df = pd.read_csv(less_cot)
print(f'Loaded: {len(df)} rows')
train_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

records, record_types = [], []
for _, row in train_df.iterrows():
    prompt = str(row['prompt'])
    answer = str(row['answer'])
    cot = str(row.get('generated_cot', ''))
    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        continue
    # Remove existing \boxed from CoT before appending canonical answer
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


def formatting_prompts_func(example):
    texts = []
    for conv in example['messages']:
        try:
            text = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False, enable_thinking=True,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {'text': texts}


print('\nApplying chat template...')
dataset = dataset.map(
    formatting_prompts_func, batched=True, num_proc=4, desc='chat_template'
)

# SFTConfig — dgxchen EXACT + max_grad_norm fix + logging 5 steps
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,                   # dgxchen
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
    max_grad_norm=1.0,                    # FIX: dgxchen had 1e9 (disabled). APIs flagged risk.
    logging_steps=5,                      # Felipe: log a cada 5 steps (~6min)
    logging_first_step=True,              # log step 0 tambem
    save_strategy='steps',                # checkpoint preventivo
    save_steps=50,                        # a cada 50 optim steps (~15-20% do run)
    save_total_limit=3,                   # manter apenas 3 checkpoints
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    dataloader_num_workers=2,
    remove_unused_columns=False,
    seed=SEED,
    report_to='none',
    packing=False,
)


# ═══════════════════════════════════════════════════════════════
# HEALTH CALLBACK — Felipe: log a cada 5 steps + gates automaticos
# ═══════════════════════════════════════════════════════════════
class HealthGateCallback(TrainerCallback):
    # Monitor de saude + gates automaticos.
    # Log cada 5 steps: loss + avg10, grad_norm + avg5, lr, VRAM used/free/peak,
    # tokens/sec, epoch progress, ETA, warnings.
    # Gates automaticos:
    #   1. NaN/Inf loss -> ABORT imediato
    #   2. Loss > 30 em step > 10 -> ABORT (exploded)
    #   3. Loss < 0.01 em step < 50 -> WARNING overfit severo
    #   4. grad_norm > 10 consecutivo 3x -> WARNING
    #   5. VRAM free menor que 2GB -> WARNING (OOM proximo)
    #   6. Loss sem melhora 100+ steps -> WARNING plateau

    def __init__(self):
        self.loss_history = deque(maxlen=10)
        self.grad_history = deque(maxlen=5)
        self.last_loss = None
        self.min_loss = float('inf')
        self.steps_since_improve = 0
        self.high_grad_count = 0
        self.train_start = None
        self.tokens_processed = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        print('=' * 70)
        print('HEALTH GATES ATIVOS — log cada 5 steps + abort automatico em NaN/explosion')
        print('=' * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        loss = logs.get('loss')
        grad_norm = logs.get('grad_norm')
        lr = logs.get('learning_rate')
        epoch = logs.get('epoch', 0.0)

        # === GATE 1: NaN/Inf ===
        if loss is not None and isinstance(loss, float):
            if math.isnan(loss) or math.isinf(loss):
                print(f'!!! CRITICAL NaN/Inf detected at step {step} — ABORTING')
                control.should_training_stop = True
                return

        # === GATE 2: Loss explosion ===
        if loss is not None and isinstance(loss, (int, float)):
            if loss > 30.0 and step > 10:
                print(f'!!! CRITICAL loss explosion {loss:.3f} at step {step} — ABORTING')
                control.should_training_stop = True
                return

        # === GATE 3: Overfit warning ===
        if loss is not None and isinstance(loss, (int, float)):
            if loss < 0.01 and step < 50:
                print(f'!!! WARN severe overfit risk: loss={loss:.4f} at step {step} (too fast)')

        # Update history
        if loss is not None:
            self.loss_history.append(float(loss))
            if float(loss) < self.min_loss:
                self.min_loss = float(loss)
                self.steps_since_improve = 0
            else:
                self.steps_since_improve += 5  # log_steps=5

        # === GATE 4: High grad_norm sustained ===
        if grad_norm is not None:
            self.grad_history.append(float(grad_norm))
            if float(grad_norm) > 10.0:
                self.high_grad_count += 1
                if self.high_grad_count >= 3:
                    print(f'!!! WARN grad_norm >10 3 consecutive steps '
                          f'(last={grad_norm:.2f}). clipping=1.0 in action.')
            else:
                self.high_grad_count = 0

        # VRAM
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            total = torch.cuda.mem_get_info()[1] / 1024**3
            used = total - free
            peak = torch.cuda.max_memory_allocated() / 1024**3

            # === GATE 5: VRAM pressure ===
            if free < 2.0:
                print(f'!!! WARN VRAM free={free:.1f}GB — OOM risk')

        else:
            free = used = peak = 0.0

        # Timing
        elapsed = time.time() - self.train_start if self.train_start else 0
        total_steps = state.max_steps if state.max_steps else 1
        progress = step / max(total_steps, 1)
        eta_sec = (elapsed / max(step, 1)) * (total_steps - step) if step > 0 else 0

        # Loss stats
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0
        avg_grad = sum(self.grad_history) / len(self.grad_history) if self.grad_history else 0

        # === GATE 6: Plateau warning ===
        plateau_warn = ''
        if self.steps_since_improve >= 100:
            plateau_warn = f' [PLATEAU {self.steps_since_improve}s]'

        # Print structured
        print(
            f'[step {step:4d}/{total_steps:4d} ep{epoch:.2f} '
            f'{progress*100:5.1f}%] '
            f'loss={loss:.4f} avg10={avg_loss:.4f} '
            f'grad={grad_norm:.3f if grad_norm is not None else 0:.3f} avg5={avg_grad:.3f} '
            f'lr={lr:.2e if lr is not None else 0:.2e} '
            f'vram={used:.1f}/{peak:.1f}/{free:.1f}GB '
            f'elapsed={int(elapsed//60)}m ETA={int(eta_sec//60)}m'
            f'{plateau_warn}'
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start if self.train_start else 0
        print('=' * 70)
        print(f'TRAINING FINISHED — elapsed={elapsed/60:.1f}min '
              f'min_loss={self.min_loss:.4f} steps={state.global_step}')
        print('=' * 70)


# dgxchen Stratified batching by type
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

trainer = StratifiedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    stratified_order=stratified_order,
    callbacks=[health_cb],
)

print('\nStarting SFT training with HEALTH GATES (log every 5 steps)...')
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f'\nTraining done: {elapsed/60:.1f} min')
print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f}GB')

# Save adapter
ADAPTER_DIR = '/content/kg1_adapter'
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f'\nAdapter saved: {ADAPTER_DIR}')

# Store for Cell 6
builtins._v78_adapter_dir = ADAPTER_DIR
print('PROCEED TO CELL 6 (package + submit)')
"""

CELL_6_SUBMIT = r"""# CELL 6: Package submission.zip + Kaggle submit
import os, json, shutil, zipfile, subprocess, sys, datetime, builtins
from pathlib import Path

ADAPTER_DIR = getattr(builtins, '_v78_adapter_dir', '/content/kg1_adapter')
BASE_MODEL_NAME = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
OUTPUT_DIR = '/content/kg1_out'
Path(OUTPUT_DIR).mkdir(exist_ok=True)
SUBMISSION_DIR = f'{OUTPUT_DIR}/submission_adapter'
Path(SUBMISSION_DIR).mkdir(exist_ok=True)

print('=' * 60)
print('PACKAGE SUBMISSION + KAGGLE SUBMIT')
print('=' * 60)

required = ['adapter_config.json', 'adapter_model.safetensors']
for fn in required:
    src = Path(ADAPTER_DIR) / fn
    dst = Path(SUBMISSION_DIR) / fn
    if not src.exists():
        raise FileNotFoundError(f'Missing: {src}')
    shutil.copy2(src, dst)
    print(f'Copied {fn} ({dst.stat().st_size/(1024**2):.1f} MB)')

# Fix adapter_config.json (inference_mode=True, base_model path)
cfg_path = Path(SUBMISSION_DIR) / 'adapter_config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['base_model_name_or_path'] = BASE_MODEL_NAME
cfg['inference_mode'] = True
cfg['lora_dropout'] = 0.0
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)

# Build zip
zip_path = f'{OUTPUT_DIR}/submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in required:
        zf.write(Path(SUBMISSION_DIR) / fn, arcname=fn)
zip_mb = os.path.getsize(zip_path) / (1024**2)
print(f'\nsubmission.zip: {zip_mb:.1f} MB')
assert zip_mb < 500, f'zip too big: {zip_mb}MB'

# Backup to GDrive if mounted
try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive', force_remount=False)
    bkp_dir = Path('/content/drive/MyDrive/kg1_v78')
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
    REPO = 'felipesp1983/kg1-nemotron-lora-v78-dgxchen'
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
            print('Slot exhausted — will NOT submit. Manual submit later:')
            print(f'  kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge \\')
            print(f'    -f {zip_path} -m "V78 dgxchen replica"')
        else:
            msg = f'V78 dgxchen replica + maxgrad1.0 {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
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
    print(f'    -f {zip_path} -m "V78 dgxchen replica"')

print('\nDONE V78. Review:')
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


OUT = 'notebooks/KG1_V78_DGXCHEN_REPLICA.ipynb'
os.makedirs('notebooks', exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1)

size = os.path.getsize(OUT)
print(f'Wrote {OUT} ({size} bytes)')

# py_compile each cell
import py_compile, tempfile
for i, c in enumerate([CELL_1_CHECK, CELL_2_INSTALL, CELL_3_DATA, CELL_4_MODEL_LORA, CELL_5_TRAIN, CELL_6_SUBMIT], 1):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as t:
        t.write(c)
        path = t.name
    try:
        py_compile.compile(path, doraise=True)
        print(f'Cell {i}: py_compile OK ({len(c.splitlines())} lines)')
    except py_compile.PyCompileError as e:
        print(f'Cell {i}: FAIL — {e}')
    finally:
        os.unlink(path)

print('\nV78 notebook ready for commit + push.')
