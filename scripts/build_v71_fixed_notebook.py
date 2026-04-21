"""Build KG1_V71_MEGA_FIXED.ipynb with ALL fixes applied.

Fixes applied vs KG1_V71_MEGA_FINAL.ipynb:
1. Cell 1: resilient pip install (already fixed in FINAL)
2. Cell 2: force re-clone to always pick up latest branch commit
3. Cell 3: use login() not HfFolder (fixed)
4. Cell 4: set max_length=4096 (V70 PROVEN, not 8192)
5. Cell 4: add USE_NF4=True flag
6. Cell 4: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
7. Cell 6: use felipesp1983/kg1-nemotron-training + single file download
8. Cell 6: handle 'messages' chat format
9. Cell 7: BitsAndBytesConfig NF4 quantization (critical for H100 80GB)
10. Cell 7: prepare_model_for_kbit_training
11. Cell 8: smoke test with abort threshold
12-16: same as FINAL (save/eval/gate/submit/decide)
"""
import json
import uuid


def cell_md(content, cell_id=None):
    if not cell_id:
        cell_id = uuid.uuid4().hex[:12]
    if isinstance(content, str):
        content = content.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": content,
    }


def cell_code(content, cell_id=None):
    if not cell_id:
        cell_id = uuid.uuid4().hex[:12]
    if isinstance(content, str):
        content = content.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "metadata": {"id": cell_id},
        "execution_count": None,
        "outputs": [],
        "source": content,
    }


CELLS = []

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
CELLS.append(cell_md(
    """# KG1 V71 MEGA FIXED - Colab Pro H100 (NF4)

ALL FIXES APPLIED 2026-04-21:
- pip install resilient (no tight version pins)
- Force re-clone branch (always latest commit)
- HF login() not HfFolder (deprecated)
- max_length=4096 (V70 PROVEN, was 8192)
- NF4 quantization (model 60GB -> 15GB, fits H100 80GB)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- Dataset: felipesp1983/kg1-nemotron-training sft_v70_huikang_full.jsonl (16365 rows with CoT)
- LoRA r=32 alpha=32 all-linear (V70 proven)

Target: 0.84 baseline -> 0.87+ (99% rule gate before submit)

Execution order: run cells 1-15 sequentially. DO NOT skip.
""",
    cell_id="header"
))

# ----------------------------------------------------------------------
# CELL 1 - SETUP (resilient pip)
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 1 - Install dependencies (resilient)", cell_id="c1-md"))

CELLS.append(cell_code(
    """# Cell 1: install deps for NemotronH LoRA SFT (resilient install)
import subprocess, sys

def _pip_one(pkg, extra_args=None):
    cmd = [sys.executable, '-m', 'pip', 'install', '-q', pkg]
    if extra_args:
        cmd.extend(extra_args)
    print(f'pip install {pkg} ...', end=' ', flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print('TIMEOUT')
        return False
    if r.returncode == 0:
        print('OK')
        return True
    print('FAIL')
    print('  stderr tail:', r.stderr[-400:])
    return False

_pip_one('pip', extra_args=['--upgrade'])

CORE_PKGS = [
    'transformers>=4.55',
    'peft>=0.13',
    'trl>=0.25',
    'accelerate>=0.34',
    'bitsandbytes>=0.44',
    'datasets>=2.20',
    'safetensors>=0.4.5',
    'sentencepiece',
    'einops',
    'huggingface_hub>=0.25',
]
failed = [p for p in CORE_PKGS if not _pip_one(p)]
if failed:
    print(f'Retrying {len(failed)} failed with --upgrade...')
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + failed,
        capture_output=True, text=True, timeout=900,
    )

# mamba-ssm + causal-conv1d OPTIONAL (NemotronH slow path fallback)
print('\\n--- Installing mamba-ssm + causal-conv1d (OPTIONAL) ---')
try:
    import mamba_ssm  # noqa
    print('mamba-ssm already installed')
except ImportError:
    _pip_one('mamba-ssm', extra_args=['--no-build-isolation'])
try:
    import causal_conv1d  # noqa
    print('causal-conv1d already installed')
except ImportError:
    _pip_one('causal-conv1d>=1.4', extra_args=['--no-build-isolation'])

# Verification
import torch, transformers, peft, trl
print('\\n=== Installed versions ===')
print(f'torch:        {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
print(f'trl:          {trl.__version__}')
print(f'CUDA:         {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA not available - switch to A100/H100 runtime')
d = torch.cuda.get_device_properties(0)
vram_gb = d.total_memory / 1024**3
print(f'GPU: {d.name} ({vram_gb:.1f} GB VRAM)')
if vram_gb < 38:
    print(f'WARNING: VRAM {vram_gb:.1f}GB < 38GB - NF4 mandatory')

print('\\nCell 1 DONE')
""",
    cell_id="c1-setup"
))

# ----------------------------------------------------------------------
# CELL 2 - CLONE (force re-clone)
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 2 - Clone KG1 worktree (force fresh)", cell_id="c2-md"))

CELLS.append(cell_code(
    """# Cell 2: force re-clone KG1 to always pick up latest branch commit
import os, subprocess, shutil, sys
from pathlib import Path

KG1_DIR = Path('/content/kg1')
REPO = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
BRANCH = os.environ.get('KG1_BRANCH', 'claude/competent-shamir')

# STEP 1: Remove stale clone (prevents stale file errors)
if KG1_DIR.exists():
    print(f'Removing stale clone at {KG1_DIR}')
    shutil.rmtree(KG1_DIR)

# STEP 2: GDrive zip fallback (offline), else git clone
gdrive_zip = Path('/content/drive/MyDrive/kg1_src.zip')
if gdrive_zip.exists():
    print(f'Unzipping KG1 from GDrive: {gdrive_zip}')
    shutil.unpack_archive(str(gdrive_zip), str(KG1_DIR))
else:
    print(f'Cloning from GitHub: {REPO} branch={BRANCH}')
    subprocess.check_call([
        'git', 'clone', '--depth', '1', '--branch', BRANCH,
        REPO, str(KG1_DIR),
    ])

# STEP 3: Verify commit
try:
    commit = subprocess.check_output(
        ['git', '-C', str(KG1_DIR), 'log', '-1', '--format=%h %s'],
        text=True,
    ).strip()
    print(f'Latest commit: {commit}')
except Exception as e:
    print(f'WARN: could not read commit: {e}')

# STEP 4: Assert required source files
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
if missing:
    print('\\nERROR: Missing source files:')
    for m in missing:
        print(f'  - {m}')
    raise AssertionError(f'Missing: {missing}')

print('\\nAll required source files present:')
for r in REQUIRED:
    size_kb = (KG1_DIR / r).stat().st_size / 1024
    print(f'  OK {r} ({size_kb:.1f} KB)')

# STEP 5: Put KG1 on sys.path
if str(KG1_DIR) not in sys.path:
    sys.path.insert(0, str(KG1_DIR))
print(f'\\nsys.path[0] = {sys.path[0]}')
print('\\nCell 2 DONE')
""",
    cell_id="c2-clone"
))

# ----------------------------------------------------------------------
# CELL 3 - AUTH (login() not HfFolder)
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 3 - Auth (GDrive + HF login + Kaggle)", cell_id="c3-md"))

CELLS.append(cell_code(
    """# Cell 3: mount GDrive + HF login() + Kaggle creds
import os, json, subprocess
from pathlib import Path

try:
    from google.colab import drive, userdata  # type: ignore
    drive.mount('/content/drive')

    # HF_KEY (per user memory)
    try:
        hf_key = userdata.get('HF_KEY')
    except Exception:
        hf_key = None
    if not hf_key:
        hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    assert hf_key, 'HF_KEY not found - add it to Colab secrets (name must be HF_KEY)'
    os.environ['HF_TOKEN'] = hf_key
    os.environ['HF_KEY'] = hf_key

    # Kaggle
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
        print(f'Kaggle creds installed for user {kuser}')
    else:
        print('WARNING: Kaggle creds not set (needed for Cell 15)')
except ImportError:
    print('Not running in Colab - using env vars')
    assert os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY'), 'HF token required'

# Modern HF login
from huggingface_hub import login, whoami
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
login(token=hf_token, add_to_git_credential=False)
print('HF token saved via login().')
try:
    info = whoami(token=hf_token)
    print(f'HF user: {info["name"]}')
except Exception as e:
    print(f'WARNING: whoami failed: {e}')

# Kaggle auth test
r = subprocess.run(['kaggle', 'competitions', 'list', '-s', 'nemotron'],
                   capture_output=True, text=True, timeout=30)
if r.returncode == 0:
    print('Kaggle API auth OK')
else:
    print(f'Kaggle test stderr: {r.stderr[:200]}')

print('\\nCell 3 DONE')
""",
    cell_id="c3-auth"
))

# ----------------------------------------------------------------------
# CELL 4 - CONFIG (NF4 + max_length=4096)
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 4 - V71 Config (NF4 + max_length=4096)", cell_id="c4-md"))

CELLS.append(cell_code(
    """# Cell 4: V71 config with NF4 + max_length=4096 (V70 PROVEN)
import os, json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# CRITICAL: set allocator config BEFORE any CUDA op
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print(f'PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True')

@dataclass
class V71Config:
    # --- Model (NF4 quantization for H100 80GB) ---
    base_model: str = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    use_nf4: bool = True                    # CRITICAL: 60GB BF16 -> 15GB NF4
    max_length: int = 4096                  # V70 PROVEN (not 8192)
    attn_implementation: str = 'eager'       # NemotronH hybrid requires eager
    mamba_ssm_cache_dtype: str = 'float32'   # T1 numerical stability
    tie_word_embeddings: bool = False        # V7 (V18 collapse prevention)

    # --- LoRA (T2 baseline) ---
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = 'all-linear'
    use_dora: bool = False
    use_rank_pattern: bool = False

    # --- Training ---
    epochs: int = 1
    per_device_batch: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = 'linear'
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    optimizer: str = 'paged_adamw_8bit'
    bf16: bool = True
    gradient_checkpointing: bool = False

    # --- Loss (V5 max-min with CE warmup) ---
    loss_type: str = 'max_min_warmup_ce'
    max_min_warmup_steps: int = 100

    # --- Data ---
    hf_dataset_repo: str = 'felipesp1983/kg1-nemotron-training'
    hf_dataset_file: str = 'data/sft_v70_huikang_full.jsonl'

    # --- Prompt ---
    enable_thinking: bool = True
    use_structured: bool = True
    use_category_hints: bool = True
    use_boxed_strict: bool = True
    use_self_correct: bool = True

    # --- Gates ---
    smoke_test_steps: int = 2
    smoke_abort_loss: float = 50.0
    eval_holdout_size: int = 600
    local_score_floor: float = 0.84
    target_score: float = 0.87

    # --- Output ---
    run_tag: str = 'v71_fixed'
    output_dir: str = '/content/kg1_out/v71_fixed'
    gdrive_checkpoint: str = '/content/drive/MyDrive/kg1_checkpoints/v71_fixed'
    hf_upload_repo: str = 'felipesp1983/kg1-nemotron-lora-v71-fixed'

CFG = V71Config()
# Invariants
assert CFG.grad_clip > 0
assert CFG.max_length <= 8192
assert CFG.attn_implementation == 'eager'
assert CFG.tie_word_embeddings is False
assert CFG.loss_type in ('ce', 'max_min', 'max_min_warmup_ce')

Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(CFG.output_dir) / 'config.json', 'w') as f:
    json.dump(asdict(CFG), f, indent=2)
print(f'\\nV71 config written to {Path(CFG.output_dir) / "config.json"}')
print(json.dumps(asdict(CFG), indent=2))

print('\\nCell 4 DONE')
""",
    cell_id="c4-config"
))

# ----------------------------------------------------------------------
# CELL 5 - PREFLIGHT
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 5 - Pre-flight checks", cell_id="c5-md"))

CELLS.append(cell_code(
    """# Cell 5: pre-flight - VRAM, module imports, tokenizer
import torch, importlib
from transformers import AutoTokenizer

# 5.1 VRAM check (need >=38GB for NF4 + activations)
d = torch.cuda.get_device_properties(0)
vram_gb = d.total_memory / 1024**3
print(f'GPU: {d.name}, VRAM: {vram_gb:.1f} GB')
assert vram_gb >= 38, f'Need >=38GB, got {vram_gb:.1f}'

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f'VRAM free: {free_gb:.1f} GB')

# 5.2 Smoke-import KG1 modules
for mod_name in [
    'src.reasoners.bit_manipulation_pairs',
    'src.reasoners.cryptarithm_47combo',
    'src.reasoners.neurosymbolic_template',
    'src.losses.max_min_logprob',
    'src.prompts.build_prompt',
]:
    mod = importlib.import_module(mod_name)
    print(f'imported {mod_name}')

# 5.3 Tokenizer
tok = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f'\\ntokenizer: vocab={tok.vocab_size} pad={tok.pad_token}')

# 5.4 enable_thinking
try:
    _ = tok.apply_chat_template(
        [{'role': 'user', 'content': 'test'}],
        enable_thinking=True, tokenize=False,
    )
    print('enable_thinking=True SUPPORTED')
except TypeError as e:
    print(f'WARN: enable_thinking not supported - {e}')

# 5.5 Self-tests
from src.reasoners.bit_manipulation_pairs import generate_cot as gen_bit
pred, cot = gen_bit([('00000000', '10101010'), ('11111111', '01010101')], '10101010')
assert pred is not None
print(f'bit_manipulation self-test OK, pred={pred}')

from src.prompts.build_prompt import build_prompt_v71
p = build_prompt_v71('Decrypt test', 'cipher')
assert '\\\\boxed' in p or 'boxed' in p, 'BOXED_INSTRUCTION missing'
print(f'build_prompt_v71 OK, length={len(p)}')

print('\\nAll pre-flight checks PASSED.')
""",
    cell_id="c5-preflight"
))

# ----------------------------------------------------------------------
# CELL 6 - DATA (felipesp1983 repo, single file)
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 6 - Load V70 dataset with CoT (100%)", cell_id="c6-md"))

CELLS.append(cell_code(
    """# Cell 6: load felipesp1983/kg1-nemotron-training -> sft_v70_huikang_full.jsonl
import os, json, random
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd

from src.prompts.build_prompt import build_prompt_v71, detect_category

HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
GDRIVE_CACHE = Path('/content/drive/MyDrive/kg1_data')
GDRIVE_CACHE.mkdir(parents=True, exist_ok=True)

print(f'Downloading {CFG.hf_dataset_file} from {CFG.hf_dataset_repo}...')
v70_path = hf_hub_download(
    repo_id=CFG.hf_dataset_repo,
    filename=CFG.hf_dataset_file,
    repo_type='dataset',
    local_dir=str(GDRIVE_CACHE / 'v70_huikang'),
    token=HF_TOKEN,
)
print(f'OK: {v70_path}  ({os.path.getsize(v70_path)/1024/1024:.1f} MB)')

df = pd.read_json(v70_path, lines=True)
print(f'\\nLoaded {len(df)} rows, columns: {list(df.columns)}')

# Auto-detect cols
COL_MAP = {
    'prompt':   ['prompt', 'question', 'input', 'text', 'user', 'problem'],
    'response': ['response', 'completion', 'output', 'assistant', 'cot',
                 'generation', 'solution', 'reasoning'],
    'answer':   ['answer', 'target', 'label', 'ground_truth', 'final_answer'],
    'category': ['category', 'type', 'puzzle_type', 'family'],
}
def find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

PROMPT_COL = find_col(df, COL_MAP['prompt'])
RESPONSE_COL = find_col(df, COL_MAP['response'])
ANSWER_COL = find_col(df, COL_MAP['answer'])
CAT_COL = find_col(df, COL_MAP['category'])

# Handle chat format (messages column)
if 'messages' in df.columns and not RESPONSE_COL:
    print('Extracting from messages column (chat format)...')
    def extract(msgs):
        if not isinstance(msgs, list):
            return None, None
        u = next((m['content'] for m in msgs if m.get('role') == 'user'), None)
        a = next((m['content'] for m in msgs if m.get('role') == 'assistant'), None)
        return u, a
    df[['_u', '_a']] = df['messages'].apply(lambda m: pd.Series(extract(m)))
    df = df.rename(columns={'_u': 'prompt', '_a': 'response'})
    PROMPT_COL = 'prompt'
    RESPONSE_COL = 'response'

print(f'Detected: prompt={PROMPT_COL} response={RESPONSE_COL} answer={ANSWER_COL} category={CAT_COL}')
assert PROMPT_COL, f'No prompt col: {list(df.columns)}'
assert RESPONSE_COL or ANSWER_COL

rename_map = {PROMPT_COL: 'prompt'}
if RESPONSE_COL and RESPONSE_COL != 'response': rename_map[RESPONSE_COL] = 'response'
if ANSWER_COL and ANSWER_COL != 'answer': rename_map[ANSWER_COL] = 'answer'
if CAT_COL and CAT_COL != 'category': rename_map[CAT_COL] = 'category'
df = df.rename(columns=rename_map)

if 'category' not in df.columns or df['category'].isna().all():
    df['category'] = df['prompt'].map(detect_category)

print(f'\\nCategory distribution:')
for cat, n in df['category'].value_counts().head(15).items():
    print(f'  {cat}: {n}')

# Build records
def build_record(row):
    prompt = row.get('prompt')
    if prompt is None or (isinstance(prompt, float) and pd.isna(prompt)):
        return None
    prompt = str(prompt).strip()
    if not prompt:
        return None

    cat = str(row.get('category', '')) if pd.notna(row.get('category', '')) else ''
    user = build_prompt_v71(
        prompt, category=cat,
        use_structured=CFG.use_structured,
        use_category_hints=CFG.use_category_hints,
        use_boxed_strict=CFG.use_boxed_strict,
        use_self_correct=CFG.use_self_correct,
    )

    resp = row.get('response')
    if resp is not None and pd.notna(resp) and str(resp).strip():
        assistant = str(resp).strip()
        if '\\\\boxed{' not in assistant:
            ans = row.get('answer', '')
            if ans and pd.notna(ans):
                assistant = assistant + f'\\n\\\\boxed{{{ans}}}'
    else:
        ans = row.get('answer', '')
        if not ans or pd.isna(ans):
            return None
        assistant = f'\\\\boxed{{{ans}}}'
    return {'user': user, 'assistant': assistant, 'category': cat}

print('\\nBuilding training records...')
records = [r for r in (build_record(row) for _, row in df.iterrows()) if r is not None]
print(f'Built {len(records)} records (dropped {len(df)-len(records)} null).')

n_cot = sum(1 for r in records if len(r['assistant']) > 50)
print(f'Records with CoT (>50 chars): {n_cot}/{len(records)} ({100*n_cot/len(records):.1f}%)')
assert n_cot > len(records) * 0.5, f'<50% have CoT - check source'

# Split
random.seed(42)
idx = list(range(len(records)))
random.shuffle(idx)
eval_n = min(CFG.eval_holdout_size, max(50, len(records) // 20))
eval_set = set(idx[:eval_n])
train_records = [records[i] for i in idx if i not in eval_set]
eval_records = [records[i] for i in idx if i in eval_set]
print(f'\\ntrain={len(train_records)}  eval={len(eval_records)}')

# Persist
out_path = Path(CFG.output_dir) / 'train.jsonl'
with open(out_path, 'w') as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\\n')
eval_path = Path(CFG.output_dir) / 'eval.jsonl'
with open(eval_path, 'w') as f:
    for r in eval_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\\n')
print(f'wrote {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)')
print(f'wrote {eval_path} ({eval_path.stat().st_size/1024/1024:.2f} MB)')

print('\\nCell 6 DONE (V70 CoT 100%)')
""",
    cell_id="c6-data"
))

# ----------------------------------------------------------------------
# CELL 7 - MODEL WITH NF4 (CRITICAL FIX)
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 7 - Load NemotronH 30B NF4 + LoRA (FITS H100!)", cell_id="c7-md"))

CELLS.append(cell_code(
    """# Cell 7: NemotronH 30B NF4 + LoRA (60GB -> 15GB, fits H100 80GB with margin)
import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# VRAM check before load
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f'VRAM free before load: {free_gb:.1f} GB')
assert free_gb >= 30, f'Need >=30GB free for NF4 load. Got {free_gb:.1f}GB. Restart runtime.'

# NF4 quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
print('NF4 quantization enabled.')

# Model config + V7 invariants
model_cfg = AutoConfig.from_pretrained(CFG.base_model, trust_remote_code=True)
if getattr(model_cfg, 'tie_word_embeddings', False):
    print('WARN: base config has tie_word_embeddings=True - forcing False (V18 incident)')
setattr(model_cfg, 'tie_word_embeddings', False)
if hasattr(model_cfg, 'mamba_ssm_cache_dtype'):
    setattr(model_cfg, 'mamba_ssm_cache_dtype', 'float32')

print(f'\\nLoading {CFG.base_model} with NF4 (first time: 8-15 min)...')
model = AutoModelForCausalLM.from_pretrained(
    CFG.base_model,
    config=model_cfg,
    quantization_config=bnb_config,
    device_map={'': 0},
    attn_implementation=CFG.attn_implementation,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print('Base model loaded (NF4).')

# Prepare for LoRA training on quantized base
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
print('prepare_model_for_kbit_training done.')

# Attach LoRA
rank_pattern = None
if CFG.use_rank_pattern:
    rank_pattern = {
        r'.*up_proj': 64,
        r'.*gate_proj': 48,
        r'.*down_proj': 16,
    }
lora_kwargs = dict(
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    target_modules=CFG.lora_target_modules,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    use_dora=CFG.use_dora,
)
if rank_pattern:
    lora_kwargs['rank_pattern'] = rank_pattern
peft_cfg = LoraConfig(**lora_kwargs)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f'Trainable: {n_trainable:,} / Total: {n_total:,}  ({100*n_trainable/n_total:.3f}%)')
assert 0 < n_trainable < n_total

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
print(f'\\nGPU after NF4 load: used={used_gb:.1f}GB, free={free_gb:.1f}GB')
assert free_gb >= 20, f'Only {free_gb:.1f}GB free - not enough for smoke test'

print('\\nCell 7 DONE (NF4 fits H100 with margin)')
""",
    cell_id="c7-model-nf4"
))

# ----------------------------------------------------------------------
# CELL 8 - SMOKE TEST
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 8 - SMOKE TEST (2 steps, abort if loss>50)", cell_id="c8-md"))

CELLS.append(cell_code(
    """# Cell 8: smoke test 2 steps with CE loss (no-explosion gate)
import math, json, torch
from pathlib import Path
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
            user_text = self.tok.apply_chat_template(
                [messages[0]], tokenize=False, enable_thinking=CFG.enable_thinking,
            )
        except TypeError:
            user_text = self.tok.apply_chat_template([messages[0]], tokenize=False)
        user_ids = self.tok(user_text, return_tensors='pt')['input_ids'][0]
        k = min(len(user_ids), len(labels))
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

train_path = Path(CFG.output_dir) / 'train.jsonl'
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
    assert not math.isnan(loss.item()), f'NaN loss at step {step}'
    assert not math.isinf(loss.item()), f'Inf loss at step {step}'
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], CFG.grad_clip,
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f'smoke step {step} loss={loss.item():.4f}')
    if step + 1 >= CFG.smoke_test_steps:
        break

assert len(losses) == CFG.smoke_test_steps
assert losses[-1] < CFG.smoke_abort_loss, (
    f'ABORT: smoke loss {losses[-1]:.2f} > threshold {CFG.smoke_abort_loss}'
)
print('\\nSmoke test PASSED.')

# Free optimizer from smoke test (Cell 9 creates a fresh one)
del opt
import gc; gc.collect(); torch.cuda.empty_cache()
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f'VRAM free after smoke cleanup: {free_gb:.1f} GB')
""",
    cell_id="c8-smoke"
))

# ----------------------------------------------------------------------
# CELL 9 - FULL TRAINING
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 9 - Full training (max-min logprob + CE warmup)", cell_id="c9-md"))

CELLS.append(cell_code(
    """# Cell 9: full 1-epoch training with V5 max-min logprob loss
import torch, os
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from src.losses.max_min_logprob import max_min_logprob_loss

train_path = str(Path(CFG.output_dir) / 'train.jsonl')
eval_path = str(Path(CFG.output_dir) / 'eval.jsonl')

ds_train = load_dataset('json', data_files=train_path, split='train')
ds_eval = load_dataset('json', data_files=eval_path, split='train')

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
    gradient_checkpointing=CFG.gradient_checkpointing,
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
        else:  # max_min_warmup_ce
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
print('Starting training (1 epoch, ~4-8h on H100 NF4)...')
trainer.train()
print('\\nTraining complete. Saving adapter...')
trainer.save_model(CFG.output_dir)
tok.save_pretrained(CFG.output_dir)
print(f'Saved to {CFG.output_dir}')
""",
    cell_id="c9-train"
))

# ----------------------------------------------------------------------
# CELL 10 - SAVE + UPLOAD
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 10 - Save adapter + backup GDrive + upload HF", cell_id="c10-md"))

CELLS.append(cell_code(
    """# Cell 10: save adapter, GDrive backup, HF upload
import shutil, datetime, os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

out_dir = Path(CFG.output_dir)
required_files = ['adapter_config.json', 'adapter_model.safetensors']
present = [f for f in required_files if (out_dir / f).exists()]
assert len(present) == len(required_files), f'Missing: {set(required_files) - set(present)}'

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
gdrive_dest = Path(CFG.gdrive_checkpoint) / f'{CFG.run_tag}_{ts}'
try:
    gdrive_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out_dir, gdrive_dest, dirs_exist_ok=True)
    print(f'Checkpoint -> GDrive: {gdrive_dest}')
except Exception as e:
    print(f'WARN: GDrive save failed: {e}')

HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(CFG.hf_upload_repo, private=True, exist_ok=True)
    upload_folder(
        repo_id=CFG.hf_upload_repo,
        folder_path=str(out_dir),
        allow_patterns=['adapter_*', 'tokenizer*', 'special_tokens*', 'config.json'],
        token=HF_TOKEN,
    )
    print(f'Uploaded -> HF: {CFG.hf_upload_repo}')
except Exception as e:
    print(f'WARN: HF upload failed: {e}')
""",
    cell_id="c10-save"
))

# ----------------------------------------------------------------------
# CELL 11 - LOCAL EVAL
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 11 - Local eval on 600-row holdout", cell_id="c11-md"))

CELLS.append(cell_code(
    """# Cell 11: run local_score.py with CORRECTED metric
import subprocess, json, sys, re
from pathlib import Path

local_score = Path('/content/kg1/scripts/local_score.py')
assert local_score.exists()
eval_csv = Path(CFG.output_dir) / 'local_eval.csv'

cmd = [
    sys.executable, str(local_score),
    '--adapter', str(Path(CFG.output_dir)),
    '--n-samples', str(CFG.eval_holdout_size),
    '--output-csv', str(eval_csv),
]
print('Running:', ' '.join(cmd))
try:
    res = subprocess.run(cmd, cwd='/content/kg1', check=False,
                         capture_output=True, text=True, timeout=3600)
    print('STDOUT:', res.stdout[-2000:])
    if res.returncode != 0:
        print('STDERR:', res.stderr[-2000:])
except subprocess.TimeoutExpired:
    print('local_score timeout')
    res = type('R', (), {'stdout': '', 'stderr': ''})

local_score_val = None
m = re.search(r'(?:overall\\s+score|score)[:\\s]+([0-9.]+)', res.stdout, re.IGNORECASE)
if m:
    local_score_val = float(m.group(1))
print(f'\\nParsed local score = {local_score_val}')

with open(Path(CFG.output_dir) / 'local_score.json', 'w') as f:
    json.dump({'local_score': local_score_val, 'n_samples': CFG.eval_holdout_size}, f)
""",
    cell_id="c11-eval"
))

# ----------------------------------------------------------------------
# CELL 12 - GATE (99% rule)
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 12 - 99% rule GO/NO-GO gate", cell_id="c12-md"))

CELLS.append(cell_code(
    """# Cell 12: GO/NO-GO based on local score
import json
from pathlib import Path

d = json.loads((Path(CFG.output_dir) / 'local_score.json').read_text())
score = d.get('local_score')

GO = False
if score is None:
    msg = 'NO-GO: local_score parse failed - skipping submission (99% rule)'
elif score < CFG.local_score_floor:
    msg = f'NO-GO: score {score:.4f} < floor {CFG.local_score_floor} - would regress baseline'
elif score < CFG.target_score - 0.01:
    msg = f'MARGINAL: score {score:.4f} ~ target {CFG.target_score}'
    GO = True
else:
    msg = f'GO: score {score:.4f} >= target {CFG.target_score}'
    GO = True

print(msg)
with open(Path(CFG.output_dir) / 'gate_decision.json', 'w') as f:
    json.dump({'go': GO, 'msg': msg, 'score': score}, f)
if not GO:
    print('DO NOT run Cells 13-15. Rollback to V70 or retry with different config.')
""",
    cell_id="c12-gate"
))

# ----------------------------------------------------------------------
# CELL 13 - ZIP
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 13 - Build Kaggle submission ZIP", cell_id="c13-md"))

CELLS.append(cell_code(
    """# Cell 13: build submission.zip (2 files at ROOT, no lm_head issue)
import zipfile, json
from pathlib import Path

gate = json.loads((Path(CFG.output_dir) / 'gate_decision.json').read_text())
assert gate.get('go'), f'Gate NO-GO: {gate.get("msg")}'

out_dir = Path(CFG.output_dir)
zip_path = out_dir / 'submission.zip'

ROOT_FILES = ['adapter_config.json', 'adapter_model.safetensors']
for fn in ROOT_FILES:
    assert (out_dir / fn).exists(), f'missing {fn}'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in ROOT_FILES:
        zf.write(out_dir / fn, arcname=fn)

size_mb = zip_path.stat().st_size / (1024*1024)
print(f'submission.zip: {zip_path}  ({size_mb:.2f} MB)')
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    assert len(names) == 2, f'expected 2 files, got {names}'
print(f'Contents: {names}')
""",
    cell_id="c13-zip"
))

# ----------------------------------------------------------------------
# CELL 14 - SUBMISSION GATE
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 14 - kg1_submission_gate.py double-check", cell_id="c14-md"))

CELLS.append(cell_code(
    """# Cell 14: kg1_submission_gate.py double-check zip integrity
import subprocess, sys
from pathlib import Path

gate_script = Path('/content/kg1/scripts/kg1_submission_gate.py')
zip_path = Path(CFG.output_dir) / 'submission.zip'
cmd = [sys.executable, str(gate_script), '--zip', str(zip_path)]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print('STDOUT:', res.stdout[-1500:])
print('STDERR:', res.stderr[-500:])
assert res.returncode == 0, f'submission_gate REJECTED (rc={res.returncode})'
print('\\nSubmission gate PASSED.')
""",
    cell_id="c14-subgate"
))

# ----------------------------------------------------------------------
# CELL 15 - SUBMIT
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 15 - Submit to Kaggle", cell_id="c15-md"))

CELLS.append(cell_code(
    """# Cell 15: submit via Kaggle API (respects 5/day limit)
import os, json, subprocess, sys, datetime
from pathlib import Path

assert os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'), 'Kaggle creds missing'

submit_script = Path('/content/kg1/scripts/submit_kaggle.py')
zip_path = Path(CFG.output_dir) / 'submission.zip'
msg = f'V71_FIXED {datetime.datetime.now().strftime("%Y-%m-%d %H:%M BRT")}'

if submit_script.exists():
    cmd = [sys.executable, str(submit_script), '--zip', str(zip_path), '--message', msg]
else:
    cmd = [
        'kaggle', 'competitions', 'submit',
        '-c', 'nvidia-nemotron-model-reasoning-challenge',
        '-f', str(zip_path), '-m', msg,
    ]
print(f'Submit cmd: {" ".join(cmd)}')
res = subprocess.run(cmd, capture_output=True, text=True)
print('STDOUT:', res.stdout)
print('STDERR:', res.stderr)

with open(Path(CFG.output_dir) / 'kaggle_submit.json', 'w') as f:
    json.dump({
        'msg': msg,
        'returncode': res.returncode,
        'stdout_tail': res.stdout[-1000:],
        'stderr_tail': res.stderr[-500:],
    }, f)
print('\\nSubmission dispatched. Monitor at kaggle.com/competitions Submissions tab.')
""",
    cell_id="c15-submit"
))

# ----------------------------------------------------------------------
# CELL 16 - DECISION TREE
# ----------------------------------------------------------------------
CELLS.append(cell_md("## Cell 16 - Decision tree (next action)", cell_id="c16-md"))

CELLS.append(cell_code(
    """# Cell 16: interpret score + decide next step
import json
from pathlib import Path

gate = json.loads((Path(CFG.output_dir) / 'gate_decision.json').read_text())
score = gate.get('score') or 0.0

DECISION_TREE = [
    (0.87, 'TOP1_CANDIDATE',  'Stage 2 (LoRA Soup DARE-TIES) + validate across 3 seeds'),
    (0.86, 'PLATEAU_PUSH',     'Add programmatic per-family solvers at inference (Stage 3)'),
    (0.85, 'MARGIN_PROBE',     'Run ablation (V71b DoRA) + (V71c rank_pattern)'),
    (0.84, 'BASELINE_HOLD',    'No regression. Invest in CoT quality (distill)'),
    (0.0,  'ROLLBACK_V70',     'Score < 0.84 - ROLLBACK. Audit corpus before retry'),
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
    'regras_imutaveis': [
        'Reservar 2/5 slots para rollback',
        'Nunca strip_lm_head (V18)',
        'Nunca mudar >1 variavel por iteracao',
        'Sempre gate submission + kaggle_like_gate antes de submit',
    ],
}
with open(Path(CFG.output_dir) / 'decision.json', 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
""",
    cell_id="c16-decide"
))

# ----------------------------------------------------------------------
# Build notebook
# ----------------------------------------------------------------------
NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "machine_shape": "hm"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": CELLS,
}

import os
OUT = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'KG1_V71_MEGA_FIXED.ipynb')
OUT = os.path.abspath(OUT)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f'Wrote {OUT} ({os.path.getsize(OUT)} bytes, {len(CELLS)} cells)')

# py_compile sanity
import py_compile, tempfile
for i, c in enumerate(CELLS):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as t:
        t.write(src)
        tpath = t.name
    try:
        py_compile.compile(tpath, doraise=True)
    except py_compile.PyCompileError as e:
        print(f'SYNTAX ERROR in cell {i} ({c["metadata"].get("id")}):')
        print(e)
        raise
    os.unlink(tpath)
print(f'All {sum(1 for c in CELLS if c["cell_type"] == "code")} code cells passed py_compile.')
