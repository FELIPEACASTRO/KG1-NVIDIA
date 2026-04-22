"""Build KG1_V80_MEGA.ipynb - SINGLE-CELL notebook for Colab (no restart needed).

Architecture:
  1. ONE CELL that does: install torch 2.5.1 + deps, downloads training script
     from GitHub raw URL, runs it via subprocess (child process has fresh torch 2.5).
  2. Child subprocess streams output to notebook live.
  3. Total time: ~15 min install + 3-4h training + submit = ~4.5h hands-off.

Training logic is in scripts/colab_mega_v80.py (separate file on GitHub, no escape issues).
"""
import json
import os

HEADER = r"""# KG1 V80 MEGA - 1 Cell Only, No Restart, dgxchen v7 EXACT

## Execução simplificada
**Apenas execute a célula abaixo.** Tudo roda em 1 cell:
- Install torch 2.5.1 + mamba-ssm + ML stack (~10 min)
- Download training script do GitHub
- Executa em child process (torch 2.5.1 fresh, sem conflito com Colab default 2.11)
- Download dataset (cached se V79 já rodou)
- Download Nemotron-30B (cached se V79 já rodou)
- Train 1 epoch ~245 steps (~2-3h H100)
- Build submission.zip
- Upload HF
- Submit Kaggle (automatic)

**Tempo total**: ~3-4h (se model/dataset cache) ou ~4-5h (cold start)

## Fixes aplicados (11 total)
Ver `scripts/colab_mega_v80.py` no repo para implementação.
- 7 divergências dgxchen v7 revertidas (dataset, attn, LoRA, epochs, grad_norm, checkpoint, formatting)
- 4 fixes de execução (mamba-ssm install, torch pin, num_workers=0, --no-deps)

## Credenciais (Colab Secrets - icone cadeado no painel esquerdo)
Adicione estes 3 secrets no Colab (Runtime > Secrets):
- `HF_KEY` = (seu HF token DEV)
- `KAGGLE_USERNAME` = felipe1983
- `KAGGLE_KEY` = (do kaggle.json)

## Hardware
- **Recomendado**: Colab Pro+ H100 80GB HBM3
- Também funciona: A100 80GB
- **Não funciona**: T4, L4, A10 (VRAM insuficiente para 30B BF16)
"""


MEGA_CELL = r"""# V80 MEGA CELL - install + download + train + submit (tudo em 1 cell)
# ATENCAO: espera ~3-4h para terminar. Output streams aqui live.
import subprocess, sys, os, json, urllib.request
from pathlib import Path

print('=' * 70)
print('V80 MEGA CELL - dgxchen v7 EXACT, 1 cell, no restart')
print('=' * 70)

# ============ 1. Load Colab secrets ============
try:
    from google.colab import userdata
    hf_key = userdata.get('HF_KEY')
    kaggle_user = userdata.get('KAGGLE_USERNAME')
    kaggle_key = userdata.get('KAGGLE_KEY')
except Exception:
    hf_key = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN')
    kaggle_user = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

assert hf_key, 'HF_KEY missing - add in Colab Secrets (cadeado esquerda)'
assert kaggle_user and kaggle_key, 'KAGGLE_USERNAME / KAGGLE_KEY missing'

os.environ['HF_TOKEN'] = hf_key
os.environ['HF_KEY'] = hf_key
os.environ['KAGGLE_USERNAME'] = kaggle_user
os.environ['KAGGLE_KEY'] = kaggle_key

# kaggle.json
kpath = Path.home() / '.kaggle' / 'kaggle.json'
kpath.parent.mkdir(parents=True, exist_ok=True)
kpath.write_text(json.dumps({'username': kaggle_user, 'key': kaggle_key}))
kpath.chmod(0o600)

print(f'HF token: ...{hf_key[-8:]}')
print(f'Kaggle user: {kaggle_user}')

# ============ 2. GPU check ============
import torch
assert torch.cuda.is_available(), 'CUDA/GPU required (use Colab H100 or A100)'
d = torch.cuda.get_device_properties(0)
total_gb = d.total_memory / 1024**3
print(f'GPU: {d.name} {total_gb:.1f}GB')
assert total_gb >= 38, f'Need 40GB+ GPU, got {total_gb:.1f}GB'

# ============ 3. Install torch 2.5.1 + mamba-ssm + ML stack ============
print()
print('Installing dependencies (torch 2.5.1 + mamba-ssm + ML stack)...')
print('Expected: ~10 min (torch wheel download + install)')


def sh(cmd, timeout=900, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        print(f'  FAIL: {" ".join(cmd[:6])}')
        print(f'  stderr: {r.stderr[-500:]}')
        raise RuntimeError('Command failed')
    return r


# Uninstall torch + Colab pre-installed torch.* packages (torch 2.11 ABI conflicts)
print('  Uninstalling existing torch + Colab pre-installed torchcodec/torchao/etc...')
for pkg in ['torch', 'torchvision', 'torchaudio',
            'torchcodec', 'torchao', 'torchdata', 'torchtune', 'torchsummary']:
    for i in range(5):
        r = sh([sys.executable, '-m', 'pip', 'uninstall', '-y', pkg], check=False)
        if 'Successfully uninstalled' not in r.stdout:
            break

# Install torch 2.5.1+cu124 via DIRECT wheel (bypass index-url resolution issues)
print('  Installing torch 2.5.1+cu124 (direct wheels)...')
for url in [
    'https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-linux_x86_64.whl',
    'https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp312-cp312-linux_x86_64.whl',
    'https://download.pytorch.org/whl/cu124/torchaudio-2.5.1%2Bcu124-cp312-cp312-linux_x86_64.whl',
]:
    sh([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', url])

# torch runtime deps
sh([sys.executable, '-m', 'pip', 'install', '-q',
    'filelock', 'jinja2', 'networkx', 'fsspec', 'sympy>=1.13', 'typing-extensions'])

# Mamba-ssm + causal-conv1d (torch 2.5 ABI wheels)
print('  Installing mamba-ssm + causal-conv1d (torch 2.5 ABI wheels)...')
sh([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
    'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'])
sh([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'])

# ML stack
print('  Installing transformers/peft/trl/accelerate/datasets/bitsandbytes...')
sh([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.48,<4.58', 'peft>=0.14,<0.18', 'trl>=0.14,<0.26',
    'accelerate>=1.0,<2.0', 'datasets>=3.2,<5',
    'bitsandbytes', 'huggingface_hub', 'safetensors', 'einops',
    'sentencepiece', 'pandas', 'kagglehub', 'einx'])

# Unsloth --no-deps (no torch upgrade)
print('  Installing unsloth + unsloth_zoo (--no-deps)...')
sh([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    'unsloth', 'unsloth_zoo', 'xformers', 'tyro', 'hf_transfer'], check=False)

# Verify via child process (child = fresh torch 2.5 import from disk)
print()
print('Verifying install via child process (torch 2.5 clean import)...')
r = subprocess.run([sys.executable, '-c', (
    "import torch, mamba_ssm; "
    "print(f'child: torch={torch.__version__} cuda={torch.version.cuda}'); "
    "print(f'child: mamba_ssm={mamba_ssm.__version__}'); "
    "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; "
    "from unsloth import FastLanguageModel; "
    "print('child: ALL IMPORTS OK')"
)], capture_output=True, text=True, timeout=120)
print(r.stdout)
if r.returncode != 0:
    print('child stderr:', r.stderr[-500:])
    raise RuntimeError('Child process verify failed - deps broken')

# ============ 4. Download training script from GitHub ============
print()
print('Downloading training script from GitHub (colab_mega_v80.py)...')
SCRIPT_URL = 'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir/scripts/colab_mega_v80.py'
SCRIPT_PATH = '/content/colab_mega_v80.py'

try:
    urllib.request.urlretrieve(SCRIPT_URL, SCRIPT_PATH)
    sz = os.path.getsize(SCRIPT_PATH)
    print(f'  Downloaded: {SCRIPT_PATH} ({sz/1024:.1f} KB)')
except Exception as e:
    print(f'  GitHub download failed: {e}')
    print('  Trying HF dataset repo fallback...')
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id='felipesp1983/kg1-nemotron-training',
        filename='scripts/colab_mega_v80.py',
        repo_type='dataset',
        token=hf_key,
    )
    import shutil
    shutil.copy2(local, SCRIPT_PATH)
    print(f'  HF fallback OK: {SCRIPT_PATH} ({os.path.getsize(SCRIPT_PATH)/1024:.1f} KB)')

# ============ 5. Execute training in subprocess (child has clean torch 2.5) ============
print()
print('=' * 70)
print('Starting V80 training in child process (streams output live)')
print('Expected: ~3-4h (download if cold + train 245 steps + submit)')
print('=' * 70)
print()

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'

proc = subprocess.Popen(
    [sys.executable, '-u', SCRIPT_PATH],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in iter(proc.stdout.readline, ''):
    print(line, end='', flush=True)
proc.wait()

print()
print('=' * 70)
print(f'V80 MEGA DONE - return code {proc.returncode}')
if proc.returncode == 0:
    print('SUCCESS: training finished, submission made')
    print('Check Kaggle score: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions')
else:
    print('FAILED: see output above for diagnosis')
print('=' * 70)
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
        cell_code(MEGA_CELL, 'mega'),
    ],
}


OUT = 'notebooks/KG1_V80_MEGA.ipynb'
os.makedirs('notebooks', exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1)

size = os.path.getsize(OUT)
print(f'Wrote {OUT} ({size} bytes)')

import py_compile, tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as t:
    t.write(MEGA_CELL)
    path = t.name
try:
    py_compile.compile(path, doraise=True)
    print(f'Mega cell: py_compile OK ({len(MEGA_CELL.splitlines())} lines)')
except py_compile.PyCompileError as e:
    print(f'Mega cell FAIL: {e}')
finally:
    os.unlink(path)

# Also verify colab_mega_v80.py compiles
with open('scripts/colab_mega_v80.py', encoding='utf-8') as f:
    mega_script = f.read()
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as t:
    t.write(mega_script)
    path = t.name
try:
    py_compile.compile(path, doraise=True)
    print(f'colab_mega_v80.py: py_compile OK ({len(mega_script.splitlines())} lines)')
except py_compile.PyCompileError as e:
    print(f'colab_mega_v80.py FAIL: {e}')
finally:
    os.unlink(path)

print('\nV80 MEGA notebook ready.')
print('Execution: open in Colab -> run the single cell -> wait ~3-4h.')
