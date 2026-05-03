#!/usr/bin/env python3
"""Build V198 micro-distillation data and Colab pack.

V198 is intentionally small and conservative. It assumes V195 was already
trained or partially checkpointed in Drive, then applies a short continuation
focused on:

* V197 strict feature gains that survived stress testing.
* V196 wrong cases as anti-regression rehearsal.
* Stable family rehearsal to avoid losing the near-perfect categories.

No Kaggle submission is created by this script or by the notebook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kg1_canonicalize_output import _extract_last_boxed_body, canonicalize_answer


CANONICAL_FAMILIES = {
    "gravity_constant",
    "unit_conversion",
    "numeral_system",
    "text_encryption",
    "bit_manipulation",
    "equation_transform",
}

BASE_SAMPLE_TARGETS = {
    "bit_manipulation": 360,
    "equation_transform": 520,
    "gravity_constant": 220,
    "unit_conversion": 220,
    "numeral_system": 220,
    "text_encryption": 220,
}


def canonical_family(raw: Any) -> str:
    family = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if family in CANONICAL_FAMILIES:
        return family
    if family == "gravity":
        return "gravity_constant"
    if family.startswith("equation") or "equation_numeric" in family:
        return "equation_transform"
    if family.startswith("cryptarithm") or "crypt" in family:
        return "equation_transform"
    raise ValueError(f"unknown family: {raw!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_identity(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    answer = str(row.get("answer") or "")
    return hashlib.sha256((prompt.strip() + "\n" + answer.strip()).encode("utf-8")).hexdigest()


def has_messages_and_answer(row: dict[str, Any]) -> bool:
    if not str(row.get("prompt") or "").strip():
        return False
    if not str(row.get("answer") or "").strip():
        return False
    messages = row.get("messages")
    return isinstance(messages, list) and len(messages) >= 2


def normalize_base_row(row: dict[str, Any], *, source: str, role: str) -> dict[str, Any]:
    out = dict(row)
    out["family"] = canonical_family(out.get("family") or out.get("subcategory"))
    out["source"] = source
    metadata = dict(out.get("metadata") or {})
    metadata["v198_source_role"] = role
    metadata.setdefault("v198_original_source", row.get("source"))
    out["metadata"] = metadata
    enforce_strict_answer_format(out)
    return out


def enforce_strict_answer_format(row: dict[str, Any]) -> None:
    """Keep copied rows compatible with the local SFT format gate."""
    family = canonical_family(row.get("family") or row.get("subcategory"))
    if family != "equation_transform":
        return
    messages = row.get("messages")
    if not isinstance(messages, list):
        return
    answer = str(row.get("answer") or "").strip()
    if not answer:
        return
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        if "final answer is:" in content.lower():
            return
        boxed = f"\\boxed{{{answer}}}"
        idx = content.rfind(boxed)
        if idx >= 0:
            prefix = content[:idx].rstrip()
            suffix = content[idx:].lstrip()
            msg["content"] = f"{prefix}\nFinal answer is: {answer}\n{suffix}".strip()
        else:
            msg["content"] = f"{content.rstrip()}\nFinal answer is: {answer}\n{boxed}".strip()
        return


def clone_row(
    row: dict[str, Any],
    *,
    source: str,
    role: str,
    suffix: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = normalize_base_row(row, source=source, role=role)
    old_id = str(out.get("id") or row_identity(row)[:16])
    out["id"] = f"v198_{suffix}_{old_id}"
    metadata = dict(out.get("metadata") or {})
    metadata["v198_clone_of"] = old_id
    if extra_metadata:
        metadata.update(extra_metadata)
    out["metadata"] = metadata
    return out


def make_strict_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert validation rows to short answer-only completions.

    This keeps eval deterministic and avoids wasting H100 time on long copied
    rationales that are not needed for validation loss.
    """
    out = dict(row)
    family = canonical_family(out.get("family") or out.get("subcategory"))
    out["family"] = family
    answer = str(out.get("answer") or "").strip()
    if answer and ("{" in answer or "}" in answer):
        boxed_answer = answer.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        if family == "equation_transform":
            completion = f"Final answer is: {answer}\n\\boxed{{{boxed_answer}}}"
        else:
            completion = f"\\boxed{{{boxed_answer}}}"
    else:
        completion = canonicalize_answer(f"\\boxed{{{answer}}}", family_hint=family) if answer else ""
    if not completion and answer:
        completion = f"\\boxed{{{answer}}}"
    body = (_extract_last_boxed_body(completion) or answer).strip()
    if body:
        out["answer"] = body
    messages = out.get("messages")
    if isinstance(messages, list):
        rewritten: list[dict[str, Any]] = []
        replaced = False
        for msg in messages:
            msg_out = dict(msg) if isinstance(msg, dict) else msg
            if isinstance(msg_out, dict) and msg_out.get("role") == "assistant":
                msg_out["content"] = completion
                replaced = True
            rewritten.append(msg_out)
        if not replaced:
            rewritten.append({"role": "assistant", "content": completion})
        out["messages"] = rewritten
    metadata = dict(out.get("metadata") or {})
    metadata["v198_eval_format"] = "strict_answer_only"
    out["metadata"] = metadata
    return out


def sample_base_rows(
    rows: list[dict[str, Any]],
    *,
    targets: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not has_messages_and_answer(row):
            continue
        family = canonical_family(row.get("family") or row.get("subcategory"))
        buckets[family].append(normalize_base_row(row, source="v198_v195_balanced_rehearsal", role="balanced_rehearsal"))

    sampled: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seen_identity: set[str] = set()
    for family, target in targets.items():
        bucket = buckets.get(family, [])
        rng.shuffle(bucket)
        added = 0
        for row in bucket:
            key = row_identity(row)
            if key in seen_identity:
                continue
            seen_identity.add(key)
            sampled.append(row)
            added += 1
            if added >= target:
                break
        counts[family] = added
    return sampled, counts


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_source_row(source_file: str, source_line: str) -> dict[str, Any] | None:
    path = Path(source_file)
    try:
        line_no = int(source_line)
    except (TypeError, ValueError):
        return None
    if not path.exists() or line_no <= 0:
        return None
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle, 1):
            if current == line_no:
                return json.loads(line)
    return None


def collect_v196_anti_regression(v196_events: Path, repeats: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for event in read_csv(v196_events):
        if event.get("event") != "wrong":
            continue
        row = read_source_row(event.get("source_file", ""), event.get("source_line", ""))
        if row is None or not has_messages_and_answer(row):
            stats["missing_source_row"] += 1
            continue
        family = canonical_family(row.get("family") or event.get("family"))
        stats[f"wrong_{family}"] += 1
        for idx in range(repeats):
            rows.append(
                clone_row(
                    row,
                    source="v198_v196_wrong_anti_regression",
                    role="anti_regression",
                    suffix=f"anti{idx}_{event.get('method','method')}",
                    extra_metadata={
                        "v198_v196_method": event.get("method"),
                        "v198_wrong_prediction": event.get("prediction"),
                        "v198_expected_answer": event.get("answer"),
                    },
                )
            )
    stats["rows_after_repeat"] = len(rows)
    return rows, dict(stats)


def collect_v197_positive_distill(v197_events: Path, repeats: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for event in read_csv(v197_events):
        if event.get("event") != "correct":
            continue
        source_file = event.get("source_file", "")
        source_name = source_file.replace("\\", "/")
        # Keep local validation clean. V197 validation wins are evidence, not training rows.
        if "/v195_focal_val.jsonl" in source_name or "/v90_val_gold_safe_stratified.jsonl" in source_name:
            stats["dropped_validation_anchor"] += 1
            continue
        if "/v94_equation_crypt_val.jsonl" in source_name or "/v93a_delta_val.jsonl" in source_name:
            stats["dropped_validation_like_anchor"] += 1
            continue
        key = (source_file, event.get("source_line", ""))
        if key in seen:
            stats["dropped_duplicate_source"] += 1
            continue
        seen.add(key)
        row = read_source_row(source_file, event.get("source_line", ""))
        if row is None or not has_messages_and_answer(row):
            stats["missing_source_row"] += 1
            continue
        family = canonical_family(row.get("family") or event.get("family"))
        stats[f"positive_{family}"] += 1
        for idx in range(repeats):
            rows.append(
                clone_row(
                    row,
                    source="v198_v197_strict_gain_distill",
                    role="strict_gain_distill",
                    suffix=f"gain{idx}_{event.get('method','method')}",
                    extra_metadata={
                        "v198_v197_method": event.get("method"),
                        "v198_base_method": event.get("base_method"),
                        "v198_prediction": event.get("prediction"),
                        "v198_expected_answer": event.get("answer"),
                    },
                )
            )
    stats["rows_after_repeat"] = len(rows)
    return rows, dict(stats)


def make_notebook(
    path: Path,
    *,
    train_sha256: str,
    val_sha256: str,
    min_train_examples: int,
) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# KG1 V198 micro-distillation Colab Pro run\n",
                "\n",
                "Short continuation after V195. It trains from the best adapter found in Drive and never submits to Kaggle automatically.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import importlib.util, os, pathlib, shutil, subprocess, sys, urllib.request, zipfile, hashlib\n",
                "ROOT = pathlib.Path('/content/kg1_v198')\n",
                "DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V198')\n",
                "PACK = DRIVE_ROOT / 'kg1_v198_colab_pack.zip'\n",
                "PACK_URL = 'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/5bea4f0/runs/v198_micro_distill_colab_pack_20260503/kg1_v198_colab_pack.zip'\n",
                "PACK_SHA256 = '33ca771ab0c4559a671d2781d8a063220d048b5ac02630a825f69c2c9d6d9012'\n",
                "OUT = DRIVE_ROOT / 'output_v198'\n",
                "BASELINE_DIR = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V195/init_adapter/final')\n",
                "V195_OUT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V195/output_v195')\n",
                "BASE_ADAPTER_MODEL_SHA256 = '3d16ba908a5c8808624f1abd8fdc2b29f92723f5c874761161c894d7e5759f21'\n",
                "BASE_ADAPTER_CONFIG_SHA256 = 'e5499f128fde60d32d0595d427e4fe84d8abe6dbde1d80886c970e8184e4b743'\n",
                "\n",
                "def sha256_path(path):\n",
                "    h = hashlib.sha256()\n",
                "    with open(path, 'rb') as f:\n",
                "        for chunk in iter(lambda: f.read(1024 * 1024), b''):\n",
                "            h.update(chunk)\n",
                "    return h.hexdigest()\n",
                "\n",
                "def adapter_ready(path):\n",
                "    cfg = path / 'adapter_config.json'\n",
                "    model = path / 'adapter_model.safetensors'\n",
                "    if not cfg.exists() or not model.exists():\n",
                "        return False\n",
                "    if cfg.stat().st_size < 100 or model.stat().st_size < 1024:\n",
                "        return False\n",
                "    try:\n",
                "        import json\n",
                "        json.loads(cfg.read_text(encoding='utf-8'))\n",
                "    except Exception:\n",
                "        return False\n",
                "    return True\n",
                "\n",
                "def ensure_baseline_adapter():\n",
                "    BASELINE_DIR.mkdir(parents=True, exist_ok=True)\n",
                "    if adapter_ready(BASELINE_DIR):\n",
                "        cfg_ok = sha256_path(BASELINE_DIR / 'adapter_config.json') == BASE_ADAPTER_CONFIG_SHA256\n",
                "        model_ok = sha256_path(BASELINE_DIR / 'adapter_model.safetensors') == BASE_ADAPTER_MODEL_SHA256\n",
                "        if cfg_ok and model_ok:\n",
                "            return BASELINE_DIR\n",
                "        print('Existing baseline adapter has SHA mismatch; deleting and redownloading fallback.')\n",
                "        for p in [BASELINE_DIR / 'adapter_config.json', BASELINE_DIR / 'adapter_model.safetensors']:\n",
                "            if p.exists():\n",
                "                p.unlink()\n",
                "    if importlib.util.find_spec('kagglehub') is None:\n",
                "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])\n",
                "    import kagglehub\n",
                "    print('Downloading public 0.86 baseline adapter to Drive fallback...')\n",
                "    kagglehub.dataset_download('aaitdads/my-0p86-adapter', path='adapter_config.json', output_dir=str(BASELINE_DIR), force_download=True)\n",
                "    kagglehub.dataset_download('aaitdads/my-0p86-adapter', path='adapter_model.safetensors', output_dir=str(BASELINE_DIR), force_download=True)\n",
                "    assert adapter_ready(BASELINE_DIR), f'Missing baseline adapter files in {BASELINE_DIR}'\n",
                "    assert sha256_path(BASELINE_DIR / 'adapter_config.json') == BASE_ADAPTER_CONFIG_SHA256\n",
                "    assert sha256_path(BASELINE_DIR / 'adapter_model.safetensors') == BASE_ADAPTER_MODEL_SHA256\n",
                "    return BASELINE_DIR\n",
                "\n",
                "DRIVE_ROOT.mkdir(parents=True, exist_ok=True)\n",
                "if PACK.exists() and sha256_path(PACK) != PACK_SHA256:\n",
                "    print('Existing Drive pack SHA mismatch; deleting stale pack and downloading the verified one.')\n",
                "    PACK.unlink()\n",
                "if not PACK.exists():\n",
                "    print('Pack not found in Drive; trying GitHub URL...')\n",
                "    try:\n",
                "        urllib.request.urlretrieve(PACK_URL, PACK)\n",
                "    except Exception as exc:\n",
                "        raise RuntimeError(f'Pack missing. Upload kg1_v198_colab_pack.zip to {PACK} or push the branch so PACK_URL is valid: {PACK_URL}') from exc\n",
                "pack_hash = sha256_path(PACK)\n",
                "print('Pack SHA256:', pack_hash)\n",
                "assert pack_hash == PACK_SHA256, f'Pack SHA mismatch: {pack_hash}'\n",
                "shutil.rmtree(ROOT, ignore_errors=True)\n",
                "ROOT.mkdir(parents=True, exist_ok=True)\n",
                "with zipfile.ZipFile(PACK) as zf:\n",
                "    zf.extractall(ROOT)\n",
                "assert (ROOT / 'data/v198/v198_micro_train.strict.jsonl').exists()\n",
                "assert (ROOT / 'data/v198/v198_micro_val.strict.jsonl').exists()\n",
                "assert (ROOT / 'scripts/hf_job_train_v90.py').exists()\n",
                "\n",
                "candidates = [\n",
                "    V195_OUT / 'final_adapter',\n",
                "    V195_OUT / 'checkpoint-110',\n",
                "    V195_OUT / 'checkpoint-75',\n",
                "    V195_OUT / 'checkpoint-55',\n",
                "]\n",
                "INIT_ADAPTER = next((p for p in candidates if adapter_ready(p)), None)\n",
                "if INIT_ADAPTER is None:\n",
                "    print('No V195 adapter/checkpoint found; falling back to 0.86 baseline adapter.')\n",
                "    INIT_ADAPTER = ensure_baseline_adapter()\n",
                "print('INIT_ADAPTER =', INIT_ADAPTER)\n",
                "print('Pack extracted to', ROOT)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content/kg1_v198\n",
                "import importlib.util, os, subprocess, sys\n",
                "os.environ.setdefault('MAX_JOBS', '4')\n",
                "os.environ.setdefault('PIP_ROOT_USER_ACTION', 'ignore')\n",
                "\n",
                "def pip_install(args):\n",
                "    print('+ pip install', ' '.join(args))\n",
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])\n",
                "\n",
                "def pip_uninstall(package_name):\n",
                "    print('+ pip uninstall -y', package_name)\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name], check=False)\n",
                "\n",
                "def install_if_missing(module_name, args):\n",
                "    if importlib.util.find_spec(module_name) is None:\n",
                "        pip_install(args)\n",
                "    else:\n",
                "        print(f'{module_name} already installed')\n",
                "\n",
                "pip_uninstall('torchao')\n",
                "pip_install(['--upgrade', 'pip', 'setuptools', 'wheel', 'packaging', 'ninja==1.13.0'])\n",
                "pip_install(['transformers==5.7.0', 'accelerate==1.13.0', 'peft==0.19.1', 'datasets==4.8.5', 'safetensors==0.7.0', 'huggingface_hub==1.13.0', 'sentencepiece==0.2.1', 'protobuf==7.34.1'])\n",
                "install_if_missing('causal_conv1d', ['causal-conv1d==1.6.1', '--no-build-isolation'])\n",
                "install_if_missing('mamba_ssm', ['mamba-ssm==2.3.1', '--no-build-isolation'])\n",
                "assert importlib.util.find_spec('torchao') is None, 'torchao still installed; restart runtime and rerun cells from top'\n",
                "import causal_conv1d, mamba_ssm\n",
                "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn\n",
                "print('mamba_ssm OK:', getattr(mamba_ssm, '__version__', 'unknown'))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, shutil\n",
                "shutil.rmtree(OUT, ignore_errors=True)\n",
                "OUT.mkdir(parents=True, exist_ok=True)\n",
                "os.environ['UPLOAD_TO_HF'] = '0'\n",
                "os.environ['MODEL_NAME'] = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'\n",
                "os.environ['DATA_FILE'] = '/content/kg1_v198/data/v198/v198_micro_train.strict.jsonl'\n",
                "os.environ['VAL_FILE'] = '/content/kg1_v198/data/v198/v198_micro_val.strict.jsonl'\n",
                "os.environ['INIT_ADAPTER_DIR'] = str(INIT_ADAPTER)\n",
                "os.environ['INIT_ADAPTER_LOAD_MODE'] = 'manual'\n",
                "os.environ['PEFT_MANUAL_LOAD_METHOD'] = 'direct'\n",
                "os.environ['OUTPUT_DIR'] = str(OUT)\n",
                "os.environ['V198_OUT'] = str(OUT)\n",
                "os.environ['RUN_ID'] = 'v198-micro-distill-v197-gates'\n",
                "os.environ['MAX_LENGTH'] = '2048'\n",
                "os.environ['BATCH_SIZE'] = '16'\n",
                "os.environ['MICRO_BATCH_SIZE'] = '1'\n",
                "os.environ['GRADIENT_CHECKPOINTING'] = '1'\n",
                "os.environ['MAX_STEPS'] = '45'\n",
                "os.environ['SAVE_EVERY_STEPS'] = '15'\n",
                "os.environ['EVAL_EVERY_STEPS'] = '15'\n",
                "os.environ['EVAL_MAX_EXAMPLES'] = '240'\n",
                "os.environ['LEARNING_RATE'] = '1e-5'\n",
                "os.environ['FINAL_LEARNING_RATE'] = '3e-6'\n",
                f"os.environ['EXPECTED_TRAIN_SHA256'] = '{train_sha256}'\n",
                f"os.environ['EXPECTED_VAL_SHA256'] = '{val_sha256}'\n",
                f"os.environ['MIN_TRAIN_EXAMPLES'] = '{min_train_examples}'\n",
                "os.environ['MIN_TOKENIZED_TRAIN_EXAMPLES'] = '1600'\n",
                "os.environ['MIN_VAL_EXAMPLES'] = '720'\n",
                "os.environ['MIN_TOKENIZED_VAL_EXAMPLES'] = '700'\n",
                "os.environ['TRAINABLE_LORA_MODULES'] = 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'\n",
                "os.environ['MAX_TRAINABLE_PARAM_RATIO'] = '0.035'\n",
                "!python scripts/hf_job_train_v90.py\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Convert final/checkpoint-30 to Kaggle layout and run the post-training ZIP gate. This does not submit to Kaggle.\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pathlib, urllib.request\n",
                "POSTTRAIN_SCRIPT = pathlib.Path('/content/kg1_v198/scripts/kg1_v198_posttrain_gate.py')\n",
                "POSTTRAIN_SCRIPT_URL = 'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/cee9825b0edd6ea2e829c94bdd7b1ff9410b30f3/scripts/kg1_v198_posttrain_gate.py'\n",
                "if not POSTTRAIN_SCRIPT.exists():\n",
                "    print('Downloading V198 posttrain gate script...')\n",
                "    urllib.request.urlretrieve(POSTTRAIN_SCRIPT_URL, POSTTRAIN_SCRIPT)\n",
                "assert POSTTRAIN_SCRIPT.exists(), f'Missing posttrain gate script: {POSTTRAIN_SCRIPT}'\n",
                "!python scripts/kg1_v198_posttrain_gate.py --root /content/kg1_v198 --output-root \"$V198_OUT\" --fail-on-block\n",
            ],
        },
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_run_dir.mkdir(parents=True, exist_ok=True)

    base_rows_raw = read_jsonl(args.v195_strict_train)
    val_rows_raw = read_jsonl(args.v195_val)
    val_rows = [make_strict_eval_row(row) for row in val_rows_raw]
    base_rows, base_counts = sample_base_rows(base_rows_raw, targets=BASE_SAMPLE_TARGETS, rng=rng)
    anti_rows, anti_stats = collect_v196_anti_regression(args.v196_stress_events, args.anti_repeats)
    gain_rows, gain_stats = collect_v197_positive_distill(args.v197_stress_events, args.gain_repeats)

    train_rows = [*base_rows, *anti_rows, *gain_rows]
    ids_seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicate_id_drops = 0
    for idx, row in enumerate(train_rows):
        if not has_messages_and_answer(row):
            continue
        row["family"] = canonical_family(row.get("family") or row.get("subcategory"))
        rid = str(row.get("id") or f"v198_row_{idx}")
        if rid in ids_seen:
            duplicate_id_drops += 1
            row = clone_row(row, source=str(row.get("source") or "v198_duplicate_fixed"), role="duplicate_id_fixed", suffix=f"dupfix{idx}")
            rid = row["id"]
        ids_seen.add(rid)
        deduped.append(row)
    rng.shuffle(deduped)

    train_path = args.output_data_dir / "v198_micro_train.jsonl"
    clean_path = args.output_data_dir / "v198_micro_train.clean.jsonl"
    strict_path = args.output_data_dir / "v198_micro_train.strict.jsonl"
    raw_val_path = args.output_data_dir / "v198_micro_val.raw.jsonl"
    val_path = args.output_data_dir / "v198_micro_val.strict.jsonl"
    manifest_path = args.output_data_dir / "v198_micro_manifest.json"
    report_path = args.output_run_dir / "V198_MICRO_DATASET_REPORT.md"
    next_path = args.output_run_dir / "V198_NEXT_ACTIONS.md"
    pack_path = args.output_run_dir / "kg1_v198_colab_pack.zip"

    write_jsonl(train_path, deduped)
    write_jsonl(clean_path, deduped)
    write_jsonl(strict_path, deduped)
    write_jsonl(raw_val_path, val_rows_raw)
    write_jsonl(val_path, val_rows)

    train_sha = sha256_file(strict_path)
    val_sha = sha256_file(val_path)
    make_notebook(
        args.notebook_path,
        train_sha256=train_sha,
        val_sha256=val_sha,
        min_train_examples=len(deduped),
    )

    family_counts = Counter(row["family"] for row in deduped)
    source_counts = Counter(str(row.get("source") or "unknown") for row in deduped)
    role_counts = Counter(str((row.get("metadata") or {}).get("v198_source_role") or "unknown") for row in deduped)

    manifest = {
        "version": "v198_micro_distill_colab_pack",
        "seed": args.seed,
        "train_rows": len(deduped),
        "val_rows": len(val_rows),
        "train_path": str(train_path),
        "clean_train_path": str(clean_path),
        "strict_train_path": str(strict_path),
        "raw_val_path": str(raw_val_path),
        "val_path": str(val_path),
        "train_sha256": sha256_file(train_path),
        "clean_train_sha256": sha256_file(clean_path),
        "strict_train_sha256": train_sha,
        "raw_val_sha256": sha256_file(raw_val_path),
        "val_sha256": val_sha,
        "base_sample_counts": base_counts,
        "v196_anti_regression_stats": anti_stats,
        "v197_positive_distill_stats": gain_stats,
        "duplicate_id_drops": duplicate_id_drops,
        "train_family_counts": dict(sorted(family_counts.items())),
        "train_source_counts": dict(source_counts.most_common()),
        "train_role_counts": dict(role_counts.most_common()),
        "notes": [
            "No Kaggle submission is produced.",
            "V197 validation anchors are excluded from training to keep local validation meaningful.",
            "V196 wrong cases are repeated with unique IDs as anti-regression rehearsal.",
            "The notebook prefers V195 final_adapter/checkpoints, then falls back to the 0.86 baseline adapter.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# V198 micro-distillation dataset",
        "",
        f"Train rows: {len(deduped)}",
        f"Validation rows: {len(val_rows)}",
        "",
        "## Train families",
        "",
    ]
    for family, count in sorted(family_counts.items()):
        report_lines.append(f"- {family}: {count}")
    report_lines.extend(["", "## Sources", ""])
    for source, count in source_counts.most_common():
        report_lines.append(f"- {source}: {count}")
    report_lines.extend(["", "## Roles", ""])
    for role, count in role_counts.most_common():
        report_lines.append(f"- {role}: {count}")
    report_lines.extend(
        [
            "",
            "## Safety",
            "",
            "- V197 local validation anchors are not trained directly.",
            "- V196 wrong cases are used as anti-regression rows.",
            "- Stable families stay represented to reduce regression risk.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    next_lines = [
        "# V198 next actions",
        "",
        "Status: Colab pack generated locally. No Kaggle submission was created.",
        "",
        "## Run order",
        "",
        "1. Copy `kg1_v198_colab_pack.zip` to `/content/drive/MyDrive/KG1_NVIDIA_V198/` or push the notebook/pack branch.",
        "2. Run `notebooks/KG1_V198_MICRO_DISTILL_COLAB_PRO.ipynb` on Colab Pro H100.",
        "3. Prefer V195 `final_adapter`; if unavailable, checkpoint-110/75/55; fallback baseline is allowed but weaker.",
        "4. After conversion, run local inference/prescore before any Kaggle submit.",
        "",
        "## Stop rule",
        "",
        "- If eval loss at step 30/45 is worse than the V195 continuation trend, stop and keep V195.",
        "- If local validation has any stable-family regression, do not submit.",
    ]
    next_path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")

    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        pack_files = [
            train_path,
            clean_path,
            strict_path,
            val_path,
            raw_val_path,
            manifest_path,
            report_path,
            next_path,
            args.notebook_path,
            Path("scripts/hf_job_train_v90.py"),
            Path("scripts/kg1_training_data_gate.py"),
            Path("scripts/kg1_sft_format_validator.py"),
            Path("scripts/hf_convert_training_to_kaggle_layout.py"),
            Path("scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py"),
            Path("scripts/kg1_v198_posttrain_gate.py"),
        ]
        for path in pack_files:
            if path.exists():
                zf.write(path, path.as_posix())

    manifest["notebook_path"] = str(args.notebook_path)
    manifest["pack_path"] = str(pack_path)
    manifest["pack_sha256"] = sha256_file(pack_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    tmp_pack = pack_path.with_suffix(".tmp.zip")
    shutil.move(pack_path, tmp_pack)
    with zipfile.ZipFile(tmp_pack, "r") as zin, zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == manifest_path.as_posix():
                continue
            zout.writestr(item, zin.read(item.filename))
        zout.write(manifest_path, manifest_path.as_posix())
    tmp_pack.unlink()
    manifest["pack_sha256"] = sha256_file(pack_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v195-strict-train", type=Path, default=Path("data/v195/v195_focal_train.strict.jsonl"))
    parser.add_argument("--v195-val", type=Path, default=Path("data/v195/v195_focal_val.jsonl"))
    parser.add_argument(
        "--v196-stress-events",
        type=Path,
        default=Path("runs/v196_zero_wrong_selector_lab_20260503/v196_zero_wrong_stress_events.csv"),
    )
    parser.add_argument(
        "--v197-stress-events",
        type=Path,
        default=Path("runs/v197_strict_feature_gate_lab_20260503/v197_stress_events.csv"),
    )
    parser.add_argument("--output-data-dir", type=Path, default=Path("data/v198"))
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=Path("runs/v198_micro_distill_colab_pack_20260503"),
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        default=Path("notebooks/KG1_V198_MICRO_DISTILL_COLAB_PRO.ipynb"),
    )
    parser.add_argument("--seed", type=int, default=198)
    parser.add_argument("--anti-repeats", type=int, default=2)
    parser.add_argument("--gain-repeats", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
