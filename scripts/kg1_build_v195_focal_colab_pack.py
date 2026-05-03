#!/usr/bin/env python3
"""Build a V195 focal SFT dataset and Colab pack.

This pack is deliberately submit-safe only after a new adapter is trained and
passes the usual local/Kaggle layout gates.  It does not create a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANONICAL_FAMILIES = {
    "gravity_constant",
    "unit_conversion",
    "numeral_system",
    "text_encryption",
    "bit_manipulation",
    "equation_transform",
}


STABLE_REHEARSAL_TARGETS = {
    "gravity_constant": 260,
    "unit_conversion": 260,
    "numeral_system": 260,
    "text_encryption": 260,
}


def canonical_family(raw: Any) -> str:
    family = str(raw or "").strip().lower()
    family = family.replace("-", "_").replace(" ", "_")
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
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
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
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_identity(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    answer = str(row.get("answer") or "")
    if prompt:
        return hashlib.sha256((prompt.strip() + "\n" + answer.strip()).encode("utf-8")).hexdigest()
    return str(row.get("id") or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())


def origin_ids(row: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("id", "original_id", "origin_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            ids.add(value)
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in ("id", "original_id", "origin_id"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def normalize_row(
    row: dict[str, Any],
    *,
    source_tag: str,
    role: str,
) -> dict[str, Any]:
    out = dict(row)
    out["family"] = canonical_family(out.get("family") or out.get("subcategory"))
    out["source"] = source_tag
    out.setdefault("id", f"{source_tag}:{row_identity(row)[:16]}")
    metadata = dict(out.get("metadata") or {})
    metadata["v195_source_role"] = role
    metadata["v195_original_source"] = row.get("source")
    metadata["v195_canonicalized_family"] = out["family"]
    out["metadata"] = metadata
    return out


def has_messages_and_answer(row: dict[str, Any]) -> bool:
    if not str(row.get("prompt") or "").strip():
        return False
    if not str(row.get("answer") or "").strip():
        return False
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    return True


def dedup_append(
    output: list[dict[str, Any]],
    seen: set[str],
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> int:
    added = 0
    for row in rows:
        key = row_identity(row)
        if key in seen:
            continue
        if not has_messages_and_answer(row):
            continue
        seen.add(key)
        output.append(row)
        added += 1
        if limit is not None and added >= limit:
            break
    return added


def exclude_validation_overlap(rows: list[dict[str, Any]], val_ids: set[str]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        if origin_ids(row) & val_ids:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def sample_by_family(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[canonical_family(row.get("family"))].append(row)

    sampled: list[dict[str, Any]] = []
    for family, target in targets.items():
        bucket = buckets.get(family, [])
        rng.shuffle(bucket)
        sampled.extend(bucket[:target])
    return sampled


def make_notebook(path: Path) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# KG1 V195 focal Colab Pro run\n",
                "\n",
                "Treino curto e controlado para tentar transformar ganhos offline verificados em adapter submetivel. Nao submete no Kaggle automaticamente.\n",
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
                "import os, zipfile, pathlib, shutil, urllib.request, hashlib, sys, subprocess, importlib.util\n",
                "ROOT = pathlib.Path('/content/kg1_v195')\n",
                "PACK = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V195/kg1_v195_colab_pack.zip')\n",
                "PACK_URL = 'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir/runs/v195_focal_colab_pack_20260503/kg1_v195_colab_pack.zip'\n",
                "PACK_SHA256 = 'ac82a921977ecb6de20e40cdc060284361d4cc1f7ca66a86a1cc0f69055b55a3'\n",
                "BASE_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V195/init_adapter/final')\n",
                "BASE_ADAPTER_MODEL_SHA256 = '3d16ba908a5c8808624f1abd8fdc2b29f92723f5c874761161c894d7e5759f21'\n",
                "BASE_ADAPTER_CONFIG_SHA256 = 'e5499f128fde60d32d0595d427e4fe84d8abe6dbde1d80886c970e8184e4b743'\n",
                "OUT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V195/output_v195')\n",
                "\n",
                "def sha256_path(path):\n",
                "    h = hashlib.sha256()\n",
                "    with open(path, 'rb') as f:\n",
                "        for chunk in iter(lambda: f.read(1024 * 1024), b''):\n",
                "            h.update(chunk)\n",
                "    return h.hexdigest()\n",
                "\n",
                "PACK.parent.mkdir(parents=True, exist_ok=True)\n",
                "if not PACK.exists():\n",
                "    print('Pack not found in Drive; downloading from GitHub...')\n",
                "    urllib.request.urlretrieve(PACK_URL, PACK)\n",
                "pack_hash = sha256_path(PACK)\n",
                "assert pack_hash == PACK_SHA256, f'Pack SHA mismatch: {pack_hash}'\n",
                "assert PACK.exists(), f'Missing pack: {PACK}'\n",
                "\n",
                "BASE_ADAPTER.mkdir(parents=True, exist_ok=True)\n",
                "adapter_model = BASE_ADAPTER / 'adapter_model.safetensors'\n",
                "adapter_config = BASE_ADAPTER / 'adapter_config.json'\n",
                "if not adapter_model.exists() or not adapter_config.exists():\n",
                "    print('Baseline adapter not found in Drive; downloading public Kaggle dataset aaitdads/my-0p86-adapter...')\n",
                "    if importlib.util.find_spec('kagglehub') is None:\n",
                "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])\n",
                "    import kagglehub\n",
                "    kagglehub.dataset_download('aaitdads/my-0p86-adapter', path='adapter_config.json', output_dir=str(BASE_ADAPTER), force_download=False)\n",
                "    kagglehub.dataset_download('aaitdads/my-0p86-adapter', path='adapter_model.safetensors', output_dir=str(BASE_ADAPTER), force_download=False)\n",
                "assert adapter_model.exists(), f'Missing {adapter_model}'\n",
                "assert adapter_config.exists(), f'Missing {adapter_config}'\n",
                "model_hash = sha256_path(adapter_model)\n",
                "config_hash = sha256_path(adapter_config)\n",
                "assert model_hash == BASE_ADAPTER_MODEL_SHA256, f'Baseline adapter_model SHA mismatch: {model_hash}'\n",
                "assert config_hash == BASE_ADAPTER_CONFIG_SHA256, f'Baseline adapter_config SHA mismatch: {config_hash}'\n",
                "print('Baseline adapter OK:', BASE_ADAPTER)\n",
                "shutil.rmtree(ROOT, ignore_errors=True)\n",
                "ROOT.mkdir(parents=True, exist_ok=True)\n",
                "with zipfile.ZipFile(PACK) as zf:\n",
                "    zf.extractall(ROOT)\n",
                "assert (ROOT / 'data/v195/v195_focal_train.strict.jsonl').exists()\n",
                "assert (ROOT / 'scripts/hf_job_train_v90.py').exists()\n",
                "assert (ROOT / 'scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py').exists()\n",
                "print('Pack extracted to', ROOT)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content/kg1_v195\n",
                "!pip install -q --upgrade pip\n",
                "!pip install -q torch transformers accelerate peft datasets safetensors huggingface_hub sentencepiece protobuf\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, pathlib\n",
                "os.environ['UPLOAD_TO_HF'] = '0'\n",
                "os.environ['MODEL_NAME'] = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'\n",
                "os.environ['DATA_FILE'] = '/content/kg1_v195/data/v195/v195_focal_train.strict.jsonl'\n",
                "os.environ['VAL_FILE'] = '/content/kg1_v195/data/v195/v195_focal_val.jsonl'\n",
                "os.environ['INIT_ADAPTER_DIR'] = str(BASE_ADAPTER)\n",
                "os.environ['INIT_ADAPTER_LOAD_MODE'] = 'manual'\n",
                "os.environ['OUTPUT_DIR'] = str(OUT)\n",
                "os.environ['V195_OUT'] = str(OUT)\n",
                "os.environ['RUN_ID'] = 'v195-focal-short-aaitdads86'\n",
                "os.environ['MAX_LENGTH'] = '2048'\n",
                "os.environ['BATCH_SIZE'] = '16'\n",
                "os.environ['MICRO_BATCH_SIZE'] = '1'\n",
                "os.environ['GRADIENT_CHECKPOINTING'] = '1'\n",
                "os.environ['MAX_STEPS'] = '110'\n",
                "os.environ['SAVE_EVERY_STEPS'] = '55'\n",
                "os.environ['EVAL_EVERY_STEPS'] = '25'\n",
                "os.environ['EVAL_MAX_EXAMPLES'] = '160'\n",
                "os.environ['LEARNING_RATE'] = '4e-5'\n",
                "os.environ['FINAL_LEARNING_RATE'] = '1e-5'\n",
                "os.environ['EXPECTED_TRAIN_SHA256'] = '8a75affddb2176c4ef46973fb7fdf2389007066a1bb96eb4fae0a7d3c9abed2b'\n",
                "os.environ['EXPECTED_VAL_SHA256'] = 'fe5530f0252cd47992eb983d063e8de90135b5def592d15daac603d985f26cad'\n",
                "os.environ['MIN_TRAIN_EXAMPLES'] = '3798'\n",
                "os.environ['MIN_TOKENIZED_TRAIN_EXAMPLES'] = '3700'\n",
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
                "Converter o adapter treinado para layout Kaggle. Isto nao submete no Kaggle.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py \\\n",
                "  --source-adapter-dir \"$V195_OUT/final_adapter\" \\\n",
                "  --output-dir \"$V195_OUT/kaggle_layout\" \\\n",
                "  --run-id v195-focal-short-aaitdads86\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Depois do treino, validar `output_v195/kaggle_layout/zip/v195-focal-short-aaitdads86_adapter_only.zip` localmente antes de qualquer submissao. O submit so deve seguir se superar o baseline em validacao local e passar gate de regressao.\n",
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
    out_data_dir = args.output_data_dir
    out_run_dir = args.output_run_dir
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_run_dir.mkdir(parents=True, exist_ok=True)

    val_rows_raw = read_jsonl(args.val_file)
    val_ids = {str(row.get("id")) for row in val_rows_raw if row.get("id")}
    val_rows = [
        normalize_row(row, source_tag="v195_val_gold_safe", role="validation_holdout")
        for row in val_rows_raw
    ]

    official_raw = read_jsonl(args.official_selector_train)
    v94_raw = read_jsonl(args.v94_train)
    v95_raw = read_jsonl(args.v95_train)
    v90_raw = read_jsonl(args.v90_full)

    raw_sets = {
        "official_selector_solver": [
            normalize_row(row, source_tag="v195_official_selector_solver", role="focal_solver_trace")
            for row in official_raw
        ],
        "v94_equation_crypt": [
            normalize_row(row, source_tag="v195_v94_equation_crypt", role="focal_equation_crypt")
            for row in v94_raw
        ],
        "v95_bit_rehearsal": [
            normalize_row(row, source_tag="v195_v95_bit_rehearsal", role="bit_rehearsal")
            for row in v95_raw
        ],
        "v90_gold_safe": [
            normalize_row(row, source_tag="v195_v90_gold_safe_rehearsal", role="stable_rehearsal")
            for row in v90_raw
        ],
    }

    overlap_drops: dict[str, int] = {}
    filtered_sets: dict[str, list[dict[str, Any]]] = {}
    for name, rows in raw_sets.items():
        kept, dropped = exclude_validation_overlap(rows, val_ids)
        filtered_sets[name] = kept
        overlap_drops[name] = dropped

    train_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    added_by_stage: dict[str, int] = {}

    added_by_stage["official_selector_solver"] = dedup_append(
        train_rows,
        seen,
        filtered_sets["official_selector_solver"],
    )

    v94_equation = [
        row for row in filtered_sets["v94_equation_crypt"] if row["family"] == "equation_transform"
    ]
    added_by_stage["v94_equation_transform"] = dedup_append(train_rows, seen, v94_equation)

    v95_bit = [
        row for row in filtered_sets["v95_bit_rehearsal"] if row["family"] == "bit_manipulation"
    ]
    rng.shuffle(v95_bit)
    added_by_stage["v95_bit_manipulation_cap"] = dedup_append(
        train_rows,
        seen,
        v95_bit,
        limit=args.bit_cap,
    )

    v90_rehearsal = sample_by_family(filtered_sets["v90_gold_safe"], STABLE_REHEARSAL_TARGETS, rng)
    added_by_stage["v90_stable_rehearsal"] = dedup_append(train_rows, seen, v90_rehearsal)

    # Final shuffle after priority insertion, but with deterministic seed.
    rng.shuffle(train_rows)

    train_path = out_data_dir / "v195_focal_train.jsonl"
    val_path = out_data_dir / "v195_focal_val.jsonl"
    manifest_path = out_data_dir / "v195_focal_manifest.json"
    report_path = out_run_dir / "V195_FOCAL_DATASET_REPORT.md"
    notebook_path = args.notebook_path
    pack_path = out_run_dir / "kg1_v195_colab_pack.zip"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    family_counts = Counter(row["family"] for row in train_rows)
    source_counts = Counter(row["source"] for row in train_rows)
    manifest = {
        "version": "v195_focal_colab_pack",
        "seed": args.seed,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_family_counts": dict(sorted(family_counts.items())),
        "train_source_counts": dict(sorted(source_counts.items())),
        "added_by_stage": added_by_stage,
        "validation_overlap_drops": overlap_drops,
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "notes": [
            "No Kaggle submission is produced by this builder.",
            "Subfamilies equation_numeric_* and cryptarithm_* are canonicalized to equation_transform.",
            "Rows overlapping v90_val_gold_safe_stratified IDs are excluded from training sources.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report = [
        "# V195 focal Colab dataset",
        "",
        f"Train rows: {len(train_rows)}",
        f"Validation rows: {len(val_rows)}",
        "",
        "## Train families",
        "",
    ]
    for family, count in sorted(family_counts.items()):
        report.append(f"- {family}: {count}")
    report.extend(["", "## Sources", ""])
    for source, count in sorted(source_counts.items()):
        report.append(f"- {source}: {count}")
    report.extend(["", "## Stage additions", ""])
    for stage, count in added_by_stage.items():
        report.append(f"- {stage}: {count}")
    report.extend(["", "## Validation overlap drops", ""])
    for name, count in overlap_drops.items():
        report.append(f"- {name}: {count}")
    report.extend(
        [
            "",
            "## Gate",
            "",
            "Training notebook uses v195_focal_train.strict.jsonl.",
            "Regenerate the pack after running kg1_training_data_gate.py and filtering format issues.",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    make_notebook(notebook_path)

    clean_path = out_data_dir / "v195_focal_train.clean.jsonl"
    strict_path = out_data_dir / "v195_focal_train.strict.jsonl"
    if clean_path.exists():
        manifest["clean_train_path"] = str(clean_path)
        manifest["clean_train_rows"] = sum(1 for _ in clean_path.open("r", encoding="utf-8"))
        manifest["clean_train_sha256"] = sha256_file(clean_path)
    if strict_path.exists():
        manifest["strict_train_path"] = str(strict_path)
        manifest["strict_train_rows"] = sum(1 for _ in strict_path.open("r", encoding="utf-8"))
        manifest["strict_train_sha256"] = sha256_file(strict_path)

    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        pack_files = [
            train_path,
            clean_path,
            strict_path,
            val_path,
            manifest_path,
            report_path,
            out_run_dir / "V195_NEXT_ACTIONS.md",
            notebook_path,
            Path("scripts/hf_job_train_v90.py"),
            Path("scripts/kg1_training_data_gate.py"),
            Path("scripts/kg1_sft_format_validator.py"),
            Path("scripts/hf_convert_training_to_kaggle_layout.py"),
            Path("scripts/kg1_convert_local_training_adapter_to_kaggle_zip.py"),
        ]
        for path in pack_files:
            if path.exists():
                zf.write(path, path.as_posix())
    manifest["notebook_path"] = str(notebook_path)
    manifest["pack_path"] = str(pack_path)
    manifest["pack_sha256"] = sha256_file(pack_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    # Refresh manifest inside the zip with the final pack hash.
    tmp_pack = pack_path.with_suffix(".tmp.zip")
    shutil.move(pack_path, tmp_pack)
    with zipfile.ZipFile(tmp_pack, "r") as zin, zipfile.ZipFile(
        pack_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
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
    parser.add_argument("--v90-full", type=Path, default=Path("data/v90/v90_gold_safe_full.clean.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/v90/v90_val_gold_safe_stratified.jsonl"))
    parser.add_argument("--v94-train", type=Path, default=Path("data/v94/v94_equation_crypt_train.jsonl"))
    parser.add_argument("--v95-train", type=Path, default=Path("data/v95/v95_bit_rehearsal_train.jsonl"))
    parser.add_argument(
        "--official-selector-train",
        type=Path,
        default=Path("runs/official_metric_solver_selector_lab_20260430/official_selector_solver_train_final_v9.jsonl"),
    )
    parser.add_argument("--output-data-dir", type=Path, default=Path("data/v195"))
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=Path("runs/v195_focal_colab_pack_20260503"),
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        default=Path("notebooks/KG1_V195_FOCAL_COLAB_PRO.ipynb"),
    )
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--bit-cap", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
