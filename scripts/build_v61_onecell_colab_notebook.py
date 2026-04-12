#!/usr/bin/env python3
"""Build a one-cell KG1 v61 Colab Pro A100/H100 training notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from build_v60_colab_notebook import cells


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "KG1_v61_COLAB_PRO_A100_H100_TRAIN_ONECELL.ipynb"


def split_source(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def make_one_cell_source() -> str:
    code_parts: list[str] = [
        r'''
# KG1 v61 ONE-CELL COLAB TRAINING
#
# Rode esta unica celula no Colab Pro/Pro+ com A100/H100.
#
# O fluxo faz:
# 1. preflight de GPU
# 2. instalacao de dependencias
# 3. autenticacao Hugging Face
# 4. download do dataset boxed; se HF retornar 404, abre upload manual
# 5. gate de dataset/parser
# 6. load do Nemotron + LoRA/QLoRA
# 7. treino
# 8. upload adapter-only para o Hugging Face
#
# Dataset local esperado para upload manual, se necessario:
# v60_selective_train_boxed_dedup.jsonl
# SHA256: bfe2421b917d761c6528c4aa7bcecc105e8067381fd13a08e066a75cb7b8ad8f
'''.strip()
    ]

    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        cell_id = cell.get("metadata", {}).get("id", f"cell_{index}")
        source = "".join(cell.get("source", []))
        code_parts.append(f"\n# ===== BEGIN {cell_id} =====\n{source}\n# ===== END {cell_id} =====")

    combined = "\n\n".join(code_parts)
    combined = combined.replace(
        "ALLOW_UPLOAD_LOCAL_DATASET = False  # use True se o arquivo ainda nao estiver no DATA_REPO",
        "ALLOW_UPLOAD_LOCAL_DATASET = True  # one-cell: se HF retornar 404, abre upload manual no Colab",
    )
    combined = combined.replace(
        'print("Falha ao baixar dataset do HF:", type(exc).__name__, str(exc)[:300])',
        'print("Falha ao baixar dataset do HF:", type(exc).__name__, str(exc)[:300])\n'
        '    print("Como fallback, selecione o arquivo v60_selective_train_boxed_dedup.jsonl no upload do Colab.")',
    )
    combined = combined.replace("raise SystemExit(", "raise RuntimeError(")
    return combined


def main() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "kg1_v61_onecell_train"},
                "outputs": [],
                "source": split_source(make_one_cell_source()),
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
