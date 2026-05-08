#!/usr/bin/env python3
"""Strict release audit for the V218 decode-rescue Colab notebook."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


EXPECTED_TRAIN_SHA256 = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
EXPECTED_VAL_SHA256 = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"


def fail(message: str) -> None:
    raise SystemExit(f"V218 audit failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    all_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)

    require(len(code_cells) == 9, f"expected 9 code cells, found {len(code_cells)}")
    require(
        sum(len(cell.get("outputs", [])) for cell in code_cells) == 0,
        "notebook must be committed without execution outputs",
    )
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        ast.parse(source)
        require("# CELL:" in source, f"code cell {index} missing # CELL marker")
        require(re.search(r"=== V218 .* START ===", source) is not None, f"code cell {index} missing START marker")
        require(re.search(r"=== V218 .* END ===", source) is not None, f"code cell {index} missing END marker")

    required_snippets = {
        "repo clone branch": "'git', 'clone', '--depth', '1', '--branch', REPO_BRANCH",
        "repo sys.path insert": "sys.path.insert(0, str(ROOT))",
        "repo sys.path log": "repo_root_on_sys_path",
        "train sha constant": EXPECTED_TRAIN_SHA256,
        "val sha constant": EXPECTED_VAL_SHA256,
        "train sha gate": "observed_train_sha256 != EXPECTED_TRAIN_SHA256",
        "val sha gate": "observed_val_sha256 != EXPECTED_VAL_SHA256",
        "safetensors fallback": "pip_install_safetensors.log",
        "adapter tensor count": "tensor_count = len(handle.keys())",
        "v194 tensor gate": "V194 adapter tensor count mismatch",
        "v217 size gate": "V217 final adapter size mismatch",
        "submit lock false": "ALLOW_KAGGLE_SUBMIT = False",
        "submit lock guard": "Kaggle submission is disabled",
        "weak total gate": "WEAK_MIN_FOR_FULL = 193",
        "weak eq gate": "WEAK_EQ_MIN_FOR_FULL = 60",
        "weak bit gate": "WEAK_BIT_MIN_FOR_FULL = 133",
        "weak trunc gate": "WEAK_MAX_TRUNC_FOR_FULL = 3",
        "decode max tokens arg": "--max-tokens",
        "decode prompt suffix arg": "--prompt-suffix",
        "decode disable thinking arg": "--disable-thinking",
        "full eval blocked": "Full eval is intentionally not automatic in V218 diagnostic notebook",
    }
    for name, snippet in required_snippets.items():
        require(snippet in all_source, f"missing required snippet: {name}")

    print(
        json.dumps(
            {
                "ok": True,
                "notebook": str(path),
                "code_cells": len(code_cells),
                "schema": "kg1_v218_decode_rescue_audit_v1",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
