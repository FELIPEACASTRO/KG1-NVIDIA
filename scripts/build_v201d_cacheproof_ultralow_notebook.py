#!/usr/bin/env python3
"""Build a cache-proof V201D notebook from the hardened V201C ultra-low plan."""

from __future__ import annotations

import json
from pathlib import Path

from build_v201c_multicandidate_notebook import build_v201c_notebook


NOTEBOOK_PATH = Path("notebooks/KG1_V201D_H100_A100_CACHEPROOF_ULTRALOW_COLAB_PRO.ipynb")
REPORT_PATH = Path("runs/v200_rank_hillclimb_20260504/V201D_CACHEPROOF_ULTRALOW_NEXT_ACTIONS.md")
VERSION = "V201D_CACHEPROOF_ULTRALOW_20260504"
EXPECTED_LABELS = [
    "A_ultralow_shuffle_1s",
    "B_equation_crypt_ultralow_1s",
    "C_bit_cipher_ultralow_1s",
]


def all_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source") or []) for cell in notebook.get("cells", []))


def replace_required_text(source: str, old: str, new: str) -> str:
    if old not in source:
        raise RuntimeError(f"Missing expected source fragment: {old!r}")
    return source.replace(old, new)


def build_v201d_notebook() -> dict:
    notebook = build_v201c_notebook()
    notebook["metadata"]["colab"]["name"] = NOTEBOOK_PATH.name
    notebook["cells"][0]["source"] = (
        "# KG1 V201D H100/A100 cache-proof ultra-low candidate notebook\n\n"
        "This notebook intentionally uses a new filename to avoid stale Colab tabs from the old V201C. "
        "It runs only ultra-low 1-step candidates from exact V194 rank-19 and prints a version marker "
        "before training. If the log shows `A_neutral_shuffle_3s`, stop immediately because you are not "
        "running this notebook.\n"
    ).splitlines(True)

    replacements = {
        "V201C_OUT =": "V201D_OUT =",
        "V201C_OUT": "V201D_OUT",
        "V201C_CANDIDATE_OUT": "V201D_CANDIDATE_OUT",
        "V201C candidate summary": "V201D candidate summary",
        "V201C final selection": "V201D final selection",
        "V201C candidates": "V201D candidates",
        "V201C production training": "V201D production training",
        "V201C, got": "V201D, got",
        "Starting V201C candidate": "Starting V201D candidate",
        "output_v201c_h100_a100_multicandidate_3x": "output_v201d_cacheproof_ultralow_3x",
        "v201c_candidates_summary.json": "v201d_candidates_summary.json",
        "v201c_final_selection.json": "v201d_final_selection.json",
    }
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source") or [])
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell["source"] = source.splitlines(True)

    train_source = "".join(notebook["cells"][6]["source"])
    train_source = replace_required_text(
        train_source,
        "import datetime, json, os, pathlib, re, subprocess, sys, urllib.request\n",
        (
            "import datetime, json, os, pathlib, re, subprocess, sys, urllib.request\n"
            f"NOTEBOOK_VERSION = '{VERSION}'\n"
            "print('NOTEBOOK_VERSION =', NOTEBOOK_VERSION)\n"
        ),
    )
    train_source = replace_required_text(
        train_source,
        "]\n\n"
        "def parse_metrics(log_text):\n",
        "]\n\n"
        "EXPECTED_CANDIDATE_LABELS = [\n"
        "    'A_ultralow_shuffle_1s',\n"
        "    'B_equation_crypt_ultralow_1s',\n"
        "    'C_bit_cipher_ultralow_1s',\n"
        "]\n"
        "actual_candidate_labels = [item['label'] for item in CANDIDATES]\n"
        "print('Candidate labels:', actual_candidate_labels)\n"
        "assert actual_candidate_labels == EXPECTED_CANDIDATE_LABELS, actual_candidate_labels\n"
        "assert 'A_neutral_shuffle_3s' not in actual_candidate_labels, actual_candidate_labels\n\n"
        "def parse_metrics(log_text):\n",
    )
    notebook["cells"][6]["source"] = train_source.splitlines(True)

    source = all_source(notebook)
    required = [
        VERSION,
        "KG1 V201D H100/A100 cache-proof ultra-low candidate notebook",
        "OUT_BASE = DRIVE_ROOT / 'output_v201d_cacheproof_ultralow_3x'",
        "V201D_OUT",
        "A_ultralow_shuffle_1s",
        "B_equation_crypt_ultralow_1s",
        "C_bit_cipher_ultralow_1s",
        "assert 'A_neutral_shuffle_3s' not in actual_candidate_labels",
        "'EVAL_MAX_EXAMPLES': '720'",
        "'MODEL_REVISION': 'cbd3fa9f933d55ef16a84236559f4ee2a0526848'",
        "No Kaggle submit was performed.",
    ]
    forbidden = [
        "Starting V201C candidate",
        "V201C_OUT",
        "output_v201c_h100_a100_multicandidate_3x",
        "A_neutral_shuffle_3s',",
        "v201c_candidates_summary.json",
        "v201c_final_selection.json",
        "kaggle competitions submit",
        "files.upload",
        "KaggleApi",
    ]
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Built V201D notebook is missing {fragment!r}")
    for fragment in forbidden:
        if fragment in source:
            raise RuntimeError(f"Built V201D notebook contains forbidden fragment {fragment!r}")
    return notebook


def main() -> int:
    notebook = build_v201d_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# V201D cache-proof ultra-low notebook\n\n"
        f"- Version marker: `{VERSION}`.\n"
        f"- Notebook: `{NOTEBOOK_PATH}`.\n"
        "- New filename prevents stale V201C Colab tabs from running the old `A_neutral_shuffle_3s` candidate.\n"
        "- Candidate labels must be exactly `A_ultralow_shuffle_1s`, `B_equation_crypt_ultralow_1s`, `C_bit_cipher_ultralow_1s`.\n"
        "- If logs show `A_neutral_shuffle_3s`, stop immediately.\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
