#!/usr/bin/env python3
"""Build the V201B baseline-neutral micro-train Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_v201_087_notebook import all_source, build_v201a_notebook, replace_required


NOTEBOOK_PATH = Path("notebooks/KG1_V201B_H100_BASELINE_NEUTRAL_MICRO_COLAB_PRO.ipynb")
REPORT_PATH = Path("runs/v200_rank_hillclimb_20260504/V201B_H100_BASELINE_NEUTRAL_MICRO_NEXT_ACTIONS.md")


def build_v201b_notebook() -> dict:
    notebook = build_v201a_notebook()
    notebook["metadata"]["colab"]["name"] = NOTEBOOK_PATH.name
    notebook["cells"][0]["source"] = (
        "# KG1 V201B H100/A100 baseline-neutral micro-train\n\n"
        "This notebook is the next roadmap step after the V201A gate blocked a local eval regression. "
        "It starts again from the exact V194 rank-19 / public 0.86 adapter, removes aggressive weighted "
        "replacement sampling, trains only attention LoRA modules for 3 steps at a lower LR, evaluates "
        "the V194 baseline before training, blocks any final eval regression, converts the candidate, "
        "and never submits to Kaggle automatically.\n"
    ).splitlines(True)

    replacements = [
        ("output_v201a_h100_solver_verified_micro_5", "output_v201b_h100_baseline_neutral_micro_3"),
        ("v201a-h100-v194-rank19-solver-verified-micro-5s", "v201b-h100-v194-rank19-baseline-neutral-micro-3s"),
        ("V201A", "V201B"),
        ("v201a", "v201b"),
        ("MAX_STEPS'] = '5'", "MAX_STEPS'] = '3'"),
        ("SAVE_EVERY_STEPS'] = '5'", "SAVE_EVERY_STEPS'] = '3'"),
        ("EVAL_EVERY_STEPS'] = '5'", "EVAL_EVERY_STEPS'] = '3'"),
        ("LEARNING_RATE'] = '3e-7'", "LEARNING_RATE'] = '2e-7'"),
        ("ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA'] = '0.005'", "ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA'] = '0.003'"),
        ("SAMPLING_MODE'] = 'weighted_replacement'", "SAMPLING_MODE'] = 'shuffle'"),
        (
            "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation=2.5,cipher=2.0,cryptarithm_deduce=3.0,cryptarithm_guess=2.0,equation_numeric_deduce=3.0,equation_numeric_guess=2.0,equation_transform=1.5'",
            "SUBCATEGORY_WEIGHTS'] = ''",
        ),
        (
            "SOURCE_WEIGHTS'] = 'v198_v196_wrong_anti_regression=2.0,v198_v197_strict_gain_distill=1.5,v198_v195_balanced_rehearsal=1.0'",
            "SOURCE_WEIGHTS'] = ''",
        ),
    ]
    for old, new in replacements:
        replace_required(notebook, old, new)

    source = all_source(notebook)
    required = [
        "KG1 V201B H100/A100 baseline-neutral micro-train",
        "V194_RANK19_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'",
        "V194_RANK19_BOOTSTRAP_TARGET = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V201/baseline_v194_rank19/submission.zip')",
        "pathlib.Path('/content/drive/MyDrive/Submit/submission.zip')",
        "pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199/baseline_v194_rank19/submission.zip')",
        "OUT_BASE = DRIVE_ROOT / 'output_v201b_h100_baseline_neutral_micro_3'",
        "RUN_ID'] = 'v201b-h100-v194-rank19-baseline-neutral-micro-3s'",
        "MAX_STEPS'] = '3'",
        "SAVE_EVERY_STEPS'] = '3'",
        "EVAL_EVERY_STEPS'] = '3'",
        "LEARNING_RATE'] = '2e-7'",
        "FINAL_LEARNING_RATE'] = '1e-7'",
        "SAMPLING_MODE'] = 'shuffle'",
        "SUBCATEGORY_WEIGHTS'] = ''",
        "SOURCE_WEIGHTS'] = ''",
        "BASELINE_EVAL_BEFORE_TRAIN'] = '1'",
        "REQUIRE_FINAL_EVAL_LTE_BASELINE'] = '1'",
        "MAX_FINAL_EVAL_REGRESSION'] = '0.0'",
        "TRAINABLE_LORA_MODULES'] = 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'",
        "kg1_v201b_posttrain_gate.py",
        "v201b_posttrain_gate_report.json",
        "V201B gated candidate ready. No Kaggle submit was performed.",
    ]
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Built V201B notebook is missing {fragment!r}")

    forbidden = [
        "--submit",
        "kaggle competitions submit",
        "KaggleApi",
        "files.upload",
        "kg1_v199b_safe_kaggle_submit.py",
        "V198_FINAL_ADAPTER_SHA256",
        "SAMPLING_MODE'] = 'weighted_replacement'",
        "bit_manipulation=2.5",
        "v198_v196_wrong_anti_regression=2.0",
        "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5",
        "SOURCE_WEIGHTS'] = 'v198_v196_wrong_anti_regression:2.0",
        "output_v201a_h100_solver_verified_micro_5",
        "v201a",
        "V201A",
        "MAX_STEPS'] = '20'",
        "MAX_STEPS'] = '10'",
        "MAX_STEPS'] = '5'",
        "LEARNING_RATE'] = '3e-6'",
        "LEARNING_RATE'] = '3e-7'",
    ]
    for fragment in forbidden:
        if fragment in source:
            raise RuntimeError(f"Built V201B notebook contains forbidden fragment {fragment!r}")
    return notebook


def main() -> int:
    notebook = build_v201b_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# V201B H100/A100 baseline-neutral micro-train\n\n"
        "- Notebook: `notebooks/KG1_V201B_H100_BASELINE_NEUTRAL_MICRO_COLAB_PRO.ipynb`.\n"
        "- V201A is blocked and must not be used as init because final eval regressed: `1.1222 > 1.1205`.\n"
        "- Production baseline remains V194/ref `52275052`, public score `0.86`, rank `19/2613`.\n"
        "- Starts only from exact V194 `submission.zip` SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.\n"
        "- Runs 3 steps at LR `2e-7 -> 1e-7` with normal shuffled sampling, no weighted replacement and no custom weight maps.\n"
        "- Trains attention LoRA modules only: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.\n"
        "- Evaluates V194 baseline before training and blocks final promotion unless `final_eval_loss <= baseline_eval_loss`.\n"
        "- Converts with `kg1_v201b_posttrain_gate.py`; no Kaggle submit cell is present.\n"
        "- Submit only if the posttrain gate is `READY`, local eval is non-regressive, and Kaggle authorization is explicit.\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
