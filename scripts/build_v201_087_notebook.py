#!/usr/bin/env python3
"""Build the V201A H100 solver-verified micro-train Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_v199_conservative_colab_notebook import (  # noqa: E402
    BRANCH_SCRIPT_BASE,
    TRAIN_SCRIPT_URL,
    build_h100_highram_notebook,
)


NOTEBOOK_PATH = Path("notebooks/KG1_V201A_H100_SOLVER_VERIFIED_MICRO_COLAB_PRO.ipynb")
REPORT_PATH = Path("runs/v200_rank_hillclimb_20260504/V201A_H100_SOLVER_VERIFIED_MICRO_NEXT_ACTIONS.md")


def replace_required(notebook: dict, old: str, new: str) -> None:
    count = 0
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source") or [])
        if old in source:
            count += source.count(old)
            cell["source"] = source.replace(old, new).splitlines(True)
    if count == 0:
        raise RuntimeError(f"Required notebook fragment not found: {old!r}")


def all_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source") or []) for cell in notebook.get("cells", []))


def build_v201a_notebook() -> dict:
    notebook = build_h100_highram_notebook()
    notebook["metadata"]["colab"]["name"] = NOTEBOOK_PATH.name
    notebook["cells"][0]["source"] = (
        "# KG1 V201A H100 solver-verified weak-category micro-train\n\n"
        "This notebook is the next >=0.87 probe from the exact V194 rank-19 / public 0.86 adapter. "
        "It validates the V194 zip before training, evaluates the V194 baseline before any update, "
        "uses a weighted weak-category sample, trains only attention LoRA modules for 5 steps, "
        "blocks local eval regressions, converts the candidate, and never submits to Kaggle automatically.\n"
    ).splitlines(True)

    replace_required(
        notebook,
        "OUT_BASE = DRIVE_ROOT / 'output_v199_h100_20'",
        "OUT_BASE = DRIVE_ROOT / 'output_v201a_h100_solver_verified_micro_5'",
    )
    replace_required(notebook, "print('V199_OUT =', OUT)", "print('V201A_OUT =', OUT)")
    replace_required(
        notebook,
        "os.environ['V199_OUT'] = str(OUT)\n",
        "os.environ['V201A_OUT'] = str(OUT)\n"
        "os.environ['V199_OUT'] = str(OUT)  # compatibility with existing gate helpers\n",
    )
    replace_required(
        notebook,
        "os.environ['RUN_ID'] = 'v199-h100-highram-v194-rank19-20s'",
        "os.environ['RUN_ID'] = 'v201a-h100-v194-rank19-solver-verified-micro-5s'",
    )
    replace_required(
        notebook,
        f"FIXED_TRAIN_SCRIPT_URL = '{TRAIN_SCRIPT_URL}'",
        f"FIXED_TRAIN_SCRIPT_URL = '{BRANCH_SCRIPT_BASE}/hf_job_train_v90.py'",
    )
    replace_required(
        notebook,
        "if 'load_peft_weights_with_direct_fallback' not in script_text:",
        "if 'load_peft_weights_with_direct_fallback' not in script_text or 'BASELINE_EVAL_BEFORE_TRAIN' not in script_text:",
    )
    replace_required(
        notebook,
        "assert 'PEFT_MANUAL_LOAD_METHOD' in script_text\n",
        "assert 'PEFT_MANUAL_LOAD_METHOD' in script_text\n"
        "assert 'BASELINE_EVAL_BEFORE_TRAIN' in script_text\n"
        "assert 'REQUIRE_FINAL_EVAL_LTE_BASELINE' in script_text\n",
    )
    replace_required(notebook, "MAX_STEPS'] = '20'", "MAX_STEPS'] = '5'")
    replace_required(notebook, "SAVE_EVERY_STEPS'] = '10'", "SAVE_EVERY_STEPS'] = '5'")
    replace_required(notebook, "EVAL_EVERY_STEPS'] = '10'", "EVAL_EVERY_STEPS'] = '5'")
    replace_required(notebook, "LEARNING_RATE'] = '3e-6'", "LEARNING_RATE'] = '3e-7'")
    replace_required(notebook, "FINAL_LEARNING_RATE'] = '8e-7'", "FINAL_LEARNING_RATE'] = '1e-7'")
    replace_required(
        notebook,
        "os.environ['ABORT_EVAL_LOSS_GT'] = '0.98'\n",
        "os.environ['ABORT_EVAL_LOSS_GT'] = '0'\n"
        "os.environ['BASELINE_EVAL_BEFORE_TRAIN'] = '1'\n"
        "os.environ['ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA'] = '0.005'\n"
        "os.environ['REQUIRE_FINAL_EVAL_LTE_BASELINE'] = '1'\n"
        "os.environ['MAX_FINAL_EVAL_REGRESSION'] = '0.0'\n",
    )
    replace_required(
        notebook,
        "os.environ['TRAINABLE_LORA_MODULES'] = 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'\n",
        "os.environ['SAMPLING_MODE'] = 'weighted_replacement'\n"
        "os.environ['SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5,cipher:2.0,cryptarithm_deduce:3.0,cryptarithm_guess:2.0,equation_numeric_deduce:3.0,equation_numeric_guess:2.0,equation_transform:1.5'\n"
        "os.environ['SOURCE_WEIGHTS'] = 'v198_v196_wrong_anti_regression:2.0,v198_v197_strict_gain_distill:1.5,v198_v195_balanced_rehearsal:1.0'\n"
        "os.environ['TRAINABLE_LORA_MODULES'] = 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'\n",
    )
    replace_required(
        notebook,
        "Convert and gate the V199 adapters. This still does not submit to Kaggle.",
        "Convert and gate the V201A adapter. This still does not submit to Kaggle.",
    )
    replace_required(
        notebook,
        "for name in ['kg1_v198_posttrain_gate.py', 'kg1_v199_posttrain_gate.py', 'nemotron_submission_preflight.py', 'kg1_submission_gate.py', 'kg1_v198_final_submit_doublecheck.py']:",
        "for name in ['kg1_v198_posttrain_gate.py', 'kg1_v201a_posttrain_gate.py', 'nemotron_submission_preflight.py', 'kg1_submission_gate.py', 'kg1_v198_final_submit_doublecheck.py']:",
    )
    replace_required(
        notebook,
        "!python scripts/kg1_v199_posttrain_gate.py --root /content/kg1_v199 --output-root \"$V199_OUT\" --fail-on-block",
        "!python scripts/kg1_v201a_posttrain_gate.py --root /content/kg1_v199 --output-root \"$V199_OUT\" --fail-on-block",
    )
    replace_required(
        notebook,
        "report_path = pathlib.Path(os.environ['V199_OUT']) / 'posttrain_kaggle_gate/v199_posttrain_gate_report.json'",
        "report_path = pathlib.Path(os.environ['V199_OUT']) / 'posttrain_kaggle_gate/v201a_posttrain_gate_report.json'",
    )
    replace_required(
        notebook,
        "assert primary_label == 'final', f'Blocked: only final adapter can be promoted for V199, got {primary_label}'",
        "assert primary_label == 'final', f'Blocked: only final adapter can be promoted for V201A, got {primary_label}'",
    )
    replace_required(
        notebook,
        "print('V199 gated candidate ready. No Kaggle submit was performed.')",
        "print('V201A gated candidate ready. No Kaggle submit was performed.')",
    )

    source = all_source(notebook)
    required = [
        "V194_RANK19_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'",
        "BEST_RANKING_BASELINE_RULE",
        "assert BEST_RANKING_BASELINE['rank'] == '19/2613'",
        "BASELINE_EVAL_BEFORE_TRAIN'] = '1'",
        "REQUIRE_FINAL_EVAL_LTE_BASELINE'] = '1'",
        "MAX_FINAL_EVAL_REGRESSION'] = '0.0'",
        "SAMPLING_MODE'] = 'weighted_replacement'",
        "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5",
        "SOURCE_WEIGHTS'] = 'v198_v196_wrong_anti_regression:2.0",
        "TRAINABLE_LORA_MODULES'] = 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'",
        "kg1_v201a_posttrain_gate.py",
    ]
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Built V201A notebook is missing {fragment!r}")
    forbidden = [
        "--submit",
        "kg1_v199b_safe_kaggle_submit.py",
        "v199-conservative-v198-final-20s",
        "V198_FINAL_ADAPTER_SHA256",
        "ABORT_EVAL_LOSS_GT'] = '0.98'",
        "LEARNING_RATE'] = '3e-6'",
        "FINAL_LEARNING_RATE'] = '8e-7'",
        "MAX_STEPS'] = '20'",
        "MAX_STEPS'] = '10'",
        "kg1_v199_posttrain_gate.py",
    ]
    for fragment in forbidden:
        if fragment in source:
            raise RuntimeError(f"Built V201A notebook contains forbidden fragment {fragment!r}")
    return notebook


def main() -> int:
    notebook = build_v201a_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# V201A H100 solver-verified weak-category micro-train\n\n"
        "- Notebook: `notebooks/KG1_V201A_H100_SOLVER_VERIFIED_MICRO_COLAB_PRO.ipynb`.\n"
        "- Production baseline remains V194/ref `52275052`, public score `0.86`, rank `19/2613`.\n"
        "- Starts only from exact V194 `submission.zip` SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.\n"
        "- Runs 5 steps at LR `3e-7 -> 1e-7`.\n"
        "- Uses weighted weak-category sampling: bit manipulation, cipher, cryptarithm, and equation numeric subcategories.\n"
        "- Trains attention LoRA modules only: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.\n"
        "- Evaluates V194 baseline before training and blocks final promotion unless `final_eval_loss <= baseline_eval_loss`.\n"
        "- Converts with `kg1_v201a_posttrain_gate.py`; no Kaggle submit cell is present.\n"
        "- After training, submit only if the posttrain gate is `READY` and Kaggle authorization is explicit.\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
