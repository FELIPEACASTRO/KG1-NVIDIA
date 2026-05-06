#!/usr/bin/env python3
"""Build the V206B answer-only H100 Colab notebook.

This notebook is derived from the vetted V206A notebook, but switches the
training data to the answer-only micro dataset and reduces the first candidate
to one ultra-low-LR update after V206A regressed.
"""

from __future__ import annotations

import json
from pathlib import Path


SOURCE_NOTEBOOK = Path(
    ".claude/worktrees/competent-shamir/notebooks/"
    "KG1_V206A_H100_LOSS_GATED_COLAB.ipynb"
)
NOTEBOOK_PATH = Path(
    ".claude/worktrees/competent-shamir/notebooks/"
    "KG1_V206B_H100_ANSWER_ONLY_LOSS_GATED_COLAB.ipynb"
)

V206_TRAIN_SHA256 = "65a810d54da73fd3859d7ee9a9edc0c35a3f89231c0033ea74f26e55f254f9f0"
V206B_TRAIN_SHA256 = "5c95e9b254a3b3a37850db1c4f75914d4b23233f0cd1e9d828886399e9a42f5d"


def replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def apply_friendly_training_block(text: str) -> str:
    old_training_block = """    write_json(REPORT_DIR / 'v206b_train_summary.json', train_summary)
    print(json.dumps(train_summary, indent=2, sort_keys=True))
    if rc != 0:
        raise RuntimeError(f'V206B training failed or gate blocked; see {TRAIN_OUT}')
    if not train_summary['passed_no_regression_gate']:
        raise RuntimeError('V206B did not pass final-eval <= baseline gate; packaging is blocked.')
else:
    print('RUN_TRAIN=False; skipping training.')
"""
    new_training_block = """    if rc != 0 or not train_summary['passed_no_regression_gate']:
        baseline = train_summary.get('baseline_eval_loss')
        final = train_summary.get('final_eval_loss')
        delta = None if baseline is None or final is None else final - baseline
        train_summary.update({
            'status': 'blocked_no_submit',
            'friendly_message': (
                'V206B finished, but the safety gate blocked this candidate. '
                'The final validation loss was worse than the V194 baseline, so packaging and Kaggle submission are skipped.'
            ),
            'package_blocked': True,
        })
        write_json(REPORT_DIR / 'v206b_training_blocked_friendly.json', train_summary)
        RUN_PACKAGE = False
        print('\\n' + '=' * 72)
        print('V206B BLOQUEADO PELO GATE DE SEGURANCA')
        print('=' * 72)
        print('O treino terminou, mas este adapter NAO deve ser submetido.')
        if baseline is not None:
            print(f'baseline_eval_loss: {baseline:.4f}')
        if final is not None:
            print(f'final_eval_loss:    {final:.4f}')
        if delta is not None:
            print(f'delta_vs_baseline: {delta:+.6f}')
        print('Motivo: o loss final ficou acima do baseline permitido.')
        print('Acao tomada: empacotamento desativado; nenhum submit Kaggle sera feito.')
        print('Resumo amigavel salvo em:', REPORT_DIR / 'v206b_training_blocked_friendly.json')
        print('=' * 72 + '\\n')
    else:
        train_summary['status'] = 'passed_no_regression_gate'
    write_json(REPORT_DIR / 'v206b_train_summary.json', train_summary)
    print(json.dumps(train_summary, indent=2, sort_keys=True))
else:
    print('RUN_TRAIN=False; skipping training.')
"""
    if old_training_block not in text:
        raise RuntimeError("Could not find V206B training gate block to replace.")
    return text.replace(old_training_block, new_training_block)


def apply_friendly_package_skip(text: str) -> str:
    old_package_skip = """else:
    print('RUN_PACKAGE=False; skipping packaging.')
"""
    new_package_skip = """else:
    if train_summary and train_summary.get('package_blocked'):
        print('RUN_PACKAGE=False; packaging skipped because V206B did not pass the no-regression gate.')
    else:
        print('RUN_PACKAGE=False; skipping packaging.')
"""
    if old_package_skip not in text:
        raise RuntimeError("Could not find V206B package skip block to replace.")
    return text.replace(old_package_skip, new_package_skip)


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    replacements = [
        ("KG1_V206A_H100_LOSS_GATED_COLAB.ipynb", "KG1_V206B_H100_ANSWER_ONLY_LOSS_GATED_COLAB.ipynb"),
        ("KG1 V206A H100 Loss-Gated Colab", "KG1 V206B H100 Answer-Only Loss-Gated Colab"),
        ("V206A_H100_LOSS_GATED_20260505", "V206B_H100_ANSWER_ONLY_LOSS_GATED_20260506"),
        ("KG1_NVIDIA_V206", "KG1_NVIDIA_V206B"),
        ("output_v206a_h100_loss_gated", "output_v206b_answer_only_h100_loss_gated"),
        ("train_v206a_3s_lr5e9", "train_v206b_answer_only_1s_lr1e9"),
        ("V206_TRAIN_SHA256", "V206B_TRAIN_SHA256"),
        (V206_TRAIN_SHA256, V206B_TRAIN_SHA256),
        ("data/v206/v206_curated_train.jsonl", "data/v206b/v206b_answer_only_micro_train.jsonl"),
        ("data/v206/v206_curated_manifest.json", "data/v206b/v206b_answer_only_manifest.json"),
        ("V206 train", "V206B answer-only train"),
        ("'MIN_TRAIN_EXAMPLES': '6548'", "'MIN_TRAIN_EXAMPLES': '1680'"),
        ("'MIN_TOKENIZED_TRAIN_EXAMPLES': '6400'", "'MIN_TOKENIZED_TRAIN_EXAMPLES': '1600'"),
        ("'MAX_STEPS': '3'", "'MAX_STEPS': '1'"),
        ("'SAVE_EVERY_STEPS': '3'", "'SAVE_EVERY_STEPS': '1'"),
        ("'EVAL_EVERY_STEPS': '3'", "'EVAL_EVERY_STEPS': '1'"),
        ("'SEED': '206'", "'SEED': '2062'"),
        ("'LEARNING_RATE': '5e-9'", "'LEARNING_RATE': '1e-9'"),
        ("'FINAL_LEARNING_RATE': '5e-9'", "'FINAL_LEARNING_RATE': '1e-9'"),
        ("'COMPUTE_PROVIDER': 'colab_v206a_h100_loss_gated'", "'COMPUTE_PROVIDER': 'colab_v206b_answer_only_h100_loss_gated'"),
        ("'RUN_ID': 'v206a-dryrun-v194-curated-8192'", "'RUN_ID': 'v206b-dryrun-answer-only-8192'"),
        ("'RUN_ID': 'v206a-v194-curated-3s-lr5e9'", "'RUN_ID': 'v206b-answer-only-1s-lr1e9'"),
        ("report['data']['train_records'] == 6548", "report['data']['train_records'] == 1680"),
        ("report['data']['tokenized_train_records'] >= 6400", "report['data']['tokenized_train_records'] >= 1600"),
        ("V206A dry-run passed", "V206B dry-run passed"),
        ("train_v206a_3s_lr5e9.log", "train_v206b_answer_only_1s_lr1e9.log"),
        ("V206A_3s_lr5e9", "V206B_answer_only_1s_lr1e9"),
        ("v206a_submission_preflight.json", "v206b_submission_preflight.json"),
        ("v206a_submission_preflight.log", "v206b_submission_preflight.log"),
        ("v206a_package_summary.json", "v206b_package_summary.json"),
        ("v206a_train_summary.json", "v206b_train_summary.json"),
        ("V206A_FINAL_RUN_SUMMARY.json", "V206B_FINAL_RUN_SUMMARY.json"),
        ("V206A", "V206B"),
        ("v206a", "v206b"),
    ]

    for cell in notebook["cells"]:
        source = "".join(cell.get("source") or [])
        source = replace_all(source, replacements)
        if "v206b_train_summary.json" in source and "V206B training failed or gate blocked" in source:
            source = apply_friendly_training_block(source)
        if "RUN_PACKAGE=False; skipping packaging" in source:
            source = apply_friendly_package_skip(source)
        if cell.get("cell_type") == "markdown" and source.startswith("# KG1 V206B"):
            source += (
                "\nV206B is a response-objective correction after V206A regressed: "
                "it trains only concise `\\boxed{answer}` completions from a 1680-row "
                "balanced micro dataset, with one `1e-9` update before the same no-regression gate.\n"
            )
        cell["source"] = source.splitlines(keepends=True)

    notebook["metadata"]["colab"]["name"] = "KG1_V206B_H100_ANSWER_ONLY_LOSS_GATED_COLAB.ipynb"

    full_text = "\n".join("".join(cell.get("source") or []) for cell in notebook["cells"])
    required = [
        "KG1 V206B H100 Answer-Only Loss-Gated Colab",
        "ALLOW_KAGGLE_SUBMIT = False",
        "V206B_TRAIN_SHA256",
        V206B_TRAIN_SHA256,
        "data/v206b/v206b_answer_only_micro_train.jsonl",
        "'MAX_STEPS': '1'",
        "'LEARNING_RATE': '1e-9'",
        "'FINAL_LEARNING_RATE': '1e-9'",
        "--adapter-zip",
        "source_v194_rank19_submission.zip",
        "V206B BLOQUEADO PELO GATE DE SEGURANCA",
        "v206b_training_blocked_friendly.json",
    ]
    missing = [item for item in required if item not in full_text]
    if missing:
        raise RuntimeError(f"Generated V206B notebook missing required markers: {missing}")
    forbidden = [
        "kaggle competitions submit",
        "ALLOW_KAGGLE_SUBMIT = True",
        "--submission-zip",
        "data/v206/v206_curated_train.jsonl",
        "'MAX_STEPS': '3'",
        "'LEARNING_RATE': '5e-9'",
        "V206B training failed or gate blocked",
        "did not pass final-eval <= baseline gate",
    ]
    present = [item for item in forbidden if item in full_text]
    if present:
        raise RuntimeError(f"Generated V206B notebook contains forbidden markers: {present}")

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
