#!/usr/bin/env python3
"""Build the V201C three-candidate micro-train Colab notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from build_v201_087_notebook import all_source, replace_required
from build_v201b_baseline_neutral_notebook import build_v201b_notebook


NOTEBOOK_PATH = Path("notebooks/KG1_V201C_H100_A100_MULTI_CANDIDATE_MICRO_COLAB_PRO.ipynb")
REPORT_PATH = Path("runs/v200_rank_hillclimb_20260504/V201C_MULTI_CANDIDATE_NEXT_ACTIONS.md")


def code(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(True)


GPU_GATE_CELL = r"""
import pathlib, re, shutil, subprocess
gpu_csv = subprocess.check_output(
    'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits',
    shell=True,
).decode().strip()
print('GPU:', gpu_csv)
parts = [part.strip() for part in gpu_csv.split(',')]
assert len(parts) >= 3, f'Unexpected nvidia-smi output: {gpu_csv}'
gpu_name = parts[0]
gpu_mem_mib = int(parts[1])
driver_version = parts[2]
is_supported_gpu = ('H100' in gpu_name) or ('A100' in gpu_name and gpu_mem_mib >= 75000)
assert is_supported_gpu, f'Use H100 or A100 80GB High-RAM for this notebook; found {gpu_name} with {gpu_mem_mib} MiB'
meminfo = pathlib.Path('/proc/meminfo').read_text(encoding='utf-8')
host_mem_kib = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1))
host_mem_gib = host_mem_kib / 1024 / 1024
disk = shutil.disk_usage('/content')
disk_free_gib = disk.free / 1024**3
print(f'Host RAM: {host_mem_gib:.1f} GiB')
print(f'/content free: {disk_free_gib:.1f} GiB')
print('Driver:', driver_version)
assert host_mem_gib >= 50, f'High-RAM runtime expected; host RAM is only {host_mem_gib:.1f} GiB'
assert disk_free_gib >= 120, f'Need at least 120 GiB free on /content for three sequential candidates; found {disk_free_gib:.1f} GiB'
"""


TRAIN_CELL = r"""
import datetime, json, os, pathlib, re, subprocess, sys, urllib.request

OUT_ROOT = OUT_BASE
if OUT_ROOT.exists():
    suffix = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')
    OUT_ROOT = pathlib.Path(str(OUT_BASE) + '_' + suffix)
OUT_ROOT.mkdir(parents=True, exist_ok=True)
print('V201C_OUT =', OUT_ROOT)
os.environ['V201C_OUT'] = str(OUT_ROOT)

FIXED_TRAIN_SCRIPT_URL = 'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir/scripts/hf_job_train_v90.py'
TRAIN_SCRIPT = pathlib.Path('/content/kg1_v199/scripts/hf_job_train_v90.py')
TRAIN_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
script_text = TRAIN_SCRIPT.read_text(encoding='utf-8') if TRAIN_SCRIPT.exists() else ''
if 'load_peft_weights_with_direct_fallback' not in script_text or 'BASELINE_EVAL_BEFORE_TRAIN' not in script_text:
    print('Runtime has stale hf_job_train_v90.py; downloading PEFT direct-load fixed script...')
    urllib.request.urlretrieve(FIXED_TRAIN_SCRIPT_URL, TRAIN_SCRIPT)
script_text = TRAIN_SCRIPT.read_text(encoding='utf-8')
assert 'load_peft_weights_with_direct_fallback' in script_text
assert 'PEFT_MANUAL_LOAD_METHOD' in script_text
assert 'BASELINE_EVAL_BEFORE_TRAIN' in script_text
assert 'REQUIRE_FINAL_EVAL_LTE_BASELINE' in script_text

BASE_ENV = {
    'UPLOAD_TO_HF': '0',
    'MODEL_NAME': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    'MODEL_REVISION': 'cbd3fa9f933d55ef16a84236559f4ee2a0526848',
    'DATA_FILE': '/content/kg1_v199/data/v198/v198_micro_train.strict.jsonl',
    'VAL_FILE': '/content/kg1_v199/data/v198/v198_micro_val.strict.jsonl',
    'INIT_ADAPTER_DIR': str(INIT_ADAPTER),
    'INIT_ADAPTER_LOAD_MODE': 'manual',
    'PEFT_MANUAL_LOAD_METHOD': 'direct',
    'MAX_LENGTH': '2048',
    'BATCH_SIZE': '16',
    'MICRO_BATCH_SIZE': '1',
    'GRADIENT_CHECKPOINTING': '1',
    'EVAL_MAX_EXAMPLES': '360',
    'ABORT_EVAL_LOSS_GT': '0',
    'BASELINE_EVAL_BEFORE_TRAIN': '1',
    'REQUIRE_FINAL_EVAL_LTE_BASELINE': '1',
    'MAX_FINAL_EVAL_REGRESSION': '0.0',
    'EXPECTED_TRAIN_SHA256': '6d2742616300818eb50c54d36019551b24f5b71c607a2b28feda7461a709def0',
    'EXPECTED_VAL_SHA256': 'e59c907c6545e5e587097a64762e3e874508e8cd74d85d5c7c79354ebe56e73c',
    'MIN_TRAIN_EXAMPLES': '1875',
    'MIN_TOKENIZED_TRAIN_EXAMPLES': '1600',
    'MIN_VAL_EXAMPLES': '720',
    'MIN_TOKENIZED_VAL_EXAMPLES': '700',
    'TRAINABLE_LORA_MODULES': 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj',
    'MAX_TRAINABLE_PARAM_RATIO': '0.035',
}

CANDIDATES = [
    {
        'label': 'A_neutral_shuffle_3s',
        'run_id': 'v201c-A-neutral-shuffle-3s',
        'max_steps': '3',
        'learning_rate': '2e-7',
        'final_learning_rate': '1e-7',
        'sampling_mode': 'shuffle',
        'subcategory_weights': '',
        'source_weights': '',
        'abort_relative_delta': '0.003',
    },
    {
        'label': 'B_equation_crypt_low_2s',
        'run_id': 'v201c-B-equation-crypt-low-2s',
        'max_steps': '2',
        'learning_rate': '1e-7',
        'final_learning_rate': '5e-8',
        'sampling_mode': 'weighted_replacement',
        'subcategory_weights': 'equation_transform=1.15,cryptarithm_deduce=1.25,cryptarithm_guess=1.10,equation_numeric_deduce=1.25,equation_numeric_guess=1.10',
        'source_weights': 'v198_v196_wrong_anti_regression=1.10,v198_v197_strict_gain_distill=1.05,v198_v195_balanced_rehearsal=1.0',
        'abort_relative_delta': '0.002',
    },
    {
        'label': 'C_bit_cipher_low_2s',
        'run_id': 'v201c-C-bit-cipher-low-2s',
        'max_steps': '2',
        'learning_rate': '1e-7',
        'final_learning_rate': '5e-8',
        'sampling_mode': 'weighted_replacement',
        'subcategory_weights': 'bit_manipulation=1.20,cipher=1.20',
        'source_weights': 'v198_v196_wrong_anti_regression=1.10,v198_v197_strict_gain_distill=1.05,v198_v195_balanced_rehearsal=1.0',
        'abort_relative_delta': '0.002',
    },
]

def parse_metrics(log_text):
    baseline_matches = re.findall(r'baseline_eval_loss=([0-9.]+)', log_text)
    final_matches = re.findall(r'Final eval loss: ([0-9.]+); best eval loss: ([0-9.]+)', log_text)
    step_eval_matches = re.findall(r'eval step=(\d+) loss=([0-9.]+) best=([0-9.]+)', log_text)
    metrics = {
        'baseline_eval_loss': float(baseline_matches[-1]) if baseline_matches else None,
        'final_eval_loss': float(final_matches[-1][0]) if final_matches else None,
        'best_eval_loss': float(final_matches[-1][1]) if final_matches else None,
        'step_evals': [
            {'step': int(step), 'loss': float(loss), 'best': float(best)}
            for step, loss, best in step_eval_matches
        ],
    }
    if metrics['baseline_eval_loss'] is not None and metrics['final_eval_loss'] is not None:
        metrics['delta_vs_baseline'] = round(metrics['final_eval_loss'] - metrics['baseline_eval_loss'], 6)
    else:
        metrics['delta_vs_baseline'] = None
    return metrics

def stream_process(cmd, cwd, env, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        return proc.wait()

results = []
for candidate in CANDIDATES:
    label = candidate['label']
    candidate_out = OUT_ROOT / label
    candidate_out.mkdir(parents=True, exist_ok=True)
    log_path = candidate_out / 'train.log'
    print('\n' + '=' * 80)
    print('Starting V201C candidate:', label)
    print('Output:', candidate_out)
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update({
        'OUTPUT_DIR': str(candidate_out),
        'V199_OUT': str(candidate_out),
        'V201C_OUT': str(OUT_ROOT),
        'V201C_CANDIDATE_OUT': str(candidate_out),
        'RUN_ID': candidate['run_id'],
        'MAX_STEPS': candidate['max_steps'],
        'SAVE_EVERY_STEPS': candidate['max_steps'],
        'EVAL_EVERY_STEPS': candidate['max_steps'],
        'LEARNING_RATE': candidate['learning_rate'],
        'FINAL_LEARNING_RATE': candidate['final_learning_rate'],
        'SAMPLING_MODE': candidate['sampling_mode'],
        'SUBCATEGORY_WEIGHTS': candidate['subcategory_weights'],
        'SOURCE_WEIGHTS': candidate['source_weights'],
        'ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA': candidate['abort_relative_delta'],
    })
    returncode = stream_process([sys.executable, 'scripts/hf_job_train_v90.py'], ROOT, env, log_path)
    log_text = log_path.read_text(encoding='utf-8', errors='replace')
    metrics = parse_metrics(log_text)
    manifest_path = candidate_out / 'final_adapter/v90_training_manifest.json'
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        gate = manifest.get('training', {}).get('baseline_gate', {})
        metrics['baseline_eval_loss'] = gate.get('baseline_eval_loss', metrics['baseline_eval_loss'])
        metrics['final_eval_loss'] = gate.get('final_eval_loss', metrics['final_eval_loss'])
        if metrics['baseline_eval_loss'] is not None and metrics['final_eval_loss'] is not None:
            metrics['delta_vs_baseline'] = round(metrics['final_eval_loss'] - metrics['baseline_eval_loss'], 6)
    passed = (
        returncode == 0
        and metrics.get('baseline_eval_loss') is not None
        and metrics.get('final_eval_loss') is not None
        and metrics['final_eval_loss'] <= metrics['baseline_eval_loss']
        and manifest_path.exists()
    )
    result = {
        'label': label,
        'run_id': candidate['run_id'],
        'output_dir': str(candidate_out),
        'returncode': returncode,
        'passed_no_regression_gate': bool(passed),
        'metrics': metrics,
        'manifest_path': str(manifest_path) if manifest_path.exists() else None,
        'log_path': str(log_path),
        'config': candidate,
    }
    (candidate_out / 'candidate_summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    results.append(result)
    print('Candidate result:', json.dumps({
        'label': label,
        'returncode': returncode,
        'passed_no_regression_gate': passed,
        'baseline_eval_loss': metrics.get('baseline_eval_loss'),
        'final_eval_loss': metrics.get('final_eval_loss'),
        'delta_vs_baseline': metrics.get('delta_vs_baseline'),
    }, indent=2))

summary = {
    'root': str(OUT_ROOT),
    'baseline': {
        'label': 'V194',
        'rank': '19/2613',
        'public_score': '0.86',
        'zip_sha256': V194_RANK19_ZIP_SHA256,
        'adapter_model_sha256': V194_RANK19_ADAPTER_MODEL_SHA256,
    },
    'candidates': results,
    'passed_candidates': [item for item in results if item['passed_no_regression_gate']],
    'policy': 'Only passed candidates may be converted; no Kaggle submit is performed automatically.',
}
summary_path = OUT_ROOT / 'v201c_candidates_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print('\nV201C candidate summary:', summary_path)
print(json.dumps([
    {
        'label': item['label'],
        'passed': item['passed_no_regression_gate'],
        'baseline': item['metrics'].get('baseline_eval_loss'),
        'final': item['metrics'].get('final_eval_loss'),
        'delta': item['metrics'].get('delta_vs_baseline'),
    }
    for item in results
], indent=2))
"""


PACKAGE_CELL = r"""
import json, os, pathlib, subprocess, sys, urllib.request

BASE = 'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir/scripts'
for name in ['kg1_convert_local_training_adapter_to_kaggle_zip.py', 'kg1_v198_posttrain_gate.py', 'kg1_v201c_posttrain_gate.py', 'nemotron_submission_preflight.py', 'kg1_submission_gate.py', 'kg1_v198_final_submit_doublecheck.py']:
    dst = pathlib.Path('/content/kg1_v199/scripts') / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    print('downloading', name)
    urllib.request.urlretrieve(f'{BASE}/{name}', dst)

out_root = pathlib.Path(os.environ['V201C_OUT'])
summary_path = out_root / 'v201c_candidates_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
ready = []
blocked = []

for item in summary['candidates']:
    label = item['label']
    candidate_out = pathlib.Path(item['output_dir'])
    if not item['passed_no_regression_gate']:
        blocked.append({'label': label, 'reason': 'failed_no_regression_gate_or_training_error', 'metrics': item.get('metrics')})
        continue

    print('\nPackaging passed candidate:', label)
    subprocess.run([
        sys.executable,
        'scripts/kg1_v201c_posttrain_gate.py',
        '--root',
        '/content/kg1_v199',
        '--output-root',
        str(candidate_out),
        '--candidate-label',
        label,
        '--fail-on-block',
    ], check=True)

    report_path = candidate_out / 'posttrain_kaggle_gate/v201c_posttrain_gate_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not report['decision']['ready']:
        blocked.append({'label': label, 'reason': 'posttrain_gate_blocked', 'decision': report['decision']})
        continue

    primary_zip = report['decision']['primary_zip']
    primary_label = report['decision']['primary_label']
    assert primary_label == 'final', f'Blocked: only final adapter can be promoted for V201C, got {primary_label}'

    preflight_json = candidate_out / 'final_preflight.json'
    subprocess.run([
        sys.executable,
        'scripts/nemotron_submission_preflight.py',
        '--adapter-zip',
        primary_zip,
        '--output-json',
        str(preflight_json),
        '--fail-on-block',
    ], check=True)

    doublecheck_json = candidate_out / 'final_submit_doublecheck.json'
    subprocess.run([
        sys.executable,
        'scripts/kg1_v198_final_submit_doublecheck.py',
        '--candidate-zip',
        primary_zip,
        '--expected-label',
        'final',
        '--posttrain-report',
        str(report_path),
        '--preflight-report',
        str(preflight_json),
        '--output-json',
        str(doublecheck_json),
        '--fail-on-block',
    ], check=True)

    ready.append({
        'label': label,
        'zip': primary_zip,
        'posttrain_report': str(report_path),
        'preflight_report': str(preflight_json),
        'doublecheck_report': str(doublecheck_json),
        'metrics': item.get('metrics'),
    })

ready_sorted = sorted(
    ready,
    key=lambda item: (
        item['metrics'].get('delta_vs_baseline', 999),
        item['metrics'].get('final_eval_loss', 999),
        item['label'],
    ),
)
selection = {
    'decision': 'READY' if ready_sorted else 'NO_READY_CANDIDATE',
    'selected': ready_sorted[0] if ready_sorted else None,
    'ready_candidates': ready_sorted,
    'blocked_candidates': blocked,
    'do_not_submit_without_explicit_authorization': True,
    'production_baseline': {
        'label': 'V194',
        'rank': '19/2613',
        'public_score': '0.86',
        'zip_sha256': V194_RANK19_ZIP_SHA256,
    },
}
selection_path = out_root / 'v201c_final_selection.json'
selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print('\nV201C final selection:', selection_path)
print(json.dumps(selection, indent=2))
print('No Kaggle submit was performed.')
"""


def build_v201c_notebook() -> dict:
    notebook = build_v201b_notebook()
    notebook["metadata"]["colab"]["name"] = NOTEBOOK_PATH.name
    notebook["cells"][0]["source"] = (
        "# KG1 V201C H100/A100 three-candidate micro-train\n\n"
        "This notebook runs three independent micro-train candidates from the exact V194 rank-19 "
        "adapter in one Colab session. Each candidate starts from V194, writes to its own output "
        "directory, runs baseline eval before training, blocks final eval regression, and is converted "
        "only if it passes. No Kaggle submit is performed automatically.\n"
    ).splitlines(True)

    replace_required(
        notebook,
        "OUT_BASE = DRIVE_ROOT / 'output_v201b_h100_baseline_neutral_micro_3'",
        "OUT_BASE = DRIVE_ROOT / 'output_v201c_h100_a100_multicandidate_3x'",
    )
    replace_required(
        notebook,
        "ALLOW_V194_REBUILD_FALLBACK = os.environ.get('ALLOW_V194_REBUILD_FALLBACK') == '1'",
        "ALLOW_V194_REBUILD_FALLBACK = False",
    )
    replace_required(
        notebook,
        "'Automatic Tinker reconstruction is disabled by default because it is not byte-stable in this runtime. '",
        "'Automatic Tinker reconstruction is disabled for V201C production training. '",
    )
    replace_required(
        notebook,
        "'Set ALLOW_V194_REBUILD_FALLBACK=1 only for forensic rebuilds, not production training.'",
        "'Stage the exact V194 rank-19 submission.zip before training.'",
    )
    replace_required(
        notebook,
        "print('ALLOW_V194_REBUILD_FALLBACK=1; using SHA-gated reconstruction path.')",
        "print('V201C fallback rebuild requested but production training requires exact V194 zip.')",
    )
    replace_required(
        notebook,
        """    if not ALLOW_V194_REBUILD_FALLBACK:
        raise RuntimeError(missing_v194_zip_message())
    print('V201C fallback rebuild requested but production training requires exact V194 zip.')
    primary = ensure_aaitdads_component()
    other = ensure_lineage_component()
    TOOLS_ROOT.mkdir(parents=True, exist_ok=True)
    soup_script = TOOLS_ROOT / 'kg1_update_space_soup_stream.py'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir/scripts/kg1_update_space_soup_stream.py', soup_script)
    if importlib.util.find_spec('safetensors') is None:
        pip_install_quiet(['safetensors==0.7.0'])
    print('Rebuilding exact V194 rank-19 adapter: 98.5% aaitdads + 1.5% lineage attention-only.')
    subprocess.run([
        sys.executable, str(soup_script),
        '--primary-adapter', str(primary),
        '--other-adapter', str(other),
        '--output-dir', str(RANK19_BUILD),
        '--config-source', str(primary / 'adapter_config.json'),
        '--primary-weight', '0.985',
        '--other-weight', '0.015',
        '--rank', '32',
        '--copy-safe-primary-non-lora',
        '--include-key-regex', r'\\.mixer\\.(in_proj|out_proj|q_proj|k_proj|v_proj|o_proj)\\.lora_A\\.',
    ], check=True)
    manifest = json.loads((RANK19_BUILD / 'update_space_soup_manifest.json').read_text(encoding='utf-8'))
    assert manifest.get('output_adapter_sha256') == V194_RANK19_ADAPTER_MODEL_SHA256, manifest
    assert manifest.get('output_zip_sha256') == V194_RANK19_ZIP_SHA256, manifest
    assert adapter_ready(INIT_ADAPTER, min_model_bytes=4_000_000_000), f'V194 rank-19 adapter was not built: {INIT_ADAPTER}'
    assert sha256_path(cfg) == V194_RANK19_ADAPTER_CONFIG_SHA256
    assert sha256_path(model) == V194_RANK19_ADAPTER_MODEL_SHA256
    assert sha256_path(zip_path) == V194_RANK19_ZIP_SHA256
    return INIT_ADAPTER
""",
        """    raise RuntimeError(missing_v194_zip_message())
""",
    )
    for old, new in [
        ("V201B", "V201C"),
        ("v201b", "v201c"),
        ("baseline-neutral micro-train", "three-candidate micro-train"),
    ]:
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source") or [])
            if old in source:
                cell["source"] = source.replace(old, new).splitlines(True)

    notebook["cells"][5]["source"] = code(GPU_GATE_CELL)
    notebook["cells"][6]["source"] = code(TRAIN_CELL)
    notebook["cells"][7]["source"] = (
        "Convert and gate only the V201C candidates that passed the no-regression gate. "
        "This still does not submit to Kaggle.\n"
    ).splitlines(True)
    notebook["cells"][8]["source"] = code(PACKAGE_CELL)

    source = all_source(notebook)
    required = [
        "KG1 V201C H100/A100 three-candidate micro-train",
        "V194_RANK19_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'",
        "V194_RANK19_ADAPTER_MODEL_SHA256 = '01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f'",
        "OUT_BASE = DRIVE_ROOT / 'output_v201c_h100_a100_multicandidate_3x'",
        "A_neutral_shuffle_3s",
        "B_equation_crypt_low_2s",
        "C_bit_cipher_low_2s",
        "'MODEL_REVISION': 'cbd3fa9f933d55ef16a84236559f4ee2a0526848'",
        "MAX_FINAL_EVAL_REGRESSION': '0.0'",
        "'REQUIRE_FINAL_EVAL_LTE_BASELINE': '1'",
        "TRAIN_SCRIPT.parent.mkdir(parents=True, exist_ok=True)",
        "kg1_convert_local_training_adapter_to_kaggle_zip.py",
        "kg1_v201c_posttrain_gate.py",
        "v201c_candidates_summary.json",
        "v201c_final_selection.json",
        "ALLOW_V194_REBUILD_FALLBACK = False",
        "No Kaggle submit was performed.",
    ]
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Built V201C notebook is missing {fragment!r}")
    forbidden = [
        "kaggle competitions submit",
        "KaggleApi",
        "files.upload",
        "output_v201a_h100_solver_verified_micro_5",
        "output_v201b_h100_baseline_neutral_micro_3",
        "kg1_v201a_posttrain_gate.py",
        "kg1_v201b_posttrain_gate.py",
        "bit_manipulation=2.5",
        "v198_v196_wrong_anti_regression=2.0",
        "SUBCATEGORY_WEIGHTS'] = 'bit_manipulation:2.5",
        "ALLOW_V194_REBUILD_FALLBACK=1",
        "V201C fallback rebuild requested",
        "Rebuilding exact V194 rank-19 adapter",
    ]
    for fragment in forbidden:
        if fragment in source:
            raise RuntimeError(f"Built V201C notebook contains forbidden fragment {fragment!r}")
    return notebook


def main() -> int:
    notebook = build_v201c_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# V201C three-candidate micro-train\n\n"
        "- Notebook: `notebooks/KG1_V201C_H100_A100_MULTI_CANDIDATE_MICRO_COLAB_PRO.ipynb`.\n"
        "- Runs three independent candidates from exact V194 rank-19, not sequential phases on one adapter.\n"
        "- Candidate A: neutral shuffle, 3 steps, LR `2e-7 -> 1e-7`.\n"
        "- Candidate B: light equation/cryptarithm weighting, 2 steps, LR `1e-7 -> 5e-8`.\n"
        "- Candidate C: light bit/cipher weighting, 2 steps, LR `1e-7 -> 5e-8`.\n"
        "- Each candidate has baseline eval before training and final eval no-regression gate.\n"
        "- Only passed candidates are converted and preflighted; no Kaggle submit cell exists.\n"
        "- H100 or A100 80GB High-RAM is required; A100 40GB is blocked.\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
