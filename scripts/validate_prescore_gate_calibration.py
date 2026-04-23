#!/usr/bin/env python3
"""Validate pre-score gate against real Kaggle scores — calibration test.

Objetivo: rodar o MESMO pre-score gate que V102 usa contra adapters que JÁ
foram submetidos ao Kaggle (ou que serão). Comparar estimativa local com
score real do Kaggle para medir CALIBRAÇÃO do gate.

Uso (no Colab Pro+ H100 com adapter já baixado):

    python scripts/validate_prescore_gate_calibration.py \\
        --adapter-dir /content/kg1_v80_stripped_v2 \\
        --dataset-path /content/kg1/data/sft_v70_huikang_full.jsonl \\
        --n-samples 100 \\
        --output-json /content/prescore_validation.json \\
        --actual-kaggle-score 0.50

Se --actual-kaggle-score for fornecido, computa delta e sugere ajuste.
Se não fornecido, só roda pre-score e salva JSON (pode comparar depois).

Requer: GPU com >=80GB VRAM (H100 HighRAM ou A100 80GB) para carregar
Nemotron-3-Nano-30B BF16.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


# === Embedded: same canonicalize + verify as V102 Cell 9 ===

def extract_final_answer_official(text):
    """Byte-exact replica de extract_final_answer Kaggle kernel."""
    if text is None:
        return 'NOT_FOUND'
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:\uff1a]\s*([^\n]+)',
        r'final answer\s*[:\uff1a]\s*([^\n]+)',
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1].strip()
    m = re.findall(r'-?\d+(?:\.\d+)?', text)
    if m:
        return m[-1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify_official(stored, predicted):
    """Byte-exact replica de verify Kaggle kernel."""
    stored = str(stored).strip()
    predicted = str(predicted).strip()
    if re.fullmatch(r'[01]+', stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()


_LATEX_TEXT_RE = re.compile(r'\\text\s*\{([^{}]*)\}')
_LATEX_MATHRM_RE = re.compile(r'\\mathrm\s*\{([^{}]*)\}')
_LATEX_SQRT_RE = re.compile(r'\\sqrt\s*\{([^{}]*)\}')
_LATEX_LEFT_RIGHT_RE = re.compile(r'\\(?:left|right)')
_LATEX_FRAC_RE = re.compile(r'\\(?:frac|dfrac|tfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}')
_BOXED_START_RE = re.compile(r'\\boxed\s*\{')
_TRAILING_DOT_ZERO_RE = re.compile(r'^(-?\d+)\.0+$')
_SCI_RE = re.compile(r'([-+]?\d+(?:\.\d+)?)[eE]([-+]?\d+)')
_THOUSAND_RE = re.compile(r'(?<=\d)[,_](?=\d{3}\b)')

_UNIT_SUFFIXES = (
    'm/s^2', 'm/s\u00b2', 'km/h', 'km/s', 'm/s',
    '\u00b0C', '\u00b0F', '\u00b0K',
    'kg', 'mg', 'g', 'km', 'cm', 'mm', 'nm', 'um', 'm',
    'ms', 'us', 'ns', 's', 'Hz', 'kHz', 'MHz', 'GHz',
    'J', 'kJ', 'MJ', 'W', 'kW', 'MW', 'N', 'Pa', 'kPa', 'MPa',
    'V', 'kV', 'A', 'mA',
)
_UNIT_RE = re.compile(
    r'\s*(?:' + '|'.join(re.escape(u) for u in _UNIT_SUFFIXES) + r')\b\s*$',
    flags=re.IGNORECASE,
)


def _strip_latex(text):
    text = _LATEX_TEXT_RE.sub(r'\1', text)
    text = _LATEX_MATHRM_RE.sub(r'\1', text)
    text = _LATEX_SQRT_RE.sub(r'\1', text)
    text = _LATEX_LEFT_RIGHT_RE.sub('', text)
    return text


def _eval_frac(text):
    def _sub(m):
        n, d = m.group(1).strip(), m.group(2).strip()
        try:
            nf, df = float(n), float(d)
            if df == 0:
                return m.group(0)
            v = nf / df
            if v == int(v):
                return str(int(v))
            return format(v, '.10g')
        except ValueError:
            return n + '/' + d
    for _ in range(3):
        new = _LATEX_FRAC_RE.sub(_sub, text)
        if new == text:
            break
        text = new
    return text


def _extract_last_boxed(raw):
    if not raw:
        return None
    last = None
    for match in _BOXED_START_RE.finditer(raw):
        start = match.end()
        depth = 1
        i = start
        while i < len(raw) and depth > 0:
            ch = raw[i]
            if ch == '\\':
                i += 2
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last = raw[start:i]
                    break
            i += 1
        if depth > 0 and last is None:
            last = raw[start:]
    return last


def _enforce_bit(text):
    s = text.strip()
    m = re.search(r'[01]{1,16}$', s)
    if m:
        bits = m.group(0)
        if len(bits) < 8:
            bits = bits.zfill(8)
        elif len(bits) > 8:
            bits = bits[-8:]
        return bits
    try:
        v = int(s.split()[0])
        if 0 <= v < 256:
            return format(v, '08b')
    except (ValueError, IndexError):
        pass
    return s


def canonicalize_answer(raw_output, family_hint=None):
    if raw_output is None:
        return '\\boxed{NOT_FOUND}'
    text = str(raw_output)
    body = _extract_last_boxed(text)
    if body is None:
        for pat in [r'The final answer is:\s*([^\n]+)', r'Final answer is:\s*([^\n]+)']:
            m = re.findall(pat, text, re.IGNORECASE)
            if m:
                body = m[-1].strip()
                break
    if body is None:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        body = lines[-1] if lines else 'NOT_FOUND'
    body = _strip_latex(body)
    body = _eval_frac(body)
    cleaned = body.strip()
    fam = (family_hint or '').strip().lower()
    if fam in {'bit_manipulation', 'bit'}:
        cleaned = _enforce_bit(cleaned)
    elif fam in {'text_encryption', 'cipher'}:
        cleaned = cleaned.strip().rstrip('.!?;,').lower()
    elif fam in {'numeral_system', 'numeral', 'roman'}:
        cleaned = cleaned.strip().rstrip('.!?;,')
    elif fam in {'gravity_constant', 'gravity', 'unit_conversion', 'unit'}:
        cleaned = _UNIT_RE.sub('', cleaned).strip()
        cleaned = _THOUSAND_RE.sub('', cleaned)
        cleaned = _SCI_RE.sub(lambda m: str(int(float(m.group(1)) * 10 ** int(m.group(2)))) if abs(float(m.group(1)) * 10 ** int(m.group(2))) < 1e16 and float(m.group(1)) * 10 ** int(m.group(2)) == int(float(m.group(1)) * 10 ** int(m.group(2))) else format(float(m.group(1)) * 10 ** int(m.group(2)), '.10g'), cleaned)
        cleaned = cleaned.strip()
    else:
        cleaned = _THOUSAND_RE.sub('', cleaned)
        cleaned = cleaned.strip()
    m = _TRAILING_DOT_ZERO_RE.match(cleaned.strip())
    if m:
        cleaned = m.group(1)
    cleaned = cleaned.replace('\u2212', '-')
    if fam in {'equation_transform', 'equation'}:
        return 'Final answer is: ' + cleaned + '\n\\boxed{' + cleaned + '}'
    return '\\boxed{' + cleaned + '}'


def detect_family_prompt(prompt):
    low = prompt.lower()
    if 'bit manipulation' in low or 'xor the bits' in low:
        return 'bit_manipulation'
    if 'decrypt the following text' in low or 'cipher' in low:
        return 'text_encryption'
    if 'numeral system' in low or 'roman numerals' in low:
        return 'numeral_system'
    if 'gravitational' in low or 'gravity constant' in low:
        return 'gravity_constant'
    if 'transformation rule' in low:
        return 'equation_transform'
    if 'unit conversion' in low or 'measurement' in low:
        return 'unit_conversion'
    if 'cryptarithm' in low:
        return 'cryptarithm'
    return 'unknown'


def extract_gt_answer(assistant_content):
    """Extract ground truth from assistant message."""
    m = re.search(r'\\boxed\{([^}]*?)\}', assistant_content)
    if m:
        return m.group(1).strip()
    return assistant_content.strip()[-200:]


KAGGLE_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def load_jsonl(path):
    """Load JSONL file with validation."""
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if isinstance(r.get('messages'), list) and len(r['messages']) >= 2:
                    out.append(r)
            except json.JSONDecodeError:
                continue
    return out


def run_prescore(model, tokenizer, holdout, n_samples, max_new_tokens, temperature, do_sample):
    """Run pre-score gate — identical logic to V102."""
    import torch
    from collections import defaultdict

    model.eval()
    # Handle gradient checkpointing
    was_grad_ckpt = True
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        was_grad_ckpt = False

    correct = 0
    per_family = defaultdict(lambda: {'correct': 0, 'total': 0})
    samples = holdout[:n_samples]
    n = len(samples)
    raw_outputs = []
    predictions = []
    ground_truths = []
    t0 = time.time()

    for i, ex in enumerate(samples):
        msgs = [dict(m) for m in ex['messages']]
        prompt_msgs = [m for m in msgs if m.get('role') != 'assistant']

        # Append Kaggle suffix (byte-exact kernel)
        for m in prompt_msgs:
            if m.get('role') == 'user' and KAGGLE_SUFFIX not in m['content']:
                m['content'] = m['content'].rstrip() + KAGGLE_SUFFIX
                break

        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
            )

        inputs = tokenizer(
            prompt_text, return_tensors='pt',
            truncation=True, max_length=4096 - max_new_tokens,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature, top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        raw = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        assistant_msgs = [m for m in msgs if m.get('role') == 'assistant']
        gt = extract_gt_answer(assistant_msgs[-1]['content']) if assistant_msgs else ''

        prompt_text_fam = ' '.join(m['content'] for m in prompt_msgs)
        family = detect_family_prompt(prompt_text_fam)

        cleaned = canonicalize_answer(raw, family)
        pred = extract_final_answer_official(cleaned)
        passed = verify_official(gt, pred)

        per_family[family]['total'] += 1
        if passed:
            correct += 1
            per_family[family]['correct'] += 1

        raw_outputs.append(raw[:500])
        predictions.append(pred)
        ground_truths.append(gt)

        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - t0
            eta = elapsed * (n - i - 1) / max(1, i + 1)
            print('  [' + str(i + 1) + '/' + str(n) + '] acc=' + format(correct / (i + 1), '.3f') + ' eta=' + format(eta / 60, '.1f') + 'm')

    # Restore gradient checkpointing
    if was_grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    score = correct / max(1, n)
    breakdown = {k: {'accuracy': round(v['correct'] / v['total'], 3) if v['total'] > 0 else 0,
                    'correct': v['correct'], 'total': v['total']}
                 for k, v in per_family.items()}

    return {
        'estimated_score': round(score, 4),
        'n_samples': n,
        'per_family_breakdown': breakdown,
        'elapsed_min': round((time.time() - t0) / 60, 1),
        'config': {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': do_sample,
            'canonicalize': True,
            'kaggle_suffix_appended': True,
        },
        'sample_predictions': [
            {'gt': gt, 'pred': p, 'raw_head': r[:150]}
            for gt, p, r in list(zip(ground_truths, predictions, raw_outputs))[:5]
        ],
    }


def analyze_calibration(result, actual_kaggle_score):
    """Compare estimated vs actual Kaggle score, suggest adjustments."""
    est = result['estimated_score']
    actual = actual_kaggle_score
    delta = est - actual
    abs_delta = abs(delta)

    analysis = {
        'estimated': est,
        'actual_kaggle': actual,
        'delta': round(delta, 4),
        'abs_delta': round(abs_delta, 4),
        'calibration_quality': None,
        'recommendations': [],
    }

    if abs_delta < 0.02:
        analysis['calibration_quality'] = 'EXCELLENT (within 2pp)'
    elif abs_delta < 0.05:
        analysis['calibration_quality'] = 'GOOD (within 5pp)'
    elif abs_delta < 0.10:
        analysis['calibration_quality'] = 'ACCEPTABLE (within 10pp)'
    else:
        analysis['calibration_quality'] = 'POOR (> 10pp off)'

    if delta > 0.10:
        analysis['recommendations'].append(
            'GATE TOO OPTIMISTIC (est ' + format(est, '.3f') +
            ' > actual ' + format(actual, '.3f') + ' by ' + format(delta, '.3f') +
            '). Possíveis causas: holdout leak, canonicalize muito agressivo, '
            'family detection errada. Ajustar threshold para est >= actual + 0.05 como buffer.'
        )
    elif delta < -0.10:
        analysis['recommendations'].append(
            'GATE TOO PESSIMISTIC (est ' + format(est, '.3f') +
            ' < actual ' + format(actual, '.3f') + ' by ' + format(abs(delta), '.3f') +
            '). Possíveis causas: max_new_tokens baixo, modelo não gera boxed completo. '
            'Aumentar max_new_tokens para 2048-3584, OR holdout set não representa test distribution.'
        )
    else:
        analysis['recommendations'].append(
            'GATE OK (delta ' + format(delta, '+.3f') + ' dentro de tolerância ±10pp).'
        )

    # Per-family analysis
    for fam, stats in result['per_family_breakdown'].items():
        if stats['total'] >= 5 and stats['accuracy'] < 0.3:
            analysis['recommendations'].append(
                'WEAK FAMILY: ' + fam + ' accuracy=' + format(stats['accuracy'], '.2f') +
                ' (n=' + str(stats['total']) + '). Investigar: canonicalize correto? family detection?'
            )

    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--adapter-dir', type=Path, required=True,
                        help='Diretório com adapter_config.json + adapter_model.safetensors')
    parser.add_argument('--dataset-path', type=Path, required=True,
                        help='JSONL com holdout (usa últimos N samples como holdout)')
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                        help='Kernel real usa 7680; 1024 é cap para speed em validation')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Kernel real usa 0.0 (greedy); match byte-exact')
    parser.add_argument('--do-sample', action='store_true', default=False)
    parser.add_argument('--base-model', default='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
    parser.add_argument('--output-json', type=Path, default=Path('./prescore_validation.json'))
    parser.add_argument('--actual-kaggle-score', type=float, default=None,
                        help='Score real do Kaggle (opcional) — computes delta')
    parser.add_argument('--hf-token', default=None,
                        help='HF token (default: env HF_TOKEN or HF_KEY)')
    parser.add_argument('--skip-nf4', action='store_true',
                        help='Force BF16 load (H100/A100 80GB only)')
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')
    if not hf_token:
        print('[WARN] No HF token — may fail for gated model')

    # Validate adapter
    adapter_cfg = args.adapter_dir / 'adapter_config.json'
    adapter_bin = args.adapter_dir / 'adapter_model.safetensors'
    if not adapter_cfg.exists() or not adapter_bin.exists():
        print('[FAIL] Adapter missing. Expected:')
        print('  ' + str(adapter_cfg))
        print('  ' + str(adapter_bin))
        return 1

    # Load dataset
    print('Loading dataset: ' + str(args.dataset_path))
    data = load_jsonl(args.dataset_path)
    print('Loaded ' + str(len(data)) + ' records')
    # Use last N as holdout (assume dataset was shuffled; last N may differ from training window)
    # For calibration, take last n_samples with a little shuffle seed
    import random
    rng = random.Random(42)
    rng.shuffle(data)
    holdout = data[-args.n_samples:] if len(data) > args.n_samples else data
    print('Holdout size: ' + str(len(holdout)))

    # Load model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print('\nLoading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_nf4 = gpu_mem_gb < 70 and not args.skip_nf4
    print('GPU VRAM: ' + str(round(gpu_mem_gb, 1)) + ' GB — NF4: ' + str(use_nf4))

    model_kwargs = dict(
        torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, token=hf_token,
    )
    if use_nf4:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print('\nLoading base model (may take 5-10 min)...')
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    print('Loaded in ' + format((time.time() - t0) / 60, '.1f') + ' min')

    # Load LoRA adapter
    print('\nLoading adapter from ' + str(args.adapter_dir) + '...')
    model = PeftModel.from_pretrained(model, str(args.adapter_dir), is_trainable=False)
    print('Adapter loaded. VRAM: ' + str(round(torch.cuda.memory_allocated() / 1e9, 1)) + ' GB')

    # Run pre-score
    print('\n=== Running pre-score gate ===')
    print('Config: N=' + str(args.n_samples) + ' max_new=' + str(args.max_new_tokens) +
          ' temp=' + str(args.temperature) + ' do_sample=' + str(args.do_sample))
    result = run_prescore(
        model, tokenizer, holdout,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    result['adapter_dir'] = str(args.adapter_dir)

    # Calibration
    if args.actual_kaggle_score is not None:
        calibration = analyze_calibration(result, args.actual_kaggle_score)
        result['calibration'] = calibration
        print('\n=== CALIBRATION ANALYSIS ===')
        print('Estimated: ' + format(calibration['estimated'], '.4f'))
        print('Actual Kaggle: ' + format(calibration['actual_kaggle'], '.4f'))
        print('Delta: ' + format(calibration['delta'], '+.4f'))
        print('Quality: ' + calibration['calibration_quality'])
        print('\nRecommendations:')
        for rec in calibration['recommendations']:
            print('  - ' + rec)

    # Save
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    print('\n[OK] Results saved: ' + str(args.output_json))
    print('\n=== FINAL SCORE ESTIMATE: ' + format(result['estimated_score'], '.4f') + ' ===')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
