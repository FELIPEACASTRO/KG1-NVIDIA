#!/usr/bin/env python3
"""Batch pre-score validation — roda gate em MÚLTIPLOS adapters + compara vs Kaggle.

Descoberta crítica: adapters com experts FULL (3GB) vs STRIPPED (~105MB) geram
outputs DIFERENTES porque Kaggle usa stripped. Pre-score local DEVE usar
stripped para matchar o que Kaggle vê.

Este script:
1. Descobre adapters disponíveis em /content/ e /content/drive
2. Strip cada adapter full (se necessário) para match Kaggle format
3. Carrega base model UMA vez (economia 5-10 min por adapter)
4. Para cada stripped adapter: roda pre-score + salva
5. Se user fornecer calibration_pairs.json (adapter → Kaggle score):
   - Computa delta por adapter
   - Constrói curva de calibração (regressão linear se >= 3 pares)
   - Recomenda ajuste V102

Uso:

    # Básico (só pre-score, sem calibration):
    python scripts/batch_validate_prescore.py \\
        --adapters-root /content \\
        --dataset-path /content/kg1/data/sft_v70_huikang_full.jsonl \\
        --n-samples 50 \\
        --output-dir /content/drive/MyDrive/prescore_validation

    # Com calibration pairs (JSON com adapter_name -> kaggle_score):
    python scripts/batch_validate_prescore.py \\
        --adapters-root /content \\
        --dataset-path /content/kg1/data/sft_v70_huikang_full.jsonl \\
        --calibration-pairs /content/kaggle_scores.json \\
        --output-dir /content/drive/MyDrive/prescore_validation

Formato calibration_pairs.json:
    {
        "/content/kg1_v80_stripped_v2": 0.50,
        "/content/kg1_v80_stripped_v3": 0.48,
        ...
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path


def find_adapters(root_paths):
    """Descobrir todos os diretórios com adapter_config.json + adapter_model.safetensors."""
    found = []
    for root in root_paths:
        if not Path(root).exists():
            continue
        for cfg_path in Path(root).rglob('adapter_config.json'):
            adapter_dir = cfg_path.parent
            bin_path = adapter_dir / 'adapter_model.safetensors'
            if not bin_path.exists():
                continue
            size_mb = bin_path.stat().st_size / 1024**2
            try:
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
                target_modules = cfg.get('target_modules', [])
                r = cfg.get('r', '?')
                alpha = cfg.get('lora_alpha', '?')
            except Exception:
                target_modules, r, alpha = [], '?', '?'
            # SHA256 of safetensors (first 12 chars for id)
            h = hashlib.sha256()
            with open(bin_path, 'rb') as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            sha = h.hexdigest()
            found.append({
                'adapter_dir': str(adapter_dir),
                'size_mb': round(size_mb, 1),
                'sha12': sha[:12],
                'r': r,
                'alpha': alpha,
                'target_modules': target_modules if isinstance(target_modules, list) else 'regex',
                'is_stripped': size_mb < 500,  # Kaggle limit proxy
            })
    return found


def strip_experts(src_dir, dst_dir):
    """Strip LoRA dos experts (remove .experts.N.up_proj + .down_proj)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(dst_dir, exist_ok=True)
    src_file = os.path.join(src_dir, 'adapter_model.safetensors')
    expert_re = re.compile(r'\.experts\.\d+\.(up_proj|down_proj)\.')

    kept_tensors = {}
    kept_keys = 0
    stripped_keys = 0
    with safe_open(src_file, framework='pt') as f:
        for k in f.keys():
            if expert_re.search(k):
                stripped_keys += 1
            else:
                kept_tensors[k] = f.get_tensor(k)
                kept_keys += 1

    dst_file = os.path.join(dst_dir, 'adapter_model.safetensors')
    save_file(kept_tensors, dst_file)

    # Copy adapter_config.json
    shutil.copy(
        os.path.join(src_dir, 'adapter_config.json'),
        os.path.join(dst_dir, 'adapter_config.json'),
    )

    return {
        'kept_keys': kept_keys,
        'stripped_keys': stripped_keys,
        'size_mb': os.path.getsize(dst_file) / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--adapters-root', nargs='+', default=['/content'],
                        help='Diretórios para buscar adapters (recursive)')
    parser.add_argument('--dataset-path', type=Path, required=True)
    parser.add_argument('--calibration-pairs', type=Path, default=None,
                        help='JSON com {adapter_dir: kaggle_score}')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Samples por adapter (50 = ~8 min em H100)')
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--auto-strip', action='store_true', default=True,
                        help='Strip experts automaticamente se adapter > 500MB')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base-model', default='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
    parser.add_argument('--max-adapters', type=int, default=10,
                        help='Max adapters to process (default 10)')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Discover adapters ===
    print('=== Step 1: Discovering adapters ===')
    adapters = find_adapters(args.adapters_root)
    print(f'Found {len(adapters)} adapters:')
    for a in adapters:
        t_mods = a['target_modules'][:3] if isinstance(a['target_modules'], list) else 'regex'
        print(f"  {a['adapter_dir']}")
        print(f"    {a['size_mb']:.1f}MB sha={a['sha12']} r={a['r']} alpha={a['alpha']} "
              f"stripped={a['is_stripped']}")

    if not adapters:
        print('[FAIL] No adapters found')
        return 1

    # Filter duplicates by SHA
    seen_sha = set()
    unique = []
    for a in adapters:
        if a['sha12'] not in seen_sha:
            seen_sha.add(a['sha12'])
            unique.append(a)
    print(f'\nUnique adapters by SHA: {len(unique)}')
    adapters = unique[:args.max_adapters]

    # === Step 2: Load calibration pairs ===
    calibration = {}
    if args.calibration_pairs and args.calibration_pairs.exists():
        calibration = json.loads(args.calibration_pairs.read_text(encoding='utf-8'))
        print(f'\nCalibration pairs loaded: {len(calibration)}')
        for k, v in calibration.items():
            print(f'  {k}: Kaggle score = {v}')

    # === Step 3: Strip adapters if needed ===
    processed_adapters = []
    for a in adapters:
        if a['size_mb'] > 500 and args.auto_strip:
            # Strip
            stripped_dir = args.output_dir / ('stripped_' + a['sha12'])
            print(f'\nStripping {a["adapter_dir"]} → {stripped_dir}')
            try:
                stats = strip_experts(a['adapter_dir'], stripped_dir)
                print(f'  kept={stats["kept_keys"]} stripped={stats["stripped_keys"]} '
                      f'size={stats["size_mb"]:.1f}MB')
                processed_adapters.append({
                    **a,
                    'effective_dir': str(stripped_dir),
                    'effective_size_mb': stats['size_mb'],
                    'was_stripped_now': True,
                })
            except Exception as e:
                print(f'  [FAIL] {e}')
                continue
        else:
            processed_adapters.append({
                **a,
                'effective_dir': a['adapter_dir'],
                'effective_size_mb': a['size_mb'],
                'was_stripped_now': False,
            })

    # === Step 4: Load base model ONCE ===
    print('\n=== Step 4: Load base model ===')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HF_KEY')

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_nf4 = gpu_mem_gb < 70
    print(f'GPU VRAM: {gpu_mem_gb:.1f}GB — NF4: {use_nf4}')

    model_kwargs = dict(
        torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, token=hf_token,
    )
    if use_nf4:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )

    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    print(f'Base model loaded in {(time.time()-t0)/60:.1f} min')

    # === Step 5: Load dataset ===
    print('\n=== Step 5: Load holdout ===')
    data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if isinstance(r.get('messages'), list) and len(r['messages']) >= 2:
                        data.append(r)
                except json.JSONDecodeError:
                    continue

    import random
    rng = random.Random(42)
    rng.shuffle(data)
    holdout = data[:args.n_samples]
    print(f'Holdout: {len(holdout)} samples')

    # Import the validate functions from the single-adapter script
    sys.path.insert(0, str(Path(__file__).parent))
    from validate_prescore_gate_calibration import (
        run_prescore, analyze_calibration, extract_final_answer_official,
        verify_official, canonicalize_answer, detect_family_prompt,
    )

    # === Step 6: Run pre-score for each adapter ===
    print('\n=== Step 6: Pre-score each adapter ===')
    all_results = []

    for i, a in enumerate(processed_adapters):
        print(f'\n--- Adapter {i+1}/{len(processed_adapters)}: {a["effective_dir"]} ---')
        try:
            # Attach adapter
            model = PeftModel.from_pretrained(
                base_model, a['effective_dir'], is_trainable=False,
            )
            print(f'  Adapter loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB')

            result = run_prescore(
                model, tokenizer, holdout,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
            result['adapter_info'] = a

            # Match with calibration (try both original and effective dirs)
            kaggle_score = calibration.get(a['adapter_dir']) or calibration.get(a['effective_dir'])
            if kaggle_score is not None:
                calib = analyze_calibration(result, kaggle_score)
                result['calibration'] = calib
                print(f'  Estimated: {result["estimated_score"]:.4f}')
                print(f'  Kaggle:    {kaggle_score:.4f}')
                print(f'  Delta:     {calib["delta"]:+.4f}')
                print(f'  Quality:   {calib["calibration_quality"]}')
            else:
                print(f'  Estimated: {result["estimated_score"]:.4f} (no Kaggle reference)')

            all_results.append(result)

            # Unload adapter for next iter
            model.unload()
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f'  [FAIL] {e}')
            all_results.append({
                'adapter_info': a,
                'error': str(e)[:500],
            })

    # === Step 7: Consolidated report ===
    print('\n=== Step 7: Consolidated Report ===')
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'n_adapters': len(processed_adapters),
        'n_samples_per_adapter': args.n_samples,
        'max_new_tokens': args.max_new_tokens,
        'adapter_results': all_results,
    }

    # Calibration curve if >= 2 points with Kaggle scores
    cal_points = [
        (r['estimated_score'], r['calibration']['actual_kaggle'])
        for r in all_results
        if 'calibration' in r
    ]
    if len(cal_points) >= 2:
        # Simple linear regression
        n = len(cal_points)
        xs, ys = zip(*cal_points)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in cal_points)
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den > 0 else 1.0
        intercept = mean_y - slope * mean_x
        # R²
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in cal_points)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        summary['calibration_curve'] = {
            'n_points': n,
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'r_squared': round(r2, 4),
            'formula': f'kaggle_score ≈ {slope:.4f} × estimated + {intercept:.4f}',
            'points': [(x, y) for x, y in cal_points],
        }
        print(f'\nCalibration curve ({n} points):')
        print(f'  kaggle ≈ {slope:.4f} * estimated + {intercept:.4f}')
        print(f'  R² = {r2:.4f}')
        print()
        print(f'Interpretation:')
        if abs(slope - 1.0) < 0.1 and abs(intercept) < 0.05:
            print('  [EXCELLENT] Gate matches Kaggle (slope ≈ 1, intercept ≈ 0)')
        elif slope < 1.0:
            print(f'  Gate is OPTIMISTIC by factor {1.0 - slope:.2f}')
            print(f'  To correct: Kaggle = gate × {slope:.3f} + {intercept:.3f}')
        else:
            print(f'  Gate is PESSIMISTIC by factor {slope - 1.0:.2f}')

    # Save
    with open(args.output_dir / 'batch_validation.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f'\n[OK] Saved: {args.output_dir / "batch_validation.json"}')

    # Table summary
    print('\n=== Summary Table ===')
    print(f'{"Adapter":40s} {"Size":>8s} {"Est":>8s} {"Kaggle":>8s} {"Delta":>8s}')
    print('-' * 76)
    for r in all_results:
        if 'error' in r:
            name = Path(r['adapter_info']['effective_dir']).name[:39]
            print(f'{name:40s} ERROR: {r["error"][:30]}')
            continue
        name = Path(r['adapter_info']['effective_dir']).name[:39]
        size = r['adapter_info']['effective_size_mb']
        est = r['estimated_score']
        if 'calibration' in r:
            kag = r['calibration']['actual_kaggle']
            delta = r['calibration']['delta']
            print(f'{name:40s} {size:>7.1f}M {est:>8.3f} {kag:>8.3f} {delta:>+8.3f}')
        else:
            print(f'{name:40s} {size:>7.1f}M {est:>8.3f} {"-":>8s} {"-":>8s}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
