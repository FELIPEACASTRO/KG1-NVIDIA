# V358 V357 Bit Ternary Transfer Dataset Summary

Generated: 2026-05-14

## Status

V358 dataset construction passed structural validation and real tokenization gate.

## Dataset

| Split | Rows | Family | Subcategory mix |
|---|---:|---|---|
| Train | `1152` | `bit_manipulation` | `832` ternary, `320` binary replay |
| Validation | `288` | `bit_manipulation` | `208` ternary, `80` binary replay |

## Hashes

- Train SHA256: `6881308c7e46167ea8752513dd6e986d14b39f04f661dbac8d9ed18d189f1a05`
- Validation SHA256: `d92f4bdf2e622be958ae09353bf3965d2a23e1e6fcea95fbd77c8bcbdf0b6b47`

## Anti-Leakage

- `id_overlap_with_reference=0`
- `prompt_sha256_overlap_with_reference=0`
- `train_val_prompt_overlap=0`

## Tokenization Gate

- Tokenizer: real `TokenizersBackend`, not toy.
- `prompt_truncation_rate=0.0`
- `completion_tokens_dropped=0`
- `fallback_masks=0`
- Train token max: `524`
- Validation token max: `524`

## Artifacts

- Manifest: `artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate/v358_v357_bit_ternary_transfer_manifest.json`
- Tokenization gate manifest: `artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate/tokenization_gate_real/v286_generic_tokenization_gate_manifest.json`

## Decision

V358 authorizes upload to HF and one V359 smoke run only. It does not authorize full eval, package, or Kaggle submit without adapter-only weak/full gain.
