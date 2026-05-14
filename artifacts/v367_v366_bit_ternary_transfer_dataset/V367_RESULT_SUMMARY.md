# V367 V366 bit ternary transfer dataset

Status: dataset CPU aprovado por tokenization gate real. HF ainda exige smoke com kill-switch.

Objetivo: converter o teacher V366 (`222/315`, `bit=159/160`) em dataset adapter-only, sem repetir a rota V358/V361. O dataset prioriza as 8 regras novas `CHO`/`MAJ3` e usa replay reduzido das regras V357/V350.

Artefatos:

- Script: `scripts/build_v367_v366_bit_ternary_transfer_dataset.py`
- Manifesto: `artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_manifest.json`
- Train: `artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_train.jsonl`
- Validation: `artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_val.jsonl`
- Tokenization gate: `artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/tokenization_gate_real/v286_generic_tokenization_gate_manifest.json`

Dataset:

- Train rows: `1128`.
- Validation rows: `282`.
- V366 new rows: `768` train, `192` validation.
- V357 replay rows: `312` train, `78` validation.
- V350 replay rows: `48` train, `12` validation.
- Unique rules: `23`.
- Completion format: `boxed_only`.
- Train/validation prompt overlap: `0`.
- Weak reference id overlap: `0`.
- Weak reference prompt hash overlap: `0`.

Tokenization:

- Real tokenizer: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16@cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Mode: `boxed_only`.
- Train token max: `285`.
- Validation token max: `285`.
- Loss tokens: `15`.
- Prompt truncation rate: `0.0`.
- Completion tokens dropped: `0`.
- Offset masks: `1128/1128` train, `282/282` validation.
- Fallback masks: `0`.

Decisao: o dataset esta tecnicamente pronto para upload/HF smoke curto. O primeiro checkpoint deve ser cancelado se `total<=192`, `bit<136`, `equation<=56` ou truncation regressiva. O objetivo real e verificar se os ganhos V366 transferem para LoRA; nao usar full/package/submit antes do weak gate.
