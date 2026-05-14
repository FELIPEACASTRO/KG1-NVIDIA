# V366 bit full-byte ternary operator gate

Status: aprovado como teacher CPU. Nao e submit direto.

Objetivo: testar uma rota mais restrita que a gramatica per-bit livre: uma unica expressao ternaria full-byte precisa explicar todos os exemplos, e a promocao e feita por familia de operador ternario.

Artefatos:

- Script: `scripts/analyze_v366_bit_fullbyte_ternary_op_gate.py`
- Manifesto: `artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_bit_fullbyte_ternary_op_gate_manifest.json`
- Decisoes: `artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_candidate_decisions.csv`
- Regras: `artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_candidate_rules.csv`
- Predicoes integradas: `artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv`

Resultado medido:

- Entrada V357: `214/315`, `equation_transform=63/155`, `bit_manipulation=151/160`.
- Candidate changes: `9`.
- Candidate gains: `8`.
- Candidate losses: `1`.
- Regras aceitas:
  - `bit_fullbyte_ternary_op_CHO`: `4` ganhos, `0` perdas.
  - `bit_fullbyte_ternary_op_MAJ3`: `4` ganhos, `0` perdas.
- Regra rejeitada:
  - `bit_fullbyte_ternary_op_AND_OR`: `0` ganhos, `1` perda.
- Saida V366: `222/315`, `equation_transform=63/155`, `bit_manipulation=159/160`.
- Accepted losses: `0`.

IDs aceitos:

- `1abaffca`
- `b8722d19`
- `7192535b`
- `1a7c8520`
- `a6192d29`
- `048cc279`
- `b8aa3072`
- `5ba26f21`

Decisao: construir V367 como dataset de transferencia pequeno, com `CHO`/`MAJ3` aceitos, replay forte de bit e hard negatives. HF continua bloqueado ate o dataset passar anti-leakage, tokenization/offset-mask e smoke com kill-switch.
