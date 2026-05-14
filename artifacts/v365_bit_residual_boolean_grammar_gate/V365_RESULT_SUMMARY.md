# V365 bit residual boolean grammar gate

Status: bloqueado.

Objetivo: testar, em CPU, se a gramatica booleana per-output-bit poderia explicar os `9` residuos de `bit_manipulation` restantes apos o teacher V357 (`151/160`) sem gerar regressao.

Artefatos:

- Script: `scripts/analyze_v365_bit_residual_boolean_grammar_gate.py`
- Manifesto: `artifacts/v365_bit_residual_boolean_grammar_gate/20260514T_cpu_gate/v365_bit_residual_boolean_grammar_gate_manifest.json`
- Decisoes: `artifacts/v365_bit_residual_boolean_grammar_gate/20260514T_cpu_gate/v365_candidate_decisions.csv`
- Regras: `artifacts/v365_bit_residual_boolean_grammar_gate/20260514T_cpu_gate/v365_candidate_rules.csv`

Resultado medido:

- Entrada V357: `214/315`, `equation_transform=63/155`, `bit_manipulation=151/160`.
- Residuos de bit antes do gate: `9`.
- Mudancas candidatas: `73`.
- Ganhos candidatos: `0`.
- Perdas candidatas: `66`.
- Candidatos aceitos: `0`.
- Saida V365: `214/315`, `equation_transform=63/155`, `bit_manipulation=151/160`.

Decisao: nao rodar HF. A gramatica booleana per-bit livre e insegura neste contrato: ela muda muitas linhas corretas e nao resolve nenhum dos residuos V357. A proxima tentativa de bit precisa ser mais restrita, usando bit-pair/bitsum/stride ou um solver full-byte com prova de classe `0` perdas.
