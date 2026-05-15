# V450 Transfer Debug Audit

Data: 2026-05-15

## Objetivo

Depurar se ainda existe erro de logica, negocio, simbolico, sintaxe ou metrica
nos caminhos que medem ACC e decidem promocao para `bit_manipulation` e
`equation_transform`.

## Resultado Curto

Nao foi encontrado erro ativo no caminho submit-safe de ACC. O caminho de
promocao esta usando `verify_answer`, que exige match exato para respostas
binarias. O erro encontrado era diagnostico: `answers_equivalent` inflava bit
quando usado para `official_correct`. Esse erro foi corrigido, bloqueado no
static gate e revalidado pelo V449.

## Componentes Auditados

| Componente | Check | Resultado |
|---|---|---|
| `src/competition_utils.py::verify_answer` | `[01]+` deve ser exact-match | passou |
| `src/competition_utils.py::verify_answer` | numero nao-binario aceita tolerancia 1% | passou |
| `src/competition_utils.py::verify_answer` | simbolico nao numerico e case-insensitive | passou |
| `scripts/evaluate_lora_adapter.py` | coluna `correct` usa `verify_answer` | passou |
| `scripts/evaluate_lora_adapters_batch.py` | batch weak/full usa `verify_answer` | passou |
| `scripts/analyze_eval_predictions.py` | `official_correct`, `first_boxed_correct` e early windows usam `verify_answer` | passou apos correcao |
| `scripts/hf_job_weak_eval_v245.py` | valida CSV weak por SHA, row count, family counts e shared contract antes de eval | passou por leitura/auditoria |
| `scripts/kg1_static_safety_gate.py` | bloqueia `official_correct` com `answers_equivalent` | passou |
| `scripts/audit_v449_acc_metric_integrity.py` | agora tem `--self-test` executavel | passou |

## Evidencia Executada

Comandos executados:

```text
python -m py_compile scripts\analyze_eval_predictions.py scripts\kg1_static_safety_gate.py scripts\audit_v449_acc_metric_integrity.py scripts\hf_job_weak_eval_v245.py scripts\evaluate_lora_adapter.py scripts\evaluate_lora_adapters_batch.py
python scripts\kg1_static_safety_gate.py --self-test
python scripts\audit_v449_acc_metric_integrity.py --self-test
python scripts\audit_v449_acc_metric_integrity.py --prediction-csv artifacts\v342_acc_first_diagnostic\v290_checkpoint6_baseline_predictions.csv --prediction-csv artifacts\v405_integrated_solver_projection\20260514T_v405_integrated_projection\v405_integrated_solver_predictions.csv --output-dir artifacts\v449_acc_metric_integrity_audit\20260515T_final_debug_check --label v449_final_debug_check
python scripts\kg1_static_safety_gate.py scripts\analyze_eval_predictions.py scripts\kg1_static_safety_gate.py scripts\audit_v449_acc_metric_integrity.py scripts\hf_job_weak_eval_v245.py scripts\evaluate_lora_adapter.py scripts\evaluate_lora_adapters_batch.py --output-json artifacts\v449_acc_metric_integrity_audit\20260515T_final_debug_check\v449_final_debug_static_gate_report.json
python scripts\kg1_static_safety_gate.py scripts\audit_v449_acc_metric_integrity.py --output-json artifacts\v449_acc_metric_integrity_audit\20260515T_final_debug_check\v449_selftest_static_gate_report.json
```

Artefatos:

- `artifacts/v449_acc_metric_integrity_audit/20260515T_final_debug_check/v449_final_debug_check_manifest.json`
- `artifacts/v449_acc_metric_integrity_audit/20260515T_final_debug_check/v449_final_debug_static_gate_report.json`
- `artifacts/v449_acc_metric_integrity_audit/20260515T_final_debug_check/v449_selftest_static_gate_report.json`

## Achado Principal

O V449 final debug confirma que a metrica permissiva inflaria `bit_manipulation`
se fosse usada para promocao:

| CSV | Coluna | Strict correct | Permissive correct | Inflacao |
|---|---|---:|---:|---:|
| V290 baseline weak | `prediction` | `192/315` | `206/315` | `+14` |
| V405 integrated projection | `prediction` | `192/315` | `206/315` | `+14` |
| V405 integrated projection | `integrated_prediction` | `201/315` | `213/315` | `+12` |

Todas as divergencias observadas vieram de `bit_manipulation`, por exemplo
respostas binarias quase iguais que seriam aceitas por tolerancia numerica, mas
devem ser rejeitadas pelo scorer estrito.

## Impacto Para O Roadmap

1. O baseline real continua `192/315`, `equation=56/155`,
   `bit=136/160`, `truncated=0`.
2. V448 real continua reprovado: `190/315`, `equation=56/155`,
   `bit=134/160`, `truncated=1`.
3. Nao ha evidencia de que a falta de ganho venha de erro de ACC no weak gate.
4. O gargalo ativo e transferencia para adapter-only, nao calculo de score.
5. Proxima acao deve ser CPU DSL/parser/raw-output, nao mais H200 SFT da mesma
   classe.

## Regras Preventivas

- `answers_equivalent` e diagnostico-only.
- `official_correct`, weak/full gate, promotion report e submit decision devem
  usar `verify_answer`.
- Qualquer script novo/alterado que mexa em ACC deve passar
  `scripts/kg1_static_safety_gate.py`.
- Se um auditor tem papel de gate, ele deve ter `--self-test`.
- Para bit, todo gabarito `[01]+` deve ser exact-match, inclusive casos de
  `equation_transform` cujo answer pareca binario (`101`, `100`).
