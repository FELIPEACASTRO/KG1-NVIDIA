# V490 Debug Double Check - 2026-05-16

## Objetivo

Revalidar, em modo debug, as pecas que influenciam diretamente `loss`,
`eval_loss`, `ACC`, gate weak e decisao de FinOps depois do resultado V488.

## Estado Verificado

| Item | Resultado | Decisao |
|---|---:|---|
| V291/V290 baseline submit-safe | weak `192/315`, equation `56/155`, bit `136/160`, trunc `0` | baseline ativo |
| V488 checkpoint-10 | weak `191/315`, equation `57/155`, bit `134/160`, trunc `1` | bloqueado |
| Delta V488 vs V291/V290 | `+1` equation, `-2` bit, `+1` truncation | regressao real |
| HF jobs ativos | nenhum | sem custo pendente |

## Debug De Metricas

Comandos reexecutados:

- `python -m py_compile ...`
- `python scripts/audit_v449_acc_metric_integrity.py --self-test`
- `python scripts/hf_job_train_v90.py --self-test`
- `python scripts/kg1_static_safety_gate.py --self-test`
- `python scripts/kg1_static_safety_gate.py <critical files>`

Resultado:

- compilacao OK;
- self-tests OK;
- static safety gate OK;
- nenhum uso ativo de `answers_equivalent` como metrica oficial;
- thresholds ativos continuam `total>192`, `equation>56`, `bit>=136`,
  `truncated=0`.

Teste manual do extrator:

| Caso | Esperado |
|---|---|
| ultimo `\boxed{00000100}` | extrai `00000100`, correto |
| `\boxed{1010}` anterior e `\boxed{1011}` final | extrai `1011`, rejeita contra `1010` |
| simbolico escapado `\boxed{]\}\\!}` | extrai `]}\!`, correto |

Conclusao: o caminho de ACC nao esta inflando acertos por selecionar um boxed
anterior. A extracao expected-aware fica restrita ao ultimo boxed e serve apenas
para preservar payload simbolico com `}`, `{` ou `\`.

## Debug Do Resultado V488

Manifestos usados:

- `artifacts/v489_solution_integrity_audit/v489_v488_metric_integrity_v2_manifest.json`
- `artifacts/v489_solution_integrity_audit/v489_v488_vs_v290_diff_manifest.json`
- `artifacts/v489_solution_integrity_audit/v489_v488_vs_v290_row_diff.csv`

Diff confirmado:

| id | familia | Delta |
|---|---|---|
| `518deb39` | equation_transform | ganho real, V488 correto e baseline errado |
| `8740ed31` | bit_manipulation | regressao real |
| `59bee375` | bit_manipulation | regressao real e truncation |

Conclusao: o plateau nao esta vindo de erro simples de score. V488 realmente
transferiu um pequeno ganho de equation, mas comprou esse ganho com perda de
bit e truncation.

## Debug De Dataset

Dataset V390/V326 usado pela linha V487:

| Split | Rows | IDs unicos | Prompt dup | Familias |
|---|---:|---:|---:|---|
| train | 5031 | 5031 | 0 | bit `4231`, equation `800` |
| val | 532 | 532 | 0 | bit `332`, equation `200` |

Flags de contaminacao no metadata:

- `gate_rows_used_for_training=False` em todas as linhas;
- `weak_gate_rows_used_for_training=False` em todas as linhas;
- `full_gate_rows_used_for_training=False` em todas as linhas.

Tokenization gate V390/V326:

- offset masks: OK;
- fallback masks: `0`;
- prompt truncation: `0`;
- completion truncation: `0`;
- `boxed_suffix` validado.

Conclusao: nao apareceu sujeira simples no dataset V390/V326. O problema nao e
duplicidade, split leak obvio ou truncation no treino. A regressao aparece no
comportamento gerado pelo adapter.

## Debug De LoRA/Trainability

V487 corrigiu o alias estrutural de `target_parameters`, mas treinou somente:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `lm_head`

Os parametros MoE `mlp.experts.gate_up_proj` e `mlp.experts.down_proj` ficaram
ativos no forward, mas nao eram obrigados a ficar treinaveis naquela receita.
O script agora registra:

- `target_parameter_trainable_lora_tensors`;
- `target_parameter_trainable_lora_params`;
- `target_parameters_trainability_mode`;
- `require_lora_target_parameters_trainable`;
- `trainable_parameter_report_after_filter`.

Conclusao: a proxima tentativa ousada nao deve repetir V487. Ela deve testar
explicitamente uma variante `target_parameters_trainability_mode=trainable` com
kill-switch de ACC no primeiro checkpoint.

## Diagnostico Honesto

O que esta correto:

- metrica weak estrita;
- gate de thresholds;
- dataset V390/V326;
- tokenizacao;
- hash/row counts;
- preflight anti-gate-row;
- observabilidade de LoRA apos o patch.

O que ainda impede ganho:

1. O treino otimiza `loss/eval_loss`, mas o ganho que precisamos e row-level
   ACC em duas familias pequenas. O loss pode cair sem alterar as respostas
   finais corretas.
2. O ganho de equation e fragil: V477 e V488 chegaram a `equation=57`, mas
   ambos perderam bit ou truncation.
3. O adapter parece sensivel a mudancas pequenas de formato/decoding; portanto
   qualquer recipe sem micro-ACC no primeiro checkpoint e desperdicio.
4. O proximo caminho util e mexer no mecanismo de transferencia, nao procurar
   mais dados genericos.

## Proximo Caminho Ousado E Responsavel

1. Criar uma variante de smoke que treine tambem `up_proj/down_proj` LoRA
   associados a `target_parameters`, com
   `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`.
2. Limitar a execucao a checkpoint curto e abortar se:
   - total `<=192`;
   - `bit<136`;
   - `truncated>0`;
   - equation nao passar de `56`.
3. Se a variante der `equation=57` e bit `136`, continuar somente ate o proximo
   checkpoint. Se repetir `bit<136`, cancelar.
4. Se a variante falhar, abandonar broad SFT e voltar ao caminho CPU: minerar
   rows `518deb39`, `8740ed31`, `59bee375` e construir hard negatives
   especificos, nao dataset amplo.

## Decisao

Nao foi encontrado novo bug silencioso que explique sozinho o plateau. O bug
real anterior era observabilidade/trainability de `target_parameters`; ele foi
corrigido no codigo, mas V487/V488 provaram que isso sozinho nao basta.

O proximo experimento valido precisa ser mais ousado no conjunto treinavel
(`up_proj/down_proj` trainable), mas mais rigido em FinOps: micro-ACC no
primeiro checkpoint e cancelamento imediato se houver regressao.
