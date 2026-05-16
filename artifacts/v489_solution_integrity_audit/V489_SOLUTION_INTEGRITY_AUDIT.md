# V489 Solution Integrity Audit

Data: 2026-05-16

## Escopo

Auditoria dos pontos que influenciam `loss`, `eval_loss`, weak ACC e decisao de
submit na linha V487/V488:

- scripts de treino, weak eval, batch eval e metrica;
- gates estaticos e pre-paid;
- artefatos pequenos da weak eval V488;
- diagramas SVG exportados em `Downloads`;
- consistencia do roadmap com os resultados medidos.

## Resultado Medido

| Metrica | V290/V291 baseline | V488 ckpt-10 | Delta |
|---|---:|---:|---:|
| Total strict weak | 192/315 | 191/315 | -1 |
| equation_transform | 56/155 | 57/155 | +1 |
| bit_manipulation | 136/160 | 134/160 | -2 |
| truncated | 0 | 1 | +1 |

Decisao: V488 continua bloqueado. O ganho de `equation_transform` nao e
submit-safe porque veio com regressao em `bit_manipulation` e uma nova
truncation.

## Integridade Da Metrica

`scripts/audit_v449_acc_metric_integrity.py` confirmou que o caminho oficial de
ACC esta correto quando usa `src.competition_utils.verify_answer`.

Achado importante: a metrica permissiva `answers_equivalent` superestima
binarios e so pode permanecer diagnostica. No par V290/V488 ela sobe para
`206/315`, mas esse numero nao e valido para gate nem submit.

Achado V489 adicional: a extracao expected-aware antiga podia examinar todos os
`\boxed{}` e escolher um payload anterior se ele batesse com o gabarito. Isso
era leakage de avaliacao. A correcao limita a desambiguacao ao ultimo
`\boxed{}` e usa `verify_answer` estrito. No V488 real, o delta esperado continua
`+1` sobre extracao simples por causa do payload simbolico `]}\!`, mas agora
esse ganho so e permitido quando esta no ultimo boxed.

## Diff Linha A Linha V488 vs V290

Mudancas reais encontradas:

| Familia | id | Tipo | Observacao |
|---|---|---|---|
| equation_transform | `518deb39` | ganho | V488 extraiu `$`, baseline errou |
| bit_manipulation | `8740ed31` | regressao | V488 escolheu `default 1` onde baseline acertava |
| bit_manipulation | `59bee375` | regressao + truncation | V488 repetiu padrao ate `finish_reason=length` |

Artefatos:

- `v489_v488_vs_v290_diff_manifest.json`
- `v489_v488_vs_v290_row_diff.csv`
- `V489_V488_VS_V290_WEAK_DIFF.md`
- `v489_v488_metric_integrity_manifest.json`

## SVGs

Os SVGs exportados representam corretamente as pecas principais da arquitetura:
launcher HF, gates, datasets, seed adapter, treino, weak eval, full eval, package
e submit.

Gap: os SVGs em `Downloads` estao estaticos e ficaram atrasados em relacao ao
`.md` versionado depois de V488. O fonte versionado atualizado e:

- `artifacts/roadmaps/KG1_SOLUTION_ARCHITECTURE_MERMAID_2026_05_16.md`

Regra: tratar SVG exportado como visualizacao, nao como fonte de verdade. A
fonte de verdade e o Mermaid versionado na branch.

## F2 Backfired / Silent Bug Encontrado

V487 provou que `target_parameters` existiam no adapter e eram carregados pela
rota PEFT nativa, mas o manifesto de treino nao registrava se esses parametros
MoE ficavam treinaveis.

Na configuracao V487:

- `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`;
- `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,lm_head`.

Portanto os LoRA de `up_proj/down_proj` podiam estar ativos no forward, mas
congelados no treino. Isso torna ambigua a leitura "target_parameters corrigidos"
como se significasse "target_parameters treinados".

Correcao implementada:

- `scripts/hf_job_train_v90.py` agora registra:
  - `target_parameter_trainable_lora_tensors`;
  - `target_parameter_trainable_lora_params`;
  - `target_parameters_trainability_mode`;
  - `require_lora_target_parameters_trainable`;
  - `trainable_parameter_report_after_filter`.
- `scripts/kg1_static_safety_gate.py` agora exige essa observabilidade no treino.
- Launchers que combinam MoE `target_parameters` com allowlist de LoRA precisam
  declarar explicitamente `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` ou `1`.
- O launcher V487 foi marcado explicitamente como `0`, porque era uma receita
  frozen-active.

## Static Gate

Achado V489 adicional: `scripts/kg1_static_safety_gate.py` tinha uma chave
duplicada para `scripts/hf_job_train_v90.py` em `CRITICAL_SNIPPETS`. Em Python,
a segunda entrada sobrescrevia a primeira e anulava checks de alias/trainability.

Correcao implementada:

- os snippets foram consolidados em uma unica entrada;
- o static gate passou a exigir a regra de extracao expected-aware sem leakage;
- o self-test cobre a exigencia de declarar
  `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE`.

## Dataset/Contamination Gate

Achado V489 adicional: o preflight SFT auditava parse, IDs, families e
assistant messages, mas nao registrava flags de contaminacao
`gate_rows_used_for_training`, `weak_gate_rows_used_for_training` e
`full_gate_rows_used_for_training`.

Correcao implementada:

- `scripts/hf_job_preflight_gate.py` agora conta flags presentes/ausentes;
- se qualquer flag presente nao for `false`, o preflight falha antes de treino;
- `scripts/kg1_static_safety_gate.py` exige esse check.

## Kaggle/External Check

O Kaggle CLI atual expõe arquivos, kernels, leaderboard e submissions, mas nao
expõe comments/discussion replies. A pagina HTML publica dos discussions carrega
conteudo via app e nao retornou posts no HTML simples.

Achado externo util e verificavel por CLI/web: dataset publico
`kishanvavdara/nemotron-reasoning-traj`, com `nemotron_traj.csv` de 9.500
trajetorias extraidas de `tonghuikang/nemotron`.

Uso permitido no roadmap: analisar como fixture/error analysis e minerar
padroes de bit/equation com anti-leakage. Nao usar direto para treino ou submit
sem gate de origem, overlap, familia, prompt hash e licenca.

## Decisao

Nao existe bug confirmado no calculo de ACC. O plateau vem de transferencia:
`eval_loss` melhora, mas o comportamento adapter-only nao incorpora os ganhos de
solver/verifier sem derrubar bit ou formato.

Proximo passo correto: antes de qualquer novo H200, rodar CPU gate/dry-run que
prove a intencao de trainability dos `target_parameters` e bloqueie receitas que
repitam `equation +1` com `bit -1/-2` ou truncation.
