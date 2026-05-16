# V510 Canonical Dataset Triage

Gerado em 2026-05-16.

Objetivo: reduzir a bagunca de datasets de treino para um unico pool ativo,
sem misturar fontes ruins, historicas ou ja bloqueadas. A decisao nao foi
concatenar tudo. O V510 inclui apenas fontes que passaram no V509 e ainda fazem
sentido para a rota atual.

## Resultado

| Item | Valor |
|---|---:|
| fontes auditadas | 20 |
| fontes incluidas | 6 |
| fontes excluidas | 14 |
| train final | 2627 |
| val final | 637 |
| duplicados removidos do train | 543 |
| duplicados removidos do val | 155 |
| reaudit V509 do V510 | passou, 0 bloqueios |
| tokenization gate real local | passou, token max 331, truncation 0 |

Arquivos:

- `v510_canonical_active_training_pool_train.jsonl`
- `v510_canonical_active_training_pool_val.jsonl`
- `v510_canonical_active_training_pool_manifest.json`
- `v510_canonical_active_training_pool_source_decisions.csv`
- `v509_reaudit/v510_canonical_reaudit_manifest.json`
- `tokenization_gate_real_local/v286_generic_tokenization_gate_manifest.json`

## Decisao Sobre Datasets Com Problema

| Dataset | Problema | Ajustar ou excluir? | Motivo |
|---|---|---|---|
| V439 final-answer-only pairs | 26 mismatches no train e 5 no val quando extraido por `extract_final_answer` label-free; casos com `\\`, `\{`, `\}` e respostas simbolicas escapadas | excluir agora | corrigir exigiria uma nova convencao symbol-safe de resposta e revalidacao de metrica; o dataset ja falhou transferencia em V440/V441 |
| V443 certified equation pair builder | arquivos train/val vazios; manifesto diz `pairs=0`, `candidate_rows=0`, `hf_gpu_allowed=false` | excluir | nao existe dado para corrigir; precisa de novo builder CPU, nao patch em dataset vazio |
| V293 v274 distill | todas as linhas usam resposta final unboxed | excluir | incompatibilidade com receita atual de boxed answer e dataset antigo amplo |
| V390/V406/V410/V416 | estruturalmente limpos, mas historicos e ja falharam transferencia | excluir | incluir de novo dilui sinal e repete caminhos ja bloqueados por FinOps |

Conclusao: V439 nao deve ser "consertado" dentro do pool atual. Ele so volta se
for reconstruido como um dataset novo, com renderizacao symbol-safe validada por
extracao label-free antes de qualquer treino. V443 nao tem conteudo e deve
permanecer fora.

## Fontes Mantidas

| Fonte | Papel |
|---|---|
| V498 numeric teacher trace pack | fonte principal atual: hard negatives numericos e replay de bit |
| V475 equation bit replay mix | fonte CPU-gated limpa para equation/bit |
| V460 numeric one-rule micro dataset | micro replay limpo para regra de sinal numerico |

## Proximo Gate Obrigatorio

Antes de qualquer HF job:

1. Usar apenas o V510 como dataset ativo.
2. Repetir tokenization/offset-mask gate no ambiente final do job se a imagem
   HF mudar.
3. Rodar pre-paid integration gate apontando explicitamente para V510.
4. Bloquear launch se qualquer linha voltar a ter mismatch de resposta,
   overlap de referencia, duplicidade conflitiva ou anti-leak flag verdadeira.
