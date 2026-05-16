# KG1 Problem Map

Atualizado: 2026-05-15

Este documento mostra onde a solucao esta travando, quais evidencias temos e
por que a proxima acao nao deve ser "treinar mais", mas sim mudar o objetivo ou
o dado antes de gastar GPU novamente.

## Resumo Executivo

O problema atual nao e infraestrutura, GPU, hash, tokenizacao ou carregamento do
adapter. Esses pontos foram validados. O problema esta no **sinal de aprendizado
que chega ao LoRA**.

Estado real submit-safe:

| Metrica | Melhor adapter-only atual |
|---|---:|
| weak total | `192/315` |
| `equation_transform` | `56/155` |
| `bit_manipulation` | `136/160` |
| truncation | `0` |

O que ja sabemos:

- `bit_manipulation` ja esta forte no weak: `136/160`.
- `equation_transform` e o gargalo principal: `56/155`.
- V436/V436B/V440 provaram que preference `mean_nll` nao esta gerando ganho
  submit-safe.
- V441 provou que focar o score somente no payload do `\boxed{...}` tambem nao
  gera sinal no primeiro checkpoint.
- V442 provou que os `133` pares V439 sao source-ok, mas `0/133` possuem
  certificado de regra label-free. O problema deixou de ser formato/loss e
  virou falta de regra congelada transferivel para o adapter.
- Solver/verifier/teacher mostrou potencial, mas esse ganho ainda nao foi
  convertido para adapter-only.
- Apos V441/V442, novo smoke defensavel exige dado novo com certificado CPU,
  nao apenas troca de loss, LR, epoch ou H200.

## Desenho Simples Das Pecas Principais

Esta visao mostra somente as pecas centrais. O ponto critico esta entre
`conhecimento verificado` e `LoRA adapter`: hoje conseguimos encontrar respostas
potenciais fora do adapter, mas ainda nao conseguimos transformar isso em pesos
que melhorem o weak gate sem regressao.

```mermaid
flowchart LR
  A[Dados e artefatos\ntrain, weak, logs, adapters] --> B[Baseline adapter-only\nV290/V226 lineage]
  B --> C[Weak gate por familia]
  C --> C1[bit_manipulation\n136/160]
  C --> C2[equation_transform\n56/155]
  C --> C3[total weak\n192/315]

  D[Conhecimento externo\nKaggle discussions, Tong, HF datasets,\nOpenRouter, papers] --> E[Solver / verifier / DSL]
  E --> F[Acertos potenciais\nex.: equation ate 60\nbit maior com bit-pair]

  F --> G{Conversao para LoRA}
  G --> H[Preference/SFT antigo\nmean-NLL sequencial]
  H --> I[Sem ganho medido\nV436/V436B/V440]

  G --> J[Proxima rota\nCPU gate solver/DSL]
  J --> K[Novo adapter candidato]
  K --> C

  C2 --> P[PROBLEMA PRINCIPAL:\nequation nao sobe no adapter]
  I --> Q[PROBLEMA SECUNDARIO:\nloss cai, ACC nao melhora]

  classDef ok fill:#ddf4ff,stroke:#0969da,stroke-width:2px,color:#111;
  classDef warn fill:#fff4cc,stroke:#9a6700,stroke-width:2px,color:#111;
  classDef bad fill:#ffdddd,stroke:#b00020,stroke-width:2px,color:#111;
  class C1,C3,F,J,K ok;
  class G,H warn;
  class C2,I,P,Q bad;
```

## Desenho Do Problema

```mermaid
flowchart TD
  A[Objetivo final: subir ranking Kaggle] --> B[Precisa virar adapter-only]
  B --> C[Melhor adapter atual]
  C --> C1[weak 192/315]
  C --> C2[bit 136/160]
  C --> C3[equation 56/155]

  C3 --> D{Onde buscar ganho?}
  D --> E[Mais SFT / mais epochs / LR sweep]
  D --> F[Preference hard negatives]
  D --> G[Solver / verifier / DSL]
  D --> H[Bit-pair / bitsum / stride]

  E --> E1[Ja testado varias vezes]
  E1 --> E2[Loss cai, ACC nao sobe]
  E2 --> P1[PROBLEMA: loss nao e proxy de ACC]

  F --> F1[V435E misto]
  F1 --> F2[133 hard negatives + 67 format-only]
  F2 --> F3[V436 piorou 6/40 -> 5/40]
  F3 --> P2[PROBLEMA: dado contaminado]

  F --> F4[V436B hard-negative-only]
  F4 --> F5[133 hard negatives limpos]
  F5 --> F6[checkpoint-3 piorou 6/24 -> 4/24]
  F6 --> P3[PROBLEMA: objetivo mean-NLL ainda desalinhado]

  F --> F7[V438 audit]
  F7 --> F8[chosen mencionava resposta errada 123/133]
  F8 --> F9[chosen mencionava label audit 133/133]
  F9 --> P4[PROBLEMA: target ensinava texto errado]

  F --> F10[V439 final-answer-only]
  F10 --> F11[target limpo: 0 contaminantes]
  F11 --> F12[V440 H200]
  F12 --> F13[baseline 8/24, ckpt-3 8/24]
  F13 --> P5[PROBLEMA: limpar target foi necessario, mas nao suficiente]
  P5 --> F14[V441 proposto]
  F14 --> F15[score so no boxed payload]

  G --> G1[Teacher/probes mostram potencial]
  G1 --> G2[Exemplo: equation pode chegar a 60 via verifier]
  G2 --> P6[PROBLEMA: ganho ainda nao foi destilado para LoRA]

  H --> H1[Bit ja em 136/160]
  H1 --> H2[Ganhos maiores exigem algoritmo/trace tipo Tong Hui Kang]
  H2 --> P7[PROBLEMA: preservar bit enquanto mexe em equation]

  P1 --> R[Proxima rota correta]
  P2 --> R
  P3 --> R
  P4 --> R
  P5 --> R
  P6 --> R
  P7 --> R

  R --> R1[CPU gate de equation DSL/solver]
  R --> R2[Objetivo focado no boxed payload]
  R --> R3[Novos pares so se solver acertar e baseline errar]
  R --> R4[GPU so se gate CPU provar +4 equation sem queda de bit]

  classDef problem fill:#ffdddd,stroke:#b00020,stroke-width:2px,color:#111;
  classDef route fill:#ddf4ff,stroke:#0969da,stroke-width:2px,color:#111;
  class P1,P2,P3,P4,P5,P6,P7 problem;
  class R,R1,R2,R3,R4 route;
```

## Onde Estamos Com Problema

### 1. `equation_transform` e o gargalo real

O weak atual mostra:

- `bit_manipulation`: `136/160`, no piso submit-safe atual de `136`.
- `equation_transform`: `56/155`, abaixo da meta operacional de `60`.
- Total: `192/315`, ainda sem margem para promover full/package com seguranca.

Isso significa que o ranking nao vai subir apenas preservando bit. Precisamos
de pelo menos alguns acertos novos em equation, sem perder bit.

### 2. Loss baixo nao resolveu a metrica que importa

Observacao acumulada:

- varios treinos reduziram `train_loss` ou `eval_loss`;
- os acertos de `equation_transform` ficaram presos em torno de `56`;
- algumas variantes ainda derrubaram bit.

Conclusao: `loss` serve para detectar treino numericamente saudavel, mas nao
serve para promover checkpoint. O gate real segue sendo weak/full por familia.

### 3. Preference hard-negative falhou em duas formas

#### V436: dataset misto

O V435E antigo tinha:

- `133` hard negatives;
- `67` format-only negatives.

Resultado V436:

| Metrica interna | Baseline | Checkpoint inicial |
|---|---:|---:|
| preference total | `6/40` | `5/40` |
| equation | `4/22` | `3/22` |

Diagnostico: o dataset misturava negativos semanticamente errados com negativos
apenas de formato. Isso contaminava o sinal.

#### V436B: hard-negative-only

Corrigimos para apenas hard negatives:

- `133` pares;
- `120` equation;
- `13` bit.

Resultado:

| Metrica interna | Baseline | Checkpoint-3 |
|---|---:|---:|
| preference total | `6/24` | `4/24` |
| equation | `4/22` | `2/22` |
| bit | `2/2` | `2/2` |

Diagnostico: mesmo com dado sem format-only, o objetivo `mean_nll` ainda moveu
equation na direcao errada.

### 4. O target escolhido tambem estava contaminado

V438 audit encontrou:

| Check | Resultado |
|---|---:|
| labels boxed semanticamente corretos | `133/133` |
| rejected igual ao adapter wrong | `133/133` |
| chosen menciona adapter prediction errado | `123/133` |
| chosen menciona public-train label audit | `133/133` |

Diagnostico: o label final estava correto, mas o texto de treino ensinava coisas
ruins junto: mencionava auditoria e repetia a resposta errada.

### 5. V439 corrigiu o texto, mas V440 mostrou que isso nao basta

V439 final-answer-only removeu a contaminacao:

- chosen: `Final answer: \boxed{ANSWER}`;
- rejected: `Final answer: \boxed{ADAPTER_WRONG}`;
- `chosen_mentions_adapter_prediction_rows=0`;
- `chosen_mentions_public_train_label_audit_rows=0`.

V440 H200 validou toda a infraestrutura:

- integration gate local/remoto OK;
- HF dataset correto e hashes batendo;
- tokenizacao sem truncation;
- offset masks OK;
- adapter inicial V290 checkpoint-6 carregado `12011/12011`;
- trainable LoRA `8,015,872` parametros, `0.0247%`.

Resultado V440:

| Metrica interna V439 validation | Baseline V290 ckpt-6 | V440 checkpoint-3 |
|---|---:|---:|
| preference total | `8/24` | `8/24` |
| equation | `7/22` | `7/22` |
| bit | `1/2` | `1/2` |

Decisao: cancelado por FinOps. Nao houve sinal material para weak/full.

Diagnostico: limpar o target era necessario, mas o objetivo `mean_nll` sobre a
sequencia curta ainda nao e forte o bastante para converter os acertos desejados
em comportamento do adapter.

### 6. Consulta API sobre V441

Consulta feita em 2026-05-15 via OpenRouter:

- `deepseek/deepseek-v3.2`: V441 e tecnicamente justificado, mas com ressalva
  de que payload-only pode nao corrigir raciocinio.
- `qwen/qwen3.6-max-preview`: V441 e a correcao mecanica direta para a diluicao
  de sinal da V440.
- `google/gemini-3.1-pro-preview`: resposta truncada, mas iniciou validando a
  mesma tese de diluicao de sinal.

Conclusao atualizada: V441 ja foi executado e cancelado. Nao deve ser repetido.
Ele testou a hipotese de diluicao da loss em tokens de boilerplate e nao trouxe
sinal interno suficiente para weak/full.

Preflight local V441:

- Compile, gate estatico e gate pre-pago: OK.
- Tokenize-only dry-run: treino `109/109`, validacao `24/24`.
- Truncation e fallback de offset mask: `0`.
- Mascara de score no payload nao vazia: treino `chosen=339`, `rejected=376`;
  validacao `chosen=75`, `rejected=78`.

Resultado V441:

| Metrica interna V441 validation | Baseline V290 ckpt-6 | V441 checkpoint-3 |
|---|---:|---:|
| preference total | `7/24` | `7/24` |
| equation | `6/22` | `6/22` |
| bit | `1/2` | `1/2` |

Decisao: cancelado por FinOps. A implementacao do payload mask funcionou, mas
nao houve sinal interno para weak/full.

### 7. Auditoria V442 pos-V441

Resultado:

| Item | Valor |
|---|---:|
| pares V439 auditados | `133` |
| source-ok rows | `133` |
| weak/full training rows | `0` |
| rule-certified rows | `0` |

Conclusao: o dataset V439 e limpo como diagnostico, mas insuficiente para novo
treino pago. Ele nao traz `rule_unique_label_free`, `program_or_rule`,
`mdl_score`, `leave_one_out_pass`, `renaming_stability_pass`,
`slot_alignment_stats` nem `rule_frozen_before_answer`.

Decisao: bloquear novos jobs de preference simples sobre V435E/V439. O proximo
passo e construir o builder CPU certificado, nao trocar LR/epoch/loss outra vez.

## O Que Isso Significa

O gargalo nao e "rodar mais H200". O gargalo e **como transformar conhecimento
verificado em gradiente util para o LoRA**.

Temos tres tipos de conhecimento:

1. `solver/verifier` consegue identificar alguns acertos potenciais;
2. literatura/discussion indica algoritmos fortes para bit e equation;
3. adapters atuais ja sabem muito, mas falham em subconjuntos especificos.

O que ainda falta:

1. isolar exatamente os subtipos de equation que podem ganhar `+4`;
2. gerar pares/traces onde a resposta correta seja aprendida sem ensinar texto
   estranho;
3. produzir acertos novos via solver/DSL antes de outro treino;
4. preservar `bit >= 136/160`.

## Proxima Acao Tecnica Correta

Nao repetir:

- V436;
- V436B;
- V440;
- V441;
- mais epochs/LR sweep sobre o mesmo `mean_nll`;
- mais preference simples sobre final answer sem novo sinal CPU;
- submit sem weak/full gain.

Historico fechado:

1. V443 CPU certified equation pair builder foi executado:
   - focar primeiro `equation_symbolic_sequence` e `equation_symbolic_short`;
   - congelar regra antes do answer;
   - exigir MDL, Leave-One-Out, renaming stability, candidate count unico e
     slot/substring alignment stats.
2. O builder encontrou `0` pares certificados, portanto a rota simples ficou
   fechada.
3. Uma nova rota so pode gerar dataset de distilacao se trouxer:
   - prompt original;
   - final answer curto derivado da regra congelada;
   - hard negative real do adapter;
   - trace deterministico curto somente se nao contaminar target.
4. So voltar a treino se um novo CPU gate produzir sinal verificavel.
5. Rodar novo HF job somente se o CPU gate provar ganho potencial e o
   integration gate aprovar tudo.

## Criterio De Sucesso

Um candidato so avanca se:

- weak total `>192/315`;
- `equation_transform >56/155`, ideal `>=60/155`;
- `bit_manipulation >=136/160`;
- truncation `0`;
- depois full official-like `>823/947`.

Sem isso, o resultado volta para o error ledger e nao vira submit.

## Atualizacao V443/V444

V443 executou o builder CPU de pares certificados para equation e retornou
`0` pares certificados. Isso localiza o problema: a melhoria desejada nao esta
em substituicoes textuais simples, slot-map global ou regras de string com LOO
e renaming stability. Essa rota fica fechada ate existir uma DSL mais forte.

V444 foi um teste minimo de transferencia supervisionada historico: usou traces
`rule_found` e `hypothesis_formed`, removeu `rule_unknown`, passou tokenizacao
sem truncation, publicou no HF e treinou um smoke H200 de quatro steps.
Depois da auditoria V472/V473, `hypothesis_formed` nao e mais aceito como fonte
ativa para novos datasets de treino sem verificacao adicional de contradicao.

Problema principal agora:

```text
Dados com regra simples certificada     -> 0 pares V443
Dados reconstruidos amplos V397/V398    -> sem ganho weak
Dados reconstruidos high-confidence V444 -> checkpoint-2 caiu para 190/315
```

Decisao:

1. V444 foi cancelado por FinOps no primeiro checkpoint avaliado.
2. Resultado checkpoint-2: total `190/315`, equation `56/155`, bit `134/160`,
   truncated `1`.
3. V448/V461/V463/V464/V468 e adapters derivados V448/V465/V469 ficam
   quarentenados/fail-closed.
4. O proximo plano nao e mais SFT reconstruido; volta para auditoria CPU de
   raw output/parse ou DSL/solver equation mais expressivo, com prova CPU antes
   de qualquer GPU.

## Atualizacao V474 - Problema De Metrica Simbolica

V474 encontrou um problema no caminho de parse/eval, nao um ganho de modelo:
respostas simbolicas com braces literais podiam ser subextraidas do
`\boxed{}`. Isso podia criar falso negativo de ACC e confundir analises de
loss/eval.

Correcao V504 aplicada:

- `extract_final_answer` label-free e o unico caminho valido para `prediction`
  promocional/submittable;
- `extract_final_answer_for_expected` fica restrito a coluna de debug
  `label_aware_debug_prediction`, nunca para ACC submit-safe;
- `evaluate_lora_adapter.py`, `evaluate_lora_adapters_batch.py` e
  `analyze_eval_predictions.py` reextraem a resposta de `raw_output` sem usar o
  `answer` conhecido;
- V286 continua usando `box_answer(answer)` para gerar datasets boxed, mas o
  gate de avaliacao precisa provar parse label-free antes de promover.

Impacto no mapa do problema: antes de culpar treino ou adapter, todo novo
dataset/resultado precisa provar que o parse simbolico esta correto pelo gate.
