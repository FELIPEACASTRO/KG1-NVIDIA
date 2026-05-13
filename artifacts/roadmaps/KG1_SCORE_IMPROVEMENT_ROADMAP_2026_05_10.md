# KG1 NVIDIA - Roadmap ativo de melhoria por familia

Atualizado em: 2026-05-13

Este arquivo e o roadmap ativo. Ele contem apenas o que sera usado daqui para frente para tentar melhorar `bit_manipulation` e `equation_transform`.

O historico completo de tentativas, buscas, logs e rotas rejeitadas foi movido para:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10_FULL_HISTORY_2026_05_13.md`

## Verdade operacional

As evidencias fortes reunidas hoje mostram que ha ganho possivel, mas o ganho comprovado ainda esta em solver/verifier, nao em LoRA puro.

| Estado | Weak / Full | Equation | Bit | Decisao |
|---|---:|---:|---:|---|
| Melhor adapter-only weak atual | `192/315` | `56/155` | `136/160` | baseline operacional LoRA |
| Melhor full adapter-only submitado | `823/947` | `56/155` | `135/160` | V291, public score `0.86` |
| V274/V275 solver/verifier | `196/315` | `60/155` | `136/160` | ganho real CPU, ainda nao LoRA puro |
| V324+V329 solver/verifier | `197/315` projetado | `61/155` | `136/160` | ganho real CPU, ainda nao LoRA puro |
| V306/V302 full verifier local | `838/947` potencial | `60/155` | `146/160` | depende de verifier/postprocessor |
| V335 LoRA mixed trace replay | `190/315` | `56/155` | `134/160` | falhou; cancelado por FinOps |

Conclusao: a busca/documentacao gerou conhecimento util e ganho tecnico real. O erro foi assumir que SFT curto/misto transferiria automaticamente essas regras para LoRA. V303, V326, V331 e V335 falsificaram essa hipotese.

## Metas

Meta minima para novo candidato:

- Weak: `>=193/315`.
- `equation_transform >=60/155`.
- `bit_manipulation >=136/160`.
- Truncation `0` ou nao regressiva.

Meta para submit novo:

- Full official-like `>823/947`.
- Sem queda nas familias criticas.
- Package permitido pelas regras do desafio.
- Manifest com diff por familia.

## Evidencias aceitas no roadmap ativo

| Evidencia | Impacto real | Uso permitido |
|---|---|---|
| V274/V275 numeric postprocessor | `+4` equation, `0` perdas, `196/315` | regra/verifier e teacher |
| V329 symbolic cryptarithm | `+1` equation adicional projetado, `0` perdas | regra/verifier e teacher |
| Tong Hui Kang bit solver | `1364/1602` train, mas `+1/-1` no weak | teacher/taxonomia, nao override direto |
| `equation-solver-swap-v1` | classes uteis, mas overlap com gates | taxonomia/fixture sintetico verificado |
| Discussions `690307`, `688461` | bit-pair/bitsum/stride e boolean gate taxonomy | implementar CPU bit gate |
| Discussions `689877`, `698293` | operadores ausentes e estrutura latente de equation | abstain/conflict count em DSL |
| Discussions `693260`, `697491` | synthetic accuracy alta pode piorar LB | traces curtos e kill-switch |
| OpenRouter/destilacao | consenso metodologico: equation e sintese/verificacao | nao autoriza GPU sozinho |

## Roadmap ativo em ordem de maior chance de ganho

### 1. V336A - CPU integrated no-loss solver gate

Objetivo: consolidar os ganhos ja medidos e procurar novo ganho sem perdas antes de qualquer GPU.

Entrada:

- Predicoes do melhor adapter-only atual: `192/315`, equation `56`, bit `136`.
- Regras V274/V275 numeric operator.
- Regra V329 symbolic cryptarithm.
- Taxonomia `equation-solver-swap-v1`: `concat`, `swap_concat`, `add`, `sub`, `abs_sub`, `mul`, `rev_both_add_rev`, `rev_both_mul_rev`, `rev_both_abs_sub_rev`.
- Taxonomia bit: Tong stride/bitsum, constantes, identity, NOT, gates 2-input, negacoes assimetricas, majority/choice/parity.

Saida obrigatoria:

- Manifest por candidato com `id`, `family`, `rule_class`, `old_prediction`, `new_prediction`, `candidate_count`, `conflict_count`, `accepted/rejected`, `reason`.
- Reproducao de V274/V275: `196/315`.
- Reproducao de V324+V329: `197/315` projetado.
- Novo ganho so vale se `losses=0`, `bit>=136` e `equation>=61`.

Bloqueio:

- Se V336A nao demonstrar ganho no-loss, nao rodar HF GPU.

### 2. V336B - Gate de permissao/package do solver/verifier

Objetivo: decidir se o ganho CPU pode virar submissao permitida ou se precisa ser absorvido por LoRA.

Checagens:

- Regras Kaggle atuais do desafio.
- Conteudo permitido no `submission.zip`.
- Se codigo/verifier/postprocessor pode acompanhar o adapter.
- Se o pacote precisa ser adapter-only.

Decisao:

- Se solver/verifier for permitido: preparar full eval official-like com V274/V275/V329/V336A.
- Se nao for permitido: seguir para V336C/V337 e tentar absorcao LoRA.

Bloqueio:

- Nenhum submit com solver/verifier sem esse gate.

### 3. V336C - Dataset minimo de transferencia, somente se V336A passar

Objetivo: criar dataset pequeno que ensine exatamente os casos em que o solver acerta e o adapter erra.

Permitido:

- Hard positives: baseline erra, solver/verifier acerta, `losses=0`.
- Hard negatives: casos parecidos em que a regra deve abstain.
- Replay minimo para proteger `bit>=136` e familias ja saturadas.
- Dois formatos no maximo:
  - `answer_only_verified`;
  - `compact_trace_verified`.

Proibido:

- Dataset publico bruto com overlap.
- Traces longos sem necessidade.
- Mistura ampla V335-like.
- SFT generico de mais epochs.

Gate:

- `id_overlap=0`.
- `prompt_sha256_overlap=0`.
- verifier correctness em todas as linhas.
- tokenization/offset-mask gate.
- manifest de source/family counts.

### 4. V337 - Tiny LoRA absorption smoke

Objetivo: testar absorcao real com gasto minimo.

Configuracao:

- Seed: melhor adapter-only atual.
- Dataset: apenas V336C.
- Checkpoint cedo.
- Weak eval imediato no primeiro checkpoint.
- A100 preferencial por FinOps; H200 so se A100 nao suportar runtime.

Continuar somente se:

- `total>192`.
- `equation>56`.
- `bit>=136`.
- truncation `0`.

Cancelar por FinOps se:

- `bit<136`;
- `total<192`;
- `equation=56`;
- OOM, erro runtime, upload travado ou perda de contrato.

### 5. V338 - Full eval, package e Kaggle submit

Executar apenas se V337 passar weak gate.

Passos:

1. Full eval official-like.
2. Comparar contra V291 `823/947`.
3. Gerar package somente se full `>823/947`.
4. Validar estrutura do pacote.
5. Submeter ao Kaggle somente com ganho medido.

Bloqueio:

- Nao submeter adapter que mantem `equation=56` por expectativa.
- Nao submeter candidato que reduz bit contra V291/V290.

### 6. V339 - Se LoRA continuar falhando

Se V337 falhar de novo:

- Parar SFT curto/misto.
- Nao gastar HF com variacao de LR, epochs ou pesos.
- Voltar para uma das duas rotas:
  - rota A: package permitido com solver/verifier;
  - rota B: aguardar/liberar fonte realmente nova de traces, como `andy279/*`, com aprovacao humana.

## Itens removidos do roadmap ativo

Os itens abaixo ficam apenas no arquivo historico. Eles nao fazem parte do plano ativo.

| Item/rota | Motivo da remocao do plano ativo |
|---|---|
| Adapter soups | testado; nao moveu equation |
| Prompt `no suffix` / thinking variants amplas | regressao severa |
| Public external adapters `gfinin`, `etencore`, outros | muito abaixo do baseline |
| ReasoningGym direto / Alice-style SFT | drift severo no weak |
| GGUF Space | nao e LoRA adapter/packageable |
| V303 bit fullbyte distill | nao transferiu verifier para LoRA |
| V326 equation+bit replay | equation ficou `56`, bit caiu |
| V331 symbolic mix | equation ficou `56`, bit caiu |
| V335 mixed trace replay | `190/315`, bit `134`; cancelado |
| Raw `kienngx` / `konbu17` / `furkankesen` datasets | overlap ou flags incorretas; usar so taxonomia |
| Generic distillation papers/datasets | P2 metodologico, sem acao direta |
| Mais epochs/LR sem novo CPU gate | gasto sem fundamento |

## Regras permanentes

- Roadmap ativo so aceita item com impacto medido ou acao concreta.
- Toda evidencia externa deve ser classificada como `ganho medido`, `taxonomia/teacher`, `bloqueado por overlap`, `rejeitado por gate` ou `P2 metodologico`.
- Enquanto job HF estiver rodando, verificar logs a cada aproximadamente `40s`.
- Se o job nao puder mais bater o gate, cancelar. A decisao FinOps correta e cancelar e nao gastar.
- `eval_loss` menor nao promove candidato.
- Nenhum notebook alterado pode ser entregue sem `scripts/notebook_release_gate.py`.
- Nenhum Kaggle submit sem weak/full gain medido.

## Proxima acao unica

Implementar `V336A - CPU integrated no-loss solver gate`.

Nao iniciar HF GPU antes disso.
