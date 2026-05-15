# KG1 V434 OpenRouter May 15 Response Review

Fonte analisada: `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026.json`.

## Resumo

O arquivo contem `16` mensagens de modelos OpenRouter para o prompt sobre como sair do teto adapter-only em `equation_transform` e `bit_manipulation`. Para decisao tecnica, considerar apenas respostas finais completas; mensagens incompletas/tool-only/reasoning-only entram como ruido.

Conclusao rigorosa: as respostas nao trazem um ganho submit-safe pronto. Elas convergem, porem, em uma mudanca tecnica importante: nao repetir SFT/teacher transfer. O proximo experimento util precisa transformar os erros reais do V291/V290 em pares de preferencia/hard negatives com filtros label-free de unicidade, MDL e Leave-One-Out.

Importante: weak/full gates nao podem ser usados como fonte de treino, selecao de pares, escolha de misses ou construcao de `chosen/rejected`. Weak/full entram apenas como avaliacao final. Qualquer par ORPO precisa vir de public train/sintetico auditado, sem overlap por `id`, `prompt_sha256` ou resposta.

## Estado Atual Confirmado

| Metrica | Valor |
|---|---:|
| Melhor adapter-only weak | `192/315` |
| `equation_transform` | `56/155` |
| `bit_manipulation` | `136/160` |
| Melhor CPU teacher/verifier | `222/315` |
| Melhor solver/verifier integrado anterior | `201/315` |

## Consenso Entre Modelos

| Achado | Avaliacao |
|---|---|
| SFT amplo em traces/teacher nao deve continuar | confirmado por historico V391/V398/V413/V416 |
| `eval_loss` baixo nao prediz ganho | confirmado por varias execucoes locais |
| Solver/verifier externo tem sinal, mas nao e submit-safe | confirmado por V336B e package gate |
| `equation_transform` precisa de desempate label-free | confirmado por V431/V432/V433 |
| `bit_manipulation` deve usar bit-pair/bitsum/stride minimalista | coerente com Tong Hui Kang, mas precisa virar comportamento do adapter |
| DPO/ORPO/hard negatives sao o caminho novo mais distinto | ainda nao medido; precisa CPU gate antes de GPU |

## Hipoteses Que Entram no Roadmap

### H1 - V291 Exact-Wrong ORPO

Gerar pares de preferencia usando a resposta errada real do V291/V290 como `rejected` e a resposta solver/verifier label-free como `chosen`, somente em dados de treino permitidos/public train/sinteticos auditados, e somente quando:

- solver passa por MDL/LOO/unicidade;
- a resposta label-free do solver e diferente da resposta V291;
- nao ha uso de id, answer no prompt, postprocessor ou cherry-pick;
- `answer` e usado apenas depois, para auditoria de metrica, nunca para construir `chosen`;
- replay de bit preserva `bit>=136`.

Esta e a proposta mais concreta do arquivo, porque ataca diretamente o erro que o adapter atual comete, ao contrario de SFT que so empurra para a resposta correta.

Status: `PRECISA DE GATE`.

### H2 - Anti-Unification / E-Generalization Para Alice

Usar anti-unification para gerar regras simbolicas minimas sobre pares `lhs/rhs` de `equation_transform`, com aceite somente se a regra for unica por MDL e estavel por Leave-One-Out.

Status: `TEACHER-ONLY` se usado como solver; `PRECISA DE GATE` se virar fonte de pares para H1.

### H3 - CEGIS/SyGuS Com Gramatica Fechada

Testar uma classe formal diferente das DSLs rejeitadas: gerar programas candidatos numa gramatica pequena, refutar por contraexemplos sinteticos e aceitar apenas programa unico.

Status: `TEACHER-ONLY` ate virar dataset/hard negative para H1.

### H4 - Bit-Pair Minimal Preference Dataset

Para bit, nao treinar CoT longa. Criar exemplos curtos que isolam relacao `bit-pair/bitsum/stride`, com hard negatives por stride/operador errado. O objetivo e manter `136/160` e tentar `+1/+2` sem regredir equation.

Status: `PRECISA DE GATE`.

### H5 - Orthogonal LoRA Merge Apenas Como Segunda Fase

Merge ortogonal/TIES/SLERP so faz sentido se existirem dois adapters com sinais reais separados. Como soups V388/V389 falharam, merge sem sinal novo continua bloqueado.

Status: `BLOQUEADO ATE H1/H4 GERAREM ADAPTERS COM SINAL`.

## Pontos Rejeitados

| Proposta | Motivo |
|---|---|
| Runtime abstention por threshold/logit | nao e submit-safe se exigir codigo externo ou postprocessor |
| Output `ABSTAIN` no teste | vira resposta errada; so serve como ferramenta de treino/gate |
| Treino a partir de weak/full misses | leakage; weak/full sao apenas avaliacao/gate |
| `row.answer` em dataset ou `chosen` | leakage; `chosen` precisa vir de regra label-free definida antes da auditoria |
| Constrained decoding, logit masks ou decoder patches | runtime externo, nao adapter-only |
| Prompt-prefix, soft-prompt ou `embed_tokens` salvo fora do contrato oficial | rejeitado ate package gate oficial provar compatibilidade |
| Treinar mais epochs ou LR sweep | historicamente reduz loss sem mover ACC |
| Promover OSLM/merge imediatamente | sem adapters novos com sinal, repete falha de soups |
| Usar solver/verifier direto no package | bloqueado por regra de adapter-only/V336B |
| Porcentagens de sucesso vindas apenas de LLMs | consenso de LLM nao e evidencia experimental |

## Proximo Experimento Recomendado

CPU-only primeiro:

1. Reproduzir predicoes V291/V290 no weak/full local.
2. Usar essas predicoes weak/full apenas como diagnostico/eval, nao como fonte de treino.
3. Em public train/sintetico auditado, gerar candidatos por anti-unification + CEGIS/SyGuS.
4. Filtrar por MDL, LOO e unicidade antes de olhar `answer`.
5. Gerar pares ORPO:
   - `chosen`: trace curto solver correto;
   - `rejected`: resposta/trace errado real do V291 ou hard negative de mesma familia.
6. Adicionar replay de bit-pair curto para preservar `bit>=136`.
7. So liberar GPU se o preflight mostrar pares suficientes e sem leakage:
   - `eq_candidates >= 4` acima do baseline;
   - `bit_guardrail >= 136`;
   - `0` weak/full leakage em treino;
   - `0` truncation no tokenization gate;
   - comparativo V434 vs V291 incluido.

## Decisao

O arquivo e util, mas nao por trazer uma resposta pronta. Ele sugere que o proximo caminho plausivel precisa ser uma mudanca de objetivo de treino: `SFT -> ORPO/DPO com hard negatives reais do V291`, filtrado por MDL/LOO/CEGIS e com replay de bit. Isto ainda e hipotese, nao evidencia experimental. Qualquer GPU antes desse CPU gate seria gasto sem evidencia.
