# KG1 V434 OpenRouter May 15 Response Review

Fonte analisada: `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026.json`.

## Resumo

O arquivo contem respostas de `16` modelos OpenRouter para o prompt sobre como sair do teto adapter-only em `equation_transform` e `bit_manipulation`.

Conclusao rigorosa: as respostas nao trazem um ganho submit-safe pronto. Elas convergem, porem, em uma mudanca tecnica importante: nao repetir SFT/teacher transfer. O proximo experimento util precisa transformar os erros reais do V291/V290 em pares de preferencia/hard negatives com filtros label-free de unicidade, MDL e Leave-One-Out.

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

Gerar pares de preferencia usando a resposta errada real do V291/V290 como `rejected` e a resposta solver/verifier correta como `chosen`, somente quando:

- solver passa por MDL/LOO/unicidade;
- a resposta correta e diferente da resposta V291;
- nao ha uso de id, answer no prompt, postprocessor ou cherry-pick;
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
| Treinar mais epochs ou LR sweep | historicamente reduz loss sem mover ACC |
| Promover OSLM/merge imediatamente | sem adapters novos com sinal, repete falha de soups |
| Usar solver/verifier direto no package | bloqueado por regra de adapter-only/V336B |

## Proximo Experimento Recomendado

CPU-only primeiro:

1. Reproduzir predicoes V291/V290 no weak/full local.
2. Para cada miss de `equation_transform`, gerar candidatos por anti-unification + CEGIS/SyGuS.
3. Filtrar por MDL, LOO e unicidade.
4. Gerar pares ORPO:
   - `chosen`: trace curto solver correto;
   - `rejected`: resposta/trace errado real do V291 ou hard negative de mesma familia.
5. Adicionar replay de bit-pair curto para preservar `bit>=136`.
6. So liberar GPU se o preflight mostrar pares suficientes e sem leakage:
   - `eq_candidates >= 4` acima do baseline;
   - `bit_guardrail >= 136`;
   - `0` weak/full leakage em treino;
   - `0` truncation no tokenization gate;
   - comparativo V434 vs V291 incluido.

## Decisao

O arquivo e util, mas nao por trazer uma resposta pronta. Ele confirma que o proximo caminho plausivel precisa ser uma mudanca de objetivo de treino: `SFT -> ORPO/DPO com hard negatives reais do V291`, filtrado por MDL/LOO/CEGIS e com replay de bit. Qualquer GPU antes desse CPU gate seria gasto sem evidencia.
