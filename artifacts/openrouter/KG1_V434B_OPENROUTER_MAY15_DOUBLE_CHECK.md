# KG1 V434B OpenRouter May 15 Double Check

Fonte: `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026.json`.

Este double check reprocessou todas as respostas do arquivo e comparou contra o V434 ja inserido no roadmap.

## Cobertura

| Item | Valor |
|---|---:|
| Modelos/respondentes | `16` |
| Respostas finais completas | `13` aprox.; mensagens incompletas/tool-only/reasoning-only tratadas como ruido |
| Maior resposta | DeepSeek V4 Pro, `51166` chars |
| Respostas com ORPO/DPO | maioria; `DPO=157`, `ORPO=145` ocorrencias no JSON |
| Respostas com MDL/LOO | forte consenso; `MDL=155`, `LOO=122` ocorrencias |
| Respostas com CEGIS/SyGuS | recorrente; `CEGIS=53`, `SyGuS=31` ocorrencias |
| Respostas com bit-pair/stride | recorrente; `bit-pair=53`, `stride=51` ocorrencias |

## Confirmacao do V434

O V434 continua correto: a recomendacao dominante e trocar `SFT teacher -> LoRA` por pares de preferencia `chosen/rejected`, usando a resposta errada real do V291/V290 como hard negative e filtrando `chosen` por MDL, Leave-One-Out e unicidade.

Nenhum modelo trouxe ganho submit-safe pronto. Todas as ideias uteis continuam em `PRECISA DE GATE`.

Correcao critica deste double check: weak/full gates nao podem ser usados para construir pares, escolher misses ou selecionar `chosen/rejected`. Eles sao apenas avaliacao. Todo dataset ORPO deve vir de public train/sintetico auditado e provar `0` overlap.

## Achados Extras do Double Check

| Achado | Fonte recorrente | Decisao |
|---|---|---|
| MDL + LOO + estabilidade por renomeacao de simbolos | GPT-5.3-Codex/Gemini-style LGG | incluir no V435 como criterio de desempate adicional |
| Bit ANF-SAT / MaxSAT esparso | GPT-5.3-Codex | P2 teacher para bit; usar so se gerar traces curtos e nao substituir o caminho principal |
| Bit-pair attention routing / target_modules restritos | Qwen3.6/DeepSeek/Gemini | P2, bloqueado ate existir dataset V435 com sinal; nao abrir treino so por isso |
| TIES/DARE/SLERP merge | DeepSeek/Qwen | bloqueado; soups lineares falharam e nao ha adapters novos com sinais separados |
| Prefix/embedding tokens | Qwen Coder/Nemotron | rejeitar para agora; risco de incompatibilidade com tokenizer/runtime e exige prefix injection |
| Hidden-state contrastive distillation | Nemotron/Qwen | rejeitar para agora; depende de acesso/modelo teacher compativel e nao resolve gate adapter-only diretamente |
| Runtime confidence/logit abstention | varios | rejeitar como runtime; usar apenas como criterio offline/gate |
| Constrained decoding/logit masks/decoder patches | varias respostas | rejeitar; runtime externo nao entra no package adapter-only |

## Ajuste Tecnico no V435

O V435 nao deve ser apenas "ORPO builder". Ele deve gerar um relatorio CPU com tres blocos:

1. `equation_pair_builder`
   - candidatos por anti-unification/E-generalization;
   - CEGIS/SyGuS pequeno;
   - score por MDL, Leave-One-Out, estabilidade por renomeacao e alinhamento por slot/operador;
   - pares `chosen/rejected` apenas quando houver regra unica.
   - proibido usar weak/full como fonte de pares.

2. `bit_guardrail_builder`
   - bit-pair/bitsum/stride curto;
   - opcional ANF-SAT/MaxSAT como teacher secundario;
   - hard negatives por stride/operador errado;
   - objetivo primario: preservar `bit>=136`, nao perseguir bit a qualquer custo.

3. `training_feasibility_gate`
   - contagem de pares limpos;
   - anti-leakage por `id`, `prompt_sha256`, familia e resposta no prompt;
   - tokenization gate;
   - comparativo contra V291/V290;
   - `hf_gpu_allowed=true` somente se houver sinal material.

## Pontos Que Nao Entram

- `ABSTAIN` como output final no teste.
- Prefix tokens novos que alterem tokenizer.
- `embed_tokens`/soft-prompt/prompt-prefix sem package gate oficial.
- Constrained decoding externo.
- Logit masks ou decoder patches.
- Solver/verifier direto no pacote.
- Merge TIES/DARE sem adapters com sinais reais.
- Treino baseado em `eval_loss`.
- Qualquer `chosen` que use `answer` antes da regra estar definida.

## Decisao

O double check nao muda o caminho principal, mas refina o V435:

- adicionar `renaming stability` ao desempate label-free;
- adicionar ANF-SAT/MaxSAT como bit teacher P2;
- manter targeted modules e merge como fases posteriores, nao como proximo job;
- continuar bloqueando GPU ate o CPU gate provar pares limpos e promissores.
