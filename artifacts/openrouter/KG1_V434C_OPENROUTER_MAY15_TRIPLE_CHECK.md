# KG1 V434C OpenRouter May 15 Triple Check

Data: 2026-05-15

Fonte principal:

- `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026.json`

Arquivos relacionados:

- `artifacts/openrouter/KG1_V434_OPENROUTER_MAY15_RESPONSE_REVIEW.md`
- `artifacts/openrouter/KG1_V434B_OPENROUTER_MAY15_DOUBLE_CHECK.md`
- `artifacts/roadmaps/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md`

## Veredito

O arquivo OpenRouter nao trouxe ganho pronto para submit. Ele reforcou uma conclusao negativa e uma rota ativa:

- Negativa: continuar SFT amplo, mais epochs, LR sweep, prompt sweep ou relaunch GPU sem novo dado nao deve continuar.
- Rota ativa: V435 CPU-only precisa construir pares submit-safe com hard negatives reais do V291/V290 em dados permitidos, certificados por MDL/LOO/renaming, antes de liberar qualquer GPU.

O melhor baseline submit-safe continua:

| Item | Valor |
|---|---:|
| Weak total | 192/315 |
| equation_transform | 56/155 |
| bit_manipulation | 136/160 |
| truncated | 0 |
| Full official-like | 823/947 |

## Achados Dos Agentes

| Frente | Achado | Decisao |
|---|---|---|
| Regras/submit-safety | Weak/full, `answer`, id, solver runtime e postprocessor nao podem ser fonte de treino ou submit | Regra permanente |
| Tecnica | ORPO/DPO e melhor rota remanescente, mas somente com hard negatives reais do V291/V290 | Ativo em V435/V436 |
| Tecnica | MDL, Leave-One-Out e renaming stability precisam virar certificado auditavel | Ativo em V435 |
| Tecnica | Operator-conditioned slot alignment ficou subvalorizado e deve virar centro de equation | Ativo em V435 |
| Transferencia | V435 nao pode ser so teacher forte; precisa provar erro real do adapter congelado antes de GPU | Gate obrigatorio |
| FinOps | GPU so se `hf_gpu_allowed=true`; cancelar no primeiro checkpoint ruim | Regra permanente |
| Roadmap | Historico V392-V434 poluia o plano ativo | Roadmap refeito e historico arquivado |

## Hipoteses Mantidas

1. `equation_pair_builder`
   - anti-unification / E-generalization;
   - CEGIS / SyGuS;
   - slot/substring alignment condicionado por operador;
   - MDL;
   - Leave-One-Out;
   - estabilidade por renomeacao.

2. `bit_guardrail_builder`
   - bit-pair;
   - bitsum;
   - stride;
   - hard negatives verificados por programa;
   - ANF-SAT/MaxSAT apenas como P2 barato.

3. ORPO/DPO curto
   - somente depois do V435;
   - usar pares aprovados;
   - preservar prompt/template oficial;
   - kill-switch no primeiro checkpoint.

## Hipoteses Removidas

| Hipotese | Motivo |
|---|---|
| Prompt-prefix, soft-prompt, `embed_tokens` | Risco de quebrar contrato adapter-only/vLLM |
| Hidden-state distillation | Exigiria runtime/modelo teacher incompatível e nao ataca package final |
| Runtime abstention/confidence/logit mask/constrained decoding | Fora do submit adapter-only |
| Mais epochs/LR sweep/broad SFT | Historico mostrou loss melhor sem ACC melhor |
| Teacher/verifier direto para LoRA sem hard negative real | Repetiu falhas V391/V398/V413/V416 |
| Solver/postprocessor final | Nao submit-safe |
| Weak/full misses como dados | Leakage/cherry-pick |

## Reforco Do Bloqueio V417

V435 so pode liberar V436 se produzir sinal material de transferencia, nao apenas acertos CPU. O novo gate deve provar:

- o V291/V290 congelado erra aqueles padroes em dados permitidos;
- a regra correta foi escolhida sem `answer`;
- o `rejected` e raw output real do adapter congelado;
- ha volume suficiente por modo de erro;
- bit guardrail esta pronto;
- tokenization/truncation estao limpos;
- nao ha overlap weak/full.

Sem isso, `hf_gpu_allowed=false`.

## Acao Tomada

O roadmap ativo foi refeito para iniciar em V435, com:

- snapshot submit-safe real;
- regras permanentes;
- V435 CPU adapter-level pair builder;
- V436 short adapter-only smoke condicionado;
- V437 full/package/submit condicionado;
- lista curta de itens removidos do plano ativo.

O historico pre-V435 foi preservado em archive para consulta, mas nao deve guiar a execucao diaria.
