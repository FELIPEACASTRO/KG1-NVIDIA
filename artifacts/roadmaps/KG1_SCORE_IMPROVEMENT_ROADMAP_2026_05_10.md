# KG1 Score Improvement Roadmap

Atualizado: 2026-05-15

Este e o roadmap ativo a partir do V435. O historico pre-V435 foi arquivado em:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V435_CLEANUP_2026_05_15.md`

O objetivo desta limpeza e remover rotas antigas, hipoteses sem transferencia e itens que nao serao mais usados. A execucao agora deve seguir apenas o plano abaixo.

## Objetivo

Melhorar o score submit-safe adapter-only nas familias fracas, sem perder o que ja esta bom.

Gate minimo para considerar ganho:

| Metrica | Baseline submit-safe | Minimo para promover |
|---|---:|---:|
| Total weak | 192/315 | > 192/315 |
| equation_transform | 56/155 | > 56/155, ideal >= 60/155 |
| bit_manipulation | 136/160 | >= 136/160 |
| truncated | 0 | 0 |
| Full official-like | 823/947 | > 823/947 |

Regra central: ganho so conta se aparecer no adapter/package. Teacher CPU, solver, verifier e postprocessor sao diagnostico ate virarem comportamento do adapter valido.

## Snapshot Atual

| Item | Resultado | Status |
|---|---:|---|
| Melhor adapter-only submit-safe, V291/V290 checkpoint-6 | weak 192/315, equation 56/155, bit 136/160, trunc 0 | referencia ativa |
| Full V291/V290 checkpoint-6 | 823/947, equation 56, bit 135, trunc 1 | melhor pacote conhecido |
| Melhor teacher CPU V414/V366 | weak 222/315, equation 63, bit 159 | nao submit-safe |
| Melhor projecao solver integrada V405 | weak 201/315, equation 63, bit 138 | nao submit-safe |
| OpenRouter V434/V434B/V434C | sem ganho pronto; trouxe filtros e rota V435 | evidencia para plano |
| V435 CPU pair gate | 0/3558 pares aprovados | bloqueia GPU; faltam raw outputs reais V291/V290 e certificados |
| V435B prompt pack | 840 prompts permitidos, 0 answers exportadas | pronto para coleta de raw outputs, nao libera treino |
| GPU/HF | bloqueado | liberar somente com V435 `hf_gpu_allowed=true` |

## Regras Permanentes

1. Weak/full sao somente avaliacao e gate. Nunca usar weak/full para construir pares, escolher misses, selecionar candidatos, gerar `chosen/rejected`, treinar, desempatar regra ou fazer cherry-pick.
2. `answer` nao pode participar da decisao de regra, filtro, tiebreak, `chosen`, selecao de trace ou selecao de candidato. A resposta correta so pode aparecer em auditoria posterior, depois da regra estar congelada.
3. `id`, prompt hash de weak/full, oracle, solver runtime, verifier runtime e postprocessor nao podem entrar no submit.
4. Submit valido e apenas adapter-only: `adapter_config.json` e `adapter_model.safetensors` no root do pacote. Sem script, tokenizer, prompt prefix, soft-prompt, `embed_tokens`, `lm_head`, decoder patch, logit mask, constrained decoding, runtime abstention ou confidence threshold.
5. Decisao e por ACC/truncation. `eval_loss`, `train_loss`, preference accuracy interna e probabilidades de LLM nao liberam submit nem GPU.
6. HF GPU so pode rodar depois de CPU gate com sinal material, manifest auditavel e `hf_gpu_allowed=true`.
7. FinOps: cancelar ou nao iniciar qualquer job que nao possa mais superar `total>192`, `equation>56`, `bit>=136`, `truncated=0`.
8. Toda nova versao precisa quadro comparativo contra V291/V290 e decisao explicita: promover, repetir CPU, cancelar ou arquivar.
9. Se notebook for criado ou alterado, precisa passar `python scripts/notebook_release_gate.py <notebook>` antes de entrega/push.
10. Se job estiver rodando, analisar logs periodicamente e aplicar kill-switch sem esperar gasto inutil.

## Achados Consolidados V434C

O triple check do arquivo `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026.json` e dos agentes convergiu para estes pontos:

| Achado | Decisao |
|---|---|
| ORPO/DPO pode ser util | Ativo somente se houver hard negatives reais do V291/V290 em dados permitidos |
| MDL, Leave-One-Out e renaming stability | Ativo como certificado label-free, nao como texto narrativo |
| Operator-conditioned slot alignment | Ativo; deve virar nucleo de `equation_pair_builder` |
| Bit-pair/bitsum/stride | Ativo como guardrail e trace curto para preservar bit |
| ANF-SAT/MaxSAT | P2 barato para bit, nao libera GPU sozinho |
| Targeted modules/freeze | P2; so apos V435 provar sinal |
| Prefix tokens, hidden-state distillation, runtime abstention | Removidos do plano ativo |
| Mais SFT amplo, mais epochs, LR sweep, H200 relaunch | Removidos ate existir dado novo |

## V435 - CPU Adapter-Level Pair Builder

Status: ativo.

Objetivo: construir um artefato CPU que prove, antes de gastar GPU, que ha pares submit-safe capazes de atacar erro real do V291/V290 e preservar bit.

V435 nao e treino. V435 e gate, contrato e gerador de pares. Se V435 nao passar, nao abrir HF GPU.

### V435.1 equation_pair_builder

Entrada permitida:

- Public train e sintetico auditado.
- Sem weak/full como fonte.
- Sem `answer` antes de congelar a regra.
- Sem linhas com overlap por `id`, `prompt_sha256` ou prompt normalizado contra weak/full.

Tecnicas permitidas:

- Anti-unification / E-generalization.
- CEGIS / SyGuS para programas pequenos.
- Slot alignment e substring alignment condicionado por operador.
- MDL para preferir regra curta.
- Leave-One-Out para rejeitar regra fragil.
- Renaming stability com renomeacoes bijetivas de simbolos.
- Rejeicao obrigatoria em caso de empate, near-tie ou candidato ambiguo.

Saida esperada:

- Pares `chosen/rejected` apenas quando a regra for unica e label-free.
- `rejected` deve vir de inferencia real do V291/V290 congelado sobre dados permitidos, com raw output, prompt hash e decode config registrados.
- Nao aceitar near-miss inventado como `rejected`.

### V435.2 bit_guardrail_builder

Objetivo: preservar `bit_manipulation>=136/160` enquanto equation tenta subir.

Tecnicas:

- Bit-pair / bitsum / stride curto inspirado na discussao de Tong Hui Kang.
- Traces deterministas curtos, preferencialmente <= 80-120 tokens.
- Hard negatives verificados por programa.
- ANF-SAT/MaxSAT somente como teacher P2 barato, se nao aumentar custo.

Regras:

- Nao perseguir ganho de bit via postprocess.
- Nao aceitar dataset de bit com contradicao de boxed answer.
- Manter proporcao minima de replay/guardrail contra pares de equation.

### V435.3 training_feasibility_gate

Artefato obrigatorio: `v435_pair_manifest.json`.

Schema minimo por par:

| Campo | Obrigatorio |
|---|---|
| source e split | sim |
| prompt_sha256 e prompt_normalized_sha256 | sim |
| family e rule_class | sim |
| program ou regra congelada | sim |
| MDL score e numero de candidatos | sim |
| Leave-One-Out result | sim |
| renaming stability stats | sim |
| slot/substring alignment stats | sim |
| chosen e rejected final answer | sim |
| V291/V290 raw wrong output | sim |
| adapter commit/path e decode config | sim |
| token lengths e truncation check | sim |
| leakage checks por id/hash/prompt | sim |
| `locked_before_answer_audit=true` | sim |

Condicoes para `hf_gpu_allowed=true`:

1. Pelo menos 4 modos de erro/regra independentes em equation, nao apenas 4 linhas.
2. Cada modo precisa hard negative real do V291/V290 em dados permitidos.
3. Volume de pares limpos suficiente por regra para justificar ORPO/DPO curto.
4. Zero overlap weak/full por `id`, `prompt_sha256` e prompt normalizado.
5. Tokenization com truncation 0 e offset/mask valido.
6. Bit guardrail pronto e sem contradicao.
7. Comparativo contra V291/V290 gerado automaticamente.
8. Manifest declara explicitamente `hf_gpu_allowed=true`.

Se qualquer condicao falhar, V435 termina com `hf_gpu_allowed=false` e nao abre GPU.

### Resultado V435 2026-05-15

Artefatos:

- `artifacts/v435_adapter_level_pair_gate/20260515T_v435_cpu_gate/v435_adapter_level_pair_gate_manifest.json`
- `artifacts/v435_adapter_level_pair_gate/20260515T_v435_cpu_gate/v435_adapter_level_pair_gate_pair_audit.csv`
- `artifacts/v435_adapter_level_pair_gate/20260515T_v435_cpu_gate/v435_adapter_level_pair_gate_decision.md`

Resultado:

| Item | Valor |
|---|---:|
| candidate pairs auditados | 3558 |
| approved pairs | 0 |
| approved equation rule modes | 0 |
| bit replay train/validation | 720 / 160 |
| programmatic bit hard negatives | 0 |
| `hf_gpu_allowed` | false |

Bloqueios principais:

- todos os pares existentes faltam raw output real do V291/V290;
- todos faltam identidade/config de decode do adapter;
- todos faltam `locked_before_answer_audit=true`;
- todos faltam certificados MDL, Leave-One-Out e renaming stability;
- bit tem replay, mas nao tem hard negatives programaticos prontos.

Decisao: nao abrir GPU de treino. O proximo passo seguro e coletar raw outputs reais do V291/V290 em prompts permitidos, sem usar answers.

### Resultado V435B 2026-05-15

Artefatos:

- `artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_manifest.json`
- `artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_prompts.jsonl`
- `artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_prompts.csv`

Resultado:

| Item | Valor |
|---|---:|
| public train rows vistos | 9500 |
| bit_manipulation vistos | 1602 |
| equation_transform vistos | 1555 |
| prompts exportados equation | 600 |
| prompts exportados bit | 240 |
| total prompts exportados | 840 |
| answers exportadas | 0 |
| weak/full rows removidas por overlap | 315 |

Decisao: o pack V435B e permitido para coleta de raw outputs reais do adapter. Ele nao e dataset de treino e nao libera V436 sozinho.

## V436 - Short Adapter-Only Smoke

Status: condicional a V435 passar.

Objetivo: testar transferencia real para adapter, com gasto minimo.

Configuracao inicial:

- Preservar config e target modules do V291/V290.
- ORPO/DPO ou preference objective curto, usando somente pares aprovados no V435.
- Evitar loss medio diluido: ponderar span final da resposta/boxed answer quando a implementacao permitir.
- Sem mudanca de tokenizer, prompt oficial, runtime ou pacote.

Kill-switch no primeiro checkpoint:

| Condicao | Acao |
|---|---|
| total <= 192 | cancelar |
| equation <= 56 | cancelar |
| bit < 136 | cancelar |
| truncated > 0 | cancelar |
| package incompatibilidade | cancelar |
| ganho weak e bit preservado | continuar para full gate |

## V437 - Full Gate, Package e Submit

Status: condicional a V436 passar.

Regras:

1. Rodar full official-like somente se weak gate superar V291/V290.
2. Package somente adapter-only root.
3. Submit somente com ganho medido contra o submit historico.
4. A descricao do submit deve declarar versao, checkpoint, weak, full, equation, bit e truncation.
5. Se full nao superar 823/947 ou trouxer regressao de bit/truncation, arquivar e nao submeter.

## P2 Somente Apos V435

Estes itens nao estao ativos agora. So entram se V435/V436 trouxer sinal real:

- Targeted module/freeze controlado para reduzir regressao de bit.
- TIES/DARE/SLERP apenas se houver adapters novos com sinais complementares medidos.
- Distilacao de teacher externa apenas como dados permitidos e auditados, sem weak/full.
- ANF-SAT/MaxSAT para bit se V435 mostrar que ha ganho barato e sem risco.
- Dataset sintetico adicional apenas se passar anti-leakage, MDL/LOO e gate de transferencia.

## Removidos Do Plano Ativo

| Item removido | Motivo |
|---|---|
| Broad SFT, mais epochs, LR sweep | Repetiu teto `equation=56` e frequentemente perdeu bit |
| V391/V398/V413/V416 rawstyle/teacher transfer | CPU bom nao transferiu para adapter |
| Prompt sweep amplo, thinking variants | Sem ganho submit-safe |
| Adapter soups V388/V389 | Sem complemento real e risco de regressao |
| DSLs V421-V433 ja fechadas | Zero ganho novo ou dependencia de label/tiebreak |
| Solver/verifier/postprocessor direto | Nao e package adapter-only |
| Runtime abstention, constrained decoding, logit masks | Fora do contrato submit-safe |
| Prefix/soft-prompt/`embed_tokens` | Risco de nao carregar no vLLM LoRA oficial |
| Hidden-state distillation | Incompativel e exigiria runtime externo |
| Public adapters drop-in | Nao superaram baseline ou quebram compatibilidade |
| Google Drive/CSV archaeology sem novo sinal | Arquivado como evidencia, nao acao |
| `competition_test.csv` como eval | Contem poucas linhas e pode sobrepor train |
| Weak/full misses como treino | Leakage/cherry-pick |
| Decisao por `eval_loss` | Nao correlacionou com ACC das familias |
| H200/A100 relaunch sem V435 | Gasto sem novo sinal |

## Proxima Acao Unica

Executar V435C adapter raw-output collection sobre o prompt pack V435B.

Objetivo:

1. Rodar o V291/V290 congelado nos 840 prompts permitidos do V435B.
2. Salvar `raw_output`, `prediction`, decode config, adapter path/commit e prompt hash.
3. Nao incluir `answer` no input do job.
4. Reexecutar V435 usando esses raw outputs para criar hard negatives reais.
5. So liberar V436 se V435 passar com `hf_gpu_allowed=true`.

Regra FinOps: V435C e inferencia, nao treino. Mesmo assim gasta GPU se rodar no HF/Kaggle. Se nao houver GPU gratuita/barata disponivel, parar aqui ate autorizacao explicita para essa coleta. Enquanto raw outputs reais nao existirem, a decisao correta e nao treinar.
