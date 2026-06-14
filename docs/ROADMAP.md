# ROADMAP TOP-1 — NVIDIA Nemotron Reasoning Challenge
_v19 LIMPO · 2026-06-10 · **FONTE-DA-VERDADE = §30 (validado no held-out real)**. Claims antigos de cobertura (§2,§6,§22-28) foram CORRIGIDOS pelo held-out; onde divergir, vale o §30. Corpus FINAL = v5 (QA+auditoria determinística). Deadline 2026-06-15 23:59 UTC._

## ⭐ TL;DR EXECUTIVO (09-jun — LER PRIMEIRO; consolida §14-19)
- **Posição: rank 39/4099 (TOP 1%), 0.86 robusto**, 65 subs, parados 06-05. Deadline 15-jun.
- **Topo (0.87-0.89) = FARM DE VARIÂNCIA** (Spearman ρ=0.556 submits×score; E[melhor de N]≈μ+σ√(2·lnN); ≥0.87 = mediana 90 subs; 96% no endgame). **NÃO é método secreto.** LB PRIVADO (decide prêmio) deve EMBARALHAR → robustos como nós SOBEM, farmadores caem 0.02-0.04.
- **Busca externa ESGOTADA** (10+ agentes: todos os países, arXiv, papers, OpenRouter+HF free): **NÃO existe atalho público pro 0.86.** Topo público real=0.85 (winner SFT).
- **Causa do ep1 (−0.26, cipher 100→0): regressão do NOSSO v217** (largamos replay/stratified) + max_length 1024 + over-concisão. NÃO é cryptarithm nem rank-32.
- **AÇÕES VIVAS (§19.F; cada uma com pré-teste $0 §19.G):** restaurar v217 (replay+max_len) · probe transposição cipher · +determinant/cross-mult equation · limpar scaffolding bit/cipher · ties_svd soup + VLLM_BATCH_INVARIANT · GRPO(exp).
- **DESCARTADO — NÃO repetir: §19** (receitas mortas, técnicas ilegais no eval, fontes esgotadas, hipóteses refutadas, dark web=proibido, pagos=bloqueados).
- **ORDEM DE EV:** (1) manter 0.86 robusto [mais forte no privado]; (2) fixes reais com pré-teste + GO; (3) re-roll variância = gamble que provavelmente regride no privado.

## 0. POSIÇÃO + OBJETIVO
- **Nós: 0.86** (`submit086`, já submetido = PISO seguro). **Deadline 2026-06-15 23:59 UTC (20:59 BRT).**
- 🔴 **LB MOVEU (08-jun, varredura):** topo **NullSira 0.89** (1 outlier, parc. variância) · 0.88×2 · **0.87×~20** · ~1419 times ≥0.86. **Nosso 0.86 caiu de competitivo → alvo MÍNIMO 0.87.**
- **Objetivo TOP 1.** 5 submits/dia. Só submeter com 99% + autorização + briefing. Teto real ≈ 0.89.
- ⭐ **ESTRATÉGIA RETA FINAL (intel 08-jun, prioridade):**
  1. **VARIÂNCIA é dominante** (±2-3% mesmo temp=0, router MoE; mesmo adapter: 83.6/84.4/84.4/85.5%). → **re-roll do melhor adapter 2-3x/dia = EV+ tanto quanto treino** (`resubmit_best.py`). O 0.89 é parc. sorte.
  2. **Anti-truncamento/concisão = receita 0.87** (CoT curto, nunca truncar antes do boxed) — JÁ é nossa receita.
  3. **PLANO B se FASE 2 regredir as fortes:** `kuangyicheng/nemotron-087-training` — micro-continuation conservadora (28 steps, **LR 1.4e-6**, só `in_proj`, retém ~48% delta, lm_head congelado). Nosso LR=1e-4 é 70x → se o pré-score do ep1 mostrar regressão em gravity/numeral/unit, pivotar pra essa receita.
  4. **MORTOS (não gastar 7 dias):** GRPO · escalar sintético 20K+ · cryptarithm do zero (NP-hard/inaprendível, muro universal) · trocar base/esperar dataset salvador (não existe).
  5. 🔴 **0.87 = COMMODITY (manada de receita pública, confirmado 08-jun):** cluster de ~20 times em EXATAMENTE 0.87 no mesmo dia = copiaram públicos: **`johnjanson/agi-for-medal-0-87-is-possible`** (warmstart `huikang/nvidia-nemotron-all-linear` + SFT curto) + **`kuangyicheng/nemotron-087-training`**. → 0.87 = novo PISO, vai lotar. **Medalha exige 0.88+.** Diferencial vs manada = **dado SOLVER-VERIFICADO** (cipher 100%/bit 89%).
  6. 🔬 **DISSECAÇÃO dos notebooks (agente 08-jun):** receita 0.87 real (kuang) = **só `in_proj` treinável + LR 1.4e-6 + 28 steps + retém 48% do delta** (ultra-conservador). "98/101/108 votos" (megatron/finding-nemo/debatreya) = **re-empacotamento do Tong, NÃO score**. **`nphuong302`** = joia (solver-verificado igual nós + ideias novas).
  7. 🔴 **ALERTA: nosso LR=1e-4 pode REGREDIR o 086** (finishers sérios usam 1e-6/1e-8, 100-1000x menor). FASE 2 é aposta maior-risco-maior-recompensa (substancial SFT cipher/bit) vs manada (nudge conservador). **Gate do pré-score decide.** PLANO B se regredir: **micro-finetune conservador** (in_proj-only, LR~1.4e-6, delta-scale 0.48, 28 steps) sobre 086.
  8. ✅ **Máscara `<think>\n` confirmada** (smoke `[scale-check] 14/75 ATIVO`). 🆕 **Learner booleano por-bit** (nphuong) = mais expressivo → bit>89% ($0). Probe de task-acc held-out durante treino.
  9. ✅ **NOSSO CORPUS NÃO É PLACEBO (verificado no dado real 08-jun):** cipher CoT = **ataque-de-dicionário load-bearing** ('.ing'→única do vocab=king) + **1576 chaves distintas ≥1500** (limiar ALICE p/ generalizar). bit CoT = regra+aplicação (derivação só declarada = melhorável p/ ep2 via Boolformer "prever a regra"). → estamos no jeito que GENERALIZA, ≠ manada (CoT genérico). [crítica do agente era de worktree ANTIGO v284, não o atual]
  10. 🏆 **VANTAGEM confirmada vs vencedor:** ele perdeu pontos com **SVD lossy (75% massa) na conversão Tinker→submit** (adapter submetido ≠ treinado). NÓS treinamos DIRETO Unsloth/PEFT (warmstart 086, sem SVD) → imunes. [validar formato MoE-LoRA carrega no vLLM — 086 já provou]
  11. 🎲 **VARIÂNCIA = 100% host-side** (VLLM_BATCH_INVARIANT é lever do host, não nosso; causa = batch composition muda routing MoE). Aceitar como ruído. **Variance farming math:** best-of-3 ≈ +0.85σ, best-of-5 ≈ +1.16σ (~+1%). Alocar ~3 submits/melhor + 2 explorando; **escolher finalistas por MÉDIA (não pico, que é enviesado)**; preferir adapter de menor σ.
  12. 🆕 **Anti-variância controlável (pós-ep1, baixo custo):** EMA/soup de checkpoints (-ep1+-ep2, MESMA init) → flat minima → menos flip de router. Selecionar checkpoint por **margem (min-logprob)**, não loss. EFlat-LoRA (ρ~0.1) no próximo treino.
  13. 📊 **TETO POR FAMÍLIA (DEFINITIVO, forense 08-jun):** headroom em pts-LB se a 100%: cryptarithm_deduce **+6.24** · cry_guess +1.55 · eq_guess +1.22 · eq_deduce +0.94 · **bit +1.85** · cipher/num/grav/unit **+0.00 (SATURADAS)**. Total cryptarithm/equation ~10pts MAS **intransferível** (697491: cry 8→90% no train → LB CAIU p/ 0.82-0.84 por esquecimento; lkevincc0 é gold-conditioned). **0.86→0.87 = formato/truncamento** (CoT concisa + **delta-scaling 0.48**), NÃO capacidade. **0.89 = VARIÂNCIA** (LB 1430→20→2→**1**; N=1 outlier; mesmo adapter ±2-3pts; nem os 0.87 sabem como o 0.89 fez).
  14. 🧭 **REVISÃO ESTRATÉGICA:** nossa FASE 2 (SFT LR 1e-4) só vale SE o 086 NÃO estiver saturado em cipher/bit. **Pré-score do ep1 decide:** (a) 086 já alto em cipher/bit? (b) ep1 subiu sem regredir? Se 086 saturado OU ep1 regrediu → **PIVOTAR pro conservador (delta-scaling 0.48 + in_proj + LR~1e-6) + variance re-roll** (caminho PROVADO 0.87). Não forçar cryptarithm (regride). Mira: 0.87 sólido + re-roll p/ spike 0.88+.
  15. 🗾 **PESQUISA JAPONESA (Sakana/FuriosaAI, 08-jun) — TIER 1 acionável:** **LoRA SOUP** (`modal_soup.py`, PEFT `add_weighted_adapter`): mesclar ep1+ep2+086 (mesmo run = basin → **linear soup ideal**, Model Soups) + mini-sweep de coef no val_clean (evolutionary-merge do Sakana, versão barata) = **3º candidato variance-robusto**, custo ~0. ⚠️ **excluir `lm_head` do soup** (conflito de escala). EMA(α=0.998, início 50%) se retreinar. **CONFIRMA (SSM-PEFT FuriosaAI ICML'25, Lemma 1):** LoRA em in_proj/out_proj CERTO; NÃO adaptar A/B/C/Δ. Hype p/ 7d: Evol-Merge/M2N2 full-model (caro), VMoER router (precisa treinar router).
  - **PLANOS DE CANDIDATOS:** A) FASE 2 ep1/ep2 (solver-data) · B) conservador delta-scale 0.48 (`modal_planB.py`) · C) **SOUP** dos melhores (`modal_soup.py`) · piso) 086. Pré-score TODOS no val_clean → submeter melhor por MÉDIA + re-roll.
  16. 🗾 **COMUNIDADE-JP (08-jun):** sem writeup-JP 0.87 novo. **CONFIRMADO nosso EDGE:** konbu17 (559 votos, 0.85-tier) tem cipher 94.3%/bit 40.3% — NÓS cipher 100%/bit 89% (dado superior ao público). Refinos: (a) **branch-weighted loss** do vencedor (peso `min(1,|logprob|/0.01)` upweight tokens incertos) — opção de treino; (b) **reserva de truncamento** (cortar CoT em 7600 c/ folga p/ `</think>\boxed{}`; confirmar geração não estoura no pré-score); (c) augmenters atômicos (spelling/concat/split). **Equation rico-catalogo NÃO aplica** (já testei = 0%, nossa família é símbolo, não numérico).

## 1. EVAL (config CRAVADA pelo host)
- **temp=0.0 · top_p=1.0 · max_tokens=7680 · max_model_len=8192 · max_lora_rank=32 · gpu_mem=0.85 · enable_thinking=True · exact-match.** Greedy, 1 amostra (→ self-consistency/majority NÃO servem).
- Prompt: sufixo `\nPlease put your final answer inside \boxed{}...` + `apply_chat_template(add_generation_prompt=True, enable_thinking=True)`. O template já injeta `<think>\n` (treino deve gerar o raciocínio DEPOIS dele — alinhamento verificado).
- **verify:** `[01]+`→string-exato (case-insens) · senão `float` isclose(rel_tol=1e-2,abs_tol=1e-5) · senão string case-insens. Extrai o ÚLTIMO `\boxed{}` não-vazio.
- 🔎 **FONTES OFICIAIS KAGGLE (confirmado 08-jun, notebooks staff Ryan Holbrook):**
  - **`ryanholbrook/nvidia-nemotron-submission-demo`** (2662 votos, staff): submissão = `model.save_pretrained()` → `zip submission.zip *` (adapter_config+safetensors). **Demo LoRA: r≤32, alpha=16, target_modules=`.*\.(in_proj|out_proj|up_proj|down_proj)$`** (Mamba+MLP; SEM lm_head/attn no baseline — nós ADICIONAMOS lm_head+attn e 086=0.86, então ajuda). Modelo=`metric/nemotron-3-nano-30b-a3b-bf16`.
  - **`ryanholbrook/nvidia-utility-script`**: ambiente de EVAL = **RTX Pro 6000 (Blackwell SM12.0, 96GB)**, CUDA 12.8 (torch nightly cu128), **flash-attn + mamba**, `enable_internet=false` (OFFLINE). → 96GB confirma que adapter ~4GB cabe (item D); eval usa **FA2** (nosso treino usa eager; nosso pré-score enforce_eager — fonte de parte da variância ±2-3%, não controlável).
  - `test.csv` = 1 linha amostra (id,prompt); `train.csv` = 9500.
  - 🏆 **CÓDIGO OFICIAL DA MÉTRICA ACHADO+CONFRONTADO (`kaggle.com/code/metric/nvidia-nemotron-metric`):** nossa impl bate **100%** — `verify()` (binário→str case-insens · float isclose rel=1e-2/abs=1e-5 · str case-insens, só `.strip()` nas bordas, espaço interno conta), `extract_final_answer` (último `\boxed{}` não-vazio via **`rfind('}')`** simples, NÃO balanceia chaves; fallbacks: "final answer is"→último número→última linha→NOT_FOUND), sufixo do prompt (byte-idêntico), `enable_thinking=True`, temp=0/top_p=1/max_tokens=7680/model_len=8192/rank≤32/gpu_mem=0.85/**dtype='auto'**/n=1 greedy. vLLM oficial: `max_num_seqs=64, enable_prefix_caching=True, enable_chunked_prefill=True` (sem seed, sem stop, sem enforce_eager).
  - 🔴 **CORREÇÃO G — score = accuracy por LINHA simples** (`num_correct/len(solution)`), **NÃO 16.7% fixo por família**. train.csv da COMPETIÇÃO é balanceado (~1580/fam) → na prática ≈ igual; mas o test PRIVADO é oculto (não assumir uniforme cegamente). NÃO confundir com a distribuição do andy279 (bit/equation dominam lá — NÃO é a competição).
  - 🔴 **CORREÇÃO truncamento:** CoT+resposta deve caber em **<7680 tokens** senão corta sem `\boxed{}` fechado → cai no fallback → quase sempre ERRA. (nossos traces ≤2800 tok ✓; monitorar comprimento de GERAÇÃO no pré-score.)
  - 🔴 **BUG repo:** `src/competition_utils.py` `OFFICIAL_INFERENCE_CONFIG` estava com defaults mortos (temp=1.0/3584/4096) → corrigido p/ 0.0/7680/8192 + max_num_seqs=64.
  - ✅ **VALIDADO POR TESTE DIFERENCIAL (08-jun, código oficial baixado + rodado):** `verify` nosso == oficial em **10/10 casos-limite** (inclui `"0"`vs`"0.0"`=False [binário], `1.5`vs`1.51`=True [tol 1%], espaço-interno conta). **Corpus inteiro = 5704/5704 = 100% sob extract+verify OFICIAIS** (dado de treino perfeito p/ a métrica real). 🔧 **GAP achado+corrigido:** nosso `extract` não tinha os **fallbacks** do oficial (sem boxed → "final answer is"/último número/última linha/NOT_FOUND) → **substituído pelas funções oficiais VERBATIM** no `hf_prescore_vllm.py` (extract+verify byte-idênticos; call-site `verify(gold,pred)`). vLLM do pré-score alinhado (`dtype=auto, max_num_seqs=64, prefix+chunked prefill`).
  - ⚠️ **NÃO duplicar sufixo:** val_clean JÁ tem o sufixo embutido; o `bp()` do pré-score NÃO adiciona de novo (= official: puzzle+sufixo→template). Verificado.
  - eval flow: submission.csv['prediction']=caminho do adapter; `score()` CARREGA o adapter + GERA + extract + verify; `accuracy=num_correct/len(solution)` (por-linha). GPU eval RTX Pro 6000 Blackwell.
  - ✅ **CONFIRMADO PELO HOST (discussões Kaggle, agente 08-jun):** (1) params da **Evaluation page SOBRESCREVEM os defaults** do metric.ipynb → **7680/0.0/8192/64/rank32/top_p1** são os REAIS (ignorar 3584/1.0/4096). (2) **6 famílias BALANCEADAS ~1/6** no train (bit 1602/grav 1597/unit 1594/cipher 1576/num 1576/eq 1555) E host diz test "roughly similar" → **cada família ≈16.7% do score** (G validado). (3) test = **"several hundred problems"** (~300-600); LB público ≈50%, final ≈50%. (4) **só dá pra submeter o LoRA** (prompt de inferência é FIXO, sem scripts). (5) **rank≤32 hard cap**, sem limite de bytes. (6) deadline **2026-06-15** (7 dias), 5 submits/dia, 4072 times, $106k.
  - ⏱️ **ORÇAMENTO ~9h de inferência** (consenso comunidade, não cravado pelo host) p/ ~300-600 linhas: a parede é **THROUGHPUT, não tokens/linha** → **CoT DENSO > CoT longo** (inflar tokens derruba throughput sem ganhar score). Reforça: equation dropada + CoT conciso = certo.

## 2. AS 6 FAMÍLIAS (train.csv 9500, ~16.7% cada) — STATUS ATUAL
| Família | Peso | Modelo hoje | NOSSO solver (CPU/$0) | Status |
|---|---|---|---|---|
_(STATUS atualizado p/ COBERTURA DO SOLVER medida no HELD-OUT real (val_clean 30/fam) — ver §30)_
| Família | Solver held-out | Corpus (CoT sound) | Status |
|---|---|---|---|
| **cipher** | **100%** (30/30) subst.+vocab77+**injetividade** | 1576/1576 | ✅ ganho grande validado (era lixo) |
| **bit** | **20%** (6/30) — classes unary/+const/XOR-mask/binop/maj/depth-2 | 1410/1428 | ✅ +10pts (era 10%); resto não-linear=teto |
| numeral | 100% (30/30) romano | 400 | ✅ |
| gravity | 100% (30/30) `d=0.5gt²` | 400 | ✅ |
| unit | 100% (30/30) `y=fator·x` | 400 | ✅ |
| **equation** | **~16%** (dígito 13% + concat-símbolo 3%) | 1100 | ⚠️ eval real é SIMBÓLICO; aritmética-sob-substituição = sub-determinada (teto provado por 6 modelos top) |

## 3. AS 3 PAREDES DO PLATÔ
1. **NÃO-DETERMINISMO ±2-3** (eval-side): kernels não-batch-invariant → flip de expert (top-5+1 shared) → Mamba-2 amplifica (bf16 9.15% desvio). vLLM JÁ tem fix Batch-Invariance, mas o **host decide ligar** → não controlamos. **Mitigação: selecionar adapter ROBUSTO (mínimo plano).**
2. **ESQUECIMENTO:** treinar família difícil regride as maxadas (rank≤32). **Medido: vnext (bit cru, sem replay) → bit −0.20.** **Mitigação: replay estratificado SEMPRE + interpolação/EMA/CorDA-KnowledgePreserved.**
3. **CORREÇÃO DOS SOLVERS (não truncamento — <0.1% passa de 7680):** gargalo = **cryptarithm/equation_guess**. cipher e bit RESOLVIDOS por solver agora.

## 4. ✅ NOSSAS VANTAGENS
- **Unsloth/PEFT DIRETO** → SEM o misalignment treino-serving do SVD do huikang (ele perde 0.877→0.85 no Tinker). Transferência fiel.
- **Pré-score vLLM próprio** (`hf_prescore_vllm.py`, `vllm==0.19.1` cu12) = mede fielmente SEM gastar submit Kaggle.

## 5. 🧪 PRÉ-SCORE REAL (val_180, vLLM = motor do eval · custo $1.85)
`086 GERAL=0.672` · `vnext(bit cru)=0.661` (Δ−0.011; **bit −0.20** = backfire). Absoluto ≠ 0.86 LB (val uniforme sobre-pondera difíceis) MAS é **régua relativa válida**. **vnext NÃO submeter.**

## 6. 💎 ACHADOS-CHAVE (5 rounds — só o que importa)
### Solvers das famílias fracas (FEITO, validado CPU/$0)
- **cipher**: decodificar contra o **dict-77** (não letra-a-letra) → 100%.
- **bit**: enumerar a **DSL de ops** (huikang `investigators/bit_manipulation.py`) → 89%.
### Gargalo = cryptarithm (consenso da competição: "quebrar 0.86 exige breakthrough em cryptarithm")
- `AB op CD = result`, símbolo→dígito único, cada SÍMBOLO-operador→uma operação.
- 🔎 **Confirmado:** nossa família é símbolo→dígito (cryptarithm), NÃO equation-literal — o reasoner RICO de equation do huikang (literal, 603 linhas) deu **0%** nos 732 casos-dígito. O "equation deduce 90%" do huikang é de OUTRA família (`equation_numeric` literal) que NÃO está no nosso train.csv.
- 🚧 **VEREDITO DEFINITIVO (7 solvers CONSTRUÍDOS + validados 08-jun): cryptarithm é MURO FUNDAMENTAL — não-crackável.** É **subdeterminado** (5 exemplos + espaço de regras grande → mais busca = mais OVERFITTING, não mais acerto). Provas: vencedor 8% · **GPT-5.4 falha** · cryptarithm_guess (op não aparece) = ambíguo.
  - **7 tentativas (todas CPU/$0):** concat-only **10% (melhor)** · CSP-5ops ~10% · literal-26ops 5% · **equation_numeric-RICO-30ops 0%** · engenhariado-26ops 3.8% · **consenso+Occam(t=.3) 5%** · **consenso+Occam(t=0) 5%/+9 erros**.
  - CHAVE: **adicionar ops PIORA** (consistência espúria); o solver **mais simples** vence. Implementadas as **2 melhorias recomendadas por agente de pesquisa** (Occam-ordering + voto de consenso) → **NÃO** bateram o baseline simples → confirma hardness, não falta de engenharia.
  - **PARAR — não é fonte-faltante nem engenharia-faltante, é hardness matemático.** Headroom = só a fração ~10% solúvel (andy279/concat-only cobrem no corpus). Score vem de **cipher+bit (feitos)**.
- **"guess" (cryptarithm/equation_guess)** = underdetermined → problema de **PRIOR bayesiano** (treinar a dar a resposta mais provável/simples), não de solver.
### Técnicas de treino (frontier 2026 — aplicar na FASE 2)
- **LOSS PONDERADA (SCALe):** peso alto no `\boxed{}`, ~0.1 no `<think>` → otimiza o exact-match. **+3-5%.** (par com min-logprob)
- **MIN-LOGPROB duplo uso:** objetivo (greedy determinístico) + **FILTRO DE DADOS** (não treinar o já-sabido → libera rank-32 pro gargalo).
- **DoRA** (>LoRA mesmo rank, **submetível via vLLM-LoRA**) + **alpha=64** (NVIDIA recomenda; usamos 32).
- **Anti-esquecimento:** replay estratificado + interpolação/EMA mesmo-rank + (init CorDA Knowledge-Preserved / O-LoRA).
- **self-distill STaR/RFT / on-policy distillation** (NVIDIA Nemotron-Cascade-2, NOSSA arq): gerar→manter verificados→re-treinar → "torna a capacidade acessível no greedy".
- **CCE** (loss sem materializar logits 131k → libera VRAM) · **template DIVERSO** (CrossThink) · **ESFT** (LoRA só nos experts ativados).
- **🆕 MUON optimizer (Kimi/Moonshot, China):** 2× eficiência vs AdamW, menos memória (1 buffer). Pra LoRA o melhor é **híbrido Muon+AdamW** (> Muon > vanilla AdamW) → melhora convergência/loss. Paper Muon-LoRA (2602.06385). Lever de treino a testar (usamos adamw_8bit).
### Consciência de arquitetura
- **Mamba: LoRA fraco em SSM** → budget vai p/ **attn+MLP+lm_head** (não os 128 experts difusos). **4-bit NÃO suportado** no Nemotron-3 → **bf16/80GB obrigatório** (explica OOM no 40GB). Arq: 23 Mamba+MoE / 6 attn; 128 experts +1 shared; 5/token.
### Fontes-mãe (dados)
- **andy279/nemotron-reasoning-challenge** (49k traces verificados; bit/equation/cipher) — reformatar (tinha bug `}}` 67% + `<think>` vazio). **Reasoning Gym** (`open-thought/reasoning-gym`) = geradores+verificadores (variantes custom; usar p/ generalização).

## 7. 🏗️ FASE 1 — EXECUTADA + AUDITADA 10× ($0, desk-test PASS)
**CORPUS `fase1_corpus.jsonl` (HF `kg1-train-bundle`): 5704 traces → 4204 EFETIVOS (equation DROPADA no treino).** Composição treinada: cipher 1576 · **bit 1428** · replay numeral/gravity/unit 1200. Desk-test com TOKENIZER REAL: verify/ACC **5703/5704=100%**, **0 bug `}}`** (2 `}}` estão no PROMPT do puzzle = dado correto), 1 `<think>`, alinhamento `<think>` train↔eval ✅, EOS `<|im_end|>` ✅, máscara `train_on_responses_only` ✅, **0 truncamento** (max 490 ≤ 1024).
- 🚫 **equation DROPADA (decisão convergente: finops + ACC + anti-backfire):** era 26% dos traces mas **45% do COMPUTE** (CoT 315-palavras confabulado), família teto-~10% (upside ~0). Evidência: arXiv 2510.16022 (podar não-generalizável = +5pp nas outras) + agente web + auditoria. Cortar = **−45% tempo/custo E +2-5pp ACC E replay vira 28.5% (ideal ~30%)**. 086 já faz ~0.47 em equation via warmstart (mantém sem treinar). Se pré-score mostrar regressão, adicionar ~300 traces deduzíveis CURTOS (concat-solver), não os confabulados.
- 🐛 **Bug silencioso achado+corrigido:** SCALe ancorado em `\boxed{` era NO-OP (tokenização dependente de contexto) → reancorado no token único `</think>` (id 13), validado nas 6 famílias. `compute_loss` custom reintroduzia o bug de normalização do grad-accum → corrigido com `num_items_in_batch` (loss smoke 74→6.22 confirma).
- ✅ **CROSS-CHECK TRIPLO (08-jun, fim do tentativa-e-erro):** agente web (literatura) + **OpenRouter 13 modelos** (gpt-oss-120b, nemotron-ultra-550b, owl-alpha...) + auditoria/finops CONVERGEM: (1) dropar equation = **13/13 net-negative**; (2) replay 25-50% (nosso 28.5% ok); (3) SCALe 2x (não 5x); (4) **única alavanca sem teto = qualidade de dados cipher+bit**. Smoke de validação H100: equation dropada ✅, scale-check 2x ATIVO ✅, sem OOM, ~66s/step → **~4.8h/época**.

## 8. ▶️ FASE 2 — TREINO (config FINAL auditada; infra PRONTA)
- **Receita FINAL (decisão tomada, fim do tentativa-e-erro):** warmstart 086 · rank32 **LoRA standard** (DoRA/alpha64 DESCARTADOS: incompatíveis com warmstart, distorceriam o piso 0.86) · train attn+MLP+lm_head · 1 época (263 steps) · LR 1e-4 · `adamw_8bit` · grad-checkpoint · replay 28.5% · **SCALe `</think>`-anchored peso 2x** (agente: ≥5x constante = instabilidade) · `num_items_in_batch` (grad-accum correto) · max_length 1024 · save_strategy=no · masking `<|im_start|>assistant\n`.
- **FINOPS:** H100 ~66s/step → **~4.8h/época** (equation dropada cortou ~45%). gc=OFF testado → MAIS LENTO (98s, Unsloth offloada grad = PCIe-bound) → **gc=ON**. tf32 ON.
- ⭐ **REGRA: DIVIDIR TREINO POR ÉPOCA.** `EpochSaver` salva+push CADA época (`OUT-ep1`, `-ep2`...) → pré-score independente + early-stop (para na época que para de melhorar; evita overtraining/esquecimento e gasto). Rodar `--epochs N`.
- 🔴 **LIÇÕES DURAS (incidentes 08-jun, NUNCA repetir):**
  1. **`modal run` SEM `--detach` morre quando o client desconecta** → matou o 1º run a 26% sem salvar (~$4 perdidos). SEMPRE `--detach`.
  2. **`--detach` + `.remote()` AINDA pode ser cancelado no disconnect** (aviso do próprio Modal). Solução real: **`train.spawn()`** (fire-and-forget, roda 100% independente). USAR SEMPRE `.spawn()` p/ treino longo.
  3. **`save_strategy="no"` = perda total se cair** → SEMPRE checkpoint rolling (`StepSaver` a cada 100 steps → `-latest`) + cleanup `/tmp` (`shutil.rmtree` pós-push, senão enche disco no multi-época).
  4. **Custo real H100 ~80-88s/step** (~$3.5-4/h) → 1 época ~6h ~$23; 3 épocas ~$65 >> créditos $30. Dimensionar épocas ao budget + early-stop por época.
  5. `modal app stop <id> --yes` (não-interativo).
  6. 🔴 **MITO DO "≤120MB" DESMENTIDO (08-jun, header do safetensors):** o adapter r32 deste Mamba-30B é **~4.26GB** (LoRA A/B sozinha = 3.55GB — in_proj/out_proj do Mamba são enormes; + 704MB de `lm_head.base_layer` redundante). NÃO existe versão 120MB. **086 (4.26GB cru) marcou 0.86 → a competição aceita ~4GB; vLLM carrega o cru (pré-score provou).** `submit_candidate.py` (zipa o cru, 2 arquivos) está CORRETO — NÃO faz strip nem precisa. **pré-score(cru)==submit(cru) → sem mismatch.** Ep1 será submetido CRU igual 086.
- **Infra validada:** **Modal** (auth+secret+app `modal_fase2_train.py` prontos; A100-80GB; **$30 grátis → treino $0**; imagem pytorch-devel cacheia mamba+modelo) **OU** HF Jobs (`a100-large` $2.50/h; smoke OK, loss 91→14.9 confirmou que aprende).
- **GATE:** **pré-score vLLM > 086 em TODAS as famílias-piso E no geral** ANTES de cogitar submit. Piso = submit086.
- **FASE 3:** interpolação/EMA mesmo-rank (free) + medir σ com re-submits → **selecionar 2 finais por CV-robusto (média−desvio), NÃO pela pública** (winner's curse).

## 9. ❌ NÃO FAZER (becos confirmados)
merge/soup SVD (cap rank32; interpolação mesmo-rank é OK) · GRPO/RL (só se pass@8 bom; viável só em cryptarithm c/ verificador) · synthetic SEM verificação · distillation NÃO-verificada · **bit cru sem replay** (backfira −0.20) · treinar router MoE (congelado) · raciocínio latente Coconut/CODI (não submetível) · test-time-training (adapter fixo) · QMC-bit (53%) · PiSSA/MiLoRA init (colapsa em reasoning).

## 10. QUIRKS DA MÉTRICA — ENGENHARIA REVERSA (08-jun, TESTADO rodando extract+verify OFICIAIS)
**Não há cheat (gold oculto, adapter não executa código, prompt fixo pelo harness). Mas o código revela brechas LEGÍTIMAS:**
- 🪤 **Armadilha `[01]+`:** QUALQUER número só com 0/1 (10,11,100,1000...) é tratado como BINÁRIO → match ESTRITO, SEM tolerância 1%. Afeta só equation (2%, dropada); gravity/unit/numeral=0. → numérico: reproduzir formato EXATO do gold (sem `.0` espúrio).
- 🛟 **Rede de segurança NUMÉRICA:** sem boxed, o fallback "último número" (`-?\d+(?:\.\d+)?`) captura o número do raciocínio → gravity/unit são ROBUSTAS a falha de boxing. **Cipher/bit/string NÃO têm essa rede** (só fallback "última linha", fraco) → **boxing confiável é CRÍTICO p/ cipher/bit** (SCALe no boxed é exatamente certo).
- 📦 **Boxing limpo (testado):** `}}` → extrai `ans}` = FALHA (emitir 1 `}`). · unidade no boxed (`54.9 m`) = FALHA (número limpo). · texto DEPOIS do boxed = OK (rfind acha o `}` da resposta). · **só o ÚLTIMO boxed conta** → NUNCA boxar depois da resposta final (um boxed errado no fim sobrescreve o certo).
- 🔢 **Binário = 8 bits EXATOS** (sem `0b`, sem perder zero à esquerda; sem tolerância) — família de menor margem. **String = case-insens, mas espaço interno conta** (cipher: espaçamento exato entre palavras).
- 🎲 **Variância re-roll (LEGÍTIMO):** temp=0 mas não-determinismo de batching MoE → ±0.02-0.03 por run → re-submeter o MELHOR adapter (5/dia) pega o "spike de 0.87" (o que os submits do 086 fizeram: "lottery for 0.87 spike"). Tática p/ o empurrão final.
- **Implicação p/ treino:** a métrica RECOMPENSA exatamente nossa receita (CoT conciso + boxing limpo + SCALe). Confirmado, não suposto.

## 11. AMBIENTE + REGRAS
- **Treino: Modal ($0 via $30 créditos) ou Colab Pro 80GB ou HF a100 ($2.50/h).** GPU comparativo em `gpu_cloud_pricing.md` (Vast.ai $0.67/RunPod $1.49 mais baratos, mas conta própria).
- Stack: torch 2.7.1 (cu12!) · unsloth · mamba_ssm 2.2.5 · transformers 4.56.2 · trl 0.22.2 · `import unsloth` 1º. Secrets: HF=`HF_KEY` (Colab) / `kg1-hf` (Modal). **vLLM treino-OK = 0.19.1 (0.20+ = cu13 quebra no driver 12.9).**
- ⭐ **REGRAS:** teste de mesa (estático + CPU grátis) ANTES de gastar · pré-treino 2 steps + pré-score antes de job · nunca submit sem 99%+autorização.

## 12. 📁 ARTEFATOS (não re-buscar)
- **Solvers/CoT nossos:** `cipher_solver.py`+`cipher_reasoner.py` (100%) · `bit_validate.py`+`bit_cot_gen.py` (89%) · `crypt_ext_solver.py` (cryptarithm CSP estendido, em validação) · `easy_reasoners.py` (numeral/gravity/unit) · `nemo_generators.py` · `fase1_assemble.py`.
- **Treino:** `modal_fase2_train.py` (Modal) · `hf_fase2_train.py` (HF Jobs, gist `FELIPEACASTRO/e060ddd...`) · `kg1_fase2_cell.py` (Colab).
- **Pré-score:** `hf_prescore_vllm.py` (extract+verify OFICIAIS verbatim, vLLM alinhado; no HF como `prescore_script.py`) · `off_metric_fns.py` (funcoes oficiais) · `score_offline.py` (scorer CLI por-familia).
- **Auto-pipeline (08-jun):** `monitor_autopre.py` (poll ep1 → dispara HF Jobs pre-score → poll resultado, hands-off) · `submit_candidate.py` (validador rank≤32 + submit CRU, sem strip) · `resubmit_best.py` (variance re-roll, budget-aware, exige autorizacao) · `test_metric_diff.py`+`reverse_engineer_score.py` (provas diferenciais).
- **Código oficial baixado:** `off_metric/nvidia-nemotron-metric.ipynb` · `off_orch/competition-metrics-orchestrator.py` · `rh_demo/` (demo host).
- **Dados:** `nemo_data/train.csv` (9500) · `fase1_data/*` (cipher/bit/crypt/andy clean + `fase1_corpus.jsonl`) · HF `kg1-train-bundle` (corpus+val_180+prescore_result+pricing).
- **Externo:** huikang repo (`investigators/bit_manipulation.py`, `cryptarithm_deduce.py`) · `andy279/nemotron-reasoning-challenge` · `nvidia/Nemotron-RL-ReasoningGym-v1` · `open-thought/reasoning-gym` · lkevincc0 equation solver.
- **Papers:** SCALe(loss-pond) · DoRA · EVA/CorDA/LoRA-GA(init) · ESFT(2407.01906) · MambaPEFT(2411.03855) · CCE(ICLR25) · Nemotron-Cascade-2(2603.19220) · CrossThink(2504.13941) · lookahead-carry(2502.19981) · STaR(2203.14465) · **Muon/MuonClip(Kimi-K2 2507.20534)+Muon-LoRA(2602.06385)** · CSP(MRV+FC+AC-3).
- **Baseline/medido:** `submit086` (0.86 PISO) · pré-score 086=0.672/vnext=0.661 (vnext NÃO submeter).

## 13. 🔒 SEGURANÇA
`.env` em `Workspace/Copa/.env` tem 5 chaves vivas (OpenRouter/HF/Kaggle/TheOdds/Modal). **Rotacionar TODAS após a competição** (expostas em prompts/memória).

## 14. 🔬 VARREDURA DEVASTADORA + RESULTADOS REAIS (09-jun-2026)

### 14.1 RESULTADOS MEDIDOS (val_clean 180 itens, métrica oficial verbatim)
- **086 baseline:** 0.756–0.761 (VARIA ±0.04 por variância MoE run-a-run; outlier velho 0.672). Por isso re-medir 086 dentro de cada run.
- **planB** (delta-scale 0.48 = receita kuangyicheng 0.87): **0.750 → delta −0.006 = EMPATE** com 086. Sem ganho. (cipher 1.0 / bit 0.37 / eq 0.13). Serve só como piso/re-roll, NÃO é 0.87 confirmado.
- **ep1** (FASE 2 SCALe): **0.506 → delta −0.256 = DESASTRE.** cipher **1.0→0.0 (COLAPSO TOTAL)**, bit 0.43→0.03. **Causa:** SCALe (peso na resposta + corpus curto) → concisão excessiva (~150 tok/prompt vs ~3800 do 086) → modelo PARA de raciocinar. ⛔ **ABANDONAR a receita SCALe.**
- **Integridade de artefato** (verificada via Range, sem baixar 4GB): r=32, 12011 tensores, 0 NaN/Inf, freeze ok (in_proj treinado, resto == 086), delta-scale aplicado. Artefato correto — o problema é a RECEITA, não o build.

### 14.2 INTEL DAS DISCUSSÕES (o lever REAL do topo)
- 🔴 **`cryptarithm_deduce` ~27.8% + base 0.85 = TOPO do LB.** Método citado: somar **dígito-a-dígito anotando carryover (vai-um)** pra localizar o erro.
- ⚖️ **Conflito honesto:** nossos experimentos de FORÇAR cryptarithm **DERRUBARAM o LB** (muro intransferível); o catálogo dígito-a-dígito rendeu em *equation* (13.6→22.7%), **não** em cryptarithm. Os pontos estão lá, mas é genuinamente difícil — quem chega a 27.8% fez o modelo **raciocinar**, não decorar.
- ✅ **CONCLUSÃO DEFINITIVA:** o ÚNICO lever que sai de 0.86 é **cryptarithm**. Soup/delta-scale/SCALe = **ruído de variância** em volta de 0.86 (comprovado: planB empatou, ep1 quebrou).

### 14.3 SOUP / ADAPTER ENSEMBLING — framework público pronto
- **lopure/is-adapter-ensembling-a-good-idea** (Kaggle 09-jun): funde huikang+konstantinboyko+kienngx com **4 estratégias** (SVD+rsLoRA, Simple Average, TIES, DARE) + **9 métricas** (exact/category acc, avg tokens, failure overlap Jaccard, pairwise disagreement, merge retention = acc_merged/max(parents), OOD subset, answer entropy). **Reutilizar o framework de avaliação.** Ainda não confirmado qual merge vence (output não trouxe verdito).

### 14.4 ACHADOS NOVOS TAGUEADOS
- 🔴 **ALTO:** lopure ensembling · **assiabenazzouz/adappter-v32-epoch-5** (adapter público que o nb "Top Score"/92v só baixa+submete) · **Naribow/nvidia-nemotron-progress-prize** (HF **31.5k downloads** — provável base do vencedor) · **matthewagi/nemotron-30b-crypt-rl-reward** (HF — RL em CRYPTARITHM = nosso muro).
- 🟡 **MÉDIO:** rohanrk1813/nvidia-comp (246v) · wethepeople918/megatronthedecepticon (99v) · biohack44 v62 "sparse-trust finisher attack" · varianceofx/synthetic-dataset-generator-27k · konstantinboyko (unsloth b2 e2).
- ⚪ **RUÍDO:** nvidia/Nemotron-CC/Math/VLM/Personas (pré-treino do modelo, não da comp) · modelos ASR/OCR/Omni/Ultra-550B · asadullah/AI-benchmark.

### 14.5 INCIDENTE FINOPS + REGRA NOVA (09-jun) ⚠️
- **Incidente:** 3 pré-scores rodando juntos (auto-disparo armado + disparo manual meu = DUPLICATAS). Desperdício direto ~$0.70 + ~$45 em bets que não pagaram (Plano A SCALe regrediu, Plano B empatou).
- 🔒 **REGRA NOVA (PERMANENTE):** NENHUM job (treino OU pré-score, Modal OU HF) sem **GO explícito por-job + custo estimado na frente**. **Auto-disparadores ELIMINADOS.** Só monitores read-only ($0). Protocolo cancelamento: monitor detecta → eu analiso → recomendo → usuário decide.
- **Ações:** Plano A CANCELADO (ep2 só pioraria, economizou ~$30/5.4h). ep1/ep2/planB descartados como subida. **086 = 0.86 floor INTACTO e submetível.**

### 14.6 PRÓXIMO PASSO (offline, $0, sem treino)
Construir/validar **CoT de cryptarithm dígito-a-dígito com carryover** (método das discussões), alvo ~27.8% nessa família **sem regredir cipher** (erro do ep1). Trabalho de DADOS na CPU primeiro; treino só com GO.

### 14.7 SWEEP DEVASTADOR v2 (09-jun, 3 agentes + OpenRouter free) — ACHADOS QUE MUDAM O JOGO
**🔴 DADOS PRONTOS PRA O GARGALO (cryptarithm):**
- **carbonteq/rg-cryptarithm-instruct-100k** (HF, MIT) — **100K cryptarithm CoT procedurais (reasoning-gym), GRPO-ready, + 4096 test.** Match direto do nosso gap. ⭐ usar.
- **reasoning-core/reasoning-gym** (HF, 1.4M ex, 104 envs incl. cryptarithm) — gerador-fonte. **nvidia/Nemotron-RL-ReasoningGym-v1** (15k, oficial).
- **alex-gapch/nemotron-cryptarithm-narrative** (HF) — CoT narrativo de cryptarithm.
- **carbonteq** test set (4096) serve como val honesto extra.

**🔴 DIAGNÓSTICO da queda do ep1 (causa + conserto principista):**
- ep1 colapsou cipher porque (a) **sem REPLAY** das famílias saturadas + (b) over-concisão do SCALe.
- ✅ **Conserto 1 — REPLAY/data-mix** (tahaalam2009 "Replay_Data 0.86" + consenso OpenRouter): treinar cryptarithm SEMPRE misturado com replay (~5-50%) de cipher/numeral/gravity/unit → não esquece.
- ✅ **Conserto 2 — truncation-budget** (Nemotron Nano2 report 2508.14444): truncar CoT a 1-2k tokens MAS preservar a resposta → ensina raciocinar dentro do budget SEM colapsar (≠ SCALe que cortou o raciocínio).

**🔴 CRYPTARITHM precisa de GUESS, não só CoT determinístico** (discussion/689915): deduce 8.2% / guess 6.7%; "cryptarithm requires training the model to make guesses". Operator default = **concatenação** se não estiver nos exemplos. Nemotron é ruim em split/concat.
- CoT-format (consenso gpt-oss-120b/glm/nemotron-super): **tabela coluna-a-coluna unidades→...**, `Σ(símbolos)+carry_in = resultado + 10·carry_out`, enumerar combos viáveis, **backtrack** se nenhum. Verificação final recomputando o boxed. Fallback determinístico p/ 2º melhor combo.

**🔴 SOUP/MERGE ingredientes + técnica:**
- **keithtyser/nemotron-086-adapters-20260605** (Kaggle) — **6 adapters rank-32 classe-0.86** → âncoras prontas p/ soup.
- **lopure/amplify-top-50-singular-value-retain-84-4-energy** (101 votos) — amplificar top-50% singular values (84.4% energia) no merge.
- **debatreyabiswas/...best0-86...under-5min** (101 votos) — 0.86 rápido.

**🟡 MÉDIO:** marksusol/etencore/PhuQuy23TNT1 (LoRA reasoning nemotron-30b, HF) · Ashima/task738_constraint_satisfaction · kienngx cot-labels (672v) · "Don't Overthink it" 2505.17813 · Adapter-Merging-Reactivates-Reasoning 2601.18350.

**🧭 REFRAME do teto:** 0.87 é alcançável SEM cryptarithm (outras famílias quase saturadas); **0.89 exige o salto de cryptarithm.** Nosso piso 0.86 já está lá.

**🎁 BÔNUS (teachers grátis):** OpenRouter tem `nvidia/nemotron-3-super-120b` e `ultra-550b` **FREE** → podem GERAR CoT de cryptarithm (mesma família, maiores) p/ distilar no nosso 30B. Custo $0.

### 14.8 [SUPERSEDED → ver 14.9 + 15.A + 16] receita carry-CoT/carbonteq descartada (cryptarithm é CSP, não carry-addition).

### 14.9 ⛔ CORREÇÃO CRÍTICA (agente cryptarithm + fontes primárias) — SUPERSEDE parte do 14.7
- **cryptarithm_deduce NÃO é SEND+MORE (carry-addition).** É **CSP símbolo→dígito** ("Alice's Wonderland"): `AB <opsym> CD = result`, 10 símbolos→dígitos bijetivo, **inferir o operador** (%=×, !=+, $=×...) + resolver **AllDifferent** consistente com ~4-5 exemplos, decodificar a query. → A CoT carry-a-carry (que pus no 14.7 do OpenRouter) **JÁ FOI TENTADA na competição e NÃO ajudou** (discussion/703240). ⛔ DESCARTAR carry-CoT.
- ✅ **JÁ TEMOS os solvers no repo** (não precisa carbonteq): `scripts/run_v329_symbolic_cryptarithm_gate.py` (CP-SAT verifier + gate de UNICIDADE) · `scripts/build_v330_symbolic_cryptarithm_distill_dataset.py` (gerador CoT **verificado antes de escrever**, 1-boxed) · `scripts/build_v594_queryop_cryptarithm_preference_dataset.py` (preference pairs por assinatura `!`→add `$`→mul `%`→mul).
- 🔴 **CONFIRMAÇÃO da queda do ep1** (discussion/703240 "8%→71% cryptarithm mas score travado 0.86"): outro time mediu que adicionar cryptarithm **DERRUBOU bit 93.9→81.6 e cryptarithm 20→0** = **rank-32 instável + catastrophic forgetting + seed noise**. É EXATAMENTE o nosso ep1. Não foi azar — é estrutural do rank-32.
- 📉 **TETO HONESTO:** o solver algorítmico do huikang só faz **deduce 14.9% / guess 8.5%** → ~15% é a fronteira. **deduce+guess = só ~3.4% do eval** → ganho relativo grande = **movimento ABSOLUTO pequeno** no LB. `cryptarithm_guess` é **info-teoricamente insolúvel** (operador nunca aparece) → tratar como Bayesian/preference, teto ~15-20%. **O vencedor NÃO usou cryptarithm sintético** (pontos vieram de cipher/bit/eq-numeric maxados).
- ✅ **RECEITA CORRIGIDA:** **algorithmic CoT** (do build_v330, NÃO LLM-written — LLM-CoT acerta resposta/erra raciocínio→ensina chutar) com CSP+backtrack+verify-em-todos+1 boxed · **slice PEQUENO** (~3-4% = peso da família) · **min-logprob filter** (libera capacidade do rank-32) · **REPLAY estratificado SEMPRE** · **plain SFT** (não SCALe, não over-weight boxed) · **LR conservador ~1e-6** + soup/EMA mesma init · **GATE pré-score:** cryptarithm↑ E bit/cipher NÃO↓.
- 🎯 **VEREDITO ESTRATÉGICO:** cryptarithm é caro (rank-32 forgetting) e rende pouco em absoluto (~3.4% peso). O EV mais alto e seguro **não** é cryptarithm — é **proteger o piso 0.86 + variância (re-roll/soup dos públicos keithtyser)**. Cryptarithm só com slice mínimo, replay e gate rígido — alto risco de regressão.

## 15. 🌐 SWEEP WEB DEVASTADOR (09-jun, 4 agentes web + OpenRouter free, $0, SEM Kaggle)

### 15.A 🔴 ANTI-FORGETTING (o conserto do ep1 — achado mais importante)
- **REFRAME (arXiv 2603.02224):** colapso cipher 100→0 do ep1 **NÃO é culpa do rank-32** — é interferência/over-tuning (`ℱ=α(1−cos²θ)+β`). Paper: "não reduza rank p/ evitar forgetting". **Causa real do ep1: LR alto + épocas demais + ZERO replay.** (corrige a tese rank-32 do 703240).
- 🔴 **Merge-based** (mais seguro, zero-forgetting por construção): treina cryptarithm em adapter SEPARADO → `add_weighted_adapter(dare_ties)` → cipher/bit ficam intactos.
- 🔴 **Narrow component tuning** (arXiv 2510.08564): LoRA só em MLP Gate&Up / in-proj, **CONGELA Down/out-proj** → +aprende / forget −0.6 a −2.1 (vs −23 do full). Aplicar: LoRA nos up/gate dos experts MoE + in-proj do Mamba, congela down/out.
- 🔴 **Replay 5-10%** de cipher+bit no mix (até 1% já ajuda) · **LR 10× menor** · **early-stop nas skills ANTIGAS** (probe held-out durante treino).
- 🟡 O-LoRA (github cmnfriend/O-LoRA, λ₁=0.5, só se ortogonalidade natural baixa) · MoFO (optim, top-5-10% momentum) · N-LoRA/LoRA-Null (null-space) · Mamba-CL (2411.15469, SSM-nativo).

### 15.B 🔴 MERGING/SOUP + VARIÂNCIA (agente #3)
- 🔴 **Método de soup CORRETO:** `add_weighted_adapter(combination_type="ties_svd", svd_rank=32, density=0.5, majority_sign_method="frequency")` — TIES resolve sinais + `_svd` **fixa rank-32**. (nosso `modal_soup.py` usava `linear` → trocar p/ ties_svd). 086/ep1/planB têm mesmos target_modules → mergeáveis.
- 🔴 **`VLLM_BATCH_INVARIANT=1`** (Thinking Machines): variância MoE = falta de batch-invariance nos kernels (NÃO o router). H100 ✓, ~30-60% + lento. **Para NÓS:** pré-score determinístico → mata o swing 086=0.672↔0.756 → gating confiável. ⚠️ **NÃO controla o eval do host.**
- ❌ self-consistency/best-of-N: **não aplicável** (host roda greedy single-pass no nosso adapter).
- ⚠️ Mamba caveat (2410.09016): LoRA padrão fraco no core SSM (HyLoRA treina conv1d) — nossos adapters miram projeções, mergeam mas herdam o ponto cego.

### 15.C 🔴 DADOS + SOLVERS cryptarithm (agentes #1 e #4)
- 🔴 **alex-gapch/nemotron-cryptarithm-narrative** (HF, 15k) — **feito p/ ESTA comp**, trace exato (parse→carry coluna-a-coluna→NOW KNOWN→[VERIFY] todos→[APPLY]→`\boxed`), add+abs_diff. ⚠️ **delimitadores `<think>`/`\boxed` malformados — LIMPAR p/ 1 canônico antes do SFT.** Asset #1.
- 🔴 **Repos GitHub p/ DIFFAR contra nosso v329/v330:** quan16369 (ops subtração-com-sinal `sub_*_neg_prefix/suffix`) · nabin2004 (catálogos `_OP_RULES`+`_NUMERIC_OPS` + reward GRPO) · Michal1337 (`cryptarithm_alice.py`) · sumit-ai-ml (eq_symbols em camadas + **alerta da armadilha `\boxed` last-numeric fallback**).
- 🔴 **Norvig pytudes Cryptarithmetic** (solver ~15 linhas, ops arbitrárias) — backbone p/ instrumentar e emitir trace no nosso formato.
- 🟡 carbonteq/rg-cryptarithm-100k (add-only, EN, SEM trace → só RL env) · jeff4000/cryptarithm_rollout (STaR-filtrável) · arnaud-m/cryptator (gerador CP unique-solution) · reasoning-gym generator+verifier.

### 15.D 🔴 METODOLOGIA cryptarithm (papers)
- **LOGIPT / solver-trace distillation** (2508.13678, 2512.17093) + **verifier-filtered SFT = +3%** + **STaR** (só traces com `\boxed` correto) = **exatamente nosso build_v330**, validado.
- **PuzzleBench/Puzzle-LM** (2402.02611): GPT-4-Turbo few-shot cryptarithm **30.5%**; com solver Z3 **93.55%** → o teto vem de offload p/ solver; como não podemos chamar solver no inference, **distilar o PROCEDIMENTO do solver no trace**.
- **Detailed-scratchpad + carry explícito unidades→dezenas** (2307.03381) corta sample-complexity. Free-CoT puro colapsa ~8% (2407.11373) → trace TEM que ser procedimento rígido checável.

### 15.E CONFIRMAÇÕES + REFRAME DO VEREDITO
- Múltiplos times convergiram no MESMO solver (símbolo→dígito backtracking) → nosso v329/v330 certo. Plateau geral ~0.85-0.86; **tail ~399 puzzles que nem GPT-5.4 resolve**. Nenhum paper/writeup não-inglês sobre o benchmark.
- 🔄 **VEREDITO REFINADO (15.A muda 14.9):** o ep1 **não provou que cryptarithm é impossível** — provou que treinamos ERRADO (SCALe full, sem replay, LR alto). Com **merge-based OU narrow-tuning+replay+LR-baixo+early-stop**, dá p/ tentar cryptarithm com **risco controlado**. MAS o ganho absoluto segue pequeno (~3.4% peso, teto ~15-25%). **Ordem de EV:** (1) variância via **ties_svd soup** dos públicos + **VLLM_BATCH_INVARIANT** no pré-score (barato, seguro); (2) cryptarithm via receita-15.A com gate rígido (médio risco/médio retorno). **Tudo só com GO + custo na frente.**

## 16. 🌏 SWEEP ASIÁTICO + NEUTRO (09-jun, agentes China/JP/KR/Índia/Vietnã + OpenRouter prompt-neutro, $0)
### 16.A HONESTIDADE: platô NÃO-quebrado em fonte pública nenhuma
China/JP/KR/Índia/Vietnã/Taiwan/Indonésia: **ZERO writeup específico desta competição**; nada quebra 0.86. Topo público real = **0.85 (winner SFT)**; resto = variância vLLM ±0.02-0.03. **Sem bala de prata** (confirmado por 8 agentes ao todo). AIMO-2 winner (famoso na China) = TIR+GenSelect+majority → **ilegal aqui** (greedy/single/no-tools).
### 16.B 🔴 CipherBank (Shanghai AI Lab, arXiv 2504.19093) — espelha nosso cipher
Eval idêntico (3-shot known-plaintext). **Reasoning models OVER-THINKAM ciphers simples e PERDEM** (o1=45 vs DeepSeek-R1=25). → famílias FÁCEIS (cipher/bit/conv) querem traces **CURTOS, mecânicos, determinísticos**, NÃO CoT longo. Transposição/posicional = ponto fraco (substituição já ok) → gastar dado sintético em transposição.
### 16.C 🔴 Truncamento come o \boxed (mecânica de inferência, fonte CN)
max_tokens é compartilhado thinking+answer → trace longo → `finish_reason=length` → trunca ANTES do `\boxed` → zero automático. Conserto: traces concisos + emitir `\boxed` cedo. (Reforça 16.B do lado infra.)
### 16.D 🔴 GOLD do código do vencedor (tonghuikang/nemotron) — diffar contra o nosso
- **train_unembed=True** — vencedor TREINA a LM head (não só attn/mlp) → fidelidade do boxed (rank ainda ≤32). **Verificar se fazemos.**
- **Stratified per-category batching** (todo batch balanceado entre famílias) → anti-forgetting barato.
- **Reference-logprob anchoring** (logprobs da época-0 = âncora KL anti-forgetting).
- **equation_numeric catalog**: determinant (d1·d4−d2·d3), cross-multiply, digit-product-sum, operando/resultado-reverso + prefixo/sufixo → **diffar contra v329/v330** (+quan16369/nabin2004).
- cipher ancora em **wonderland.txt**; bit = **per-bit family+stride**. (winner `reasoners/cipher.py` VAZIO → cipher sub-investido = headroom em transposição.)
- Config: rank32, LR **2e-4 step-linear**, **1 época**, max_len 8192, batch64, Adam β(0.9,0.95), no grad clip.
### 16.E 🔴 LEVER GENUINAMENTE NOVO: GRPO > SFT sob rank-32
- **TinyLoRA (2602.04118):** GRPO recupera ~90% do full-FT com **1000× menos params**; **SFT falha quando updates são minúsculos**. Rank-32 HARD cap → **RL extrai mais por-parâmetro** → candidato real a quebra-platô. Reward = `\boxed` exact-match (verificável). Caveat: precisa vLLM rank≥4 merge-each-step.
- **NÃO usar PiSSA p/ RL** (2511.08567: RLVR aprende OFF das direções principais). LoRA/rsLoRA init correto. 🟡 DoRA low-rank (+1-4%, r≤16)+rsLoRA · LoRA-Soups CAT (bake p/ 1 adapter) · adapter-merge reativa reasoning (2601.18350).
### 16.F SÍNTESE NOVA — estratégia POR-FAMÍLIA + EV
- **Famílias FÁCEIS (cipher/bit/conv):** traces CURTOS/mecânicos (overthinking + truncamento = inimigos). **Família DURA (cryptarithm):** dedução estruturada completa (CSP, teto ~15%).
- **Quick-wins de fidelidade (baratos):** `train_unembed=True` + stratified batching + ref-logprob anchor + diffar catálogo equation.
- **Ordem de EV final:** (1) variância barata/segura (**ties_svd soup** + **VLLM_BATCH_INVARIANT** no pré-score); (2) **quick-wins de fidelidade**; (3) **GRPO sob rank-cap** (experimental, novo); (4) cryptarithm-slice (médio risco). **Platô real; tudo só com GO + custo na frente.**

## 17. 🔬 AUDITORIA INTERNA DA NOSSA SOLUÇÃO (09-jun, 2 agentes code-audit + OpenRouter neutro) — MAIOR VALOR
### 17.A 🔴 BUGS/GAPS ACIONÁVEIS (file:line + micro-fix, todos $0/CPU)
- **[SILENT-BUG/ALTO] `max_length=1024` (landmine latente):** `modal_fase2_train.py:195` + `modal_planB.py:76`. HOJE seguro (corpus kept máx ~325 tok), mas se equation/CoT maior reentrar → trunca `</think>\boxed{}` **silenciosamente** → modelo aprende a NUNCA emitir boxed. Sem reserva de budget. **Fix:** `max_length≥4096` + assert `len(row)≤budget` antes do train.
- **[GAP/ALTO] SEM stratified batching + SEM replay = o vetor do forgetting:** FASE-2 usa SFTTrainer stock (shuffle default, sem sampler/pesos). Mix desbalanceado (~42% bit / 34% cipher / 9.5 grav / 9.5 unit / 5 crypt / 0 numeral). **REGRESSÃO do NOSSO PRÓPRIO v217** (que tinha `weighted_replacement` + `source_weights` + `v215_replay_anchor` — FASE-2 largou isso). 1 época a LR 1e-4 sobre mix bit/cipher-heavy = cipher sobrescrito. **Fix:** restaurar WeightedRandomSampler/replay-anchor; cap por-família ≤25-30%.
- **[GAP/MÉDIO] concisão (SCALe ans_weight 2.0 + max_length 1024 + sem replay)** = receita exata do colapso cipher. **Fix:** SCALe + replay + budget maior + **LR menor p/ continuação warmstart** (1e-4 full-epoch sobre 086 é agressivo).
- **[GAP/ALTO solvers] Cipher SÓ substituição, ZERO transposição:** `cipher_solver_v2.py` + `all_families_solver.py:299`. val é só-substituição → 100% FALSO; se teste tiver transposição (tipo fraco CipherBank) = miss silencioso. **Fix:** probe `dec==enc[::-1]`/swap-pares/shift antes de commitar.
- **[GAP/ALTO solvers] catálogo equation falta `determinant`(d1·d4−d2·d3) + `cross-multiply`(d1·d3+d2·d4):** real em `kg1_v274_numeric_postprocessor.py:98-128` (NÃO v329/v330). v329 descarta negativos (`:269`). **Fix:** +2 lambdas em `numeric_rule_functions()`.
- **[BUG-adj solvers] `\boxed` last-number fallback:** extractor oficial sem box pega o ÚLTIMO número; traces bit/cipher cospem números antes (`Bit 0: solver=1`, `Map size: 5`) → se box trunca, retorna número errado. **Fix:** tirar números do scaffolding OU NOT_FOUND sem box fechado.
- **[BUG-adj solvers] bit Level-3 empate → default 0 (cara-coroa):** `all_families_solver.py:428`. **Fix:** abster no empate em vez de chutar 0.
### 17.B ✅ VERIFICADO LIMPO (não mexer)
lm_head treinado (LoRA rank-32 nele; warmstart-086 config) · masking correto (markers batem, v217 confirma 0 mismatch) · **1 `\boxed` canônico/trace** (5704 rows, 0 mismatch, sem `}}`) · **grad-accum num_items CORRETO** · **prompt suffix byte-exato** (= oficial) · **val_clean 0 leak** (0/180) · ZIP packaging robusto · uniqueness gate forte (build_v330) · `_guess` abstém · **bit solver FUNCIONAL** (per-bit family+stride, memória "DEAD" era formulação errada).
### 17.C 🩹 MICRO-FIXES PRIORIZADOS ($0/CPU, sem treino p/ implementar)
1. **Restaurar replay/stratified sampling** (cap família ≤30% + replay-anchor) → conserta vetor #1 do ep1.
2. **`max_length≥4096` + assert de budget** → desarma a landmine.
3. **Probe de transposição no cipher** → fecha gap oculto (pontos reais).
4. **+determinant +cross-multiply** no catálogo equation → cobertura do muro.
5. **Limpar números do scaffolding bit/cipher / NOT_FOUND sem box** → anti-fallback-trap.
6. **bit Level-3 empate → abster** · LR menor warmstart · (opc. `modules_to_save=["lm_head"]` p/ paridade com winner).
### 17.D 🎯 VEREDITO
Auditoria interna = **maior valor de todas as buscas**: bugs/gaps NOSSOS, concretos, fix de poucas linhas, $0/CPU. **Causa do ep1 100% explicada** = regressão do replay/stratified (que NÓS já tínhamos no v217) + max_length 1024 + concisão. Fixes #1-2 consertam o ep1; #3-4 abrem cobertura nova; #5-6 fecham fugas de fidelidade. Implementar todos é CPU/offline; re-treinar/medir só com GO.

### 17.E 📦 RECEITA v217 VALIDADA — extraída do NOSSO histórico (`artifacts/v217_doublecheck_tokenize_dry_run.log`) — RESTAURAR p/ consertar o ep1
- **Treino:** `max_length=4096` (NÃO 1024), batch 4, grad_accum 4, micro 1, `max_prompt_truncation_rate=0.0`, `max_trainable_param_ratio=0.08`.
- **LoRA:** r32 / alpha32 / dropout 0.0; targets `k,up,down,out,v,q,lm_head,o,in_proj` + experts `gate_up_proj,down_proj`; trainable filter `in,k,o,out,q,v`.
- **`sampling.mode="weighted_replacement"`** com **`v215_replay_anchor` (share ~8.3%)** + `source_weights`{base_bit 1.05, base_eq 1.0, synth_bit 1.05, synth_numeric 0.9, synth_symbolic 1.0} + `subcategory_weights`{bit 1.05, eq_numeric 0.9, eq_symbolic_bin/unary 1.05}.
- **shares medidos:** base_eq 30.7% · synth_symbolic 31% · base_bit 18% · **replay_anchor 8.3%** · synth_bit 6.9% · synth_numeric 5%.
- ➡️ **A FASE-2 jogou TUDO isso fora** (SFTTrainer stock + max_length 1024) → por isso o colapso. **Restaurar esta config = conserto direto e validado do ep1, $0 p/ montar.**

### 16.G HF free inference (09-jun): Llama-3.3-70B + Qwen-2.5-72B, prompt neutro = playbook GENÉRICO (data-aug/HP-tuning/regularização/curriculum/ensemble), ZERO novo. Fecha o ciclo: Kaggle + web(todos continentes) + arXiv + papers universidade + nossos dados + OpenRouter-free + HF-free → TODOS convergem; nenhuma fonte pública quebra 0.86.

## 18. 📊 DOUBLE-CHECK DO RANKING + EFEITO MANADA (09-jun, LB ao vivo, 4099 times)
### 18.A Distribuição de scores (a manada REAL é o 0.86)
- 0.89: **1** (NullSira) · 0.88: **3** (Domdolus, vli, Kh0a) · 0.87: **22** · **0.86: 1.449 🐑** · 0.85: 512 · 0.84: 249.
- ~2.210 times em 0.84-0.86. Só **26 times (0,6%)** acima de 0.86. O "salto do 0.87" existe mas é pequeno perto do paredão 0.86.
### 18.B Nossa posição: **Rank 39 / 4.099 (top 1%), 0.86**, último submit 06-05 (65 subs)
- Estamos na **FRENTE da manada 0.86** (1.449 nesse score) graças ao **desempate por timestamp** (batemos 0.86 cedo). Posição forte.
### 18.C O que explica o efeito manada (3 camadas)
1. **0.86 = receitas públicas** (huikang 0.85, kuangyicheng 0.87-nb, konbu17, kienngx, keithtyser, dgxchen) → todos convergem.
2. **0.87 (22) = delta-scale + variância** (nosso planB rodou essa receita → deu ≈086 = é variância, não método).
3. **0.88-0.89 (4) = LOTERIA DE VARIÂNCIA por contagem de submits:** NullSira 49 · Domdolus 93 · Kh0a 168 · JK-Piece 215 subs. Com ±2-3% MoE, re-roll de 100-215 submits + seleção do melhor = +1-2% **no LB público** (provável overfit do público → risco de embaralhar no privado/final).
### 18.D Timing
Quase todos os top têm último submit 08-09/jun → empurraram **nos últimos dias** (re-roll de endgame perto do deadline) = o "efeito manada de um dia pro outro" observado.
### 18.E Conclusão
O topo (0.87-0.89) NÃO é método secreto — é **receita pública (0.86) + farm de variância (re-roll de endgame)**. 2 caminhos pra subir: (1) jogar a variância (re-roll 086 / ties_svd soup — barato, loteria); (2) edge real (fixes v217/transposição/equation — sobe por acerto, não sorteio). Piso 0.86 = top 1% já garantido.

### 18.F TRIPLE-CHECK estatístico + 3 modelos free (gpt-oss-120b, nemotron-120b, Qwen-72B-HF, prompt neutro)
- **Submits×score:** ≥0.87 mediana 90 subs · 0.86 mediana 11 · 0.84-0.85 mediana 5 · <0.84 mediana 2. **Spearman ρ=0.556 (p≈0).** 96% dos ≥0.87 submeteram nos últimos 3 dias.
- **Matemática do valor-extremo (consenso dos 3):** E[melhor de N] ≈ μ + σ·√(2·ln N), σ≈0.02-0.025 → 10 subs +0.03, 30 +0.05, 100 +0.07. Explica 0.86→0.87 (poucos re-rolls) e 0.86→0.89 (100+).
- **Veredito:** topo = **best-of-N variance farming, NÃO método**. LB PRIVADO vai embaralhar: farmadores caem 0.02-0.04 (0.87 público → ~0.84-0.85 privado); **robustos com poucos submits SOBEM relativamente**.
- **🟢 NÓS = perfil robusto** (rank 39, 0.86, 65 subs, parados 06-05, sem farm) → tendemos a SEGURAR/SUBIR no privado vs os 0.87-0.89 farmados. **Nosso 0.86 honesto pode valer mais no final.**
- **Decisão de submits restantes:** re-roll = ganho público +0.02-0.03 MAS ~30-40% de chance do draw final ser baixo no privado (gamble). Manter robusto = estável. Trade-off documentado.

## 19. ⛔ DESCARTADO / JÁ ANALISADO — NÃO REPETIR (registro anti-loop, 09-jun)
> Tudo abaixo já foi testado/medido/refutado. NÃO re-analisar sem motivo novo concreto.

### 19.A Receitas/treino DESCARTADAS (com evidência)
- **SCALe full (ep1)** → −0.26, cipher 100→0. Over-concisão + sem replay + max_length 1024. MORTO.
- **carry-by-carry CoT p/ cryptarithm** → já tentado na comp, NÃO ajuda (é CSP símbolo→dígito, não SEND+MORE). disc/703240.
- **equation rich-catalog literal nos dígitos** → testado = 0% (nossa família é símbolo). 
- **forçar cryptarithm sem replay/gate** → DERRUBA o LB (703240: bit 94→82, crypt 20→0).
- **DoRA / alpha64 sobre warmstart-086** → distorce o piso 0.86.
- **max_length=1024** → landmine (trunca boxed); usar ≥4096.
- **SFTTrainer stock sem replay/stratified** → causa do forgetting (regressão do nosso v217).

### 19.B Técnicas INAPLICÁVEIS (regras do desafio — greedy + 1 boxed + só adapter)
- self-consistency / best-of-N / majority-vote / entropy-branch / multi-rollout → eval é **greedy single-pass**. ILEGAL.
- activation steering / steering vectors → submetemos **só adapter LoRA**, sem código de inferência.
- TIR / code execution / tool use (AIMO-2) → **sem tools** no eval.
- PiSSA init p/ RL → aprende off-principal (~aleatório). gradient-based rank allocation → pior que uniforme. treinar router MoE → não.

### 19.C Fontes externas ESGOTADAS (10+ agentes, convergiram, ZERO plateau-breaker)
Kaggle (disc/nb/data/models) · web global (Américas/UE/Rússia/China/Japão/Coreia/Índia/Vietnã/Taiwan/Indonésia) · arXiv + papers universidade · OpenRouter free · HF free. **Topo público real = 0.85 (winner SFT); 0.86 = commodity de receita pública. NÃO existe atalho público.** konbu17/kuangyicheng/huikang/kienngx/dgxchen = KNOWN. carbonteq/rg-100k = add-only EN sem trace (só RL env).

### 19.D Hipóteses REFUTADAS
- "Top1/2 = cartel de Tóquio mesma universidade trocando dados" → **FALSO** (rank2 = indonésios; rank1 anônimo; sem evidência).
- "Manada 0.87 = vazamento/recipe secreta" → **FALSO** (= receita pública + farm de variância; ρ=0.556 + valor-extremo E[max]≈μ+σ√(2lnN)).
- "rank-32 é a causa do forgetting" → **FALSO** (2603.02224: interferência/over-tuning; nós regredimos do v217).
- "bit majority/choice DEAD = impossível" → **FALSO** (per-bit family+stride funciona, ~working).
- "cipher 100% = sem gap" → **PARCIAL** (gap oculto: transposição não coberta).

### 19.E PROIBIDO (ético/legal)
- dark web / deep web / dados vazados → ilegal, desclassifica. NUNCA.
- modelos pagos sem GO de custo (bloqueados 402 de qualquer forma).

### 19.F ✅ NÃO descartado — AÇÕES VIVAS (única coisa que sobrou)
1. Restaurar config **v217** (replay 8.3% + max_length 4096) → conserta ep1.
2. Probe **transposição** no cipher. 3. **+determinant/+cross-multiply** equation.
4. Limpar scaffolding bit/cipher (anti boxed-trap). 5. **ties_svd soup** + **VLLM_BATCH_INVARIANT** (pré-score) p/ variância.
6. **GRPO sob rank-cap** (experimental, único lever novo). 7. **Manter 0.86 robusto** (forte no privado vs farmadores).

### 19.G RED-TEAM do plano (modelos free gpt-oss-120b/nemotron-120b/Qwen-72B, ângulo crítico) — refinos antes de confiar
- 🔴 **Replay 8.3% (v217) pode ser baixo** (92% dado novo). ANTES de re-treinar: **teste "replay-only"** (re-treina só famílias antigas → confirma que o colapso é prevenido). Se cair, aumentar replay / baixar share de dado novo.
- 🔴 **max_length 4096+assert: NÃO dropar silenciosamente** linhas longas (= casos difíceis) — LOGAR/tratar overflow. **2048 pode bastar** (menos VRAM/variância).
- 🟡 **determinant/cross-mult + merge ties_svd = baixo-ROI** (3.4% peso) + risco de saturar rank-32/ops espúrias → ablação isolada antes.
- 🟡 **Stripper de números:** testar em 100 exemplos, nenhum box correto pode sumir (carries são scaffolding legítimo).
- 🟡 **GRPO:** reward esparso + alta variância + reward-hacking → trial curto synthetic antes de comprometer.
- **PRINCÍPIO:** cada fix tem um **teste isolado barato (CPU/$0) ANTES** de gastar GPU. Não confiar cego.

## 20. 🔧 FIX #1 IMPLEMENTADO + QA GO (09-jun) — corrige o ep1
- **`modal_fase3_train.py`** (corrige FASE-2): balance por familia (eff_cap clampa, NUNCA esvazia) cap<=30% · max_len 2048 + DROP de over-budget (nao truncar boxed) · plain SFT (sem over-weight SCALe) · LR 5e-5 · unknown=fatal · _save_push re-raise + assert.
- **`desk_test_fase3.py`**: 17/17 PASS. Cobre os bugs da equipe QA (balance drain, cap inviavel, familia zerada, gold ausente, invariante no corpus full, classify negativos).
- **Equipe QA (silent-failure-hunter + test-analyzer):** acharam balance-drain (cap<4fam=0 linhas) + budget-log-and-truncate + _save_push-swallow → TODOS corrigidos + regressao testada. SAFE confirmado: markers do template casam (masking ok), sem double-think.
- **Distribuicao corrigida:** bit 42->30% · cipher 34->30% · **numeral incluido (era 0%!)** · 30/30/13/13/13 = 3000 traces, zero truncamento.
- **Status:** CODIGO com GO do QA. **Treino GPU = gasto, exige GO do usuario + custo (separado do GO do QA).**

## 21. 🧠 CONSULTA AO ARSENAL GRATUITO (prompt neutro) + DIAGNÓSTICO MICRO/MACRO (09-jun)
### 21.A Consulta (neutra, sem indução): gpt-oss-120b + nemotron-super-120b + gemini-2.5-flash + cerebras-glm-4.7 (qwen 429)
**Convergência dos 4:** o gargalo é dado/CoT das 2 famílias fracas (bit ~40% + cryptarithm ~13%); sugeriram: (a) mais dado diverso p/ bit+cryptarithm; (b) **CoT estruturado tipo "scratchpad"/state-table** (`Current_Map:{A:1,B:?}` + truth-tables) — reduz "raciocínio" a "lookup" p/ adapter low-rank; (c) **filtrar traces por SOUNDNESS lógica** (descartar resposta-certa-com-raciocínio-errado, que o greedy expõe); (d) curriculum/oversample das fracas.
### 21.B Análise (sinal vs ruído) — VALIDADO
- ✅ **Já fazemos (confirma a abordagem):** (b) e (c) = nosso **solver-CoT** é **sound por construção** (v330 transcreve o solver CP-SAT + gate de unicidade) e **estruturado** (NOW KNOWN/VERIFY/APPLY). Os modelos neutros chegaram à NOSSA receita sozinhos.
- ❌ **Ruído/inaplicável:** "subir rank" (já estamos no cap 32) · "aux head/operator classifier" (submissão é só-adapter, sem código de inferência) · "treinar router MoE" (proibido/Unsloth) · "oversample 10x / parar famílias 1-4" (= colapso por forgetting, o erro do ep1).
- 🟡 **Único candidato extra plausível (NÃO testado):** diversificar corpus de **bit** (bit-width/ops variados) + **gate de verificação de passos** (step-soundness) além do gate de resposta. Validação = construir verificador por-família (CPU/$0); teste real = treino (GPU+GO).
### 21.C VEREDITO
O arsenal neutro **NÃO achou lever público novo** — convergiu na nossa própria abordagem (solver-CoT sound+estruturado) e o resto é já-feito/inaplicável. Reforça: o caminho é **nossos fixes (§17/§20) + dado de bit**, não um atalho externo.

### 21.D Rodada HF (modelos gratuitos via router PRO) — re-execução incluindo HF
- Responderam: **DeepSeek-V3.2 + Qwen3-235B-A22B-Thinking** (gpt-oss-120b/GLM-4.7/Llama-3.3 deram **403 Cloudflare** no router — alguns providers exigem User-Agent; nota operacional p/ o arsenal).
- 🎯 **CROSS-VALIDAÇÃO do bug #2:** Qwen-235B-Thinking, neutro, apontou **subtração/resultados-negativos** como gap do cryptarithm → bate com `run_v329:269` que DESCARTA negativos. Defeito real confirmado por modelo independente.
- 🟡 Corrobora diversificar **bit** com padrões raros (0xFF/0x00).
- ❌ Ruído/inaplicável: subir rank (no cap), "operator priming" (operador é oculto no teste), head auxiliar.
- DeepSeek-V3.2 (honesto): "se nada melhorar o held-out, o platô não quebra dentro das restrições."
- **VEREDITO (com HF):** arsenal completo (OpenRouter+Gemini+Cerebras+HF) **não achou lever público novo**; convergiu na nossa abordagem e **cross-validou o bug #2 (negativos)**. Caminho = consertar as peças #1-4 + dado de bit.

### 21.E Consulta CIRÚRGICA (arsenal completo) + VALIDAÇÃO por teste (09-jun)
- 5/6 responderam (Qwen-235B 504). Convergiram em CoT algorítmico c/ **passos de REJEIÇÃO/eliminação** (já é o v330).
- 🎯 **ACHADO NOVO TESTADO E REFUTADO:** Cerebras propôs "tokenizer merge bits 0101→1 token, space-separar p/ 1 token/bit". **TESTE no tokenizer real: FALSO** — `01010101`→8 tokens (já 1/bit); `0 1 0 1...`→15 (PIOR). **bit 40% NÃO é tokenização. NÃO space-separar bits.**
- 🟡 **Nuance REAL (testada):** símbolos do cryptarithm PODEM mergear (`#[`→1 token). Micro-fix plausível: space-separar SÍMBOLOS na CoT do cryptarithm (não os bits). Validável; teste real = treino.
- **VEREDITO:** o "100x" surfou 1 ideia testável → refutada por teste; nenhum lever novo válido. Caminho segue: peças #1-4 + dado de bit (diverso, não re-tokenizado).

---

## §22 — RED-TEAM 10000x DO NOSSO PLANO + VALIDAÇÃO EMPÍRICA NO CORPUS REAL (2026-06-09)

**Mudança de método (o que destravou):** em vez de perguntar aos modelos gratuitos "como melhorar X" (rodadas anteriores convergiram em nada novo), desta vez o prompt **incluiu o NOSSO PLANO** e pediu **CRÍTICA/red-team** (neutro, sem indução). 6/6 modelos responderam (gpt-oss-120b, nemotron-super-120b, Gemini-2.5-flash, Cerebras glm-4.7, HF DeepSeek-V3.2, HF Qwen3-235B-Thinking). Script: `C:\tmp\consult_redteam.py`. Respostas: `%TEMP%\redteam\*.txt`.

### Sinal convergente dos 6 modelos (signal, não ruído)
1. **O CoT (formato/qualidade) das famílias fracas é o lever dominante — não os fixes de solver-op.** 6/6 convergiram: bit/equation precisam de **dedução verbosa + auto-verificação contra os exemplos**, não trace-código/afirmação. (gpt-oss "RULE: line + self-check"; Cerebras "verbose NL, not code log"; Qwen "self-verification: verify vs Example 1, restart if mismatch"; DeepSeek "if CoT is a search trace, learns no generalizable reasoning".)
2. **Fix cipher transposition = desperdício** (5/6): cipher já ~100%, sem transposição no held-out.
3. **Fix determinant/cross-multiply = possivelmente mal-direcionado** (Cerebras+Qwen): op-set de equation é simples.
4. **abstain do bit deve ser em DATA-TIME (dropar/dedup linha ambígua), NUNCA runtime** (gpt-oss, catch crítico): "output nothing" no box = contado ERRADO.
5. Divergente/ruído: magnitude de LR (uns dizem ↑, outros ↓), MoA/2-LoRAs, MoE routing loss, attention-only targeting → DESCARTADO.

### VALIDAÇÃO EMPÍRICA (medido no corpus REAL `fase1_corpus.jsonl`, não suposto) — o NÚCLEO
Baixei o corpus de treino real (5704 linhas, `messages`+`answer`) e medi:

- **equation (n=1500): 79% (1180) têm marcadores de confusão/fabricação/VAZAMENTO de resposta.** Amostra do CoT mais longo (literal): *"the user explicitly says 'Test Input 65!76 = 11', we must output 11"*, *"Probably the puzzle ... typed 47 incorrectly. We'll ignore and assert the rule matches"*, *"we could fabricate an intermediate step"*. → O gerador **vaza a resposta** no prompt do teacher e os **puzzles-fonte são inconsistentes** (`29!81=47` não fecha com +,−,×,b−a) → o trace é racionalização reversa para uma resposta já dada. **Held-out não tem resposta vazada → o modelo não tem procedimento de dedução aprendido → 13%.** ESTE é o núcleo (peça exata: campo `completion` das linhas equation, gerador build_v304/v330-style com answer-leak).
- **bit (n=1428): 0% (0/1428) verificam a regra contra os exemplos.** Todos afirmam "the deduced rule is SHR(2)" e aplicam (regra e aplicação corretas, mas sem derivação/verificação) → modelo erra a INFERÊNCIA da regra sem rede de segurança → 40%.
- **Truncamento:** equation median=288 / **máx=1244 palavras (~1900 tok); 127 linhas >680 palavras truncam @max_len=1024** perdendo o `\boxed{}`. bit máx=53 palavras. → a justificativa "truncamento geral" (fix #5 antigo) era FALSA; truncamento é real só na **cauda de equation**.
- (Artefato descartado: medição "missing boxed=1500" foi erro de escape de regex — as amostras claramente contêm `\boxed{11}`.)

### Convergência externa (web-investigator concorrentes + literatura)
- **Blog NVIDIA + discussões (disc/689915):** vencedores distilam CoT de teacher forte (DeepSeek-R1/QwQ); *"top solutions require significant progress on cryptarithm"* = nosso núcleo.
- **Literatura (arXiv 2509.00768 rejection sampling; 2604.02819 student-in-the-loop):** padrão = gerar trace SEM a resposta e **manter só traces cujo resultado bate com o gold** ("parse teacher answer, compare to gold, keep only correct"); manter estágios disjuntos p/ evitar leakage. = cura exata do nosso 79% de poison.

### DECISÃO (corrige o roadmap; substitui a ênfase anterior em fixes de solver-op)
- 🔴 **NOVO FIX #1 (núcleo):** regerar CoT de equation com **rejection sampling sem vazamento** (operador-hipótese testado contra todos exemplos → bijeção → verificação → decode query; manter só os corretos).
- 🟠 **NOVO FIX #2:** CoT de bit ganha passo de **verificação da regra contra TODOS os exemplos** antes de aplicar.
- 🟡 **FIX #3 (max_len 2048):** mantido, mas só justificado pela cauda de equation.
- 🟢 Fix negativos (#4): mantido, baixa magnitude.
- ⬇️ **REBAIXADOS:** cipher-transposition (~0 valor, 5/6), determinant-catalog (mal-direcionado). Verificar held-out antes; provável drop.
- 🧪 **ABLAR (não auto-adotar):** proporção fraca 30%↔40-50% (conflita c/ esquecimento prévio), LR warmup+cosine, augmentation por permutação de símbolos.
- **FINOPS:** treinar `modal_fase3` no corpus ATUAL = repetir o colapso e queimar ~$27. **Regerar o CoT (offline/$0) é pré-requisito do treino.**

---

## §23 — QUADRUPLE-CHECK CIRÚRGICO: "resolver o cryptarithm" (2026-06-09)

**Pergunta focada:** todas as formas de resolver a família equation/cryptarithm (gargalo, 13%) e sair do platô. Arsenal: 6/7 modelos gratuitos (gpt-oss-120b, nemotron-super-120b, Gemini-2.5-flash, Cerebras glm-4.7, HF DeepSeek-V3.2, HF Qwen3-235B-Thinking; deepseek-r1:free 404). Script `C:\tmp\consult_crypt.py`. Prompt embutiu exemplos REAIS verbatim + nossa medição empírica.

### NÚCLEO MEDIDO (no corpus real `fase1_corpus.jsonl`, 1400 puzzles equation c/ query)
| Métrica | Valor | O que significa |
|---|---|---|
| **LIMPO** (op-substitution único, query-op visto, decode==gold) | **17% (236)** | único núcleo aproveitável hoje |
| sem solução (operator-substitution) | 77% (1080) | quebrado OU digit-sub (H2 deu ~0 limpo) → maioria quebrado |
| query-op nunca visto nos exemplos | 4-16%* | estruturalmente impossível inferir |
| ambíguo (>1 op p/ query) | 1% | sem resposta única |
| CoT com vazamento/confusão/fabricação | 79% | "we must output 11", "we'll ignore and assert" |
| H1 operator-sub real-digit consistente | 15% · H2 digit-bijection+literal | ~4% |

(*4% medido no modelo estrito desta rodada; 16% no modelo amplo da §22 — ambos confirmam o defeito.)

### CONVERGÊNCIA 6/6 (forte): **a DATA está quebrada; o caminho é reparo de dado, não modelagem**
- **4/6 craquearam P1 identicamente:** `]`=× (99×18=1782, 43×50=2150) e `!`=(a+b)+1 (59+45+1=105; query 85+42+1=**128**=gold). → estrutura intencional = **substituição de operador (símbolo→operação, inclui offset) e/ou bijeção de dígitos**.
- **5/6 declararam P2/P3/P5 QUEBRADOS** ("corrupted mixture of literal arithmetic, cryptarithmetic and random noise" — Cerebras; "95*19=1805≠1995, off by 190, noise" — DeepSeek; Nemotron PROVOU contradição f(8)=0 em P2).
- **Prescrição unânime (= padrão de literatura, CSP clássico + rejection sampling, AIMA cap.5):**
  1. **CSP validator-solver** (já temos `run_v329_*` CP-SAT; ampliar catálogo de ops c/ offsets + ramo de bijeção de dígitos).
  2. **FILTRO DURO** nas linhas equation: manter só **solução ÚNICA + query-op visto + decode==gold** (hoje só ~17% passam → causa do 13%).
  3. **REGERAR volume** por rejection sampling: op aleatório + bijeção/glyph-set aleatório → 4-5 exemplos → solver confirma solução única → garante query-op presente → calcula resposta (**nunca vaza**).
  4. **CoT = trace dedutivo real do solver:** listar glyphs → testar cada op candidata contra TODOS exemplos → confirmar mapeamento único → aplicar à query → único `\boxed{}`. SEM "we must output X".
  5. **Currículo:** ≥4-5 exemplos, solução única primeiro.
- **gpt-oss (especulativo, quantificado):** descartar só os 16% query-op-unseen já levaria base 0.13→~0.30; pipeline completo limpo → potencial >0.9.
- **REJEITADO (ilegal):** solver em tempo de inferência ("model outputs run-solver") — viola adapter-only/no-tools (gpt-oss e Cerebras alertaram).
- **CAVEAT honesto (Cerebras/DeepSeek/Qwen):** se o TESTE OCULTO também estiver quebrado igual, teto <100%. **Pendência $0:** verificar se os puzzles oficiais/sample da competição são limpos (provável: o gerador oficial gera solúvel; só a NOSSA renderização de treino está corrompida). Confirmar antes do treino.

### DECISÃO (substitui ênfase anterior; tudo offline/$0)
🔴 **CRYPTARITHM FIX = (1) solver CSP como FILTRO + (2) regerar por rejection sampling + (3) CoT sem vazamento.** É o maior lever isolado da competição (família 13%, headroom ~+0.04-0.10). Pré-requisito absoluto do treino — treinar no corpus atual = repetir o colapso e queimar ~$27.

---

## §24 — QUINTUPLE-CHECK + IMPLEMENTAÇÃO DO FIX DE CRYPTARITHM (2026-06-09)

### Fontes inexploradas testadas computacionalmente (validação, não suposição)
- **Operador GLOBAL** (mesmo glyph = mesma op em todo o dataset): **REFUTADO** — 26 glyphs, melhor op por glyph explica só 17-43%; global 23%. (`C:\tmp\global_op.py`)
- **Modelo COMBINADO H3** (bijeção de dígitos σ + op por glyph) nos 977 puzzles que falham operator-substitution: **só 6% têm solução, 0% única** → os 77% são GENUINAMENTE quebrados, não sub-modelados. (`C:\tmp\crack_h3.py`)
- **Consulta 4/8 modelos** (gpt-oss-120b, nemotron, Gemini, DeepSeek-V3.2): **convergência total**. gpt-oss fez proof-by-exhaustion (10!×12×… ~2.7e12 hipóteses) e provou P2/P3/P5 insolúveis; Gemini achou `*`=(a×b)+b×10 em P2 mas `+` é irredutivelmente inconsistente. Confirmam: **filtrar + regerar**.
- **Fontes inexploráveis nomeadas (convergente):** program synthesis/ILP (DreamCoder, SyGuS-Comp), symbolic regression, **ARC**, **SCAN (Lake&Baroni 2018)**, **ListOps**, **BIG-bench symbolic-math/cryptonite**, **DeepMind Mathematics Dataset**. Uso: pré-treino/augmentation de indução-de-função (especulativo, $0 testável offline).

### IMPLEMENTAÇÃO ENTREGUE (offline/$0): `scripts/kg1_cryptarithm_clean_pipeline.py`
Módulo standalone, sem dependências, 3 funções + self-test:
1. **solve()** — infere a op de cada glyph (operator-substitution, dígitos reais; catálogo {add, |a-b|, mul, (a+b)±1, concat}); retorna insolúvel p/ no-op / query-op-unseen / ambíguo; expõe `unique_ops` (rótulo não-ambíguo).
2. **filter_corpus()** — gate ESTRUTURAL por padrão-de-query (não pela palavra "equation"); mantém só puzzles c/ op única por glyph + query-op visto + decode==gold; reescreve o CoT limpo.
3. **generate_clean()/generate_many()** — rejection sampling: op+glyphs aleatórios → 3-5 exemplos → solver confirma único → query-op presente → resposta calculada → **rejeita se o token-resposta aparece antes da Query**. CoT ensina FALSIFICAÇÃO (try op→reject→try→match), sem vazamento, 1 só `\boxed{}`.

### QA (code-review agent pr-review-toolkit + 14 desk-tests)
- Agent confirmou solver/rejection-sampler SOUND (0 respostas erradas em 167.960 linhas geradas). Achou 3 defeitos → **TODOS corrigidos e verificados**:
  - **C1** (filtro dependia da palavra "equation" → linhas quebradas com outra redação passavam direto) → gate por padrão-de-query; `equation_passthrough=0`.
  - **I2** (~14% das geradas tinham o token-resposta antes da Query) → rejeição no gerador; `leak=0/3716`.
  - **I3** (`sorted()[0]` colapsava subtração em |a-b|) → ops `a-b`/`b-a` removidas (indistinguíveis de |a-b| em resultados não-negativos) + exige `unique_ops`; `solver_disagree=0`.
- **Self-test 14/14 PASS** (inclui T11=C1, T12=I2, T13=I3, T14=determinismo).

### MÉTRICAS VALIDADAS (corpus real `fase1_corpus.jsonl`, 1500 equation)
filtro: **284 limpos (19%)** mantidos · 937 no-op (quebrado) · 102 query-op-unseen · 151 não-parseável · 21 decode-errado · 5 rótulo-ambíguo → **todos os 1216 sujos DESCARTADOS** (não vazam). Refil: **3716 regerados limpos**. Corpus final: `C:\tmp\clean_equation.jsonl` (8204 linhas = 4204 outras famílias + 4000 equation limpo). Verificação: bad_box=0, solver_disagree=0, leak=0, passthrough-equation=0 → **CLEAN**.

### PENDÊNCIA $0 antes do treino (caveat honesto)
Confirmar que os puzzles do TESTE OFICIAL têm a mesma estrutura limpa (operator-substitution) — provável (nosso gerador de treino é que estava quebrado). Se o teste também for quebrado, teto <100%. Depois: este corpus limpo alimenta `modal_fase3` (GPU, só com GO+custo ~$27). Treinar no corpus ANTIGO = repetir o colapso.

---

## §25 — DOUBLE-CHECK PAGO/HF + DESCOBERTA DA CIFRA DE REVERSAO (2026-06-09)

### Pagos: TODOS sem fundos ($0 gasto)
- OpenRouter (Qwen3-max, DeepSeek-R1, Gemini-2.5-pro, Claude-opus-4.1, GPT-5, Kimi-k2-thinking): **402 "never purchased credits"**.
- Chaves diretas (reference_api_keys.md): OpenAI gpt-5.x **429**, Anthropic opus **400 low balance**, DeepSeek **402**, Gemini-2.5-pro **429**. Nenhum pago acessivel. **Não comprar créditos (ação financeira do usuário).**

### HF PRO (melhores/recentes): Qwen3-235B-Thinking, Kimi-K2.6, GLM-4.7, Llama-4-Maverick, gpt-oss-120b (DeepSeek-R1 504)
### 🔴 ACHADO MAIOR — corrige conclusão anterior: a familia é uma CIFRA DE REVERSAO DE DIGITOS
**Qwen3-235B-Thinking CRACKOU** o que TODOS os outros (modelos grátis + nossa busca "exaustiva" H3 + proof-by-exhaustion do gpt-oss) tinham declarado QUEBRADO. Regra real por glyph = (transformação × operação):
  `c = T( op( T(a), T(b) ) )`, com **T ∈ {identidade, REVERSÃO de dígitos}**.
- **P2** `+` = (rev, somar): `88+81` → rev(88)+rev(81)=88+18=106 → rev(106)=**601** ✓ (todas as 4 linhas; query 14+66→**701**=gold). `*` = concat(b,a): "19"+"95"=1995.
- **P3** `-` = (rev, |a−b|): query 65-54 → rev(56−45=11)=**11**=gold.
- **Verificado computacionalmente** (`C:\tmp\crack_reverse.py`): adicionar reversão sobe a cobertura **17% → 35%** dos puzzles reais; base-change (6-16) não acrescenta nada (+0%). Restante ~42% provavelmente usa bijeção+reversão (futuro).

### Correção honesta
A conclusão "77% quebrado" das §22-24 estava **ERRADA** — era cegueira do nosso solver (não testava reversão), não defeito do dado. Foi exatamente o que o usuário buscava ao escalar para modelos melhores: o modelo mais forte achou a transformação que os menores e a busca local não acharam.

### IMPLEMENTADO (v2): `scripts/kg1_cryptarithm_clean_pipeline.py`
- Solver/filtro/regerador agora modelam **regra = (transform{id,rev} × op)**. `apply_rule`, `_consistent_rules`, CoT que mostra "reverse(a)=.., reverse(b)=.., op=.., reverse(result)=..".
- **Self-test 17/17 PASS** (T15 P2→701, T16 P3→11 reversão; T1 P1 id; +C1/I2/I3/determinismo).
- Filtro no corpus real: **kept 284 → 482** (224 reversão); regen 3518 (mix id+rev). Corpus `C:\tmp\clean_equation.jsonl` (8204 linhas). Verificação por arquivo (sem mangling de shell): **bad_box=0, solver_disagree=0, leak=0, passthrough=0 → CLEAN**; 2368 puzzles usam reversão.
- (Nota: `bad_box=4000` em verificadores inline foi artefato de shell comendo `\` em `\boxed`; os testes in-process e a checagem por arquivo confirmam 0.)

### Treinos sugeridos pelos modelos (especulativo, anotado): pré-treino de indução-de-função (ARC/SCAN/ListOps/DeepMind-Math) antes do SFT (Qwen/gpt-oss). Caveat: confirmar estrutura do TESTE oficial.

---

## §26 — REMAINDER + BIT (peça #2) + CORPUS FASE3 PRONTO (2026-06-09, offline/$0)

### Remainder do cryptarithm (mapeado, decisão tomada)
- id+rev_full = **35%** é o teto limpo. Variantes (rev-operandos-só, rev-resultado-só) **PIORAM** (35%→33%, ambiguidade). Base-change (6-16): **+0%**. Bijeção-dígito no restante: ~13% (caro+ambíguo). → manter **só id+rev_full**; o resto regenera. Sem mais lever barato.

### Peça #2 (bit): `scripts/kg1_bit_clean.py` (10/10 self-test)
- Solver por-posição (copy/not/const/2-input boolean) deriva a regra dos exemplos, **VERIFICA contra todos** e aplica. SOUND-gate: só reescreve CoT quando a regra derivada == gold (nunca emite CoT errado — testado B10).
- Bit é sub-determinado por 3-5 exemplos (por isso satura ~40%): solver forçado-único só 15%; greedy bate gold em **57%**. Reescrevemos esses **814/1428** com CoT que mostra derivação+verificação; 614 ficam intactos (respostas já corretas).

### CORPUS FASE3 montado + desk-test PASS: `C:\tmp\fase3_corpus.jsonl` (5704 linhas)
- equation: 482 filtrados limpos (224 reversão) + 1018 regerados = **1500 LIMPOS** (era 1500 quebrados).
- bit: 814 CoT verificados + 614 originais. 4 famílias fortes intactas. Distribuição = original (cipher 1576, equation 1500, bit 1428, numeral/gravity/unit 400).
- Desk-test: **bad_box=0, eq_unsound=0, bit_unsound=0, regen_leak=0 → PASS**.
- **Upload (não-destrutivo):** `fase3_corpus.jsonl` no `felipesp1983/kg1-train-bundle`. `modal_fase3_train.py` agora aponta p/ ele. Pré-flight `classify_family`: **0 unknown** (treino não recusa).

### ⛔ GATE HUMANO (próximo passo exige autorização de custo)
Tudo offline/$0 está PRONTO. Falta só o **treino GPU** (`modal_fase3`, ~$27) — exige GO+custo. Depois: pré-score gate (cipher≥0.97, ≥086) → submit (regra 99%). Treino agora roda em dado LIMPO (reversão-aware), não no que colapsou.

---

## §27 — P4/P5: consulta (5 modelos) + TESTES + implementacao (2026-06-09)

**Contexto:** smoke FASE3 PASSOU ([DONE] kg1-fase3-fix-smoke2; o RemoteError vazio anterior era infra transitorio). Full em espera (corpus melhorado antes).

### Consulta (Qwen3-235B, Kimi-K2.6, GLM-4.7, gpt-oss; DeepSeek-R1 504) + MEDICAO no corpus
- **P5 (ambiguo, 50 puzzles):** Qwen propos prior "digit-count" -> **TESTADO: só 26%** (a alegacao "98.1%" do Qwen era ALUCINADA). Kimi+GLM convergiram em "Constant-T / transform constante no puzzle"; **TESTADO: majority-transform = 68%, constant-T = 68%** (iguais). → **prior VALIDADO: majority-transform (68%)**.
- **P4 (query-op nunca visto, 217 puzzles):** **information-theoretically impossivel** (operador nunca demonstrado). Melhor chute medido: `|a-b|`=**23%** (Qwen disse rev-add 28% -> medido **11%**, alucinacao). Oraculo (se soubesse a op)=70%. → **mantido DROPADO** (chutar arrisca poluir; ganho ~23% de 15% da familia nao compensa).
- **bit-endianness** (Qwen: "73.5% little-endian, 40%->68%"): **TESTADO: 57%->58%** (nulo, alucinado).
- **Licao:** numeros dos modelos sao ALUCINADOS; só o teste no nosso dado vale. 3 modelos convergiram no conceito certo (constant-T), mas a versao precisa veio da nossa medicao.

### IMPLEMENTADO (validado, deterministico)
- `solve(..., tiebreak=True)`: prior majority-transform p/ ambiguos (determinístico, empate->lexicografico). Filtro usa tiebreak -> recupera ambiguos cujo resultado bate com gold.
- Regerador: 85% puzzles single-transform (ensina a convencao "1 transform por puzzle") + 15% misto (cobre tipo-P2).
- **Equation filtrado: 284 (v2 buggy) -> 482 (reversao) -> 610 (string-fiel) -> 644 (tiebreak).** Self-test 22/22 (T21 tiebreak no-regressao). Corpus FASE3 reconstruido (desk-test PASS: bad_box=0/eq_unsound=0/bit_unsound=0/leak=0) e re-upload.

### Impacto honesto
Ganho de P4/P5 e' MODESTO (+~34 equation reais + bias de convencao). O grande lever ja era a reversao (13%->~35%). P4 e' teto duro (~15% da familia ~insoluvel). Score: o pre-score medira; faixa honesta inalterada (otimista +0.02 a +0.04).

---
## §28 — BIT CRACKADO: whole-rule binop(A,B)+maj/choice = 77% sound (2026-06-09)
Estrutura do bit = combine(deslocamento_A(x), deslocamento_B(x)[,C]) — analogo a' reversao do cryptarithm.
Medido (sound, 0 erro): unaria 10% -> binop(A,B) 65% -> +maj/choice 77%. Supera greedy-por-bit 56% (unsound) e sound-por-bit 15%.
GF(2)-linear (sugestao DeepSeek) TESTADO e REJEITADO: unsound (312 erros) pois AND/OR/maj nao sao lineares.
22% restante: Qwen(504)/DeepSeek/Kimi/GLM nao crackearam -> teto genuino (sub-determinado).
IMPLEMENTADO: kg1_bit_clean.py v2 (solver whole-rule + CoT verifica+aplica), 7/7 testes. Corpus: 1116/1428 bit reescritos sound (binop788/maj183/unaria145). Desk-test PASS. Re-upload.

---
## §29 — VALIDADO NO HELD-OUT (val_clean 30/fam) + ganhos novos (2026-06-09)
REALIDADE medida no held-out real (corrige claims antes inflados que vinham do corpus enviesado):
- gravity/unit/numeral = 100% (deterministico). cipher/bit/equation eram o gap.
- CIPHER: substituicao monoalfabetica + vocab(77 palavras). Solver: mapa puro 38% -> +vocab-completion 98.5% -> +INJETIVIDADE(bijecao) **100%** (held-out 30/30, treino 1576/1576, 0 erro). [lead da auditoria de codigo; testado e confirmado]. kg1_cipher_clean.py.
- BIT: classe que FALTAVA = U(x) XOR/+/- const e binop(A,B) XOR const (lead do Kimi, params dele errados, classe certa). Held-out 10%->**20%** sound, 0 erro. depth-3, GF(2)-afim, perm+NOT, majority-rot: TODOS testados e REJEITADOS (unsound/0). kg1_bit_clean.py tiers T1-T6.
- EQUATION: forma real do eval e' SIMBOLICA (86% held-out), nao digito. Sem mapa global (refutado). So concat-glifo solvel (~3%)+digito(13%)=~16% teto. Aritmetica-sob-substituicao = sub-determinada (6 modelos top + meus testes nao crackearam).
AUDITORIA DO DATASET (programatica + 6 modelos): dataset limpo. Achados corrigidos: 400 dups equation removidas; 244->0 instrucao \boxed duplicada; bit CoT ganhou nota "8-bit leading zeros". boxed==gold 100%, sem leak real (os 499 'answer no prompt' = substring coincidente em exemplos). Imbalance tratado por balance_rows(cap30%).
CORPUS FINAL v5: 5304 traces (cipher1576/equation1100/bit1428/numeral400/gravity400/unit400). Smoke OK. Upload felipesp1983/kg1-train-bundle.
Antes->Depois cobertura solver (held-out): cipher 38%->100% | bit 10%->20% | equation ~16% | grav/unit/num 100%.

---
## §30 — ESTADO FINAL VALIDADO = FONTE-DA-VERDADE (2026-06-10)
**Tudo abaixo foi medido/validado; onde §2/§6/§22-29 divergirem, vale ESTE.**

### Corpus FINAL = `fase3_corpus_v5.jsonl` (5304 traces) — no HF felipesp1983/kg1-train-bundle
cipher 1576 (100% sound) · equation 1100 (dedup) · bit 1428 (1410 sound) · numeral/gravity/unit 400.

### Cobertura do SOLVER (held-out real val_clean 30/fam) — ANTES->DEPOIS desta sessao
| classe | baseline | final | ganho |
|---|---|---|---|
| cipher | 38% | **100%** | +62pts (+163%) |
| bit | 10% | **20%** | +10pts (+100%) |
| equation | ~16% | ~16% | 0 (teto) |
| gravity/unit/numeral | 100% | 100% | mantido |
(NB: cobertura do SOLVER = teto do que o modelo pode aprender; accuracy do MODELO sai no pre-score pos-treino.)

### Solvers (implementados, validados, QA-revisados)
- `scripts/kg1_cipher_clean.py`: subst. monoalfabetica + vocab-completion + INJETIVIDADE sempre + parser lowercase + guard exemplos-vazios. Selftest PASS.
- `scripts/kg1_bit_clean.py`: tiers Occam T1 unary | T2 U(x)XOR/+const | T3 binop | T4 binop XOR const | T5 maj/choice(perm) | T6 depth-2 + nota 8-bit no CoT. Selftest PASS.
- SOUNDNESS: corpus nunca ensina errado pois improve_*_corpus so reescreve quando pred==gold (gold-gate airtight, confirmado por 2 agentes QA).

### QA + auditoria deterministica (caminho critico) — TUDO OK
- Equipe QA (2 agentes Claude): achados corrigidos (cipher injetividade-sempre, empty-guard, lowercase; bit choice-perm; docstring honesta). Gold-gate airtight.
- Simulacao deterministica end-to-end: dataset->mascara(train_on_responses_only 6/6 OK)->loss(0 degenerado)->ACC com GRADER OFICIAL **100%/familia (5304/5304)**->pacote(0 multi-boxed). Sem backfire/bug silencioso.
- Auditoria dataset: 0 dup, 0 dup-instr, boxed==gold 100%, sem leak real. Budget tokenizer-real max 804<1024.

### SCORE (como e gerado — NAO e' o loss): grader oficial competition_utils.py
inferencia greedy temp=0 max_tokens=7680 -> extract_final_answer(ultimo \boxed) -> verify_answer(binario=exato | numerico=tol1% | senao string NFKC/case) -> media por-linha 6 familias.
Loss = cross-entropy so no response (prompt mascarado); molda o modelo, NAO e' o score.

### Treino (config FINAL, infra pronta, smoke PASS)
H100 · warmstart 086 (piso 0.86) · epochs=1 · max_len 1024 · lr 5e-5 · timeout 16h · salva s120/s240/ep1/final (retry+confirmacao remota) · ~9-12h ~$24-48.
**PENDENTE = unica acao humana: OK do Felipe p/ disparar (regra >30min).** Pos-treino: pre-score por familia vs 086 -> antes/depois MEDIDO -> submit so com 99%+autorizacao.

### LIXO/SUPERSEDED (nao confiar; corrigido pelo held-out)
- "bit 77%/89% sound" (§28,§2-antigo) -> real held-out **20%** (corpus era enviesado).
- "equation reversao 35%/cracked" (§24-25) -> eval real e' SIMBOLICO; real **~16%** (aritmetica-sob-substituicao sub-determinada).
- "cipher exige dict / 65% sem-letra" -> resolvido 100% com injetividade.

---
## §31 — VEREDITO FASE3 + MECANISMO DA REGRESSAO + RE-ROLLS (2026-06-10)
**PRE-SCORE MEDIDO (val_clean, grader oficial): FASE3-final REPROVADO.** 086 vs final:
cipher 1.000->0.133 | bit 0.400->0.233 | equation 0.133->0.067 | det 1.000=1.000 | GERAL 0.756->0.572.
Gate barrou o submit; 086 intacto. Mesmo padrao da fase2 (2x reproduzido) = MECANISMO, nao azar.

**MECANISMO (consenso 4 fontes: eu + DeepSeek-V4-Pro + Qwen3-Coder-480B + estrutura):**
loss perfeita = imitacao em teacher-forcing, nao execucao. Cipher vivia NO ADAPTER (mapeamentos
especificos, fragil; 1 letra errada = frase toda errada no exact-match); numeral/gravity/unit vivem
no BASE (formulas, robustas). Com cipher ja em loss~0 no warm-start, lr 5e-5 alto + 30% dos tokens
sendo cipher-estilo-novo + gradientes de equation/bit vazando pelo subespaco COMPARTILHADO do rank-32
=> rotacao/sobrescrita do procedimento que funcionava ("differential catastrophic rotation").
**RECEITA DO PROXIMO TREINO (consenso): lr 1e-5 a 1e-6; saturadas FORA (ou <=10% replay no estilo
ORIGINAL do modelo, nao o nosso); poucos steps; foco bit/equation; considerar congelar lm_head;
bit via SELF-DISTILLATION (gerar com o proprio 086 + gold-gate = zero deslocamento).**

**EQUATION encerrada:** CSP v5 final = 4/30 sound 0-erro (semantica: sub_ab/sub_ba c/ sinal=glifo
F/E por direcao do op + length-prefilter + split central); corpus-mode 17%; zero-pad REFUTADO (v6 3/30);
U1/U3 = q-glifo-inedito; consultas (20+ modelos em 4 rodadas) sem crack alem do nosso. Teto pratico.

**BIT T7 (2026-06-10 tarde): classe AFIM MOD-256 descoberta e integrada** — out=POST((k*PRE(x)+c)%256),
PRE/POST invertiveis (id/NOT/rev/ROL/ROR), k,c 0..255; c determinado pela 1a linha por (pre,post,k).
Held-out: tiered 6/30 + afim {4,6} novos = **8/30 (27%), 0 erro**. selftest 11/11. kg1_bit_clean.py T7.
B1/B2 continuam sem-fit (nem afim) — duros genuinos.
**Patamares AUDITADOS ao vivo (2026-06-10, decomposicao por classe):** antigas 4/30 (13%) ->
+XOR-const 6/30 (20%) -> +afim-256 8/30 (27%). GANHO COMPROVADO bit = +14 pontos (13->27),
nao +17 (baseline historico "10%" corrigido p/ 13% pela decomposicao).

**ACOES DO DIA:** 2x variance re-roll do 086 submetidos (refs 53536911/53537075, PENDING) — auditoria
profunda do zip 14/14 OK (sha256 byte-a-byte, 12011 tensores, r=32). Quota 2/5.
Velocidade (consenso historico supersedido): bs2-4+ga8-4, grad-ckpt off. Para score/prescore ativo,
usar contrato oficial: max_tokens 7680 e max_num_seqs 64. Infra: manter Modal=treino / HF Jobs=score
(libs SEMPRE pinadas) / Colab=emergencia.

---
## §32 — CONSOLIDACAO FINAL DOS ACHADOS (2026-06-10 noite; 6 rodadas de painel, ~50 retornos, tudo TESTADO)
**RE-ROLLS:** zip IDENTICO 2x no dia: 0.86 e 0.85 => variancia do host PROVADA (±0.01, sigma~0.007).
13x0.86 + 1x0.85 = estamos no MEIO do bin 0.86; P(0.87)/roll ~<7%. LB: 0.89x1, 0.88x3, 0.87x28,
0.86x1474; nos #45. Receitas publicas (huikang ProgressPrize #88@0.86; notebook "087" autor@0.86) =
plateau 0.85-0.86. Solver bit huikang = SUBCONJUNTO do nosso; cryptarithm dele (concat-pad4) provado
equivalente ao nosso p/ operandos 2-dig. Corpus dele 17,963 publico (risco style-clash se usado cru).
**SOLVERS = FRENTE FECHADA:** cipher 100% | det 100% | bit 27% (T7 afim-256; B1p/B2p provados alem de
9 classes; 8 modelos auto-refutaram) | equation 13%/17% (U2: digitos fixados por contagem, ops {$,}};
busca EXAUSTIVA 10!x20opsx2T = 0 solucoes; 2 "cracks" de modelos refutados linha-a-linha; zero-pad e
digit-wise refutados por medicao).
**RECEITA FASE4 v2 (consenso v8, 9 paineis):** lr 1e-6 (mediana; varios <=5e-7), steps 60-120, mix
75% self-distill (k=8, MENOR CoT correto por puzzle) / 15% solver-CoTs / 10% replay estilo-086;
freeze lm_head+embed; padding_side=right (guard Mamba2); tripwire canario por familia a cada 40 steps.
ANTES de treinar: medir OVERLAP modelo-vs-solver per-item no bit (se disjuntos, uniao ate 18/30=0.60).
**FORMATO CoT:** "revise-in-place" (grader pega ULTIMO boxed => autocorrecao gratis); nunca abster.
**SELECAO FINAL (consenso 9/9):** 2 adapters DIFERENTES de val equivalente > mesmo campeao 2x.
Trocar slot 2 por candidato novo SO se val>=0.77 + zero regressao por familia + >=0.86 publico.
**0.88+ real (especulativo, rankeado):** soup/merge especialistas; rank-residual sobre 086 congelado;
minimal-token self-distill; DPO greedy-sharpening; hipotese nula (0.88/0.89=sorte) segue viva.
**PRESCORE v2:** per-item logging (mede overlap); o antigo limite de 3500 tokens era LOWER bound.
Contrato ativo: prescore/score com `max_tokens=7680`; re-run truncados continua obrigatorio.

**§32-B — Q4 RESPONDIDA (2026-06-10, via Chrome na aba Leaderboard; pergunta que 7 rodadas nao fizeram):**
"This leaderboard is calculated with approximately 50% of the test data. The final results will be
based on the OTHER 50%, so the final standings may be different." => FINAL = SPLIT PRIVADO DISJUNTO.
IMPLICACOES: (1) re-roll de variancia no LB publico NAO transfere ao final (era loteria no split
errado); (2) manada 0.87 publica pode reordenar inteira no shake-up (single-roll 0.87 = sorte no
publico, regride ao real no privado); (3) nosso 0.86 com 13 amostras = estimativa ROBUSTA do nivel
real (~0.855-0.86) — plateau vira VANTAGEM estatistica: times acima por sorte caem, nos passamos;
(4) selecao final: maximizar SCORE REAL ESPERADO (nao o publico) — 2 adapters DIFERENTES robustos
(consenso 9/9 reforcado); (5) o que move o final e SO capacidade real => FASE4/especialista-merge
ganham prioridade sobre qualquer jogo de LB publico.

---
## §33 — CONSOLIDACAO 11-12/06: JUIZ DECIFRADO + DOUTRINA CERTIFICADA + FASE4-v2 COMPLETA E SUBMETIDA (fonte-da-verdade ATUAL; supersede §32 onde conflitar)

### 33.A O JUIZ, DECIFRADO (provado, nao suposto — log publico do utility script do host via kaggle CLI)
- Motor do juiz: **vllm==0.17.1 + torch 2.10 em RTX Pro 6000** (nao 0.19.1 como assumiamos).
- Display do LB **TRUNCA (floor) em 2 casas**: 0.8699 vira 0.86 => display 0.87 exige real >=0.870.
- Ancora RECALIBRADA: 086 real ~ **0.864** (nao 0.860; re-rolls 0.86/0.85 consistentes); sigma_host ~ 0.004.
- sd(publico-privado) ~ **0.57pp** (hipergeometrico, splits disjuntos 50/50; certificado pelo tribunal).

### 33.B DOUTRINA DE SCORE CERTIFICADA (tribunal externo: Qwen3-235B + Kimi-K2 + GPT-OSS-120B free + agentes internos; convergencia quadrupla)
- **P(privado>=0.87 | display 0.87) ~ 46-65% = CARA-OU-COROA.** Display 0.87 NAO decide selecao sozinho.
- **P(privado>=0.87 | display 0.88) ~ 86-99% = SEGURO.** => **ALVO OPERACIONAL: display 0.88.**
- Conversao val->test equation: shrink 0.8 era TETO otimista => **SHRINK_GAIN_EQ = 0.5** (preditor v3.3).
- POLITICA DE SLOTS (simulada; domina filtrar-por-preditor 55.6% vs 30.3% de P(slot2>=0.87 privado)):
  todo candidato GATE_PASS = submit-MEDICAO (informacao gratis, slots nao acumulam); sabado re-submit
  do lider (mede sigma_host de graca); domingo slot2 = MAIOR DISPLAY MEDIDO (slot1 = 086 TRAVADO).
  Cada submit segue exigindo GO explicito do Felipe ("prossiga" = GO).

### 33.C PRESCORE v4 = CLONE DO JUIZ (motor de medicao; backtest com resposta conhecida)
- Config espelho: vllm==0.17.1, revision do base PINADA (cbd3fa9f...), max_num_seqs=64, sem
  enforce_eager, gpu_mem=0.85, max_model_len 8192. Custo ~$2.5/rodada (HF A100 — HF SO medicao, treino NUNCA).
- Sentinelas: probe L0 (3 itens com/sem LoRA; identicos = adapter nao aplicado = ABORT), C2 sanity,
  SPRT, gate por familia, log POR ITEM (alimenta o preditor pareado).
- BACKTEST (vexplicit2, real conhecido 0.85): regua de 209 itens e **CEGA a +-1pp** (net +2 nao viu
  o -1pp); IC95 do delta +-2.6-3.7pp => prescore = **VETO + ORDENADOR**, nunca preditor fino.
- Regua ABSOLUTA validada: bit_oficial 19/29 ~ 65% ~ perfil test 0.70 (consistencia regua<->LB).

### 33.D PREDITOR v3.3 (score ANTES do submit; selftests 5/5; red-team ULTRACODE matou v1 e v2)
- Mecanica: discordantes PAREADOS item-a-item (b=so-candidato, c=so-086; Dirichlet-Jeffreys eps=0.5)
  + CONTROLE clone-emparelhado (clone do 086 corrige vies estrutural) + shrink assimetrico (perdas e
  bit_oficial x1.0; ganhos eq x0.5; ganhos sinteticos x0.8) + truncamento de display + sigma_host.
- Decisao por SINAL LIQUIDO: net >= +4-6 itens = sinal real; +-2 = ruido (calibrado no backtest).
- 1o uso real (s40): net = +0 => "SEM SINAL — nao gastar slot" (politica funcionou de primeira).
- v3.4 (12/06, IMPLEMENTADO+TESTADO 9/9): camada PUBLICO->PRIVADO codificada (`p_private`,
  posterior fechada real|display x SD_SPLIT=0.57pp, 3 priors cons/mod/flat) — REPRODUZ a banda
  certificada sem ajuste (0.87 -> 49/65/75%; 0.88 -> 76/91/98%). Uso pos-display:
  `python score_predictor.py --pprivate <display>`. LIBERADO: repo scripts/ + Hub bundle.

### 33.E SAGA COLAB: 6 FALHAS -> 6 VACINAS (notebook v17-FULL + core v6; TODAS determinciticas e testadas)
| # | Falha | Vacina permanente |
|---|---|---|
| 1 | unsloth `_get_statistics` TimeoutError 120s | `UNSLOTH_DISABLE_STATISTICS=1` |
| 2 | Colab PRE-IMPORTA huggingface_hub (mix disco/memoria) | PURGE de sys.modules dos 9 pacotes pinados ANTES do 1o import |
| 3 | Preempcao silenciosa de VM (sem traceback) | escada de checkpoints s40/80/120/140/160 + forense via commits do Hub |
| 4 | kernels CUDA mamba_ssm/causal_conv1d com ABI quebrada (decode caiu p/ python 0.7-2.6 tok/s) | rebuild forcado (26.7min) + probe correto `import torch, <kernel>` (sem torch SEMPRE falha libc10 = falso alarme) |
| 5 | numpy e C-ext: pin PyPI subiu 2.0.2->2.4.6 e unsloth abortou | pinar numpy A VERSAO DA VM (2.0.2) nos 2 lotes + rede RESTART-NECESSARIO |
| 6 | bitsandbytes registra operadores torch (re-import em kernel quente = erro) | restart de sessao OBRIGATORIO pos-install; diagnosticos SO em kernel reiniciado (traceback Jupyter segura 74GB VRAM) |
- Estrutural: decode no Colab segue ~25x mais lento que Modal MESMO com kernels saos (loop python do
  MoE) => **CANARY_OFF=1** (canario fora do caminho critico; vigia semantica = prescore externo do s40).
  TREINO e saudavel (caminho Triton; baseline 2.208 = Modal 4x). Operacao 100% via Chrome MCP
  (restart/play/dialogos por coordenadas; dropdown congela screenshot CDP => clique cego).

### 33.F FASE4-v2 TREINADA E SUBMETIDA (Colab A100, 160/160 steps, 247min, loss 23->7.6, sentinelas calados)
- Corpus 1.648 **100% prompts oficiais** (check permanente X2): 769 eq pos-purga-leak + 289 bit
  dedutivo + 590 replay (60 ancoras sinteticas da FASE1 REMOVIDAS). val_bit_official RECONSTRUIDO
  verbatim do train.csv (29/29 oficial+gold, corrupcao chr(8) sanada, leak 0; checks X3a-e).
- Fix FAB#H (mecanismo da regressao FASE3): traces fechavam `<think></think>` vazio vs eval ABRE
  `<think>` => mismatch de template; corrigido (fix_think_wrap.py) + validado no tokenizer (max 1672 tok).
- Receita: warm-start 086, lr 5e-6, 160 steps, bs2xga8, warmup 0.1, max_len 2048, freeze conforme §32.
- Kill-switch s40 (prescore v4): GATE_PASS, **ZERO regressao de familia** (sem veneno FASE2/FASE3).
- Checkpoints TODOS no Hub (kg1-fase4-cons-s40/80/120/140/160 + final).
- **s160 SUBMETIDO** ref 53586867 (12/06 02:44 UTC, PENDING): zip 3.820MB, 2 arquivos na raiz,
  config diff vs 086 = VERDE (mesmos 9 target_modules incl. lm_head, r=32, alpha=32).

### 33.G ESTADO AO VIVO (12/06 ~03:00 UTC) + PLANO ATE O FIM
- RODANDO: score Kaggle s160 (sentinela 10min) + prescore s160 (HF job 6a2b6eac, ~$2.5) + 2 agentes
  de gap (reweighting / hipergeometrico).
- 12/06: consumir prescore s160 -> preditor v3.3 -> briefing -> submits-MEDICAO s120/s140 (GO por submit).
- 14/06 sab: re-submit do lider (mede sigma_host) + Open Contribution Award (lembrete 09:00 BRT agendado).
- 15/06 dom <=15:00 BRT: SELECAO FINAL — slot1 = 086 (travado), slot2 = maior display medido
  (checklist: docs/SELECAO_FINAL_CHECKLIST.md). Deadline 20:59 BRT.
- Budget: HF ~$5-7 gastos de ~$10 (teto medicoes); treino = Colab (CU ok). Regras vivas: GO p/ job
  >30min; pos-job `hf jobs ps` + cancelar zumbi; 99% de certeza antes de submit que vira selecao.

### 33.H INVESTIGACAO DE RUPTURA 12/06 madrugada (2 agentes + forense per-item do s160)
- MEDIDO: s160 display 0.86 (ref 53586867); prescore 206/209 identico ao 086 (net +1 bit);
  val fraco e' MUITO mais duro que test (val bit 37% vs test 0.70; val eq 13% vs test 0.46);
  truncamento val 44-45% ambos (test nao espelha: saturadas=100%).
- ENTERRO NOVO (bit): desempate de ambiguidade por coerencia/EM — train CV prometia +29pp
  (34.7->63.6 nos ambiguos; EM 61-63% total) MAS regua oficial: estrategia 14/29 < modelo 19/29
  e 0/10 nos misses dos modelos => modelo ja supera a estrategia; confirma "prior 0/8" do K por
  rota independente. NAO ensinar tie-break ao modelo (rebaixaria). Corpus 846 traces ARQUIVADO
  (C:\tmp\fase5_bit_coherence_traces.jsonl, nao usar).
- ACHADO MAIOR (eq, agent_eqasym): gerador 100%% engenheirado-reverso: 13 ops (add/add+-1/sub/
  subba/+-|a-b|/mul/mul+-1/catab/catba/maxmod) x transforms {id, reversao-string operandos+result,
  zeros preservados}, negativo = glifo do operador como sinal. Familia = 732 numeric + 823 glyph
  (cifra digito->glifo). 086 ja esta ~no teto do numeric (dedutivel+guess = 40.1%% da familia;
  q_op_unseen 136 = 18.6%% indeterminavel); GAP INTEIRO = glyph (~11%% captura atual).
  CSP cracka 90.8%% do glyph; gold-UNICO 281/823; corpus cobria 201 => +80 itens novos sound.
  Token-economics: SEM alavanca por encurtar CoT (mediana 349 tok << 7680; truncamento = runaway
  em item insoluvel; resgate exigiria stop-rule+guess).
- FASE5-GLYPH: build dos 80 oficiais => SO 7 sound (65 ambiguos p/ cadeia rigida, 5 leak
  bloqueado, 3 hostis) = PORTA OFICIAL ESGOTADA (corpus 769 ja era o maximo extraivel).
- VAL EQ DESVENDADO (12/06): 30 = 4 numeric + 26 glyph (glyph-pesado); solver alcanca 11/30,
  modelo 4/30 => 7 itens dedutiveis que o modelo PERDE = criterio de sucesso VISIVEL p/ FASE5
  (a regua nao e' cega p/ glyph; FASE4 capturou 0/7 = parede de DOSE, nao de regua).
- CHAVE SINTETICA DESTRANCADA: template UNICO provado (esqueleto bfb3fa68f0 nos 1555;
  n_ex 3/4/5 = 488/540/527); gerador sintetico byte-fiel com ARBITRO = capture congelado
  (so entra o que a cadeia oficial resolve sound). Piloto 40/40 skel_fail=0 gold_fail=0.
  Producao 1200 traces (4 procs, seeds 11/22/33/44; dist. ops/transform/alfabetos MEDIDAS
  do train). Caveat X2: prompts sinteticos com hash de esqueleto byte-identico + arbitro;
  treino FASE5 SO com GO explicito (custo ~3-5h Colab + $2.5 prescore).
- Matematica do rompimento: dose glyph 254->~1500 (6x) na unica habilidade com headroom;
  captura 11%->25-40%% => eq 0.46->0.53-0.61 => real 0.878-0.891. Risco: r32 nao absorver
  busca nem com 6x dose => portfolio D (086+s160) intacto como fallback.

### 33.I EXERCITO 12/06 (3 jurados internos EXECUTARAM + 17 modelos externos; artefatos juror1/2/3 + army_*.md em C:\tmp)
- JURADO 3 (MC 400k, gate-validado vs p_private): REGRA DOMINGO REVISADA — s160 DOMINA 086;
  C(FASE5) display>=0.86+gate => selecionar S160+C (P>=0.87 priv: 38/66/91% p/ 0.86/0.87/0.88);
  C<=0.85 => 086+s160 (~28%). NUNCA 086+C. Gravado em SELECAO_FINAL_CHECKLIST.
- JURADO 2 (censo nos traces reais): M2 LINHA MAGICA — 100% dos 590 traces glyph ESCONDEM a
  busca ("searching finds a consistent reading") => 0 gradiente na habilidade = explicacao
  mecanica do 254->0/7. M1: synth v1 errou estrato (1.8% 3-op vs 57% oficial) => v1 DESCARTADO
  (336). Receita: quota 3/2/1op=428/300/22, deducao EXTERNALIZADA row-a-row verificada,
  numeric cortado 515->230 (no teto), replay 740 cipher x2, 450 steps lr5e-6 max_len=2048
  OBRIGATORIO, CKPTS 150..450, curriculum sort=NO-OP (RandomSampler), dial real = duplicacao.
  GATE-87: receita completa P~0.17 (0.10-0.25); como estava P~0.05.
- JURADO 1 (ataques executados): TETO BIT 0.72 = FALSO (plato de DESTILACAO): solver producao
  23/29 na regua oficial vs modelo 19/29; train 95.3% hibrido; prior Occam tier 97.8% nos
  catalog-ambiguos => headroom bit +2.3-4.1pp, MAIOR alavanca da janela. q_op_unseen: nao e
  info-free (sem-reposicao p<5e-5; politica glifo+exclusao 30.1% vs 21.3%) mas vale so +0.13pp.
  Glyph canonico: TETO CONFIRMADO (12 convencoes, max 45% ~ acaso).
- FASE5 PLANO CONSOLIDADO: synth glyph v2 (quota+deducao explicita; producao 760 em curso,
  piloto 12/12 verify_fail=0) + REBUILD dos 254 oficiais com deducao externalizada + traces
  BIT catalog-ambiguos com prior de tier explicito no CoT + numeric 230 + replay 740.
  Treino 450 steps SO com GO Felipe.
- CORRECOES DE NUMERO aceitas do exercito externo (army_consolidado.md): teto eq 0.61 ->
  ~0.563 deterministico + credito probabilistico (documentar split); captura glyph atual
  11% -> 14.6% (back-solve J2 confirma). Nao mudam nenhuma decisao. Ideia nova unica
  sobrevivente: depth-2 SISTEMATICO nos 3 no-fit do bit (CPU<1h, ganho max +0.17pp,
  fila opcional). 4 modelos externos fabricaram [DERIVED] -> rebaixados.

### 33.J JURADO VERMELHO 12/06 — TETO BIT FURADO 2a VEZ (WORD-UNIFORM) + soup math
- B1: regra WORD-UNIFORM (mesma fn booleana 4-taps shift-com-zero em TODAS as colunas) resolve
  os 3 no-fit + idx23 da regua: solver 23/29 -> 27/29 (93.1%); train 94.4% -> 98.6% (+67, 0
  contradicoes, p<1e-9). MEMORY "BIT TETO PROVADO" FALSIFICADA (corrigida). Falha antiga: depth-2
  por-coluna 3-vars nao cobria regra de 4 taps amarrada entre colunas. Marginal: +0.35-0.70pp.
- ACAO: traces W-tier dos 67 train rescued (agente construindo; regua-val NUNCA treina).
- R1 soup: lm_head/embed CONGELADOS no core => per-module morto; soup last-3 uniforme +0.2pp
  [0,+0.5]; medir EXATAMENTE 1 merge ($2.5) + 1 condicional WiSE alpha 0.25; dominancia
  pre-comprometida p/ slot sabado. R3: medir Dt (truncado-sem-box) GRATIS nos artefatos de
  prescore antes de qualquer dose guess-box. B2 prior-ambiguo glyph: MORTO (49/100, nao passa 50%).
- RODADA 3 CONSOLIDADA (army3_consolidado.md): soup last-3+1-medicao CONFIRMADO (v31 verbatim;
  u550 refez o SE ~2.35pp que mata greedy-soup); 7 wildcards dedupados W1-W8: 4 exigem GPU
  (nao cabem), 1 ja-coberto (WiSE), 1 fabricacao (SVD+Procrustes nao altera dW); ZERO wildcard
  CPU<1h novo. ANTI-parafrase (q80) reforca gate R4. Dt (truncado-sem-box) = unica medicao
  pendente, GRATIS no proximo prescore. G1-G6 MORTO (residuo bit = ambiguidade irredutivel);
  retry CSP: rescues ja estavam no corpus (esteira FASE4 > CSP do agente). EXPLORACAO ESGOTADA
  COM PROVA: pacote FASE5 IMUTAVEL (mix 2301 + core v7 8578e2a1 + nb v18) aguardando GO.

### 33.K META 1o LUGAR — ordem-estatistica do campo (12/06; MC field_orderstat_fast.py, gate p_private OK)
- Lider 0.90 publico (NullSira) = real ESTIMADO ~0.8856 (farm ~220 submits) = QUASE no teto provado 0.888.
  NAO e' so sorte: alguem resolveu quase todo o resolvivel. Vencedor do campo no privado: P50=0.88 (P10 0.87/P90 0.89).
- P(NOSSO 1o lugar) x real: 0.866=0.5% | 0.872=4% | 0.876=10-13% | 0.880=22-27% | 0.884=40-47% | 0.888=60-68%.
- DIVERSIDADE no topo vale +5-8pp (corr 0.5 vs 0.95) => regra de selecao re-otimizada p/ titulo (par diverso).
- CONCLUSAO: FASE5 deixou de ser opcional; titulo exige real>=0.880 (topo da projecao 0.872-0.884) + par diverso + sorte de split. Sem treino: P(1o)~0.5%.
# STATUS 2026-06-13: HISTORICO/FORENSE
Nao use este arquivo como runbook ativo, fonte de permissao de GPU, pacote de adapter, submit Kaggle ou claim de score. O caminho ativo e gated esta em `docs/KG1_SCORE_DETERMINANTS_AUDIT_2026_06_13.md`, `scripts/kg1_score_path_operational_audit.py` e `scripts/kg1_active_gate_registry_audit.py`.
