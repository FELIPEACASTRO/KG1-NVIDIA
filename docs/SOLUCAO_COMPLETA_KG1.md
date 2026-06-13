# 📕 KG1 — Solução Completa NVIDIA Nemotron Reasoning Challenge
### Documento mestre: macro → hiper-micro · acertos · erros · tentativas · scores · treinos · resultados

> **Atualizado:** 2026-06-13 · **Autor:** Felipe (felipe1983) + Claude
> **Objetivo absoluto:** TOP 1 (0.90+). **Deadline:** domingo 2026-06-15 20:59 BRT
> **Melhor resultado provado:** **0.86** (submit086) · **Teto observado no campo:** ~0.85-0.86

---

## 🗺️ PARTE 1 — VISÃO MACRO (o desafio em 1 página)

### 1.1 O que é a competição

```
┌─────────────────────────────────────────────────────────────┐
│  NVIDIA Nemotron Model Reasoning Challenge (Kaggle)          │
│                                                              │
│  TAREFA:   treinar um ADAPTER LoRA para o modelo base        │
│            nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16        │
│            que resolva 6 famílias de "puzzles de raciocínio" │
│                                                              │
│  RESTRIÇÃO: rank LoRA <= 32 (hard cap)                       │
│  ENTREGA:   submission.zip = adapter_config.json +           │
│             adapter_model.safetensors (2 arquivos na raiz)   │
│                                                              │
│  AVALIAÇÃO: vLLM greedy (temp=0), max_tokens=7680,           │
│             exact-match do \boxed{...} final                 │
│  DISPLAY:   trunca no PISO de 2 decimais (0.869 → "0.86")    │
│  PRIVADO:   re-score em 50% disjunto do teste oculto         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 As 6 famílias de puzzle (e onde está o dinheiro)

| Família | Natureza | Nosso domínio | Comentário |
|---|---|---|---|
| **cipher** | decodificar cifra | ✅ ~1.00 | resolvido cedo |
| **numeral** | conversão de base/numeral | ✅ ~1.00 | resolvido |
| **gravity** | física simples | ✅ ~1.00 | resolvido |
| **unit** | conversão de unidades | ✅ ~1.00 | resolvido |
| **equation** | resolver/decifrar equação | ⚠️ ~0.46 | **GAP #1** |
| **bit** | lógica booleana / cryptarithm | ⚠️ ~0.70 | **GAP #2** |

> 🎯 **A corrida toda gira em torno de equation + bit.** As outras 4 já estão no teto. Subir de 0.86 → 0.90 = fechar esses 2 gaps SEM quebrar as 4 que já funcionam.

### 1.3 A anatomia de um score (por que 0.86 trava)

```
score ≈ média ponderada das 6 famílias
       = (cipher 1.0 + numeral 1.0 + gravity 1.0 + unit 1.0
          + equation 0.46 + bit 0.70) / 6  ≈ 0.86

Para chegar a 0.90 precisamos, p.ex.:
   equation 0.46 → 0.75  E  bit 0.70 → 0.85
   SEM derrubar as 4 famílias de 1.0
```

### 1.4 A realidade do leaderboard (descoberta crítica)

- O "0.877" que perseguíamos era **taxa de TRAIN**, não LB.
- **Vencedor real do campo ≈ 0.85-0.86.** O topo é dominado por **variância** do vLLM (não-determinismo MoE-LoRA gera ±0.02-0.03 entre rodadas idênticas).
- Ou seja: **a corrida pelo topo é parte receita, parte loteria de variância.**

---

## 🧩 PARTE 2 — A SOLUÇÃO (componentes ponto a ponto)

### 2.1 Stack técnico

| Camada | Escolha | Por quê |
|---|---|---|
| Modelo base | Nemotron-3-Nano-30B-A3B (MoE híbrido Mamba+Attn) | imposto pela competição |
| Arquitetura | 23 Mamba (M) + 23 MoE (E) + 6 Attn (*); 128 experts roteados, 6 por token | hybrid_override_pattern (verificado no config.json live 13/jun) |
| Fine-tuning | LoRA rank 32, alpha 32, 9 target_modules | hard cap rank≤32 |
| Treino | Unsloth + TRL SFTTrainer, A100/H100 | 2x mais rápido, -60% VRAM |
| Ambiente | Colab Pro H100/A100 (treino) + HF Jobs (prescore) | budget |
| Avaliação local | vLLM 0.17.1 greedy (paridade c/ juiz oficial) | mesma engine do host |

### 2.2 O corpus de treino (a matéria-prima)

```
EVOLUÇÃO DO CORPUS:
  andy279 (49.290 traces brutos)
     │  purga de leak + dedup + auditoria
     ▼
  corpus FASE1 → ... → corpus1648-official (s160, score 0.86)
     │  expansão glyph+tier
     ▼
  fase5_mix.jsonl = 2301 traces  ←── ÚLTIMA TENTATIVA (FALHOU)
     ├─ syn2 .............. 705 (synth equação, quota 3/2/1-op 428/300/22)
     ├─ bit_tier ......... 101 (222 cortados por max_len>2032)
     ├─ glyph_rebuilt .... 254 (dedução explícita, 0 magic-line)
     ├─ numeric .......... 230 (estratificado)
     ├─ replay ........... 576 (anti-esquecimento)
     ├─ replay_cipher_x2 . 150 (dobra cipher p/ proteger família 1.0)
     └─ bit_wuniform ..... 63  (regra W-UNIFORM 4-taps, solver 27/29)
```

### 2.3 Os achados de engenharia reversa (o conhecimento duro)

| Achado | Descoberta | Status |
|---|---|---|
| **equation v2** | held-out 20% = 45.3%; val 10/30 com 0-erro | implementado |
| **bit W-UNIFORM** | regra WORD-UNIFORM 4-taps → solver 27/29 (98.6% train) | implementado |
| **mapa glifo→op** | ENTERRADO (teto 41%, era artefato) | descartado |
| **bit teto falsificado** | "teto" antigo era erro de parametrização | corrigido |
| **q_op_unseen ~18%** | classe indeterminável | limite teórico |
| **FAB#H fix** | traces tinham `<think></think>` vazio (mismatch template) | corrigido |

---

## 🏗️ PARTE 3 — AS FASES (cronologia hiper-micro de cada treino)

### FASE 0 — Baseline e descoberta do 0.86 (abril)

```
v40 (06/abr) score 0.58  → primeiro SFT, loss 7.12
v49 (09/abr) score 0.64  → nuclear lr1e-4
v50c (11/abr) score 0.66 → NO-STRIP v30-exact, 3.5GB
v51 (11-12/abr) 0.47-0.53 → série STRIP falhou (smart-strip QUEBRAVA o adapter)
  ⚠️ ERRO-CHAVE: o "strip" do safetensors corrompia pesos → -0.30 de score
```

**🔴 Lição FASE 0:** STRIP do adapter mata o score. Desde então: **submissão CRUA (sem strip)**, validada (086 provou 0.86 cru).

### FASE 0.5 — Replicar a receita campeã (abril)

```
v73 (15/abr) 0.64  → A100-40GB NF4, loss 1.01
v120 (23/abr) 0.85 → Modal Surfer, réplica huikang/nemotron, target 0.86
(24/abr) 0.86 ★ → PRIMEIRO 0.86! (submissão sem descrição, fix de layout)
v158 (25/abr) 0.86 → re-empacotamento "FASE 0 - V120 baseline (name fix)"
```

**🟢 Acerto:** replicar a receita pública de referência (huikang) nos deu o **primeiro 0.86** (24/abr); v158 (25/abr) consolidou.

### FASE 1-3 — Tentativas de superar 0.86 via soup/merge (maio)

```
v087 (27/abr) 0.86  → delta safeoom s80
aaitdads (01/mai) 0.86 → adapter local853
v191 (02/mai) 0.78  → SVD soup w50 (REGREDIU)
v192 (02/mai) 0.86  → W98 conservative soup (keep backbone+lmhead)
v193 (02/mai) 0.86  → attention-only micro-merge lineage1
v194 (03/mai) 0.86  → attention-only micro-merge lineage1p5
V199B (04/mai) 0.86 → exact-sha final
V281 (11/mai) 0.86  → safe rank19
V291/V290 ckpt-6 (11/mai) 0.86 ★ = submit086 (CAMPEÃO ATUAL, mais validado)
```

**🟡 Padrão FASE 1-3:** dezenas de merges/soups, TODOS travam em 0.86 display. O merge conserva, mas não cria habilidade nova. Soup agressivo (SVD w50) REGREDIU para 0.78.

### FASE 4 — Re-treino com corpus oficial (junho)

```
v_concise (05/jun) 0.22 ☠️ → corpus pequeno anti-truncation DESTRUIU o modelo
v_explicit2 (05/jun) 0.85 → EXPLICIT-compressed bit-gates
vfull (05/jun) 0.86 → full-reasoning LR8e-5 r32 (over-compression corrigido)
re-roll (05/jun) 0.86 → loteria de variância do baseline
re-roll (10/jun) 0.86 / 0.85 → 2 re-rolls (FASE3 gated out por prescore)
s160 (12/jun) 0.86 ★ → FASE4v2 corpus1648-official kill-switch-PASSED
```

**🟢 Acerto FASE 4:** s160 passou no kill-switch e bateu 0.86 → é nosso 2º melhor candidato.
**🔴 Erro FASE 4:** v_concise 0.22 — corpus pequeno + anti-truncation agressivo = colapso.

### FASE 5 — Glyph+Tier (a tentativa final, FALHOU) — 12-13/jun

Esta é a fase mais documentada porque foi a última grande aposta.

```
HIPÓTESE: corpus glyph_rebuilt (dedução explícita) + bit W-UNIFORM
          + replay reforçado → fechar gaps equation/bit
RECEITA:  warm-start s160, lr 1e-5 (re-dose D19 de 5e-6), 420 steps
```

#### Cronologia hiper-micro da FASE 5 (a saga das tentativas)

| Tentativa | O que aconteceu | Causa-raiz |
|---|---|---|
| FULL #1 | morreu em ~2min | `AttributeError HF_HUB_ENABLE_HF_TRANSFER` (D7) |
| FULL #2 | morreu em ~1.8min | `TypeError sum(n...)` linha 65 (bug D22) |
| FULL #3 | rodou s8, interrompido | usuário abriu 2ª aba → KeyboardInterrupt |
| reabertura | rodou SMOKE acidental | gist v18 NÃO tinha os 4 hotfixes (edições unsaved perdidas) |
| **v19** | SMOKE PASS | HOTFIX-PACK 4/4 gravado no gist (à prova de reload) |
| **v20 FULL** | rodou até s140, **MORTO no gate** | **regressão de skill (ver abaixo)** |

#### Os 4 HOTFIXES da FASE 5 (engenharia de robustez)

```
HOTFIX1: monkeypatch constants.HF_HUB_ENABLE_HF_TRANSFER = False
         (hub em memória 1.17 vs disco 0.36.2 → AttributeError)
HOTFIX2: R14 gate bypass (mix sem chave "answer" → gate reproduzido local 2301/2301)
HOTFIX3: LOGPROB-WATCH @60 steps (observacional, gold-logprob dos ensináveis)
HOTFIX4: META#A len-fix → sum(len(n)...) em vez de sum(n...) (bug D22)
```

#### A trajetória de treino do v20 FULL (números reais)

```
step   1: loss 15.36 | gnorm 22 | lr 0 (warmup)
step  43: loss  8.97 | lr atinge pico 1e-5
step  60: loss  7.13 | LOGPROB-WATCH -3.4977 (PRE era -3.8431) → +0.345 nats
step 120: loss  2.69 | LOGPROB-WATCH -2.8843 → +0.96 nats ACUMULADO ⚠️
step 140: loss  2.19 | SAVE s140 → kill-gate
```

#### 💀 O VEREDITO (prescore s140 vs 086, motor vLLM 0.17.1)

```
[C2] 10/10 cipher no 086 → harness VÁLIDO
[086  b0] 15/20 ok (trunc 6)
[cand b0] 10/20 ok (trunc 3)   ← s140 PERDEU 5 itens
[SPRT] W=0 L=5 LLR=-2.555 → ABORT (pior, 95% confiança)
[GATE] cipher: 086=3 → cand=0 → GATE_PASS=False
verdict = CAND_PIOR_EARLY_EXIT
```

**Decisão executada:** treino morto no s140 (economizou ~5.5h A100), runtime excluído, job HF cancelado, zumbis varridos.

#### 📉 A confirmação brutal no LB

```
submit s140 → SCORE PÚBLICO = 0.53   (−0.33 vs 0.86)
```

---

## 🧠 PARTE 4 — A GRANDE LIÇÃO DA FASE 5 (o erro mais caro e instrutivo)

### 4.1 O paradoxo que enganou

```
LOGPROB-WATCH dizia:  +0.96 nats → "transfer ótimo, modelo aprendendo!"
SCORE REAL no LB:     0.53        → COLAPSO de 38%
```

### 4.2 O que realmente aconteceu (diagnóstico)

> **Esquecimento catastrófico mascarado por métrica enganosa.**
>
> O modelo aprendeu a "soar" como os traces de treino (logprob sobe) MAS isso **canibalizou habilidades que já estavam em 1.00**. A prova: cipher caiu de 3/3 → 0/3. O s140 truncou MENOS que o 086 (3 vs 6) e mesmo assim acertou MENOS — ou seja, **não era verbosidade, era perda de raciocínio.**

### 4.3 As 3 lições gravadas em pedra

```
1. LOGPROB-WATCH está MORTA como sinal de qualidade.
   Subiu lindo e o score despencou. Só prescore (geração+grader) vale como gate.

2. lr 1e-5 nessa receita = VENENO.
   O "re-dose D19" que parecia "acoplar" acelerou a destruição.
   O smoke "ACOPLADO" mediu a coisa errada.

3. A direção glyph+tier/FASE5 está FALIDA.
   Não é afinar — o corpus/abordagem corrompe o modelo base. Não vale mais GPU.
```

---

## 📊 PARTE 5 — TABELA-MESTRE DE TODOS OS SCORES (50 submissões)

### Distribuição

```
0.86 ██████████████ 14   ← TETO (melhor display)
0.85 ███ 3
0.84 █████ 5
0.78 █ 1
0.66 █ 1
0.64 ██ 2
0.62 █ 1
0.61 ██ 2
0.58 ██ 2
0.53 ███ 3            ← inclui s140 (FASE5)
0.50 ██████████ 10    ← maioria = série STRIP quebrada
0.22-0.54 (resto)     ← experimentos que colapsaram
```

### Marcos (os que importam)

| Data | Score | Marco |
|---|---|---|
| 06/abr | 0.58 | primeiro SFT (v40) |
| 11/abr | 0.66 | melhor pré-receita-campeã (v50c NO-STRIP) |
| 23/abr | 0.85 | v120 réplica huikang (Modal) |
| **24/abr** | **0.86** | **PRIMEIRO 0.86** (submissão sem descrição; v158 replicou 25/abr) |
| 11/mai | 0.86 | **submit086 = V291/V290 ckpt-6 (CAMPEÃO)** ⭐ |
| 12/jun | 0.86 | **s160 FASE4v2 (kill-switch-passed)** ⭐ |
| 13/jun | 0.53 | s140 FASE5 (reality-check, falhou) |

---

## ✅ PARTE 6 — ACERTOS vs ERROS (balanço honesto)

### 🟢 ACERTOS

```
1. Submissão CRUA (descobrir que STRIP quebra o adapter)         → +0.30
2. Replicar receita campeã huikang (v120/v158)                   → primeiro 0.86
3. Disciplina de PRESCORE antes de submit (regra 99%)            → evitou queimar slots
4. Kill-gates e sentinelas (mataram treinos ruins cedo)          → economia de GPU
5. Engenharia reversa equation v2 (45.3% held-out)               → conhecimento real
6. bit W-UNIFORM solver (27/29)                                  → conhecimento real
7. HOTFIX-PACK no gist (robustez a reload)                       → reprodutibilidade
8. Observabilidade (heartbeat + tripwire)                        → diagnóstico remoto
```

### 🔴 ERROS

```
1. Série STRIP v51 (0.47-0.53) — só descoberto após ~8 submits  → -slots
2. v_concise 0.22 — corpus pequeno + anti-truncation agressivo  → colapso
3. SVD soup w50 0.78 — merge agressivo regrediu                 → -0.08
4. CONFIAR no LOGPROB-WATCH como sinal (FASE5)                   → falso otimismo
5. lr 1e-5 re-dose (FASE5) — acelerou esquecimento catastrófico → s140=0.53
6. Abrir 2ª aba Colab → matou FULL #3                           → perda de tempo
7. Edições unsaved no gist (hotfixes perdidos em reload)        → 3 tentativas falhas
```

### 🟡 NEUTROS / APRENDIZADOS ESTRUTURAIS

```
- Merge/soup CONSERVA 0.86 mas nunca CRIA habilidade → teto estrutural
- O topo do LB é variância vLLM (±0.02-0.03), não receita pura
- "0.877" era train, não LB → expectativa recalibrada
```

---

## 🎯 PARTE 7 — ESTADO FINAL E ESTRATÉGIA DE FECHAMENTO

### 7.1 Onde estamos (13/jun)

```
✅ Melhor ativo provado:  submit086 = 0.86 (V291/V290 ckpt-6)
✅ 2º melhor:             s160 FASE4v2 = 0.86 (kill-switch-passed)
☠️ FASE5 enterrada:       s140 = 0.53 (esquecimento catastrófico)
⏰ Deadline:              domingo 15/jun 20:59 BRT (~44h)
```

### 7.2 Decisão tomada: ACEITAR 0.86 + seleção final diversa

```
SLOT FINAL 1 = submit086 (V291/V290 ckpt-6)
   └─ o mais validado de todos, baseline 0.86 mais provada

SLOT FINAL 2 = s160 (FASE4v2 corpus1648-official)
   └─ receita mais recente que PASSOU no kill-switch
   └─ linhagem descorrelacionada do 086 (hedge de order-statistics)

LÓGICA: 2 linhagens 0.86 máxima diversas. Se o re-score privado
        (50% disjunto) penalizar uma, a outra segura. Hedge clássico.
```

### 7.3 Por que NÃO mais treinos

> A evidência da FASE5 (0.53) é **condenatória**. A direção de re-treino agressivo destrói o modelo. As 4 famílias de 1.0 são frágeis a qualquer fine-tuning pesado. O caminho seguro para **manter top 10%** é consolidar o 0.86 com seleção diversa, não arriscar mais regressões.

### 7.4 Alternativa de baixo risco (se quiser usar slots restantes)

```
Variance re-roll do 086: re-submeter o próprio campeão buscando a
loteria 0.87 do vLLM não-determinístico. Custo = slots. Risco = ZERO
(é o mesmo adapter provado). Já tentado antes; pode repetir.
```

---

## 📚 PARTE 8 — GLOSSÁRIO DE REGRAS INTERNAS (as "Dxx" e gates)

| Regra | O que é |
|---|---|
| **Regra 99%** | nunca submeter sem 99% de certeza de melhora |
| **GATE-87** | todo job GPU responde antes "chegamos a ≥0.87?" com P medida |
| **Kill-gate s140** | prescore valida bit>0 E eq>0, senão mata o run |
| **D7** | backslash/env em heredoc quebra (HF_HUB bug) |
| **D19** | re-dose de lr (2x) quando treino inerte |
| **D22** | blocos if-not-SMOKE nunca testados pelo smoke → bugs latentes |
| **C2 SANITY** | 10 cipher no 086 antes de tudo; se <9/10, harness errado |
| **SPRT** | early-exit estatístico (mata candidato pior cedo) |
| **paridade de motor** | prescore usa vLLM 0.17.1 (mesma do juiz oficial) |

---

## 🏁 RESUMO EXECUTIVO (1 parágrafo)

> Partimos de 0.58 (abril) e chegamos a **0.86** replicando a receita pública campeã (huikang/Modal) em ~24/abr. De lá, **dezenas de tentativas de superar 0.86** via merge/soup/re-treino TODAS travaram ou regrediram — o 0.86 é um **teto estrutural** desta arquitetura+restrição (rank≤32). A última grande aposta (FASE5 glyph+tier, lr 1e-5) **falhou catastroficamente** (s140=0.53) por esquecimento de habilidades, ensinando que LOGPROB-WATCH é métrica enganosa e que re-treino pesado destrói as famílias de 1.0. **Decisão final: consolidar 0.86 com 2 submissões de linhagens diversas (086 + s160)** para maximizar a robustez no re-score privado e garantir top 10%, em vez de arriscar mais regressões. O 0.90 não foi alcançado — a evidência mostra que exigiria fechar equation+bit sem tocar nas 4 famílias de 1.0, algo que nenhuma das ~50 tentativas conseguiu.

---

*Fim do documento. Gerado a partir do histórico real de 50 submissões Kaggle + logs de treino/prescore.*
