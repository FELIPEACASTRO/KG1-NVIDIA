# 🎯 ROADMAP — Score V1244 dentro do deadline

> ⏰ **Cronômetro** (eu atualizo este bloco toda vez que mexer no roadmap):

```
ATUALIZADO EM:   2026-06-14 20:45 BRT (23:45 UTC)
DEADLINE FINAL:  2026-06-15 20:59 BRT  (23:59 UTC)   [VERIFICADO no Kaggle CLI]
TEMPO RESTANTE:  ~24h 13min
```

- **Competição:** NVIDIA Nemotron Model Reasoning Challenge · prêmio US$ 106.388 · 4.300 times.
- **Objetivo:** score ≥ 0.87 (subir do piso 086 = 0.86) → TOP 1.
- **Regras-chave:** submissão até o deadline · 5 submits/dia (reset 00:00 UTC = 21:00 BRT) · seleção das finais exige autorização do Felipe · eval oculto greedy boxed exact-match budget 7680.

---

## ⏱️ Orçamento de tempo (cabe folgado em 24h; o aperto é não retrabalhar)
```
Treino REAL (160 steps):    ~2-3h   (o SMOKE dá o número exato)
Eval Notebook B (full947):  ~1-2h   (086 + candidato)
Seleção + submit:           ~min    (com autorização)
────────────────────────────────────
GPU total: ~4-5h  →  sobra MUITA folga. Margem p/ 1 retrabalho se preciso.
```

---

## 🗺️ FASES (status ao vivo)

```
[✅ FEITO]   F0. Solução validada: dataset 979 (triple-check), gates auditados+corrigidos,
                live-log+heartbeat+watchdog, torch 2.10 alinhado, mamba/causal via wheel.

[🔄 AGORA]   F1. SMOKE (8 steps) — rodando. Carregando o 30B (heartbeat vivo).
                PRODUTO: tempo/step real + confirma pipeline (adapter sobe em runs/<RUN_ID>/).

[⏳ NEXT]    F2. BRIEFING + GO do Felipe (treino REAL >30min = regra).
                Eu entrego: custo + ETA medido (do smoke) + P(≥0.87) honesto.

[ ]          F3. TREINO REAL (MODE='REAL', 160 steps, 4 checkpoints) → candidato.

[ ]          F4. EVAL Notebook B: candidato (final+checkpoints) vs 086 no full947 → melhor.

[ ]          F5. SELEÇÃO + SUBMIT (autorização do Felipe):
                086 (PISO 0.86 garantido) + melhor candidato V1244 (se passar do 086).
```

---

## 🛡️ Regra de ouro do deadline (anti-zerar)
```
O 086 (0.86) DEVE estar selecionado como UMA das finais cedo (piso garantido), independente
do candidato V1244 ficar pronto. Nunca chegar no deadline sem o piso travado.
```

## 🎯 Como o V1244 sobe o score (onde está o ganho — do mapa de puzzles)
```
GANHO real:  bit (classes d2/3/b — o CoT mira nelas) + equation numérico.
PROTEGER:    unit/numeral/gravity/text (086 ~100% — replay 30% não deixa esquecer).
TETO:        equation simbólico q_unseen (~13) — sub-determinado, não dá. Não gastar nisso.
```

---
*Mantido pelo Claude. O bloco ⏰ Cronômetro é atualizado a cada revisão do roadmap.*
