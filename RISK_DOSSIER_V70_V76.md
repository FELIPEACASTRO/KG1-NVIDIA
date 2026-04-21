# RISK DOSSIER — Roadmap V70.5 -> V76

**Data**: 2026-04-20
**Floor absoluto**: 0.84 (eval local hold-out 600)
**Objetivo**: TOP 1 — score final projetado >= 0.90
**Politica**: qualquer stage que viola Go/No-Go criteria e abortado e roll-back para ultimo V(X-1) funcional.

---

## V70.5 — max_length 4096 -> 8192

### Risco Primario
Expansao do contexto dobra a memoria de atencao e pode quebrar tokenizer em amostras cipher com >4096 tokens (sequencia cortada no passado agora gera gradiente), causando OOM e/ou recompute instavel.
- **Probabilidade**: HIGH (80%)
- **Impact**: moderate (backoff 4096 mantem V70 floor)
- **Root cause**: Nemotron-Nano 9B BF16 + Mamba SSM cache cresce O(L); atencao cresce O(L^2). Em H100 80GB, peak VRAM com 4096 foi ~62GB, projetado 78-82GB em 8192.

### Pre-flight Checks
1. `python -c "from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('nvidia/NVIDIA-Nemotron-Nano-9B-v2'); lens=[len(tok(r['text']).input_ids) for r in load_ds()[:500]]; print(f'p95={np.percentile(lens,95)}, p99={np.percentile(lens,99)}, max={max(lens)}')` — exigir p99 < 8000.
2. Smoke train 2 steps com `max_length=8192, per_device_batch=1, grad_accum=16` e capturar `torch.cuda.max_memory_allocated()/1e9`.
3. Rodar `python scripts/validate_tokenization.py --max_length 8192 --sample 200` e verificar zero truncamentos nas 20 amostras mais longas.
4. Confirmar `gradient_checkpointing_enable(use_reentrant=False)` ativo no modelo (inspect via `model.is_gradient_checkpointing`).
5. Validar que `attn_implementation="flash_attention_2"` carregou sem warning (grep log: `FlashAttention-2 not available` deve retornar 0 hits).

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] VRAM peak (smoke 2 steps) <= 75.0 GB
- [ ] Step 1 loss finito (nao-NaN, < 10.0)
- [ ] Step 2 loss < step 1 loss * 1.05 (sem explosao)
- [ ] p99 token length < max_length (zero truncation warnings)
- [ ] Throughput >= 0.15 step/s (se < 0.1, custo extrapola budget)

### Mitigation Actions
- If VRAM peak > 75GB: reduzir `per_device_batch=1, grad_accum=32`, reativar checkpointing.
- If p99 > 8000: deixar `max_length=6144` (intermediate).
- If throughput < 0.1 step/s: manter 4096 e avancar para V71.1 sem essa mudanca.
- **Kill switch**: peak VRAM > 79GB OU loss NaN em 2 steps -> rollback imediato a V70.

### Rollback Procedure
```bash
git checkout tags/v70-floor
rm -rf adapter_v70_5/
huggingface-cli download felipesp1983/kg1-nemotron-lora-v70-floor \
    --local-dir adapter_v70/ --revision main
```

### Expected vs Abort
Expected delta: +0.004 (coverage de long-context cases)
Abort if: local_eval < 0.840 - 0.005 = 0.835
Submit OK if: local_eval >= 0.840 + 0.003 = 0.843

---

## V71.1 — max-min-logprob loss

### Risco Primario
Trocar cross-entropy por max-min-logprob (pick hardest token) pode gerar gradientes de magnitude 10-100x maiores em tokens raros, levando a NaN/Inf no optimizer e collapse do adapter em <50 steps.
- **Probabilidade**: HIGH (75%)
- **Impact**: catastrophic (adapter inutil, perda de 4-6h compute)
- **Root cause**: max-min-logprob e nao-convexa; sem warm-up, LR alto (2e-4) causa explosion.

### Pre-flight Checks
1. Validar implementacao da loss em unit test: `pytest tests/test_maxmin_loss.py -v` — exigir 100% pass.
2. Rodar 20 steps em subset de 200 samples com `lr=1e-4 (half)` e checar `grad_norm` < 10.0 a cada step.
3. Adicionar `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` no training_step.
4. Flag de fallback: `--loss_type cross_entropy` como fallback automatico se NaN detectado.
5. Log `min_logprob` e `max_logprob` por step; se range > 20.0, ativar smoothing `loss = 0.5*maxmin + 0.5*ce`.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Step 20 loss < step 0 loss * 0.85
- [ ] grad_norm_max (steps 1-20) <= 10.0
- [ ] Zero NaN/Inf detections em logs
- [ ] LR efetiva >= 1e-5 (scheduler nao colapsou)
- [ ] Eval parcial (100 samples) >= baseline V70.5 - 0.01

### Mitigation Actions
- If grad_norm > 10 em 3 steps consecutivos: reduzir LR para 5e-5 e ativar warmup 100 steps.
- If NaN em qualquer step: abortar run, fallback para `cross_entropy` com LR original.
- If eval parcial < baseline - 0.015: ativar composite loss `0.5*ce + 0.5*maxmin`.
- **Kill switch**: 2 NaN events OR eval parcial < baseline - 0.02 -> rollback V70.5.

### Rollback Procedure
```bash
git checkout v70.5-verified
rm -rf adapter_v71_1/
# reuse V70.5 adapter para stages seguintes
cp -r adapter_v70_5/ adapter_current/
```

### Expected vs Abort
Expected delta: +0.006
Abort if: local_eval < 0.843 - 0.005 = 0.838
Submit OK if: local_eval >= 0.843 + 0.003 = 0.846

---

## V71.2 — bit_manipulation "pairs of bits" CoT

### Risco Primario
Categoria bit_manipulation ja esta em 80.1% acc (ponto forte). Qualquer alteracao na CoT pode regredir: tokenizer fragmenta bits em sub-tokens se nao houver espacos, e CoT nova pode introduzir ambiguidade.
- **Probabilidade**: MED (50%)
- **Impact**: moderate (regressao 0.005-0.01 em cat forte)
- **Root cause**: BPE tokenizer mescla `01011` em 1-2 tokens; modelo perde estrutura bit-a-bit.

### Pre-flight Checks
1. Rodar `python scripts/inspect_tokenization.py --category bit_manipulation --n 50` — exigir que cada bit vire token separado apos pre-processamento.
2. Comparar acc antes/depois em split dedicado: `val_split_bit_manipulation.jsonl` (120 amostras).
3. Pre-inserir espacos: `text = re.sub(r'([01])(?=[01])', r'\1 ', text)` — validar reversivel.
4. Gerar 10 samples sinteticos da nova CoT e fazer manual review (2 humanos minimo).
5. Garantir que `val_split_bit_manipulation.jsonl` NAO foi exposto ao treino (hash check vs train set).

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Acc bit_manipulation pos-train >= 0.791 (80.1% - 0.01 floor)
- [ ] Per-token len media < len media * 1.3 (sem explosao de tokens)
- [ ] Zero amostras com mais de 2 bits em mesmo token
- [ ] Eval global >= baseline - 0.003
- [ ] CoT final parsable por regex existente em src/baseline_solvers.py

### Mitigation Actions
- If acc bit < 0.791: reverter preprocessing, usar CoT original V70.5.
- If tokens/sample > 1.3x: reduzir verbosity da CoT nova.
- If CoT nao parsable: adicionar normalizacao no teacher_cot.py antes do treino.
- **Kill switch**: acc bit_manipulation < 0.78 em eval completo -> rollback para V71.1.

### Rollback Procedure
```bash
git checkout v71.1-verified
rm -rf adapter_v71_2/
python scripts/regenerate_cot.py --category bit_manipulation --version v71.1
```

### Expected vs Abort
Expected delta: +0.008
Abort if: local_eval < 0.849 - 0.005 = 0.844
Submit OK if: local_eval >= 0.849 + 0.003 = 0.852

---

## V71.3 — cryptarithm 47-combo + VER

### Risco Primario
Espaco de combinacoes digito-letra e 10! = 3.6M. Cobrir 47 template combos pode nao capturar formato especifico da competicao. VER pode classificar mapping ambiguo como valido, poluindo dataset.
- **Probabilidade**: HIGH (70%)
- **Impact**: moderate (cat cryptarithm: ~3% global, regressao max -0.01)
- **Root cause**: 47 << 3.6M; busca heuristica pode preferir solucao errada quando >1 mapping satisfaz.

### Pre-flight Checks
1. Rodar VER em 500 cryptarithm do train set: exigir `precision >= 0.95` (95% dos "valid" tem exatamente 1 solution).
2. Hard timeout `per_puzzle=30s` no solver; amostras que time-out vao para fallback Tong.
3. Cobrir os 47 combos + fallback: `python scripts/cryptarithm_coverage.py` deve retornar coverage >= 0.85.
4. Checar que VER retorna `None` (nao crash) quando ambiguo, testado em 20 casos ambigue manualmente.
5. Split de validacao `val_cryptarithm.jsonl` (80 samples) nunca usado no train.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] VER precision em 500 samples >= 0.95
- [ ] Coverage dos 47 combos em val set >= 0.75
- [ ] Mean time per puzzle (solver) <= 10s
- [ ] Zero crashes em 500 samples
- [ ] Acc cryptarithm pos-train > acc pre-train (sem regressao)

### Mitigation Actions
- If VER precision < 0.95: adicionar validacao extra (re-solve com SAT solver).
- If coverage < 0.75: fallback para `src/baseline_solvers.py::solve_cryptarithm_tong` em casos nao cobertos.
- If time > 10s: reduzir busca a 30s timeout e cache em LRU.
- **Kill switch**: regressao cryptarithm > 0.02 -> rollback V71.2.

### Rollback Procedure
```bash
git checkout v71.2-verified
rm -rf adapter_v71_3/
python scripts/regenerate_cryptarithm.py --solver tong_simple
```

### Expected vs Abort
Expected delta: +0.005
Abort if: local_eval < 0.852 - 0.005 = 0.847
Submit OK if: local_eval >= 0.852 + 0.003 = 0.855

---

## V71 — PEFT 3 fixes (rsLoRA + PiSSA + modules_to_save)

### Risco Primario
Tres mudancas simultaneas no PEFT config sao de alto risco:
(1) PiSSA init faz SVD de pesos originais, instavel se rank > 32;
(2) `modules_to_save=["lm_head"]` triplica VRAM (lm_head e 128k * 4096 * 2bytes = ~1GB * 3 = 3GB);
(3) rsLoRA com alpha scaling pode conflitar com camadas Mamba (dt_proj).
- **Probabilidade**: HIGH (85% de pelo menos 1 falhar)
- **Impact**: catastrophic se PiSSA explode; moderate se apenas lm_head VRAM estoura.
- **Root cause**: Nemotron-Nano 9B usa tied embeddings (input_emb == lm_head.weight). `modules_to_save` duplica isso.

### Pre-flight Checks
1. Inspecionar `model.config.tie_word_embeddings` — se True, NAO adicionar lm_head em `modules_to_save`.
2. Smoke run PiSSA isolado: 10 steps, `rank=16`, checar `grad_norm < 5`.
3. VRAM estimate: `python scripts/estimate_vram.py --rank 16 --modules_to_save lm_head` — exigir < 72GB estimado.
4. Validar rsLoRA nao toca `dt_proj` (Mamba state projection) — verificar `target_modules` explicito.
5. Testar init PiSSA SVD em 3 seeds; variancia de loss step 0 entre seeds < 0.5.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] tie_word_embeddings validado (True -> exclude lm_head)
- [ ] VRAM peak smoke <= 74GB
- [ ] PiSSA step 10 loss < step 0 loss * 0.9 (descendo)
- [ ] grad_norm max (steps 1-10) <= 5.0
- [ ] Sem `dt_proj` em target_modules (grep config)

### Mitigation Actions
- If PiSSA step 10 loss >= step 0 loss: reverter para init LoRA normal (revoke PiSSA).
- If VRAM > 74GB: remover `modules_to_save` (lm_head stays frozen).
- If grad_norm > 5: reduzir LR 2x + warmup 200 steps.
- If Mamba state explode: excluir `dt_proj`, `x_proj` de target_modules.
- **Kill switch**: loss NaN OR VRAM > 78GB em 10 steps -> rollback V71.3.

### Rollback Procedure
```bash
git checkout v71.3-verified
rm -rf adapter_v71/
# remove peft config modifications
git checkout v71.3 -- scripts/train_peft_config.py
```

### Expected vs Abort
Expected delta: +0.010 (acumulado 3 fixes)
Abort if: local_eval < 0.855 - 0.005 = 0.850
Submit OK if: local_eval >= 0.855 + 0.003 = 0.858

---

## V72 — multi-teacher datasets

### Risco Primario
andy279 dataset esta em status "denied access" (HF 403). kienngx tem qualidade CoT variavel (cipher lixo, per memory). Teacher bias pode introduzir padroes que nao generalizam.
- **Probabilidade**: MED (55%)
- **Impact**: moderate (usar so DeepSeek-V3 e fallback viavel)
- **Root cause**: Ensemble nao-ponderado poisoning-vulnerable a teacher de baixa qualidade.

### Pre-flight Checks
1. Verificar acesso andy279: `huggingface-cli download andy279/kg1-data --dry-run` — se 403, skip.
2. Validar qualidade kienngx em 100 samples cipher com `src/teacher_cot.py::validate_cot` — exigir >= 70% valid.
3. Voting ensemble: cada sample precisa de >=2 teachers concordando, se nao, descartar ou usar DeepSeek-V3 sozinho.
4. Dataset final stats: print `len(ds), category_distribution, teacher_source_distribution` — exigir no teacher com > 60% share.
5. Dedup cross-teacher: checar `jaccard_similarity` entre CoTs de mesmo problema — < 0.9 significa divergencia real.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Dataset final >= 8000 samples
- [ ] Nenhuma categoria com < 500 samples
- [ ] Teacher dominance: no source > 60%
- [ ] Validation CoT parsable rate >= 0.85
- [ ] Ensemble agreement >= 2 teachers em >= 70% samples

### Mitigation Actions
- If andy279 denied: usar hierarquia DeepSeek-V3 > kienngx > Gemini, sem andy279.
- If kienngx cipher invalido > 30%: excluir cipher de kienngx, usar so DeepSeek.
- If agreement < 70%: aumentar threshold para voting (precisa 3 teachers em categorias criticas).
- **Kill switch**: dataset < 5000 samples apos filtros -> rollback V71.

### Rollback Procedure
```bash
git checkout v71-verified
rm -rf data/multi_teacher/
# reuse V71 dataset
cp -r data/v71_final/ data/current/
```

### Expected vs Abort
Expected delta: +0.008 (diversity gain)
Abort if: local_eval < 0.865 - 0.005 = 0.860
Submit OK if: local_eval >= 0.865 + 0.003 = 0.868

---

## V72.5 — RFT + Structured + Curriculum

### Risco Primario
3 sub-fixes interagindo: RFT filtra amostras < threshold (pode cortar >80% do dataset se mal-calibrado); template rigido engessa geracao; curriculum order errada pode hurtar (easy-first vs hard-first).
- **Probabilidade**: HIGH (70% pelo menos 1 sub-fix falha)
- **Impact**: moderate (cada sub-fix e ablatable)
- **Root cause**: RFT reward tipicamente < 0.3, threshold padrao deixa 20-30% dos dados.

### Pre-flight Checks
1. RFT threshold sweep: rodar reward em 1000 samples, plotar histograma; ajustar threshold para manter >= 30% dos dados.
2. Template Pydantic schema validado em 200 samples: `json.loads(template.format(**sample))` deve succeed 100%.
3. Curriculum scheduler: verificar distribuicao por step (easy 0-30%, medium 30-70%, hard 70-100%) via `curriculum_schedule_check.py`.
4. Ablation prep: 3 runs separados (RFT-only, Structured-only, Curriculum-only) com 200 steps cada para isolar efeitos.
5. Rollback plan por sub-fix: flags `--no-rft`, `--no-template`, `--no-curriculum` funcionais.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] RFT threshold mantem >= 30% dos samples
- [ ] Template parse success >= 0.95
- [ ] Curriculum progression smooth (loss monotonic nao-crescente por bucket)
- [ ] Ablation: cada sub-fix isolado >= baseline - 0.003
- [ ] Combined eval >= baseline V72 + 0.002

### Mitigation Actions
- If RFT corta > 70%: relax threshold para p40 (percentile 40 da reward distribution).
- If template parse < 0.95: adicionar tolerancia fuzzy (regex fallback).
- If curriculum hurt: trocar para anti-curriculum (hard-first) ou revert uniform sampling.
- **Kill switch**: eval combined < V72 baseline -> rollback V72.

### Rollback Procedure
```bash
git checkout v72-verified
# por sub-fix:
python scripts/train.py --no-rft --no-template --no-curriculum
```

### Expected vs Abort
Expected delta: +0.007
Abort if: local_eval < 0.872 - 0.005 = 0.867
Submit OK if: local_eval >= 0.872 + 0.003 = 0.875

---

## V73 — LoRA soup TIES merge

### Risco Primario
TIES merge assume linearidade entre adapters, mas Mamba `dt_proj` governa state dinamico; merge de dt_proj destroi trajetoria aprendida. Se `tie_word_embeddings=True`, merge de lm_head e tecnicamente invalido.
- **Probabilidade**: HIGH (80%)
- **Impact**: catastrophic (soup pode ser inferior ao melhor componente sozinho)
- **Root cause**: TIES usa magnitude pruning + sign election; Mamba SSM nao respeita essas assuncoes.

### Pre-flight Checks
1. Validar `model.config.tie_word_embeddings` e excluir lm_head do merge se True.
2. Build blocklist: `EXCLUDE_FROM_MERGE = ["dt_proj", "x_proj", "A_log", "D"]` — verificar via `peft_config.target_modules`.
3. Ablation per combination: mergear pares (v71+v72, v71+v72.5, v72+v72.5) e single best antes de triple merge.
4. Pre-merge eval cada adapter isolado, registrar baseline individual.
5. Magnitude pruning threshold: testar `density=[0.2, 0.5, 0.7]`, escolher melhor via small val.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Soup eval >= max(component_eval_i) for i in components
- [ ] Nenhum dt_proj em merge weights
- [ ] tie_word_embeddings check passed
- [ ] Ablation: cada pair merge >= best_single - 0.003
- [ ] Triple merge density otima selecionada por sweep

### Mitigation Actions
- If soup < best_single: reverter para best_single adapter.
- If Mamba state break: excluir todos Mamba-specific modules do merge.
- If lm_head merge invalido: forcar tied behavior manualmente.
- **Kill switch**: triple merge < best_pair -> usar best_pair como V73.

### Rollback Procedure
```bash
git checkout v72.5-verified
rm -rf adapter_v73_soup/
# fallback: usar best single adapter
cp -r adapter_v72.5/ adapter_v73/
```

### Expected vs Abort
Expected delta: +0.005
Abort if: local_eval < 0.877 - 0.005 = 0.872
Submit OK if: local_eval >= 0.877 + 0.003 = 0.880

---

## V73.5 — per-family LoRA

### Risco Primario
Treinar 3 adapters (per family: arithmetic, logic, combinatorics) triplica compute (~$18 de $200). Cada adapter pode underfit (menos dados por family ~2500). Merge entre familias pode interferir.
- **Probabilidade**: MED (50%)
- **Impact**: moderate (budget risk primario)
- **Root cause**: Data por family ~2500 pode ser insuficiente para rank 16.

### Pre-flight Checks
1. Budget hard cap: `kg1_budget_check.py --remaining_budget 20` — abort se < $20.
2. Per-family data count: exigir >= 2000 samples por family (caso contrario, merge duas families).
3. Early stopping por-family: `patience=3` evals, se nao melhorar 3x consecutivos, parar.
4. Router implementado: `python scripts/family_router.py --test` — exigir acc classificacao family >= 0.90.
5. Merge interference test: treinar 2 families, mergear, eval — deve manter >= 0.003 de cada individual.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Budget remaining >= $20 antes de start
- [ ] Router acc >= 0.90
- [ ] Cada family adapter eval individual >= 0.83
- [ ] Combined (router + adapters) eval >= single-adapter V73
- [ ] Total compute spent <= $18

### Mitigation Actions
- If budget < $20: skip V73.5, prosseguir V74.
- If router acc < 0.90: usar oracle routing (gt category) como teto; V73.5 vira opcional.
- If family individual < 0.83: merger duas families menores.
- **Kill switch**: budget overrun > $20 OR router acc < 0.85 -> skip V73.5.

### Rollback Procedure
```bash
git checkout v73-verified
rm -rf adapter_v73_5_family_*/
# usar V73 single adapter
cp -r adapter_v73_soup/ adapter_current/
```

### Expected vs Abort
Expected delta: +0.006
Abort if: local_eval < 0.882 - 0.005 = 0.877
Submit OK if: local_eval >= 0.882 + 0.003 = 0.885

---

## V74 — post-hoc repair (regex normalizer)

### Risco Primario
Regex de normalizacao pode sobre-normalizar: casos ja corretos sao alterados e quebrados (e.g., `0x1F` virando `31`, quando submission espera `0x1F`).
- **Probabilidade**: MED (40%)
- **Impact**: minor (ablation on/off simples)
- **Root cause**: regex nao contexto-consciente; pattern match em string pode hit false positive.

### Pre-flight Checks
1. Ablation on/off: rodar eval com `--repair on` e `--repair off` em hold-out, comparar.
2. Zero degradation check: para cada categoria, `acc_with_repair >= acc_without_repair`.
3. Test cases unitarios: 50 casos de cada padrao regex, exigir 100% pass em `tests/test_repair.py`.
4. Edge cases: numeros hex/bin/decimal coexistindo, validar regex nao confunde.
5. Dry-run em 100 samples submission-ready, diff manual amostragem de 20.

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Eval com repair >= eval sem repair (em cada categoria)
- [ ] Zero test failures em `tests/test_repair.py`
- [ ] Overall delta >= +0.002 (caso contrario nao vale adicionar)
- [ ] Nenhuma categoria regride (delta_cat >= -0.002)
- [ ] Edge case tests pass 100%

### Mitigation Actions
- If delta global < 0.002: desabilitar repair, manter V73.5 como final.
- If alguma cat regride: aplicar repair apenas para cats onde ganha.
- If edge case fail: adicionar guard clause antes do regex.
- **Kill switch**: qualquer categoria com regressao > 0.005 -> disable repair.

### Rollback Procedure
```bash
# Simples: flag off
python scripts/submit_kaggle.py --no-post-hoc-repair
```

### Expected vs Abort
Expected delta: +0.003
Abort if: local_eval < 0.885 - 0.005 = 0.880
Submit OK if: local_eval >= 0.885 + 0.003 = 0.888

---

## V75 — ThinkPRM verifier

### Risco Primario
Kaggle kernel tem restricao de memoria para multi-LoRA (solver LoRA + verifier LoRA carregados juntos). Se nao suportar, fatal. Verifier mal-treinado gera false positives que hurt o ensemble.
- **Probabilidade**: HIGH (90% que Kaggle tem issue)
- **Impact**: catastrophic se multi-LoRA falha (backoff total para V74)
- **Root cause**: Kaggle T4 = 16GB VRAM. Modelo 9B BF16 = 18GB. Sem quantizacao, impossivel.

### Pre-flight Checks
1. VALIDAR multi-LoRA em Kaggle kernel: teste isolado `kaggle_multi_lora_test.ipynb` com adapter_solver + adapter_verifier carregados.
2. Fallback tier S: `confidence threshold` simples (sem verifier) — implementar como Plan B.
3. Quantizacao: testar `load_in_8bit=True` + 2 LoRAs — se VRAM < 14GB, OK.
4. Verifier treino: acc em val set >= 0.85 antes de integrar ensemble.
5. Latency check: inferencia com verifier <= 2x baseline (submit time limit Kaggle).

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Multi-LoRA load success em Kaggle T4 (verificado)
- [ ] VRAM total em inference <= 15GB
- [ ] Verifier val acc >= 0.85
- [ ] Inference latency < 2x baseline
- [ ] Ensemble eval > solver-only + 0.003

### Mitigation Actions
- If multi-LoRA falha Kaggle: usar confidence threshold (Plan B) — sem verifier.
- If VRAM > 15GB: quantizacao 4bit forcada.
- If verifier acc < 0.85: treinar mais dados OR usar Plan B.
- **Kill switch**: Kaggle multi-LoRA impossivel -> skip V75 inteiramente.

### Rollback Procedure
```bash
git checkout v74-verified
rm -rf adapter_verifier/
# ativar Plan B
python scripts/submit_kaggle.py --confidence-threshold 0.7
```

### Expected vs Abort
Expected delta: +0.010 (verifier) ou +0.003 (Plan B confidence)
Abort if: local_eval < 0.888 - 0.005 = 0.883
Submit OK if: local_eval >= 0.888 + 0.003 = 0.891

---

## V76 — ensemble submit

### Risco Primario
Overfit na metrica de validacao local (hold-out 600). LB (private) tem variancia; selecionar submit by local_eval pode underperform.
- **Probabilidade**: MED (45%)
- **Impact**: moderate (reserve 2/5 submits permite correcao)
- **Root cause**: Val hold-out 600 samples -> intervalo confianca +/- 0.008 (95%). Diferenca <0.005 pode ser noise.

### Pre-flight Checks
1. Submit budget: verificar reservas >= 2/5 do dia — nunca gastar ultima slot em ensemble nao-testado.
2. Weighted eval: multiplos criterios (local_eval 70% + per-cat min 30%).
3. Cross-val: rodar eval em 3 folds distintos, exigir std < 0.005.
4. Kaggle-like gate: `kg1_kaggle_like_gate.py --target 0.890` — se < target - 0.01, nao submit.
5. Diff check vs ultimo submit: nao submitar se diff de candidato < 0.002 de submit anterior (waste de slot).

### Go/No-Go Criteria (MATHEMATICAL)
Proceed if ALL:
- [ ] Slots restantes >= 2
- [ ] Kaggle-like gate >= 0.880
- [ ] Local eval >= 0.888
- [ ] Cross-val std < 0.005
- [ ] Per-cat min >= 0.70 (nenhuma catastrophically baixa)

### Mitigation Actions
- If cross-val std > 0.005: nao confiavel, rodar mais seeds.
- If Kaggle-like gate < 0.880: volta para V75 ou V74 best.
- If LB < local: reserve slot 4 para best known good (V74 ou V75 confirmado).
- **Kill switch**: se submit 1 de 2 der LB < 0.84 -> rollback imediato para last known good.

### Rollback Procedure
```bash
# Submit ultimo known good via last working submit:
huggingface-cli download felipesp1983/kg1-submission-v70-floor
kaggle competitions submit -c nvidia-nemotron -f submission_v70.zip -m "rollback V70"
```

### Expected vs Abort
Expected delta: LB >= 0.90 (se tudo alinhar)
Abort if: LB < 0.84 (floor)
Submit OK if: LB >= 0.88 (progresso mesmo que sub-target)

---

## Sumario de Thresholds por Stage

| Stage | Baseline | Expected | Floor | Abort if < | Submit OK if >= |
|-------|----------|----------|-------|------------|-----------------|
| V70   | 0.840    | -        | 0.840 | 0.835      | 0.843           |
| V70.5 | 0.840    | +0.004   | 0.840 | 0.835      | 0.843           |
| V71.1 | 0.843    | +0.006   | 0.840 | 0.838      | 0.846           |
| V71.2 | 0.849    | +0.008   | 0.840 | 0.844      | 0.852           |
| V71.3 | 0.852    | +0.005   | 0.840 | 0.847      | 0.855           |
| V71   | 0.855    | +0.010   | 0.840 | 0.850      | 0.858           |
| V72   | 0.865    | +0.008   | 0.840 | 0.860      | 0.868           |
| V72.5 | 0.872    | +0.007   | 0.840 | 0.867      | 0.875           |
| V73   | 0.877    | +0.005   | 0.840 | 0.872      | 0.880           |
| V73.5 | 0.882    | +0.006   | 0.840 | 0.877      | 0.885           |
| V74   | 0.885    | +0.003   | 0.840 | 0.880      | 0.888           |
| V75   | 0.888    | +0.010   | 0.840 | 0.883      | 0.891           |
| V76   | 0.891    | -        | 0.840 | LB<0.840   | LB>=0.880       |

**Regra global**: qualquer stage que falhe Go/No-Go faz rollback para o anterior verificado. Floor absoluto 0.840 nunca deve ser violado em submit.
