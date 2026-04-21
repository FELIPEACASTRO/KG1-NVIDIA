# PRE-FLIGHT CHECKLIST — KG1 Roadmap V70.5 -> V76

**Data**: 2026-04-20
**Floor**: 0.840
**Uso**: executavel em ordem antes de cada stage. Marcar [x] cada item. Qualquer [ ] -> NO-GO.

---

## GLOBAL — Antes de qualquer stage

### Budget & Infra
- [ ] Budget remanescente >= $20 (`python scripts/kg1_budget_check.py`)
- [ ] HF token valido (`huggingface-cli whoami` retorna felipesp1983)
- [ ] Kaggle creds validos (`kaggle competitions list` sem erro)
- [ ] GPU disponivel: Colab Pro A100/H100 OR HF Jobs (nao Spot)
- [ ] Slot Kaggle submit >= 2/5 reservados para dia atual
- [ ] Disk space >= 50GB livre no worktree
- [ ] Colab secrets presentes: HF_KEY, KAGGLE_KEY, KAGGLE_USERNAME

### Data & Splits
- [ ] Val split `data/val_holdout_600.jsonl` existe e tem 600 linhas exatas
- [ ] Hash val set NAO intersecta train (SHA256 check via `scripts/validate_no_leak.py`)
- [ ] Per-cat val count balanceado (min 40 per-cat)
- [ ] Dataset train >= 5000 linhas apos filtros
- [ ] Tokenizer `nvidia/NVIDIA-Nemotron-Nano-9B-v2` carregado sem warning

### Git & Rollback
- [ ] Branch atual fresh (`git status` sem arquivos pendentes criticos)
- [ ] Tag do ultimo stage verificado existe (`git tag | grep v7`)
- [ ] HF model `felipesp1983/kg1-nemotron-lora-v70-floor` downloadavel
- [ ] Ultimo adapter good salvo localmente E em HF
- [ ] Script de rollback testado em dry-run (`--dry-run` flag)

---

## V70.5 — max_length 4096 -> 8192

### Before training
- [ ] Tokenizer p99 length < 8000 (`scripts/tokenize_stats.py --max 8192`)
- [ ] Smoke 2 steps com max_length=8192 OK (VRAM peak capturado)
- [ ] VRAM peak smoke <= 75GB
- [ ] `gradient_checkpointing=True` ativo
- [ ] `flash_attention_2` carregado sem erro
- [ ] `per_device_batch=1, grad_accum=16` configurado

### During training
- [ ] Loss decrescente nos primeiros 50 steps
- [ ] Throughput >= 0.15 step/s
- [ ] VRAM estavel (delta entre step 10 e step 100 < 2GB)
- [ ] Zero OOM events em logs
- [ ] Checkpoint salvo a cada 200 steps

### After training
- [ ] Local eval 600 hold-out >= 0.843
- [ ] Per-cat acc: nenhuma regride > 0.01 vs V70
- [ ] VRAM eval peak <= 60GB
- [ ] `kg1_submission_gate.py` GO

---

## V71.1 — max-min-logprob loss

### Before training
- [ ] `pytest tests/test_maxmin_loss.py -v` 100% pass
- [ ] Smoke 20 steps em 200 samples OK
- [ ] `lr=1e-4` (halved) configurado
- [ ] `grad_clip_norm=1.0` ativo
- [ ] Fallback flag `--loss_type cross_entropy` disponivel
- [ ] Logs incluem `min_logprob` e `max_logprob` por step

### During training
- [ ] `grad_norm` max < 10.0 em todos os steps
- [ ] Zero NaN/Inf events em logs
- [ ] LR efetiva >= 1e-5 (scheduler saudavel)
- [ ] Loss step 20 < loss step 0 * 0.85
- [ ] Logs de monitor rodando (tensorboard OR wandb)

### After training
- [ ] Local eval >= 0.846
- [ ] Eval parcial (100 samples) >= baseline V70.5 - 0.01
- [ ] Per-cat: nenhuma regressao > 0.015
- [ ] `kg1_submission_gate.py` GO

---

## V71.2 — bit_manipulation pairs-of-bits CoT

### Before training
- [ ] `scripts/inspect_tokenization.py --category bit_manipulation` OK (bits separados)
- [ ] `val_split_bit_manipulation.jsonl` (120 samples) presente
- [ ] Hash val != hash train (zero leak)
- [ ] 10 samples da nova CoT review manual por 2 pessoas
- [ ] Preprocessing reversible (`space_separate_bits` func tested)
- [ ] CoT parsable por regex existente

### During training
- [ ] Loss bit_manipulation tracking separado
- [ ] Tokens/sample ratio < 1.3x original
- [ ] Zero samples com > 2 bits por token
- [ ] Checkpoint salvo a cada 200 steps

### After training
- [ ] Acc bit_manipulation >= 0.791 (80.1% - 0.01 floor)
- [ ] Eval global >= 0.852
- [ ] Zero regressao > 0.01 em cat nao-bit
- [ ] `kg1_submission_gate.py` GO

---

## V71.3 — cryptarithm 47-combo + VER

### Before training
- [ ] VER precision em 500 samples >= 0.95
- [ ] `scripts/cryptarithm_coverage.py` >= 0.85
- [ ] Hard timeout 30s per puzzle configurado
- [ ] Fallback Tong solver testado em 20 casos
- [ ] `val_cryptarithm.jsonl` (80 samples) presente e sem leak
- [ ] Zero crashes em 500 sample run

### During training
- [ ] Mean time per puzzle <= 10s
- [ ] VER precision mantido >= 0.95
- [ ] Dataset cryptarithm final >= 1500 samples

### After training
- [ ] Acc cryptarithm pos-train > acc pre-train
- [ ] Eval global >= 0.855
- [ ] Per-cat: nenhuma cat regride > 0.01
- [ ] `kg1_submission_gate.py` GO

---

## V71 — PEFT 3 fixes

### Before training
- [ ] `model.config.tie_word_embeddings` checado (True -> exclude lm_head)
- [ ] `scripts/estimate_vram.py` estimado < 72GB
- [ ] Target modules exclui `dt_proj`, `x_proj`, `A_log`, `D` (Mamba)
- [ ] Smoke PiSSA 10 steps: `grad_norm < 5`, loss step 10 < step 0 * 0.9
- [ ] 3 seeds de PiSSA init: variancia loss step 0 < 0.5
- [ ] rsLoRA alpha config validado

### During training
- [ ] VRAM peak <= 74GB
- [ ] grad_norm max <= 5.0 em todos steps
- [ ] Loss monotonic nao-crescente em rolling mean 50 steps
- [ ] Zero NaN/Inf
- [ ] Checkpoint a cada 200 steps

### After training
- [ ] Local eval >= 0.858
- [ ] Eval individual por sub-fix (ablation) registrado
- [ ] VRAM inference <= 60GB
- [ ] `kg1_submission_gate.py` GO

---

## V72 — multi-teacher datasets

### Before training
- [ ] andy279 access status verificado (se 403, skip andy279)
- [ ] kienngx cipher quality >= 70% valid (100 sample check)
- [ ] Voting: cada sample com >= 2 teachers concordando
- [ ] Dataset final >= 8000 samples
- [ ] Teacher dominance: nenhuma fonte > 60%
- [ ] Jaccard similarity cross-teacher < 0.9 (divergencia real)

### During training
- [ ] Per-category loss tracking separado
- [ ] Nenhuma categoria com < 500 samples
- [ ] Checkpoint a cada 200 steps

### After training
- [ ] Local eval >= 0.868
- [ ] Per-cat: delta >= -0.005 em cada cat
- [ ] Validation CoT parsable rate >= 0.85
- [ ] `kg1_submission_gate.py` GO

---

## V72.5 — RFT + Structured + Curriculum

### Before training
- [ ] RFT threshold sweep done, mantem >= 30% samples
- [ ] Template parse success 200 samples >= 0.95
- [ ] Curriculum scheduler validated (`scripts/curriculum_check.py`)
- [ ] 3 ablation runs (RFT/Template/Curriculum only) 200 steps each
- [ ] Flags `--no-rft`, `--no-template`, `--no-curriculum` testadas
- [ ] Loss monotonic nao-crescente por bucket curriculum

### During training
- [ ] Loss trajectory saudavel por bucket
- [ ] Template parse rate mantido em logs
- [ ] Zero NaN/Inf

### After training
- [ ] Local eval >= 0.875
- [ ] Combined eval > best ablation single
- [ ] Per-cat: delta >= -0.003
- [ ] `kg1_submission_gate.py` GO

---

## V73 — LoRA soup TIES

### Before training
- [ ] `tie_word_embeddings` checked, lm_head excluido do merge se True
- [ ] Blocklist `EXCLUDE_FROM_MERGE = ["dt_proj", "x_proj", "A_log", "D"]`
- [ ] Cada component adapter eval registrado (baseline)
- [ ] Density sweep `[0.2, 0.5, 0.7]` testado
- [ ] Pair merges (v71+v72, v71+v72.5, v72+v72.5) eval registrados
- [ ] TIES config final selecionado

### During merge (sem training)
- [ ] Zero NaN weights apos merge
- [ ] VRAM load merged adapter <= 60GB
- [ ] Density selecionada por sweep

### After merge
- [ ] Soup eval >= max(component_eval_i)
- [ ] Triple merge >= best_pair_merge
- [ ] Per-cat: nenhuma cat regride > 0.003 vs best component
- [ ] Local eval >= 0.880
- [ ] `kg1_submission_gate.py` GO

---

## V73.5 — per-family LoRA

### Before training
- [ ] Budget remaining >= $20
- [ ] Data per family >= 2000 samples
- [ ] Router implementado e testado (`family_router.py --test`)
- [ ] Router acc classification >= 0.90
- [ ] Early stopping `patience=3` configurado
- [ ] Merge interference test done (2-family merge)

### During training (3 runs paralelos OR sequenciais)
- [ ] Budget tracking a cada 30 min
- [ ] Loss per-family saudavel
- [ ] Early stop acionado se nao melhora 3x

### After training
- [ ] Cada family adapter eval >= 0.83
- [ ] Combined (router + adapters) >= V73 single adapter
- [ ] Total compute <= $18
- [ ] Local eval >= 0.885
- [ ] `kg1_submission_gate.py` GO

---

## V74 — post-hoc repair

### Before integration
- [ ] `pytest tests/test_repair.py` 100% pass
- [ ] 50 test cases por padrao regex validados
- [ ] Edge cases hex/bin/decimal testados
- [ ] Dry-run em 100 samples com diff manual

### Ablation (sem training)
- [ ] Eval `--repair on` vs `--repair off` comparado
- [ ] Zero degradation em cada categoria
- [ ] Overall delta >= +0.002 (se < 0.002, NAO integrar)

### Before submit
- [ ] Per-cat: nenhuma cat regride > 0.005
- [ ] Local eval >= 0.888
- [ ] `kg1_submission_gate.py` GO

---

## V75 — ThinkPRM verifier

### Before integration
- [ ] Multi-LoRA Kaggle test: `kaggle_multi_lora_test.ipynb` sucesso
- [ ] VRAM Kaggle T4 inference <= 15GB (com 2 LoRAs)
- [ ] Quantizacao 8bit testada (se necessario)
- [ ] Verifier val acc >= 0.85
- [ ] Inference latency < 2x baseline
- [ ] Plan B (confidence threshold) implementado como fallback

### Before training verifier
- [ ] Dataset verifier balanced (positives/negatives ~50/50)
- [ ] Val split verifier nao leaked no train

### After integration
- [ ] Ensemble eval > solver-only + 0.003
- [ ] Local eval >= 0.891
- [ ] Kaggle submit time estimate <= limit (9h)
- [ ] `kg1_submission_gate.py` GO

---

## V76 — ensemble submit

### Before submit
- [ ] Slots Kaggle restantes >= 2
- [ ] `kg1_kaggle_like_gate.py --target 0.890` >= 0.880
- [ ] Local eval >= 0.888
- [ ] Cross-val 3 folds std < 0.005
- [ ] Per-cat min acc >= 0.70
- [ ] Weighted eval combining all criteria done
- [ ] Diff vs ultimo submit >= 0.002 (nao desperdicar slot)

### Submit format
- [ ] ZIP tem 2 arquivos raiz (per memory `reference_submission_format.md`)
- [ ] `submission.zip` dentro do ZIP final
- [ ] Kaggle CLI command preparado
- [ ] Hash submission registrado

### After submit
- [ ] LB score registrado
- [ ] Se LB < 0.84: rollback plan ativado imediatamente
- [ ] Se LB >= 0.88: submit seguinte planejado (se slots restantes)
- [ ] Reserve slot 4 para rollback para V70 floor se algo quebrar
- [ ] Documentar resultado em `.learnings/`

---

## Emergency Rollback — Template Global

```bash
# 1. Check ultimo good
git tag | grep v7 | sort -V | tail -5

# 2. Checkout last known good
git checkout tags/v<LAST_GOOD>

# 3. Restore adapter
rm -rf adapter_current/
huggingface-cli download felipesp1983/kg1-nemotron-lora-v<LAST_GOOD> \
    --local-dir adapter_current/

# 4. Smoke eval para confirmar
python scripts/evaluate_lora_adapter.py \
    --adapter adapter_current/ \
    --val data/val_holdout_600.jsonl \
    --target 0.840

# 5. Se smoke passa, submit rollback (usa slot 4 reservado)
kaggle competitions submit -c nvidia-nemotron \
    -f submission_v<LAST_GOOD>.zip -m "rollback to v<LAST_GOOD>"
```

**Regra de ouro**: nunca submitar sem 99% certeza (per memory `feedback_99percent_rule.md`). Se duvida, NO-GO.
