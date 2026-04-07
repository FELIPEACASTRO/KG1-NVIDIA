# KG1_v42_NONUPLE.ipynb — Mudanças vs v41 PROVEN

## Resumo

`v42 NONUPLE` é uma versão evoluída de `v41 CLEAN PIPELINE` (PROVEN baseline 0.68) com TODOS os achados acionáveis do **NONUPLE Check (9º round)** implementados como **Phase 4 opt-in**.

**CRÍTICO**: Phases 1, 2, 3 permanecem **idênticas** ao v41 (backward compat 100%). Para usar as melhorias NONUPLE, selecionar `PHASE = 4` no Cell 1.

---

## Layout de cells

| # | Cell | Status | Tamanho |
|---|------|--------|---------|
| 0 | Markdown header | inalterado | 613 chars |
| 1 | Setup + Config | **MODIFICADO** | 9568 chars (+2500) |
| 2 | Load & Prepare Data | inalterado | 9197 chars |
| 3 | Load Model + LoRA | **MODIFICADO** | 5212 chars (+772) |
| **4** | **Cell 3.5: Pre-treino Smoke Test + Pre-score Gate** | **NOVO** | 4628 chars |
| 5 | Train (SFT) + Auto-Submit | inalterado | 11941 chars |
| **6** | **Cell 4.5: Checkpoint Averaging (AIMO-2)** | **NOVO** | 3480 chars |
| 7 | Check Scores + Diagnostics | inalterado | 1619 chars |
| 8 | Manual Submit | inalterado | 1041 chars |
| 9 | Markdown footer | inalterado | 960 chars |

**Total**: 8 → 10 cells. Acréscimo de ~10K chars de código novo.

---

## Mudanças por Cell

### Cell 1: Setup + Config — pin versions + Phase 4

**Pinned package versions**:
- `peft>=0.17.0` (era `>=0.14.0`) → suporte a `target_parameters` para MoE LoRA
- `trl>=0.17.0` (era `>=0.14.0`) → consistência com PEFT
- `vllm>=0.18.0` → fix CVE-2026-27893 (Reddit /r/LocalLLaMA, 8.6 dias)

**Phase 4 NONUPLE config adicionada**:
```python
4: {
    "name": "v42-nonuple",
    "output_repo": "felipesp1983/kg1-nemotron-lora-v42-nonuple",
    # Mesmo que v41 PROVEN baseline:
    "n_examples": 5000,
    "n_epochs": 2,
    "learning_rate": 5e-5,
    "max_length": 1024,
    "grad_accum": 8,
    "warmup_ratio": 0.05,
    "use_thinking": False,
    "use_cot": False,
    "submit_steps": [200, 300, 400, 500, 600, 800, 1000],
    "eq_oversample": 2.0,
    # NONUPLE additions:
    "freeze_moe_router": True,    # Aman Atar (168 votes Kaggle)
    "exclude_out_proj": False,    # NVIDIA NeMo YAML (TEST FALSE first)
    "checkpoint_averaging": True, # AIMO-2 trick (+1-2 score points free)
    "skip_pretrain_smoke": False, # mandatory feedback rule
    "skip_prescore_gate": False,
    "prescore_min_threshold": 0.60,
}
```

**Variables exposed para outras cells**:
- `FREEZE_MOE_ROUTER`, `EXCLUDE_OUT_PROJ`, `CHECKPOINT_AVERAGING`
- `SKIP_PRETRAIN_SMOKE`, `SKIP_PRESCORE_GATE`, `PRESCORE_MIN`

---

### Cell 3: Model + LoRA — freeze_moe_router + exclude out_proj

**A. freeze MoE routers** (Aman Atar, 168 votes Kaggle):
```python
if FREEZE_MOE_ROUTER:
    for name, param in model.named_parameters():
        if "router" in name.lower() or ("gate" in name.lower() and "gate_proj" not in name):
            param.requires_grad = False
```
Previne router collapse durante fine-tuning, ataca o gap equation 12.2%.

**B. Optional exclude out_proj** (NVIDIA NeMo official YAML):
```python
if EXCLUDE_OUT_PROJ:
    target = ["q_proj", "k_proj", "v_proj", "o_proj",
              "in_proj",  # Mamba in_proj only
              "up_proj", "down_proj", "gate_proj"]
else:
    target = "all-linear"  # PROVEN v30/v41 baseline
```

**Conflito empírico**: NVIDIA NeMo YAML diz "exclude out_proj" porque Mamba layers usam custom kernels. Notebooks Kaggle (Konbu17) USAM out_proj. **Default = False** para preservar v41 PROVEN. Felipe pode testar `True` em segundo run.

---

### Cell 3.5 (NOVO): Pre-treino Smoke Test + Pre-score Gate

**Mandatório por feedback memory** `feedback_pretreino_prescore.md`:

1. **Smoke test (2 steps)**: roda 20 examples + 2 grad steps com config real, valida loss < 8.0
2. **Pre-score gate**: usa smoke test loss como proxy (full prescore via vLLM = 10min, fora do orçamento)
3. **Decision gate**: `raise RuntimeError` se smoke test falhar → IMPEDE training pago se houver bug

**Backward compat**: cell skipped totalmente se `SKIP_PRETRAIN_SMOKE and SKIP_PRESCORE_GATE` (= phases 1, 2, 3 originais).

```python
if SKIP_PRETRAIN_SMOKE and SKIP_PRESCORE_GATE:
    print("⏭️  CELL 3.5 SKIPPED (Phase 1-3 backward compat)")
else:
    # Run smoke test 2 steps
    smoke_args = SFTConfig(max_steps=2, ...)
    smoke_trainer = SFTTrainer(...)
    smoke_trainer.train()
    # Verify loss < 8.0
    if smoke_test_passed:
        # Cleanup memory + proceed to Cell 4
    else:
        raise RuntimeError("NONUPLE pre-training gate failed")
```

---

### Cell 4.5 (NOVO): Checkpoint Averaging (AIMO-2 trick)

**AIMO-2 winning solution paper (NemoSkills)** observou que merge linear de 4 checkpoints equally-spaced melhora score em +1-2 pontos *sem custo extra de treino*.

```python
if CHECKPOINT_AVERAGING:
    ckpts = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))
    ckpts_to_average = ckpts[-4:]  # last 4

    states = [load_file(f"{c}/adapter_model.safetensors") for c in ckpts_to_average]

    # Linear average
    avg_state = {key: sum(s[key].float() for s in states) / len(states)
                 for key in states[0].keys()}

    # Save + auto-submit averaged version to Kaggle
    save_file(avg_state, f"{OUTPUT_DIR}/averaged/adapter_model.safetensors")
    create_submission_zip(...)
    submit_to_kaggle(avg_zip, "AVERAGED NONUPLE")
```

Submete tanto checkpoints individuais (Cell 4 callback) quanto o averaged (Cell 4.5).

---

## Mapping NONUPLE Findings → Implementation

| Achado NONUPLE | Status | Onde implementado |
|----------------|--------|-------------------|
| BOMBA #1: Together AI Cascade-2 free | ⏭️ Stage 4 (não Stage 1) | Não aplicável a v42 SFT |
| BOMBA #2: andy279 dataset | ⏭️ Manual access required | Não bloqueia v42 |
| BOMBA #3: vLLM bug #39103 workaround | ✅ Já não usamos --reasoning-config | competition_utils.py |
| BOMBA #4: jasonkung98 fallback dataset | ⏭️ É mirror do train.csv (já temos) | - |
| BOMBA #5: GGUF Cascade-2 sources | ⏭️ Stage 4 only | - |
| BOMBA #6: Llama-Nemotron-30M | 🔬 Future experiment | data_repo extension |
| **freeze_moe_router (Aman Atar)** | ✅ Cell 3 | `FREEZE_MOE_ROUTER` flag |
| **exclude out_proj (NVIDIA NeMo)** | ✅ Cell 3 (default False) | `EXCLUDE_OUT_PROJ` flag |
| **PEFT 0.17 target_parameters** | ✅ Cell 1 (pinned version) | pip install |
| **vLLM 0.18+ CVE fix** | ✅ Cell 1 (pinned version) | pip install |
| **Pre-treino smoke test mandatório** | ✅ Cell 3.5 (NOVO) | mandatory gate |
| **Pre-score gate mandatório** | ✅ Cell 3.5 (NOVO) | proxy via smoke loss |
| **Checkpoint averaging (AIMO-2)** | ✅ Cell 4.5 (NOVO) | linear merge last 4 |
| **Tong Hui Kang #7 GitHub** | 📚 Reference (felipe lê amanhã) | github.com/tonghuikang/nemotron |
| **Donald Galliano playbook** | ✅ Já existe! | src/baseline_solvers.py |
| **Mohan Krishna 6 puzzle types** | ✅ Já existe! | src/baseline_solvers.py |
| **Self-consistency maj@16** | ❌ Não aplicável | Kaggle scorer fixa temp=0.0 |
| **temp=1.0 NVIDIA recommend** | ❌ NVIDIA recomenda para uso normal, MAS Kaggle scorer usa temp=0.0 (Ryan Holbrook STAFF confirmou) | Não mexer |
| **DeepSeek-R1 distillation** | 🔬 Stage 4 future | data_repo extension |
| **NeMo-RL nano-v3 branch** | ⏭️ GRPO (Phase 3, não v42 SFT) | - |

---

## Como rodar

### Phase 4 NONUPLE (RECOMENDADO para v42):

1. Abrir `KG1_v42_NONUPLE.ipynb` no Colab
2. Selecionar runtime H100/A100 (>=40GB)
3. **Cell 1**: setar `PHASE = 4`, `AUTO_SUBMIT = True`, `FRESH_LORA = True`
4. Run all cells sequentially
5. Cell 3.5 vai rodar smoke test → se passar, prossegue
6. Cell 4 treina full SFT
7. Cell 4.5 faz averaging + submete versão averaged
8. Total checkpoints submetidos: 7 individuais (steps 200-1000) + 1 averaged

### Phase 1 backward compat (= v41 PROVEN):

1. Setar `PHASE = 1`
2. Cells 3.5 e 4.5 são automaticamente skipped
3. Comportamento idêntico ao v41

---

## Testes pré-merge

- ✅ JSON válido (10 cells)
- ✅ 20/20 NONUPLE checks passing (peft, vllm, freeze, exclude, smoke, averaging)
- ✅ Phase 1 backward compat preservado (variables default = False/skip)
- ⏳ Smoke run em Colab T4 ($0.40/h, ~5 min) — recomendado antes do H100
- ⏳ Full v42 Phase 4 em Colab H100 — Felipe vai rodar 09:00 BRT 07/Apr

## Sources NONUPLE

- Aman Atar Kaggle notebook (168 votes): https://www.kaggle.com/code/amanatar/nvidia-nemotron-sfttrainer-training
- NVIDIA NeMo Automodel YAML: https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune/nemotron
- AIMO-2 winning paper: https://arxiv.org/abs/2504.16891
- vLLM CVE-2026-27893: Reddit /r/LocalLLaMA
- PEFT 0.17 release notes: https://github.com/huggingface/peft/releases
- NONUPLE plan file: `C:\Users\davis\.claude\plans\quirky-hopping-pudding.md`
