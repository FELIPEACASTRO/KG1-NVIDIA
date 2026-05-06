# KG1 NVIDIA Nemotron - V214 Probe/Solver Update - 2026-05-06

## Status

V213 foi executado contra os raw outputs reais do V194 encontrados no Google Drive.

Fonte Drive:

- Pasta: `v194_baseline_eval`
- `v194_baseline_predictions.csv`
- `v194_baseline_per_task.csv`
- `v194_baseline_eval_report.json`

Copias locais:

- `artifacts/drive_exports/v194_baseline_predictions.csv`
- `artifacts/drive_exports/v194_baseline_per_task.csv`
- `artifacts/drive_exports/v194_baseline_eval_report.json`

## Boxed Rewrite Probe

Script:

- `scripts/boxed_rewrite_probe.py`

Artefatos:

- `artifacts/boxed_rewrite_probe_v194_2026-05-06/v194_boxed_rewrite_probe_rows.csv`
- `artifacts/boxed_rewrite_probe_v194_2026-05-06/v194_boxed_rewrite_probe_weak_errors.csv`
- `artifacts/boxed_rewrite_probe_v194_2026-05-06/v194_boxed_rewrite_probe_extractor_disagreement.csv`
- `artifacts/boxed_rewrite_probe_v194_2026-05-06/v194_boxed_rewrite_probe_summary.json`
- `artifacts/boxed_rewrite_probe_v194_2026-05-06/v194_boxed_rewrite_probe_decision.md`

Resultado reproduzido:

- Full: `822/947 = 0.868004`
- Weak: `190/315`
- Strong: `632/632`
- `bit_manipulation`: `135/160`
- `equation_transform`: `55/155`

Buckets dos 125 erros weak:

- `ALGEBRA_MANIP`: `100`
- `ARITHM_BOUNDARY`: `24`
- `LOOP_TRUNC`: `1`
- `FORMAT_EXTRACT`: `0`

Conclusao:

- Parser/formato nao e gargalo material.
- Safe extractor recovery: `0/125`.
- Semantic tail recovery apos endurecimento anti-falso-positivo: `0/125`.
- Decisao: `reasoning_first_solvers_dataset`.

Implicacao:

- Nao fazer format-first.
- Nao treinar template/boxed como branch principal.
- Prosseguir para solvers/verificadores e dados verificados.

## Legacy Solver Probe

Script:

- `scripts/legacy_solver_probe.py`

Artefatos:

- `artifacts/legacy_solver_probe_v194_2026-05-06/legacy_solver_probe_rows.csv`
- `artifacts/legacy_solver_probe_v194_2026-05-06/legacy_solver_probe_v194_error_gains.csv`
- `artifacts/legacy_solver_probe_v194_2026-05-06/legacy_solver_probe_v194_success_losses.csv`
- `artifacts/legacy_solver_probe_v194_2026-05-06/legacy_solver_probe_summary.json`
- `artifacts/legacy_solver_probe_v194_2026-05-06/legacy_solver_probe_decision.md`

Resultado:

- `bit_manipulation`:
  - V194: `135/160`
  - solver legado: `159/160`
  - ganhos em erros V194: `24`
  - perdas em acertos V194: `0`
  - decisao: `promote_as_verified_fix_source`
- `equation_transform`:
  - V194: `55/155`
  - solver legado: `57/155`
  - ganhos em erros V194: `3`
  - perdas em acertos V194: `1`
  - decisao: `diagnostic_only`

## Roadmap Atualizado

### Prioridade 1: bit_manipulation verified fixes

Usar solver bit legado como fonte/verificador para construir fixes, nao como mecanismo de submissao.

Proximos passos:

1. Extrair os 24 ids de `bit_manipulation` onde V194 erra e solver acerta.
2. Gerar completions curtas no estilo V194 para esses 24 prompts.
3. Validar:
   - exatamente uma `\boxed{...}`;
   - 8 bits exatos;
   - sem texto apos `\boxed{...}`;
   - sem loop;
   - completion curta.
4. Criar replay forte a partir dos sucessos V194:
   - strong success replay;
   - weak success replay de bit;
   - nao incluir equation_transform ainda como fix massivo.
5. Antes de treino, criar dataset preview e validador.

### Prioridade 2: equation_transform solver audit

O solver legado nao e confiavel para equation_transform como fonte ampla.

Proximos passos:

1. Manter os 3 ganhos como diagnostico, nao como dataset principal.
2. Auditar os 97 erros onde V194 e solver falham.
3. Separar subtipos:
   - numeric operator;
   - symbolic/operator-position;
   - punctuation/braces;
   - sign-only;
   - ambiguous.
4. Construir solver/verificador incremental apenas para subclusters com regra clara.

### Prioridade 3: training apenas depois do dataset gate

Treino candidato continua bloqueado ate:

- dataset verificado gerado;
- replay forte selecionado;
- validator passar;
- adapter_config V194 auditado;
- tokenizer/template auditado.

Mix inicial sugerido apos dataset gate:

- `>=60%` strong-success replay;
- `20-25%` weak-success replay;
- `15-20%` verified fixes;
- neste primeiro ciclo, fixes devem favorecer `bit_manipulation`, porque ha evidencia de `24/25` ganhos sem perdas.

## Gates Mantidos

- Sem Kaggle submit sem aprovacao humana.
- Sem treino antes de validator/dataset gate.
- Candidate local precisa superar V194:
  - minimo review: `>=825/947`;
  - strict candidate: `>=828/947`;
  - preferido: `>=830/947`;
  - weak preferred: `>=195/315`;
  - weak strict: `>=198/315`;
  - strong default: `632/632`.

## Proxima Acao Operacional

Criar o dataset preview V214 para `bit_manipulation`:

- `v214_bit_solver_fix_candidates.csv`
- `v214_bit_fix_training_preview.jsonl`
- `v214_replay_pool_manifest.json`
- `v214_dataset_validation_report.json`

Nao treinar ainda.

## Execucao Do Dataset Preview

Script:

- `scripts/build_v214_bit_fix_preview.py`

Artefatos:

- `artifacts/v214_bit_fix_preview_2026-05-06/v214_bit_solver_fix_candidates.csv`
- `artifacts/v214_bit_fix_preview_2026-05-06/v214_bit_fix_forensic_not_train.jsonl`
- `artifacts/v214_bit_fix_preview_2026-05-06/v214_bit_replay_preview_train_allowed.jsonl`
- `artifacts/v214_bit_fix_preview_2026-05-06/v214_replay_pool_manifest.json`
- `artifacts/v214_bit_fix_preview_2026-05-06/v214_dataset_validation_report.json`

Resultado:

- `24` referencias forenses de bit verificadas.
- Essas `24` referencias sao V194 local validation rows e estao marcadas como `train_allowed=false`.
- `160` exemplos de replay bit vindos de `data/v206/v206_curated_train.jsonl` foram selecionados e validados.
- O builder excluiu overlap contra os `947` rows V194 por assinatura `prompt+answer`.
- Foram detectados e removidos `8` overlaps de validacao no primeiro pool.
- Overlap final do replay selecionado contra V194: `0`.
- O pool fonte `data/v206/v206_curated_train.jsonl` contem:
  - `2050` bit rows;
  - `2109` equation rows;
  - `632+` strong/replay rows distribuidos nas familias fortes.

Decisao:

- Nao treinar nos 24 erros V194 diretamente, para nao contaminar o gate local.
- Usar esses 24 apenas para taxonomia/template/pattern mining.
- Para treino, usar fonte curada nao-val ou sinteticos solver-verificados.

## Auditoria Adapter V194

Artefato:

- `artifacts/V194_ADAPTER_AUDIT_2026-05-06.md`

Resultado:

- `r=32`
- `lora_alpha=32`
- `lora_dropout=0.0`
- `modules_to_save=null`
- `target_modules` inclui `lm_head`
- `target_parameters` inclui MoE experts:
  - `mlp.experts.gate_up_proj`
  - `mlp.experts.down_proj`

Decisao:

- Continuation de V194 e possivel, mas deve preservar config real.
- `lm_head` em V194 nao rejeita o baseline, pois ele reproduz `822/947`, mas precisa ser documentado como excecao/risco.
- Nao fazer surgery/strip de `lm_head` sem solve-rate gate completo.

## Micro-Replay Candidate V214

Script:

- `scripts/build_v214_micro_replay_candidate.py`

Artefatos:

- `data/v214/v214_micro_replay_candidate.jsonl`
- `data/v214/v214_micro_replay_candidate_manifest.json`

Resultado:

- `880` rows.
- `880/880` verified.
- `880/880` single-boxed.
- Overlap contra os `947` rows V194: `0`.
- O builder removeu `371` overlaps potenciais com V194.
- Mix:
  - `gravity_constant`: `150`
  - `numeral_system`: `150`
  - `text_encryption`: `150`
  - `unit_conversion`: `150`
  - `bit_manipulation`: `160`
  - `equation_transform`: `120`

Decisao:

- Este e um candidato review-only, nao um treino lancado.
- Antes de qualquer H100/Colab training, ainda falta:
  - revisar manifest;
  - auditar tokenizer/template trainability;
  - confirmar que o objetivo V214 nao repete V206A/V206B;
  - preparar treino curto com logs explicitos;
  - manter submit bloqueado.

## V214 H100 Micro-Replay Colab

Scripts/artefatos:

- `scripts/split_v214_micro_replay_candidate.py`
- `data/v214/v214_micro_train.jsonl`
- `data/v214/v214_micro_val.jsonl`
- `data/v214/v214_micro_split_manifest.json`
- `scripts/build_v214_h100_micro_replay_colab.py`
- `notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

Split interno:

- treino: `792` rows.
- validacao interna de loss: `88` rows.
- overlap treino/validacao interna: `0`.
- uso: diagnostico de loss/trainability apenas; nao substitui gate V194 solve-rate.

Notebook V214:

- monta Drive;
- bootstrapa scripts/dados V214 em `/content/kg1`;
- audita dataset e hashes;
- audita adapter V194 no Drive;
- constroi CSVs weak/full/strong a partir do gate V194;
- roda dry-run de trainability;
- so treina se `KG1_V214_RUN_TRAIN=1`;
- se treinar, roda um step de continuation V194 com LR `3e-7`;
- usa filtro trainable LoRA `q_proj,k_proj,v_proj,o_proj,in_proj,out_proj`;
- roda weak eval antes de full eval;
- so roda full eval se weak `>=191/315`;
- nunca empacota e nunca submete.

Gates atuais do notebook:

- dry-run precisa escrever `dry_run_model_recipe_report.json`;
- weak precisa superar V194 para liberar full eval;
- strict candidate exige `full >=828/947`, weak `>=198/315`, strong `632/632`;
- preferido segue `>=830/947`;
- qualquer promocao continua exigindo revisao humana.

URL Colab:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

Observacao: o notebook foi preparado para o branch publicado `v214-h100-micro-replay`. Se o notebook for mergeado para `master`, o segmento do branch no URL pode ser alterado.
