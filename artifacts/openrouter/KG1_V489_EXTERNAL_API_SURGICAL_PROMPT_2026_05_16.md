# KG1 V489 External API Surgical Prompt

Use este prompt em modelos externos somente para obter propostas testaveis. Nao
aceite sugestoes que dependam de solver/verifier/postprocessor no runtime de
submit, weak/full leakage ou selecao manual por gabarito.

```text
Voce e um auditor senior de ML/LLM competitivo. Analise o caso abaixo sem
alucinar. Responda apenas com hipoteses testaveis, checks de codigo/dados e
mudancas que possam melhorar ACC adapter-only no desafio Kaggle NVIDIA Nemotron
Model Reasoning Challenge.

CONTEXTO DO DESAFIO
- O submit valido e adapter-only LoRA/PEFT para Nemotron; nao podemos usar
  runtime solver, verifier, postprocessor, logit mask, constrained decoding ou
  cherry-pick baseado em gabarito hidden.
- Familias problematicas no weak gate de 315 linhas:
  - bit_manipulation: baseline submit-safe 136/160
  - equation_transform: baseline submit-safe 56/155
  - total baseline submit-safe 192/315, truncated=0
- Promocao minima: total > 192, equation > 56, bit >= 136, truncated=0. Full
  official-like precisa bater o baseline conhecido.

ESTADO ATUAL MEDIDO
- Melhor submit-safe conhecido: V291/V290 checkpoint-6:
  - weak 192/315
  - equation_transform 56/155
  - bit_manipulation 136/160
  - truncated 0
- V477 ckpt-2:
  - weak 192/315
  - equation 57/155
  - bit 135/160
  - truncated 0
  - bloqueado por regressao de bit
- V475 CPU solver/verifier projection:
  - weak 196/315
  - equation 60/155
  - bit 136/160
  - ainda nao submit-safe porque depende de solver/verifier
- V487 H200 treino:
  - melhor checkpoint por eval_loss: checkpoint-10, eval_loss 1.3519
  - carregou PEFT target_parameters via alias Nemotron
- V488 weak eval do V487 checkpoint-10:
  - weak 191/315
  - equation 57/155
  - bit 134/160
  - truncated 1
  - bloqueado
- V489 row diff:
  - equation gain: id 518deb39
  - bit regressions: ids 8740ed31 e 59bee375
  - 59bee375 truncou em 7680 tokens

O QUE JA TENTAMOS E NAO RESOLVEU
- Mais SFT amplo, mais epochs/steps e LR sweeps: reduziram eval_loss, mas nao
  melhoraram ACC submit-safe.
- Datasets de traces/CoT, hard negatives e answer-span com pesos altos:
  transferiram mal para adapter-only ou derrubaram bit.
- Foco excessivo em equation: gerou equation +1, mas bit -1/-2.
- `answers_equivalent` permissivo parecia melhorar numeros, mas superconta bit
  strings; nao e valido para ACC.
- Treinos com target_parameters carregados mas frozen-active nao provaram treino
  real de `up_proj/down_proj`.

BUGS/GAPS CORRIGIDOS EM V489
- `extract_final_answer_for_expected` agora so pode desambiguar o ultimo
  `\boxed{}` usando `verify_answer`; antes podia escolher boxed anterior e vazar
  validacao.
- `scripts/audit_v449_acc_metric_integrity.py` agora audita raw extraction:
  `simple_extracted` vs `expected_aware_extracted`.
- `scripts/hf_job_train_v90.py` agora grava no manifesto:
  `target_parameter_trainable_lora_tensors`,
  `target_parameter_trainable_lora_params`,
  `target_parameters_trainability_mode`,
  `trainable_parameter_report_after_filter`.
- Launchers com MoE target_parameters e allowlist precisam declarar
  `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` ou `1`.
- `scripts/kg1_static_safety_gate.py` tinha chave duplicada que anulava checks
  de `hf_job_train_v90.py`; foi corrigido.

REGRAS DE FINOPS/GATE
- Nao gastar GPU se o CPU gate nao mostrar sinal novo.
- Cancelar se nao houver caminho para superar:
  total > 192, equation > 56, bit >= 136, truncated=0.
- H200 no maximo 1h sem autorizacao adicional.
- Todo novo job precisa passar:
  static safety gate, pre-paid integration gate, HF preflight gate, objective
  alignment gate, tokenization/offset-mask, hashes/datasets/family counts.

PERGUNTAS QUE QUERO QUE VOCE RESPONDA
1. Onde ainda pode haver bug silencioso que explique eval_loss melhorando sem
   ACC melhorar?
2. A receita deve treinar `up_proj/down_proj` target_parameters de fato
   (`REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`) ou manter frozen-active e
   mexer em outra parte? Justifique com risco de regressao em bit.
3. Que CPU gate objetivo voce implementaria antes de novo H200 para prever
   se equation sobe sem bit cair?
4. Como transformar os 4 ganhos CPU de equation em comportamento LoRA
   adapter-only sem solver runtime?
5. Como evitar truncation em bit sem perder capacidade de raciocinio de
   equation?
6. Que dataset minimalista voce treinaria agora, com quais pesos, filtros,
   hard negatives e kill-switch?
7. Que linhas de codigo/metricas voce auditaria primeiro para confirmar que
   score, ACC, extraction, thresholds e family counts estao corretos?

FORMATO DA RESPOSTA
- Liste no maximo 10 achados.
- Para cada achado, informe:
  - evidencia esperada;
  - mudanca concreta no codigo/dataset/gate;
  - como testar em CPU;
  - criterio de abortar;
  - risco de regressao em bit/equation;
  - estimativa realista de ganho weak.
- Nao recomende broad SFT, mais epochs, usar gabarito weak/full para treinar,
  runtime postprocessor/verifier, constrained decoding no submit, ou modelos/pesos
  publicos que violem regras.
```
