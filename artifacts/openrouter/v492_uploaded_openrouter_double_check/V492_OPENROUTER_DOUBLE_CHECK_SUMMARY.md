# V492 OpenRouter Uploaded Double Check Summary

Data: 2026-05-16

Fonte analisada:

- `C:\Users\davis\Downloads\OpenRouter Chat Sat May 16 2026 (2).json`
- O conteudo bruto foi analisado localmente e nao foi mantido no repositorio;
  este arquivo e o resumo acionavel.

## Escopo

O arquivo e um export do OpenRouter com 12 respostas de modelos alem do prompt
original. Ele tambem contem muita metadata de modelos, provedores, termos legais
e URLs de API. Essa metadata foi classificada como ruido para ACC porque nao
altera treino, gate, dataset, metrica ou submit.

## Consenso Acionavel

1. O principal suspeito tecnico continua sendo MoE `target_parameters`
   frozen-active. V487/V488 carregavam `mlp.experts.gate_up_proj` e
   `mlp.experts.down_proj`, mas a allowlist treinavel era
   `q_proj,k_proj,v_proj,o_proj,lm_head`. O proximo smoke deve exigir modo
   `trainable` para `up_proj/down_proj`.

2. `lm_head` deve sair do smoke principal. V477/V488 mostram o padrao
   equation +1 junto com bit regression/truncation. Varios modelos apontaram
   `lm_head` como risco direto para distribuicao de tokens `0/1` e EOS.

3. `ANSWER_SPAN_LOSS_WEIGHT` deve ser `1.0` no proximo smoke promocional.
   Pesos altos podem baixar `eval_loss` sem melhorar ACC estrito.

4. O ganho V488 em `equation_transform=57` precisa ser auditado como raw
   output. O resultado pode ser ganho de extracao expected-aware e nao melhoria
   real do adapter.

5. O proximo job precisa ser fail-fast:

   - `MAX_STEPS=4`
   - eval no step/checkpoint 2
   - abortar se `bit_manipulation < 136`
   - abortar se `truncated > 0`
   - abortar se `equation_transform <= 56`
   - promover apenas se `total > 192`, `equation > 56`, `bit >= 136`,
     `truncated = 0`

## Configuracao Recomendavel Para O Proximo Smoke

```text
TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj
LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
ANSWER_SPAN_LOSS_WEIGHT=1.0
LEARNING_RATE=2.0e-8
FINAL_LEARNING_RATE=5.0e-9
MAX_STEPS=4
EVAL_EVERY_STEPS=2
SAVE_EVERY_STEPS=2
```

`lm_head` deve permanecer fora de `TRAINABLE_LORA_MODULES` no smoke principal.

## Rejeitado Ou Nao Promocional

- Treinar diretamente em IDs weak/full regredidos ou ganhos
  (`518deb39`, `8740ed31`, `59bee375`) como labels. Esses IDs podem servir para
  diagnostico de diff/gate, nao para treino submit-safe.
- Usar `ANSWER_SPAN_LOSS_WEIGHT > 1.0` em job promocional.
- Manter `lm_head` treinavel sem ablation isolada.
- Promover resultado que melhora apenas por `expected_aware_extracted`.
- Continuar H200 depois do primeiro checkpoint se o resultado ja nao puder
  bater o gate.

## Impacto No Roadmap

O roadmap foi atualizado para V491/V492:

- P3 agora e o smoke MoE trainable com `lm_head` congelado.
- O gate exige `target_parameters_trainability_mode="trainable"`.
- O roadmap removeu a rota de treino amplo e de ganho por parser como
  promocional.
- A proxima decisao e binaria: se P3 nao preservar bit e nao melhorar equation,
  broad SFT fica bloqueado e so restam ablations curtas ou novo dado
  verificavel nao contaminado.
