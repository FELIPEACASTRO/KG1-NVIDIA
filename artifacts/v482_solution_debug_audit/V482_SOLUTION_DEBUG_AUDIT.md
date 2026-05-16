# V482 Solution Debug Audit

Data: 2026-05-16

## Pergunta

Por que os treinos recentes mexem em loss/eval_loss, mas nao geram ganho
adapter-only submit-safe em `bit_manipulation` e `equation_transform`?

## Conclusao

O gargalo imediato nao esta no calculo de ACC nem no dataset V479. O erro
operacional mais forte esta na preservacao da configuracao PEFT do adapter
inicial:

- o adapter inicial V290 checkpoint-6 declara `target_parameters`;
- o job V480 treinou com `LORA_TARGET_PARAMETERS` vazio;
- o job V480 deixou `REQUIRE_LORA_TARGET_PARAMETER_MATCH=0`;
- todos os checkpoints V480 foram salvos com `target_parameters=null`.

Isso significa que V480 nao preservou integralmente a receita PEFT declarada do
adapter que deveria continuar a linhagem 0.86. A avaliacao V481 confirma que a
mudanca nao foi submit-safe: equation sobe pontualmente para `57/155`, mas bit
cai para `134/160` e alguns checkpoints truncam.

## Evidencia

### Adapter config

| Adapter | `adapter_config.json` bytes | sha256 | `target_parameters` |
|---|---:|---|---|
| V290 seed `checkpoint-6` | 1210 | `a3d74c5a52ce75f71a8406222d877b9760ea18a40a772bcf407686c8ea19f11d` | `["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]` |
| V480 `checkpoint-2` | 1149 | `ca52d6d86aa6be727be6af3b7ce1d8c7a1743c429034a7e3e742f0ec3e8fefe7` | `null` |
| V480 `checkpoint-4` | 1149 | `ca52d6d86aa6be727be6af3b7ce1d8c7a1743c429034a7e3e742f0ec3e8fefe7` | `null` |
| V480 `checkpoint-6` | 1149 | `ca52d6d86aa6be727be6af3b7ce1d8c7a1743c429034a7e3e742f0ec3e8fefe7` | `null` |
| V480 `checkpoint-8` | 1149 | `ca52d6d86aa6be727be6af3b7ce1d8c7a1743c429034a7e3e742f0ec3e8fefe7` | `null` |

### V480 HF log

O preflight remoto viu corretamente que o adapter inicial tinha
`target_parameters`:

`init_adapter_gate = ... "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]`

Mas o treino executou com:

- `LoRA target_parameters: disabled`;
- `Require LoRA target-parameter match: False`;
- `target_parameter_lora_params: {}`;
- `target_parameter_lora_tensors: {}`.

### V480 loss

O baseline eval-loss antes do treino foi `0.9725`. Todos os checkpoints ficaram
piores:

| Step | eval_loss |
|---:|---:|
| baseline | 0.9725 |
| 2 | 0.9761 |
| 4 | 0.9752 |
| 6 | 0.9738 |
| 8/final | 0.9739 |

Mesmo antes do weak ACC, o proprio validation loss sinalizou que o delta nao era
saudavel.

### V481 weak ACC

| Checkpoint | Total | equation_transform | bit_manipulation | truncated | Decisao |
|---|---:|---:|---:|---:|---|
| V480 checkpoint-2 | 191/315 | 57/155 | 134/160 | 1 | rejeitado |
| V480 checkpoint-4 | 190/315 | 56/155 | 134/160 | 0 | rejeitado |
| V480 checkpoint-6 | 191/315 | 57/155 | 134/160 | 1 | rejeitado |

Baseline submit-safe ativo segue `192/315`, equation `56/155`, bit `136/160`,
trunc `0`.

## Pecas auditadas

| Peca | Resultado |
|---|---|
| Dataset V479 | hashes, rows, family/subcategory counts e tokenizacao passaram; nao e o bug principal |
| V478 objective alignment | passou em V479; corrigiu o bug de peso V476 |
| ACC weak | coerente com logs; V481 `candidate_per_task` soma corretamente cada familia |
| Loss/eval_loss | util como early warning, mas nao substitui ACC; em V480 piorou contra baseline |
| Adapter config | divergencia real: V290 tem `target_parameters`, V480 salvou `null` |
| Gates antigos | nao bloqueavam mismatch de `target_parameters`; corrigido agora |

## Correcoes implementadas

1. `scripts/hf_job_preflight_gate.py` agora, quando
   `KG1_STRICT_INIT_ADAPTER_CONFIG=1`, compara `target_modules` e
   `target_parameters` do adapter inicial contra o ambiente do job.
2. O mesmo preflight agora bloqueia init adapter com `target_parameters` se
   `REQUIRE_LORA_TARGET_PARAMETER_MATCH=0`.
3. `scripts/kg1_static_safety_gate.py` agora bloqueia launchers ativos que
   limpem `LORA_TARGET_PARAMETERS` ou desliguem a verificacao para MoE
   `target_parameters`.
4. O launcher base V391 agora passa
   `KG1_LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`
   e usa `REQUIRE_LORA_TARGET_PARAMETER_MATCH=1`.
5. V480 foi endurecido para falhar em debug se algum snippet antigo reaparecer.

## Proximo passo tecnico

Nao relancar V480 como estava. A proxima tentativa paga so faz sentido como
smoke minimo config-preserving:

1. recriar o job com target_parameters preservados;
2. rodar preflight remoto e confirmar no log:
   - `target_parameters` nao vazio;
   - `Require LoRA target-parameter match: True`;
   - contadores `target_parameter_lora_tensors` nao vazios;
   - checkpoints salvos com `target_parameters` igual ao seed;
3. rodar `MAX_STEPS=2` ou menor smoke;
4. fazer weak eval imediata;
5. continuar apenas se bater `total>=193`, `equation>=57`, `bit>=136`,
   `truncated=0`.

Se o smoke preservando config tambem falhar, o caminho LoRA SFT curto fica
bloqueado novamente e a rota volta para solver/verifier CPU ou preference com
hard negatives, nao para mais epochs.
