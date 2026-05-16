# V483 OpenRouter Plateau Debug Consensus

Data: 2026-05-16

## Painel Chamado

| Modelo | Status | Custo OpenRouter |
|---|---|---:|
| `openai/gpt-5.5-pro` | ok, mas sem texto final util | `$0.514710` |
| `openai/gpt-5.3-codex` | ok | `$0.031257` |
| `anthropic/claude-opus-4.7` | ok | `$0.078065` |
| `google/gemini-3.1-pro-preview` | ok | `$0.034848` |
| `qwen/qwen3-max-thinking` | ok | `$0.006912` |
| `deepseek/deepseek-v4-pro` | ok, resposta ruidosa; usar apenas convergencias | `$0.010689` |

Custo total registrado: `$0.67648015`.

## Consenso Util

As respostas uteis convergiram em quatro pontos:

1. O `target_parameters=null` em V480 e um bug/gap real de continuidade PEFT,
   nao apenas detalhe cosmetico do manifest.
2. O proximo passo nao e mais epochs, mais dados ou outro H200 longo. O proximo
   passo e provar que a linhagem V290 e carregada e salva preservando:
   `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`.
3. O caminho preferido para continuacao e `PeftModel.from_pretrained(base,
   seed_adapter_dir, is_trainable=True)`. Se o codigo usar `LoraConfig` novo +
   `set_peft_model_state_dict`, ele precisa provar equivalencia exata de config
   e cobertura de tensores antes do treino.
4. File size de `adapter_model.safetensors` nao prova equivalencia. A comparacao
   correta e por `adapter_config.json`, lista de keys, shapes, dtypes, cobertura
   dos nomes `mlp.experts.*`, contagem de parametros trainable e round-trip de
   save/reload.

## Decisao Tecnica

Manter V482 como root-cause candidato principal e executar somente um caminho
config-preserving:

1. CPU preflight de round-trip:
   - carregar base + V290 com PEFT nativo;
   - listar trainable params;
   - exigir LoRA em `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`;
   - salvar checkpoint temporario;
   - reabrir e comparar `adapter_config.json`, keys/shapes/dtypes e coverage.
2. Se o CPU preflight falhar, nao abrir GPU.
3. Se o CPU preflight passar, rodar no HF um smoke de no maximo 2 steps.
4. Weak eval imediato. Continuar apenas com:
   - `total>=193`;
   - `equation_transform>=57`;
   - `bit_manipulation>=136`;
   - `truncated=0`.

## O Que Parar

- Treinos longos julgados por `eval_loss`.
- Broad SFT/replay sem novo sinal CPU.
- Launchers que recriem `LoraConfig` manualmente sem provar equivalencia.
- Jobs onde `target_parameter_lora_tensors` seja `{}`.
- Uso de solver/verifier como se fosse submit-safe adapter-only.

## Evidencia Ainda Necessaria

- Lista de keys/shapes/dtypes do `adapter_model.safetensors` V290 vs checkpoint
  salvo por round-trip.
- Dump de `named_parameters()` trainable depois do load PEFT nativo.
- Log do proximo smoke provando `target_parameter_lora_tensors` nao vazio.
- Segmento de avaliacao que gerou `accuracy=0.0000`, para separar bug de log de
  bug de scoring.

## Artefatos

- Prompt: `artifacts/v483_openrouter_plateau_consult/V483_OPENROUTER_PROMPT.md`
- Manifest: `artifacts/v483_openrouter_plateau_consult/v483_openrouter_manifest.json`
- Respostas por modelo: `artifacts/v483_openrouter_plateau_consult/*.md`
