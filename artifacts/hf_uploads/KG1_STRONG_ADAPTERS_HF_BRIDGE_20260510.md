# KG1 Strong Adapters HF Bridge - 2026-05-10

Objetivo: remover a dependencia operacional do Google Drive para os pesos fortes conhecidos antes de novos jobs HF.

## Repositorio HF

- Repo: `felipesp1983/kg1-strong-adapters-v194-v226`
- URL: `https://huggingface.co/felipesp1983/kg1-strong-adapters-v194-v226`
- Tipo: model repo
- Visibilidade: privado
- SHA remoto validado: `1bb23fdbc3f5ccadd36b91e8f7db9d7474bf6312`
- Arquivos remotos validados: `11`

## Adapters

| Pasta HF | Origem Drive | Weak conhecido | Tamanho | SHA256 | Tensor count | Status |
|---|---|---:|---:|---|---:|---|
| `v226_checkpoint1` | `KG1_NVIDIA_V226/output_v226_equation_checkpoint_sweep/train_v226_v194_micro_lr2e9_s6/checkpoint-1` | `191/315`, equation `55`, bit `136`, trunc `0` | `4259063856` | `f4e2083d83f13a102cd86e5d1295a8603264856c17ec35c357188e1acde6ea79` | `12011` | validado |
| `v194_protected` | `KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter` | `190/315`, equation `54`, bit `136`, trunc `0` | `4259069440` | `01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f` | `12011` | validado |

## Arquivos Esperados

- `.gitattributes`
- `README.md`
- `strong_adapters_validation_manifest.json`
- `v226_checkpoint1/adapter_config.json`
- `v226_checkpoint1/adapter_model.safetensors`
- `v226_checkpoint1/README.md`
- `v226_checkpoint1/chat_template.jinja`
- `v226_checkpoint1/tokenizer.json`
- `v226_checkpoint1/tokenizer_config.json`
- `v194_protected/adapter_config.json`
- `v194_protected/adapter_model.safetensors`

## Limpeza Local

- Staging temporario removido apos validacao: `%TEMP%/kg1_drive_strong_adapters_hf_20260510`
- Bytes removidos do staging: `8535403351`
- Nenhum processo `python`/`rclone`/`hf` ficou rodando depois da ponte.

## Uso Correto

1. Usar esse repo privado como fonte HF para weak eval canonico dos pesos fortes.
2. Nao promover nenhum treino novo sem comparar contra `v226_checkpoint1`.
3. Qualquer smoke training deve partir de um initializer forte e executar weak eval imediato.
4. O score amplo `0.86` permanece evidencia historica de familias nao criticas; as metas atuais continuam sendo `equation_transform` e `bit_manipulation`.
