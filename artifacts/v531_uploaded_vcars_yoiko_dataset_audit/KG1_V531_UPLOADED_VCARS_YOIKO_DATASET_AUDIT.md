# KG1 V531 - Auditoria Dos Anexos V-CARS E Yoiko

Data: 2026-05-17

## Arquivos Auditados

| Arquivo | SHA256 | Tamanho | Classificacao |
|---|---|---:|---|
| `C:\Users\davis\Downloads\archive (7).zip` | `c122ef226e5a17f2beae096cbd8c1a631ddf9932be3b364d67a16962de6c1801` | `3418307265` bytes | pacote V-CARS/offline deps, nao dataset direto |
| `C:\Users\davis\Downloads\archive (8).zip` | `df000b9391501fd4feb413a969b7ea7f760d7b4b8b36b120f296a33ea9f19bc8` | `3212044621` bytes | adapter LoRA publico Yoiko ver5 |
| `C:\Users\davis\Downloads\vcars-external-data-metadata.json` | `eeef5463872305da3a71bff277408cbed1ed52b50383e00d9c5ca318494b5b63` | `6027` bytes | metadados Kaggle `williamlu931/vcars-external-data` |
| `C:\Users\davis\Downloads\nvidia-nemotron-yoiko-ver5-metadata.json` | `b0a0512adea12fcf9b91e7ee2793197c1f8971e04a73c857f24998fbbcc9c9a3` | `2792` bytes | metadados Kaggle `yoikoarmor/nvidia-nemotron-yoiko-ver5` |

## Resultado Do `archive (7).zip`

Conteudo:

- `V-CARS_Nemotron_Data_RL_Project_Spec_V1.2.md`;
- `V-CARS_Nemotron_Data_RL_Project_Spec_V1.3.md`;
- `kaggle_run_R0001_with_offline_deps.ipynb`;
- `46` wheels offline, incluindo `torch`, `triton`, `mamba_ssm`,
  `causal_conv1d`, `z3_solver` e bibliotecas CUDA;
- `773395.crdownload`, um fragmento de download, sem valor para treino.

O notebook contem `28` arquivos `vcars/*` embutidos. A auditoria dos modulos
mostra:

- `vcars/data/family_detector.py`: classificador heuristico por assinatura de
  prompt. Ele mapeia `bit manipulation` para `binary` e `secret set of
  transformation` para `symbolic`.
- `vcars/verification/parse_answer.py`: extrator `boxed -> phrase -> number ->
  lastline`.
- `vcars/verification/metric.py`: verificador binario estrito, numerico com
  `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` e string case-insensitive.
- `vcars/audit/audit_train_labels.py`: auditor completo apenas para roman e
  cipher; binary/gravity/unit/symbolic ficam `pending_m2`.
- `vcars/verification/reward.py`: placeholder com `NotImplementedError`.

Achado util:

- O pacote e util como referencia de contrato, notebook smoke, offline deps e
  disciplina de artifact protocol.
- Ele reforca regras que ja estao no KG1: nao aceitar `eval_loss` como proxy de
  ACC, usar paired comparison, family-wise metrics, `run_manifest`,
  `family_metrics`, failures export e gate antes de GRPO/RL.

Limite:

- Nao ha solver novo implementado para `bit_manipulation`.
- Nao ha solver novo implementado para `equation_transform`.
- Nao ha dataset direto de treino para as duas familias alvo.
- Nao justifica GPU ou submit por si so.

## Resultado Do `vcars-external-data-metadata.json`

Metadados:

- dataset Kaggle: `williamlu931/vcars-external-data`;
- licenca: Apache 2.0;
- descricao em chines: dataset extra publico para a competicao;
- `recordSet` aponta para `sentences.csv` do Tatoeba, com campos de lingua
  natural.

Decisao:

- Nao usar no treino atual. Conteudo externo de linguagem natural nao ataca os
  gaps atuais de `bit_manipulation` e `equation_transform`.
- Pode servir no futuro apenas como referencia de licenca/provenance, nao como
  fonte P0.

## Resultado Do `archive (8).zip`

Conteudo:

- `NVIDIA-nemotron-yoiko-ver5/README.md`;
- `NVIDIA-nemotron-yoiko-ver5/adapter_config.json`;
- `NVIDIA-nemotron-yoiko-ver5/adapter_model.safetensors`.

Adapter config:

| Campo | Valor |
|---|---|
| `base_model_name_or_path` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| `peft_type` | `LORA` |
| `task_type` | `CAUSAL_LM` |
| `r` | `32` |
| `lora_alpha` | `32` |
| `lora_dropout` | `0.0` |
| `target_modules` | `.*\.(in_proj|out_proj|up_proj|down_proj)$` |
| `target_parameters` | `null` |
| `modules_to_save` | `null` |
| `inference_mode` | `true` |
| `peft_version` | `0.18.1` |

Safetensors header:

- tensor count: `11960`;
- dtype: `F32` em todos os tensores;
- total LoRA params: `880138240`;
- tamanho F32 estimado: `3520552960` bytes;
- non-LoRA tensors: `0`;
- `in_proj`: `46` tensors;
- `out_proj`: `46` tensors;
- `up_proj`: `5934` tensors;
- `down_proj`: `5934` tensors;
- camadas observadas: `46`, indices `0..51`;
- camadas expert MoE: `23`, com ate `128` experts por camada.

Achado util:

- Este e o unico anexo V531 com possibilidade real de ganho de ACC sem novo
  treino, porque ja e um adapter LoRA completo.
- Ele e compatibile em termos basicos: rank `32`, alpha `32`, base Nemotron
  correto, sem `modules_to_save`, e somente tensores LoRA.
- Ele deve entrar como candidato de weak eval curto, nao como fonte de treino.

Riscos:

- O zip esta com subpasta `NVIDIA-nemotron-yoiko-ver5/`; para submit KG1 o zip
  precisa ter `adapter_config.json` e `adapter_model.safetensors` no root.
- O README nao declara score, familia, dados de treino nem avaliacao.
- `target_modules` e uma regex string, enquanto alguns gates KG1 esperam lista
  canonica de modulos. Antes de qualquer avaliacao, o gate precisa tratar regex
  PEFT como forma valida ou marcar explicitamente a diferenca.
- O adapter nao tem `target_parameters`; isso diverge da linha KG1 V485/V493
  que exige target parameters MoE. Como a cobertura MoE aparece via
  `up_proj/down_proj`, a avaliacao deve ser empirica via weak eval.
- Por ser F32 e grande, a avaliacao deve ser curta e com FinOps kill-switch.

## Ganho Potencial

Nao ha ganho medido ainda.

Classificacao objetiva:

| Fonte | Ajuda bit? | Ajuda equation? | Uso correto |
|---|---|---|---|
| `archive (7).zip` V-CARS | indireto | indireto | notebook contract, gates, offline deps, auditoria |
| `vcars-external-data-metadata.json` | nao | nao | ignorar no plano atual |
| `archive (8).zip` Yoiko adapter | possivel | possivel | weak eval label-free curto |
| `nvidia-nemotron-yoiko-ver5-metadata.json` | indireto | indireto | provenance/licenca do adapter |

## Decisao V531

1. Nao treinar com V-CARS external data.
2. Nao usar Tatoeba/sentences como fonte para as duas familias alvo.
3. Registrar Yoiko ver5 como candidato P1 de avaliacao adapter-only.
4. Antes de avaliar:
   - validar `adapter_config.json`;
   - validar header safetensors;
   - aceitar ou normalizar `target_modules` regex;
   - empacotar em root-level temp zip somente dentro do job;
   - apagar extraidos temporarios ao final.
5. So promover se weak eval label-free superar baseline com:
   - `total > 191/315`;
   - `bit >= 136/160`;
   - `equation >= 56/155`;
   - `trunc = 0`;
   - `8740ed31 = 01101000`.

Sem esse weak eval, V531 e apenas candidato, nao ganho submit-safe.
