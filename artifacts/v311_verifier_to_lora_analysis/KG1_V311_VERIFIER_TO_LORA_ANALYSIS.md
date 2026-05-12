# KG1 V311 - Analise verifier/postprocessor para LoRA puro

Gerado em: 2026-05-12

Fonte analisada: `C:\Users\davis\Downloads\ANALISE_DESAFIO_IAS_15.txt`

## Veredito

O arquivo contem uma direcao tecnica util, mas nao uma conversao literal. Nao existe caminho confiavel para "compilar" regex, solver, verifier ou postprocessor Python diretamente para pesos LoRA. O caminho utilizavel e destilar comportamento: usar postprocessor/verifier como professor offline para gerar targets canonicos, exemplos corrigidos e negativos, depois treinar o adapter para emitir a resposta final sem codigo externo em inferencia.

Isso e coerente com o que ja fizemos, mas aponta dois gaps concretos:

1. O V303/V304 ja tentou destilar regras de bit/equation, mas ainda nao usa pares de preferencia `chosen/rejected` nem hard negatives sistematicos.
2. O V304 usa `Final answer: <answer>`; para alinhar melhor com o prompt oficial e extractor, a proxima versao deve padronizar exatamente uma saida `Final answer: \boxed{<answer>}` ou `\boxed{<answer>}`, sem texto depois.

## Evidencias verificadas no repo

- `src/competition_utils.py` implementa extractor/verifier no estilo Kaggle: ultimo `\boxed{}` primeiro, fallback para frases de final answer, depois numero/linha final.
- `scripts/hf_job_train_v90.py` ja usa loss mask nos tokens de resposta e tambem suporta segmentos pretokenizados com mascara. Isso valida a recomendacao de loss masking do arquivo.
- `scripts/run_v302_combined_postprocessor_gate.py` e `scripts/run_v306_solver_promotion_gate.py` provaram sinal local: `823/947` baseline para `838/947`, com `bit_manipulation 135 -> 146` e `equation_transform 56 -> 60`, sem losses no gate local. Esse sinal ainda nao e adapter-only.
- `scripts/build_v303_bit_fullbyte_distill_dataset.py` e `scripts/build_v304_solver_trace_distill_dataset.py` ja sao tentativas de transformar parte do postprocessor/verifier em dataset LoRA.
- `scripts/package_hf_adapter_submission.py` bloqueia pacote que dependa de postprocessor externo; isso esta correto para adapter-only.

## Evidencias verificadas externamente

- PEFT documenta que MoE experts podem ser `nn.Parameter`, exigindo `target_parameters` para LoRA em parametros como `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`.
- A mesma documentacao alerta que LoRA em MoE experts pode ter overhead grande e restricoes de multiplos adapters.
- vLLM documenta `max_lora_rank` e restricao de target modules por sufixo; isso reforca que adapter package precisa bater com os modulos suportados no runtime.
- O paper LoRA confirma a natureza da tecnica: pesos base congelados e matrizes treinaveis de baixa patente. Isso nao fornece garantia deterministica de verifier.
- O paper QLoRA mostra que LoRA em todas as camadas lineares pode ser necessario para aproximar full fine-tuning. Isso apoia o diagnostico de que attention-only/lm_head tende a ser capacidade insuficiente.

## V310 como double check pratico

O V310 implementou exatamente o gate que faltava: exigir que `LORA_TARGET_PARAMETERS` gerasse tensors LoRA treinaveis para `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`.

Resultado observado no job HF `6a036e497618f125ee2b78ec`:

- init adapter V290 checkpoint-6 carregou `12011/12011` pesos;
- tokenizacao V304 passou sem truncation e com offset masks completos;
- o gate falhou antes do treino:
  - `LORA_TARGET_PARAMETERS were configured but no matching LoRA tensors were found: mlp.experts.gate_up_proj, mlp.experts.down_proj`.

Esse erro e bom: ele prova que a rota PEFT incremental atual nao estava efetivamente treinando os experts, mesmo quando os campos apareciam na config. Nao devemos gastar H200 em "mais steps" desse mesmo caminho.

## O que podemos usar agora

### P0 - Usar como proxima direcao de dataset

Criar um V311/V312 CPU-only que gere um pack verifier-distilled a partir do sinal V306:

- positivos: completions canonicas para os `15` ganhos V306 e suas variantes sinteticas fora do weak;
- negativos: predicao original errada, resposta com formato errado, multiplos boxes, bit quase certo, numeric sign/digit-order errado;
- formato: sempre exatamente uma resposta final canonica;
- metadata: familia, regra, origem do verifier, motivo de rejeicao, e flag `weak_gate_rows_used_for_training=false`;
- gate: zero overlap com weak por id/prompt hash, zero conflito de assistant, tokenizacao real, e auditoria de formato.

### P1 - Preferencia/DPO, somente depois do pack CPU

O arquivo recomenda DPO/ORPO/KTO. Isso e tecnicamente plausivel, mas ainda nao e o primeiro passo porque:

- nao temos um builder de pares `chosen/rejected` versionado e auditado;
- TRL/DPO aumenta custo e complexidade no HF;
- primeiro precisamos provar, em CPU, que os pares sao limpos e nao vazam weak/full labels.

### P1 - Padronizar output canonico

Para novas destilacoes, preferir:

```text
Final answer: \boxed{ANSWER}
```

ou apenas:

```text
\boxed{ANSWER}
```

Nao usar resposta final solta quando a meta e adapter-only no prompt oficial, porque o prompt oficial pede `\boxed{}` e o extractor da competicao prioriza boxed.

### P2 - LoRA+ / all-linear / rank maior

O arquivo cita LoRA+, `all-linear`, DPO e ranks maiores. Para esta competicao:

- `r > 32` nao e aceitavel se o runtime/competicao mantiver `max_lora_rank=32`.
- `all-linear`/experts so deve voltar depois de resolver o gap de empacotamento/conversao Huikang/Tinker-style ou provar suporte PEFT/vLLM real.
- LoRA+ pode ser investigado, mas nao e prioridade antes de dados e formato melhores.

## O que nao usar

- Nao tentar converter codigo Python do verifier em pesos.
- Nao usar `merge_and_unload()` como caminho principal de submissao se o formato exigido e adapter-only.
- Nao aceitar postprocessor externo no pacote official-like.
- Nao rodar treino longo so porque eval loss caiu; V308 provou que eval loss melhor pode piorar ACC por familia.
- Nao usar `no-strip`/experts sem gate de compatibilidade com o runtime de avaliacao.

## Proxima acao recomendada

Depois de concluir ou abortar formalmente o V310, implementar um builder CPU-only:

`scripts/build_v311_verifier_distillation_preference_pack.py`

Objetivo:

- transformar V306/V302 em dataset/pair-pack auditavel;
- medir quantos exemplos positivos, negativos e correcoes podem ser gerados sem usar weak rows como treino;
- rodar tokenization/format gate;
- so depois decidir se vale um smoke HF pequeno.

## Implementacao CPU realizada

O builder foi implementado e executado sem HF/GPU:

- script: `scripts/build_v311_verifier_distillation_preference_pack.py`;
- output: `artifacts/v311_verifier_distillation_preference_pack/20260512T1535Z/`;
- seed gain rows: `15`;
- preference rows: `60`;
- familias: `bit_manipulation=11`, `equation_transform=4`;
- status: `training_authorization=blocked_seed_only_until_synthetic_out_of_gate_variants`.

Esse pack e intencionalmente bloqueado para treino direto. Ele serve como semente auditada para gerar variantes sinteticas fora dos gate rows, exatamente para evitar transformar o gate local em memorization dataset.
