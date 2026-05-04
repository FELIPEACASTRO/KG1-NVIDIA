# V200 Rank-Hillclimb Roadmap

## Regra principal

O baseline de producao continua sendo V194/ref `52275052`, score publico `0.86`, zip SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.

Nenhum candidato substitui esse baseline sem score Kaggle confirmado:

- Score `< 0.86`: descartar.
- Score `= 0.86`: manter em quarentena, promover somente se ranking/empate for melhor.
- Score `> 0.86`: promover como novo baseline obrigatorio.

V199B/ref `52325494` saiu com score publico `0.86`. Ele nao regrediu, mas tambem nao superou o baseline. Portanto fica em quarentena e nao e promovido sem confirmacao de ranking/empate melhor.

## Objetivo V200-V202

Fazer hill-climbing conservador a partir do melhor ranking confirmado, sem repetir o erro V198.

O criterio nao e "treinou e loss caiu"; o criterio e:

1. Comecar do melhor baseline confirmado.
2. Alterar uma unica hipotese por candidato.
3. Passar gate local contra o baseline.
4. Submeter apenas candidatos com delta local nao regressivo.
5. Promover apenas se o Kaggle confirmar score/rank melhor.

## Candidatos

### V200A - micro attention-only 5 steps

Hipotese: o V199B foi conservador, mas 10 steps ainda pode ter mexido demais. Testar metade da exposicao.

- Status: notebook gerado em `notebooks/KG1_V200A_H100_MICRO_ATTENTION_COLAB_PRO.ipynb`.
- Gates estaticos: `v200a_notebook_doublecheck.json` PASS e `v200a_lineage_gate.json` PASS.
- Init: V194 exact zip SHA `49886191...`.
- Steps: 5.
- LR: `5e-7 -> 2e-7`.
- Trainable modules: `q_proj,k_proj,v_proj,o_proj,in_proj,out_proj`.
- Freeze: `lm_head,up_proj,down_proj`.
- Dados: mesmo pacote strict V198/V195/V196/V197 ja validado.
- Gate minimo: final eval <= baseline eval e best eval <= baseline eval.
- Submit: apenas se zip passa SHA/layout/tensor/preflight gate.

### V200B - update interpolation V194 + V199B

Hipotese: V199B tem sinal util, mas update integral pode regredir no LB. Aplicar delta menor.

- Init/base: V194.
- Delta source: V199B adapter SHA `444dd40c...`.
- Pesos candidatos: 2.5%, 5%, 7.5%, 10%.
- Sem treino.
- Gate: cada interpolacao deve preservar tensor count `12011`, target modules e prefixos Kaggle.
- Submit: no maximo 1 candidato, o melhor por eval local.

### V201 - targeted anti-regression only

Hipotese: ganhos devem vir de categorias fracas sem afetar categorias fortes.

- Init: melhor baseline confirmado, nao V199B ate score confirmado.
- Dataset: somente exemplos corrigidos e solver-verificados de bit/cipher/equation edge cases.
- Steps: 5-8.
- LR: `3e-7 -> 1e-7`.
- Trainable modules: atencao apenas.
- Gate extra: categorias fortes nao podem piorar no eval estratificado.

### V202 - no-train packaging/control submit

Hipotese: antes de grandes mudancas, confirmar se o empacotamento local/stage nao altera o artefato.

- Nao treinar.
- Revalidar V194 exact SHA e V199B exact SHA.
- Nao submeter se ja houver submissao identica recente.
- Usar somente para forensic se algum score divergir do esperado.

## Gates obrigatorios

Antes de qualquer submit:

- ZIP tem exatamente `adapter_config.json` e `adapter_model.safetensors`.
- Nome final enviado ao Kaggle e `submission.zip`.
- ZIP SHA registrado.
- Adapter model SHA registrado.
- Tensor count `12011`.
- Prefixos proibidos ausentes.
- Target modules completos.
- Base model correto.
- Nenhum segredo, checkpoint, optimizer ou arquivo de treino dentro do ZIP.
- Candidate nao e V198 regressivo.
- Candidate nao e baseline inalterado, exceto quando explicitamente marcado como forensic/control.

## Ordem operacional

1. Manter V194/ref `52275052` como baseline de producao.
2. Manter V199B/ref `52325494` em quarentena: score `0.86`, ranking ainda nao confirmado como melhor.
3. Gerar V200A e V200B a partir de V194, nao de V199B.
4. Submeter no maximo um candidato por vez.
5. Atualizar `best_baseline_registry.json` apos cada score completo.

## Stop conditions

Parar imediatamente se:

- Novo score for `< 0.86`.
- Gate local aprovar por margem menor que `0.0005` e houver mudanca ampla de pesos.
- Qualquer candidate vier de V198 ou de linhagem sem SHA confirmado.
- O Kaggle aceitar submit mas o score sumir/ficar nulo por periodo prolongado; nesse caso fazer forensic antes de novo submit.
