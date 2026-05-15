# Prompt Para APIs Externas - KG1 V438 Error Ledger

Voce deve atuar como engenheiro senior de ML/Kaggle focado em diagnostico
causal. Nao alucine. Se algo nao estiver provado pelos dados abaixo, marque
como hipotese. O objetivo e propor correcoes testaveis para melhorar
`equation_transform` e preservar `bit_manipulation` em um pacote LoRA
adapter-only.

## Contexto Do Desafio

Competicao: NVIDIA Nemotron Model Reasoning Challenge.

O pacote submetido deve ser adapter-only. Nao podemos depender de script de
inferencia, postprocessor, verifier runtime, logit mask, constrained decoding,
confidence threshold, prompt externo, tokenizer custom, soft prompt ou qualquer
codigo adicional. O ganho so conta se aparecer no adapter/package.

Metricas internas weak atuais do melhor estado submit-safe:

- Total weak: `192/315`.
- `equation_transform`: `56/155`.
- `bit_manipulation`: `136/160`.
- Truncated: `0`.

Gate minimo para promover qualquer novo candidato:

- Total weak `>192/315`.
- `equation_transform >56/155`, ideal `>=60/155`.
- `bit_manipulation >=136/160`.
- Truncated `0`.
- Depois, full official-like precisa superar `823/947`.

## O Que Ja Falhou

Nao recomende estas rotas sem uma mudanca tecnica nova e verificavel:

- Broad SFT, mais epochs, LR sweep.
- Repetir treino em H200/A100 sem novo gate CPU.
- Usar weak/full misses como treino, filtro, tiebreak ou cherry-pick.
- Usar solver/verifier/postprocessor no submit.
- Constrained decoding, logit masks, runtime abstention ou confidence threshold.
- Prompt sweep amplo e thinking variants.
- Adapter soups sem sinal complementar medido.
- Decidir por `eval_loss` ou `train_loss`.

## Descobertas Reais Recentes

### E001 - Dataset V435E antigo misturava negativos invalidos

V435E antigo tinha `200` pares:

- `133` hard negatives semanticamente errados.
- `67` format-only negatives.

Isso contaminou preference mean-NLL. V436 rodou nesse dataset misto e piorou no
primeiro checkpoint:

- Baseline preference: `6/40`.
- Step 4: `5/40`.
- Equation: `4/22 -> 3/22`.

Esse caminho foi bloqueado.

### E002 - V436B hard-negative-only tambem piorou

Corrigimos o dataset para hard-negative-only:

- `133` pares totais.
- `equation_transform=120`.
- `bit_manipulation=13`.
- Todos `negative_type=hard_negative_adapter_exact_wrong`.

V436B em H200:

- Job: `felipesp1983/6a073c74e48bea4538b9e652`.
- Adapter inicial: V290 checkpoint-6.
- LoRA treinavel: `q_proj,k_proj,v_proj,o_proj,lm_head`.
- `8,015,872` parametros treinaveis, `0.0247%` do modelo.
- Tokenizacao sem truncation.
- Adapter carregado `12011/12011`.

Resultado:

| Metrica | Baseline | Checkpoint-3 |
|---|---:|---:|
| preference total | `6/24` | `4/24` |
| equation preference | `4/22` | `2/22` |
| bit preference | `2/2` | `2/2` |

Foi cancelado por FinOps. Nao houve weak/full eval.

### E003 - Erro estrutural descoberto pelo V438

V438 auditou os pares hard-negative-only sem GPU.

Depois de usar a normalizacao oficial de `\\boxed{}`:

- `answer_box_mismatch_rows=0`.
- `rejected_box_mismatch_rows=0`.
- Ou seja: os labels finais estao corretos.

Mas a estrutura do texto e ruim para transferencia:

- `chosen_mentions_adapter_prediction_rows=123/133`.
- `chosen_mentions_public_train_label_audit_rows=133/133`.
- `chosen_tokens_mean=34.08`.
- `rejected_tokens_mean=26.80`.
- `chosen/rejected token ratio=1.2767`.

Exemplo de target atual:

```text
Verification:
Check the hidden transformation against every example. The frozen adapter candidate 'WRONG' is rejected by the public-train label audit.
Final answer: \boxed{ANSWER}
```

Exemplo de rejected atual:

```text
Rejected adapter candidate:
This is the exact final answer selected by the frozen adapter on the prompt-only probe.
Final answer: \boxed{WRONG}
```

Hipotese forte: o chosen ensina o modelo a repetir a resposta errada dentro do
texto e a falar de auditoria, o que nao existe no submit. O objetivo mean-NLL
pode piorar equation porque o texto correto e mais longo/estranho e contem o
erro como substring.

## Pergunta Principal

Qual e a melhor correcao tecnica, testavel primeiro em CPU, para transformar
esses hard negatives reais em ganho de adapter-only, sem runtime externo?

Responda com uma proposta concreta. Nao fale genericamente.

## Restrições Obrigatorias

1. Nao usar weak/full como treino, tiebreak, filtro ou selecao.
2. Nao usar answer antes de a regra/dataset estar congelado, exceto em linhas
   public-train permitidas ja auditadas.
3. Nao propor submit com solver/verifier/postprocessor.
4. Nao propor apenas "treinar mais".
5. Nao promover por loss; precisa passar ACC gate.
6. Toda proposta deve ter kill-switch no primeiro checkpoint.

## Opcoes Que Quero Que Voce Avalie

Avalie especificamente estas alternativas e diga qual e a melhor:

1. Trocar preference chosen/rejected por final-answer-only equalizado:
   - chosen: `Final answer: \\boxed{ANSWER}`
   - rejected: `Final answer: \\boxed{WRONG}`
2. SFT answer-only ultra-curto usando apenas chosen, sem rejected.
3. DPO/ORPO apenas no span final `\\boxed{...}`, mascarando todo o prefixo.
4. Contraste por token apenas no payload dentro de `\\boxed{}`.
5. Filtrar para subcategorias com maior chance:
   - `equation_numeric_operator_to_number`
   - `equation_numeric_operator_to_symbolic`
   - `equation_symbolic_sequence`
   - `equation_symbolic_short`
   - `bit_adapter_exact_wrong`
6. Manter replay minimo de bit para preservar `bit>=136`.
7. Adicionar hard negatives sinteticos de mesmo formato/mesmo comprimento.

## Formato Da Resposta

Responda exatamente nestas secoes:

1. **Diagnostico Causal**
   - O que provavelmente causou V436B piorar?
   - O que esta provado e o que e hipotese?

2. **Correcao Recomendada**
   - Escolha uma unica rota principal.
   - Diga o formato exato das linhas de treino.
   - Diga quais campos remover do target.

3. **Gate CPU Antes De GPU**
   - Liste os checks obrigatorios.
   - Inclua metricas minimas para liberar H200.

4. **Config De Treino Curto**
   - Modulos LoRA treinaveis.
   - LR, steps, batch, loss.
   - Stop condition no primeiro checkpoint.

5. **Riscos**
   - Como a proposta pode piorar bit?
   - Como detectar isso cedo?

6. **Implementacao**
   - Pseudocodigo ou patch-level plan.
   - Nada de sugestoes vagas.

7. **O Que Nao Fazer**
   - Liste explicitamente rotas que parecem tentadoras mas devem continuar bloqueadas.

Se voce nao souber, diga "nao sei" e explique qual evidencia faltaria.
