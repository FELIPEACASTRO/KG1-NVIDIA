# KG1 V447 Public Submission Mining Consensus

Data: 2026-05-15

## Prompt

Prompt usado nas APIs:

- `artifacts/openrouter/v447_public_submission_mining_consensus/v447_prompt.txt`

Modelos consultados via OpenRouter:

| Modelo | Status | Custo reportado |
|---|---|---:|
| `openai/gpt-5.2` | ok | `$0.040194` |
| `anthropic/claude-sonnet-4.6` | ok | `$0.048519` |
| `deepseek/deepseek-r1-0528` | ok | `$0.00664205` |
| `qwen/qwen3-max-thinking` | ok | `$0.00538122` |
| `google/gemini-3.1-pro-preview` | ok | `$0.032004` |
| `perplexity/sonar-reasoning-pro` | ok | `$0.02497` |
| `google/gemini-2.5-pro` | falhou | sem conteudo retornado |

## Consenso

A ideia de minerar notebooks/submissoes publicas e valida, mas nao deve ser o
caminho principal imediato. O consenso dos modelos foi:

1. Nao continuar SFT/LoRA generico com mais epochs, LR sweep ou preference sem
   dado novo. Isso ja baixou loss sem melhorar ACC.
2. Priorizar o sinal V446, porque ele ja encontrou material target-aligned
   local e auditavel: `1310` traces aceitos preliminarmente, sendo `848`
   `bit_manipulation` e `462` `equation_transform`.
3. Usar mineracao de notebooks publicos em paralelo, CPU-only, para extrair
   tecnica, formato de trace, hard negatives e datasets permitidos. Nao usar
   adapter/peso/submissao de terceiros como artefato final.
4. Qualquer achado de notebook publico so vira treino se passar pelo mesmo gate:
   provenance, licenca, anti-leakage, tokenizacao, target alignment e comparacao
   contra baseline.
5. GPU so e justificavel depois de dataset builder + tokenization/pair gate.
   O primeiro checkpoint precisa manter `bit>=136`, `truncated=0` e mostrar
   caminho para `equation>56`.

## Decisao

O caminho correto agora e:

1. Transformar V446 em dataset treinavel minimo.
2. Rodar tokenization/pair gate e scaffold LoRA oficial.
3. Minerar notebooks publicos em CPU, mas apenas para complementar lacunas do
   V446.
4. Abrir no maximo um smoke H200 curto se os gates passarem.

## Roadmap Consensual

| Ordem | Acao | Objetivo | Gate |
|---:|---|---|---|
| 1 | V447 dataset builder do V446 | converter `1310` traces aceitos em train/val submit-safe | schema, family counts, boxed, target, no trunc |
| 2 | Tokenization + pair gate | garantir que o trace cabe no runner oficial | `0` prompt/completion truncation |
| 3 | Scaffold LoRA/vLLM dummy | provar que o pacote carrega adapter-only | adapter config/tensors e vLLM load ok |
| 4 | Mining CPU de notebooks publicos | extrair tecnicas faltantes, nao artefatos | licenca/provenance/anti-leakage |
| 5 | Merge apenas de achados aprovados | evitar ruido e regressao | accepted rows por familia e manifest |
| 6 | H200 smoke <= 1h | medir transferencia real para adapter | checkpoint-1 kill-switch |
| 7 | Weak/full gate | decidir submit | weak `>192`, equation `>56`, bit `>=136`, trunc `0`, full `>823` |

## Proibido

- Usar solver/verifier/postprocessor/parser/runtime customizado no submit.
- Treinar em weak/full, seus misses, suas respostas ou hashes como fonte de
  target.
- Usar adapters/pesos de terceiros diretamente.
- Rodar GPU para repetir receita que ja falhou.
- Promover por `loss`, `eval_loss`, preference accuracy interna ou opiniao de
  LLM.

## Proxima Acao

Implementar V447 dataset builder em CPU a partir do `candidate_audit_csv` do
V446, mantendo apenas rows aceitas, com split estratificado e manifests de
hashes. Depois rodar tokenization/pair gate. Se passar, preparar smoke H200
curto com kill-switch no primeiro checkpoint.
