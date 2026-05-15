# V448 Weak Eval Result

Data: 2026-05-15

## Contexto

V448 testou se o dataset V447 clean, derivado do target-alignment V446/Tong,
transferia traces limpos para o adapter V290 checkpoint-6 sem perder
`bit_manipulation`.

Job avaliado:

- HF job: `https://huggingface.co/jobs/felipesp1983/6a07832e3308d79117b90a27`
- Adapter repo: `felipesp1983/kg1-nemotron-lora-v448-nemo-h200-v447-clean-trace-v290ckpt6`
- Checkpoint avaliado: `checkpoint-3`
- Commit esperado no launcher: `da3a99a7b88c794f193d99cd0c712688a4c709da`

## Resultado

| Metrica | Baseline submit-safe | V444 ckpt-2 | V448 ckpt-3 | Gate |
|---|---:|---:|---:|---|
| Total weak | `192/315` | `190/315` | `190/315` | reprova |
| `equation_transform` | `56/155` | `56/155` | `56/155` | reprova |
| `bit_manipulation` | `136/160` | `134/160` | `134/160` | reprova |
| `truncated` | `0` | `1` | `1` | reprova |
| Accuracy | `0.6095` | `0.6032` | `0.6032` | reprova |

Detalhe capturado dos logs HF:

| Familia | Rows | Correct | Accuracy | Truncated |
|---|---:|---:|---:|---:|
| `bit_manipulation` | `160` | `134` | `0.8375` | `1` |
| `equation_transform` | `155` | `56` | `0.3613` | `0` |
| Overall | `315` | `190` | `0.6032` | `1` |

## Decisao FinOps

O checkpoint-3 falhou todas as condicoes de promocao:

- nao superou `192/315`;
- nao superou `equation=56/155`;
- perdeu bit contra `136/160`;
- introduziu truncation.

O job foi cancelado antes de avaliar checkpoint-6. Nao fazer full eval, package
ou submit.

## Conclusao

V448 fechou a hipotese de que traces V447 clean, por si so, resolvem a
transferencia para adapter-only. A rota agora deve voltar para CPU:

1. auditoria estrita de metric/parser/raw-output;
2. equation DSL v2 com abstain e regra unica;
3. bit guardrail exact-binary;
4. novo GPU somente se houver sinal CPU diferente e material.
