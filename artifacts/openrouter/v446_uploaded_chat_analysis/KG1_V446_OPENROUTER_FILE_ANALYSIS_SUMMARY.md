# KG1 V446 OpenRouter Uploaded File Analysis

Arquivo analisado: `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026 (1).json`

Data da analise: 2026-05-15

## Escopo

O export contem 16 respostas/slots de modelos. Desses, 11 tem resposta final
utilizavel; 5 ficaram apenas em reasoning/web-search/truncamento e nao foram
usados como base de decisao.

Respostas finais extraidas em:

- `artifacts/openrouter/v446_uploaded_chat_analysis/responses/manifest.json`
- `artifacts/openrouter/v446_uploaded_chat_analysis/responses/*.md`

## Consenso Util

| Tema | Conclusao |
|---|---|
| FinOps e gates | Corretos e devem ficar estritos. Loss/eval_loss/preference accuracy nao promovem. |
| Proxima GPU | Nao abrir H200/A100 sem novo artefato CPU que prove sinal material. |
| Gargalo real | Transferencia para LoRA: o conhecimento CPU/solver nao virou comportamento autoregressivo estavel. |
| `equation_transform` | Precisa de alvo treinavel com estado intermediario: trace canonico, DSL composicional ou trajetoria nativa verificada. |
| `bit_manipulation` | Deve ser preservado por replay/anchor; nao tentar relearn amplo. |
| Parser/extractor | V445 ja fechou essa rota: re-extrair boxed nao muda `192/56/136/0`. |
| SFT amplo/preference | Fechados ate existir dado novo; repetem teto `equation=56` e risco de `bit<136`. |

## Achados Acionaveis

1. Adicionar gate de alinhamento de distribuicao do target antes de GPU.
   Um target so entra se for uma continuacao plausivel para o modelo/base:
   target nativo por rejection sampling ou target canonico com logprob/score
   pre-registrado contra a base/V291.

2. Implementar duas rotas CPU concorrentes, ambas sem weak/full como treino:
   - DSL v2 composicional com certificacao, LOO, renaming/metamorphic checks.
   - Native rejection sampling em public train: gerar varias trajetorias,
     manter somente as que o verificador/label publico confirma como corretas,
     curtas e sem truncation risk.

3. Qualquer dataset equation precisa vir com replay/anchor de bit.
   Piso minimo exploratorio: `>=200` bit-preserve rows. Para H200 final, alvo
   preferido: `>=800` anchors se houver disponibilidade limpa.

4. Adicionar leakage forte alem de SHA:
   denylist por `id`, `prompt_sha256`, prompt normalizado e `13-gram` contra
   weak/full antes de qualquer treino.

5. Pre-registrar mistura/family ratio e kill-switch.
   Nao permitir ajuste apos olhar weak/full. Primeiro checkpoint deve manter:
   `equation>=58`, `bit>=136`, `truncated=0`; caso contrario cancelar.

6. Validar o loader antes de gastar GPU.
   Um scaffold LoRA nao treinado deve carregar via caminho oficial vLLM/LoRA e
   gerar em prompt dummy antes do launcher pago.

## Rejeicoes

| Sugestao | Decisao |
|---|---|
| Treinar em weak misses/120 misses como fonte | Rejeitado: leakage/cherry-pick. Weak/full sao somente gate. |
| Parser/verifier como parte do adapter | Rejeitado: submit exige adapter-only, sem runtime externo. |
| Broad SFT, mais epochs, LR sweep | Rejeitado: ja falhou e nao cria novo sinal. |
| Modulo/layer sweep sem dado certificado | Rejeitado por agora: custo sem evidencia. P2 apenas apos target gate passar. |
| Synthetic generico sem certificacao | Rejeitado: risco de ruido e regressao. |
| Relaxar bit para 134/135 | Rejeitado: baseline submit-safe exige `bit>=136`. |

## Decisao

O roadmap deve seguir CPU-first. O proximo artefato implementavel nao e outro
treino, e sim um `target_alignment_gate` que diga se existe target limpo,
plausivel e submit-safe antes de gastar H200.

Se DSL v2 e native rejection sampling nao gerarem cobertura/precisao suficiente,
o resultado honesto e manter V291/V290 checkpoint-6 como teto adapter-only atual.
