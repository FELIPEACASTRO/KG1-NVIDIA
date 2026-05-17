# KG1 V535 Specialist Triple Check

Data: 2026-05-17

## Objetivo

Revisao cirurgica dos achados V531-V534 e do roadmap ativo pela otica de:

- QA/MLOps/FinOps;
- dados/ML/leakage;
- engenharia de competicao/submit-safe.

## Diagnostico Consolidado

## Oticas de Especialistas

| Otica | Pergunta revisada | Conclusao |
|---|---|---|
| QA / release gate | O artefato bloqueado falha fechado ou apenas escreve um warning? | Agora falha fechado quando chamado com `--fail-on-blocked`; validado com retorno esperado `2`. |
| MLOps / FinOps | Algum resultado justifica GPU paga? | Nao. V532 esta bloqueado (`hf_gpu_allowed=false`) e GPU continua proibida ate CPU produzir novo sinal source-only. |
| Data engineering | Algum arquivo row-level fraco ficou disponivel para consumo acidental? | CSV row-level V532 removido; CSVs de overlap weak do V533 removidos; permanecem apenas agregados. |
| ML / metricas | Loss, verifier score ou canonicalization conseguem substituir ACC? | Nao. O seletor com score alto perdeu `28` linhas; ACC label-free segue criterio soberano. |
| Submit-safe | Algum adapter externo ou candidate pool pode virar submit direto? | Nao. Yoiko/Huikang e pools externos ficam diagnostico/proveniencia ate decisao separada. |
| F2/backfire | As linhas de regressao conhecidas estao protegidas? | Roadmap protege `8740ed31=01101000` e `59bee375=10010101`; equation `518deb39=$` e tratado como smoke guard, nao como label de treino. |

| Area | Achado | Decisao |
|---|---|---|
| Leakage | O V532 gravava `answer` e `competition_match_audit_only` no CSV de decisoes, apesar de declarar esses campos como auditoria. | Corrigido: o CSV row-level V532 foi removido; weak labels ficam apenas para agregados de auditoria. |
| Bug silencioso | `baseline_label_free_prediction()` aceitava fallback silencioso para `prediction` quando `raw_output` faltava. | Corrigido: agora falha fechado, salvo flag explicita `--diagnostic-allow-stored-prediction`. |
| F2/backfire | V532 cai de `55/155` para `29/155`, com `2` ganhos e `28` perdas. | Rebaixado para diagnostico-only; nao e descoberta promocional de equation. |
| Verifier score | Muitas perdas do V532 tem `verifier_valid=True` e `verifier_score=1.0`; logo esse score local nao e suficiente para substituir baseline. | Roadmap exige abstain/manter baseline e prova rule-level/source-only antes de qualquer troca. |
| Thresholds | Roadmap misturava smoke anti-backfire, weak promocional e package. | Corrigido: smoke falha fechado em `total>=193/equation>=57/bit>=136/trunc=0`; weak promocional exige `>=196/equation>=60/bit>=136/trunc=0`; package exige full `>823/947`. |
| Adapters externos | Plano permitia avaliar adapters publicos, mas itens removidos diziam nunca usar peso publico. | Corrigido: Yoiko/Huikang adapter eval fica diagnostico/proveniencia; nao vira submissao direta sem decisao separada. |
| V532 scope | `selection_v2` e `solver_swap_v1` foram inventariados, mas nao usados como direct selector. | Marcado como escopo V535 separado: source-only rule/canonicalization audit, sem usar weak overlap como treino. |
| Huikang bit | Huikang tem `2000` traces sinteticas CHO/MAJ no ZIP local e `100` oficiais com `0` mismatch; weak overlap cobre `8` misses atuais. | Usar como referencia P0 de trace style, excluindo weak/full; nao copiar rows weak/full para treino. |
| Row-level weak proxy | Mesmo sem `answer`, CSV row-level com `id`, predicoes e correctness vira proxy do label weak. | Corrigido: V532 agora apaga `v532_external_equation_candidate_decisions.csv` e publica apenas diagnostico agregado por family/source. |
| Reproducibilidade | O V532 registrava apenas `zip_size` dos datasets Kaggle. | Corrigido: manifest agora inclui `zip_sha256` para cada zip baixado. |
| Guard bit | F2/backfire recorrente inclui `8740ed31` e `59bee375`, mas o guard citava apenas `8740ed31`. | Roadmap atualizado para proteger tambem `59bee375=10010101` em paid/promotional gates. |

## Estado Pos-Correcao

- `scripts/run_v532_external_equation_candidate_gate.py` compila.
- V532 foi rerodado e regenerou:
  - `v532_external_equation_candidate_family_diagnostics.csv`;
  - `v532_external_equation_candidate_summary.csv`;
  - `v532_external_equation_candidate_manifest.json`;
  - `KG1_V532_EXTERNAL_EQUATION_CANDIDATE_GATE.md`.
- `v532_external_equation_candidate_decisions.csv` foi apagado porque
  expunha weak row-level labels por proxy.
- O resultado numerico nao mudou: `29/155` selecionado vs `55/155` baseline,
  `2` ganhos, `28` perdas.
- O status operacional mudou: V532 e diagnostico-only e nao pode alimentar
  treino/submit por weak gain rows.
- Os CSVs row-level de overlap weak do V533/Huikang tambem foram removidos da
  arvore ativa; fica apenas resumo agregado sem labels por linha.
- O modo bloqueante foi validado localmente:
  `python scripts/run_v532_external_equation_candidate_gate.py --fail-on-blocked`
  retornou `2`, como esperado, e o wrapper de validacao tratou esse retorno
  como sucesso do fail-closed.

## Plano Certeiro Atual

1. **Bit primeiro, CPU-only:** construir V534 source-only com Konbu
   high-confidence + Huikang CHO/MAJ, excluindo weak/full por `id`,
   `prompt_sha256` e `prompt+answer_sha256`.
2. **Equation depois, CPU-only:** usar V532 apenas para estudar falhas de
   canonicalization/ranking e construir hard negatives source-only; nao usar
   candidate pool como patch.
3. **Adapters externos:** apenas diagnostico barato de compatibilidade/ACC; se
   houver sinal, abrir decisao separada. Nao submeter peso publico diretamente.
4. **GPU:** bloqueada ate CPU provar novo sinal aprendivel, com zero leakage e
   gates de tokenizacao/offset/trace.

## Regra F2/Backfire Atualizada

Qualquer nova rota que:

- aumente equation mas reduza bit;
- reduza loss sem melhorar ACC;
- use label-aware extraction;
- dependa de weak/full labels;
- ou substitua baseline por candidate sem abstain/no-loss proof

entra em crisis mode: bloquear, registrar, corrigir gate e nao gastar GPU.
