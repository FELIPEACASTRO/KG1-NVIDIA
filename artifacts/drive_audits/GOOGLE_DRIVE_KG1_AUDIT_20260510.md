# Google Drive KG1 Audit - 2026-05-10

Objetivo: inventariar os diretorios KG1 do Google Drive e separar artefatos aproveitaveis de ruido operacional, com foco em `equation_transform` e `bit_manipulation`.

## Inventario

- Arquivo principal: `google_drive_kg1_inventory_latest.json`
- Files catalogados: `1879`
- Tamanho total: `301.446 GiB`
- Adapters completos: `85`
- Reports/manifests: `232`
- CSVs: `423`
- JSONLs: `54`
- Notebooks: `11`
- Zips: `29`

## Usar Com Prioridade

| Artefato | Evidencia | Uso |
|---|---|---|
| `KG1_NVIDIA_V226/.../checkpoint-1` | Weak `191/315`, equation `55/155`, bit `136/160`, trunc `0` | Publicar/validar no HF como initializer forte |
| `KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter` | Weak `190/315`, equation `54/155`, bit `136/160`, trunc `0` | Guardrail forte de bit e baseline historico |
| `KG1_NVIDIA_V217/.../final_adapter` | Weak `190/315`, equation `55/155`, bit `135/160`, trunc `0` | Comparativo, nao substituto |
| V230 miss packs | Oracle row-level `197/315`, equation `57/155`, bit `140/160` | Minerar regras e verifiers, nao deployar oracle |

## Rejeitar Como Promoção

| Linha | Motivo |
|---|---|
| V227 final/checkpoint | V229 weak agregado `16/315` |
| V206/V207 delta/answer-only | Abaixo do baseline e truncation alto |
| V214/V216/V217/V223 evals antigos | Scores `107-137/315` com truncation alto |
| V218/V219 decode rescue | `18/315` ou `6/315` |
| Public adapters Kienngx COT antigos | Truncation severo e weak baixo |
| V251/V252/V253 HF public adapters | Melhor `19/315`; desalinhados com gate canonico |

## Achados Diagnosticos

- V207A full/validation gate do V194: `822/947`, com familias nao criticas em `100%`, mas `equation_transform` segue `55/155` e `bit_manipulation` `135/160`.
- V225 equation-only decode: `think_strict_boxed` atingiu `56/155` em equation para V194/V217, um ganho pequeno mas concreto sobre `54-55/155`.
- V230-V238 confirmam que o caminho correto e parser/solver/verifier para equation simbolico/misto antes de qualquer treino longo.

## Proximo Uso Operacional

1. Transferir/publicar V226 checkpoint-1 e V194 protegido para HF com hashes, config e tensor-count verificados.
2. Rodar weak eval HF canonico nesses pesos fortes para eliminar dependencia do Drive.
3. Executar qualquer smoke training apenas partindo de initializer forte, com gate imediato.
4. Continuar mineracao CPU-only de `equation_transform` simbolico/misto usando V230 miss packs e evidencias V237/V238.
