# KG1 Score Improvement Roadmap

Atualizado: 2026-05-14

Este arquivo agora e o roadmap ativo e limpo. O historico detalhado anterior foi arquivado em `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V379_CLEANUP_2026_05_14.md`.

## Objetivo

Melhorar `equation_transform` e `bit_manipulation` sem regressao no adapter-only submit.

Meta minima para gastar HF:

- Manter `bit_manipulation >= 136/160`.
- Sair do teto adapter-only `equation_transform=56/155`.
- Primeiro checkpoint deve manter `total >= 192/315`, `truncated=0`.
- Se qualquer job nao puder mais bater o gate, cancelar por FinOps.
- A partir do V392, novo treino LoRA so pode iniciar se existir prova previa de transferencia no proprio caminho adapter/package. Projecao CPU/teacher isolada nao autoriza GPU.

## Regra Operacional Agressiva 2026-05-14

Precisamos buscar subida no ranking ainda hoje, `2026-05-14`. A decisao V392 e ser agressivo no que tem maior chance de virar submit hoje e cortar o que ja falhou repetidamente.

- Prioridade maxima: ganho adapter-only submetivel medido em weak/full gate, nao `eval_loss` isolado.
- Caminho de hoje: partir do melhor adapter/package conhecido e procurar ganho por prompt/template/extractor/decoding e selecao de checkpoint, antes de qualquer novo treino.
- Treino LoRA fica pausado ate um gate demonstrar que o proprio modelo muda respostas em `equation_transform` sem perder bit. Dataset teacher, solver/verifier ou postprocessor que so melhora fora do adapter nao basta.
- Se algum sweep sem treino bater `total > 192`, `equation_transform > 56`, `bit_manipulation >= 136` e `truncated=0`, promover para full/official-like eval e package gate no mesmo dia.
- Se o sweep sem treino nao bater o baseline, o proximo passo e auditoria row-level dos misses para descobrir exatamente quais `4` equation rows podem ser convertidas em ganho adapter-only; nao iniciar HF training para "ver se aprende".
- Proibido submit de postprocessor/verifier/teacher-only. Submit hoje so pode vir de adapter/package que passe os gates.
- Regra permanente: toda nova versao criada deve incluir um quadro comparativo contra a versao anterior, com metricas, delta e decisao de promocao/rejeicao.

## Estado Atual Medido

| Estado | Total weak | equation_transform | bit_manipulation | Status |
|---|---:|---:|---:|---|
| Melhor adapter-only | `192/315` | `56/155` | `136/160` | baseline real para submit |
| V274/V275 postprocessor/verifier | `196/315` | `60/155` | `136/160` | nao e adapter-only; usar como teacher/diagnostico |
| V366/V336 CPU teacher/verifier | `222/315` | `63/155` | `159/160` | nao submit-safe; melhor teacher CPU |
| V372 HF trace-style smoke | `191/315` | `56/155` | `135/160` | rejeitado; nao continuar |
| V375 equation residual clustering | `92` eq misses restantes | `82 symbolic_punct`, `10 numeric_operator` | n/a | diagnostico |
| V378 solver parquet sobre V375 | `82/92` cobertos | `79/82` corretos | n/a | melhor novo sinal |
| V380 oracle diagnostic | `301/315` | `142/155` | `159/160` | nao submit-safe; usa resposta/teacher |
| V380 reexecuted teacher | `292/315` | `133/155` | `159/160` | dataset teacher permitido; HF ainda bloqueado |
| V380 strict independent | `222/315` | `63/155` | `159/160` | ganho real independente `+0`; nao submeter |
| V381 filtered teacher dataset | n/a | `840` eq sintéticas | `280` bit replay | passou dataset + tokenization gate real; pronto para micro-train HF |
| V382/V383 V381 teacher smoke | `191/315` melhor parcial | `56/155` | `135/160` | rejeitado; checkpoints 2/4/6 nao bateram baseline; V383 cancelado por FinOps |
| V384 V382 V221 prompt weak eval | melhor `193/315` | `56/155` | `137/160` | rejeitado; `truncated=1` e equation nao subiu |
| V387 V382 checkpoint-4 full official-like | full `823/947` | `56/155` | `135/160` | rejeitado; empatou V291 e falhou package gate `>=824/947` |
| V388/V389 V291/V382 adapter soups | melhor `191/315` | `56/155` | `135/160` | rejeitado; todos os soups regrediram bit/total e nenhum moveu equation |
| V390 CPU equation gate + bit replay mix | projecao CPU `198/315` | `62/155` | `136/160` guardrail | autorizado para smoke HF curto; ainda nao e adapter-only medido |
| V390 A100 runtime attempt | n/a | n/a | n/a | bloqueado corretamente por gate: CUDA 13 em A100 |
| V391 H200 relaunch + weak eval | `191/315` | `56/155` | `135/160` | rejeitado; checkpoints 2/4 iguais, eval cancelado por FinOps |
| V392 roadmap reset | n/a | n/a | n/a | pausar LoRA; priorizar baseline lock + sweep sem treino + gate de transferencia real |
| V393 prompt/template sweep | melhor `192/315` | `56/155` | `136/160` | encerrado por FinOps; sem ganho sobre V392 lock |
| V394 equation row inventory | CPU projection `198/315` | `62/155` | `136/160` | sem sinal novo vs V390; nao autoriza GPU |
| V395 HF CPU aggressive symbolic gate | CPU integrated `199/315` | `63/155` | `136/160` | confirma sinal V336/V343; `0` ganho novo adapter-only; GPU ainda bloqueada |
| V396 Google Drive artifact audit | n/a | n/a | n/a | Drive contem artefatos validos de linhagem/diagnostico, mas nenhum supera V291/V290 adapter-only |
| V397 reconstructed SFT transfer dataset | n/a | n/a | n/a | novo corpus adapter-transfer: `2578` train / `264` val, weak overlap `0`, tokenization real passou com `0` truncation |
| V398 reconstructed SFT H200 smoke | melhor `191/315` | `56/155` | `135/160` | rejeitado; nao transferiu, perdeu bit e nao moveu equation |
| V399 V398 pairwise CPU audit | melhor `191/315` | `56/155` | `135/160` | V398 tem `0` candidate-only equation e so perde bit; encerrar V397/V398 |
| V400 algorithmic prompt sweep | melhor `175/315` | `40/155` | `135/160` | rejeitado; prompt algoritmico explicito causou truncation e colapso de ACC |
| V401 baseline raw-output audit | n/a | `0` boxed recoverable | `0` boxed recoverable | sem ganho; misses nao sao erro de extrator simples |
| V402 local candidate scoreboard | weak max `192/315`; full max `823/947` | `56/155` | `136/160` weak | sem candidato local acima do baseline V291/V290 |
| V403 formal solver abstain audit | CPU projection `+2` bit | `0` ganho equation v2 | `136 -> 138` se postprocess externo fosse permitido | sinal solver limpo, nao submit-safe; usar so como fixture/trace |
| V404 expanded symbolic cryptarithm audit | CPU projection `+1` equation | `56 -> 57` isolado | n/a | reconfirma o unico ganho simbolico conhecido `99d6a3b5`; sem nova classe segura |
| V405 integrated solver projection | CPU projection `201/315` | `63/155` | `138/160` | melhor combinacao solver-first com abstain; nao adapter-only submit-safe |

Conclusao: `eval_loss` baixo nao e criterio de promocao. O criterio e ACC por familia no weak/full gate. A rota "resolver nos mesmos" finalmente tem ganho mensuravel (`+9` weak em CPU), mas esse ganho ainda e solver/verifier externo; para submit, ele precisa virar comportamento do adapter/package ou ser permitido explicitamente pelas regras de runtime.

Resultado V383 em `2026-05-14`: checkpoint-2 = `190/315`, `equation=56`, `bit=134`, `truncated=1`; checkpoint-4 = `191/315`, `equation=56`, `bit=135`, `truncated=1`; checkpoint-6 = `190/315`, `equation=56`, `bit=134`, `truncated=1`. Como nenhum dos tres podia superar o baseline `192/315`, `equation=56`, `bit=136`, `truncated=0`, o job foi cancelado antes de avaliar `checkpoint-8/10`.

Auditoria V385 de medicao ACC em `2026-05-14`: o weak scorer atual esta correto para comparar candidatos adapter-only. O CSV weak validado pelo proprio runner tem `315` rows, `160` bit, `155` equation, SHA `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6` e contrato `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`. O merge e `one_to_one` por `id`, a familia vem do CSV de solucao, `bit_manipulation` usa igualdade binaria exata e truncation vem de `finish_reason == "length"`. Gap encontrado: V383 usou sufixo curto e e diagnostico; V384 e a comparacao correta contra o prompt historico V221.

Double check V386 de medicao ACC em `2026-05-14`: weak315 e full947 foram cruzados contra o `train.csv` oficial baixado via Kaggle CLI; `id`, `prompt` e `answer` bateram com `0` ausentes e `0` mismatches. O re-score local dos CSVs V384 baixados do HF reproduziu exatamente o `batch_candidate_summary`: `v382_ckpt4_v221prompt = 193/315, equation=56, bit=137, truncated=1`; `v382_ckpt6_v221prompt = 190/315, equation=55, bit=135, truncated=1`. Conclusao: o baixo ACC atual nao e bug de medicao nem sujeira de dataset; e falha real do candidato. A linha V381/V382/V384 nao deve ser promovida.

Decisao agressiva V387 para ranking hoje: apesar de V384 falhar no weak gate por `truncated=1` e `equation=56`, o checkpoint-4 era o unico adapter-only novo com `total=193/315` e `bit=137/160`. O full official-like V387 em H200 terminou com `823/947`, `equation_transform=56/155`, `bit_manipulation=135/160`, `truncated=1`, `full_candidate_gate=False`. Isso empata a V291, nao melhora ranking e nao autoriza package/submission. A linha V381/V382/V384/V387 esta encerrada para submissao.

Decisao V388/V389 em `2026-05-14`: os soups adapter-only V291/V382 foram gerados em CPU e avaliados no weak gate H200. Resultados: `0.95/0.05 = 191/315, equation=56, bit=135, truncated=2`; `0.90/0.10 = 190/315, equation=56, bit=134, truncated=1`; `1.05/-0.05 = 191/315, equation=56, bit=135, truncated=1`. Nenhum passou `total>192`, `equation>56`, `bit>=136`, `truncated=0`. Conclusao: combinacao linear simples entre V291 e V382 nao gera ganho submit-safe e esta encerrada.

Decisao V390 em `2026-05-14`: o V324 CPU gate encontrou `6` novos ganhos equation sem conflitos (`equation` projetado `56 -> 62`, weak projetado `192 -> 198`) e preservou `bit=136`. O dataset V390/V326 tem `5031` train rows (`4231` bit replay + `800` equation) e `532` validation rows, passou tokenization gate com `0` truncation e offset mask completo. O smoke HF deve ser curto e agressivo (`12` steps), com weak eval nos checkpoints `2/4/6/8/10/12`. Promocao so se `total>192`, `equation>56`, `bit>=136`, `truncated=0`; caso contrario cancelar/encerrar por FinOps.

Runtime V390/V391 em `2026-05-14`: o primeiro launch V390 em A100 foi barrado pelo preflight (`torch cuda=13.0`, `NVIDIA A100-SXM4-80GB`). Isso confirma que o gate novo esta correto e evita gasto inutil. A continuacao autorizada e V391 em H200, mantendo o mesmo dataset e os mesmos thresholds; nao repetir A100 com CUDA 13.

Resultado V391 em `2026-05-14`: o H200 treinou a receita V390 com `12` steps. O weak eval V221-contract mediu `checkpoint-2 = 191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, `truncated=0`; `checkpoint-4 = 191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, `truncated=0`. Como ambos ficaram abaixo do melhor adapter-only (`192/315`, `equation=56`, `bit=136`) e nenhum sinal de equation apareceu, o eval foi cancelado antes de gastar com os checkpoints seguintes. Conclusao: a projecao CPU V390 (`198/315`, `equation=62`) nao transferiu para LoRA.

Decisao operacional pos-V387: usar Kaggle GPU apenas como alternativa barata para validacao ou fallback, nao para repetir treino SFT amplo. O proximo gasto em HF/Kaggle GPU precisa vir depois de CPU gate novo que mostre `equation>56`, `bit>=136` e `truncated=0` no weak, ou de um candidato full official-like com expectativa objetiva de `>=824/947`.

Decisao V392: nao ha justificativa tecnica para continuar a linha "teacher/verifier -> SFT LoRA" sem um gate novo que prove transferencia de resposta. V382/V383, V384/V387, V388/V389 e V391 repetiram o mesmo padrao: loss/teacher melhora ou projecao CPU parece boa, mas o adapter continua em `equation=56` e frequentemente perde `bit`. O plano ativo passa a privilegiar o caminho mais rapido para ranking hoje: identificar o melhor package historico, fazer sweeps sem treino, e so voltar ao treino se o gate mostrar ganho adapter-only antes da GPU.

Decisao V393 em `2026-05-14`: o sweep sem treino foi encerrado por FinOps. `v221_boxed_suffix` empatou o baseline travado (`192/315`, `equation=56`, `bit=136`, `truncated=0`), portanto nao gera novo submit. `no_suffix` regrediu severamente (`158/315`, `equation=55`, `bit=103`, `truncated=1`), provando que remover a instrucao boxed quebra `bit_manipulation` e nao melhora equation. As variantes restantes (`strict_disable_thinking`, `strict_2048_tokens`) foram canceladas porque, apos esse resultado, a chance de superar `equation>56` com `bit>=136` nao justificava continuar gastando H200.

Decisao V394 em `2026-05-14`: o inventario row-level sobre o baseline travado V290 checkpoint-6 confirmou `99` misses de `equation_transform`: `16` numeric operator e `83` symbolic punctuation. O V324 achou `6` ganhos CPU/verifier (`274def88`, `528ec0d8`, `7688e06e`, `c5b058d6`, `d1bd7478`, `fb623471`), projetando `equation=62/155` e `weak=198/315` com `bit=136/160`. Double check contra V390: `0` IDs novos, mesma projecao `62`, mesmo guardrail `136`. Como V391 ja testou esse sinal em LoRA e nao transferiu, V394 nao autoriza novo HF job. A proxima acao e expandir DSL simbolica nos `93` misses restantes, especialmente `equation_symbolic_punct`.

Decisao V395 em `2026-05-14`: o job HF CPU `https://huggingface.co/jobs/felipesp1983/6a0647a9e48bea4538b9d78a` rodou em `cpu-upgrade` (`8 vCPU`, `32 GB`, `US$0.0005/min`) e completou. O V324 agressivo nao achou IDs novos alem dos `6` numericos conhecidos. O V329 amplo reconfirmou o unico ganho simbolico conhecido `99d6a3b5`. A integracao no-loss ficou `199/315`, `equation=63/155`, `bit=136/160`, `losses=0`, mas e sinal solver/verifier ja conhecido de V336/V343, nao ganho adapter-only. Portanto V395 nao autoriza treino GPU repetindo a mesma transferencia. Os artefatos foram enviados para `felipesp1983/kg1-v395-cpu-symbolic-gate-artifacts`.

Decisao V396 em `2026-05-14`: a auditoria seletiva no Google Drive confirmou que existem muitos artefatos validos, mas nenhum novo adapter-only melhor que V291/V290. V221 e V226 continuam validos como registros de candidatos (`190/315` e `191/315`), V230-V238 continuam uteis como miss packs/parser/solver diagnostics, e V199 e uma linhagem historica production-ready com ZIP validado. Em contrapartida, V227/V228/V229 foram descartados por regressao severa (`16/315` no V229 final; V228 com `111/160` truncados na primeira janela), e os adapters publicos em `KG1_PUBLIC_ADAPTERS` nao sao drop-in: Huikang falha por incompatibilidade de modulos LoRA e Kienngx 3000 fica em `32/315`. O resumo versionado esta em `artifacts/v396_drive_artifact_audit/KG1_V396_GOOGLE_DRIVE_ARTIFACT_AUDIT.md`. Nao ha autorizacao de novo treino GPU a partir do Drive.

Decisao V397 em `2026-05-14`: o `sft_reconstructed.jsonl` local foi auditado contra `competition_train.csv`: `9500/9500` finais extraidos batem com os labels oficiais. Excluindo os `315` rows do weak gate, restam `2578` train e `264` validation focados em `bit_manipulation` e `equation_transform`. O gate real V286 passou com `0` overlap train/val, `0` weak-id leakage, `0` prompt truncation e `0` completion truncation em `max_length=8192`. Este e o primeiro corpus recente realmente diferente da linha V381/V391; ele autoriza apenas um smoke H200 curto V398, com `4` steps e weak eval nos checkpoints `2/4`. Promocao somente se `weak total > 192`, `equation > 56`, `bit >= 136`, `truncated=0`; caso contrario cancelar/encerrar por FinOps.

Resultado V398 em `2026-05-14`: treino H200 curto a partir do V290 checkpoint-6 completou, mas falhou no weak gate. `checkpoint-2 = 190/315`, `equation=56`, `bit=134`, `truncated=1`; `checkpoint-4 = 191/315`, `equation=56`, `bit=135`, `truncated=0`. Ambos ficam abaixo do baseline adapter-only `192/315`, `equation=56`, `bit=136`, `truncated=0`. Decisao: nao promover, nao fazer full eval, nao fazer submit e nao alongar V397/V398. O quadro comparativo esta em `artifacts/v398_hf_nemo_h200_sft_reconstructed_launch/V398_VS_PREVIOUS.md`.

Decisao V399 em `2026-05-14`: a auditoria pairwise CPU comparou V398 checkpoint-2/4 contra o baseline travado V290 checkpoint-6 (`192/315`, `equation=56`, `bit=136`, `truncated=0`). Resultado: checkpoint-2 tem `0` acertos novos de `equation_transform`, perde `2` bit e trunca `1`; checkpoint-4 tem `0` acertos novos de `equation_transform`, perde `1` bit e nao trunca. Portanto nao existe nem row-level complementaridade para minerar. A linha V397/V398 esta encerrada definitivamente. Artefatos: `artifacts/v399_v398_pairwise_complementarity/20260514T_v399_pairwise/V399_V398_PAIRWISE_COMPLEMENTARITY.md`.

Decisao V400 em `2026-05-14`: o sweep H200 sem treino testou dois prompts algoritmicos curtos sobre o baseline V290 checkpoint-6. `symbolic_equation_first = 175/315`, `equation=40`, `bit=135`, `truncated=27`; `bit_stride_guarded = 7/315`, `equation=7`, `bit=0`, `truncated=227`. Conclusao: colocar a literatura/DSL diretamente no prompt e contraproducente; induz geracao longa e quebra formato. Encerrar sweeps de prompt algoritmico amplo. Artefato comparativo: `artifacts/v400_hf_h200_algorithmic_prompt_sweep_launch/V400_VS_BASELINE.md`.

Decisao V401 em `2026-05-14`: auditoria CPU nos `123` misses do baseline V290 checkpoint-6 verificou se a resposta correta ja estava em `raw_output` mas perdida pela extracao. Resultado: `0` respostas corretas em simple `\boxed{}` nos misses de `equation_transform` e `0` em `bit_manipulation`. Existem ocorrencias brutas (`19` equation, `4` bit), mas spot-check mostra caracteres/numeros em raciocinio ou exemplos intermediarios, nao resposta final recuperavel. Conclusao: nao ha ganho submit-safe por trocar extrator; o gargalo e geracao do adapter. Artefato: `artifacts/v401_baseline_raw_output_audit/20260514T_v401_raw_output/V401_BASELINE_RAW_OUTPUT_AUDIT.md`.

Decisao V402 em `2026-05-14`: varredura CPU dos `batch_candidate_summary.json` locais nao encontrou candidato historico acima do baseline. Melhor weak local: empate `192/315`, `equation=56`, `bit=136`, `truncated=0` (`v321_hybrid_attn_lmhead_checkpoint_2_v221_contract` e `v290_checkpoint_6_v221_contract`). Melhor full-like local: V291 `823/947`, `truncated=1`. Conclusao: nao ha package local esquecido que suba ranking hoje sem novo sinal. Artefato: `artifacts/v402_local_candidate_scoreboard/20260514T_v402_scoreboard/V402_LOCAL_CANDIDATE_SCOREBOARD.md`.

Decisao V403 em `2026-05-14`: a rota "resolver nos mesmos" foi formalizada como solver-first com abstain. O solver bit global exato encontrou `112` rows aceitas no weak, com `2` ganhos e `0` perdas quando limitado a regras byte-globais exatas. Ganhos: `4ada9150` (`01111111 -> 01111011`, `OR(ROL2, SHL4)`) e `4c327b55` (`11011110 -> 11011100`, `XOR(SHL1, SHR4)`). O mesmo solver, se usar `CONSENSUS`/`UNSOLVED`, gera `14` perdas e portanto fica proibido. O parser antigo `equation_solver_v2` nao produziu ganhos independentes em equation. Conclusao: SyGuS/CEGIS/solver com abstain e a direcao correta, mas V403 ainda e sinal CPU/postprocessor, nao ganho adapter-only submitavel. Artefato: `artifacts/v403_formal_solver_abstain_audit/20260514T_v403_solver_abstain/V403_FORMAL_SOLVER_ABSTAIN_AUDIT.md`.

Decisao V404 em `2026-05-14`: a auditoria expandida de symbolic cryptarithm varreu `83` misses simbolicos. Foram gerados `4` candidatos; apenas `1` passou (`99d6a3b5`, `(<)) -> ?()<`, classe `symbolic_cryptarithm_v404_mul`) e `3` falharam antes do class gate (`abs_diff`, `concat_ba`, `sub_ba`). Conclusao: o simbolico seguro atual continua sendo somente o ganho ja conhecido de V329; regras simbolicas fracas nao devem ser promovidas por gerarem perdas. Artefato: `artifacts/v404_expanded_symbolic_cryptarithm_audit/20260514T_v404_symbolic_cryptarithm/V404_EXPANDED_SYMBOLIC_CRYPTARITHM_AUDIT.md`.

Decisao V405 em `2026-05-14`: a integracao no-loss de V324 + V329 + V403 produziu a melhor projecao solver-first ate agora: `201/315` weak (`+9`), `equation_transform=63/155` (`+7`), `bit_manipulation=138/160` (`+2`), `0` conflitos. Ganhos aceitos: `274def88`, `528ec0d8`, `7688e06e`, `99d6a3b5`, `c5b058d6`, `d1bd7478`, `fb623471` em equation e `4ada9150`, `4c327b55` em bit. Isto prova que ha acertos recuperaveis por sintese formal com abstain. Tambem prova o limite operacional: nao e submit adapter-only. Proximo passo do roadmap: transformar essas `9` regras aceitas em fixtures/traces curtos para um teste minimo de transferencia adapter-only; se o primeiro checkpoint nao superar `192/315` mantendo `bit>=136` e `truncated=0`, abortar por FinOps. Artefato: `artifacts/v405_integrated_solver_projection/20260514T_v405_integrated_projection/V405_INTEGRATED_SOLVER_PROJECTION.md`.

## Google Drive Artifact Audit 2026-05-14

Achados que entram no plano:

- `KG1_NVIDIA_V221`: registry weak AB com V194 `190/315`, V217 `190/315`, Kienngx `183/315`; util para oracle/complementarity, nao para novo submit.
- `KG1_NVIDIA_V226`: checkpoint-1 `191/315`, bit `136`, equation `55`; predecessor direto usado em V230.
- `KG1_NVIDIA_V230`: complementarity e miss packs; oracle row-level melhora, mas nao passa weak gate.
- `KG1_NVIDIA_V231` a `V238`: workbench/parser/solver diagnostics; no maximo `1` row deployable em V238, insuficiente para submit.
- `KG1_NVIDIA_V199`: `final_submit_doublecheck.json` valida um ZIP historico com layout root-only, rank `32`, target modules completos e `12011` tensors. Fica como linhagem/auditoria, nao como baseline atual.

Achados descartados:

- `KG1_NVIDIA_V227/V228/V229`: regressao de prompt/treino; V229 final `16/315`, eq `9`, bit `7`.
- `KG1_PUBLIC_ADAPTERS/huikang_default_v20`: falha de carregamento vLLM por target modules de mixer incompativeis.
- `KG1_PUBLIC_ADAPTERS/kienngx_cot_labels_3000samples_adapter`: `32/315`, truncation `202/315`.
- `Submit/submission.zip`: historico e anterior ao package V291; nao substitui o baseline local travado.

Impacto: o Drive melhora a confianca na linhagem e evita repetir linhas ruins, mas nao oferece ganho adapter-only novo. O proximo gasto em GPU continua bloqueado ate existir prova de transferencia adapter-only.

## Fontes Web Reauditadas 2026-05-14

- `tonghuikang/nemotron`: repo publico da submissao Progress Prize. Ele confirma que a solucao vencedora nao foi treino generico; usou reasoners, corpus, token-level traces, metricas por categoria e treino SFT controlado. O que entra no plano: copiar o metodo de instrumentacao e gates, nao copiar bruto sem validacao.
- `tonghuikang/nemotron/reasoners/equation_numeric.py`: fonte concreta para DSL de equation numeric. Operacoes uteis: concatenacao, reverse concatenation, soma, diferenca absoluta, subtracao nos dois sentidos, multiplicacao, `+1/-1`, divisao inteira, modulo, reverse division/modulo, operacoes por digito, cross multiply, determinante e abs determinant. Isso vira inventario row-level de misses, nao SFT amplo.
- `tonghuikang/nemotron/reasoners/bit_manipulation.py` e discussao associada: confirma abordagem de bit por relacoes de bits, familias unary/binary/constant e stride. Como nossa linha "Tong bit direct replacement" ja caiu no V374, o uso correto agora e gerar traces curtos/verificaveis e probes, nao substituir o solver nem treinar bruto.
- `nvidia/Nemotron-RL-ReasoningGym-v1`: dataset oficial NVIDIA com `15000` amostras em `104` ambientes procedurais/verificaveis, licenca CC-BY-4.0. Uso permitido: fixtures/probes e sanity checks de raciocinio; nao entra em treino direto do desafio sem gate anti-overlap e prova de ganho nas familias alvo.
- `NVIDIA-NeMo/Nemotron`: hub de receitas/datasets Nemotron. Ajuda em runtime/infra e confirma valor de dados verificaveis; nao fornece, sozinho, ganho row-level para o submit.

## Literatura Solver/Verifier Reauditada 2026-05-14

Filtro usado: entra no roadmap somente estudo que mude uma decisao tecnica para `bit_manipulation`, `equation_transform` ou gate solver/verifier. A revisao inclui trabalhos e labs asiaticos, mas a conclusao operacional nao depende de nacionalidade do paper: precisamos de sintese verificavel em CPU antes de GPU.

| Fonte | Instituicao/lab | Achado operacional | Acao no roadmap |
|---|---|---|---|
| Kaggle/Tong Hui Kang + `tonghuikang/nemotron` | competidor Progress Prize | A solucao forte combina reasoners, corpus, traces e metricas por categoria; nao e "SFT amplo". | Manter V394/V395 como inventario row-level + DSL/verifier antes de qualquer treino. |
| SyGuS / syntax-guided synthesis | comunidade PL/SMT | Restringir a gramatica torna sintese tratavel e melhora otimizacao. | `equation_transform` deve usar DSL pequena e verificada, nao busca livre nem prompt generico. |
| Program Synthesis via Bi-directional Reduced-product Abstract Interpretation | Seoul National University/KAIST line of work | Busca bidirecional e interpretacao abstrata reduzem espaco de programas por restricoes de entrada e saida. | Em `bit`, inferir dependencia output-bit -> input-bit/pares/constantes antes de enumerar expressoes. |
| Euphony / learned probabilistic search for synthesis | KAIST, UPenn, Mayur Naik line | Em benchmarks SyGuS, inclui `750` tarefas BitVec e usa gramatica + modelo probabilistico para guiar busca em vez de enumeracao cega. | V395 deve ordenar candidatos bit por gramatica/probabilidade e validar por exemplos, nao tentar prompt ou treino amplo. |
| Math-Shepherd | Peking/Tsinghua/DeepSeek lineage | Verificador/process reward melhora reranking e RL quando ha multiplas saidas e supervisao automatica. | Usar teacher/verifier para escolher candidatos e gerar hard negatives; nao contar como ganho submetivel sem adapter gate. |
| Cumulative Reasoning | Tsinghua University, Andrew Yao/Yang Yuan team | Arquitetura proposer/verifier/reporter valida passos antes de acumular contexto, com ganhos em logica, Game of 24 e MATH. | Para `equation`, separar gerador de candidatos, verificador simbolico e decisor; nao confiar em uma unica amostra do modelo. |
| InternLM-Math | Shanghai AI Laboratory | Unifica CoT, reward model, formal reasoning, data augmentation e code interpreter como solver/verifier/prover/augmenter. | Para `equation`, priorizar code/DSL verifier e data augmentation validada, nao mais epochs. |
| DeepSeekMath/DeepSeek-Prover | DeepSeek AI, pesquisadores chineses | Ganho matematico vem de corpus filtrado, RL/verificacao e/ou proof feedback; answer final sozinho nao prova transferencia. | Se houver novo LoRA, dataset precisa ter resposta correta, baseline errada, trace curto deterministico e hard negative. |
| VERSE/PLSE@NUS | National University of Singapore | Programa de pesquisa em sintese com garantias, verificação multimodal e Lean confirma que corretude precisa ser checada por ferramenta, nao inferida por texto. | O V394 deve emitir certificado simples por row: operador/DSL escolhido, execucao nos exemplos e resposta prevista. |
| DreamCoder / neural-guided synthesis / HYSYNTH | MIT/UCSD/general PL literature | LLM pode guiar busca, mas programa final precisa ser checado em DSL. | LLM/OpenRouter serve para propor operadores e priorizar busca; CPU verifier decide. |

Decisao honesta: nenhuma literatura revisada autoriza "treinar mais" como proxima acao. O caminho de maior chance para subir hoje e:

1. V393 esta encerrado: prompt/template sweep nao gerou ganho submit-safe.
2. executar V394 inventario row-level de `equation` com DSL expandida e certificado por row;
3. executar V397 guardrail de `bit` por bit-pair/bitsum/stride, com ordenacao de busca inspirada em BitVec synthesis;
4. so abrir novo HF/Kaggle GPU se CPU gate mostrar `equation>56`, `bit>=136`, `truncated=0` ou uma variante prompt-safe com expectativa objetiva de full `>=824/947`.

## Fontes Auditadas

### Fonte ativa 1 - `solver_results.parquet`

Arquivo original: `C:\Users\davis\Downloads\nemotron_dataset_final\solver_results.parquet`

Status: o parquet bruto nao deve ser tratado como dependencia operacional atual. A evidencia versionada e reproduzivel no repo esta nos CSVs derivados V378/V380.

Evidencia:

- `823` rows de `equation_transform`.
- `800/823` corretas pelo scorer do projeto.
- `741/823` rows tem `conditioned_on_answer=True`; sao evidencia de reparo, nao prova independente.
- `82/823` rows nao foram condicionadas na resposta; este e o subconjunto preferencial para promocao de regra.
- Cobre `82/92` residuos V375.
- `79/82` corretos nesses residuos.
- Categorias: `arithmetic`, `little_endian`, `mixed_concat`, `pure_concat`, `mixed_concat_little_endian`, `query_unseen_concat`.
- V380 reexecutou os `solver_ops` no prompt: `70/79` ganhos reproduzidos como teacher (`36 arithmetic`, `27 little_endian`, `7 mixed_concat`).
- V380 encontrou `0` ganhos strict-independent. Portanto o sinal ainda nao desbloqueia submit nem treino HF direto.

Uso permitido no plano:

- Gerar candidatos CPU para os residuos de `equation_transform`, priorizando rows nao condicionadas.
- Usar `solver_ops`, `solver_mapping`, `solver_category` como regra/teacher somente depois de reexecucao independente no prompt.
- Nunca usar bruto: ha `23` erros.

### Fonte ativa 2 - `filtered_merged_dataset.csv`

Arquivo: `C:\Users\davis\Downloads\nemotron_dataset_final\kaggle_logprob\results\filtered_merged_dataset.csv`

Evidencia:

- `8703` rows.
- `7044` IDs unicos.
- `1659` duplicatas/reweighting.
- `821` rows sao duplicatas exatas.
- Labels `8703/8703` corretos.
- CoT `8691/8703` correto.
- `equation_transform`: `1499` IDs unicos, `2438` rows, CoT `2428/2438`.
- `bit_manipulation`: `1354` IDs unicos, `1754` rows, CoT `1752/1754`.
- Cobre `92/92` residuos V375; `91/92` CoT correto.

Uso permitido no plano:

- Selecionar um melhor trace por ID, filtrado por resposta correta, loss, tamanho e tokenizer.
- Gerar dataset curto de transferencia para LoRA depois que o solver gate CPU passar.
- Nao usar bruto por causa de duplicatas exatas, duplicatas por ID/reweighting e traces longos.

### Fonte ativa 3 - `sft_train_full_9500.jsonl`

Arquivo: `C:\Users\davis\Downloads\nemotron_dataset_final\sft_train_full_9500.jsonl`

Evidencia:

- `9500/9500` correto.
- `9500` IDs unicos.
- Cobertura completa das seis familias.
- `9500/9500` rows tem multiplos spans `\boxed{}`.
- `364` rows tem resposta declarada errada antes do `\boxed{}` final corrigido: `238` em `bit_manipulation`, `126` em `equation_transform`.
- `173` respostas de `equation_transform` contem chaves literais; exigem escaping/extrator oficial.

Uso permitido no plano:

- Fallback de trace correto por ID.
- Comparar formato/prompt/template.
- Nao fazer SFT amplo.
- Se usado, limpar spans intermediarios e manter exatamente um final answer validado.

## Fontes Diagnosticas

| Fonte | Decisao |
|---|---|
| `kaggle_sft_data/dataset_generated.csv` | labels corretos, CoT `9197/9500`; usar para comparacao/hard negatives, nao como treino bruto |
| `kaggle_trajectories/nemotron_traj.csv` | geracao direta so `4542/9500`; nunca usar como label, apenas hard-negative/confidence |
| `sft_train_converted.jsonl` | duplicado pelo `filtered_merged_dataset.csv`; `6923` rows com think-tags malformados; usar so para referencia de formato, nunca bruto |
| `sft_train_reconstructed.jsonl` | contem `8463` rows sinteticas/desconhecidas; fora do plano ativo |
| `sft_reconstructed.jsonl` | `9500/9500`, mas supersedido por `sft_train_full_9500` e logprob filtered |
| `nemotron_hacker_dataset` | subconjunto duplicado do `nemotron_dataset_final`; fora do plano ativo |

## Gaps Corrigidos no Double Check

- `nemotron_dataset_final` tem `13` arquivos, `509113679` bytes.
- `nemotron_hacker_dataset` tem `7` arquivos, `401052223` bytes.
- Arquivos comuns entre os diretorios: `6`.
- Hash mismatches entre arquivos comuns: `0`.
- `nemotron_dataset_final` e o superset ativo.
- `competition_train.csv` tem `9500` rows reais; contagem por linha fisica e invalida porque prompts tem quebras de linha.
- `competition_test.csv` tem `3` rows sample e `3/3` IDs/prompts aparecem no train (`00066667`, `000b53cf`, `00189f6a`); nao e eval.
- O pacote final tem overlap com V217 train em `1476` prompts: `654` bit, `246` equation e `144` em cada familia facil.
- O pacote final tem overlap com V217 val em `103` prompts: `30` bit, `9` equation e `16` em cada familia facil.
- Qualquer dataset futuro precisa filtrar `id`, `prompt_sha256`, prompt normalizado e split V217 val antes de treino ou validacao.
- O relatorio menciona `tong_with_logprob.csv` e `yours_with_logprob.csv`, mas esses arquivos nao existem em nenhum dos dois diretorios auditados. Nao contam como evidencia.

## Triple Check dos Anexos 2026-05-14

Arquivos analisados:

- `Dataset andy279_nemotron-reasoning-challenge - Relatorio Completo de Extracao.md`.
- `Relatorio de Extracao_ Dataset andy279_nemotron-reasoning-challenge.md`.
- `Nemotron Reasoning Challenge - SFT Data.md`.
- `Relatorio_ Dataset andy279_nemotron-reasoning-challenge.md`.
- `competition_train.csv`.

Achados que ficam no plano:

- `competition_train.csv` anexado e identico ao `competition_train.csv` do pacote final: SHA256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`, `9500` rows, `9500` IDs unicos, `0` duplicatas.
- Contagem oficial por familia no train: `bit_manipulation=1602`, `equation_transform=1555`, `gravity_constant=1597`, `numeral_system=1576`, `text_encryption=1576`, `unit_conversion=1594`.
- Os anexos descrevem o SFT original `andy279` como `49290` exemplos de treino / `7200` puzzles unicos e validacao com `1165` exemplos / `1123` puzzles.
- Os anexos citam `399` transformations nao resolvidas no split de validacao original. Isto reforca que `equation_transform` e gargalo de solver/DSL/verificacao, nao de treino generico.
- Os anexos citam forte sinal original para as familias alvo: `17285` traces de bit, `10741` transformation, `1602` solver-guided bit e `1101` solver-guided transformation.
- Esses arquivos SFT originais nao estao disponiveis localmente; portanto nao entram como fonte ativa de treino. Se acesso for aprovado depois, entram apenas por novo gate de hash, resposta, duplicata, overlap e tokenizacao.
- A frase dos relatorios sobre ter `100% dos dados essenciais` nao e aceita como fato operacional: os proprios anexos dizem que o original tem `49290` train e `1165` validation, enquanto os dados locais cobrem `17963`/`9500` derivados e nao incluem a validacao original.
- O README SFT descreve uma regra de qualidade que agora e obrigatoria no V381: limpar artefatos LaTeX dentro de `\boxed{}`, reextrair a resposta final, recomputar corretude pelo scorer e manter somente tentativas corretas.
- O claim de `competition_test.csv` com `34` puzzles foi contradito pela auditoria local: o arquivo auditado tem `3` rows e as `3` aparecem no train. Continua proibido como eval.
- Os anexos citam raw traces multi-attempt (`all_traces_merged.jsonl`, `solver_bit_manipulation_traces_merged.jsonl`, `solver_transformation_traces_merged.jsonl`), mas esses arquivos nao existem nos diretorios locais auditados. So entram no plano se forem adquiridos e auditados em novo gate.
- Um dos relatorios contem padrao de token HF; relatórios brutos nao devem ser versionados. Apenas metadados redigidos entram no repo.

Ganho medido novo desses anexos:

- Adapter-only: `+0`.
- CPU teacher: `+0`.
- Ganho esperado: indireto e condicional. A utilidade real e reduzir erro no V381/V382; ainda nao autoriza HF nem submit.

## Roadmap Ativo V392

### Step 1 - V392 baseline/package lock

Status: concluido em CPU.

Objetivo: parar de comparar contra nomes incertos. Localizar e travar o melhor package/submission historico que gerou o plateau `0.86`/ranking 19, ou declarar formalmente que o arquivo exato ainda nao esta disponivel localmente.

Entrada:

- Artefatos locais de package/submission.
- Manifests de Kaggle/HF ja versionados.
- Inventario Google Drive ja auditado.

Regras:

- Registrar nome, caminho, SHA256, adapter repo/subfolder, commit e metricas weak/full do melhor submit conhecido.
- Comparar contra o baseline operacional atual (`192/315`, `equation=56`, `bit=136`) e contra V291/V290/V226 quando existirem manifests.
- Se o package exato do ranking 19 nao for encontrado, bloquear qualquer claim de "melhorar o submit 0.86" ate ele ser localizado.
- Criar quadro comparativo V392 vs V391/V291 antes de qualquer novo experimento.

Saida esperada:

- Manifest pequeno de baseline lock.
- Tabela com `total`, `equation_transform`, `bit_manipulation`, `truncated`, full score quando existir e decisao.

Resultado V392:

- Manifest: `artifacts/v392_baseline_package_lock/20260514T_baseline_lock/v392_baseline_package_lock_manifest.json`.
- Baseline travado: V291/V290 checkpoint-6 adapter-only package.
- Kaggle CLI mostra o time `Felipe Angelo` na posicao `19` da pagina retornada, com submission `2026-05-11 22:19:17.163000`, descricao `V291 V290 checkpoint-6 adapter-only full823 trunc1 official-like gate`, public score `0.86`.
- Package: `artifacts/v291_submission_package/v291_h200_checkpoint6_823_20260511T212028Z/submission.zip`, SHA256 `293b414f316330db7ac12c4f3001e7796b0a087ed5dd86af6e13d98620b43433`.
- Weak baseline: `192/315`, `equation=56/155`, `bit=136/160`, `truncated=0`.
- Full official-like baseline: `823/947`, `equation=56/155`, `bit=135/160`, `truncated=1`.

### Step 2 - V393 sweep sem treino do melhor adapter/package

Status: encerrado por FinOps em `2026-05-14`.

Objetivo: buscar ganho hoje sem gastar em treino que ja falhou. Testar variantes de prompt/template/extractor/decoding sobre o melhor adapter/package existente.

Hipotese:

- V384 mostrou que prompt pode mover `bit` para `137/160`, mas falhou por `truncated=1` e `equation=56`.
- A rota mais barata e rapida e corrigir truncation/extracao e testar prompts curtos, nao treinar outro LoRA.

Regras:

- Sem solver/postprocessor runtime.
- Sem alterar answer por regra externa.
- Temperatura `0`, seed fixo, contrato V221 weak315.
- Variantes pequenas e rastreaveis:
  - prompt oficial historico;
  - prompt curto boxed-only;
  - prompt sem sufixo extra;
  - limite de max tokens mais restritivo se reduzir truncation sem cortar answer;
  - extractor oficial vs extractor estrito apenas para medir, sem escolher resposta manual.
- Promover somente se `total > 192`, `equation > 56`, `bit >= 136`, `truncated=0`.
- Se uma variante tiver `total=193` mas `equation=56`, so promover para full se tambem tiver `bit>=137` e `truncated=0`.

Saida esperada:

- Batch weak eval comparavel.
- Se passar: full official-like e package no mesmo dia.
- Se falhar: encerrar sweep e ir para Step 3.

Implementacao V393:

- Launcher: `artifacts/v393_hf_h200_v290_prompt_sweep_launch/launch_v393_hf_h200_v290_prompt_sweep.py`.
- Diff obrigatório: `artifacts/version_diffs/V393_PROMPT_SWEEP_VS_V392.md`.
- Adapter travado: `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.
- Variantes: `baseline_v290_repro`, `v221_boxed_suffix`, `no_suffix`, `strict_disable_thinking`, `strict_2048_tokens`.
- Gate de promocao continua: `total>192`, `equation>56`, `bit>=136`, `truncated=0`.

Resultado V393:

| Variante | Total | equation | bit | trunc | Decisao |
|---|---:|---:|---:|---:|---|
| `baseline_v290_repro` | `190/315` | `56/155` | `134/160` | `1` | rejeitado; pior que V392 lock |
| `v221_boxed_suffix` | `192/315` | `56/155` | `136/160` | `0` | empatou baseline; sem promocao |
| `no_suffix` | `158/315` | `55/155` | `103/160` | `1` | regressao severa; prova que remover sufixo boxed quebra bit |
| `strict_disable_thinking` | n/a | n/a | n/a | n/a | cancelado por FinOps apos regressao de `no_suffix` |
| `strict_2048_tokens` | n/a | n/a | n/a | n/a | cancelado por FinOps apos regressao de `no_suffix` |

Erro encontrado na primeira execucao: o launcher tentava ler `batch_candidate_summary.json` como lista (`payload[0]`), mas o formato real atual e objeto com chave `rows`. Correcao aplicada no launcher: aceitar os dois formatos e permitir `--skip-variant`.

Extensao agressiva V400:

| Variante | Total | equation | bit | trunc | Decisao |
|---|---:|---:|---:|---:|---|
| `symbolic_equation_first` | `175/315` | `40/155` | `135/160` | `27` | rejeitado; instrução algoritmica gera saida longa e perde equation |
| `bit_stride_guarded` | `7/315` | `7/155` | `0/160` | `227` | rejeitado; colapso de formato, bit totalmente perdido |

Conclusao: V393/V400 nao liberam submit nem full eval. Prompt/template sozinho nao move `equation_transform` acima de `56`; remover, apertar demais ou inserir DSL/literatura diretamente no prompt tende a destruir `bit_manipulation` e aumentar truncation. A linha de prompt sweep esta encerrada.

### Step 3 - V394 row-level equation miss inventory

Status: concluido em CPU; sem autorizacao de GPU.

Objetivo: descobrir exatamente os `4` ganhos necessarios para `equation_transform 56 -> 60` sem depender de "treinar mais".

Entrada:

- Misses `equation_transform` do melhor adapter.
- `solver_results`/V378/V380 derivados.
- DSL auditada em `tonghuikang/nemotron/reasoners/equation_numeric.py`.
- `filtered_merged_dataset.csv` somente como referencia de trace correto, nao como treino bruto.

Regras:

- Para cada miss, classificar se e:
  - numeric operator DSL;
  - concat/reverse concat;
  - signed/negative formatting;
  - little-endian/reversal;
  - symbolic punctuation/braces;
  - extractor/boxed formatting;
  - prompt compliance failure.
- Separar "o solver sabe" de "o adapter consegue produzir".
- Gerar candidato de prompt/exemplar somente para rows onde o erro parece de formato/extracao/prompt compliance.
- Nao contar ganho de solver externo como ganho submit-safe.

Saida esperada:

- CSV de misses com categoria, old prediction, expected answer, hipotese e acao.
- Lista curta de no maximo `10` rows candidatas para prompt-level rescue.
- Decisao: se existir rota prompt-safe, voltar ao Step 2 com variante focada; se nao existir, nao gastar GPU.

Resultado V394:

- Input: `artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/input/v290_ckpt6_weak_predictions.csv`, SHA256 `910a051d8b8e652e37c0b0814ac59fe4a400b95cb432945b6a0244f97f5b31bf`.
- V324 sobre V290 checkpoint-6: `6` ganhos CPU aceitos, `0` conflitos, `projected_equation=62`, `projected_weak=198`, `bit_guardrail=136`.
- V375: `99` misses, `26` clusters, `57` priority rows; subtipos `16` numeric operator e `83` symbolic punctuation.
- V394 consolidation: `6` CPU solver verified gains, `93` unresolved misses.
- Comparacao obrigatoria vs V390: `accepted_cpu_gain_ids` identicos, `delta=0`; `projected_equation_correct=62`, `delta=0`; `bit_guardrail=136`, `delta=0`.
- Decisao: `reconfirmed_existing_cpu_signal_no_new_gpu_authorization`. Nao lançar HF training com os mesmos seis exemplos numericos; expandir DSL simbolica nos unresolved rows.

Artefatos V394:

- Script: `scripts/analyze_v394_equation_row_level_inventory.py`.
- Manifest: `artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v394_inventory/v394_equation_row_level_inventory_manifest.json`.
- Inventario: `artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v394_inventory/v394_equation_row_level_inventory.csv`.
- Comparativo: `artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v394_inventory/v394_vs_v390_comparison.csv`.

### Step 4 - V395 HF CPU aggressive symbolic gate

Status: concluido no Hugging Face CPU; sem autorizacao de GPU.

Objetivo: obedecer a diretriz de usar CPU do HF para validar uma busca mais agressiva antes de gastar GPU.

Resultado:

- Job: `https://huggingface.co/jobs/felipesp1983/6a0647a9e48bea4538b9d78a`.
- Flavor: `cpu-upgrade`, `8 vCPU`, `32 GB`, `US$0.0005/min`.
- V324 agressivo: `6` ganhos numericos conhecidos, `0` conflitos, `projected_equation=62`.
- V329 amplo: `1` ganho simbolico conhecido (`99d6a3b5`), `0` conflitos, `projected_equation=63`.
- V336 integrado: `199/315`, `equation=63/155`, `bit=136/160`, `losses=0`.
- Decisao: nao houve sinal novo alem de V336/V343; nao treinar LoRA com a mesma lista de `7` IDs.

Artefatos:

- Local: `artifacts/v395_hf_cpu_aggressive_symbolic_gate_results/v395-hf-cpu-aggressive-symbolic-gate-20260514T220636Z/`.
- HF dataset: `felipesp1983/kg1-v395-cpu-symbolic-gate-artifacts`, path `v395-hf-cpu-aggressive-symbolic-gate-20260514T220636Z`.

### Step 5 - V397 bit guardrail/probe

Status: CPU primeiro.

Objetivo: manter `bit>=136` enquanto se tenta subir equation. Bit ja esta no limite bom; a tarefa e evitar regressao.

Entrada:

- Bit misses e hits do melhor adapter.
- Ideias Tong: bit-pair, bitsum, stride, unary/binary/constant families.

Regras:

- Nao repetir "Tong bit direct replacement" porque V374 ja refutou essa linha.
- Usar algoritmo Tong somente para gerar probes/traces curtos e identificar quais prompts causam regressao.
- Qualquer variante equation precisa rodar bit guardrail antes de full eval.

Saida esperada:

- Bit guardrail CSV.
- Bloqueio automatico de qualquer candidato com `bit<136`.

### Step 6 - V398 treino LoRA somente com prova de transferencia

Status: V398 executado e rejeitado; bloqueado para continuidade.

Objetivo: permitir treino apenas se houver evidencia que o modelo consegue internalizar a correcao.

Gate obrigatorio antes de HF/Kaggle GPU:

- Um pilot adapter/package ou prompt probe deve mostrar pelo menos `+1` weak real sem regressao.
- Dataset deve ser curto, um trace por ID, sem overlap proibido, resposta final unica e scorer validado.
- Primeiro checkpoint deve ser avaliado antes de continuar.
- Cancelar se `equation<=56` ou `bit<136`.

Decisao atual:

- V391 provou que `198/315` em projecao CPU nao basta. V398 provou que o corpus reconstruido V397 tambem nao basta. V399 provou que V398 nao tem nenhum acerto complementar de `equation_transform`. Portanto o proximo passo nao pode ser "mais epochs", "LR diferente" ou "mais H200" nessa linha.

### Step 6B - V405/V406 solver-first transfer

Status: CPU projection concluida; dataset V406 construido e gateado; sem ganho adapter-only ainda.

Comparativo obrigatorio:

| Versao | Weak total | equation | bit | Conflitos | Submit-safe? |
|---|---:|---:|---:|---:|---|
| V291/V290 checkpoint-6 | `192/315` | `56/155` | `136/160` | n/a | sim |
| V405 integrated CPU solver projection | `201/315` | `63/155` | `138/160` | `0` | nao |
| V406 solver-first transfer dataset | nao treinado | nao medido | nao medido | n/a | nao ainda |

V405 integrou os ganhos CPU sem conflito:

- equation gains: `274def88`, `528ec0d8`, `7688e06e`, `99d6a3b5`, `c5b058d6`, `d1bd7478`, `fb623471`;
- bit gains: `4ada9150`, `4c327b55`;
- projecao: `201/315`, `equation=63`, `bit=138`, `0` conflitos.

V406 transformou esse sinal em dataset de transferencia sem treinar nos weak/full rows:

- treino: `2064` rows (`1024` bit, `1040` equation);
- validacao: `516` rows (`256` bit, `260` equation);
- overlap com weak/full: `0` por ID e `0` por prompt hash;
- tokenization gate real: `passed`, truncation `0`, offset masks `100%`.

Decisao: V406 e o primeiro candidato responsavel para um smoke adapter-only curto. Ainda nao autoriza submit. Autoriza apenas um treino curto com kill-switch no primeiro checkpoint.

Promocao minima:

- weak `>192/315`;
- `equation>56`;
- `bit>=136`;
- `truncated=0`.

Cancelar por FinOps se o primeiro checkpoint nao bater esses limites.

### Step 6C - V407 literature/Kaggle double check

Status: concluido; achados acionaveis inseridos.

Fontes auditadas:

- Kaggle/Nemotron: `konbu17/bit-manipulation-solver-cot-generator`, `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`, `huikang/end-to-end-finetuning-for-lb-0-85`;
- Kaggle/program synthesis: `michaelhodel/program-synthesis-starter-notebook`, `marcshade/three-tier-dsl-based-program-synthesis`, `francisbanda/arc-agi-2-mdl-program-synthesis-solver`;
- literatura: FlashFill/PBE, SyGuS/CEGIS, Z3 bit-vectors, DreamCoder/neural-guided synthesis.

Achados reais:

- `bit_manipulation` deve expandir o gate exato para funcoes booleanas assimetricas `INHIB(a,b)=a AND NOT b` e `IMPL(a,b)=NOT a OR b`; variantes reversas vem de pares ordenados.
- `MAJ`, `CH`, `XOR3` so entram como fallback raro e com verificacao em todos os exemplos.
- CoT de bit deve ser high-confidence; low-confidence/bruteforce fica fora de treino porque historicamente causa regressao.
- `equation_transform` deve seguir PBE/SyGuS: DSL pequena, busca de programa curto, verificacao contra todos os exemplos e abstencao quando houver ambiguidade.
- ARC/program-synthesis reforca tie-break por programa curto/MDL e nao por primeira hipotese fraca.

Mudanca de plano:

1. implementar gate CPU V407/V408 para bit assimetrico e equation symbolic PBE;
2. se o gate achar novo sinal no-loss, atualizar V406/V408 dataset;
3. rodar somente smoke adapter-only curto;
4. promover para full/package/submit apenas se bater V291 no weak gate.

Artefato: `artifacts/v407_literature_kaggle_doublecheck/KG1_V407_LITERATURE_KAGGLE_DOUBLECHECK.md`.

### Step 7 - Full/package/submit

Status: somente depois de weak gate.

Regras:

- Full eval somente se weak passar.
- Package somente se full official-like melhorar o baseline conhecido.
- Kaggle submit somente com ganho medido e tabela comparativa contra o submit historico.
- Se criterio de desempate favorecer submissao anterior, nao enviar regressao mesmo que haja curiosidade experimental.

## Regras Permanentes

- Nenhum HF sem CPU gate com sinal novo.
- Nenhum submit sem ganho medido.
- Nenhuma decisao por `eval_loss`; decisao por ACC e truncation.
- Jobs HF em execucao devem ser verificados a cada aproximadamente `40s`.
- Se o job nao puder mais bater o gate, cancelar por FinOps.
- Nao copiar datasets grandes para o repo.
- Artefatos versionados devem ser pequenos: manifests, CSVs de auditoria, scripts.
- Todo `.ipynb` novo ou alterado deve passar `scripts/notebook_release_gate.py`.

## Itens Removidos do Plano Ativo

| Item | Motivo |
|---|---|
| Mais epochs/LR sem novo CPU gate | ja reduziu loss sem melhorar ACC |
| Broad SFT em todos os traces | risco de regressao; historicamente nao transferiu |
| `nemotron_traj.csv` como label | so `4542/9500` correto |
| `sft_train_reconstructed.jsonl` completo | contem rows sinteticas/desconhecidas |
| `sft_train_converted.jsonl` bruto | duplicatas/reweighting |
| `sft_train_full_9500.jsonl` bruto | multiplos boxed spans e `364` respostas declaradas erradas antes do final corrigido |
| `competition_test.csv` como eval | `3/3` rows aparecem no train |
| Claims de `100% dos dados essenciais` | contradizem a ausencia local de `sft_train.jsonl`, `sft_val.jsonl` e validacao original |
| Raw traces citados nos relatorios | arquivos nao existem nos diretorios locais auditados |
| `dataset_generated.csv` bruto | CoT errado em `303` rows |
| `problems.jsonl` bruto | apenas `8333/9500` correto |
| Tong bit direct replacement | V374 caiu para `bit=136` e teve perdas contra V366 |
| HF V371/V372 trace-style | checkpoint-1 `191/315`, `bit=135` |
| Prompt/thinking variants amplas | regressao severa |
| Adapter soups V291/V382 | V389 mostrou `190-191/315`, `equation=56`, `bit=134-135`, truncation `1-2`; linha encerrada |
| V390/V391 equation+bit replay LoRA direto | CPU projection `198/315` nao transferiu; V391 ficou `191/315`, `equation=56`, `bit=135` |
| H200 relaunch sem novo dado | V391 confirmou que trocar hardware nao muda ACC quando a hipotese de dados nao transfere |
| HF training baseado apenas em `eval_loss` | historicamente loss caiu sem mover `equation_transform`; promocao e por ACC |
| Web/API buscas genericas | so retornam ao plano se virarem regra, dataset ou gate verificavel |
| V227/V228/V229 targeted equation sweep | Drive audit V396 confirmou regressao severa: V229 `16/315`, V228 com `111/160` truncados na janela inicial |
| Public adapters Huikang/Kienngx do Drive como drop-in | Huikang falha por incompatibilidade de target modules; Kienngx 3000 mede `32/315` com truncation alto |

## Proxima Acao Unica

Seguir rota solver-first agressiva, mas com gate:

1. implementar V407/V408 CPU gate para `bit_manipulation`:
   - `INHIB`, `IMPL`, pares ordenados, `MAJ`, `CH`, `XOR3`;
   - aceitar somente candidatos que batem todos os exemplos e nao causam perdas.
2. implementar V407/V408 CPU gate para `equation_transform`:
   - DSL PBE/SyGuS para concat, reverse concat, signed format, literal insert/delete, pontuacao/brackets e pequenos hibridos aritmeticos;
   - aceitar somente programa unico/curto com verificacao total.
3. se houver novo ganho CPU no-loss, atualizar V406 em V408 e rodar smoke HF/Kaggle curto.
4. se o primeiro checkpoint nao superar V291 (`192/315`, `equation=56`, `bit=136`, trunc `0`), cancelar por FinOps.
5. manter V291/V290 checkpoint-6 como unico package submitavel ate aparecer ganho adapter-only medido.

Nao rodar broad SFT, prompt sweep ou job guiado por `eval_loss`. A decisao e por ACC, truncation e comparativo contra V291.
