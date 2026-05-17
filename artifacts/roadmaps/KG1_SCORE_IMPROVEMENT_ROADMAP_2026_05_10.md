# KG1 Score Improvement Roadmap

Atualizado: 2026-05-17

Este e o roadmap ativo e limpo apos a revisao V484 dos arquivos OpenRouter de
16/05/2026. O historico detalhado foi arquivado em:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V435_CLEANUP_2026_05_15.md`
- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V484_OPENROUTER_CLEANUP_2026_05_16.md`

A partir deste arquivo, itens antigos so valem como evidencia historica. O plano
executavel e somente o que esta abaixo.

## Estado Real

| Metrica | Melhor submit-safe atual | Promocao minima |
|---|---:|---:|
| Total weak adapter raw label-free | 191/315 | > 192/315 |
| equation_transform adapter raw label-free | 55/155 | alvo inicial 60/155 |
| bit_manipulation | 136/160 | >= 136/160 |
| truncated | 0 | 0 |
| Full official-like conhecido | 823/947 | > 823/947 |

Valores melhores que dependem de solver, verifier, postprocessor, teacher CPU,
oracle ou selecao por weak/full nao sao submit-safe ate virarem comportamento do
adapter LoRA valido.

Ultima evidencia operacional relevante:

| Versao | Resultado | Decisao |
|---|---:|---|
| V291/V290 checkpoint-6 | stored weak 192, equation 56, bit 136, trunc 0; V505 label-free recalcula 191, equation 55, bit 136 | baseline forense; promocao precisa metrica label-free |
| V477 ckpt-2 | weak 192, equation 57, bit 135, trunc 0 | nao promove; ganhou equation mas perdeu bit |
| V475 CPU solver projection | weak 196, equation 60, bit 136 | sinal CPU; ainda nao submit-safe |
| V480/V483 linha recente | loss mexe, ACC nao sai do plateau | suspeita forte de PEFT continuity bug |
| V485 seed PEFT metadata gate | `hf_gpu_allowed=true`; 12011 tensors; target params 5934/5934; `modules_to_save=[]` | seed V290/V291 estruturalmente liberado |
| V487 treino H200 | treino completo, checkpoint-10 melhor `eval_loss=1.3519`; target params 5934/5934 ativos | continuidade PEFT corrigida, mas nao prova ACC |
| V488 ckpt-10 weak eval | weak 191, equation 57, bit 134, trunc 1 | nao promove; target params nao eram o unico gargalo |
| V489 audit integridade | metrica ACC estrita correta; V488 teve +1 equation, -2 bit, +1 trunc; F2 frozen-active nao era visivel no manifesto; expected-aware antigo podia vazar boxed anterior | corrigir observabilidade/guard/extracao antes de novo GPU |
| V490 debug double check | compilacao, self-tests, static gate, dataset V390/V326, tokenization e metric path OK; HF jobs ativos 0 | proximo passo deve mudar mecanismo treinavel, nao repetir V487 |
| V491 OpenRouter consult | GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro, Qwen 3.6 Max convergem em MoE trainability, freeze `lm_head`, loss weight 1.0 e kill-switch cedo | roteiro de smoke alterado; nao repetir treino attention+`lm_head` |
| V492 uploaded OpenRouter double check | 12 modelos adicionais reforcam MoE `up_proj/down_proj` frozen-active como principal suspeita; tambem alertam que o +1 equation pode ser extracao, nao aprendizado bruto | roadmap limpo para um unico experimento fail-fast, depois pivot/stop |
| V493 H200 smoke | treino completo; `target_parameters_trainability_mode=trainable`; `up_proj/down_proj` treinaveis; `lm_head` congelado; eval loss `1.9233 -> 1.9152`; checkpoint-2 uploaded | loss saudavel mas ganho nao comprovado; seguir para V494 weak eval |
| V494 loss/ACC sync audit | loss path correto como CE mascarada; ACC path correto como geracao+extracao+`verify_answer`; loss nao e proxy matematico de ACC; V245 precisa controles long-context explicitos | static gate atualizado; rodar weak eval promocional com `KG1_MAX_TOKENS=7680`, thinking on e gate bloqueante |
| V494 V493 checkpoint-2 weak eval | weak 190, equation 57, bit 133, trunc 1; simple extraction 189; strict-vs-permissive bit overcount 15 | nao promove; invalida o mix antigo V390/V326 nesse mecanismo, mas nao testa o dataset V475 CPU-gated |
| V494 dataset mismatch audit | V493/V494 usou `data/v390_equation_bit_replay_mix`, nao `data/v475_equation_bit_replay_mix`; V475 tem 1312/328 linhas, token max 331, trunc 0 e projecao CPU `equation 56 -> 60` | justificou V495; apos V496, V475 SFT tambem esta bloqueado sem novo sinal CPU |
| V495 H200 V475 smoke | treino tecnico OK; MoE `up_proj/down_proj` treinaveis; `lm_head` congelado; eval loss `1.695015 -> 1.694518` | loss saudavel, mas muito pequeno; decisao dependeu do V496 weak eval |
| V496 V495 checkpoint-2 weak eval | weak 191, equation 57, bit 134, trunc 1; diff real vs V290: +1 equation (`518deb39`), -2 bit (`8740ed31`, `59bee375`) | nao promove; V475 SFT transfer tambem bloqueada; proximo passo e CPU teacher/guardrail, nao mais H200 SFT amplo |
| V497 CPU residual transfer audit | baseline 192; V324 CPU projeta 196; V496 projeta 191; 4 ganhos CPU nao transferiram; 1 ganho simbolico V496; 2 perdas bit | confirma que o gargalo e transferencia teacher->LoRA, nao metrica/loss; proximo passo e trace numeric curto + bit guardrail |
| V498 numeric teacher trace pack | 1712/428 linhas; 3 regras numeric com hard negatives; bit replay guardrail; zero overlap id/prompt/prompt+answer com weak/full; token max 331; offset masks completos | dataset elegivel para um unico smoke H200 curto, nao para treino amplo |
| V499 H200 smoke | dataset V498 uploaded no HF commit `c7e27fd39c598dd23cb25481f567787bdff50820`; V478 objetivo passou; H200 rodou, mas final eval `2.8125 -> 2.8162` e answer-span inativo | bloqueado; nao weak-eval/package/submit |
| V501 H200 answer-span | answer-span ativo: train `1712` exemplos e `15197` tokens; MoE `gate_up/down` treinaveis; `lm_head` congelado; final eval `1.9919 -> 1.9923` | bloqueado por kill-switch; nao rodar weak/full/package/submit |
| V504 crisis root-cause audit | achou bugs reais em metrica/gates: ACC label-aware, thresholds weak antigos, weak defaults diagnosticos, full best selection, anti-leak flags ausentes, pre-paid gate sem schema SFT | corrigir gates e revalidar candidatos com extracao label-free antes de qualquer novo GPU |
| V505 label-free revalidation | 30 CSVs varridos; 22 weak315; raw-output adapter top `191/315`, `equation=55`, `bit=136`; reference-only solver/postprocessor chega a `222/315`, mas nao e adapter-only | bloquear promocao de CSV sem `raw_output`; proximo ganho precisa converter sinal solver em comportamento do adapter ou pacote valido |
| V506 reference signal gap | compara melhor adapter raw vs melhor reference-only: `31` targets (`23` bit, `8` equation), `0` reference-loss risk, `93` ambos errados | inventario alvo para transferencia; ainda nao e submit-safe |
| V507 V274 label-free projection | melhor adapter raw + V274 em `raw_output` label-free: `195/315`, `equation=59`, `bit=136`, `0` perdas; `196/equation=60` dependia de overcount simbolico expected-aware em `4bb8c6cd` | sinal de postprocessor, nao adapter-only; usar como alvo, nao como submit |
| V508 adapter raw V274 sweep | 8 candidatos adapter raw varridos; melhor geral `195`, `equation=60`, `bit=135`; melhor com guardrail `bit>=136` e `195`, `equation=59` | nao ha candidato adapter-only submit-safe novo; equation 60 atual perde bit |
| V509 dataset integrity audit | 20 datasets auditados; V439 bloqueado por `31` mismatches de resposta simbolica/escape; V443 bloqueado por dataset vazio; V475/V498/V460 passam integridade | nao treinar com V439/V443; consolidar dataset ativo limpo |
| V510 canonical active training pool | dataset unico criado: `2627` train, `637` val; incluiu V498/V475/V460; removeu `543` duplicados train e `155` val; reaudit V509 com `0` bloqueios | usar V510 como unica fonte ativa antes de tokenization/pre-paid gate |
| V510 tokenization real local | tokenizer oficial local passou: `offset_masks=2627/637`, `prompt_truncation_rate=0`, `completion_truncation=0`, token max `331`, sem overlap weak/full | V510 esta pronto para launch fail-fast, desde que pre-paid/debug tambem passem |
| V511 canonical H200 smoke | job `felipesp1983/6a08dc43e48bea4538ba02ce` completou em `0.05h`; MoE `gate_up/down` treinaveis `5934/5934`, `lm_head` congelado, checkpoint/final adapter uploaded; eval loss `2.8125 -> 2.8128` | bloqueado por FinOps; nao weak-eval/package/submit; V510 nao trouxe sinal de transferencia |
| V512 Kaggle discussions audit | `140` topicos e `586` posts varridos via API; `392` hits; achados concretos: THK bit-pair/bitsum/stride, min-logprob, prompt-loss masking, logprob/learnability test, trace curto, duplicate-CoT/format-clash, solvers simbolicos como oracle | reforca CPU-first e trace learnability; nao autoriza novo broad SFT |
| V513 trace learnability gate | Local + HF CPU job `felipesp1983/6a08e383e48bea4538ba03ba` reproduziram `blocked_no_gpu`; V510 tem `742/742` bit rows como `Final answer: boxed` sem trace; bit assistant p50 `3` palavras, `0` bit traces; equation tem traces curtos p50 `31` palavras; tokenization segue OK | bloqueia GPU a partir de V510 como esta; proximo passo e substituir bit answer-only por traces deterministicas bit-pair/bitsum/stride antes de qualquer novo treino |
| V514 traceable bit V510 dataset | V510 refeito apenas no bloco bit: `581/742` bit rows convertidas para traces verificadas (`466` train, `115` val); `161` bit rows sem prova foram descartadas; equation V510 mantido; tokenization real passou com `0` trunc, offset masks `2484/619`, token max `553/541`; V513 recheck passou com `0` blockers | primeiro dataset estruturalmente melhor que V510; ainda nao e submit-safe nem autoriza GPU sem HF CPU reproduction, objective/pre-paid gate e smoke minimo |
| V514 HF CPU attempt 1 | job `felipesp1983/6a08e6fe3308d79117b915bb` falhou antes dos gates por dependencia ausente: `pandas` exigido pelo `run_v296_bit_stride_solver_audit.py`; nenhum treino/eval/package/submit rodou | launcher corrigido para instalar `pandas>=2.0.0`; relancar CPU apos commit/push |
| V514 HF CPU attempt 2 | job `felipesp1983/6a08e83de48bea4538ba0468` reproduziu o build V514 no HF CPU, mas o tokenizer gate falhou por `jinja2` ausente em `apply_chat_template`; nenhum treino/eval/package/submit rodou | launcher corrigido para instalar `jinja2>=3.1.0`; relancar CPU apos commit/push |
| V514 HF CPU attempt 3 | job `felipesp1983/6a08e9ad3308d79117b91609` completou no HF CPU; build V514 reproduzido; tokenization real passou com `0` trunc e offset masks `2484/619`; V513 recheck passou com `0` blockers; artefatos enviados para `felipesp1983/kg1-v514-traceable-bit-v510-artifacts/v514-hf-cpu-traceable-bit-20260516T220219Z` | HF CPU reproduction concluido; proximo passo e V515 CPU residual full-byte, nao GPU amplo |
| V515 V514 fullbyte residual | recuperou somente rows bit residuais com `fullbyte_unique_prediction`: `+7` train e `+1` validation; V515 total `2491/620`; tokenization real passou com `0` trunc, offset masks `2491/620`, token max `553/541`; V513 recheck passou com `0` blockers | ganho pequeno de cobertura verificavel; reproduzir no HF CPU antes de qualquer GPU |
| V515 HF CPU reproduction | job `felipesp1983/6a08edcf3308d79117b9167f` completou; V515 reproduzido no HF CPU; V286 passou com `0` trunc e offset masks `2491/620`; V513 passou com `0` blockers; upload para `felipesp1983/kg1-v515-v514-fullbyte-residual-artifacts/v515-hf-cpu-fullbyte-residual-20260516T221957Z` | V515 e o dataset ativo mais limpo; ainda nao e ACC submit-safe; proximo passo e gate objetivo/pre-paid antes de qualquer GPU |
| V515 objective alignment gate | pesos iguais falham: `bit=18.99%`, `equation=81.01%`; pesos fonte bit `1.5x` passam sem findings: `bit=26.01%`, `equation=73.99%` | qualquer launcher V515 deve usar exatamente essa ponderacao ou passar novo gate objetivo; pesos iguais ficam bloqueados por risco de backfire |
| V516 label-free equation regate | baseline correto e `191/315`, `equation=55`, `bit=136`; gate equation achou os mesmos `4` ganhos no-loss conhecidos e `0` conflitos, projetando `195/315`, `equation=59`; V324 agora bloqueia CSV com `raw_output` se `prediction` nao for label-free | nao e novo dado de equation; esses IDs ja estao cobertos no V475/V510/V515, entao o gargalo e transferencia |
| V517 H200 V515 smoke attempt 1 | job `felipesp1983/6a08f4713308d79117b916c8` falhou barato em `phase=artifacts`: `DATA_REPO` Python apontava para V515, mas `COMMAND_SCRIPT` ainda exportava repo antigo `felipesp1983/kg1-nemotron-training`; nenhum modelo carregou e nenhum treino rodou | corrigido no launcher e transformado em gate: `kg1_pre_paid_job_integration_gate.py --expected-data-repo` agora exige constante e export do repo esperado antes de qualquer job pago |
| V517 H200 V515 smoke retry | job `felipesp1983/6a08f6043308d79117b916de` completou; `DATA_REPO` correto; MoE `gate_up/down` treinaveis; `lm_head` congelado; checkpoint-2 uploaded; eval loss `3.2771 -> 3.2720`; tempo `0.05h` | perda caiu pouco, mas loss nao promove; obrigatorio rodar V518 weak eval label-free do checkpoint-2 antes de qualquer full/package/submit |
| V518 V517 checkpoint-2 weak eval | job `felipesp1983/6a08f97ce48bea4538ba05d2` concluiu a inferencia weak e salvou diagnosticos; resultado label-free `191/315`, `equation=56/155`, `bit=135/160`, `trunc=0`; label-aware debug seria `192/315`, mas nao e promocional | F2/backfire confirmado: queda de bit `136 -> 135` e total nao supera baseline; V517/V518 bloqueados para full/package/submit; qualquer nova GPU precisa de novo sinal CPU e gate anti-regressao |
| V519 V518 row-level backfire audit | baixou diagnosticos V518, comparou contra V516 label-free, destilou apenas 6 linhas mudadas e apagou CSVs brutos; ganho real `518deb39` equation `{ -> $`; perda real `8740ed31` bit `01101000 -> 01111000`; a mesma troca ja aparecia em V488/V494/V496 | novo gate CPU `scripts/kg1_weak_backfire_row_guard.py` protege `8740ed31=01101000`; qualquer novo checkpoint que perder essa linha fica bloqueado mesmo que loss caia |
| V520 local candidate mining | varreu 224 CSVs locais e reavaliou 26 candidatos weak comparaveis contra V516 label-free; nenhum adapter-only local supera `191/315` preservando `bit>=136`, `trunc=0` e `8740ed31`; top scores `201-222` sao solver/postprocessor/integrated | nao existe submit pronto escondido no repositorio; proximo caminho real e transformar sinal de solver em adapter sem repetir a troca `518deb39` por `8740ed31` |
| V521 transfer blocker audit | audit CPU varreu V390/V475/V510/V515/V304; V475/V510/V515 estao bloqueados como estao porque ja falharam em V496/V511/V518; V515 tem `406/473` bit traces train mas bit share nao ponderado e `18.99%`, e mesmo assim V518 perdeu `8740ed31`; V304 nao tem blocker estrutural, mas e historico/broad e nao e pool ativo | GPU fica `blocked_until_new_cpu_transfer_signal`; proximo passo obrigatorio e V522 CPU source-target alignment/learnability gate, nao novo H200 com os mesmos dados |
| V522 source-target alignment audit | melhor referencia teacher V380 tem `31` ganhos no-loss vs baseline label-free e `0` perdas: `23` bit e `8` equation; bit gains concentram em `bit_exact_global_ternary_unique_prediction=13`, `CHO=4`, `MAJ3=4`, `OR=1`, `XOR=1`; V304 contem cobertura fonte alta (`CHO=506`, `MAJ3=709`, `PAR3=105` em treino), enquanto V515 tem cobertura muito pequena (`CHO=4`, `MAJ3=3`) | dataset build permitido, GPU ainda bloqueada; V523 deve criar trace pack fonte-only direcionado a CHO/MAJ3/global ternary + equation classes V516 atuais, sem usar weak labels como treino |
| V523 targeted source trace pack | dataset fonte-only criado e limpo: `1026` train, `219` val; treino `706` bit e `320` equation; `0` overlap weak/full, `0` duplicidade; V286 real passou com token max `749`, offset masks `1026/219`, trunc `0`; V513 passou com `0` blockers, bit traces p50 `99` palavras e equation p50 `52` | e o primeiro dataset novo apos o plateau que passa gates estruturais; GPU ainda depende do V524 porque quota por tokens pode enviesar o loss |
| V524 quota/token objective audit | calculo por literatura/objetivo mostrou que V523 tem `68.8%` bit por linhas, mas `90.7%` bit por tokens de loss (`329702` bit vs `33920` equation), enquanto o sinal V522 e `23/31=74.2%` bit | corrigido no trainer: `hf_job_train_v90.py` agora suporta `LOSS_NORMALIZATION_MODE=example_mean`; qualquer job V523 deve usar esse modo ou encurtar bit traces antes de GPU |
| V525 OpenRouter objective consult | 5 modelos (`gpt-5.5`, Claude Opus 4.7, Gemini 3.1 Pro, Qwen 3.6 Max, DeepSeek V4 Pro) confirmaram que `token_mean` nao pode ser usado com V523; todos recomendam `example_mean`, guard de `8740ed31`, e kill-switch de primeiro checkpoint; 2/5 preferem rebuild V525 antes de GPU | decisao adotada: rodar CPU dry-run de contribuicao por familia; se V523+`example_mean` ainda for dominado por bit, construir V525 com traces bit curtas e token-mass `bit<=70-78%`; se passar, permitir apenas smoke H200 curto com gate `total>=193`, `equation>=57`, `bit>=136`, `trunc=0` |
| V526 example_mean dry-run | `example_mean` passou: bit share por tokens cairia de `90.67%` para `68.81%`, delta vs referencia `5.38pp`, boxed/control/answer checks `0` blockers | autoriza somente um smoke H200 curto V523 com `LOSS_NORMALIZATION_MODE=example_mean` e kill-switch no primeiro checkpoint; nao autoriza treino longo |
| V528 notebooks 0.86/0.85 | notebooks `0.86` sao principalmente packaging do Tinker adapter `kienngx/.../tinker-adapter/1`; tecnicas uteis estao nos notebooks Tong/Pear/Konbu/PJT/ZZYS, nao nos zips de score | usar `0.86` apenas para schema/provenance/package; usar solver/CoT para novo sinal CPU |
| V529 todos os kernels baixados | `704` kernels puxados, `702` parseados; lista filtrada indica P0: `pjt222/nemotron-cot-review`, `pearpn25/bit-cot-85-1364-sample`, `konbu17/bit-manipulation-solver-cot-generator`, `zzys0316/full-pipeline...`; notebooks 0.86 nao resolvem familias | proximo passo efetivo e V530 CPU solver harness antes de qualquer H200 amplo; GPU so se houver novo sinal label-free ou smoke V523 estritamente limitado |
| V530 anexos/datasets | `archive.zip`/Konbu v2 cobre `1508/1602` linhas bit do `competition_train`; `success.csv` tem `1134` CoTs corretas, `671` high-confidence, `0` mismatch de prompt/resposta; `failed.csv` nao pode ser positivo; `archive (1)/(6)` tem `3000` sinteticos, mas `1341` `solver_correct=False`; `archive (2)/(3)` sao dataset oficial duplicado; `archive (4)` adiciona só `28` CoTs numeral/Roman | nova fonte P0 para bit somente; converter/encurtar success high-confidence para KG1 com `\boxed{}` e `example_mean`; nenhum ganho direto para equation |
| V531 anexos V-CARS/Yoiko | `archive (7)` e pacote V-CARS/offline deps/notebook smoke, sem solver novo para bit/equation; `vcars-external-data` aponta para Tatoeba/sentences e nao e fonte P0; `archive (8)` contem LoRA Yoiko ver5 rank 32 alpha 32, base Nemotron correto, `11960` tensores F32, `880138240` params, sem non-LoRA tensors | nao treinar com V-CARS/Tatoeba; registrar Yoiko como candidato P1 de weak eval adapter-only curto, apos gate de config/header e empacotamento root-level temporario; nao e ganho submit-safe sem weak label-free |
| V532 Kaggle dataset/topics/comments search | `70` datasets listados por CLI; `18` candidatos baixados/analisados em temp e apagados; P0/P1 concreto: Konbu BM/ET CoT, `itskshivam` candidate_pool/critic/router, `sohamp13` 3-way selector, `furkankesen` solver-swap, `adityakrishnanmohan` hard triad; topics/comments refresh: `58` topicos, `357` posts, `238` hits | novo plano muda equation para candidate/verifier/canonicalization CPU gate; nao usar bundles com mismatches como gold; Huikang foi auditado localmente no V533 sem commitar ZIP/pesos |
| V532 external equation candidate gate | `critic_v2`, `router_v1`, `selection_v2`, `solver_swap_v1` baixados em temp; candidate pools cobrem `155/155` weak equation, mas seletor label-free por `verifier_score/canonicalization/sympy/rank` cai de `55/155` para `29/155`, com `2` ganhos e `28` perdas | nao promotavel; usar esses datasets como fonte de features/canonizacao/hard negatives, nao como seletor direto nem treino direto |
| V533 Huikang local package | `archive (9).zip` auditado sem extrair pesos; `adapter_v26`: LoRA `all-linear`, r32/alpha32, `418` tensores F32, `386072576` params; `bit_manipulation_3input_traces`: `100` oficiais, `0` mismatch; `2000` sinteticos CHO/MAJ no ZIP local, apesar da metadata citar `10000`; overlap weak bit `10`, incluindo `8` misses atuais | P0 para bit CHO/MAJ trace mining source-only; nao copiar weak rows para treino promocional; adapter v26 e P1 weak eval estatico, nao submit-safe sem gate |

## Decisao Atual V533

Artefatos:

- `artifacts/v525_openrouter_objective_consult/KG1_V525_OPENROUTER_OBJECTIVE_PROMPT.md`;
- `artifacts/v525_openrouter_objective_consult/v525_openrouter_model_responses.json`;
- `artifacts/v525_openrouter_objective_consult/KG1_V525_OPENROUTER_OBJECTIVE_CONSENSUS.md`;
- `artifacts/v525_openrouter_objective_consult/KG1_V525_OPENROUTER_OBJECTIVE_DECISION.md`;
- `artifacts/v526_example_mean_objective_dry_run/KG1_V526_EXAMPLE_MEAN_OBJECTIVE_DRY_RUN.md`;
- `artifacts/v528_score086_notebook_double_check/KG1_V528_SCORE086_NOTEBOOK_DOUBLE_CHECK.md`;
- `artifacts/v529_all_downloaded_notebook_helpfulness/KG1_V529_ALL_DOWNLOADED_NOTEBOOK_HELPFULNESS.md`;
- `artifacts/v530_uploaded_bit_cot_dataset_audit/KG1_V530_UPLOADED_BIT_COT_DATASET_AUDIT.md`;
- `artifacts/v531_uploaded_vcars_yoiko_dataset_audit/KG1_V531_UPLOADED_VCARS_YOIKO_DATASET_AUDIT.md`;
- `artifacts/v532_kaggle_dataset_search_audit/KG1_V532_KAGGLE_DATASET_DOWNLOAD_AUDIT.md`;
- `artifacts/v532_kaggle_dataset_search_audit/discussion_refresh/V512_KAGGLE_DISCUSSION_AUDIT_SUMMARY.md`;
- `artifacts/v532_external_equation_candidate_gate/KG1_V532_EXTERNAL_EQUATION_CANDIDATE_GATE.md`;
- `artifacts/v533_huikang_artifacts_audit/KG1_V533_HUIKANG_ARTIFACTS_AUDIT.md`.

O consenso externo e a auditoria dos notebooks baixados nao autorizam treino
longo nem broad SFT. Eles autorizam duas frentes, nesta ordem:

1. Frente CPU V530, obrigatoria para buscar ganho real:
   - portar/implementar as ideias P0 dos notebooks `pjt222`, `pearpn25`,
     `konbu17` e `zzys0316`;
   - bit: per-bit/bitsum/stride, INHIB/IMPL, CH/CHO, MAJ3, XOR3, GF(2), ANF;
   - equation: concat/reverse concat, operands/result reversal, `+1/-1`,
     divisao/modulo, prefix/suffix operator encoding, `Z_94`/mod-94;
   - medir label-free em source/weak diagnostic, sem usar weak/full como
     treino;
   - gerar traces curtas somente de source rows verificadas.
   - atualizacao V530: usar `archive.zip` como fonte P0 adicional de bit,
     comecando por `bit_manipulation_cot_success.csv` e `confidence=high`;
     `bit_manipulation_cot_failed.csv` entra somente como diagnostico/hard
     negative.
   - atualizacao V533: usar Huikang CHO/MAJ como fonte P0 adicional, mas
     remover qualquer row weak/full do treino promocional; os 8 weak misses
     cobertos sao diagnostico de cobertura, nao exemplos de treino.
2. Frente CPU equation V532, agora obrigatoria antes de qualquer GPU:
   - baixar/analisar apenas os arquivos pequenos necessarios dos datasets
     `itskshivam/nemotron-equation-candidate-critic-v2`,
     `itskshivam/nemotron-equation-candidate-critique-router-v1`,
     `sohamp13/nemotron-equation-candidate-selection-v2` e
     `furkankesen/equation-solver-swap-v1`;
   - criar um gate local que ranqueia candidatos por features label-free:
     `verifier_valid`, `verifier_score`, `canonicalization_status`,
     `profile_normalized_prediction`, `sympy_parse_success`,
     `best_program_family` e voto/top-k;
   - manter `competition_match`, `answer` e `expected_answer` apenas como
     campos de auditoria posterior, nunca como criterio de selecao;
   - testar somente contra os misses atuais de `equation_transform` e exigir
     ganho row-level label-free sem trocar bit por equation;
   - bloquear qualquer CSV com mismatch oficial ou sem prova de answer
     canonicalizada.
   - resultado V532 atual: seletor simples nao promove (`29/155` vs baseline
     `55/155`); proxima acao e derivar regras de canonizacao/hard negatives,
     nao usar o pool como patch direto.
3. Frente GPU V523, opcional e estritamente limitada:
   - `LOSS_NORMALIZATION_MODE=example_mean` ativo;
   - perda por exemplo calculada como `CE_sum / active_label_tokens`;
   - labels decodificados com `\\boxed{` literal, sem `\b`/control chars;
   - prompt tokens com peso zero;
   - offset masks completos;
   - V526 ja passou este dry-run (`example_mean_bit_share=0.688109`).
4. Frente adapter externo V531, opcional e mais barata que treino:
   - avaliar Yoiko ver5 apenas como candidato adapter-only, sem treino;
   - antes de rodar, validar `adapter_config`, header safetensors, regex
     `target_modules`, ausencia de non-LoRA tensors e zip root-level;
   - usar weak eval label-free curto, com FinOps kill-switch se cair abaixo do
     baseline nas primeiras metricas;
   - promover somente se superar baseline raw atual sem perder bit/truncation.
5. Qualquer smoke V523/V530/V531/V532 precisa falhar fechado se:
   - `total < 193/315`;
   - `equation < 57/155`;
   - `bit < 136/160`;
   - `trunc != 0`;
   - `8740ed31 != 01101000`;
   - `518deb39 != $`.

Promocao/package/submit continua bloqueado ate haver ganho label-free real em
weak/full. `eval_loss` menor sem ACC melhor nao promove.

## Achados Principais V484-V492

Os dois arquivos OpenRouter de 16/05/2026 reforcam um ponto tecnico mais forte
que qualquer nova busca de dados: a linha de treino precisa provar continuidade
PEFT/LoRA antes de gastar GPU.

Consenso acionavel:

| Achado | Status | Acao |
|---|---|---|
| `target_parameters` perdido ou carregado por caminho manual pode quebrar continuidade do adapter MoE | risco alto, agora bloqueado por gate | usar `PeftModel.from_pretrained(..., is_trainable=True)` como padrao |
| `adapter_config.json` precisa bater com env de treino | obrigatorio | gate compara `r`, `alpha`, `target_modules`, `target_parameters` |
| key/shape/dtype de `adapter_model.safetensors` precisa ser auditado | obrigatorio antes de GPU | criar/rodar CPU round-trip gate |
| `modules_to_save` nao pode carregar pesos cheios | obrigatorio | permitir `lm_head` apenas como LoRA em `target_modules`, nunca como modulo salvo inteiro |
| `answer_span_loss_weight=12.0` pode mascarar ACC | risco nao comprovado, mas recorrente no double check | logar componentes e usar micro-ACC como kill-switch; nao aumentar peso sem evidencia |
| `eval_loss` baixo nao comprova ACC | regra permanente | promover so por weak/full ACC e truncation |
| Mais epochs, LR sweep ou H200 longo sem novo gate e desperdicio | removido | FinOps cancela antes de custo |

Evidencia externa verificada:

- PEFT documenta `target_parameters` para parametros MoE que nao sao `nn.Linear`.
- PEFT documenta que, quando o adapter original usa `target_parameters`, a
  injecao a partir de `state_dict` exige a config PEFT correta.
- A suite PEFT possui testes especificos para `target_parameters` em modelos
  Llama4/GPT-OSS, incluindo `mlp.experts.gate_up_proj` e
  `mlp.experts.down_proj`.
- PR PEFT `#2710` corrigiu problemas de `target_parameters` e menciona riscos
  de erro silencioso com multiplos adapters; portanto nossa regra deve ser
  fail-closed.

Fontes: `artifacts/v484_openrouter_uploaded_audit/V484_OPENROUTER_UPLOAD_AUDIT.md`.

Atualizacao V491/V492: o double check com o prompt completo e o arquivo
`C:\Users\davis\Downloads\OpenRouter Chat Sat May 16 2026 (2).json` reforcou
quatro conclusoes praticas:

| Achado | Decisao no plano |
|---|---|
| V487/V488 carregavam MoE `target_parameters`, mas a allowlist treinavel era `q/k/v/o/lm_head`; isso deixa `up_proj/down_proj` em modo `frozen_active` | proximo smoke deve exigir `target_parameters_trainability_mode="trainable"` e tensores treinaveis nao nulos para `up_proj/down_proj` |
| `lm_head` treinavel e suspeito direto de flip de bit e truncation, porque altera distribuicao de `0/1` e EOS | remover `lm_head` de `TRAINABLE_LORA_MODULES` no proximo smoke; so reativar como ablation documentada |
| `ANSWER_SPAN_LOSS_WEIGHT=12.0` pode baixar loss sem melhorar ACC estrito | pin em `1.0` para smokes oficiais; qualquer valor maior vira experimento separado e nao promocional |
| O `equation=57` de V488 precisa ser revalidado em raw output, porque pode vir de extracao expected-aware e nao de melhoria real do adapter | todo ganho novo precisa ter diff `raw_output`, `simple_extracted`, `expected_aware_extracted` e familia antes de promocao |

Consenso util dos modelos: a proxima tentativa nao e "mais treino"; e um teste
de mecanismo. Se o MoE treinavel com `lm_head` congelado nao preservar
`bit>=136` no primeiro checkpoint, broad SFT segue bloqueado por FinOps.

Atualizacao V494: o teste de mecanismo foi executado. O adapter treinou com
`target_parameters_trainability_mode="trainable"`, `up_proj/down_proj`
treinaveis, `lm_head` congelado e `ANSWER_SPAN_LOSS_WEIGHT=1.0`, mas a weak
eval longa retornou `190/315`, `equation_transform=57/155`,
`bit_manipulation=133/160` e `truncated=1`. A auditoria local confirmou que a
metrica estrita esta sincronizada e que o valor de loss nao deve mais ser usado
como preditor de ACC. O unico ganho liquido vs V290 checkpoint-6 foi
`518deb39` em equation, contra tres perdas em bit (`5b9964c7`, `8740ed31`,
`59bee375`). Alem disso, a extracao expected-aware adicionou 1 acerto em
equation (`4bb8c6cd`) que nao existe na extracao simples.

Correcao critica do double check: V493/V494 nao treinou o dataset V475. O
launcher usou `data/v390_equation_bit_replay_mix/20260514T193847Z`, com 5031
linhas de treino e pressao efetiva `bit=0.926178`, `equation=0.073822`. O
V475 CPU-gated que projetava `equation 56 -> 60` e `weak 196` e outro dataset:
`data/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix`,
com 1312/328 linhas, `equation=800/200`, `bit=512/128`, token max `331`,
truncation `0`, offset masks completos e zero overlap com weak/full. Isso
justificou o smoke V495 no V475, mas V496 mostrou que esse sinal CPU nao
transferiu para LoRA submit-safe.

Atualizacao V495/V496: o smoke curto V475 foi executado. O treino confirmou que
as pecas tecnicas estavam no lugar (`target_parameters` MoE treinaveis,
`lm_head` congelado, `ANSWER_SPAN_LOSS_WEIGHT=1.0`, hashes V475 corretos), mas
a weak eval bloqueou promocao:

| Metrica | V290 checkpoint-6 | V496 V495 checkpoint-2 | Delta |
|---|---:|---:|---:|
| Total weak | 192/315 | 191/315 | -1 |
| equation_transform | 56/155 | 57/155 | +1 |
| bit_manipulation | 136/160 | 134/160 | -2 |
| truncated | 0 | 1 | +1 pior |

Auditoria V496:

- metric path OK (`verify_answer` estrito);
- V496 mudou 17 linhas, mas so 3 mudaram corretude;
- ganho real: `518deb39` em equation;
- perdas reais: `8740ed31` e `59bee375` em bit;
- o extra `4bb8c6cd` de expected-aware extraction ja existia no baseline V290,
  portanto nao e aprendizado novo;
- V496 gerou `1,504,306` completion tokens em `516.9s`; o gargalo e geracao
  longa com thinking/max_tokens oficiais, nao falta de H200.

Decisao: V475 SFT transfer tambem esta bloqueada. A rota mais rapida agora e
CPU-first: descobrir teacher/verifier de equation que projete pelo menos
`equation>=60`, `bit>=136`, `trunc=0` antes de qualquer novo H200.

Atualizacao V497: a auditoria CPU residual separou o problema em tres grupos:

| Grupo | Evidencia | Acao |
|---|---|---|
| 4 ganhos V324 numeric nao transferidos | `7688e06e`, `274def88`, `d1bd7478`, `c5b058d6` sao corretos no CPU solver e continuam errados no adapter | transformar em traces numericos deterministas curtos, com hard negatives e sem weak/full labels como alvo de treino |
| 1 ganho V496 simbolico isolado | `518deb39` virou correto no adapter, mas nao veio do V324 numeric gate | usar apenas como padrao de diagnostico de formato/pontuacao; nao treinar diretamente pelo label weak |
| 2 perdas bit bloqueantes | `8740ed31` virou `01111000`; `59bee375` virou `2` | qualquer novo dataset/job deve falhar se nao preservar essas duas linhas e saida binaria exata |

Top clusters residuais continuam majoritariamente `equation_symbolic_punct`
opacos. Portanto, para ganhar hoje, o alvo mais barato e mais controlavel nao e
"resolver todos os simbolicos"; e transferir corretamente os 4 numeric gains ja
verificados pelo CPU gate sem perder bit.

Atualizacao V498/V499: o pacote V498 foi criado para atacar somente as tres
classes numericas que explicam os quatro ganhos CPU de V324. Ele nao usa linhas
weak/full como treino; essas linhas entram apenas como fingerprints proibidos.
O gate real V286 passou com `prompt_truncation_rate=0`, `completion_tokens_dropped=0`,
`offset_masks=train 1712/1712` e `validation 428/428`. O V478 local e o debug
HF do V499 confirmaram que os pesos do launcher nao repetem a sub-representacao
de bit: o objetivo efetivo fica `bit=0.390244` e `equation=0.609756`. Isso
autoriza um unico smoke H200 curto. Se o primeiro checkpoint repetir o padrao
V496 (`equation=57` com `bit<136` ou truncation), cancelar por FinOps e voltar
ao CPU teacher, sem broad SFT.

Atualizacao V511: o smoke canonico com V510 executou no H200 e validou as pecas
de infraestrutura que antes eram suspeitas: dataset HF correto, MoE
`target_parameters` treinaveis, `lm_head` congelado, adapter seed carregado,
checkpoint salvo e upload final completo. O resultado tecnico, porem, nao
melhorou o objetivo local: `baseline_eval_loss=2.8125` e `final_eval_loss=2.8128`.
Pela regra FinOps, esse candidato nao deve receber weak/full eval pago nem
treino mais longo. O V511 prova que o gargalo atual nao e mais "job nao roda";
e transferencia de sinal solver/trace para comportamento do adapter.

Atualizacao V512: a auditoria via API das discussions do Kaggle carregou todos
os `140` topicos listados nos arquivos locais e varreu `586` posts. Achados que
entram no plano:

| Fonte | Achado concreto | Impacto no plano |
|---|---|---|
| Discussion `690307`, Tong Hui Kang | bit-pair/bitsum/stride resolve a maior parte de bit sem enumerar todas as expressoes; o proprio autor declara que a transferencia depende do modelo reproduzir exatamente a CoT | bit continua guardrail; novo bit dataset so vale se gerar traces curtos e stride-verificados, nao full brute force longo |
| Discussion `689915`, Tong Hui Kang | vencedor usou SFT, CoT deterministica, objetivo de maximizar minimo logprob e codigo gerado por solver; nao apostou em RL/distill generico | proximo dataset deve passar teste de learnability/logprob antes de GPU |
| Discussion `690756`, Mark Cooper | full-byte unary/global tem baixa divergencia; per-bit/pair pode cobrir mais, mas sem restricao global diverge muito | recuperar os `161` bit descartados no V514 com gate CPU residual full-byte/3-input, aceitando apenas predicao unica sem conflito |
| Discussion `697491`, Taha/Russell | dataset mais correto por solver pode piorar LB; oversampling, trace dificil, duplicate CoT e format clash causam regressao; traces acima de ~1300 tokens tendem a ser caros e pouco transferiveis | todo novo trace precisa auditoria de duplicidade textual, comprimento e logprob base; evitar mudar formato global |
| Discussion `694710`, Taha/CPMP | prompt-loss masking/pretokenized response mask e requisito; loss baixo sem mask correta e ilusao | manter offset-mask gate obrigatorio e nao interpretar loss sem ACC |
| Discussion `690891`, Mark Cooper | equation/cryptarithm devem ser separados em solver coverage e modelo gerar CoT; heuristicas para guess sao limite informacional | equation deve ser CPU solver + trace learnability; nao broad SFT |
| Discussions `684192`/`689877` | operador da query pode estar ausente nos exemplos; nesses casos a regra pode ser ambigua ou nao-constrangida | criar gate semantico equation com `query_operator_seen`, `same_operator_examples`, `candidate_count`, `conflict_count`, `derivable_vs_guess` |
| Discussion `698293`, lkevincc/Taha | solver simbolico gold-conditioned mostra estrutura latente, mas nao e submit-safe e nao transfere automaticamente para LoRA | usar como oracle de pesquisa/rotulagem, nunca como runtime ou label weak/full |
| Discussion `694556`, Murugesan/NguyenThanhNhan | categorias simbolicas tem muitos duplicados e DSL publica cobre so parte; muitos traces sao fallback/guess | dataset simbolico precisa deduplicacao forte e marca de "derivavel" vs "guess" |

Decisao apos V511/V512/V513/V514: parar H200 de treino amplo sobre o V510
answer-only. O caminho mais rapido e barato agora e CPU-only ate o V514 ser
reproduzido no HF e provar:

- zero duplicate-CoT conflitante;
- max trace preferencial abaixo de `1300` tokens;
- response-mask/offset-mask completo;
- bit replay com CoT deterministica curta, nao somente `Final answer`;
- bit residual testado com full-byte/ternary/3-input apenas quando houver
  predicao unica sem conflito;
- equation rows marcadas como derivaveis vs guess, com operador da query visto
  ou ambiguidade explicitamente bloqueada;
- base logprob/learnability melhor ou igual ao baseline por subfamilia;
- CPU projection `equation>=60`, `bit>=136`, `trunc=0`;
- nenhum ganho dependente de expected-aware extraction.

## Regras Ativas

1. Submit valido e somente adapter-only: `adapter_config.json` e
   `adapter_model.safetensors` no pacote. Sem runtime solver, verifier,
   postprocessor, logit mask, prompt hack ou threshold.
2. Weak/full sao apenas avaliacao e gate. Nao podem construir dataset, pares,
   chosen/rejected, desempate de regra ou cherry-pick.
3. ACC de promocao usa predicao label-free:
   `extract_final_answer(raw_output)` seguido de
   `src.competition_utils.verify_answer`. `answers_equivalent` e
   diagnostico-only.
4. Extracao expected-aware e diagnostico-only para datasets com label; ela nao
   pode produzir a coluna `prediction` usada em gate, package ou submit.
5. Nenhum job pago roda se `target_parameters` estiver ausente, divergente ou
   carregado por modo manual sem round-trip CPU aprovado.
6. `modules_to_save` deve ficar vazio no seed e no pacote final. `lm_head` pode
   aparecer em `target_modules` como LoRA, mas nao como peso cheio salvo.
7. Se um launcher usa `LORA_TARGET_PARAMETERS` MoE junto com allowlist
   `TRAINABLE_LORA_MODULES`, ele deve declarar explicitamente se esses
   `target_parameters` precisam ser treinaveis:
   `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` ou `1`.
8. Todo launcher/job/notebook novo ou alterado passa por
   `scripts/kg1_static_safety_gate.py`.
9. Antes de job pago, rodar `scripts/kg1_pre_paid_job_integration_gate.py` no
   schema correto (`preference` ou `sft`) e `scripts/hf_job_preflight_gate.py`;
   o preflight deve falhar se qualquer linha de treino vier marcada como
   gate/weak/full usada para treino ou se as flags anti-leakage estiverem
   ausentes.
10. Dataset de treino ativo deve vir do V510 ou de uma versao posterior com
   manifesto equivalente. V439 fica excluido ate ser reconstruido com
   renderizacao simbolica label-free validada; V443 fica excluido por estar
   vazio.
11. FinOps: cancelar job que nao possa mais superar `total>192`,
   `equation>56`, `bit>=136`, `truncated=0`.
11. H200 pode ser usada ate 1 hora por execucao. Acima disso exige autorizacao
   humana.
12. Todo erro novo entra em `KG1_ERROR_LEDGER_2026_05_15.md` antes de novo job
   pago.
13. Toda versao nova precisa quadro comparativo contra V291/V290.
14. Job promocional com `LORA_TARGET_PARAMETERS` deve declarar
    `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`, incluir `up_proj/down_proj`
    na allowlist treinavel e falhar se o manifesto nao registrar modo
    `trainable`.
15. `lm_head` fica fora de `TRAINABLE_LORA_MODULES` nos smokes de melhoria.
    Reativar `lm_head` exige ablation isolada, porque V477/V488 sugerem risco
    de flip de bit e truncation.
16. `ANSWER_SPAN_LOSS_WEIGHT` fica em `1.0` nos smokes promocionais. Valores
    maiores sao diagnostico-only ate provarem ganho de ACC estrito.
17. Nenhum ganho baseado apenas em `expected_aware_extracted` promove pacote.
    O diff precisa mostrar que a saida bruta ou a extracao simples tambem nao
    introduz regressao ilegal.
18. `PRETOKENIZED_VAL_COPY_ONLY=1` e diagnostico-only. Ele invalida
    independencia de `eval_loss` porque pode copiar treino para validacao; o
    static gate bloqueia isso em jobs/notebooks promocionais.
19. Weak eval promocional com `scripts/hf_job_weak_eval_v245.py` deve declarar
    controles long-context: `KG1_DISABLE_THINKING=0`, `KG1_NO_PROMPT_SUFFIX=0`,
    `KG1_MAX_TOKENS=7680`, `KG1_MAX_MODEL_LEN=8192`,
    `KG1_MAX_NUM_SEQS=64`; os defaults agora tambem seguem esse modo.
20. Weak eval promocional falha por padrao se nao passar
    `total>=196`, `equation>=60`, `bit>=136`, `trunc=0`. Para sweep barato,
    marcar explicitamente `KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=1`.
21. V499 e V501 sao artefatos de forense, nao candidatos. Qualquer launcher que
    referencie os repos desses adapters deve falhar no static/pre-paid gate.
22. Regra F2/backfire permanente: antes de qualquer job ou notebook
    promocional, analisar se houve `F2 backfired`, bug silencioso ou regressao
    mascarada por loss. Se houver suspeita concreta, entrar em crisis mode:
    bloquear promocao, registrar no ledger, reverter/ajustar a rota afetada e
    corrigir o gate para impedir repeticao. O static/pre-paid gate deve exigir
    `KG1_CRISIS_MODE_BACKFIRE_GUARD=1` em launchers pagos.

## Plano Cronologico Ativo

### P0 - Congelar Gasto Pendente

Objetivo: parar custo enquanto a continuidade PEFT nao estiver provada.

Executar:

- Nao abrir novo H200/A100 para repetir V476/V477/V480.
- Cancelar jobs que ja nascem abaixo do gate ou dependem de `eval_loss`.
- Manter apenas CPU/HF CPU barato ate o round-trip passar.

Promove para P1 quando: nenhum launcher ativo usa `INIT_ADAPTER_LOAD_MODE=manual`
com `LORA_TARGET_PARAMETERS`.

### P1 - Corrigir Continuidade PEFT

Objetivo: garantir que o adapter inicial V290/V291 e carregado no mesmo espaco
estrutural que gerou 192/315 e que o manifesto declare claramente quais LoRA
ficaram treinaveis.

Executar:

- Padrao de treino: `PeftModel.from_pretrained(base, init_adapter, is_trainable=True)`.
- Bloquear `INIT_ADAPTER_LOAD_MODE=manual` para adapters com
  `target_parameters`.
- Verificar no gate:
  - `adapter_config.json` preserva `target_parameters`.
  - `adapter_config.json` tem `modules_to_save` vazio.
  - LoRA tensors de `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`
    existem.
  - `target_parameter_lora_tensors` nao e vazio.
  - `target_parameter_trainable_lora_tensors` e
    `target_parameters_trainability_mode` ficam registrados no manifesto.
  - nomes treinaveis contem os modulos obrigatorios.
  - nao ha warnings de missing adapter keys.
  - SHA256 de `adapter_config.json` e fingerprints de keys/shapes/dtypes sao
    registrados antes e depois do treino.

Promove para P2 quando: CPU preflight e static gate passam sem excecao.

### P2 - CPU Round-Trip Gate V484/V485

Status: implementado e aprovado no adapter seed V290 checkpoint-6.

Objetivo: provar equivalencia estrutural antes de qualquer GPU.

Implementar/rodar um gate CPU que:

- baixa/carrega o adapter seed V290/V291;
- carrega com `PeftModel.from_pretrained(..., is_trainable=True)`;
- salva em diretorio temporario;
- recarrega;
- compara `adapter_config.json`, lista de keys, shapes, dtypes e contagem de
  tensores LoRA;
- confirma `modules_to_save=[]` ou `null`;
- roda um micro forward/backward em batch dummy quando aplicavel e confirma se
  os parametros LoRA esperados estao treinaveis ou explicitamente
  `frozen_active`;
- emite manifesto com `hf_gpu_allowed=true` somente se tudo bater.

Implementacao atual:

- usa metadados Hub/safetensors para evitar download multi-GB;
- valida `target_modules`, `target_parameters`, `modules_to_save`, keys,
  shapes, dtypes, contagens LoRA e fingerprints;
- aceita o alias estrutural real do Nemotron em que
  `mlp.experts.gate_up_proj` aparece como LoRA em
  `mixer.experts.<id>.up_proj`;
- aceita apenas o `lm_head.base_layer.weight` conhecido do seed como tensor
  nao-LoRA fingerprintado; qualquer `modules_to_save` segue bloqueado;
- foi conectado ao launcher V391 antes do download de dataset e antes de
  qualquer treino pago.

Resultado V485 seed:

| Campo | Valor |
|---|---:|
| resolved revision | `75909c9b40d8b7fa846d379d9d764fa33daeb9e2` |
| adapter_model bytes | 4259063856 |
| tensor_count | 12011 |
| target_parameter_lora_tensors gate_up | 5934 |
| target_parameter_lora_tensors down | 5934 |
| modules_to_save | `[]` |
| allowed non-LoRA key | `base_model.model.lm_head.base_layer.weight` |
| hf_gpu_allowed | `true` |

Manifest: `artifacts/v485_peft_roundtrip_gate/v485_seed_adapter_manifest.json`.

Promove para P3 quando: round-trip manifesto aprovado.

### P3 - V493 Checkpoint-2 Weak Eval V494

Objetivo: testar se a correcao de mecanismo V493 transferiu para ACC real. O
loss melhorou pouco (`1.9233 -> 1.9152`), entao a unica decisao valida vem do
weak eval.

Status: executado em H200.

- Rodou V494 H200 weak eval no `checkpoint-2` do repo
  `felipesp1983/kg1-nemotron-lora-v493-nemo-h200-moe-trainable-no-lmhead-v290ckpt6`.
- Usou controles long-context:
  - `KG1_DISABLE_THINKING=0`
  - `KG1_NO_PROMPT_SUFFIX=0`
  - `KG1_MAX_TOKENS=7680`
  - `KG1_MAX_MODEL_LEN=8192`
  - `KG1_MAX_NUM_SEQS=64`
- Gate enforcado:
  - `total > 192`
  - `equation_transform > 56`
  - `bit_manipulation >= 136`
  - `truncated = 0`

Resultado:

| Metrica | V290 checkpoint-6 baseline | V494 checkpoint-2 | Delta |
|---|---:|---:|---:|
| Total weak | 192/315 | 190/315 | -2 |
| equation_transform | 56/155 | 57/155 | +1 |
| bit_manipulation | 136/160 | 133/160 | -3 |
| truncated | 0 | 1 | +1 pior |

Auditoria de metrica:

- `scripts/audit_v449_acc_metric_integrity.py` passou com
  `decision=metric_path_ok`.
- `simple_correct=189` e `expected_aware_correct=190`; logo 1 acerto de
  equation (`4bb8c6cd`) depende de extracao expected-aware e nao pode ser
  tratado como ganho bruto do adapter.
- Strict vs permissive divergiu em 15 linhas de bit; `verify_answer` estrito
  esta correto e impede overcount numerico em strings binarias.
- Diff vs V290 checkpoint-6: ganho `518deb39` em equation; perdas
  `5b9964c7`, `8740ed31` e `59bee375` em bit.

Decisao atualizada: falhou para o mix antigo V390/V326 e tambem para o V475
CPU-gated apos V495/V496. Nao repetir V390/V326 nem V475 em H200. SFT pago fica
bloqueado ate existir novo sinal CPU independente que projete `equation>=60`,
`bit>=136`, `trunc=0` e `total>192`.

### V391/V486 Objective Balance Update

V391 foi lancado em H200 mas parou antes do treino, no gate V478. Isso foi
correto: os pesos `equation_numeric_* = 10.00` faziam equation dominar o
objetivo efetivo.

| Versao | Status | Bit share efetivo treino | Equation share efetivo treino | Decisao |
|---|---|---:|---:|---|
| V391 | rejeitado antes do treino | 0.135975 | 0.864025 | nao treinar; objetivo desequilibrado |
| V486 | probe CPU aprovado | 0.207788 | 0.792212 | candidato a smoke curto |

Artefatos:

- `artifacts/version_diffs/V486_VS_V391.md`
- `artifacts/v486_objective_weight_probe/V486_OBJECTIVE_WEIGHT_PROBE.md`
- `artifacts/v486_objective_weight_probe/eq_6.json`

Regra atualizada: qualquer novo HF job deve passar o gate V478 e manter
`bit_manipulation` com pressao efetiva minima. `eval_loss` continua sendo
diagnostico secundario; promocao depende de weak micro-ACC.

Atualizacao V487: V486 passou V485/V478 e falhou antes do treino porque o
script de treino nao aplicava o alias estrutural de `target_parameters`
(`mlp.experts.gate_up_proj` salvo/carregado como `mixer.experts.<id>.up_proj`).
O matcher foi alinhado com V485 e V487 e o relancamento correto.

Artefato: `artifacts/version_diffs/V487_VS_V486.md`.

Atualizacao V488: o treino V487 completou em H200 e confirmou LoRA ativa para
os `target_parameters`, mas a weak eval focada do checkpoint-10 produziu
`191/315`, `equation_transform=57/155`, `bit_manipulation=134/160` e
`truncated=1`. Portanto a continuidade PEFT era um bug real, mas nao era
suficiente para romper o plateau. A rota de repetir o mesmo SFT/mesmo objetivo
esta bloqueada por FinOps ate existir novo sinal CPU que preserve bit e
truncation.

Atualizacao V489/V492: o diff linha a linha confirmou que a metrica estrita esta
correta e que V488 teve exatamente um ganho observado de equation (`518deb39`) e
duas regressoes reais de bit (`8740ed31`, `59bee375`), sendo uma com truncation.
O ganho de equation ainda precisa ser classificado como aprendizado bruto do
adapter ou efeito de extracao expected-aware antes de qualquer promocao. A
auditoria tambem mostrou um gap de F2/observabilidade: V487 carregava
`target_parameters`, mas a allowlist treinavel era `q/k/v/o/lm_head`; logo
`up_proj/down_proj` ficavam frozen-active, nao comprovadamente treinados. O
script de treino agora grava `target_parameters_trainability_mode` e os contadores
de tensores trainaveis por `target_parameter`; launchers futuros precisam
declarar explicitamente se esperam `target_parameters` treinaveis. A auditoria
tambem corrigiu dois bugs silenciosos de validação: uma chave duplicada no
static gate que anulava checks de `hf_job_train_v90.py`, e a extracao
expected-aware que agora so pode desambiguar o ultimo boxed.

Artefato: `artifacts/v489_solution_integrity_audit/V489_SOLUTION_INTEGRITY_AUDIT.md`.

Atualizacao V490/V492: o double check em modo debug confirmou que o gap atual
nao e um erro simples de ACC, split, tokenizacao ou threshold. V488 observou
1 linha a mais de equation (`518deb39`) e perdeu 2 linhas de bit (`8740ed31`,
`59bee375`), com truncation em `59bee375`. O dataset V390/V326 tem 5031/532
linhas, IDs/prompts unicos, zero overlap train/val e flags `gate/weak/full`
como `False`. A proxima tentativa so faz sentido se for estruturalmente
diferente: testar `target_parameters` MoE como treinaveis, nao apenas
frozen-active, e auditar se qualquer ganho de equation aparece na saida bruta.

Artefato: `artifacts/v490_debug_double_check/V490_DEBUG_DOUBLE_CHECK_2026_05_16.md`.

### P3 - Smoke HF Minimo V511

Objetivo: testar uma rota submit-safe depois da limpeza de dados V509/V510,
sem repetir os datasets antigos que ja falharam. O smoke V511 combina:

- dataset unico V510, com V498 + V475 + V460 e V439/V443 excluidos;
- MoE LoRA treinavel em `up_proj/down_proj`;
- `lm_head` congelado;
- `ANSWER_SPAN_LOSS_WEIGHT=1.0`;
- bit replay com peso efetivo suficiente para preservar `136/160`;
- fail-fast em 2 steps.

Status final: executado e bloqueado.

Evidencia pronta:

- dataset HF: `felipesp1983/kg1-nemotron-training`
- dataset commit: `40e71a686d9970c3c842d26dcf89200fc4990a51`
- train: `2627` rows,
  SHA256 `9033e794bad98679f26bb2fc7f1eb5d4d7f32d06ef6231ee6e0fffc66fc70d3b`
- val: `637` rows,
  SHA256 `062514b8a74ba3656df44ad99667ba63dda69f56d41a20ffb0500f17393ceea8`
- objective alignment:
  - train bit effective share `0.475163`
  - train equation effective share `0.524837`
  - val bit effective share `0.441860`
  - val equation effective share `0.558140`
- static safety gate: passou.
- pre-paid integration gate: passou.
- launcher debug: passou, H200 `0.083333 USD/min`, `timeout=3600`.
- job HF: `felipesp1983/6a08dc43e48bea4538ba02ce`.
- runtime: `0.05h`.
- trainability: `target_parameters_trainability_mode="trainable"`,
  `mlp.experts.gate_up_proj=5934` e `mlp.experts.down_proj=5934`
  trainable LoRA tensors.
- result: `baseline_eval_loss=2.8125`, `final_eval_loss=2.8128`.
- upload: checkpoint-2 e final adapter enviados para
  `felipesp1983/kg1-nemotron-lora-v511-nemo-h200-v510-canonical-v290ckpt6`.

Decisao:

- nao weak-eval pago;
- nao packagear;
- nao submeter;
- nao continuar V511 em treino longo;
- usar o log como evidencia de que a pipeline roda, mas o dataset/objetivo nao
  transferiu sinal.

Configuracao minima do launcher:

  - `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`
  - `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`
  - `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`
  - `ANSWER_SPAN_LOSS_WEIGHT=1.0`
  - `lm_head` ausente da allowlist treinavel
  - `LEARNING_RATE=2.0e-8`
  - `FINAL_LEARNING_RATE=5.0e-9`
  - `MAX_STEPS=2`
  - `SAVE_EVERY_STEPS=2`
  - `EVAL_EVERY_STEPS=2`
  - `KG1_CRISIS_MODE_BACKFIRE_GUARD=1`

Pesos do V511:

- source weights:
  - `v498_bit_replay_guardrail_from_v475=3.00`
  - `v475_v217_bit_replay_guardrail=3.00`
  - `v217_bit_replay_guardrail=3.00`
  - `v498_numeric_teacher_trace_pack=1.00`
  - `v475_v325_equation_no_loss_distill=1.00`
  - `v460_v459_numeric_one_rule_micro_dataset=1.00`
- subcategory weights:
  - bit replay `1.00`
  - equation direct/colon/minus/hard-negative families `1.00`
  - V274 guarded numeric minus signed `1.00`

Gate:

| Metrica | Requisito |
|---|---:|
| Total weak | > 192 |
| equation_transform | > 56 |
| bit_manipulation | >= 136 |
| truncated | 0 |

Manifesto obrigatorio antes de aceitar o resultado:

- `target_parameters_trainability_mode == "trainable"`;
- `target_parameter_trainable_lora_tensors` nao vazio para
  `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`;
- `trainable_by_module["up_proj"] > 0` e `trainable_by_module["down_proj"] > 0`;
- `trainable_by_module["lm_head"] == 0`;
- `ANSWER_SPAN_LOSS_WEIGHT` deve estar explicito e auditado; para treino
  promocional de ACC, o valor `1.0` agora e considerado smoke-only, nao
  suficiente para alinhar loss com resposta final;
- `simple_extracted` vs `expected_aware_extracted` auditado por familia.

Se o smoke der `equation=57` com perda de bit, nao promove. Esse caso ja
ocorreu duas vezes: V477 (`equation=57`, `bit=135`, `trunc=0`) e V488
(`equation=57`, `bit=134`, `trunc=1`). O ganho isolado de uma linha em
equation nao vale se derruba o guardrail de bit ou truncation.

### P4 - CPU Learnability Gate Antes De Novo GPU

Objetivo: transformar ganhos CPU de equation/bit em comportamento aprendivel
do adapter, sem repetir V493/V495/V511.

Executar somente em CPU enquanto nao houver novo sinal. V511 nao repetiu a
regressao de ACC porque nao recebeu weak eval, mas o loss local piorou
`2.8125 -> 2.8128`; portanto tambem nao autoriza GPU longo. O objetivo agora e
descobrir um teacher/verifier/trace que produza ganhos de equation sem usar
weak/full como label de treino e sem perder bit.

- construir dataset hard-negative curto com apenas regras CPU verificadas;
- target final-answer-only ou trace deterministico curto, sem auditoria textual
  longa;
- rejected = resposta exata errada do baseline;
- chosen = resposta correta curta;
- leakage gate por `id`, `prompt_sha256`, prompt normalizado e n-gram.
- novo gate de learnability antes de GPU:
  - nenhum duplicate-CoT conflitante;
  - trace max preferencial abaixo de `1300` tokens;
  - base logprob nao pior que baseline por subfamilia;
  - response-mask/offset-mask `100%`;
  - diff label-free separado de expected-aware.

Nao executar:

- SFT amplo;
- mais epochs sem novo sinal;
- dataset sintetico que ja falhou transfer;
- treino com labels selecionados por weak/full.
- hard negatives diretamente extraidos de IDs weak/full como labels de treino.
  `518deb39`, `8740ed31` e `59bee375` so podem ser usados como diagnostico de
  diff/gate, nao como exemplos de treino promocional.

Como P3 falhou no V390/V326 e depois V495/V496 falhou no V475:

- nao continuar o mesmo job;
- nao abrir nova H200 para repetir V493/V494/V495/V496, V390/V326 ou V475
  sem novo sinal CPU;
- rodar CPU-only para auditar os residuos de equation e bit:
  - mapear os 99 misses de equation por padrao simbolico;
  - marcar cada equation row como `derivable` ou `guess` usando
    `query_operator_seen`, quantidade de exemplos do mesmo operador, numero de
    candidatos e conflitos; rows ambiguas nao viram teacher promocional;
  - separar ganho bruto vs ganho por extracao;
  - identificar qualquer regra que resolva pelo menos +4 equation com zero
    perda de bit quando convertida para trace/teacher;
  - produzir dataset curto somente se o solver CPU independente passar
    leakage/contract gate.
- broad SFT fica encerrado ate existir novo sinal CPU submit-safe e aprendivel.

### P5 - Bit Como Guardrail

Objetivo: preservar os 136/160 enquanto equation sobe.

Executar:

- manter replay de bit em todo treino de equation;
- validar que qualquer ganho de equation nao derruba bit;
- usar bit-pair/bitsum/stride somente para gerar traces curtos e verificados se
  houver cobertura nova maior que a linha V304.
- antes de GPU, rodar `V514b` CPU residual: tentar recuperar os `161` bit rows
  descartados no V514 com solver full-byte/ternary/3-input; aceitar somente
  predicao unica, sem conflito e com trace curto. Se nao houver cobertura nova,
  manter V514 como esta e nao inflar dataset com guess.
- V515 deve materializar esse residual como dataset/gate pequeno:
  - comparar os IDs bit originais do V510 contra os IDs convertidos pelo V514;
  - rodar o solver full-byte/global apenas nos descartados;
  - aceitar somente regra `fullbyte_unique_prediction` com verificacao exata;
  - anexar traces curtos ao V514 e rerodar V286 + V513;
  - nao abrir GPU se o ganho de cobertura for zero ou se tokenizacao/trace
    regredir.

Nao abrir job bit-only enquanto teacher CPU nao transfere para adapter.

### P6 - Package/Submit

Objetivo: submeter apenas quando existir ganho real.

Condicao minima:

- weak > 192;
- equation > 56;
- bit >= 136;
- trunc 0;
- full official-like > 823/947 ou evidencia equivalente via gate oficial-like.

Sem isso, nao packagear e nao submeter.

## Itens Removidos Do Plano Ativo

| Item | Motivo |
|---|---|
| Repetir H200 longo por eval_loss | loss nao correlacionou com ACC |
| Mais epochs/steps sem novo dado | quatro dias de plateau; custo sem sinal |
| V435E misto e format negatives | contaminado e bloqueado |
| V447/V448 trace SFT limpo | nao transferiu para adapter |
| V464/V468/V469 derivados | rota contaminada/quarentenada |
| Public adapters/submissions de terceiros | somente tecnica, nunca peso/submissao direta |
| Solver/verifier no runtime submit | contra regra adapter-only |
| Prompt hack, logit mask, constrained decoding | nao submit-safe |
| OpenRouter/provider/legal URLs | ruido; nao afeta ACC |
| `lm_head` treinavel no smoke principal | risco de bit flip/truncation; somente ablation |
| Treino promocional com `ANSWER_SPAN_LOSS_WEIGHT=1.0` | V499 mostrou `0` exemplos com answer-span weighting e loss final sem melhora |
| Treinar diretamente nos IDs weak/full que regrediram ou ganharam | viola regra de usar weak/full apenas como gate |
| Promover ganho visto apenas por expected-aware extractor | pode ser melhoria de parser, nao de adapter |

## Proxima Acao Imediata

1. Manter V290 checkpoint-6 como unico adapter submit-safe ate haver weak/full
   gate melhor com metrica label-free.
2. V499 e V501 estao bloqueados:
   - V499: final eval loss `2.8125 -> 2.8162`, answer-span inativo;
   - V501: answer-span ativo, MoE trainavel e `lm_head` congelado, mas final
     eval loss `1.9919 -> 1.9923`.
3. Nao rodar weak/full/package/submit em V499/V501. A decisao FinOps correta e
   nao gastar com ACC caro quando o kill-switch local ja bloqueou o candidato.
4. Revalidar inventario de candidatos anteriores usando somente
   `submit_safe_label_free_prediction`. Qualquer ganho que exista apenas em
   `label_aware_debug_prediction` e descartado.
5. Atualizar/usar os gates corrigidos antes de qualquer novo job:
   - `kg1_static_safety_gate.py`;
   - `kg1_pre_paid_job_integration_gate.py --dataset-schema sft|preference`;
   - `hf_job_preflight_gate.py` com flags anti-leakage obrigatorias;
   - weak eval promocional official-like e bloqueante por default.
6. V515 local e HF CPU ja recuperaram `+8` bit traces residuais e passaram
   V286/V513. O gate objetivo mostrou que pesos iguais voltam a subponderar
   bit (`18.99%`) e superponderar equation (`81.01%`). A unica ponderacao V515
   aceita ate aqui e fonte bit `1.5x`, que produz `bit=26.01%` e
   `equation=73.99%`.
7. V516 corrigiu o gate de equation para baseline label-free: o sinal real
   atual e `+4` equation (`55 -> 59`) nos mesmos IDs ja usados em V475/V510/V515.
   Isso nao autoriza outro treino focado em equation sozinho; so autoriza um
   smoke se houver mudanca de mecanismo, como V515 bit traceability + pesos bit
   corrigidos.
8. Se surgir novo candidato, weak eval promocional deve usar thinking ligado,
   `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64` e falhar se nao
   passar primeiro pelo gate minimo anti-backfire (`total>=193`,
   `equation>=57`, `bit>=136`, `trunc=0`). Para package/submit, a meta segue
   mais alta: aproximar `total>=196`, `equation>=60`, `bit>=136`, `trunc=0`.
9. V517/V518 encerram a linha V515 GPU atual: H200, `MAX_STEPS=2`,
   dataset V515 HF CPU, bit source weights `1.5x`, `lm_head` congelado,
   MoE `gate_up/down` treinaveis, `KG1_CRISIS_MODE_BACKFIRE_GUARD=1`.
   O weak eval V518 do checkpoint-2 retornou `191/315`, `equation=56`,
   `bit=135`, `trunc=0`, portanto regrediu bit contra o baseline label-free
   `191/315`, `equation=55`, `bit=136`. Esta linha fica em crisis mode:
   nenhum full/package/submit, nenhum GPU repetido, e o proximo passo volta
   para CPU/debug ate aparecer sinal novo sem regressao.

## Atualizacao V500 - Auditoria De Parametros V499

Artefatos:

- `artifacts/v500_v499_parameterization_audit/KG1_V500_V499_PARAMETERIZATION_AUDIT.md`;
- `artifacts/v500_v499_parameterization_audit/v500_v499_parameterization_audit_manifest.json`.

Conclusao: as pecas de execucao do V499 estao sincronizadas, mas os valores nao
estao corretos para esperar ganho de ACC. O run comprovou que o caminho tecnico
funciona, nao que o adapter aprendeu os quatro ganhos de equation. O novo gate
obrigatorio e: treino para ACC precisa ativar answer-span weighting e provar que
esse weighting foi aplicado no tokenizador antes de gastar H200.

## Atualizacao V502 - Double Check De Parametrizacao

Artefatos:

- `artifacts/v502_solution_parameterization_double_check/KG1_V502_SOLUTION_PARAMETERIZATION_DOUBLE_CHECK.md`;
- `artifacts/v502_solution_parameterization_double_check/v502_solution_parameterization_double_check_manifest.json`;
- `artifacts/v501_hf_nemo_h200_v498_answer_span_weighted_launch/launch_v501_hf_nemo_h200_v498_answer_span_weighted.py`.

Resultado do double check:

- dataset V498: `1712` train, `428` val, zero duplicados, zero falha de
  extracao/verificacao do `Final answer`, zero caracteres suspeitos nos
  answers;
- tokenizacao: `token_max=331`, `MAX_LENGTH=1024` seguro, zero truncation,
  offset masks completos;
- ACC metric: raw CoT nao e aceito diretamente; avaliacao precisa extrair
  resposta final antes de `verify_answer`, como esperado;
- V499: tecnicamente correto, mas bloqueado por objetivo local
  (`2.8125 -> 2.8162`, delta `+0.0037`, answer-span inativo);
- V501: corrigido para `ANSWER_SPAN_LOSS_WEIGHT=4.0`,
  `ANSWER_SPAN_MIN_WEIGHTED_TOKENS=1000`, `MAX_STEPS=4`, V290 checkpoint-6,
  MoE `gate_up/down` treinaveis, `lm_head` congelado, bit replay efetivo
  `39.02%`;
- `kg1_static_safety_gate.py` atualizado para permitir peso de answer-span
  apenas em rota explicitamente `answer_span_weighted` e bloquear qualquer caso
  sem min-token gate.

Decisao:

- V499 nao deve receber weak eval pago.
- V501 foi lancado como smoke H200 curto depois do debug de parametrizacao e
  foi bloqueado pelo kill-switch: final eval `1.9919 -> 1.9923`.
- Nao rodar weak/full/package/submit em V499/V501; ambos sao artefatos
  forenses, nao candidatos.

## Atualizacao V503/V504 - Causa Raiz Do Plateau

Artefatos:

- `artifacts/v503_live_hf_job_parameterization/KG1_V503_LIVE_HF_JOB_PARAMETERIZATION.md`;
- `artifacts/v504_crisis_root_cause_audit/KG1_V504_CRISIS_ROOT_CAUSE_AUDIT.md`;
- `artifacts/v504_crisis_root_cause_audit/v504_crisis_root_cause_audit_manifest.json`.

Conclusao: as pecas de treino V501 estavam tecnicamente presentes, mas a rota
de promocao tinha bugs silenciosos suficientes para explicar falsas esperancas
de ganho:

| Problema | Correcao |
|---|---|
| `prediction` podia ser extraida com `extract_final_answer_for_expected`, usando label esperado | `prediction` agora e label-free; expected-aware fica somente em `label_aware_debug_prediction` |
| expected-aware aceitava prefixo inseguro como `\boxed{30 wrong}` para expected `30` | helper agora aceita apenas delimitador real `}` |
| weak eval V245 tinha defaults diagnosticos (`max_tokens=96`, thinking off) e threshold `equation>=57` | defaults agora sao official-like e promocao exige `total>=196`, `equation>=60`, `bit>=136`, `trunc=0` |
| weak eval podia terminar com exit 0 sem candidato promovido se env nao estivesse setado | promocao e bloqueante por padrao; sweeps baratos exigem `KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=1` |
| full eval escolhia maior `correct` antes de aplicar truncation gate | agora escolhe primeiro entre candidatos que passam correct/truncation |
| pre-paid gate era preference-only e nao cobria V498/V501 SFT | adicionado `--dataset-schema sft` e save/eval steps configuraveis |
| flags anti-leakage ausentes eram apenas contadas | agora bloqueiam preflight promocional |
| launchers pagos podiam ser criados sem declarar que os guards F2/backfire/FinOps estavam ativos | `kg1_static_safety_gate.py` e `kg1_pre_paid_job_integration_gate.py` agora exigem `KG1_CRISIS_MODE_BACKFIRE_GUARD=1` |
| CSV de solver/postprocessor podia parecer candidato promocional | V505 separa `raw_output` adapter-only de reference-only; CSV sem `raw_output` nao promove |

O que ficou provado como nao raiz: hashes V498, row counts, offset masks,
truncation do dataset, MoE `gate_up/down` treinaveis, `lm_head` congelado,
memoria H200 e custo.

Decisao operacional: V499 e V501 entram na lista de adapters bloqueados. A
revalidacao V505 ja foi executada e nao encontrou candidato adapter-only acima
do gate. O melhor CSV `raw_output` de adapter ficou em `191/315`, enquanto
solvers/postprocessors reference-only chegam a `222/315` mas nao sao
submittable. Sem novo candidato adapter-only `>192/315`, nao ha novo GPU; volta
para CPU teacher/verifier discovery com obrigacao de converter sinal em
comportamento LoRA ou pacote permitido.

## Atualizacao V506 - Alvo Real De Transferencia

Artefatos:

- `artifacts/v506_reference_signal_gap/v506_reference_signal_gap_manifest.json`;
- `artifacts/v506_reference_signal_gap/v506_reference_signal_gap_rows.csv`;
- `artifacts/v506_reference_signal_gap/v506_reference_gain_targets.csv`.

V506 comparou o melhor adapter `raw_output` label-free contra o melhor sinal
reference-only. Resultado:

| Status | Total | bit | equation |
|---|---:|---:|---:|
| both_correct | 191 | 136 | 55 |
| reference_gain_target | 31 | 23 | 8 |
| both_wrong | 93 | 1 | 92 |
| reference_loss_risk | 0 | 0 | 0 |

Quebra dos `31` targets:

- `13` bit por `bit_exact_global_ternary_unique_prediction`;
- `4` bit por `bit_fullbyte_ternary_op_CHO`;
- `4` bit por `bit_fullbyte_ternary_op_MAJ3`;
- `1` bit por `bit_exact_global_binary_XOR`;
- `1` bit sem regra preenchida;
- `8` equation sem `source_rule` preenchido no CSV reference-only, exigindo
  inspeção/rotulagem antes de qualquer dataset sintetico.

Interpretacao: existe sinal tecnico forte para subir `191 -> 222` no weak, mas
ele ainda esta fora do comportamento adapter-only. A tarefa agora nao e "mais
treino" generico; e converter esses `31` targets em saida do adapter sem
quebrar os `191` atuais. Qualquer job pago precisa declarar
`KG1_CRISIS_MODE_BACKFIRE_GUARD=1`, passar pre-paid gate, e so pode ser
promovido apos weak eval `raw_output` label-free.
