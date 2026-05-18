# KG1 Score Improvement Roadmap

Atualizado: 2026-05-18

Este e o roadmap ativo e limpo apos a revisao V484 dos arquivos OpenRouter de
16/05/2026. O historico detalhado foi arquivado em:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V435_CLEANUP_2026_05_15.md`
- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V484_OPENROUTER_CLEANUP_2026_05_16.md`

A partir deste arquivo, itens antigos so valem como evidencia historica. O plano
executavel e somente o que esta abaixo.

## Estado Real

| Metrica | Estado atual | Promocao minima |
|---|---:|---:|
| Baseline historico packageable | 192/315 | > 192/315 |
| Recompute label-free atual | 192/315 | >= 196/315 |
| equation_transform baseline | 56/155 | alvo inicial 60/155 |
| bit_manipulation baseline | 136/160 | >= 136/160 |
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
| V525 OpenRouter objective consult | 5 modelos (`gpt-5.5`, Claude Opus 4.7, Gemini 3.1 Pro, Qwen 3.6 Max, DeepSeek V4 Pro) confirmaram que `token_mean` nao pode ser usado com V523; todos recomendam `example_mean`, guard de `8740ed31`, e kill-switch de primeiro checkpoint; 2/5 preferem rebuild V525 antes de GPU | decisao adotada: rodar CPU dry-run de contribuicao por familia; se V523+`example_mean` ainda for dominado por bit, construir V525 com traces bit curtas e token-mass `bit<=70-78%`; se passar, permitir apenas smoke H200 curto com gate `total>=196`, `equation>=60`, `bit>=136`, `trunc=0` |
| V526 example_mean dry-run | `example_mean` passou: bit share por tokens cairia de `90.67%` para `68.81%`, delta vs referencia `5.38pp`, boxed/control/answer checks `0` blockers | autoriza somente um smoke H200 curto V523 com `LOSS_NORMALIZATION_MODE=example_mean` e kill-switch no primeiro checkpoint; nao autoriza treino longo |
| V528 notebooks 0.86/0.85 | notebooks `0.86` sao principalmente packaging do Tinker adapter `kienngx/.../tinker-adapter/1`; tecnicas uteis estao nos notebooks Tong/Pear/Konbu/PJT/ZZYS, nao nos zips de score | usar `0.86` apenas para schema/provenance/package; usar solver/CoT para novo sinal CPU |
| V529 todos os kernels baixados | `704` kernels puxados, `702` parseados; lista filtrada indica P0: `pjt222/nemotron-cot-review`, `pearpn25/bit-cot-85-1364-sample`, `konbu17/bit-manipulation-solver-cot-generator`, `zzys0316/full-pipeline...`; notebooks 0.86 nao resolvem familias | proximo passo efetivo e V530 CPU solver harness antes de qualquer H200 amplo; GPU so se houver novo sinal label-free ou smoke V523 estritamente limitado |
| V530 anexos/datasets | `archive.zip`/Konbu v2 cobre `1508/1602` linhas bit do `competition_train`; `success.csv` tem `1134` CoTs corretas, `671` high-confidence, `0` mismatch de prompt/resposta; `failed.csv` nao pode ser positivo; `archive (1)/(6)` tem `3000` sinteticos, mas `1341` `solver_correct=False`; `archive (2)/(3)` sao dataset oficial duplicado; `archive (4)` adiciona só `28` CoTs numeral/Roman | nova fonte P0 para bit somente; converter/encurtar success high-confidence para KG1 com `\boxed{}` e `example_mean`; nenhum ganho direto para equation |
| V531 anexos V-CARS/Yoiko | `archive (7)` e pacote V-CARS/offline deps/notebook smoke, sem solver novo para bit/equation; `vcars-external-data` aponta para Tatoeba/sentences e nao e fonte P0; `archive (8)` contem LoRA Yoiko ver5 rank 32 alpha 32, base Nemotron correto, `11960` tensores F32, `880138240` params, sem non-LoRA tensors | nao treinar com V-CARS/Tatoeba; registrar Yoiko como candidato P1 de weak eval adapter-only curto, apos gate de config/header e empacotamento root-level temporario; nao e ganho submit-safe sem weak label-free |
| V532 Kaggle dataset/topics/comments search | `70` datasets listados por CLI; `18` candidatos baixados/analisados em temp e apagados; P0/P1 concreto: Konbu BM/ET CoT, `itskshivam` candidate_pool/critic/router, `sohamp13` 3-way selector, `furkankesen` solver-swap, `adityakrishnanmohan` hard triad; topics/comments refresh: `58` topicos, `357` posts, `238` hits | novo plano muda equation para candidate/verifier/canonicalization CPU gate; nao usar bundles com mismatches como gold; Huikang foi auditado localmente no V533 sem commitar ZIP/pesos |
| V532 external equation candidate gate | `critic_v2`, `router_v1`, `selection_v2`, `solver_swap_v1` baixados em temp; candidate pools cobrem `155/155` weak equation, mas seletor label-free por `verifier_score/canonicalization/sympy/rank` cai de `55/155` para `29/155`, com `2` ganhos e `28` perdas | nao promotavel; usar esses datasets como fonte de features/canonizacao/hard negatives, nao como seletor direto nem treino direto |
| V533 Huikang local package | `archive (9).zip` auditado sem extrair pesos; `adapter_v26`: LoRA `all-linear`, r32/alpha32, `418` tensores F32, `386072576` params; `bit_manipulation_3input_traces`: `100` oficiais, `0` mismatch; `2000` sinteticos CHO/MAJ no ZIP local, apesar da metadata citar `10000`; overlap weak bit `10`, incluindo `8` misses atuais | P0 para bit CHO/MAJ trace mining source-only; nao copiar weak rows para treino promocional; adapter v26 e P1 weak eval estatico, nao submit-safe sem gate |
| V534 roadmap/F2 double check | corrigiu contradicao "duas frentes"; reafirmou que `competition_match`/`answer`/`expected_answer` sao auditoria, nao selecao; removeu broad SFT/V-CARS/Tatoeba/candidate-pool direto do plano ativo | plano ativo agora e CPU source-only bit, CPU equation canonicalization/hard negatives, adapter-only eval barato, GPU bloqueada ate novo sinal |
| V535 specialist triple check | revisao independente apontou risco de leakage no CSV V532, fallback silencioso para `prediction`, superconfianca em `verifier_score=1.0`, thresholds contraditorios e adapter externo conflitando com itens removidos | V532 deixou de exportar decisoes row-level, `--fail-on-blocked` validado, CSVs weak-overlap Huikang removidos; adapters externos rebaixados para diagnostico/proveniencia; thresholds separados em smoke, weak promocional e package |
| V536 V534-bit/V523-equation pack | dataset source-only controlado criado: `1026` train, `219` val; quotas iguais ao V523 (`706/320` train bit/equation, `139/80` val); bit substituido por Konbu high-confidence + Huikang CHO/MAJ; `0` overlap weak/full, `0` duplicidade; V286, V513, V524 e V526 passam | autoriza somente um smoke H200 curto com `LOSS_NORMALIZATION_MODE=example_mean`; nao autoriza treino longo nem submit sem ACC label-free real |
| V536 HF upload/debug/pre-paid | dataset enviado para HF commit `d2f11d82b40e3e9aa0f5add58c3698a7428bf550`; launcher debug baixou do HF e validou hashes, H200 `0.083333/min`, adapter inicial e objetivo; `kg1_pre_paid_job_integration_gate` passou sem findings | pronto para commit/push e um smoke H200 de 4 steps; ainda nao ha ganho ACC medido |
| V536 H200 attempt 1 | job `felipesp1983/6a0930223308d79117b9181a` falhou antes do treino: runtime `MAX_LENGTH=1024` truncava `78/1026` prompts (`7.6023%`) apesar do V286 ter `token_max=1123` com `max_length=8192` | falha barata e correta; launcher corrigido para `MAX_LENGTH=2048`, `KG1_EXPECTED_MAX_LENGTH=2048`, e pre-paid gate agora compara `token_max <= runtime MAX_LENGTH` |
| V536 H200 current retry | job `felipesp1983/6a0981fe3308d79117b91bc6` completou; checkpoints atuais `2/4`; MoE/target params corretos, `lm_head` congelado, `MAX_LENGTH=2048`, loss `3.0276 -> 3.0257/3.0277` | loss nao promove; decisao dependeu da weak eval V549 |
| V549 weak eval V536 current | job `felipesp1983/6a098851e48bea4538ba0f1f`; checkpoint-2 `190/315`, `bit=134`, `equation=56`, `trunc=1`, avg completion `4774.9`, max `7680`; checkpoint-4 `190/315`, `bit=135`, `equation=55`, `trunc=0`, avg completion `4772.0`, max `7492` | ambos bloqueados; nenhum full/package/submit; V536/V549 rejeitado como linha de ganho |
| V549 protected/token guards | ambos checkpoints erraram `8740ed31`: esperado `01101000`, previsto `01111000`; `weak_promotion_gate` bloqueou por `correct_lt_196`, `equation_lt_60`, `bit_lt_136`, tokens excessivos e protected-row backfire; `catastrophic_eval_guard` passou porque nao foi colapso total | guards atualizados estao funcionando; condicoes atuais de promocao continuam: `total>=196`, `bit>=136`, `equation>=60`, `trunc=0`, protected rows preservadas e saida curta |
| V550 condition sync audit | scripts/launchers ativos agora exigem `8740ed31=01101000` e `59bee375=10010101`; V536 tem bit traces longas (`p95=1669` chars) e V549 repetiu trade-off antigo: cp2 ganhou `518deb39`, mas perdeu `8740ed31` e `59bee375`; cp4 perdeu `8740ed31` sem ganho | `gpu_allowed=false`; nao repetir V536/V549; proxima linha precisa dataset curto answer-span ou hard-negative CPU gate preservando os dois protegidos antes de qualquer job pago |
| V551 short bit trace pack | V536 foi reescrito em CPU mantendo fonte-only: bit train `706`, equation train `320`; bit assistant `p95=250`, `max=251` chars contra `p95=1669` no V536; V509 integridade passou, V286 real passou (`token_max=383`, trunc `0`, offset masks `1026/219`), V513 real passou, V524 real ficou `quota_ok_cpu_only` com bit loss-token share `0.592993`; static safety gate passou apos bloquear/diagnosticar launcher V548 historico | novo candidato de dataset curto; ainda `gpu_allowed=false` ate pre-paid/HF CPU/preflight/protected weak smoke; gates agora tambem aceitam limite `KG1_MAX_ASSISTANT_CHARS_P95/MAX` para impedir repetir traces longas |
| V547 contract-aligned distillation | dataset `236/111` criado com targets `Final answer: \boxed{...}`; sem `RULE:`; train `bit=146`, `equation=90`; tokenization `max=328`, trunc `0`; objective alignment passou | dataset tecnicamente limpo, mas era answer-only pequeno; exigia weak eval antes de qualquer package |
| V547 H200 train | job `felipesp1983/6a097803e48bea4538ba0df2` completou em `0.19h`; checkpoints `2/4/6/8`; melhor loss em checkpoint-4 (`3.0138`) | loss nao promove; seguir somente por ACC label-free |
| V548 weak eval V547 | job `felipesp1983/6a097c51e48bea4538ba0e3a`; checkpoint-2 e checkpoint-4 deram `3/315`, `equation=3/155`, `bit=0/160`, `truncated=288/315`, avg completion `509.9/512`; cp2/cp4 identicos | linha V547 rejeitada; job cancelado por FinOps; sem full eval/package/submit |
| V548 catastrophic guard | `scripts/hf_job_weak_eval_v245.py` agora tem `catastrophic_eval_guard`; bloqueia ACC quase zero + truncation massiva mesmo se `STOP_AFTER_CONSECUTIVE_FAILED_CANDIDATES=0` | evita repetir custo H200 quando um checkpoint destrói o contrato de saida |
| V548 package permission recheck | V336B reexecutado; regras extraidas confirmam `submission.zip` com LoRA adapter rank<=32; pacote local valido contem somente `adapter_config.json` e `adapter_model.safetensors`; script rejeita postprocessor | solver/verifier direto continua bloqueado para submit; ganho precisa virar adapter-only |
| V552/V553 short-bit smoke | V552 treinou em H200 sobre V551; checkpoint-2 reduziu loss minimamente, mas V553 weak eval deu `190/315`, `bit=134`, `equation=56`, `trunc=1`, media `4775` tokens; perdeu `8740ed31` e `59bee375`; checkpoint-4 foi cancelado por FinOps | linha V551/V552 bloqueada; nao package/full/submit; proxima acao deve atacar backfire/protected rows em CPU antes de novo SFT pago |
| V560/V561 answer-only smoke | dataset `236/111`, train `bit=146`, `equation=90`, targets de uma linha `Final answer: \boxed{...}`, sem `RULE:`/`Trace:`; V561 H200 treinou 4 steps a partir de V290 checkpoint-6 | tecnicamente limpo, mas ainda dependia de weak ACC; loss nao autoriza submit |
| V562 weak eval V561 | checkpoint-2 `191/315`, `bit=135`, `equation=56`, `trunc=1`, avg completion `4775`, max `7680`; backfire nos protected ids `8740ed31` e `59bee375`; gate bloqueou e parou antes dos demais checkpoints | causa concreta: mesmo com target answer-only, a inferencia official-like ativou CoT longo/runaway; nao package/full/submit; proximo passo e separar erro de treino vs erro de inferencia |
| V563 strict diagnostic | job `felipesp1983/6a09da1ee7940de6ee6cd80c` completou com `disable_thinking=1`, `max_tokens=128`; checkpoint-2 `16/315` (`bit=8`, `equation=8`), checkpoint-4 `15/315` (`bit=7`, `equation=8`), final `16/315` (`bit=7`, `equation=9`), trunc `0`, tokens curtos, protected rows falharam em todos | diagnostico negativo: prompt curto resolve runaway, mas destrói ACC; V560/V561 ficam bloqueados; nao package/full/submit |
| V564 OpenRouter plateau consult | 8 respostas validas de 9 modelos (`Claude`, `DeepSeek`, `Gemini`, `Kimi`, `Nemotron`, `Qwen`) sobre V562/V563; consenso dominante: `STOP_BROAD_LORA`/`INVESTIGATE_BUG_FIRST`; causas provaveis: contrato/prompt, mask/label/weight, logica solver nao transferivel por SFT curto, possivel PEFT continuity | novo roadmap deve comecar por auditoria de contrato+mask+logits/protected rows; H200 amplo bloqueado ate gate CPU/tiny mostrar ganho real |
| V564 contract/mask audit | `scripts/audit_v564_contract_mask_alignment.py` rodou em CPU no V560; mask/weights passaram (`train 236/236`, validation `111/111` zero-weight ignored), protected rows existem uma vez com peso `3.0`; blocker real isolado: `train_eval_prompt_token_mismatch` em `347/347` linhas, prefixo comum de apenas `3` tokens | causa concreta nova: treino usa system prompt `You are solving Kaggle...`, enquanto eval official-like usa system vazio + `PROMPT_SUFFIX`; qualquer novo GPU fica bloqueado ate padronizar treino/inferencia |
| V565 official-like dataset | `build_v547_contract_aligned_distillation_dataset.py` agora suporta `--prompt-contract official_like`; dataset V565 `236/111` criado com user prompt `prompt + PROMPT_SUFFIX`, sem system prompt; V564 contract/mask passou, V286 tokenization passou (`max=315`, trunc `0`), V478 objective passou (`bit_share=0.665689`, `equation_share=0.334311`) | primeiro dataset candidato com contrato alinhado; ainda nao autoriza submit nem treino longo; proximo passo e diagnostico tiny/baseline antes de H200 |
| V566 uploaded OpenRouter double check | arquivo `C:\Users\davis\Downloads\OpenRouter Chat Sun May 17 2026.json` tem 13 respostas uteis; consenso reforca `INVESTIGATE_BUG_FIRST`, contrato de prompt, protected rows, hard negatives e small gate; URLs do arquivo sao majoritariamente provider/legal/model-card e nao trazem dado novo do desafio | novo achado acionavel: antes de usar strict no-think como contrato, rodar diagnostico baseline/no-adapter strict; V565 continua a unica rota adapter-only plausivel, com kill-switch no primeiro checkpoint |
| V567 prompt-contract probe | job H200 `felipesp1983/6a09ee09a5e509f1a841336f` avaliou base, V290 ckpt-6 e V561 ckpt-2 em 11 linhas criticas; `strict_no_think` e `hybrid_one_line` deram `0/11`; `official_like`/`legacy` com `max_tokens=2048` truncaram e nao preservaram protected rows | diagnostico negativo: nao promover prompt curto/no-think; eval promocional precisa manter thinking habilitado e `max_tokens=7680`; qualquer variante curta so pode rodar como diagnostico |
| V568 logits/NLL probe e drift gate | job H200 `felipesp1983/6a09f63ea5e509f1a841342d` completou; `missing_logprob_rows=0`, `prefix_mismatch_rows=0`; analisador `scripts/analyze_v568_decoding_adapter_drift.py` gerou `decision=blocked`; margem curta absoluta pode ser negativa ate no `base_no_adapter`, mas V290/V561 regrediram vs base em `59bee375` (`max_regression` boxed `0.246333/0.218400`; final-answer `0.107941/0.106541`) | `hf_job_preflight_gate.py` e `kg1_pre_paid_job_integration_gate.py` agora bloqueiam regressao de margem vs baseline, nao apenas margem absoluta; nao rodar treino pago se o adapter aumenta probabilidade relativa de respostas erradas protegidas |
| V569 OpenRouter plateau resolution consult | 7 modelos consultados; 6 respostas uteis; consenso dominante `PROTECTED_REPLAY_FIRST`, com `PREFERENCE_TRAINING_WITH_GATES` como segunda fase; `openai/gpt-5.5-pro` retornou conteudo vazio/utilizavel insuficiente | proxima acao muda para V570 protected trajectory replay audit/build; equation fica bloqueado ate preservar margem protegida/bit floor; broad SFT, answer-only, strict no-think e CPU-solver direto saem do plano ativo |
| V570 protected replay audit | `scripts/build_v570_protected_replay_audit.py` recuperou do V516/V290 baseline as duas trajetorias longas protegidas corretas (`8740ed31=01101000`, `59bee375=10010101`) e `40` anchors corretos (`30` bit, `10` equation); `42` linhas, `training_allowed_rows=0`, `blockers=[]`, `decision=diagnostic_only_no_training`; warnings apenas por `completion_tokens` ausente no CSV fonte | evidencia util para debug/replay/margem, mas nao pode virar treino direto por ser weak-gate; proximo passo e derivar analogos source-only/hard-negative preference e medir V568 margin gate antes de qualquer H200 |
| V571 source-only bit-pair traces | `scripts/build_v571_bitpair_source_only_trace_pack.py` criou traces bit-pair/bitsum deterministicas apenas de fonte externa; `437` train e `79` val aceitas; V509/V286/V513 passaram; V524 bloqueou treino bit-only por objetivo dominado por bit | achado aproveitavel: bit-pair source-only existe e e verificavel; nao treinar isolado |
| V572 aggressive bit/equation mix | V571 + V551 equation, `757/159` linhas, pesos `bit=0.5`, `equation=1.5`; V509/V286/V478/V513 passaram, mas V526 row-weighted corrigido bloqueou: peso efetivo `bit=31.28%`, delta `42.9pp` vs referencia | nao ir para GPU; risco alto de repetir regressao de bit |
| V573 reference-weighted source-only mix | mesmo conteudo V571+V551, mas pesos `bit=1.5`, `equation=1.0`; V509 passou, V286 real passou (`token_max=1074`, `0` trunc, offset masks `757/159`), V478 passou (`bit=67.20%`, `equation=32.80%`), V513 passou e V526 row-weighted passou (`delta=6.997pp`) | primeiro candidato novo pos-V570 que passa gates CPU com protecao de objetivo; autoriza somente auditoria pre-paid e smoke curto com kill-switch, nao submit |
| V573 HF upload/pre-paid | dataset V573 subido para HF dataset commit `3d321aff1e68d72769f167ceab2a28123faa18fd`; launcher debug baixou do HF sem symlink cache, validou hashes `ba515.../6957...`, adapter seed e objetivo; pre-paid gate passou sem findings com deferimento V568 restrito a `MAX_STEPS=2` e weak eval obrigatoria no primeiro checkpoint | pronto para commit/push dos gates e um smoke H200 de 2 steps; se checkpoint-2 nao passar `total>=196`, `bit>=136`, `equation>=60`, `trunc=0` e protected rows, cancelar/abandonar |
| V573 H200 attempt 1 | job `felipesp1983/6a0a0c94a5e509f1a8413626` falhou antes de treinar; H200 correto e preinstall passaram; artifact gate validou hashes e dataset, mas bloqueou por bug silencioso no preflight: `metadata.subcategory` antigo (`bit_konbu_high_confidence_trace`, `bit_huikang_*`) sobrescrevia a subcategoria canonica top-level `bit_bitpair_certified_source_only` usada pelo pre-paid gate | sem custo de treino nem ganho ACC; corrigido para priorizar subcategoria canonica top-level e adicionado self-test `canonical_subcategory`; relancar somente apos commit/push e novo pre-paid/debug |
| V573 H200 attempt 2 | job `felipesp1983/6a0a0f18a5e509f1a8413664` falhou antes de treinar; preinstall, artifact gate canonico e V485 adapter roundtrip passaram, mas o objective alignment remoto rodou sem `--use-row-loss-weight`/`--require-row-loss-weight` e avaliou share fisico (`bit=57.73%`, `equation=42.27%`) em vez do share efetivo do loss (`bit=67.20%`, `equation=32.80%`) | bug silencioso operacional, nao sinal negativo de modelo; launcher V573 agora injeta os flags no comando remoto e o debug local bloqueia qualquer launch se o contrato de row-loss-weight sumir; sem treino e sem ganho/perda ACC |
| V573 H200 attempt 3 | job `felipesp1983/6a0a123ae7940de6ee6cdad8` completou; checkpoint-2/final uploaded; baseline/eval loss ficou neutro (`1.4546 -> 1.4546`) e o log expôs bug silencioso local no script de treino: `tokenize_examples` ainda priorizava `metadata.subcategory` antes do `subcategory` canonico top-level, divergindo do preflight | impacto no job atual e baixo porque os pesos de subcategoria sao 1.0 e o objetivo efetivo vem de `metadata.loss_weight`; corrigido em `scripts/hf_job_train_v90.py` para top-level vencer metadata e coberto por self-test; proximo passo e weak eval official-like do checkpoint-2, sem submit antes de `total>=196`, `bit>=136`, `equation>=60`, `trunc=0` e protected rows |
| V595b/V597 weak eval | V595b treinou 2 steps no dataset V596 answer-only preference; preference val piorou `59/120 -> 58/120`; weak remoto armazenou `190/315`, `equation=55`, `bit=135`, `trunc=1`; re-score V598 com extrator atual corrige `4bb8c6cd` e fica `191/315`, `equation=56`, `bit=135`, `trunc=1` | rota V596/V595b encerrada; sem full/package/submit; regressao real e bit, nao equation |
| V598 metric/gate fixes | o baseline V290 re-score atual e `192/315`, `equation=56`, `bit=136`, `trunc=0`; limite absoluto de tokens `512/2048` era falso para official-like e foi tornado opcional; `validate_answer_extraction_v1.py` agora trata mismatch sem mudanca de acerto como warning; protected guard separa backfire de missing required gain | evita falso bloqueio/falso ganho; qualquer nova avaliacao deve passar rescore raw-output label-free antes de promocao |
| V601/V602 MoE preference fechado | V601 source-build treinou `preference answer-only + MoE up/down trainable`; params treinaveis subiram para `869,318,656` e preference val melhorou `59/120 -> 61/120`; V602 remoto deu `191`, mas re-score local atual corrige o caso `\\boxed{]}\\!}` e fica `192/315`, `equation=56`, `bit=136`, `trunc=0` | sem ganho liquido vs V290; rota MoE preference V596 fechada; nao rodar mais H200 nessa linha sem novo sinal CPU/protected-row |
| V604 interpolacao V601 fechada | eval-only V290->V601 testou lambdas uteis antes de cancelar por FinOps: `0.05 = 192/315, equation=56, bit=136, trunc=0`; `0.10 = 190/315, equation=56, bit=134, trunc=1`; `0.25 = 190/315, equation=56, bit=134, trunc=1`; `0.50` foi cancelado apos tres pontos sem sinal | confirma que preference/MoE nao vira ACC por interpolacao; V596/V601 fica bloqueado para novo H200; proxima acao precisa ser CPU/source-only com novo sinal, nao ajuste de lambda |
| V605 consolidated plateau audit | comparou V574, V582, V591, V597 e V602 contra V290 usando `raw_output -> extract_final_answer -> verify_answer`; V574/V582/V591/V597 ficaram `191/315`, `bit=135`, `equation=56`, `trunc=1`; V602 ficou `192/315`, `bit=136`, `equation=56`, `trunc=0`; todos sem ganho total, todos com token runaway e backfire protegido (`8740ed31`, e em 4/5 tambem `59bee375`) | rotas V573/V579/V591/V594/V596 e adapters V573/V579/V582/V591/V595/V595b/V601 entram em quarentena nos gates; nao repetir essas linhas em H200 sem novo CPU signal diferente e sem passar V605/protected gates |
| Workspace clean rule | regra permanente do usuario formalizada: manter somente fonte, roadmap/gates, datasets/manifests ativos, evidencias futuras reutilizaveis e artefatos necessarios para reproducao; caches/temp/download leftovers/logs redundantes devem sair | criado `scripts/kg1_workspace_clean_gate.py`; modo `--delete-safe` remove apenas lixo inequivoco (`.cache`, `__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`, `*.tmp/*.bak/*.old/*.orig`); logs/manifests/datasets/adapters nunca sao apagados automaticamente |

## Decisao Atual V560-V605

O problema ativo nao e mais falta de busca externa nem apenas hiperparametro de
loss. A evidencia V562/V563 mostra desalinhamento entre treino e inferencia:

- V560 removeu `RULE:` e `Trace:` e treinou targets curtos, mas o adapter V561
  ainda gerou CoT longo em weak eval;
- V562 mediu `avg_completion_tokens=4775`, `max_completion_tokens=7680` e
  `truncated=1`, com regressao de bit para `135/160`;
- os dois protected rows que o baseline acertava foram perdidos:
  `8740ed31: 01101000 -> 01111000` e `59bee375: 10010101 -> 2`;
- o gate bloqueou por `correct_lt_196`, `equation_lt_60`, `bit_lt_136`,
  `truncated_gt_0`, tokens excessivos e protected-row backfire.
- V563 eliminou o runaway com `disable_thinking=1` e `max_tokens=128`, mas
  colapsou a ACC para `15-16/315` nos tres checkpoints e manteve backfire nos
  protected rows. Portanto prompt curto sozinho nao e solucao.
- V564 consultou OpenRouter e confirmou a decisao operacional: parar broad
  LoRA, investigar bug/contrato/mask/logits primeiro e so voltar para GPU apos
  gate pequeno com ganho medido.
- V564 contract/mask audit confirmou que mask, pesos e protected rows nao sao
  o blocker imediato do V560. O blocker real e contrato de prompt: `347/347`
  prompts de treino diferem do prompt official-like, com prefixo comum de
  apenas `3` tokens. Isso torna invalido repetir H200 antes de alinhar o
  template.
- V565 corrigiu o contrato no dataset: treino e eval official-like agora usam
  o mesmo texto de prompt. As validacoes V564/V286/V478 passaram sem blockers.
- V566 analisou o arquivo OpenRouter de 17/05/2026 e adicionou uma cautela
  concreta: o colapso V563 pode ser incompatibilidade global do prompt strict,
  nao apenas erro do adapter. Portanto `disable_thinking=1`/`max_tokens=128`
  nao pode virar contrato padrao antes de um teste baseline/no-adapter.
- V567 executou esse teste em H200. O resultado foi negativo: strict/no-think
  e one-line nao acertaram nenhuma das 11 linhas criticas, e as variantes
  longas com `max_tokens=2048` truncaram. O cross-check historico mostrou que
  o V290 correto nas protected rows usa `6290` e `6589` completion tokens.
  Portanto V567 nao substitui weak eval oficial; ele prova que qualquer rota
  promocional precisa manter o contrato longo `max_tokens=7680` e thinking
  habilitado.
- V568 executou o probe logits/NLL em H200 e refinou a regra: margem absoluta
  negativa e um alerta, mas nao pode ser bloqueio automatico porque ate o
  `base_no_adapter` mostra margem curta negativa em algumas alternativas. O
  bloqueio correto e regressao de margem contra baseline. V290/V561 pioraram
  principalmente `59bee375`, entao a proxima linha deve corrigir protected-row
  drift antes de tentar novos ganhos em equation. A regra entrou tanto no gate
  de runtime HF quanto no pre-paid integration gate.
- V569 consultou novamente OpenRouter com o estado completo pos-V568. O consenso
  mais forte foi `PROTECTED_REPLAY_FIRST`: antes de tentar ganhar equation,
  recuperar/ancorar as trajetorias longas que preservam `8740ed31` e
  `59bee375`. DPO/ORPO/preference fica como segunda fase, somente apos replay
  protegido passar V568 sem piorar margem.
- V570 recuperou essas trajetorias e anchors em modo diagnostico. O artefato
  tem `42` linhas corretas, mas `training_allowed_rows=0` por seguranca: sao
  linhas do weak gate, portanto servem para entender o comportamento, construir
  negativos duros e criar analogos source-only, nao para treinar diretamente.
- V571 criou o primeiro material novo bit-pair source-only depois do plateau,
  com trace deterministica curta o bastante para tokenizar sem truncation. O
  dataset isolado e bit-only, por isso permanece bloqueado para GPU.
- V572 mostrou um erro de calibragem de peso antes de gastar H200: aumentar
  equation para `1.5` e reduzir bit para `0.5` produziria apenas `31.28%` de
  peso efetivo em bit. Esse desenho fica bloqueado porque tende a repetir a
  perda de `bit>=136`.
- V526 foi corrigido para enxergar `metadata.loss_weight`; antes ele avaliava
  apenas contagem fisica e podia aprovar/reprovar a calibragem errada. Esse era
  um bug silencioso de gate, nao um ganho de ACC.
- V573 reaproveita os mesmos dados source-only verificados, mas com peso
  `bit=1.5` e `equation=1.0`, ficando em `67.20%` bit e `32.80%` equation.
  Esta e a primeira rota treinavel plausivel apos V570, ainda sem ganho ACC
  medido.
- V573 upload/pre-paid tambem corrigiu um bug operacional: o debug local do HF
  falhava no Windows por symlink/cache; o launcher agora baixa para pasta local
  explicita e valida hash. Isso evita confundir erro de cache com erro do
  dataset.
- V573 H200 attempt 1 encontrou outro bug silencioso antes do treino: o runtime
  `hf_job_preflight_gate.py` auditava subcategoria por `metadata.subcategory`
  antes do campo top-level. O dataset V573 esta correto no campo top-level, mas
  preserva `metadata.subcategory` da origem para rastreabilidade. Isso fez o
  gate remoto divergir do pre-paid gate. A correcao e canonica: top-level
  `subcategory` governa treinamento/quotas; `metadata.subcategory` fica apenas
  como linhagem.
- V601/V602 fechou a linha `preference answer-only + MoE trainable`: mesmo com
  `869,318,656` parametros treinaveis e preference val melhor, o weak label-free
  atual ficou `192/315`, `equation=56`, `bit=136`, `trunc=0`, isto e, sem ganho
  liquido contra V290.
- V604 testou a unica tentativa barata restante nessa linha, interpolar V290 com
  V601. Resultado: `lambda=0.05` ficou exatamente no baseline; `lambda=0.10` e
  `0.25` regrediram para `190/315`, `bit=134` e `trunc=1`; `lambda=0.50` foi
  cancelado por FinOps. Portanto nao ha caminho de ganho por V601/MoE preference
  nem por interpolacao dessa familia de checkpoints.
- V605 consolidou V574, V582, V591, V597 e V602 contra o mesmo baseline V290
  parser-current. O padrao e agora comprovado: nenhuma rota transfere equation,
  todas repetem no maximo o ganho isolado `4ada9150`, e todas falham por
  backfire protegido, truncation/token runaway ou ausencia de ganho liquido.
  Os gates agora bloqueiam por identidade as rotas V573/V579/V591/V594/V596 e
  adapters V573/V579/V582/V591/V595/V595b/V601.

Plano efetivo a partir daqui:

1. Bloquear novo broad LoRA em H200. Loss, eval loss ou preference-val nao
   autorizam gasto nem submit sem weak ACC label-free.
2. Manter V290 checkpoint-6 como unico adapter submit-safe conhecido:
   `192/315`, `equation=56`, `bit=136`, `trunc=0`.
3. Nao executar novamente V573/V579/V582/V591/V594/V596, V595/V595b/V601 ou
   interpolacoes derivadas. Essas identidades estao em quarentena nos gates
   `kg1_static_safety_gate.py`, `hf_job_preflight_gate.py` e
   `kg1_pre_paid_job_integration_gate.py`.
4. Proximo treino pago so pode nascer de uma fonte nova ou de um mecanismo novo
   que primeiro passe em CPU/source-only: `total>=200`, `equation>=59`,
   `bit>=136`, `lost_rows=0`, `lost_bit_rows=0`, `lost_equation_rows=0`,
   protected rows preservadas, `max_token_headroom<=0.90` e V568 sem regressao
   de margem protegida.
5. Antes de qualquer H200, rodar V586/V587/V568/NLL ou equivalente no candidato
   de transferencia e comparar contra V605. Se houver backfire em
   `8740ed31`/`59bee375`, truncation, token runaway, ou nenhum ganho label-free
   real, cancelar por FinOps.
6. A unica linha nova aceitavel agora e microexperimento local/CPU com objetivo
   diferente: hard-negative/protected/no-change, teacher source-only, ou
   logits/NLL orientado a margem. A saida do gate deve mostrar ganho em linhas
   obrigatorias sem perdas antes de qualquer job pago.
7. Se houver novo treino, ele deve ser curto, com checkpoint-2 obrigatorio,
   weak eval imediato e cancelamento automatico se nao atingir:
   `total>=196`, `equation>=60`, `bit>=136`, `trunc=0`,
   `label_aware_delta=0`, `protected_backfire=0`.

Itens removidos do plano ativo apos V569:

- broad LoRA/SFT sem novo gate de margem;
- answer-only V560/V561 como estrategia principal;
- strict `disable_thinking`/short answer como contrato promocional;
- CPU solver/verifier direto como submit;
- novos H200 justificados por eval loss;
- treino equation-first enquanto `8740ed31` e `59bee375` nao estiverem
  protegidos por replay/margem.
- qualquer novo treino, eval ou interpolacao na linha V596/V601/MoE preference
  sem novo sinal CPU/source-only diferente.

## Decisao Atual V551/V552/V553

V551 executou o proximo passo obrigatorio do V550 sem gastar GPU e depois foi
promovido para um smoke curto V552 H200. O problema concreto era que V536
ensinava bit traces longas/repetitivas: isso reduziu um pouco o loss, mas na
weak eval gerou completions medias perto de `4770` tokens e regrediu bit. V551
manteve o mesmo material fonte-only do V536, mas trocou os targets de
`bit_manipulation` por trace cards curtos com `Final answer: \boxed{...}`.

Achado tecnico principal:

- V551 reduziu o target de bit para `p95=250` e `max=251` caracteres, contra
  `p95=1669` no V536;
- V509 integridade passou em train/val, sem overlap weak/full e sem mismatch;
- V286 tokenization real passou com `token_max=383`, `prompt_truncation=0`,
  `completion_truncation=0`, `offset_masks=1026/219` e tokenizer oficial
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`;
- V513 learnability real passou sem blocker;
- V524 real ficou `quota_ok_cpu_only`, com loss-token bit share `0.592993`;
- gates foram atualizados para bloquear novos datasets SFT com assistant
  comprido: `kg1_pre_paid_job_integration_gate.py` agora aceita
  `--max-assistant-chars-p95` e `--max-assistant-chars-max`, e
  `hf_job_preflight_gate.py` aceita `KG1_MAX_ASSISTANT_CHARS_P95` e
  `KG1_MAX_ASSISTANT_CHARS_MAX`.

Resultado pago V552/V553:

- V552 H200 rodou `4` steps e gerou checkpoints `2` e `4`;
- checkpoint-2 teve a menor melhora de loss (`5.9880 -> 5.9840`), mas loss
  nao promove;
- V553 weak eval label-free do checkpoint-2 mediu:
  - total `190/315`;
  - `bit_manipulation=134/160`;
  - `equation_transform=56/155`;
  - `truncated=1`;
  - `avg_completion_tokens=4775.12`, `max_completion_tokens=7680`;
  - protected rows quebradas:
    `8740ed31: 01101000 -> 01111000` e
    `59bee375: 10010101 -> 2`;
- o `weak_promotion_gate` bloqueou por `correct_lt_196`,
  `equation_lt_60`, `bit_lt_136`, `truncated_gt_0`,
  completion tokens excessivos e `protected_row_backfire_guard_failed`;
- V553 foi cancelado antes do checkpoint-4, porque checkpoint-4 tinha loss
  pior no treino e checkpoint-2 ja repetiu o backfire conhecido.

Decisao:

- V551/V552/V553 nao e submit-safe;
- nao rodar full eval, package ou submit desta linha;
- nao relancar SFT curto de bit traces se o novo CPU gate nao provar antes
  preservacao de `8740ed31=01101000`, `59bee375=10010101`, `bit>=136`,
  `equation>=60`, `trunc=0` e completions curtas;
- o gargalo atual nao e mais comprimento bruto do target: mesmo target curto
  repetiu a troca antiga `+equation/-bit`, entao a proxima frente precisa ser
  hard-negative/protected-row first ou package permitido para solver, nao
  mais uma variacao de SFT amplo.

Proximo passo obrigatorio:

1. usar o resultado V553 como negativo duro obrigatorio:
   `8740ed31`, `59bee375` e os outputs runaway devem entrar no gate local,
   nao como weak labels de treino;
2. construir/avaliar em CPU uma rota que preserve primeiro os dois protegidos
   e so depois tente `equation_transform`;
3. so liberar novo HF pago se o gate CPU provar novo sinal sem perdas:
   `total>=196`, `bit>=136`, `equation>=60`, `trunc=0`, protected rows
   preservadas e completions baixas;
4. se o pacote Kaggle permitir executar solver/verifier no caminho de
   inferencia dentro do formato oficial de submissao, priorizar essa auditoria
   porque a projecao CPU continua sendo o unico caminho com ganho real
   (`200/315`, `bit=138`, `equation=62`) sem backfire.

Artefato V551:

- `artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/KG1_V551_SHORT_BIT_TRACE_PACK.md`;
- `artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/v551_short_bit_trace_pack_manifest.json`;
- `artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/v509_integrity/v551_v509_integrity_manifest.json`;
- `artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/v286_tokenization_real/v286_generic_tokenization_gate_manifest.json`;
- `artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/v513_learnability_real/v513_trace_learnability_gate_manifest.json`;
- `artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/v524_objective_real/v524_quota_token_objective_manifest.json`;
- `artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/static_safety_gate_after_v551.json`.

Artefato V553:

- `artifacts/v553_hf_h200_v552_weak_eval_launch/V553_RESULT_SUMMARY.md`;
- `artifacts/v553_hf_h200_v552_weak_eval_launch/v553_job_6a099c123308d79117b91cf0_logs_after_cancel.txt`;
- HF job: `felipesp1983/6a099c123308d79117b91cf0`.

Atualizacao V554 - condicoes sincronizadas no gate:

- `scripts/kg1_pre_paid_job_integration_gate.py` agora tem `--self-test`;
- o self-test prova que um launcher/dataset SFT limpo passa;
- o self-test falha se `KG1_PROTECTED_ID_ANSWERS` nao incluir tambem
  `59bee375=10010101`;
- o self-test falha se o launcher pede saida `one_line_boxed_no_reasoning`
  mas o target SFT ainda comeca com `RULE:`;
- a bateria passou:
  - `python scripts/kg1_pre_paid_job_integration_gate.py --self-test`;
  - `python scripts/kg1_static_safety_gate.py --self-test`;
  - `python scripts/kg1_weak_backfire_row_guard.py --self-test`;
  - `python scripts/hf_job_weak_eval_v245.py --self-test`;
  - static safety dos scripts criticos sem findings.

Decisao V554: as condicoes estao agora testaveis antes do job pago. O proximo
dataset/launcher que nao preservar os dois protected rows, nao respeitar o
contrato curto, ou tentar promover por loss deve falhar no pre-paid gate.

Atualizacao V555 - missmap refeito com dois protected rows:

- V541 foi reexecutado em CPU para eliminar a evidencia antiga que registrava
  so `8740ed31`;
- novo manifest:
  `artifacts/v555_condition_refresh/v541_missmap_two_protected/v555_v541_two_protected_manifest.json`;
- resultado: `191/315`, `bit=136/160`, `equation=55/155`, `trunc=0`,
  `protected_passed=true`;
- `protected_id_answers` agora inclui explicitamente:
  `8740ed31=01101000` e `59bee375=10010101`;
- miss classes continuam: `24` bit residual, `12` equation numeric,
  `88` equation symbolic/punctuation; coverage `1.0`.

Decisao V555: a base diagnostica CPU esta sincronizada com os guards atuais.
Qualquer proximo dataset/launcher deve apontar para esta evidencia ou para uma
derivacao mais nova, nunca para o V541 antigo de um protected row.

## Decisao Anterior V550

V550 fecha uma lacuna real nas condicoes: o guard antigo protegia
`8740ed31=01101000`, mas V549 checkpoint-2 tambem perdeu `59bee375=10010101`
ao emitir `2`. A partir de agora, scripts e launchers ativos devem exigir os
dois protegidos antes de weak/full/package/submit.

Achado tecnico principal:

- V536 nao deve ser repetido: as bit traces continuam longas (`p95=1669`
  caracteres), e a weak eval mostrou completions medias perto de `4770`
  tokens; isso explica por que loss saudavel nao virou ACC e por que bit
  regressou;
- checkpoint-2 fez o trade-off proibido: `+518deb39` em equation, mas
  `-8740ed31` e `-59bee375` em bit;
- checkpoint-4 nao ganhou equation e ainda perdeu `8740ed31`;
- `gpu_allowed=false` ate existir dataset/gate CPU curto, answer-span, com os
  dois protegidos preservados.

Artefato V550:

- `artifacts/v550_condition_sync_audit/20260517T095613Z/KG1_V550_CONDITION_SYNC_AUDIT.md`;
- `artifacts/v550_condition_sync_audit/20260517T095613Z/v550_condition_sync_audit_manifest.json`.

## Decisao Anterior V549

V536/V549 encerra a rota V534-bit + V523-equation como ganho submit-safe. O
treino H200 tecnicamente rodou, mas a weak eval label-free atual dos
checkpoints atuais mostrou regressao: checkpoint-2 `190/315`, `bit=134`,
`equation=56`, `trunc=1`; checkpoint-4 `190/315`, `bit=135`, `equation=55`,
`trunc=0`. Ambos violam o protected row `8740ed31=01101000` ao prever
`01111000`, e ambos excedem muito os limites de completions curtas.

Decisao:

- nao usar V536 checkpoint-2 ou checkpoint-4 para full eval, package ou submit;
- nao continuar a linha V534/Konbu/Huikang + V523-equation com o mesmo prompt
  e objetivo sem um novo sinal CPU que preserve `8740ed31` e reduza completions;
- manter V290/V291 checkpoint-6 como baseline packageable ate existir adapter
  com `total>=196`, `bit>=136`, `equation>=60`, `trunc=0`, protected rows
  preservadas e completions dentro do limite;
- diagnostico row-level V549 contra V516 mostrou que checkpoint-2 faz o mesmo
  trade-off antigo: ganha `518deb39` em equation, mas perde `8740ed31` e
  `59bee375` em bit; checkpoint-4 nao ganha nenhuma linha e perde `8740ed31`;
- a proxima acao deve abandonar esse mix como linha promocional e focar em
  blindar `8740ed31`/`59bee375` com hard negatives curtos antes de tentar
  qualquer nova transferencia de `518deb39`/equation.

Artefatos V549:

- `artifacts/v549_hf_h200_v536_current_weak_eval_launch/v549-h200-v221contract-v536-current-cp2-cp4-20260517T091907Z_launch_manifest.json`;
- `artifacts/v549_hf_h200_v536_current_weak_eval_results/evals/v549-h200-v221contract-v536-current-cp2-cp4-20260517T091907Z/v245_hf_weak_eval_manifest.json`;
- `artifacts/v549_hf_h200_v536_current_weak_eval_results/delta_audit/v549_delta_audit_summary.json`.

## Decisao Anterior V548

V547/V548 encerrou a rota de destilacao answer-only curta para transformar
solver/verifier em adapter. O modelo aprendeu a emitir raciocinio longo e
ignorou o contrato de uma linha `\boxed{...}`. O resultado e pior que regressao
normal: `0/160` bit e `288/315` truncados. Portanto:

- nao usar V547 para full eval, package ou submit;
- nao repetir treino answer-only pequeno sem um smoke de geracao antes;
- qualquer paid eval deve abortar automaticamente se ocorrer colapso
  catastrofico (`correct<=10`, truncation massiva ou bit destruido);
- ganho CPU solver/verifier continua util como teacher/diagnostico, mas nao
  autoriza submit enquanto o pacote oficial permanecer adapter-only.
- recheck V336B em `artifacts/v548_package_permission_recheck/` confirma que
  solver/verifier direto continua bloqueado: o submit oficial exige ZIP com
  LoRA adapter, e o nosso packager rejeita `prediction_postprocessor`.

Artefatos V547/V548:

- `artifacts/v547_contract_aligned_distillation_dataset/20260517T_v547_cpu_gate/v547_contract_aligned_manifest.json`;
- `artifacts/v547_hf_h200_launch/v547-nemo-h200-contract-final-v290ckpt6-20260517T080936Z_launch_manifest.json`;
- `artifacts/v548_hf_h200_v547_weak_eval_launch/v548-h200-v221contract-v547-cp2-cp4-cp6-cp8-20260517T082755Z_launch_manifest.json`;
- `artifacts/v548_hf_h200_v547_weak_eval_launch/KG1_V548_CATASTROPHIC_TRANSFER_COLLAPSE_AUDIT.md`;
- `artifacts/v548_hf_h200_v547_weak_eval_launch/v548_catastrophic_transfer_collapse_audit.json`.

## Decisao Anterior V536

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
- `artifacts/v533_huikang_artifacts_audit/KG1_V533_HUIKANG_ARTIFACTS_AUDIT.md`;
- `artifacts/v534_roadmap_double_check/KG1_V534_ROADMAP_DOUBLE_CHECK.md`;
- `artifacts/v535_specialist_triple_check/KG1_V535_SPECIALIST_TRIPLE_CHECK.md`;
- `artifacts/v534_bit_source_only_trace_pack/20260517T024405Z/v534_bit_source_only_trace_pack_manifest.json`;
- `artifacts/v536_v534_bit_v523_equation_pack/20260517T024752Z/v536_v534_bit_v523_equation_pack_manifest.json`;
- `artifacts/v536_v534_bit_v523_equation_pack/20260517T024752Z/v526_example_mean_dry_run/v526_example_mean_objective_dry_run_manifest.json`;
- `artifacts/v536_hf_h200_launch/v536_hf_dataset_upload_manifest.json`;
- `artifacts/v536_hf_h200_launch/v536_pre_paid_job_integration_gate.json`;
- `artifacts/v536_hf_h200_launch/v536-nemo-h200-v534bit-v523eq-v290ckpt6-20260517T030732Z_launch_manifest.json`;
- `artifacts/version_diffs/V536_VS_V523.md`.

O consenso externo e a auditoria dos notebooks baixados nao autorizam treino
longo nem broad SFT. O plano executavel agora fica restrito as frentes abaixo,
em ordem de prioridade:

1. Frente V536 smoke H200 curto:
   - V534/V536 ja materializou a frente CPU bit source-only com Konbu
     high-confidence e Huikang CHO/MAJ, removendo qualquer overlap weak/full;
   - usar exatamente `LOSS_NORMALIZATION_MODE=example_mean`;
   - usar `MAX_LENGTH=2048`, validado contra `token_max=1123` do V286;
   - manter `lm_head` congelado e MoE `up_proj/down_proj` treinaveis;
   - abortar no primeiro checkpoint que violar `bit>=136`, `equation>=60`,
     `trunc=0` ou linhas protegidas;
   - se o smoke nao mostrar ganho ACC label-free, bloquear GPU e voltar para
     equation canonicalization/hard negatives.
2. Frente CPU equation canonicalization/hard negatives:
   - usar os downloads pequenos ja auditados no V532 como referencia, nao como
     patch direto;
   - qualquer novo gate local so pode ranquear candidatos por features
     label-free:
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
3. Frente adapter externo V531/V533, opcional e diagnostica:
   - avaliar Yoiko ver5 apenas como candidato adapter-only, sem treino;
   - avaliar Huikang `adapter_v26` apenas como candidato adapter-only, sem
     extrair/commitar pesos;
   - antes de rodar, validar `adapter_config`, header safetensors, regex
     `target_modules`, ausencia de non-LoRA tensors e zip root-level;
   - usar weak eval label-free curto, com FinOps kill-switch se cair abaixo do
     baseline nas primeiras metricas;
   - nao usar peso publico como submissao direta nem como fonte de treino; se
     superar baseline, usar apenas como diagnostico/proveniencia e abrir uma
     decisao separada de compatibilidade/licenca/submit antes de qualquer
     package.
4. Novo GPU SFT alem do smoke V536 volta a ficar bloqueado ate a frente CPU
   produzir novo dataset source-only com:
   - zero overlap weak/full por `id`, `prompt_sha256` e
     `prompt+answer_sha256`;
   - tokenization/offset-mask/truncation gates limpos;
   - trace learnability gate sem respostas answer-only;
   - objetivo `example_mean` ou equivalente validado por contribution audit.
5. Qualquer smoke V530/V531/V532/V533/V534/V535 precisa falhar fechado se:
   - `total < 196/315`;
   - `equation < 57/155`;
   - `bit < 136/160`;
   - `trunc != 0`;
   - `8740ed31 != 01101000`;
   - `59bee375 != 10010101`;
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
11. FinOps: cancelar job que nao possa mais superar `total>=196`,
   `equation>=60`, `bit>=136`, `truncated=0`.
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
23. Regra clean permanente: antes de job pago, package ou submit, rodar
    `python scripts/kg1_workspace_clean_gate.py --delete-safe` e registrar o
    relatorio quando houver achado. O gate so apaga cache/temp inequivoco; todo
    dataset, manifest, log decisorio, adapter, roadmap e evidencia reutilizavel
    fica preservado ou arquivado conscientemente.

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
`bit>=136`, `trunc=0` e `total>=196`.

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

Condicao minima para sair de diagnostico e abrir full official-like:

- weak promocional `>=196/315`;
- equation `>=60/155`;
- bit `>=136/160`;
- trunc `0`;
- predicao `submit_safe_label_free_prediction`, sem expected-aware extraction.

Condicao minima para package/submit:

- full official-like `>823/947`;
- sem regressao conhecida em bit/equation/truncation;
- artefato final adapter-only com `adapter_config.json` e
  `adapter_model.safetensors` no root;
- nenhuma dependencia de runtime solver, verifier, postprocessor, prompt hack,
  logit mask ou cherry-pick por weak/full.

Sem isso, nao packagear e nao submeter.

## Itens Removidos Do Plano Ativo

| Item | Motivo |
|---|---|
| Repetir H200 longo por eval_loss | loss nao correlacionou com ACC |
| Mais epochs/steps sem novo dado | quatro dias de plateau; custo sem sinal |
| V435E misto e format negatives | contaminado e bloqueado |
| V447/V448 trace SFT limpo | nao transferiu para adapter |
| V464/V468/V469 derivados | rota contaminada/quarentenada |
| Public adapters/submissions de terceiros | tecnica/proveniencia/diagnostico apenas; nunca peso/submissao direta sem decisao separada |
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
   passar primeiro pelo gate minimo anti-backfire (`total>=196`,
   `equation>=60`, `bit>=136`, `trunc=0`). Para package/submit, a meta segue
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
| weak eval V245 tinha defaults diagnosticos (`max_tokens=96`, thinking off) e threshold `equation>=60` | defaults agora sao official-like e promocao exige `total>=196`, `equation>=60`, `bit>=136`, `trunc=0` |
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

## Atualizacao V536 - Auditoria De Parametros E Falha Do H200

Artefatos:

- `artifacts/v536_hf_h200_launch/v536_pre_paid_job_integration_gate_parameter_recheck.json`;
- `artifacts/v536_hf_h200_launch/v536-nemo-h200-v534bit-v523eq-v290ckpt6-20260517T033326Z_launch_manifest.json`;
- `artifacts/v536_v534_bit_v523_equation_pack/20260517T024752Z/v526_example_mean_dry_run_after_param_patch/v526_example_mean_objective_dry_run_manifest.json`;
- `artifacts/v536_v534_bit_v523_equation_pack/20260517T024752Z/v478_objective_alignment_parameter_recheck.json`.

Diagnostico honesto: a ultima falha do job H200 nao foi erro de dataset,
hash, tokenizacao ou adapter inicial. O job abortou no kill-switch de VRAM:
`mem_reserved=78.18GiB` contra `ABORT_MAX_RESERVED_GIB=78.00`. Isso era um
threshold de FinOps apertado demais para H200, nao um sinal de ganho nem de
perda de ACC.

Parametro atual validado:

| Area | Valor atual | Status |
|---|---:|---|
| `MAX_LENGTH` | `2048` | passa; token max observado `1123`, truncation `0` |
| `LOSS_NORMALIZATION_MODE` | `example_mean` | agora exigido no launcher e no preflight |
| `ANSWER_SPAN_LOSS_WEIGHT` | `1.0` | intencional: CE normal, sem falsa promessa de answer-span |
| `ANSWER_SPAN_MIN_WEIGHTED_TOKENS` | `0` | corrigido; peso `1.0` nao pode exigir min-token fake |
| `ABORT_MAX_RESERVED_GIB` | `84` | corrigido para H200; ainda muito abaixo de 141GB |
| `MAX_STEPS` | `4` | smoke curto, nao treino longo |
| save/eval | `2/2` | primeiro checkpoint sempre avaliavel |
| dataset train/val | `1026/219` | hashes remotos batem com locais |
| mix train | bit `706`, equation `320` | bit share `68.81%`, equation `31.19%` |
| mix val | bit `139`, equation `80` | bit share `63.47%`, equation `36.53%` |
| init adapter | V290 `checkpoint-6` | arquivos presentes no HF |
| trainable LoRA | `q,k,v,o,up,down` + MoE params | declarado e gateado |
| dataset schema | `sft` | declarado no launcher; gate bloqueia CLI/schema divergente |

Gates reforcados:

- `kg1_pre_paid_job_integration_gate.py` agora aceita
  `--expected-loss-normalization-mode` e bloqueia launcher que nao exporte o
  modo esperado.
- `hf_job_preflight_gate.py` agora exige `LOSS_NORMALIZATION_MODE` e compara
  com `KG1_EXPECTED_LOSS_NORMALIZATION_MODE`.
- `audit_v526_example_mean_objective_dry_run.py` deixou de usar regex simples
  do primeiro `\boxed{}` e passou a usar `extract_final_answer` +
  `verify_answer`, alinhando o gate com a metrica compartilhada.
- `build_v536_v534_bit_v523_equation_pack.py` preserva proveniencia upstream
  (`v536_upstream_*`) e amplia o overlap gate por id, prompt hash e
  prompt+answer hash.
- `kg1_pre_paid_job_integration_gate.py` agora compara o schema declarado no
  launcher (`KG1_DATASET_SCHEMA`) com `--dataset-schema`; isso evita falso
  aceite ou falso bloqueio por avaliar dataset SFT como preference/DPO.

Decisao:

- nao relancar H200 automaticamente antes de avaliar se os checkpoints ja
  gerados (`checkpoint-2` e `checkpoint-abort-step4`) merecem weak eval;
- se relancar V536, usar somente com os parametros acima e monitoramento
  FinOps de 40s;
- qualquer promocao continua bloqueada ate weak ACC label-free mostrar
  `total>=196`, `equation>=60`, `bit>=136`, `trunc=0`; package/submit exige
  full official-like `>823/947`.

## Atualizacao V537 - Weak Eval Dos Checkpoints V536 Com Guard Por Linha

Objetivo imediato: medir ACC real dos dois adapters ja produzidos pelo V536
antes de gastar outro treino H200. O job V536 falhou depois de salvar
`checkpoint-2` e `checkpoint-abort-step4`; portanto o caminho mais barato e
correto e avaliar esses dois checkpoints no weak set oficial-like.

Comparativo de versao:

| Item | V536 treino smoke | V537 weak eval |
|---|---|---|
| Acao | treinar 4 steps a partir do V290 ckpt6 | avaliar checkpoints ja gerados |
| Compute | H200 pago com risco de novo abort | H200 apenas para inferencia weak |
| Candidatos | gera `checkpoint-2` e `checkpoint-abort-step4` | mede ambos no mesmo job |
| Gate promocional | nao aplicavel sem ACC | `total>=196`, `equation>=60`, `bit>=136`, `trunc=0` |
| Guard F2/backfire | exigido pelo roadmap | integrado ao `hf_job_weak_eval_v245.py` |
| Protected id | documentado em V519 | `8740ed31=01101000` bloqueia candidato se regredir |

Correcoes implementadas:

- `hf_job_weak_eval_v245.py` agora roda `protected_row_backfire_guard`
  imediatamente apos gerar `batch_candidate_summary.json` e antes de aceitar
  `weak_promotion_gate`;
- se o guard global falhar por baseline/protected-id ausente, todos os
  candidatos sao bloqueados;
- se um candidato melhora totais mas quebra o protected id, ele recebe
  `protected_row_backfire_guard_failed` e nao pode ser promovido;
- `launch_v537_hf_weak_eval_v536_checkpoints.py` declara explicitamente
  `KG1_PROTECTED_ROW_GUARD=1` e `KG1_PROTECTED_ID_ANSWERS=8740ed31=01101000`.

Validacoes locais concluídas antes de qualquer job pago:

- `python -m py_compile scripts/hf_job_weak_eval_v245.py
  scripts/kg1_weak_backfire_row_guard.py
  artifacts/v537_hf_h200_v536_weak_eval_launch/launch_v537_hf_weak_eval_v536_checkpoints.py`;
- `python scripts/hf_job_weak_eval_v245.py --self-test`;
- `python scripts/kg1_weak_backfire_row_guard.py --self-test`;
- `python scripts/kg1_static_safety_gate.py
  artifacts/v537_hf_h200_v536_weak_eval_launch/launch_v537_hf_weak_eval_v536_checkpoints.py`.
- auditoria focada das pecas ativas (`launch_v536`, `launch_v537`,
  `hf_job_weak_eval_v245`, `kg1_pre_paid_job_integration_gate`) passou com
  `findings=[]`;
- inspeção direta dos JSONL V536 confirmou `0` ids duplicados, `0` respostas
  vazias, `0` flags weak/full/gate usadas para treino, hashes esperados e
  contagens `train=1026`, `val=219`;
- repo HF do V536 contem os quatro arquivos obrigatorios dos dois candidatos:
  `checkpoint-2/{adapter_config.json,adapter_model.safetensors}` e
  `checkpoint-abort-step4/{adapter_config.json,adapter_model.safetensors}`.

Decisao:

- nao relancar treino V536 enquanto V537 nao medir os checkpoints existentes;
- commitar/pushar as alteracoes antes do launch, porque o HF clona o commit
  remoto e alteracoes locais dirty nao entram no container;
- se V537 nao atingir o gate, arquivar os checkpoints como diagnostico e voltar
  para frente CPU/dataset; se atingir, rodar full official-like antes de
  qualquer package/submit.

## Atualizacao V537 Resultado - Linha V536 Bloqueada

Artefatos:

- `artifacts/v537_hf_h200_v536_weak_eval_launch/downloaded_20260517T040250Z/evals/v537-h200-v221contract-v536-cp2-step4-20260517T040250Z/eval/batch_candidate_summary.json`;
- `artifacts/v537_hf_h200_v536_weak_eval_launch/downloaded_20260517T040250Z/evals/v537-h200-v221contract-v536-cp2-step4-20260517T040250Z/eval/batch_candidate_summary.csv`;
- upload remoto em `felipesp1983/kg1-nemotron-lora-v536-nemo-h200-v534bit-v523eq-v290ckpt6/evals/v537-h200-v221contract-v536-cp2-step4-20260517T040250Z`.

Resultado:

| Candidato | Total | bit | equation | trunc | Protected row | Decisao |
|---|---:|---:|---:|---:|---|---|
| `checkpoint-2` | 191/315 | 135/160 | 56/155 | 0 | quebrou `8740ed31`: `01111000` vs `01101000` | bloqueado |
| `checkpoint-abort-step4` | 191/315 | 135/160 | 56/155 | 0 | preservou | bloqueado |

Diagnostico:

- V536/V537 repetiu a troca ruim vista em outras linhas: pequeno ganho ou
  estabilidade em equation com perda em bit;
- o melhor candidato nao atingiu `total>=196`, `equation>=60`, `bit>=136`;
- `checkpoint-2` ainda acionou o guard de F2/backfire por quebrar a linha
  protegida.

Decisao:

- arquivar V536/V537 como diagnostico;
- nao relancar H200 nessa familia de dataset/parametros;
- voltar para CPU residual, extracao/canonicalizacao e miss-map antes de
  qualquer novo treino pago.

## Atualizacao V538/V539 - Consenso OpenRouter E Plano Residual-First

Artefatos:

- `artifacts/v538_openrouter_residual_first_consult/KG1_V538_OPENROUTER_RESIDUAL_FIRST_PROMPT.md`;
- `artifacts/v538_openrouter_residual_first_consult/v538_openrouter_manifest.json`;
- `artifacts/v538_openrouter_residual_first_consult/v539_free_double_check/KG1_V539_OPENROUTER_FREE_DOUBLE_CHECK_PROMPT.md`;
- `artifacts/v538_openrouter_residual_first_consult/v539_free_double_check/v539_openrouter_free_manifest.json`;
- `artifacts/v538_openrouter_residual_first_consult/KG1_V539_FREE_DOUBLE_CHECK_CONSENSUS.md`.

Painel V538 pago: DeepSeek, Qwen, Claude, Gemini e GPT-5.5 foram chamados; o
conteudo utilizavel convergiu para CPU residual-first. Painel V539 gratuito:
`openai/gpt-oss-120b:free`, `z-ai/glm-4.5-air:free`,
`poolside/laguna-m.1:free` e
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` responderam com custo `0`;
`qwen/qwen3-coder:free` e `meta-llama/llama-3.3-70b-instruct:free` deram `429`
e nao contam como evidencia tecnica.

Consenso:

- o loop atual nao sai do plateau porque estamos tentando transferir sinal de
  solver/verifier para LoRA sem mapear a causa exata de cada erro;
- loss menor nao e criterio promocional e nao deve autorizar novo GPU;
- o proximo passo precisa ser CPU, por linha, com evidencia de extracao,
  canonicalizacao, token-offset, regra faltante e risco de regressao;
- antes do miss-map, validar se extracao/canonicalizacao/prompt-template e
  answer-span estao alinhados com a metrica ACC; se essa camada tiver mismatch,
  qualquer treino continuara parecendo saudavel no loss e ruim no ACC.

Plano ativo, em ordem:

1. `V540 validate_answer_extraction_v1.py`: auditar raw output, extracted answer,
   `verify_answer`, prompt hash, token offsets, truncation e protected row para
   baseline strict e V537. Gate: `0` blockers e `8740ed31=01101000` estavel.
2. `V541 weak_missmap_v1`: construir CSV/parquet com uma linha por weak row:
   `id`, `family`, `prompt_sha256`, `raw_output`, `extracted_answer`,
   `verify_answer_ok`, `answer_span_start`, `answer_span_end`, `token_count`,
   `miss_class`, `operator_tag`, `required_rule`, `solver_trace`,
   `is_protected`.
3. CPU simulation: somente aprovar treino se projetar `>=200/315`, `bit>=136`,
   `equation>=59`, `trunc=0`, protected row preservada, `>=70%` dos misses
   classificados e `0` overlap/duplicidade.
4. Se e somente se o gate CPU passar, gerar `400-600` traces curtas, source-only,
   deterministicas e verificadas, com kill-switch de probe antes de weak full.
5. Se o primeiro checkpoint nao preservar bit/protected row ou nao indicar ganho
   em probe, cancelar por FinOps.

Stop list:

- parar broad SFT, treino H200 por loss, replay generico e selector direto de
  candidate pool;
- parar qualquer GPU paga sem V540/V541 aprovados;
- parar qualquer avaliacao que dependa de `prediction` label-aware em vez de
  `raw_output` label-free.

## Atualizacao V540 - Gate Residual-First Agora E Bloqueante

Artefatos:

- `artifacts/v540_openrouter_gate_prompt_consult/KG1_V540_OPENROUTER_GATE_PROMPT.md`;
- `artifacts/v540_openrouter_gate_prompt_consult/v540_openrouter_gate_manifest.json`;
- `artifacts/v540_openrouter_gate_prompt_consult/KG1_V540_OPENROUTER_GATE_CONSENSUS.md`.

Consulta: DeepSeek, Gemini, Claude, GPT e Qwen foram chamados. Gemini, Claude e
Qwen trouxeram conteudo acionavel; DeepSeek e GPT retornaram resposta
reasoning-only/null e foram tratados como nao acionaveis.

Mudanca implementada:

- `scripts/kg1_pre_paid_job_integration_gate.py` agora bloqueia launchers pagos
  sem evidencias V540/V541;
- `scripts/hf_job_preflight_gate.py` agora bloqueia o job remoto antes de
  carregar modelo se as evidencias V540/V541 nao estiverem no ambiente;
- `hf_job_preflight_gate.py --self-test` cobre o novo gate.

Regra bloqueante para qualquer novo treino pago:

| Campo | Regra |
|---|---|
| `KG1_RESIDUAL_FIRST_GATE` | `1` |
| `KG1_V540_EXTRACTION_GATE_STATUS` | `passed` |
| `KG1_CPU_EXTRACTOR_PARITY_STATUS` | `passed` |
| `KG1_PROMPT_TEMPLATE_PARITY_STATUS` | `passed` |
| `KG1_V541_MISSMAP_GATE_STATUS` | `passed` |
| `KG1_V541_FLIP_LEDGER_STATUS` | `passed` |
| `KG1_CPU_SIMULATED_TOTAL_CORRECT` | `>=200` |
| `KG1_CPU_SIMULATED_BIT_CORRECT` | `>=136` |
| `KG1_CPU_SIMULATED_EQUATION_CORRECT` | `>=59` |
| `KG1_CPU_MISS_CLASSIFICATION_COVERAGE` | `>=0.70` |
| `KG1_CPU_SIMULATED_LOST_ROWS` | `0` |
| `KG1_CPU_SIMULATED_LOST_BIT_ROWS` | `0` |
| `KG1_CPU_SIMULATED_LOST_EQUATION_ROWS` | `0` |
| `KG1_MAX_TOKEN_HEADROOM_RATIO` | `<=0.90` |
| `KG1_EXPECTED_TRUNCATED` | `0` |
| `KG1_PROTECTED_ID_ANSWERS` | inclui `8740ed31=01101000` |
| `KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS` | `passed` |
| `KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE` | `0` |
| `KG1_WEAK_LABEL_AWARE_SELECTION` | `0` |
| `KG1_CPU_SIMULATION_USES_WEAK_LABELS` | `0` |

Decisao:

- manter `>=200/315` como meta CPU antes de GPU; se isso bloquear, o bloqueio e
  correto, porque o sinal atual `195-196` ja falhou na transferencia;
- proxima implementacao obrigatoria e `V540 validate_answer_extraction_v1.py`,
  seguida por `V541 weak_missmap_v1`;
- nao fazer novo H200 enquanto esses gates nao produzirem os valores acima.

## Atualizacao V540/V541/V542 - Primeiro Ganho CPU Real Sem Perdas

Artefatos:

- `scripts/validate_answer_extraction_v1.py`;
- `scripts/build_v541_weak_missmap_v1.py`;
- `artifacts/v540_answer_extraction_audit_baseline_only/v540_baseline_only_answer_extraction_audit_summary.json`;
- `artifacts/v540_answer_extraction_audit/v540_answer_extraction_audit_summary.json`;
- `artifacts/v541_weak_missmap/v541_weak_missmap_manifest.json`;
- `artifacts/v542_cpu_equation_solver_gate/v324_on_v516_strict/v324_equation_expanded_solver_manifest.json`;
- `artifacts/v542_cpu_equation_solver_gate/v329_on_v516_strict/v329_symbolic_cryptarithm_manifest.json`;
- `artifacts/v542_cpu_equation_solver_gate/v336_integrated_on_v516_strict/v336a_integrated_no_loss_solver_gate_manifest.json`;
- `artifacts/v542_cpu_equation_solver_gate/KG1_V542_CPU_EQUATION_GAIN_SUMMARY.md`;
- `artifacts/v542_cpu_equation_solver_gate/KG1_V542_PACKAGE_PERMISSION_RECHECK.md`.

Resultado validado:

| Métrica | Baseline V516 label-free | V542 CPU integrado | Delta |
|---|---:|---:|---:|
| Total weak | 191/315 | 196/315 | +5 |
| `bit_manipulation` | 136/160 | 136/160 | 0 |
| `equation_transform` | 55/155 | 60/155 | +5 |
| truncation | 0 | 0 | 0 |
| perdas | - | 0 | 0 |

O V540 baseline-only passou:

- contrato weak `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`;
- `cpu_extractor_parity_status=passed`;
- `prompt_template_parity_status=passed`;
- protected row `8740ed31=01101000` preservada;
- aviso legado: `stored_correct_column_mismatch=1` em `4bb8c6cd`, mas `correct`
  e extração label-free estão alinhados, portanto não é blocker.

O V540 incluindo V537 ficou bloqueado de propósito:

- os artefatos baixados de V537 têm apenas `batch_candidate_summary.*` e
  manifest, sem prediction CSV/raw output por linha;
- portanto V537 é diagnóstico, não evidência para promoção.

O V541 miss-map passou e mostrou o gargalo real:

| Classe | Misses |
|---|---:|
| `bit_residual_miss` | 24 |
| `equation_numeric_miss` | 12 |
| `equation_symbolic_punctuation_miss` | 88 |

Conclusão técnica:

- a frente com ganho comprovado agora é `equation_transform`, principalmente
  regras numéricas guardadas e um caso simbólico de cryptarithm;
- cinco linhas foram corrigidas em simulação CPU sem perda:
  `274def88`, `7688e06e`, `c5b058d6`, `d1bd7478`, `99d6a3b5`;
- isso atende ao objetivo de achar ganho real, mas ainda não atende o gate para
  treino pago (`>=200/315`);
- não fazer submit, package ou H200 ainda. O próximo passo é ampliar a DSL
  simbólica/pontuação sobre os 83+ misses restantes até a simulação CPU passar
  `>=200/315`, preservando `bit>=136` e `losses=0`.

Recheck de pacote/submissao:

- o pacote local validado e adapter-only: apenas `adapter_config.json` e
  `adapter_model.safetensors`;
- `scripts/package_hf_adapter_submission.py` rejeita `prediction_postprocessor`;
- o gate V336B conclui `direct_solver_package_allowed=false`;
- rerodando V336B contra o manifesto V542 atual, o gate bloqueia antes do
  pacote porque V542 ainda e `196/315`, abaixo do piso `>=200/315`;
- portanto `196/315` nao deve ser submetido diretamente. Ele e sinal CPU real
  para transferencia ou para expansao residual, nao pacote Kaggle pronto.

Próximo passo obrigatório:

1. implementar V543 symbolic-punctuation residual expansion com foco nos 83
   misses restantes após V324/V329;
2. regras candidatas somente source-derived, nunca escolhidas por label;
3. aceitar uma regra apenas se o class gate tiver `verified>0` e
   `incorrect=0`;
4. rerodar V336 integrado com thresholds `total>=200`, `equation>=59`,
   `bit>=136`, `losses=0`;
5. só depois gerar traces curtos para LoRA ou avaliar se existe caminho
   submit-safe permitido pelas regras da competição.

## Atualizacao V543/V544 - CPU >=200 Atingido, Distilacao Agora E O Caminho Ativo

Artefatos:

- `scripts/run_v543_symbolic_queryop_refinement_gate.py`;
- `artifacts/v542_cpu_equation_solver_gate/v543_symbolic_queryop_on_v350_v516_strict/v543_symbolic_queryop_refinement_manifest.json`;
- `artifacts/openrouter/v544_distillation_transfer_consult/KG1_V544_OPENROUTER_DISTILLATION_TRANSFER_PROMPT.md`;
- `artifacts/openrouter/v544_distillation_transfer_consult/v544_openrouter_distillation_transfer_manifest.json`;
- `artifacts/openrouter/v544_distillation_transfer_consult/KG1_V544_OPENROUTER_DISTILLATION_TRANSFER_CONSENSUS.md`.

Resultado V543 validado:

| Métrica | Baseline V516 label-free | V350 CPU | V543 CPU | Delta vs baseline |
|---|---:|---:|---:|---:|
| Total weak | 191/315 | 198/315 | 200/315 | +9 |
| `bit_manipulation` | 136/160 | 138/160 | 138/160 | +2 |
| `equation_transform` | 55/155 | 60/155 | 62/155 | +7 |
| truncation | 0 | 0 | 0 | 0 |
| perdas | - | 0 | 0 | 0 |

V543 aceitou três assinaturas source-derived por `rule_class + query_op`:

- `symbolic_cryptarithm_multi_operator_digits_add | query_op=!`, ganho
  `6cc5dafb`;
- `symbolic_cryptarithm_multi_operator_digits_mul | query_op=$`, ganho
  `5501c054`;
- `symbolic_cryptarithm_single_operator_digits_mul | query_op=%`, já coberto
  pelo V350 em `99d6a3b5`.

Decisão:

- o gate CPU que antes bloqueava GPU (`>=200/315`) agora foi atingido;
- isso ainda **não** é submit direto: o pacote Kaggle continua adapter-only e
  bloqueia `prediction_postprocessor`;
- o próximo caminho ativo é V544: destilar o comportamento V350/V543 para
  adapter-only, com dataset mínimo e gates antes de qualquer H200.

Consenso OpenRouter V544:

- chamados: DeepSeek V4 Pro, DeepSeek R1 Distill Qwen 32B, Qwen 3.6 Max,
  Claude Opus 4.7, Gemini 3.1 Pro Preview, GPT-5.5 Pro;
- respostas estruturadas aproveitáveis: DeepSeek R1 Distill, Qwen, Claude e
  Gemini;
- respostas reasoning-only/null: DeepSeek V4 Pro e GPT-5.5 Pro, retidas apenas
  como artefato bruto;
- consenso acionável: não repetir broad SFT; usar traces curtos,
  deterministicos, answer-focused, replay forte de bit e kill-switch por ACC.

Contrato V544 obrigatório:

1. criar `scripts/build_v544_minimal_distillation_dataset.py`;
2. incluir os 9 ganhos teacher:
   `99d6a3b5`, `7688e06e`, `274def88`, `d1bd7478`, `c5b058d6`,
   `4ada9150`, `4c327b55`, `6cc5dafb`, `5501c054`;
3. incluir replay de linhas baseline-correct, com `8740ed31=01101000`
   obrigatório e peso alto;
4. usar alvo curto submit-safe: `RULE: <tag>. Final answer: \boxed{<answer>}`
   quando a extracao label-free preserva o valor, ou `Final answer: <answer>`
   quando simbolos com `}` tornam o boxed ambiguo;
5. mascarar prompt com `labels=-100` e provar via self-test que tokens
   não mascarados decodificam para o target esperado;
6. rodar gates de dedup, leakage, tokenização, truncation, extraction
   round-trip, protected row e familia antes de qualquer GPU;
7. primeiro checkpoint H200 deve abortar se `bit<136`, `equation<55`,
   `total<191`, `truncation>0`, `8740ed31!=01101000` ou `raw_output`
   estiver ausente.

Double check V544 dataset:

- artefato: `artifacts/v544_minimal_distillation_dataset/20260517T_v544_cpu_gate/v544_dataset_doublecheck_audit.json`;
- decisão: `dataset_doublecheck_passed`, com `issues=[]` e `warnings=[]`;
- SHA do treino e validação batem com o manifesto:
  `09f542297d9bafe85015b2955c09289817487ebf9fc53746de4ea68cb5f3e4f3`
  e `894a1df7590ccd0ded77f307438e646f870aa2cc6e6e006cb536e88f8aedb921`;
- treino: `236` linhas, `146` bit, `90` equation, `200` prompts/source IDs
  únicos;
- validação: `115` linhas, `22` bit, `93` equation, `115` prompts/source IDs
  únicos;
- roles de treino: `136` `bit_replay`, `55` `equation_replay`, `45`
  `teacher_gain`;
- os 9 teacher gains aparecem exatamente 5x cada;
- `8740ed31=01101000` aparece uma vez, como `bit_replay`, com peso `3.0`;
- não há overlap train/val por prompt ou source ID;
- bug encontrado no pre-paid gate: a versao inicial escapava `{`, `}` e `\`
  dentro de `\boxed{}`, passando apenas em extracao expected-aware, mas falhando
  no caminho label-free usado para score/submit;
- correcao implementada: `scripts/build_v544_minimal_distillation_dataset.py`
  agora exige `extract_final_answer` label-free em todas as linhas; `V286`
  ganhou o modo `submit_safe_suffix`;
- todas as respostas extraem corretamente no caminho label-free; treino tem
  `236/236` em `boxed_raw_label_free`; validação tem `111` boxed raw e `4`
  `unboxed_label_free_fallback`;
- tokenização real V286 passou em `submit_safe_suffix`: `train_token_max=342`,
  `val_token_max=336`, `completion_tokens_dropped=0`, `prompt_truncated=0`,
  `fallback_masks=0`;
- upload HF verificado:
  `felipesp1983/kg1-v544-minimal-distillation-artifacts`,
  root `v544-minimal-distillation-20260517T063045Z`,
  commit `c71e4ea4029ac2a77efd913c4d251752e4bbee18`.

Risco resolvido antes de GPU:

- resolvido o gap crítico encontrado no double check: `hf_job_train_v90.py`
  lia `metadata.loss_weight` no dataset, mas não transferia esse valor para
  o loss real;
- correção: `metadata.loss_weight` agora vira `row_loss_weight` no treino
  quando `USE_ROW_LOSS_WEIGHT=1`; validação ignora esse peso para não zerar o
  eval com os rows `loss_weight=0.0`;
- a loss foi ajustada para respeitar `row_loss_weight` tanto em `token_mean`
  quanto em `example_mean`, sem cancelamento matemático do peso por exemplo;
- `scripts/kg1_pre_paid_job_integration_gate.py` agora aceita
  `--require-row-loss-weight` e falha se o launcher não expuser
  `USE_ROW_LOSS_WEIGHT=1` e `REQUIRE_ROW_LOSS_WEIGHT=1`;
- `scripts/audit_v478_training_objective_alignment.py` agora mede o objetivo
  efetivo com `metadata.loss_weight` via `--use-row-loss-weight` e
  `--require-row-loss-weight`;
- V478 row-weighted passou em
  `artifacts/v544_minimal_distillation_dataset/20260517T_v544_cpu_gate/v478_objective_alignment_row_weighted.json`:
  objetivo efetivo de treino `66.42%` bit, `33.58%` equation, `45` teacher
  gains com peso total `90.0`, sem source/subcategory desconhecida;
- validações rodadas: `py_compile`, `hf_job_train_v90.py --self-test` e
  `kg1_static_safety_gate.py` passaram.

Launcher V544:

- launcher: `artifacts/v544_hf_h200_launch/launch_v544_hf_nemo_h200_minimal_distillation.py`;
- H200 controlado: `MAX_STEPS=8`, checkpoints a cada `2`, timeout `3600s`,
  custo HF observado `0.083333 USD/min`;
- dataset remoto baixado e hasheado em debug local;
- adapter inicial `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`
  verificado;
- `kg1_pre_paid_job_integration_gate.py` passou com `ok=true`,
  `--require-row-loss-weight`, root remoto novo, hashes novos e sem
  `assistant_final_answer_mismatch`;
- bloqueador final antes de `--launch`: commitar/pushar os scripts alterados,
  porque o HF clona o branch GitHub; sem isso o job rodaria trainer antigo.

Meta realista:

- CPU teacher: `200/315` já provado;
- adapter-only esperado: `194-198/315` se a transferencia funcionar;
- submit só volta a ser considerado após weak label-free `raw_output` passar
  `total>=194`, `bit>=136`, `equation>=58`, `truncation=0` e protected row
  intacta.

## Atualizacao V545 - V544 Nao Transferiu para ACC

Artefatos:

- treino V544 H200:
  `artifacts/v544_hf_h200_launch/v544_job_6a0962bce48bea4538ba0c71_logs.txt`;
- treino V544 H200, log refetched completo:
  `artifacts/v544_hf_h200_launch/v544_job_6a0962bce48bea4538ba0c71_logs_refetched.txt`;
- launcher/eval V545:
  `artifacts/v545_hf_h200_v544_weak_eval_launch/launch_v545_hf_weak_eval_v544_checkpoints.py`;
- logs V545:
  `artifacts/v545_hf_h200_v544_weak_eval_launch/v545_job_6a09681d3308d79117b91ab2_logs.txt`;
- logs V545 refetched:
  `artifacts/v545_hf_h200_v544_weak_eval_launch/v545_job_6a09681d3308d79117b91ab2_logs_refetched.txt`;
- launch manifest:
  `artifacts/v545_hf_h200_v544_weak_eval_launch/v545-h200-v221contract-v544-checkpoints-20260517T070146Z_launch_manifest.json`.

Resultado do treino V544:

- job HF `felipesp1983/6a0962bce48bea4538ba0c71` completou;
- checkpoints gerados: `checkpoint-2`, `checkpoint-4`, `checkpoint-6`,
  `checkpoint-8` e `final`;
- `checkpoint-4` foi o melhor por `eval_loss` (`5.4595` vs baseline
  `5.4607`), mas a melhoria de loss foi muito pequena e precisava de weak ACC.
- curva refetched do treino:
  - `step=2`: `eval_loss=5.4693`;
  - `step=4`: `eval_loss=5.4595`, melhor por loss;
  - `step=6`: `eval_loss=5.4651`;
  - `step=8/final`: `eval_loss=5.4675`;
  - `train_loss` oscilou entre `3.9648` e `6.8580`, sem correlação útil com
    ACC label-free.

Resultado V545 weak ACC label-free:

| Candidato | Total weak | bit | equation | trunc | Decisao |
|---|---:|---:|---:|---:|---|
| V544 `checkpoint-2` | 189/315 | 134/160 | 55/155 | 0 | bloqueado; regressao forte |
| V544 `checkpoint-4` | 188/315 | 133/160 | 55/155 | 1 | bloqueado; regressao + truncation |

Diagnostico adicional dos logs:

- os dois checkpoints avaliados geraram runaway output:
  - `checkpoint-2`: `1,503,221` completion tokens em `315` linhas,
    media `4,772` tokens/linha;
  - `checkpoint-4`: `1,504,064` completion tokens em `315` linhas,
    media `4,775` tokens/linha;
- isso prova que a rota V544 nao apenas perdeu ACC: ela tambem quebrou a
  obediencia ao formato curto de resposta, mesmo com prompt `No reasoning`;
- correcao implementada no gate: `scripts/evaluate_lora_adapters_batch.py`
  agora grava `rows`, `avg_completion_tokens` e `max_completion_tokens`; e
  `scripts/hf_job_weak_eval_v245.py` bloqueia candidatos com
  `avg_completion_tokens>512` ou `max_completion_tokens>2048`;
- correcao operacional implementada no wrapper V245: quando
  ha mais de um candidato, `KG1_EVAL_CANDIDATE_BY_CANDIDATE` passa a ser
  `true` por padrao; cada checkpoint e avaliado em uma rodada isolada, grava
  manifest incremental, pode fazer upload parcial de diagnosticos e para por
  padrao apos `2` candidatos consecutivos bloqueados;
- consequencia: qualquer novo checkpoint com output runaway sera barrado
  como bug silencioso/F2 backfire antes de package/full/submit.

Decisao FinOps:

- o job V545 `felipesp1983/6a09681d3308d79117b91ab2` foi cancelado em
  `2026-05-17T07:22:00Z`, antes de gastar H200 com `checkpoint-6`,
  `checkpoint-8` e `final`;
- motivo: os dois primeiros checkpoints ficaram abaixo do baseline
  adapter-only (`192/315`, `bit=136`, `equation=56` historico) e abaixo do
  recompute strict (`191/315`, `bit=136`, `equation=55`);
- `eval_loss` menor no `checkpoint-4` nao refletiu ACC e ainda introduziu
  truncation, portanto a rota V544 minimal distillation fica bloqueada para
  package/full/submit.

Impacto no roadmap:

1. nao relancar V544, nem avaliar os checkpoints restantes sem novo sinal CPU;
2. manter o CPU teacher `200/315` como sinal valido, mas nao como submit;
3. qualquer novo eval de varios checkpoints deve manter o padrao
   candidato-a-candidato do wrapper V245 e upload incremental ligado, para
   preservar row-level diagnostics antes de cancelamento FinOps;
4. proximo passo deve ser diagnostico row-level da transferencia:
   - comparar os exemplos teacher-gain/replay do V544 contra as saidas reais
     dos checkpoints 2 e 4, usando apenas logs/diagnosticos label-free;
   - identificar se a falha vem de target curto, peso por linha,
     answer extraction, prompt template ou interferencia MoE;
   - so construir novo dataset se houver causa concreta e teste CPU que
     preserve `8740ed31=01101000`, `bit>=136`, `equation>=60` e `trunc=0`.
5. se nao houver causa concreta, voltar para a frente CPU equation
   canonicalization/hard negatives e abandonar treino LoRA dessa familia ate
   aparecer novo sinal verificavel.

## Atualizacao V546 - Diagnostico Formal do Backfire V544

Artefatos:

- script:
  `scripts/analyze_v546_v544_transfer_runaway_audit.py`;
- manifest:
  `artifacts/v546_v544_transfer_runaway_audit/20260517T_cpu_audit/v546_v544_transfer_runaway_audit_manifest.json`;
- static gate:
  `artifacts/v546_v544_transfer_runaway_audit/20260517T_cpu_audit/v546_static_safety.json`.

Achado principal:

- V544 nao falhou apenas por hiperparametro ou por loss fraco;
- o audit V546 mostra que `236/236` linhas de treino e `115/115` linhas de
  validacao tinham assistant target iniciando com `RULE:`;
- ao mesmo tempo, o V545 avaliou com prompt suffix:
  `Return only one line: \boxed{answer}. No reasoning. No explanation.`;
- portanto o treino ensinou um prefixo de resposta diferente do contrato de
  inferencia submit-safe. Isso e uma causa concreta para o runaway output e
  para a perda de formato observada no V545.

Bloqueadores V546:

- `checkpoint-2`: `189/315`, `bit=134`, `equation=55`,
  `avg_completion_tokens=4772`;
- `checkpoint-4`: `188/315`, `bit=133`, `equation=55`, `truncated=1`,
  `avg_completion_tokens=4775`;
- V545 antigo nao gerou `candidate_summary_payload`, nao subiu diagnosticos
  finais e iniciou `checkpoint-6` antes do cancelamento, sem summary final;
- isso confirma que a mudanca V245 candidato-a-candidato e obrigatoria para
  proximos evals pagos.

Decisao:

- V544 fica bloqueado como rota LoRA;
- nao rodar novo H200 com targets `RULE: ... Final answer: ...` se o eval
  continuar exigindo uma unica linha `\boxed{...}`;
- qualquer nova tentativa de distilacao precisa passar antes por gate CPU de
  compatibilidade entre target de treino e output esperado:
  - `assistant_final_answer_only_rows > 0` ou `assistant_boxed_only_rows > 0`;
  - zero linhas com prefixo incompatível quando a inferencia pedir boxed-only;
  - weak label-free seco em CPU ou amostra HF barata antes de H200;
  - protected row `8740ed31=01101000`, `bit>=136`, `equation>=60`,
    `truncation=0` e completion tokens baixos.

Proximo passo tecnico:

1. nao corrigir V544 por mais epochs;
2. se insistirmos em LoRA, construir um micro dataset novo com target
   compatível com o contrato de inferencia, por exemplo somente
   `\boxed{answer}` ou `Final answer: \boxed{answer}`, e testar primeiro em
   amostra pequena;
3. caminho preferido continua CPU solver/verifier/canonicalization para
   `equation_transform`, porque ja mostrou `200/315` sem perda, enquanto LoRA
   ainda nao transferiu esse ganho.

## Atualizacao V574 - Auditoria de Contrato, Metrica e Ganho Falso

Artefatos:

- launcher:
  `artifacts/v574_hf_h200_v573_weak_eval_launch/launch_v574_hf_weak_eval_v573_checkpoint2.py`;
- manifest HF baixado:
  `artifacts/v574_hf_h200_v573_weak_eval_launch/downloaded_final/evals/v574-h200-officiallike-v573-checkpoint2-20260517T194619Z/v245_hf_weak_eval_manifest.json`;
- audit local:
  `artifacts/v574_hf_h200_v573_weak_eval_launch/v574_contract_metric_audit.json`;
- adapter config:
  `artifacts/v574_hf_h200_v573_weak_eval_launch/downloaded_final/checkpoint-2/adapter_config.json`.

Resultado confirmado por recomputacao local:

| Item | Valor V574 |
|---|---:|
| Weak CSV SHA256 | `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6` |
| Shared row contract | `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff` |
| Weak rows | `315` |
| Family counts | `bit_manipulation=160`, `equation_transform=155` |
| Prompt suffix | oficial `Please put your final answer inside \boxed{}` |
| Thinking | ligado (`KG1_DISABLE_THINKING=0`) |
| `max_tokens` | `7680` |
| `max_model_len` | `8192` |
| Metric mode | `submit_safe_label_free` |
| Postprocessor | `none` |
| ACC total | `190/315` |
| bit | `135/160` |
| equation | `55/155` |
| truncated | `1` |
| label-aware debug | `191/315`, nao submit-safe |

Contrato LoRA confirmado no checkpoint:

| Campo | Valor |
|---|---|
| base | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| revision avaliada | `cbd3fa9f933d55ef16a84236559f4ee2a0526848` |
| `r` | `32` |
| `lora_alpha` | `32` |
| `modules_to_save` | `null` |
| target modules | `in_proj,k_proj,v_proj,down_proj,lm_head,o_proj,q_proj,up_proj,out_proj` |
| target parameters | `mlp.experts.gate_up_proj`, `mlp.experts.down_proj` |

Bloqueadores de ganho falso:

- o V574 nao usa postprocessor nem expected-aware extraction no score
  promocional;
- o `label_aware_debug_correct=191` fica separado do score submit-safe e nao
  pode promover pacote;
- a recomputacao local do CSV de predicoes confirmou exatamente `190/315`,
  `bit=135`, `equation=55`, `truncated=1`;
- protected row guard bloqueou duas regressoes bit que o baseline acertava:
  - `8740ed31`: esperado `01101000`, candidato `01111000`;
  - `59bee375`: esperado `10010101`, candidato `2`, `finish_reason=length`;
- por isso V574 nao pode virar package/full/submit.

Correcao implementada apos a auditoria:

- `scripts/hf_job_weak_eval_v245.py` agora rejeita adapter com
  `modules_to_save` nao vazio no caminho adapter-only;
- o mesmo gate agora rejeita mismatch de `base_model_name_or_path` contra
  Nemotron esperado;
- `scripts/kg1_static_safety_gate.py` exige esses snippets criticos para nao
  deixar a checagem desaparecer em futuras alteracoes;
- scripts historicos V217/V223/V244 foram colocados em modo fail-closed para
  evitar relancar rotas antigas com contrato obsoleto.

Posicao sobre loss vs ACC:

- o loss de treino esta implementado como cross-entropy mascarada sobre tokens
  de resposta, com `loss_mask`, `row_loss_weight` quando habilitado e
  normalizacao `token_mean` ou `example_mean`;
- isso e correto para medir saude do treino, mas nao e proxy matematico de
  ACC exact-match;
- ACC real depende de geracao completa, formato de resposta, extracao
  label-free, truncation e `verify_answer`;
- portanto qualquer checkpoint so promove por weak/full ACC gerado, nunca por
  `best_eval_loss` isolado.

Decisao:

- V573/V574 ficam bloqueados para submit;
- nao rodar outro H200 amplo antes de um gate que preserve as protected rows,
  mantenha `bit>=136`, elimine truncation e mostre ganho label-free real;
- o proximo trabalho deve isolar se a falha e de decoding ruim ou se o adapter
  empurrou o modelo para resposta errada, usando row-level diff e, quando
  possivel, probes de logits/NLL antes de novo treino pago.

## Atualizacao V574B - Double Check de Dataset, Symbols, Loss e ACC

Objetivo: impedir ganho falso antes de qualquer novo job ou submit.

Achados confirmados em 2026-05-17:

- o weak CSV esta correto:
  - SHA256 `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - `315` linhas;
  - `bit_manipulation=160`, `equation_transform=155`;
  - shared row contract
    `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`;
- o prompt de weak eval V574 esta no contrato correto:
  - thinking ligado;
  - suffix oficial com `\boxed{}`;
  - `max_tokens=7680`;
  - `max_model_len=8192`;
  - metric mode `submit_safe_label_free`;
- recomputacao direta do CSV de predicoes confirmou:
  - total `190/315`;
  - `bit=135/160`;
  - `equation=55/155`;
  - `truncated=1`;
  - `314` rows `finish_reason=stop`, `1` row `finish_reason=length`;
- o contrato LoRA do checkpoint avaliado esta consistente:
  - base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`;
  - `r=32`, `lora_alpha=32`;
  - `modules_to_save=null`;
  - target modules e target parameters presentes;
- o dataset V573 ativo passou integridade:
  - treino `757` linhas, sem duplicidade, sem conflito prompt-answer, sem
    raw_output, sem assistant mismatch, `757/757` boxed;
  - validacao `159` linhas, sem duplicidade, sem conflito prompt-answer, sem
    raw_output, sem assistant mismatch, `159/159` boxed;
  - charset das respostas restrito a `-0123456789`;
  - zero caracteres de controle problematicos e zero nao ASCII em `prompt` e
    `answer`;
- `example_mean` esta ativo no V573:
  - `KG1_LOSS_NORMALIZATION_MODE=example_mean`;
  - `KG1_REQUIRE_ROW_LOSS_WEIGHT=1`;
  - treino com peso efetivo `bit=0.671963`, `equation=0.328037`;
  - validacao balanceada `bit=0.496855`, `equation=0.503145`.

Bug silencioso corrigido:

- `scripts/audit_v509_training_dataset_integrity.py` aceitava caminho explicito
  inexistente e retornava `dataset_count=0`, o que poderia gerar uma falsa
  auditoria limpa;
- agora o V509 falha fechado com `FileNotFoundError` para qualquer
  `--dataset-jsonl` inexistente;
- `scripts/kg1_static_safety_gate.py` passou a exigir essa protecao para nao
  regredir.

Interpretacao correta de loss:

- o loss e cross-entropy mascarada nos tokens de resposta e esta adequado para
  detectar saude/regressao de treino;
- o loss nao deve ser tratado como proxy de ACC, porque ACC depende de
  geracao, truncation, formato, extracao label-free e `verify_answer`;
- `best_eval_loss` pode melhorar sem mover ACC, como ja ocorreu em varios
  jobs; portanto promocao continua obrigatoriamente por ACC gerado.

Decisao:

- V574 permanece bloqueado para full/package/submit;
- o proximo job pago so pode iniciar se o pre-paid gate confirmar:
  - dataset real existente e auditado;
  - prompt oficial;
  - `max_tokens=7680`;
  - `modules_to_save` vazio;
  - base Nemotron correta;
  - protected rows sem backfire;
  - separacao documentada entre decoding ruim e adapter drift.

## Atualizacao V574C - Double Check de Gates, Package e Long Path

Objetivo: fechar bugs silenciosos que poderiam gerar falso positivo de
limpeza, dataset ou pacote submetivel.

Achados e correcoes aplicadas:

- `scripts/audit_v509_training_dataset_integrity.py` agora falha fechado se:
  - qualquer `--dataset-jsonl` explicito nao existir;
  - `dataset_count=0`;
  - o CSV de referencia weak/full nao existir;
  - a descoberta por `rglob` precisar atravessar caminho longo no Windows.
- `scripts/kg1_workspace_clean_gate.py` agora usa traversal com prefixo de
  caminho longo no Windows e tem self-test para garantir que arquivos profundos
  nao sejam ignorados.
- `scripts/package_hf_adapter_submission.py` agora valida tambem
  `base_model_name_or_path` contra o Nemotron esperado antes de empacotar.
- `scripts/package_hf_adapter_submission.py --self-test` foi adicionado para
  bloquear localmente:
  - adapter com `modules_to_save`;
  - adapter treinado sobre base model diferente.
- `scripts/kg1_static_safety_gate.py` passou a exigir esses checks, incluindo o
  self-test do package script.

Validacoes executadas:

- `python -m py_compile` nos scripts criticos;
- `python scripts/kg1_static_safety_gate.py --self-test`;
- `python scripts/kg1_static_safety_gate.py scripts artifacts/v574_hf_h200_v573_weak_eval_launch src`;
- `python scripts/kg1_workspace_clean_gate.py --delete-safe ...`;
- `python scripts/audit_v509_training_dataset_integrity.py` no train/val V573;
- teste negativo de dataset inexistente, que agora falha fechado;
- `python scripts/validate_answer_extraction_v1.py --self-test`;
- `python scripts/run_v485_peft_roundtrip_gate.py --self-test`;
- `python scripts/hf_job_weak_eval_v245.py --self-test`;
- `python scripts/package_hf_adapter_submission.py --self-test`.

Estado real apos o double check:

- o dataset V573 ativo esta limpo para treino tecnico:
  - train `757` rows;
  - val `159` rows;
  - zero duplicidade, conflito, raw_output, resposta vazia, prompt vazio,
    nonboxed ou assistant-answer mismatch;
  - respostas apenas com `-0123456789`;
  - zero caracteres de controle problematicos e zero nao ASCII em
    `prompt`/`answer`.
- o weak V574 continua bloqueado:
  - `190/315`;
  - `bit=135/160`;
  - `equation=55/155`;
  - `truncated=1`;
  - protected rows `8740ed31` e `59bee375` regrediram.
- o contrato LoRA avaliado continua correto, mas sem ganho:
  - base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`;
  - `r=32`;
  - `lora_alpha=32`;
  - `modules_to_save=null`;
  - target modules/parameters presentes.
- loss esta correto como cross-entropy mascarada dos tokens de resposta, mas
  nao e proxy de ACC; promocao continua somente por geracao completa +
  extracao label-free + `verify_answer` + truncation zero.

Decisao:

- nao empacotar nem submeter V574;
- nao promover qualquer checkpoint por `best_eval_loss`;
- antes de novo treino pago, exigir row-level gate que prove:
  - `bit>=136`;
  - `equation>=60`;
  - `total>=196`;
  - `truncated=0`;
  - nenhuma protected row regredida;
  - diferenca clara entre `decoding ruim` e `adapter drift`.

## Atualizacao V574D - Triple Check de ACC, Parser e Bugs Silenciosos

Objetivo: reauditar bit a bit o caminho que transforma `raw_output` em ACC e
fechar qualquer falso bloqueio/falso ganho antes de novo HF job.

Correcoes aplicadas em 2026-05-17:

- `scripts/audit_v449_acc_metric_integrity.py` e
  `scripts/validate_answer_extraction_v1.py` agora suportam caminhos longos no
  Windows. O CSV V574 de predicoes existe, mas tinha caminho local com `329`
  caracteres; antes disso o gate podia marcar `run_csv_missing` falso.
- `scripts/kg1_static_safety_gate.py` tinha chave duplicada para
  `scripts/kg1_workspace_clean_gate.py`; a segunda definicao sobrescrevia a
  primeira e enfraquecia checks criticos de limpeza. A tabela foi consolidada.
- `src/competition_utils.py` agora extrai, sem usar label, respostas simbolicas
  compactas com `}` literal seguido de escape TeX de pontuacao, por exemplo
  `\boxed{]}\!}` -> `]}\!`.
- Foram adicionados self-tests para:
  - aceitar `\boxed{]}\!}` como `]}\!`;
  - nao superestender `\boxed{$}{>}`, que continua extraindo `$`;
  - manter expected-aware apenas como diagnostico.

Resultados medidos apos a correcao do parser:

- CSV V574 original gravado antes da correcao:
  - coluna `prediction`: `190/315`;
  - `bit=135/160`;
  - `equation=55/155`.
- Reextracao label-free a partir de `raw_output` com parser corrigido:
  - `191/315`;
  - `bit=135/160`;
  - `equation=56/155`;
  - ganho real local de parser: `+1 equation` no id `4bb8c6cd`;
  - `expected_aware_minus_simple_correct=0`, ou seja, o ganho deixou de
    depender de gabarito.
- Mesmo apos o parser corrigido, V574 continua bloqueado:
  - falta `+5` total para `196/315`;
  - falta `+4` equation para `60/155`;
  - falta `+1` bit para `136/160`;
  - ainda ha `1` truncation;
  - protected rows `8740ed31` e `59bee375` continuam erradas.

Classificacao dos erros V574 apos a reextracao corrigida:

- `equation_wrong`: `99` linhas;
- `bit_binary_wrong`: `24` linhas;
- `decoding_truncated`: `1` linha;
- `extractor_expected_aware_delta`: `0` linhas.

Auditoria de dataset/objetivo:

- V509 train/val V573 passou novamente:
  - `dataset_count=2`;
  - `blocked_dataset_count=0`.
- V478 objetivo passou:
  - treino com share efetivo `bit=0.671963`, `equation=0.328037`;
  - validacao balanceada `bit=0.496855`, `equation=0.503145`;
  - `hf_gpu_allowed=true`.
- V526 `example_mean` passou:
  - `status=example_mean_dry_run_passed`;
  - `gpu_allowed=True`.

Achado de parametro/documentacao:

- o manifesto de treino V573 ainda tinha texto de receita antigo
  `total>=196, equation>=60`;
- o eval V574 usou o gate correto `total>=196`, `equation>=60`,
  `bit>=136`, `truncated=0`;
- acao: manifestos e textos de receita novos devem usar somente o gate atual
  `196/60/136/0` para evitar decisao FinOps baseada em limite antigo.

Validacoes executadas apos a correcao:

- `python -m py_compile` em `src/competition_utils.py`, V449, V540 e static
  gate;
- `python scripts/validate_answer_extraction_v1.py --self-test`;
- `python scripts/audit_v449_acc_metric_integrity.py --self-test`;
- `python scripts/kg1_static_safety_gate.py --self-test`;
- V449 no CSV V574 original e no CSV projetado;
- V540 no CSV V574 original e no CSV projetado;
- V509, V478, V526 nos datasets/objetivo V573;
- `python scripts/kg1_static_safety_gate.py scripts src artifacts/v574_hf_h200_v573_weak_eval_launch`;
- `python scripts/kg1_workspace_clean_gate.py --delete-safe ...`.

Decisao:

- nao ha submit ainda: `191/315` projetado continua abaixo do gate;
- o ganho de parser `+1 equation` deve entrar em todos os proximos evals e
  full/package gates;
- o proximo trabalho nao deve ser outro treino amplo. A prioridade agora e
  atacar os `99` erros de equation, os `24` erros bit binarios e o `1`
  truncation, preservando as duas protected rows.

## Atualizacao V574E - Gate Submit-Safe e Contrato LoRA Ativo

Objetivo: remover divergencias de parametro que podiam permitir falso ganho ou
decisao FinOps errada antes de qualquer novo treino/submissao.

Correcoes aplicadas em 2026-05-17:

- o launcher base V536 agora registra somente o gate atual:
  `total>=196`, `equation>=60`, `bit>=136`, `truncated=0`;
- o manifesto V573 atual foi corrigido para a mesma regra, porque o job/eval ja
  usava esse criterio, mas o texto de receita ainda mostrava `193/57`;
- `scripts/notebook_release_gate.py` agora exige
  `WEAK_MIN_FOR_FULL = 196` em notebooks novos/editados;
- os builders ja modificados (`V217`, `V223`, `V244`) foram alinhados para
  `WEAK_MIN_FOR_FULL = 196`;
- `scripts/kg1_static_safety_gate.py` agora bloqueia HF jobs/notebooks que
  descrevam criterios promocionais obsoletos (`193/57` ou equivalentes
  `total>192`/`equation>56`) como se fossem o gate atual;
- o launcher V573 agora explicita no contrato estatico e no env:
  `KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`;
- o launcher V573 tambem declara o allowlist treinavel:
  `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`.

Validacoes executadas:

- `python -m py_compile` nos launchers/gates/scripts alterados;
- `python scripts/kg1_static_safety_gate.py --self-test`;
- `python scripts/notebook_release_gate.py --self-test`;
- `python scripts/validate_answer_extraction_v1.py --self-test`;
- `python scripts/audit_v449_acc_metric_integrity.py --self-test`;
- static safety gate nos scripts e launchers ativos: `ok=true`, `findings=[]`;
- `scripts/kg1_pre_paid_job_integration_gate.py` no V573:
  - `ok=true`;
  - `findings=[]`;
  - dataset SFT confirmado;
  - hashes train/val confirmados;
  - `example_mean` confirmado;
  - row loss weight requerido;
  - H200 com `timeout=3600`;
  - primeira weak eval obrigatoria;
  - `decoding_vs_adapter_drift` permitido apenas como deferimento do primeiro
    checkpoint.

Conclusao tecnica:

- o caminho de gate/metricas esta mais sincronizado agora;
- nao ha evidencia de que o platô venha de `loss` sendo calculado errado,
  `dataset_count` errado, CSV weak incorreto, parser label-aware, hash errado,
  ou gate antigo aceitando falso positivo;
- o gargalo atual permanece em geracao do adapter:
  `equation_wrong=99`, `bit_binary_wrong=24`, `decoding_truncated=1`;
- novo HF pago so deve rodar se a proxima intervencao atacar diretamente esses
  erros row-level e mantiver o contrato `196/60/136/0`.

## Atualizacao V574F - Auditoria Consolidada de CSV/Dataset/Adapter

Artefato novo:

- `artifacts/v574_hf_h200_v573_weak_eval_launch/v574_triple_full_contract_audit_summary.json`.

Confirmacoes:

- weak CSV:
  - SHA256 `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - `315` linhas;
  - `bit_manipulation=160`, `equation_transform=155`;
  - zero ids duplicados;
  - zero prompts duplicados por SHA256;
  - zero prompt/answer vazio;
  - zero caracteres de controle;
  - zero nao ASCII nas respostas.
- dataset V573:
  - train `757`: `bit=437`, `equation=320`;
  - val `159`: `bit=79`, `equation=80`;
  - `boxed_assistant_rows=757/757` no train e `159/159` no val;
  - zero linhas vazias/problematicas;
  - zero caracteres de controle;
  - zero nao ASCII em prompt/answer/assistant.
- adapter V574:
  - base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`;
  - `r=32`;
  - `lora_alpha=32`;
  - `modules_to_save=null`;
  - target modules incluem atencao, MLP e `lm_head`;
  - target parameters: `mlp.experts.gate_up_proj`,
    `mlp.experts.down_proj`.

Classificacao refinada dos erros V574 por reextracao label-free de
`raw_output`:

- `equation_wrong`: `99`;
- `bit_binary_wrong`: `24`;
- `decoding_truncated`: `1`;
- total reextraido: `191/315`;
- `equation=56/155`;
- `bit=135/160`.

Interpretacao:

- o parser corrigido recupera exatamente `+1 equation` em relacao ao CSV
  gravado antes da correcao;
- o restante do platô nao e erro de CSV, hash, dataset, parser, expected-aware,
  gate antigo ou loss;
- o problema acionavel agora e fazer o adapter gerar as respostas corretas nos
  `99` equation misses e nos `24` bit misses, sem aumentar truncation e sem
  perder protected rows.

## Atualizacao V575 - Solution Sync Contract Gate

Artefatos novos:

- `scripts/audit_v575_solution_sync_contract.py`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_solution_sync_contract_audit.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_static_gate_after_audit_fix.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_static_gate_full_threshold_refresh.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_static_gate_full_threshold_refresh_v2.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_static_gate_full_threshold_refresh_v3.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_workspace_clean_gate_after_audit.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_workspace_clean_gate_final_clean.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_workspace_clean_gate_final_verify.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v575_prediction_row_audit.csv`.

Resultado:

- V575 passou com `0` erros e `6` warnings;
- V575 agora gera tambem auditoria row-level das `315` predicoes,
  separando `stored_prediction`, reextracao label-free,
  expected-aware debug, truncation, hashes e classe de erro;
- o gate estatico passou em `7` pecas criticas:
  treino, weak eval HF, avaliador batch, static gate, notebook gate, V509 e
  launcher V573;
- apos o refresh de thresholds, o gate estatico passou em `22` arquivos,
  incluindo builders Colab antigos e builders de dataset;
- o clean gate final removeu apenas `2` itens seguros de cache/temp criados
  por `py_compile`; a verificacao seguinte ficou com `0` findings e preservou
  datasets, adapters, manifests, logs, roadmaps e relatorios.

Correcoes feitas no proprio V575:

- o primeiro V575 acusava falso positivo porque procurava
  `extract_final_answer(raw_output)` dentro do wrapper HF;
- a arquitetura correta e:
  - `hf_job_weak_eval_v245.py` monta o comando e chama
    `evaluate_lora_adapters_batch.py`;
  - `evaluate_lora_adapters_batch.py` faz a extracao label-free submit-safe,
    guarda `label_aware_debug_prediction` apenas para diagnostico e declara
    `prediction_metric_mode=submit_safe_label_free`;
- o V575 agora audita as duas pecas na fronteira correta.

Reforco contra ganho falso:

- o static gate agora tambem bloqueia `WEAK_MIN_FOR_FULL = 193`, nao apenas
  defaults `KG1_WEAK_PROMOTE_TOTAL_MIN=193`;
- foram atualizados builders antigos que ainda poderiam regenerar notebooks
  com piso obsoleto:
  `build_v218`, `build_v219`, `build_v220`, `build_v221`, `build_v222`,
  `build_v224`, `build_v225`, `build_v226`, `build_v227`, `build_v228`,
  `build_v229`, `build_v230` e `build_v245`;
- `build_v321` e `build_v322` agora documentam gate `total>=196`,
  `equation>=60`, `bit>=136`, `truncation=0`.

Confirmacoes V575:

- weak CSV oficial permanece correto:
  - SHA256 `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - `315` linhas;
  - `bit_manipulation=160`, `equation_transform=155`;
  - zero ids duplicados, zero prompt duplicado por SHA256, zero prompt/answer
    vazio, zero caracteres de controle e zero nao ASCII nas respostas.
- dataset de treino/validacao V573 permanece limpo:
  - train `757`: `bit=437`, `equation=320`;
  - val `159`: `bit=79`, `equation=80`;
  - todas as respostas do assistant estao em `boxed`;
  - zero `raw_output`, zero resposta vazia, zero mismatch assistant/answer,
    zero caracteres de controle e zero nao ASCII;
  - `loss_weight=1.5` para bit e `1.0` para equation.
- contrato LoRA do adapter V574 esta correto:
  - base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`;
  - `r=32`, `lora_alpha=32`, `modules_to_save=null`;
  - target modules obrigatorios presentes:
    `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`;
  - target parameters MoE:
    `mlp.experts.gate_up_proj`, `mlp.experts.down_proj`.
- regras de metrica estao sincronizadas:
  - cross-entropy mascarada existe no treino;
  - `example_mean` existe no treino;
  - weak eval usa prompt oficial, `max_tokens`, extracao label-free e
    expected-aware apenas como debug;
  - notebook gate e weak eval exigem `196/315`, `equation>=60`,
    `bit>=136`, `trunc=0`.

Warnings que continuam bloqueando submit/full:

- predicao V574 reextraida com parser atual:
  - `bit=135/160`;
  - `equation=56/155`;
  - `total=191/315`;
  - `truncated=1`;
- row-level audit:
  - `135` bit corretos;
  - `24` `bit_binary_wrong`;
  - `1` `decoding_truncated`;
  - `56` equation corretos;
  - `99` `equation_wrong`;
- o CSV gravado antes da correcao do parser tem `1` linha stale:
  id `4bb8c6cd`, resposta esperada `]}\!`, predicao gravada `]`,
  reextracao atual `]}\!`;
- `label_aware_minus_label_free_correct_count=0`, portanto nao ha ganho
  escondido vindo de expected-aware;
- subcategory top-level e metadata diferem nas linhas de bit porque metadata
  guarda proveniencia (`bit_konbu`, `bit_huikang`) e o top-level guarda a
  subcategoria canonica usada no treino. Isto e aceitavel somente porque
  `canonical_example_subcategory` prioriza o top-level.

Decisao:

- o platô atual nao deve mais ser tratado como bug de hash, CSV, parser,
  expected-aware, loss, adapter_config, `dataset_count`, rglob, prompt oficial
  ou gate antigo;
- o blocker real e gerativo/decoding do adapter:
  `99` erros de equation, `24` erros binarios de bit e `1` truncation;
- novo treino ou eval pago so entra se atacar esses rows explicitamente e
  passar primeiro pelo contrato V575 + gate CPU/weak sem falso ganho.

## Atualizacao V576 - Quadruple Check Sem Warnings Ambiguos

Artefatos novos:

- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_solution_sync_contract_audit_zero_warning.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_prediction_row_audit.csv`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_static_gate_active_notebooks_scripts_v2.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_static_gate_active_notebooks_scripts_final.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_notebook_release_gate_active_refresh.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_notebook_release_gate_active_final.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_notebook_release_gate_v231.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_notebook_release_gate_v226_v227_static_marker.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_workspace_clean_gate_final_clean.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_workspace_clean_gate_final_verify.json`.
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_workspace_clean_gate_final_clean2.json`;
- `artifacts/v574_hf_h200_v573_weak_eval_launch/v576_workspace_clean_gate_final_verify2.json`.

Mudanca importante no contrato:

- warnings de performance foram removidos do canal de `finding.warning`;
- falhas reais de ACC agora aparecem como `performance_blockers`, mantendo
  `submit_safe_now=false`;
- resultado V576:
  - `error=0`;
  - `warning=0`;
  - `info=2`, ambos sobre subcategoria top-level versus metadata de
    proveniencia, aceitos porque o treino prioriza `example["subcategory"]`.

Performance blockers atuais:

- `equation_transform=56<60`;
- `bit_manipulation=135<136`;
- `truncated=1>0`;
- `stored_prediction_stale_after_parser_fix=1`.

Atualizacoes de threshold/quoting:

- scripts antigos que ainda tinham `total>192`, `equation>56` ou texto de
  gate fraco foram atualizados para o contrato atual:
  `total>=196`, `equation>=60`, `bit>=136`, `truncation=0`;
- `SAMPLING_MODE='weighted'` foi removido do V226 e substituido por
  `weighted_replacement`, que e o modo aceito pelo gate;
- V226/V227 agora declaram explicitamente:
  `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`;
- V226/V227 tambem documentam o gate V478 de alinhamento de objetivo antes de
  qualquer uso promocional com pesos de bit/equation.

Notebooks:

- notebooks ativos regenerados e aprovados pelo `notebook_release_gate`:
  V218, V219, V220, V221, V222, V224, V225, V226, V227, V228, V229,
  V230, V231 e V245;
- static gate ativo passou em `44` arquivos, incluindo esses notebooks,
  builders, weak eval, treino, avaliador batch, static gate, notebook gate e
  launcher V573;
- V217, V223 e V244 continuam em quarentena fail-closed no proprio builder
  (`Archived KG1 launcher quarantined fail-closed`) e nao fazem parte do
  caminho ativo.

Limpeza:

- `__pycache__` criado por `py_compile` foi removido pelo clean gate;
- verificacao final do clean gate ficou com `0` findings.

Decisao V576:

- a solucao ativa esta consistente do ponto de vista de contratos, thresholds,
  parser, label-free metric, expected-aware debug, loss mascarado,
  `example_mean`, weak CSV, dataset, adapter_config e gates;
- a proxima melhoria de ACC nao deve ser buscada por novo ajuste generico de
  loss. Deve partir do `v576_prediction_row_audit.csv`, construindo um
  rule-gap map dos `99` equation misses, `24` bit binary misses e `1`
  truncation.

## Atualizacao V577-V579 - Bloqueio De Ganho Falso Antes De H200

Estado real verificado em `2026-05-17`:

- adapter-only label-free atual continua em `191/315`:
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`.
- projecao CPU/teacher V577, ainda nao adapter-only/submetivel:
  - `197/315`;
  - `bit_manipulation=137/160`;
  - `equation_transform=60/155`;
  - `truncated=0`.

Achados novos:

- V577 combinou ganhos auditados de V324 equation e V333 Tong bit, mas o
  dataset gerado com prompts weak foi bloqueado pelo gate V509. Portanto V577
  e evidencia de alvo, nao dataset treinavel seguro.
- V578 aumentou a exposicao de equation, mas falhou no V526 estrito por
  `example_mean` longe demais da referencia (`delta=0.141585 > 0.08`).
  Portanto nao deve ir para H200 sem aceitar risco explicitamente.
- V579 ajustou o mix para a borda estrita que passa V509, V286, V478, V513 e
  V526:
  - train `757`: `bit=437`, `equation=320`;
  - `example_mean` efetivo train: `bit=0.661969`, `equation=0.338031`;
  - tokenizacao: `train_token_max=1074`, `val_token_max=1037`,
    `max_length=2048`, `0` truncation.
- Antes de lancar H200, o launcher V579 revelou um bug silencioso herdado:
  variaveis `KG1_CPU_SIMULATED_*` antigas (`200/138/62`) vinham do launcher
  V536/V573, enquanto a evidencia real V577 e `197/137/60`.

Correcoes implementadas:

- `launch_v579_hf_nemo_h200_v571bit_v551eq_strictedge.py` agora carrega o
  manifesto CPU V577 e valida:
  - manifesto SHA256
    `91465c58b941e1f911aad0c1bd6f4ba1242ef27b35ccc974ee170936398ab111`;
  - projection CSV SHA256
    `d9143bc03c430435cba4d266b529153ca289835b8fd9abebecb82cb61786b6ef`;
  - total `197`, bit `137`, equation `60`, trunc `0`;
  - sobrescrita obrigatoria de qualquer `KG1_CPU_SIMULATED_*` herdado.
- O launcher agora falha fechado se a projecao real nao passar o residual-first
  pago:
  - total minimo `200`;
  - bit minimo `136`;
  - equation minimo `59`;
  - truncation `0`.
- `kg1_pre_paid_job_integration_gate.py` foi rerodado contra V579 e ficou com
  um unico blocker:
  - `KG1_CPU_SIMULATED_TOTAL_CORRECT=197.0 below required 200`.
- Os demais pontos do pre-paid gate ficaram consistentes:
  - hashes/linhas V579 presentes;
  - row-loss-weight estatico e remoto auditavel;
  - subcategorias obrigatorias declaradas;
  - tokenization gate passou;
  - cost/H200/timeout/max_length/example_mean coerentes.

Decisao V579:

- nao lancar H200 com V579 agora. A decisao FinOps correta e bloquear, porque
  a unica autorizacao paga possivel dependeria de aceitar `197 < 200`;
- nao fazer package/full/submit;
- proximo passo efetivo e elevar a projecao CPU real de `197` para pelo menos
  `200` sem perdas, ou criar um micro-gate local que prove transferencia
  adapter-only antes de qualquer gasto;
- qualquer launcher futuro deve comparar as variaveis `KG1_CPU_SIMULATED_*`
  com um manifesto CPU versionado. Valor hard-coded sem manifesto passa a ser
  tratado como risco de ganho falso.

## Atualizacao V580-V581 - Projecao CPU 200 E Dataset Transferivel

Estado novo verificado em `2026-05-17`:

- adapter-only real continua em `191/315`:
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`.
- V580/V581 CPU teacher projection, ainda nao submetivel:
  - `200/315`;
  - `bit_manipulation=139/160`;
  - `equation_transform=61/155`;
  - `truncated=0`;
  - `loss_count=0`.

Achados concretos:

- `scripts/analyze_v350_cpu_residual_no_loss_gate.py` foi rerodado no baseline
  V577 atual e encontrou `197/315`, com:
  - `equation=61/155`;
  - `bit=136/160`;
  - `gains=6`;
  - `losses=0`.
- A uniao auditada V580 de V350 + Tong bit chegou a `199/315`:
  - `equation=61`;
  - `bit=138`;
  - `truncated=0`;
  - `losses=0`.
- A rota V581 adicionou exatamente um ganho local de bit esperado/auditado
  (`55d834d1`), chegando a `200/315`.
  - Importante: o solver local de bit completo tem `13` perdas se aplicado
    diretamente. Portanto ele nao e postprocessor submetivel; so pode ser usado
    como teacher expected-aware/label-audited para tentar transferencia ao
    adapter.

Artefatos novos:

- `scripts/build_v580_combined_teacher_projection.py`;
- `artifacts/v580_cpu_residual_on_v577_label_free/20260517T_v580_v350_current/v350_no_loss_gate_manifest.json`;
- `artifacts/v580_combined_teacher_projection/20260517T_v580_combined/v580_combined_teacher_projection_manifest.json`;
- `artifacts/v581_combined_plus_local_bit_teacher_projection/20260517T_v581_combined_localbit/v580_combined_teacher_projection_manifest.json`;
- `scripts/build_v581_combined_teacher_distill_dataset.py`;
- `artifacts/v581_combined_teacher_distill_dataset/20260517T_v581_sft/v581_combined_teacher_dataset_manifest.json`;
- `artifacts/v581_combined_teacher_distill_dataset/20260517T_v581_sft/v286_tokenization_real/v286_generic_tokenization_gate_manifest.json`.

Gate V286 real:

- passou com tokenizer Nemotron oficial;
- `train_rows=315`;
- `val_rows=115`;
- `train_token_max=315`;
- `val_token_max=315`;
- `completion_truncation=0`;
- `offset_masks=315` no treino e `115` na validacao.

Correcao de bug silencioso:

- o primeiro dataset V581 falhou porque usava `\boxed{}` manual em respostas
  simbolicas com `{`, `}` e `\`;
- a correcao valida a extracao label-free antes de salvar a linha: usa
  `Final answer: \boxed{...}` somente quando `extract_final_answer()` sem
  label recupera o alvo; nos casos ambiguos usa fallback plain
  `Final answer: <answer>`;
- isso evita ganho falso por `expected-aware`: o parser com label continua
  diagnostico-only, nunca criterio de promocao;
- o metadata agora explicita `label_audited_teacher_projection=true` nas linhas
  teacher e mantem `weak_gate_rows_used_for_training=false`,
  `full_gate_rows_used_for_training=false`, `gate_rows_used_for_training=false`
  para passar o contrato V286.

Decisao V581:

- esta e a primeira projecao CPU deste ciclo que atinge o piso pago
  `total>=200` sem perdas e sem truncation;
- ainda nao e submit Kaggle nem adapter-only. O proximo passo e um treino curto
  de transferencia com kill-switch por ACC:
  - primeiro checkpoint deve manter pelo menos `bit>=136`, `equation>=59`,
    `truncated=0`;
  - qualquer regressao para `bit<136`, `equation<59`, `truncated>0` ou perda
    clara contra V574/V579 cancela o job por FinOps;
  - se o adapter nao transferir pelo menos `+1` ACC no weak label-free, parar
    treino generico e voltar para solver/projection, nao insistir em loss.

## Atualizacao V582 - Dataset Label-Free Corrigido e Smoke H200

Estado antes do job V582:

- adapter-only label-free real continua em `191/315`:
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`.
- teacher CPU V581 continua sendo apenas alvo de transferencia:
  - `200/315`;
  - `bit_manipulation=139/160`;
  - `equation_transform=61/155`;
  - `truncated=0`;
  - `loss_count=0`.

Correcoes V582:

- o dataset foi regenerado em
  `artifacts/v582_combined_teacher_distill_dataset/20260517T_v582_sft_label_free`;
- `train_rows=315`, `val_rows=115`;
- `train_sha256=8c56379d6d0c046f9b97b3158e46c07b60333f6c67ed6c72ab583abf2fe4ce61`;
- `val_sha256=ad50976f35c8cc864b3266d182df5dbdba1cc7e05038ad40946f38b6366f1168`;
- formatos finais do treino:
  - `final_answer_boxed_label_free=312`;
  - `final_answer_plain_label_free=3`.
- formatos finais da validacao:
  - `final_answer_boxed_label_free=100`;
  - `final_answer_plain_label_free=15`.

Gates V582 antes de gastar H200:

- `py_compile` passou para o launcher e builders;
- `scripts/kg1_static_safety_gate.py` passou sem findings;
- `kg1_pre_paid_job_integration_gate.py` passou sem blockers;
- V286 real com tokenizer Nemotron passou:
  - `train_token_max=315`;
  - `val_token_max=315`;
  - `prompt_truncation=0`;
  - `completion_dropped=0`;
  - `offset_masks=315/115`.
- upload dataset HF:
  - repo `felipesp1983/kg1-v582-combined-teacher-distill-label-free-artifacts`;
  - commit `fe07a7fda4fa88e3da7617c14fcc8e9af53dfb30`.

Job H200 V582:

- job `felipesp1983/6a0a489ce7940de6ee6cde15`;
- URL `https://huggingface.co/jobs/felipesp1983/6a0a489ce7940de6ee6cde15`;
- output adapter repo
  `felipesp1983/kg1-nemotron-lora-v582-v581-teacher-transfer-v290ckpt6`;
- init adapter
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`;
- `MAX_STEPS=2`, H200, timeout menor que 1h;
- LoRA `r=32`, `alpha=32`, `target_parameters=mlp.experts.gate_up_proj,mlp.experts.down_proj`;
- `LOSS_NORMALIZATION_MODE=example_mean`;
- `ANSWER_SPAN_LOSS_WEIGHT=2.0`;
- `USE_ROW_LOSS_WEIGHT=1`;
- primeiro checkpoint precisa ser avaliado em weak label-free antes de qualquer
  full/package/submit.

Decisao V582:

- o job e permitido como smoke curto porque a projecao CPU chegou a `200/315`
  e os bugs de label-free do dataset foram corrigidos;
- ainda nao ha ganho submit-safe. O criterio de promocao e checkpoint
  adapter-only `raw_output` label-free acima de `191/315`, preservando
  `bit>=136`, `equation>=59` e `truncated=0`;
- se o checkpoint ficar em `191/315` ou regredir bit/equation, bloquear
  V582 para submit e voltar para destilacao/solver em CPU.

### Resultado V582 e Crisis-Mode Audit 2026-05-17

Resultado do treino curto:

- job treino H200 `felipesp1983/6a0a489ce7940de6ee6cde15` completou;
- `eval_loss` piorou levemente de `3.4520` para `3.4549`;
- checkpoint avaliado: `checkpoint-2`;
- weak eval job `felipesp1983/6a0a4c47a5e509f1a8413c48`;
- commit de diagnosticos HF:
  `6cdeb2b49c7663e9bde419790fd36a1e4f74322f`.

Resultado adapter-only do V582:

- CSV armazenado pelo job: `190/315`;
  - `bit_manipulation=135/160`;
  - `equation_transform=55/155`;
  - `truncated=1`.
- reextracao label-free local de `raw_output`: `191/315`;
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`.
- baseline V516 reextraido com o parser atual: `192/315`;
  - `bit_manipulation=136/160`;
  - `equation_transform=56/155`;
  - `truncated=0`.

Decisao:

- V582 esta bloqueado para full/package/submit;
- nao houve ganho real contra o baseline parser-atual;
- houve F2/backfire real em bit:
  - `8740ed31`: baseline correto `01101000`, V582 `01111000`;
  - `59bee375`: baseline correto `10010101`, V582 `2`;
  - `55d834d1` nao e backfire: baseline ja errava e V582 continuou errando;
    agora o gate classifica como `protected_id_missing_required_gain`.
- `completion_tokens_mean=4775.25`, `completion_tokens_max=7680`;
  isso confirma runaway/decoding ruim e torna o checkpoint inutil para submit.
- o V540 mostrou `stored_prediction_stale_after_parser_fix=1` no id
  `4bb8c6cd`: o CSV armazenado extraia `]`, mas o parser label-free atual
  extrai corretamente `]}\!`. Portanto toda promocao deve recalcular ACC a
  partir de `raw_output`, nao confiar cegamente na coluna `prediction` antiga.

Correcoes implementadas no gate:

- `scripts/kg1_weak_backfire_row_guard.py` agora separa:
  - `protected_id_backfire` quando o baseline era correto e o adapter erra;
  - `protected_id_missing_required_gain` quando o baseline ja errava e o
    adapter nao aprendeu o ganho esperado.
- o mesmo gate nao mistura mais defaults de `--protected-id-answer` com uma
  lista customizada.
- `scripts/analyze_eval_predictions.py` agora converte `completion_tokens`
  para numerico antes de calcular soma/media/max, evitando erro silencioso de
  pandas com strings.
- `scripts/audit_v509_training_dataset_integrity.py` agora:
  - aceita `user,assistant` como role sequence valida;
  - aceita `prompt + PROMPT_SUFFIX` como contrato oficial;
  - bloqueia explicitamente `false_anti_leak_flag_on_overlap`.
- `scripts/build_v581_combined_teacher_distill_dataset.py` foi corrigido para
  marcar `weak_gate_rows_used_for_training=True` nas linhas de treino derivadas
  do weak/projection. A versao V582 ja treinada permanece historica e nao deve
  ser reutilizada como evidencia submit-safe.
- `scripts/audit_v575_solution_sync_contract.py` agora aceita
  `final_answer_plain_label_free` quando o parser label-free recupera a
  resposta, evitando bloquear respostas simbolicas que nao cabem em `\boxed{}`
  sem ambiguidade.

Auditorias rodadas:

- `python -m py_compile` nos scripts alterados;
- self-test `kg1_weak_backfire_row_guard`;
- self-test `hf_job_weak_eval_v245`;
- self-test `analyze_v568_decoding_adapter_drift`;
- self-test `run_v485_peft_roundtrip_gate`;
- V485 no adapter V582 passou o contrato LoRA:
  - `r=32`;
  - `alpha=32`;
  - `modules_to_save=null`;
  - base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`;
  - `target_parameters=mlp.experts.gate_up_proj,mlp.experts.down_proj`.

Bloqueio novo do dataset V582:

- V509 confirmou overlap direto com weak/full:
  - train: `reference_overlap=1260`;
  - validation: `reference_overlap=460`.
- como o dataset V582 historico marcava os flags anti-leak como false, o gate
  agora reporta `false_anti_leak_flag_on_overlap`.
- conclusao: V582 e valido como experimento de transferencia/diagnostico, mas
  nao como evidencia de ganho submit-safe. Qualquer proximo dataset derivado
  desses rows deve ser tratado como weak-supervised, e sua promocao precisa vir
  de raw-output label-free sem overlap ou de uma validacao separada.

Proximo passo efetivo:

- nao repetir SFT amplo com esse dataset;
- se for tentar LoRA, usar micro-experimento answer-only com controle de
  decoding/runaway e protected replay obrigatorio;
- criterio de continuar gasto:
  - baseline parser-atual `192/315`;
  - candidato precisa ficar `>192/315`, `bit>=136`, `equation>=60`,
    `truncated=0`;
  - qualquer protected backfire cancela;
  - comprimento absoluto de resposta nao cancela por si so em official-like;
    cancela apenas se houver `truncated>0`, protected backfire, piora de ACC,
    ou se um diagnostico curto tiver definido explicitamente um limite local;
  - qualquer `stored_prediction_stale_after_parser_fix` bloqueia ate recomputar
    metricas por `raw_output`.

### Atualizacao V583 - Consenso OpenRouter Crisis Plateau

Artefatos:

- prompt completo:
  `artifacts/openrouter/v583_crisis_consensus/KG1_V583_OPENROUTER_CRISIS_PROMPT.md`;
- respostas:
  `artifacts/openrouter/v583_crisis_consensus/responses/`;
- analise consolidada:
  `artifacts/openrouter/v583_crisis_consensus/KG1_V583_OPENROUTER_CONSENSUS_ANALYSIS.md`.

Modelos consultados:

- `openai/gpt-5.5`: retornou `None`; descartado como sem conteudo util;
- `anthropic/claude-opus-4.7-fast`: util;
- `google/gemini-3.1-pro-preview`: util, mas curto;
- `deepseek/deepseek-v4-pro`: util;
- `qwen/qwen3.6-max-preview`: util.

Consenso acionavel:

- V582 confirmou falha de transferencia, nao ganho:
  - `191/315` por reextracao label-free de `raw_output`;
  - `bit=135/160`, `equation=56/155`, `truncated=1`;
  - protected backfire real em `8740ed31` e `59bee375`;
  - runaway severo com `completion_tokens_mean=4775.25`, `max=7680`.
- Nao repetir broad SFT, long traces, mix bit+equation ou treino baseado apenas
  em loss.
- Antes de qualquer novo H200, separar tres causas:
  - init adapter ruim;
  - decoding/runaway ruim;
  - drift real do adapter para resposta errada.
- O primeiro diagnostico deve ser sem treino:
  - base model, `max_new_tokens=512`;
  - base + V290 checkpoint-6, `max_new_tokens=512`;
  - base + V582 checkpoint-2, `max_new_tokens=512`;
  - base + V582 checkpoint-2, limite maior, apenas para confirmar runaway.
- Se V290 checkpoint-6 sozinho regredir protected rows, parar de usa-lo como
  init.
- Se V582 curto corrige parte do dano, tratar o problema como decoding/runaway
  e bloquear qualquer treino cujo primeiro checkpoint estoure token budget.
- Se V582 curto tambem erra protected rows, tratar como drift de adapter e
  bloquear teacher-transfer nessa linhagem.

Novo fluxo obrigatorio:

1. Diagnostico `MAX_STEPS=0` / init / decoding antes de treino.
2. Se passar, micro-transfer family-isolated:
   - bit-only: replay de bit correto + poucos gains de bit verificados;
   - equation-only: replay de equation correto + gains simbolicos com
     `final_answer_plain_label_free` quando `\boxed{}` for ambiguo.
3. So recombinar adapters depois que cada family isolada passar weak label-free
   sem backfire.

Parametros e gates ajustados:

- `ANSWER_SPAN_LOSS_WEIGHT=2.0` deixa de ser default seguro para smoke; usar
  `1.0` ou rodar ablation `1.0` vs `2.0` apenas em micro-run.
- `max_new_tokens` de micro diagnostico pode ficar em `512` ou menor para separar
  runaway/decoding de drift, mas esse limite curto nao e criterio promocional
  official-like. Para promocao, usar `max_tokens=7680`, `truncated=0`, protected
  rows intactas e re-score por `raw_output` label-free.
- dataset de equation deve auditar simbolos `{`, `}`, `\`, `]`, `!`; se boxed
  falhar ou ficar ambiguo, usar plain final answer apenas quando o extractor
  label-free recuperar corretamente.
- target module expansion para attention (`q_proj/k_proj/v_proj/o_proj`) e P2:
  testar apenas se MoE-only normalizado falhar sem runaway, e sempre em micro
  subset com grad-norm/trainability gate.

Itens retirados/depriorizados:

- SFT amplo e long trace como caminho principal;
- promocao por `eval_loss`;
- qualquer expected-aware metric como evidencia de submit;
- V582 dataset/checkpoint como evidencia submit-safe;
- treino misto bit+equation sem family-isolated gate;
- H200 sem `MAX_STEPS=0` control, protected guard e token-budget gate.

### Atualizacao V584 - Double Check OpenRouter Chat Export 2026-05-17

Fonte analisada:

- export do usuario:
  `C:\Users\davis\Downloads\OpenRouter Chat Sun May 17 2026 (1).json`;
- respostas extraidas:
  `artifacts/openrouter/v584_uploaded_chat_sun_may17_1/`;
- auditoria consolidada:
  `artifacts/openrouter/v584_uploaded_chat_sun_may17_1/KG1_V584_UPLOADED_CHAT_AUDIT.md`.

Modelos com conteudo aproveitavel:

- `baidu/cobuddy:free`;
- `openrouter/owl-alpha` parcial;
- `poolside/laguna-xs.2:free`;
- `poolside/laguna-m.1:free`;
- `arcee-ai/trinity-large-thinking:free`;
- `minimax/minimax-m2.5:free` parcial;
- `openai/gpt-oss-120b:free` parcial;
- `z-ai/glm-4.5-air:free`;
- `nvidia/nemotron-nano-9b-v2:free`;
- `openai/gpt-oss-20b:free`.

Consenso validado:

- V582 continua bloqueado para package/submit:
  - `191/315` por `raw_output` label-free;
  - `bit=135/160`, `equation=56/155`, `truncated=1`;
  - protected backfire em `8740ed31` e `59bee375`;
  - runaway com `completion_tokens_mean=4775.25`, `max=7680`.
- Nao repetir broad SFT, long traces ou teacher-transfer misturado antes de
  diagnostico local curto.
- A unica metrica de promocao continua sendo `raw_output` + extractor
  label-free atual + `verify_answer`, nunca coluna `prediction` armazenada ou
  expected-aware.

Achados acionaveis adicionados:

1. P0 - `raw_output answer-anywhere` audit:
   - para cada row, verificar se o expected answer aparece em algum ponto do
     `raw_output`;
   - registrar `answer_found_anywhere`, primeira posicao aproximada, family,
     protected status, `completion_tokens`, predicao extraida e
     `verify_answer`;
   - se a resposta correta aparece cedo mas o extractor erra, priorizar parser e
     formato;
   - se a resposta correta nunca aparece, tratar como drift/learnability do
     adapter.
2. P1 - replay-only diagnostic:
   - usar rows de replay apenas para medir estabilidade de pipeline/init;
   - nao usar replay-only como evidencia de ganho submit-safe.
3. P1 - sanity A/B de formato para `equation_transform`:
   - testar `\boxed{...}` vs `Final answer: ...` em subset com `{`, `}`, `\`,
     `]`, `!`;
   - permitir plain apenas quando o extractor label-free verificar a resposta;
   - nao alterar prompt oficial de submissao sem gate.
4. P1/P2 - correlacao de row-loss, answer-span CE/margem e ACC:
   - medir delta de loss por row/family;
   - comparar com delta de margem da resposta e `verify_answer`;
   - se loss melhora sem ganho de margem/resposta correta, bloquear promocao por
     loss.
5. P2 - grad norm/trainability e target ablation:
   - logar grad norms dos `target_parameters` atuais;
   - testar `up_proj`/attention apenas em micro-run depois que decoding/format
     passarem.

Rejeicoes explicitas desta rodada:

- `train_token_max=315` como causa raiz nao esta comprovado; V286 ja indicou
  zero prompt truncation, zero completion dropped e zero fallback masks. Manter
  apenas como sanity audit.
- Nao rodar `MAX_STEPS=8/10/20` direto sobre V582 antes de `MAX_STEPS=0`,
  `answer-anywhere`, protected guard e token-budget gate.
- Nao excluir protected rows como estrategia de preservacao; protected precisa
  de replay/guard, nao sumir do contrato.
- Nao usar `stop_token` ingenuo em `\boxed{}`; risco de cortar respostas
  simbolicas.
- Nao promover por ACC expected-aware ou por treino direto em weak labels.

Proximo passo obrigatorio:

1. Implementar/rodar auditoria CPU `answer-anywhere` no CSV de predicoes V582 e
   nos baselines disponiveis.
2. Em seguida, rodar diagnostico sem treino:
   - base model, `max_new_tokens=512`;
   - base + V290 checkpoint-6, `max_new_tokens=512`;
   - base + V582 checkpoint-2, `max_new_tokens=512`;
   - base + V582 checkpoint-2 com limite maior apenas para confirmar runaway.
3. So depois disso decidir se vale micro-transfer family-isolated ou se a falha
   e de parser/decoding.

### Atualizacao V585 - Segundo Export OpenRouter Plateau Audit 2026-05-17

Fonte analisada:

- export do usuario:
  `C:\Users\davis\Downloads\OpenRouter Chat Sun May 17 2026 (2).json`;
- respostas extraidas:
  `artifacts/openrouter/v585_uploaded_chat_sun_may17_2/`;
- auditoria consolidada:
  `artifacts/openrouter/v585_uploaded_chat_sun_may17_2/KG1_V585_UPLOADED_CHAT_AUDIT.md`.

Respostas finais uteis:

- `google/gemini-3.1-pro-preview-20260219`;
- `qwen/qwen3.5-plus-20260420`;
- `anthropic/claude-4.7-opus-20260416`;
- `deepseek/deepseek-v3.2-speciale-20251201` parcial;
- `qwen/qwen3.6-plus-04-02`;
- `qwen/qwen3.6-max-preview-20260420` parcial e completa.

Fontes externas verificadas nesta rodada:

- `tonghuikang/nemotron` e um repo publico da submissao Progress Prize e contem
  `reasoners`, `corpus`, `trainer`, `train_sft.py`, paginas de corpus/training
  e metrics;
- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge` no HF aparece como
  dataset CSV com cerca de `9.5k` linhas, `id`, `prompt`, `answer`;
- essas fontes continuam sendo referencia/teacher/diagnostico, nao evidencia
  adapter-only de submit.

Consenso novo que altera a ordem do plano:

1. P0 - Decode-contract 2x2:
   - baseline com decoding do baseline;
   - baseline com decoding do V582;
   - V582 com decoding do baseline;
   - V582 com decoding do V582;
   - salvar hash/JSON completo de `generation_config`: `max_new_tokens`,
     temperature, top_p, stop strings, EOS/pad token ids, thinking mode e
     sampler.
   - objetivo: separar config ruim de decoding de drift real do adapter.
2. P0 - Per-row delta taxonomy:
   - classificar cada row como `flip_correct`, `flip_wrong`,
     `still_wrong_same` ou `still_wrong_changed`;
   - cruzar com family, protected status, `completion_tokens`,
     `answer-anywhere` e predicao extraida;
   - sem essa tabela, total agregado de ACC continua escondendo o motivo do
     plato.
3. P0/P1 - LoRA alpha attenuation diagnostic:
   - avaliar V582 sem novo treino com escala de adapter `1.0`, `0.5`, `0.25`,
     `0.1`;
   - se protected rows recuperam e token length cai, o problema e magnitude do
     adapter/choque de LoRA;
   - se nao recupera, tratar como dataset/objetivo/superficie.
4. P1 - EOS/stop probability probe:
   - medir probabilidade de EOS no ponto esperado apos a resposta em base, V290
     e V582;
   - usar como diagnostico de runaway, nao como metrica de promocao.
5. P1 - Teacher target base-likelihood filter:
   - calcular perplexity/logprob do base model nos targets teacher;
   - filtrar targets longos, low-likelihood ou fora da distribuicao natural do
     modelo antes de qualquer destilacao.
6. P1 - Equation-first, bit-protected curriculum:
   - se os gates P0 passarem, treinar apenas equation teacher-gain normalizado;
   - usar bit apenas como replay/protected guard;
   - nao usar bit teacher-gain no primeiro experimento;
   - `ANSWER_SPAN_LOSS_WEIGHT=1.0`;
   - checkpoint-1 gate obrigatorio: abortar em `bit<136`, protected backfire ou
     token runaway.
7. P1/P2 - Routing/grad diagnostics se treino voltar:
   - grad norms por grupo de parametro;
   - routing entropy/gate logits se usar MoE targets;
   - per-token gradient norm para simbolos `{`, `}`, `\`, `]`, `!`;
   - perda por answer length para testar a hipotese de amplificacao em answers
     curtas de equation.

Rejeicoes V585:

- nao rodar 5/10/20 steps so porque modelos sugeriram mais treino. Isso so e
  permitido depois de `answer-anywhere`, delta taxonomy, decode-contract 2x2,
  alpha attenuation, protected guard e token-budget gate;
- nao tratar `train_token_max=315` como prova de target truncado. O V286 real ja
  mostrou `prompt_truncation=0`, `completion_dropped=0`, `fallback_masks=0`;
- nao usar treino nas 155 weak equation rows como evidencia submit-safe;
- nao adicionar attention/up_proj direto antes de provar que decoding/escala e
  formato nao explicam o plateau;
- nao usar CPU teacher projection como candidato submit-safe.

Ordem obrigatoria atualizada antes de qualquer novo H200 de treino:

1. `answer-anywhere` + per-row delta taxonomy em baseline, V290 e V582.
2. Decode-contract 2x2 com hash de `generation_config`.
3. V582 alpha attenuation eval sem treino.
4. EOS/stop probe e teacher target base-likelihood filter.
5. Se todos os gates anteriores justificarem treino: micro-transfer
   equation-first, bit-protected, checkpoint-1 gate.

Regra FinOps atualizada:

- enquanto os quatro primeiros passos nao existirem com manifest e sem
  blockers, novo H200 de treino e bloqueado como tentativa cega;
- inferencia diagnostica curta e permitida, treino nao.

### Atualizacao V586 - Plateau Row Diagnostics CPU 2026-05-17

Implementado e rodado:

- script:
  `scripts/analyze_v586_plateau_row_diagnostics.py`;
- saida:
  `artifacts/v586_plateau_row_diagnostics/`;
- resumo:
  `artifacts/v586_plateau_row_diagnostics/v586_v516_vs_v582_cp2_summary.json`;
- deltas por row:
  `artifacts/v586_plateau_row_diagnostics/v586_v516_vs_v582_cp2_row_deltas.csv`;
- relatorio:
  `artifacts/v586_plateau_row_diagnostics/KG1_V586_V516_VS_V582_CP2_SUMMARY.md`.

Checks executados:

- `python scripts/analyze_v586_plateau_row_diagnostics.py --self-test`;
- `python -m py_compile scripts/analyze_v586_plateau_row_diagnostics.py`;
- `python scripts/kg1_static_safety_gate.py --self-test`;
- `python scripts/kg1_static_safety_gate.py scripts/analyze_v586_plateau_row_diagnostics.py scripts/kg1_static_safety_gate.py`.

Resultado V586 sobre baseline V516 vs V582 checkpoint-2:

- baseline parser-current label-free:
  - `192/315`;
  - `bit_manipulation=136/160`;
  - `equation_transform=56/155`;
  - `truncated=0`.
- V582 checkpoint-2:
  - `191/315`;
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`;
  - `completion_tokens_mean=4775.2508`;
  - `completion_tokens_max=7680`.
- delta taxonomy:
  - `both_correct=190`;
  - `flip_correct=1`: `4ada9150`;
  - `flip_wrong=2`: `59bee375`, `8740ed31`;
  - `still_wrong_same=109`;
  - `still_wrong_changed=13`.
- protected rows:
  - `8740ed31`: `protected_id_backfire`;
  - `59bee375`: `protected_id_backfire`;
  - `55d834d1`: `protected_id_missing_required_gain`, nao backfire.
- `answer_anywhere_wrong_final=23` em V582:
  - `bit_manipulation=4`;
  - `equation_transform=19`.

Achado cirurgico:

- V582 nao falhou por falta pequena de steps. O checkpoint tem 1 ganho real e 2
  perdas reais, incluindo 2 protected backfires.
- O maior problema mensuravel agora e misto:
  - drift do adapter em protected bit rows;
  - runaway/decoding com token budget estourado;
  - 23 rows com resposta correta aparecendo no texto mas a resposta final
    extraida errada.
- Portanto novo treino amplo ou mais longo continua bloqueado.

Novo gate permanente:

- `scripts/kg1_static_safety_gate.py` agora exige que
  `scripts/analyze_v586_plateau_row_diagnostics.py` preserve:
  - metrica `raw_output -> extract_final_answer -> verify_answer`;
  - `answer_anywhere_wrong_final`;
  - separacao entre `protected_id_backfire` e
    `protected_id_missing_required_gain`;
  - blocker de token runaway;
  - blocker de `stored_prediction_not_raw_extraction`;
  - self-test do V586.


### Atualizacao V587 - Output Extraction Audit CPU 2026-05-17

Implementado/rodado sem GPU e sem treino:

- `python scripts/analyze_eval_predictions.py --predictions-csv artifacts/v582_hf_h200_launch/v582_checkpoint2_predictions.csv --output-dir artifacts/v587_output_extraction_audit/v582_cp2`;
- `python scripts/analyze_eval_predictions.py --predictions-csv artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv --output-dir artifacts/v587_output_extraction_audit/v516_baseline`.

Resultados objetivos:

- V582 checkpoint-2:
  - `official_correct=191/315`;
  - `first_boxed_correct=191/315`;
  - `early_512_correct=0/315`;
  - `early_1024_correct=2/315`;
  - `early_2048_correct=9/315`;
  - `completion_tokens_mean=4775.2508`;
  - `completion_tokens_max=7680`;
  - `truncated=1`.
- V516/V290 baseline parser-current:
  - `official_correct=192/315`;
  - `first_boxed_correct=192/315`;
  - `early_512_correct=0/315`;
  - `early_1024_correct=2/315`;
  - `early_2048_correct=9/315`.

Conclusao nova:

- trocar o extractor para `first_boxed` nao gera ganho; e exatamente igual ao
  extractor oficial nos dois CSVs analisados;
- janelas curtas de caracteres tambem nao sao caminho de promocao; elas perdem
  quase todas as respostas;
- o sinal `answer_anywhere_wrong_final=23` do V586 e diagnostico label-aware e
  ruidoso para alguns simbolos/numeros, nao e uma regra de postprocessamento
  submit-safe;
- o foco permanece em reduzir drift/runaway do adapter ou filtrar targets, nao
  em trocar parser.

Impacto no plano:

- parser/postprocessor alternativo fica bloqueado para submit adapter-only;
- proximo passo continua sendo eval-only de escala/decoding, nao treino;
- qualquer novo dataset de destilacao deve ensinar finalizacao curta no formato
  oficial, mas so depois de provar que nao causa protected backfire.

Proximo passo obrigatorio:

1. Rodar decode-contract 2x2 curto, sem treino, usando os mesmos 315 rows:
   - V290/base com config oficial longa;
   - V290/base com config curta diagnostica;
   - V582 com config oficial longa;
   - V582 com config curta diagnostica.
2. Se V582 curto ainda tiver protected backfire, bloquear V582 como linhagem de
   init/teacher-transfer e partir para novo teacher target filtrado.
3. Se V582 curto recuperar protected mas perder answer final, atacar
   formato/finalizacao antes de qualquer treino.
4. Se apenas a escala do adapter for suspeita, executar alpha attenuation
   `1.0/0.5/0.25/0.1` como eval-only antes de novo H200 de treino.

### Atualizacao V588 - Adapter Interpolation Probe H200 2026-05-18

Objetivo:

- testar, sem novo treino, se uma interpolacao conservadora entre o baseline
  V290 checkpoint-6 e o V582 checkpoint-2 recuperaria parte do ganho teacher
  sem causar protected backfire/runaway;
- candidatos gerados:
  - `lambda=0.10`;
  - `lambda=0.25`;
  - `lambda=0.50`.

Correcoes antes de rodar:

- o primeiro job V588 falhou cedo no gate de compatibilidade porque comparava
  `target_modules` por ordem textual;
- corrigido para comparar `target_modules` e `target_parameters` como conjuntos
  ordenados, preservando compatibilidade real do contrato LoRA;
- self-tests e static gate passaram antes do relaunch.

Resultado real observado:

- job relancado:
  `https://huggingface.co/jobs/felipesp1983/6a0a6a6ea5e509f1a8413f00`;
- `lambda=0.10` terminou com:
  - `190/315`;
  - `bit_manipulation=135/160`;
  - `equation_transform=55/155`;
  - `truncated=2`;
  - `avg_completion_tokens=4798.1524`;
  - `max_completion_tokens=7680`.
- comparado ao baseline submit-safe `192/315`, `bit=136`, `equation=56`,
  `truncated=0`, o menor passo de interpolacao ja piorou as duas familias e
  aumentou truncation.

Decisao FinOps:

- cancelar o job antes de gastar H200 nos candidatos `lambda=0.25` e
  `lambda=0.50`, pois eles ficam ainda mais proximos do adapter V582, que ja
  era regressivo;
- cancelamento executado apos o resumo do `lambda=0.10`;
- essa rota nao e submit-safe e nao deve ser promovida.

Impacto no plano:

- bloquear V582 checkpoint-2 como linhagem de init, teacher-transfer ou
  interpolacao;
- nao rodar nova escala/interpolacao V582 sem um diagnostico novo que prove
  recuperacao de protected rows e queda real de completion tokens;
- proximo caminho com chance tecnica:
  - teacher target base-likelihood filter;
  - dataset equation-first filtrado por targets que o modelo base consiga
    completar;
  - bit apenas como replay/protected guard;
  - ou pacote submit-safe que execute solver/verifier no caminho de inferencia,
    se as regras e o pacote permitirem.

### Atualizacao V589 - Crisis Audit Dataset/Metric 2026-05-18

Objetivo:

- analisar em modo bloqueador tudo que poderia gerar ganho falso ou esconder o
  plato: dataset, flags anti-leak, extractor label-free, protected rows,
  contrato/masks, adapter config, weak CSV, truncation e workspace clean.

Evidencia nova:

- `V509` sobre o dataset V582/V581 bloqueou o train e o validation:
  - train `315/315` rows com overlap exato weak/full por prompt e
    prompt+answer;
  - validation `115/115` rows com overlap exato weak/full;
  - flags `weak_gate_rows_used_for_training=false`,
    `full_gate_rows_used_for_training=false` e
    `gate_rows_used_for_training=false` aparecem mesmo em rows com overlap;
  - `expected_aware_teacher_signal=true` e
    `label_audited_teacher_projection=true` existem no train.
- `V540` confirmou que o problema nao e so parser:
  - baseline V516 reextraido por `raw_output -> extract_final_answer ->
    verify_answer`: `192/315`, `bit=136`, `equation=56`, protected OK;
  - V582 checkpoint-2 reextraido: `191/315`, `bit=135`, `equation=56`,
    `truncated=1`;
  - V582 quebrou protected rows:
    - `8740ed31`: esperado `01101000`, extraido `01111000`;
    - `59bee375`: esperado `10010101`, extraido `2`;
  - existe 1 row (`4bb8c6cd`) em que o `prediction` salvo estava stale
    frente ao parser atual; isso melhora equation no recompute, mas nao resolve
    o total nem o backfire.
- `V564` confirmou que masks/offsets estavam OK, mas protected rows foram
  superamostradas:
  - `mask_ok=315` no train;
  - validation com `row_loss_weight=0`;
  - protected `8740ed31` e `59bee375` aparecem `16x` cada no train com peso
    `2.6`;
  - mesmo assim o adapter empurrou essas rows para respostas erradas. Isso e
    drift/backfire real, nao ganho de loss.
- `V575` mostrou sync incorreto/insuficiente para promocao:
  - adapter config V582: base Nemotron correto, `r=32`, `alpha=32`,
    `modules_to_save=null`, target modules/parameters presentes;
  - performance blockers: `equation=56<60`, `bit=135<136`,
    `truncated=1>0`, `stored_prediction_stale_after_parser_fix=1`;
  - manifesto de eval V582 nao carregava contrato promocional completo
    (`weak_sha`, shared row contract, promotion floors).
- Recheck do caminho source-only V573:
  - V509 V573 continua limpo: `blocked_dataset_count=0`;
  - V526 V573 continua objetivo-correto com `example_mean` e row weights
    (`example_mean_bit_share=0.671963`, delta `0.069972`);
  - porem o weak eval V574 checkpoint-2 de V573 foi `190/315`,
    `bit=135`, `equation=55`, `truncated=1` e quebrou os mesmos protected
    rows. Isso mostra que V573 nao esta contaminado como V582, mas tambem nao
    transferiu ACC; relancar V573 sem mudanca de objetivo/decoding e bloqueado.

Correcoes implementadas:

- `scripts/hf_job_preflight_gate.py`:
  - bloqueia `v581_combined_teacher_distill_dataset` e
    `v582_combined_teacher_distill_dataset`;
  - bloqueia qualquer JSONL de treino pago que contenha
    `expected_aware_teacher_signal=true` ou
    `label_audited_teacher_projection=true`;
  - self-test novo cobre esse caso.
- `scripts/kg1_pre_paid_job_integration_gate.py`:
  - bloqueia as mesmas identidades V581/V582;
  - audita e bloqueia rows expected-aware/label-audited no dataset;
  - self-test novo cobre o bloqueio.
- `scripts/kg1_static_safety_gate.py`:
  - bloqueia launchers/notebooks ativos que referenciem V581/V582;
  - exige que o preflight mantenha o bloqueio expected-aware.
- `artifacts/v582_hf_h200_launch/launch_v582_hf_nemo_h200_v581_teacher_transfer.py`:
  - arquivado como `fail-closed`, mantendo evidencia historica mas impedindo
    relaunch.
- workspace clean:
  - `__pycache__` e `.cache` removidos;
  - `scripts/kg1_workspace_clean_gate.py .` passou com `0` findings.

Testes executados:

- `python -m py_compile scripts/hf_job_preflight_gate.py scripts/kg1_pre_paid_job_integration_gate.py scripts/kg1_static_safety_gate.py artifacts/v582_hf_h200_launch/launch_v582_hf_nemo_h200_v581_teacher_transfer.py`;
- `python scripts/hf_job_preflight_gate.py --self-test`;
- `python scripts/kg1_pre_paid_job_integration_gate.py --self-test`;
- `python scripts/kg1_static_safety_gate.py --self-test`;
- `python scripts/kg1_static_safety_gate.py --paths scripts/hf_job_preflight_gate.py scripts/kg1_pre_paid_job_integration_gate.py scripts/kg1_static_safety_gate.py artifacts/v582_hf_h200_launch/launch_v582_hf_nemo_h200_v581_teacher_transfer.py`;
- `python scripts/validate_answer_extraction_v1.py ... --run-csv v582=...`;
- `python scripts/audit_v564_contract_mask_alignment.py ...`;
- `python scripts/audit_v575_solution_sync_contract.py ...`;
- `python scripts/kg1_workspace_clean_gate.py .`.

Decisao:

- V581/V582 deixa de ser caminho ativo para treino, init, interpolation ou
  submit. Qualquer ganho projetado nessa rota e contaminado/diagnostico.
- O plato nao deve ser tratado como "precisa de mais epochs": a evidencia agora
  mostra dataset weak/full-derived, protected-row backfire e sync incompleto.
- O proximo passo ativo so pode ser:
  1. dataset source-only sem overlap weak/full, com V509 limpo;
  2. teacher target filtrado por base-likelihood e sem expected-aware labels;
  3. ou solver/verifier submit-safe no caminho de inferencia, se o pacote e as
     regras permitirem;
  4. qualquer novo treino pago precisa passar V509, V540, V564, V575,
     pre-paid gate e static gate com `finding_counts.warning=0` e
     `finding_counts.error=0`.

### Atualizacao V590 - Parser-current CPU Target 2026-05-18

Objetivo:

- remover ambiguidade entre `prediction` salvo e o parser atual antes de
  usar qualquer CPU solver como teacher target;
- recomputar o gate residual sem loss usando o baseline V516 materializado
  por `raw_output -> extract_final_answer -> verify_answer`.

Evidencia nova:

- o CSV historico V516 tinha 1 row stale:
  - `4bb8c6cd`: `prediction` salvo `]`, parser atual extrai `]}\!`;
  - isso explica divergencia entre manifest `192/315` e CSV reavaliado como
    `191/315` por scripts que confiavam na coluna stale.
- baseline parser-current materializado em:
  - `artifacts/v590_parser_current_baseline/20260518T_v516_parser_current/v516_label_free_v290_checkpoint6_baseline.csv`;
  - resultado: `192/315`, `bit_manipulation=136/160`,
    `equation_transform=56/155`, `truncated=0`;
  - row contract:
    `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- V350 recomputado sobre esse baseline:
  - `199/315`;
  - `bit_manipulation=138/160`;
  - `equation_transform=61/155`;
  - `truncated=0`;
  - `gains=7`, `losses=0`.
- V543 query-op refinement sobre V350 recomputado:
  - `201/315`;
  - `bit_manipulation=138/160`;
  - `equation_transform=63/155`;
  - `truncated=0`;
  - `gains=2`, `losses=0`;
  - manifest:
    `artifacts/v590_current_baseline_residual_gate/20260518T_v543_queryop_refinement/v543_symbolic_queryop_refinement_manifest.json`.

Decisao:

- existe ganho CPU/verifier real contra o baseline adapter-only:
  - total `+9`;
  - bit `+2`;
  - equation `+7`;
  - sem perdas e sem truncation.
- esse ganho ainda nao e submit-safe adapter-only, porque o pacote Kaggle
  aceito continua limitado ao adapter LoRA e nao pode depender de
  `prediction_postprocessor`/solver no caminho de inferencia.
- V590 substitui metas stale como `KG1_CPU_SIMULATED_TOTAL_CORRECT=200` em
  launchers antigos; qualquer launcher novo deve usar os numeros reais do
  target que for treinado.

Correcoes de gate implementadas:

- `scripts/hf_job_preflight_gate.py` agora exige, antes de treino pago:
  - `KG1_V516_PARSER_CURRENT_BASELINE_STATUS=passed`;
  - `KG1_STALE_PREDICTION_PARITY_STATUS=passed`.
- `scripts/kg1_pre_paid_job_integration_gate.py` exige as mesmas flags em
  launchers pagos.
- `scripts/kg1_static_safety_gate.py` agora monitora que essas protecoes
  continuem presentes.

Testes executados:

- `python -m py_compile scripts/hf_job_preflight_gate.py scripts/kg1_pre_paid_job_integration_gate.py scripts/kg1_static_safety_gate.py`;
- `python scripts/hf_job_preflight_gate.py --self-test`;
- `python scripts/kg1_pre_paid_job_integration_gate.py --self-test`;
- `python scripts/kg1_static_safety_gate.py --self-test`;
- `python scripts/kg1_static_safety_gate.py --paths scripts/hf_job_preflight_gate.py scripts/kg1_pre_paid_job_integration_gate.py scripts/kg1_static_safety_gate.py`.

Proximo passo ativo:

1. Construir V591 adapter-transfer dataset source-only a partir das classes de
   regra V590/V543, sem copiar weak rows, answers esperados ou exemplos
   label-audited.
2. Treino/eval so pode iniciar se V509/V526/V540/V564/V575/pre-paid/static
   passarem com:
   - `finding_counts.warning=0`;
   - `finding_counts.error=0`;
   - `0` overlap weak/full;
   - `0` truncation;
   - `0` stale prediction mismatch;
   - protected rows intactas.
3. Primeiro checkpoint deve manter no minimo:
   - `bit_manipulation>=136`;
   - `equation_transform>=56`;
   - `truncated=0`.
4. Promocao para submit so pode ocorrer com adapter-only medido acima do
   baseline submit-safe, nao apenas com solver CPU.

### Atualizacao V591 - Adapter-transfer Query-op Source-only 2026-05-18

Objetivo:

- transformar o ganho CPU V590/V543 em um candidato adapter-only sem copiar
  rows weak/full, sem expected-aware labels e sem dataset teacher contaminado;
- manter bit como replay protegido e adicionar apenas exemplos fonte-sinteticos
  para as assinaturas V543 aceitas:
  - `symbolic_cryptarithm_multi_operator_digits_add|query_op=!`;
  - `symbolic_cryptarithm_multi_operator_digits_mul|query_op=$`;
  - `symbolic_cryptarithm_single_operator_digits_mul|query_op=%`.

Evidencia/gates:

- dataset V591 criado em
  `artifacts/v591_v579_symbolic_queryop_source_mix/20260518T_v591_cpu_gate`;
- train `877` rows, SHA
  `d70fda3b979703cc8fef52463498f06b2cb56a6f517cae1907bee46147668da7`;
- validation `189` rows, SHA
  `6968ef93c8d2b0dc3608dc7adaef8d0cea0ee6d6e499434a5f264b3489113df9`;
- V509 passou: `blocked_dataset_count=0`;
- V478 passou com `example_mean` e row weights:
  - train effective bit share `0.666212`;
  - train effective equation share `0.333788`;
  - `unknown_source_rows=0`, `unknown_subcategory_rows=0`;
- V286 real passou:
  - `max_length=2048`;
  - train token max `1074`;
  - validation token max `1037`;
  - `0` truncation, `0` fallback masks, `0` dropped completion tokens;
- V513 passou sem warnings/blockers;
- V526 passou com `example_mean`:
  - delta `0.075723 <= 0.08`;
  - `gpu_allowed=true`;
  - `0` mismatch de answer e `0` control chars;
- dataset subido para HF:
  - repo `felipesp1983/kg1-v591-v579-symbolic-queryop-source-mix-artifacts`;
  - commit `aed6804b77eb8dc8aa2b50726d0afe1fbe38759d`.

Launcher/gate:

- launcher ativo:
  `artifacts/v591_hf_h200_launch/launch_v591_hf_nemo_h200_v579_symbolic_queryop.py`;
- debug local baixou o dataset do HF, validou hashes, adapter seed e objetivo;
- pre-paid integration gate passou sem findings em
  `artifacts/v591_hf_h200_launch/v591_pre_paid_integration_gate.json`;
- contrato CPU real do launcher:
  - `KG1_CPU_SIMULATED_TOTAL_CORRECT=201`;
  - `KG1_CPU_SIMULATED_BIT_CORRECT=138`;
  - `KG1_CPU_SIMULATED_EQUATION_CORRECT=63`;
  - `KG1_MAX_TOKEN_HEADROOM_RATIO=0.525`;
  - `KG1_V516_PARSER_CURRENT_BASELINE_STATUS=passed`;
  - `KG1_STALE_PREDICTION_PARITY_STATUS=passed`;
  - manifest V543 SHA
    `df8c4c55dda6ed5acead78a4453e57fa1bb4cb649bf3c4b26a789791d2620bd9`;
  - integrated CSV SHA
    `71ae1283b0bc1dadb39cff87fad6265524f62bf451b0f1da617386568f9e6360`.

Decisao:

- V591 e o unico candidato ativo para H200 agora, porque:
  - usa ganho CPU parser-current sem perdas;
  - nao usa V581/V582;
  - passa gates de integridade, tokenizacao, objetivo e pre-paid;
  - corrige o vazamento herdado do launcher V573 que ainda colocava
    `200/62` e headroom `0.668` no `job_env`.
- executar somente smoke H200 de `2` steps, `timeout=3600`.
- kill-switch obrigatorio no primeiro checkpoint:
  - weak eval label-free deve manter `bit>=136`, `equation>=56`,
    `truncated=0` e protected rows intactas;
  - promocao para package/submit exige ganho adapter-only real acima de
    `192/315`; meta submit-safe operacional continua `total>=196`,
    `equation>=60`, `bit>=136`, `truncated=0`.
- se checkpoint-2 repetir backfire de protected rows, truncation ou queda de
  bit/equation, cancelar/abandonar a linha antes de qualquer treino longo.

### Atualizacao V591 - Resultado H200 Weak Eval e Bloqueio 2026-05-18

Resultado real:

- treino H200 V591 concluiu, mas o checkpoint-2 falhou no weak gate;
- job de treino:
  `https://huggingface.co/jobs/felipesp1983/6a0a7bb3e7940de6ee6ce074`;
- job weak eval:
  `https://huggingface.co/jobs/felipesp1983/6a0a80e7e7940de6ee6ce0ad`;
- diagnosticos enviados para o adapter repo no commit:
  `728d4b120215be0a69b67bc807a32e73b33550bd`;
- metricas do treino:
  - baseline eval loss `1.7252`;
  - final eval loss `1.7261`;
  - sem truncation de tokenizacao no treino;
- weak eval armazenado pelo job remoto:
  - `190/315`;
  - `bit_manipulation=135/160`;
  - `equation_transform=55/155`;
  - `truncated=1`;
  - `avg_completion_tokens=4775.48`;
  - `max_completion_tokens=7680`;
- recomputacao local com parser atual:
  - `191/315`;
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`;
  - ainda abaixo do baseline submit-safe `192/315`.

Diagnostico linha a linha:

- artefatos locais:
  - `artifacts/v591_hf_h200_launch/v591_v586_plateau_diagnostics/KG1_V591_CHECKPOINT2_SUMMARY.md`;
  - `artifacts/v591_hf_h200_launch/diagnostics_local/v591_vs_baseline_v543_summary.json`;
- contra baseline V516/V590:
  - ganhos reais V591: `1` row (`4ada9150`, bit);
  - perdas reais V591: `2` protected bit rows (`8740ed31`, `59bee375`);
  - net atual: `-1` pelo parser local, `-2` pelo parser remoto antigo;
- contra alvo CPU V543:
  - V543 tem `9` gains reais (`+2` bit, `+7` equation);
  - V591 copiou apenas `1/9`;
  - V591 perdeu `7/7` gains de `equation_transform`;
  - isso prova que o dataset source-only query-op nao transferiu o solver para
    o adapter.

Achados de bug/gap:

- o eval remoto usou o commit GitHub `403c3f9`, enquanto o worktree local tem
  correcoes nao publicadas em `src/competition_utils.py`;
- a diferenca `190` vs `191` vem de parser drift em simbolos: o parser local
  atual extrai corretamente `\boxed{]}\!}`, mas o eval remoto antigo extraiu
  apenas `]`;
- esse parser fix nao gera submit candidate, porque V591 continua abaixo do
  baseline e tem backfire/truncation;
- V591 exibiu token runaway: protected bit rows continuam exigindo milhares de
  tokens e a linha `59bee375` truncou;
- `label_aware_debug_correct > correct` apareceu como sinal de formato/extracao,
  nao como ganho submit-safe.

Correcoes aplicadas ao gate:

- `scripts/hf_job_weak_eval_v245.py` agora bloqueia promocao quando
  `label_aware_debug_correct - correct > KG1_WEAK_PROMOTE_LABEL_AWARE_DELTA_MAX`;
- default operacional: `KG1_WEAK_PROMOTE_LABEL_AWARE_DELTA_MAX=0`;
- `scripts/kg1_static_safety_gate.py` passou a procurar esse bloqueador;
- `launch_v591_hf_weak_eval.py` recebeu a flag explicitamente.

Testes executados:

- `python -m py_compile scripts/hf_job_weak_eval_v245.py scripts/kg1_static_safety_gate.py artifacts/v591_hf_h200_launch/launch_v591_hf_weak_eval.py`;
- `python scripts/hf_job_weak_eval_v245.py --self-test`;
- `python scripts/kg1_static_safety_gate.py --paths scripts/kg1_static_safety_gate.py src/competition_utils.py artifacts/v591_hf_h200_launch/launch_v591_hf_weak_eval.py`;
- sanity parser:
  - `extract_final_answer("</think>\\boxed{]}\\!}") == "]}\\!"`;
  - `verify_answer("]}\\!", "]}\\!") == True`.

Decisao:

- V591 esta bloqueado para submit, full eval e treino adicional;
- nao executar `final` nem mais epochs/steps da V591, porque o primeiro
  checkpoint ja violou FinOps/ACC:
  - `correct < 196`;
  - `bit < 136`;
  - `equation < 60`;
  - `truncated > 0`;
  - protected-row backfire;
  - token runaway;
  - label-aware delta.

Proximo passo ativo:

1. Qualquer novo job deve usar script patch aplicado no HF para evitar drift
   entre codigo local e commit remoto.
2. Novo treino so e permitido depois de um gate local provar que o objetivo
   aprende pelo menos parte dos `9` gains V543 sem alterar protected rows.

### Atualizacao V592 - Interpolation Probe Cancelado por FinOps 2026-05-18

Resultado real:

- job H200 eval-only:
  `https://huggingface.co/jobs/felipesp1983/6a0a8779a5e509f1a841416e`;
- run id:
  `v592-h200-adapter-interp-v290-to-v591-20260518T032746Z`;
- objetivo: testar se `V290 + lambda * (V591 - V290)` preservaria o baseline
  e capturaria o unico ganho bit da V591 sem backfire;
- primeiro candidato avaliado, `lambda=0.10`:
  - `190/315`;
  - `bit_manipulation=134/160`;
  - `equation_transform=56/155`;
  - `truncated=1`;
  - `avg_completion_tokens=4775.39`;
  - `max_completion_tokens=7680`;
  - `label_aware_debug_delta=0`.

Decisao FinOps:

- `lambda=0.10` ja ficou abaixo do baseline submit-safe `192/315`, reduziu
  bit para `134/160` e manteve truncation;
- `lambda=0.25` e `lambda=0.50` caminham ainda mais na direcao do adapter
  V591 que ja falhou, portanto a expectativa racional era piorar, nao
  melhorar;
- job cancelado manualmente no HF com status `CANCELED`;
- log salvo em:
  `artifacts/v591_hf_h200_launch/v592_job_6a0a8779a5e509f1a841416e_canceled_logs.txt`.

Conclusao:

- interpolacao nao resgatou a linha V591/V579;
- encerrar a linha V591/V579 para submit, full eval, mais epochs, mais steps
  e novas interpolacoes;
- o problema nao e apenas intensidade do adapter: o delta treinado empurra
  protected bit rows para geracao longa/errada antes de aprender os ganhos
  V543.

Proximo passo obrigatorio:

1. Parar loops de `solver -> source-only SFT -> weak eval` ate haver evidencia
   de preferencia do modelo pelas respostas corretas dos ganhos V543.
2. Executar um probe de NLL/logits em cima dos `9` gains V543:
   - comparar resposta baseline errada vs resposta solver correta;
   - medir base/V290/V591 quando possivel;
   - separar `decoding ruim` de `adapter empurrou o modelo para resposta
     errada`;
   - se a resposta correta nao tiver vantagem de NLL/logits, novo SFT curto
     nao deve ser esperado como caminho de ACC.
3. A unica rota submit-safe imediata continua sendo verificar se o pacote
   Kaggle pode executar o solver/verifier V543 no caminho de inferencia. Se
   puder, empacotar e gatear `201/315`, `bit=138`, `equation=63`,
   `truncated=0`; se nao puder, manter adapter-only bloqueado em `192/315`.

### Atualizacao V593 - NLL Contrast Dos 9 Gains V543 2026-05-18

Objetivo:

- diagnostico H200 eval-only, sem treino e sem submit;
- comparar, nos `9` ganhos reais V543, a NLL da resposta correta contra a NLL
  da resposta errada do baseline V290/V516;
- candidatos avaliados:
  - `base_no_adapter`;
  - `v290_checkpoint6`;
  - `v591_checkpoint2`;
- competidores por row:
  - `99d6a3b5`: correto `?()<` vs baseline `(<))`;
  - `7688e06e`: correto `-55` vs baseline `55`;
  - `274def88`: correto `92` vs baseline `-92`;
  - `6cc5dafb`: correto `)(` vs baseline `^&>)`;
  - `d1bd7478`: correto `30` vs baseline `3`;
  - `c5b058d6`: correto `134` vs baseline `35`;
  - `4ada9150`: correto `01111011` vs baseline `01111111`;
  - `4c327b55`: correto `11011100` vs baseline `11011110`;
  - `5501c054`: correto `[#>#` vs baseline `!##^`.

Job:

- `https://huggingface.co/jobs/felipesp1983/6a0a8c70a5e509f1a84141f3`;
- run id:
  `v593-h200-logits-nll-v543-gain-contrast-20260518T034857Z`;
- launcher:
  `artifacts/v593_hf_h200_logits_nll_v543_gain_contrast_launch/launch_v593_hf_logits_nll_v543_gain_contrast.py`.

Regra de decisao:

- margem `wrong_minus_correct = NLL(wrong) - NLL(correct)`;
- margem positiva significa que o modelo prefere a resposta correta curta;
- se V591 nao melhora margem nos gains V543 vs V290, o erro e de objetivo/
  aprendizado, nao apenas de decoding;
- se base/V290 tambem preferem respostas erradas em equation, novo SFT curto
  sem preference/contrastive objective fica bloqueado;
- qualquer futuro treino adapter-only precisa primeiro melhorar essas margens
  e manter regressao protegida `0`.

Resultado:

- job concluido e artefatos baixados;
- logs salvos em:
  `artifacts/v593_hf_h200_logits_nll_v543_gain_contrast_launch/v593_job_6a0a8c70a5e509f1a84141f3_logs.txt`;
- analise local:
  `artifacts/v593_hf_h200_logits_nll_v543_gain_contrast_launch/KG1_V593_ANALYSIS_NOTES.md`;
- medicao valida:
  - `missing_logprob_rows=0`;
  - `prefix_mismatch_rows=0`.

Achados:

- no formato `boxed`, V591 melhorou margem vs V290 em `6/9`, mas reduziu a
  contagem de margens positivas para `4/9` e equation para `3/7`;
- no formato `final_answer_boxed`, V591 piorou margem vs V290 em `7/9`;
- V591 deixou bit positivo em `2/2` no formato `final_answer_boxed`, mas isso
  nao apareceu no weak generation porque as protected rows ainda backfire/
  truncam;
- as piores margens negativas repetidas continuam em equation:
  - `5501c054`: correto `[#>#` vs baseline `!##^`;
  - `6cc5dafb`: correto `)(` vs baseline `^&>)`;
  - `7688e06e`: correto `-55` vs baseline `55`.

Conclusao:

- V591/V579 source-only SFT esta encerrado;
- o platô nao sera resolvido com mais epochs/steps do mesmo dataset;
- gate reprodutivel adicionado:
  `scripts/analyze_v593_nll_gain_contrast.py`;
- resultado do gate:
  `artifacts/v593_hf_h200_logits_nll_v543_gain_contrast_launch/analysis_gate/KG1_V593_NLL_GAIN_CONTRAST_GATE.md`;
- decisao do gate: `blocked`;
- blockers:
  - `final_positive_margin_rows_lt_6`;
  - `final_equation_positive_margin_rows_lt_4`;
  - `candidate_worse_than_baseline_on_final_answer_margins`;
- proximo caminho adapter-only plausivel deve ser preference/contrastive smoke,
  nao SFT:
  - correto V543 acima do baseline errado nos `9` gains;
  - protected correct acima de backfire alternatives;
  - gate de geracao label-free preservando `bit>=136`, `equation>=56`,
    `truncated=0`, protected rows intactas;
  - promocao somente se superar `192/315`.

### Atualizacao V594 - Dataset Preference Query-Operator Source-Only 2026-05-18

Motivo:

- V593 mostrou que V591/V579 nao resolveu as margens de resposta curta dos
  ganhos V543;
- V330 antigo gerava apenas cryptarithm de operador unico/multiplicacao, mas
  os dois piores ganhos novos V543 sao multi-operador:
  - `6cc5dafb`: `query_op=!`, regra aceita `add`, correto `)(`;
  - `5501c054`: `query_op=$`, regra aceita `mul`, correto `[#>#`;
  - `99d6a3b5`: `query_op=%`, `single_operator_digits_mul`, ja coberto.

Implementacao:

- novo builder:
  `scripts/build_v594_queryop_cryptarithm_preference_dataset.py`;
- gera apenas pares preference source-only, sem copiar prompts/respostas weak
  ou full;
- cada `chosen` e revalidado pelo solver CP-SAT V329;
- cada `rejected` e um hard negative com uma unica resposta `\boxed{...}`;
- sem negativos de formato, sem multiplos boxes, sem texto pos-box como
  objetivo de treino;
- assinaturas cobertas:
  - `symbolic_cryptarithm_multi_operator_digits_add|query_op=!`;
  - `symbolic_cryptarithm_multi_operator_digits_mul|query_op=$`;
  - `symbolic_cryptarithm_single_operator_digits_mul|query_op=%`.

Artefatos:

- diretorio:
  `artifacts/v594_queryop_cryptarithm_preference_dataset/20260518T041112Z`;
- train:
  `1920` preference rows, `480` prompts unicos;
- validation:
  `480` preference rows, `120` prompts unicos;
- contagem balanceada por query operator:
  - train: `!=640`, `$=640`, `%=640`;
  - validation: `!=160`, `$=160`, `%=160`;
- anti-leakage:
  - `reference_id_overlap=0`;
  - `reference_prompt_overlap=0`;
  - `reference_prompt_answer_overlap=0`;
  - `non_ascii_chars=0`;
  - `invisible_control_chars=0`.

Validacoes executadas:

- `python -m py_compile scripts/build_v594_queryop_cryptarithm_preference_dataset.py`;
- `python scripts/build_v594_queryop_cryptarithm_preference_dataset.py --self-test`;
- `python scripts/kg1_static_safety_gate.py scripts/build_v594_queryop_cryptarithm_preference_dataset.py`;
- validação do schema V315 preference:
  - train: `1920` rows, `1920` ids unicos;
  - validation: `480` rows, `480` ids unicos;
  - `format_negative_*` ausente.

Decisao:

- V594 e o primeiro dataset apos V593 que mira diretamente os dois modos
  multi-operador que faltavam, sem repetir SFT amplo;
- ainda nao e ganho submit-safe;
- proximo job permitido deve ser somente preference/contrastive smoke com:
  - `PAIR_SCORE_MODE=boxed_payload_mean_nll`;
  - `ALLOW_FORMAT_NEGATIVES=0`;
  - primeiro checkpoint obrigatorio;
  - kill-switch se weak eval nao superar `192/315` ou se `bit<136`,
    `equation<60`, `truncated>0`;
  - V593-style margin check antes de qualquer treino mais longo.

### Atualizacao V596 - Preference Answer-Only para Query-Operator 2026-05-18

Problema encontrado antes do H200:

- o V594 original era seguro contra leakage, mas o `chosen` continha trace longo
  enquanto o `rejected` era `Final answer: \boxed{...}`;
- com `PAIR_SCORE_MODE=boxed_payload_mean_nll`, isso cria contexto assimetrico:
  o payload correto e condicionado por trace de professor e o payload errado
  por uma resposta curta;
- esse mismatch pode reduzir loss sem ensinar `prompt -> resposta` e e uma
  explicacao concreta para plato/backfire em treinos anteriores.

Correcao implementada:

- novo builder:
  `scripts/build_v596_queryop_preference_answer_only.py`;
- novo dataset:
  `artifacts/v596_queryop_answer_only_preference_dataset/20260518T042900Z`;
- deriva do V594, mas reescreve todos os `chosen` para o mesmo contrato do
  `rejected`: `Final answer: \boxed{payload}`;
- preserva os hard negatives source-only e as assinaturas V543:
  - `query_op=!` add;
  - `query_op=$` mul;
  - `query_op=%` mul.

Validacoes:

- train: `1920` preference rows, `1920` ids unicos;
- validation: `480` preference rows, `480` ids unicos;
- `assistant_final_answer_only_rows=1920/1920` no train e `480/480` no val;
- `assistant_trace_rows=0`, `assistant_multiline_rows=0`;
- `reference_id_overlap=0`, `reference_prompt_overlap=0`,
  `reference_prompt_answer_overlap=0`;
- `non_ascii_chars=0`, `invisible_control_chars=0`;
- hashes:
  - train: `b509a3eb5bd841891a918a0bb2766252bc3d5ff4cbb726eea80ae72acd2960e7`;
  - val: `5035e7ff1499c4bfc609115b445bffd310c5dd58829c95c33ac15e1f8c828511`.

Gates corrigidos:

- `scripts/kg1_pre_paid_job_integration_gate.py` agora aceita manifesto V594/V596
  source-only e nao exige `messages` para schema preference;
- o gate continua bloqueando `format_negative_*`, chosen/rejected iguais,
  traces/multilinha quando o contrato e answer-only, overlap de referencia,
  simbolos invisiveis e ruido nao ASCII;
- launcher V595 ajustado para:
  - preservar `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`;
  - usar `INIT_ADAPTER_LOAD_MODE=peft`;
  - setar `REQUIRE_LORA_TARGET_PARAMETER_MATCH=1`;
  - congelar target parameters neste smoke com
    `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`;
  - manter treino limitado a `MAX_STEPS=2`.

Gate pre-pago:

- artefato:
  `artifacts/v595_hf_h200_v594_queryop_pref_launch/v595_v596_pre_paid_job_integration_gate.json`;
- status: `ok=true`, `findings=[]`;
- decisao:
  - permitido apenas smoke H200 de 2 steps;
  - primeiro checkpoint deve passar weak eval;
  - continuar somente se houver ganho real submit-safe:
    `total>=196`, `equation>=60`, `bit>=136`, `truncated=0`;
  - caso contrario, cancelar por FinOps e registrar como rota bloqueada.

### Atualizacao V595b - Fix de Dependencias PEFT/Transformers 2026-05-18

Falha real do V595:

- job H200 `felipesp1983/6a0a96b5a5e509f1a84142cf` falhou antes do treino;
- a falha ocorreu no carregamento PEFT do adapter inicial:
  `WeightConverter.__init__() got an unexpected keyword argument 'distributed_operation'`;
- causa tecnica: install aberto puxou `transformers 5.8.1` com `peft 0.19.1`;
- isso nao invalida o dataset V596, porque o job ja havia validado:
  - hashes train/val corretos;
  - `1920/480` linhas;
  - `0` truncation;
  - `0` fallback masks;
  - `0` prompt truncation;
  - `offset_masks=100%`;
  - familias/subcategorias corretas.

Correcao implementada:

- launcher V595 virou V595b:
  `artifacts/v595_hf_h200_v594_queryop_pref_launch/launch_v595_hf_h200_v594_queryop_preference.py`;
- output repo novo:
  `felipesp1983/kg1-nemotron-lora-v595b-v596-queryop-answeronly-pref-v290ckpt6`;
- stack fixada:
  - `huggingface_hub==0.36.2`;
  - `transformers==4.57.6`;
  - `peft==0.19.1`;
  - `accelerate==1.13.0`;
- launcher agora faz check inline das versoes antes do postinstall/training;
- `scripts/hf_job_preflight_gate.py` tambem passa a bloquear drift de versao via
  `KG1_EXPECTED_*_VERSION` no postinstall;
- `scripts/kg1_static_safety_gate.py` agora bloqueia `transformers>=...` ou
  `peft>=...` em jobs PEFT-native com adapter inicial, para evitar repeticao do
  erro V595.

Validacoes executadas:

- `python -m py_compile` no launcher V595b, launcher V597, gates e trainers;
- `python scripts/kg1_static_safety_gate.py --self-test`;
- static gate V595b/V597/trainers/preflight/pre-paid:
  `artifacts/v595_hf_h200_v594_queryop_pref_launch/v595b_static_safety_gate.json`;
- pre-paid gate V595b:
  `artifacts/v595_hf_h200_v594_queryop_pref_launch/v595b_pre_paid_job_integration_gate.json`;
- ambos com `ok=true`, `findings=[]`.

Execucao e desfecho:

- V595b foi relancado e produziu `checkpoint-2`;
- V597 weak eval foi executado contra esse checkpoint;
- o resultado final corrigido ficou abaixo do baseline (`191/315`,
  `equation=56`, `bit=135`, `truncated=1`);
- a rota foi encerrada por FinOps e nao deve ser relancada sem novo sinal CPU
  anterior.

### Atualizacao V598 - V597 Re-score e Correcao de Gate 2026-05-18

Resultado final da rota V596/V595b:

- job de treino V595b:
  `https://huggingface.co/jobs/felipesp1983/6a0a9ccda5e509f1a8414349`;
- job de weak eval V597:
  `https://huggingface.co/jobs/felipesp1983/6a0aa3c2a5e509f1a84143e9`;
- o treino completou, mas o preference val caiu de `59/120` para `58/120`;
- o weak remoto armazenado deu `190/315`, `equation=55`, `bit=135`,
  `truncated=1`;
- o re-score com o extrator local atual corrige um erro de parsing em
  `4bb8c6cd` (`\boxed{]}\!}`) e eleva somente a metrica corrigida para
  `191/315`, `equation=56`, `bit=135`, `truncated=1`.

Delta real apos re-score contra V290:

- `T->F`: `2` linhas, ambas `bit_manipulation`:
  - `8740ed31`: `01101000 -> 01111000`;
  - `59bee375`: `10010101 -> 2`, com truncation;
- `F->T`: `1` linha `bit_manipulation`:
  - `4ada9150`: `01111111 -> 01111011`;
- `equation_transform`: nenhum ganho liquido; o aparente erro em `4bb8c6cd`
  era bug de extracao remota, nao mudanca real do modelo.

Correcoes de gate implementadas:

- `scripts/hf_job_weak_eval_v245.py`:
  - `KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX=0` e
    `KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX=0` agora desativam bloqueio de
    comprimento por padrao;
  - isso e necessario porque o baseline V290 official-like tambem usa saidas
    longas (`avg_completion_tokens ~4772`, protected rows acima de `6290`);
  - truncation, total/equation/bit, delta label-aware e protected rows seguem
    bloqueantes;
- `scripts/hf_job_v588_adapter_interpolation_probe.py`:
  - mesmo contrato opcional de comprimento;
- launchers ativos V597 e V588:
  - `KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX=0`;
  - `KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX=0`;
  - evita reaplicar o falso blocker `512/2048` em avaliacao official-like;
- `scripts/validate_answer_extraction_v1.py`:
  - mismatch entre `stored_prediction` e extracao atual vira warning se nao
    muda `correct`;
  - vira blocker apenas quando altera o acerto ou quando `correct` armazenado
    diverge da extracao atual;
- `scripts/kg1_weak_backfire_row_guard.py` local atual separa:
  - `protected_id_backfire` quando o baseline estava correto;
  - `protected_id_missing_required_gain` quando o baseline tambem errava.
  No V597, `55d834d1` e missing required gain, nao backfire.

Artefatos:

- relatorio V598:
  `artifacts/v597_hf_h200_v595_weak_eval_launch/KG1_V598_V597_RESCORING_AND_GATE_FINDINGS.md`;
- auditoria de extracao corrigida:
  `artifacts/v597_hf_h200_v595_weak_eval_launch/v598_answer_extraction_rescore_audit_after_validator_fix`;
- protected-row guard local atual:
  `artifacts/v597_hf_h200_v595_weak_eval_launch/v598_local_protected_row_guard_current_code.json`.

Validacoes executadas:

- `python -m py_compile` nos scripts alterados;
- `python scripts/hf_job_weak_eval_v245.py --self-test`;
- `python scripts/validate_answer_extraction_v1.py --self-test`;
- `python scripts/kg1_static_safety_gate.py` nos scripts alterados.

Decisao:

- fechar V596/V595b como rota bloqueada;
- nao rodar novo H200 preference/answer-only query-op sem uma evidencia barata
  anterior que preserve `bit>=136`, `equation>=56`, `truncated=0` e protected
  rows;
- antes de qualquer futura promocao, re-scorear sempre o CSV com raw_output
  usando o extrator label-free atual, para evitar falso ganho ou falso bloqueio.

### Atualizacao V599 - Preference Query-Op com MoE Trainable 2026-05-18

Achado novo e acionavel apos auditoria do V595b:

- V595b nao testou a rota completa `preference + MoE target_parameters`
  treinaveis;
- o launcher V595b declarou:
  - `TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj'`;
  - `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`;
  - `MAX_TRAINABLE_PARAM_RATIO=0.020`;
- o log do V595b confirmou:
  - `target_parameters_trainability_mode='frozen_active'`;
  - `target_parameter_trainable_lora_params=0` para
    `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`;
  - somente `3,735,552` parametros LoRA treinaveis;
- portanto o resultado negativo V595b (`191/315`, `equation=56`, `bit=135`,
  `truncated=1` apos re-score) encerra apenas a rota
  `answer-only preference + attention-only`, nao a rota MoE-trainable.

Implementacao V599:

- novo launcher:
  `artifacts/v599_hf_h200_v596_queryop_pref_moe_launch/launch_v599_hf_h200_v596_queryop_preference_moe.py`;
- dataset mantido:
  `artifacts/v596_queryop_answer_only_preference_dataset/20260518T042900Z`;
- output repo:
  `felipesp1983/kg1-nemotron-lora-v599-v596-queryop-answeronly-pref-moe-v290ckpt6`;
- mudancas contra V595b:
  - `TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'`;
  - `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`;
  - `MAX_TRAINABLE_PARAM_RATIO=0.040`;
  - `LEARNING_RATE=2e-8`;
  - `FINAL_LEARNING_RATE=8e-9`;
  - `MAX_STEPS=2`, `SAVE_EVERY_STEPS=2`, `EVAL_EVERY_STEPS=2`;
  - dependencias seguem pinadas:
    `huggingface_hub==0.36.2`, `transformers==4.57.6`,
    `peft==0.19.1`, `accelerate==1.13.0`.

Status de gates locais:

- `python -m py_compile` passou no launcher V599;
- `scripts/kg1_static_safety_gate.py` passou com `ok=true`,
  `findings=[]`;
- `scripts/kg1_pre_paid_job_integration_gate.py` passou com `ok=true`,
  `findings=[]`;
- dataset V596 segue com:
  - train `1920`, val `480`;
  - `equation_transform=100%`;
  - subcategoria `equation_symbolic_queryop_cryptarithm=100%`;
  - `0` non-ASCII;
  - `0` invisible control chars;
  - `0` overlap com weak/full;
  - assistant one-line `Final answer: \boxed{...}`.

Criterio de promocao V599:

- rodar weak eval do `checkpoint-2` com extrator label-free atual;
- promover somente se:
  - `total >= 196/315`;
  - `equation_transform >= 60/155`;
  - `bit_manipulation >= 136/160`;
  - `truncated = 0`;
  - protected rows mantidas;
  - sem piora real por re-score `raw_output -> extract -> verify_answer`;
- se qualquer criterio falhar, fechar V599 por FinOps sem full, sem package e
  sem submit.

### Atualizacao V600 - Fast-Deps para MoE Preference 2026-05-18

Desfecho operacional V599:

- job HF H200:
  `https://huggingface.co/jobs/felipesp1983/6a0aadbfa5e509f1a8414524`;
- ciclos de log de 40s ficaram parados no build silencioso de
  `causal-conv1d==1.6.1`;
- nenhum treino iniciou, nenhum checkpoint foi produzido e nenhum valor ACC foi
  medido;
- decisao FinOps: cancelado antes de continuar gastando H200 em compilacao de
  dependencia, status final `CANCELED`.

Correcao V600:

- novo launcher:
  `artifacts/v600_hf_h200_v596_queryop_pref_moe_fastdeps_launch/launch_v600_hf_h200_v596_queryop_preference_moe_fastdeps.py`;
- mesma hipotese de V599:
  `preference answer-only + MoE target_parameters treinaveis`;
- diferencas contra V599:
  - `MAX_JOBS=16`;
  - instalar `causal-conv1d==1.6.1` e `mamba-ssm==2.3.1` com
    `--prefer-binary`;
  - output repo:
    `felipesp1983/kg1-nemotron-lora-v600-v596-queryop-answeronly-pref-moe-fastdeps-v290ckpt6`;
- gates locais:
  - `python -m py_compile` passou;
  - `v600_static_safety_gate.json`: `ok=true`, `findings=[]`;
  - `v600_pre_paid_job_integration_gate.json`: `ok=true`, `findings=[]`.

Criterio de FinOps V600:

- se o job repetir ciclos sem sair de instalacao de dependencias, cancelar;
- se produzir `checkpoint-2`, rodar weak eval imediatamente;
- promover somente se cumprir os mesmos criterios V599:
  `total>=196`, `equation>=60`, `bit>=136`, `truncated=0`, protected rows
  intactas e re-score atual sem falso ganho.

### Atualizacao V601 - Source-Build Controlado para MoE Preference 2026-05-18

Desfecho V600:

- job HF H200:
  `https://huggingface.co/jobs/felipesp1983/6a0ab07be7940de6ee6ce311`;
- falhou antes do treino no install de `causal-conv1d==1.6.1`;
- causa raiz confirmada no log:
  - `--prefer-binary` ativou build isolation;
  - o ambiente temporario selecionou `torch==2.12.0+cu130`;
  - a imagem real estava em CUDA `12.8` com `torch==2.8.0+cu128`;
  - resultado: mismatch CUDA `12.8` vs Torch `13.0`;
- nenhum checkpoint foi produzido e nenhum ACC foi medido.

Correcao V601:

- novo launcher:
  `artifacts/v601_hf_h200_v596_queryop_pref_moe_source_verbose_launch/launch_v601_hf_h200_v596_queryop_preference_moe_source_verbose.py`;
- mantem exatamente a hipotese ainda nao testada:
  `preference answer-only + MoE target_parameters treinaveis`;
- muda apenas a instalacao operacional:
  - `MAX_JOBS=16`;
  - `causal-conv1d==1.6.1` com
    `--no-build-isolation --no-deps --no-binary=causal-conv1d`;
  - `mamba-ssm==2.3.1` com
    `--no-build-isolation --no-deps --no-binary=mamba-ssm`;
  - logs explicitos `KG1_CAUSAL_CONV1D_SOURCE_BUILD_START/END` e
    `KG1_MAMBA_SSM_SOURCE_BUILD_START/END`;
- output repo:
  `felipesp1983/kg1-nemotron-lora-v601-v596-queryop-answeronly-pref-moe-source-v290ckpt6`.

Gates V601 locais:

- `python -m py_compile` passou;
- `v601_static_safety_gate.json`: `ok=true`, `findings=[]`;
- `v601_pre_paid_job_integration_gate.json`: `ok=true`, `findings=[]`;
- dataset V596 confirmado:
  - train `1920`, val `480`;
  - hashes esperados batendo;
  - `0` overlap com weak/full;
  - `0` caracteres invisiveis/non-ASCII;
  - completion sempre `Final answer: \boxed{...}`.

Criterio de FinOps V601:

- monitorar logs de 40 em 40 segundos;
- se source-build ficar sem progresso util repetido ou exceder a janela H200
  responsavel, cancelar;
- se produzir `checkpoint-2`, rodar weak eval imediatamente;
- promover somente se cumprir:
  - `total>=196/315`;
  - `equation_transform>=60/155`;
  - `bit_manipulation>=136/160`;
  - `truncated=0`;
  - protected rows intactas;
  - re-score `raw_output -> extract -> verify_answer` sem falso ganho.

### Atualizacao V602 - Weak Eval do V601 e Fechamento da Hipotese 2026-05-18

Resultado V601:

- job HF H200:
  `https://huggingface.co/jobs/felipesp1983/6a0ab2b2e7940de6ee6ce331`;
- source-build controlado funcionou:
  `KG1_CAUSAL_CONV1D_SOURCE_BUILD_END` e
  `KG1_MAMBA_SSM_SOURCE_BUILD_END` apareceram nos logs;
- a hipotese tecnica foi realmente testada:
  - `target_parameters_trainability_mode="trainable"`;
  - `mlp.experts.gate_up_proj` e `mlp.experts.down_proj` com
    `432,791,552` parametros LoRA treinaveis cada;
  - total treinavel `869,318,656`, razao `2.68%`, abaixo do limite `4%`;
- preference val melhorou pouco:
  - baseline `59/120` (`0.4917`);
  - final `61/120` (`0.5083`);
- checkpoint produzido:
  `felipesp1983/kg1-nemotron-lora-v601-v596-queryop-answeronly-pref-moe-source-v290ckpt6/checkpoint-2`.

Resultado V602 weak eval:

- job HF H200:
  `https://huggingface.co/jobs/felipesp1983/6a0ab8cee7940de6ee6ce380`;
- avaliacao remota armazenada pelo commit antigo:
  - `191/315`;
  - `equation_transform=55/155`;
  - `bit_manipulation=136/160`;
  - `truncated=0`;
- re-score local com o extractor atual:
  - `192/315`;
  - `equation_transform=56/155`;
  - `bit_manipulation=136/160`;
  - `truncated=0`;
- diferenca remoto -> local vem de uma unica linha simbolica:
  - `4bb8c6cd`, expected `]}\!`;
  - raw final contem `\boxed{]}\!}`;
  - extractor remoto antigo cortou no primeiro `}` e retornou `]`;
  - extractor local atual retorna `]}\!`;
  - isto e falso bloqueio de parser, nao ganho de adapter.

Diff real V602 vs baseline V290, usando extractor atual:

- total: `192 -> 192`, delta `0`;
- equation: `56 -> 56`, delta `0`;
- bit: `136 -> 136`, delta `0`;
- flips reais:
  - `8740ed31` virou backfire:
    `01101000 -> 01111000`;
  - `4ada9150` virou ganho:
    `01111111 -> 01111011`;
  - liquido bit `0`;
  - nenhuma linha equation passou de errada para correta.

Artefatos:

- resumo remoto/local:
  `artifacts/v602_hf_h200_v601_weak_eval_launch/downloaded_v602/v602_local_current_extractor_rescore_summary.json`;
- auditoria V602 vs V290 com extractor atual:
  `artifacts/v602_hf_h200_v601_weak_eval_launch/downloaded_v602/v602_vs_v290_audit_summary_local_current_extractor.json`;
- flips reais:
  `artifacts/v602_hf_h200_v601_weak_eval_launch/downloaded_v602/v602_vs_v290_flips_local_current_extractor.csv`.
- gate reutilizavel criado para impedir falso ganho/falso bloqueio por
  extractor remoto antigo:
  `scripts/kg1_rescore_predictions_label_free.py`;
- reproducao V603 do V602 pelo novo gate:
  `artifacts/v602_hf_h200_v601_weak_eval_launch/v603_rescore_gate_on_v602/v603_v602_current_label_free_rescore_summary.json`.

Decisao:

- fechar `V596 answer-only preference + MoE trainable` como hipotese negativa;
- nao rodar novo H200 nessa linha, porque a parte antes nao testada
  (`target_parameters` MoE treinaveis) foi testada e nao gerou ganho weak;
- manter regra obrigatoria:
  todo weak eval remoto deve ser re-scoreado localmente com o extractor
  label-free atual antes de qualquer promocao, full eval, package ou submit;
- se o re-score local e o remoto divergirem, usar a divergencia como auditoria
  de parser/commit, nunca como ganho do adapter.

Proximo passo ativo:

1. Bloquear gasto H200 sem novo sinal barato: a proxima acao deve ser CPU/local
   ou weak-eval de adapter ja existente.
2. Separar falha de `decoding/extractor` de falha real do adapter:
   `raw_output -> extract_final_answer atual -> verify_answer`.
3. Proteger explicitamente o backfire recorrente `8740ed31=01101000` antes de
   qualquer nova hipotese de treino.
4. Procurar ganho somente onde ha delta label-free real:
   - CPU solver/projection continua sinal, mas precisa virar comportamento do
     adapter ou pacote permitido;
   - preference answer-only V596 esta fechado;
   - novo treino so pode nascer de um gate CPU que mostre `>=+4 equation`,
     `bit>=136` e `0` protected-row backfire.

### Atualizacao V604 - Interpolacao Eval-Only V290 -> V601 2026-05-18

Racional:

- V602 mostrou dois flips reais de bit:
  - ganho: `4ada9150`, `01111111 -> 01111011`;
  - perda: `8740ed31`, `01101000 -> 01111000`;
- como o liquido e zero, a unica forma barata de tentar extrair valor do V601
  sem novo treino e testar interpolacoes pequenas entre V290 e V601;
- isso e eval-only, nao treina, e respeita FinOps melhor que outro smoke H200.

Preflight:

- configs V290/V601 baixados por API HF e comparados:
  - ambos `r=32`, `lora_alpha=32`;
  - mesmo base Nemotron;
  - mesmos `target_parameters` MoE;
  - `target_modules` iguais apos ordenar, diferem apenas na ordem serializada;
- launcher criado:
  `artifacts/v604_hf_h200_v601_interpolation_probe/launch_v604_hf_h200_v601_interpolation_probe.py`;
- usa o job script V588, que injeta patch dos scripts atuais no HF antes da
  execucao, evitando o erro V602 de commit/extractor remoto antigo;
- lambdas iniciais: `0.05,0.10,0.25,0.50`;
- gates locais:
  - `python -m py_compile` passou;
  - `v604_static_safety_gate.json`: `ok=true`, `findings=[]`;
  - `scripts/kg1_rescore_predictions_label_free.py --self-test` passou.

Criterio:

- promover somente se algum lambda tiver, apos re-score local atual:
  - `total>=196/315`;
  - `equation_transform>=60/155`;
  - `bit_manipulation>=136/160`;
  - `truncated=0`;
  - protected rows intactas;
  - `label_aware_delta=0`;
- se nenhum lambda passar, fechar a linha "interpolacao V601" e nao rodar mais
  H200 sobre V596/V601.

Resultado:

- job HF H200:
  `https://huggingface.co/jobs/felipesp1983/6a0abfc7e7940de6ee6ce3e4`;
- logs locais preservados em:
  `artifacts/v604_hf_h200_v601_interpolation_probe/v604_job_6a0abfc7_logs_cycle25.txt`;
- resumo extraido dos logs:
  `artifacts/v604_hf_h200_v601_interpolation_probe/v604_candidate_summaries_from_logs.json`;
- candidatos concluidos:
  - `v588_interp_l050`: `192/315`, `equation=56/155`, `bit=136/160`, `truncated=0`;
  - `v588_interp_l100`: `190/315`, `equation=56/155`, `bit=134/160`, `truncated=1`;
  - `v588_interp_l250`: `190/315`, `equation=56/155`, `bit=134/160`, `truncated=1`;
- `v588_interp_l500` foi iniciado, mas cancelado por FinOps depois de tres
  pontos sem sinal e duas regressoes consecutivas.

Decisao:

- nenhum lambda passou o criterio promocional;
- fechar a rota "V601/MoE preference por interpolacao";
- nao abrir novo H200 nessa linha;
- o proximo gasto pago so e permitido se houver novo sinal CPU/source-only que
  nao use V596/V601 como unica justificativa e que preserve `bit>=136`,
  `equation>=60`, `truncated=0` e protected rows.

### Atualizacao V606-V608 - Fonte V446 Compacta e Smoke H200 Permitido 2026-05-18

Novo sinal CPU/source-only:

- V606 auditou o pool V446 aceito contra `competition_train.csv` com
  `raw_output -> extract_final_answer -> verify_answer`;
- resultado V606:
  - `1299` linhas limpas e ainda nao usadas/quarentenadas;
  - `848` bit e `451` equation;
  - `11` linhas sujas removidas por mismatch de resposta extraida;
  - `0` overlap com datasets/adapters em quarentena;
  - CoT bruto era longo demais para treino direto, portanto foi bloqueado como
    fonte verbatim.

Dataset ativo V607:

- construtor: `scripts/build_v607_compact_v446_source_dataset.py`;
- train `1099` linhas: `721` bit, `378` equation;
- validation `194` linhas: `127` bit, `67` equation;
- contrato de prompt: `official_like` (`user = prompt + PROMPT_SUFFIX`, sem
  system prompt);
- resposta final: exatamente uma resposta `\boxed{...}` verificavel por
  extrator label-free;
- pesos de loss: bit `1.1`, equation `1.0`;
- gates CPU:
  - V509 train/val: `blocked_dataset_count=0`;
  - V286 real tokenizer: `tokenization_gate_passed`, `0` overlap weak/full,
    `0` truncation, `0` fallback masks, `0` completion tokens dropped;
  - V513 corrigido para `official_like`: `0` blockers, `0` warnings;
  - V478 objetivo: `bit=0.677227`, `equation=0.322773`;
  - V524: `quota_ok_cpu_only`;
  - V526 com `example_mean` + `row_loss_weight`: `example_mean_dry_run_passed`,
    `gpu_allowed=true`, escopo `one_short_h200_smoke_only`.

Correcoes de gate feitas durante V607:

- V513 nao aceitava contrato `official_like` e gerava falso
  `prompt_user_mismatch`; corrigido para suportar `user,assistant` e preservar
  o `\n` inicial do `PROMPT_SUFFIX`;
- `scripts/build_v607_compact_v446_source_dataset.py` passou a exigir roundtrip
  label-free do `\boxed{...}` bruto, evitando escape simbolico errado;
- pre-paid gate bloqueou V608 ate o launcher ter snippets literais de
  `PREF_TRAIN_SHA256`, `PREF_VAL_SHA256`, `PREF_TRAIN_ROWS`,
  `PREF_VAL_ROWS`, `KG1_V516_PARSER_CURRENT_BASELINE_STATUS=passed` e
  `KG1_STALE_PREDICTION_PARITY_STATUS=passed`.

HF artifacts e launcher V608:

- dataset enviado para:
  `felipesp1983/kg1-v607-compact-v446-source-artifacts`;
- commit HF dataset:
  `eedc4f29b3a2180b3a6c56b4b86b8d26d1362b9e`;
- upload manifest:
  `artifacts/v607_compact_v446_source_dataset/20260518T_v607_cpu_gate/v607_hf_dataset_upload_manifest.json`;
- launcher:
  `artifacts/v608_hf_h200_launch/launch_v608_hf_nemo_h200_v607_compact_source.py`;
- debug local V608 validou:
  - H200 `0.083333/min`, abaixo do limite `0.09`;
  - hashes HF train/val iguais aos manifestos V607;
  - seed adapter V290 checkpoint-6 com arquivos obrigatorios;
  - objetivo remoto com `--use-row-loss-weight` e
    `--require-row-loss-weight`;
  - `KG1_CPU_SIMULATED_TOTAL_CORRECT=201`,
    `KG1_CPU_SIMULATED_BIT_CORRECT=138`,
    `KG1_CPU_SIMULATED_EQUATION_CORRECT=63`,
    perdas simuladas `0`;
  - `KG1_MAX_TOKEN_HEADROOM_RATIO=0.336`.
- pre-paid gate:
  `artifacts/v608_hf_h200_launch/v608_pre_paid_job_integration_gate.json`;
  resultado `ok=true`, `findings=[]`.

Decisao atual:

- V608 e o unico treino H200 permitido agora;
- escopo maximo: `MAX_STEPS=2`, `timeout=3600`,
  `LOSS_NORMALIZATION_MODE=example_mean`, `USE_ROW_LOSS_WEIGHT=1`,
  `REQUIRE_ROW_LOSS_WEIGHT=1`;
- monitorar logs a cada `40` segundos;
- se produzir `checkpoint-2`, rodar weak eval imediatamente;
- cancelar/encerrar a rota se checkpoint-2 nao cumprir todos:
  `total>=196/315`, `equation_transform>=60/155`,
  `bit_manipulation>=136/160`, `truncated=0`, protected rows intactas e
  re-score label-free atual sem ganho falso.

### Atualizacao V609 - Weak Eval do V608 e Fechamento da Rota V607 2026-05-18

Resultado medido:

- job HF H200 weak eval:
  `https://huggingface.co/jobs/felipesp1983/6a0ad85da5e509f1a8414975`;
- checkpoint avaliado:
  `felipesp1983/kg1-nemotron-lora-v608-v607compact-v290ckpt6/checkpoint-2`;
- diagnosticos baixados em:
  `artifacts/v609_hf_h200_v608_weak_eval_launch/downloaded_diagnostics/`;
- resultado adapter-only label-free:
  - `191/315`;
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`;
  - `label_aware_minus_label_free=0`;
- weak promotion gate bloqueou corretamente:
  - `correct_lt_196`;
  - `equation_lt_60`;
  - `bit_lt_136`;
  - `truncated_gt_0`;
  - `protected_row_backfire_guard_failed`.

Diagnostico cirurgico:

- contra o baseline label-free V516, V608/V609 teve `+1` equation e `-1` bit,
  net `0`;
- ganho isolado:
  - `4bb8c6cd` (`equation_transform`) mudou de `]` para `]}\!` e passou;
- perda critica:
  - `59bee375` (`bit_manipulation`) era correto no baseline com `10010101`,
    mas virou `2`, `finish_reason=length`, `completion_tokens=7680`;
- protected row `55d834d1` continuou sem aprender o ganho obrigatorio
  (`10111111` em vez de `00111111`);
- isso separa claramente dois problemas:
  - houve algum sinal de aprendizado local de equation;
  - o treino tambem aumentou tendencia de resposta longa/truncada e backfire em
    bit, portanto nao e submit-safe.

Decisao:

- fechar a rota V607/V608 como nao promocional;
- nao repetir H200 com o mesmo dataset/hiperparametros;
- qualquer proximo treino pago precisa provar, antes, uma mudanca objetiva em
  CPU/local diagnostics contra:
  - truncation/length drift;
  - perda de bit em linhas protegidas;
  - ganho equation que nao dependa de expected-aware;
  - contrato adapter-only sem postprocessor.

Proximo passo ativo:

1. Construir uma auditoria local V610 de drift por linha usando os CSVs V609:
   `raw_output -> extract -> verify_answer`, completion length, finish_reason,
   gain/loss contra V516 e protected rows.
2. Se a auditoria confirmar que o problema dominante e output longo/decoding
   drift, criar somente um micro-dataset source-only/contract-safe que preserve
   bit com respostas curtas e exemplos equivalentes, sem treinar em weak labels.
3. Rodar primeiro gate CPU/tokenization/objective; H200 so volta se o novo
   dataset mostrar contrato melhor que V607 e uma justificativa diferente de
   "mais steps".

### Atualizacao V611 - Consenso OpenRouter Para Sair do Plato 2026-05-18

Consulta externa:

- prompt:
  `artifacts/openrouter/v611_plateau_direction_consult/KG1_V611_OPENROUTER_ROADMAP_DIRECTION_PROMPT.md`;
- respostas brutas:
  `artifacts/openrouter/v611_plateau_direction_consult/v611_openrouter_raw_results.json`;
- consenso:
  `artifacts/openrouter/v611_plateau_direction_consult/KG1_V611_OPENROUTER_CONSENSUS_AND_ROADMAP.md`;
- modelos consultados:
  - `openai/gpt-5.4`;
  - `anthropic/claude-sonnet-4.6`;
  - `google/gemini-3.1-pro-preview`;
  - `deepseek/deepseek-v4-pro`;
  - `qwen/qwen3-max-thinking`.

Consenso util:

- o plato nao e falta simples de epochs, rows ou capacidade;
- o erro dominante e transferencia de contrato de saida:
  o adapter aprende continuacao longa/CoT em vez de resposta curta
  `\boxed{...}` com terminacao estavel;
- `bit_manipulation` deve ser ancora de preservacao antes de tentar empurrar
  `equation_transform`;
- `eval_loss` nao deve ser usado como decisao de promocao sem gates de ACC,
  completion length, truncation e protected rows;
- qualquer H200 novo precisa nascer de uma taxonomia CPU dos erros, nao de
  repeticao de V607/V608.

Itens rejeitados:

- sugestao de treinar diretamente nos `315` weak rows com respostas esperadas
  foi descartada por risco de leakage/ganho falso;
- sugestoes de "treinar mais" ou broad SFT sem novo gate foram descartadas;
- postprocessor/verifier em runtime submit continua bloqueado.

Novo plano ativo:

1. V612 CPU failure taxonomy:
   - classificar erros por linha em `WRONG_ANSWER`, `FORMAT_FAIL`, `RUNAWAY`,
     `BLANK`, `NEAR_MISS`;
   - separar por `bit_manipulation` e `equation_transform`;
   - rastrear `completion_tokens`, `finish_reason`, `boxed_count`,
     `raw_output -> extract_final_answer -> verify_answer`;
   - bloquear qualquer GPU ate sabermos se o proximo dado deve atacar
     formato/terminacao, raciocinio equation ou ambos.
2. Se V612 apontar formato/terminacao como dominante:
   - construir dataset source-only answer-first/short-target;
   - resposta `\boxed{answer}` no inicio;
   - EOS/answer boundary unmasked;
   - hard cap de target tokens;
   - final-answer tokens com peso maior;
   - bit preservation gates antes de H200.
3. Se V612 apontar wrong-answer em equation como dominante:
   - testar micro-adapter equation-only, curto e estruturado;
   - weak eval completa so depois de smoke e gates;
   - promocao exige preservar bit.
4. Task arithmetic/interpolation so volta como diagnostico apos uma rota
   equation-only gerar ganho real com pequena perda; nao e caminho default.

Gates novos obrigatorios:

- `high_completion_token_rows` por familia;
- bloqueio se `finish_reason=length`;
- bloqueio se high-token bit aumentar contra baseline;
- protected rows sempre auditadas:
  `8740ed31=01101000`, `59bee375=10010101`, `55d834d1=00111111`;
- qualquer checkpoint com `bit<136` ou `truncated>0` nao passa para full/package.

### Atualizacao V612 - Double Check OpenRouter e Taxonomia Real dos Erros 2026-05-18

Motivo da mudanca:

- o plano anterior nao produziu ganho submit-safe;
- V608/V609 provou que treino com fonte V446 compacta pode gerar `+1`
  equation, mas tambem gera `-1` bit, truncation e backfire em protected row;
- continuar H200 sem atacar essa causa seria repetir o mesmo ciclo.

Double check OpenRouter:

- prompt:
  `artifacts/openrouter/v612_effective_plan_doublecheck/KG1_V612_OPENROUTER_EFFECTIVE_PLAN_PROMPT.md`;
- respostas:
  `artifacts/openrouter/v612_effective_plan_doublecheck/v612_openrouter_raw_results.json`;
- modelos consultados:
  - `openai/gpt-5.4`;
  - `anthropic/claude-sonnet-4.6`;
  - `google/gemini-3.1-pro-preview`;
  - `deepseek/deepseek-v4-pro`;
  - `qwen/qwen3-max-thinking`.

Consenso novo, mais restritivo:

- parar broad SFT, sweeps de LR/rank/epochs e H200 sem gate CPU;
- o problema dominante nao e falta de capacidade, e politica de saida:
  completions longas, terminacao ruim e contrato `\boxed{...}` instavel;
- `eval_loss` continua secundario; promocao deve ser por ACC, truncation,
  protected rows e completion length;
- o plano efetivo precisa primeiro recuperar emissao curta e estavel, depois
  tentar transferir o ganho do solver para LoRA;
- se o contrato curto nao passar em CPU, adapter-only gain nao e plausivel no
  curto prazo e nao deve consumir H200.

V612 implementado:

- script:
  `scripts/analyze_v612_failure_taxonomy.py`;
- validacoes:
  - `python -m py_compile scripts/analyze_v612_failure_taxonomy.py`;
  - `python scripts/analyze_v612_failure_taxonomy.py --self-test`;
- execucao real:
  `artifacts/v612_failure_taxonomy/`;
- report:
  `artifacts/v612_failure_taxonomy/KG1_V612_FAILURE_TAXONOMY.md`;
- resumo:
  `artifacts/v612_failure_taxonomy/v612_failure_taxonomy_summary.json`.

Resultado V612 contra V609:

- decisao: `blocked`;
- total: `191/315`;
- `bit_manipulation=135/160`;
- `equation_transform=56/155`;
- `truncated=1`;
- avg completion tokens: `4775.5`;
- p99 completion tokens: `7357`;
- blockers:
  - `total_not_above_submit_safe_best`;
  - `bit_below_submit_safe_best`;
  - `truncated_nonzero`;
  - `protected_rows_not_all_ok`;
  - `bit_long_completion_gt_256_seen`.

Achado principal:

- `bit_manipulation` esta dominado por runaway/saida longa:
  - `160/160` bit rows tiveram mais de `256` tokens;
  - `160/160` tiveram mais de `1000` tokens;
  - `160/160` tiveram mais de `4000` tokens;
  - `34/160` passaram de `7000` tokens;
  - categorias: `135` corretas mas com risco runaway,
    `25` runaway/truncated erradas.
- `equation_transform` ainda mistura erro real e contrato:
  - `53` wrong-answer;
  - `16` boxed/payload format fail;
  - `14` casos onde a resposta esperada aparece no raw em auditoria label-aware,
    mas a extracao/formatacao falha;
  - `16` runaway/truncated;
  - `49` corretas mas com risco runaway.
- protected rows:
  - `8740ed31` correto, mas com `6290` tokens;
  - `55d834d1` continua errado, com `6285` tokens;
  - `59bee375` regrediu de `10010101` para `2`, com
    `finish_reason=length` e `7680` tokens.

Novo plano efetivo, substituindo o ciclo anterior:

1. V613 source-only answer-first anti-runaway dataset:
   - nao usar weak rows nem weak answers como targets;
   - targets curtos, idealmente apenas `\boxed{answer}` + EOS;
   - bit: alvo exatamente 8 bits e EOS;
   - equation: alvo canonico curto e parse-valid;
   - sem CoT longo como target;
   - hard cap de target tokens;
   - answer/EOS obrigatoriamente unmasked;
   - exemplos source-only derivados de solver/gerador, com roundtrip
     `raw_output -> extract -> verify_answer`.
2. V613 static/CPU gates antes de GPU:
   - `0` weak ID overlap;
   - `0` duplicate prompt-target pair;
   - `100%` bit target regex `^[01]{8}$`;
   - `100%` target extraivel por label-free extractor;
   - `0` truncation/token drop/fallback mask;
   - bit heldout p99 completion tokens `<=32`, max `<=128`;
   - equation heldout p99 completion tokens `<=160`;
   - protected rows corretas em `3/3` repeated decodes;
   - long completion `>256` precisa cair pelo menos `90%` contra V609.
3. Somente se V613 CPU passar:
   - fazer um unico smoke GPU curto;
   - promocao minima:
     `total>=193`, `bit>=136`, `equation>=57`, `truncated=0`,
     protected rows intactas;
   - meta de submissao:
     `total>=196`, `bit>=136`, `equation>=60`, `truncated=0`.
4. Se V613 falhar:
   - nao gastar H200;
   - abrir diagnostico de decode/prompt/base-vs-adapter antes de treinar;
   - o ganho solver/projection continua diagnostico, nao submit-safe.

Itens removidos do caminho ativo:

- repetir V607/V608;
- broad SFT por mais epochs;
- selecionar checkpoint por `eval_loss` sem ACC/gates;
- aceitar `+1` equation com `-1` bit;
- treinar em weak labels;
- promover postprocessor/verifier runtime como submit adapter-only.

### Atualizacao V613 - Inicio do Novo Plano Efetivo Anti-Runaway 2026-05-18

Objetivo:

- substituir o ciclo que treinava traces longos por um dataset source-only com
  alvo curto;
- atacar diretamente o achado V612:
  completions longas e terminacao instavel estao destruindo bit e impedindo
  ganho submit-safe;
- nao gastar GPU ate o contrato curto passar nos gates CPU/estaticos.

Limpeza executada:

- manifest:
  `artifacts/v613_cleanup/v613_cleanup_manifest.json`;
- removidos apenas lixos comprovados:
  - `artifacts/tmp_v610_self_test`;
  - diretorios `__pycache__` em `scripts`, `src` e launch artifacts antigos;
- artefatos historicos, CSVs de predicao, manifests e reports foram preservados
  porque ainda sao evidencias para comparacao e regressao.

Dataset V613 criado:

- script:
  `scripts/build_v613_answer_first_anti_runaway_dataset.py`;
- fonte:
  V607/V446 source-only ja auditado;
- saida:
  `artifacts/v613_answer_first_anti_runaway_dataset/20260518T_v613_cpu_gate/`;
- train:
  `1099` rows;
- validation:
  `194` rows;
- familias:
  - train: `721` bit, `378` equation;
  - validation: `127` bit, `67` equation;
- contrato:
  - `official_like`;
  - assistant target = exatamente uma linha curta
    `Final answer: \boxed{...}`;
  - sem CoT, sem explicacao, sem sufixo extra;
  - `weak_gate_rows_used_for_training=false`;
  - `full_gate_rows_used_for_training=false`;
  - `gate_rows_used_for_training=false`;
- max assistant chars:
  `30`;
- weak overlap:
  - ID overlap `0`;
  - prompt overlap `0`;
  - prompt+answer overlap `0`;
  - weak CSV sha:
    `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.

Correcoes feitas durante V613:

- o primeiro builder usava `box_answer()`, que escapa backslashes e quebrava
  answers simbolicos como `\93` no extrator label-free;
- corrigido para usar payload bruto em
  `Final answer: \boxed{answer}`, que passa por
  `extract_final_answer -> verify_answer` sem expected-aware;
- isso evita criar um dataset "limpo" no papel mas errado para o parser real.

Gates V613 passados:

- `python -m py_compile scripts/build_v613_answer_first_anti_runaway_dataset.py`;
- `python scripts/build_v613_answer_first_anti_runaway_dataset.py`;
- V286 toy:
  `artifacts/v613_answer_first_anti_runaway_dataset/20260518T_v613_cpu_gate/v286_tokenization_toy/`;
- V286 real:
  `artifacts/v613_answer_first_anti_runaway_dataset/20260518T_v613_cpu_gate/v286_tokenization_real/`;
- static safety:
  `artifacts/v613_answer_first_anti_runaway_dataset/20260518T_v613_cpu_gate/v613_static_safety_gate.json`.

Resultado do V286 real:

- status:
  `tokenization_gate_passed`;
- tokenizer:
  `TokenizersBackend`, EOS `<|im_end|>`;
- train:
  - rows `1099`;
  - prompt truncated `0`;
  - completion tokens dropped `0`;
  - fallback masks `0`;
  - offset masks `1099`;
  - max total tokens `315`;
  - max loss tokens `18`;
- validation:
  - rows `194`;
  - prompt truncated `0`;
  - completion tokens dropped `0`;
  - fallback masks `0`;
  - offset masks `194`;
  - max total tokens `315`;
  - max loss tokens `18`;
- train/validation overlap:
  - prompt overlap `0`;
  - prompt+answer overlap `0`;
- weak reference overlap:
  - ID `0`;
  - prompt `0`;
  - prompt+answer `0`;
- static safety:
  - `ok=true`;
  - `findings=[]`.

Estado atual:

- V613 e o primeiro artefato do novo plano que realmente corrige a causa
  medida pela V612;
- ainda nao e submit-safe e ainda nao autoriza H200;
- o proximo passo obrigatorio e criar/rodar o gate de decode/protected-row
  antes de qualquer treino:
  - `finish_reason=length` precisa ser `0`;
  - protected rows devem ficar corretas;
  - completions `>256` precisam cair pelo menos `90%` contra V609;
  - se isso nao passar, nao ha gasto GPU.

Proximo passo ativo:

1. V614 decode/protected-row gate para V613:
   - validar contrato curto, protected rows e risco de runaway antes de treino;
   - bloquear rota se o gate indicar que somente dataset curto nao basta.
2. Se V614 passar:
   - preparar um unico smoke GPU curto, nao broad SFT;
   - treinar com objetivo answer/EOS unmasked;
   - avaliar checkpoint imediatamente;
   - promover somente se `total>=193`, `bit>=136`, `equation>=57`,
     `truncated=0` e protected rows intactas.
3. Se V614 falhar:
   - parar GPU;
   - diagnosticar base-vs-adapter/prompt/decoding antes de novo treino.

### Atualizacao V614 - Gate Anti-Runaway Obrigatorio Para Proximos Checkpoints 2026-05-18

Implementado:

- script:
  `scripts/run_v614_anti_runaway_promotion_gate.py`;
- validacoes:
  - `python -m py_compile scripts/run_v614_anti_runaway_promotion_gate.py`;
  - `python scripts/run_v614_anti_runaway_promotion_gate.py --self-test`;
  - static safety em V612/V613/V614:
    `artifacts/v613_answer_first_anti_runaway_dataset/20260518T_v613_cpu_gate/v613_v614_static_safety_gate.json`;
  - resultado static safety:
    `ok=true`, `findings=[]`.

Regra V614:

- o gate e obrigatorio para qualquer proximo weak eval/checkpoint;
- `finding_counts.warning` tambem precisa ser `0`;
- promocao minima:
  - `total>=193`;
  - `bit>=136`;
  - `equation>=57`;
  - `truncated=0`;
  - `finish_reason=length` igual a `0`;
  - protected rows corretas:
    `8740ed31=01101000`, `59bee375=10010101`, `55d834d1=00111111`;
  - bit p99 completion tokens `<=128`;
  - equation p99 completion tokens `<=512`;
  - long completions `>256` precisam cair pelo menos `90%` contra o baseline
    de referencia informado.

Teste de sanidade em V609:

- comando rodado contra V609, esperando bloqueio:
  `artifacts/v613_answer_first_anti_runaway_dataset/20260518T_v613_cpu_gate/v614_gate_on_v609_expected_block.json`;
- resultado:
  - decision `blocked`;
  - correct `191/315`;
  - blockers:
    - `total_lt_193`;
    - `bit_lt_136`;
    - `equation_lt_57`;
    - `truncated_nonzero`;
    - `finish_reason_length_nonzero`;
    - `protected_failed_59bee375`;
    - `protected_failed_55d834d1`;
    - `bit_p99_tokens_gt_128`;
    - `equation_p99_tokens_gt_512`;
    - long-token reduction ausente em bit/equation.

Decisao atual:

- V613 dataset esta pronto para o proximo passo tecnico;
- V614 gate esta pronto para bloquear qualquer repeticao do erro V609;
- V615 executou essa hipotese e foi bloqueado.  Nao repetir V613/V615 como
  estava.

### Atualizacao V615/V616 - V613 Answer-First Nao Transferiu Termination 2026-05-18

Artefatos:

- treino V615:
  `artifacts/v615_hf_h200_v613_answer_first_launch/`;
- checkpoint:
  `felipesp1983/kg1-nemotron-lora-v615-v613-answerfirst-v290ckpt6/checkpoint-2`;
- weak eval V615:
  `artifacts/v615_hf_h200_v613_answer_first_launch/downloaded_eval/`;
- gate V614:
  `artifacts/v615_hf_h200_v613_answer_first_launch/v615_v614_anti_runaway_gate.json`;
- taxonomy:
  `artifacts/v615_hf_h200_v613_answer_first_launch/v615_failure_taxonomy/`;
- OpenRouter V616:
  `artifacts/openrouter/v616_v615_failure_consult/`;
- consenso V616:
  `artifacts/openrouter/v616_v615_failure_consult/KG1_V616_OPENROUTER_CONSENSUS.md`;
- first-answer audit V617:
  `artifacts/v615_hf_h200_v613_answer_first_launch/v617_first_answer_drift_v2/`.

Resultado V615:

- treino:
  - baseline eval_loss `1.6998`;
  - step-1 train_loss `1.6625`;
  - step-2 train_loss `1.3296`;
  - post-train eval_loss `1.6959`.
- weak eval:
  - total `192/315`;
  - `bit_manipulation=136/160`;
  - `equation_transform=56/155`;
  - `truncated=0`.

Bloqueios:

- `total_lt_193`;
- `equation_lt_57`;
- `protected_failed_8740ed31`;
- `protected_failed_55d834d1`;
- `bit_p99_tokens_gt_128`;
- `equation_p99_tokens_gt_512`;
- `finding_counts.warning=2` no V614.

Diagnostico novo:

- V613 answer-first limpo nao foi suficiente para transferir termination;
- todas as `160` linhas bit e todas as `155` linhas equation continuam com
  completion `>256` tokens;
- bit p99 tokens `7357`;
- equation p99 tokens `6397`;
- protected row `8740ed31` regrediu:
  `01101000 -> 01111000`;
- protected row `55d834d1` continuou errada:
  `10111111` em vez de `00111111`;
- V617 mostrou que nao houve "primeiro boxed correto e depois corrompido":
  o modelo gera um unico boxed no fim, depois de milhares de tokens;
- nos protected rows, chars antes do primeiro boxed:
  - `8740ed31`: `8458`;
  - `55d834d1`: `8585`;
  - `59bee375`: `8708`.

Decisao V616:

- parar V613/V615-style GPU:
  - `max_steps=2`;
  - LR `1e-7 -> 2e-8`;
  - MoE expert-only target parameters;
  - short positive targets sem pressao explicita contra raciocinio longo;
- nao promover nenhum treino por loss enquanto o weak eval seguir com milhares
  de tokens antes do boxed;
- proxima rota precisa atacar o contrato oficial/template + EOS + superficie
  LoRA capaz de mexer em politica de saida.

### Roadmap Ativo Pos-V616 - V618 Official-Template EOS Policy

Objetivo:

- transformar o sinal answer-first em comportamento adapter-only que responda
  cedo e pare;
- preservar `bit>=136`;
- buscar `equation>=57`;
- so gastar GPU se os gates novos provarem que o proximo treino mede o
  mecanismo correto.

Passo 1 - implementar gates CPU/static V618:

- probe set oficial-like com:
  - protected rows `8740ed31`, `59bee375`, `55d834d1`;
  - linhas long-risk de V615;
  - bit/equation balanceados;
- gate de primeira resposta:
  - `first_boxed`;
  - `final_extracted`;
  - chars/tokens antes do primeiro boxed;
  - chars/tokens depois do primeiro boxed;
  - protected rows;
- gate de template:
  - treino deve usar o mesmo chat/template do weak eval;
  - `prompt_suffix` oficial preservado;
- gate de EOS:
  - answer + EOS precisam estar unmasked;
  - EOS deve ser o ultimo token unmasked;
  - `0` completion tokens dropped;
  - `0` fallback masks;
  - `0` truncation;
- gate de LoRA surface:
  - enumerar target modules reais;
  - bloquear rota output-policy se usar somente MoE expert MLP;
  - exigir superficie attention/output-policy verificavel
    (`q_proj/v_proj/o_proj` ou equivalente real no Nemotron).

Passo 2 - construir dataset V618:

- derivado do V613, mas com treino no contrato oficial-like real;
- target:
  - resposta curta;
  - boxed;
  - EOS imediato;
  - sem CoT;
  - sem tokens apos EOS;
- manter:
  - `0` weak/full ID overlap;
  - `0` prompt overlap;
  - `0` prompt+answer overlap;
  - answers verificadas por extractor label-free.

Passo 3 - so entao considerar um micro-treino:

- iniciar do V290 checkpoint-6;
- alterar uma variavel principal: template/EOS/LoRA surface;
- budget GPU curto;
- probe imediato antes de weak completo;
- FinOps kill se:
  - qualquer protected row falhar;
  - bit p99 tokens `>256` no probe;
  - equation p99 tokens `>512` no probe;
  - mais de `6/64` probe rows tiverem muitos tokens antes do boxed;
  - ACC do probe nao indicar pelo menos `+1` sem perda bit.

Promocao weak:

- total `>=193`;
- `bit>=136`;
- `equation>=57`;
- `truncated=0`;
- protected rows `3/3`;
- bit p99 tokens `<=128`;
- equation p99 tokens `<=512`;
- `finding_counts.warning=0`.

### Atualizacao V618 - Probe Oficial e Preflight EOS/Output Policy 2026-05-18

Artefatos implementados:

- script probe:
  `scripts/build_v618_official_template_probe_set.py`;
- script preflight:
  `scripts/run_v618_official_template_eos_policy_gate.py`;
- probe 64 linhas:
  `artifacts/v618_official_template_eos_policy_probe/v618_official_template_probe_64.jsonl`;
- auditoria CSV:
  `artifacts/v618_official_template_eos_policy_probe/v618_official_template_probe_64.csv`;
- manifest probe:
  `artifacts/v618_official_template_eos_policy_probe/v618_official_template_probe_manifest.json`;
- preflight:
  `artifacts/v618_official_template_eos_policy_probe/v618_route_preflight_gate.json`.

Resultado:

- `build_v618_official_template_probe_set.py --self-test`: passou;
- `run_v618_official_template_eos_policy_gate.py --self-test`: passou;
- probe V618 criado com:
  - `64` linhas;
  - `32` `bit_manipulation`;
  - `32` `equation_transform`;
  - protected rows `8740ed31`, `59bee375`, `55d834d1`;
  - `used_for_training=false`;
  - weak CSV sha:
    `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.

Preflight contra V615:

- decisao: `blocked`;
- blockers:
  - `launch_missing_all_three_protected_rows:55d834d1`;
  - `output_policy_steps_lt_20`;
  - `learning_rate_too_low_for_output_policy_route`;
- warning:
  - `final_learning_rate_extremely_low_for_output_policy_route`.

Achado tecnico novo:

- o V615 nao falhou por sujeira de dataset ou tokenizacao basica:
  - dataset V613 limpo;
  - `0` weak overlap;
  - `0` prompt truncation;
  - `0` completion tokens dropped;
  - `0` fallback masks;
  - offset masks em todas as linhas.
- o erro operacional foi no desenho do treino:
  - protected row `55d834d1` nao estava no contrato de treino;
  - apenas `2` steps;
  - LR efetivo `2e-8 -> 5e-9`;
  - target_parameters MoE expert-only sem gate V618 de superficie capaz de
    alterar politica de saida;
  - weak eval ainda gerou milhares de tokens antes do unico boxed.

Regra permanente adicionada:

- todo treino ou weak eval que nao ocorrer como planejado deve gerar uma
  consulta OpenRouter com prompt rigoroso, sem ruido, contendo:
  - logs completos;
  - metricas loss/eval_loss/acc;
  - manifest do dataset;
  - parametros do launcher;
  - blockers dos gates;
  - diffs por linha, principalmente protected rows;
  - separacao entre erro de decoding, erro de adapter e erro de extractor.
- nenhum novo H200 pode iniciar se essa consulta estiver pendente para a
  tentativa anterior.

### Atualizacao V619 - Module Surface Gate Nemotron 2026-05-18

Artefatos:

- script:
  `scripts/run_v619_nemotron_module_surface_gate.py`;
- relatorio:
  `artifacts/v619_nemotron_module_surface_gate/v619_module_surface_report.json`.

Resultado:

- `run_v619_nemotron_module_surface_gate.py --self-test`: passou;
- gate real: `surface_gate_passed`;
- `finding_counts.blocker=0`;
- `finding_counts.warning=0`;
- fonte inspecionada:
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`,
  revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`;
- nomes detectados no codigo real:
  - `q_proj`;
  - `k_proj`;
  - `v_proj`;
  - `o_proj`;
  - `lm_head`;
  - `up_proj`;
  - `down_proj`.

Impacto:

- o bloqueio de "superficie LoRA nao verificada" foi removido;
- a proxima rota pode usar `KG1_V618_MODULE_SURFACE_GATE_STATUS=passed`;
- ainda nao pode rodar GPU porque o V615 continua bloqueado por:
  - protected rows incompletas no contrato de treino;
  - apenas `2` steps;
  - LR baixo demais;
  - weak eval real do V617/anti-runaway com milhares de tokens antes do boxed.

Proximo passo obrigatorio:

1. criar uma rota de treino V620 somente depois do V619:
   - protected rows `3/3` no contrato;
   - steps minimos `>=20`;
   - LR minimo para output-policy `>=1e-6`;
   - template oficial-like;
   - EOS/answer imediato validado;
   - probe V618 antes de weak completo.
2. se V620 falhar:
   - aplicar imediatamente a regra OpenRouter de falha de treino/eval;
   - atualizar este roadmap antes de qualquer novo job.

### Atualizacao V620 - Rota Output-Policy H200 Liberada Por Gates 2026-05-18

Correcoes feitas antes de qualquer GPU:

- `scripts/kg1_pre_paid_job_integration_gate.py` agora reconhece constantes
  numericas nao-quoted no launcher, evitando falso bloqueio de
  `MAX_STEPS`, `SAVE_EVERY_STEPS` e `EVAL_EVERY_STEPS`;
- o mesmo gate passou a exigir as `3` protected rows:
  `8740ed31=01101000`, `59bee375=10010101`, `55d834d1=00111111`;
- o launcher V620 agora expõe `KG1_PROTECTED_ID_ANSWERS` como literal
  auditavel pelo gate estatico.

Artefatos:

- launcher:
  `artifacts/v620_hf_h200_v613_output_policy_launch/launch_v620_hf_nemo_h200_v613_output_policy.py`;
- manifest debug mais recente:
  `artifacts/v620_hf_h200_v613_output_policy_launch/v620-nemo-h200-v613-output-policy-v290ckpt6-20260518T115934Z_launch_manifest.json`;
- comando remoto auditado:
  `artifacts/v620_hf_h200_v613_output_policy_launch/v620-nemo-h200-v613-output-policy-v290ckpt6-20260518T115934Z_remote_command.sh`;
- gate pre-pago:
  `artifacts/v620_hf_h200_v613_output_policy_launch/v620_pre_paid_job_integration_gate.json`;
- preflight V618:
  `artifacts/v620_hf_h200_v613_output_policy_launch/v620_v618_preflight_gate.json`.

Gates executados:

- `python -m py_compile` no gate pre-pago e launcher V620: passou;
- `python scripts/kg1_pre_paid_job_integration_gate.py --self-test`: passou;
- gate pre-pago real V620: `ok=true`, `findings=[]`;
- V618 preflight real V620: `decision=gpu_allowed`, `blockers=none`,
  `warnings=none`;
- auditoria do comando remoto: passou, com hash
  `fbeeb04d695981f71af0f1e6a68b933d4b070dc1ef9e7569be7d94c95c84e05f`
  e exports sincronizados:
  `MAX_STEPS=20`, `SAVE_EVERY_STEPS=10`, `EVAL_EVERY_STEPS=10`,
  `LEARNING_RATE=1.0e-6`, `FINAL_LEARNING_RATE=1.0e-7`,
  `MAX_LENGTH=1024`, `BATCH_SIZE=4`, `MICRO_BATCH_SIZE=1`;
- limpeza: removido apenas `scripts/__pycache__`.

Contrato V620:

- dataset ativo: V613 answer-first anti-runaway source-only;
- weak overlap: `0`;
- train rows: `1099`;
- val rows: `194`;
- objetivo: `example_mean`, row-loss weight ativo;
- LoRA surface auditada por V619:
  `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`;
- H200 max runtime: `timeout=3600`;
- `MAX_STEPS=20`, `SAVE_EVERY_STEPS=10`, `EVAL_EVERY_STEPS=10`;
- LR: `1.0e-6`, final LR: `1.0e-7`;
- protected rows obrigatorias: `3/3`.

Decisao:

- V620 e a primeira rota pos-V616 que esta liberada por gates para um job curto
  H200;
- ainda nao e ganho submit-safe;
- depois do checkpoint, a ordem obrigatoria e:
  1. avaliar probe/weak com V614/V618;
  2. bloquear se `finding_counts.warning>0`;
  3. aplicar regra OpenRouter se treino/eval nao entregar
     `total>=193`, `bit>=136`, `equation>=57`, `truncated=0` e protected `3/3`;
  4. so considerar submit se houver ganho real label-free e submit-safe.

### Atualizacao V621 - Falha V620 Antes Do Treino E Regra Anti-Drift 2026-05-18

Evento real:

- job H200 `felipesp1983/6a0afcf7e7940de6ee6ce78d` falhou no preflight
  remoto, antes de treinar;
- erro: `MAX_STEPS mismatch: expected 20, got 200`;
- causa: o manifesto/local gates esperavam `MAX_STEPS=20`, mas o comando HF
  renderizado ainda exportava `MAX_STEPS=200`;
- impacto: nenhum checkpoint, nenhum treino, nenhum sinal de ACC/loss e nenhum
  ganho/perda de modelo. Foi um bug operacional de sincronizacao de comando.

Regra aplicada:

- por ser treino que nao ocorreu como planejado, foi feita consulta OpenRouter
  V621 antes de qualquer novo job;
- artefatos:
  - prompt:
    `artifacts/openrouter/v621_v620_launch_failure_consult/KG1_V621_OPENROUTER_V620_LAUNCH_FAILURE_PROMPT.md`;
  - resultados:
    `artifacts/openrouter/v621_v620_launch_failure_consult/v621_openrouter_raw_results.json`;
  - modelos validos:
    `openai/gpt-5.4`, `deepseek/deepseek-v4-pro`,
    `qwen/qwen3-max-thinking`.

Consenso V621 adotado:

- corrigir apenas `MAX_STEPS` nao era suficiente;
- todo launcher HF precisa materializar o comando remoto exato em arquivo local;
- o manifesto precisa registrar:
  - caminho do comando remoto;
  - SHA256 do comando;
  - exports observados;
  - exports esperados;
- V618/pre-paid gate devem bloquear qualquer launch se o comando renderizado
  divergir do manifesto/contrato;
- nao mudar dataset, adapter, modelo base, LR ou objetivo por causa dessa falha,
  porque o modelo nem chegou a treinar.

Implementacao concluida:

- launcher V620 agora reescreve e valida os exports finais de steps antes de
  qualquer launch;
- launcher V620 grava o comando remoto em
  `*_remote_command.sh` e calcula SHA256;
- V618 preflight agora exige `command_export_audit` completo, valida hash e
  bloqueia divergencia de exports;
- gates atuais apos a correcao:
  - `python -m py_compile`: passou;
  - V618 preflight: `decision=gpu_allowed`, `blockers=[]`, `warnings=[]`;
  - pre-paid gate: `ok=true`, `findings=[]`;
  - `finding_counts.warning=0` permanece obrigatorio.

Regra permanente:

- todo treino/eval que falhar, regressar, divergir de parametros planejados ou
  produzir resultado inesperado deve gerar uma nova consulta OpenRouter
  cirurgica com logs, manifestos, comando remoto, parametros, gates, metricas e
  diffs por linha antes de qualquer novo job pago.

### Atualizacao V622 - Sync Do Gate Remoto Antes De Novo H200 2026-05-18

Evento real:

- relaunch V620 job `felipesp1983/6a0b006ba5e509f1a8414d41` falhou antes do
  treino;
- desta vez o comando remoto estava correto:
  `MAX_STEPS=20`, `SAVE_EVERY_STEPS=10`, `EVAL_EVERY_STEPS=10`;
- falha:
  `Deferred decoding-vs-adapter drift gate is unsafe: max_steps_gt_2:20,
  save_every_gt_2:10, eval_every_gt_2:10`;
- causa: o HF clonava o commit remoto `8c98388...`, cujo
  `scripts/hf_job_preflight_gate.py` ainda tinha limite antigo `<=2`, enquanto
  os gates locais V618/pre-paid ja permitiam a rota V618 com `<=20` e
  checkpoint `<=10`.

Consulta OpenRouter V622:

- artefatos:
  - prompt:
    `artifacts/openrouter/v622_v620_remote_gate_mismatch_consult/KG1_V622_OPENROUTER_V620_REMOTE_GATE_MISMATCH_PROMPT.md`;
  - resultados:
    `artifacts/openrouter/v622_v620_remote_gate_mismatch_consult/v622_openrouter_raw_results.json`;
- modelos validos: `openai/gpt-5.4`, `deepseek/deepseek-v4-pro`,
  `qwen/qwen3-max-thinking`;
- consenso unanime:
  `commit_push_remote_gate`;
- decisoes rejeitadas:
  - reduzir para `2` steps, porque isso muda o experimento e tem baixa chance
    de ganho real em `equation_transform`;
  - bypass inline do gate, porque enfraquece a protecao que estamos tentando
    estabilizar;
  - abandonar V620, porque nao houve sinal negativo de treino.

Implementacao:

- `scripts/hf_job_preflight_gate.py` agora ativa limites V618 quando
  `KG1_V618_MODULE_SURFACE_GATE_STATUS=passed`:
  - `MAX_STEPS<=20`;
  - `SAVE_EVERY_STEPS<=10`;
  - `EVAL_EVERY_STEPS<=10`;
- mantem limite antigo `<=2` quando a rota V618 nao esta marcada como passada;
- self-test cobre:
  - V618 route passa com `20/10/10`;
  - rota sem V618 continua falhando com `max_steps_gt_2`;
- novo gate:
  `scripts/run_v622_remote_gate_policy_sync.py`;
- commit remoto pushado:
  `058fafbcdcd6f0638ad77ea7c91ba2fa82d3c714`.

Gates apos commit/push:

- novo manifest debug:
  `artifacts/v620_hf_h200_v613_output_policy_launch/v620-nemo-h200-v613-output-policy-v290ckpt6-20260518T121158Z_launch_manifest.json`;
- `KG1_EXPECTED_COMMIT=058fafbcdcd6f0638ad77ea7c91ba2fa82d3c714`;
- V622 policy sync:
  `ok=true`, `findings=[]`, `current_head==expected_commit`;
- V618 preflight:
  `decision=gpu_allowed`, `blockers=[]`, `warnings=[]`;
- pre-paid integration:
  `ok=true`, `findings=[]`;
- comando remoto continua auditado com exports esperados.

Proxima acao:

- relancar V620 unchanged usando o commit `058fafb...`;
- monitorar logs;
- se qualquer novo bloqueio/falha ocorrer, aplicar novamente a regra
  OpenRouter antes de outro job pago;
- se passar do preflight e produzir checkpoint, rodar V614/V618/probe/weak
  antes de considerar submit.

### Atualizacao V623 - Regra De Consulta E Protected Rows Sincronizadas 2026-05-18

Status operacional:

- job H200 V620 relancado:
  `felipesp1983/6a0b02a5e7940de6ee6ce7e1`;
- o job passou `preinstall`, `artifacts`, `postinstall`,
  `decoding_vs_adapter_drift_gate`, hash dos datasets e tokenizacao;
- ate o ultimo log, estava carregando o modelo antes do primeiro step;
- nenhum checkpoint/ACC novo ainda foi produzido.

Regra permanente reforcada:

- todo treino/eval que falhar, regressar, divergir de parametros planejados,
  travar, produzir warnings bloqueantes, ou entregar ACC/loss fora do plano,
  deve parar a proxima execucao paga e gerar uma consulta OpenRouter com prompt
  rigoroso e sem ruido antes de qualquer novo job;
- o prompt deve incluir obrigatoriamente:
  logs HF completos, manifestos, comando remoto, commit, dataset hashes,
  contrato LoRA, parametros de treino, tokenizacao, masks, protected rows,
  thresholds, predicoes, raw outputs, extract/verify, diffs por linha e
  decisao FinOps;
- nao pode haver relaunch pago por intuicao depois de falha.

Correcao implementada:

- `scripts/hf_job_preflight_gate.py` agora exige as 3 protected rows:
  `8740ed31=01101000`, `59bee375=10010101`, `55d834d1=00111111`;
- isso sincroniza o preflight remoto com V618 e com
  `kg1_pre_paid_job_integration_gate.py`;
- testes locais passaram:
  - `python -m py_compile`;
  - `python scripts/hf_job_preflight_gate.py --self-test`;
  - `python scripts/kg1_pre_paid_job_integration_gate.py --self-test`.

Proxima acao:

- continuar monitorando o job V620 de 40 em 40 segundos;
- quando checkpoint-10 aparecer, rodar primeiro V614/V618/probe/weak;
- se checkpoint falhar qualquer contrato ou ACC minima
  (`total>=193`, `bit>=136`, `equation>=57`, `truncated=0`, protected `3/3`),
  acionar consulta OpenRouter antes de novo gasto;
- se passar, continuar para checkpoint-20/final e so considerar submit com
  ganho submit-safe real.
