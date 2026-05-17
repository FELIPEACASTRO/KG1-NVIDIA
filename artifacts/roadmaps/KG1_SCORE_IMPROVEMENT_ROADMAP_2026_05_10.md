# KG1 Score Improvement Roadmap

Atualizado: 2026-05-17

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
| Recompute strict label-free recente | 191/315 | >= 193/315 |
| equation_transform baseline | 56/155 historico; 55/155 strict | alvo inicial 60/155 |
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
| V525 OpenRouter objective consult | 5 modelos (`gpt-5.5`, Claude Opus 4.7, Gemini 3.1 Pro, Qwen 3.6 Max, DeepSeek V4 Pro) confirmaram que `token_mean` nao pode ser usado com V523; todos recomendam `example_mean`, guard de `8740ed31`, e kill-switch de primeiro checkpoint; 2/5 preferem rebuild V525 antes de GPU | decisao adotada: rodar CPU dry-run de contribuicao por familia; se V523+`example_mean` ainda for dominado por bit, construir V525 com traces bit curtas e token-mass `bit<=70-78%`; se passar, permitir apenas smoke H200 curto com gate `total>=193`, `equation>=57`, `bit>=136`, `trunc=0` |
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
| V549 protected/token guards | ambos checkpoints erraram `8740ed31`: esperado `01101000`, previsto `01111000`; `weak_promotion_gate` bloqueou por `correct_lt_193`, `equation_lt_57`, `bit_lt_136`, tokens excessivos e protected-row backfire; `catastrophic_eval_guard` passou porque nao foi colapso total | guards atualizados estao funcionando; condicoes atuais de promocao continuam: `total>=193`, `bit>=136`, `equation>=57`, `trunc=0`, protected rows preservadas e saida curta |
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
| V573 HF upload/pre-paid | dataset V573 subido para HF dataset commit `3d321aff1e68d72769f167ceab2a28123faa18fd`; launcher debug baixou do HF sem symlink cache, validou hashes `ba515.../6957...`, adapter seed e objetivo; pre-paid gate passou sem findings com deferimento V568 restrito a `MAX_STEPS=2` e weak eval obrigatoria no primeiro checkpoint | pronto para commit/push dos gates e um smoke H200 de 2 steps; se checkpoint-2 nao passar `total>=193`, `bit>=136`, `equation>=57`, `trunc=0` e protected rows, cancelar/abandonar |

## Decisao Atual V560-V573

O problema ativo nao e mais falta de busca externa nem apenas hiperparametro de
loss. A evidencia V562/V563 mostra desalinhamento entre treino e inferencia:

- V560 removeu `RULE:` e `Trace:` e treinou targets curtos, mas o adapter V561
  ainda gerou CoT longo em weak eval;
- V562 mediu `avg_completion_tokens=4775`, `max_completion_tokens=7680` e
  `truncated=1`, com regressao de bit para `135/160`;
- os dois protected rows que o baseline acertava foram perdidos:
  `8740ed31: 01101000 -> 01111000` e `59bee375: 10010101 -> 2`;
- o gate bloqueou por `correct_lt_193`, `equation_lt_57`, `bit_lt_136`,
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

Plano efetivo a partir daqui:

1. Bloquear novo broad LoRA em H200. Loss nao autoriza gasto nem submit.
2. Concluido: auditoria local V564 de contrato/mask separou falso positivo de
   validacao e isolou o mismatch train-vs-eval como blocker real.
3. Concluido: V565 reconstruiu o dataset answer-only com contrato official-like
   e passou contract/mask/tokenization/objective gates.
4. Concluido: V567 descartou strict no-think/one-line e tambem mostrou que
   `max_tokens=2048` e insuficiente para as protected rows.
5. Gate novo: weak eval promocional fica bloqueado se `KG1_DISABLE_THINKING=1`
   ou `KG1_MAX_TOKENS<7680`, exceto quando o job declarar diagnostico explicito.
6. Gate novo: treino promocional pago fica bloqueado ate o diagnostico V568
   separar decoding ruim de drift do adapter com `0` logprobs faltantes,
   protected rows completos e regressao de margem vs baseline dentro do limite
   (`KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION <=
   KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION`). Margem absoluta negativa e
   alerta, nao bloqueio padrao.
7. Concluido: V569 confirmou que o proximo passo nao e outro broad SFT; e
   recuperar/reproduzir trajetorias protegidas longas e medir margens.
8. Concluido: V570 recuperou `raw_output` correto do V290 ckpt-6 para
   `8740ed31` e `59bee375`, mais `40` anchors corretos, com prompt hash,
   completion hash, resposta final, truncation e decisao `diagnostic_only`.
9. Gate V570:
   protected final answers exatos, `max_tokens=7680`, thinking habilitado,
   tokenization sem truncation, e reproducao V568 dentro da tolerancia. Como as
   linhas sao weak-gate, o gate tambem exige `training_allowed_rows=0` ate haver
   analogos source-only ou pares preference sem contaminacao.
10. Concluido parcial: V571/V573 construiram analogos source-only verificaveis
   para bit-pair e mantiveram equation V551 limpa. Isso nao usa as linhas
   weak-gate V570 como treino.
11. Concluido: V573 upload/debug/pre-paid passou. O job so pode declarar
   `LOSS_NORMALIZATION_MODE=example_mean`, `USE_ROW_LOSS_WEIGHT=1`,
   `REQUIRE_ROW_LOSS_WEIGHT=1`, `MAX_LENGTH=2048`, `MAX_STEPS=2`, protected
   rows `8740ed31=01101000` e `59bee375=10010101`, e weak eval obrigatoria no
   primeiro checkpoint.
12. Executar smoke V573 H200 somente apos commit/push dos gates usados pelo job.
   O checkpoint-2 deve ser avaliado em weak official-like com `max_tokens=7680`.
13. Executar auditoria de logits/protected rows apos contrato padronizado:
   medir top-k e probabilidade das respostas corretas em `8740ed31` e
   `59bee375`; qualquer adapter que reduza esses logits fica bloqueado.
14. Se o diagnostico de logits/protected passar, rodar apenas smoke curto V573
   com primeiro checkpoint kill-switch; se falhar, nao gastar H200.
15. So entao montar microexperimento hard-negative/protected:
   baseline-wrong/solver-right como positivos e baseline-correct protected como
   no-change; sucesso minimo `total>=193`, `bit>=136`, `equation>=57`,
   protected rows OK e trunc `0`; falha aborta GPU.

Itens removidos do plano ativo apos V569:

- broad LoRA/SFT sem novo gate de margem;
- answer-only V560/V561 como estrategia principal;
- strict `disable_thinking`/short answer como contrato promocional;
- CPU solver/verifier direto como submit;
- novos H200 justificados por eval loss;
- treino equation-first enquanto `8740ed31` e `59bee375` nao estiverem
  protegidos por replay/margem.

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
- o `weak_promotion_gate` bloqueou por `correct_lt_193`,
  `equation_lt_57`, `bit_lt_136`, `truncated_gt_0`,
  completion tokens excessivos e `protected_row_backfire_guard_failed`;
- V553 foi cancelado antes do checkpoint-4, porque checkpoint-4 tinha loss
  pior no treino e checkpoint-2 ja repetiu o backfire conhecido.

Decisao:

- V551/V552/V553 nao e submit-safe;
- nao rodar full eval, package ou submit desta linha;
- nao relancar SFT curto de bit traces se o novo CPU gate nao provar antes
  preservacao de `8740ed31=01101000`, `59bee375=10010101`, `bit>=136`,
  `equation>=57`, `trunc=0` e completions curtas;
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
   `total>=193`, `bit>=136`, `equation>=57`, `trunc=0`, protected rows
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
  com `total>=193`, `bit>=136`, `equation>=57`, `trunc=0`, protected rows
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
   - abortar no primeiro checkpoint que violar `bit>=136`, `equation>=57`,
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
   - `total < 193/315`;
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
  `total>=193`, `equation>=57`, `bit>=136`, `trunc=0`; package/submit exige
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
| Gate promocional | nao aplicavel sem ACC | `total>=193`, `equation>=57`, `bit>=136`, `trunc=0` |
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
- o melhor candidato nao atingiu `total>=193`, `equation>=57`, `bit>=136`;
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
     preserve `8740ed31=01101000`, `bit>=136`, `equation>=57` e `trunc=0`.
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
  - protected row `8740ed31=01101000`, `bit>=136`, `equation>=57`,
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
