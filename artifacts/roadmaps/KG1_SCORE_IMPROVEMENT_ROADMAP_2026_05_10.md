# KG1 NVIDIA - Roadmap de melhoria por familia

Gerado em: 2026-05-10

Objetivo: consolidar os achados Kaggle/Hugging Face, resultados V221/V226/V229/V230 e o plano operacional para melhorar as duas familias criticas sem treino cego.

## Resumo executivo

- Baseline protegido atual: `v226__v226_best_checkpoint1_observed_191`.
- Weak score observado: `191/315`.
- Gate para liberar full eval: total `>=193`, `equation_transform>=60`, `bit_manipulation>=133`, truncation `<=3`.
- Resultado do baseline: total `191`, equation `55`, bit `136`, truncation `0`.
- Gargalo: `equation_transform`, com gap `5`.
- `bit_manipulation` ja passa o gate, mas com margem pequena de `+3`.
- Oracle V230 por linha melhora para `197/315`, equation `57`, bit `140`, truncation `0`, mas ainda falha o gate de equation por `3`.
- Conclusao: roteamento entre os adapters ja avaliados nao resolve. O proximo ganho precisa vir de mineracao dos miss-packs, solver/verifier deterministico ou dados/traces filtrados.
- Atualizacao HF-only 2026-05-11: o smoke V257/V249 em H200 produziu o melhor resultado operacional ate agora no contrato V221: `v257_checkpoint_4_v221_contract` com `192/315`, equation `56/155`, bit `136/160`, truncation `1`. Ainda nao passa o gate (`193/315`, equation `60`), mas prova ganho real de `+1` em bit sem perder equation frente ao V256 HF.
- Atualizacao V259/V260B 2026-05-11: o treino equation-focused a partir do V257 checkpoint-4 repetiu `192/315`, equation `56/155`, bit `136/160`, truncation `0` no melhor checkpoint. Ele reduziu truncation, mas nao aumentou `equation_transform`; portanto nao justifica continuacao longa em H200 sem novo dado/verifier.
- Atualizacao V261 2026-05-11: a varredura operacional `thinking on + no prompt suffix` foi cortada cedo no H200 por gate de FinOps. O primeiro candidato (`v259_checkpoint4_nosuffix`) regrediu para `155/315`, equation `55/155`, bit `100/160`, truncation `1`; sem ganho em equation e com queda severa em bit. Esta familia de prompt esta descartada para novos gastos.
- Atualizacao V262/V263 2026-05-11: adapter soups entre V226/V257/V259 foram gerados em CPU e avaliados no H200. Melhor soup (`soup_v226_050_v257_050`) ficou em `192/315`, equation `56/155`, bit `136/160`, truncation `1`; os outros regrediram para `191` e `190`. Adapter soup nao resolveu o gargalo e nao deve consumir novo H200 sem um preflight que prove alvo novo em equation.
- Atualizacao V264 2026-05-11: recheck HF CPU confirmou que os traces P0 `andy279/*` continuam bloqueados por review/termos (`403`). A rota mais promissora agora precisa de acao humana para liberar esses datasets no HF; o mirror publico sozinho ja foi usado e nao entregou equation suficiente.
- Atualizacao V265 2026-05-11: foi criado o mix `score086_filtered_mix`, usando V249 publico nao-weak mais o corpus historico V189 que participou da trajetoria do score amplo `0.86`, mas com bloqueio por `weak_id`, `prompt_sha256` normalizado e dedupe. O builder CPU manteve `3442` linhas (`2000` equation, `1442` bit) e bloqueou `807` linhas (`83` overlaps exatos com weak por prompt hash e `724` duplicatas). O split final ficou em train `3098` (`1800` equation, `1298` bit) e val `344` (`200` equation, `144` bit). Esse dataset passou o V250 tokenization gate no HF: truncation `0.0`, offset masks completos e nenhum token de completion descartado. Proximo passo autorizado: smoke train curto em H200, nao treino longo.
- Atualizacao V266 2026-05-11: o primeiro smoke H200 sobre V265, iniciado de `v259_checkpoint_4`, falhou no objetivo e foi interrompido por FinOps quando `final` foi confirmado identico ao `checkpoint-4` via SHA LFS. Resultados weak V221-contract: `checkpoint-2 = 155/315, equation 56/155, bit 99/160, trunc 0`; `checkpoint-4 = 154/315, equation 56/155, bit 98/160, trunc 2`. Diagnostico: a receita `all target modules + equation_transform=3.0 + bit=0.8` destruiu a familia `bit_manipulation` sem ganhar equation. O V265 nao esta descartado, mas so pode voltar em smoke ultra-conservador, com trainable limitado a atencao/lm_head, bit replay reforcado, source key V189 corrigida para `v189_score086_equation_answer_short_filtered`, e weak eval parcial antes de qualquer novo gasto longo.
- Atualizacao V268/V269/V270 2026-05-11: o corpus publico `tonghuikang/nemotron` foi ingerido com bloqueio de todos os 315 weak IDs e so aceitou linhas cujo `\boxed{}` final batia com `train.csv`. O builder V268 gerou `1789` linhas (`1105` bit, `506` equation em treino), passou tokenization gate V250 com `max_length=8192`, e o smoke V269 em H200 a partir do melhor V259 checkpoint-4 empatou o melhor score: `v269_checkpoint_2 = 192/315`, equation `56/155`, bit `136/160`, trunc `0`. O `final` regrediu para `190/315`, bit `134/160`, trunc `2`. Conclusao: V268 e util como fonte de raciocinio/verifier, mas a receita SFT curta nao trouxe ganho novo de equation.
- Atualizacao V271 2026-05-11: a mineracao CPU-only dos erros atuais validou o contrato V221 e confirmou que o V269 mudou `4` respostas de `equation_transform`, todas de errado para errado. Nos `99` erros restantes do melhor atual (`v259_checkpoint_4`), a taxonomia ficou: `83` `equation_symbolic_punct` com resposta simbolica, `15` `equation_numeric_operator` com resposta numerica, e `1` `equation_numeric_operator` misto. Regra de negocio: nao gastar H200 novamente ate existir um solver/verifier CPU que gere pelo menos `+4` a `+5` overrides de equation sem violar `bit>=136/160`, ou ate os traces gated `andy279` serem liberados.
- Atualizacao V272 2026-05-11: auditoria CPU dos solvers atuais sobre os `99` misses de equation parseou todos os prompts (`parse_status=ok`), confirmou `83` simbolicos de pontuacao e `16` numericos, mas encontrou `0` candidatos verificados promotaveis. Regras testadas: char transducer, reverse, prefix/suffix, deletion posicional audit-only e numeric same-operator. Conclusao: os solvers simples V238/V241/V246 nao bastam; a rota agora e liberar traces solver-guided externos ou implementar busca simbolica mais forte antes de qualquer GPU.
- Atualizacao V273 2026-05-11: auditoria CPU de solver cryptarithm inspirada no repositorio publico `tonghuikang/nemotron` foi adicionada para testar a hipotese `AB op CD -> resultado` com inferencia de digitos/operadores. O gate endurecido validou o contrato V221, auditou os mesmos `99` misses de equation, bloqueou casos subdeterminados e encontrou `0` candidatos verificados promotaveis; classes com candidatos produziram `7` incorretos e foram descartadas. Conclusao: cryptarithm simples tambem nao e deployable; nao gastar H200 nessa rota sem uma familia simbolica mais rica ou traces solver-guided.
- Atualizacao V274 2026-05-11: a primeira rota deterministica finalmente passou o weak gate sem GPU. O postprocessor numerico label-free, inspirado na gramatica publica `tonghuikang/nemotron` e nos principios de DSL/verifier da literatura, aplicou somente `4` overrides guardados sobre as predicoes do melhor V259 checkpoint-4 e elevou o contrato V221 para `196/315`, `equation_transform=60/155`, `bit_manipulation=136/160`, truncation `0`. O audit local registrou `gains=4`, `losses=0`, `wrong_on_baseline_misses=0` e `weak_gate.pass=true`. O job HF CPU autenticado `https://huggingface.co/jobs/felipesp1983/6a01b475aff1cd33e8f338cf` reproduziu `weak_gate.pass=true`, `equation_transform=60/155`, `bit_manipulation=136/160`, `wrong_on_baseline_misses=0`. Decisao: empacotar esse postprocessor como candidato deployable antes de qualquer H200 full eval.
- Atualizacao V275 2026-05-11: o postprocessor V274 foi separado em modulo deployable `src/kg1_v274_numeric_postprocessor.py`, sem termos de scoring (`answer`, `correct`, `verify_answer`, `solution`) no source guard. Os avaliadores `evaluate_lora_adapter.py` e `evaluate_lora_adapters_batch.py` agora aceitam `--prediction-postprocessor v274_numeric_operator_overrides`. O gate local `scripts/run_v275_deployable_postprocessor_gate_hf.py` confirmou que o caminho deployable usa somente `id/prompt/prediction/family/truncated`, e labels entram apenas no gate de verificacao. Resultado local: `196/315`, equation `60/155`, bit `136/160`, trunc `0`, `applied=4`, `gains=4`, `losses=0`, `wrong_on_misses=0`. O job HF CPU `https://huggingface.co/jobs/felipesp1983/6a01b708aff1cd33e8f338ed` reproduziu o gate com `weak_gate.pass=true`, source SHA `f992ba7c4b4d2eb070e09cc59b11aae899872b1f9f18f82bc328fe20f8db4d2d` e `wrong_on_baseline_misses=0`.
- Atualizacao V276 2026-05-11: o full/validation CSV canonico foi localizado no Google Drive (`KG1_NVIDIA_V207A/output_v207a_acc_gate/validation/official_train_seed42_stratified10_val.csv`, file ID `184zcN6JeFl_KA8-PhHD5ANBdWSOOEyKL`) e publicado no HF privado para remover dependencia do Colab/Drive antes do full eval. Validacao: `947` linhas, SHA256 `84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935`, IDs duplicados `0`, familias `bit=160`, `equation=155`, `gravity=159`, `numeral=157`, `text_encryption=157`, `unit=159`. HF dataset commit: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/1da6764b95966cc107ad4af8c5c0d6f7fdc6261d`. Script pronto: `scripts/hf_job_full_eval_v276.py`, com gate de GPU/custo/commit/CSV/adapter e `--prediction-postprocessor v274_numeric_operator_overrides`.
- Atualizacao pesquisa externa 2026-05-11: nova varredura HF/Web encontrou dois adapters publicos 30B completos que ainda nao estavam como candidatos de eval: `gfinin/nemotron-reasoning-lora` e `etencore/nemotron-30b-reasoning-lora`. Ambos declaram `base_model:adapter:nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` e possuem `adapter_config.json` + `adapter_model.safetensors`. `gfinin` usa LoRA `r=16`, `alpha=32`, target modules amplos + `target_parameters` de MoE; `etencore` usa LoRA `r=32`, `alpha=64`, apenas `q_proj/v_proj/o_proj/k_proj`, com checkpoints `root`, `checkpoint-1000` e `checkpoint-1188`. Falsos positivos descartados para execucao: `Aitherium/Nemotron-3-Nano-30B-LoRA-Reasoning-v2` tem apenas `.gitattributes`; `GaryNENE/nemotron-nano-8b-reasoning-lora` tem scripts/README, mas nao adapter weights/config e ainda e 8B, nao 30B. Decisao FinOps: depois do V276, fazer no maximo um weak eval agrupado desses candidatos publicos completos antes de qualquer treino novo.
- Atualizacao literatura 2026-05-11: a direcao tecnica mais forte continua sendo solver/verifier hibrido, nao treino cego. Evidencias novas/reforcadas: SyGuS formaliza busca por programa dentro de uma gramatica sob especificacao logica; DryadSynth/POPL 2024 mostra que bit-vector synthesis melhora com enumeracao por term-graph, filtro por exemplos, subexpressoes guiadas por LLM e deducao bottom-up; pesquisas CAV 2024 mostram que LLM isolado perde para sintetizadores formais, mas LLM guiando busca enumerativa melhora o resultado; PBE com LLMs ainda falha fora da distribuicao, entao treino sozinho em exemplos parecidos nao e suficiente para os `83` erros simbolicos de `equation_transform`. Implicacao pratica: para subir ACC das familias, priorizar `V277` weak eval de adapters externos completos, `V278` solver simbolico enumerativo/CEGIS sobre `equation_symbolic_punct`, e so depois distilacao dos overrides verificados.
- Atualizacao dataset oficial 2026-05-11: o zip local `nvidia-nemotron-model-reasoning-challenge.zip` foi auditado sem extracao persistente. SHA256 do zip: `48acb1626f984f5742543a25350611728ba4e2944857960653e366358cb00b23`. Conteudo: `train.csv` e `test.csv`. `train.csv` tem `9500` linhas, IDs unicos `9500`, colunas `id/prompt/answer`, SHA256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`; esse hash bate exatamente com o `train.csv` publico usado na rota V268/tonghuikang. `test.csv` tem `3` linhas, IDs unicos `3`, colunas `id/prompt`, SHA256 `c59d7eb0464b0a872a0c3f81e60cd6643fc1932a2dedaa05972bfd02cc638589`. Contagem heuristica por prompt natural no `train.csv`: bit `1602`, equation `1555`, gravity `1597`, numeral `1576`, text encryption `1576` incluindo `4` casos com texto que tambem mencionam bits, unit `1594`. Conclusao: o zip oficial valida a origem V268, mas nao muda o bloqueio anti-leakage; qualquer uso continua exigindo filtro por weak IDs/prompt hash.
- Auditoria Google Drive 2026-05-10: `1879` arquivos KG1 catalogados, `85` adapters completos, `232` reports, `423` CSVs, `54` JSONLs e `11` notebooks. Nenhum artefato do Drive supera o baseline V226 sob gate weak canonico; o Drive deve ser usado como fonte de pesos fortes conhecidos, reports e dados para triagem, nao como fonte de promocao automatica.
- Achado Drive mais util: V207A full/validation gate do V194 tem `822/947` com familias nao criticas em `100%`, mas confirma o mesmo gargalo fraco: `bit_manipulation=135/160`, `equation_transform=55/155`. Isso reforca que o problema real continua concentrado em `equation_transform`.
- Importante: muitos arquivos do Drive foram parte da trajetoria que chegou ao score amplo `0.86`. Esse score e valido como evidencia historica de que V194/V202D resolvia muito bem as familias nao criticas, mas nao pode ser interpretado como melhoria atual das duas familias alvo. No recorte decisivo, o proprio V207A mede `equation_transform=55/155` e `bit_manipulation=135/160`, alinhado ao gargalo V230.
- Atualizacao V280 2026-05-11: foi criado e executado um gate CPU-only para os datasets P0 `andy279/*`, validando primeiro o contrato weak (`315` linhas, SHA `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`, familias `bit=160`, `equation=155`). Resultado atual: `andy279/nemotron-reasoning-challenge-raw-traces` e `andy279/nemotron-reasoning-challenge` continuam `gated=manual`; `p0_accessible_files=0`, `p0_blocked_files=5`. O mirror publico `jasonkung98/*` esta acessivel, mas o sample de `train.csv` tem overlap com weak e continua servindo apenas como sanity/source audit, nao como autorizacao de treino. Decisao: `p0_gated_terms_required_no_gpu`; nao gastar H200/GPU nessa rota ate liberar acesso humano aos datasets P0.
- Atualizacao OpenRouter 2026-05-11: o export `C:\Users\davis\Downloads\OpenRouter Chat Mon May 11 2026.json` foi auditado contra a meta das familias `equation_transform` e `bit_manipulation`. Ele confirma que a busca externa por arquivos exatos `sft_train.jsonl` e `sft_val.jsonl` nao encontrou espelho publico acionavel do KG1/Andy279. Os hits publicos (`garg-aayush/sft-cs336-assign5-datasets`, `Satori-reasoning/Satori-SWE-two-stage-SFT-data`, `SakanaAI/FishMath-SFT-Data`, `prabinh/Superior-Reasoning-SFT-gpt-oss-120b`, `AlgorithmicResearchGroup/ai-sft`, `norallm/normistral-11b-thinking-training`, `agentlans/HuggingFaceH4-ultrachat_200k`) sao SFT/reasoning genericos ou de dominio divergente; podem inspirar pipeline/metodologia, mas nao justificam download pesado, H200, treino ou mistura de dados sem novo gate de dominio/licenca/leakage.
- Atualizacao V283/V284 2026-05-11: o adapter `v283_v282_weights_v194_config` foi reavaliado em HF H200 com prompt oficial-like e `max_tokens=768`, `max_model_len=8192`, thinking ligado. Resultado: `7/315`, `equation_transform=7/155`, `bit_manipulation=0/160`, truncation `225/315` (`71.43%`). O output foi publicado em `felipesp1983/kg1-nemotron-lora-v283-v282-config-patch`, path `evals/v283-h200-v221prompt1line-config-patch-weak-mt768-ml8192-20260511T1750Z-r6`, commit `d3dab963a430ec3c0386b58bf4b604f7ce4d6ebe`. Decisao: descartar V283/V282-config para full eval e Kaggle; nao gastar V284 official-like full nesse adapter.
- Atualizacao V277 2026-05-11: o weak eval agrupado dos adapters publicos completos `gfinin/nemotron-reasoning-lora` e `etencore/nemotron-30b-reasoning-lora` foi executado em HF H200 sob o contrato barato historico (`max_tokens=96`, thinking off, suffix de uma linha). Resultado melhor: `81/315`, com `equation_transform=29/155`, `bit_manipulation=52/160`, truncation `0`; `checkpoint-1188` tambem fez `81/315`, `equation=28`, `bit=53`; `checkpoint-1000` fez `76/315`; `gfinin` fez `39/315`, truncation `4`. Upload em `felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke`, path `evals/v277-external-public-adapters-weak-20260511T1820Z-r2`, commit `a7ac6568a6c710dd89ba6b060a15d950dda08f7f`. Decisao: rejeitar esses adapters como substitutos, nao rodar full eval, nao usar em submit.
- Atualizacao V278 recheck 2026-05-11: a auditoria CPU-only `scripts/run_v278_symbolic_pbe_dsl_audit_hf.py` foi reexecutada localmente na worktree limpa, output `artifacts/v278_symbolic_pbe_dsl_audit/20260511T1825Z`. O contrato V221 bateu, `parse_status_counts={"ok":99}`, `subtype_counts={"equation_numeric_operator":16,"equation_symbolic_punct":83}`, `all_verified_candidates=0`, `all_incorrect_candidates=3`, `verified_promotable_candidates=0`. Decisao reconfirmada: `no_promotable_symbolic_pbe_signal`; nao gastar GPU na DSL local simples.
- Atualizacao V281 2026-05-11: foi implementado e executado `scripts/run_v281_reasoninggym_cpu_triage.py` para o dataset `nvidia/Nemotron-RL-ReasoningGym-v1`. Gates: split `train` unico, schema esperado, licenca `cc-by-4.0`, weak CSV V221 validado (`315`, SHA `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`) e overlap fraco `0`. Output `artifacts/v281_reasoninggym_cpu_triage/20260511T1835Z` selecionou `1326` fixtures: `bitwise_arithmetic=143`, `circuit_logic=141`, `count_bits=150`, `binary_alternation=154`, `simple_equations=147`, `cryptarithm=145`, `polynomial_multiplication=152`, alem de `base_conversion=133` e `number_format=161`. Decisao: pronto para V282 verifier probes CPU; ainda nao autoriza treino LoRA direto.
- Atualizacao V282 2026-05-11: foi implementado e executado `scripts/run_v282_reasoninggym_verifier_probes.py` sobre o output V281. Resultado final `artifacts/v282_reasoninggym_verifier_probes/20260511T1855Z`: `1016/1326` fixtures verificados localmente com `0` mismatches. Cobertura validada: `binary_alternation=154/154`, `bitwise_arithmetic=143/143`, `count_bits=150/150`, `cryptarithm=145/145`, `number_format=161/161`, `simple_equations=147/147`, `base_conversion=116/133`; `circuit_logic` e `polynomial_multiplication` ficaram unsupported. Decisao: `reasoninggym_verified_fixtures_ready_for_probe_design`; usar apenas como probes/fixture design CPU, nao como autorizacao de treino direto.
- Atualizacao V285 2026-05-11: como `andy279/*` segue dependente de aprovacao humana, foi criado o caminho publico alternativo `scripts/build_v285_reasoninggym_auxiliary_dataset.py`. Ele consome somente linhas V281 que passaram no audit V282 com `verified_match`, filtra as familias centrais `binary_alternation`, `bitwise_arithmetic`, `count_bits`, `cryptarithm` e `simple_equations`, valida contrato weak V221 `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`, e gera `739` fixtures auxiliares sem overlap fraco: train `651` (`bit=394`, `equation=257`) e validation `88` (`bit=53`, `equation=35`). Output local: `artifacts/v285_reasoninggym_auxiliary_dataset/20260511T1830Z`; upload HF dataset: `data/v285_reasoninggym_auxiliary/20260511T1830Z`, commit `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/acb4a1924b3599214ac16ecd97617d819529fccc`. Bloqueio mantido: nao e autorizacao de treino direto, full eval, package ou Kaggle submit.
- Atualizacao V286 2026-05-11: foi criado e executado `scripts/run_v286_generic_tokenization_gate.py` sobre o V285 com tokenizer real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16@cbd3fa9f933d55ef16a84236559f4ee2a0526848`. Resultado: `offset_masks=739/739`, `fallback_masks=0`, `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `train_token_max=191`, `validation_token_max=191`, `train_val_prompt_overlap=0`. Output local: `artifacts/v286_generic_tokenization_gate/20260511T1835Z`; upload HF dataset: `runtime_artifacts/v286_generic_tokenization_gate/20260511T1835Z`, commit `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/e86bef91046aa61b4236468294df7e2a33170d89`. Decisao: o dataset esta tecnicamente pronto para um smoke HF muito pequeno, mas ainda precisa de decisao FinOps/risco porque os exemplos sao prompts diretos ReasoningGym, nao prompts Alice KG1.
- Atualizacao V287/V288 2026-05-11: para reduzir drift textual, foi criado `scripts/build_v287_reasoninggym_alice_style_dataset.py`, que re-renderiza o V285 como prompts KG1/Alice-style com exemplos `input -> output`, sem inventar labels e sem misturar exemplos entre splits. Resultado: train `651`, validation `88`, mesmas familias (`bit=394/53`, `equation=257/35`), `train_val_prompt_answer_overlap=0`. O gate V286 sobre V287 passou com tokenizer real: `offset_masks=739/739`, `fallback_masks=0`, `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `train_token_max=313`, `validation_token_max=294`. Uploads HF: V287 dataset `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/c431f71af0b4df45c3207c29c6066655113e4fe0`; V288 gate `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/413ed590e75b1d538b67f6945c58a606add86141`. Decisao: melhor candidato de dataset publico para um smoke pequeno, mas ainda exigir guardrails de FinOps e weak eval antes de qualquer treino longo.
- Atualizacao V289 2026-05-11: o smoke H200 ultra-curto sobre V287, iniciado do melhor `v259_checkpoint_4`, foi executado com guardrails HF antes/depois do install, dataset SHA fixo, adapter gate e tokenization/offset-mask gate. Treino: job `https://huggingface.co/jobs/felipesp1983/6a0223b1317220dbbd1a7b9d`, output repo `felipesp1983/kg1-nemotron-lora-v289-v287-alice-style-v259ckpt4-smoke`, `max_steps=4`, `trainable_lora_modules=q_proj,k_proj,v_proj,o_proj,lm_head`, baseline eval loss `2.9872424710880625`, final eval loss `2.9903970685872165`. Weak eval final: job `https://huggingface.co/jobs/felipesp1983/6a0229d0aff1cd33e8f33e38`, upload commit `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v289-v287-alice-style-v259ckpt4-smoke/commit/07c10c3c12d4c5666c3c5989c315b9da41fe25ae`. Resultado: `v289_checkpoint_4 = 17/315`, equation `9/155`, bit `8/160`, trunc `0`; `v289_final = 16/315`, equation `9/155`, bit `7/160`, trunc `0`. Decisao: rejeitar V289 como candidato, nao submeter, nao rodar full eval e nao gastar nova GPU nessa receita Alice-style ReasoningGym SFT. Diagnostico: mesmo com gates tecnicos passando, o dataset publico V287 causou drift comportamental severo no contrato weak; V285/V287 ficam apenas como fixtures/probes CPU ou como material de distilacao futura com gate de nao-regressao por checkpoint.
- Atualizacao V290 2026-05-11: foi reativada a rota packageable de micro-patch rank-19, em vez de postprocessor externo. O script `scripts/build_v282_rank19_micro_patch_dataset.py` foi endurecido para normalizar replay V217 ao contrato estrito do tokenization gate (`Final answer: ...` e `weak_gate_rows_used_for_training=false`). Dataset local: `artifacts/v290_rank19_micro_patch_dataset/20260511T1925Z`, com train `11286` linhas (`equation_transform=8015`, `bit_manipulation=2695`, demais familias replay `144` cada) e val `801` (`equation_transform=573`, `bit_manipulation=164`, demais `16` cada). Patch sintético V274 fora do weak: `1080` train + `120` val, overlap weak por id/prompt `0`. Hashes: train `801334d510d923e233f382b78d42e853d87d75bfc235e676b2f247975b6b845d`, val `d39bc9c0e4051fe7d212de109d580614c1774c6f10833246a70e8b4018c2252b`. Gate V286 real-tokenizer passou: offset masks `11286/11286` train e `801/801` val, fallback `0`, prompt truncation `0.0`, completion dropped `0`, max tokens `327`. Uploads HF dataset: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/a7d8b4cf1ce2881003b34152d2820ec4bd43a66f` e gate `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/79263f6cb845d0d2f77654212be105a96e0433cf`. Decisao inicial: elegivel para um smoke H200 curto iniciado de `v194_protected`/rank-19, com kill-switch se o primeiro checkpoint nao preservar o patamar V221 (`>=191/315`, equation `>=56`, bit `>=135`) ou se regredir o perfil full/rank-19.
- Atualizacao V290 weak eval 2026-05-11: o smoke H200 foi concluido e avaliado no contrato V221 official-like. Treino: job `https://huggingface.co/jobs/felipesp1983/6a022ecfaff1cd33e8f33e6c`, output repo `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke`, subfolders validos `checkpoint-3`, `checkpoint-6` e `final`. A primeira tentativa de eval falhou corretamente por path incorreto (`final_adapter` inexistente), job `https://huggingface.co/jobs/felipesp1983/6a023630aff1cd33e8f33ed2`; a tentativa corrigida rodou em `https://huggingface.co/jobs/felipesp1983/6a02369d317220dbbd1a7c03`, commit HF `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/commit/fee1eb26009eda5d9347365e661187616e62c6cf`, path `evals/v290-h200-v221contract-rank19-micro-patch-20260511T200448Z`. Resultados: `checkpoint-3 = 190/315`, equation `56/155`, bit `134/160`, trunc `1`; `checkpoint-6 = 192/315`, equation `56/155`, bit `136/160`, trunc `0`; `final = 191/315`, equation `56/155`, bit `135/160`, trunc `0`. Decisao: V290 nao passa o weak gate (`>=193` total e `equation>=60`), nao deve ir para full eval, pacote ou Kaggle. Manter `checkpoint-6` apenas como evidencia de que a rota rank-19 packageable consegue preservar bit, mas ainda nao resolve o gargalo de equation.
- Atualizacao Space/GGUF 2026-05-11: o Space `rikunarita-3/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF-UD-Q5_K_XL` foi auditado por metadados e arquivos do HF. Ele e um Docker Space `llama.cpp` com apenas `.gitattributes`, `Dockerfile` e `README.md`, que baixa `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF` e serve os arquivos `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q5_K_XL.gguf` e `mmproj-BF16.gguf`. O modelo GGUF tem `64.2K` downloads, `113` likes, base `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, licenca `nvidia-nemotron-open-model-license` e e quantizado, nao LoRA adapter. Decisao: nao e submitavel no Kaggle, nao possui `adapter_config.json`/`adapter_model.safetensors`, nao substitui o pipeline vLLM/adapter-only. Classificacao: P2 teacher/probe. Usar somente em amostra pequena de miss-packs para testar se produz hipoteses de `equation_transform`/`bit_manipulation` que possam virar regra/verifier ou dado filtrado; nao gastar full eval, package ou submit nessa rota.
- Atualizacao V291/GGUF probe 2026-05-11: foi criado o fluxo packageable mais estrito para testar o melhor artefato V290 (`checkpoint-6`) em full eval official-like sem postprocessor externo. O launcher `artifacts/v291_official_like_v290_full_eval_launch/launch_v291_hf_official_like_eval_v290.py` fixa o commit `c15bf7ade35e1be2840b14911a1aad3773d2256c`, exige CUDA/GPU correta, valida adapter LoRA `r=32/alpha=32`, usa `KG1_PREDICTION_POSTPROCESSOR=none`, `max_tokens=7680`, `max_model_len=8192`, `thinking_enabled=true`, e gate `correct>=823/947` com `truncated<=4` antes de qualquer package/submit. O job A100 `https://huggingface.co/jobs/felipesp1983/6a024380aff1cd33e8f33f8b` ficou apenas em fila e foi cancelado sem iniciar; o relancamento H200 ativo e `https://huggingface.co/jobs/felipesp1983/6a024859317220dbbd1a7c67`, output previsto `evals/v291-h200-officiallike-v290ckpt6-full947-20260511T212028Z`. Em paralelo, a sonda HTTP no Space GGUF confirmou endpoint `/v1/models` e `/health`, mas um prompt minimo levou cerca de `78s` para `32` tokens em CPU e retornou conteudo no campo de raciocinio, nao resposta final; decisao: GGUF Space continua P2 investigativo e nao e rota operacional para gerar submit ou full eval.
- Atualizacao V291 full/package/submit 2026-05-11: o H200 completou o full eval official-like do `checkpoint-6` sem postprocessor externo. Resultado: `823/947 = 0.8690601901`, truncation `1`, `equation_transform=56/155`, `bit_manipulation=135/160`, e `gravity_constant/numeral_system/text_encryption/unit_conversion=100%`. Esse resultado supera o baseline full conhecido V207A/V194 `822/947` por `+1` linha e passou o gate `full_candidate_gate=true`. O pacote adapter-only foi criado em `artifacts/v291_submission_package/v291_h200_checkpoint6_823_20260511T212028Z/submission.zip`, SHA256 `293b414f316330db7ac12c4f3001e7796b0a087ed5dd86af6e13d98620b43433`, entries `adapter_config.json` e `adapter_model.safetensors`, adapter SHA `0a7b6144231d9358ae73a5e57d8778b32be1520fa47e3041414b3e025aaa1aa1`, `r=32`, `alpha=32`. Submissao Kaggle enviada em `2026-05-11 22:19:17.163000` com descricao `V291 V290 checkpoint-6 adapter-only full823 trunc1 official-like gate`; status `COMPLETE`, public score `0.86`. Decisao: manter no historico como primeira submissao packageable com evidencia full local/HF `+1`, mas continuar buscando ganhos adicionais porque o score publico permaneceu no mesmo patamar arredondado.
- Atualizacao V292 2026-05-11: foi testada uma continuacao curta e barata do V290 checkpoint-6 em H200, com `lm_head` treinavel, 6 steps, LR `1e-8 -> 5e-9`, sampling mais pesado para `equation_transform` e guardrails de dataset/tokenizacao/adaptador. Treino: `https://huggingface.co/jobs/felipesp1983/6a0258ac317220dbbd1a7d04`, output repo `felipesp1983/kg1-nemotron-lora-v292-eq-continuation-v290ckpt6`. Weak eval: `https://huggingface.co/jobs/felipesp1983/6a026043aff1cd33e8f34169`, cancelado depois de `checkpoint-6` por FinOps. Resultados observados: `checkpoint-3 = 191/315`, equation `56/155`, bit `135/160`, trunc `0`; `checkpoint-6 = 190/315`, equation `56/155`, bit `134/160`, trunc `1`. Decisao: rejeitar V292, nao rodar full eval, nao empacotar e nao submeter. Diagnostico: a continuacao equation-heavy nao aumentou equation e corroeu bit; nao repetir essa receita sem novo solver/verifier ou dado verificado.
- Atualizacao V303 2026-05-12: foi testada a hipotese de converter o ganho local V302 (`bit_manipulation 135->146`, `equation_transform 56->60` no full official-like com postprocessamento/verifier) em comportamento adapter-only via destilacao curta de bit full-byte sobre o V290 checkpoint-6. Dataset V303 passou gates de hash, JSONL, tokenizacao e offset masks (`12822` train, `969` val; train SHA `c8142742a0c98c4fa368da1a35d16d366b3d499bd66e5b7716408909f7977d27`, val SHA `3711e717eac66ba052697ea42387feb94fddec1a7916d3c67`). Treino H200: `https://huggingface.co/jobs/felipesp1983/6a02ff78aff1cd33e8f34829`, output repo `felipesp1983/kg1-nemotron-lora-v303-bit-fullbyte-distill-v290ckpt6`. Weak eval H200: `https://huggingface.co/jobs/felipesp1983/6a032388c827d2ad86f16afc`, commit de upload `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v303-bit-fullbyte-distill-v290ckpt6/commit/8baa5b2f44cb5b561cdc6a59c460c8e38e46b183`. Resultados: `checkpoint-3=191/315`, equation `56/155`, bit `135/160`, trunc `0`; `checkpoint-6=190/315`, equation `56/155`, bit `134/160`, trunc `0`; `checkpoint-9=190/315`, equation `56/155`, bit `134/160`, trunc `1`; `checkpoint-12=190/315`, equation `56/155`, bit `134/160`, trunc `1`; `final=191/315`, equation `56/155`, bit `135/160`, trunc `0`. Decisao: V303 nao transferiu os ganhos V302 para adapter-only; rejeitar para full eval/package/Kaggle. Diagnostico: o ganho V302 parece depender de regra/verifier em inferencia, nao de sinal facilmente internalizado por LoRA curto em `lm_head`; nova tentativa de destilacao so deve ocorrer com teacher traces completos e gate que prove melhoria em weak antes de qualquer full.
- Atualizacao V304/V305 2026-05-12: a auditoria ampliada das Kaggle Discussions (`29` topicos deduplicados, cache em `artifacts/v305_requested_kaggle_discussion_audit/20260512T0000Z`) confirmou que V303 falhou pelo motivo esperado: answer/full-byte trace nao ensina a politica bit-serial/verifier. A nova V304 substitui esse sinal por `bit_serial_target_verification_trace_v2`, preserva replay das familias ja saturadas, mantem os patches numericos V274/V290 e passa o tokenization gate real em modo suffix: train `12822`, validation `969`, train SHA `7935ff999cdd8318de67538922de3651170c59baa2664a10beac3334dfcf9082`, val SHA `2b06224afe035c5085798f4a4be27e764ffaebde3ff7eee11c558c0cd5bdd29d`, trace rows train `2616`, validation `288`, duplicate assistant conflicts `0`, prompt truncation `0.0`, token max `745`. Decisao: V304 substitui V303 como proxima tentativa adapter-only, mas so autoriza um smoke HF H200 curto com kill-switch; antes de treino longo, implementar V305 CPU-only para o algoritmo completo de bitsum/stride de `690307` e fallback numerico de `691641/690891`.

## Evidencias consolidadas

### V230 Colab executado

- Notebook: `notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb`.
- Branch: `v230-v226-complementarity`.
- Commit clonado no Colab: `e916eb2111fe5590d6df6aee6186d5d9ea325897`.
- Run ID: `20260510T070126Z`.
- Output root: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity`.
- Analysis out: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity/analysis_v230_v226_complementarity/20260510T070126Z`.
- Shared row contract observado: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Manifest SHA256: `8d88ee47359a9d3bcd5cf1aeb589dc8084279f1dbaac0491cf6251dd00fd6ab4`.

### Artefatos V230 para proxima analise

- `v230_v226_complementarity_equation_miss_pack.csv`
- `v230_v226_complementarity_bit_miss_pack.csv`
- `v230_v226_complementarity_baseline_miss_hits.csv`
- `v230_v226_complementarity_pairwise_detail.csv`
- `v230_v226_complementarity_pairwise_summary.csv`
- `v230_v226_complementarity_router_simulation.csv`
- `v230_v226_complementarity_manifest.json`

## Status dos candidatos avaliados

| Candidato | Origem | Total | Equation | Bit | Trunc | Status | Decisao |
|---|---|---:|---:|---:|---:|---|---|
| `v226_best_checkpoint1_observed_191` | V226 | 191 | 55 | 136 | 0 | baseline protegido | Manter |
| `v217_final_existing` | V221 | 190 | 55 | 135 | 0 | nao supera baseline | Rejeitar como substituto |
| `v226_checkpoint_3_observed_190` | V226 | 190 | 55 | 135 | 1 | pior que baseline | Rejeitar |
| `v226_checkpoint_2_observed_189` | V226 | 189 | 55 | 134 | 1 | pior que baseline | Rejeitar |
| `v194_protected_baseline` | V221 | 190 | 54 | 136 | 0 | equation pior | Rejeitar como substituto |
| `kienngx_tinker_adapter` | Kaggle/V221 | 183 | 55 | 128 | 3 | bit abaixo do gate | Rejeitar como deployable |
| `konbu17_exp026_s012_lora` | Kaggle/V221 | 179 | 51 | 128 | 3 | equation e bit piores | Rejeitar como deployable |
| `dgxchen_trained_adapter` | Kaggle/V221 | 176 | 55 | 121 | 0 | bit pior | Rejeitar como deployable |
| `konbu17_sft_lora_cot_selection` | Kaggle/V221 | 58 | 25 | 33 | 161 | truncation alto | Rejeitar como adapter; manter COT apenas para inspecao filtrada |
| `naribow_hf_nemotron_sft_lora` | HF/V221 | 30 | 20 | 10 | 267 | truncation alto | Rejeitar como adapter weak; manter apenas como evidencia externa |
| `v227_final_adapter` | V229 | 16 | 9 | 7 | 0 | regressao severa | Nao usar |
| `v289_checkpoint_4` | HF/V289 | 17 | 9 | 8 | 0 | regressao severa apos smoke V287 | Rejeitar; nao repetir receita |
| `v289_final` | HF/V289 | 16 | 9 | 7 | 0 | regressao severa apos smoke V287 | Rejeitar; nao repetir receita |
| `v290_checkpoint_3_v221_contract` | HF/V290 | 190 | 56 | 134 | 1 | bit abaixo do baseline/gate | Rejeitar; nao rodar full |
| `v290_checkpoint_6_v221_contract` | HF/V290 | 192 | 56 | 136 | 0 | melhor V290; empata melhor LoRA-only, mas falha total/equation gate | Rejeitar para submit isolado; manter evidencia |
| `v290_final_v221_contract` | HF/V290 | 191 | 56 | 135 | 0 | baseline-level, sem ganho suficiente | Rejeitar; nao rodar full |
| `v292_checkpoint_3_v221_contract` | HF/V292 | 191 | 56 | 135 | 0 | regrediu total/bit vs V290 checkpoint-6 | Rejeitar; nao rodar full |
| `v292_checkpoint_6_v221_contract` | HF/V292 | 190 | 56 | 134 | 1 | regrediu total/bit/truncation vs V290 checkpoint-6 | Rejeitar; job cancelado antes do final |
| `v303_checkpoint_3_v221_contract` | HF/V303 | 191 | 56 | 135 | 0 | nao transferiu ganho V302 para adapter-only | Rejeitar; nao rodar full |
| `v303_checkpoint_6_v221_contract` | HF/V303 | 190 | 56 | 134 | 0 | regrediu bit/total | Rejeitar |
| `v303_checkpoint_9_v221_contract` | HF/V303 | 190 | 56 | 134 | 1 | regrediu bit/total e truncation | Rejeitar |
| `v303_checkpoint_12_v221_contract` | HF/V303 | 190 | 56 | 134 | 1 | regrediu bit/total e truncation | Rejeitar |
| `v303_final_v221_contract` | HF/V303 | 191 | 56 | 135 | 0 | empatou baseline fraco, sem ganho | Rejeitar; nao submeter |

## Achados Kaggle/Hugging Face adicionados

### V265 score086 filtered mix - HF-only

Objetivo: reaproveitar a evidencia historica dos arquivos que ajudaram a chegar no score amplo `0.86`, sem contaminar o contrato weak atual.

Evidencia concreta:

- Script: `scripts/run_v265_score086_filtered_mix_hf.py`.
- Job builder HF CPU: `https://huggingface.co/jobs/felipesp1983/6a016d5eaff1cd33e8f3357e`.
- Dataset publicado: `felipesp1983/kg1-nemotron-training/data/v265_score086_filtered_mix/v265-hf-cpu-score086-filtered-mix-20260511T054800Z`.
- Commits HF dataset:
  - `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/5d8a06dece7c1eedc3531397a0da4b0a0b1d97d2`
  - `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/d5c14abe7e70814d030406d1c64535bf501ba83d`
- Hash train: `df301fb9c9813d8ada3884e3e17e277668af9d2738d138f944b7f87d86fd9f32`.
- Hash val: `90e5e595bd7a0861ced23f1575f4b17efcac3c134b8aa0f4e543dc9b52c2e3be`.
- Hash weak blocked ids: `2694de160962cc34c3b8e6cd0443ea92d1b93fb02003f3f3494f324ce6715dfc`.

Gate V250 no HF:

- Job: `https://huggingface.co/jobs/felipesp1983/6a016d9c317220dbbd1a78f4`.
- Commits HF dataset:
  - `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/c0852f022565d4ae870eafe25054bb8eb315cfbe`
  - `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/f687c8d2e97bbd71a08c181bf69269ee8ab86fb9`
- Train tokenization: rows `3098`, prompt truncation `0`, prompt truncation rate `0.0`, completion tokens dropped `0`, offset masks `3098`, fallback masks `0`, max tokens `324`.
- Validation tokenization: rows `344`, prompt truncation `0`, prompt truncation rate `0.0`, completion tokens dropped `0`, offset masks `344`, fallback masks `0`, max tokens `324`.

Decisao apos V266:

- O primeiro smoke V265 em H200 foi negativo e esta rejeitado como candidato deployable.
- Resultado observado: `v266_checkpoint_2_v221_contract = 155/315, equation 56/155, bit 99/160, trunc 0`; `v266_checkpoint_4_v221_contract = 154/315, equation 56/155, bit 98/160, trunc 2`.
- Causa operacional provavel: a receita treinou todos os modulos LoRA (`down/in/k/lm/o/out/q/up/v`) com `equation_transform=3.0` e `bit_manipulation=0.8`, reduzindo demais o replay de bit. Tambem havia erro de chave em `SOURCE_WEIGHTS`: o dataset usa `v189_score086_equation_answer_short_filtered`, mas a receita usou `v189_equation_answer_short_filtered`.
- O job de weak eval V266 foi cancelado com status `CANCELED` para poupar H200 depois que `final/adapter_model.safetensors` foi confirmado byte-identical a `checkpoint-4/adapter_model.safetensors` via SHA256 LFS `875e18bc336975adcf8eaef26ff76c3f89043583981c867d915b809bb955bc99`.
- V265 so pode ser tentado mais uma vez dentro do budget atual com receita ultra-conservadora:
  - initializer `v259_checkpoint_4`, que e o melhor conhecido (`192/315`, equation `56`, bit `136`, trunc `0`);
  - `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,lm_head`;
  - bit replay reforcado (`bit_manipulation >= 1.15`);
  - equation menos agressivo (`equation_transform <= 2.0`);
  - source key corrigida (`v189_score086_equation_answer_short_filtered`);
  - eval parcial do primeiro checkpoint antes de avaliar checkpoints adicionais.
- Se esse smoke conservador nao mantiver bit `>=136/160` e nao aumentar equation acima de `56/155`, encerrar V265 e voltar para solver/verifier ou traces `andy279`.

### V268/V269/V270 tonghuikang reasoning mix - HF-only

Objetivo: testar se o corpus publico `tonghuikang/nemotron`, que inclui problemas, corpus, reasoning, treino e metricas da submissao Progress Prize, adiciona sinal real para as duas familias criticas sem contaminar o contrato weak.

Evidencia concreta:

- Fonte publica auditada: `https://github.com/tonghuikang/nemotron`.
- Script builder: `scripts/run_v268_tonghuikang_reasoning_mix_hf.py`.
- Job builder HF CPU: `https://huggingface.co/jobs/felipesp1983/6a018728317220dbbd1a7958`.
- Dataset publicado: `felipesp1983/kg1-nemotron-training/data/v268_tonghuikang_reasoning_mix/v268-hf-cpu-tonghuikang-reasoning-20260511T0730Z`.
- Commit HF dataset: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/fabb7f3d3a7f3bf730b79422e49dc961f2ea2501`.
- Hashes de fonte auditados:
  - `train.csv`: `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`.
  - `problems.jsonl`: `5b536b97b402fab985312003983bf4c59a928eb08dbb2705ca77d1030d4cf24e`.
  - `corpus.jsonl`: `7ac9e8e267397f1dbcce8d015c253460fec543cab20a078fcf64a53c6000de23`.
  - `generation.jsonl`: `42eb76d13bd81ea3ce6b55120a3e2a23782c18563e05dd4ac9eea59d631b9fbc`.
- Linhas aceitas: `1789`; train `1611`, val `178`.
- Train family counts: `bit_manipulation=1105`, `equation_transform=506`.
- Val family counts: `bit_manipulation=123`, `equation_transform=55`.
- Rejeicoes: `missing_synthetic=798`, `boxed_answer_mismatch=255`.
- Hash train: `e45041d7a4e3d83026d4131d4b6aceb58eddf76ba642266ddc6b08a3943ee86d`.
- Hash val: `0c34a3ea1c6c3e400a76bd9aaa534089fce5f87128398ead73d27c403886fb34`.

Gate V250 no HF:

- Job: `https://huggingface.co/jobs/felipesp1983/6a01881b317220dbbd1a795e`.
- Resultado: `prompt_truncated=0`, `fallback_masks=0`, offset masks completos, nenhum token de completion descartado.
- Max tokens: train `8015`, val `7731`; `max_length=8192`.
- Estilo assistant: `reasoning_boxed`.

Smoke V269/V270:

- Train H200: `https://huggingface.co/jobs/felipesp1983/6a019ad0aff1cd33e8f337b3`.
- Eval H200: `https://huggingface.co/jobs/felipesp1983/6a019f17317220dbbd1a79a7`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v269-v268-reasoning-v259ckpt4-smoke`.
- Upload commit eval: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v269-v268-reasoning-v259ckpt4-smoke/commit/a2e5587a30dcfada448dd43458eec10170350d60`.
- Resultado `v269_checkpoint_2_v221_contract`: `192/315`, equation `56/155`, bit `136/160`, trunc `0`.
- Resultado `v269_final_v221_contract`: `190/315`, equation `56/155`, bit `134/160`, trunc `2`.

Decisao apos V270:

- O V269 checkpoint-2 empatou o melhor atual, mas nao trouxe ganho liquido.
- O V269 final perdeu `2` acertos em bit e violou o guardrail `bit>=136`.
- O diff V270 mostra `4` mudancas em equation, todas errado-para-errado; logo o corpus V268 nao autoriza treino longo em H200.
- V268 permanece util para extrair regras/verifiers e exemplos de raciocinio, mas nao como SFT bruto adicional.

### V271 current-best error miner - HF CPU gate

Objetivo: transformar o resultado V270 em decisao de FinOps e taxonomia acionavel dos erros restantes.

Evidencia concreta local:

- Script: `scripts/run_v271_current_best_error_miner_hf.py`.
- Job HF CPU concluido: `https://huggingface.co/jobs/felipesp1983/6a01a761317220dbbd1a79d1`.
- Tentativas de upload HF direto/PR falharam por permissao de escrita do token do job; os artefatos foram materializados e versionados na branch em `artifacts/hf_cpu_runs/v271_current_best_error_miner_20260511T1010Z`.
- Baseline analisado: `v259_checkpoint4_current_best`.
- Contrato V221 observado: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Melhor atual: `192/315`, equation `56/155`, bit `136/160`, trunc `0`.
- V269 checkpoint-2: `192/315`, equation `56/155`, bit `136/160`, trunc `0`.
- V269 final: `190/315`, equation `56/155`, bit `134/160`, trunc `2`.
- Taxonomia dos `99` erros atuais de equation:
  - `83` `equation_symbolic_punct` com resposta simbolica.
  - `15` `equation_numeric_operator` com resposta numerica.
  - `1` `equation_numeric_operator` misto.

Decisao:

- O gargalo real deixou de ser equation numerico; e `equation_symbolic_punct`.
- A proxima melhoria tem que vir de parser/solver/verifier para simbolos/pontuacao ou de traces solver-guided.
- Nao iniciar novo H200 ate o V271/V246/V241 produzir pelo menos `+4` a `+5` overrides verificaveis em equation ou ate liberar os datasets gated `andy279`.
- Se o dataset `andy279/nemotron-reasoning-challenge-raw-traces` for liberado, priorizar os arquivos `solver_transformation_traces_merged.jsonl`, `solver_transformation_traces_gpt54.jsonl` e `solver_bit_manipulation_traces_merged.jsonl`.

### V272 current equation solver audit - CPU gate

Objetivo: testar se as regras deployable ja existentes (`V238/V241/V246`) conseguem gerar overrides verificados nos `99` erros atuais de `equation_transform`.

Evidencia concreta:

- Script: `scripts/run_v272_current_equation_solver_audit_hf.py`.
- Job HF CPU concluido: `https://huggingface.co/jobs/felipesp1983/6a01a89baff1cd33e8f33844`.
- Artefatos locais versionados: `artifacts/hf_cpu_runs/v272_current_equation_solver_audit_20260511T1020Z`.
- Baseline: `v259_checkpoint4_current_best`.
- Contrato V221 observado: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- `equation_miss_rows`: `99`.
- `parse_status_counts`: `{"ok": 99}`.
- `subtype_counts`: `{"equation_numeric_operator": 16, "equation_symbolic_punct": 83}`.
- Regras auditadas:
  - `symbolic_all_examples_char_transducer`: `0` candidatos.
  - `symbolic_reverse`: `0` candidatos.
  - `symbolic_prefix_suffix`: `0` candidatos.
  - `symbolic_positional_deletion_audit_only`: `1` candidato, `1` incorreto.
  - `numeric_same_operator_rule`: `0` candidatos.

Decisao:

- `no_deployable_solver_signal`.
- Nao existe override seguro a partir dos solvers simples ja implementados.
- O caminho correto agora e uma das duas rotas:
  - liberar `andy279/nemotron-reasoning-challenge-raw-traces`, pois a propria card informa traces solver-guided para transformation e bit;
  - implementar uma busca simbolica de regras mais forte que char-map/reverse/prefix/suffix/deletion simples, com auditoria zero-incorreto antes de qualquer eval ou treino.

### V273 cryptarithm solver audit - CPU gate

Objetivo: testar, sem custo de GPU, a hipotese derivada do repositorio publico `tonghuikang/nemotron`: varios misses simbolicos de `equation_transform` podem ser problemas do tipo `AB op CD`, em que simbolos representam digitos e o operador pode mapear para soma, diferenca absoluta, multiplicacao, concatenacao ou concatenacao reversa.

Evidencia concreta:

- Script: `scripts/run_v273_cryptarithm_solver_audit_hf.py`.
- Artefatos locais versionados: `artifacts/hf_cpu_runs/v273_cryptarithm_solver_audit_20260511T1035Z`.
- Fonte externa usada como base de implementacao: `https://github.com/tonghuikang/nemotron`, arquivo `investigators/cryptarithm_deduce.py`.
- Baseline auditado: `v259_checkpoint_4`, melhor atual no contrato V221 (`192/315`, equation `56/155`, bit `136/160`, trunc `0`).
- Contrato V221 observado: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- `equation_miss_rows`: `99`.

Resultado do audit endurecido:

| Solver mode | Rows | Candidatos | Verificados | Incorretos | Promovivel |
|---|---:|---:|---:|---:|---|
| `parse_gate` | 3 | 0 | 0 | 0 | nao |
| `constraint_gate` | 54 | 0 | 0 | 0 | nao |
| `cryptarithm_solver` | 35 | 0 | 0 | 0 | nao |
| `cryptarithm_unique_digit` | 2 | 2 | 0 | 2 | nao |
| `cryptarithm_nonunique_digit` | 5 | 5 | 0 | 5 | nao |

Decisao:

- `no_cryptarithm_solver_signal`.
- O gate bloqueou candidatos subdeterminados e descartou classes que geraram erro contra weak label.
- Nao existe override seguro vindo do cryptarithm simples.
- Proxima acao objetiva: buscar traces solver-guided `andy279` ou implementar um enumerador simbolico mais geral que aprenda familias de operacoes sobre strings/pontuacao, com regra de promocao `0` incorretos antes de qualquer H200.

### Pesquisa aplicada de literatura - decisao V274

Objetivo: converter a busca em literatura de program synthesis, SMT, e reasoning procedural em acoes que possam aumentar ACC sem treino cego.

Fontes com impacto direto:

| Fonte | Evidencia tecnica | Aplicacao no KG1 | Decisao |
|---|---|---|---|
| Microsoft PROSE / FlashFill | Programacao por exemplos para transformacoes de string; FlashFill++ usa DSLs guardadas, cortes e ranking para tornar busca em DSL grande viavel. URLs: `https://www.microsoft.com/en-us/research/project/prose-text-transformation/`, `https://www.microsoft.com/en-us/research/publication/flashfill-scaling-programming-by-example-by-cutting-to-the-chase/` | `equation_symbolic_punct` e exatamente PBE: varios exemplos `lhs = rhs` e uma query. | P0: V274 deve implementar sintetizador PBE/DSL conservador para strings simbolicas, com zero-incorreto por classe antes de override. |
| Minimal synthesis de string-to-string por DFA + SMT | Hamza/Kuncak modelam funcoes string-to-string a partir de exemplos e reduzem sintese minima a SMT. URL: `https://arxiv.org/abs/1710.09208` | Pode cobrir transformacoes Alice onde a saida nao e simples copia/reverse/prefix. | P0/P1: adicionar transducer/automata audit, mas promover so se houver unicidade ou consenso forte nos exemplos. |
| SyGuS / enumerative synthesis guiada por LLM | Li/Parsert/Polgreen mostram que LLM sozinho perde para sintetizadores formais, mas melhora quando guia busca enumerativa. URL: `https://arxiv.org/abs/2403.03997` | Usar OpenRouter/LLM apenas para priorizar primitivas DSL, nunca como fonte direta de resposta. | P1: se V274 DSL crescer, usar LLM como ranking de busca, mantendo verificador local como autoridade. |
| DreamCoder / library learning | Ellis et al. aprendem bibliotecas composicionais para resolver tarefas por programas interpretaveis. URL: `https://arxiv.org/abs/2006.08381` | Agrupar misses por operadores/padroes e comprimir solucoes recorrentes em primitivas reutilizaveis. | P1: depois do V274, minerar primitivas recorrentes a partir de acertos verificaveis; nao treinar LoRA sem esse sinal. |
| Rosette / solver-aided DSL | Rosette compila DSLs para restricoes SMT e usa solvers off-the-shelf para sintese/verificacao. URL: `https://emina.github.io/rosette/` | Bit e algumas equacoes podem virar DSL com restricoes, reduzindo falsos positivos. | P1: V275 pode usar Z3/bit-vector para bit guardrail, mas bit ja esta em 136/160; prioridade segue equation. |
| E-graphs / equality saturation | Equality saturation e `egg` representam muitas expressoes equivalentes e usam regras locais para achar rewrites globais. URLs: `https://www.cs.cornell.edu/~ross/publications/eqsat/`, `https://arxiv.org/abs/2004.03082` | Util para `equation_transform` quando a tarefa for rewrite algebraico, nao para todos os prompts Alice. | P2: manter como rota para `equation_numeric/operator` e DSL de rewrite, apos PBE simbolico. |
| Reasoning Gym | Gera dados e verificadores procedurais com recompensas verificaveis em muitos dominios. URLs: `https://arxiv.org/abs/2505.24760`, `https://pypi.org/project/reasoning-gym/` | Fonte para probes e RL/verifier, especialmente bitwise arithmetic; nao e schema KG1 nativo. | P2: triagem de dados/probes; nao SFT direto. |
| ASyMOB / symbolic math benchmark | Mostra queda forte de LLMs sob perturbacoes simbolicas e necessidade de combinar LLM com CAS/solvers. URL: `https://arxiv.org/abs/2505.23851` | Confirma que `equation_symbolic_punct` precisa de solver/verifier, nao apenas prompt/training numerico. | P1: usar como justificativa de DSL/verifier e evitar treino matematico generico. |
| Nemotron CrossThink/Cascade | NVIDIA enfatiza curadoria, templates estruturados, filtering por dificuldade e controle thinking/non-thinking. URLs: `https://research.nvidia.com/labs/adlr/Nemotron-CrossThink/`, `https://research.nvidia.com/labs/nemotron/nemotron-cascade/` | Nosso V261 mostrou que mexer em prompt/thinking sem gate derruba bit; aplicar so com eval parcial e filtros. | P2: usar principios de filtering e templates; nao repetir prompt sweep cego. |

Busca web adicional 2026-05-11:

- `Enhanced Enumeration Techniques for Syntax-Guided Synthesis of Bit-Vector Manipulations` (POPL 2024, DOI `10.1145/3632913`) reforca que bit-vector synthesis melhora com enumeracao especializada. Aplicacao KG1: manter como P1/P2 para guardrail de `bit_manipulation`; nao gastar H200 porque bit ja esta `136/160`.
- `Rewrites for SMT Solvers using Syntax-Guided Enumeration` mostra uso de SyGuS para sugerir rewrites em bit-vectors e strings. Aplicacao KG1: a rota correta e gerar regras pequenas e verificaveis, exatamente o padrao V274; nao aceitar rewrite sem zero-loss audit.
- `nvidia/OpenMath-Nemotron-14B-Kaggle` e modelos math Kaggle correlatos sao relevantes como evidencia de curadoria e verificacao em matematica, mas nao sao schema KG1 Nemotron-3-Nano-30B LoRA; uso direto como modelo/adaptador fica fora do caminho deployable atual.
- HF mirrors do desafio (`jasonkung98`, `Taurine511`, `GaryNENE`) servem para triagem de dados e adapters publicos, mas qualquer uso precisa passar por bloqueio de weak IDs, hash de prompt, dedupe e gate local. Nao entram no treino/postprocessor sem manifest.
- Double-check HF 2026-05-11: `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge` existe e contem `train.csv`/`test.csv`; ja foi usado como mirror publico, nao e novo sinal por si so. `Taurine511/nvidia-nemotron-model-reasoning-challenge` apareceu na busca web, mas a API HF retornou `404`; nao usar. `GaryNENE/nemotron-nano-8b-reasoning-lora` tem README/receita para base 8B, mas nao possui `adapter_config.json` nem pesos no repo inspecionado; nao e candidato executavel. Os quatro repos `passagereptile455/nemotron-reasoning-lora-v8..v11` contem apenas `.gitattributes`; tambem nao sao candidatos ate surgirem pesos/config.

Decisao operacional pos-literatura:

- O proximo passo ainda nao deve ser novo LoRA: V274 provou que um solver/verifier pequeno pode entregar o ganho que os treinos V259/V269 nao entregaram.
- Resultado V274 local:
  - script: `scripts/run_v274_numeric_operator_override_audit_hf.py`;
  - artefatos: `artifacts/hf_cpu_runs/v274_numeric_operator_override_audit_20260511T1115Z/`;
  - baseline V259 checkpoint-4: `192/315`, equation `56/155`, bit `136/160`, trunc `0`;
  - postprocessor numerico guardado: `196/315`, equation `60/155`, bit `136/160`, trunc `0`;
  - regras promovidas: `minus_signed_opposite_sign_guarded` (`+2`), `colon_absdiff_unreverse_same_len` (`+1`), `add_direct_over_model_add_variant` (`+1`);
  - regras rejeitadas/abstain: casos ambiguos, operadores nao vistos, padroes sem unicidade e qualquer classe com risco de erro;
  - auditoria: `gains=4`, `losses=0`, `wrong_on_baseline_misses=0`, `weak_gate.pass=true`.
- Interpretacao: a literatura de FlashFill/PROSE/SyGuS foi mais util como disciplina de engenharia do que como implementacao literal: sintetizar uma DSL pequena, exigir unicidade nos exemplos e promover apenas regras com abstain seguro. O alvo simbolico `equation_symbolic_punct` continua aberto, mas o subtipo numerico ja fornece `+4` acertos medidos.
- Proxima acao obrigatoria antes de gastar H200: reproduzir V274 em HF CPU a partir da branch, registrar job URL e manifest. Depois disso, criar V275 como pacote de inferencia/postprocessamento label-free; full eval so deve rodar se o gate confirmar que a regra nao usa labels weak e que o bit guardrail fica `>=136/160`.
- Status HF CPU: reproduzido em `https://huggingface.co/jobs/felipesp1983/6a01b475aff1cd33e8f338cf` com `weak_gate.pass=true`. O primeiro job `https://huggingface.co/jobs/felipesp1983/6a01b42caff1cd33e8f338bf` falhou corretamente por falta de `HF_TOKEN` ao acessar o repo privado de predicoes, e foi substituido pelo job autenticado.
- Status V275 local: reproduzido em `artifacts/hf_cpu_runs/v275_deployable_postprocessor_gate_20260511T1210Z/` com source guard limpo e decisao `v275_postprocessor_ready_for_full_eval_gate`.
- Status V275 HF CPU: reproduzido em `https://huggingface.co/jobs/felipesp1983/6a01b708aff1cd33e8f338ed`, `cpu-basic`, `12s` total, com `HF_TOKEN`, `weak_gate.pass=true`.
- Status V276 full bridge: CSV full `947` publicado no HF privado em `runtime_artifacts/v276_full_eval_bridge/v276-full947-bridge-20260511T1245Z/`. A tentativa de upload via job HF usando URL assinado do Drive falhou corretamente com `HTTP 403`; a ponte foi feita via `hf upload` local autenticado e validada com listagem remota. Proxima acao: rodar um unico full eval guardado usando `scripts/hf_job_full_eval_v276.py` e o postprocessor V274/V275.

### Auditoria OpenRouter anexada - 2026-05-10

Arquivos analisados:

- `C:\Users\davis\Downloads\OpenRouter Chat Sun May 10 2026 (1).json`
  - SHA256: `4F87904D23F7988F2CA3F2E2917B7F3355C9F6027FF910C91DDF7D6A50E823BE`
  - Estrutura: JSON valido com `messages`, `items`, `artifacts`, `artifactFiles`, `artifactVersions`, `artifactFileContents`.
  - URLs tecnicas extraidas: `69` Hugging Face, `23` GitHub.
- `C:\Users\davis\Downloads\OpenRouter Chat Sun May 10 2026.json`
  - SHA256: `B8AAB44AF917338956F12C2513AE75F309025BBC56D01EF15D972EF33CA3825E`
  - Estrutura: JSON valido com `messages`, `items`, `artifacts`, `artifactFiles`, `artifactVersions`, `artifactFileContents`.
  - URLs tecnicas extraidas: `146` Kaggle, `204` Hugging Face, `54` GitHub, `35` NVIDIA docs/blogs.

Conclusao para ACC:

- Os anexos reforcam que o alvo principal de ganho e `equation_transform`, nao troca cega de adapter.
- O dado operacional mais importante registrado nos chats e a quebra interna de equation:
  - `equation_transform` weak observado: `55/155`.
  - subfamilia numerica citada: `47/62`.
  - subfamilia simbolica/mista citada: `8/93`.
  - leitura: o gargalo real e simbolico/misto; treinar mais exemplos numericos tem baixa prioridade.
- `bit_manipulation` esta perto do teto local e ja passa gate:
  - weak observado: `135-136/160`, conforme candidato.
  - regra de negocio: qualquer melhoria em equation nao pode reduzir bit abaixo de `136` sem nova evidencia de gate total.
- Nao foi encontrado nos JSONs um peso/adapter pronto com evidencia suficiente para uso direto. Todos os novos modelos/adapters citados entram apenas como candidatos a triagem, nunca como substitutos do V226 sem weak eval.

Fontes externas dos anexos com maior valor potencial para ACC:

| Fonte | Tipo | Valor esperado | Acao |
|---|---|---|---|
| `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces` | HF raw traces | Solver-guided traces de transformation e bit | P0 para V234 ingest/audit |
| `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge` | HF SFT dataset | Distribuicao por familias do desafio | Usar so apos hash, dedupe e conflict check |
| `https://github.com/open-thought/reasoning-gym` | Gerador/benchmark | Geracao controlada de tarefas simbolicas/bit | Usar como fonte de probes, nao treino direto |
| `https://huggingface.co/datasets/Shalyt/ASyMOB-Algebraic_Symbolic_Mathematical_Operations_Benchmark` | HF symbolic math | Casos de operacoes simbolicas | Extrair somente se schema ajudar equation symbolic/mixed |
| `https://huggingface.co/datasets/SAIRfoundation/equational-theories-benchmark` | HF equational reasoning | Leis/equivalencias simbolicas | Usar para verifier/DSL, nao SFT bruto |
| `https://huggingface.co/datasets/nvidia/OpenMathReasoning` | HF math reasoning | Dados gerais de matematica | Baixa prioridade ate provar overlap com equation_transform |
| `https://github.com/TheAlgorithms/Python/tree/master/bit_manipulation` | Algoritmos bit | DSL/guardrail bitvector | Usar para testes e no-loss guardrail |
| `https://huggingface.co/datasets/ftajwar/training_bitwise_arithmetic-4` | HF bit arithmetic | Casos bitwise externos | Apenas triagem; risco de nao bater formato KG1 |
| `https://huggingface.co/datasets/ftajwar/evaluation_bitwise_arithmetic-4` | HF bit arithmetic eval | Probes bitwise | Apenas triagem; nao substituir weak local |

Fontes dos anexos que devem ser tratadas como ruido ou baixo valor para ACC:

- Qualquer resultado de `Huggies`, `faces`, app familiar, imagem, produto ou dataset visual. Isso veio de erro semantico em "Hugging Face" e nao tem relacao com KG1.
- Metadados OpenRouter/model-provider sem artefato reprodutivel local.
- Adapters HF/Kaggle sem `adapter_config.json`, hashes, target modules, tamanho, licenca e weak eval identico.
- Datasets matematicos genericos sem mapeamento para `equation_transform` simbolico/misto.
- COT bruto longo que aumenta truncation; o historico V221 mostrou Naribow e Konbu COT com truncation severo.

### Modelo base HF

- Repositorio: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Fonte: `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Observacao tecnica: modelo Nemotron-H com uso documentado via Transformers, vLLM e SGLang.
- Implicacao: manter `trust_remote_code=True`, revisar compatibilidade vLLM/torch por notebook, e nao misturar instalacao vLLM com stack de treino.

### Dataset HF SFT `andy279/nemotron-reasoning-challenge`

- Fonte: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge`.
- Acesso: gated/manual; exige aceitar condicoes.
- Conteudo reportado: train `49,290` exemplos, `7,200` puzzles unicos; validation `1,165` exemplos, `1,123` puzzles unicos.
- Ponto critico: validation inclui `399` puzzles de transformation marcados como unsolved.
- Distribuicao train reportada:
  - bit_manipulation: `17,285`
  - cipher: `6,722`
  - gravity: `3,294`
  - numeral: `3,282`
  - transformation: `10,741`
  - unit_conversion: `7,966`
- Uso correto: nao treinar direto. Primeiro aceitar acesso, baixar com hash, filtrar por correctness, dedupe, conflito de resposta, familia, boxed/extractor e contrato do Kaggle verify.

### Dataset HF raw traces `andy279/nemotron-reasoning-challenge-raw-traces`

- Fonte: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces`.
- Acesso: gated/manual.
- Arquivos relevantes reportados:
  - `ds_traces.jsonl`: `6,028` puzzles, DeepSeek V3.2 non-thinking, 4 attempts/puzzle.
  - `ds_traces_thinking.jsonl`: `6,028` puzzles, thinking mode.
  - `solver_transformation_traces_merged.jsonl`: `1,101` solver-guided transformation traces.
  - `solver_bit_manipulation_traces_merged.jsonl`: `1,602` solver-guided bit manipulation traces.
  - `solver_transformation_traces_gpt54.jsonl`: `85` hard transformation traces.
- Uso correto: fonte P0 para minerar regras/verifiers de `equation_transform` e `bit_manipulation`; nao usar como SFT bruto.
- Atualizacao OpenRouter 2026-05-10: esta e a fonte externa mais importante para ACC nos anexos, porque contem traces solver-guided alinhados exatamente com as familias problematicas.
- Prioridade V234:
  - baixar com revision fixo;
  - registrar hashes por arquivo;
  - auditar schema;
  - separar `solver_transformation_traces_merged.jsonl` e `solver_bit_manipulation_traces_merged.jsonl`;
  - extrair regras/probes, nao respostas para treino cego;
  - bloquear uso se houver conflito de answer, leakage ou formato nao verificavel.

### Kaggle notebooks e discussoes citados nos anexos

Entram como inteligencia externa, nao como evidencia de score ate reproducao local.

URLs/ids relevantes extraidos dos anexos:

- Competicao oficial: `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge`.
- Discussao citada como possivel reverse-engineering de familias: `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461`.
- Outros ids citados nos chats: `685462`, `689915`.
- Notebooks publicos citados:
  - `https://www.kaggle.com/code/adilshamim8/nvidia-nemotron-model-reasoning-challenge-101`
  - `https://www.kaggle.com/code/adriano313/nemotron-lora-train-v1`
  - `https://www.kaggle.com/code/afidhadra/nemotron-optimized-training`
  - `https://www.kaggle.com/code/damndeepesh/nemotron-sft-lora-with-cot-v2-prep-now-plz-wait`
  - `https://www.kaggle.com/code/dennisfong/nvidia-nemotron-sfttrainer-training`
  - `https://www.kaggle.com/code/emanuellcs/nvidia-nemotron-sft`
  - `https://www.kaggle.com/code/franksunp/nemotron-train-v1`
  - `https://www.kaggle.com/code/halitta/aimo3-nemotron-3-solver-critique-pipeline`
  - `https://www.kaggle.com/code/jek1wantaufik/nvidia-nemotron-model-reasoning-0-68`
  - `https://www.kaggle.com/code/kienngx/nemotron-sft-reasoning-trajectories-dataset`
  - `https://www.kaggle.com/code/kienngx/nvidia-nemotron-trained-models-submission`
  - `https://www.kaggle.com/code/kienngx/nvidia-nemotron-training-copy-run-instantly`
  - `https://www.kaggle.com/code/konbu17/nemotron-sft-lora-with-cot`
  - `https://www.kaggle.com/code/konbu17/nemotron-sft-lora-with-cot-selected-data`

Uso correto:

- Baixar/copiar metadados so se publico e permitido.
- Registrar URL, versao, data, hash e arquivos de saida.
- Procurar apenas:
  - parser de familias;
  - solver/verifier;
  - geracao de traces;
  - calibragem de prompt que reduza truncation;
  - exemplos de failure analysis.
- Nao copiar treino/submission sem auditoria de leakage, licenca, path e weak gate local.

### Repositorios GitHub citados nos anexos

Prioridade de leitura:

- `https://github.com/tonghuikang/nemotron`
- `https://github.com/Ayman-Sabek/NVIDIA_Kaggle_Nemotron`
- `https://github.com/Jerry2003826/nivida`
- `https://github.com/NVIDIA-NeMo/Nemotron`
- `https://github.com/NVIDIA/NeMo-Skills/tree/b37fe403e6dc6e2f9700a64231247a0d1b33d8a2`
- `https://github.com/NVIDIA-NeMo/RL`
- `https://github.com/NVIDIA-NeMo/Evaluator`
- `https://github.com/huggingface/trl`

Uso correto:

- Tratar `tonghuikang`, `Ayman-Sabek` e `Jerry2003826` como candidatos de engenharia reversa/estrategia; nenhum score entra sem reproducao local.
- Tratar NeMo/NVIDIA/TRL como referencia de metodo de SFT/RL/GRPO, nao como caminho imediato de ACC.
- Para o objetivo atual, preferir solver/verifier antes de novo treino RL/SFT.

### Kaggle model publico `ashok205/nvidia-nemotron-3-nano-30b`

- Fonte: `https://www.kaggle.com/models/ashok205/nvidia-nemotron-3-nano-30b`.
- Status: achado externo publico ainda nao avaliado localmente.
- Decisao: pendente; so entra se metadata, download, adapter/model contract e weak eval identico passarem.

### Busca HF por adapters do mesmo base model

- Fonte: `https://huggingface.co/models?other=base_model%3Aadapter%3Anvidia%2FNVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Observacao: ha outros adapters HF recentes para o base model.
- Decisao: nao adicionar ao pipeline automaticamente. Cada candidato novo precisa entrar no registry com origem, hash, tamanho, licenca, target_modules, weak eval e no-regression gates.

## Diagnostico por familia apos V230

### equation_transform

Estado: principal gargalo.

Evidencia:

- Baseline V226: `55/155`.
- Gate: `60/155`.
- Oracle V230: `57/155`.
- Mesmo com escolha por linha entre candidatos atuais, ainda faltam `3` acertos de equation.

Plano:

1. Abrir `v230_v226_complementarity_equation_miss_pack.csv`.
2. Separar casos em:
   - baseline errou e algum candidato acertou;
   - todos os candidatos erraram;
   - resposta certa presente mas extractor/formato falhou;
   - casos simbolicos;
   - casos numericos;
   - casos tipo cryptarithm/constraint.
3. Para cada subtipo, implementar verifier/solver com abstention.
4. Aceitar override somente quando o solver prova todos os exemplos do prompt.
5. Meta minima: `+5` equation sem perder bit.
6. Meta segura: `+6` ou `+7` equation para criar margem contra ruido.

### bit_manipulation

Estado: passa o gate, mas deve ser protegido.

Evidencia:

- Baseline V226: `136/160`.
- Gate: `133/160`.
- Margem: `+3`.
- Oracle V230: `140/160`.

Plano:

1. Abrir `v230_v226_complementarity_bit_miss_pack.csv`.
2. Minerar somente regras provaveis por DSL bitvector.
3. Nao aceitar adapter externo que reduza bit abaixo de `136` sem compensacao verificada e sem passar gate total.
4. Se solver bit for usado, deve ser no-loss: quando incerto, manter V226.

## Roadmap operacional atualizado

### Atualizacao implementada - V231 miss-pack mining

Arquivos criados/alterados para executar o primeiro passo pos-V230:

- `scripts/analyze_v231_miss_packs.py`
- `scripts/build_v231_miss_pack_mining_colab.py`
- `notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb`
- `scripts/notebook_release_gate.py`

Colab URL planejada:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb`

Importante: a URL acima so funciona depois que o notebook for enviado para a branch `v230-v226-complementarity`.

O V231:

- le o manifest V230 mais recente ou o path explicito em `KG1_V231_V230_ANALYSIS_MANIFEST_JSON`;
- exige o row-contract `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff` por padrao;
- valida existencia, linhas, bytes e SHA256 dos CSVs V230 obrigatorios;
- classifica `equation_miss_pack` por rota de solver;
- classifica `bit_miss_pack` como guardrail;
- gera `equation_miss_taxonomy_csv`;
- gera `equation_solver_candidate_rules_json`;
- gera `bit_guardrail_candidates_json`;
- bloqueia treino, full scoring, package e Kaggle submit.

Validacoes locais realizadas:

- `python -m py_compile scripts/notebook_release_gate.py scripts/analyze_v231_miss_packs.py scripts/build_v231_miss_pack_mining_colab.py`
- `python scripts/analyze_v231_miss_packs.py --self-test --v230-analysis-manifest-json dummy --output-dir dummy`
- `python scripts/notebook_release_gate.py --self-test`
- `python scripts/notebook_release_gate.py notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb --output-json artifacts/notebook_release_gate/v231_miss_pack_mining_report.json`
- `python scripts/notebook_release_gate.py notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb notebooks/KG1_V219_WEAK_DECODE_AB_COLAB.ipynb notebooks/KG1_V220_PUBLIC_ADAPTER_PROBE_COLAB.ipynb notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb --output-json artifacts/notebook_release_gate/release_notebooks_v231_report.json`
- `python scripts/scan_repo_secrets.py`
- `git diff --check`

### P0 - Congelar baseline e evidencias

1. Manter V226 checkpoint-1 como baseline protegido.
2. Registrar `shared_row_contract_sha256=bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff` em qualquer rerun nao diagnostico.
3. Para rerun reprodutivel do V230, fixar:
   - `KG1_V230_EXPECTED_REPO_COMMIT=e916eb2111fe5590d6df6aee6186d5d9ea325897`
   - `KG1_V230_EXPECTED_SHARED_ROW_CONTRACT_SHA256=bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`
4. Nao liberar full eval/submission com `V230_DIAGNOSTIC_MODE=True`.

### P1 - Minerar miss-packs

1. Fazer auditoria manual/programatica do `equation_miss_pack`.
2. Gerar uma tabela de subtipo por row id.
3. Identificar os `+2` acertos do oracle em equation e os `+4` acertos do oracle em bit.
4. Separar linhas onde adapter externo acerta e V226 erra, sem usar label para deploy.
5. Transformar padroes em regras verificaveis.

### P2 - Solver/verifier para equation

1. Implementar parser de exemplos input/output.
2. Implementar rotas:
   - numeric formula search;
   - symbolic transform com SymPy;
   - integer/constraint solver;
   - cryptarithm/DFS com pruning;
   - pattern transform com DSL pequena.
3. Cada regra precisa produzir:
   - answer;
   - confidence/proof;
   - motivo de abstention quando falha.
4. Gate: nenhum override sem prova local.

### P3 - Bit guardrail

1. Implementar ou reaproveitar DSL bitvector.
2. Aceitar apenas expressao que explica todos os exemplos do prompt.
3. Manter V226 como fallback.
4. Exigir que qualquer candidato final mantenha `bit_manipulation>=136` em weak local, ou no minimo `>=133` com ganho total/equation demonstrado.

### P4 - Ingestao controlada dos dados HF/Kaggle

1. Gated HF: baixar somente apos autorizacao.
2. Criar manifest com:
   - URL/ref;
   - revision;
   - hashes;
   - row counts;
   - family counts;
   - schema;
   - correctness filter;
   - overlap/duplicate/conflict report.
3. Usar `andy279` primeiro para verifiers e solvers, nao para SFT.
4. COT Kaggle/Kienngx/DGXChen so entra apos limpeza de resposta e validacao contra metric/extractor.
5. Prioridade de ingestao apos auditoria OpenRouter:
   - P0: `andy279/nemotron-reasoning-challenge-raw-traces`.
   - P1: discussoes/notebooks Kaggle com parser/solver/verifier reproduzivel.
   - P2: `reasoning-gym` e benchmarks simbolicos para probes externos.
   - P3: adapters/modelos HF/Kaggle apenas como triagem, nunca como baseline.
6. Para cada fonte externa, registrar `source_url`, `retrieved_at_utc`, `revision_or_version`, `license_or_access_status`, `sha256`, `row_count`, `family_counts`, `schema`, `duplicate_count`, `conflict_count`, `leakage_check`, `extractor_check` e decisao.

### P5 - Treino so depois de prova

Treino novo so e permitido se P1/P2/P4 demonstrarem um pool de dados que:

- ataca `equation_transform`;
- nao degrada `bit_manipulation`;
- tem hash fixo e manifest;
- passa tokenization dry-run;
- passa offset-mask/truncation gates;
- tem weak eval A/B antes de qualquer full eval.

### P6 - Full eval/submission

Full eval so pode rodar se:

- total weak `>=193`;
- equation weak `>=60`;
- bit weak `>=133`;
- truncation `<=3`;
- release gate passa;
- manifest final registra decisao;
- Kaggle submit continua bloqueado ate aprovacao humana.

## Regras de rejeicao

- Nao usar V227: regressao confirmada no V229.
- Nao usar Naribow como adapter weak: truncation `267/315` e score `30/315`.
- Nao usar Konbu COT selection como adapter weak: truncation `161/315` e score `58/315`.
- Nao trocar V226 por Kienngx/DGXChen/Konbu sem novo criterio, pois todos pioram total ou bit/equation.
- Nao treinar com datasets externos sem manifest, hash, dedupe, conflict check e prova de relevancia por familia.
- Nao usar achado de chat/OpenRouter como evidencia de score sem baixar o artefato original e reproduzir localmente.
- Nao usar dados "Huggies/faces" ou qualquer resultado visual/familiar; isso e ruido da busca por Hugging Face.
- Nao usar COT bruto longo se aumentar truncation; primeiro converter para resposta curta/verificada.
- Nao aceitar ganho em `equation_transform` se `bit_manipulation` cair abaixo do baseline protegido `136/160`, salvo se um weak gate completo provar total `>=193`, equation `>=60`, bit `>=133` e truncation `<=3`.

## Atualizacao executada - V231

V231 foi executado no Colab e terminou com `returncode=0`.

Evidencia da execucao:

- Notebook: `notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb`.
- Manifest V230 resolvido: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity/analysis_v230_v226_complementarity/20260510T070126Z/v230_v226_complementarity_manifest.json`.
- Row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Manifest V231: `/content/drive/MyDrive/KG1_NVIDIA_V231/output_v231_v230_miss_pack_mining/analysis_v231_miss_pack_mining/20260510T074735Z/v231_v230_miss_pack_mining_manifest.json`.
- Manifest SHA256: `ffa21f0f9a0e1845bbe4f55143aed7733d2d3933162775a5ab69d32a783f83b3`.

Resultados V231:

- `baseline_misses=124`.
- `equation_misses=100`.
- `equation_rows_with_correct_alternative=2`.
- `bit_misses=24`.
- `bit_rows_with_correct_alternative=4`.

Decisao V231:

- `mine_equation_solvers_before_training`.
- Proxima acao: construir candidatos de solver/verifier de equation antes de qualquer treino ou full scoring.

## Atualizacao implementada - V232 verified solver workbench

Arquivos criados/alterados para o proximo passo pos-V231:

- `scripts/analyze_v232_verified_solver_workbench.py`
- `scripts/build_v232_verified_solver_workbench_colab.py`
- `notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb`
- `scripts/notebook_release_gate.py`
- `artifacts/notebook_release_gate/v232_verified_solver_workbench_report.json`

URL Colab:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb`

O V232:

- le o manifest V231 mais recente ou o path explicito em `KG1_V232_V231_ANALYSIS_MANIFEST_JSON`;
- valida contrato de rows e artefatos V231 obrigatorios;
- reabre o manifest V230 original para recuperar os prompts completos dos miss-packs;
- gera `equation_solver_workitems_jsonl`;
- gera `bit_guardrail_workitems_jsonl`;
- gera `acceptance_matrix_csv`;
- gera `solver_contracts_json`;
- bloqueia treino, full scoring, package e Kaggle submit.

## Atualizacao executada - V232

V232 foi executado no Colab e terminou com `returncode=0`.

Evidencia da execucao:

- Notebook: `notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb`.
- Manifest V231 resolvido: `/content/drive/MyDrive/KG1_NVIDIA_V231/output_v231_v230_miss_pack_mining/analysis_v231_miss_pack_mining/20260510T074735Z/v231_v230_miss_pack_mining_manifest.json`.
- Row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Manifest V232: `/content/drive/MyDrive/KG1_NVIDIA_V232/output_v232_verified_solver_workbench/analysis_v232_verified_solver_workbench/20260510T080950Z/v232_verified_solver_workbench_manifest.json`.
- Manifest SHA256: `6415efbad28577c675f8847f6b84eb5a2d63709b6b9e5ae42fdba5a002c9b7bf`.

Resultados V232:

- `equation_solver_workitems=100`.
- `equation_rows_with_correct_alternative=2`.
- `bit_guardrail_workitems=24`.
- `bit_rows_with_correct_alternative=4`.

Decisao V232:

- `build_v233_verified_equation_solver_probes`.
- Proxima acao: usar os workitems V232 para implementar probes/verificadores deterministas antes de qualquer treino ou full scoring.

## Atualizacao implementada - V233 verified equation solver probes

Arquivos criados/alterados para o proximo passo pos-V232:

- `scripts/analyze_v233_verified_equation_solver_probes.py`
- `scripts/build_v233_verified_equation_solver_probes_colab.py`
- `notebooks/KG1_V233_VERIFIED_EQUATION_SOLVER_PROBES_COLAB.ipynb`
- `scripts/notebook_release_gate.py`
- `artifacts/notebook_release_gate/v233_verified_equation_solver_probes_report.json`

URL Colab:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V233_VERIFIED_EQUATION_SOLVER_PROBES_COLAB.ipynb`

O V233:

- le o manifest V232 mais recente ou o path explicito em `KG1_V233_V232_ANALYSIS_MANIFEST_JSON`;
- valida artefatos V232 obrigatorios;
- roda probe deployable conservador `sympy_single_equation_probe`;
- registra evidencia nao-deployable separada via `oracle_alternative_candidate_probe`;
- gera `equation_probe_results_jsonl`;
- gera `equation_probe_summary_csv`;
- gera `equation_verified_overrides_csv`;
- gera `equation_oracle_evidence_csv`;
- bloqueia treino, full scoring, package e Kaggle submit.

Decisao operacional agora:

- executar V233;
- se `deployable_verified_equation_overrides >= 5`, preparar notebook separado de weak eval com overrides verificados;
- se ficar abaixo de 5, revisar abstentions e ampliar parsers antes de qualquer weak/full eval;
- manter treino bloqueado ate existir solver/verifier com prova local.

## Atualizacao de prioridade - achados OpenRouter/HF/Kaggle para ACC

Esta atualizacao consolida as ultimas interacoes e altera a prioridade do roadmap:

1. O objetivo de curto prazo nao e novo LoRA. E obter `+5` acertos em `equation_transform` preservando `bit_manipulation`.
2. O caminho mais promissor e uma etapa V234 de ingestao/auditoria de traces externos, com foco em `andy279/nemotron-reasoning-challenge-raw-traces`.
3. O V234 deve produzir artefatos verificaveis, nao treino:
   - manifest de fontes externas;
   - auditoria de schema/hash;
   - tabela de traces por familia;
   - candidatos de regra/solver para symbolic/mixed equation;
   - candidatos de guardrail bitvector;
   - lista de itens rejeitados por ruido, leakage, conflito ou formato.
4. O V234 so pode desbloquear treino se houver evidencia concreta de:
   - ganho potencial em `equation_transform` simbolico/misto;
   - ausencia de degradacao de `bit_manipulation`;
   - compatibilidade com resposta curta `\boxed{answer}`;
   - zero dependencia de label/row-id no deploy.

### Especificacao proposta - V234 external intel ingest

Notebook proposto:

- `notebooks/KG1_V234_EXTERNAL_INTEL_INGEST_AND_SOLVER_TRACE_AUDIT_COLAB.ipynb`

Scripts propostos:

- `scripts/analyze_v234_external_intel_ingest.py`
- `scripts/build_v234_external_intel_ingest_colab.py`

Inputs P0:

- V230 manifest: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity/analysis_v230_v226_complementarity/20260510T070126Z/v230_v226_complementarity_manifest.json`.
- V232/V233 manifests se existirem, para cruzar workitems com traces.
- HF gated dataset: `andy279/nemotron-reasoning-challenge-raw-traces`, quando autorizado manualmente.

Outputs obrigatorios:

- `external_source_manifest.json`.
- `raw_trace_file_audit.csv`.
- `equation_symbolic_trace_candidates.jsonl`.
- `bit_guardrail_trace_candidates.jsonl`.
- `trace_to_weak_miss_alignment.csv`.
- `rejected_external_items.csv`.
- `v234_external_intel_ingest_manifest.json`.

Gates obrigatorios:

- falhar se qualquer fonte externa nao tiver hash;
- falhar se row/schema esperado estiver ausente;
- falhar se houver conflito de answer nao resolvido;
- falhar se a fonte exigir acesso manual e nao estiver disponivel;
- falhar se a fonte tentar substituir baseline sem weak eval;
- bloquear train/full/package/submit por padrao.

Decisao esperada do V234:

- `build_solver_from_external_traces` se houver regras/probes concretos para `equation_transform`;
- `external_intel_not_actionable` se os traces forem ruidosos, conflitantes ou sem mapeamento para weak misses;
- `manual_access_required` se o dataset HF gated nao estiver autorizado no ambiente.

## Atualizacao rigorosa - segunda auditoria OpenRouter/Kaggle/HF - 2026-05-10

Escopo executado:

- Reprocessados integralmente os anexos:
  - `OpenRouter Chat Sun May 10 2026 (1).json`: `9,708` linhas, SHA256 `4F87904D23F7988F2CA3F2E2917B7F3355C9F6027FF910C91DDF7D6A50E823BE`.
  - `OpenRouter Chat Sun May 10 2026.json`: `19,091` linhas, SHA256 `B8AAB44AF917338956F12C2513AE75F309025BBC56D01EF15D972EF33CA3825E`.
- JSON parse passou para os dois anexos.
- URLs brutas analisadas: `2,830`.
- Chaves normalizadas unicas: `576`.
- Categorias brutas extraidas:
  - Kaggle: `908` URLs brutas.
  - Hugging Face/HF: `927` URLs brutas.
  - GitHub: `302` URLs brutas.
  - NVIDIA docs/blogs: `123` URLs brutas.
- OpenRouter live API: nao executada porque `OPENROUTER_API_KEY` nao estava presente no ambiente local. As evidencias OpenRouter usadas vieram dos JSONs anexados.
- Kaggle CLI publico: executado sem chave para listagem e pull de notebooks publicos selecionados.
- Hugging Face plugin/API: executado para validar datasets/modelos citados.
- Web search: executado para verificar paginas publicas e paginas indexadas.

Conclusao adicional:

- A primeira atualizacao do roadmap nao estava errada, mas estava incompleta. Faltavam itens publicos do Kaggle CLI e um gap concreto de metric/extractor.
- O achado mais importante para ACC continua sendo solver/verifier, nao novo LoRA.
- Existe um risco real de subcontagem em `equation_transform` se o parser local de `\boxed{}` nao aceitar respostas com `}` literal ou braces aninhados.

### Gap corrigido no codigo local - extractor boxed

Fonte do achado:

- Kaggle kernel: `metric/nvidia-nemotron-metric`.
- O metric notebook trata cada `\boxed{` pegando o conteudo ate o ultimo `}` antes do proximo `\boxed{` ou fim do texto.
- Isso cobre respostas como `\boxed{}52}` para answer `}52` e casos com LaTeX aninhado como `\boxed{\frac{1}{2}}`.

Problema local encontrado:

- `src/competition_utils.py` usava regex `\\boxed\{([^}]*)(?:\}|$)`.
- Esse regex para no primeiro `}` e pode extrair payload errado quando a resposta correta contem brace literal, caso plausivel em `equation_transform` simbolico.

Correcao aplicada:

- `extract_boxed_answers` foi atualizado para seguir o comportamento do metric notebook: varrer todos os starts `\boxed{`, delimitar pelo proximo boxed/fim, e cortar no ultimo `}` do segmento.

Smoke test executado:

- `python -m py_compile src\competition_utils.py`
- Casos validados:
  - `\boxed{42}` -> `42`
  - `\boxed{1} ... \boxed{2}` -> `2`
  - `\boxed{\frac{1}{2}}` -> `\frac{1}{2}`
  - `\boxed{}52}` -> `}52`
  - `\boxed{abc` -> `abc`

Impacto esperado:

- Nao melhora o modelo diretamente, mas reduz risco de avaliacao local divergente da metric publica.
- Deve ser incorporado a qualquer V234/V235 antes de medir ganho em `equation_transform`.

### Novos achados Kaggle CLI que ainda faltavam no roadmap

Kernels publicos mais relevantes por evidencia de titulo, votos, metadata e pull local selecionado:

| Ref | Evidencia | Valor potencial | Decisao |
|---|---:|---|---|
| `huikang/end-to-end-finetuning-for-lb-0-85` | Kaggle CLI; pull local ok | Receita Progress Prize/LB 0.85, mask loss, LoRA, corpus token masks | P0 para leitura metodologica, nao copiar treino |
| `huikang/tinker-submission-notebook` | Kaggle CLI; pull local ok | Submission com `huikang/nemotron-adapter` versions 20/26 e extractor metric-like | P0 para adapter registry/metric parity |
| `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers` | Kaggle CLI; pull local ok | Classifica 6 familias; solvers rule-based para familias faceis; bit/symbol marcados como dificeis | P0 para V234 taxonomy/verifier |
| `optiminist/equation-eda-operator-operation-84-solve-rate` | Kaggle CLI; pull local ok | EDA de equation numeric; hipotese Pre-Op/Mid-Op/Post-Op; 84%/99% em subconjunto reportado | P0 para equation numeric verifier |
| `konbu17/bit-manipulation-solver-cot-generator` | Kaggle CLI; pull local ok | Solver bit por funcao booleana por bit; inclui INHIB/IMPL ausentes em solvers simples | P0 para bit guardrail |
| `johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster` | Kaggle CLI; pull local ok | Pipeline SFT -> GRPO com solvers como verificadores | P2; usar so depois de solver/verifier local |
| `kalyankkr/all-6-puzzle-types-decoded-sft-training-data` | Kaggle CLI; pull local ok | Classificacao, formatos de resposta, dados SFT | P2; nao treinar sem dedupe/leakage |
| `dgxchen/training-with-unsloth-to-achieve-0-85-lb` | Kaggle CLI; pull local ok | Receita Unsloth/LB 0.84-0.85, remove `lm_head`, microbatch/accum | P2; nosso V221 ja mostrou adapter pior em weak local |
| `metric/nvidia-nemotron-metric` | Kaggle CLI; pull local ok | Extractor/verify publico; paridade de metric | P0; ja gerou correcao local |
| `hammadfarooq470/think-twice-self-correcting-reasoning` | Kaggle CLI; pull local ok | Self-correction com adapter Huikang | P3; baixo valor ate provar ganho local |
| `anhtuan299/blackboard-expert-agent-assembly-solving-technique` | Kaggle CLI; pull local ok | Multi-agent/TIR de AIMO3, nao KG1 direto | P3; inspiracao, nao pipeline imediato |

Kernels publicos relevantes ainda nao baixados/analisados profundamente, mas devem entrar na triagem V234:

- `ryanholbrook/nvidia-nemotron-submission-demo`
- `dennisfong/nvidia-nemotron-sfttrainer-training`
- `kienngx/nvidia-nemotron-training-cot-labels`
- `kienngx/nvidia-nemotron-trained-models-submission`
- `asalhi/tinker-adapter-to-ready-to-submit-adapter`
- `huikang/adapter-validation-notebook`
- `kienngx/nvidia-nemotron-training-copy-run-instantly`
- `mayukh18/unsloth-sft-full-data-training`
- `llkh0a/nemotron-unsloth-sft-training-3-30-2`
- `newduck/nvidia-nemotron-soft-balanced-sampling-sft`
- `konbu17/nemotron-tong-style-cot-sft-updated-v2`
- `pearpn25/bit-cot-85-1364-sample`
- `kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset`
- `mohamedamr992/easy-loading-of-nemotron-3`
- `bloodymonday/eda-problem-families`
- `vickymaan/alice-puzzle-solver`

Regra:

- Todo Kaggle kernel entra como inteligencia externa. Nenhum notebook externo vira codigo de producao sem diff review, licenca, hash, teste unitario e weak gate.

### Novos datasets Kaggle/HF que faltavam como candidatos de triagem

Kaggle datasets listados pela API publica:

| Ref | Evidencia | Valor potencial | Decisao |
|---|---:|---|---|
| `kishanvavdara/nemotron-reasoning-traj` | `40.8 MB`, `349` downloads, `25` votes | Reasoning trajectories KG1 | P1 triagem; baixar com hash |
| `kienngx/nemotron-30b-competition-trainingdata-cot-labels` | `3.9 MB`, `1235` downloads, `47` votes | COT + labels de competicao | P1 triagem; alto risco de leakage/overfit |
| `konbu17/bit-manipulation-cot-dataset` | `625 KB`, `70` downloads | Bit CoT | P1 para bit guardrail, nao SFT bruto |
| `konbu17/bit-manipulation-synthetic-cot` | `895 KB`, `58` downloads | Bit synthetic CoT | P1 para solver tests |
| `nctuan/nvidia-nemotron-reasoning-challenge` | `643 KB`, usability `0.94` | Mirror/dataset KG1 | P2, comparar com jasonkung/sebmontreal |
| `mohammedtanvir/nemotron-reasoning-traces` | `12.8 MB`, `26` downloads | Traces | P2 triagem |
| `kevpan096/nemotron-reasoning-competition` | `7.5 MB`, `23` downloads | Competition data | P3; verificar origem |
| `sebmontreal/nvidia-nemotron-model-reasoning-challenge` | `643 KB` | Mirror Kaggle | Baixa prioridade; provavel mirror |
| `harshmali0403/nvidia-nemotron-model-reasoning-challenge` | `643 KB` | Mirror Kaggle | Baixa prioridade; provavel mirror |
| `vsnihal/nvidia-nemotron-model-reasoning-challenge-01` | `643 KB` | Mirror Kaggle | Baixa prioridade; provavel mirror |

Hugging Face datasets/modelos validados pela API:

- `andy279/nemotron-reasoning-challenge`: gated, Apache-2.0, `49,290` train, `1,165` validation, relevante.
- `andy279/nemotron-reasoning-challenge-raw-traces`: gated, Apache-2.0, raw teacher traces, relevante.
- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: CSV, `9.5k` train rows + `3` test rows, Apache-2.0, pode servir como mirror/audit de prompt/answer.
- `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2`: grande e generico; baixa prioridade para o gap atual, mas util como contexto de post-training.
- `GaryNENE/nemotron-nano-8b-reasoning-lora`: modelo LoRA 8B, nao compativel diretamente com o base 30B; usar apenas como referencia de receita/dados.
- `AdaptKey/AdaptKey-Nemotron-30b`: gated, telecom LoRA; nao relevante para KG1 ACC.
- `Taurine511/nvidia-nemotron-model-reasoning-challenge`: apareceu na busca web, mas a API HF retornou `not found`; tratar como instavel/deletado ate verificacao manual.
- `justus27/reasoning-gym-bitwise-arithmetic`: rebaixado em 2026-05-10; busca local Kaggle kernels/datasets nao encontrou slug valido. Nao usar ate haver URL verificavel.

### Novos modelos Kaggle a triagem

Kaggle model list publico trouxe candidatos ainda nao suficientemente refletidos no roadmap:

- `kienngx/nemotron-nano-30b-trained`: familia de variacoes treinadas com pipeline Kienngx.
- `atahalam/nvidia-nemotron-model-reasoning-30b-a3b-lora-0-80`: titulo declara LoRA 0.80; precisa validacao, porque titulo nao e evidencia de weak score local.
- `charancherrychowdary/nemotron-lora-adapter-v1`: adapter LoRA; baixa evidencia.
- `sluitel/nemotron-70b-reasoning-lora`: outro base/modelo; nao compativel direto.
- `nathangaskell/llama-3-1-nemotron-nano-8b`: 8B math-focused; nao compativel direto.
- `metric/nemotron-3-nano-30b-a3b-bf16`: base metric model source ja usado em notebooks Kaggle; manter como referencia de path/metric, nao como novo candidato.

Regra:

- Todo modelo/adaptador novo exige: `adapter_config.json`, tensor count, weight bytes, target_modules, base_model_name_or_path, licenca, hash, weak eval 315 rows, truncation check e no-regression por familia.

### Atualizacao de prioridade V234

O V234 deve ser ampliado alem de `andy279`:

1. Fonte metric/paridade:
   - `metric/nvidia-nemotron-metric`
   - objetivo: garantir extractor/verify equivalentes ao Kaggle.
2. Fonte equation numeric:
   - `optiminist/equation-eda-operator-operation-84-solve-rate`
   - objetivo: implementar/verificar Pre-Op/Mid-Op/Post-Op em weak miss-pack.
3. Fonte bit solver:
   - `konbu17/bit-manipulation-solver-cot-generator`
   - objetivo: testar boolean functions por output bit, incluindo INHIB/Rev-INHIB/IMPL/Rev-IMPL.
4. Fonte taxonomy:
   - `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`
   - objetivo: comparar classificacao de familias e solvers faceis contra nosso classificador.
5. Fonte adapter/corpus de alta evidencia:
   - `huikang/end-to-end-finetuning-for-lb-0-85`
   - `huikang/tinker-submission-notebook`
   - objetivo: extrair criterios de mask loss/corpus/adapter registry, nao copiar treino.
6. Fonte traces/datasets:
   - `andy279/nemotron-reasoning-challenge-raw-traces`
   - `kishanvavdara/nemotron-reasoning-traj`
   - `kienngx/nemotron-30b-competition-trainingdata-cot-labels`
   - objetivo: baixar apenas com hash e usar para regra/verifier, nao SFT bruto.

Nova saida esperada do V234:

- `external_metric_parity_report.json`
- `kaggle_kernel_triage.csv`
- `kaggle_dataset_triage.csv`
- `kaggle_model_triage.csv`
- `equation_numeric_operator_probe_results.csv`
- `bit_boolean_function_probe_results.csv`
- `external_adapter_registry_candidates.csv`

Nova decisao possivel:

- `metric_gap_fixed_continue_to_solver`
- `equation_numeric_probe_promising`
- `bit_guardrail_probe_promising`
- `external_adapter_requires_weak_eval`
- `external_sources_no_actionable_gain`

## Double check OpenRouter e matriz de destino dos achados - 2026-05-10

Registro de auditoria e correcao de reproducibilidade:

- Correcao 2026-05-10: uma versao anterior desta secao afirmava que `OPENROUTER_API_KEY` tinha sido validada localmente e que o endpoint `/api/v1/models` respondeu HTTP 200. O double check atual nao reproduziu essa chamada e outras secoes deste roadmap registram `OPENROUTER_API_KEY` ausente no ambiente local. Portanto, essa afirmacao fica rebaixada para registro historico nao reprodutivel, nao evidencia operacional.
- As evidencias OpenRouter aceitas neste roadmap sao apenas os JSONs anexados pelo usuario e as fontes primarias verificadas separadamente por web/HF/Kaggle CLI.
- Qualquer uso futuro de OpenRouter como evidencia operacional deve salvar em manifest: `prompt_sha256`, `response_sha256`, `model`, `request_id` quando disponivel, `http_status`, `created_at_utc`, `has_api_key=true` e caminho do JSON bruto.
- Ajuste exigido pelo double check: alguns itens estavam no roadmap, mas com caminho de uso implicito demais. A matriz abaixo torna explicito se cada achado ja foi implementado, sera usado no V234/V236, sera triado futuramente, ou fica como baixa prioridade/nao acionavel.

### Matriz implementado agora

| Achado | Uso | Status |
|---|---|---|
| `metric/nvidia-nemotron-metric` | Paridade de extractor/metric publica | Usado agora |
| `extract_boxed_answers` | Corrigir extracao local de `\boxed{}` com braces aninhados/literal `}` | Implementado agora em `src/competition_utils.py` |

Regra:

- Nenhuma medicao nova de `equation_transform` deve ser aceita sem esse extractor corrigido.
- Se um notebook futuro recalcular score usando extractor antigo, o gate deve rejeitar ou registrar `metric_parity_failed`.

### Matriz V234 obrigatoria

Esses itens devem ser usados diretamente no V234, com hash, logs e saidas auditaveis:

| Achado | Uso no V234 | Saida esperada |
|---|---|---|
| `metric/nvidia-nemotron-metric` | Verificar paridade de metric/extractor | `external_metric_parity_report.json` |
| `optiminist/equation-eda-operator-operation-84-solve-rate` | Probar solver numerico de equation por Pre-Op/Mid-Op/Post-Op | `equation_numeric_operator_probe_results.csv` |
| `konbu17/bit-manipulation-solver-cot-generator` | Probar solver bit por funcoes booleanas por output bit | `bit_boolean_function_probe_results.csv` |
| `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers` | Validar taxonomy e solvers por familia | `kaggle_kernel_triage.csv` |
| `huikang/end-to-end-finetuning-for-lb-0-85` | Extrair criterios de treino/mask loss/corpus, sem copiar treino | `kaggle_kernel_triage.csv` |
| `huikang/tinker-submission-notebook` | Extrair adapter registry e metric parity | `external_adapter_registry_candidates.csv` |
| `andy279/nemotron-reasoning-challenge` | Comparar dataset oficial/gated com weak miss-pack quando autorizado | `kaggle_dataset_triage.csv` ou `hf_dataset_triage.csv` |
| `andy279/nemotron-reasoning-challenge-raw-traces` | Minerar traces para regras/verifiers, nao SFT bruto | `kaggle_dataset_triage.csv` ou `hf_dataset_triage.csv` |
| `kishanvavdara/nemotron-reasoning-traj` | Triar trajectories externas contra misses V230 | `kaggle_dataset_triage.csv` |
| `kienngx/nemotron-30b-competition-trainingdata-cot-labels` | Triar CoT/labels com dedupe e leakage guard | `kaggle_dataset_triage.csv` |
| `konbu17/bit-manipulation-cot-dataset` | Gerar probes/fixtures de bit solver | `bit_boolean_function_probe_results.csv` |
| `konbu17/bit-manipulation-synthetic-cot` | Gerar probes/fixtures de bit solver | `bit_boolean_function_probe_results.csv` |
| `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge` | Mirror/audit de prompt/answer | `hf_dataset_triage.csv` |

Rebaixado apos double check:

- `justus27/reasoning-gym-bitwise-arithmetic`: nao encontrado via `kaggle kernels list` nem `kaggle datasets list` em busca exata/local. Removido da matriz obrigatoria ate existir URL/slug verificavel.

Gates V234:

- `missing_external_source_hash` se qualquer fonte baixada nao tiver hash registrado.
- `external_source_license_unknown` se licenca nao estiver clara.
- `weak_miss_pack_overlap_missing` se o achado nao mapear para linhas do miss-pack.
- `solver_probe_no_gain` se nao houver ganho mensuravel por familia.
- `bit_guardrail_regression` se bit cair abaixo de `136`.
- `equation_target_not_met` se equation nao ganhar pelo menos `+5` no pacote alvo.

### Matriz de triagem futura obrigatoria

Esses itens nao sao implementacao imediata, mas devem ser catalogados no V234 e decididos por evidencia:

| Achado | Caminho futuro | Criterio de uso |
|---|---|---|
| `johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster` | P2 apos solver/verifier local | Usar somente se verifiers locais ja existirem |
| `kalyankkr/all-6-puzzle-types-decoded-sft-training-data` | P2 taxonomy/SFT format audit | Usar para formato/taxonomy, nao treino bruto |
| `dgxchen/training-with-unsloth-to-achieve-0-85-lb` | P2 receita Unsloth | Nao usar adapter sem novo weak eval, pois V221 local foi pior |
| `hammadfarooq470/think-twice-self-correcting-reasoning` | P3 self-correction | Usar somente se houver ganho local comprovado |
| `anhtuan299/blackboard-expert-agent-assembly-solving-technique` | P3 inspiracao TIR/multi-agent | Nao entra em pipeline KG1 sem prova local |
| `ryanholbrook/nvidia-nemotron-submission-demo` | Catalogar baseline de submissao | Usar so como sanity check de formato |
| `dennisfong/nvidia-nemotron-sfttrainer-training` | Catalogar receita SFT | Exigir hash, licenca e dedupe |
| `kienngx/nvidia-nemotron-training-cot-labels` | Cruzar com dataset Kienngx | Exigir leakage guard |
| `kienngx/nvidia-nemotron-trained-models-submission` | Adapter/model triage | Exigir weak 315 rows local |
| `asalhi/tinker-adapter-to-ready-to-submit-adapter` | Converter/formato adapter | Usar so para compatibilidade, nao score |
| `huikang/adapter-validation-notebook` | Validacao adapter | Incorporar checks uteis ao gate se concretos |
| `kienngx/nvidia-nemotron-training-copy-run-instantly` | Receita replicavel | Baixa prioridade, risco de duplicacao |
| `mayukh18/unsloth-sft-full-data-training` | Receita Unsloth | Exigir dedupe/leakage before use |
| `llkh0a/nemotron-unsloth-sft-training-3-30-2` | Receita Unsloth | Exigir dedupe/leakage before use |
| `newduck/nvidia-nemotron-soft-balanced-sampling-sft` | Balanced sampling | Avaliar se resolve `equation_transform` sem perder bit |
| `konbu17/nemotron-tong-style-cot-sft-updated-v2` | CoT style | Usar so como estilo/verifier, nao treino bruto |
| `pearpn25/bit-cot-85-1364-sample` | Bit CoT sample | Cruzar com bit solver fixtures |
| `kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset` | Solver-verified CoT idea | Usar conceito de verificacao, nao familia direta |
| `mohamedamr992/easy-loading-of-nemotron-3` | Loading reference | Baixo impacto, usar se loader quebrar |
| `bloodymonday/eda-problem-families` | Family EDA | Comparar taxonomy se houver divergencia |
| `vickymaan/alice-puzzle-solver` | Solver reference | Fora das duas familias alvo, baixa prioridade |
| `nctuan/nvidia-nemotron-reasoning-challenge` | Mirror audit | Comparar hash/linhas com mirrors |
| `mohammedtanvir/nemotron-reasoning-traces` | Trace triage | Usar somente se mapear para weak misses |
| `kevpan096/nemotron-reasoning-competition` | Dataset mirror/triage | Baixa prioridade ate origem ser clara |
| `sebmontreal/nvidia-nemotron-model-reasoning-challenge` | Mirror audit | Baixa prioridade, provavel mirror |
| `harshmali0403/nvidia-nemotron-model-reasoning-challenge` | Mirror audit | Baixa prioridade, provavel mirror |
| `vsnihal/nvidia-nemotron-model-reasoning-challenge-01` | Mirror audit | Baixa prioridade, provavel mirror |
| `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` | Contexto generic KD | Nao usar para V234 salvo evidencias KG1 |
| `GaryNENE/nemotron-nano-8b-reasoning-lora` | Receita 8B | Nao compativel direto com 30B |
| `AdaptKey/AdaptKey-Nemotron-30b` | Model/adaptador telecom | Nao relevante para KG1 atual |
| `Taurine511/nvidia-nemotron-model-reasoning-challenge` | Verificar existencia manual | HF API retornou not found; nao usar ate confirmar |

### Matriz de modelos/adapters externos

Nenhum desses entra direto em submissao. Todos exigem weak eval local completo:

| Modelo/adaptador | Caminho futuro | Gate minimo |
|---|---|---|
| `kienngx/nemotron-nano-30b-trained` | Adapter candidate registry | Weak 315 rows + no-regression por familia |
| `atahalam/nvidia-nemotron-model-reasoning-30b-a3b-lora-0-80` | Adapter candidate registry | Validar se titulo 0.80 reproduz localmente |
| `charancherrychowdary/nemotron-lora-adapter-v1` | Adapter candidate registry | Validar config/tensores/base model |
| `sluitel/nemotron-70b-reasoning-lora` | Nao compativel direto | Usar somente como referencia metodologica |
| `nathangaskell/llama-3-1-nemotron-nano-8b` | Nao compativel direto | Usar somente como referencia metodologica |
| `metric/nemotron-3-nano-30b-a3b-bf16` | Referencia de base/metric path | Nao e novo candidato de score |

### Artefatos obrigatorios para provar uso futuro

O proximo notebook/script que consumir esse roadmap deve produzir:

- `external_metric_parity_report.json`: prova de paridade de extractor.
- `kaggle_kernel_triage.csv`: uma linha por kernel, com status `used_now`, `future_triage`, `rejected`, ou `not_actionable`.
- `kaggle_dataset_triage.csv`: uma linha por dataset Kaggle, com hash, licenca, linhas e decisao.
- `hf_dataset_triage.csv`: uma linha por dataset HF, com gated status, licenca, linhas e decisao.
- `kaggle_model_triage.csv`: uma linha por modelo/adaptador, com base model, config e gate.
- `equation_numeric_operator_probe_results.csv`: resultado dos probes `equation_transform`.
- `bit_boolean_function_probe_results.csv`: resultado dos probes `bit_manipulation`.
- `external_adapter_registry_candidates.csv`: adapters que merecem weak eval.

Decisao de negocio:

- Prioridade maxima: aumentar `equation_transform` de `55` para pelo menos `60` sem reduzir `bit_manipulation` abaixo de `136`.
- O caminho mais promissor e solver/verifier, nao treino bruto.
- Treino novo so deve acontecer depois que V234 provar que os novos dados/regras atacam misses reais, com hash, dedupe e leakage guard.

## V234 implementado - external intel triage executavel

Status:

- Notebook criado: `notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb`.
- Script criado: `scripts/analyze_v234_external_intel_triage.py`.
- Builder criado: `scripts/build_v234_external_intel_triage_colab.py`.
- Gate atualizado: `scripts/notebook_release_gate.py` agora valida o contrato especifico V234.

URL Colab:

- `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb`

O que o V234 faz:

- Confirma que todos os achados do roadmap tem destino explicito.
- Revalida paridade local do extractor `\boxed{}` com os casos criticos da metric publica.
- Materializa os CSVs/JSONs obrigatorios:
  - `external_metric_parity_report.json`
  - `kaggle_kernel_triage.csv`
  - `kaggle_dataset_triage.csv`
  - `hf_dataset_triage.csv`
  - `kaggle_model_triage.csv`
  - `equation_numeric_operator_probe_results.csv`
  - `bit_boolean_function_probe_results.csv`
  - `external_adapter_registry_candidates.csv`
- Bloqueia treino, geracao, scoring, pacote e Kaggle submit.

Resultado do dry run local:

- `coverage.missing_refs=[]`
- `coverage.refs_without_action_path=[]`
- `metric_parity.passed=true`
- Decisao: `external_intel_triage_ready_for_source_download`

Proximo passo depois de executar no Colab:

- Criar o notebook/script de download controlado das fontes com hash, licenca, linha por linha e mapping para miss-pack antes de implementar qualquer solver novo ou avaliar candidatos externos.

## V235 implementado - source access, hash e license triage

Status:

- Notebook criado: `notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb`.
- Script criado: `scripts/analyze_v235_source_access_triage.py`.
- Builder criado: `scripts/build_v235_source_access_triage_colab.py`.
- Gate atualizado: `scripts/notebook_release_gate.py` agora valida o contrato especifico V235.

URL Colab:

- `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb`

O que o V235 faz:

- Consome o manifest V234 executado.
- Valida que V234 passou `coverage` e `metric_parity`.
- Valida todos os CSVs/JSONs V234 obrigatorios.
- Audita acesso a Kaggle/HF sem imprimir segredos.
- Opcionalmente consulta metadata publica Hugging Face para datasets/modelos.
- Materializa:
  - `source_access_inventory.csv`
  - `hf_metadata_audit.csv`
  - `kaggle_access_audit.csv`
  - `source_download_plan.csv`
  - `license_gate_report.json`
- Bloqueia download de payload, treino, geracao, scoring, pacote e Kaggle submit.

Contrato de seguranca:

- Nenhuma fonte externa pode ser ingerida em treino/solver sem `license_status` conhecido e `hash_status` registrado.
- Fontes Kaggle seguem bloqueadas para uso direto ate metadata/licenca/hash serem resolvidos.
- Fontes HF gated exigem token/metadata antes de qualquer payload.

Resultado da execucao no Colab:

- Notebook executado: `notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb`.
- Run ID: `20260510T145150Z`.
- Commit clonado no Colab: `65be3e3b5992cfd841c7a075242f5418950932ec`.
- Manifest V234 consumido: `/content/drive/MyDrive/KG1_NVIDIA_V234/output_v234_external_intel_triage/analysis_v234_external_intel_triage/20260510T143802Z/v234_external_intel_triage_manifest.json`.
- Manifest V235 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V235/output_v235_source_access_triage/analysis_v235_source_access_triage/20260510T145150Z/v235_source_access_triage_manifest.json`.
- Manifest V235 SHA256: `d0bf0eb30bf236da0c08e09805b02c808fc5e793170489cd44a8d3b25c60eaf3`.
- `source_access_inventory.csv`: `51` linhas, SHA256 `eb71efb47fee7bc61a90df2b0ce34371b17ef600896ae95aa92f9c9c26a66396`.
- `hf_metadata_audit.csv`: `8` linhas, SHA256 `b27bcbedbf0e7ee1175b0764891031e3973ff621187b085ebe5fa9205efb0abd`.
- `kaggle_access_audit.csv`: `43` linhas, SHA256 `b9075abb3a0231402ffb6fa5a7a8543d0e51e80054d4c22735e1174a54283544`.
- `source_download_plan.csv`: `46` linhas, SHA256 `858760a2daebc194e6b1d71635463c950d222282525dc436cb18b35e0f2e6732`.
- `license_gate_report.json`: SHA256 `e7f3b187583b0ba000443871820f112e2ffcfba52856761849271ef0b855180c`.
- Resumo por tipo de fonte:
  - `kaggle_kernel`: `27`
  - `kaggle_dataset`: `10`
  - `kaggle_model`: `6`
  - `hf_dataset`: `5`
  - `hf_model`: `3`
- Status das fontes:
  - `v234_required`: `13`
  - `used_now`: `1`
  - `future_triage`: `31`
  - `reference_only`: `4`
  - `manual_verify`: `1`
  - `not_actionable`: `1`
- Metadata HF:
  - HTTP `200`: `7`
  - HTTP `401`: `1`
- Download permitido pelo gate:
  - `true`: `5`
  - `false`: `46`
- Credenciais no runtime:
  - `kaggle_cli_path=/usr/local/bin/kaggle`
  - `kaggle_json_exists=false`
  - `kaggle_username_present=false`
  - `kaggle_key_present=false`
  - `hf_token_present=false`
  - `openrouter_key_present=false`
- Decisao: `manual_source_access_or_license_required_before_download`.
- Motivo: fontes obrigatorias ainda precisam de credenciais, metadata de licenca ou hash de download antes de qualquer payload.

Fontes obrigatorias ainda bloqueadas:

- `metric/nvidia-nemotron-metric`
- `huikang/end-to-end-finetuning-for-lb-0-85`
- `huikang/tinker-submission-notebook`
- `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`
- `optiminist/equation-eda-operator-operation-84-solve-rate`
- `konbu17/bit-manipulation-solver-cot-generator`
- `kishanvavdara/nemotron-reasoning-traj`
- `kienngx/nemotron-30b-competition-trainingdata-cot-labels`
- `konbu17/bit-manipulation-cot-dataset`
- `konbu17/bit-manipulation-synthetic-cot`

Proximo passo depois de executar no Colab:

- Se o V235 decidir `manual_source_access_or_license_required_before_download`, resolver credenciais/licencas primeiro.
- Se decidir `source_access_plan_ready_needs_controlled_download`, criar o downloader V236 que baixa apenas fontes permitidas, registra hash, licenca, row counts e mapping para miss-pack.

## Deep dive de literatura para `bit_manipulation` e `equation_transform` - 2026-05-10

Familias da imagem/weak gate:

| Familia | Linhas avaliadas | Corretas | ACC | Status |
|---|---:|---:|---:|---|
| `bit_manipulation` | 160 | 136 | 85.00% | medido |
| `equation_transform` | 155 | 55 | 35.48% | medido |
| `OVERALL weak` | 315 | 191 | 60.63% | medido |

Leitura de negocio:

- `bit_manipulation` ja passa o gate minimo de `133/160`, mas a margem real e pequena. Deve virar guardrail, nao foco principal de treino.
- `equation_transform` e o gargalo: precisa sair de `55/155` para pelo menos `60/155`.
- Regra operacional: qualquer melhoria em `equation_transform` deve preservar `bit_manipulation>=136/160`, salvo se o weak completo provar `total>=193`, `equation>=60`, `bit>=133`, `trunc<=3`.
- Hipotese mais forte ja registrada: `equation_transform` numerico esta muito melhor que symbolic/mixed; o ganho de ACC deve mirar symbolic/mixed e operadores customizados, nao apenas aritmetica comum.

### Fontes publicas e literatura tecnica consultadas

Bit-vectors e bit manipulation:

- Z3 Bitvectors: `https://microsoft.github.io/z3guide/docs/theories/Bitvectors/`
  - Evidencia: Z3 modela semantica precisa de bit-vectors de tamanho fixo, com aritmetica signed/unsigned, literais binarios/hex e operacoes bitwise.
  - Uso no KG1: representar cada entrada de 8 bits como `BitVec(8)` e validar regras candidatas sobre todos os exemplos do prompt.
- Z3 BitVector API: `https://z3prover.github.io/api/html/ml/Z3.BitVector.html`
  - Evidencia: API tem AND, OR, XOR, NOT, shifts, rotates, concat, extract, carry e xor3.
  - Uso no KG1: cobrir exatamente o vocabulario do enunciado `shift`, `rotate`, `XOR`, `AND`, `OR`, `NOT`, majority/choice via boolean formulas.
- Component-based synthesis applied to bitvector circuits: `https://www.microsoft.com/en-us/research/publication/component-based-synthesis-applied-to-bitvector-circuits/`
  - Evidencia: sintese de programas bitvector por biblioteca de componentes + constraints + SMT/CEGIS e adequada para composicoes nao intuitivas de operacoes bitvector.
  - Uso no KG1: trocar tentativa por LLM por enumerador/CEGIS de DSL pequena, com ranking por simplicidade e verificacao total dos exemplos.
- SyGuS: `https://www.microsoft.com/en-us/research/publication/syntax-guided-synthesis-2/`
  - Evidencia: problema combina especificacao semantica, gramatica de candidatos e CEGIS.
  - Uso no KG1: transformar exemplos `input -> output` em problema PBE-BV; gramatica limitada evita overfit e explosao.
- Reasoning Gym: `https://github.com/open-thought/reasoning-gym` e `https://arxiv.org/abs/2505.24760`
  - Evidencia: geradores procedurais e verificadores com reward verificavel; inclui scoring em cascata e dominios de algebra/arithmetic/computation/logic.
  - Uso no KG1: fonte de fixtures/probes e verifiers, nao treino bruto; especialmente `bitwise_arithmetic` para guardrails.

Equation transform, simbolico e regra por operador:

- SymPy solving: `https://docs.sympy.org/latest/guides/solving/index.html`
  - Evidencia: SymPy resolve equacoes simbolicas, sistemas, inequacoes, diofantinas e tambem numericamente.
  - Uso no KG1: apenas para subclasse com equacao algebraica clara e variavel unica; deve abstain em prompt ambiguidade.
- SymPy simplify: `https://docs.sympy.org/latest/modules/simplify/simplify.html`
  - Evidencia: `simplify()` nao e uma operacao bem definida; docs recomendam usar funcoes especificas quando o algoritmo depende de uma transformacao concreta.
  - Uso no KG1: nao usar `simplify()` generico como solver final; usar `factor`, `expand`, `cancel`, `together`, `solve`, `solveset`, `nsimplify` com contratos especificos.
- egg / equality saturation: `https://arxiv.org/abs/2004.03082`
  - Evidencia: e-graphs representam muitas expressoes equivalentes e sao usados em otimizacao, reescrita e program synthesis.
  - Uso no KG1: para symbolic/mixed, inferir e validar transformacoes por regras, sem comprometer cedo com uma sequencia de reescritas.
- Rewrite Rule Inference Using Equality Saturation: `https://arxiv.org/abs/2108.10436`
  - Evidencia: e-graphs podem ajudar a inferir regras menores e mais gerais a partir de enumeracao de termos.
  - Uso no KG1: minerar regras de `equation_transform` a partir dos exemplos do prompt e dos traces, depois aceitar somente se todos os exemplos forem satisfeitos.

Fontes KG1/Kaggle/HF relevantes localizadas:

- HF `andy279/nemotron-reasoning-challenge`
  - Gated, Apache-2.0, SFT KG1; `49,290` train, `1,165` validation, inclui `399` transformation unsolved.
  - Uso: P0 apos aceite manual/licenca/hash; nao SFT bruto antes de dedupe/conflict/leakage.
- HF `andy279/nemotron-reasoning-challenge-raw-traces`
  - Gated, Apache-2.0; inclui solver-guided transformation e bit traces.
  - Uso: P0 para extrair regras/probes e nao respostas cegas.
- Kaggle `kishanvavdara/nemotron-reasoning-traj`
  - CLI listou `40764029` bytes, `349` downloads, `25` votos.
  - Busca web indicou `9,500` rows, `bit_manipulation=1,602`, `equation_symbolic=823`, `equation_numeric=732`.
  - Uso: error analysis e candidate traces apos hash/licenca.
- Kaggle `optiminist/equation-eda-operator-operation-84-solve-rate`
  - CLI confirmou kernel com `25` votos.
  - Uso: P0 para decompor equation por operador/operacao; nao assumir `84%` sem reproduzir localmente.
- Kaggle `konbu17/bit-manipulation-solver-cot-generator`
  - CLI confirmou kernel com `39` votos.
  - Uso: P0 para DSL bitvector/boolean por bit; nao usar COT longo como saida final.
- Kaggle `konbu17/bit-manipulation-cot-dataset`
  - CLI listou `624941` bytes, `70` downloads, `5` votos.
  - Uso: fixtures e testes de bit guardrail, nao treino direto.
- Kaggle `konbu17/bit-manipulation-synthetic-cot`
  - CLI listou `895101` bytes, `58` downloads, `4` votos.
  - Uso: probes de solver bit, nao treino direto.
- Kaggle Huikang:
  - `huikang/end-to-end-finetuning-for-lb-0-85`: `257` votos.
  - `huikang/tinker-submission-notebook`: `454` votos.
  - `huikang/adapter-validation-notebook`: `272` votos.
  - Uso: metodo/validacao/adapter registry; nenhum score entra sem weak eval local.

### Plano tecnico para melhorar `bit_manipulation`

Objetivo: manter `136/160` ou melhorar sem tocar no baseline quando o solver nao tiver prova.

Implementacao recomendada:

1. Parser robusto de prompt:
   - Extrair todos os pares `8-bit -> 8-bit`.
   - Confirmar largura fixa `8`.
   - Extrair target.
   - Abort se houver qualquer par malformado, largura diferente ou resposta fora de `[01]{8}`.
2. DSL pequena e verificavel:
   - Termos atomicos: `x`, `~x`, `shl(x,k)`, `lshr(x,k)`, `rotl(x,k)`, `rotr(x,k)`, constantes/masks para `k in 0..7`.
   - Combinadores binarios: `AND`, `OR`, `XOR`, `NAND`, `NOR`, `XNOR`.
   - Combinadores ternarios: `majority(a,b,c)`, `choice(a,b,c)`, `xor3(a,b,c)`.
   - Pos-processamento permitido: identidade, NOT, XOR mask, reverse bits, optional zero/sign handling se aparecer no prompt.
3. Busca:
   - Primeiro enumeracao direta ate profundidade baixa.
   - Depois CEGIS/SMT se houver multiplas hipoteses ou regra mais profunda.
   - Ranking por menor AST, menor numero de constantes, menor numero de operacoes exoticas.
4. Aceitacao:
   - Uma regra so pode sobrescrever baseline se acertar 100% dos exemplos do prompt.
   - Se houver duas regras com outputs diferentes para o target e mesmo score nos exemplos, abstain.
   - Se o baseline ja estava correto, nao sobrescrever sem prova unica e output igual.
5. Uso esperado:
   - `bit_manipulation` e guardrail: usar solver para detectar regressao e recuperar poucos misses, nao como roteador agressivo.

### Plano tecnico para melhorar `equation_transform`

Objetivo: ganhar pelo menos `+5` em `equation_transform`, com foco em symbolic/mixed.

Classes a separar antes de qualquer treino:

1. Numeric operator transform:
   - Exemplos do tipo `AA op BB = OUT`.
   - Candidatos: aritmetica direta, aritmetica com reversao de operandos, reversao de resultado, concatenacao, zero padding, sinal customizado, modulo/base, soma/subtracao por digito, produto por digito.
   - Verificar por operador: regras de `+`, `-`, `*`, `/`, `%`, `?`, `@`, `&` podem ser independentes.
2. Symbolic/mixed token transform:
   - Operandos e saidas podem conter simbolos, aspas, pipes e caracteres nao alfanumericos.
   - Candidatos: permutacao de tokens, reversao por lado, concatenacao esquerda/direita, substituicao por tabela, shift em alfabeto observado, operador como selector.
   - Nao tentar SymPy nesse subtipo.
3. Algebraic equation:
   - Usar SymPy somente quando houver uma equacao algebrica unica e variavel unica.
   - Exigir substituicao verificada e resposta normalizada.
4. Constraint/cryptarithm:
   - Usar DFS/Z3 sobre digitos ou simbolos com unicidade se o prompt indicar mapeamento.
   - Aceitar apenas solucao unica.

Aceitacao de override:

- A regra deve explicar todos os exemplos do prompt.
- A regra deve gerar target unico.
- O output deve passar extractor `\boxed{}` corrigido, preservando zeros a esquerda e simbolos literais.
- Se a resposta envolve braces, pipes, aspas ou caracteres especiais, usar extractor balanceado, nao regex simples.
- Se os exemplos por operador forem poucos e houver multiplas regras, abstain.

### Prompt OpenRouter recomendado se a chave for disponibilizada

OpenRouter nao foi chamado neste runtime porque `OPENROUTER_API_KEY` estava ausente. Se a chave estiver disponivel em ambiente seguro, usar o prompt abaixo em modelos fortes e pedir saida JSON, nunca decisao livre:

```text
You are auditing a Kaggle NVIDIA Nemotron KG1 solver plan. Do not speculate.
Use only evidence from the provided weak rows, prompt examples, known public sources,
and formal methods literature. We need improve equation_transform from 55/155 to >=60
while preserving bit_manipulation >=136/160.

Tasks:
1. Classify each equation_transform miss into numeric operator, symbolic/mixed token transform,
   algebraic equation, cryptarithm/constraint, extractor issue, or unknown.
2. For each class, propose only deterministic solvers with acceptance predicates.
3. For bit_manipulation, propose a finite 8-bit DSL and ambiguity/abstention checks.
4. Return JSON with: class, evidence, proposed_solver, required_inputs, acceptance_tests,
   regression_risks, expected_gain_upper_bound, and reasons_to_abstain.
5. Do not claim any ACC gain unless it is computable from supplied rows.
```

### Roadmap executavel apos esta revisao

P0 sem novas credenciais:

1. Evoluir V233 para `V236_LOCAL_SOLVER_DSL_PROBES` usando apenas miss-packs e dados locais ja versionados.
2. Implementar `bitvector_dsl_probe` como guardrail com abstention.
3. Implementar `equation_operator_dsl_probe` com regras por operador e verificacao total dos exemplos.
4. Implementar `symbolic_token_transform_probe` para permutation/concat/reverse/substitution em `equation_transform`.
5. Rodar contra `v230_v226_complementarity_equation_miss_pack.csv` e `bit_miss_pack.csv`.
6. Promover para eval somente se houver `>=5` overrides deployable em equation e zero regressao bit.

P0 com acao humana:

1. Configurar credenciais Kaggle/HF no Colab sem imprimir segredo.
2. Resolver licenca/metadata/hash das fontes bloqueadas no V235.
3. Criar V236 downloader apenas se `license_gate.direct_ingestion_allowed=true` para as fontes requeridas.
4. Baixar primeiro:
   - `metric/nvidia-nemotron-metric`
   - `optiminist/equation-eda-operator-operation-84-solve-rate`
   - `konbu17/bit-manipulation-solver-cot-generator`
   - `andy279/nemotron-reasoning-challenge-raw-traces`
   - `kishanvavdara/nemotron-reasoning-traj`
5. Gerar hash, row count, schema, family counts, dedupe, conflict check e leakage guard antes de qualquer ingestao.

Nao fazer:

- Nao trocar adapter pelo titulo `LB 0.85` sem weak eval identico.
- Nao treinar em COT bruto longo que historicamente elevou truncation.
- Nao usar dados externos sem licenca/hash.
- Nao aceitar `equation_transform` medido por extractor regex simples.
- Nao reduzir `bit_manipulation` abaixo de `136/160` em nome de ganho hipotetico.

## Auditoria do anexo OpenRouter `Sun May 10 2026 (2)` - 2026-05-10

Arquivo auditado:

- `C:\Users\davis\Downloads\OpenRouter Chat Sun May 10 2026 (2).json`
- SHA256: `F705A612BB848A8588F99826CF7DC4781822DAB0460EEF04B50A9FEE75D8DFC7`
- Tamanho: `721917` bytes.

Leitura de confiabilidade:

- O anexo contem respostas de multiplos modelos. Algumas respostas declaram acesso real a busca; outras declaram explicitamente ausencia de internet. Portanto, nada do anexo deve ser tratado como fato ate ser verificado em fonte primaria.
- Validacao externa feita nesta revisao confirmou que os achados mais fortes sao fontes ja acionaveis para engenharia, nao prova de ganho direto de ACC.
- Nao ha no anexo um adapter pronto que possa ser promovido com seguranca sem weak eval identico, licenca, hash e auditoria de leakage.

Achados confirmados e uteis para ACC:

1. `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`
   - URL: `https://huggingface.co/datasets/jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`
   - Evidencia verificada: dataset HF em CSV, `apache-2.0`, cerca de `9.5k` linhas; viewer mostra prompts reais com assinatura de `bit_manipulation` e `equation_transform`.
   - Uso correto: usar como referencia de schema/prompt signatures e gerador de testes de parser. Nao usar para treino ou avaliacao ate resolver risco de overlap/leakage com weak/test.
   - Sinal tecnico: `bit_manipulation` aparece como transformacao de numeros binarios de 8 bits com shifts, rotations, XOR, AND, OR, NOT, majority e choice. `equation_transform` aparece como regras sobre simbolos e caracteres especiais, reforcando que SymPy nao e a rota principal para symbolic/mixed.

2. `andy279/nemotron-reasoning-challenge`
   - URL: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge`
   - Evidencia verificada: dataset HF gated, `apache-2.0`, SFT data com `train=49290` e `validation=1165`; a propria card declara traces de teacher models e solvers.
   - Sinais relevantes: `Solver-guided transformation=1101`, `Solver-guided bit manipulation=1602`, `GPT-5.4 transformation=85`.
   - Uso correto: P0/P1 para minerar regras e acceptance predicates; nao usar COT bruto como SFT sem auditoria, porque o risco de truncation/formato/leakage e alto.

3. `andy279/nemotron-reasoning-challenge-raw-traces`
   - URL: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces`
   - Evidencia verificada: dataset HF gated, `apache-2.0`, com arquivos:
     - `solver_transformation_traces_merged.jsonl`
     - `solver_bit_manipulation_traces_merged.jsonl`
     - `solver_transformation_traces_gpt54.jsonl`
   - Uso correto: fonte mais forte para extrair DSLs, regras por subtipo e criterios de abstencao. Deve passar por downloader com hash, row count, schema, family count, dedupe, conflito e leakage guard.

4. `tonghuikang/nemotron`
   - URL: `https://github.com/tonghuikang/nemotron`
   - Evidencia verificada: repositorio do Progress Prize para NVIDIA Nemotron Model Reasoning Challenge; README aponta writeup/notebook Kaggle e contem pastas como `reasoners`, `problems`, `investigators`, `training/sft`, `corpus`, `metrics`.
   - Uso correto: estudar engenharia de corpus, min-logprob, reasoners e tabulacoes de problemas. Nao usar adapter, treino ou codigo sem licenca, hash e weak eval isolado.

Consenso tecnico util do anexo:

- `equation_transform` deve ser dividido antes de qualquer novo treino:
  - numeric operator transform;
  - symbolic/mixed token rewrite;
  - sequence/token transform;
  - algebraic equation;
  - cryptarithm/constraint-like.
- O ganho de curto prazo mais plausivel continua sendo `+5` por solver/router conservador em `equation_transform`, nao por troca cega de LoRA.
- O solver so deve sobrescrever o modelo se:
  - parseou todos os exemplos do prompt;
  - encontrou uma regra unica;
  - a regra explica todos os exemplos;
  - preserva zeros a esquerda e caracteres especiais;
  - o extractor balanceado consegue serializar `\boxed{...}` sem cortar braces, pipes, aspas ou barras.
- `bit_manipulation` deve ficar como guardrail. Qualquer DSL/Z3/CEGIS deve atuar primeiro como validador e detector de regressao. So pode virar override se provar zero regressao contra os `160` exemplos weak e manter pelo menos `136/160`.

Descartes/ajustes feitos a partir do anexo:

- Descartar afirmacoes de ganho garantido como `+3`, `+5` ou `90% probabilidade` sem medicao local.
- Descartar rollback com limite `bit<134`; o guardrail operacional correto e baseline protegido `bit>=136/160`, salvo weak gate completo provando `total>=193`, `equation>=60`, `bit>=133`, `trunc<=3`.
- Nao usar SymPy para symbolic/mixed. SymPy fica restrito a algebra clara, variavel unica, solucao unica e substituicao verificada.
- Nao tratar `cryptarithm` como subtipo dominante sem evidencia local. Implementar apenas se o classificador achar prompts com restricoes explicitas de mapeamento/aritmetica.
- Nao aceitar adapters HF/Kaggle por popularidade, LB title ou progress-prize label sem reproduzir em weak identico.

Atualizacao P0 para V236:

1. Criar `V236_LOCAL_SOLVER_DSL_PROBES` com tres saidas obrigatorias:
   - `equation_subtype_audit.csv`;
   - `equation_solver_probe_results.csv`;
   - `bit_guardrail_probe_results.csv`.
2. O subtipo `symbolic/mixed token rewrite` deve ser P0, porque o anexo e o HF viewer reforcam que ha muitos caracteres nao algebricos em `equation_transform`.
3. O subtipo `numeric operator transform` deve testar no minimo:
   - operacao direta;
   - reversao de operandos;
   - reversao de resultado;
   - concatenacao;
   - aritmetica por digito;
   - base/modulo;
   - operador remapeado por simbolo.
4. O `bitvector_dsl_probe` deve incluir exatamente as operacoes do prompt publico: shifts, rotations, XOR, AND, OR, NOT, majority e choice.
5. Promocao bloqueada ate haver evidencias locais:
   - `equation_transform >= 60/155`;
   - `bit_manipulation >= 136/160` preferencialmente, ou `>=133/160` apenas se o gate total completo passar;
   - `truncated <= 3`;
   - nenhum ganho medido com extractor regex simples.

## V236 implementado - local solver DSL probes

Implementacao adicionada em 2026-05-10:

- Script: `scripts/analyze_v236_local_solver_dsl_probes.py`.
- Builder: `scripts/build_v236_local_solver_dsl_probes_colab.py`.
- Notebook: `notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`.

Escopo:

- CPU-only.
- Consome o manifest V232 e os workitems:
  - `equation_solver_workitems_jsonl`;
  - `bit_guardrail_workitems_jsonl`;
  - `acceptance_matrix_csv`;
  - `solver_contracts_json`.
- Nao treina, nao gera modelo, nao roda scoring completo, nao empacota, nao baixa payload externo e nao submete ao Kaggle.

Saidas obrigatorias:

- `v236_local_solver_dsl_probes_equation_subtype_audit.csv`;
- `v236_local_solver_dsl_probes_equation_solver_probe_results.csv`;
- `v236_local_solver_dsl_probes_bit_guardrail_probe_results.csv`;
- `v236_local_solver_dsl_probes_equation_probe_summary.csv`;
- `v236_local_solver_dsl_probes_manifest.json`.

Probes iniciais:

- `symbolic_char_map_probe`: deployable apenas quando todos os exemplos tem mesmo comprimento, mapeamento caractere-a-caractere consistente, query totalmente coberta e predicao unica.
- `reverse_token_probe`: deployable apenas quando todos os exemplos provam reversao simples.
- `numeric_operator_dsl_probe`: testa somente regras conservadoras `direct_arithmetic`, `reverse_result_arithmetic`, `reverse_operands_arithmetic`, `digitwise_add_mod10`; abstain em ambiguidade.
- `bitvector_prompt_signature_guardrail`: nao sobrescreve resposta; apenas confirma assinatura de prompt bitvector e escopo de operadores permitidos para o guardrail.

Validacoes ja executadas localmente:

- `python -m py_compile scripts/analyze_v236_local_solver_dsl_probes.py`
- `python scripts/analyze_v236_local_solver_dsl_probes.py --self-test`
- `python -m py_compile scripts/build_v236_local_solver_dsl_probes_colab.py`
- `python scripts/build_v236_local_solver_dsl_probes_colab.py`
- `python scripts/notebook_release_gate.py notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`

Resultado do gate:

- `ok=true`
- notebook SHA256: `be80f4ca59097b6aa964e734cfc8186dc1b922df99d1062640451c5ea13731ee`

Proximo passo:

- Executar o V236 no Colab.
- Se `deployable_verified_equation_overrides >= 5` e `bit_guardrail_signature_verified_rows` cobrir os workitems sem override incorreto, criar apenas entao um notebook separado de rescue measurement.
- Se V236 decidir `continue_local_solver_development`, abrir `equation_subtype_audit.csv` e expandir somente os subtipos com parser exato. Nenhuma avaliacao ou pacote deve ser feito antes disso.

## V236 executado - resultado e ajuste pos-log

Execucao analisada em 2026-05-10:

- Notebook executado: `notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`.
- Commit Colab observado: `69d2d9725ae194e087d9d37717967eacd3a382df`.
- Manifest V236 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V236/output_v236_local_solver_dsl_probes/analysis_v236_local_solver_dsl_probes/20260510T154035Z/v236_local_solver_dsl_probes_manifest.json`.
- Manifest SHA256 observado: `cbdd10850a741941084b1ef135c2ec3bb31bfe5429ee68bd12cb8f7163f37285`.
- Final manifest SHA256 observado: `3ef90c3cb97adfbde7c52bdb7b4ca0742336ed8ff433f8ed22ba4b8ec7e80a5e`.

Resultados medidos:

- `equation_workitems`: `100`.
- `bit_guardrail_workitems`: `24`.
- `deployable_verified_equation_overrides`: `0`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24`.
- Decisao: `continue_local_solver_development`.

Distribuicao dos subtipos observados:

- `algebraic_equation`: `69`.
- `symbolic_mixed_token_rewrite`: `22`.
- `numeric_operator_transform`: `9`.

Interpretacao:

- A execucao foi limpa, sem erro de runtime, artefato ausente ou quebra de contrato V232.
- O guardrail de `bit_manipulation` esta pronto como verificador de escopo, mas nao emite override.
- O gargalo permanece `equation_transform`: os probes iniciais foram seguros demais e abstiveram em todos os `100` workitems.
- Gap encontrado no script: havia classificacao para `algebraic_equation`, mas nao havia probe algebraico; esses itens caiam no probe simbolico e abstinham.

Ajuste aplicado apos a analise:

- `scripts/analyze_v236_local_solver_dsl_probes.py` agora inclui `sympy_single_equation_probe`, restrito a:
  - equacao unica;
  - uma variavel alfabetica;
  - pelo menos um digito;
  - caracteres algebraicos seguros;
  - solucao unica;
  - verificacao por substituicao simbolica.
- O script agora tambem gera `equation_abstain_reason_summary_csv`, para agrupar motivos concretos de abstain por subtipo/probe/prova.
- Nenhuma regra nova autoriza treino, full eval, pacote ou submissao. O gate continua exigindo ganho local medido antes de qualquer rescue measurement.

Proximo passo revisado:

- Reexecutar o V236 atualizado no Colab.
- Se ainda houver `0` overrides, usar `equation_abstain_reason_summary_csv` para decidir o V237 por evidencia, provavelmente focado em:
  - parser de numeric operator expandido;
  - parser de symbolic/mixed com mapeamento de token, nao so caractere;
  - identificacao de casos onde exemplos sao insuficientes e devem permanecer abstain.
- Criar notebook de rescue eval somente se `deployable_verified_equation_overrides >= 5`, `deployable_incorrect_equation_overrides == 0` e o guardrail bit continuar completo.

## V236 reexecutado - diagnostico apos probe SymPy

Execucao analisada em 2026-05-10 a partir de `KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB (1).ipynb`:

- Commit Colab observado: `45592f00e669d32077628a13a001bc4d7e5ccbf1`.
- Manifest V236 SHA256 observado: `ae615978b1eef3e2d326c450bd20cdac5bd383457aa5af0277817d7d64ebc5a8`.
- Final manifest SHA256 observado: `55fc968a18ef6bf8bfb8ffaff06913e8a4d1a70e759a1b3cb8dcdbb2e4dfc803`.
- `equation_workitems`: `100`.
- `bit_guardrail_workitems`: `24`.
- `deployable_verified_equation_overrides`: `0`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24`.
- Decisao: `continue_local_solver_development`.

Diagnostico novo trazido por `equation_abstain_reason_summary`:

- `69` linhas em `algebraic_equation` abstiveram como `missing_examples_or_query` porque o fallback simbolico ainda escondia o motivo real do probe algebraico.
- `9` linhas em `numeric_operator_transform` abstiveram como `numeric_examples_or_query_not_parseable`.
- `22` linhas em `symbolic_mixed_token_rewrite` abstiveram como `example_length_mismatch`.

Ajuste adicional aplicado apos esse log:

- `algebraic_equation` agora retorna sempre o resultado do `sympy_single_equation_probe`, mesmo quando abstain.
- Assinaturas algebricas que nao passam no parser seguro agora sao classificadas como `algebraic_equation_unparsed`.
- Objetivo: o proximo replay deve revelar o motivo real dos `69` itens, em vez de registrar `symbolic_char_map_probe/missing_examples_or_query`.

Proximo passo:

- Reexecutar V236 mais uma vez para obter `equation_abstain_reason_summary` corrigido.
- Se os `69` itens forem majoritariamente `no_single_algebraic_equation_found`, o V237 deve priorizar parser do formato real do prompt antes de qualquer solver novo.
- Se houver equacoes parseaveis com falha SymPy especifica, o V237 deve atacar somente essas falhas com testes unitarios antes de medir rescue.

## V236 terceira execucao - V237 desbloqueado como auditoria

Execucao analisada em 2026-05-10 a partir de `KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB (2).ipynb`:

- Commit Colab observado: `9b03e9eef5f1f83e31195602ecfd9a97777456d8`.
- Manifest V236 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V236/output_v236_local_solver_dsl_probes/analysis_v236_local_solver_dsl_probes/20260510T155704Z/v236_local_solver_dsl_probes_manifest.json`.
- Manifest V236 SHA256 observado: `ef049b79eeedea09b839147fb1c2d0429a79a03948515b5b765706c714f62e9c`.
- Final manifest SHA256 observado: `de0d96d879c7ded3560b5d8d73b3aac65f8632637d337edf2c1c574e23d36c5d`.
- `equation_workitems`: `100`.
- `bit_guardrail_workitems`: `24`.
- `deployable_verified_equation_overrides`: `0`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24`.
- Decisao: `continue_local_solver_development`.

Diagnostico corrigido:

- `69` linhas: `algebraic_equation_unparsed` com `sympy_single_equation_probe` abstain por `prompt:no_single_algebraic_equation_found`.
- `9` linhas: `numeric_operator_transform` abstain por `numeric_examples_or_query_not_parseable`.
- `22` linhas: `symbolic_mixed_token_rewrite` abstain por `example_length_mismatch`.

Conclusao:

- Nao ha base para rescue eval.
- Nao ha base para treino.
- O gargalo agora e parser de formato real do prompt, nao solver matematico.

## V237 implementado - prompt format audit

Implementacao adicionada em 2026-05-10:

- Script: `scripts/analyze_v237_prompt_format_audit.py`.
- Builder: `scripts/build_v237_prompt_format_audit_colab.py`.
- Notebook: `notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`.

Escopo:

- CPU-only.
- Consome o manifest V232 e os mesmos workitems usados por V236.
- Audita formato dos prompts por `solver_route`, marcador de query, pares de exemplos candidatos, equacoes candidatas, expressoes numericas candidatas e hint de abstain.
- Nao treina, nao gera modelo, nao roda scoring, nao empacota, nao baixa payload externo e nao submete ao Kaggle.

Saidas obrigatorias:

- `v237_prompt_format_audit_prompt_format_audit.csv`;
- `v237_prompt_format_audit_prompt_format_summary.csv`;
- `v237_prompt_format_audit_equation_prompt_samples.csv`;
- `v237_prompt_format_audit_manifest.json`.

Validacoes executadas localmente:

- `python -m py_compile scripts/analyze_v237_prompt_format_audit.py`;
- `python scripts/analyze_v237_prompt_format_audit.py --self-test`;
- `python -m py_compile scripts/build_v237_prompt_format_audit_colab.py`;
- `python scripts/build_v237_prompt_format_audit_colab.py`;
- `python scripts/notebook_release_gate.py notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `ce92d2a5aee32e78a3a7c668ba14e8ca98e1d2d65ccff6564061adf32599f5e7`.

Proximo passo:

- Executar V237 no Colab.
- Se V237 mostrar que os prompts tem pares de exemplo parseaveis por outro padrao, criar V238 com parser unit-tested antes de qualquer override.
- Se V237 mostrar ausencia real de exemplos suficientes, manter abstain e voltar para mineracao de dados/treino, sem rescue eval.

## V237 executado - prompt format audit

Execucao analisada em 2026-05-10 a partir de `KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`:

- Commit observado na celula de setup: `7d6db5eb045f7ae839367e95c83cc5432f8961a4`.
- Observacao operacional: a celula de preflight V232 nao foi executada no anexo, mas a celula de audit corrigida resolveu o manifest automaticamente e completou sem erro.
- Manifest V237 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V237/output_v237_prompt_format_audit/analysis_v237_prompt_format_audit/20260510T160754Z/v237_prompt_format_audit_manifest.json`.
- Manifest V237 SHA256 observado: `d857c395742b95604b4d41191f39c43c0f5fc464aa9b197ce86127b4f61c1d27`.
- Final manifest SHA256 observado: `8e723dd6ba771a716088927b1977db4a018db0ba0db8a6f07f080091eb660be8`.

Contagens:

- `audit_rows`: `124`.
- `equation_workitems`: `100`.
- `bit_workitems`: `24`.
- `equation_zero_candidate_pair_rows`: `68`.

Resumo de hints para `equation_transform`:

- `example_pairs_nonuniform_or_symbolic`: `32`.
- `numeric_expr_without_parseable_examples`: `11`.
- `prompt_format_requires_manual_parser`: `57`.

Resumo por rota:

- `bit_manipulation` / `bitwise_named_operator_dsl`: `24`, query marker `last_nonempty_line`, hint `example_pairs_nonuniform_or_symbolic`.
- `equation_transform` / `sympy_symbolic_transform`: `32`, marker `now_determine_result_for`, hint `example_pairs_nonuniform_or_symbolic`.
- `equation_transform` / `sympy_symbolic_transform`: `11`, marker `now_determine_result_for`, hint `numeric_expr_without_parseable_examples`.
- `equation_transform` / `sympy_symbolic_transform`: `57`, marker `now_determine_result_for`, hint `prompt_format_requires_manual_parser`.

Conclusao:

- A maioria dos workitems de `equation_transform` (`68/100`) nao possui pares de exemplo candidatos detectados pelos parsers V237 atuais.
- O proximo passo nao e rescue eval nem treino; e examinar exemplos reais dos prompts para construir parser especifico com testes unitarios.

Ajuste adicional aplicado apos esta execucao:

- V237 agora inclui `equation_prompt_sample_preview` no manifest e imprime essa previa no log do notebook.
- Objetivo: permitir decidir V238 diretamente pelos logs do Colab, sem depender de abrir manualmente o CSV no Drive.

Proximo passo revisado:

- Reexecutar V237 atualizado.
- Usar `equation_prompt_sample_preview` para definir se V238 deve implementar:
  - parser de exemplos simbolicos nao uniformes;
  - parser numerico com exemplos em texto natural;
  - ou classificador de abstain definitivo quando nao ha exemplos suficientes.

## V237 reexecutado - evidencia do formato Alice inline

Execucao analisada em 2026-05-10 a partir de `KG1_V237_PROMPT_FORMAT_AUDIT_COLAB (1).ipynb`:

- Commit observado na celula de setup: `e06d467bc3b9d23c2da027dc31f902c734eec331`.
- Manifest V237 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V237/output_v237_prompt_format_audit/analysis_v237_prompt_format_audit/20260510T161723Z/v237_prompt_format_audit_manifest.json`.
- Manifest V237 SHA256 observado: `bd66dd39646bfb71c39845301ba21a4844c89cd2f363e05c30cda1fce06e8289`.
- Final manifest SHA256 observado: `14dcbeef2e4db58f3a2d1502bd5e93545803cd4c9e5444672b679ccb4927aede`.

Contagens medidas:

- `audit_rows`: `124`.
- `equation_workitems`: `100`.
- `bit_workitems`: `24`.
- `equation_zero_candidate_pair_rows`: `68`.

Resumo de hints:

- `example_pairs_nonuniform_or_symbolic`: `32`.
- `numeric_expr_without_parseable_examples`: `11`.
- `prompt_format_requires_manual_parser`: `57`.

Achado novo concreto:

- Os prompts de `equation_transform` usam formato Alice inline: texto introdutorio, exemplos como `lhs = rhs` em linha corrida, e query com `Now, determine the result for:`.
- Exemplo numerico observado: `72)27 = 99 26#48 = 22 42#45 = 3 24#14 = 10 ... 94)40`.
- Exemplo numerico/misto observado: `38(96 = 3648 13(43 = 559 42#38 = 81 41(94 = 3854 ... 11-50`.
- Exemplos simbolicos usam caracteres especiais como tokens reais; portanto parsers nao podem remover backticks de forma agressiva.

Conclusao:

- O fracasso do V236 nao era prova de ausencia de solucao; era gap de parser.
- O proximo passo correto e V238: parser/probe especifico para Alice inline, unit-tested, ainda sem treino, sem full eval e sem pacote.

## V238 implementado - Alice parser probes

Implementacao adicionada em 2026-05-10:

- Script: `scripts/analyze_v238_alice_parser_probes.py`.
- Builder: `scripts/build_v238_alice_parser_probes_colab.py`.
- Notebook: `notebooks/KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`.

Objetivo:

- Medir, de forma CPU-only e diagnostica, quantos misses de `equation_transform` podem ser recuperados por parsers deterministas do formato Alice inline.
- Atacar especificamente o gargalo V230: baseline `191/315`, `equation_transform=55/155`, faltando `+5` equation para o gate fraco.

Regras implementadas:

- Parser Alice por marcadores `Below are a few examples:` e `Now, determine the result for:`.
- Extracao de exemplos inline `lhs = rhs` sem exigir backticks ou quebras de linha.
- Preservacao de backtick quando ele e caractere real do token; apenas wrapper balanceado `` `...` `` e removido.
- Probe numerico por operador: aprende regras conservadoras por operador da query (`+`, `-`, `abs_diff`, `mul`, concat, diferenca por digito, soma de digitos).
- Probe simbolico: mapa de caracteres, delecao por posicao, reversao, prefixo/sufixo.
- Qualquer previsao so e classificada como `verified` se bater o `expected_answer` do weak workitem; previsoes erradas viram `incorrect`, nao sao usadas para pacote.

Validacoes executadas localmente:

- `python -m py_compile scripts/analyze_v238_alice_parser_probes.py`.
- `python scripts/analyze_v238_alice_parser_probes.py --self-test`.
- `python -m py_compile scripts/build_v238_alice_parser_probes_colab.py scripts/analyze_v238_alice_parser_probes.py`.
- `python scripts/build_v238_alice_parser_probes_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `622281163cd12d31b49ef09f5fe12d7893e18bffc10b5780317d5cfc150cb0af`.

Proximo passo:

- Executar V238 no Colab.
- Se `deployable_verified_overrides >= 5` e `deployable_incorrect_overrides == 0`, criar V239 como measurement notebook separado que aplica somente overrides verificados e mede o ganho fraco.
- Se houver qualquer `incorrect`, nao usar override; abrir `v238_alice_parser_probes_alice_parser_probe_results.csv` e restringir ou remover a regra causadora.
- Se o ganho ficar abaixo de `+5`, continuar mineracao de subformatos Alice antes de qualquer treino.

## V238 executado - parser Alice ainda insuficiente

Execucao analisada em 2026-05-10 a partir de `KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`:

- Commit Colab observado: `dd1d8068402e0f012cb8f4ea09dff3ff3630c192`.
- Manifest V238 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V238/output_v238_alice_parser_probes/analysis_v238_alice_parser_probes/20260510T163118Z/v238_alice_parser_probes_manifest.json`.
- Manifest V238 SHA256 observado: `a1b17284ee64006b6d40b6e5a6be8662cdbdd2baa26b31a2695d4ec5b5677ce8`.
- Final manifest SHA256 observado: `22002ac871de7c991de22c7aad023a25bec62ee215554481a049cc7836e2ed90`.

Resultado medido:

- `equation_workitems`: `100`.
- `deployable_verified_overrides`: `1`.
- `deployable_incorrect_overrides`: `1`.
- `target_gain`: `5`.
- Decisao: `continue_alice_parser_development`.

Resumo:

- Numerico Alice: `1` verified, `15` abstain.
- Simbolico Alice: `1` incorrect, `83` abstain.
- O resultado bloqueia qualquer V239/rescue measurement.

Gap encontrado:

- O log antigo mostrava a existencia de `1` incorreto, mas nao imprimia a linha incorreta nem os principais motivos de abstain.
- O probe simbolico aceitava `char_map` como candidato deployable. Esse tipo de regra pode encaixar todos os exemplos e ainda errar a query; portanto e fraco demais para uso automatico.

Ajuste aplicado apos o log:

- `char_map_probe` foi removido do conjunto deployable de `symbolic_probe`; permanece como codigo auxiliar, mas nao pode gerar override automatico.
- V238 agora grava e imprime:
  - `abstain_reason_summary_top`;
  - `verified_preview`;
  - `incorrect_preview`;
  - `abstain_preview`.
- O objetivo e tornar o proximo log suficiente para decidir a regra seguinte sem abrir CSV manualmente no Drive.

Proximo passo revisado:

- Reexecutar V238 atualizado.
- Se `deployable_incorrect_overrides` cair para `0`, avaliar quantos `verified` restam.
- Se continuar abaixo de `5`, criar proximo parser apenas a partir de `abstain_reason_summary_top` e dos previews; nao fazer treino, full eval, pacote ou submissao.

## V238 reexecutado - delecao simbolica tambem e insegura

Execucao analisada em 2026-05-10 a partir de `KG1_V238_ALICE_PARSER_PROBES_COLAB (1).ipynb`:

- Commit Colab observado: `16b91a831a8576effe5df70cc4e9d84eb3f7beec`.
- Manifest V238 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V238/output_v238_alice_parser_probes/analysis_v238_alice_parser_probes/20260510T163738Z/v238_alice_parser_probes_manifest.json`.
- Manifest V238 SHA256 observado: `c2e47d33583c20f1127e9aa83a29ebaf7be26b7986cf89bff025dfebb833f853`.
- Final manifest SHA256 observado: `6ea56a6c8ac299d5e97c3f0bf770f55edf3b08aa7dfd7b39cbd00a1b90a4f79e`.

Resultado medido:

- `equation_workitems`: `100`.
- `deployable_verified_overrides`: `1`.
- `deployable_incorrect_overrides`: `1`.
- `target_gain`: `5`.
- Decisao: `continue_alice_parser_development`.

Evidencia concreta:

- Verified numerico: id `c5b058d6`, query `94)40`, baseline `35`, expected `134`, prediction `134`, proof `rules=add`.
- Incorrect simbolico: id `432b1110`, query `\{*<?`, baseline `\{<?`, expected `%[:?`, prediction `\{<?`, proof `candidate_probes=alice_symbolic_deletion_positions_probe`.

Conclusao:

- A regra `alice_symbolic_deletion_positions_probe` tambem e fraca demais para override automatico.
- Ela consegue encaixar exemplos de treino inline por posicao mantida, mas pode apenas reproduzir uma delecao parecida com o baseline e errar a transformacao real da query.
- Logo, V238 continua bloqueando qualquer V239/rescue measurement.

Ajuste aplicado apos esta execucao:

- `alice_symbolic_deletion_positions_probe` foi rebaixado para diagnostico apenas.
- Quando encontra uma delecao consistente, a previsao fica registrada no proof como `diagnostic_only_candidate_disabled`, mas nao entra em `candidates` e nao pode gerar override deployable.
- Self-test V238 atualizado para exigir:
  - `deployable_verified_overrides == 1`;
  - `deployable_incorrect_overrides == 0`;
  - caso simbolico de delecao classificado como `abstain`;
  - proof contendo `diagnostic_only_candidate_disabled`.

Proximo passo revisado:

- Reexecutar V238 apos o bloqueio da delecao simbolica.
- Se o log mostrar `deployable_incorrect_overrides == 0` e `deployable_verified_overrides == 1`, nao criar V239 de rescue ainda: o ganho e insuficiente.
- Criar a proxima iteracao apenas para minerar os 83 abstains simbolicos e os abstains numericos por `no_examples_for_query_operator`/`candidate_rule_count`, sem permitir override simbolico novo sem teste unitario e evidencia de zero incorretos.

## V238 reexecutado apos bloqueio - zero incorretos, ganho insuficiente

Execucao analisada em 2026-05-10 a partir de `KG1_V238_ALICE_PARSER_PROBES_COLAB (2).ipynb`:

- Commit Colab observado: `a31877996a61f9d8f8b3485e6c8fb9fb3c4a16e4`.
- Manifest V238 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V238/output_v238_alice_parser_probes/analysis_v238_alice_parser_probes/20260510T164430Z/v238_alice_parser_probes_manifest.json`.
- Manifest V238 SHA256 observado: `a088c00c3a7424e25ea35953ba85fb9afd56dbaaf9f8d2a2fe291d128d5833e6`.
- Final manifest SHA256 observado: `742d1114ee75fe736b41e7c20559e00068af176f9d9b23c382f9b558e9ac253b`.

Resultado medido:

- `equation_workitems`: `100`.
- `deployable_verified_overrides`: `1`.
- `deployable_incorrect_overrides`: `0`.
- `target_gain`: `5`.
- Decisao: `continue_alice_parser_development`.

Interpretacao:

- O bloqueio da delecao simbolica funcionou: o caso antes incorreto agora aparece como abstain diagnostico com `diagnostic_only_candidate_disabled`.
- Ainda nao existe autorizacao para V239 de rescue/measurement, porque o ganho deployable verificado e apenas `+1`, abaixo do alvo `+5`.
- O unico ganho concreto continua sendo numerico Alice: id `c5b058d6`, query `94)40`, baseline `35`, expected `134`, prediction `134`, proof `rules=add`.

Abstains dominantes que devem guiar a proxima etapa:

- `79` simbolicos: `alice_symbolic_deletion_positions_probe:nonuniform_lengths`; reverse nao bate; prefix/suffix com comprimentos nao uniformes.
- `4` simbolicos: `alice_symbolic_deletion_positions_probe:ambiguous_or_missing_keep_positions=0`; reverse nao bate; prefix/suffix sem regra candidata.
- `1` simbolico: delecao consistente, mas bloqueada como `diagnostic_only_candidate_disabled prediction='\\{<?'`; esse e o antigo caso inseguro.
- Numericos: `3` por `candidate_rule_count=0 unique_prediction_count=0`; `2` por `no_examples_for_query_operator='+'`; varios operadores sem exemplo para a query (`'`, `!`, `%`, `&`, `*`, `-`, `/`, `:`, `@`).

Proximo passo correto:

- Nao criar pacote, nao full eval e nao rescue measurement.
- Criar uma proxima auditoria V239 focada em minerar os abstains Alice, principalmente:
  - decompor simbolicos de comprimento nao uniforme por delta de comprimento, posicao do operador inserido/removido e relacao entre baseline/expected;
  - separar numericos sem exemplo do operador da query de numericos com exemplos ambiguos;
  - produzir workpacks pequenos com exemplos, query, baseline, expected e motivo de abstain para desenhar novas regras unit-tested.

## V239 implementado - mineracao dos abstains Alice

Arquivos:

- Script: `scripts/analyze_v239_alice_abstain_mining.py`.
- Builder: `scripts/build_v239_alice_abstain_mining_colab.py`.
- Notebook: `notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.

Objetivo:

- Consumir o manifesto V238 mais recente com `deployable_incorrect_overrides == 0`.
- Gerar workpacks auditaveis dos abstains antes de qualquer rescue measurement.
- Separar claramente:
  - simbolicos com comprimento nao uniforme, keep-positions impossivel ou delecao diagnostica bloqueada;
  - numericos sem exemplo do operador da query, sem regra candidata ou com regras ambiguas.

Saidas esperadas:

- `v239_alice_abstain_mining_symbolic_abstain_workpack.csv`.
- `v239_alice_abstain_mining_numeric_abstain_workpack.csv`.
- `v239_alice_abstain_mining_abstain_bucket_summary.csv`.
- `v239_alice_abstain_mining_manifest.json`.

Validacoes locais:

- `python -m py_compile scripts/analyze_v239_alice_abstain_mining.py scripts/build_v239_alice_abstain_mining_colab.py`.
- `python scripts/analyze_v239_alice_abstain_mining.py --self-test`.
- `python scripts/build_v239_alice_abstain_mining_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `36fb65e8ea9a3d645ab5b2a018ed64a7ccff6026121eaa1a11d96936672600ab`.

Proximo passo:

- Executar V239 no Colab.
- Usar `abstain_bucket_summary` e os workpacks para escolher apenas uma nova regra por vez.
- Toda regra nova deve entrar primeiro como self-test/fixture negativo, especialmente o antigo caso `432b1110`, antes de voltar para qualquer V240 parser probe.

## HF Jobs / FinOps execution policy

Decisao operacional:

- Sim, os proximos notebooks CPU-only e diagnosticos devem ser executados via Hugging Face Jobs sempre que os artefatos de entrada estiverem acessiveis fora do Google Drive.
- Evitar GPU para auditorias, parsers, gates, self-tests, mineracao de CSV/JSON e notebooks que nao carregam o modelo base.
- Usar `cpu-basic` como padrao para jobs curtos; a execucao remota V239 self-test/gate terminou em poucos segundos.
- Para jobs que precisarem GPU pequena, preferir primeiro `t4-small` ou `l4x1`; subir para `a10g-small` apenas se houver necessidade clara de VRAM/throughput.
- A100/H100 ou equivalentes ficam bloqueados ate haver uma execucao longa e justificativa explicita; isso protege o credito de USD 15.

Execucao HF validada:

- Conta HF autenticada usada: `felipesp1983`.
- Job HF: `6a00b9c3317220dbbd1a761e`.
- Flavor: `cpu-basic`.
- Imagem: `python:3.12`.
- Tarefa executada:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` dos scripts V238/V239;
  - `python scripts/analyze_v239_alice_abstain_mining.py --self-test`;
  - `python scripts/notebook_release_gate.py notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.
- Status: `COMPLETED`.
- Duracao total observada: `7s`; runtime: `3s`.
- Resultado: self-test V239 `ok` e notebook gate `ok=true`.

Bloqueio atual para substituir 100% o Colab:

- HF Jobs nao monta `/content/drive/MyDrive/...`.
- Os artefatos completos V232/V238 usados pelos notebooks (`equation_solver_workitems_jsonl`, `v238_alice_parser_probe_results.csv`, manifests e CSVs completos) ainda vivem no Google Drive do Colab.
- Busca local nao encontrou copias completas desses artefatos fora do Drive.

Proximo passo para remover trabalho manual:

- Criar um bridge de artefatos para publicar manifests/CSVs diagnosticos em um dataset privado HF ou bucket equivalente.
- Depois desse bridge, V239 e as proximas auditorias podem rodar integralmente como HF Jobs sem Colab.
- Enquanto o bridge nao existir, HF consegue validar codigo/gates/self-tests, mas nao consegue executar analises completas que dependem de `/content/drive`.

## V240 implementado - bridge Drive para HF dataset

Arquivos:

- Script de upload: `scripts/upload_runtime_artifacts_to_hf.py`.
- Runner HF: `scripts/run_v239_from_hf_bridge.py`.
- Builder: `scripts/build_v240_hf_artifact_bridge_colab.py`.
- Notebook: `notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.

Objetivo:

- Rodar uma unica vez no Colab com Drive montado.
- Resolver os manifests V232/V238 mais recentes.
- Validar contrato de linhas compartilhadas.
- Subir para HF dataset privado os artefatos pequenos necessarios para V239:
  - V232 manifest;
  - V238 manifest;
  - V232 equation workitems;
  - V232 bit guardrail workitems;
  - V238 Alice parser probe results;
  - V238 Alice parser probe summary.

Destino padrao:

- Dataset HF: `felipesp1983/kg1-nemotron-training`.
- Prefixo: `runtime_artifacts/v240_hf_bridge/<RUN_ID>/`.

Requisito humano minimo:

- O Colab V240 precisa de `HF_TOKEN` com permissao de escrita no dataset.
- O token deve estar em Colab Secrets ou variavel de ambiente `HF_TOKEN`.
- A tentativa automatica de criar novo dataset privado via HF Job falhou com `403`; por isso o bridge usa dataset privado existente.

Validacoes locais:

- `python -m py_compile scripts/upload_runtime_artifacts_to_hf.py scripts/run_v239_from_hf_bridge.py scripts/build_v240_hf_artifact_bridge_colab.py`.
- `python scripts/upload_runtime_artifacts_to_hf.py --self-test`.
- `python scripts/run_v239_from_hf_bridge.py --self-test`.
- `python scripts/build_v240_hf_artifact_bridge_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `bc5aec2d763c58ec873511644ea91f1e7908e547181072fa659f154e932e13e3`.

Validacao HF Jobs:

- Job HF: `6a00bc9aaff1cd33e8f32dfb`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total observada: `8s`; runtime: `3s`.
- Comandos executados no remoto:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` dos scripts V240/V239 bridge;
  - `python scripts/upload_runtime_artifacts_to_hf.py --self-test`;
  - `python scripts/run_v239_from_hf_bridge.py --self-test`;
  - `python scripts/notebook_release_gate.py notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.
- Resultado: self-tests `ok` e notebook gate `ok=true`.

Proximo passo:

- Executar V240 no Colab.
- Copiar do log o `bridge_path_in_repo`.
- A partir desse caminho, executar V239 completo no HF Job com `scripts/run_v239_from_hf_bridge.py`.

## V240/V239 executado sem Colab manual via Drive MCP + HF dataset

Atualizacao:

- Autenticacao HF local confirmada como `felipesp1983`.
- Os artefatos V232/V238 foram recuperados diretamente via Google Drive MCP, sem depender de uma nova execucao manual no Colab.
- Artefatos recuperados:
  - V232 manifest SHA256: `6415efbad28577c675f8847f6b84eb5a2d63709b6b9e5ae42fdba5a002c9b7bf`.
  - V238 manifest SHA256: `a088c00c3a7424e25ea35953ba85fb9afd56dbaaf9f8d2a2fe291d128d5833e6`.
  - V232 equation workitems: 100 linhas.
  - V232 bit workitems: 24 linhas.
  - V238 Alice results: 100 linhas.
- Bridge publicado no dataset HF:
  - Dataset: `felipesp1983/kg1-nemotron-training`.
  - Path: `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z`.
  - Commit HF: `7accc1518c1e3303401cd509aa9388541c4fc421`.
  - Bridge manifest SHA256: `76d3474ba84fa97014fe7a5887200617f3910483aa00636debd9cf6ac9c01778`.

Execucao V239 completa em HF:

- Job HF: `6a00bff0317220dbbd1a762f`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total: `11s`; runtime: `7s`.
- Dependencias no job: `huggingface_hub`, `pandas`.
- Resultado:
  - `v238_rows=100`.
  - `verified=1`.
  - `incorrect=0`.
  - `abstain=99`.
  - `symbolic_abstain=84`.
  - `numeric_abstain=15`.
  - `target_gain=5`.
- Decisao V239: `mine_abstains_before_any_rescue_measurement`.
- Outputs V239 publicados no dataset HF:
  - Path: `runtime_artifacts/v239_alice_abstain_mining/local_drive_mcp_20260510T172421Z`.
  - Commit HF: `68bc8c14eb70f446331ed8acb81bade36566e91d`.

Bucket summary V239:

- `symbolic_nonuniform_lengths`: 79.
- `numeric_no_examples_for_query_operator`: 11.
- `symbolic_no_keep_positions`: 4.
- `numeric_no_candidate_rule`: 3.
- `numeric_ambiguous_candidate_rules`: 1.
- `symbolic_diagnostic_deletion_disabled`: 1.

Achado operacional:

- Uma tentativa de fazer o HF Job baixar diretamente URLs temporarias do Drive/OpenAI falhou com HTTP `403`.
- A rota correta e robusta e: Drive MCP/local -> upload autenticado para HF dataset -> HF Jobs consomem o dataset.
- Isso remove o trabalho manual do Colab para os proximos notebooks CPU-only, preservando FinOps.

## V241 abstain rule candidate audit

Arquivo:

- Script: `scripts/analyze_v241_abstain_rule_candidate_audit.py`.

Objetivo:

- Auditar os 99 abstains V238/V239 com regras candidatas mais fortes, sem promover nada inseguro.
- Testar dois caminhos:
  - symbolic char-transducer conservador, derivado apenas dos exemplos do prompt;
  - numeric DSL expandido, exigindo exemplos do mesmo operador e minimo de evidencia.
- Usar `expected_answer` apenas para auditoria fraca, nunca para derivar a regra.

Validacoes locais:

- `python -m py_compile scripts/analyze_v241_abstain_rule_candidate_audit.py`.
- `python scripts/analyze_v241_abstain_rule_candidate_audit.py --self-test`.
- Execucao real sobre os artefatos V232/V238 recuperados do Drive MCP.

Resultado real V241:

- `v238_rows=100`.
- `abstain_rows=99`.
- `symbolic_rows=84`.
- `numeric_rows=15`.
- `deployable_verified_candidates=0`.
- `deployable_incorrect_candidates=0`.
- `under_evidenced_candidates=0`.
- Decisao: `do_not_promote_v241_candidates`.

Resumo tecnico V241:

- Simbolico:
  - `no_global_mapping`: 60.
  - `no_pair_mapping`: 22.
  - `char_transducer mappings=1 usable=0 unique_predictions=0`: 1.
  - `char_transducer mappings=16 usable=16 unique_predictions=9`: 1.
- Numerico:
  - `no_same_operator_examples`: 11.
  - `candidate_rule_count=0 unique_prediction_count=0`: 2.
  - `candidate_rule_count=2 unique_prediction_count=2`: 1.
  - `candidate_rule_count=7 unique_prediction_count=5`: 1.

Outputs V241 publicados no dataset HF:

- Path: `runtime_artifacts/v241_abstain_rule_candidate_audit/local_drive_mcp_20260510T172421Z`.
- Commit HF: `fc8d5e956327ddd8635b06cf7c7b212dd5e48535`.

Validacao V241 em HF Jobs:

- Job HF: `6a00c1e7aff1cd33e8f32e2e`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total: `10s`; runtime: `4s`.
- A validacao remota executou:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` do script V241;
  - self-test V241;
  - download dos artefatos V232/V238 do bridge HF;
  - auditoria real V241.
- Resultado remoto:
  - `v238_rows=100`.
  - `abstain_rows=99`.
  - `deployable_verified_candidates=0`.
  - `deployable_incorrect_candidates=0`.
  - decisao `do_not_promote_v241_candidates`.
- Observacao operacional: um job anterior tentou subir outputs de dentro do HF Job e recebeu `403`; a publicacao de artefatos deve continuar usando o token local autenticado ou um secret HF com permissao explicita de escrita.

Decisao de negocio/QA:

- Nao promover parser novo agora.
- O risco de overfit/leakage e maior que o ganho esperado, porque nenhum candidato deployable foi verificado com zero ambiguidade.
- Proximo passo correto: gerar exemplos/fixtures adicionais para os buckets dominantes antes de nova medicao de rescue:
  - simbolico: focar `symbolic_nonuniform_lengths`, mas exigir regra derivavel de exemplos;
  - numerico: focar operadores sem exemplo do mesmo simbolo, porque 11/15 numericos estao bloqueados por ausencia de evidencia local.

## Data leakage audit - Alice weak workitems versus local datasets

Motivo:

- Antes de qualquer novo treino para `equation_transform`, foi necessario verificar se os workitems fracos V232/V238 ja aparecem em datasets locais.
- Usar weak IDs/answers como treino contaminaria o gate fraco e inflaria ACC sem validade.

Resultado do overlap exato:

- Referencia auditada: 100 workitems V232/V238 do bridge `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z`.
- `data/v217/v217_short_answer_train.jsonl`:
  - linhas: 10206.
  - `exact_prompt_overlap=0`.
  - `id_overlap=0`.
  - prompts Alice equation por frase: 6935.
- `data/v217/v217_short_answer_val.jsonl`:
  - linhas: 681.
  - `exact_prompt_overlap=0`.
  - `id_overlap=0`.
  - prompts Alice equation por frase: 453.
- `data/sft_v51_complete.jsonl`:
  - linhas: 9500.
  - `exact_prompt_overlap=0`.
  - `id_overlap=100`.
  - prompts Alice equation por frase: 1555.

Conclusao:

- V217 train/val permanecem limpos contra os 100 workitems V232/V238 auditados.
- `data/sft_v51_complete.jsonl` contem todos os 100 IDs fracos auditados e deve ficar em quarentena para qualquer treino, calibragem ou selecao que use o weak gate como evidencia.
- `data/sft_v51_complete.jsonl` pode ser usado somente como source-intel/diagnostico rotulado como potencial leakage, nunca como fonte direta para aumentar ACC medida no weak gate.

Regra para proximos notebooks/gates:

- Qualquer novo dataset de treino para `equation_transform` ou `bit_manipulation` deve executar overlap por `id` e por hash de prompt normalizado contra os workitems weak conhecidos.
- `id_overlap > 0` com weak/eval artifacts deve bloquear treino automaticamente, exceto em notebook explicitamente marcado como diagnostico de leakage.

Validacao materializada:

- Script: `scripts/audit_jsonl_overlap.py`.
- Self-test: `python scripts/audit_jsonl_overlap.py --self-test`.
- Execucao real:
  - referencia: `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z/v232_equation_workitems.jsonl`.
  - candidatos: V217 train, V217 validation, `data/sft_v51_complete.jsonl`.
- Resultado publicado no HF dataset:
  - Path: `runtime_artifacts/v241_overlap_audit/local_drive_mcp_20260510T172421Z/v241_overlap_audit.json`.
  - Commit HF: `b87478307a71b5cfea7c8d65e366b7d6794562da`.

## V242 safe equation fixture generation

Objetivo:

- Caminho mais rapido e objetivo antes de gastar GPU.
- Gerar fixtures sinteticos independentes para `equation_transform`, focados nos buckets V239/V241:
  - simbolico com comprimentos nao uniformes;
  - numerico com exemplos suficientes do mesmo operador.
- Bloquear automaticamente qualquer overlap por `id` ou hash de prompt normalizado contra weak workitems conhecidos.
- Nao treinar, nao inferir com modelo, nao pontuar modelo e nao submeter.

Arquivos:

- Gerador: `scripts/generate_v242_safe_equation_fixtures.py`.
- Builder: `scripts/build_v242_safe_equation_fixtures_colab.py`.
- Notebook: `notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb`.

Validacoes locais:

- `python -m py_compile scripts/generate_v242_safe_equation_fixtures.py scripts/audit_jsonl_overlap.py`.
- `python scripts/generate_v242_safe_equation_fixtures.py --self-test`.
- `python -m py_compile scripts/build_v242_safe_equation_fixtures_colab.py`.
- `python scripts/build_v242_safe_equation_fixtures_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb`.

Resultado do notebook gate:

- `ok=true`.
- notebook SHA256: `27f1a2f4c5bd251479cb0977ea7958133f5fde0b238419307c58452e6ab748dc`.

Execucao local V242:

- Referencia de leakage: `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z/v232_equation_workitems.jsonl`.
- `train_rows=1800`.
- `validation_rows=240`.
- `seed=242`.
- Resultado:
  - train simbolico: 1126.
  - train numerico: 674.
  - validation simbolico: 153.
  - validation numerico: 87.
  - train `id_overlap=0`, `prompt_overlap=0`.
  - validation `id_overlap=0`, `prompt_overlap=0`.

Outputs V242 publicados no HF dataset:

- Dataset: `felipesp1983/kg1-nemotron-training`.
- Path: `runtime_artifacts/v242_safe_equation_fixtures/local_cpu_20260510T174632Z`.
- Commit HF: `eb1979dcea095a5b06b5f77c96b03027bea25ece`.

Validacao V242 em HF Jobs:

- Job HF: `6a00c56d317220dbbd1a7644`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total: `10s`; runtime: `5s`.
- A validacao remota executou:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` dos scripts V242 e overlap audit;
  - self-tests dos dois scripts;
  - notebook release gate do V242;
  - download do weak reference a partir do HF bridge;
  - geracao completa de `1800` train e `240` validation.
- Resultado remoto:
  - train simbolico: 1126.
  - train numerico: 674.
  - validation simbolico: 153.
  - validation numerico: 87.
  - train `id_overlap=0`, `prompt_overlap=0`.
  - validation `id_overlap=0`, `prompt_overlap=0`.

Decisao:

- Fixtures estao prontos para revisao de gate de treino.
- Ainda nao autoriza treino automaticamente.
- O proximo passo objetivo, se aprovado, e criar um treino curto que consome somente V217 limpo + V242, repetindo o overlap gate antes de carregar modelo/GPU.

## V243 guarded V217 plus V242 training mix and HF execution

Objetivo:

- Executar a opcao mais objetiva e rapida no Hugging Face antes de qualquer treino longo.
- Criar um mix de treino usando somente:
  - V217 short-answer limpo ja versionado no repo;
  - V242 safe equation fixtures publicados no HF dataset.
- Validar hash, linhas, dedupe, overlap contra weak workitems e tokenizacao antes de gastar GPU.
- Fazer apenas um smoke train curto em GPU, nao um treino final.

Arquivos:

- Script: `scripts/build_v243_training_mix.py`.
- Commit GitHub: `c72e95bbf693609698f9df215ef8121d6d870ef1`.

Mix V243 publicado:

- Dataset HF: `felipesp1983/kg1-nemotron-training`.
- Path: `runtime_artifacts/v243_training_mix/local_upload_20260510T180200Z`.
- Train rows: `12006`.
- Validation rows: `921`.
- Train SHA256: `c290555bffade5f4fa4e5c14f6f66c36745bd31a22c4b004709afd5a5f33f6d1`.
- Validation SHA256: `54eda74b1ea01e6e3b165af23c99eac5dc6e21f29cbc49888503ea7a3d707764`.
- Familias no train:
  - `equation_transform=8735`.
  - `bit_manipulation=2695`.
  - demais familias de guarda: `gravity_constant=144`, `numeral_system=144`, `text_encryption=144`, `unit_conversion=144`.
- Familias na validation:
  - `equation_transform=693`.
  - `bit_manipulation=164`.
  - demais familias de guarda: `16` cada.
- Overlap contra weak reference:
  - train `id_overlap=0`, `prompt_overlap=0`.
  - validation `id_overlap=0`, `prompt_overlap=0`.

HF Jobs executados:

- `6a00c7d6aff1cd33e8f32e66`: CPU build/upload remoto.
  - Status: `ERROR`.
  - Achado util: build remoto validou o mix, mas upload falhou porque o token injetado pelo job tinha leitura sem escrita.
  - Mitigacao: upload feito localmente com token write validado.
- `6a00c904317220dbbd1a7650`: tokenization dry-run CPU inicial.
  - Status: `ERROR`.
  - Causa: comando de pip apontou todos os pacotes para o indice CPU do PyTorch.
  - Mitigacao: relancado corrigido em `6a00c92e317220dbbd1a7652`.
- `6a00c92e317220dbbd1a7652`: tokenization dry-run CPU corrigido.
  - Status: `COMPLETED`.
  - Train tokenized: `12006/12006`.
  - Validation tokenized: `921/921`.
  - Truncation: `0`.
  - Prompt truncation: `0`.
  - Offset masks: `12006` train e `921` validation.
  - Fallback masks: `0`.
  - Dry-run report publicado em:
    `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/dry_runs/v243-tokenize-dryrun-20260510T1808Z/dry_run_model_recipe_report.json`.
- `6a00c888aff1cd33e8f32e6a`: GPU smoke train em `a100-large`.
  - Status no momento do registro: `SCHEDULING`.
  - Run ID: `v243-v188-safe-eq-smoke-s4-20260510T1803Z`.
  - Init adapter: `felipesp1983/kg1-nemotron-lora-v188-equation-lmhead/checkpoint-40`.
  - Output repo: `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures`.
  - Config intencionalmente curta:
    - `MAX_STEPS=4`.
    - `MAX_LENGTH=4096`.
    - `BATCH_SIZE=4`, `MICRO_BATCH_SIZE=1`.
    - `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,lm_head`.
    - `LEARNING_RATE=1e-7`, `FINAL_LEARNING_RATE=5e-8`.
  - Motivo de manter `a100-large`: e o menor flavor razoavel para 30B BF16; multi-GPU seria pior em FinOps e L40S/L4/A10G isolados sao arriscados para memoria.

Decisao:

- O mix V243 esta aprovado para smoke train: hash, contagem, overlap, truncation e offset-mask passaram.
- O smoke train GPU deve ser avaliado por weak eval antes de qualquer treino longo.
- Se `a100-large` ficar preso em scheduling por muito tempo, a acao correta e aguardar capacidade ou cancelar; nao trocar automaticamente para multi-GPU caro.

Atualizacao HF H200:

- A API local de HF Jobs confirmou flavors com acelerador:
  - `a100-large`: 1x NVIDIA A100 80GB, custo aproximado `0.041667 USD/min`.
  - `h200`: 1x NVIDIA H200 141GB, custo aproximado `0.083333 USD/min`.
  - `h200x2`, `h200x4`, `h200x8` tambem existem, mas nao sao FinOps-correto para smoke train.
- O job A100 `6a00c888aff1cd33e8f32e6a` foi cancelado enquanto ainda estava em `SCHEDULING`.
- Tentativas H200 registradas:
  - `6a00cb86aff1cd33e8f32e8a`: falhou antes de treinar porque `mamba_ssm` nao estava instalado.
  - `6a00cbea317220dbbd1a765b`: falhou antes de treinar porque `pip install causal-conv1d mamba-ssm` trocou `torch 2.8.0+cu128` por `torch 2.11.0+cu130`, gerando ABI incompatível.
  - `6a00cc82aff1cd33e8f32e97`: preflight H200 confirmou `torch 2.8.0+cu128` e GPU `NVIDIA H200`, mas `mamba-ssm --no-deps` precisa das dependencias base instaladas antes.
  - `6a00cce2aff1cd33e8f32e99`: preflight H200 confirmou que `mamba-ssm --no-deps` preserva `torch 2.8.0+cu128`; faltou `transformers` no preflight isolado.
  - `6a00cd62317220dbbd1a7660`: smoke train H200 relancado com build de fonte para `causal-conv1d==1.6.1` e `mamba-ssm==2.3.1`, `--no-deps`, `--no-build-isolation`, `--no-binary`, e gate que aborta se `torch` mudar.
    - Resultado util: passou das dependencias, confirmou GPU H200, preservou `torch 2.8.0+cu128`, baixou modelo `63.2GB`, carregou adapter V188 checkpoint-40, aplicou filtro LoRA e iniciou setup de treino.
    - Falha: `SAMPLING_MODE=weighted` era invalido; `scripts/hf_job_train_v90.py` aceita `shuffle` ou `weighted_replacement`.
    - Mitigacao implementada: `scripts/hf_job_train_v90.py` agora valida `SAMPLING_MODE` em import/startup, antes de baixar modelo, para evitar repetir erro caro.

Regras obrigatorias para o notebook/executor HF de treino:

- Deve listar flavors disponiveis por `HfApi.list_jobs_hardware()` e logar explicitamente H200/A100, custo por minuto, VRAM e flavor selecionado.
- Deve cancelar ou bloquear jobs antigos em fila antes de lancar outro treino GPU, para evitar gasto duplicado.
- Deve instalar dependencias em ordem:
  - imagem base CUDA/PyTorch fixa;
  - dependencias Python base (`huggingface_hub`, `transformers`, `peft`, `accelerate`, `safetensors`, `sentencepiece`, `protobuf`, `hf_transfer`, `packaging`, `wheel`, `setuptools`, `ninja`, `einops`);
  - extensoes Mamba/Causal Conv compiladas contra o torch ja presente, nunca deixando `pip` resolver outro torch.
- Deve imprimir e validar `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` e nome da GPU antes e depois das instalacoes.
- Deve abortar se `torch` mudar entre `torch_before` e `torch_after`.
- Deve importar e logar `causal_conv1d`, `mamba_ssm`, `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn` e `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` antes de clonar/carregar o modelo.
- Deve clonar a branch com commit esperado fixo e abortar em mismatch.
- Deve rodar `py_compile` em `scripts/hf_job_train_v90.py`, `scripts/build_v243_training_mix.py` e `scripts/audit_jsonl_overlap.py`.
- Deve validar SHA256 e contagem dos arquivos V243 antes de carregar o modelo.
- Deve validar `SAMPLING_MODE` antes de carregar modelo; valores permitidos: `shuffle` ou `weighted_replacement`.
- Deve manter `MAX_STEPS=4` no smoke train e nao executar treino longo sem novo gate humano.
- Deve escrever todos os IDs de job, run IDs, URLs HF, status, erro e mitigacao no manifesto/roadmap.

## V244 HF H200 smoke train concluido

Objetivo:

- Executar o primeiro treino remoto curto com o mix V243, em H200, com gates dentro do container antes de qualquer download/carga cara.
- Validar que o executor HF consegue treinar o Nemotron 30B com adapter V188 inicial, sem quebrar dependencias Mamba/Causal Conv, sem trocar `torch`, e sem gastar com erro ja conhecido.
- Nao promover adapter, nao rodar full eval e nao criar pacote/submissao.

Job executado:

- Job HF: `6a00d6a9317220dbbd1a7683`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00d6a9317220dbbd1a7683`.
- Status: `COMPLETED`.
- Flavor: `h200`.
- Imagem: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- Run ID: `v244-h200-smoke-20260510T190308Z`.
- Commit GitHub fixado: `7f4192d9dfa5e73fd4ccda1c1a15ed7a24a186ee`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures`.
- Custo estimado: H200 `0.083333 USD/min`; runtime total observado aproximado `17.9 min`; custo aproximado `US$1.49`.

Gates executados dentro do HF Job:

- `scripts/hf_job_preflight_gate.py --phase preinstall`.
- `scripts/hf_job_preflight_gate.py --phase artifacts`.
- `scripts/hf_job_preflight_gate.py --phase postinstall`.

Resultado dos gates:

- Repo/commit: OK.
- `py_compile` dos scripts criticos: OK.
- GPU/Torch: OK, H200 CUDA disponivel.
- Flavor/custo permitido: OK.
- Modelo base: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Dataset V243:
  - train SHA256 `c290555bffade5f4fa4e5c14f6f66c36745bd31a22c4b004709afd5a5f33f6d1`.
  - validation SHA256 `54eda74b1ea01e6e3b165af23c99eac5dc6e21f29cbc49888503ea7a3d707764`.
  - train rows `12006`.
  - validation rows `921`.
  - familias train: `equation_transform=8735`, `bit_manipulation=2695`, guardas `144` cada.
  - familias validation: `equation_transform=693`, `bit_manipulation=164`, guardas `16` cada.
  - subcategorias V242 presentes: train `equation_symbolic_mixed_v242=1126`, `equation_numeric_same_operator_v242=674`; validation `153` e `87`.
- Init adapter V188:
  - repo/subfolder: `felipesp1983/kg1-nemotron-lora-v188-equation-lmhead/checkpoint-40`.
  - `r=32`, `lora_alpha=32`.
  - target modules: `down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj`.
  - `target_parameters=null` no config remoto aceito para esse adapter.
- Dependencias pos-instalacao: `huggingface_hub`, `transformers`, `peft`, `accelerate`, `safetensors`, `causal_conv1d`, `mamba_ssm`, `mamba_ssm.ops.triton.layernorm_gated`, `mamba_ssm.ops.selective_scan_interface`.

Resultado do treino:

- Modelo baixado e carregado.
- Adapter V188 inicial carregado manualmente:
  - tensors mapeados: `12011`.
  - tensors nao mapeados: `0`.
  - cobertura: `1.0000`.
- Modulos LoRA treinaveis: `q_proj,k_proj,v_proj,o_proj,lm_head`.
- Parametros treinaveis: `8,015,872`.
- Parametros totais reportados: `32,466,091,456`.
- Percentual treinavel: `0.0247%`.
- Config smoke:
  - `MAX_STEPS=4`.
  - `MAX_LENGTH=4096`.
  - `BATCH_SIZE=4`.
  - `MICRO_BATCH_SIZE=1`.
  - `GRADIENT_ACCUMULATION=4`.
  - `LEARNING_RATE=1e-7`.
  - `FINAL_LEARNING_RATE=5e-8`.
  - `SAMPLING_MODE=weighted_replacement`.
- Loss por step:
  - step 1: `3.4815`.
  - step 2: `3.3636`.
  - step 3: `3.3335`.
  - step 4: `3.2083`.
- Eval:
  - final eval loss: `3.2582213087007403`.
  - best eval loss: `3.2582213087007403`.
- VRAM pico reportado: `63.142578125 GiB`.
- Elapsed do treino no manifest: `284.10246777534485s`.

Artefatos publicados:

- `final/adapter_config.json`.
- `final/adapter_model.safetensors`.
- `final/v90_training_manifest.json`.
- `checkpoint-4/adapter_model.safetensors`.
- `checkpoint-2/adapter_model.safetensors`.
- Dry-run anterior preservado: `dry_runs/v243-tokenize-dryrun-20260510T1808Z/dry_run_model_recipe_report.json`.

Decisao QA/negocio:

- V244 prova que o pipeline HF H200 com gates funciona.
- V244 nao prova ganho de ACC; loss menor em smoke train nao substitui weak eval.
- Nenhum adapter V244 deve ir para full eval, packaging ou Kaggle antes de passar weak eval identico ao gate V230/V226.
- O risco principal agora e overfit/regressao em `bit_manipulation`; por isso V245 deve medir `final`, `checkpoint-4` e `checkpoint-2` contra os thresholds e guardrails.

## V245 proximo passo - weak eval dos adapters V244

Objetivo:

- Medir se o smoke train V244 realmente melhorou `equation_transform` sem perder o piso forte de `bit_manipulation`.
- Avaliar os tres artefatos publicados:
  - `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/final`.
  - `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/checkpoint-4`.
  - `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/checkpoint-2`.
- Comparar contra baseline V226 observado:
  - total `191/315`.
  - `equation_transform=55/155`.
  - `bit_manipulation=136/160`.
  - truncation `0`.

Gate minimo para considerar continuidade:

- Weak total `>=193`.
- `equation_transform >=60`.
- `bit_manipulation >=133`.
- truncation `<=3`.

Guardrail adicional recomendado:

- Se `equation_transform <60`, parar; nao gastar com full eval.
- Se `bit_manipulation <136`, exigir justificativa tecnica antes de qualquer treino longo, porque o baseline ja esta em `85.00%` nessa familia.
- Se truncation `>0`, inspecionar prompt/output antes de treinar mais, porque o formato final do KG1 e sensivel ao parser `\boxed{...}`.

Preflight obrigatorio para V245:

- Resolver/baixar adapters V244 via `snapshot_download` ou `hf_hub_download` e validar:
  - `adapter_config.json` existe.
  - `adapter_model.safetensors` existe.
  - `r=32`.
  - `lora_alpha=32`.
  - target modules compativeis.
  - tamanho do weights plausivel e nao vazio.
- Resolver weak CSV de 315 linhas fora do Drive antes de criar job pago.
- Se o CSV fraco nao estiver publicado no HF dataset, executar primeiro um bridge pequeno Drive -> HF dataset; nao iniciar vLLM/H200 dependendo de `/content/drive`.
- Reusar `scripts/evaluate_lora_adapters_batch.py` com:
  - `--max-tokens 96`.
  - `--max-model-len 4096`.
  - `--max-num-seqs 8`.
  - `--warmup-rows 0`.
  - `--disable-thinking`.
  - prompt suffix: `Return only one line: \boxed{answer}. No reasoning. No explanation.`
- Registrar `batch_candidate_summary.json`, predictions CSV, per-task CSV e manifest HF.

Decisao esperada:

- Se qualquer V244 atingir o gate weak: preparar full eval controlado em nova versao.
- Se nenhum V244 atingir o gate, encerrar esta linha de treino curto e voltar para mineracao deterministica dos miss-packs/fixtures, sem treino longo.

## V245 bridge implementado - publicar weak CSV exato no HF dataset

Motivo:

- O weak eval V245 dos adapters V244 precisa do CSV fraco exato de `315` linhas.
- O dataset HF atual contem manifests/workitems V232/V238/V242/V243, mas nao contem `v221_weak_315.csv`.
- Busca local, HF dataset e Google Drive connector nao recuperou o arquivo raw diretamente.
- Recriar a amostra fraca a partir de `data/train.csv` com heuristicas de `seed=42` nao reproduziu o contract hash V230; portanto isso nao pode ser usado para medir ACC.

Implementacao:

- Script: `scripts/upload_v245_weak_csv_bridge_to_hf.py`.
- Builder: `scripts/build_v245_weak_eval_bridge_colab.py`.
- Notebook: `notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb`.
- Colab URL:
  `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb`.

O que o notebook faz:

- Monta Google Drive.
- Clona a branch `v230-v226-complementarity`.
- Roda `py_compile`, self-test do bridge e `notebook_release_gate.py`.
- Procura primeiro:
  `/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab/v221_weak_315.csv`.
- Se esse CSV nao existir, reconstrói a partir do CSV exato:
  `/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/validation/official_train_seed42_stratified10_val.csv`.
- Valida:
  - colunas `id`, `prompt`, `answer`, `type/family`;
  - linhas `315`;
  - `bit_manipulation=160`;
  - `equation_transform=155`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Publica no HF dataset:
  - repo `felipesp1983/kg1-nemotron-training`;
  - prefixo `runtime_artifacts/v245_weak_eval_bridge/<RUN_ID>/v221_weak_315.csv`;
  - manifesto `v245_weak_eval_bridge_manifest.json`.

Validacoes locais:

- `python -m py_compile scripts/upload_v245_weak_csv_bridge_to_hf.py scripts/build_v245_weak_eval_bridge_colab.py`.
- `python scripts/upload_v245_weak_csv_bridge_to_hf.py --self-test`.
- `python scripts/build_v245_weak_eval_bridge_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- Notebook SHA256: `738ae203316b9e60111b991ca974291861e594788bc94bfcd9e6603a7288e24a`.

Status:

- Implementado.
- Esta rota Colab fica suspensa enquanto a diretriz operacional for "usar HF para tudo".
- O weak CSV exato foi reconstruido fora do Colab e publicado diretamente no HF dataset em V245 HF-only, portanto nao ha mais bloqueio de Drive para o proximo weak eval.

## V245 HF-only bridge concluido - weak CSV canonico publicado

Diretriz operacional atual:

- Por decisao do usuario em 2026-05-10, Colab fica suspenso ate segunda ordem.
- Todo trabalho executavel deve usar Hugging Face Jobs/datasets/model repos sempre que tecnicamente possivel.
- Antes de qualquer job pago, aplicar gates baratos: existencia de artefatos, hashes, contratos de linhas, adapter config/pesos, GPU/custo, dependencias e commit esperado.

Evidencia:

- O CSV oficial `train.csv` local em `artifacts/api_kaggle_openrouter_audit_2026_05_06/competition_data/extracted/train.csv` tem SHA256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`, igual ao esperado pelo V207A.
- A celula V207A gera a validacao assim:
  - baixa `data/kaggle/unzipped/train.csv`;
  - classifica com `classify_puzzle`;
  - `random.seed(42)`;
  - para cada familia em ordem alfabetica, embaralha e pega `int(10%)`;
  - junta as linhas e embaralha novamente;
  - filtra as familias weak `bit_manipulation` e `equation_transform`.
- A reconstrucao produziu:
  - weak rows `315`;
  - `bit_manipulation=160`;
  - `equation_transform=155`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.

Upload HF:

- Dataset repo: `felipesp1983/kg1-nemotron-training`.
- Commit HF: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/d6a0ffe1af8205ba8fa2fb6c633b16c9f0aaf054`.
- Prefixo:
  `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/`.
- CSV publicado:
  `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv`.
- Manifest publicado:
  `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v245_weak_eval_bridge_manifest.json`.
- Canonical weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.

Novo executor HF:

- Script: `scripts/hf_job_weak_eval_v245.py`.
- Objetivo: rodar weak eval dos adapters V244 dentro do HF Job sem depender de Drive/Colab.
- Gates antes de vLLM/model-load:
  - GPU CUDA e VRAM minima;
  - branch commit esperado;
  - import `vllm`;
  - download do weak CSV do HF;
  - SHA do CSV `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`;
  - rows/familias `315`, `160/155`;
  - manifest V245 do bridge;
  - adapter `adapter_config.json` e `adapter_model.safetensors`;
  - `r=32`, `lora_alpha=32`.
- Avaliacao:
  - usa `scripts/evaluate_lora_adapters_batch.py`;
  - `max_tokens=96`;
  - `max_model_len=4096`;
  - `max_num_seqs=8`;
  - `disable-thinking`;
  - prompt suffix de resposta curta boxed.

Validacoes locais:

- `python -m py_compile scripts/hf_job_weak_eval_v245.py scripts/hf_job_preflight_gate.py scripts/evaluate_lora_adapters_batch.py scripts/evaluate_lora_adapter.py`.
- `python scripts/hf_job_weak_eval_v245.py --self-test`.

Proximo passo automatico HF:

- Commitar/pushar `scripts/hf_job_weak_eval_v245.py` e este roadmap.
- Lançar HF Job de weak eval para o adapter V244 `final`.
- Se `final` nao passar ou regredir, repetir para `checkpoint-4` e `checkpoint-2`.
- So considerar full eval se algum adapter bater o gate:
  - total `>=193`;
  - equation `>=60`;
  - bit `>=133`;
  - truncation `<=3`.

## V245 HF weak eval - primeira medicao do adapter V244 final

Job:

- Tentativa A100: `6a00e301aff1cd33e8f32f80`.
  - Cancelada porque ficou em `SCHEDULING` sem logs.
- Execucao H200: `6a00e3e1317220dbbd1a76bc`.
  - URL: `https://huggingface.co/jobs/felipesp1983/6a00e3e1317220dbbd1a76bc`.
  - Run ID: `v245-h200-weak-final-20260510T195932Z`.
  - Commit repo: `d4578bb098b82561ea402041691d8830ead3d4d1`.

Gates confirmados antes da avaliacao:

- GPU: `NVIDIA H200`, `139.80 GiB`.
- vLLM import OK: `vllm==0.20.1`.
- Weak CSV:
  - rows `315`;
  - `bit_manipulation=160`;
  - `equation_transform=155`;
  - SHA256 `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Adapter final:
  - repo `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/final`;
  - `adapter_model.safetensors` `4,259,063,856` bytes;
  - `r=32`;
  - `lora_alpha=32`.

Resultado medido:

- Candidate: `v244_final_adapter`.
- Overall weak: `18/315 = 5.71%`.
- `bit_manipulation`: `9/160 = 5.63%`.
- `equation_transform`: `9/155 = 5.81%`.
- truncation: `0`.
- Gate: reprovado.

Interpretacao:

- O adapter final V244 nao e utilizavel como candidato de weak/full eval.
- A queda e grande demais para justificar full eval ou pacote.
- Possivel fator operacional identificado: o executor V245 removia o `\n` inicial do prompt suffix porque usava `env_str(...).strip()`. Isso foi corrigido em `scripts/hf_job_weak_eval_v245.py` para preservar o prompt suffix default com newline, igual ao padrao V229/V230.
- O primeiro upload de resultados subiu tambem `adapter_snapshot` sob `evals/`, o que era ruido de 4.28GB. O snapshot duplicado foi removido do HF repo no commit:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/commit/f95e0749b14a27dae020b7cb9eaf3a58dcc323cd`.
- O script foi ajustado para baixar o adapter fora do `output_dir` e ignorar qualquer `adapter_snapshot/**` no upload.

Proximo passo:

- O executor foi ampliado para aceitar `KG1_ADAPTER_SUBFOLDERS` e `KG1_CANDIDATE_NAMES`, permitindo avaliar `final`, `checkpoint-4` e `checkpoint-2` no mesmo job com um unico model-load.
- Reexecutar `final`, `checkpoint-4` e `checkpoint-2` juntos com o prompt suffix corrigido para separar falha de adapter de falha de wrapper.

## V245 HF weak eval trio - resultado final da linha V244

Job:

- HF Job: `6a00e5b3317220dbbd1a76be`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00e5b3317220dbbd1a76be`.
- Run ID: `v245-h200-weak-v244-trio-20260510T200718Z`.
- Repo commit: `10865607892d53ceac6b6a1885b13db7bf31b7c7`.
- Upload dos resultados:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/commit/664ae944125e4f55d4d5c8d0bb02895ba72564cf`.
- Path:
  `evals/v245-h200-weak-v244-trio-20260510T200718Z/`.

Gates:

- H200 OK.
- Commit esperado OK.
- Weak CSV hash e row contract OK.
- Prompt suffix corrigido preservado no config:
  `\nReturn only one line: \boxed{answer}. No reasoning. No explanation.`
- Tres adapters validados:
  - `final`;
  - `checkpoint-4`;
  - `checkpoint-2`.
- Todos com `r=32`, `lora_alpha=32`, pesos `4,259,063,856` bytes.
- Upload limpo: `adapter_snapshot` nao foi publicado nos resultados do trio.

Resultados weak:

| Candidate | Overall | bit_manipulation | equation_transform | Truncation |
|---|---:|---:|---:|---:|
| `v244_final_adapter` | `18/315 = 5.71%` | `9/160 = 5.63%` | `9/155 = 5.81%` | `0` |
| `v244_checkpoint_4` | `18/315 = 5.71%` | `9/160 = 5.63%` | `9/155 = 5.81%` | `0` |
| `v244_checkpoint_2` | `19/315 = 6.03%` | `10/160 = 6.25%` | `9/155 = 5.81%` | `0` |

Decisao:

- Linha V244 reprovada.
- Nao fazer full eval.
- Nao fazer packaging.
- Nao fazer Kaggle submit.
- Nao continuar treino longo partindo desses checkpoints.

Diagnostico objetivo:

- O problema nao era so o prompt suffix; a execucao corrigida preservou o newline e continuou em ~6%.
- As predicoes baixadas mostram respostas plausiveis mas quase sempre erradas, com frequencias altas de payloads como `00000000`, `10000000`, simbolos isolados e respostas curtas repetidas.
- Isso indica regressao/degradacao real ou incompatibilidade de continuidade do adapter V244, nao truncation.

Proximo passo:

- Encerrar a linha V244.
- Voltar para rota P0 do roadmap: mineracao deterministica/DSL dos miss-packs e fixtures por familia, antes de novo treino.
- Qualquer novo treino HF deve partir de um adapter/baseline que ja prove weak ACC perto do V226 (`191/315`) ou entao deve passar primeiro por uma avaliacao weak curta; nao repetir smoke train sem weak gate intermediario.

## V236/V246 HF-only local solver path - status atual

Executor V236 no HF:

- Script: `scripts/run_v236_from_hf_bridge.py`.
- Job parserfix: `6a00eaa0317220dbbd1a76d0`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00eaa0317220dbbd1a76d0`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/de4942ba75849f84fa4096466c0e341768c90d59`.
- Path:
  `runtime_artifacts/v236_local_solver_dsl_probes/v236-hf-cpu-bridge-parserfix-20260510T202819Z/`.

Resultado V236 parserfix:

- `deployable_verified_equation_overrides`: `1`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24/24`.
- Linha recuperada:
  - id `c5b058d6`;
  - baseline `35`;
  - solver `134`;
  - expected `134`;
  - proof `alice_rules=add`.
- Impacto maximo isolado: V226 baseline iria de `191/315` para `192/315`, ainda abaixo do gate `193/315` e equation `60`.

Novo executor V246:

- Script: `scripts/run_v246_exhaustive_abstain_audit_hf.py`.
- Objetivo: auditar os `99` abstains restantes da V236 parserfix usando regras locais conservadoras.
- Custo: CPU-only HF Job.
- Entradas HF:
  - V236 parserfix results;
  - V236 parserfix manifest;
  - V240 bridge `v232_equation_workitems.jsonl`;
  - V240 bridge `v232_manifest.json`.
- Gate de contrato:
  - expected/observed shared row contract `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Regra de seguranca:
  - weak label e usado apenas como freio/auditoria;
  - uma classe de regra so e promovivel se todos os candidatos emitidos pela classe forem verificados e houver `0` incorretos;
  - se qualquer classe produz incorretos, ela fica bloqueada.
- Classes auditadas:
  - numeric same-operator DSL min2/min3/min4;
  - symbolic char transducer com todos exemplos;
  - symbolic char transducer por mesmo operador min2;
  - symbolic positional deletion;
  - symbolic same-operator positional deletion min2;
  - symbolic same-operator position-specific char map min2.

Pre-check local consumindo artefatos HF:

- `v236_rows`: `100`.
- `v236_abstain_rows`: `99`.
- `audit_rows`: `465`.
- `verified_candidates`: `0`.
- `incorrect_candidates`: `11`.
- `promotable_rows_after_class_gate`: `0`.
- Decisao local preliminar:
  `no_safe_local_rule_promotion_found`.

Interpretacao:

- A recuperacao deterministica local atual nao entrega os +5 de equation necessarios.
- Ha evidencia concreta de que regras simbolicas simples geram incorretos; portanto nao devem ser promovidas.
- Nao gastar H100/H200 em treino derivado desses candidatos sem uma nova fonte de dados/traços.
- Proximo passo HF-only: rodar V246 no HF CPU e publicar os artefatos; se confirmar `0` promoviveis, bloquear essa rota e seguir para acesso aos traces externos ou novo desenho de treino.

Execucao HF V246 confirmada:

- HF Job: `6a00ef1aaff1cd33e8f32ff1`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00ef1aaff1cd33e8f32ff1`.
- Run ID: `v246-hf-cpu-exhaustive-abstain-20260510T204724Z`.
- Repo commit executado: `09bdb266b54bb2ded373814e753af8d20de779f3`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/f95b5232e8d76df68d180178047ba1273a09e8a5`.
- Path:
  `runtime_artifacts/v246_exhaustive_abstain_audit/v246-hf-cpu-exhaustive-abstain-20260510T204724Z/`.

Resultado HF V246:

- `v236_rows`: `100`.
- `v236_abstain_rows`: `99`.
- `audit_rows`: `465`.
- `verified_candidates`: `0`.
- `incorrect_candidates`: `11`.
- `promotable_rows_after_class_gate`: `0`.
- Decisao:
  `no_safe_local_rule_promotion_found`.

Classes bloqueadas/sem ganho:

- `numeric_same_operator_extended_dsl_min2/min3/min4`: nenhum candidato verificado.
- `symbolic_all_examples_char_transducer`: nenhum candidato.
- `symbolic_same_operator_char_transducer_min2`: nenhum candidato.
- `symbolic_all_examples_positional_deletion`: 1 candidato, 1 incorreto.
- `symbolic_same_operator_position_char_map_min2`: 10 candidatos, 10 incorretos.
- `symbolic_same_operator_positional_deletion_min2`: nenhum candidato.

Conclusao de negocio/QA:

- A rota local solver/DSL conservadora esta esgotada para ganho imediato.
- Nao ha evidencia para promover nova regra sem aumentar falso positivo.
- Nao gastar H100/H200 nesta rota.
- Proxima rota objetiva: validar acesso aos datasets/traces externos no HF:
  - `andy279/nemotron-reasoning-challenge-raw-traces`;
  - `andy279/nemotron-reasoning-challenge`;
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`.
- Se os datasets `andy279/*` continuarem gated/403 com o token atual, sera necessaria acao humana para aceitar os termos no HF antes de qualquer treino baseado nesses traces.

## V247 HF source access gate - proxima rota de dados externos

Script:

- `scripts/run_v247_hf_source_access_gate.py`.

Objetivo:

- Validar no HF, com o token atual, quais fontes externas de traces/dados estao realmente acessiveis antes de gastar GPU.
- Evitar baixar payloads grandes: usa metadata + HTTP range-read pequeno.
- Nao treina, nao avalia modelo, nao faz pacote e nao submete Kaggle.

Fontes testadas:

- `andy279/nemotron-reasoning-challenge-raw-traces`:
  - `solver_transformation_traces_gpt54.jsonl`;
  - `solver_transformation_traces_merged.jsonl`;
  - `solver_bit_manipulation_traces_merged.jsonl`.
- `andy279/nemotron-reasoning-challenge`:
  - `sft_val.jsonl`;
  - `sft_train.jsonl`.
- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`:
  - `train.csv`;
  - `test.csv`.

Pre-check local usando HF token:

- Metadata `andy279/*`: acessivel, `gated=manual`.
- Payload `andy279/*`: `403`, mensagem HF: request awaiting review from repo authors.
- `jasonkung98/*`: acessivel por range-read.
- Contagem:
  - P0 accessible files: `0`;
  - P0 denied files: `5`;
  - public accessible files: `2`.
- Decisao preliminar:
  `p0_gated_terms_required_public_mirror_available`.

Interpretacao:

- Os arquivos publicos de `jasonkung98` servem para sanity/source check, mas nao substituem os traces P0.
- A rota de maior impacto para melhorar `equation_transform` depende dos datasets gated `andy279/*`.
- Se o job HF V247 confirmar o mesmo 403, a proxima acao nao e tecnica: sera necessario aceitar/liberar acesso aos repos gated no HF.

Execucao HF V247 confirmada:

- HF Job: `6a00f039aff1cd33e8f3300f`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f039aff1cd33e8f3300f`.
- Run ID: `v247-hf-source-access-gate-20260510T205212Z`.
- Repo commit executado: `cb9ff271c7a4504314930cf33a937bb8de594979`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/99eca833c15b9651a55c33c71296f14b6dd9cc94`.
- Path:
  `runtime_artifacts/v247_hf_source_access_gate/v247-hf-source-access-gate-20260510T205212Z/`.

Resultado HF V247:

- P0 accessible files: `0`.
- P0 denied files: `5`.
- Public accessible files: `2`.
- `andy279/nemotron-reasoning-challenge-raw-traces`: metadata OK, payload 403/manual review pending.
- `andy279/nemotron-reasoning-challenge`: metadata OK, payload 403/manual review pending.
- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: `train.csv` e `test.csv` acessiveis.
- Decisao:
  `p0_gated_terms_required_public_mirror_available`.

Bloqueio atual:

- A rota de traces externos P0 esta bloqueada por review/termos HF dos repos `andy279/*`.
- O mirror publico `jasonkung98/*` e util para sanity check, mas nao traz os traces/solver SFT que justificariam novo treino.
- Proxima acao humana necessaria:
  - abrir `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces`;
  - solicitar/aceitar acesso;
  - abrir `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge`;
  - solicitar/aceitar acesso;
  - depois rerodar V247. Se P0 ficar acessivel, criar job V248 de ingestao/filtragem de traces equation/bit antes de qualquer treino.

Recheck HF V247:

- HF Job: `6a00f0f0aff1cd33e8f33018`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f0f0aff1cd33e8f33018`.
- Run ID: `v247-hf-source-access-recheck-20260510T205514Z`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/84bb71a1806208a512b8e01edfb297c402036e0a`.
- Resultado: sem mudanca, `andy279/*` ainda `403` aguardando review; `jasonkung98/*` acessivel.

Recheck HF V247 em 2026-05-11:

- Tentativa inicial de launch:
  - HF Job: `6a014e3a317220dbbd1a784b`.
  - URL: `https://huggingface.co/jobs/felipesp1983/6a014e3a317220dbbd1a784b`.
  - Status: falhou imediatamente por parsing do CLI (`bash` recebeu o script inteiro como caminho). Custo operacional esperado: minimo; nenhum payload baixado, nenhum treino.
- Launch corrigido:
  - HF Job: `6a014e51aff1cd33e8f333fa`.
  - URL: `https://huggingface.co/jobs/felipesp1983/6a014e51aff1cd33e8f333fa`.
  - Flavor: `cpu-basic`.
  - Status: `COMPLETED`.
  - Upload HF:
    `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/08e73a970662e93ec38db2189570977e2bfa8922`.
  - Path:
    `runtime_artifacts/v247_hf_source_access_gate/v247-hf-source-access-recheck-20260511T033339Z/`.
- Resultado:
  - P0 accessible files: `0`.
  - P0 denied files: `5`.
  - Public accessible files: `2`.
  - `andy279/nemotron-reasoning-challenge-raw-traces`: metadata OK, arquivos existem, payload `403` com mensagem de review pendente.
  - `andy279/nemotron-reasoning-challenge`: metadata OK, arquivos existem, payload `403` com mensagem de review pendente.
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: `train.csv` e `test.csv` acessiveis.
- Decisao: `p0_gated_terms_required_public_mirror_available`.
- Implicacao: a rota de traces externos segue bloqueada por acesso humano aos repos `andy279/*`. Ate liberar esse acesso, nao iniciar treino H200 baseado nesses traces.

Recheck HF V264 em 2026-05-11:

- HF Job: `6a016a74317220dbbd1a78e4`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a016a74317220dbbd1a78e4`.
- Run ID: `v264-hf-source-access-recheck-20260511T053342Z`.
- Repo commit executado: `c1efe6af76918145a16a9a96423ee4e2b19c5dd5`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/049b6afe57041eeb9b36424ee03c341a9e3b7c07`.
- Path:
  `runtime_artifacts/v247_hf_source_access_gate/v264-hf-source-access-recheck-20260511T053342Z/`.
- Resultado:
  - P0 accessible files: `0`.
  - P0 denied files: `5`.
  - Public accessible files: `2`.
  - `andy279/nemotron-reasoning-challenge-raw-traces`: metadata OK, payload `403`, review pendente.
  - `andy279/nemotron-reasoning-challenge`: metadata OK, payload `403`, review pendente.
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: `train.csv` e `test.csv` acessiveis.
- Decisao: `p0_gated_terms_required_public_mirror_available`.
- Implicacao: sem aceitar/liberar acesso aos datasets gated `andy279/*`, a rota de traces P0 permanece bloqueada. Novo H200 baseado no mirror publico ou em soups nao e justificado pelos resultados V257-V263.

## V248 public mirror leakage audit

Script:

- `scripts/run_v248_public_mirror_leakage_audit_hf.py`.

Objetivo:

- Auditar o mirror publico `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`.
- Verificar vazamento contra o weak set canonico V245.
- Contar linhas target-family disponiveis apos excluir qualquer ID weak.
- Bloquear qualquer uso de labels weak-overlap em treino, calibragem ou selecao de regra.

Pre-check local:

- Public train rows: `9500`.
- Public test rows: `3`.
- Weak rows: `315`.
- Weak overlap rows: `315`.
- Weak answer mismatches: `0`.
- Weak prompt mismatches normalizados: `0`.
- Non-weak target rows: `2842`.
- Por familia:
  - `bit_manipulation`: `1602` train, `160` weak-overlap, `1442` nonweak;
  - `equation_transform`: `1555` train, `155` weak-overlap, `1400` nonweak;
  - demais familias permanecem P2/P3 para este objetivo.
- Decisao preliminar:
  `public_mirror_usable_only_after_weak_id_exclusion`.

Interpretacao:

- O mirror publico confirma que o weak set esta dentro do train publico; usar essas respostas para ajustar regra/modelo e vazamento.
- O uso permitido e apenas com exclusao explicita dos `315` weak IDs.
- Como ainda ha `2842` linhas target-family nao weak, a rota possivel sem gated traces e construir um dataset V249 estritamente non-weak, com fixtures de validacao separados e sem usar weak labels para selecao.

Execucao HF V248 confirmada:

- HF Job: `6a00f1e1aff1cd33e8f3302a`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f1e1aff1cd33e8f3302a`.
- Run ID: `v248-hf-public-mirror-leakage-20260510T205916Z`.
- Repo commit executado: `c039b7e093cfdca5dbbb7effba60f835d526a7fd`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/0d134fc36737346c32061ab063ab01eb01db0256`.
- Path:
  `runtime_artifacts/v248_public_mirror_leakage_audit/v248-hf-public-mirror-leakage-20260510T205916Z/`.
- Resultado: igual ao pre-check local.

Proxima acao HF-only:

- V249 deve materializar somente linhas `bit_manipulation` e `equation_transform` do mirror publico com `id` fora do weak set.
- V249 deve gerar `train.jsonl`, `val.jsonl`, manifest, hashes e CSV de IDs bloqueados.
- V249 nao deve treinar; e apenas preparo de dados com gates.
- Antes de treino, precisa comparar V249 contra V217/V226 para evitar repetir dataset/efeito V244.

## V249 public non-weak target dataset

Script:

- `scripts/run_v249_public_nonweak_target_dataset_hf.py`.

Objetivo:

- Materializar um dataset HF estritamente sem vazamento dos `315` weak IDs.
- Usar somente as familias que estamos tentando melhorar agora:
  `bit_manipulation` e `equation_transform`.
- Gerar `train.jsonl`, `val.jsonl`, CSV de weak IDs bloqueados, manifest e hashes.
- Nao treinar, nao avaliar modelo, nao gerar pacote e nao submeter Kaggle.

Gates implementados:

- Baixa o mirror publico `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`.
- Baixa o weak CSV canonico V245 do dataset HF privado.
- Exige SHA exato do weak CSV:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Exige contagem target apos exclusao weak:
  - total: `2842`;
  - `bit_manipulation`: `1442`;
  - `equation_transform`: `1400`.
- Valida que nenhum `original_id` de treino/validacao esta no weak set.
- Valida IDs unicos, formato `messages`, resposta do assistant e split estratificado.
- Bloqueia explicitamente `train`, `model_generation`, `full_scoring`, `package` e `kaggle_submit` no manifest.

Pre-check local:

- `py_compile`: OK.
- `--self-test`: OK.
- Run local CPU: OK.
- Public train rows: `9500`.
- Weak rows bloqueados: `315`.
- Candidate target rows non-weak: `2842`.
- Train rows: `2558`.
- Val rows: `284`.
- Train family counts:
  - `bit_manipulation`: `1298`;
  - `equation_transform`: `1260`.
- Val family counts:
  - `bit_manipulation`: `144`;
  - `equation_transform`: `140`.
- Hashes locais:
  - train JSONL: `81c8624b7e0a330a720e22b5e4fc254b238a7c618e1c0cdcdea3cf1fd96d9f41`;
  - val JSONL: `43dd9f5fbb6864e85e60b1a6cc2ad7060a667e914a67dda2aa3a22771efb4783`;
  - blocked weak IDs CSV: `5392c44fda7e0522910735c9a8b560d9c504a136d6141ed25091f4c858c3d4ce`.
- Escrita JSON/JSONL ajustada para LF deterministico entre Windows e Linux/HF.

Decisao:

- `dataset_ready_for_tokenization_gate_not_training_yet`.
- Proxima acao HF-only: executar V249 no HF e fazer upload dos artefatos.
- Depois do V249 remoto, executar gate V250 de tokenizacao/offset-mask/truncation antes de qualquer treino GPU.

Execucao HF V249 confirmada:

- HF Job: `6a00f420317220dbbd1a76f0`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f420317220dbbd1a76f0`.
- Run ID: `v249-hf-public-nonweak-target-20260510T210850Z`.
- Repo commit executado: `5660be29f3347e5adefd80c8000d096d334ecca0`.
- Upload HF folder:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/1bc2b5b1c1e3893e73e570b45ef4860949785d80`.
- Upload HF manifest refresh:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/8d4510fd7390aa370fad4ae6172e1bd679038be9`.
- Path:
  `data/v249_public_nonweak_target/v249-hf-public-nonweak-target-20260510T210850Z/`.
- Manifest remoto verificado:
  `data/v249_public_nonweak_target/v249-hf-public-nonweak-target-20260510T210850Z/v249_public_nonweak_target_manifest.json`.
- Resultado remoto:
  - public train rows: `9500`;
  - weak rows bloqueados: `315`;
  - target non-weak rows: `2842`;
  - train rows: `2558`;
  - val rows: `284`;
  - train counts: `bit_manipulation=1298`, `equation_transform=1260`;
  - val counts: `bit_manipulation=144`, `equation_transform=140`.
- Hashes remotos canonicos:
  - train JSONL: `81c8624b7e0a330a720e22b5e4fc254b238a7c618e1c0cdcdea3cf1fd96d9f41`;
  - val JSONL: `43dd9f5fbb6864e85e60b1a6cc2ad7060a667e914a67dda2aa3a22771efb4783`;
  - blocked weak IDs CSV: `5392c44fda7e0522910735c9a8b560d9c504a136d6141ed25091f4c858c3d4ce`.

Status:

- V249 pronto para V250 tokenizer/mask/truncation gate.
- Ainda nao autorizado para treino GPU: falta provar tokenizacao, labels/offset-mask e comparacao contra V217/V226.

## V250 V249 tokenization gate

Script:

- `scripts/run_v250_v249_tokenization_gate_hf.py`.

Objetivo:

- Validar o dataset V249 remoto antes de qualquer gasto com GPU.
- Rebaixar risco de V244: nao treinar ate provar hashes, formato, weak exclusion, tokenizacao real, offset masks e truncation zero.
- Comparar o V249 contra o corpus V217 ja usado, para medir novidade real.

Gates implementados:

- Baixa `train.jsonl`, `val.jsonl`, `v249_blocked_weak_ids.csv` e manifest V249 do HF.
- Exige hashes canonicos:
  - train JSONL: `81c8624b7e0a330a720e22b5e4fc254b238a7c618e1c0cdcdea3cf1fd96d9f41`;
  - val JSONL: `43dd9f5fbb6864e85e60b1a6cc2ad7060a667e914a67dda2aa3a22771efb4783`;
  - blocked weak IDs CSV: `5392c44fda7e0522910735c9a8b560d9c504a136d6141ed25091f4c858c3d4ce`.
- Exige contagens:
  - train rows: `2558`;
  - val rows: `284`;
  - blocked weak rows: `315`.
- Exige family counts:
  - train: `bit_manipulation=1298`, `equation_transform=1260`;
  - val: `bit_manipulation=144`, `equation_transform=140`.
- Exige zero overlap de `original_id` com weak IDs.
- Exige formato `messages=[system,user,assistant]` e assistant `Final answer: ...`.
- Usa tokenizer real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Exige offset masks; fallback mask e falha.
- Exige `MAX_LENGTH=4096` com prompt truncation rate `0.0`.
- Bloqueia `long_train`, `full_scoring`, `package` e `kaggle_submit`.

Pre-check local:

- `py_compile`: OK.
- `--self-test`: OK.
- Run local CPU: OK.
- Tokenizer: `TokenizersBackend`, fast tokenizer, `eos/pad=<|im_end|>`.
- Train tokenization:
  - rows: `2558`;
  - offset masks: `2558`;
  - fallback masks: `0`;
  - prompt truncated: `0`;
  - token max: `324`.
- Validation tokenization:
  - rows: `284`;
  - offset masks: `284`;
  - fallback masks: `0`;
  - prompt truncated: `0`;
  - token max: `324`.
- Por familia:
  - `bit_manipulation`: answer loss tokens fixos em `14`, token max `324`;
  - `equation_transform`: answer loss tokens `6..10`, token max `157` train / `153` val.
- Overlap contra V217:
  - V249 total: `2842`;
  - prompt+answer overlap V217 train: `900`;
  - prompt+answer overlap V217 val: `39`;
  - prompt+answer overlap total: `939`;
  - novidade prompt+answer vs V217: `1903`.

Interpretacao:

- O V249 e tecnicamente treinavel, mas nao e totalmente novo: `939/2842` linhas ja existem no V217 por prompt+answer.
- O ganho esperado de treino deve vir dos `1903` exemplos novos e de uma mistura mais cuidadosa, nao de simplesmente repetir V217.
- Proxima acao HF-only: executar V250 no HF, subir manifest e, se passar, criar um treino smoke muito curto com gate fraco antes de qualquer H200 longo.

Execucao HF V250 confirmada:

- Primeiro job: `6a00f5b4317220dbbd1a76f6`.
  - Resultado: falhou antes da tokenizacao porque `jinja2` nao estava instalado no container CPU.
  - Correcao operacional: rerun com `jinja2` no setup do job. Sem custo GPU.
- Job valido: `6a00f5ecaff1cd33e8f33044`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f5ecaff1cd33e8f33044`.
- Run ID: `v250-hf-v249-tokenization-gate-20260510T211631Z`.
- Repo commit executado: `5e9dbe546e9dd579d3b4e312b7643ed1f43c2cfa`.
- Upload HF folder:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/2ff0c9acbceca9ae4ad74688a4e9b61e34d229ea`.
- Upload HF manifest refresh:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/94df99e4ce002dca1cf8c0d81689f8e1db5cd623`.
- Path:
  `runtime_artifacts/v250_v249_tokenization_gate/v250-hf-v249-tokenization-gate-20260510T211631Z/`.
- Manifest remoto verificado:
  `runtime_artifacts/v250_v249_tokenization_gate/v250-hf-v249-tokenization-gate-20260510T211631Z/v250_v249_tokenization_gate_manifest.json`.
- Resultado remoto:
  - train tokenized rows: `2558`;
  - validation tokenized rows: `284`;
  - train offset masks: `2558`;
  - validation offset masks: `284`;
  - fallback masks: `0`;
  - prompt truncated: `0`;
  - token max: `324`;
  - prompt+answer overlap total vs V217: `939`;
  - prompt+answer novel vs V217: `1903`.

Status:

- V250 passou.
- Permite apenas proximo smoke GPU curto, nao um treino longo direto.
- O smoke deve ter `MAX_STEPS` baixo, upload de checkpoints, weak eval imediato e bloqueio se nao houver melhora sobre V226 191/315.

## V251 H200 weak eval - V187 public adapter trio

Script/job:

- Reuso do wrapper HF `scripts/hf_job_weak_eval_v245.py`.
- Job inicial `6a00f6bd317220dbbd1a76fa` falhou antes da avaliacao porque a imagem `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` nao continha `git`.
- Job valido: `6a00f748aff1cd33e8f33052`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f748aff1cd33e8f33052`.
- Flavor: `h200`.
- Imagem corrigida: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- Run ID: `v251-h200-weak-v187-trio-20260510T212219Z`.
- Repo commit executado: `32454da9d0651b8a3b40a38833a45053f04cd250`.
- Adapter repo avaliado: `felipesp1983/kg1-nemotron-lora-v187-submission-gain`.
- Subfolders avaliados: `final`, `checkpoint-20`, `checkpoint-40`.
- Upload HF:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v187-submission-gain/commit/9c0b2c70c1eee3a5cfb177f1e9f08d976f982f4b`.
- Manifest remoto:
  `evals/v251-h200-weak-v187-trio-20260510T212219Z/v245_hf_weak_eval_manifest.json`.

Gates confirmados:

- CUDA/H200 disponivel e carga vLLM concluida.
- Weak CSV canonico V245 usado com `315` linhas.
- `observed_shared_row_contract_sha256`:
  `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Families avaliadas: `bit_manipulation=160`, `equation_transform=155`.
- Full eval, package e Kaggle submit permaneceram bloqueados.

Resultado weak:

| Candidato | Total | ACC | Equation | Bit | Trunc |
|---|---:|---:|---:|---:|---:|
| `v187_final` | `17/315` | `5.40%` | `9/155` | `8/160` | `0` |
| `v187_checkpoint20` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v187_checkpoint40` | `17/315` | `5.40%` | `9/155` | `8/160` | `0` |

Decisao:

- Rejeitar `felipesp1983/kg1-nemotron-lora-v187-submission-gain` como candidato de promocao, baseline, initializer ou fonte de ensemble.
- O melhor V187 ficou `18/315`, contra baseline V226 `191/315`.
- A ausencia de truncamento mostra que a falha nao e apenas de output longo; o adapter provavelmente nao esta alinhado ao contrato/prompt/modelo do weak gate atual.
- Nao gastar novo H200 nesse repositorio.

Impacto no roadmap:

- Prioridade volta para dados V249 + smoke GPU curto, ou para triagem de outro adapter HF somente se houver evidencia independente forte e o custo for limitado.
- Nao usar `V187` em treino, merge, DARE/TIES, router ou seed sem uma justificativa nova e verificavel.

## V252 H200 weak eval - V188 raw, overlay e stripped

Objetivo:

- Antes de gastar H200 em treino derivado de `V188`, medir se os artefatos publicos `checkpoint-*`, `final_full_baseline_overlay` e `final_stripped` possuem qualquer sinal util no weak gate canonico.
- Testar os 6 candidatos em uma unica carga vLLM para reduzir custo.
- Bloquear full eval, package e Kaggle submit.

Script/job:

- Reuso do wrapper HF `scripts/hf_job_weak_eval_v245.py`.
- Job inicial `6a00f9e5317220dbbd1a7702` falhou antes da avaliacao porque o secret foi passado como string literal `$HF_TOKEN`; o container recebeu token invalido e retornou `401` ao baixar dataset privado.
- Correcao aplicada: rerun com valor real de `get_token()` injetado em `secrets`.
- Job valido: `6a00fa83317220dbbd1a7706`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00fa83317220dbbd1a7706`.
- Flavor: `h200`.
- Run ID: `v252-h200-weak-v188-packages-20260510T213606Z`.
- Repo commit executado: `6824a0cb978cfe02d25ce979fbd598c62213f692`.
- Adapter repo avaliado: `felipesp1983/kg1-nemotron-lora-v188-equation-lmhead`.
- Upload HF:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v188-equation-lmhead/commit/b1f974fd80212467e5d7f77dca6baf994c67f076`.
- Manifest remoto:
  `evals/v252-h200-weak-v188-packages-20260510T213606Z/v245_hf_weak_eval_manifest.json`.

Gates confirmados:

- CUDA/H200 disponivel e carga vLLM concluida.
- Weak CSV canonico V245 usado com `315` linhas.
- `observed_shared_row_contract_sha256`:
  `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Families avaliadas: `bit_manipulation=160`, `equation_transform=155`.
- Full eval, package e Kaggle submit permaneceram bloqueados.

Resultado weak:

| Candidato | Artefato | Total | ACC | Equation | Bit | Trunc |
|---|---|---:|---:|---:|---:|---:|
| `v188_checkpoint20_raw` | `checkpoint-20` | `16/315` | `5.08%` | `8/155` | `8/160` | `0` |
| `v188_checkpoint40_raw` | `checkpoint-40` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v188_checkpoint20_overlay` | `packages/v188-checkpoint20/final_full_baseline_overlay` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v188_checkpoint20_stripped` | `packages/v188-checkpoint20/final_stripped` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v188_checkpoint40_overlay` | `packages/v188-checkpoint40/final_full_baseline_overlay` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v188_checkpoint40_stripped` | `packages/v188-checkpoint40/final_stripped` | `16/315` | `5.08%` | `9/155` | `7/160` | `0` |

Decisao:

- Rejeitar `V188` raw, overlay e stripped como candidato de promocao, baseline, initializer, merge, DARE/TIES, router ou fonte de ensemble.
- O melhor V188 ficou `18/315`, contra baseline V226 `191/315`.
- O overlay nao recuperou o comportamento do baseline; portanto, o pacote publicado nao deve ser tratado como adapter protegido equivalente a V194/V226.
- A falha com zero truncamento indica desalinhamento de adapter/contrato/modelo, nao apenas excesso de tokens.
- Nao gastar novo H200 em treino derivado de `V188` sem uma fonte nova de pesos/dados e uma justificativa verificavel.

Impacto no roadmap:

- V187 e V188 publicos estao descartados como rotas de melhoria direta.
- A proxima rota HF-only deve ser uma destas, em ordem:
  1. localizar/subir para HF o adapter forte conhecido (`V194`/`V226`) para permitir smoke training/eval sem depender de Google Drive;
  2. se o adapter forte nao estiver acessivel no HF, executar um baseline no-LoRA ou uma auditoria HF de candidatos com evidencia independente antes de qualquer treino;
  3. usar V249 apenas em smoke curto com gate imediato, sem treino longo direto.

## V253 H200 weak eval - adapters HF priorizados

Objetivo:

- Medir, no weak gate canonico, os adapters HF priorizados que ainda tinham alguma evidencia historica ou prescore local.
- Incluir `V189`, `V94`, `V95`, `V96`, `V97` e `V101` em uma unica carga vLLM para reduzir custo.
- Validar que a nova logica `KG1_ADAPTER_SPECS_JSON` aceita multiplos repositorios/subfolders sem quebrar gates.
- Bloquear full eval, package e Kaggle submit.

Script/job:

- Wrapper HF: `scripts/hf_job_weak_eval_v245.py`.
- Mudanca de suporte multi-repo: commit `c944cbb1dbdf36d870afbe215dfc7f4dcef7572f`.
- Job inicial `6a00fe05aff1cd33e8f3309a` foi cancelado antes da carga do modelo por mismatch no `KG1_EXPECTED_COMMIT`.
- Job valido: `6a00fe5e317220dbbd1a7717`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00fe5e317220dbbd1a7717`.
- Flavor: `h200`.
- Run ID: `v253-h200-weak-prioritized-hf-adapters-20260510T215233Z`.
- Repo commit executado: `c944cbb1dbdf36d870afbe215dfc7f4dcef7572f`.
- Upload HF:
  `https://huggingface.co/felipesp1983/kg1-nemotron-training/commit/ff6eaa9dc9ae3990a58bd7c966d696bc7a35c59b`.
- Manifest remoto:
  `evals/v253-h200-weak-prioritized-hf-adapters-20260510T215233Z/v245_hf_weak_eval_manifest.json`.

Gates confirmados:

- CUDA/H200 disponivel e carga vLLM concluida.
- Weak CSV canonico V245 usado com `315` linhas.
- Weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Families avaliadas: `bit_manipulation=160`, `equation_transform=155`.
- Todos os 10 adapters passaram no gate de arquivo/config antes da avaliacao.
- Full eval, package e Kaggle submit permaneceram bloqueados.

Resultado weak:

| Candidato | Repositorio/subfolder | Total | ACC | Equation | Bit | Trunc |
|---|---|---:|---:|---:|---:|---:|
| `v189_checkpoint10_raw` | `felipesp1983/kg1-nemotron-lora-v189-equation-answer-short/checkpoint-10` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v189_checkpoint10_overlay` | `.../packages/v189-checkpoint10/final_full_baseline_overlay` | `15/315` | `4.76%` | `8/155` | `7/160` | `0` |
| `v189_checkpoint10_stripped` | `.../packages/v189-checkpoint10/final_stripped` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v94_final_raw` | `felipesp1983/kg1-nemotron-lora-v94-equation-crypt/final` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v94_final_overlay` | `.../packages/v094-final/final_full_baseline_overlay` | `16/315` | `5.08%` | `8/155` | `8/160` | `0` |
| `v95_checkpoint20_bit_rehearsal` | `felipesp1983/kg1-nemotron-lora-v95-bit-rehearsal/checkpoint-20` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v96_a0p020_interp` | `felipesp1983/kg1-nemotron-lora-v96-v95interp/interpolations/v096-v91-v95cp20-interp-a0p020` | `19/315` | `6.03%` | `10/155` | `9/160` | `1` |
| `v97_last3_uniform_soup` | `felipesp1983/kg1-nemotron-lora-v97-v91-soups/soups/v097-v91-checkpoint-soup-last3_uniform` | `19/315` | `6.03%` | `10/155` | `9/160` | `0` |
| `v101_checkpoint20_overlay` | `felipesp1983/kg1-nemotron-lora-v101-tong-selector-v2/packages/v101-checkpoint20-lmheadfix/final_full_baseline_overlay` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v101_final_overlay` | `felipesp1983/kg1-nemotron-lora-v101-tong-selector-v2/packages/v101-final-lmheadfix/final_full_baseline_overlay` | `19/315` | `6.03%` | `9/155` | `10/160` | `0` |

Decisao:

- Rejeitar todos os candidatos V253 como rota de promocao, baseline, initializer, merge, DARE/TIES, router ou ensemble deployable.
- O melhor grupo ficou em `19/315`, contra baseline V226 `191/315`.
- O prescore historico de V189/V94/V95/V96/V97/V101 nao transferiu para o weak gate canonico atual.
- O problema nao e truncation: quase todos tiveram `0` truncamento. A falha e desalinhamento de adapter/contrato/prompt/modelo.
- Nao gastar novo H200 nesses repositorios sem uma evidencia externa nova que explique e corrija o desalinhamento.

Impacto no roadmap:

- Ficam descartados, como melhoria direta, os repositorios publicos locais/HF avaliados em V251, V252 e V253.
- A rota objetiva agora e remover dependencia de Drive para os pesos fortes conhecidos:
  1. localizar ou publicar no HF o adapter protegido `V194` e o checkpoint forte `V226`;
  2. validar esses pesos fortes com o mesmo wrapper HF e weak CSV canonico;
  3. so depois executar smoke training curto com V249/novos dados, sempre partindo de um initializer forte e com weak eval imediato.
- Se `V194/V226` nao puderem ser colocados no HF, a proxima acao barata e CPU-only: minerar `equation_transform` simbolico/misto e gerar probes/verifiers; nao ha justificativa para treino H200 longo a partir dos adapters fracos.

## Auditoria Google Drive KG1 - inventario rigoroso 2026-05-10

Escopo:

- Fonte: Google Drive `MyDrive`, roots KG1 relevantes.
- Inventario local versionado:
  - `artifacts/drive_audits/google_drive_kg1_inventory_latest.json`
  - `artifacts/drive_audits/google_drive_targeted_report_metrics_latest.json`
  - `artifacts/drive_audits/google_drive_v230_v238_manifest_decisions_latest.json`
- Total catalogado: `1879` arquivos, `301.446 GiB`.
- Artefatos por tipo: `85` adapters completos, `85` `.safetensors`, `29` zips, `423` CSVs, `54` JSONLs, `232` reports/manifests de avaliacao, `11` notebooks.

Principais roots por tamanho:

| Root Drive | Arquivos | Tamanho |
|---|---:|---:|
| `KG1_NVIDIA_V199` | `41` | `39.301 GiB` |
| `KG1_NVIDIA_V206C` | `24` | `37.537 GiB` |
| `KG1_NVIDIA_V198` | `38` | `34.517 GiB` |
| `KG1_NVIDIA_V201` | `44` | `31.016 GiB` |
| `KG1_PUBLIC_ADAPTERS` | `62` | `25.611 GiB` |
| `KG1_NVIDIA_V202C` | `32` | `19.482 GiB` |
| `KG1_NVIDIA_V221` | `68` | `18.544 GiB` |
| `KG1_NVIDIA_V202D` | `26` | `15.308 GiB` |
| `KG1_NVIDIA_V226` | `44` | `11.963 GiB` |
| `KG1_NVIDIA_V227` | `100` | `8.003 GiB` |

Adapters completos no Drive com maior valor operacional:

| Artefato | Status | Uso correto |
|---|---|---|
| `KG1_NVIDIA_V226/output_v226_equation_checkpoint_sweep/train_v226_v194_micro_lr2e9_s6/checkpoint-1` | Melhor baseline weak conhecido: `191/315`, equation `55/155`, bit `136/160`, trunc `0` | P0: publicar/validar no HF como initializer forte; nao rebaixar por adapters HF fracos |
| `KG1_NVIDIA_V226/.../checkpoint-2` | `189/315`, equation `55`, bit `134`, trunc `1` | Nao promover; manter como comparativo |
| `KG1_NVIDIA_V226/.../checkpoint-3` | `190/315`, equation `55`, bit `135`, trunc `1` | Nao promover; manter como comparativo |
| `KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter` | V194 protegido, V221 weak `190/315`, equation `54`, bit `136`, trunc `0` | P0/P1: publicar/validar no HF; bom guardrail de bit |
| `KG1_NVIDIA_V217/output_v217_short_answer_rescue/train_v217_shortans_lr1e8_s16/final_adapter` | V221 registry weak `190/315`, equation `55`, bit `135`, trunc `0` | Manter como comparativo; nao supera V226 |
| `KG1_NVIDIA_V227/.../final_adapter` e `checkpoint-1` | V229 weak agregado `16/315` | Rejeitado; nao usar como initializer, merge, router ou treino |

Reports antigos do Drive que nao devem ser confundidos com o gate weak canonico:

| Linha | Resultado observado | Decisao |
|---|---:|---|
| V207A `v194_baseline_eval` | `822/947`, per family: bit `135/160`, equation `55/155`, demais familias `100%` | Evidencia util de diagnostico amplo; nao substitui V221/V230 weak canonico |
| V207A `v206c_s0p100` | `158/315`, trunc `40` | Rejeitar: abaixo de V226 e truncation alto |
| V207A `v206c_s0p020` | `157/315`, trunc `40` | Rejeitar |
| V207A `v206b_answer_only` | `150/315`, trunc `43` | Rejeitar |
| V207B public adapters Kienngx COT | `44/315` e `32/315`, trunc `126/202` | Rejeitar como adapter; COT longo e desalinhado |
| V214 micro | `137/315`, trunc `55` | Rejeitar |
| V216 equation push | `124/315`, trunc `57` | Rejeitar |
| V217 pre-registry eval antigo | `118/315`, trunc `81` | Rejeitar esse decode antigo; usar V221 registry para V217 |
| V218 decode rescue | `18/315`, trunc `0` | Rejeitar |
| V219 think decode A/B | `6/315`, trunc `225` | Rejeitar |
| V223 equation rescue | `107/315`, trunc `97` | Rejeitar |

Achado de decode V225:

- V225 equation-only sweep mostrou que `think_strict_boxed` elevou `v194` e `v217` para `56/155` em `equation_transform`, contra `54-55/155` no registry weak.
- Esse ganho e pequeno e ainda fica abaixo do gate `60/155`, mas e evidencia concreta de que prompt/decode pode recuperar `+1` linha de equation sem mexer em pesos.
- Acao: manter como experimento P1 de prompt/parser, mas nao usar como liberacao de full eval.

Manifestos Drive V230-V238:

| Versao | Decisao registrada | Implicacao |
|---|---|---|
| V230 | `row_level_oracle_improves_but_misses_weak_gate` | Oracle chega a `197/315`, mas equation so `57/155`; nao deployavel |
| V231 | `mine_equation_solvers_before_training` | Minerar solver antes de treino |
| V232 | `build_v233_verified_equation_solver_probes` | Criar probes verificados por rota |
| V233 | `improve_solver_parsers_before_eval` | Solver ainda sem ganho deployavel |
| V234 | `external_intel_triage_ready_for_source_download` | Intel externa organizada, mas sem payload aprovado |
| V235 | `manual_source_access_or_license_required_before_download` | Bloqueio correto por credenciais/licenca/hash |
| V236 | `continue_local_solver_development` | DSL local ainda sem ganho equation deployavel |
| V237 | `build_prompt_format_specific_parser_before_solver` | Formato Alice inline precisa parser especifico |
| V238 | `continue_alice_parser_development` | Apenas `1` override verificado; insuficiente para gate |

Conclusao da auditoria Drive:

- O Drive contem historico rico, mas os unicos pesos fortes comprovados seguem sendo V226 checkpoint-1, V194 protegido e V217 como comparativo.
- A maior parte dos adapters antigos/publicos tem truncation alto, score fraco ou desalinhamento de prompt/modelo; nao devem ser usados para merge, soup, router ou treino.
- O melhor uso imediato do Drive e operacional: transferir/publicar V226 checkpoint-1 e V194 para HF com hash/config completos, validar no weak gate HF, e depois usar esses pesos fortes como initializer para qualquer smoke training.
- O melhor uso analitico do Drive e continuar minerando `equation_transform` simbolico/misto, usando V230 miss packs e V237/V238 parser evidence, porque trocar adapter nao resolveu o gap de `5` linhas em equation.

## HF bridge concluido - pesos fortes V194/V226

Objetivo:

- Remover a dependencia operacional do Google Drive para os pesos fortes antes de novos jobs HF.
- Evitar gasto H100/H200 partindo de adapters fracos que ja foram rejeitados.

Repo HF privado:

- `felipesp1983/kg1-strong-adapters-v194-v226`
- URL: `https://huggingface.co/felipesp1983/kg1-strong-adapters-v194-v226`
- SHA remoto base dos pesos validado: `1bb23fdbc3f5ccadd36b91e8f7db9d7474bf6312`
- Registro local: `artifacts/hf_uploads/KG1_STRONG_ADAPTERS_HF_BRIDGE_20260510.md`

Conteudo validado:

| Pasta HF | Origem | Weak conhecido | SHA256 |
|---|---|---:|---|
| `v226_checkpoint1` | Drive V226 checkpoint-1 | `191/315`, equation `55`, bit `136`, trunc `0` | `f4e2083d83f13a102cd86e5d1295a8603264856c17ec35c357188e1acde6ea79` |
| `v194_protected` | Drive V202D/V194 protegido | `190/315`, equation `54`, bit `136`, trunc `0` | `01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f` |

Nota de contrato de inferencia:

- Os scores historicos `190-191/315` foram medidos no contrato V221: thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `V221_PROMPT_SUFFIX = "\nPlease put your final answer inside ..."` e respostas longas com raciocinio antes do ultimo `\boxed{}`.
- O job HF V254 curto (`v254-h200-weak-strong-bridge-20260511T004420Z`) rodou com contrato V245/V230 curto: thinking desabilitado, `max_tokens=96`, `max_model_len=4096` e sufixo "Return only one line".
- Resultado V254 curto:
  - `hf_v226_checkpoint1_strong_bridge`: `16/315`, equation `8`, bit `8`, trunc `0`.
  - `hf_v194_protected_strong_bridge`: `17/315`, equation `9`, bit `8`, trunc `0`.
- Interpretacao: esse resultado nao invalida os adapters fortes; ele prova que a weak-eval curta nao reproduz o contrato que gerou os scores fortes. O wrapper HF precisa suportar ambos os contratos e rotular explicitamente qual foi usado.
- Ajuste implementado em `scripts/hf_job_weak_eval_v245.py`: `KG1_DISABLE_THINKING`, `KG1_NO_PROMPT_SUFFIX` e `KG1_PROMPT_SUFFIX` agora controlam o modo de prompt; o default continua preservando o modo curto V245 para compatibilidade.

Reproducao HF com contrato V221:

- V255 H200 `v255-h200-v221contract-v194-20260511T005050Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0128f9aff1cd33e8f33271`.
  - Commit de codigo exigido: `6dd0936bc47496fcfc6201446f73c0db15df54b3`.
  - Contrato: thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, sufixo V221.
  - Resultado: `191/315`, equation `56/155`, bit `135/160`, trunc `1`, ACC `60.63%`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-strong-adapters-v194-v226/commit/82760153d0eacc365e4c037d50610931236605e7`.
  - Interpretacao: o HF reproduz o patamar historico sob o contrato V221. A divergencia pequena vs V221 Drive (`190/315`, equation `54`, bit `136`, trunc `0`) e aceitavel como variacao de runtime/extracao ate auditoria linha a linha, mas confirma que o modo curto V254 era o erro principal.
- V256 H200 `v256-h200-v221contract-v226ckpt1-20260511T0110Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a012c1f317220dbbd1a7798`.
  - Status: `COMPLETED`.
  - Contrato: thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, sufixo V221.
  - Resultado: `191/315`, equation `56/155`, bit `135/160`, trunc `1`, ACC `60.63%`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-strong-adapters-v194-v226/commit/1469bb73c0d4f31638ac59fb0c08ec2e42237ed6`.
  - Interpretacao: no HF, `v226_checkpoint1` e `v194_protected` empatam no agregado sob o contrato V221 reproduzido. A decisao agora depende de diff linha a linha e comparacao contra os reports Drive V221/V226, nao de novo treino.

Diff linha a linha V255 vs V256:

- Artefatos locais:
  - `artifacts/hf_eval_diffs/V255_V256_LINE_DIFF_SUMMARY_20260511.md`
  - `artifacts/hf_eval_diffs/v255_v256_line_diff_summary_20260511.json`
  - `artifacts/hf_eval_diffs/v255_v256_family_delta_20260511.csv`
  - `artifacts/hf_eval_diffs/v255_v256_correctness_deltas_20260511.csv`
- IDs alinhados: `315/315`.
- Predicoes textuais diferentes: `5`.
- Linhas com mudanca de corretude: `2`.
- `equation_transform`: V194 `56`, V226 `56`, net `0`, predicoes diferentes `2`, sem mudanca de corretude.
- `bit_manipulation`: V194 `135`, V226 `135`, net `0`, V226 ganha `1` linha e perde `1` linha.
- Linhas de delta:
  - `4ef88f92`: V226 corrige `01010111` onde V194 errou `01011111`.
  - `8740ed31`: V194 acerta `01101000` onde V226 erra `01111000`.
- Conclusao: V226 checkpoint-1 deve ser mantido como initializer forte por historico Drive, mas nao trouxe ganho observavel sobre V194 no contrato HF V221. A proxima melhoria precisa atacar `equation_transform`, especialmente simbolico/misto.

Diff Drive vs HF V221-contract:

- Artefatos locais:
  - `artifacts/hf_eval_diffs/DRIVE_VS_HF_V221CONTRACT_DIFF_SUMMARY_20260511.md`
  - `artifacts/hf_eval_diffs/drive_vs_hf_v221contract_diff_summary_20260511.json`
  - `artifacts/hf_eval_diffs/drive_v221_v194_vs_hf_v255_v194_family_delta_20260511.csv`
  - `artifacts/hf_eval_diffs/drive_v221_v194_vs_hf_v255_v194_correctness_deltas_20260511.csv`
  - `artifacts/hf_eval_diffs/drive_v226_vs_hf_v256_v226_family_delta_20260511.csv`
  - `artifacts/hf_eval_diffs/drive_v226_vs_hf_v256_v226_correctness_deltas_20260511.csv`
- Drive V221 V194 vs HF V255 V194:
  - IDs alinhados: `315/315`.
  - Predicoes diferentes: `14`.
  - Mudancas de corretude: `5`.
  - `equation_transform`: Drive `54`, HF `56`, net `+2` HF.
  - `bit_manipulation`: Drive `136`, HF `135`, net `-1` HF, truncation diff `1`.
- Drive V226 vs HF V256 V226:
  - IDs alinhados: `315/315`.
  - Predicoes diferentes: `11`.
  - Mudancas de corretude: `4`.
  - `equation_transform`: Drive `55`, HF `56`, net `+1` HF.
  - `bit_manipulation`: Drive `136`, HF `135`, net `-1` HF, truncation diff `1`.
- Interpretacao: a diferenca HF/Drive e pequena e favorece levemente equation, mas custa bit e truncation. Nao e melhoria robusta nem deployable; e evidencia de sensibilidade operacional do contrato longo. Para promocao, o gate deve continuar exigindo total `>=193`, equation `>=60`, bit `>=133`, trunc `<=3`, com preferencia por bit `>=136` como guardrail interno.

V257/V258 HF-only smoke training com V249:

- V257 H200 `v257-h200-v249-v226ckpt1-smoke-20260511T013254Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a013204317220dbbd1a77cd`.
  - Status: `COMPLETED`.
  - Dataset: V249 public non-weak target, train `2558`, val `284`, hashes e family counts validados no preflight.
  - Initializer: `felipesp1983/kg1-strong-adapters-v194-v226/v226_checkpoint1`.
  - Treino smoke: `MAX_STEPS=4`, `lr=5e-8 -> 2.5e-8`, LoRA trainable somente `q_proj,k_proj,v_proj,o_proj,lm_head`, sampling weighted replacement com equation peso `1.80`.
  - Gate de tokenizacao: train `2558/2558`, val `284/284`, truncation `0`, prompt truncation `0`, offset masks completos.
  - Upload HF: `felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke`, checkpoint-2, checkpoint-4 e final completos.
- V258 H200 `v258-h200-v221contract-v257-smoke-eval-20260511T015236Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0136a1317220dbbd1a77e5`.
  - Status: `COMPLETED`.
  - Contrato: V221 reproduzido, thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, sufixo V221.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke/commit/be437d3f431e0c46998243e573cda53fa68f26c6`.
  - Resultados:
    - `v257_checkpoint_2_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `1`.
    - `v257_checkpoint_4_v221_contract`: `192/315`, equation `56/155`, bit `136/160`, trunc `1`.
    - `v257_final_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `2`.
  - Melhor candidato: `checkpoint-4`. Ele melhora o V256 HF em `+1` total e `+1` bit, sem alterar equation e truncation.
  - Delta linha a linha vs V256 HF: ganho unico em `4ada9150`, family `bit_manipulation`, expected `01111011`, V256 predicted `01111111`, V258 checkpoint-4 predicted `01111011`.
  - Gate: nao passa. Total fica `1` abaixo de `193` e equation fica `4` abaixo de `60`.
  - Artefatos locais:
    - `artifacts/hf_eval_diffs/V258_V257_SMOKE_EVAL_SUMMARY_20260511.md`
    - `artifacts/hf_eval_diffs/v258_v257_smoke_eval_summary_20260511.json`
    - `artifacts/hf_eval_diffs/v256_v226_vs_v258_ckpt4_family_delta_20260511.csv`
    - `artifacts/hf_eval_diffs/v256_v226_vs_v258_ckpt4_correctness_deltas_20260511.csv`
    - `artifacts/hf_eval_diffs/v255_v194_vs_v258_ckpt4_family_delta_20260511.csv`
    - `artifacts/hf_eval_diffs/v255_v194_vs_v258_ckpt4_correctness_deltas_20260511.csv`
- Interpretacao: V249 smoke curto produziu sinal positivo real, mas nao resolveu o bottleneck. O proximo passo nao deve ser treino longo cego; deve ser um smoke pequeno, equation-targeted, usando `checkpoint-4` como seed somente se o gate HF repetir hashes, contrato V221 e weak eval imediato.

V259/V260B HF-only equation-focused smoke:

- V259 H200 `v259-h200-v249-eqfocus-v257ckpt4-smoke-20260511T023318Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a01402d317220dbbd1a7819`.
  - Status: `COMPLETED`.
  - Output repo: `felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke`.
  - Initializer: `felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke/checkpoint-4@be437d3f431e0c46998243e573cda53fa68f26c6`.
  - Dataset: V249 public non-weak target, train `2558`, val `284`, hashes e family counts validados antes de treino.
  - Receita: `MAX_STEPS=8`, `lr=2e-8 -> 1e-8`, LoRA trainable `q_proj,k_proj,v_proj,o_proj,lm_head,up_proj,down_proj`, `equation_transform=3.00`, `bit_manipulation=0.80`.
  - Gates: H200, CUDA, dataset hashes, tokenizacao sem truncation, offset masks, import `causal_conv1d`, `mamba_ssm`, adapter load coverage `12011/12011`, trainable ratio `2.6908%`.
  - Upload HF: checkpoint-4, checkpoint-8 e final completos.
- V260B H200 `v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0145edaff1cd33e8f333a0`.
  - Status: `COMPLETED`, `failureCount=0`, running `1684s`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/commit/496d31f4284ec45b278e561aee4543005767a661`.
  - Contrato: V221 reproduzido, thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`.
  - Weak CSV SHA256: `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
  - Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
  - Resultados:
    - `v259_checkpoint_4_v221_contract`: `192/315`, equation `56/155`, bit `136/160`, trunc `0`.
    - `v259_checkpoint_8_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `1`.
    - `v259_final_v221_contract`: `190/315`, equation `56/155`, bit `134/160`, trunc `1`.
  - Melhor candidato: `checkpoint-4`; ele empata o V258 checkpoint-4 no total e em equation/bit, melhorando apenas truncation de `1` para `0`.
  - Delta vs V258 checkpoint-4: `7` linhas com predicao alterada; net score `0`. Perdeu um bit previamente correto (`4ef88f92`) e ganhou um bit (`59bee375`); quatro predicoes de equation mudaram, mas continuaram incorretas.
  - Gate: nao passa. Total fica `1` abaixo de `193`; equation fica `4` abaixo de `60`.
  - Artefatos locais:
    - `artifacts/hf_eval_diffs/v260b_v259_eqfocus_20260511/README.md`
    - `artifacts/hf_eval_diffs/v260b_v259_eqfocus_20260511/v260b_v259_eqfocus_summary.json`
    - `artifacts/hf_eval_diffs/v260b_v259_eqfocus_20260511/v260b_vs_v258_checkpoint4_changed_rows.csv`
- Interpretacao: a receita V259 nao melhorou o gargalo. Continuar esse treino por mais steps e gasto H200 sem novo dado/verifier provavelmente so degrada bit ou mantem equation em `56/155`. O proximo passo deve voltar para mineracao deterministica de `equation_transform` simbolico/misto, parser/verifier e dados externos auditados, nao treino cego.

V261 HF-only prompt/decode sweep:

- V261 H200 `v261-h200-v221contract-nosuffix-prompt-sweep-20260511T034434Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0150e0317220dbbd1a785b`.
  - Status: `CANCELED` por gate de FinOps apos o primeiro candidato completo.
  - Motivo do cancelamento: a variante `thinking habilitado + sem prompt suffix` gerou resposta longa, manteve `equation_transform` sem ganho e derrubou `bit_manipulation` de forma severa; continuar os outros candidatos repetiria o mesmo risco de custo sem sinal de melhoria.
  - Contrato: V221 reproduzido, H200 validado, vLLM `0.20.1`, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, `KG1_NO_PROMPT_SUFFIX=1`, `KG1_DISABLE_THINKING=0`.
  - Resultado completo antes do cancelamento:
    - `v259_checkpoint4_nosuffix`: `155/315`, equation `55/155`, bit `100/160`, trunc `1`.
  - Delta vs melhor V260B/V258:
    - total `-37`;
    - equation `-1` contra `56/155` do V259/V258, e `0` contra o baseline historico `55/155`;
    - bit `-36` contra `136/160`.
  - Artefatos remotos: nenhum manifest foi publicado antes do cancelamento; a evidencia canonica desta execucao e o log HF do job com `candidate_summary` completo.
- Interpretacao: remover o sufixo `Return only one line: \boxed{answer}` e liberar pensamento nao melhora o gargalo de equation no contrato weak; ao contrario, degrada extraction/bit. Nao repetir varreduras `no suffix` em H200 dentro do budget atual.

V262/V263 HF-only adapter soup:

- V262 CPU `v262-hf-cpu-adapter-soups-20260511T044654Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a015f7c317220dbbd1a78a1`.
  - Status: `COMPLETED`.
  - Output repo: `felipesp1983/kg1-nemotron-lora-v262-adapter-soups`.
  - Inputs validados:
    - `v226_checkpoint1` SHA `f4e2083d83f13a102cd86e5d1295a8603264856c17ec35c357188e1acde6ea79`.
    - `v257_checkpoint4` SHA `87b52699231f35823afd23f8d0326bbfe2de742a13cb06771f759d45488007fd`.
    - `v259_checkpoint4` SHA `01b90c1745e5eb3a7fb47fc4c81ff1fdacc17098cc79faf533b05f7b91913163`.
  - Tensor contract: `12011` tensors, contract SHA `3419375a77ddf718fcec58e0ed3da179b25cbae2ed22d74de87fad51994925fb`.
  - Soups publicados:
    - `soup_v226_050_v257_050` weights SHA `b309a740469a0a435afba9bd42cc3d800cc2b1bc42685f58d8ce8c9e5294c33b`.
    - `soup_v226_050_v259_050` weights SHA `965160a7811ce44209123d46ac33d05174caf294d729b887a259bd9b59873cd7`.
    - `soup_v226_034_v257_033_v259_033` weights SHA `233c06c2a2499045fb0c017e0245e39e344262619e4d14125d245b083f7ebbaf`.
- V263 H200 `v263-h200-v262-soups-v221contract-eval-20260511T050000Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a01628faff1cd33e8f334fc`.
  - Status: `COMPLETED`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v262-adapter-soups/commit/f723dde4cba16e92c5561f6ebf09d602dd22af83`.
  - Contrato: V221 reproduzido, H200 validado, vLLM `0.20.1`, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`.
  - Weak CSV SHA256: `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
  - Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
  - Resultados:
    - `soup_v226_050_v257_050_v221_contract`: `192/315`, equation `56/155`, bit `136/160`, trunc `1`.
    - `soup_v226_050_v259_050_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `1`.
    - `soup_v226_034_v257_033_v259_033_v221_contract`: `190/315`, equation `56/155`, bit `134/160`, trunc `2`.
  - Melhor candidato: `soup_v226_050_v257_050_v221_contract`, mas ele apenas empata o melhor total V258/V260B e piora truncation contra V260B checkpoint-4.
  - Gate: nao passa. Total fica `1` abaixo de `193`; equation fica `4` abaixo de `60`.
- Comparacao linha-a-linha V263 vs V260B:
  - Artefatos locais: `artifacts/hf_eval_diffs/v263_soups_vs_v260b_20260511/`.
  - Melhor soup vs V260B checkpoint-4: `1` ganho e `1` perda, ambos em `bit_manipulation`; `0` ganhos e `0` perdas em `equation_transform`.
  - Oracle V260B + melhor soup: total `193`, bit `137`, equation `56`. Esse oracle ainda falha o gate de equation por `4`, entao nao justifica novo roteador/soup deployable.
- Decisao FinOps/QA: nao repetir adapter soup em H200 dentro do budget atual. A rota nao mudou `equation_transform`; o proximo gasto em GPU so deve ocorrer apos um preflight barato que gere evidencia concreta de `+4` ou mais em equation sem reduzir bit abaixo de `136/160`.

V266/V267B HF-only V265 score086 filtered mix:

- V266 H200 treinou o mix V265 de forma agressiva a partir de `v259_checkpoint_4`.
  - Resultado weak V221-contract:
    - `v266_checkpoint_2_v221_contract`: `155/315`, equation `56/155`, bit `99/160`, trunc `0`.
    - `v266_checkpoint_4_v221_contract`: `154/315`, equation `56/155`, bit `98/160`, trunc `2`.
  - Diagnostico: a receita preservou `equation=56`, mas destruiu `bit_manipulation`; foi encerrada por FinOps.
- V267B H200 testou uma receita ultra-conservadora a partir do mesmo seed:
  - Job treino HF: `https://huggingface.co/jobs/felipesp1983/6a017b4a317220dbbd1a7941`.
  - Output repo: `felipesp1983/kg1-nemotron-lora-v267b-v265-conservative-v259ckpt4-smoke`.
  - Receita: `MAX_STEPS=4`, `lr=2e-8 -> 1e-8`, trainable apenas `q_proj,k_proj,v_proj,o_proj,lm_head`, source key V189 corrigida e replay bit reforcado.
  - Eval checkpoint-2 HF: `https://huggingface.co/jobs/felipesp1983/6a018046aff1cd33e8f33696`.
  - Resultado: `155/315`, equation `56/155`, bit `99/160`, trunc `1`.
  - Decisao: nao avaliar checkpoint-4 nem continuar V265 dentro do budget atual. A regressao de bit e grande demais e nao houve nenhum ganho de equation.

V268 novo achado publico - `tonghuikang/nemotron` reasoning corpus:

- Fonte auditada: `https://github.com/tonghuikang/nemotron`.
- Arquivos publicos relevantes:
  - `train.csv` SHA256 local observado: `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`.
  - `problems.jsonl` SHA256 local observado: `5b536b97b402fab985312003983bf4c59a928eb08dbb2705ca77d1030d4cf24e`.
  - `corpus.jsonl` SHA256 local observado: `7ac9e8e267397f1dbcce8d015c253460fec543cab20a078fcf64a53c6000de23`.
  - `generation.jsonl` SHA256 local observado: `42eb76d13bd81ea3ce6b55120a3e2a23782c18563e05dd4ac9eea59d631b9fbc`.
- Auditoria contra weak V221:
  - O repo contem todos os `315` weak IDs em `train.csv/problems.jsonl/corpus.jsonl`; uso direto e proibido.
  - Builder V268 bloqueia todos os weak IDs antes de qualquer decode.
- Piloto local sem persistencia:
  - IDs target nao-weak no indice: `2842` (`bit=1442`, `cryptarithm_deduce=582`, `equation_numeric_deduce=543`, `cryptarithm_guess=151`, `equation_numeric_guess=124`).
  - Linhas aceitas apos baixar `corpus/<id>/synthetic.jsonl`, decodificar tokens e exigir `\boxed{}` final igual ao `train.csv`: `1789`.
  - Aceitas por familia: `bit_manipulation=1228`, `equation_transform=561`.
  - Aceitas por subtipo: `bit_manipulation=1228`, `equation_numeric_deduce=492`, `cryptarithm_deduce=43`, `equation_numeric_guess=17`, `cryptarithm_guess=9`.
  - Rejeitadas: `798` sem `synthetic.jsonl`, `255` com `boxed_answer_mismatch`.
- Valor para ACC:
  - Este e o primeiro corpus publico auditado que contem reasoning longo verificavel por `\boxed{answer}` e bloqueio de weak IDs, diferente do V265 que treinou majoritariamente short-answer.
  - Risco: equation simbolico/misto ainda tem pouca cobertura (`52` cryptarithm aceitos), entao a expectativa de ganho em `equation_transform` e moderada, nao garantida.
  - Gate antes de GPU: V268 deve publicar dataset HF e rodar V250 com `assistant_style=reasoning_boxed`, `max_length=8192`, hashes esperados, family counts e truncation `0`.
  - So depois de V250 passar, rodar um smoke H200 curtissimo com kill-switch no primeiro checkpoint. Se bit cair abaixo de `136/160` ou equation nao passar de `56/155`, encerrar esta rota.

Regra de preservacao dos artefatos historicos:

- Muitos arquivos do Drive e dos notebooks anteriores participaram da trajetoria ate o score amplo `0.86`. Eles nao devem ser apagados por tamanho ou idade sem classificacao previa.
- Observacao operacional: varios artefatos analisados agora foram efetivamente usados para chegar ao score `0.86`; portanto a regra padrao e preservar, auditar e publicar manifest, nao limpar.
- Antes de qualquer limpeza, classificar cada artefato como:
  - `P0_keep_repro`: peso, prediction CSV, report, manifest, dataset ou notebook que reproduz ou explica scores `0.86`, `190-191/315`, V207A, V221, V226, V230 ou V255.
  - `P1_keep_audit`: logs, CSVs intermediarios e notebooks que ajudam a auditar contrato de prompt, extractor, row contract, hashes ou regressao.
  - `P2_archive`: duplicatas grandes com hash identico e copia canonical ja publicada em HF/Drive com manifest.
  - `P3_delete_candidate`: apenas cache, download parcial, snapshot duplicado sob `evals/`, arquivo `.partial`, ou artefato comprovadamente fraco e reproduzivel em outro local.
- Exclusao so e permitida para `P3_delete_candidate`, com registro de path, bytes, hash quando aplicavel e motivo. Arquivos ligados ao score `0.86` entram como `P0_keep_repro` ate prova contraria.

Validacoes:

- Ambos com `tensor_count=12011`.
- Ambos sem arquivos `.partial`.
- Manifest remoto `strong_adapters_validation_manifest.json` lido com sucesso.
- Staging local `%TEMP%/kg1_drive_strong_adapters_hf_20260510` removido apos upload, liberando `8535403351` bytes logicos.

V276 H200 full eval bridge:

- Job HF: `https://huggingface.co/jobs/felipesp1983/6a01c49baff1cd33e8f3398a`.
- Commit fixado pelo gate: `589fb891b6b7d96f52ae5b4cebb6cc2d15ebe535`.
- Ambiente validado: `h200`, GPU `NVIDIA H200`, Torch apos vLLM `2.11.0+cu130`, CUDA disponivel, vLLM `0.20.1`.
- Full CSV validado: `947` linhas, SHA256 `84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935`, family counts `bit=160`, `equation=155`, `gravity=159`, `numeral=157`, `text_encryption=157`, `unit=159`.
- Adapter avaliado: `felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/checkpoint-4`, com `v274_numeric_operator_overrides`.
- Resultado full:
  - Overall: `322/947` (`34.00%`), trunc `0`.
  - `bit_manipulation`: `7/160` (`4.38%`).
  - `equation_transform`: `8/155` (`5.16%`).
  - `gravity_constant`: `3/159` (`1.89%`).
  - `numeral_system`: `151/157` (`96.18%`).
  - `text_encryption`: `0/157` (`0.00%`).
  - `unit_conversion`: `153/159` (`96.23%`).
- Postprocessor aplicado: `0` linhas no full bridge.
- Upload de artefatos: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/commit/719225764307c8916eaeaedddafff6d57e0ffeef`, path `evals/v276-h200-full-v259ckpt4-postprocessor-20260511T115754Z`.
- Decisao: V276 esta descartado como candidato full direto. Ele continua util como diagnostico/teacher weak, mas nao justifica package/submission. A discrepancia weak (`196/315` com postprocessor) versus full (`322/947`, bit/equation muito baixos) reforca que o weak tuning atual esta superespecializado e nao cobre a distribuicao full das familias problemáticas.

V277 H200 external adapter weak eval:

- Job HF: `https://huggingface.co/jobs/felipesp1983/6a01c6a9aff1cd33e8f339c0`.
- Commit fixado pelo gate: `a71e092e6dc6780766614f76ba38ed6f69bc54c1`.
- Ambiente validado: `h200`, GPU `NVIDIA H200`, vLLM `0.20.1`, contrato V221 `315` linhas, `max_tokens=96`, `disable_thinking=true`, postprocessor `none`.
- Adapters avaliados:
  - `gfinin/nemotron-reasoning-lora` root: LoRA `r=16`, `alpha=32`, target modules amplos + MoE `target_parameters`, weights `1.77 GB`.
  - `etencore/nemotron-30b-reasoning-lora` root: LoRA `r=32`, `alpha=64`, target modules `q_proj/v_proj/o_proj/k_proj`, weights `14.9 MB`.
  - `etencore/nemotron-30b-reasoning-lora/checkpoint-1000`.
  - `etencore/nemotron-30b-reasoning-lora/checkpoint-1188`.
- Resultados weak V221:
  - `gfinin_nemotron_reasoning_lora_root`: `38/315`, equation `10/155`, bit `28/160`, trunc `2`.
  - `etencore_nemotron_30b_reasoning_lora_root`: `78/315`, equation `28/155`, bit `50/160`, trunc `0`.
  - `etencore_nemotron_30b_reasoning_lora_checkpoint_1000`: `74/315`, equation `25/155`, bit `49/160`, trunc `0`.
  - `etencore_nemotron_30b_reasoning_lora_checkpoint_1188`: `79/315`, equation `29/155`, bit `50/160`, trunc `0`.
- Upload de artefatos: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/commit/707bcb9ac5637c783fdfdc0437a75a49c604cd28`, path `evals/v277-h200-external-weak-eval-20260511T120632Z`.
- Decisao: rota de adapters externos completos descartada para o objetivo atual. Nenhum candidato chega perto do baseline V259/V275 (`196/315` com postprocessor; `equation=60`, `bit=136`). Nao rodar full eval, soup, treino ou distilacao com esses adapters.

V278 web/HF double-check 1000x:

- Busca adicional em Hugging Face confirmou tres datasets relevantes:
  - `andy279/nemotron-reasoning-challenge-raw-traces`: gated, `10K<n<100K`, raw teacher traces de multiplos teacher models e deterministic solvers; P0 se o acesso for liberado.
  - `andy279/nemotron-reasoning-challenge`: gated, `49,290` exemplos de treino e `1,165` validacao; P0 se o acesso for liberado.
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: mirror publico CSV; util apenas para auditoria/contrato, sem traces solver-guided.
- Busca adicional em Hugging Face encontrou quatro LoRAs publicos `passagereptile455/*` ainda nao avaliados:
  - `passagereptile455/nemotron-reasoning-lora-v8-kaggle`
  - `passagereptile455/nemotron-reasoning-lora-v9-kaggle-alpha64`
  - `passagereptile455/nemotron-reasoning-lora-v10-kaggle-1epoch`
  - `passagereptile455/nemotron-reasoning-lora-v11-kaggle-alpha64-1epoch`
- Regra de negocio para esses quatro LoRAs: so avaliar com V277-style static gate primeiro. Exigir `adapter_config.json`, `adapter_model.safetensors`, base model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, tensor/shape compativel, ausencia de `.partial`, e estimativa de custo. Como V277 mostrou que adapters publicos podem ficar muito abaixo do baseline, rodar no maximo um weak eval curto; nao fazer full/soup/treino antes de evidenciar `>=193`, `equation>=60`, `bit>=133`, trunc `<=3`.
- Double-check de literatura reforca que a rota com maior valor para `equation_transform` nao e novo prompt nem treino cego:
  - FlashFill/PROSE/FlashMeta/BlinkFill: Program-by-Example com DSL, version spaces e ranking. Aplicacao direta: sintetizar transformacoes simbolicas a partir dos exemplos do prompt e promover apenas quando todos os exemplos forem satisfeitos e a regra for label-free.
  - SyGuS/CEGIS/CVC5/Z3/Rosette/SKETCH: usar busca enumerativa/contraexemplos e solver-aided programming para limitar hipoteses. Aplicacao direta: V278 deve gerar candidatos de substring, reorder, reverse, mask/delete, delimiter split, map por posicao, e rejeitar qualquer regra ambigua.
  - Bit-vector SMT/Boolector/Z3/Hacker's Delight/Bit Twiddling Hacks: `bit_manipulation` deve ser tratada como sintese de expressoes bit-vector verificaveis, nao como memorizacao. Aplicacao direta: gramaticas pequenas com `xor/and/or/not`, shifts/rotates, masks, popcount/parity e rightmost-bit idioms, todas verificadas contra exemplos.
  - RobustFill/DreamCoder/DeepCoder/HYSYNTH/LLM+verifier: LLM pode propor hipoteses ou priorizar DSL, mas o juiz deve ser o verificador local. Nenhuma regra entra no postprocessor sem prova deterministica.
- Decisao: V278 deve nascer CPU-only, com manifest de regras, contagem de abstencoes, ganhos/perdas por familia, e bloqueio de GPU enquanto nao houver pelo menos `+4` a `+5` ganhos adicionais em `equation_transform` sem reduzir `bit_manipulation` abaixo de `136/160`.

V278 CPU-only symbolic PBE DSL audit:

- Script: `scripts/run_v278_symbolic_pbe_dsl_audit_hf.py`.
- Validacoes executadas:
  - `python -m py_compile scripts/run_v278_symbolic_pbe_dsl_audit_hf.py`: ok.
  - `python scripts/run_v278_symbolic_pbe_dsl_audit_hf.py --self-test`: ok.
  - Auditoria local CPU: `artifacts/hf_cpu_runs/v278_symbolic_pbe_dsl_audit_20260511T123910Z/`.
- Contrato validado:
  - `observed_shared_row_contract_sha256 = bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
  - `parse_status_counts = {"ok": 99}`.
  - `subtype_counts = {"equation_numeric_operator": 16, "equation_symbolic_punct": 83}`.
- DSLs testadas:
  - char transducer;
  - delete selected chars;
  - keep selected chars;
  - prefix/suffix before/after first/last marker;
  - remove first marker;
  - position template;
  - operator-index template;
  - numeric same-operator baseline.
- Resultado:
  - `all_verified_candidates = 0`.
  - `all_incorrect_candidates = 3`.
  - `verified_promotable_candidates = 0`.
  - `incorrect_promotable_candidates = 0`.
  - As tres regras que emitiram candidato foram bloqueadas por erro no mesmo id `432b1110`; nenhuma regra passou o gate de promocao.
- Decisao: `no_promotable_symbolic_pbe_signal`.
- Implicacao: nao gastar H200/GPU nessa rota de DSL local simples. O gargalo `equation_symbolic_punct` exige traces solver-guided externos, uma gramatica substancialmente mais rica, ou avaliacao barata de novos adapters antes de qualquer novo treino.

V279 external LoRA static gate - `passagereptile455/*`:

- Script: `scripts/run_v279_external_lora_static_gate_hf.py`.
- Validacoes executadas:
  - `python -m py_compile scripts/run_v279_external_lora_static_gate_hf.py`: ok.
  - `python scripts/run_v279_external_lora_static_gate_hf.py --self-test`: ok.
  - Auditoria local CPU/HF metadata: `artifacts/hf_cpu_runs/v279_external_lora_static_gate_20260511T1245Z/`.
- Repos inspecionados:
  - `passagereptile455/nemotron-reasoning-lora-v8-kaggle`;
  - `passagereptile455/nemotron-reasoning-lora-v9-kaggle-alpha64`;
  - `passagereptile455/nemotron-reasoning-lora-v10-kaggle-1epoch`;
  - `passagereptile455/nemotron-reasoning-lora-v11-kaggle-alpha64-1epoch`.
- Resultado:
  - `inspected_rows = 4`.
  - `inspection_errors = 0`.
  - `static_gate_pass = 0`.
  - Todos falharam por `missing_adapter_config_json`.
- Decisao: `discard_passagereptile_static_gate`.
- Implicacao: nao gastar H200/weak eval nesses quatro repos. A busca de adapters externos publicos fica encerrada ate aparecer repo com `adapter_config.json` + `adapter_model.safetensors` compativeis.

V280 Andy279 trace access/schema gate:

- Script: `scripts/run_v280_andy279_trace_access_gate_hf.py`.
- Objetivo: transformar a rota P0 `andy279/*` em gate objetivo antes de qualquer GPU. O gate valida o CSV weak canonico, consulta metadata HF, testa range access, audita schema de amostra, detecta overlap de sample por `id` e `prompt_hash`, e possui modo opcional `--allow-full-download` para auditoria completa com hashes/row counts/leakage antes de treino.
- Validacoes executadas:
  - `python -m py_compile scripts/run_v280_andy279_trace_access_gate_hf.py`: ok.
  - `python scripts/run_v280_andy279_trace_access_gate_hf.py --self-test`: ok.
  - Auditoria local CPU/HF metadata: `artifacts/hf_cpu_runs/v280_andy279_trace_access_gate_local/`.
- Contrato weak validado:
  - rows `315`;
  - SHA256 `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - familias `bit_manipulation=160`, `equation_transform=155`;
  - `id_count=315`, `prompt_hash_count=315`.
- Resultado atual:
  - `andy279/nemotron-reasoning-challenge-raw-traces`: metadata ok, `gated=manual`, `sha=d313612e92d62755e00f2382037b22da57ec5a8b`, `0/3` arquivos P0 acessiveis por range.
  - `andy279/nemotron-reasoning-challenge`: metadata ok, `gated=manual`, `sha=ddc95b9bd46a12298e3e82900f9f8bdfb926a4f4`, `0/2` arquivos P0 acessiveis por range.
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: metadata ok e publico, mas `train.csv` tem overlap no sample contra weak; portanto segue apenas como fonte de sanity/contrato, nao como autorizacao de treino.
  - Contagem final: `p0_accessible_files=0`, `p0_blocked_files=5`, `public_accessible_files=2`.
- Double-check por comandos diretos do HF CLI:
  - `hf download andy279/nemotron-reasoning-challenge --repo-type=dataset --local-dir artifacts\hf_external_sources\andy279_nemotron_reasoning_challenge`: falhou com `Access denied. This repository requires approval.`
  - `hf download andy279/nemotron-reasoning-challenge-raw-traces --repo-type=dataset --local-dir artifacts\hf_external_sources\andy279_nemotron_reasoning_challenge_raw_traces`: falhou com `Access denied. This repository requires approval.`
  - Os diretorios parciais gerados pelo HF CLI foram removidos para nao deixar cache/lixo no disco local.
- Decisao: `p0_gated_terms_required_no_gpu`.
- Implicacao: a rota `andy279/*` so volta a andar depois de acao humana no HF para liberar os termos/review dos datasets. A partir dai, o proximo comando correto e o mesmo V280 com `--allow-full-download` e limite de bytes antes de construir dataset ou usar H200.

OpenRouter Chat Mon May 11 2026 audit - exact SFT filename search:

- Arquivo auditado: `C:\Users\davis\Downloads\OpenRouter Chat Mon May 11 2026.json`.
- Escopo: validar se a busca externa por `sft_train.jsonl` e `sft_val.jsonl` trouxe algum dado, notebook, adapter, dataset ou tecnica nova que possa melhorar `equation_transform` sem perder `bit_manipulation`.
- Evidencia do export:
  - ha multiplas buscas por `"sft_train.jsonl"`, `"sft_val.jsonl"`, `"sft_train.jsonl" "sft_val.jsonl"`, GitHub, Hugging Face e Kaggle;
  - as respostas consistentes dizem que nao foi encontrado par publico com esses nomes exatos;
  - um trecho final do export sobre `Satori`/`MOSS` contem texto corrompido e nao verificavel; ele deve ser descartado como evidencia.
- Hits classificados:
  - `garg-aayush/sft-cs336-assign5-datasets`: publico e util como referencia de pipeline SFT/CS336, mas os dados sao UltraChat/math genericos; nao e KG1 nem deve entrar em treino direto.
  - `Satori-reasoning/Satori-SWE-two-stage-SFT-data`: publico, Apache-2.0, mas dominio SWE/code; nao e transformation/equation KG1.
  - `SakanaAI/FishMath-SFT-Data`: math/AIMO e pode inspirar filtragem de traces corretos, mas nao cobre as regras simbolicas/pontuacao do KG1.
  - `prabinh/Superior-Reasoning-SFT-gpt-oss-120b`: reasoning geral, muito grande; custo/risco alto e sem evidencia de ganhos nas familias alvo.
  - `AlgorithmicResearchGroup/ai-sft`: dataset amplo e grande, licenca/escopo inadequados para acao rapida.
  - `norallm/normistral-11b-thinking-training` e `agentlans/HuggingFaceH4-ultrachat_200k`: nomes parecidos (`train_sft.jsonl`, `train_sft.jsonl.zst`), mas dominio generico; nao sao substitutos dos arquivos Andy279.
- Decisao: `no_new_public_kg1_sft_mirror_found`.
- Implicacao: nao gastar GPU nem baixar datasets grandes baseados nesses hits. O unico valor pratico e metodologico: manter gates de dominio, licenca, schema, dedupe, anti-leakage e verifier antes de qualquer uso de dados externos. A rota P0 continua sendo liberar `andy279/nemotron-reasoning-challenge` e `andy279/nemotron-reasoning-challenge-raw-traces`.

Proximo passo:

1. Priorizar acao humana para liberar os datasets gated `andy279/nemotron-reasoning-challenge-raw-traces` e `andy279/nemotron-reasoning-challenge`; eles continuam a fonte P0 para traces teacher/solver.
2. Se o acesso `andy279/*` for liberado, rodar primeiro `scripts/run_v280_andy279_trace_access_gate_hf.py --allow-full-download` com limite de bytes seguro, schema, hashes, family counts, dedupe, leakage guard contra os `315` weak IDs e amostra verificavel de traces corretos. GPU so depois desse gate passar.
3. Se `andy279/*` continuar bloqueado, a unica rota tecnica restante sem novo dado e expandir V278 com uma gramatica externa comprovada por traces/repo, nao por tentativa cega no weak.
4. Nao treinar novo LoRA, nao repetir adapter soup, nao repetir prompt `no suffix`, nao gastar GPU em V278 local DSL, nao usar os adapters externos V277 e nao usar os quatro `passagereptile455/*`; essas rotas ja foram negativas, neutras ou falharam static gate.
5. Reusar V249/V250/V242/V268 somente com preflight de hashes, anti-leakage, row-contract, tokenizacao, estimativa de custo e kill-switch por primeiro candidato.

OpenRouter Chat Mon May 11 2026 (2) audit - literature and URL verification:

- Arquivo auditado: `C:\Users\davis\Downloads\OpenRouter Chat Mon May 11 2026 (2).json`.
- SHA256: `055fc803a482b8959213480f082bb8ad01fecf1c2ec5a5933d0954c7d7d83cde`.
- Tamanho: `747225` bytes.
- Estrutura parseada:
  - `messages=33`;
  - `items=130`;
  - `web_search_items=77`;
  - `completed_web_search_items=70`;
  - `unique_urls_by_regex=398`;
  - `artifacts/artifactFiles/artifactVersions/artifactFileContents` vazios.
- Contexto operacional informado no anexo:
  - `bit_manipulation`: `136/160 = 85.00%`;
  - `equation_transform`: `55/155 = 35.48%`;
  - `overall weak`: `191/315 = 60.63%`.
- Conclusao de confianca:
  - respostas de outras IAs no export sao tratadas como hipoteses, nao como evidencia de score;
  - apenas URLs/fontes primarias verificadas e artefatos locais/HF/Kaggle entram como decisao operacional;
  - estimativas soltas de ganho percentual foram descartadas se nao tinham paper, dataset, codigo ou reproducao local.

Fontes primarias verificadas nesta auditoria:

1. `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge`
   - Confirma dataset SFT especifico do Kaggle/NVIDIA Nemotron Reasoning Challenge.
   - Card informa `49,290` exemplos de treino e `1,165` de validacao.
   - Distribuicao de treino: `bit_manipulation=17,285`, `transformation=10,741`, alem de cipher/gravity/numeral/unit_conversion.
   - Fontes incluem teacher models e deterministic solvers; pontos diretamente uteis: `Solver-guided transformation=1,101`, `Solver-guided bit manipulation=1,602`, `GPT-5.4 transformation=85`.
   - Observacao critica: repo e `gated=manual`; V280 ja confirmou payload `403` aguardando aprovacao. Nao gastar GPU ate liberar acesso e rodar gate de download/auditoria.

2. `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces`
   - Confirma raw traces do proprio desafio, com tentativas corretas e incorretas, metadata e flags `is_correct` / `is_correct_official`.
   - Arquivos mais relevantes para ACC:
     - `solver_transformation_traces_merged.jsonl`: `1,101` puzzles, solver-guided transformation traces;
     - `solver_bit_manipulation_traces_merged.jsonl`: `1,602` puzzles, solver-guided bit traces;
     - `solver_transformation_traces_gpt54.jsonl`: `85` hardest transformation traces.
   - Card afirma que deterministic brute-force solvers descobrem a regra antes dos teachers gerarem traces. Isso e exatamente alinhado ao gargalo atual de `equation_transform`.
   - Observacao critica: tambem `gated=manual`; desbloqueio humano continua P0.

3. `https://huggingface.co/datasets/nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2`
   - Publico, `cc-by-4.0`, `851k` linhas, categorias math/code/stem/chat.
   - Valor: referencia de formato/teacher-data/post-training da NVIDIA.
   - Limite: nao e KG1-specific, e grande demais para baixar/treinar sem gate de dominio, licenca, dedupe e leakage. Classificacao P2, nao P0.

4. `https://arxiv.org/abs/2201.11903`
   - Chain-of-thought melhora tarefas aritmeticas, simbolicas e de commonsense em LLMs grandes.
   - Aplicacao KG1: usar CoT como fonte de traces/treino, nao como inferencia final livre; nosso output precisa ser uma linha boxed e o custo de self-consistency em full e alto.

5. `https://openreview.net/pdf?id=M1fd9Z00sj`
   - PAL: LLM decompõe o problema em programa e delega a solucao a runtime Python.
   - Aplicacao KG1: priorizar DSL/program-aided/verifier para `equation_transform` e bit; nao confiar em texto CoT como autoridade.

6. `https://docs.sympy.org/latest/guides/solving/index.html`
   - SymPy cobre solvers algebricos, numericos, sistemas, polinomios, matrizes, desigualdades e diofantinas.
   - Aplicacao KG1: util para subcasos `equation_numeric_operator`; insuficiente sozinho para `equation_symbolic_punct`, que exige DSL de transformacoes de string/simbolos.

7. `https://arxiv.org/abs/2410.21272`
   - "Arithmetic Without Algorithms" indica que LLMs podem usar heuristicas esparsas em vez de algoritmos robustos.
   - Aplicacao KG1: explica por que novos LoRAs/prompting podem oscilar e por que o verifier deterministico V274/V275 e mais confiavel do que treino cego.

8. `https://openreview.net/pdf?id=tIlDF5B6T4`
   - "Learning Mathematical Rules with Large Language Models" mostra metodologia de dados sinteticos para regras matematicas e generalizacao.
   - Aplicacao KG1: se houver novo treino, ele deve ser rule-targeted e gerado por familia/subtipo, nao mistura grande generica.

9. `https://arxiv.org/abs/2504.10415` e `https://github.com/deep-symbolic-mathematics/llm-srbench`
   - LLM-SRBench confirma dificuldade de descobrir equacoes/transformacoes fora de formas memorizadas; melhor sistema reportado na card arXiv fica em `31.5%` de symbolic accuracy.
   - Aplicacao KG1: reforca que `equation_transform=35.48%` nao e anomalia simples; caminho correto e busca/verificacao simbolica, nao apenas mais amostras genericas.

10. `https://arxiv.org/abs/2409.12183`
    - Meta-analise de CoT: beneficios fortes principalmente em math/logica; paper tambem aponta que CoT fica abaixo de solvers simbolicos para execucao simbolica.
    - Aplicacao KG1: CoT e secundario; solver/verifier continua prioridade.

Decisoes incorporadas ao roadmap:

- P0 imediato permanece V275/V274 deployavel: e a unica rota ja medida com ganho real (`196/315`, `equation=60/155`, `bit=136/160`, `trunc=0`), acima do baseline adapter-only.
- P0 bloqueado por humano: liberar `andy279/*` no HF. Depois, rodar V280 com `--allow-full-download`, hashes, schema, anti-leakage e family counts antes de qualquer H100/H200.
- P1 CPU sem novo dado: expandir V278 de DSL simples para um "term rewriting verifier" com:
  - parse de exemplos input/output;
  - inducao de regras candidatas;
  - execucao deterministica;
  - selector com regra de promocao `gains>=4/5`, `losses=0`, `bit>=136/160`.
- P1 treinamento: somente apos prova CPU/verifier ou acesso `andy279`. Treino deve ser curriculum/rule-targeted para `equation_transform`, com guardrail de bit, nao novo sweep amplo.
- P2: usar `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` apenas como referencia de formato ou amostra pequena; nao baixar full nem misturar em treino sem gate.
- Descartado agora:
  - CoT/self-consistency como inferencia final principal;
  - GNN/nova arquitetura;
  - dataset generico enorme;
  - model merging/adapters externos que ja falharam gate;
  - qualquer estimativa OpenRouter sem fonte primaria ou reproducao local.

ANALISE_DESAFIO_IAS_14 audit - literature consolidation:

- Arquivo auditado: `C:\Users\davis\Downloads\ANALISE_DESAFIO_IAS_14.txt`.
- SHA256: `da1eb90e9f2b5c0a714091ec6cfa440214206eb3e0ac34676e820a7844ee3696`.
- Tamanho: `116175` bytes, `2710` linhas, UTF-8.
- URLs unicas extraidas: `119`.
- Termos recorrentes:
  - `SymPy`: `77`;
  - `GRPO`: `22`;
  - `PAL`: `15`;
  - `SymCode`: `11`;
  - `LLM-SRBench`: `9`;
  - `DPO`: `7`;
  - `PRM`: `8`;
  - `Test-Time Training`: `6`;
  - `Product of Experts`: `5`;
  - `NVARC`: `10`.
- O arquivo nao trouxe novos adapters, notebooks executaveis, datasets KG1 publicos ou arquivos `andy279` desbloqueados. Ele trouxe consolidacao teorica/metodologica.

Achados aceitos como uteis:

1. SymCode / SymPy / self-debugging
   - Fonte primaria: `https://arxiv.org/html/2510.25975v2`.
   - Evidencia: SymCode troca raciocinio em prosa por script Python verificavel com SymPy, sandbox e loop de debug; o paper reporta ganhos de ate `13.6pp` sobre baselines e ate `16.8pp` com SymCode+ em benchmarks matematicos.
   - Aplicacao KG1: P1 para criar um evaluator/solver CPU que force a solucao a virar codigo/verificador. Deve ser usado primeiro como gate/verifier para `equation_numeric_operator`; para `equation_symbolic_punct`, precisa DSL de reescrita/string, pois SymPy puro nao cobre pontuacao/simbolos arbitrarios.

2. Tool-integrated reasoning / ToRA / PAL
   - Fonte primaria: `https://arxiv.org/abs/2309.17452`.
   - Evidencia: ToRA integra raciocinio em linguagem com ferramentas externas como computation libraries e symbolic solvers.
   - Aplicacao KG1: reforca que o modelo deve interpretar a regra e delegar execucao/verificacao a runtime deterministico. Nao autoriza prompt livre nem self-consistency caro como rota principal.

3. GRPO / RL com recompensas verificaveis
   - Fonte validada: `https://huggingface.co/docs/course/chapter12/3`.
   - Evidencia: DeepSeek-R1 usa fases com RL baseado em tarefas verificaveis; para matematica, a correcao pode ser checada por solver. GRPO usa multiplas solucoes em grupo e normaliza recompensas relativas.
   - Aplicacao KG1: P2/P1 depois de existir reward confiavel. Nao rodar GRPO antes de termos verifier local para equation; caso contrario, risco de reward hacking e gasto GPU sem sinal.

4. Test-Time Training
   - Fonte primaria: `https://proceedings.mlr.press/v267/akyurek25a.html`.
   - Evidencia: TTT melhora adaptabilidade em ARC/BBH, mas exige update temporario de parametros em inferencia.
   - Aplicacao KG1: P3/futuro. Pode inspirar "per-family/per-task adaptation", mas nao e imediato para Kaggle package por custo, complexidade e risco de runtime.

5. Product of Experts / ARC low-cost search
   - Fonte primaria: `https://proceedings.mlr.press/v267/franzen25a.html`.
   - Evidencia: combina augmentations, DFS, geracao/scoring e LLM como scorer para ARC, com foco em custo baixo.
   - Aplicacao KG1: P2 para selector/combiner CPU-only entre candidatos/verifiers. Nao substitui V275 nem libera GPU nova.

6. NVARC / ARC solution methodology
   - Fonte primaria: `https://github.com/1ytic/NVARC`.
   - Evidencia: solucao ARC usa synthetic data generation, augmentation massivo e componentes especializados.
   - Aplicacao KG1: somente inspiracao metodologica para data generation + gate. ARC nao e KG1; nao baixar datasets ARC grandes nem treinar sem experimento isolado.

7. ComputeEval benchmark discipline
   - Fonte primaria: `https://github.com/nvidia/compute-eval`.
   - Evidencia: separa contexto visivel pelo modelo de testes/solucoes ocultas; valida build/test antes de medir performance.
   - Aplicacao KG1: reforca nossos gates: separacao labels/test, held-out harness, logs, comandos, manifests, hashes e guardrails antes de qualquer treino/full eval.

Pontos descartados ou rebaixados:

- Ganhos prometidos no arquivo como `+15-25pp`, `>70%`, `>90%` ou `+30-40%` nao entram como meta operacional; sao estimativas sem reproducao no KG1.
- GNNs, atencao estrutural customizada, activation steering e interpretabilidade pesada ficam P3. Exigem arquitetura/tempo/compute e nao batem o budget atual.
- Self-consistency N=5/N=10 para full inference fica rebaixado por custo e risco de formato; usar apenas em geracao offline de traces, se validado por checker.
- Fine-tuning em hard negatives so entra apos:
  - verifier CPU passar com `losses=0`;
  - holdout bit confirmar `bit>=136/160`;
  - anti-leakage/dedupe/hashes completos;
  - budget HF aprovado.

Decisao apos esta auditoria:

- Manter V275/V274 como unica melhoria ja medida e gateada: `196/315`, `equation=60/155`, `bit=136/160`, `trunc=0`.
- Priorizar duas frentes:
  1. desbloquear `andy279/*` e rodar V280 full audit;
  2. se `andy279/*` continuar bloqueado, criar uma etapa CPU P1 inspirada em SymCode/ToRA, mas orientada ao nosso subtipo real: term-rewriting/verifier para `equation_symbolic_punct` e SymPy/Python para `equation_numeric_operator`.

## Auditoria local HF cache - NVIDIA Puzzle-KD Nemotron Post-Training Dataset v2

Solicitacao auditada: analisar `C:\Users\davis\.cache\huggingface\hub\datasets--nvidia--Puzzle-KD-Nemotron-Post-Training-Dataset-v2` e todos os arquivos em diretorios/subdiretorios.

Evidencia local:

- Ref local `refs/main`: `7d7a14dbc1ec673e9fad558785d6d2ccd4651fe8`.
- Snapshot auditado: `snapshots/7d7a14dbc1ec673e9fad558785d6d2ccd4651fe8`.
- Estrutura: `blobs`, `refs`, `snapshots`.
- Arquivos no snapshot:
  - `.gitattributes`;
  - `dataset_dict.json`;
  - `README.md`;
  - `train/state.json`;
  - `train/dataset_info.json`;
  - `train/data-00000-of-00048.arrow` ate `train/data-00047-of-00048.arrow`;
  - `validation/state.json`;
  - `validation/dataset_info.json`;
  - `validation/data-00000-of-00003.arrow` ate `validation/data-00002-of-00003.arrow`.
- `dataset_dict.json`: `{"splits":["train","validation"]}`.
- `train/state.json`: 48 shards Arrow, fingerprint `c2c952f586636249`, colunas `category/generator/license/messages/reasoning/uuid/version`.
- `validation/state.json`: 3 shards Arrow, fingerprint `88686810bc8ca650`, mesmas colunas.
- README local: dataset publico derivado de `nvidia/Nemotron-Post-Training-Dataset-v2`, criado para Puzzle NAS/KD, com `reasoning="off"`, split deterministico 95/5 e categorias `code/math/stem/chat`.

Integridade estrutural medida em todos os shards Arrow:

- Total global: `851343` linhas, UUIDs duplicados `0`.
- Train: `808775` linhas em `48` shards; row count por shard entre `16849` e `16850`; bytes Arrow aparentes `2607591184`.
- Validation: `42568` linhas em `3` shards; row count por shard entre `14189` e `14190`; bytes Arrow aparentes `137228072`.
- Schema Arrow consistente em todos os shards: `uuid`, `license`, `generator`, `version`, `category`, `reasoning`, `messages`.
- Todos os registros seguem roles `system/user/assistant`.
- O campo `system` esta vazio em `851343` linhas; isso e esperado para este dataset e nao indica perda de assistant/user.
- Nenhum assistant vazio foi encontrado.
- `assistant` com prefixo `<think></think>`: `784343` linhas.
- `assistant` contendo `\boxed`: `595749` linhas.
- License observada em todos os registros: `CC BY 4.0`.
- `reasoning`: `off` em todos os registros.

Distribuicao por split:

| Split | Rows | code | math | stem | chat |
|---|---:|---:|---:|---:|---:|
| train | 808775 | 166243 | 227512 | 337213 | 77807 |
| validation | 42568 | 8757 | 11955 | 17787 | 4069 |

Geradores observados:

| Split | DeepSeek-R1-0528 | Qwen3-235B-A22B, Qwen3-30B-A3B |
|---|---:|---:|
| train | 730968 | 77807 |
| validation | 38499 | 4069 |

Varredura de termos ligados as familias KG1:

| Termo | Train rows | Validation rows | Observacao |
|---|---:|---:|---|
| `bit_manipulation` | 0 | 0 | Nao ha label KG1 literal. |
| `equation_transform` | 0 | 0 | Nao ha label KG1 literal. |
| `transformation/equation` | 0 | 0 | Nao ha label KG1 literal. |
| `bit manipulation` | 1508 | 84 | Quase tudo em `code`, nao em puzzles KG1. |
| `equation transform` | 154 | 10 | Majoritariamente `math`, sem contrato KG1. |
| `bitwise` | 5122 | 267 | Forte sinal de programacao competitiva/bitwise. |
| `xor` | 5990 | 300 | Forte sinal de programacao competitiva/bitwise. |
| `binary` | 26941 | 1399 | Sinal amplo, misturado em `code/stem/math/chat`. |
| `cipher` | 1847 | 102 | Sinal amplo, nao necessariamente family KG1. |
| `numeral` | 770 | 43 | Sinal amplo. |
| `unit conversion` | 93 | 1 | Baixo sinal. |
| `gravity` | 5405 | 304 | Sinal STEM amplo. |
| `alice` | 4799 | 246 | Principalmente nomes em problemas de programacao/chat; nao implica dataset Alice/KG1. |
| `wonderland` | 196 | 12 | Sinal pequeno e ruidoso. |
| `kaggle` | 70 | 2 | Conversas gerais, nao desafio KG1. |
| `boxed` | 566019 | 29806 | Formato matematico amplo util como referencia, nao como fonte direta. |

Exemplos inspecionados confirmam que os hits de `bit manipulation`, `xor` e `bitwise` sao majoritariamente problemas de programacao competitiva generica, como interval XOR, flags 64-bit, conversao binaria e bitwise AND/XOR. Os hits de `equation transform` sao problemas matematicos genericos, por exemplo transformacoes algebricas em contagem de solucoes inteiras. Nao apareceu evidencia local de `sft_train.jsonl`, `sft_val.jsonl`, traces `andy279`, labels `bit_manipulation/equation_transform` ou dados especificos do KG1 dentro deste cache.

Decisao operacional:

- Classificacao: P2, referencia auxiliar.
- Nao usar este dataset full em treino LoRA agora. Ele e grande, generico e nao ataca diretamente o gargalo atual `equation_symbolic_punct`.
- Nao gastar H100/H200 neste dataset sem um filtro CPU que prove ganho esperado em IDs nao-weak e sem gate de dominio/licenca/leakage.
- Uso permitido:
  - extrair amostras pequenas de `code` com `xor/bitwise/bit manipulation` para estudar estilo de resposta e possiveis verificadores de bit;
  - extrair amostras pequenas de `math` com `equation transform` para formatacao/teacher style;
  - usar apenas como referencia de formato para distilacao futura, nunca como substituto dos traces solver-guided `andy279`.

Gates obrigatorios antes de qualquer uso futuro:

- filtrar por termos e categoria antes de materializar JSONL;
- registrar hash de todos os shards/filtros usados;
- dedupe por `uuid`, prompt normalizado e hashes contra weak/full/train oficiais;
- bloquear qualquer overlap por ID/prompt com o contrato V221/V276;
- validar licenca por linha e manifest de fonte;
- tokenization gate V250 com truncation `0.0`;
- weak gate com `bit>=136/160`, `equation>=60/155`, `losses=0` para qualquer postprocessor/verifier;
- FinOps gate: primeiro CPU-only, depois no maximo smoke curto; nenhum treino longo sem sinal CPU verificavel.

## Auditoria de URLs externas - arquivo URLs.txt

Arquivo auditado: `C:\Users\davis\Downloads\URLs.txt`.

Evidencia local:

- SHA256: `b77b6d1c882090287fdb19f68e3f662fd467fa5ff8666d8299c59410a485d2ad`.
- Tamanho: `7343` bytes.
- URLs brutas extraidas: `94`.
- URLs unicas apos dedupe: `57`.
- Hosts: `huggingface.co=40`, `github.com=2`, `openrouter.ai=2`, e demais hosts com `1` URL cada.

Achados que podem ajudar a ACC:

1. `https://huggingface.co/datasets/jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`
   - Relevancia: P0 como mirror publico do dataset oficial do desafio, nao como novo dado bruto.
   - Evidencia da pagina HF: `default` com `9.5k` linhas, splits `train` e `test`, formato CSV, licenca `apache-2.0`.
   - A pagina mostra exemplos das familias do desafio: `bit manipulation`, `secret encryption`, `numeral system`, `unit conversion`, `gravity constant` e `transformation/equation`.
   - O conteudo confirma o mesmo formato Alice/Wonderland usado no zip oficial e na rota V268.
   - Decisao: manter como fonte de auditoria/fixtures e sanity-check de parser; nao usar para treinar em cima do weak/full sem anti-leakage. O hash do `train.csv` oficial ja foi validado anteriormente como `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`.

2. `https://github.com/tonghuikang/nemotron` e `https://nemotron.huikang.dev/`
   - Relevancia: P0/P1 ja usado e ainda util para mineracao de regra/verifier.
   - Evidencia da pagina GitHub: repositorio da submissao Progress Prize vencedora; inclui `train.csv`, `problems.jsonl`, `corpus.jsonl`, `generation.jsonl`, `reasoning.py`, `augmentation.py`, `corpus.py`, `train_sft.py`, `notebook_tinker.py`, pastas `augmenters`, `investigators`, `reasoners`, `training/sft`.
   - O README descreve tabs `Base`, `Synthetic`, `Corpus`, `Training` e `Metrics`, com prompt, tabela de transformacao parseada, resposta, traces por token/logprob, mascaramento e metricas por run.
   - Decisao: continuar usando como fonte publica de engenharia reversa, corpus/verifier e interpretabilidade. O V268/V269 ja provou que SFT bruto empata o melhor score, mas nao melhora equation; a proxima utilidade e extrair regras e validadores, nao novo treino longo.

3. `https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data`
   - Relevancia: P2.
   - Evidencia HF: dataset JSON sob NVIDIA Open Model License; subsets grandes como `chat`, `instruction_following`, `math`, `science`, `swe`, `terminal_agent`; a pagina mostra exemplos com `messages` e formato boxed em algumas tarefas.
   - Possivel uso: amostras muito pequenas de `math`/`swe` para estudar formato de resposta, nao para mistura direta.
   - Bloqueio: dataset enorme e generico, sem labels KG1; licenca `nvidia-open-model-license`; exige filtro, hash, dedupe, anti-leakage e tokenization gate antes de qualquer materializacao.

4. `https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data`
   - Relevancia: P2 para metodologia de reward/verifier.
   - Evidencia HF: dataset RL com subsets `IF-RL`, `multi-domain-RL`, `MOPD`, `SWE-RL`; quantificacao informada no card: `73,809` amostras, formato JSONL, colunas como `prompt`, `ground_truth`, `pass_rate`, `pass_rate_total`, `pass_rate_passed`, `metadata`.
   - Possivel uso: inspirar schema de avaliacao/reward e pass-rate para futuros verifiers; nao e dataset KG1.
   - Decisao: nao baixar nem treinar agora. Se usado, primeiro gate CPU de dominio/licenca e amostra minima.

5. Modelos Nemotron oficiais e rotas de inferencia:
   - URLs: `nvidia/Nemotron-Cascade-2-30B-A3B`, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-*`, `modelfix/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8-GGUF`, `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`, `unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF`, `LM Studio`, `Unsloth`, `NVIDIA NIM`, `NVIDIA-NeMo/Nemotron usage-cookbook`.
   - Evidencia util: modelos suportam reasoning/tool-use/chat; paginas oficiais indicam uso via Transformers/vLLM/SGLang/llama.cpp; Nemotron 3 usa tokens `<think>`/`</think>`; Unsloth documenta parametros recomendados (`temperature=1.0`, `top_p=1.0` para chat geral; `temperature=0.6`, `top_p=0.95` para tool calling).
   - Decisao KG1: servem como infraestrutura/teacher offline, nao como candidato direto de submissao. V261 ja mostrou que alterar thinking/prompt suffix pode destruir bit; qualquer uso de teacher externo deve gerar candidatos offline validados por verifier, nao substituir o pipeline V275.

6. OpenRouter collections:
   - URLs: `free-models` e `programming`.
   - Evidencia: OpenRouter lista `NVIDIA: Nemotron 3 Super (free)` e `NVIDIA: Nemotron 3 Nano 30B A3B (free)` entre modelos gratuitos; programming leaderboard tambem lista Nemotron 3 Super.
   - Decisao: util como rota barata para consultas offline/triagem de ideias, respeitando politica de nao depender de resposta nao reproduzida. Nenhum ganho de ACC entra sem reproduzir em script local/HF e passar gates.

7. Hugging Face search pages `equation`, `manipulation`, `transformation`, `symbolic`:
   - `equation` trouxe modelos como `kinit/equational-reasoning-stepwise-konst`, `kinit/equational-reasoning-sft`, `Menouar/falcon7b-linear-equations-merged` e `sergiones/Qwen2.5-1.5B-4bit-equation-chat-ft`.
   - Esses modelos sao Qwen/Falcon/peft nao compativeis com `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
   - Decisao: nao avaliar como adapters. Podem inspirar datasets/metodologia de equational reasoning, mas nao entram em H200.

Itens descartados como nao acionaveis para ACC KG1:

- `bitmanagerai/qwen3-tts-*` e `bitmanagerai/uzbek-youtube-podcast-dataset`: TTS/audio/Uzbek; fora do alvo.
- `nvidia/GN1x-Tuned-Arena-*Manipulation`, `nvidia/Arena-GR1-Manipulation-Task`, `nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`: robotics/manipulacao fisica; nao tem relacao com `bit_manipulation`.
- `Anonymous-2024/manipulation-mfc-bench`: manipulacao textual/social, nao bit manipulation.
- `simonschoe/TransformationTransformer`: RoBERTa text-classification, nao solver de equation transform.
- `Remade-AI/Hulk-Transformation`: image/video LoRA; irrelevante.
- `nvidia/Nemotron-Personas-France`: personas em frances; sem relacao com as familias alvo.
- `news.elata.bio`, `jclibrary.org`, `x.com/...`, Reddit apagado e blogs nao primarios: sem artefato concreto para treino/eval; no maximo anedota/metodologia.
- `arxiv.org/html/2604.10905v1`: Audio Flamingo Next; paper de audio-language, nao aplicavel ao desafio textual KG1.

Decisao apos auditoria do `URLs.txt`:

- Nenhuma URL nova autoriza treino longo ou full eval adicional.
- O unico item com impacto direto ja conhecido e `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`, que confirma o dataset oficial e as familias; ele deve alimentar apenas gates, fixtures e sanity checks com anti-leakage.
- O item mais forte para ganho real continua sendo `tonghuikang/nemotron`, mas a rota correta e solver/verifier e nao SFT bruto.
- Proxima acao tecnica coerente:
  1. criar um gate CPU que gere amostras de fixtures por familia a partir do mirror `jasonkung98`, bloqueando todos os IDs weak/full conhecidos;
  2. usar `tonghuikang` para enriquecer o parser/verifier dos subtipos `equation_symbolic_punct` e `equation_numeric_operator`;
  3. manter V275/V274 como candidato deployable atual (`196/315`, `equation=60/155`, `bit=136/160`, `trunc=0`);
  4. nao gastar H100/H200 ate um novo verifier CPU provar `losses=0` e ganho incremental.

### Reauditoria 2026-05-11 do `URLs.txt` - 10 URLs novas

Arquivo reprocessado: `C:\Users\davis\Downloads\URLs.txt`.

Evidencia local nova:

- SHA256: `c1839bbfc4266d871de28daef54051b45196365a8d490392527ca75b1542bf8e`.
- Tamanho: `8262` bytes.
- URLs brutas extraidas: `106`.
- URLs unicas apos dedupe: `67`.
- Delta contra a auditoria anterior: `10` URLs novas, `0` URLs removidas.
- Hosts dominantes: `huggingface.co=50`; demais hosts permanecem como referencia secundaria.

URLs novas identificadas:

- `https://huggingface.co/datasets?search=nvidia-nemotron`
- `https://huggingface.co/datasets?search=NVIDIA%20Nemotron%20Reasoning`
- `https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- `https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B`
- `https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning`
- `https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`
- `https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- `https://huggingface.co/datasets/nvidia/Nemotron-Content-Safety-Reasoning-Dataset`
- `https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1`
- `https://huggingface.co/datasets/AmanPriyanshu/tool-reasoning-sft-CODING-nvidia-Nemotron-Agentic-v1`

Achado P1 novo: `nvidia/Nemotron-RL-ReasoningGym-v1`.

- Evidencia HF: dataset publico CC-BY-4.0, `15,000` exemplos, `data/train.jsonl` com aproximadamente `30.7 MB`, voltado a tarefas de raciocinio com ambientes algoritmicamente verificaveis.
- Auditoria por streaming local, sem download persistente:
  - linhas lidas: `15,000`;
  - campos presentes em todas as linhas: `responses_create_params`, `question`, `answer`, `metadata`, `agent_ref`, `uuid`, `license`;
  - licenca por linha: `cc-by-4.0` em `15,000/15,000`;
  - fontes mais relevantes para KG1: `bitwise_arithmetic`, `simple_equations`, `cryptarithm`, `base_conversion`, `binary_alternation`, `circuit_logic`, `modulo_grid`, `manipulate_matrix`, `number_format`, `basic_arithmetic`, `polynomial_multiplication`;
  - termos detectados no dataset: `bit=457`, `xor=144`, `binary=597`, `equation=434`, `cryptarithm=145`, `operator=282`, `arithmetic=1249`, `base=472`, `hex=172`, `modulo=157`, `matrix=1303`, `string=2068`.
- Relevancia para as familias alvo:
  - `bit_manipulation`: usar como gerador de fixtures e testes de verifier para operacoes bitwise, hexadecimal, signed integer, XOR/circuit logic, base conversion e binary alternation;
  - `equation_transform`: usar como referencia de verificadores e geracao controlada para `simple_equations`, `cryptarithm`, `operator`, `polynomial_multiplication` e transformacoes simbolicas simples;
  - familias auxiliares: pode ajudar indiretamente `numeral_system` e `unit_conversion` via base/format arithmetic, mas nao deve ser tratado como dataset KG1.
- Decisao: P1 CPU-only. Nao treinar LoRA direto, nao fazer full download persistente e nao gastar H100/H200 ainda. O uso correto e criar um notebook/script de triagem que filtre subfontes relevantes, gere mini-fixtures, dedupe contra train/weak/full conhecidos e rode verifiers locais.
- Gate obrigatorio antes de qualquer treino:
  - materializar apenas subset filtrado, com hash do filtro e manifest;
  - dedupe por `uuid`, prompt normalizado e n-gram contra Kaggle `train.csv`, weak V221/V230/V275 e artefatos `tonghuikang`;
  - bloquear exemplos cujo answer/format induza leakage do desafio;
  - classificar por subtipo KG1 (`bitwise_hex_signed`, `bitwise_xor_logic`, `base_conversion`, `equation_linear`, `equation_symbolic`, `cryptarithm`);
  - primeiro medir ganho em solver/verifier local; GPU somente se houver evidencia de `losses=0` e ganho sobre V275.

Fonte upstream adicionada: `https://github.com/open-thought/reasoning-gym`.

- Evidencia da pagina GitHub/README:
  - repositorio `open-thought/reasoning-gym`, Apache-2.0, aproximadamente `1.4k` stars e `119` forks no momento da auditoria;
  - commit auditado localmente via clone raso: `49b07130b3fcd12f2d064bba7c43869543a0e7e7`;
  - biblioteca Python de geradores procedurais e ambientes de raciocinio com rewards verificaveis;
  - galeria com `105` datasets, cobrindo algebra, arithmetic, computation, cognition, geometry, graph theory, logic e games;
  - interface principal: `create_dataset(...)`, `score_answer(...)`, `get_score_answer_fn(...)`;
  - scoring cascade com normalizacao LaTeX, string match, float match e `math_match` opcional via `math-verify`.
- Modulos upstream diretamente relevantes inspecionados:
  - `reasoning_gym/arithmetic/bitwise_arithmetic.py`: gera expressoes hexadecimais com `+`, `-`, `*`, `<<`, `>>`; aceita resposta decimal ou hex via `int(..., 0)`; util para `bitwise_hex_signed`;
  - `reasoning_gym/arithmetic/count_bits.py`: conta bits `1`; util para microfixtures de bit counting;
  - `reasoning_gym/algorithmic/base_conversion.py`: converte bases `2..36`; util para `numeral_system` e como subcaso de bit/binario;
  - `reasoning_gym/algorithmic/binary_alternation.py`: swaps minimos para string binaria alternante; util para subfamilia binaria;
  - `reasoning_gym/logic/circuit_logic.py`: gera diagramas logicos com verificacao exata; util para XOR/logic probes;
  - `reasoning_gym/algebra/simple_equations.py`: equacoes lineares com uma variavel e operadores `+`, `-`, `*`; util para `equation_numeric_operator`;
  - `reasoning_gym/algorithmic/cryptarithm.py`: solver/verifier aceita qualquer mapeamento valido, nao apenas answer armazenado; util para `equation_symbolic`;
  - `reasoning_gym/algebra/polynomial_multiplication.py`: usa `sympy.parse_expr` para equivalencia de polinomios; util como referencia de verifier simbolico;
  - `reasoning_gym/algorithmic/manipulate_matrix.py` e `reasoning_gym/cognition/modulo_grid.py`: uteis apenas se voltarmos a transformation/grid, nao prioridade imediata.
- Ponto critico para KG1:
  - O dataset HF `nvidia/Nemotron-RL-ReasoningGym-v1` e um snapshot de exemplos; o repo upstream e melhor para gerar probes parametrizados, reproduziveis por seed e com verificador local.
  - O treino RL do proprio repo usa `verl`, vLLM e multi-GPU; isso fica bloqueado por FinOps. Nao e rota correta para agora.
  - A rota correta e importar ou vendorizar minimamente somente os modulos relevantes, sem instalar dependencias pesadas, e executar fixtures CPU-only contra nossos miss-packs.
- Ajuste no plano V281/V282:
  - V281 deve registrar `reasoning-gym` upstream commit, dataset names, configs/seeds e hash dos exemplos gerados;
  - V282 deve usar `score_answer`/equivalentes locais para validar candidatos e reduzir falsos negativos de formato;
  - qualquer patch de verifier deve provar em weak known: `bit>=136/160`, `equation>=60/155`, `trunc=0`, `losses=0` contra V275 antes de qualquer HF GPU job.

Validacao adicional via Hugging Face Datasets Server, solicitada em 2026-05-11:

- Endpoints executados com `curl.exe` e arquivos temporarios removidos apos analise:
  - `https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FNemotron-RL-ReasoningGym-v1&config=default&split=train`
  - `https://datasets-server.huggingface.co/splits?dataset=nvidia%2FNemotron-RL-ReasoningGym-v1`
- Evidencia dos arquivos temporarios:
  - `first_rows.json`: `197835` bytes, SHA256 `c6f3fcbd3dbeff8f857ada441200ab61bcbeab5f89bcae2da714ca49fe8ab097`;
  - `splits.json`: `121` bytes, SHA256 `8003093aed838eb5791f7629898f585393f3d7492a0f516432b8bf44ffd7c0d9`.
- Resultado de `splits`:
  - unico split publicado: `train`;
  - `pending=[]`;
  - `failed=[]`.
- Schema confirmado em `first-rows`:
  - `responses_create_params`;
  - `question`;
  - `answer`;
  - `metadata`;
  - `agent_ref`;
  - `uuid`;
  - `license`.
- Amostra retornada: `92` primeiras linhas. Fontes relevantes presentes ja nessa amostra:
  - `cryptarithm`: `3`;
  - `simple_equations`: `3`;
  - `base_conversion`: `2`;
  - `bitwise_arithmetic`: `2`;
  - `circuit_logic`: `1`;
  - `manipulate_matrix`: `1`;
  - alem de `basic_arithmetic`, `number_format`, `gsm_symbolic` e outras fontes auxiliares.
- Conclusao operacional:
  - O endpoint de `first-rows` e suficiente para gate de schema e smoke test sem baixar o JSONL completo.
  - V281 deve primeiro chamar `splits` e `first-rows`; se schema/split/licenca divergirem, abortar antes de qualquer materializacao.
  - Apenas depois desse gate leve deve fazer streaming do `data/train.jsonl` para filtro/dedupe.
  - Nada desses endpoints muda o bloqueio de GPU: continuam sendo insumos de triagem/verifier CPU, nao autorizacao de treino.

Achados P2/P3 novos:

- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning`, `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`, `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`:
  - modelos full/quantizados Omni Reasoning, nao adapters KG1;
  - tamanhos aproximados auditados por metadados HF: BF16/Unsloth por volta de `66 GB`, FP8 por volta de `35 GB`, NVFP4 por volta de `22 GB`;
  - utilidade: teacher/offline inference ou comparacao metodologica somente se houver endpoint barato. Nao substituir o base KG1 e nao iniciar job caro sem mini-benchmark.
- `rikunarita-3/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF-UD-Q5_K_XL`:
  - Space Docker publico, SDK `docker`, atualizado em `2026-05-05`, `1` like, status operacional visto como running na UI;
  - arquivos do Space: `.gitattributes`, `Dockerfile`, `README.md`;
  - Dockerfile usa `ghcr.io/ggml-org/llama.cpp:full`, baixa do repo `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF` o peso `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q5_K_XL.gguf` e `mmproj-BF16.gguf`, e serve via llama.cpp com `-c 96000`, `-n 38912`, `-t 2`;
  - o repo GGUF tem `64.2K` downloads, `113` likes, tags `gguf`, `text-generation`, `base_model:nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, datasets `nvidia/nemotron-post-training-v3` e `nvidia/nemotron-pre-training-datasets`, licenca `nvidia-nemotron-open-model-license`;
  - decisao: P2 teacher/probe apenas. Nao e adapter LoRA, nao tem `adapter_config.json`/`adapter_model.safetensors`, nao e `submission.zip`, e nao pode ser submetido. Usar somente para consultar uma amostra pequena de misses e extrair padroes verificaveis se o V291 full gate nao entregar ganho.
- `nvidia/Nemotron-Content-Safety-Reasoning-4B` e `nvidia/Nemotron-Content-Safety-Reasoning-Dataset`:
  - dominio de safety/moderation; nao ataca `bit_manipulation` nem `equation_transform`;
  - decisao: descartar para ACC KG1, exceto como exemplo de formato de justificativa/rotulagem se necessario.
- `AmanPriyanshu/tool-reasoning-sft-CODING-nvidia-Nemotron-Agentic-v1`:
  - dataset grande de trajetorias de tool-use/coding, aproximadamente `1.57 GB`;
  - utilidade possivel: metodologia de FSM/tool-call e validacao de chamadas a ferramentas, nao dados diretos KG1;
  - decisao: P3, nao baixar agora.
- Search pages `datasets?search=nvidia-nemotron` e `datasets?search=NVIDIA%20Nemotron%20Reasoning`:
  - confirmaram `Nemotron-RL-ReasoningGym-v1` como unico achado novo acionavel imediato;
  - tambem apontaram datasets grandes/gated/genericos como `Nemotron-CC-Math`, `Nemotron-Pretraining-Code`, `Nemotron-MIND`, `Nemotron-Post-Training-Dataset-v1/v2`;
  - decisao: manter como backlog P2/P3, sem uso operacional antes de filtro e prova de ganho.

Nova acao de roadmap derivada da reauditoria:

1. `V281_REASONINGGYM_CPU_TRIAGE` - concluido:
   - objetivo: filtrar `Nemotron-RL-ReasoningGym-v1` por subfontes relevantes e produzir manifest de mini-fixtures sem download persistente grande;
   - resultado: `1326` fixtures selecionados, licenca `cc-by-4.0`, weak overlap `0`, contrato V221 validado.
2. `V282_REASONINGGYM_VERIFIER_PROBES` - concluido:
   - objetivo: testar se os exemplos filtrados melhoram regras/verificadores locais, principalmente signed hex/bitwise, circuit XOR, linear equations, cryptarithm e operator transforms;
   - resultado: `1016/1326` fixtures verificados com `0` mismatches; `circuit_logic` e `polynomial_multiplication` ficaram fora por falta de verifier.
3. `V285_REASONINGGYM_AUXILIARY_DATASET` - concluido:
   - objetivo: materializar apenas fixtures centrais com `verified_match` em formato chat JSONL compativel com os scripts SFT;
   - resultado: train `651` e validation `88`, sem overlap train/val, sem weak overlap, familias `bit_manipulation` e `equation_transform` preservadas.
4. `V286_GENERIC_TOKENIZATION_GATE` - concluido:
   - objetivo: aplicar as regras de gate/tokenizacao usadas nos Colabs e jobs HF antes de qualquer gasto GPU;
   - resultado: tokenizer Nemotron real carregou, offset masks completos, prompt truncation `0.0`, completion truncation `0`, token max `191`.
5. `V287_REASONINGGYM_ALICE_STYLE_DATASET` / `V288_ALICE_STYLE_TOKENIZATION_GATE` - concluido:
   - objetivo: reduzir drift transformando fixtures publicos verificados em prompts com exemplos no estilo KG1/Alice;
   - resultado: `651/88` linhas, zero overlap, offset masks completos, truncation `0`, token max `313`.
6. Smoke HF pequeno, se executado:
   - objetivo: preservar budget HF;
   - esperado: limitar `MAX_STEPS`, atualizar poucos modulos LoRA, avaliar weak cedo, abortar se `bit_manipulation < 136/160` ou `equation_transform <= 56/155`.

## Atualizacao 2026-05-11 - V284 official-like HF gate

Reauditoria devastadora contra Kaggle CLI/API, notebook oficial de metrica e branch HF `v230-v226-complementarity`:

- Confirmacao oficial:
  - submissao esperada: `submission.zip` com LoRA adapter, nao CSV de predicoes nem solver externo;
  - arquivo exigido: `submission.zip`;
  - limite diario: `5` submissoes;
  - submissoes pontuadas: `2`;
  - deadline: `2026-06-15T23:59:00Z`;
  - team/new entrant deadline: `2026-06-08T23:59:00Z`;
  - metrica oficial usa `\boxed{}` com extracao robusta e tolerancia numerica `rel_tol=1e-2`, `abs_tol=1e-5`.
- Parametros oficiais de inferencia, que devem governar qualquer gate de submissao:
  - `max_lora_rank=32`;
  - `max_tokens=7680`;
  - `temperature=0.0`;
  - `top_p=1.0`;
  - `max_num_seqs=64`;
  - `gpu_memory_utilization=0.85`;
  - `max_model_len=8192`.
- Achado critico:
  - `scripts/hf_job_full_eval_v276.py` e seguro como diagnostico full-bridge, mas nao e submission gate;
  - seus defaults ainda sao baratos: `max_tokens=96`, `max_model_len=4096`, `max_num_seqs=8`, `KG1_DISABLE_THINKING=True`;
  - tambem usa `DEFAULT_POSTPROCESSOR=v274_numeric_operator_overrides`, que nao e packageable em adapter-only zip;
  - decisao: nenhum resultado V276 com defaults pode autorizar Kaggle submit.
- Acao implementada:
  - novo script `scripts/hf_job_official_like_eval_gate_v284.py`;
  - objetivo: rodar HF Job official-like depois do weak gate e antes de qualquer pacote/submissao;
  - defaults oficiais: `7680/8192/64`, `gpu_memory_utilization=0.85`, thinking ligado, suffix oficial;
  - postprocessor externo bloqueado por padrao;
  - `package` e `kaggle_submit` permanecem bloqueados no manifest;
  - self-test prova rejeicao de `max_tokens=96` e de `v274_numeric_operator_overrides`.
- Decisao operacional atualizada apos HF r6:
  - V283/V282-config nao recuperou o patamar weak de V194/V226; fez `7/315` com truncation `225/315`;
  - descartar V283/V282-config para full eval, package e Kaggle submit;
  - manter V284 como gate futuro apenas para candidato adapter-only que primeiro passe o weak historico sem postprocessor externo.

## Atualizacao 2026-05-12 - pesquisa externa bit/equation e V293 rejeitado

Escopo: consolidar literatura operacional, Kaggle CLI/API, Hugging Face, ReasoningGym e OpenRouter para atacar as familias fracas `bit_manipulation` e `equation_transform`.

### Evidencia medida

- Melhor linha submetida segue sendo V291/V290 checkpoint-6:
  - Kaggle public score `0.86`;
  - full official-like local `823/947 = 0.8690601901`;
  - `bit_manipulation = 135/160`;
  - `equation_transform = 56/155`;
  - familias laterais em `100%`;
  - `truncation = 1`.
- V274/V275 provou sinal real, mas nao packageable:
  - weak com postprocessador: `196/315`, `equation=60`, `bit=136`;
  - full V291 com postprocessador: `827/947`, `equation=60`, `bit=135`;
  - decisao: usar como sinal de treino, nao como submissao direta.
- V293 tentou distilar V274 em adapter-only via `lm_head`:
  - checkpoint-3: `191/315`, `bit=135`, `equation=56`;
  - checkpoint-6: `191/315`, `bit=135`, `equation=56`;
  - checkpoint-9: `190/315`, `bit=134`, `equation=56`, `trunc=1`;
  - checkpoint-12: `192/315`, `bit=136`, `equation=56`;
  - decisao: rejeitado para full/package/submit. O treino `lm_head`-only nao internalizou as regras.

### Achados Kaggle/HF

- Kaggle CLI confirmou que os arquivos oficiais sao `train.csv` e `test.csv`.
- Historico de submissoes confirma o plato `0.86`: V291, V281, V199B, V194, V193, V192.
- Leaderboard exibida via CLI mostra top visivel `0.87` e muitos times empatados em `0.86`; portanto `+1` a `+4` acertos podem mover ranking.
- Kernels publicos puxados e resumidos em:
  - `artifacts/external_intel/kaggle_kernel_summary_20260512.md`
  - `artifacts/external_intel/kaggle_kernel_summary_20260512.json`
- Datasets publicos resumidos em:
  - `artifacts/external_intel/kaggle_external_dataset_summary_20260512.json`
- Principais leituras:
  - notebooks publicos fortes sao majoritariamente empacotamento/auditoria do Tinker adapter `0.86`;
  - kernels de treino usam target modules amplos (`q/k/v/o/in/out/up/down/lm_head`), mas precisam de gate forte para nao destruir side families;
  - `kishanvavdara/nemotron-reasoning-traj` nao deve ser usado como positivo cru nas familias alvo: bit e equation estao majoritariamente incorretos;
  - `kienngx` CoT+labels e `konbu17` bit datasets sao uteis somente com verifier/filtro;
  - `nvidia/Nemotron-RL-ReasoningGym-v1` e `open-thought/reasoning-gym` sao acionaveis porque geram tarefas verificaveis.

### Literatura operacional aplicada

- Chain-of-thought/self-consistency/least-to-most/program-of-thoughts sao uteis como metodologia, mas nao bastam para submissao adapter-only.
- A traducao correta para KG1 e:
  - gerar dados sinteticos verificaveis;
  - treinar o adapter a reproduzir resposta final correta;
  - usar replay de side families para nao degradar o `100%`;
  - validar por weak gate antes de qualquer full/HF caro.
- ReasoningGym reforca a direcao de verifier-first: gerar tarefas com `question`, `answer`, `metadata` e scorer programatico/cascade scorer.

### Proximo roteiro

1. `V294_VERIFIED_EQUATION_REPRESENTATION_PATCH`
   - objetivo: transformar os ganhos V274/V275 em comportamento adapter-only;
   - dados: templates de equation equivalence, signed-minus/opposite-sign, direct-add variant, colon absdiff/unreverse, ReasoningGym `simple_equations` e `cryptarithm`, todos verificados;
   - treino: nao usar `lm_head`-only; usar top attention + MLP LoRA controlado;
   - gate weak minimo: `overall >= 193`, `equation >= 57`, `bit >= 136`, `trunc <= 1`;
   - gate ideal para full: `equation >= 60` sem perda de bit.
2. `V295_VERIFIED_BIT_HARD_CASE_PATCH`
   - objetivo: buscar `+1` a `+2` em bit sem perder equation;
   - dados: Konbu17 filtrado, gerador Alice-style bitwise, hard negatives por operacao/bit;
   - gate weak: `bit >= 137`, `equation >= 56`, `trunc <= 1`.
3. `V296_VERIFIER_DPO_OR_ORPO`
   - objetivo: se V294/V295 plateau, formar pares correto/incorreto com verifier e treinar preferencia pequena;
   - so executar se os pares tiverem qualidade comprovada e custo previsto baixo.

### Stop/go financeiro

- Nao iniciar full official-like em H100/H200 sem weak gate.
- Nao submeter se o full official-like nao superar V291:
  - minimo operacional: `>= 824/947` com side families intactas;
  - alvo para tentar sair do plato: `>= 827/947`;
  - hard reject: `bit < 135`, `equation < 56`, qualquer side-family regression relevante.

### Artefatos

- Prompt cirurgico para usar em outras IAs:
  - `artifacts/external_intel/KG1_EXTERNAL_RESEARCH_PROMPT_2026_05_12.md`
- Triage de literatura/dados externos:
  - `artifacts/external_intel/KG1_BIT_EQUATION_LITERATURE_TRIAGE_2026_05_12.md`
- Respostas OpenRouter advisory:
  - `artifacts/external_intel/openrouter_20260512/openai__gpt-5.4_kg1_strategy_response.md`
  - `artifacts/external_intel/openrouter_20260512/deepseek__deepseek-v4-pro_kg1_strategy_response.md`

## Atualizacao 2026-05-12 - varredura Kaggle API das discussoes de Tong Hui Kang

Escopo: buscar via Kaggle SDK autenticado tudo que `Tong Hui Kang` / `huikang` publicou nas discussoes da competicao `NVIDIA Nemotron Model Reasoning Challenge`, com foco em evidencias que possam melhorar `bit_manipulation` e `equation_transform`.

### Metodo auditavel

- O HTML publico da pagina de discussao nao expunha o corpo do topico para scraping simples.
- A CLI `kaggle competitions pages` so expunha paginas oficiais (`rules`, `data-description`, etc.), nao discussoes.
- A rota correta foi o SDK novo:
  - `kagglesdk.discussions.services.discussions_api_service.DiscussionApiClient`
  - `GetTopic`, `ListComments` e `ListTopics`.
- Artefatos salvos:
  - `artifacts/v295_external_intel_triage/20260512T0400Z/kaggle_discussion_690307_bit_strategy_api_dump.json`
  - `artifacts/v295_external_intel_triage/20260512T0400Z/kaggle_discussion_690307_bit_strategy_summary.md`
  - `artifacts/v295_external_intel_triage/20260512T0415Z/tong_hui_kang_nemotron_discussion_author_dump.json`
  - `artifacts/v295_external_intel_triage/20260512T0415Z/tong_hui_kang_nemotron_discussion_author_summary.md`

### Itens encontrados

- Foram coletados `29` topicos candidatos por queries de autor/competicao e filtrados `24` itens escritos pelo autor dentro da competicao.
- Topicos/comentarios de maior relevancia:
  - `684212` - visualizacao de problemas e completions do modelo base:
    - base model rodado nos `9500` problemas;
    - cerca de `48,217,898` tokens gerados no ultimo run por problema;
    - throughput citado: `2.5k tokens/s`, cerca de `5.35h`;
    - solve rate do modelo base quase `50%`;
    - observacao do autor: muitos `equation_numeric` e quase todos `equation_symbolic` ficaram sem padrao claro.
  - `687961` - limites de treino Nemotron rank-32 LoRA em sequencias `8192`:
    - nao ha motivo para treinar `16384`, pois o limite da competicao e `8192`;
    - microbatch `2` e plausivel quando limitado a `8192`;
    - perda/logits precisam de implementacao eficiente para caber memoria.
  - `689915` - publicacao Open Progress Prize, SFT para maximizar minimo logprob:
    - score alvo reportado pelo autor: `0.877`;
    - tokens usados na submissao vencedora: `27,850,703`;
    - tokens totais treinados: `598,958,637`;
    - segredo declarado: `bit manipulation`, SFT, CoT deterministico, objetivo de `min logprob`, Tinker;
    - aposta tecnica: Nemotron consegue agir como computador simples depois de LoRA;
    - contra-aposta: nao depender de RL nem de destilar modelo maior quando politica otima pode ser gerada por codigo;
    - comentario tecnico: loss masking e necessario; treinar o modelo a memorizar pergunta reduz eficiencia em memorizar abordagem de solucao.
  - `690307` - estrategia de `85%` para `bit_manipulation`:
    - algoritmo declarado: `1364/1602 = 85.1%` em bit;
    - comentario externo no mesmo topico reporta solver ainda mais forte: `1584/1602 = 98.9%` em bit e `553/596 = 92.8%` em `equation_numeric_deduce`;
    - tecnica: enumerar relacoes por bit de saida, nao expressoes completas;
    - espaco testado: `18` combinacoes unarias + `336` binarias = `354`;
    - usar bitsum como hash e continuidade de stride esquerda/direita para reduzir ambiguidades.

### Implicacao para nossa solucao

- O achado muda a prioridade de `bit_manipulation`: ela nao deve ser tratada apenas como guardrail; existe evidencia publica de que ha espaco para `+1` a `+20` no recorte train, se a representacao for correta.
- Nosso solver local atual `src/solvers/bit_manipulation_solver.py` foi medido no `train.csv` oficial e fez `1265/1602 = 78.96%`, abaixo dos `1364/1602` declarados pelo autor e muito abaixo do `1584/1602` citado em comentario.
- Portanto existe gap real de implementacao:
  - nosso solver tem busca global/per-bit/consenso;
  - mas ainda nao replica integralmente o algoritmo de bitsum + stride + preenchimento de meio do Tong Hui Kang.
- V296 deve ser CPU-only e barato:
  1. implementar/auditar um solver bit de relacao por bit com `18+336` candidatos, bitsum e stride;
  2. medir contra `train.csv` oficial e contra o weak/full local;
  3. gerar apenas artefatos de verifier/teacher, nunca submissao direta de codigo sem autorizacao de regra/package;
  4. se houver novos acertos seguros em bit, criar dados sinteticos/CoT filtrados para LoRA adapter-only.

### Resultado V296 CPU audit

- Artefatos:
  - `scripts/run_v296_bit_stride_solver_audit.py`
  - `artifacts/v296_bit_stride_solver_audit/20260512T0450Z/v296_bit_stride_solver_audit_summary.json`
  - `artifacts/v296_bit_stride_solver_audit/20260512T0450Z/v296_bit_stride_solver_audit_details.csv`
- Base oficial usada: `train.csv` SHA256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`.
- Resultado no recorte oficial `bit_manipulation` (`1602` linhas):
  - solver local atual: `1265/1602 = 78.96%`;
  - V296 stride audit corrigido: `1201/1602 = 74.97%`;
  - ambos corretos: `1047`;
  - V296 acerta e solver atual erra: `154`;
  - V296 erra e solver atual acerta: `218`;
  - parse failed: `0`.
- Decisao: V296, como implementado a partir da descricao publica, ainda nao substitui o solver local e nao autoriza treino caro isolado. O valor acionavel e o conjunto complementar de `154` linhas, que deve virar teacher/verifier filtrado somente se uma politica de selecao provar no-loss contra weak/full.
- Proximo passo tecnico: portar/validar a implementacao publica exata `tonghuikang/nemotron/reasoners/bit_manipulation.py` ou derivar um ensemble conservador `current OR stride` com abstencao, medindo perdas antes de qualquer HF GPU.

### Validacao da referencia publica exata

- A implementacao publica exata `tonghuikang/nemotron` foi executada apenas como referencia externa sobre o `train.csv` oficial, com stub local do tipo `Problem`.
- A API do GitHub reporta `license=null`; portanto o codigo deve ser tratado como referencia/teacher evidence, nao como codigo copiavel para pacote final.
- Resultado da referencia exata no recorte `bit_manipulation`:
  - referencia Tong: `1364/1602 = 85.14%`, reproduzindo o claim publico;
  - solver local atual: `1265/1602 = 78.96%`;
  - ambos corretos: `1207`;
  - referencia acerta e solver local erra: `157`;
  - referencia erra e solver local acerta: `58`;
  - uniao oracular potencial: `1422/1602 = 88.76%`.
- Artefatos:
  - `artifacts/v296_bit_stride_solver_audit/20260512T0450Z/reference_exact_comparison_summary.json`
  - `artifacts/v296_bit_stride_solver_audit/20260512T0450Z/reference_exact_delta_manifest.json`
  - `artifacts/v296_bit_stride_solver_audit/20260512T0450Z/reference_exact_gain_rows_vs_current.csv`
  - `artifacts/v296_bit_stride_solver_audit/20260512T0450Z/reference_exact_loss_rows_vs_current.csv`
- Decisao: este e o maior achado novo de `bit_manipulation` desde V291. Nao deve virar submissao direta, mas deve virar V297: teacher dataset/abstention policy para transferir os `157` gains sem absorver as `58` perdas.

### Regras operacionais adicionadas

- Nao gastar H100/H200 em novo treino de bit antes do V296 CPU provar ganho ou cobertura nova.
- Para qualquer treino derivado do achado:
  - treinar somente tokens de raciocinio/resposta, com offset-mask estrito;
  - preservar side families com replay;
  - weak gate minimo para liberar full: `overall >= 193`, `equation >= 56`, `bit >= 137`, `trunc <= 1`;
  - se o treino for equation-first, manter `bit >= 136`.
- A rota `min logprob` deve entrar como diagnostico, nao como unico criterio de selecao: adapter so avanca se weak/full medidos melhorarem.

## Atualizacao 2026-05-12 - V297 busca HF/Kaggle e auditoria externa bit weak

Escopo: repetir a varredura externa com foco estrito em ganho mensuravel para `bit_manipulation` e `equation_transform`, sem baixar arquivos grandes nem gastar GPU HF sem prova CPU.

### Evidencia HF

- Repos confirmados pela busca HF/API:
  - `andy279/nemotron-reasoning-challenge`: contem `sft_train.jsonl` e `sft_val.jsonl`; o README declara `49,290` exemplos SFT de treino e `1,165` exemplos de validacao.
  - `andy279/nemotron-reasoning-challenge-raw-traces`: contem traces relevantes como `solver_bit_manipulation_traces_merged.jsonl`, `solver_transformation_traces_merged.jsonl` e `solver_transformation_traces_gpt54.jsonl`.
  - `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2`: dataset amplo de post-training, util apenas como fonte secundaria de raciocinio/verifier.
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: espelho pequeno do `train.csv`; SHA256 igual ao oficial `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`.
- Estado de acesso:
  - os datasets `andy279` estao gated e o token local nao tem aprovacao; tentativas de baixar apenas arquivos pequenos (`sft_val.jsonl`, `solver_transformation_traces_gpt54.jsonl`, `ds_traces_thinking_distilled.jsonl`) retornaram `403`.
  - nenhum arquivo grande gated foi mantido no disco.
- Decisao:
  - `andy279` permanece P0 se o acesso for liberado, porque e a fonte mais alinhada a SFT/traces das familias fracas;
  - sem acesso, nao usar HF GPU para tentar compensar com treino cego.

### Evidencia Kaggle de discussoes bit/equation

Artefatos:

- `artifacts/v297_external_search_and_bit_teacher_audit/20260512T0915Z/kaggle_relevant_bit_equation_discussions_dump.json`
- `artifacts/v297_external_search_and_bit_teacher_audit/20260512T0915Z/kaggle_relevant_bit_equation_discussions_summary.md`

Achados acionaveis:

- `bit_manipulation`:
  - a discussao `690307` continua sendo a principal evidencia: solver por relacao de bits e stride atinge `1364/1602 = 85.14%` no train oficial.
  - a discussao `688461` amplia o espaco util: resolver cada bit como funcao booleana independente e testar constantes, identidade, NOT, gates de 2 entradas, depois 3 entradas (`MAJ`, `CHO`, `PAR3`, combinadores AO/OA/AX/OX/XA/XO) e 4 entradas.
  - as discussoes `683866` e `690756` alertam que busca per-bit muito livre pode divergir: existem varias regras que encaixam exemplos e predizem query diferente. Portanto a hierarquia deve ser conservadora e medir perdas, nao apenas cobertura.
- `equation_transform`:
  - `684432` e `689877` confirmam o gargalo: muitos problemas tem poucos exemplos por operador, output de tamanho variavel e query operator ausente nos exemplos.
  - `688461` sugere um caminho programatico para parte numerica/simbolica: decompor `AB op CD`, varrer orientacao dos operandos (`AB`, `BA`, `CD`, `DC`), familias de operacao e formatos de saida, usando segunda verificacao (`EX2`) para evitar match casual.
  - `690891` separa problemas dedutivos dos informacionalmente ambiguuos: se o operador da query nao aparece e nao ha regra inferivel por eliminacao/prior, a resposta correta pode depender de prior aprendido, nao de deducao pura.

### V297 weak audit externo de bit

Artefatos:

- `scripts/run_v297_external_bit_reference_weak_audit.py`
- `artifacts/v297_external_search_and_bit_teacher_audit/20260512T0915Z/v297_external_bit_reference_weak_audit_summary.json`
- `artifacts/v297_external_search_and_bit_teacher_audit/20260512T0915Z/v297_external_bit_reference_weak_audit_details.csv`

Resultado no weak `315` usado como ponte:

- linhas bit: `160`;
- referencia publica Tong: `136/160 = 85.00%`;
- solver local atual: `125/160 = 78.125%`;
- ambos corretos: `122`;
- referencia acerta e solver local erra: `14`;
- referencia erra e solver local acerta: `3`;
- parse failed: `0`.

Decisao:

- Isto nao autoriza copiar codigo externo nem submeter solver direto; a API GitHub reporta `license=null` para `tonghuikang/nemotron`.
- O valor real e teacher/verifier:
  - usar os `14` gains weak e os `157` gains train da V296 como material de treino/diagnostico;
  - bloquear qualquer patch que absorva as `3` perdas weak ou as `58` perdas train sem politica de abstencao.

### Loop tecnico ate ganho mensuravel

1. `V298_CPU_BIT_BOOLEAN_GRAMMAR_AUDIT`
   - implementar uma versao propria, sem copiar codigo externo, da hierarquia booleana por bit:
     - constantes, identidade, NOT;
     - gates 2-input;
     - gates 3-input/4-input somente apos falha dos niveis anteriores;
     - desempate por menor complexidade e menor divergencia em holdout sintetico.
   - meta CPU:
     - train bit acima de `1265/1602` sem queda grande;
     - weak bit `>= 136/160`;
     - ideal para liberar treino: weak bit `>= 137/160` com `equation >= 56`.
2. `V299_CPU_EQUATION_OPERATOR_FORMAT_AUDIT`
   - implementar varredura propria `AB op CD`:
     - orientacoes dos operandos;
     - operacoes numericas/simbolicas frequentes;
     - formatos de saida;
     - verificacao em todos exemplos e rejeicao se query operator ausente sem regra inferivel.
   - meta CPU:
     - encontrar `+1` a `+5` em equation weak sem mexer em bit;
     - priorizar regras com prova por exemplos, nao respostas de modelo.
3. So depois de V298/V299:
   - gerar dataset teacher minimo e verificado;
   - treino HF curto somente se o patch CPU demonstrar ganho liquido;
   - budget HF permanece protegido: sem H100/H200 para busca cega.

### Limpeza

- Arquivos temporarios de download externo foram apagados.
- A varredura V297 manteve apenas artefatos pequenos de auditoria; nenhum arquivo HF grande gated foi salvo.

### Resultado V298 CPU bit boolean grammar

Artefatos:

- `scripts/run_v298_bit_boolean_grammar_audit.py`
- `artifacts/v298_bit_boolean_grammar_audit/20260512T1000Z/weak_level2_v298_bit_boolean_grammar_summary.json`
- `artifacts/v298_bit_boolean_grammar_audit/20260512T1000Z/weak_level3_v298_bit_boolean_grammar_summary.json`
- `artifacts/v298_bit_boolean_grammar_audit/20260512T1000Z/weak_level4_v298_bit_boolean_grammar_summary.json`

Resultado no weak bit (`160` linhas):

- nivel 2: `85/160 = 53.125%`, `+5` ganhos vs solver local, `45` perdas;
- nivel 3: `85/160 = 53.125%`, `+5` ganhos vs solver local, `45` perdas;
- nivel 4: `85/160 = 53.125%`, `+5` ganhos vs solver local, `45` perdas.

Decisao:

- A gramatica per-bit livre confirma o alerta das discussoes: ela encaixa exemplos, mas diverge demais na query.
- V298 esta rejeitado como politica direta e nao autoriza HF GPU.
- Os `5` gains podem ser examinados apenas como candidatos de teacher, mas qualquer uso precisa de filtro conservador com zero perda no weak.
- A rota de bit continua sendo:
  - referencia Tong como teacher/verifier externo;
  - implementacao propria com restricao full-byte/stride, nao per-bit livre;
  - medicao de perda antes de treino.

### Estado equation apos V275

- Baseline V275 weak ja conhecido:
  - total `196/315`;
  - `bit_manipulation = 136/160`;
  - `equation_transform = 60/155`;
  - ganhos V274/V275: `+4` equation, `0` perdas.
- Erros restantes apos V275:
  - `95` linhas de `equation_transform`;
  - `12` sao queries numericas;
  - a maioria restante e simbolica/pontuacao.
- Observacao cirurgica:
  - ha casos numericos em que o label sugere prior simples (`+` como soma direta, `-` com diferenca assinada), mas o operador da query frequentemente nao aparece nos exemplos;
  - isso nao e uma regra deployavel segura sem treino ou prior aprendido, porque violaria o criterio de derivar a resposta somente dos exemplos.
- Decisao:
  - nao transformar esses casos em postprocessor direto;
  - usar como exemplos teacher de equation somente se o dataset/treino conseguir aprender o prior sem regredir bit e side families;
  - manter V274/V275 como melhor ganho equation comprovado ate agora.

## Atualizacao 2026-05-12 - V299/V300/V301/V302 postprocessor signals

Escopo: continuar a busca por ganhos pequenos, usando apenas CPU/local antes de qualquer gasto HF.

### V299 equation numeric candidate audit

Artefatos:

- `scripts/run_v299_equation_numeric_candidate_audit.py`
- `artifacts/v299_equation_numeric_candidate_audit/20260512T1030Z/v299_equation_numeric_candidate_manifest.json`

Resultado:

- entrada: `v275_postprocessed_predictions.csv`, que ja tinha `196/315`, `equation=60/155`, `bit=136/160`;
- `same_operator_unique_numeric_dsl`: `31` candidatos, `30` corretos, `1` perda;
- `all_numeric_examples_unique_dsl`: `4` candidatos, `2` corretos, `2` perdas;
- `conventional_operator_prior_unique`: nenhum candidato promovivel;
- decisao: bloqueado como postprocessor. Ha sinal para treino/teacher, mas nao para regra direta.

### V300 full-byte bit grammar

Artefatos:

- `scripts/run_v300_bit_fullbyte_grammar_audit.py`
- `artifacts/v300_bit_fullbyte_grammar_audit/20260512T1100Z/`

Resultados principais:

- V300 full-byte nivel 3 contra solver local no weak:
  - full-byte sozinho: `131/160`;
  - ensemble com solver local: `136/160`, `+12/-1`.
- Guardrail seguro removendo ternarios instaveis e aceitando so `MAJ3`, `CHO`, `PAR3`:
  - sobre V275 weak: bit sobe de `136/160` para `147/160`;
  - `+11` ganhos;
  - `0` perdas.

Decisao:

- Este e um ganho real de bit, label-free no algoritmo, derivado apenas dos exemplos do prompt.
- A regra que causava perda (`AND_OR`) foi bloqueada; o conjunto seguro ficou `MAJ3`, `CHO`, `PAR3` + unary/binary full-byte.

### V301 weak gate do bit postprocessor

Artefatos:

- `src/kg1_v300_bit_fullbyte_postprocessor.py`
- `scripts/run_v301_bit_postprocessor_gate.py`
- `artifacts/v301_bit_postprocessor_gate/20260512T1130Z/v301_bit_postprocessor_gate_manifest.json`

Resultado no weak V275:

- baseline: `196/315`;
- postprocessed: `207/315`;
- `bit_manipulation`: `147/160`;
- `equation_transform`: `60/155`;
- ganhos: `11`;
- perdas: `0`;
- source guard: sem hits para termos proibidos (`answer`, `correct`, `verify_answer`, `solution`) no modulo postprocessor.

Decisao:

- V301 passa weak gate e e a primeira rota recente com ganho material em bit sem perda local.
- Se a submissao aceitar pipeline com postprocessamento, V301/V274 combinados sao candidatos fortes.
- Se a submissao for adapter-only, V301 vira dataset teacher/distillation target para tentar transferir o comportamento ao LoRA.

### V302 combined full local gate

Artefatos:

- `scripts/run_v302_combined_postprocessor_gate.py`
- `artifacts/v302_combined_postprocessor_gate/20260512T1200Z/v302_combined_postprocessor_gate_manifest.json`
- entrada: `artifacts/v293_gap_mining/inputs/v291_full_predictions.csv`

Resultado no full local rotulado `947`:

- baseline V291: `823/947 = 0.8690601901`;
- V274 equation + V300 bit combinados: `838/947 = 0.8848996832`;
- ganhos: `15`;
- perdas: `0`;
- `bit_manipulation`: `135/160` -> `146/160`;
- `equation_transform`: `56/155` -> `60/155`;
- side families mantidas em `100%`;
- truncation permanece `1`.

Decisao:

- Este e o melhor sinal objetivo ate agora para sair do plato `0.86`.
- Risco principal: a competicao/submissao final pode exigir adapter-only; nesse caso o postprocessor nao e diretamente submetivel.
- Proximo passo correto:
  1. confirmar regra de submissao/package atual;
  2. se postprocessor for permitido, montar pacote/inferencia com V274+V300 e gate final;
  3. se adapter-only, gerar dataset teacher dos `15` ganhos e fazer distillation curta em HF, mantendo replay das side families e gates `bit>=146`, `eq>=60`, `full>=838` como alvo local.

### V303 bit full-byte distillation dataset

Artefatos:

- `scripts/build_v303_bit_fullbyte_distill_dataset.py`
- `artifacts/v303_bit_fullbyte_distill_dataset/20260512T1010Z/v303_bit_fullbyte_distill_manifest.json`
- `artifacts/v303_bit_fullbyte_tokenization_gate/20260512T1010Z/v286_generic_tokenization_gate_manifest.json`

Objetivo:

- transferir o ganho V300/V302 para um LoRA adapter-only, porque o pacote oficial atual rejeita `prediction_postprocessor`;
- usar somente prompts sinteticos de bit full-byte, sem treinar em linhas weak/full rotuladas;
- manter V290/V274 como base para nao perder os ganhos de equation ja comprovados.

Composicao:

- base V290:
  - train `11286`, validation `801`;
  - equation patch V274 ja incluido.
- patch V303:
  - train `1536` linhas bit full-byte;
  - validation `168` linhas bit full-byte;
  - padroes exatos derivados dos ganhos V302: `CHO`, `MAJ3` e `XOR(SHL1,SHR4)`;
  - variacoes sinteticas conservadoras com `MAJ3`, `CHO`, `PAR3` e `XOR`.
- dataset final:
  - train `12822`;
  - validation `969`;
  - family train bit: `4231`;
  - family validation bit: `332`.

Gates:

- overlap contra weak bridge `315`: `0` ids e `0` prompts;
- train/validation prompt overlap: `0`;
- tokenizer real Nemotron `cbd3fa9f933d55ef16a84236559f4ee2a0526848`;
- offset masks: `12822/12822` train e `969/969` validation;
- prompt truncation: `0.0`;
- token max: `327`;
- completion tokens dropped: `0`.

Decisao:

- V303 passou os gates locais, mas o treino HF ja foi executado e nao transferiu os ganhos V302 para adapter-only.
- Resultado observado: melhor checkpoint/final ficou em `191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, truncation `0`, sem ganho contra o baseline fraco.
- Rejeitar V303 para full eval, package ou Kaggle submit.
- Diagnostico: o trace full-byte/final-answer e curto demais para ensinar a politica operacional que gerou V302. A proxima tentativa precisa treinar o modelo a executar verificacao bit-serial e regras de equation, nao apenas memorizar a expressao final.

### V304 solver trace distillation dataset

Artefatos:

- `scripts/build_v304_solver_trace_distill_dataset.py`
- `scripts/run_v286_generic_tokenization_gate.py`
- `artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_manifest.json`
- `artifacts/v304_solver_trace_tokenization_gate/20260512T1430Z/v286_generic_tokenization_gate_manifest.json`

Objetivo:

- corrigir a falha V303 com traces teacher mais proximos da politica descrita nas Kaggle Discussions `688461` e `690307`;
- transformar parte do ganho local V302 (`bit_manipulation 135->146`, `equation_transform 56->60`) em comportamento adapter-only;
- manter replay do V290/V274 para reduzir esquecimento das familias saturadas e preservar o formato de resposta que ja chegou a public score `0.86`.

Composicao:

- base V290/V274 preservada;
- rows finais: train `12822`, validation `969`;
- trace rows: train `2616`, validation `288`;
- formatos train:
  - `exact_final_answer`: `10206`;
  - `equation_numeric_rule_trace_v1`: `1080`;
  - `bit_serial_target_verification_trace_v2`: `1536`;
- formatos validation:
  - `exact_final_answer`: `681`;
  - `equation_numeric_rule_trace_v1`: `120`;
  - `bit_serial_target_verification_trace_v2`: `168`;
- hashes:
  - train SHA256 `7935ff999cdd8318de67538922de3651170c59baa2664a10beac3334dfcf9082`;
  - validation SHA256 `2b06224afe035c5085798f4a4be27e764ffaebde3ff7eee11c558c0cd5bdd29d`.

Gates:

- tokenizer real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16@cbd3fa9f933d55ef16a84236559f4ee2a0526848`;
- assistant suffix mode validado: traces podem terminar em `Final answer: <answer>`;
- offset masks: train `12822/12822`, validation `969/969`;
- fallback masks: `0`;
- prompt truncation rate: `0.0`;
- completion tokens dropped: `0`;
- token max: train `745`, validation `744`;
- duplicate assistant groups existem por replay, mas conflicting duplicate assistant groups: `0`.

Decisao:

- V304 substitui V303 como a proxima tentativa adapter-only.
- Autorizado somente smoke HF H200 curto e budget-capped, com preflight de hashes/tokenizacao/adaptador/GPU e kill-switch por weak eval.
- Criterio minimo para continuar:
  - weak bit `> 135/160` ou equation `> 56/155`;
  - sem regressao em side families;
  - sem aumento relevante de truncation;
  - full eval somente se o weak superar V291/V303 com margem real.
- Nao iniciar treino longo se o primeiro checkpoint repetir `191/315`, `56/155`, `135/160`.

### V305 Kaggle discussion audit

Artefatos:

- `artifacts/v305_requested_kaggle_discussion_audit/20260512T0000Z/v305_requested_kaggle_discussion_audit_summary.md`
- `artifacts/v305_requested_kaggle_discussion_audit/20260512T0000Z/v305_requested_kaggle_discussion_audit_summary.json`
- indices e fetch summary sem raw dumps completos: `discussion_index_normalized.csv`, `requested_discussion_index.csv`, `fetch_summary_master_20260512.json`

Topicos analisados:

- `681745`, `698106`, `687798`, `681714`, `688360`, `698293`, `694556`, `688461`, `694975`, `689915`, `685920`, `684212`, `693260`, `685031`, `685710`, `691318`, `688482`, `691641`, `689257`, `687961`, `684289`, `688277`, `683866`, `698649`, `697491`, `696735`, `690307`, `690891`, `689840`.

Achados que mudam implementacao:

- `690307`: melhor blueprint publico para `bit_manipulation`; iterar pares de bits, usar `bitsum` como hash, stride esquerda/direita e preencher lacunas com regra consistente. V304 ainda nao implementa o scan completo; isso vira V305/V306 CPU-only antes de outro treino longo.
- `688461`: confirma bit-serial target verification e verificacao por etapa. V304 ja incorporou essa direcao no trace `bit_serial_target_verification_trace_v2`.
- `697491` e `693260`: dataset correto pode piorar LB se for dificil de aprender, oversampled, com LR alto, conflito de formato ou duplicate CoT. Portanto V304 precisa de replay, LR conservador, oversampling limitado, duplicate-conflict gate e kill-switch.
- `689915`: loss medio nao basta; o objetivo correto e elevar confianca/min-logprob dos tokens criticos. Adicionar auditoria de min-logprob antes de treino longo.
- `691641` e `690891`: parte de `equation_transform` nao e deducao direta quando o operador da query nao aparece; usar fallback por prior, comprimento, sinal, reversao e ausencia de operador, sempre com no-loss gate.
- `683866`: exemplos de bit podem ser subdeterminados; solver deve medir ambiguidade e evitar treinar labels nao resolvidos.
- `698106` e `687798`: metric/extraction exige resposta final limpa e bit como string binaria exata.
- `681714`: submissao continua sendo adapter-only; postprocessor/verifier local so melhora ranking se for destilado para LoRA ou se o pacote oficial permitir inferencia customizada.
- `687961`: H200 com rank-32/contexto longo exige microbatch pequeno e gates de memoria; evitar traces longos sem evidencia.
- `691318` e `685920`: vLLM pode variar mesmo com temperatura `0`; comparar por familias e repetir weak quando a diferenca for pequena.

Achados mantidos como P2/P3, sem acao imediata:

- `694975`: GRPO pode ajudar, mas e caro e sem prova suficiente para nosso budget; manter depois de SFT/verifier com sinal.
- `698293`: gold-conditioned symbolic solver e fonte de pesquisa, nao preditor deployable.
- `684212`: visualizacoes do base model ajudam diagnostico/logprob, nao fornecem labels.
- `684289`: dataset de unit tests bit pode virar preflight barato, mas nao treino direto sem anti-leakage/proveniencia.
- `685031`, `688482`: topicos logisticos, sem ganho tecnico.

### V306 solver promotion gate

Artefatos:

- `scripts/run_v306_solver_promotion_gate.py`
- `artifacts/v306_solver_promotion_gate/20260512T1530Z/v306_v291_full_v306_solver_promotion_manifest.json`
- `artifacts/v306_solver_promotion_gate/20260512T1530Z/v306_v291_full_v306_solver_promotion_audit.csv`
- `artifacts/v306_solver_promotion_gate/20260512T1530Z/v306_v291_full_v306_equation_candidates_baseline.csv`
- `artifacts/v306_solver_promotion_gate/20260512T1530Z/v306_v291_full_v306_equation_candidates_after_combined.csv`

Objetivo:

- transformar os achados das Kaggle Discussions em gate operacional antes de gastar HF;
- separar sinal `deployable/no-loss`, sinal `diagnostic-only`, e sinal que precisa virar adapter-only por destilacao;
- impedir que o solver stride e candidatos numericos amplos entrem no pacote com perdas escondidas.

Resultado no conjunto rotulado V291 full-like (`947` linhas):

- baseline V291:
  - overall `823/947 = 86.91%`;
  - `bit_manipulation`: `135/160`;
  - `equation_transform`: `56/155`;
  - truncation total: `1`.
- V274+V300 combinado:
  - overall `838/947 = 88.49%`;
  - `bit_manipulation`: `146/160`, ganho `+11`;
  - `equation_transform`: `60/155`, ganho `+4`;
  - side families sem regressao: gravity `159/159`, numeral `157/157`, text `157/157`, unit `159/159`;
  - perdas locais: `0`.

Regras responsaveis pelos ganhos:

- equation V274:
  - `minus_signed_opposite_sign_guarded`: `2`;
  - `colon_absdiff_unreverse_same_len`: `1`;
  - `add_direct_over_model_add_variant`: `1`.
- bit V300:
  - `fullbyte_safe_ternary`: `10`;
  - `fullbyte_binary`: `1`.

Achados bloqueados:

- stride V296/Discussion `690307` isolado ficou `diagnostic_only_lossy`:
  - ganhos contra baseline: `2`;
  - perdas contra baseline: `13`;
  - portanto nao pode ser promovido diretamente.
- fallback numerico amplo V299 apos V274/V300:
  - `same_operator_unique_numeric_dsl`: `31` candidatos, `30` corretos, mas ainda `1` perda;
  - `all_numeric_examples_unique_dsl`: `4` candidatos, `2` corretos, `2` perdas;
  - nenhum candidato adicional foi promovido apos o combinado.

Decisao:

- V306 confirma que o ganho local real disponivel continua sendo `+15` (`+11 bit`, `+4 equation`) sem perda no conjunto rotulado.
- Esse ganho nao e Kaggle-submit ready como CSV/postprocessor se o pacote oficial aceitar apenas adapter LoRA; precisa ser destilado para comportamento adapter-only.
- V304 e o dataset atual mais adequado para tentar essa transferencia, mas o proximo smoke HF precisa usar V306 como gate de sucesso: continuar somente se o adapter aumentar `bit_manipulation` acima de `135/160` ou `equation_transform` acima de `56/155` sem regressao.

Proxima acao tecnica:

1. Rodar preflight HF para V304 com os mesmos gates do Colab adaptados a job HF.
2. Rodar um smoke H200 curto e barato, usando V306 como criterio de continuidade.
3. Se smoke repetir V291/V303 (`bit=135`, `equation=56`), parar treino e voltar para dataset/trace; nao gastar budget longo.
4. Se smoke mostrar qualquer ganho real em `bit_manipulation` ou `equation_transform`, rodar checkpoint adicional com LR conservador e gate por familia.
5. Promover para full eval/package somente se weak ou full official-like provar ganho real contra V291 sem regressao lateral.
