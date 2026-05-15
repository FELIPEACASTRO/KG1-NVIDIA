# KG1 V411B New Sources Triple Check

Gerado em: 2026-05-14

## Escopo

Este triple check revisou somente fontes que ainda nao tinham sido usadas como base direta do roadmap V411. O objetivo foi procurar uma decisao tecnica nova para `bit_manipulation` e `equation_transform`, sem adicionar ruido ao plano.

Baseline de decisao:

| Referencia | Weak total | equation_transform | bit_manipulation | Submit-safe |
|---|---:|---:|---:|---|
| V291/V290 checkpoint-6 | `192/315` | `56/155` | `136/160` | sim |
| V409 CPU projection | `202/315` | `63/155` | `139/160` | nao |
| V410 transfer dataset | nao treinado | nao medido | nao medido | nao |

## Fontes Academicas e Top Universities

Filtro aplicado: uma fonte entra no roadmap apenas se alterar a implementacao de um gate, solver, verifier, dataset ou criterio de promocao. Buscas academicas genericas que so repetem "use program synthesis" foram descartadas.

Fontes verificadas e impacto:

| Fonte | Evidencia | Impacto KG1 |
|---|---|---|
| MIT CSAIL / Solar-Lezama, Sketch e curso de Program Synthesis ([curso](https://people.csail.mit.edu/asolar/SynthesisCourse/index.htm), [Sketch](https://people.csail.mit.edu/asolar/sketch2012/)) | Sketch permite escrever programas parciais com buracos e sintetizar detalhes; o curso destaca a interacao entre busca e verificacao. | `equation_transform` e `bit_manipulation` devem ser tratados como PBE/SyGuS: gramatica pequena, busca por programa curto, verifier obrigatorio e abstencao quando ambiguo. |
| UPenn / Berkeley / MIT, SyGuS ([site](https://sygus.org/), [paper](https://www.cis.upenn.edu/~alur/SyGuS13.pdf)) | SyGuS formaliza sintese com especificacao semantica e gramatica sintatica. | V412 deve declarar explicitamente a gramatica de candidatos por familia e verificar todos os exemplos antes de aceitar uma predicao. |
| UW / Rosette ([site](https://emina.github.io/rosette/)) | Rosette compila DSLs para restricoes SMT para sintese e verificacao. | Se a enumeracao Python empacar, a extensao correta e solver-aided DSL, nao prompt maior nem treino bruto. |
| Stanford STOKE / superoptimization ([repo](https://github.com/StanfordPL/stoke)) | Busca programas e usa testcases + verificacao para provar equivalencia. | Para bit, usar vetores de valores e contraexemplos sinteticos; nunca aceitar candidato so porque passou poucos exemplos fracos. |
| Microsoft PROSE / PBE ([site](https://www.microsoft.com/en-us/research/project/prose-framework/)) | Dada uma DSL e exemplos I/O, sintetiza programas consistentes e ranqueados. | Em equation, ranquear por menor programa/MDL e exigir unicidade ou predicao consensual entre candidatos. |
| ETH Zurich / SMT and synthesis ([program verification](https://www.pm.inf.ethz.ch/education/courses/program-verification.html)) | Program verification usa SMT/Z3; pesquisas de sintese e verificacao reforcam geracao por especificacao. | Validar V412 com differential/fuzz tests para evitar bug de solver ou overfit de DSL. |
| NUS SynGuar / PBE generalization ([paper](https://www.comp.nus.edu.sg/~prateeks/papers/SynGuar.pdf)) | A literatura alerta que PBE com poucos exemplos pode nao generalizar. | LUT e equation DSL precisam de guardrails de cobertura, unicidade e abstencao. |

Conclusao academica: as melhores universidades reforcam o mesmo caminho tecnico, mas nao adicionam um "truque de treino". O ganho vem de sintese verificavel e dataset de transferencia somente depois do gate.

## Novas Fontes Kaggle/HF Auditadas

Kaggle sources baixadas e inspecionadas:

- `manderson240/nemotron-pure-symbolic-solver-v29`
- `manderson240/nemotron-pure-symbolic-solver-v28`
- `johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster`
- `amanatar/nemotron-ultimate-sft-grpo-v3`
- `torpidoff/full-pipeline-nvidia-nemotron-3-reasoning`
- `aliafzal9323/nemotron-30b-deterministic-solvers-cot-fine-tu`
- `kalyankkr/all-6-puzzle-types-decoded-sft-training-data`
- `khoinguyennguyen/aimo3-blackboard-cross-model-multi-agent-solver`
- `anhtuan299/blackboard-expert-agent-assembly-solving-technique`
- `mohit98765/arc-2026-llm-guided-program-synthesis-v5`
- `mohit98765/arc-2026-mcts-program-synthesis-v9`
- `mohit98765/arc-2026-dreamcoder-lite-library-learning-v10`

HF sources revisadas:

- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: espelho/empacotamento util como referencia de dados, sem ganho novo sobre nossos CSVs auditados.
- `andy279/nemotron-reasoning-challenge-raw-traces`: potencialmente util, mas gated. So entra se houver acesso e passar anti-leakage, family-count, prompt-hash e row-contract gate.

## Sweep Adicional MIT/Top Universities/Asia

Pedido adicional em 2026-05-14: buscar MIT e grandes universidades globais, incluindo China, Taiwan, Japao, Coreia e Coreia do Norte.

Resultado util:

- Peking University PL Lab / [PISTool](https://www.jiry17.site/pistool/): biblioteca de sintese indutiva com suporte a tarefas SyGuS e version-space algebra. Impacto: reforca o desenho V412 com cache/VSA e enumeracao por DSL, mas nao adiciona uma regra nova alem de LUT k=2/k=3 e DSL equation v29 ja registradas.
- KAIST / [program synthesis artifacts](https://prosys.kaist.ac.kr/publications/pldi18b.pdf): compara dominios de string manipulation, bit-vector manipulation e circuit transformation contra EUSolver/SyGuS, reforcando uso de modelos/probabilidade para priorizar busca. Impacto: P2 para priorizacao depois do V412; nao autoriza GPU nem novo treino.
- Princeton / SyGuS para bit-vector lemma generation: reforca que problemas bit-vector devem ser tratados com sintese + verifier, mas nao adiciona operador novo.
- Taiwan/Japao/Coreia: buscas por NTU/NTHU/Academia Sinica, University of Tokyo, Kyoto, Tokyo Tech, Osaka, KAIST, Seoul National University e POSTECH retornaram material geral de SMT/synthesis ou ruido de outras areas. Nada novo com impacto direto maior que PISTool/KAIST.
- Coreia do Norte: nao foi encontrada fonte tecnica aberta e verificavel de universidade norte-coreana sobre program synthesis/SMT/SyGuS aplicavel ao KG1.

Decisao: nenhum item novo foi inserido no roadmap alem do V411B. As novas fontes confirmam a rota V412; nao trazem ganho concreto adicional.

## Achados Acionaveis

### 1. Bit: LUT booleana k=2/k=3 com guardrails

O notebook `manderson240/nemotron-pure-symbolic-solver-v29` contem a ideia mais util desta rodada: para cada bit de saida, procurar uma funcao booleana de 2 ou 3 bits de entrada por tabela verdade, primeiro em tabelas nomeadas e depois em tabelas restantes. Isso complementa o V408/V409, que ja cobria `INHIB`, `IMPL` e pares ordenados, mas ainda nao tinha um modo conservador de LUT parcial.

Implementacao correta para V412:

- enumerar por output bit:
  - direct/NOT/constant ja existentes;
  - operadores binarios nomeados, com pares ordenados;
  - LUT k=2 sobre pares de bits;
  - LUT k=3 somente quando k=2 falhar;
- aceitar LUT somente se:
  - bater todos os exemplos do prompt;
  - tiver cobertura minima de padroes observados;
  - for unica entre candidatos ou todos os candidatos consistentes predizerem o mesmo bit de teste;
  - passar fuzz/differential local em familias sinteticas;
  - nao causar perda contra baseline em weak audit.

Risco: uma tabela verdade arbitraria pode overfit com poucos exemplos. Portanto a regra nao pode ser "primeiro match vence"; precisa de unicidade/abstencao.

### 2. Equation: catalogo numeric/operator maior, mas com unicidade

O mesmo `manderson240` v29 tem o melhor catalogo concreto novo para equation:

- unary: identidade, negacao, digit-sum, digit-product, reverse digits, len digits, square, cube, `+1`, `-1`, `*2`, `*10`, `//10`, `%10`, `//2`;
- binary: `+`, `-`, abs diff, `*`, `//`, `%`, max/min, bitwise `| & ^`, digit-sum combos, concat/reverse concat, digit-length, digit-product, gcd, lcm, squares sum/diff, `(a+b)*(a-b)`;
- ternary: soma, diferencas, produtos, `(a+b)*c`, `a*(b+c)`, `a+b*c`, max/min, bitwise, digit-sum ternario, concat abc, `a*b-c`, `abs(a-b-c)`;
- per-operator lookup: `concat_ab`, `concat_ba`, `a+b`, `abs`, `a*b`, `a-b`, `b-a`, div/mod, digit-sum, gcd, `+1/-1`, `*2`.

Implementacao correta para V412:

- separar lanes:
  - numeric/operator;
  - symbolic/punctuation/brackets;
- gerar candidatos por custo crescente;
- verificar todos os exemplos;
- aplicar tie-break por menor programa/MDL;
- aceitar apenas programa unico ou predicao identica entre candidatos minimos;
- abstencao quando houver ambiguidade.

Risco: o codigo v29 usa heuristicas de primeiro match em alguns pontos. Para nosso gate, a ideia entra, mas com verifier mais rigoroso.

### 3. GRPO/RL nao e proximo passo imediato

Os notebooks GRPO novos confirmam reward por resposta exata, formato boxed e comprimento. Isso e util se formos para RL, mas nao resolve o gargalo atual:

- GRPO sem solver/verifier novo repete o problema de loss/reward nao virar ACC nas duas familias.
- O custo HF/Kaggle nao se justifica antes de V412 encontrar ganho CPU no-loss adicional.
- Se usado depois, deve ser smoke curto, com kill-switch por `weak total > 192`, `equation > 56`, `bit >= 136`, `truncated=0`.

### 4. Blackboard/MCTS/DreamCoder sao P2, nao caminho de hoje

Os notebooks ARC/AIMO reforcam arquitetura de geracao de hipoteses + verifier, mas nao fornecem regra direta para KG1. Eles entram apenas como opcao posterior:

- MCTS limitado sobre DSL quando enumeracao exaustiva ficar cara;
- biblioteca de programas recorrentes depois de V412/V413;
- votacao somente entre candidatos verificados.

## Decisao

V411B nao autoriza submit e nao autoriza treino longo. Ele altera a implementacao do proximo gate CPU:

- V412 bit deve adicionar LUT k=2/k=3 com unicidade, cobertura e abstencao.
- V412 equation deve incorporar o catalogo numeric/operator do v29, mas reescrito com verifier rigoroso e tie-break MDL.
- HF/Kaggle GPU so volta se V412 produzir ganho no-loss novo e V413 converter isso em dataset de transferencia com gate real.

## Ganho Esperado

Estimativa honesta, antes de executar V412:

| Caminho | Ganho possivel weak | Confianca | Observacao |
|---|---:|---|---|
| Bit LUT k=2/k=3 com abstain | `+0` a `+2` bit | media | pode recuperar casos alem de `INHIB/IMPL`; risco de overfit se sem guardrail |
| Equation catalogo v29 rigoroso | `+0` a `+4` equation | media | melhor chance de achar os 4 rows ate `60/155` em CPU; adapter-transfer ainda nao provado |
| GRPO/RL direto | indefinido | baixa | nao executar sem V412/V413 |

## Limpeza

Os downloads brutos de Kaggle usados nesta auditoria devem ser removidos do workspace apos a extracao dos achados. O repo deve manter somente este relatorio e o manifest pequeno.
