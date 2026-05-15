# KG1 External AI Research Prompt

Use este prompt em outras IAs para buscar ganhos reais nas familias `bit_manipulation` e `equation_transform`.

```text
Voce e um pesquisador senior de program synthesis, SMT/SyGuS, reasoning datasets e competicoes Kaggle. Analise o problema abaixo com rigor, sem alucinacao e sem sugerir treinamento generico se nao houver evidencia verificavel.

Contexto do desafio:
- Competicao: NVIDIA Nemotron Model Reasoning Challenge.
- O dataset contem puzzles de raciocinio com `prompt` e `answer`.
- As familias problematicas atuais sao:
  - `bit_manipulation`: weak gate atual `136/160` = 85.00%.
  - `equation_transform`: weak gate atual `56/155` = 36.13%.
  - Overall weak adapter-only: `192/315`.
- Melhor submit-safe atual: adapter-only V291/V290 checkpoint-6, weak `192/315`, `equation=56/155`, `bit=136/160`, full official-like `823/947`, public score `0.86`, rank historico 19.
- Projecoes CPU/verifier melhores, mas NAO submit-safe:
  - V409: `202/315`, `equation=63/155`, `bit=139/160`.
  - V405: `201/315`, `equation=63/155`, `bit=138/160`.
- Postprocessor/verifier V274/V275 chegou a `196/315`, `equation=60`, `bit=136`, mas nao e adapter-only.
- Precisamos melhorar ranking hoje com ganhos reais, mesmo pequenos, mas sem regredir bit nem truncation.

O que ja foi testado e falhou:
- Mais epochs/LR e queda de eval_loss nao melhoraram ACC.
- Teacher/verifier -> SFT LoRA direto nao transferiu: continua `equation=56`, frequentemente perde bit.
- Prompt/template sweep nao moveu equation; alguns prompts colapsaram ACC/truncation.
- Adapter soups V291/V382 regrediram bit/total.
- Raw-output extractor audit encontrou `0` respostas boxed recuperaveis nos misses; nao e erro simples de parser.
- Broad SFT e GRPO sem novo solver/verifier nao devem ser sugeridos.

O que ja sabemos que funciona em CPU/verifier:
- Para bit:
  - abordagem Tong Hui Kang: relacoes por bits, bit-pair, bitsum, stride, operadores unary/binary/constant.
  - V408/V409 adicionou operadores assimetricos `INHIB(a,b)=a AND NOT b`, `IMPL(a,b)=NOT a OR b`, pares ordenados e abstencao.
  - Novo achado V411B: usar LUT booleana por output bit com k=2/k=3 input bits, mas apenas com guardrails de cobertura, unicidade ou predicao consensual, verifier em todos os exemplos e abstencao.
- Para equation:
  - tratar como PBE/SyGuS, nao treino generico.
  - separar numeric/operator lane de symbolic/punctuation lane.
  - DSL atual precisa expandir: concat/reverse concat, signed format, literal insert/delete, brackets/punctuation, `+1/-1`, multiply/divide/mod, gcd/lcm, digit-sum, digit-product, reverse digits, square/cube, ternary ops e depth-2 pequenas.
  - Aceitar apenas programa unico/curto por MDL ou predicao identica entre candidatos minimos.

Fontes ja consideradas:
- Kaggle discussion de Tong Hui Kang sobre estrategia de bit manipulation.
- `tonghuikang/nemotron`.
- `manderson240/nemotron-pure-symbolic-solver-v29` com LUT k=2/k=3 e catalogo equation numeric/operator.
- MIT CSAIL Sketch / Solar-Lezama.
- SyGuS (UPenn/Berkeley/MIT).
- UW Rosette.
- Stanford STOKE.
- Microsoft PROSE.
- Peking University PISTool.
- KAIST program synthesis / EUSolver comparison.
- DryadSynth/SyGuS, CVC5/SyGuS, SMT/Z3 bit-vectors.

Pergunta principal:
Quais tecnicas concretas, implementaveis em CPU gate, podem aumentar `equation_transform` de `56/155` para pelo menos `60/155` e/ou `bit_manipulation` de `136/160` para acima de `136`, sem regressao e sem depender de postprocessor proibido?

Formato obrigatorio da resposta:
1. Liste apenas achados acionaveis. Para cada achado, indique:
   - familia afetada (`bit_manipulation`, `equation_transform` ou ambas);
   - fonte concreta com URL, paper, repo ou notebook;
   - algoritmo/regra exata;
   - como verificar em CPU sem labels do test;
   - risco de overfit/leakage/truncation;
   - criterio de aceitar/rejeitar;
   - ganho esperado em weak rows, se mensuravel.
2. Nao recomende "treinar mais", "usar LR maior", "mais epochs", "prompt melhor" ou GRPO amplo, a menos que explique exatamente por que isso superaria o teto `equation=56`.
3. Priorize:
   - sintese por exemplos;
   - SMT/SyGuS;
   - version-space algebra;
   - e-graphs/term-graph/value-vector enumeration;
   - MDL/shortest-program tie-break;
   - abstencao quando ambiguo;
   - fixtures/traces deterministas curtos para transferir solver para LoRA.
4. Se encontrar fonte de MIT, Stanford, Berkeley, CMU, Cornell, Princeton, ETH, Cambridge, Oxford, Imperial, UCL, Tsinghua, Peking, SJTU, Zhejiang, USTC, NUS, NTU Taiwan, NTHU Taiwan, Academia Sinica, University of Tokyo, Kyoto, Osaka, Tokyo Tech, KAIST, SNU, POSTECH ou outras top universities, explique exatamente se ela muda ou nao nosso plano.
5. Separe "ganho comprovavel agora" de "hipotese futura". Nao misture.
6. Termine com uma tabela:
   - Step;
   - Implementacao minima;
   - Custo computacional;
   - Threshold para continuar;
   - Threshold para cancelar por FinOps.

Restricoes:
- Nao usar test labels.
- Nao propor submit sem weak/full gate.
- Nao propor postprocessor/verifier externo como submissao se a regra do desafio exigir adapter-only.
- Toda sugestao precisa ser convertivel em script Python CPU antes de GPU.
```
