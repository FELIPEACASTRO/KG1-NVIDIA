# KG1 V528 - Double Check Dos Notebooks 0.86/0.85

Data: 2026-05-17

## Escopo

Analise dos notebooks publicos do Kaggle que explicitamente citam score `0.86`
ou `0.85`, mais notebooks Tong/Huikang relacionados que explicam a linhagem de
adapter/CoT usada pelos pacotes de alto score.

Artefatos brutos:

- `score086_notebook_inventory.csv`
- `afr1ste__nemotron-0-86-tinker-adapter-guide.json`
- `safar1__lb-score-0-86.json`
- `mohamedamr992__0-86-adapter-packaging-workflow.json`
- `huikang__end-to-end-finetuning-for-lb-0-85.json`
- `huikang__tinker-submission-notebook.json`
- `huikang__adapter-validation-notebook.json`
- `dgxchen__training-with-unsloth-to-achieve-0-85-lb.json`
- `konbu17__nemotron-tong-style-cot-sft-updated-v2.json`
- `teasue05__tinker-submission-notebook.json`

## Conclusao Curta

Os notebooks com `0.86` explicito nao sao notebooks que resolvem
`bit_manipulation` ou `equation_transform`. Eles sao principalmente notebooks de
empacotamento/auditoria que submetem o adapter publico:

`kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1`

O aprendizado util para nossas duas familias vem dos notebooks de treino
`0.85`/Tong-style e dos notebooks de solver/CoT, nao dos empacotadores `0.86`.

## O Que Os Notebooks 0.86 Ensinaram

| Notebook | Tipo | Achado real | Acao no KG1 |
|---|---|---|---|
| `afr1ste/nemotron-0-86-tinker-adapter-guide` | empacotamento/auditoria | score `0.86` vem de pacotear corretamente o Tinker adapter; zip precisa conter `adapter_config.json` e `adapter_model.safetensors` na raiz | manter gate de zip root, schema, namespace e cobertura de target modules antes de submit |
| `safar1/lb-score-0-86` | empacotamento | lista varias versoes Kien e escolhe `version_14 = .../triton/tinker-adapter/1` | tratar como referencia de provenance; nao como novo metodo de treino |
| `mohamedamr992/0-86-adapter-packaging-workflow` | empacotamento | escolhe `model_name = "tinker-adapter"` e copia os dois arquivos do adapter para `submission.zip` | reforca que o 0.86 e adapter artifact, nao CSV nem runtime solver |
| `huikang/tinker-submission-notebook` | conversao/auditoria | alinha config, expande experts, converte `gate_proj + x_proj -> in_proj` via SVD, renomeia namespace, valida safetensors | manter checks de key namespace, expert expansion e in_proj conversion em qualquer conversao |
| `huikang/adapter-validation-notebook` | validacao | valida adapter com vLLM/LoRA, `max_lora_rank=32`, prompt oficial com `\boxed{}` | manter validacao official-like antes de considerar submit |

## O Que Realmente Ajuda Bit Manipulation

Dos notebooks Tong-style, PearPN25, Konbu e PJT:

1. `bit_manipulation` precisa ser tratado como familia de regras booleanas por
   bit, nao como texto generico.
2. O conjunto minimo de candidatos deve incluir:
   - whole-byte: XOR mask, rotate, reverse, permutation, NOT;
   - unario: ID, NOT, C0, C1;
   - binario: AND, OR, XOR, XNOR, NAND, NOR;
   - assimetricos: INHIB, reverse-INHIB, IMPL, reverse-IMPL;
   - ternarios: MAJ, CH/CHO, XOR3;
   - GF(2) affine e degree-2 ANF como caminhos compactos quando enumeracao
     local fica ambigua.
3. O solver deve usar evidencia estrutural:
   - dominant shift;
   - stride `+1/+1`;
   - permutation consistency;
   - operation priors;
   - ambiguity scoring.
4. Traces longas de bit podem dominar a loss. Para KG1, o trace de bit deve ser
   curto e normalizado por exemplo (`example_mean`), nao por massa de tokens.

Impacto no plano: V525/V526 devem priorizar `example_mean`, traces de bit
curtas e coverage novo acima do V304/V523, com hard guard de `bit>=136` e
proteção do caso `8740ed31`.

## O Que Realmente Ajuda Equation Transform

Dos notebooks PJT e ZZYS:

1. `equation_transform` deve ser separado em subtipos numeric/symbolic antes de
   treinar.
2. O DSL numeric deve cobrir:
   - soma, subtracao A-B/B-A, abs diff;
   - multiplicacao;
   - divisao e modulo;
   - concat e reverse concat;
   - reversed operands;
   - reversed result;
   - `+1/-1` e offsets pequenos;
   - prefix/suffix operator encoding;
   - bitwise XOR/AND/OR quando os operandos sao inteiros.
3. O DSL symbolic deve cobrir aritmetica em printable ASCII, incluindo
   candidatos tipo `Z_94/mod-94`, operacoes por operador e padroes de pontuacao.
4. Treino generico por mais epochs nao e a rota. A rota e gerar traces
   deterministicamente corretas para regras que o baseline erra, com hard
   negatives e sem usar labels weak/full como treino.

Impacto no plano: qualquer novo pacote equation precisa primeiro demonstrar em
CPU quais regras novas resolve, quantos source rows cobrem essas regras, e so
depois virar SFT curto. O gate de promocao continua `equation>=57` para smoke e
meta `equation>=60` para submit-safe.

## O Que Nao Devemos Fazer

- Nao interpretar notebook `0.86` de packaging como tecnica nova para familia.
- Nao trocar o adapter inteiro por outro adapter publico apenas porque o zip e
  valido.
- Nao rodar GPU so porque um notebook publico cita `0.85/0.86`.
- Nao usar solver/verifier em runtime submit.
- Nao usar weak/full labels como dados de treino.
- Nao otimizar apenas eval_loss token-level; a metrica precisa refletir ACC por
  row/familia.

## Decisao V528

1. Proteger o Tinker-like adapter schema como baseline de empacotamento.
2. Usar notebooks `0.86` apenas para auditoria de package/provenance.
3. Usar notebooks Tong-style/solver para implementar V525/V526:
   - bit: boolean candidate coverage + shift/stride/GF2/ANF;
   - equation: DSL numeric/symbolic expandida;
   - treino: `example_mean`, family/token balance e checkpoint kill-switch.
4. Nenhum novo submit deve ser feito por causa dos notebooks `0.86` ate haver
   ganho local label-free em weak/full.

