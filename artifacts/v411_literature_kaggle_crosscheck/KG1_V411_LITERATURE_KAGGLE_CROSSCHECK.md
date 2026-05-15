# KG1 V411 Literature and Kaggle Cross-Check

Date: 2026-05-14

## Scope

Double-check focused on `bit_manipulation` and `equation_transform`, including:

- Kaggle kernels from this challenge;
- Kaggle kernels from adjacent program-synthesis/TIR challenges;
- program synthesis literature relevant to bit-vectors, string/number transformations, and example-based equation rules.

Current measured state:

| Candidate | Weak total | equation_transform | bit_manipulation | Submit-safe? |
|---|---:|---:|---:|---|
| V291/V290 checkpoint-6 adapter package | `192/315` | `56/155` | `136/160` | yes |
| V409 integrated CPU solver projection | `202/315` | `63/155` | `139/160` | no, solver/verifier projection |
| V410 solver-first transfer dataset | not trained | not measured | not measured | no, dataset only |

## Sources Checked

Kaggle/Nemotron:

- `konbu17/bit-manipulation-solver-cot-generator`
- `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`
- `huikang/end-to-end-finetuning-for-lb-0-85`
- `kienngx/nemotron-sft-reasoning-trajectories-dataset`
- `konbu17/nemotron-tong-style-cot-sft-updated-v2`

Kaggle adjacent competitions:

- `michaelhodel/program-synthesis-starter-notebook`
- `marcshade/three-tier-dsl-based-program-synthesis`
- `francisbanda/arc-agi-2-mdl-program-synthesis-solver`
- `sorokin/aimo2-tir-rm`
- `nihilisticneuralnet/39-50-aimo3-condition-mining-tir-w-python`

Literature/reference:

- Syntax-Guided Synthesis / CEGIS.
- Programming by Example and FlashFill-style synthesis from examples.
- Z3 bit-vector theory and fixed-width bit-vector semantics.
- Enhanced enumeration for SyGuS bit-vector manipulation.
- Semantics-Guided Synthesis.
- DreamCoder/neural-guided program synthesis.

## Findings

### 1. Bit manipulation remains a synthesis problem, not a loss problem

The strongest Kaggle-specific source is still Konbu17/Tong-style bit solving:

- output bits should be solved independently when possible;
- candidate functions must include asymmetric `INHIB(a,b)=a AND NOT b` and `IMPL(a,b)=NOT a OR b`;
- ordered pairs naturally cover reverse variants;
- `MAJ`, `CH`, and `XOR3` are fallback candidates only if every example bit verifies;
- low-confidence or ambiguous bit CoT should not enter SFT.

The literature adds one implementation refinement beyond V408:

- use candidate-signature caching / term-graph style enumeration;
- filter candidates by their value vector over examples before composing them;
- rank candidates by short expression length and recurrence, not by model confidence.

This is directly useful for a V412/V413 CPU gate, especially for ternary bit rows that V408 intentionally avoided.

### 2. Equation transform must be handled as PBE/SyGuS

The cross-check found no credible evidence that broad SFT, more epochs, or lower `eval_loss` is the right lever.

Actionable rule:

- define a small grammar;
- enumerate candidates in increasing complexity;
- verify every input-output example;
- apply to the query only when the program is unique or the tie-break is deterministic and lossless;
- abstain on ambiguity.

For `equation_transform`, the next DSL should be split into two lanes:

- numeric lane: affine, `+1/-1`, multiply/divide/mod, concat digits, reverse concat digits, sign formatting;
- symbolic lane: operator substitution, bracket/punctuation movement, literal insertion/deletion, variable ordering, expression-side swap, simple template rewrite.

### 3. Adjacent Kaggle submissions reinforce solver/verifier gates

ARC program-synthesis kernels are not directly transferable, but they confirm:

- small DSL beats unconstrained search;
- brute force explodes quickly without pruning;
- MDL/shortest-program tie-break is safer than first-match;
- every candidate must verify against all training examples.

AIMO/TIR kernels add a useful pattern:

- generate multiple hypotheses;
- execute or verify them with tools;
- use majority/reward only after candidates are syntactically valid and executable.

For this challenge, that means TIR/reward-model ideas should become CPU verifier or trace-generation helpers. They should not replace weak/full ACC gates.

### 4. Training-data lessons

Kienngx and Konbu notebooks both converge on the same point:

- data quality matters more than quantity;
- only verified-correct CoT should be trained;
- hard families need solver knowledge injected into traces;
- equation transform was explicitly under-solved in public CoT training notebooks.

This supports V410, but does not prove V410 will transfer. Promotion still requires first-checkpoint ACC improvement.

## Decision

V411 does not create a new submit-safe adapter. It confirms the correct next path:

1. keep V291 as the only submit-safe package;
2. use V409 as the CPU target projection;
3. use V410 as the first transfer dataset;
4. add one more CPU gate before any larger GPU job:
   - bit: candidate-signature cache plus verified ternary fallback;
   - equation: numeric/symbolic PBE grammar with uniqueness and abstention.

## Expected Gain

Measured today:

- submit-safe adapter-only: still `192/315`, `equation=56`, `bit=136`;
- CPU projection: `202/315`, `equation=63`, `bit=139`.

Expected next measurable gain:

- CPU gate may find incremental no-loss candidates, likely small (`+0` to `+4`) unless the symbolic equation DSL covers more misses;
- adapter-only gain remains unproven until a V410/V412 smoke checkpoint beats V291.

## Implementation Impact

Do next:

- V412 CPU equation/bit synthesis cross-check:
  - candidate signatures over all examples;
  - shortest-program tie-break;
  - strict no-loss integration;
  - output accepted candidates as solver traces.
- V413/V414 transfer update only if V412 finds new no-loss signal.
- HF/Kaggle GPU smoke only after tokenization and no-leak gates pass.

Reject:

- broad SFT;
- longer training just because loss improves;
- reward/majority voting without an exact verifier;
- any submit before weak/full ACC improves over V291.

