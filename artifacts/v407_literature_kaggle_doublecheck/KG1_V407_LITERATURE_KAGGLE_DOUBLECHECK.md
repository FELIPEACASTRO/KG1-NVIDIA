# KG1 V407 Literature and Kaggle Double Check

Date: 2026-05-14

## Scope

Objective: re-check literature and Kaggle notebooks/submissions related to `bit_manipulation` and `equation_transform`, with emphasis on methods that can produce measurable ACC gains instead of lower `eval_loss` only.

Current measured state:

| Candidate | Weak total | equation_transform | bit_manipulation | Submit-safe? |
|---|---:|---:|---:|---|
| V291/V290 checkpoint-6 adapter package | `192/315` | `56/155` | `136/160` | yes |
| V405 integrated CPU solver projection | `201/315` | `63/155` | `138/160` | no, solver/verifier projection |
| V406 solver-first transfer dataset | not trained yet | not measured | not measured | candidate dataset only |

## Checked Sources

Kaggle/Nemotron:

- `konbu17/bit-manipulation-solver-cot-generator`
- `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`
- `huikang/end-to-end-finetuning-for-lb-0-85`

Kaggle/program-synthesis analogues:

- `michaelhodel/program-synthesis-starter-notebook`
- `marcshade/three-tier-dsl-based-program-synthesis`
- `francisbanda/arc-agi-2-mdl-program-synthesis-solver`

Research/official references:

- Flash Fill / Programming by Example: small DSL plus input-output examples.
- SyGuS / CEGIS: grammar-constrained program synthesis with semantic verification.
- Z3 bit-vector theory: exact fixed-width bit-vector semantics for proof/verification.
- DreamCoder / neural-guided synthesis: neural component guides search, verifier decides.

## Findings

### Bit Manipulation

The strongest new actionable detail is from the Konbu17 bit CoT notebook. It expands the boolean function set beyond symmetric `AND/OR/XOR/NAND/NOR/XNOR`:

- `INHIB(a,b) = a AND NOT b`
- `IMPL(a,b) = NOT a OR b`
- ordered pairs cover reverse variants;
- `MAJ`, `CH`, `XOR3` are useful only as low-frequency fallback;
- high-confidence rows, where all output bits are resolved without brute-force ambiguity, are more valuable for SFT than low-confidence heuristic rows.

This fits our measured behavior:

- V403 exact global bit rules produced `+2` weak bit gains with `0` losses.
- Consensus/fallback bit rules would have caused losses and must stay rejected.
- Next bit work should not be broad SFT. It should be a CPU no-loss gate over missing bit rows using asymmetric boolean functions and strict abstention.

### Equation Transform

The literature and Kaggle analogues converge on the same answer: this is programming-by-example / synthesis, not generic language modeling.

Actionable constraints:

- define a small DSL;
- enumerate short candidates;
- verify against all examples in the prompt;
- tie-break by shortest/simple program;
- abstain if multiple candidates fit or the query application is not uniquely determined.

For our weak misses, this means:

- numeric operator DSL from `equation_numeric.py` remains valid but already harvested most known gains;
- unresolved symbolic rows need a FlashFill-style string/symbol DSL, not more epochs;
- candidates must include concat/reverse-concat, signed formatting, punctuation/bracket movement, fixed symbol substitution, literal insertion/deletion, and small arithmetic/string hybrids;
- only no-loss CPU discoveries should be converted into traces or adapter training examples.

### Other Kaggle Competitions

ARC program-synthesis notebooks are not directly transferable, but they reinforce three implementation choices:

- explicit DSL primitives are required;
- brute-force search explodes unless constrained by examples and short-program priors;
- MDL/shortest-program tie-break is safer than first-match or model-confidence tie-break.

This should change our local gates, not start a new broad training job.

## Decision

No source justifies repeating broad SFT or judging by `eval_loss`.

The correct next implementation path is:

1. V407/V408 CPU gate for bit:
   - extend exact bit solver with `INHIB`, `IMPL`, reverse ordered variants, and optional `MAJ/CH/XOR3`;
   - accept only rows that satisfy every example and produce no losses against baseline;
   - keep consensus/fallback blocked.
2. V407/V408 CPU gate for equation:
   - implement a PBE/SyGuS-style symbolic DSL over the `99` equation misses;
   - accept only unique short programs with all examples matched;
   - target at least `+4` equation with `0` losses.
3. V406 adapter transfer smoke:
   - use the already tokenization-gated V406 dataset only after CPU gate confirms no-loss signal;
   - stop at first checkpoint if `bit<136` or `equation<=56`.

## Expected Gain

Conservative expectation:

- CPU solver/verifier path: can preserve V405 projection `201/315`, with `equation=63`, `bit=138`.
- Adapter-only path: unproven until V406/V408 smoke. Minimum promotion gate remains `>192/315`, `equation>56`, `bit>=136`, `truncated=0`.

This double check does not create a submit-safe gain by itself. It narrows the next work to exact no-loss synthesis gates and one short transfer experiment.

