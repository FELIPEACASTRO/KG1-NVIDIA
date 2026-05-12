# Local Bit Solver Audit Against Tong Hui Kang Strategy

Date: 2026-05-12

## Source

- Official Kaggle competition download via `kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge`.
- Raw files were downloaded only into `artifacts/tmp_v296_official_kaggle_download` for audit and must not be kept as long-term workspace data.
- Observed `train.csv` SHA256: `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`.

## Local Solver Measured Result

Command target: `src/solvers/bit_manipulation_solver.py` imported as `BitManipulationSolver`.

- Official train rows: `9500`.
- Bit rows detected: `1602`.
- Correct: `1265/1602`.
- Accuracy: `78.96379525593009%`.
- Wrong: `337`.
- Runtime: about `51.2s` on local CPU.
- Global-rule count reported by current solver: `1053`.

## External Public Claims For Comparison

From Kaggle discussion `690307`, retrieved through authenticated Kaggle SDK:

- Tong Hui Kang reports `1364/1602 = 85.1%` bit manipulation solved by his algorithm.
- A comment by Taha reports `1584/1602 = 98.9%` bit manipulation solved.

## Gap

The current KG1 local bit solver is materially below the public algorithmic target:

- Gap to Tong Hui Kang reported solver: `99` train rows.
- Gap to Taha reported solver: `319` train rows.

The current implementation uses global transforms, per-bit unary matching, and binary consensus, but does not fully replicate the public `bitsum + stride + middle-fill` chain-of-thought algorithm.

## Decision

Create a V296 CPU-only audit before any HF GPU spend:

1. Reimplement the `18` unary + `336` binary per-output-bit candidate grid.
2. Use bitsum hashes for candidate/output columns.
3. Implement left/right stride matching and deterministic middle fill.
4. Measure against official train and weak/full local rows.
5. Use any new verified bit coverage only as verifier/teacher data for adapter-only training, unless competition packaging rules later permit solver code.
