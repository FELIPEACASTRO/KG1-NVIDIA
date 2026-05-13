# KG1 V333 Kienngx Kaggle + Attachment Audit - 2026-05-13

## Sources

Kaggle notebook:

- URL: `https://www.kaggle.com/code/kienngx/nemotron-sft-reasoning-trajectories-dataset`
- Kaggle id: `kienngx/nemotron-sft-reasoning-trajectories-dataset`
- Status via CLI: `COMPLETE`
- Kernel id number: `114879652`
- Docker image: `gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9`
- Machine shape: `NvidiaRtxPro6000`

Datasets referenced by the notebook:

- `kishanvavdara/nemotron-reasoning-traj`
- `mayukh18/nemotron-packages`
- `konbu17/nemotron-sft-lora-cot-selection`
- `dennisfong/nvidia-nemotron-offline-packages`
- `rubyducklove/nvidia-cutlass`

Local attachments:

- `C:\Users\davis\Downloads\tls.pdf`
- `C:\Users\davis\Downloads\tl.pptx`
- `C:\Users\davis\Downloads\2605.11666v1.pdf`

## Kaggle Notebook Findings

The notebook's core claim is data quality over volume:

- generate CoT / reasoning trajectories;
- verify final answer by rule-based scoring;
- keep only verified-correct samples;
- train Nemotron-3-Nano-30B-A3B-BF16 with LoRA.

Important implementation details extracted from the notebook:

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Training dataset path in code: `/kaggle/input/datasets/kishanvavdara/nemotron-reasoning-traj/nemotron_traj.csv`.
- The code filters `df[df['correctness']=='true']`.
- Type mapping:
  - `bit_manipulation` -> `Bit Manipulation`;
  - `equation_numeric` and `equation_symbolic` -> `Equation Transformation`.
- LoRA:
  - `r=32`;
  - `alpha=32`;
  - target modules regex `.*\.(in_proj|out_proj|up_proj|down_proj)$`;
  - dropout `0.05` during training, forced to `0.0` for package.
- SFT:
  - `num_train_epochs=2`;
  - `per_device_train_batch_size=2`;
  - `gradient_accumulation_steps=8`;
  - `learning_rate=1e-5`;
  - `max_length=4096`;
  - `packing=False`.

Risk:

- The markdown says the prepared dataset contains `607` bit rows and `200` equation rows.
- The executable code using `nemotron_traj.csv` samples only:
  - `128` `Bit Manipulation` rows;
  - `178` `Equation Transformation` rows.
- Therefore the notebook description and executable training recipe are not perfectly aligned.

## Dataset Triage

### `kishanvavdara/nemotron-reasoning-traj`

Downloaded file for triage only:

- File: `nemotron_traj.csv`
- Size: `139,456,207` bytes
- SHA256: `01da9b309daedf18c9bcff9e0766b3deb7d736a1d350c73ded47775a8b66685e`
- Rows: `9500`
- Columns:
  - `id`
  - `prompt`
  - `generated`
  - `correct answer`
  - `generated answer`
  - `correctness`
  - `problem type`

Family counts:

- `bit_manipulation`: `1602`
- `gravity`: `1597`
- `unit_conversion`: `1594`
- `numeral`: `1576`
- `cipher`: `1576`
- `equation_symbolic`: `823`
- `equation_numeric`: `732`

Correctness counts:

- `true`: `4423`
- `partial`: `339`
- `false`: `4738`

Correct rows in critical families:

- `bit_manipulation`: `128`
- `equation_numeric`: `176`
- `equation_symbolic`: `2`

Weak leakage risk:

- Overlap with canonical V221 weak IDs: `315/315`.
- This is expected because the CSV covers the official 9500 train rows.
- It cannot be used as raw SFT for our weak/full pipeline.

Decision:

- Use only as a source for template/trajectory analysis after hard anti-leak by `id` and normalized prompt hash.
- It is not a new deployable adapter path and not a direct training authorization.

### `konbu17/nemotron-sft-lora-cot-selection`

Downloaded only safe small files for triage:

- `train_split_with_cot.csv`
- `adapter_config.json`

Not downloaded:

- `adapter_model.safetensors` (`3,522,350,336` bytes).

`train_split_with_cot.csv`:

- Size: `9,361,848` bytes
- SHA256: `31a8667f23f68a1745cb777f63b32e0b32e8aea324f12935d23540cf5356d3e7`
- Rows: `6558`
- Unique IDs: `6558`
- Columns:
  - `id`
  - `prompt`
  - `answer`
  - `type`
  - `generated_cot`

Family counts:

- `Gravitational Constant`: `1511`
- `Numeral Conversion`: `1491`
- `Text Encryption`: `1407`
- `Unit Conversion`: `1342`
- `Bit Manipulation`: `607`
- `Equation Transformation`: `200`

Cross-check against `nemotron_traj.csv`:

- All `6558` IDs exist in the raw 9500-row trajectory file.
- Label mismatches against `correct answer`: `0`.
- But many selected CoTs correspond to raw `correctness=false` in `nemotron_traj.csv`:
  - `Bit Manipulation`: `514 false`, `14 partial`, `79 true`;
  - `Equation Transformation`: `80 false`, `6 partial`, `114 true`.

Interpretation:

- `train_split_with_cot.csv` is not simply the `correctness=true` subset of `nemotron_traj.csv`.
- It appears to contain solver-generated or separately curated CoTs for rows where the raw trajectory model was often wrong.
- This is useful as teacher trace material, but not safe for direct training without anti-leak.

Weak leakage risk:

- Overlap with canonical V221 weak IDs: `74/315`.
- By family:
  - `Bit Manipulation`: `56`;
  - `Equation Transformation`: `18`.

Decision:

- Direct SFT from this CSV is blocked for our pipeline.
- It can be mined after excluding weak/full IDs and normalized prompt hashes.
- For bit, the short `CERTAIN` bit-value traces are not enough by themselves; they should be replaced or enriched with Tong-style bit-pair/bitsum/stride trace.
- For equation, the 200 traces are valuable for DSL taxonomy, but only non-overlapping IDs can be used for synthetic rule generation.

## Attachment Findings

### `2605.11666v1.pdf`

This is the EvoTD paper:

- Title: `Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling`.
- Core method:
  - skill-based seeding;
  - parametric attribute mutation;
  - skill crossover;
  - multi-objective fitness check;
  - verifier/RLVR-style training signal;
  - Zone of Proximal Development learnability filter.

Direct KG1 implication:

- Use it as a design pattern for V335:
  - seed skills from V324/V329/V333/V334 accepted solvers;
  - mutate complexity while preserving rule class;
  - generate executable examples;
  - verify every generated sample;
  - reject trivial, impossible, ambiguous, or conflicting examples;
  - train only after CPU gates show new coverage.

### `tls.pdf` and `tl.pptx`

Both contain `Tensor Logic: The Language of AI` by Pedro Domingos.

Useful ideas:

- represent rules as transparent equations/programs;
- combine symbolic logic programming with tensor/einsum-style computation;
- backward chaining/forward chaining as execution strategies;
- transparent/reliable reasoning rather than opaque memorization.

Direct KG1 implication:

- This supports the existing solver/verifier-first roadmap.
- It does not provide immediate new data, adapter weights, or a family-specific algorithm.
- Practical use is P2:
  - model our DSL rules as auditable typed programs;
  - keep rule provenance and conflicts explicit;
  - avoid relying on LoRA loss as proof of correctness.

## Roadmap Decision

Concrete action to add:

1. V333 bit CPU gate remains priority:
   - implement Tong-style bit-pair/bitsum/stride solver;
   - use `konbu17` bit traces only as wording/template reference after anti-leak.

2. V334 equation DSL should mine non-overlapping `konbu17` equation traces:
   - extract rule classes from the 200 equation rows;
   - exclude weak/full IDs and prompt hashes;
   - compare with V324/V329 accepted classes;
   - add only conflict-free candidates.

3. V335 EvoTD-style fixture builder:
   - create synthetic tasks from accepted rules;
   - apply executor/verifier/ZPD filters;
   - block HF GPU until CPU gate shows new measurable signal.

Blocked:

- No direct training from `nemotron_traj.csv`.
- No direct training from `train_split_with_cot.csv`.
- No download of the `3.5GB` adapter unless a new weak gate or specific comparison requires it.
