# V377 Nemotron Hacker Dataset Audit

## Verdict

- ZIP was streamed in place; no large extraction was left on disk.
- The package contains useful SFT/trajectory material, but the attached report describes gated-access bypass and 'leak' provenance. That makes the data compliance-sensitive.
- Technically useful rows must remain blocked until legal/rules provenance is cleared and the CPU data gate proves no leakage/domain drift.
- No HF GPU job, package, or Kaggle submit is authorized by this audit.

## File Hashes

- `zip`: bytes `94515464`, sha256 `6272e30e717c1b257801b82ae98a76980ecb14845f3306862a9be0a888ee476f`
- `report_md`: bytes `6884`, sha256 `869a7e0b1f294c48803bc4cdc2014baecb421a42c3b6676a1d3f68363c8b6314`
- `train_csv`: bytes `3069304`, sha256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`

## ZIP Contents

- `kaggle_trajectories/nemotron_traj.csv`: bytes `139456207`, sha256 `01da9b309daedf18c9bcff9e0766b3deb7d736a1d350c73ded47775a8b66685e`
- `sft_train_reconstructed.jsonl`: bytes `96207995`, sha256 `1434811d65cbf68fd17d391cb2fce77426183666f59d7cf6ca4fddf8b4fe5a91`
- `sft_train_converted.jsonl`: bytes `59672150`, sha256 `69aee7ddaed6b7d55a6792f03ad960e44a10e1974cffe2f184514646f68fd228`
- `sft_train_full_9500.jsonl`: bytes `57034518`, sha256 `8b5cead5c539a81761ec00cb43aa32ae5f2b277a4d7a252b98135d5693912a35`
- `kaggle_sft_data/dataset_generated.csv`: bytes `48670122`, sha256 `dd3160e264585cb1902cdf481c31ac62a2d945a0eb481068eee9f45a56351f0a`
- `HACKER_REPORT.md`: bytes `6884`, sha256 `869a7e0b1f294c48803bc4cdc2014baecb421a42c3b6676a1d3f68363c8b6314`
- `README.md`: bytes `4347`, sha256 `ccedc438981d8aa5a1b1b659f7c8d6bd58af6eda4329838c32dd4b132436a3e1`

## Measured Dataset Signals

### `sft_train_converted.jsonl`

- rows: `8703`
- known official train rows: `8703`
- unique known official IDs: `7044`; duplicate known-ID rows: `1659`
- unknown/synthetic rows: `0`
- project metric correct: `8703`; wrong: `0`
- base_loss stats: `{'min': 0.1318104539082245, 'p50': 0.4474952736446086, 'p90': 1.2121590133261892, 'p99': 1.561454154349662, 'max': 1.7144912010015445, 'mean': 0.5355024396995169}`
- by family: `{'bit_manipulation': {'rows': 1754, 'correct': 1754, 'wrong': 0, 'unknown': 0}, 'equation_transform': {'rows': 2438, 'correct': 2438, 'wrong': 0, 'unknown': 0}, 'text_encryption': {'rows': 1656, 'correct': 1656, 'wrong': 0, 'unknown': 0}, 'unit_conversion': {'rows': 1070, 'correct': 1070, 'wrong': 0, 'unknown': 0}, 'gravity_constant': {'rows': 1055, 'correct': 1055, 'wrong': 0, 'unknown': 0}, 'numeral_system': {'rows': 730, 'correct': 730, 'wrong': 0, 'unknown': 0}}`
- unique IDs by family: `{'bit_manipulation': 1354, 'equation_transform': 1499, 'gravity_constant': 975, 'numeral_system': 650, 'text_encryption': 1576, 'unit_conversion': 990}`

### `sft_train_full_9500.jsonl`

- rows: `9500`
- known official train rows: `9500`
- unique known official IDs: `9500`; duplicate known-ID rows: `0`
- unknown/synthetic rows: `0`
- project metric correct: `9500`; wrong: `0`
- by family: `{'bit_manipulation': {'rows': 1602, 'correct': 1602, 'wrong': 0, 'unknown': 0}, 'text_encryption': {'rows': 1576, 'correct': 1576, 'wrong': 0, 'unknown': 0}, 'numeral_system': {'rows': 1576, 'correct': 1576, 'wrong': 0, 'unknown': 0}, 'unit_conversion': {'rows': 1594, 'correct': 1594, 'wrong': 0, 'unknown': 0}, 'gravity_constant': {'rows': 1597, 'correct': 1597, 'wrong': 0, 'unknown': 0}, 'equation_transform': {'rows': 1555, 'correct': 1555, 'wrong': 0, 'unknown': 0}}`
- unique IDs by family: `{'bit_manipulation': 1602, 'equation_transform': 1555, 'gravity_constant': 1597, 'numeral_system': 1576, 'text_encryption': 1576, 'unit_conversion': 1594}`

### `sft_train_reconstructed.jsonl`

- rows: `17963`
- known official train rows: `9500`
- unique known official IDs: `9500`; duplicate known-ID rows: `0`
- unknown/synthetic rows: `8463`
- project metric correct: `9500`; wrong: `0`
- by family: `{'bit_manipulation': {'rows': 1602, 'correct': 1602, 'wrong': 0, 'unknown': 0}, 'text_encryption': {'rows': 1585, 'correct': 1576, 'wrong': 0, 'unknown': 9}, 'numeral_system': {'rows': 1576, 'correct': 1576, 'wrong': 0, 'unknown': 0}, 'unit_conversion': {'rows': 1594, 'correct': 1594, 'wrong': 0, 'unknown': 0}, 'gravity_constant': {'rows': 1602, 'correct': 1597, 'wrong': 0, 'unknown': 5}, 'equation_transform': {'rows': 1555, 'correct': 1555, 'wrong': 0, 'unknown': 0}, 'unknown': {'rows': 8449, 'correct': 0, 'wrong': 0, 'unknown': 8449}}`
- unique IDs by family: `{'bit_manipulation': 1602, 'equation_transform': 1555, 'gravity_constant': 1597, 'numeral_system': 1576, 'text_encryption': 1576, 'unit_conversion': 1594}`

### `HACKER_REPORT.md`

- rows: `None`

### `README.md`

- rows: `None`

### `kaggle_sft_data/dataset_generated.csv`

- rows: `9500`
- known official train rows: `9500`
- unique known official IDs: `9500`; duplicate known-ID rows: `0`
- answer labels vs official: matches `9500`, mismatches `0`
- by family: `{'bit_manipulation': {'rows': 1602, 'correct': 1602, 'wrong': 0, 'unknown': 0}, 'text_encryption': {'rows': 1576, 'correct': 1576, 'wrong': 0, 'unknown': 0}, 'numeral_system': {'rows': 1576, 'correct': 1576, 'wrong': 0, 'unknown': 0}, 'unit_conversion': {'rows': 1594, 'correct': 1594, 'wrong': 0, 'unknown': 0}, 'gravity_constant': {'rows': 1597, 'correct': 1597, 'wrong': 0, 'unknown': 0}, 'equation_transform': {'rows': 1555, 'correct': 1555, 'wrong': 0, 'unknown': 0}}`
- unique IDs by family: `{'bit_manipulation': 1602, 'equation_transform': 1555, 'gravity_constant': 1597, 'numeral_system': 1576, 'text_encryption': 1576, 'unit_conversion': 1594}`
- generated CoT by family: `{'bit_manipulation': {'rows': 1602, 'correct': 1364, 'wrong': 238, 'unknown': 0}, 'text_encryption': {'rows': 1576, 'correct': 1576, 'wrong': 0, 'unknown': 0}, 'numeral_system': {'rows': 1576, 'correct': 1576, 'wrong': 0, 'unknown': 0}, 'unit_conversion': {'rows': 1594, 'correct': 1594, 'wrong': 0, 'unknown': 0}, 'gravity_constant': {'rows': 1597, 'correct': 1597, 'wrong': 0, 'unknown': 0}, 'equation_transform': {'rows': 1555, 'correct': 1490, 'wrong': 65, 'unknown': 0}}`

### `kaggle_trajectories/nemotron_traj.csv`

- rows: `9500`
- known official train rows: `9500`
- unique known official IDs: `9500`; duplicate known-ID rows: `0`
- generated answer vs official: correct `4542`, wrong `4958`
- generated by family: `{'bit_manipulation': {'rows': 1602, 'correct': 99, 'wrong': 1503, 'unknown': 0}, 'text_encryption': {'rows': 1576, 'correct': 525, 'wrong': 1051, 'unknown': 0}, 'numeral_system': {'rows': 1576, 'correct': 1522, 'wrong': 54, 'unknown': 0}, 'unit_conversion': {'rows': 1594, 'correct': 1217, 'wrong': 377, 'unknown': 0}, 'gravity_constant': {'rows': 1597, 'correct': 995, 'wrong': 602, 'unknown': 0}, 'equation_transform': {'rows': 1555, 'correct': 184, 'wrong': 1371, 'unknown': 0}}`

## Actionable Decision

- Treat `sft_train_full_9500.jsonl` and `sft_train_converted.jsonl` as candidate trace sources only after source/rules clearance.
- `sft_train_converted.jsonl`/`dataset_generated.csv` are interesting because they are filtered/logprob-style and smaller, but they must not bypass anti-leakage and tokenizer gates.
- `nemotron_traj.csv` is best used as hard-negative/confidence metadata, not as label source.
- The direct next step is still CPU-only V377/V378 filtered trace gate: prove rows, hashes, families, length, loss buckets, and no-loss weak behavior before any HF spend.

## Residual Equation Coverage

- V375 residual equation rows checked: `92`.
- `sft_train_converted.jsonl` coverage: `92/92`, all project-scorer correct.
- `sft_train_full_9500.jsonl` coverage: `92/92`, all project-scorer correct.
- `dataset_generated.csv` CoT coverage on those residuals: `91/92` project-scorer correct.
- Coverage CSV: `artifacts/v377_nemotron_hacker_dataset_audit/v377_v375_residual_coverage.csv`.

## Candidate Trace Counts

- `sft_train_converted.jsonl` unique IDs:
  - `bit_manipulation`: `1354` unique IDs, `1754` total rows, all best traces project-scorer correct.
  - `equation_transform`: `1499` unique IDs, `2438` total rows, all best traces project-scorer correct.
- Filter caveat:
  - Many bit traces are long; actual tokenizer gate is required before treating them as trainable.
  - For equation, `823` unique best traces are under the rough `7680` character threshold, but this is not a tokenization proof.
