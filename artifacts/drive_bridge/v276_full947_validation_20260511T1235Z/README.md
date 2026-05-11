# V276 full 947 validation bridge

Date: 2026-05-11

Source Drive file:

- `KG1_NVIDIA_V207A/output_v207a_acc_gate/validation/official_train_seed42_stratified10_val.csv`
- Drive file ID: `184zcN6JeFl_KA8-PhHD5ANBdWSOOEyKL`

Local validation:

- rows: `947`
- duplicate IDs: `0`
- SHA256: `84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935`
- columns: `id`, `prompt`, `answer`, `family`
- family counts:
  - `bit_manipulation`: `160`
  - `equation_transform`: `155`
  - `gravity_constant`: `159`
  - `numeral_system`: `157`
  - `text_encryption`: `157`
  - `unit_conversion`: `159`

HF private dataset upload:

- Dataset: `felipesp1983/kg1-nemotron-training`
- Prefix: `runtime_artifacts/v276_full_eval_bridge/v276-full947-bridge-20260511T1245Z/`
- Commit: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/1da6764b95966cc107ad4af8c5c0d6f7fdc6261d`
- Uploaded files:
  - `official_train_seed42_stratified10_val.csv`
  - `v276_full947_validation_manifest.json`

Local cleanup:

- The raw CSV was removed from the Git working tree after HF upload.
- Only this README and the validation manifest are kept locally for traceability.

