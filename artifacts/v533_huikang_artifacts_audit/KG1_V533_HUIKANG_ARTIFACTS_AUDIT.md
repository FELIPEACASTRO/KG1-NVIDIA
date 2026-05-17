# KG1 V533 Huikang Artifacts Audit

- zip: `C:\Users\davis\Downloads\archive (9).zip`
- zip_size: `1495393103`
- sha256: `5dc5cf5e252d45cf9ef54b7fd4e13ed7ed91bbe6742ddebf5d9c833ccbcc2f46`
- entries: `47947`
- uncompressed_bytes: `1989183041`
- Kaggle source: `https://www.kaggle.com/datasets/samvalladares/huikang-nemotron-artifacts/versions/7`
- Metadata note: the Kaggle metadata mentions `10,000` synthesized 3-input bit traces, but the local `archive (9).zip` contains `2000` rows in `bit_manip_3input_synthesized_traces.jsonl`. This audit uses the local file contents as ground truth.

## Adapter v26 Static Header

- base_model_name_or_path: `None`
- r/alpha/dropout: `32` / `32` / `0`
- target_modules: `all-linear`
- target_parameters: `None`
- tensor_count: `418`
- dtype_counts: `{'F32': 418}`
- total_params: `386072576`
- key_group_counts: `{'out_proj': 46, 'experts': 230, 'down_proj': 46, 'up_proj': 46, 'lm_head': 2}`
- non_lora_key_count: `0`

## JSONL Corpus Signals

| file | rows | official overlap | mismatches | boxed rows | family/category signal | decision |
|---|---:|---:|---:|---:|---|---|
| `bit_manip_3input_synthesized_traces.jsonl` | 2000 | 0 | 0 | 2000 | `[('bit_like', 2000)]` / `[('bit_manipulation', 2000)]` | P0 bit trace reference; mine source-only trace style and operators |
| `bit_manipulation_3input_traces.jsonl` | 100 | 100 | 0 | 100 | `[('bit_like', 100)]` / `[('bit_manipulation', 100)]` | P0 bit trace reference; mine source-only trace style and operators |
| `corpus.jsonl` | 15979 | 6355 | 0 | 0 | `[('other', 8275), ('bit_like', 2059), ('cipher_like', 1768), ('equation_like', 1147)]` / `[('matching', 4515), ('bit_manipulation', 2059), ('cipher', 1756), ('splitting', 1500), ('concatenation', 1500)]` | P1 broad corpus reference; filter before use |

## Weak Overlap Diagnostic

This is a diagnostic only; weak rows must not be copied directly into training
packs used to judge promotion.

- `bit_manipulation_3input_traces.jsonl` overlaps `10` weak bit rows.
- Of those, `8` are current baseline misses:
  - `048cc279`
  - `1a7c8520`
  - `1abaffca`
  - `5ba26f21`
  - `7192535b`
  - `a6192d29`
  - `b8722d19`
  - `b8aa3072`
- Weak-overlap rule mix:
  - `MAJ`: `6`
  - `CHO`: `4`
- Full official 100-row rule mix:
  - `MAJ`: `52`
  - `CHO`: `48`
- Synthetic 2000-row rule mix:
  - `CHO`: `1078`
  - `MAJ`: `922`
- `corpus.jsonl` overlaps `136` weak bit rows and `78` weak equation rows.
  It includes only `1` current weak bit miss (`4ada9150`), and `22` current
  weak equation misses. The bit rows have very long unmasked token counts and
  are not a clean short-trace source without filtering.

## Decision

- Do not extract or commit `adapter_v26_model.safetensors`; it is a large external adapter and must go through adapter-only weak eval if used.
- Use `bit_manip_3input_synthesized_traces.jsonl` and the non-weak part of
  `bit_manipulation_3input_traces.jsonl` as P0 references for CHO/MAJ 3-input
  bit trace style/operator coverage.
- Use `corpus.jsonl` only after filtering by official contract and family; it is broad and not automatically better than our current curated packs.
- No direct submit-safe ACC gain is claimed by this audit; it creates actionable source material for CPU trace/gate work.
