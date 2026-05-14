# KG1 V373 HF Jasonkung98 Dataset Audit

Source: https://huggingface.co/datasets/jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge

## Verdict

This dataset is an exact public mirror of the official Kaggle challenge files already present in `C:\Users\davis\Downloads\nvidia-nemotron-model-reasoning-challenge.zip`.

It is useful as a small, HF-accessible source for reproducible CPU/HF jobs, but it is not a new signal source and does not add hidden-test labels or new measured accuracy.

## Repository Metadata

- Repo SHA: `0c8b90ec46067cdd943a83771be602c4c86092e2`
- Updated: `2026-03-22 15:26:32+00:00`
- License tag: `apache-2.0`
- Size tag: `1K<n<10K`
- Format: CSV

## File Hashes

| File | HF bytes | HF SHA256 | Official ZIP SHA256 | Match |
| --- | ---: | --- | --- | --- |
| `train.csv` | `3,069,304` | `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa` | `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa` | yes |
| `test.csv` | `1,461` | `c59d7eb0464b0a872a0c3f81e60cd6643fc1932a2dedaa05972bfd02cc638589` | `c59d7eb0464b0a872a0c3f81e60cd6643fc1932a2dedaa05972bfd02cc638589` | yes |

## Content Audit

`train.csv`:

- Rows: `9500`
- Columns: `id`, `prompt`, `answer`
- Duplicate ids: `0`
- Null prompts/answers: `0`

`test.csv`:

- Rows: `3`
- Columns: `id`, `prompt`
- All `3` sample test prompts exactly overlap train rows. This is the public sample test file, not the hidden scoring set.

## Family Counts In Train

| Family | Rows |
| --- | ---: |
| `bit_manipulation` | `1602` |
| `gravity_numeric` | `1597` |
| `unit_conversion` | `1594` |
| `cipher` | `1576` |
| `numeral` | `1576` |
| `equation_transform` | `1555` |

## Relevance To Current Roadmap

- Safe use: as a mirror/fallback input for HF jobs when Kaggle files are not mounted.
- Safe use: regenerate family counts, prompt hashes, fixtures, and solver probes.
- Unsafe assumption: treating this as new external knowledge. It is only the official public data already available.
- Not useful for immediate submit: no new hidden-test answers and no measured adapter-only gain.
- Required guardrail: keep existing anti-leakage checks by `id`, `prompt_sha256`, family counts, and exact hash.

## Action

Keep this source as infrastructure evidence only. It does not justify a new GPU training job by itself.
