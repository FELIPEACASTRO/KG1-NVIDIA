# V376 Andy279 SFT File Audit

## Verdict

- `competition_train.csv` is byte-identical to the official Kaggle train file already audited.
- `sft_reconstructed.jsonl` is a full 9500-row public-train SFT reconstruction. Under the project extractor/scorer, all `9500/9500` rows match official train labels.
- ZIP contains additional diagnostics/traces (`problems.jsonl`, `corpus.jsonl`, `generation.jsonl`, `sft_train_reconstructed.jsonl`) that are useful for CPU triage and hard-negative analysis, but they do not provide hidden-test labels or submit-ready gains.
- This source may justify a CPU/tokenization/data-quality gate, not an immediate HF GPU job.
- The local Portuguese report contained a plaintext Hugging Face token. Generated artifacts were redacted; do not commit the original report or any raw secret-bearing copy.

## Key Counts

- train rows: `9500`; families: `{'bit_manipulation': 1602, 'cipher': 1576, 'numeral': 1576, 'unit_conversion': 1594, 'gravity': 1597, 'equation_transform': 1555}`
- standalone SFT rows: `9500`; known train ids: `9500`; project-scorer mismatches: `0`
- naive boxed-regex mismatch count was `173`, but this is a false audit signal caused by symbolic/braced answers and is not the metric to use.
- standalone SFT categories: `{'bit_manipulation': 1602, 'cipher': 1576, 'numeral': 1576, 'unit_conversion': 1594, 'gravity': 1597, 'cryptarithm_deduce': 659, 'equation_numeric_deduce': 596, 'cryptarithm_guess': 164, 'equation_numeric_guess': 136}`
- standalone SFT statuses: `{'rule_found': 8333, 'hypothesis_formed': 249, 'rule_unknown': 918}`
- generation rows: `9500`; by family: `{'bit_manipulation': {'rows': 1602, 'any_correct': 153, 'latest_correct': 144}, 'cipher': {'rows': 1576, 'any_correct': 573, 'latest_correct': 525}, 'numeral': {'rows': 1576, 'any_correct': 1531, 'latest_correct': 1522}, 'unit_conversion': {'rows': 1594, 'any_correct': 1259, 'latest_correct': 1217}, 'gravity': {'rows': 1597, 'any_correct': 1055, 'latest_correct': 995}, 'equation_transform': {'rows': 1555, 'any_correct': 191, 'latest_correct': 186}}`
- corpus rows: `17963`; included counts: `{'True': 17963}`; categories: `{'matching': 4515, 'bit_manipulation': 1602, 'cipher': 1576, 'numeral': 1576, 'unit_conversion': 1594, 'splitting': 1500, 'spelling': 648, 'gravity': 1597, 'cryptarithm_deduce': 659, 'concatenation': 1500, 'equation_numeric_deduce': 596, 'lstrip': 300, 'cryptarithm_guess': 164, 'equation_numeric_guess': 136}`

## Actionable Use

- Use only after anti-leakage by `problem_id/id`, prompt hash, family counts, and exact train hash.
- Do not train broad SFT directly: previous broad trace transfer attempts lowered ACC despite lower loss.
- Potential V377 path: filter only `bit_manipulation`/`equation_transform` rows with `status=rule_found`, short enough traces, and exact official boxed answer; then run tokenization/offset-mask gate before any HF.
- Use `generation.jsonl` as hard-negative/confidence metadata, especially rows with failed extracted answers despite known label.
- Treat `sft_train_reconstructed.jsonl` synthetic/unknown rows as blocked until they are classified by prompt hash/family and checked for leakage/domain drift.

## Files
- `competition_train_csv`: bytes `3069304`, sha256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`
- `sft_reconstructed_jsonl`: bytes `54328626`, sha256 `e385421e85881a406144e2e1166961ef21b0b9063077c908f41e626e01681609`
- `sft_md`: bytes `4347`, sha256 `ccedc438981d8aa5a1b1b659f7c8d6bd58af6eda4329838c32dd4b132436a3e1`
- `zip`: bytes `41281191`, sha256 `79a68541451bd6d7c13b6a97f9da424df80727827d593723e0eb5ca11d3d5a26`
- `report_md`: bytes `5551`, sha256 `6b16e14312a965b918cf04748c6428a73e50d8429dfe70e6652e9ff806234c14`

## Metric Revalidation With Project Extractor

- `sft_reconstructed.jsonl`: `9500/9500` metric-correct, `0` wrong, `0` no final answer.
- By family: `{'bit_manipulation': {'rows': 1602, 'correct': 1602, 'wrong': 0}, 'text_encryption': {'rows': 1576, 'correct': 1576, 'wrong': 0}, 'numeral_system': {'rows': 1576, 'correct': 1576, 'wrong': 0}, 'unit_conversion': {'rows': 1594, 'correct': 1594, 'wrong': 0}, 'gravity_constant': {'rows': 1597, 'correct': 1597, 'wrong': 0}, 'equation_transform': {'rows': 1555, 'correct': 1555, 'wrong': 0}}`
- `sft_train_reconstructed.jsonl`: known official rows `9500`, synthetic/unknown rows `8463`, metric-correct known `9500`, wrong known `0`.
- `problems.jsonl` submissions under project scorer: `8333/9500` correct; by family `{'bit_manipulation': {'rows': 1602, 'correct': 1364, 'wrong': 238}, 'text_encryption': {'rows': 1576, 'correct': 1576, 'wrong': 0}, 'numeral_system': {'rows': 1576, 'correct': 1576, 'wrong': 0}, 'unit_conversion': {'rows': 1594, 'correct': 1594, 'wrong': 0}, 'gravity_constant': {'rows': 1597, 'correct': 1597, 'wrong': 0}, 'equation_transform': {'rows': 1555, 'correct': 626, 'wrong': 929}}`.

Decision: never use raw `problems.submission` as labels. For SFT, filter by project extractor correctness, family, status, trace length, and prompt/hash gates before any tokenization/HF job.
