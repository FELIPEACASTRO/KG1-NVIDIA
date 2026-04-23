# Kaggle Metric — Reverse Engineering + Exploits (NVIDIA Nemotron Reasoning Challenge)

Status: `2026-04-22` — derived from the OFFICIAL metric source
(`C:/Users/davis/AppData/Local/Temp/tc/metric_official_exact.py`), the
`ultra_consensus_report.md` (15 APIs, 10 effective) and the `train_analysis.md`
devastator pass over 9,500 rows.

This document is the single source of truth for the V81 canonicalization work.

---

## 1. Official metric — verbatim excerpt

```python
def extract_final_answer(text: str | None) -> str:
    if text is None:
        return 'NOT_FOUND'

    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:\uff1a]\s*([^\n]+)',
        r'final answer\s*[:\uff1a]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()
```

Scoring loop: `num_correct += int(verify(str(gt), str(extract_final_answer(raw))))`
then `score = num_correct / len(solution)`. No partial credit, no per-family
weighting — every row counts the same.

---

## 2. Validated bugs (with empirical tests)

| # | Name | Evidence | Consensus |
|---|---|---|---|
| B1 | Nested-brace truncation | `\\boxed\{([^}]*)` stops at first `}` so `\boxed{\frac{1}{2}}` extracts `\frac{1` | 10/10 |
| B2 | Binary-decimal collision | `re.fullmatch('[01]+', stored)` forces strict compare on `'10'`, `'100'`, `'1000'` — breaks rel_tol | 5/10 |
| B3 | Scientific notation loss | fallback regex `-?\d+(?:\.\d+)?` does not match `1e-3` | 8/10 |
| B4 | Last-boxed wins | `non_empty[-1]` means a post-thought self-correction overrides the correct answer | 10/10 |
| B5 | Unclosed boxed | alternation `(?:\}|$)` eats to EOS when the model fails to close the brace | 6/10 |
| B6 | Trailing punctuation | `.strip()` only whitespace — `XLVII.` != `XLVII` | 8/10 |
| B7 | Unit suffix breaks float cast | `float("24 cm")` raises; string compare fails too | 9/10 |
| B8 | Thousand separator | `verify('1234','1,234') = False` | 6/10 |
| B9 | Last-number fallback hijack | `"Result 42. Confidence 0.93"` extracts `0.93` | 5/10 |
| B10 | Large-integer float precision | `verify('9007199254740993','9007199254740992') = True` (irrelevant for competition) | 1/10 |

Bug B1 makes **94 of 1,555 equation_transform rows physically
unwinnable** when the gold contains a literal `}` — the only recovery is to
**skip boxed entirely** and use `Final answer is: X` (variant C hits 100%).

---

## 3. Validated exploits

| Code | Name | Shape |
|---|---|---|
| X1 | 1% numeric tolerance | `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` — any number in `[0.99X, 1.01X]` passes (NOT binary-shaped ints) |
| X2 | Last-boxed wins (can be used for k-vote) | emit `\boxed{v1}` ... `\boxed{vK}`, the final one counts |
| X3 | Case-insensitive compare | `\boxed{xlvii}` passes when gold is `XLVII` — useful for cipher/roman |
| X4 | "Final answer is:" bypass for B1 | emit the line before the `\boxed{}` to bypass nested-brace truncation |

---

## 4. Format best-practices (per family)

### 4.1 bit_manipulation (16.9% of rows, all 8-bit binary)

- Exactly 8 binary digits, zero-padded. `01010001` not `1010001` (B2 triggers on gold `10`).
- Emit as `\boxed{01010001}` — no spaces, no leading `0b`.
- SFT rule: `format(int(v), "08b")` every training label.

### 4.2 text_encryption (16.6%, lowercase phrases with spaces)

- Lowercase ASCII only (gold is lowercase; verify is case-insensitive anyway).
- Keep internal spaces, strip trailing punctuation (B6).
- Emit as `\boxed{the quick brown fox}` — no quotes, no `\text{}` wrappers.

### 4.3 equation_transform (16.4%, mixed int/symbolic, BRACE TRAP)

- Prefer emitting the `Final answer is: X` line ABOVE the `\boxed{}` — bypasses B1.
- If the label is numeric, strip all LaTeX cosmetic wrappers.
- If the label is symbolic (55.8% of this family), **do not** wrap in `\frac`, `\text`, `\mathrm`.

### 4.4 gravity_constant, unit_conversion (33.6% combined, all floats)

- Plain decimal with enough precision for rel_tol=1e-2 (2 decimals is usually enough).
- No units (`m/s`, `kg`, etc.) — B7 breaks `float()`.
- No thousand separators (B8). No scientific notation (B3).

### 4.5 numeral_system (16.6%, Roman numerals)

- Uppercase roman numerals (or lowercase — verify is case-insensitive).
- Strip trailing punctuation.
- No parentheses, no `\text{}`.

---

## 5. Calibration: local -> Kaggle

Consensus (9/12 APIs): `Kaggle ≈ local - 0.02`.

Root-cause breakdown (weighted):

- 50% sampling noise (Kaggle runs with temperature=1.0 by default).
- 33% test distribution shift (train.csv != test.csv slice).
- 17% extractor mismatch (our old local gate had a stricter boxed regex).

With canonicalization + `--temperature 0.0` + exactly-one-terminal-boxed, the
V80 Mega runs show the gap closing to `local - 0.005` to `local - 0.015`.

Random-Forest pre-score (seeded on historical local/Kaggle pairs) is
implemented in `scripts/kg1_prescore_rf.py` and outputs a
`predicted_kaggle_score` + `risk_flags` list for each adapter checkpoint.

---

## 6. Expected gain table (canonicalization only, per family, vs V80 MEGA baseline)

| Family | V80 raw | V81 canon | delta | source |
|---|---:|---:|---:|---|
| bit_manipulation | 0.801 | ~0.92 | +0.12 | fix zfill(8), strip .0 (B2) |
| text_encryption | ~0.42 | ~0.58 | +0.16 | trim punctuation (B6), lowercase (X3) |
| equation_transform | 0.122 | ~0.48 | +0.36 | "Final answer is:" bypass (B1) |
| gravity_constant | ~0.88 | ~0.93 | +0.05 | strip units (B7), no sci-notation (B3) |
| numeral_system | ~0.91 | ~0.95 | +0.04 | trim punctuation (B6) |
| unit_conversion | ~0.89 | ~0.94 | +0.05 | strip units (B7) |
| **overall** | **0.68** | **~0.79** | **+0.11** | weighted (all equal, 16.67%) |

These are the upper-bound gains from FORMATTING alone, assuming the underlying
semantic accuracy is unchanged. Any further gain must come from better SFT data
or semantic improvements in the solver.

---

## 7. Pipeline integration hooks

The V81 notebook (`notebooks/KG1_V81_CANONICALIZED.ipynb`) wires the helpers in
at three points:

1. **Dataset build (pre-train)**: `kg1_sft_format_validator.py` blocks the
   adapter training if clean_rate < 0.98 on the prepared JSONL.
2. **Post-inference (Kaggle submit)**: every `raw_output` from vLLM passes
   through `canonicalize_answer(raw, family_hint=detect_family(prompt))` before
   it is written to `submission.csv`.
3. **Pre-submit gate**: `kg1_prescore_rf.prescore_submission` produces a
   `predicted_kaggle_score`; we abort the submission if the prediction is below
   the 99%-certainty threshold described in the `feedback_99percent_rule.md`
   memory.

---

## 8. References

- `C:/Users/davis/AppData/Local/Temp/tc/metric_official_exact.py` — the exact metric source.
- `C:/Users/davis/AppData/Local/Temp/tc/submission_demo_exact.py` — submission flow.
- `C:/Users/davis/AppData/Local/Temp/tc/chat_template_analysis.md` — tokenizer/chat template.
- `C:/Users/davis/AppData/Local/Temp/tc/train_analysis.md` — dataset-level fingerprint.
- `C:/Users/davis/AppData/Local/Temp/tc/ultra_consensus_report.md` — 15-API consensus.
