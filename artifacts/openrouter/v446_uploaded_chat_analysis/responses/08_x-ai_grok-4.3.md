# x-ai/grok-4.3

variant: `x-ai/grok-4.3-20260430`

cost: `0.0049331` duration_ms: `2212` tokens: `940`

```json
{
  "verdict": "No justified next GPU run exists under current constraints. All proposed LoRA paths repeat documented failure modes.",
  "key_diagnosis": "Equation_transform failures are dominated by generation errors (79 boxed_wrong + 18 literal_nonfinal). All prior SFT and preference runs on rule-derived or hard-negative data produced either truncation, bit regression, or no equation lift. No non-leaking CPU signal has been shown to survive LoRA distillation without violating the promotion gates.",
  "ranked_next_experiments": [],
  "required_cpu_gates": [
    "Must produce >=201 weak total on held-out public rows with equation >=62, bit >=136, truncated=0 using only CPU inference + exact string match to public labels.",
    "Artifact must be exactly 100 rows of (prompt, final boxed answer) pairs with zero tokenization/offset issues and provable 0 leakage to test set families.",
    "Exact row hashes and family counts must be published before any GPU request."
  ],
  "no_go_conditions": [
    "Any experiment that re-uses SFT, preference, or reconstructed data from V397–V445 families",
    "Any CPU-generated data that cannot first beat 192/315 + equation 60 on the exact public weak set with strict boxed match",
    "Any plan requiring >0.5 GPU hours before a passing CPU gate",
    "Any data selection that uses weak or full labels to filter rows"
  ],
  "changes_to_current_gates": [
    "Add mandatory CPU-only weak-set audit with exact string match before every GPU request (currently missing formal gate).",
    "Raise equation promotion threshold to >=62/155 before any H200 spend (current 60 is too permissive given observed variance).",
    "Enforce one-hour H200 hard kill after checkpoint-1 regardless of internal metrics."
  ],
  "one_hour_h200_precondition": "A CPU artifact that first achieves >=201/315 weak (equation >=62, bit >=136, truncated=0) on the exact public weak set using only the allowed adapter inference path and exact string match, with zero row overlap to any prior training data."
}
```
