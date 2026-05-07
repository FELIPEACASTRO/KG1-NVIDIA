# V216 Score Push Dataset

Status: **PASS**

## Train
- rows: 10210
- path: `C:\Users\davis\Workspace\KG1 -NVIDIA\data\v216\v216_score_push_train.jsonl`
- sha256: `8cfd065c102187b12c131aae7475c35e28073721175b4e6108004b0afc4d5d03`
- family_counts: `{'bit_manipulation': 2699, 'equation_transform': 6935, 'gravity_constant': 144, 'numeral_system': 144, 'text_encryption': 144, 'unit_conversion': 144}`

## Validation
- rows: 681
- path: `C:\Users\davis\Workspace\KG1 -NVIDIA\data\v216\v216_score_push_val.jsonl`
- sha256: `80efe71260c8589b998699543c85aff3ff140bc90e431dfa0ec33bce3e0921c0`
- family_counts: `{'bit_manipulation': 164, 'equation_transform': 453, 'gravity_constant': 16, 'numeral_system': 16, 'text_encryption': 16, 'unit_conversion': 16}`

## Recommended Gates
- Promote only if weak total >= 193, equation_transform >= 60, bit_manipulation >= 133.
- Run full validation only after weak gate passes.
- Package only if full validation >= 831 and truncation <= 4.

## Recommended Training Env
- SOURCE_WEIGHTS: `v216_synthetic_kg1_symbolic_rules=0.55,v216_synthetic_kg1_numeric_rules=0.45,v216_synthetic_kg1_bit_rules=0.25,v216_base_clean_safe_strict_equation=0.85,v216_base_clean_safe_strict_bit=0.75,v215_replay_anchor=0.9`
- SUBCATEGORY_WEIGHTS: `equation_symbolic_binary=1.2,equation_symbolic_unary=1.2,equation_numeric=0.85,bit_manipulation=0.75`

