# V202D Strict Promotion Gate

Notebook generated:
- `notebooks\KG1_V202D_H100_A100_STRICT_PROMOTION_GATE_COLAB_PRO.ipynb`

Use after V202C passed A/B/C. V202D re-evaluates V194, A, and B on larger deterministic splits and packages only a selected candidate.

Policy:
- no training
- no Kaggle submit
- no promotion unless overall loss is <= V194 on all required splits
- no category regression above 0.0005
