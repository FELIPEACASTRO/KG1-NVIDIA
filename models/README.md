# Machine Learning models for Kaggle Nemotron metric pre-score prediction

This directory contains trained models that predict whether a model output will pass the Kaggle scoring metric.

## Models

| File | Type | Brier | Size | Recommended |
|---|---|---|---|---|
| ml_lgb_metric_consensus.pkl | LightGBM | 0.00178 | 1.7 MB | **PRODUCTION** (best Brier + fast) |
| ml_xgb_metric_consensus.pkl | XGBoost | 0.00180 | 855 KB | Alt fast |

## Performance

- Accuracy: 99.81%
- Precision: 99.55%
- Recall: 100.00%
- F1: 99.78%
- ROC-AUC: 0.9999
- Brier: 0.00178 (LGB)

Dataset: 66,500 samples (9500 train.csv rows × 7 format variants).
Cross-validation: GroupKFold(5) by row_id, 0.9981 ± 0.0004 acc.

## Top 5 features (by avg importance across 4 models)

1. output_length_chars (0.235)
2. has_trailing_dot_zero (0.157) - binary collision fix
3. has_comma_in_output (0.136) - thousand separator
4. boxed_is_empty (0.088) - blocks fallback
5. has_latex_command (0.057) - nested brace truncation

## Usage

```python
from scripts.kg1_ml_ensemble_prescore import MLEnsemblePrescorer
p = MLEnsemblePrescorer(lightweight=True)  # loads LGB + XGB only
prob = p.predict(features_dict)
```

## Feature list

See `ml_ensemble_metadata.json` for the complete 30 features.

## Report

See `../docs/ML_ENSEMBLE_REPORT.md` for full analysis.

