"""
Kaggle Playground Series S6E3 - Predict Customer Churn
Metric: ROC AUC Score
Target: Top 1% solution with ensemble of LightGBM + XGBoost + CatBoost
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import optuna
import warnings
import gc
import os

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "C:/tmp/kaggle-ps-s6e3"
N_FOLDS = 10
SEED = 42
OPTUNA_TRIALS = 60

np.random.seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

train_id = train['id']
test_id = test['id']

# Target: convert Yes/No to 1/0
train['Churn'] = (train['Churn'] == 'Yes').astype(int)
y = train['Churn'].values

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Churn rate: {y.mean():.4f}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("Feature engineering...")

def feature_engineering(df):
    df = df.copy()

    # --- Binary encoding ---
    binary_maps = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
    }
    for col, mapping in binary_maps.items():
        df[col] = df[col].map(mapping)

    # --- Ternary service columns: Yes=2, No=1, No service=0 ---
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].map({'Yes': 2, 'No': 1, 'No phone service': 0, 'No internet service': 0})

    # --- Internet & Contract ---
    df['InternetService'] = df['InternetService'].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    # --- Payment method ---
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })

    # ============================
    # ENGINEERED FEATURES
    # ============================

    # Total services count
    df['NumServices'] = (
        df['PhoneService'] +
        (df['MultipleLines'] == 2).astype(int) +
        (df['OnlineSecurity'] == 2).astype(int) +
        (df['OnlineBackup'] == 2).astype(int) +
        (df['DeviceProtection'] == 2).astype(int) +
        (df['TechSupport'] == 2).astype(int) +
        (df['StreamingTV'] == 2).astype(int) +
        (df['StreamingMovies'] == 2).astype(int)
    )

    # Internet services count (only internet-related)
    df['NumInternetServices'] = (
        (df['OnlineSecurity'] == 2).astype(int) +
        (df['OnlineBackup'] == 2).astype(int) +
        (df['DeviceProtection'] == 2).astype(int) +
        (df['TechSupport'] == 2).astype(int) +
        (df['StreamingTV'] == 2).astype(int) +
        (df['StreamingMovies'] == 2).astype(int)
    )

    # Security services (protective)
    df['NumSecurityServices'] = (
        (df['OnlineSecurity'] == 2).astype(int) +
        (df['OnlineBackup'] == 2).astype(int) +
        (df['DeviceProtection'] == 2).astype(int) +
        (df['TechSupport'] == 2).astype(int)
    )

    # Streaming services
    df['NumStreamingServices'] = (
        (df['StreamingTV'] == 2).astype(int) +
        (df['StreamingMovies'] == 2).astype(int)
    )

    # Tenure interactions
    df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['tenure_TotalCharges'] = df['tenure'] * df['TotalCharges']

    # Average monthly charge over tenure
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Charge difference (current vs average)
    df['ChargeDiff'] = df['MonthlyCharges'] - df['AvgMonthlyCharge']

    # Tenure bins
    df['tenure_bin'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 72], labels=[0, 1, 2, 3, 4]).astype(float)

    # Monthly charges per service
    df['ChargePerService'] = df['MonthlyCharges'] / (df['NumServices'] + 1)

    # Has internet but no security
    df['NoProtection'] = ((df['InternetService'] > 0) & (df['NumSecurityServices'] == 0)).astype(int)

    # High-risk combo: month-to-month + fiber + electronic check
    df['HighRisk'] = (
        (df['Contract'] == 0) &
        (df['InternetService'] == 2) &
        (df['PaymentMethod'] == 0)
    ).astype(int)

    # Senior citizen interactions
    df['Senior_Contract'] = df['SeniorCitizen'] * df['Contract']
    df['Senior_MonthlyCharges'] = df['SeniorCitizen'] * df['MonthlyCharges']
    df['Senior_tenure'] = df['SeniorCitizen'] * df['tenure']

    # Family status
    df['HasFamily'] = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int)
    df['FullFamily'] = ((df['Partner'] == 1) & (df['Dependents'] == 1)).astype(int)

    # Tenure groups
    df['NewCustomer'] = (df['tenure'] <= 6).astype(int)
    df['LoyalCustomer'] = (df['tenure'] >= 48).astype(int)

    # Contract value (total expected)
    contract_months = df['Contract'].map({0: 1, 1: 12, 2: 24})
    df['ContractValue'] = contract_months * df['MonthlyCharges']

    # MonthlyCharges bins
    df['MC_bin'] = pd.qcut(df['MonthlyCharges'], q=10, labels=False, duplicates='drop')

    # Tenure * Contract interaction
    df['tenure_Contract'] = df['tenure'] * df['Contract']

    # Total charges ratio to expected
    df['ChargesRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'] + 1)

    # Internet type * services
    df['Internet_Services'] = df['InternetService'] * df['NumInternetServices']

    return df

train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

# Drop id and target
drop_cols = ['id', 'Churn']
features = [c for c in train_fe.columns if c not in drop_cols]
X = train_fe[features].values
X_test = test_fe[features].values

print(f"Features: {len(features)}")
print(f"Feature names: {features}")

# ============================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# --- LightGBM Tuning ---
print("\n" + "="*60)
print("TUNING LightGBM...")
print("="*60)

def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        'random_state': SEED,
    }

    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

        if fold >= 2:  # Early pruning: use 3 folds for speed
            break

    return np.mean(scores)

lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
lgb_study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"LightGBM best CV AUC: {lgb_study.best_value:.6f}")
lgb_best_params = lgb_study.best_params

# --- XGBoost Tuning ---
print("\n" + "="*60)
print("TUNING XGBoost...")
print("="*60)

def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': SEED,
        'verbosity': 0,
    }

    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50
        )
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

        if fold >= 2:
            break

    return np.mean(scores)

xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"XGBoost best CV AUC: {xgb_study.best_value:.6f}")
xgb_best_params = xgb_study.best_params

# --- CatBoost Tuning ---
print("\n" + "="*60)
print("TUNING CatBoost...")
print("="*60)

def cat_objective(trial):
    params = {
        'iterations': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': SEED,
        'verbose': 0,
        'eval_metric': 'AUC',
        'task_type': 'CPU',
    }

    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

        if fold >= 2:
            break

    return np.mean(scores)

cat_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
cat_study.optimize(cat_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"CatBoost best CV AUC: {cat_study.best_value:.6f}")
cat_best_params = cat_study.best_params

# ============================================================
# FULL TRAINING WITH BEST PARAMS (ALL FOLDS)
# ============================================================
print("\n" + "="*60)
print("FULL TRAINING WITH BEST PARAMS...")
print("="*60)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 5000,
        'random_state': SEED,
        **lgb_best_params
    }
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
    pred_lgb += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS

    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'n_estimators': 5000,
        'random_state': SEED,
        'verbosity': 0,
        **xgb_best_params
    }
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
    oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
    pred_xgb += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

    # CatBoost
    cat_params = {
        'iterations': 5000,
        'random_seed': SEED,
        'verbose': 0,
        'eval_metric': 'AUC',
        'task_type': 'CPU',
        **cat_best_params
    }
    m_cat = CatBoostClassifier(**cat_params)
    m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
    pred_cat += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS

    auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    auc_cat = roc_auc_score(y_val, oof_cat[val_idx])
    print(f"  LGB: {auc_lgb:.6f} | XGB: {auc_xgb:.6f} | CAT: {auc_cat:.6f}")

    gc.collect()

# Individual model scores
auc_lgb_full = roc_auc_score(y, oof_lgb)
auc_xgb_full = roc_auc_score(y, oof_xgb)
auc_cat_full = roc_auc_score(y, oof_cat)
print(f"\nFull CV - LGB: {auc_lgb_full:.6f} | XGB: {auc_xgb_full:.6f} | CAT: {auc_cat_full:.6f}")

# ============================================================
# OPTIMAL ENSEMBLE WEIGHTS (grid search on OOF)
# ============================================================
print("\n" + "="*60)
print("OPTIMIZING ENSEMBLE WEIGHTS...")
print("="*60)

best_auc = 0
best_w = (0, 0, 0)

for w1 in np.arange(0.1, 0.9, 0.05):
    for w2 in np.arange(0.1, 0.9 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc = auc
            best_w = (w1, w2, w3)

print(f"Best weights - LGB: {best_w[0]:.2f}, XGB: {best_w[1]:.2f}, CAT: {best_w[2]:.2f}")
print(f"Best ensemble CV AUC: {best_auc:.6f}")

# Final prediction
final_pred = best_w[0] * pred_lgb + best_w[1] * pred_xgb + best_w[2] * pred_cat

# ============================================================
# SUBMISSION
# ============================================================
print("\n" + "="*60)
print("GENERATING SUBMISSION...")
print("="*60)

submission = pd.DataFrame({'id': test_id, 'Churn': final_pred})
submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)

print(f"Submission saved: {DATA_DIR}/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Prediction range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")
print(f"Prediction mean: {final_pred.mean():.6f}")
print(f"\nDONE!")
