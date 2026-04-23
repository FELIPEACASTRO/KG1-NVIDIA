"""
Inference API for the ML Consensus Ensemble pre-scorer.

Loads the 4 base models (RF, XGB, LGB, CatBoost) plus the SoftVotingEnsemble
wrapper and exposes a single predict entry point.

Usage:
    from ml_ensemble_prescore import MLEnsemblePrescorer
    p = MLEnsemblePrescorer()
    prob = p.predict(features_dict)  # float in [0, 1]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


# Default: repo-local `models/` (portable); fallback: Felipe's local temp cache
_DEFAULT_MODEL_DIRS = [
    Path(__file__).resolve().parent.parent / "models",  # repo models/
    Path("C:/Users/davis/AppData/Local/Temp/tc/"),       # Felipe local cache
    Path("/tmp/tc/"),                                     # Colab/Linux mirror
]


def _resolve_model_dir() -> Path:
    """Pick the first directory that has the lightweight models."""
    required = "ml_lgb_metric_consensus.pkl"  # lightweight + required
    for d in _DEFAULT_MODEL_DIRS:
        if d.exists() and (d / required).exists():
            return d
    # If nothing found, still return the first candidate (raise will happen on load)
    return _DEFAULT_MODEL_DIRS[0]


MODEL_DIR = _resolve_model_dir()


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Mirror of the class defined in train_ml_ensemble.py so joblib can unpickle."""

    def __init__(self, estimators):
        self.estimators = estimators
        self.classes_ = np.array(self.estimators[0][1].classes_)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for _, m in self.estimators]
        return np.mean(probas, axis=0)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


class MLEnsemblePrescorer:
    """Wraps the 5 saved models + metadata."""

    def __init__(self, model_dir: Path = None, lightweight: bool = True) -> None:
        """Load models.

        Args:
            model_dir: override path (default: auto-resolve repo models/ -> tmp cache)
            lightweight: if True, skips RF + CatBoost + full ensemble (saves 18MB + load time).
                Per ml_ensemble_report.md, LGB has BEST Brier (0.00178) and is 2-4x faster
                than RF. Use LGB or XGB in production; ensemble adds <0.001 improvement.
        """
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.models = {}
        # Always load the fast/small ones (recommended for production)
        self.models["lgb"] = joblib.load(self.model_dir / "ml_lgb_metric_consensus.pkl")
        self.models["xgb"] = joblib.load(self.model_dir / "ml_xgb_metric_consensus.pkl")
        if not lightweight:
            rf_path = self.model_dir / "ml_rf_metric_consensus.pkl"
            cat_path = self.model_dir / "ml_cat_metric_consensus.pkl"
            ens_path = self.model_dir / "ml_ensemble_metric_consensus.pkl"
            if rf_path.exists():
                self.models["rf"] = joblib.load(rf_path)
            if cat_path.exists():
                self.models["cat"] = joblib.load(cat_path)
            if ens_path.exists():
                self.models["ensemble"] = joblib.load(ens_path)
        with open(self.model_dir / "ml_ensemble_metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.features = self.metadata["features"]

    def _prepare_frame(self, features: Dict) -> pd.DataFrame:
        row = {k: features.get(k, 0) for k in self.features}
        return pd.DataFrame([row])

    def predict(self, features: Dict, model: str = None) -> float:
        """Return P(label_would_pass=1) for a single feature dict.

        Default model is 'lgb' (fastest + best Brier per ML report).
        Fallback order: ensemble > lgb > xgb if requested not loaded.
        """
        if model is None:
            model = "ensemble" if "ensemble" in self.models else "lgb"
        elif model not in self.models:
            # graceful fallback
            for fb in ("ensemble", "lgb", "xgb"):
                if fb in self.models:
                    model = fb
                    break
        X = self._prepare_frame(features)
        m = self.models[model]
        return float(m.predict_proba(X)[0, 1])

    def predict_all(self, features: Dict) -> Dict[str, float]:
        """Return probas for every model name."""
        X = self._prepare_frame(features)
        return {name: float(m.predict_proba(X)[0, 1]) for name, m in self.models.items()}

    def batch_predict(
        self, df_features: pd.DataFrame, model: str = "ensemble"
    ) -> np.ndarray:
        """Vectorized batch predict. df_features must carry all required columns
        (missing ones are filled with 0)."""
        missing = [c for c in self.features if c not in df_features.columns]
        df = df_features.copy()
        for c in missing:
            df[c] = 0
        df = df[self.features].fillna(0)
        return self.models[model].predict_proba(df)[:, 1]


def _demo() -> None:
    p = MLEnsemblePrescorer()
    print(f"features ({len(p.features)}): {p.features[:6]}...")
    demo = {
        "output_length_chars": 50,
        "output_length_tokens_est": 15,
        "has_comma_in_output": 0,
        "has_trailing_dot_zero": 0,
        "boxed_is_empty": 0,
        "has_latex_command": 0,
        "has_nested_braces": 0,
        "num_boxes_in_output": 1,
        "output_format_boxed": 1,
        "output_format_final_answer": 0,
        "output_format_plain": 0,
        "answer_length": 3,
        "family_bit_manipulation": 1,
        "answer_is_numeric": 1,
    }
    per_model = p.predict_all(demo)
    print(f"Per-model P(pass): {per_model}")
    print(f"Ensemble P(pass): {p.predict(demo):.4f}")


if __name__ == "__main__":
    _demo()
