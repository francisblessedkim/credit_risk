# src/predict.py
from __future__ import annotations
import joblib
import pandas as pd
from typing import Tuple

def load_model(model_path: str):
    """Load a trained sklearn/xgboost model (or Pipeline) from joblib."""
    return joblib.load(model_path)

def score_df(model, X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Given a model and feature matrix X, return (probabilities, labels).
    Assumes binary classification with predict_proba available.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # Fallback for models without predict_proba
        # Use decision_function if available; otherwise .predict as hard labels
        if hasattr(model, "decision_function"):
            # map decision scores to 0..1 via a simple logistic; still not calibrated
            import numpy as np
            scores = model.decision_function(X)
            proba = 1 / (1 + np.exp(-scores))
        else:
            yhat = model.predict(X)
            proba = pd.Series(yhat, index=X.index).astype(float)

    yhat = (proba >= 0.5).astype(int)
    return pd.Series(proba, index=X.index, name="pred_prob"), pd.Series(yhat, index=X.index, name="pred_label")
