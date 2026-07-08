"""Model evaluation utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_prediction_scores(model, X):
    """Return continuous scores for ROC-AUC / PR-AUC when the model supports them."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def evaluate_binary_classifier(model, X, y) -> Dict[str, float]:
    """Evaluate a fitted binary classifier on common classification metrics."""
    y_pred = model.predict(X)
    y_score = get_prediction_scores(model, X)

    scores = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
    }
    if y_score is not None:
        scores["roc_auc"] = roc_auc_score(y, y_score)
        scores["average_precision"] = average_precision_score(y, y_score)
    else:
        scores["roc_auc"] = np.nan
        scores["average_precision"] = np.nan
    return scores


def scores_to_frame(scores: Dict[str, float], model_name: str) -> pd.DataFrame:
    """Convert a score dictionary into a tidy dataframe."""
    return pd.DataFrame([
        {"model": model_name, "metric": metric, "value": value}
        for metric, value in scores.items()
    ])
