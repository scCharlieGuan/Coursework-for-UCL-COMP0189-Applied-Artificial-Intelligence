"""Model definitions and hyperparameter grids."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_scoring():
    """Return the metrics used during GridSearchCV."""
    return {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "precision": "precision",
        "recall": "recall",
    }


def get_models_and_grids(random_state: int = 42, enabled: dict | None = None):
    """Return model objects and grids based on the notebook experiments."""
    enabled = enabled or {}

    def is_enabled(name: str) -> bool:
        value = enabled.get(name, {"enabled": True})
        return bool(value.get("enabled", True)) if isinstance(value, dict) else bool(value)

    models = {}
    if is_enabled("LinearSVM"):
        models["LinearSVM"] = (
            LinearSVC(dual="auto", max_iter=20000, random_state=random_state),
            {"clf__C": np.logspace(-3, 2, 10)},
        )

    if is_enabled("LogisticRegression"):
        models["LogisticRegression"] = (
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_state),
            {
                "clf__C": np.logspace(-3, 2, 8),
                "clf__class_weight": [None, "balanced"],
            },
        )

    if is_enabled("RandomForest"):
        models["RandomForest"] = (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "clf__n_estimators": [200, 400, 600],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 5, 10],
                "clf__max_features": ["sqrt", "log2"],
            },
        )

    if is_enabled("HistGradientBoosting"):
        models["HistGradientBoosting"] = (
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "clf__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "clf__max_depth": [2, 3, 5, None],
                "clf__max_leaf_nodes": [15, 31, 63],
            },
        )
    return models
