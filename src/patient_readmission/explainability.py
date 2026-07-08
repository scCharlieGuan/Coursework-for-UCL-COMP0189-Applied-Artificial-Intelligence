"""Feature-importance and interpretation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_transformed_feature_names(fitted_pipe, X_reference) -> pd.Index:
    """Return feature names after the preprocessing step."""
    preprocess = fitted_pipe.named_steps["preprocess"]
    X_trans = preprocess.transform(X_reference)
    if not isinstance(X_trans, pd.DataFrame):
        raise TypeError("Expected preprocess output to be a pandas DataFrame.")
    return X_trans.columns


def extract_linear_feature_importance(fitted_pipe, X_reference) -> pd.DataFrame:
    """Extract absolute and signed coefficients from a fitted linear model pipeline."""
    clf = fitted_pipe.named_steps["clf"]
    feature_names = get_transformed_feature_names(fitted_pipe, X_reference)
    if not hasattr(clf, "coef_"):
        raise TypeError("The classifier does not expose coef_.")
    coefs = np.ravel(clf.coef_)
    return pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)


def extract_tree_feature_importance(fitted_pipe, X_reference) -> pd.DataFrame:
    """Extract feature_importances_ from tree-based models such as RandomForest."""
    clf = fitted_pipe.named_steps["clf"]
    feature_names = get_transformed_feature_names(fitted_pipe, X_reference)
    if not hasattr(clf, "feature_importances_"):
        raise TypeError("The classifier does not expose feature_importances_.")
    return pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def compute_permutation_importance(
    fitted_pipe,
    X_reference,
    y_reference,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance in the transformed feature space."""
    preprocess = fitted_pipe.named_steps["preprocess"]
    clf = fitted_pipe.named_steps["clf"]
    X_trans = preprocess.transform(X_reference)
    feature_names = X_trans.columns
    result = permutation_importance(
        estimator=clf,
        X=X_trans,
        y=y_reference,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
