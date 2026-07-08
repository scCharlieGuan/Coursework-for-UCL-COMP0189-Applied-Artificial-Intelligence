"""Feature-related helpers."""

from __future__ import annotations

import pandas as pd

from .data_loader import make_binary_target


def split_features_target(df: pd.DataFrame, target_col: str = "readmitted"):
    """Return X and binary y from a dataframe containing readmitted labels."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = make_binary_target(df[target_col])
    return X, y


def get_groups(X: pd.DataFrame, group_col: str = "patient_nbr") -> pd.Series:
    """Return patient-level groups for group-aware cross-validation."""
    if group_col not in X.columns:
        raise KeyError(f"Group column '{group_col}' not found in features.")
    return X[group_col].copy()
