"""Data loading and group-aware train/test splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Read the raw diabetes readmission CSV file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. Put diabetic_data.csv under data/raw/ first."
        )
    return pd.read_csv(path)


def load_id_mapping(mapping_path: str | Path) -> Optional[pd.DataFrame]:
    """Read the optional ID mapping file if it exists."""
    path = Path(mapping_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def make_binary_target(y: pd.Series) -> pd.Series:
    """Convert readmitted labels into binary labels: NO=0, <30/>30=1."""
    mapping = {"NO": 0, "<30": 1, ">30": 1}
    out = y.map(mapping)
    if out.isna().any():
        unknown = sorted(y[out.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected readmitted labels: {unknown}")
    return out.astype(int)


def split_train_val_test(
    df: pd.DataFrame,
    group_col: str = "patient_nbr",
    target_col: str = "readmitted",
    outer_splits: int = 5,
    validation_size: float = 0.2,
    random_state: int = 42,
    create_validation: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Split data with patient-level separation to reduce leakage risk.

    The test set is one fold from StratifiedGroupKFold. If requested, a validation
    set is then taken from the remaining training data using GroupShuffleSplit.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    groups = df[group_col].to_numpy()

    outer_cv = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_all_idx, test_idx = next(outer_cv.split(X, y, groups=groups))
    train_all = df.iloc[train_all_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    if not create_validation:
        return train_all, None, test

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )
    train_idx, val_idx = next(
        gss.split(
            train_all,
            train_all[target_col],
            groups=train_all[group_col],
        )
    )
    train = train_all.iloc[train_idx].reset_index(drop=True)
    val = train_all.iloc[val_idx].reset_index(drop=True)

    _assert_group_disjoint(train, val, test, group_col)
    return train, val, test


def _assert_group_disjoint(
    train: pd.DataFrame,
    val: Optional[pd.DataFrame],
    test: pd.DataFrame,
    group_col: str,
) -> None:
    """Check that the same patient does not appear in multiple splits."""
    train_groups = set(train[group_col])
    test_groups = set(test[group_col])
    if not train_groups.isdisjoint(test_groups):
        raise AssertionError("Patient overlap found between train and test.")
    if val is not None:
        val_groups = set(val[group_col])
        if not train_groups.isdisjoint(val_groups):
            raise AssertionError("Patient overlap found between train and validation.")
        if not val_groups.isdisjoint(test_groups):
            raise AssertionError("Patient overlap found between validation and test.")


def save_splits(
    train: pd.DataFrame,
    val: Optional[pd.DataFrame],
    test: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save split datasets as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path / "train.csv", index=False)
    if val is not None:
        val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)


def load_processed_splits(processed_dir: str | Path):
    """Load processed train/validation/test CSV files."""
    processed_path = Path(processed_dir)
    train = pd.read_csv(processed_path / "train.csv")
    val_path = processed_path / "val.csv"
    val = pd.read_csv(val_path) if val_path.exists() else None
    test = pd.read_csv(processed_path / "test.csv")
    return train, val, test
