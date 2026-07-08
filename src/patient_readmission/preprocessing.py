"""Preprocessing pipeline for diabetes readmission data."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

INVALID_VALUES = ["?", "Unknown/Invalid", ""]

DEFAULT_COLUMNS_TO_DROP = [
    "encounter_id",
    "patient_nbr",
    "acetohexamide",
    "troglitazone",
    "examide",
    "citoglipton",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "weight",
    "nateglinide",
    "chlorpropamide",
    "acarbose",
    "glyburide-metformin",
    "tolbutamide",
    "miglitol",
    "tolazamide",
]

CATEGORICAL_COLS = [
    "gender",
    "max_glu_serum",
    "A1Cresult",
    "race",
    "medical_specialty",
    "payer_code",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
    "change",
    "diabetesMed",
    "glipizide",
    "glimepiride",
    "metformin",
    "repaglinide",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

AGE_MAPPING = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}


def replace_invalid_values(X: pd.DataFrame) -> pd.DataFrame:
    """Replace dataset-specific invalid markers with NaN."""
    return X.copy().replace(INVALID_VALUES, np.nan)


def drop_columns(X: pd.DataFrame, columns_to_drop: Iterable[str] | None = None) -> pd.DataFrame:
    """Drop ID, near-constant, and very sparse columns used in the notebook."""
    columns = list(columns_to_drop or DEFAULT_COLUMNS_TO_DROP)
    return X.copy().drop(columns=columns, errors="ignore")


def log1p_time_in_hospital(X: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to time_in_hospital."""
    X = X.copy()
    if "time_in_hospital" in X.columns:
        X["time_in_hospital"] = np.log1p(X["time_in_hospital"])
    return X


def encode_age_midpoint(X: pd.DataFrame) -> pd.DataFrame:
    """Convert age intervals such as [70-80) into interval midpoints."""
    X = X.copy()
    if "age" in X.columns:
        X["age"] = X["age"].map(AGE_MAPPING)
    return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop a fixed list of columns while keeping sklearn pipeline compatibility."""

    def __init__(self, columns_to_drop: Iterable[str] | None = None):
        self.columns_to_drop = list(columns_to_drop or DEFAULT_COLUMNS_TO_DROP)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy().drop(columns=self.columns_to_drop, errors="ignore")

    def set_output(self, *, transform=None):
        return self


class TopCategoryGrouper(BaseEstimator, TransformerMixin):
    """Keep top-k training categories and map rarer categories to 'Other'.

    This replaces the notebook version that recomputed top-k categories during
    transform, which can leak test-set distribution information into preprocessing.
    """

    def __init__(self, columns: List[str] | None = None, top_k: int = 10):
        self.columns = columns or ["medical_specialty", "payer_code"]
        self.top_k = top_k

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.top_categories_: Dict[str, set] = {}
        for col in self.columns:
            if col in X.columns:
                top = X[col].value_counts(dropna=False).nlargest(self.top_k).index
                self.top_categories_[col] = set(top)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, top_categories in self.top_categories_.items():
            if col in X.columns:
                X[col] = X[col].where(X[col].isin(top_categories), "Other")
        return X

    def set_output(self, *, transform=None):
        return self


class BasicCategoricalCleaner(BaseEstimator, TransformerMixin):
    """Fill categorical missing values and cast selected ID columns to category."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        fill_values = {
            "gender": "Unknown",
            "race": "Unknown",
            "max_glu_serum": "Unknown",
            "A1Cresult": "Unknown",
            "medical_specialty": "Missing",
            "payer_code": "Missing",
        }
        existing_fill_values = {k: v for k, v in fill_values.items() if k in X.columns}
        X = X.fillna(existing_fill_values)

        id_like_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
        existing_id_cols = [c for c in id_like_cols if c in X.columns]
        if existing_id_cols:
            X[existing_id_cols] = X[existing_id_cols].astype("category")
        return X

    def set_output(self, *, transform=None):
        return self


def icd9_to_category(code: object) -> str:
    """Map an ICD-9 code to a coarse disease category."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if s == "" or s in {"?", "Unknown", "None", "nan"}:
        return "Missing"
    s_upper = s.upper()
    if s_upper.startswith("E"):
        return "Injury"
    if s_upper.startswith("V"):
        return "Supplementary"
    try:
        val = float(s_upper)
    except ValueError:
        return "Other"

    if 1 <= val <= 139:
        return "Infectious/Parasitic"
    if 140 <= val <= 239:
        return "Neoplasms"
    if 240 <= val <= 279:
        return "Diabetes" if 250 <= val < 251 else "Endocrine/Metabolic"
    if 280 <= val <= 289:
        return "Blood"
    if 290 <= val <= 319:
        return "Mental"
    if 320 <= val <= 389:
        return "Nervous/Sense"
    if 390 <= val <= 459:
        return "Circulatory"
    if 460 <= val <= 519:
        return "Respiratory"
    if 520 <= val <= 579:
        return "Digestive"
    if 580 <= val <= 629:
        return "Genitourinary"
    if 630 <= val <= 679:
        return "Pregnancy"
    if 680 <= val <= 709:
        return "Skin"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 740 <= val <= 759:
        return "Congenital"
    if 760 <= val <= 779:
        return "Perinatal"
    if 780 <= val <= 799:
        return "Symptoms"
    if 800 <= val <= 999:
        return "Injury"
    return "Other"


def add_diag_categories(X: pd.DataFrame) -> pd.DataFrame:
    """Replace diag_1/2/3 ICD-9 codes by coarse diagnosis categories."""
    X = X.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in X.columns:
            X[col] = X[col].apply(icd9_to_category)
    return X


def select_diag_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Select diagnosis columns in a fixed order."""
    return X.loc[:, ["diag_1", "diag_2", "diag_3"]]


def build_preprocess_pipeline(
    top_k_categories: int = 10,
    columns_to_drop: Iterable[str] | None = None,
) -> Pipeline:
    """Build the sklearn preprocessing pipeline used before model training."""
    drop_transformer = DropColumnsTransformer(columns_to_drop=columns_to_drop)

    diag_category_pipe = Pipeline([
        ("add_diag_cat", FunctionTransformer(add_diag_categories, feature_names_out="one-to-one")),
        ("select_cat_cols", FunctionTransformer(select_diag_cols, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    one_hot_preprocess = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            (
                "diag_cat_onehot",
                diag_category_pipe,
                ["diag_1", "diag_2", "diag_3"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=True,
    )

    pipe = Pipeline(steps=[
        ("replace_invalid", FunctionTransformer(replace_invalid_values, feature_names_out="one-to-one")),
        ("drop_columns", drop_transformer),
        ("log1p_time_in_hospital", FunctionTransformer(log1p_time_in_hospital, feature_names_out="one-to-one")),
        ("age_preprocess", FunctionTransformer(encode_age_midpoint, feature_names_out="one-to-one")),
        ("categorical_cleaner", BasicCategoricalCleaner()),
        ("top_category_grouper", TopCategoryGrouper(top_k=top_k_categories)),
        ("onehot", one_hot_preprocess),
    ])
    pipe.set_output(transform="pandas")
    return pipe


def compute_columns_to_drop(
    X_train: pd.DataFrame,
    missing_threshold: float = 0.5,
    near_constant_threshold: float = 0.99,
    high_cardinality_threshold: int | None = None,
):
    """Compute candidate columns to drop using training data only."""
    X_train = replace_invalid_values(X_train)
    cols_to_drop = []
    drop_details = []
    for col in X_train.columns:
        series = X_train[col]
        missing_ratio = series.isna().mean()
        if missing_ratio > missing_threshold:
            cols_to_drop.append(col)
            drop_details.append(f"{col}: missing ratio={missing_ratio:.4f}")
            continue
        nunique = series.nunique(dropna=False)
        if nunique <= 1:
            cols_to_drop.append(col)
            drop_details.append(f"{col}: constant column")
            continue
        max_ratio = series.value_counts(normalize=True, dropna=False).iloc[0]
        if max_ratio > near_constant_threshold:
            cols_to_drop.append(col)
            drop_details.append(f"{col}: near constant, main value ratio={max_ratio:.4f}")
            continue
        if high_cardinality_threshold is not None and series.dtype == "object":
            cardinality = series.nunique()
            if cardinality > high_cardinality_threshold:
                cols_to_drop.append(col)
                drop_details.append(f"{col}: high cardinality={cardinality}")
    return sorted(set(cols_to_drop)), drop_details
