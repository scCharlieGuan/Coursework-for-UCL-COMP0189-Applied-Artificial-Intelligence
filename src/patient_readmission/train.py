"""Training and hyperparameter tuning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from .evaluate import evaluate_binary_classifier, get_prediction_scores


def _make_split_signature(cv, X, y, groups) -> str:
    """Create a short signature so repeated CV splits can be compared."""
    parts = []
    for fold_id, (_, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        parts.append(f"fold{fold_id}:" + ",".join(map(str, val_idx[:20])))
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _score_from_metric(refit_metric: str, y_true, y_pred, y_score) -> float:
    """Return the scalar score corresponding to the refit metric."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if refit_metric == "roc_auc":
        return roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    if refit_metric == "average_precision":
        return average_precision_score(y_true, y_score) if y_score is not None else np.nan
    if refit_metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if refit_metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if refit_metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if refit_metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    raise ValueError(f"Unsupported refit_metric: {refit_metric}")


def tune_save_best_per_model(
    X_train,
    y_train,
    groups_train,
    X_test,
    y_test,
    preprocess_pipe,
    models_and_grids: Dict[str, Tuple[object, dict]],
    scoring: dict,
    refit_metric: str = "roc_auc",
    inner_splits: int = 5,
    n_repeats: int = 3,
    base_random_state: int = 42,
    output_dir: str | Path = "models",
    n_jobs: int = -1,
):
    """Tune each model with group-aware CV and save the best final pipeline.

    For every repeat, GridSearchCV is run with a different random split seed. The
    best hyperparameters across repeats are then refit on all training data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    repeat_details = {}
    saved_paths = {}

    for model_name, (clf, param_grid) in tqdm(models_and_grids.items(), desc="Models"):
        cv_scores = []
        test_scores = []
        repeat_records = []
        best_cv_score = -np.inf
        best_params = None
        best_grid = None

        for repeat_id in range(n_repeats):
            rs = base_random_state + repeat_id
            split_cv = StratifiedGroupKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=rs,
            )
            split_signature = _make_split_signature(split_cv, X_train, y_train, groups_train)

            cv_for_gs = StratifiedGroupKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=rs,
            )
            pipe = Pipeline([
                ("preprocess", clone(preprocess_pipe)),
                ("clf", clone(clf)),
            ])
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit_metric,
                cv=cv_for_gs.split(X_train, y_train, groups=groups_train),
                n_jobs=n_jobs,
                verbose=0,
                return_train_score=False,
            )
            gs.fit(X_train, y_train)

            cv_score = float(gs.best_score_)
            cv_scores.append(cv_score)
            if cv_score > best_cv_score:
                best_cv_score = cv_score
                best_params = gs.best_params_
                best_grid = gs

            best_model_repeat = gs.best_estimator_
            y_pred = best_model_repeat.predict(X_test)
            y_score = get_prediction_scores(best_model_repeat, X_test)
            test_score = float(_score_from_metric(refit_metric, y_test, y_pred, y_score))
            test_scores.append(test_score)

            repeat_records.append({
                "model": model_name,
                "repeat": repeat_id,
                "random_state": rs,
                "split_signature": split_signature,
                "best_cv_score": cv_score,
                "test_score": test_score,
                "best_params": json.dumps(gs.best_params_, sort_keys=True),
            })

        if best_params is None:
            raise RuntimeError(f"No best parameters found for {model_name}.")

        final_pipe = Pipeline([
            ("preprocess", clone(preprocess_pipe)),
            ("clf", clone(clf)),
        ])
        final_pipe.set_params(**best_params)
        final_pipe.fit(X_train, y_train)
        save_path = output_path / f"best_{model_name}.joblib"
        joblib.dump(final_pipe, save_path)
        saved_paths[model_name] = str(save_path)

        test_metrics = evaluate_binary_classifier(final_pipe, X_test, y_test)
        row = {
            "model": model_name,
            f"cv_{refit_metric}_mean": float(np.nanmean(cv_scores)),
            f"cv_{refit_metric}_std": float(np.nanstd(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0,
            f"test_{refit_metric}_mean": float(np.nanmean(test_scores)),
            f"test_{refit_metric}_std": float(np.nanstd(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0,
            f"best_cv_{refit_metric}_over_repeats": best_cv_score,
            "best_params_over_repeats": json.dumps(best_params, sort_keys=True),
            "saved_pipeline_path": str(save_path),
        }
        row.update({f"final_test_{k}": v for k, v in test_metrics.items()})
        summary_rows.append(row)
        repeat_details[model_name] = pd.DataFrame(repeat_records)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        f"test_{refit_metric}_mean", ascending=False
    ).reset_index(drop=True)
    return summary_df, saved_paths, repeat_details
