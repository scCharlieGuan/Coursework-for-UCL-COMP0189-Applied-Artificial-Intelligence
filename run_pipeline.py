"""Main entry point for the patient readmission prediction pipeline.

Run from the project root:
    python run_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.patient_readmission.config import ensure_dirs, find_project_root, load_config, resolve_path
from src.patient_readmission.data_loader import (
    load_processed_splits,
    load_raw_data,
    save_splits,
    split_train_val_test,
)
from src.patient_readmission.features import get_groups, split_features_target
from src.patient_readmission.models import get_models_and_grids, get_scoring
from src.patient_readmission.preprocessing import build_preprocess_pipeline
from src.patient_readmission.train import tune_save_best_per_model


def main() -> None:
    project_root = find_project_root(Path(__file__).resolve().parent)
    config = load_config(project_root / "configs" / "config.yaml")

    paths = config["paths"]
    raw_data_path = resolve_path(project_root, paths["raw_data"])
    processed_dir = resolve_path(project_root, paths["processed_dir"])
    models_dir = resolve_path(project_root, paths["models_dir"])
    tables_dir = resolve_path(project_root, paths["tables_dir"])
    ensure_dirs(processed_dir, models_dir, tables_dir)

    raw_df = load_raw_data(raw_data_path)
    split_cfg = config["split"]
    train_df, val_df, test_df = split_train_val_test(
        raw_df,
        outer_splits=split_cfg["n_splits_outer"],
        validation_size=split_cfg["validation_size"],
        random_state=config["random_state"],
        create_validation=split_cfg.get("create_validation", True),
    )
    save_splits(train_df, val_df, test_df, processed_dir)

    train_df, _, test_df = load_processed_splits(processed_dir)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    groups_train = get_groups(X_train)

    preprocess_cfg = config["preprocessing"]
    preprocess_pipe = build_preprocess_pipeline(
        top_k_categories=preprocess_cfg["top_k_categories"],
        columns_to_drop=preprocess_cfg["columns_to_drop"],
    )

    training_cfg = config["training"]
    models_and_grids = get_models_and_grids(
        random_state=config["random_state"],
        enabled=config.get("models", {}),
    )
    scoring = get_scoring()

    summary_df, saved_paths, repeat_details = tune_save_best_per_model(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_test=X_test,
        y_test=y_test,
        preprocess_pipe=preprocess_pipe,
        models_and_grids=models_and_grids,
        scoring=scoring,
        refit_metric=training_cfg["refit_metric"],
        inner_splits=training_cfg["inner_splits"],
        n_repeats=training_cfg["n_repeats"],
        base_random_state=config["random_state"],
        output_dir=models_dir,
        n_jobs=training_cfg["n_jobs"],
    )

    summary_path = tables_dir / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    for model_name, df in repeat_details.items():
        df.to_csv(tables_dir / f"repeat_details_{model_name}.csv", index=False)

    print("Training complete.")
    print(f"Saved model summary to: {summary_path}")
    print(pd.DataFrame.from_dict(saved_paths, orient="index", columns=["path"]))
    print(summary_df)


if __name__ == "__main__":
    main()
