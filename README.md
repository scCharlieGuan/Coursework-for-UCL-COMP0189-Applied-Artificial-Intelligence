# COMP0189 Applied Artificial Intelligence Coursework

## Patient Readmission Prediction

This project develops and evaluates machine learning pipelines for predicting diabetes patient readmission using the diabetes hospital encounter dataset. The work covers dataset exploration, leakage-aware data splitting, preprocessing, model selection, feature importance analysis, and feature-selection-based retraining.

## Project Structure

```text
.
|-- main.ipynb                         # Main notebook containing EDA, preprocessing, modelling and evaluation
|-- COMP0189_coursework_1.pdf          # Coursework specification
|-- dataset_diabetes/
|   |-- diabetic_data.csv              # Original dataset
|   |-- IDs_mapping.csv                # ID mapping file
|   |-- train.csv                      # Generated training subset
|   `-- test.csv                       # Generated held-out test set
|-- saved_models/
|   |-- best_LinearSVM.joblib          # Best Linear SVM pipeline
|   |-- best_RF.joblib                 # Best Random Forest pipeline
|   `-- best_HGB.joblib                # Best HistGradientBoosting pipeline
|-- saved_models_task5/
|   `-- best_rf_fs_pipeline.joblib     # Random Forest pipeline with feature selection
|-- rf_fs_summary_*.json               # Task 5 repeated-run summaries
|-- missing_ratio_plot.png             # Missing-value visualization
`-- upload/
    |-- Report.docx / Report.pdf       # Final report files
    |-- Figures.docx / Figures.pdf     # Figure files
    `-- main.ipynb                     # Submitted notebook copy
```

## Dataset

The dataset contains 101,766 hospital encounters and 49 predictive features, with `readmitted` used as the target variable. The original target is converted into a binary classification problem:

- `NO` -> 0, no readmission
- `<30` and `>30` -> 1, readmitted

The dataset includes both numerical and categorical variables. It also contains repeated encounters from the same patient, so patient-level grouping is required to avoid data leakage between training and testing.

Key dataset characteristics:

- 101,766 samples
- 49 input features
- Sample-to-feature ratio: approximately 2076.86
- 37 string-type columns and 13 integer-type columns in the original data
- Important missing-value indicators include `?` and `Unknown/Invalid`
- High missingness appears in fields such as `weight`, `max_glu_serum`, `A1Cresult`, `medical_specialty`, and `payer_code`

## Methodology

### Data Splitting

The dataset is split using group-aware strategies based on `patient_nbr`. This ensures that records from the same patient do not appear in both training and test sets.

The workflow is:

1. Use `StratifiedGroupKFold` to create an approximately 80/20 train-test split.
2. Use `GroupShuffleSplit` to sample a smaller training subset from the training portion to reduce computation cost.
3. Verify that training and test patient groups are disjoint.

### Preprocessing

The preprocessing pipeline is implemented with scikit-learn transformers and is embedded inside model pipelines to avoid leakage during cross-validation.

Main preprocessing steps include:

- Replace invalid or missing-value markers such as `?` and `Unknown/Invalid` with `NaN`
- Drop identifier, near-constant, or extremely sparse features where appropriate
- Preserve clinically meaningful variables such as `max_glu_serum` and `A1Cresult`
- Transform `time_in_hospital` using a log transformation due to skewness
- Treat ID-like variables such as admission and discharge codes as categorical features
- Apply one-hot encoding to categorical variables
- Reduce cardinality for high-cardinality features such as `medical_specialty` and `payer_code`
- Apply special handling to diagnosis variables `diag_1`, `diag_2`, and `diag_3`

## Models

Three main classifiers are trained and tuned:

- Linear SVM (`LinearSVC`)
- Random Forest
- HistGradientBoostingClassifier

Hyperparameter tuning is performed using `GridSearchCV` with repeated group-aware 5-fold cross-validation. The refit metric is ROC-AUC. Additional metrics include average precision, accuracy, F1, precision, and recall.

The main hyperparameter grids are:

- Linear SVM: `C`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`
- HistGradientBoosting: `learning_rate`, `max_depth`, `max_leaf_nodes`

## Evaluation Results

The best saved models were evaluated on the held-out test set.

| Model | Test ROC-AUC | Notes |
| --- | ---: | --- |
| Linear SVM | 0.6916 | Stable across repeated CV runs, with best `C = 0.0464` |
| Random Forest | 0.7032 | Best overall baseline model among the three main classifiers |
| HistGradientBoosting | 0.7025 | Comparable to Random Forest |
| Random Forest + Feature Selection | 0.7016 | Slightly lower ROC-AUC, but with reduced feature space |

For the feature-selection Random Forest pipeline, repeated runs produced the following averaged results:

| Metric | CV Mean | CV Std | Test Mean | Test Std |
| --- | ---: | ---: | ---: | ---: |
| ROC-AUC | 0.6835 | 0.0008 | 0.7014 | 0.0006 |
| Average Precision | 0.6379 | 0.0005 | 0.6550 | 0.0006 |
| Accuracy | 0.6361 | 0.0016 | 0.6456 | 0.0002 |
| F1 | 0.5406 | 0.0013 | 0.5634 | 0.0007 |
| Precision | 0.6356 | 0.0027 | 0.6518 | 0.0001 |
| Recall | 0.4704 | 0.0010 | 0.4961 | 0.0012 |

## Feature Importance

Feature importance is analysed using model-specific methods:

- Linear SVM: absolute and signed coefficients
- Random Forest: impurity-based feature importance
- HistGradientBoosting: permutation importance

Task 5 additionally retrains a Random Forest model with `SelectFromModel`, using Random Forest as the feature selector. The best repeated configuration used:

```text
feature_selection__estimator__max_depth = 8
feature_selection__max_features = None
feature_selection__threshold = "median"
```

## How to Run

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib scipy tqdm joblib
```

Then open and run:

```text
main.ipynb
```

The notebook reads data from `dataset_diabetes/`, trains and evaluates the models, saves fitted pipelines into `saved_models/` and `saved_models_task5/`, and generates evaluation outputs and figures.

## Reproducibility

The modelling workflow uses fixed random seeds where applicable, including `random_state=42` as the base seed for repeated cross-validation. Since patient groups are used during splitting and cross-validation, the reported performance is designed to reflect generalization to unseen patients rather than unseen encounters from already observed patients.

## Summary

This coursework demonstrates a full applied AI workflow for clinical readmission prediction. The final system combines leakage-aware dataset splitting, robust preprocessing, repeated cross-validation, hyperparameter optimization, model comparison, feature interpretation, and feature-selection-based retraining. Random Forest and HistGradientBoosting achieved the strongest predictive performance, while the feature-selection experiment provided a more compact model with only a small reduction in ROC-AUC.
