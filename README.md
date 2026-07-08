# COMP0189 Applied Artificial Intelligence Coursework

## Patient Readmission Prediction/ 患者再入院预测

This project develops and evaluates machine learning pipelines for predicting diabetes patient readmission using the diabetes hospital encounter dataset. The work covers dataset exploration, leakage-aware data splitting, preprocessing, model selection, feature importance analysis, and feature-selection-based retraining.

## English Version

### 1. Project Overview

This project predicts whether a diabetes patient is likely to be readmitted to hospital after discharge. It was originally developed in `main.ipynb` for the COMP0189 Applied Artificial Intelligence coursework, and has now been reorganised into a cleaner, more reproducible machine learning project structure.

The project covers the full applied machine learning workflow: data loading, exploratory understanding, leakage-aware data splitting, preprocessing, model training, hyperparameter tuning, evaluation, feature importance analysis, and result saving.

The prediction target is `readmitted`, which originally has three values:

- `NO`: the patient was not readmitted.
- `<30`: the patient was readmitted within 30 days.
- `>30`: the patient was readmitted after more than 30 days.

In this project, the task is treated as a binary classification problem:

- `NO` is converted to `0`.
- `<30` and `>30` are converted to `1`.

Therefore, the model predicts whether a patient will be readmitted, rather than separating 30-day and post-30-day readmission.

---

### 2. Dataset

The dataset contains diabetes hospital encounter records. It includes patient information, admission details, diagnosis codes, treatment information, medication usage, and the readmission label.

Main dataset characteristics:

- 101,766 hospital encounters.
- 49 predictive features.
- A mixture of numerical and categorical variables.
- Repeated records from the same patient.
- Missing or invalid values represented by markers such as `?` and `Unknown/Invalid`.
- High missingness in variables such as `weight`, `max_glu_serum`, `A1Cresult`, `medical_specialty`, and `payer_code`.

Important fields include:

- Patient and encounter IDs: `encounter_id`, `patient_nbr`.
- Demographic variables: `race`, `gender`, `age`.
- Admission information: `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital`.
- Medical service variables: `num_lab_procedures`, `num_procedures`, `num_medications`.
- Diagnosis variables: `diag_1`, `diag_2`, `diag_3`.
- Medication variables: `metformin`, `insulin`, `glipizide`, and others.
- Target variable: `readmitted`.

---

### 3. Project Structure

```text
patient_readmission_prediction_refactored/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── main.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   └── patient_readmission/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── evaluate.py
│       ├── explainability.py
│       ├── features.py
│       ├── models.py
│       ├── preprocessing.py
│       ├── train.py
│       └── visualization.py
├── main.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

Main components:

| Path | Purpose |
|---|---|
| `configs/config.yaml` | Stores paths, random seeds, split settings, preprocessing settings, and model training parameters. |
| `data/raw/` | Stores the original dataset files. |
| `data/processed/` | Stores processed train, validation, and test splits. |
| `notebooks/main.ipynb` | Keeps the original experimental notebook for reference. |
| `src/patient_readmission/data_loader.py` | Loads raw data, creates the binary target, and performs patient-level data splitting. |
| `src/patient_readmission/preprocessing.py` | Handles missing values, column removal, age conversion, diagnosis grouping, and one-hot encoding. |
| `src/patient_readmission/features.py` | Separates features, labels, and patient groups. |
| `src/patient_readmission/models.py` | Defines models and hyperparameter grids. |
| `src/patient_readmission/train.py` | Runs cross-validation, hyperparameter tuning, repeated experiments, and model saving. |
| `src/patient_readmission/evaluate.py` | Calculates metrics such as Accuracy, F1, Precision, Recall, ROC-AUC, and Average Precision. |
| `src/patient_readmission/explainability.py` | Provides tools for coefficient analysis, feature importance, and permutation importance. |
| `src/patient_readmission/visualization.py` | Saves common plots such as feature importance charts. |
| `run_pipeline.py` | Main entry point for running the full machine learning pipeline. |
| `main.py` | A simple compatibility entry point that calls the pipeline. |
| `models/` | Stores trained `.joblib` model files. |
| `reports/tables/` | Stores model summary tables and repeated-run details. |
| `reports/figures/` | Stores generated figures. |

---

### 4. Machine Learning Workflow

The project follows the workflow below.

#### Step 1: Load the Data

The raw diabetes hospital encounter dataset is loaded from `data/raw/diabetic_data.csv`.

#### Step 2: Create the Binary Target

The original `readmitted` column is converted into a binary label:

```text
NO        -> 0
<30, >30  -> 1
```

This means the project focuses on general readmission prediction.

#### Step 3: Split the Data Without Patient Leakage

The dataset contains repeated hospital encounters from the same patient. If the same patient appears in both training and test data, the model may learn patient-specific patterns and produce overly optimistic results.

To reduce this risk, the project uses `patient_nbr` as the grouping variable. Records from the same patient are kept in the same split. The workflow uses group-aware splitting methods such as `StratifiedGroupKFold` and `GroupShuffleSplit`, and checks that patient groups do not overlap across splits.

#### Step 4: Preprocess the Features

The preprocessing pipeline is built with scikit-learn transformers and is placed inside model pipelines. This helps avoid data leakage during cross-validation.

Main preprocessing steps include:

- Replacing invalid values such as `?`, `Unknown/Invalid`, and empty strings with missing values.
- Dropping ID columns, constant columns, near-constant columns, and extremely sparse columns when appropriate.
- Preserving clinically meaningful variables such as `max_glu_serum` and `A1Cresult`.
- Applying a `log1p` transformation to `time_in_hospital` because it is skewed.
- Converting age intervals into numerical midpoints.
- Treating admission and discharge code variables as categorical variables.
- Grouping ICD-9 diagnosis codes into broader disease categories.
- Reducing cardinality for high-cardinality variables such as `medical_specialty` and `payer_code`.
- Applying one-hot encoding to categorical variables.

#### Step 5: Train and Tune Models

The project trains and tunes several models:

- Linear SVM / LinearSVC.
- Logistic Regression.
- Random Forest.
- HistGradientBoostingClassifier.

Hyperparameter tuning is performed with `GridSearchCV`. The main refit metric is ROC-AUC. Additional metrics include Average Precision, Accuracy, F1, Precision, and Recall.

#### Step 6: Evaluate the Models

The final models are evaluated on a held-out test set. The main metrics are:

- **ROC-AUC**: measures how well the model ranks readmitted patients above non-readmitted patients.
- **Average Precision / PR-AUC**: useful for imbalanced classification and positive-class detection.
- **Accuracy**: overall percentage of correct predictions.
- **F1**: balance between precision and recall.
- **Precision**: among predicted readmission cases, how many are correct.
- **Recall**: among actual readmission cases, how many are detected.

For this task, Accuracy alone is not enough, because class imbalance can make Accuracy misleading. ROC-AUC and Average Precision are usually more informative.

---

### 5. Main Results from the Original Experiment

The original notebook reported the following test ROC-AUC results:

| Model | Test ROC-AUC | Notes |
|---|---:|---|
| Linear SVM | 0.6916 | Stable across repeated cross-validation; best `C = 0.0464`. |
| Random Forest | 0.7032 | Best overall baseline model among the three main classifiers. |
| HistGradientBoosting | 0.7025 | Very close to Random Forest. |
| Random Forest + Feature Selection | 0.7016 | Slightly lower ROC-AUC, but uses a smaller feature space. |

For the Random Forest feature-selection pipeline, the repeated-run average results were:

| Metric | CV Mean | CV Std | Test Mean | Test Std |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.6835 | 0.0008 | 0.7014 | 0.0006 |
| Average Precision | 0.6379 | 0.0005 | 0.6550 | 0.0006 |
| Accuracy | 0.6361 | 0.0016 | 0.6456 | 0.0002 |
| F1 | 0.5406 | 0.0013 | 0.5634 | 0.0007 |
| Precision | 0.6356 | 0.0027 | 0.6518 | 0.0001 |
| Recall | 0.4704 | 0.0010 | 0.4961 | 0.0012 |

The best repeated feature-selection setting was:

```text
feature_selection__estimator__max_depth = 8
feature_selection__max_features = None
feature_selection__threshold = "median"
```

Overall, Random Forest and HistGradientBoosting achieved the strongest predictive performance. The feature-selection model produced a more compact feature space with only a small reduction in ROC-AUC.

---

### 6. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you only want to run the original notebook version, the core packages are:

```bash
pip install pandas numpy scikit-learn matplotlib scipy tqdm joblib
```

---

### 7. How to Run

First, make sure the raw dataset is placed at:

```text
data/raw/diabetic_data.csv
```

Then run the full refactored pipeline from the project root:

```bash
python run_pipeline.py
```

Alternatively, run:

```bash
python main.py
```

After running the pipeline, the project should generate files such as:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/best_*.joblib
reports/tables/model_summary.csv
reports/tables/repeat_details_*.csv
```

The original notebook is also kept at:

```text
notebooks/main.ipynb
```

---

### 8. How to Read the Output

The most important output file is:

```text
reports/tables/model_summary.csv
```

Common columns include:

- `cv_roc_auc_mean`: average validation ROC-AUC across repeated cross-validation.
- `cv_roc_auc_std`: variation in validation ROC-AUC across repeated cross-validation.
- `test_roc_auc_mean`: average test ROC-AUC from repeated tuning runs.
- `final_test_roc_auc`: ROC-AUC of the final retrained best model on the test set.
- `final_test_average_precision`: PR-AUC of the final model on the test set.
- `best_params_over_repeats`: best hyperparameter settings found across repeated runs.
- `saved_pipeline_path`: location of the saved trained pipeline.

A higher ROC-AUC means the model has better ranking ability. A higher Average Precision means the model is better at identifying readmitted patients, especially under class imbalance.



---

### 9. Possible Future Improvements

Possible next steps include:

- Tune the model grids more carefully, especially for Random Forest and HistGradientBoosting.
- Try stronger handling of class imbalance, such as class weights, threshold tuning, or resampling.
- Redefine the task as strict 30-day readmission prediction, where `<30` is positive and `NO` / `>30` are negative.
- Improve diagnosis-code feature engineering.
- Add SHAP or LIME for more detailed model explanation.
- Add MLflow or Weights & Biases for experiment tracking.
- Add unit tests for data loading, preprocessing, splitting, and evaluation functions.



---

<br>

# 中文版

## 1. 项目简介

本项目用于预测糖尿病患者出院后是否会再次入院。项目最初是在 COMP0189 Applied Artificial Intelligence 课程作业中的 `main.ipynb` 中完成的，现在已经被整理成一个更清晰、更容易复现、也更方便维护的机器学习项目结构。

这个项目覆盖了一个完整的应用机器学习流程，包括：数据读取、数据理解、防止数据泄漏的数据切分、数据预处理、模型训练、超参数调优、模型评估、特征重要性分析以及结果保存。

预测目标是 `readmitted`。原始标签有三个取值：

- `NO`：患者没有再次入院。
- `<30`：患者在 30 天内再次入院。
- `>30`：患者在 30 天之后再次入院。

在当前项目中，这个任务被处理为二分类问题：

- `NO` 转换为 `0`。
- `<30` 和 `>30` 转换为 `1`。

也就是说，模型预测的是“患者是否会再次入院”，而不是进一步区分“30 天内再入院”和“30 天后再入院”。

---

## 2. 数据说明

数据集来自糖尿病患者的住院记录，包含患者基本信息、入院信息、诊断编码、治疗信息、用药情况以及是否再次入院的标签。

数据集的主要特点包括：

- 共有 101,766 条住院记录。
- 有 49 个预测特征。
- 同时包含数值变量和类别变量。
- 同一个患者可能有多次住院记录。
- 缺失值或无效值可能用 `?`、`Unknown/Invalid` 等形式表示。
- `weight`、`max_glu_serum`、`A1Cresult`、`medical_specialty`、`payer_code` 等字段存在较高比例的缺失。

常见字段包括：

- 患者和住院记录 ID：`encounter_id`、`patient_nbr`。
- 人口统计变量：`race`、`gender`、`age`。
- 入院相关信息：`admission_type_id`、`discharge_disposition_id`、`admission_source_id`、`time_in_hospital`。
- 医疗服务变量：`num_lab_procedures`、`num_procedures`、`num_medications`。
- 诊断变量：`diag_1`、`diag_2`、`diag_3`。
- 药物变量：`metformin`、`insulin`、`glipizide` 等。
- 预测目标：`readmitted`。

---

## 3. 项目结构

```text
patient_readmission_prediction_refactored/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── main.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   └── patient_readmission/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── evaluate.py
│       ├── explainability.py
│       ├── features.py
│       ├── models.py
│       ├── preprocessing.py
│       ├── train.py
│       └── visualization.py
├── main.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

主要文件说明：

| 路径 | 作用 |
|---|---|
| `configs/config.yaml` | 统一管理路径、随机种子、数据切分参数、预处理参数和模型训练参数。 |
| `data/raw/` | 存放原始数据文件。 |
| `data/processed/` | 存放处理后的训练集、验证集和测试集。 |
| `notebooks/main.ipynb` | 保留原始 notebook，作为实验记录和参考。 |
| `src/patient_readmission/data_loader.py` | 读取原始数据、生成二分类标签、按患者 ID 进行数据切分。 |
| `src/patient_readmission/preprocessing.py` | 处理缺失值、删除列、转换年龄、诊断编码归类、One-Hot 编码等。 |
| `src/patient_readmission/features.py` | 拆分特征、标签和患者分组信息。 |
| `src/patient_readmission/models.py` | 定义模型和超参数搜索范围。 |
| `src/patient_readmission/train.py` | 负责交叉验证、超参数调优、重复实验和模型保存。 |
| `src/patient_readmission/evaluate.py` | 计算 Accuracy、F1、Precision、Recall、ROC-AUC、Average Precision 等指标。 |
| `src/patient_readmission/explainability.py` | 提供模型系数、特征重要性和 permutation importance 等解释工具。 |
| `src/patient_readmission/visualization.py` | 保存常用图表，例如特征重要性图。 |
| `run_pipeline.py` | 运行完整机器学习流程的主入口。 |
| `main.py` | 简单兼容入口，内部调用主流程。 |
| `models/` | 保存训练好的 `.joblib` 模型。 |
| `reports/tables/` | 保存模型结果汇总表和重复实验细节。 |
| `reports/figures/` | 保存生成的图表。 |

---

## 4. 机器学习流程

项目的主要流程如下。

### 第一步：读取数据

从 `data/raw/diabetic_data.csv` 读取糖尿病患者住院记录。

### 第二步：生成二分类标签

将原始的 `readmitted` 字段转换为二分类标签：

```text
NO        -> 0
<30, >30  -> 1
```

因此，本项目关注的是一般意义上的“是否再次入院”。

### 第三步：按照患者进行数据切分，避免数据泄漏

数据中同一个患者可能有多条住院记录。如果同一个患者同时出现在训练集和测试集中，模型可能会学到与具体患者有关的信息，从而让测试结果显得过于乐观。

为了降低这种风险，本项目使用 `patient_nbr` 作为分组变量。同一个患者的所有记录会被放在同一个数据 split 中。项目使用 `StratifiedGroupKFold` 和 `GroupShuffleSplit` 等 group-aware 方法，并检查不同数据集之间的患者 ID 是否重叠。

### 第四步：特征预处理

预处理流程使用 scikit-learn transformer 构建，并放入模型 pipeline 中。这样可以避免在交叉验证中提前使用验证集或测试集的信息。

主要预处理步骤包括：

- 将 `?`、`Unknown/Invalid`、空字符串等无效值替换为缺失值。
- 删除 ID 列、常数列、近似常数列和极端稀疏列。
- 保留有临床意义的变量，例如 `max_glu_serum` 和 `A1Cresult`。
- 对偏态分布的 `time_in_hospital` 做 `log1p` 转换。
- 将年龄区间转换为数值中点。
- 将入院、出院等编码类变量作为类别变量处理。
- 将 ICD-9 诊断编码归入更粗的疾病类别。
- 对 `medical_specialty` 和 `payer_code` 等高基数类别变量进行类别合并。
- 对类别变量做 One-Hot 编码。

### 第五步：训练和调参模型

项目训练并调优了多个模型：

- Linear SVM / LinearSVC。
- Logistic Regression。
- Random Forest。
- HistGradientBoostingClassifier。

超参数调优使用 `GridSearchCV`。主要优化指标是 ROC-AUC，同时也记录 Average Precision、Accuracy、F1、Precision 和 Recall。

### 第六步：评估模型

最终模型会在独立测试集上进行评估。主要指标包括：

- **ROC-AUC**：衡量模型是否能把再次入院患者排在未再次入院患者前面。
- **Average Precision / PR-AUC**：更适合观察类别不平衡场景下模型识别正类的能力。
- **Accuracy**：整体预测正确率。
- **F1**：综合考虑 Precision 和 Recall。
- **Precision**：模型预测为再次入院的样本中，有多少是真的再次入院。
- **Recall**：真实再次入院的样本中，有多少被模型找出来。

对于再入院预测这类任务，不建议只看 Accuracy。因为如果类别分布不均衡，Accuracy 可能会掩盖模型对少数类的识别能力。

---

## 5. 原始实验主要结果

原始 notebook 中报告的测试集 ROC-AUC 结果如下：

| 模型 | Test ROC-AUC | 说明 |
|---|---:|---|
| Linear SVM | 0.6916 | 重复交叉验证表现较稳定，最佳 `C = 0.0464`。 |
| Random Forest | 0.7032 | 三个主要基线模型中整体表现最好。 |
| HistGradientBoosting | 0.7025 | 与 Random Forest 非常接近。 |
| Random Forest + Feature Selection | 0.7016 | ROC-AUC 略低，但使用了更少的特征。 |

Random Forest 特征选择 pipeline 的重复实验平均结果如下：

| 指标 | CV Mean | CV Std | Test Mean | Test Std |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.6835 | 0.0008 | 0.7014 | 0.0006 |
| Average Precision | 0.6379 | 0.0005 | 0.6550 | 0.0006 |
| Accuracy | 0.6361 | 0.0016 | 0.6456 | 0.0002 |
| F1 | 0.5406 | 0.0013 | 0.5634 | 0.0007 |
| Precision | 0.6356 | 0.0027 | 0.6518 | 0.0001 |
| Recall | 0.4704 | 0.0010 | 0.4961 | 0.0012 |

最佳重复实验中的特征选择设置为：

```text
feature_selection__estimator__max_depth = 8
feature_selection__max_features = None
feature_selection__threshold = "median"
```

整体来看，Random Forest 和 HistGradientBoosting 的预测表现最强。特征选择版本的 Random Forest 虽然 ROC-AUC 略有下降，但特征空间更小，模型也更简洁。

---

## 6. 安装依赖

建议先创建虚拟环境：

```bash
python -m venv .venv
```

在 Windows PowerShell 中激活：

```bash
.venv\Scripts\Activate.ps1
```

在 macOS 或 Linux 中激活：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果只想运行原始 notebook，核心依赖包括：

```bash
pip install pandas numpy scikit-learn matplotlib scipy tqdm joblib
```

---

## 7. 运行方法

首先确认原始数据已经放在：

```text
data/raw/diabetic_data.csv
```

然后在项目根目录运行完整重构流程：

```bash
python run_pipeline.py
```

也可以运行：

```bash
python main.py
```

运行后会生成类似下面的文件：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/best_*.joblib
reports/tables/model_summary.csv
reports/tables/repeat_details_*.csv
```

原始 notebook 也保留在：

```text
notebooks/main.ipynb
```

---

## 8. 如何理解输出结果

最主要的结果文件是：

```text
reports/tables/model_summary.csv
```

其中常见字段包括：

- `cv_roc_auc_mean`：重复交叉验证中的平均验证集 ROC-AUC。
- `cv_roc_auc_std`：重复交叉验证中验证集 ROC-AUC 的波动程度。
- `test_roc_auc_mean`：多次调参后测试集 ROC-AUC 的平均值。
- `final_test_roc_auc`：最终重新训练的最佳模型在测试集上的 ROC-AUC。
- `final_test_average_precision`：最终模型在测试集上的 PR-AUC。
- `best_params_over_repeats`：重复实验中找到的最佳超参数组合。
- `saved_pipeline_path`：训练好的 pipeline 保存位置。

ROC-AUC 越高，说明模型整体排序能力越好。Average Precision 越高，说明模型越能有效识别再次入院患者，尤其适合类别不平衡场景。

---

## 9. 后续改进方向

后续可以从以下方向继续改进：

- 更细致地调整模型参数，尤其是 Random Forest 和 HistGradientBoosting。
- 尝试更强的类别不平衡处理方法，例如 class weight、阈值移动或重采样。
- 将任务重新定义为严格的 30 天再入院预测，即 `<30` 为正类，`NO` 和 `>30` 为负类。
- 改进诊断编码相关的特征工程。
- 加入 SHAP 或 LIME，提供更细致的模型解释。
- 使用 MLflow 或 Weights & Biases 管理实验记录。
- 为数据读取、预处理、切分和评估函数添加单元测试。

---

