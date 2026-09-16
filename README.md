# Credit Risk & Default Prediction

An end-to-end Data Science project for predicting whether a credit card customer is likely to default on payment in the following month.

The project covers the complete machine learning workflow, including data understanding, cleaning, exploratory data analysis, feature engineering, model comparison, class imbalance handling, hyperparameter tuning, classification threshold optimization, and final model evaluation.

---

## 1. Project Overview

Credit default prediction is an important problem in financial risk management.

The objective of this project is to build a machine learning model that can identify customers with a higher probability of payment default using historical customer, repayment, billing, and payment information.

The target variable is:

- `0` — No default in the following month
- `1` — Default in the following month

The project focuses not only on overall prediction accuracy but also on detecting customers who actually default.

For this reason, metrics such as **Recall, F1-score, ROC-AUC, and PR-AUC** are considered alongside Accuracy.

---

## 2. Dataset

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

Dataset size:

- **30,000 customers**
- **23 original input features**
- **1 target variable**

The dataset contains information about:

- Credit limit
- Customer demographics
- Repayment status
- Monthly bill amounts
- Monthly payment amounts
- Default status in the following month

### Main Feature Groups

#### Customer Information

- `LIMIT_BAL`
- `SEX`
- `EDUCATION`
- `MARRIAGE`
- `AGE`

#### Repayment History

- `PAY_0`
- `PAY_2`
- `PAY_3`
- `PAY_4`
- `PAY_5`
- `PAY_6`

#### Bill Amounts

- `BILL_AMT1` — `BILL_AMT6`

#### Payment Amounts

- `PAY_AMT1` — `PAY_AMT6`

#### Target

- `DEFAULT_PAYMENT_NEXT_MONTH`

---

## 3. Project Workflow

```text
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Validation / Test Split
        ↓
Baseline Modeling
        ↓
Class Imbalance Experiments
        ↓
Random Forest
        ↓
XGBoost
        ↓
Hyperparameter Tuning
        ↓
Threshold Optimization
        ↓
Final Test Evaluation
        ↓
Feature Importance
        ↓
Model Persistence
```

---

## 4. Data Cleaning

Several data quality issues were investigated before modeling.

### Education

Undocumented education values were grouped into the `Others` category.

### Marriage

Undocumented marriage values were grouped into the `Others` category.

### Missing Values

No missing-value imputation was required.

### Repayment Status

Special repayment-status values were preserved rather than removed because they represent meaningful historical account behavior.

### Bill and Payment Values

Negative bill amounts and zero payment amounts were retained because they may represent valid financial behavior rather than data errors.

### Data Leakage Prevention

Explicit leakage checks were added to ensure that the target variable or target-derived features were not included in the model input.

---

## 5. Exploratory Data Analysis

EDA was performed to investigate relationships between customer behavior and payment default.

The analysis focused on:

- Target class distribution
- Customer age
- Credit limit
- Demographic variables
- Repayment delays
- Bill amounts
- Payment amounts
- Credit utilization
- Default behavior across repayment patterns

Because the target variable is moderately imbalanced, Accuracy alone was not used as the primary model evaluation metric.

---

## 6. Feature Engineering

Additional behavioral features were created from the historical repayment and transaction information.

Examples include:

- `NUM_DELAYED_MONTHS`
- `MAX_PAYMENT_DELAY`
- `HAS_PAYMENT_DELAY`
- `RECENT_DELAY`
- `AVG_BILL_AMT`
- `MAX_BILL_AMT`
- `AVG_PAY_AMT`
- `TOTAL_PAY_AMT`
- `ZERO_PAYMENT_MONTHS`
- `CREDIT_UTILIZATION`
- `AVG_CREDIT_UTILIZATION`
- `PAYMENT_TO_BILL_RATIO`
- `BILL_CHANGE`

The original variables were preserved alongside the engineered features.

---

## 7. Data Splitting

The dataset was split using stratified sampling:

- **70% Training**
- **15% Validation**
- **15% Test**

| Dataset    | Records |
| ---------- | ------: |
| Training   |  21,000 |
| Validation |   4,500 |
| Test       |   4,500 |

The test dataset was kept untouched during model development, model selection, hyperparameter tuning, and threshold optimization.

---

## 8. Models Evaluated

The following models were evaluated:

1. Dummy Classifier
2. Logistic Regression
3. Logistic Regression with balanced class weights
4. Random Forest
5. Random Forest with balanced class weights
6. XGBoost
7. XGBoost with class imbalance handling

### Validation Performance

| Model                          | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ------------------------------ | -------: | --------: | -----: | -----: | ------: | -----: |
| Dummy Classifier               |   0.7789 |    0.0000 | 0.0000 | 0.0000 |  0.5000 | 0.2211 |
| Logistic Regression            |   0.8098 |    0.6433 | 0.3136 | 0.4216 |  0.7556 | 0.5109 |
| Logistic Regression (Balanced) |   0.7464 |    0.4455 | 0.6000 | 0.5113 |  0.7601 | 0.5081 |
| Random Forest                  |   0.8140 |    0.6391 | 0.3648 | 0.4645 |  0.7589 | 0.5340 |
| Random Forest (Balanced)       |   0.7978 |    0.5492 | 0.4764 | 0.5102 |  0.7624 | 0.5320 |
| XGBoost                        |   0.8160 |    0.6538 | 0.3568 | 0.4616 |  0.7792 | 0.5532 |
| XGBoost (Balanced)             |   0.7587 |    0.4658 | 0.6231 | 0.5331 |  0.7784 | 0.5554 |

XGBoost provided the strongest overall predictive performance among the evaluated model families.

Balanced XGBoost achieved particularly strong Recall, F1-score, and PR-AUC, making it a strong candidate for further optimization.

---

## 9. Hyperparameter Tuning

XGBoost was selected for further optimization.

`RandomizedSearchCV` was used to tune important parameters including:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

Average Precision was used as the optimization metric because of the imbalanced classification problem.

Class imbalance was handled using:

```python
scale_pos_weight = negative_samples / positive_samples
```

---

## 10. Classification Threshold Optimization

The default classification threshold of `0.50` was not assumed to be optimal.

Thresholds between `0.20` and `0.70` were evaluated using the validation set.

The selected threshold was **0.55**.

At this threshold, the validation results were approximately:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 78.44% |
| Precision | 51.09% |
| Recall    | 59.10% |
| F1-score  | 54.80% |

The threshold of `0.55` produced the highest validation F1-score and provided a better balance between Precision and Recall.

---

## 11. Final Model

The final model uses:

- Tuned XGBoost
- Class imbalance handling
- Engineered behavioral features
- Classification threshold = **0.55**

After model and threshold selection were completed, the training and validation datasets were combined to retrain the final model.

The model was then evaluated once on the previously untouched test dataset.

---

## 12. Final Test Results

| Metric    | Final Test Score |
| --------- | ---------------: |
| Accuracy  |       **78.38%** |
| Precision |       **51.00%** |
| Recall    |       **59.14%** |
| F1-score  |       **54.77%** |
| ROC-AUC   |       **78.55%** |
| PR-AUC    |       **56.61%** |

The final test results are very close to the validation results.

This suggests that the selected model and classification threshold generalize consistently to unseen data.

The final model detects approximately **59% of actual default cases** while maintaining approximately **51% precision**.

---

## 13. Model Explainability

XGBoost feature importance is used to identify which features contribute most strongly to the model predictions.

Feature importance results are saved in:

```text
reports/feature_importance.csv
```

---

## 14. Project Structure

```text
credit-risk-default-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebook/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_preparation.ipynb
│   ├── 06_baseline_modeling.ipynb
│   ├── 07_tree_models.ipynb
│   ├── 08_model_tuning.ipynb
│   └── 09_final_model.ipynb
│
├── src/
│   ├── __init__.py
│   └── modeling_utils.py
│
├── reports/
│   ├── baseline_results.csv
│   ├── model_comparison.csv
│   ├── xgb_tuning_results.csv
│   ├── threshold_results.csv
│   ├── final_model_selection.csv
│   ├── final_test_results.csv
│   └── feature_importance.csv
│
├── models/
│   ├── final_xgboost_model.joblib
│   └── model_config.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 15. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## 16. Installation

Clone the repository:

```bash
git clone <repository-url>
cd credit-risk-default-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 17. Running the Project

The notebooks are designed to be executed in order:

```text
01_data_understanding.ipynb
        ↓
02_data_cleaning.ipynb
        ↓
03_exploratory_data_analysis.ipynb
        ↓
04_feature_engineering.ipynb
        ↓
05_model_preparation.ipynb
        ↓
06_baseline_modeling.ipynb
        ↓
07_tree_models.ipynb
        ↓
08_model_tuning.ipynb
        ↓
09_final_model.ipynb
```

---

## 18. Key Takeaways

- Accuracy can be misleading in imbalanced classification problems.
- Increasing minority-class weight can significantly improve Recall.
- Improving Recall usually introduces a Precision trade-off.
- Tree-based and boosting models can capture relationships that linear models may miss.
- Classification threshold selection is an important part of model development.
- Validation data should be used for model and threshold selection.
- The test dataset should remain untouched until final evaluation.
- Data leakage can produce unrealistically strong model performance and must be actively checked.

---

## 19. Future Improvements

Possible future extensions include:

- SHAP-based model explainability
- Probability calibration
- Cost-sensitive decision optimization
- Fairness analysis across customer groups
- REST API deployment using FastAPI
- Interactive prediction application using Streamlit
- Docker-based deployment
- Model monitoring

---

## Conclusion

This project demonstrates an end-to-end approach to credit default prediction, from raw data exploration through final model evaluation.

After comparing multiple classification approaches, XGBoost was selected as the final model family.

Hyperparameter tuning and classification threshold optimization improved the balance between identifying default customers and limiting false-positive predictions.

The final model achieved a test ROC-AUC of **78.55%**, PR-AUC of **56.61%**, and Recall of **59.14%** on previously unseen data.
