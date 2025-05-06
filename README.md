# Responsible Data Science: Hospital Mortality Prediction

Machine learning models for hospital mortality prediction with calibration, interpretability, and fairness analysis.

---

## Overview

This project trains and evaluates multiple machine learning models to predict hospital mortality using patient data. It also includes model calibration, interpretability (via SHAP), and fairness analysis (via Fairlearn).

---

## Dataset

Due to data confidentiality, the dataset is **not included** in this repository.  
Please ensure the dataset file is located at: Data/training_v2.csv



---

## Python Version

Tested on **Python 3.8+**

---

## Required Packages

Install dependencies from `requirements.txt`.  
(Required packages include: `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `fairlearn`)

---

## Scripts Overview

### `LogisticRegression.py`
- Trains a Logistic Regression model with L2 regularization
- Handles missing values, scaling, encoding
- Evaluates using ROC-AUC and Brier score
- Calibrates using sigmoid method
- Saves SHAP summary plot as `shap_summary_logreg.png`

### `RandomForest.py`
- Trains a calibrated Random Forest classifier
- Same preprocessing steps
- Saves SHAP plot: `shap_summary_rf.png`

### `SVM.py`
- Trains and tunes an SVM model via GridSearchCV
- Calibrates and evaluates

### `XGBoost_model.py`
- Calibrated XGBoost model
- Evaluates ROC-AUC and Brier
- Saves SHAP plot: `shap_summary.png`

### `XGBoost.py`
- Same as above, with additional fairness metrics
- Includes APACHE IV model comparison
- Plots confusion matrix and evaluates demographic parity/equalized odds

### `evaluation.py`
- Utility functions for:
  - Confusion matrix, classification report, false rates
  - Fairness metrics like demographic parity and equalized odds

---

## Author

Group 32 — TU/e Responsible Data Science  
Maintained by [@idilmy](https://github.com/idilmy)
