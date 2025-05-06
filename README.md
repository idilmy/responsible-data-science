# Responsible-Data-Science-Assignment
This is the repository of Group 32 for the Assignment in the course "Responsible Data Science"

Overview
--------
This project trains and evaluates multiple machine learning models to predict hospital mortality using patient data. It also includes model calibration, interpretability, and fairness analysis.

Dataset
-------
Make sure the dataset file is located at:
    Data/training_v2.csv

Python Version
--------------
Tested on Python 3.8+

Required Packages
-----------------
Install dependencies in the requirements.txt
(Required packages include: pandas, scikit-learn, xgboost, shap, matplotlib, seaborn, fairlearn)

Scripts
-------

1. LogisticRegression.py
   - Trains a Logistic Regression model with L2 regularization
   - Handles preprocessing: missing values, scaling, encoding
   - Evaluates using ROC-AUC and Brier score
   - Calibrates the model using sigmoid method
   - Saves SHAP summary plot as shap_summary_logreg.png

2. RandomForest.py
   - Trains a calibrated Random Forest classifier
   - Same preprocessing as above
   - Evaluates using ROC-AUC and Brier score
   - Saves SHAP plot: shap_summary_rf.png

3. SVM.py
   - Trains and tunes a Support Vector Machine (SVM) via GridSearchCV
   - Same preprocessing as above
   - Calibrates the best model and evaluates performance

4. XGBoost_model.py
   - Trains a calibrated XGBoost classifier
   - Same preprocessing as above
   - Evaluates using ROC-AUC and Brier score
   - Saves SHAP plot: shap_summary.png
  
5. XGBoost.py
   - Trains a calibrated XGBoost classifier
   - Handles preprocessing: encoding, scaling, missing values
   - Drops irrelevant and data leakage columns
   - Evaluates model performance using ROC-AUC, Brier score, confusion matrix, and classification report
   - Generates SHAP summary plot saved as shap_summary.png
   - Compares model predictions against APACHE IV scoring
   - Evaluates group fairness (selection rate, equalized odds) on 'gender' and 'ethnicity'

6. evaluation.py
   - Utility functions to evaluate classification models
   - Includes confusion matrix, classification report, false rates
   - Also computes group fairness metrics (demographic parity, equalized odds)

