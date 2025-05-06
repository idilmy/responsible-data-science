# Necessary imports
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# Import functions from evaluation.py
from evaluation import plot_confusion_matrix, print_classification_report, print_false_rates, \
    compute_equalized_odds, compute_selection_rate, confusion_matrix

# Load the dataset
file_path = "C:\\Users\\20225009\\Documents\\Responsible Data Science\\dataset\\Data\\training_v2.csv"  # insert correct path according to the directory the data is being stored at
df = pd.read_csv(file_path)

# Define the target variable
TARGET = "hospital_death"

# Drop columns that don't contribute to prediction
drop_cols = ["encounter_id", "patient_id", "hospital_id", "icu_id", 'apache_4a_icu_death_prob']
df.drop(columns=drop_cols, inplace=True)


# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols.remove(TARGET)

# Dictionary to store label encodings
label_mappings = {}

# Encode categorical variables
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

    # Save the mapping for later inspection
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Print mappings
for col, mapping in label_mappings.items():
    print(f"\nEncoding for '{col}':")
    for category, code in mapping.items():
        print(f"  {category}: {code}")

# Handle missing values
df.fillna(df.median(), inplace=True)

# Standardize numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Train/test split
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

apache4_prob = X_test['apache_4a_hospital_death_prob'].copy()  # Save separately
X_train = X_train.drop(columns=['apache_4a_hospital_death_prob'])  # Drop from training set
X_test = X_test.drop(columns=['apache_4a_hospital_death_prob'])

# Define model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
calibrated_model = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=5)
calibrated_model.fit(X_train, y_train)

# Evaluate model
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Brier Score (Calibration Check): {brier:.4f}")

# Plot SHAP
xgb_trained_model = calibrated_model.calibrated_classifiers_[0].estimator
explainer = shap.Explainer(xgb_trained_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()

## Model performance evaluation
threshold = 0.6 # the probability threshold used for the classification

# Apply threshold to get binary predictions
y_pred = (y_pred_proba >= threshold).astype(int)
apache4_pred = (apache4_prob >= threshold).astype(int)

# Evaluate performance
print("### Model Performance Evaluation ###")
plot_confusion_matrix(y_test, y_pred)
print_classification_report(y_test, y_pred)
print_false_rates(y_test, y_pred)

# Group fairness evaluation
print("\n### Group Fairness Evaluation ###")

# Ensure sensitive feature columns exist
sensitive_features = ["gender", "ethnicity"]  # Adjust based on dataset columns

for feature in sensitive_features:
    if feature in X_test.columns:
        print(f"\nFairness metrics for: {feature}")
        compute_selection_rate(y_test, y_pred, X_test[feature])
        compute_equalized_odds(y_test, y_pred, X_test[feature])
    else:
        print(f"\nWarning: Sensitive feature '{feature}' not found in dataset.")

# Next, we evaluate the APACHE IV model
print("### APACHE IV Performance Evaluation ###")
plot_confusion_matrix(y_test, apache4_pred)
print_classification_report(y_test, apache4_pred)
print_false_rates(y_test, apache4_pred)

# Group fairness evaluation
print("\n### APACHE IV Group Fairness Evaluation ###")

# Ensure sensitive feature columns exist
sensitive_features = ["gender", "ethnicity"]  # Adjust based on dataset columns

for feature in sensitive_features:
    if feature in X_test.columns:
        print(f"\nFairness metrics for: {feature}")
        compute_selection_rate(y_test, apache4_pred, X_test[feature])
        compute_equalized_odds(y_test, apache4_pred, X_test[feature])
    else:
        print(f"\nWarning: Sensitive feature '{feature}' not found in dataset.")