import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # For saving plots in headless environments
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.svm import SVC

# Load the dataset
file_path = "Data/training_v2.csv"
df = pd.read_csv(file_path)

# Define the target variable
TARGET = "hospital_death"

# Drop columns that don't contribute to prediction or may cause label leakage
drop_cols = [
    "encounter_id", "patient_id", "hospital_id", "icu_id",
    "readmission_status", "apache_2_diagnosis", "apache_3j_diagnosis", "apache_post_operative",
    "arf_apache", "bilirubin_apache", "bun_apache", "creatinine_apache", "fio2_apache",
    "gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache", "glucose_apache",
    "heart_rate_apache", "hematocrit_apache", "intubated_apache", "map_apache", "paco2_apache",
    "paco2_for_ph_apache", "pao2_apache", "ph_apache", "resprate_apache", "sodium_apache",
    "temp_apache", "urineoutput_apache", "ventilated_apache", "wbc_apache",
    "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob",
    "apache_3j_bodysystem", "apache_2_bodysystem"
]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols.remove(TARGET)

# Encode categorical variables
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle missing values
df.fillna(df.median(), inplace=True)

# Standardize numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Train/test split
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}
svm = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(svm, param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best SVM Parameters: {grid_search.best_params_}")
best_svm = grid_search.best_estimator_

# Calibrate the model
calibrated_svm = CalibratedClassifierCV(best_svm, method="sigmoid", cv=5)
calibrated_svm.fit(X_train, y_train)

# Evaluation
y_pred_proba_svm = calibrated_svm.predict_proba(X_test)[:, 1]
roc_auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
brier_svm = brier_score_loss(y_test, y_pred_proba_svm)

print(f"[SVM] ROC-AUC Score: {roc_auc_svm:.4f}")
print(f"[SVM] Brier Score: {brier_svm:.4f}")