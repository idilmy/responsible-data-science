import pandas as pd
import shap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

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

# Define Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

# Calibrate the model
calibrated_model = CalibratedClassifierCV(lr_model, method="sigmoid", cv=5)
calibrated_model.fit(X_train, y_train)

# Evaluate model
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)
print(f"Logistic Regression - ROC-AUC Score: {roc_auc:.4f}")
print(f"Logistic Regression - Brier Score: {brier:.4f}")

# Plot SHAP summary
lr_trained_model = calibrated_model.calibrated_classifiers_[0].estimator
explainer = shap.Explainer(lr_trained_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary_logistic.png", dpi=300, bbox_inches='tight')
plt.close()
