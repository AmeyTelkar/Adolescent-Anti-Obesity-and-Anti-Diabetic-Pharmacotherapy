import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
DATASET_PATH = "ObesityAll_14_Drugs_Adolescent_14Columns_Imputed.xlsx"

FEATURE_COLS = [
    'age_years', 'sex', 'weight_kg', 'drug_seq', 'route', 'role_cod',
    'drugname_normalized', 'rxcui', 'dose_amt', 'dose_unit', 'dose_form',
    'indi_pt', 'pt_term'
]
TARGET_COL = 'outc_cod'
QUARTER_COL = 'source_quarter'

NUMERIC_COLS = ['age_years', 'weight_kg', 'drug_seq', 'dose_amt']
CATEGORICAL_COLS = [
    'sex', 'route', 'role_cod', 'drugname_normalized',
    'rxcui', 'dose_unit', 'dose_form', 'indi_pt', 'pt_term'
]

# Temporal splits
TRAIN_QUARTERS = [f"{y}Q{q}" for y in range(2021, 2024) for q in range(1, 5)]  # 2021-2023
VAL_QUARTERS = [f"2024Q{q}" for q in range(1, 5)]                            # 2024
TEST_QUARTERS = [f"2025Q{q}" for q in range(1, 5)]                           # 2025

# ============================================================
# Load and Split
# ============================================================
print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_excel(DATASET_PATH)

train_df = df[df[QUARTER_COL].isin(TRAIN_QUARTERS)].copy()
val_df = df[df[QUARTER_COL].isin(VAL_QUARTERS)].copy()
test_df = df[df[QUARTER_COL].isin(TEST_QUARTERS)].copy()

print(f"Train (2021-2023): {len(train_df)} records")
print(f"Validation (2024): {len(val_df)} records")
print(f"Test (2025):       {len(test_df)} records")

# ============================================================
# Preprocess (NO DATA LEAKAGE - FIT ON TRAIN ONLY)
# ============================================================

# 1. Numeric Imputation
medians = train_df[NUMERIC_COLS].median()
for col in NUMERIC_COLS:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(medians[col])
    val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(medians[col])
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(medians[col])

# 2. Categorical Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_df[CATEGORICAL_COLS] = encoder.fit_transform(train_df[CATEGORICAL_COLS].astype(str))
val_df[CATEGORICAL_COLS] = encoder.transform(val_df[CATEGORICAL_COLS].astype(str))
test_df[CATEGORICAL_COLS] = encoder.transform(test_df[CATEGORICAL_COLS].astype(str))

# 3. Target Encoding
le_y = LabelEncoder()
y_train = le_y.fit_transform(train_df[TARGET_COL].astype(str))
y_val = le_y.transform(val_df[TARGET_COL].astype(str))
y_test = le_y.transform(test_df[TARGET_COL].astype(str))

# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURE_COLS])
X_val = scaler.transform(val_df[FEATURE_COLS])
X_test = scaler.transform(test_df[FEATURE_COLS])

# ============================================================
# Temporal Evaluation
# ============================================================
print("\n============================================================")
print("XGBOOST TEMPORAL EVALUATION")
print("============================================================")

xgb_m = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss')
xgb_m.fit(X_train, y_train)

# Validation
preds_val = xgb_m.predict(X_val)
f1_val = f1_score(y_val, preds_val, average='macro', zero_division=0)
print(f"[XGBoost Validation (2024)] Macro-F1: {f1_val:.4f}")

# Test
preds_test = xgb_m.predict(X_test)
f1_test = f1_score(y_test, preds_test, average='macro', zero_division=0)
print(f"[XGBoost Temporal Test (2025)] Macro-F1: {f1_test:.4f}")

labels_present = np.unique(y_test)
target_names_present = le_y.inverse_transform(labels_present)
print("\nTest (2025) Classification Report:")
print(classification_report(y_test, preds_test, labels=labels_present, target_names=target_names_present, zero_division=0))

# ============================================================
# Random Split Comparison (No Leakage)
# ============================================================
print("\n============================================================")
print("XGBOOST RANDOM SPLIT COMPARISON")
print("============================================================")

X_all_raw = df[FEATURE_COLS].copy()
y_all_raw = df[TARGET_COL].copy()

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_all_raw, y_all_raw, test_size=0.2, random_state=42, stratify=y_all_raw
)

# Impute
medians_r = X_r_train[NUMERIC_COLS].median()
for col in NUMERIC_COLS:
    X_r_train[col] = pd.to_numeric(X_r_train[col], errors='coerce').fillna(medians_r[col])
    X_r_test[col] = pd.to_numeric(X_r_test[col], errors='coerce').fillna(medians_r[col])

# Encode
encoder_r = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_r_train[CATEGORICAL_COLS] = encoder_r.fit_transform(X_r_train[CATEGORICAL_COLS].astype(str))
X_r_test[CATEGORICAL_COLS] = encoder_r.transform(X_r_test[CATEGORICAL_COLS].astype(str))

le_y_r = LabelEncoder()
y_r_train_enc = le_y_r.fit_transform(y_r_train.astype(str))
y_r_test_enc = le_y_r.transform(y_r_test.astype(str))

# Scale
scaler_r = StandardScaler()
X_r_train_sc = scaler_r.fit_transform(X_r_train)
X_r_test_sc = scaler_r.transform(X_r_test)

xgb_r = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss')
xgb_r.fit(X_r_train_sc, y_r_train_enc)
preds_r = xgb_r.predict(X_r_test_sc)
f1_random = f1_score(y_r_test_enc, preds_r, average='macro', zero_division=0)
print(f"[XGBoost Random Split] Macro-F1: {f1_random:.4f}")

labels_r_present = np.unique(y_r_test_enc)
target_names_r_present = le_y_r.inverse_transform(labels_r_present)
print("\nRandom Split Classification Report:")
print(classification_report(y_r_test_enc, preds_r, labels=labels_r_present, target_names=target_names_r_present, zero_division=0))
