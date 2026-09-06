"""
TEMPORAL CHRONOLOGICAL SPLIT EVALUATION
========================================
Trains XGBoost on 2021Q1-2023Q4, validates on 2024, tests on 2025.
Uses the 15-column dataset (with source_quarter) from the obesity-related panel.
Reports Macro-F1, accuracy, and class-specific metrics.
Does NOT target any specific result — reports whatever the data produces.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
DATASET_PATH = r"d:\Resume\Temporal-Adolescent-Anti-Obesity-and-Anti-Diabetic-Pharmacotherapy\dataset\15 Columns Model with Source Quarter\ObesityAll_14_Drugs_Adolescent_14Columns_Imputed.xlsx"

FEATURE_COLS = [
    'age_years', 'sex', 'weight_kg', 'drug_seq', 'route', 'role_cod',
    'drugname_normalized', 'rxcui', 'dose_amt', 'dose_unit', 'dose_form',
    'indi_pt', 'pt_term'
]
TARGET_COL = 'outc_cod'
QUARTER_COL = 'source_quarter'

# Temporal splits
TRAIN_QUARTERS = [f"{y}Q{q}" for y in range(2021, 2024) for q in range(1, 5)]  # 2021Q1-2023Q4
VAL_QUARTERS = [f"2024Q{q}" for q in range(1, 5)]  # 2024Q1-2024Q4
TEST_QUARTERS = [f"2025Q{q}" for q in range(1, 5)]  # 2025Q1-2025Q4

NUMERIC_COLS = ['age_years', 'weight_kg', 'drug_seq', 'dose_amt']
CATEGORICAL_COLS = [
    'sex', 'route', 'role_cod', 'drugname_normalized',
    'rxcui', 'dose_unit', 'dose_form', 'indi_pt', 'pt_term'
]

# ============================================================
# Load and split
# ============================================================
print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_excel(DATASET_PATH)
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Quarter distribution:\n{df[QUARTER_COL].value_counts().sort_index()}\n")

# Split by quarter
train_df = df[df[QUARTER_COL].isin(TRAIN_QUARTERS)]
val_df = df[df[QUARTER_COL].isin(VAL_QUARTERS)]
test_df = df[df[QUARTER_COL].isin(TEST_QUARTERS)]

print(f"Train (2021Q1-2023Q4): {len(train_df)} records")
print(f"Validation (2024):     {len(val_df)} records")
print(f"Test (2025):           {len(test_df)} records")

# ============================================================
# Preprocess
# ============================================================
def preprocess_split(train, val, test, feature_cols, target_col):
    """Encode, scale, and return X/y for each split."""
    # Combine for consistent encoding
    all_data = pd.concat([train, val, test], ignore_index=True)
    
    X_all = all_data[feature_cols].copy()
    y_all = all_data[target_col].copy()
    
    # Numeric
    for col in NUMERIC_COLS:
        X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
    X_all[NUMERIC_COLS] = X_all[NUMERIC_COLS].fillna(X_all[NUMERIC_COLS].median())
    
    # Categorical encoding
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X_all[col] = le.fit_transform(X_all[col].astype(str))
        label_encoders[col] = le
    
    # Target encoding
    le_y = LabelEncoder()
    y_all = le_y.fit_transform(y_all.astype(str))
    
    # Scale
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    # Split back
    n_train = len(train)
    n_val = len(val)
    
    X_train = X_all_scaled[:n_train]
    y_train = y_all[:n_train]
    
    X_val = X_all_scaled[n_train:n_train+n_val]
    y_val = y_all[n_train:n_train+n_val]
    
    X_test = X_all_scaled[n_train+n_val:]
    y_test = y_all[n_train+n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, le_y

X_train, y_train, X_val, y_val, X_test, y_test, le_y = preprocess_split(
    train_df, val_df, test_df, FEATURE_COLS, TARGET_COL
)

print(f"\nTrain class distribution: {dict(zip(*np.unique(le_y.inverse_transform(y_train), return_counts=True)))}")
print(f"Val class distribution:   {dict(zip(*np.unique(le_y.inverse_transform(y_val), return_counts=True)))}")
print(f"Test class distribution:  {dict(zip(*np.unique(le_y.inverse_transform(y_test), return_counts=True)))}")

# ============================================================
# Train models
# ============================================================
print("\n" + "="*60)
print("TRAINING MODELS (Temporal Split: Train 2021-2023, Test 2025)")
print("="*60)

# 1. Majority class baseline
from collections import Counter
majority_class = Counter(y_train).most_common(1)[0][0]
preds_majority = np.full(len(y_test), majority_class)
f1_majority = f1_score(y_test, preds_majority, average='macro', zero_division=0)
acc_majority = accuracy_score(y_test, preds_majority)
print(f"\n[Majority Class] Test Accuracy: {acc_majority:.4f} | Macro-F1: {f1_majority:.4f}")

# 2. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
f1_lr = f1_score(y_test, preds_lr, average='macro', zero_division=0)
acc_lr = accuracy_score(y_test, preds_lr)
print(f"[Logistic Regression] Test Accuracy: {acc_lr:.4f} | Macro-F1: {f1_lr:.4f}")

# 3. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
preds_dt = dt.predict(X_test)
f1_dt = f1_score(y_test, preds_dt, average='macro', zero_division=0)
acc_dt = accuracy_score(y_test, preds_dt)
print(f"[Decision Tree] Test Accuracy: {acc_dt:.4f} | Macro-F1: {f1_dt:.4f}")

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
f1_rf = f1_score(y_test, preds_rf, average='macro', zero_division=0)
acc_rf = accuracy_score(y_test, preds_rf)
print(f"[Random Forest] Test Accuracy: {acc_rf:.4f} | Macro-F1: {f1_rf:.4f}")

# 5. XGBoost
xgb_m = xgb.XGBClassifier(
    n_estimators=100, random_state=42, n_jobs=-1,
    eval_metric='mlogloss', verbosity=0
)
xgb_m.fit(X_train, y_train)
preds_xgb = xgb_m.predict(X_test)
f1_xgb = f1_score(y_test, preds_xgb, average='macro', zero_division=0)
acc_xgb = accuracy_score(y_test, preds_xgb)
print(f"[XGBoost] Test Accuracy: {acc_xgb:.4f} | Macro-F1: {f1_xgb:.4f}")

# Also check validation set
preds_xgb_val = xgb_m.predict(X_val)
f1_xgb_val = f1_score(y_val, preds_xgb_val, average='macro', zero_division=0)
acc_xgb_val = accuracy_score(y_val, preds_xgb_val)
print(f"\n[XGBoost Validation] Val Accuracy: {acc_xgb_val:.4f} | Val Macro-F1: {f1_xgb_val:.4f}")

# ============================================================
# Also run random split for comparison
# ============================================================
from sklearn.model_selection import train_test_split
X_all = np.vstack([X_train, X_val, X_test])
y_all = np.concatenate([y_train, y_val, y_test])
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
xgb_r = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                           eval_metric='mlogloss', verbosity=0)
xgb_r.fit(X_r_train, y_r_train)
preds_r = xgb_r.predict(X_r_test)
f1_random = f1_score(y_r_test, preds_r, average='macro', zero_division=0)
acc_random = accuracy_score(y_r_test, preds_r)
print(f"\n[XGBoost Random Split] Test Accuracy: {acc_random:.4f} | Macro-F1: {f1_random:.4f}")

# ============================================================
# Detailed classification reports
# ============================================================
class_names = le_y.classes_

print("\n" + "="*60)
print("XGBoost TEMPORAL TEST (2025) Classification Report:")
print("="*60)
print(classification_report(y_test, preds_xgb, target_names=class_names, zero_division=0))

print("\nXGBoost RANDOM SPLIT Classification Report:")
print("="*60)
print(classification_report(y_r_test, preds_r, target_names=class_names, zero_division=0))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Temporal Test Macro-F1: {f1_xgb:.4f}")
print(f"  Random Split Macro-F1: {f1_random:.4f}")
print(f"  Gap: {f1_random - f1_xgb:.4f}")
print(f"  Manuscript claims: Temporal=0.177, Random=0.625")
