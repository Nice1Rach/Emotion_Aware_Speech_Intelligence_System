"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Train and evaluate a Random Forest classifier on emotional speech features.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from pathlib import Path

# ── Paths ─────────────────────────────────────────
TRAIN_CSV = Path("features/splits/train.csv")
TEST_CSV = Path("features/splits/test.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load training and test data ──────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# ── Encode labels ────────────────────────────────
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# ── Scale features ───────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Train Random Forest Classifier ───────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_enc)

# ── Evaluate model ───────────────────────────────
y_pred = rf.predict(X_test_scaled)
print("\nRandom Forest Evaluation")
print(confusion_matrix(y_test_enc, y_pred))
print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

# ── Save model and preprocessing tools ───────────
dump(rf, MODEL_DIR / "rf_model.joblib")
dump(scaler, MODEL_DIR / "rf_scaler.joblib")
dump(encoder, MODEL_DIR / "label_encoder.joblib")

print("RF model, scaler, and encoder saved.")