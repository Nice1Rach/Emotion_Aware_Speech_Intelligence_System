"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Train and evaluate an MLP classifier on emotional speech features.
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
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

# ── Train MLP classifier ─────────────────────────
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train_enc)

# ── Evaluate model ───────────────────────────────
y_pred = mlp.predict(X_test_scaled)
print("\nMLP Classifier Evaluation")
print(confusion_matrix(y_test_enc, y_pred))
print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

# ── Save model and preprocessing tools ───────────
dump(mlp, MODEL_DIR / "mlp_model.joblib")
dump(scaler, MODEL_DIR / "mlp_scaler.joblib")
dump(encoder, MODEL_DIR / "mlp_label_encoder.joblib")

print("MLP model, scaler, and encoder saved.")