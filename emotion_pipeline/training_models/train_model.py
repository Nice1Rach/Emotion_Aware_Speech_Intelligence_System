"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Train and evaluate multiple classifiers (Random Forest, MLP, Naive Bayes)
on emotional speech feature data and save each trained model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from pathlib import Path

# ── Paths ─────────────────────────────────────────────
TRAIN_PATH = Path("features/splits/train.csv")
TEST_PATH = Path("features/splits/test.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load training and testing CSVs, split features/labels, and encode labels.
    Returns: X_train, X_test, y_train_enc, y_test_enc, label_encoder
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Split features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Encode labels numerically
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    return X_train, X_test, y_train_enc, y_test_enc, encoder

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, encoder):
    """
    Fit model, evaluate performance, print metrics, and save trained model.
    """
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print(f"\nEvaluation Report for {model_name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save trained model
    model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
    dump(model, model_path)
    print(f"{model_name} model saved to {model_path}\n")

if __name__ == "__main__":
    # Load dataset
    X_train, X_test, y_train, y_test, encoder = load_data()

    # ── Define models to train ─────────────────────────
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "Naive Bayes": GaussianNB()
    }

    # ── Train each model ──────────────────────────────
    for name, model in models.items():
        train_and_evaluate_model(model, name, X_train, y_train, X_test, y_test, encoder)