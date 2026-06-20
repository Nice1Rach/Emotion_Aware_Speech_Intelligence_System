#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Voting ensemble classifier using RF, LR, KNN, and MLP.
Resolves class imbalance using SMOTE and class weighting.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse

# ── Custom utility import ──
from emotion_pipeline.processing.utils.utils_train import load_and_preprocess

warnings.simplefilter(action='ignore', category=UserWarning)

def main():
    # ── Load data ──
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess()

    # ── Combine for SMOTE then re-split ──
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    print(f"Original class distribution:\n{pd.Series(y).value_counts()}\n")

    sm = SMOTE(random_state=42)
    resample_result = sm.fit_resample(X, y)
    if isinstance(resample_result, tuple) and len(resample_result) == 3:
        X_resampled, y_resampled, _ = resample_result
    else:
        X_resampled, y_resampled = resample_result
    print(f"After SMOTE resampling:\n{pd.Series(np.ravel(y_resampled)).value_counts()}\n")

    # ── Stratified re-split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    # ── Base classifiers ──
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    knn = KNeighborsClassifier()
 
    # ── Base MLP with shared settings ──
    base_mlp = MLPClassifier(
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.2,
        learning_rate_init=0.001,
        momentum=0.9,
        nesterovs_momentum=True,
        batch_size='auto',
        activation='relu',
        random_state=42,
        n_iter_no_change=5
    )

    # ── Hyperparameter tuning grid ──
    param_grid = {
        'hidden_layer_sizes': [(128, 64), (256, 128, 64), (512, 256, 128)],
        'alpha': [1e-3, 1e-4, 1e-5],
        'learning_rate': ['constant', 'adaptive'],
        'solver': ['adam', 'lbfgs'],
    }

    # ── Grid Search ──
    grid = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=2,         # Optional: parallelize
        verbose=2          # Optional: show progress
    )
    grid.fit(X_train, y_train)
    print("Best MLP Parameters:", grid.best_params_)

    # ── Voting Ensemble ──
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('knn', knn), ('mlp', grid.best_estimator_)],
        voting='soft'
    )

    # ── Pipeline with scaler ──
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('voting', voting_clf)
    ])

    # ── Train ──
    print("Training VotingClassifier with SMOTE + class weights...")
    pipeline.fit(X_train, y_train)

    # ── Predict ──
    y_pred = pipeline.predict(X_test)

    # ── Evaluate ──
    print("\nClassification Report:\n")
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, digits=2)
    print(report)

    # Optional save to file
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/classification_report.txt", "w") as f:
        f.write(str(report))

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # ── Save Model ──
    model_path = Path("models/voting_ensemble_balanced.joblib")
    dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    labels = encoder.classes_

    # ── Plot 1: Standard Confusion Matrix ──
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix – Voting Ensemble")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plot_path1 = Path("plots/confusion")
    plot_path1.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path1 / "confusion_matrix_voting.png", dpi=300)
    plt.close()

    # ── Plot 2: Orange Heatmap Variant ──
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.title("Heatmap – Voting Ensemble")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plot_path2 = Path("plots/heatmaps")
    plot_path2.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path2 / "voting_confusion_heatmap.png", dpi=300)
    plt.close()

    print(f"Heatmaps saved to:\n• {plot_path1/'confusion_matrix_voting.png'}\n• {plot_path2/'voting_confusion_heatmap.png'}")

    # Binarize with exact class names, not integer indexes
    label_classes = list(encoder.classes_)
    y_test_bin = label_binarize(y_test, classes=label_classes, sparse_output=False)
    # Ensure y_test_bin is a dense numpy array
    y_test_bin = np.asarray(y_test_bin)
    if y_test_bin.ndim == 1:
        y_test_bin = y_test_bin.reshape(-1, 1)
    y_score = np.asarray(pipeline.predict_proba(X_test))

    print("y_test_bin shape:", y_test_bin.shape)
    print("y_score shape:", y_score.shape)
    print("label_classes:", label_classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, label in enumerate(label_classes):
        if i >= y_test_bin.shape[1] or i >= y_score.shape[1]:
            print(f"⚠️ Skipping ROC for {label}: not enough data.")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        
    # Plot ROC curves (after collecting all fpr/tpr)
    plt.figure(figsize=(10, 8))
    for i in fpr:  # Only plot classes for which ROC was computed
        label = label_classes[i]
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{label} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Emotion Class')
    plt.legend(loc='lower right')
    plt.tight_layout()
    Path("plots/roc").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/roc/roc_curves.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()