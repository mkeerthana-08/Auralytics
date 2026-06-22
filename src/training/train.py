"""
Training module for Engine Whisperer.

Fits a Random Forest classifier on the feature matrix produced by
``src.feature_extraction.extract.build_feature_matrix``.

Design choices:
  • Random Forest  — handles tabular numeric features well, robust to noisy
    data, no feature scaling required, and easy to explain:
    "a majority vote among 200 independent decision trees".
  • 80/20 split    — ensures reported accuracy reflects unseen examples.
  • Target         — ≥ 85 % accuracy on the held-out test split.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

CLASSES = ["healthy", "worn", "critical"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_features(csv_path: str) -> tuple:
    """
    Load a feature CSV produced by ``build_feature_matrix``.

    Returns
    -------
    X : float32 array, shape (n_samples, n_features)
    y : str array,     shape (n_samples,) — values in CLASSES
    """
    df = pd.read_csv(csv_path)
    X  = df.drop("label", axis=1).values.astype(np.float32)
    y  = df["label"].values
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size:    float = 0.20,
    random_state: int   = 42,
) -> tuple:
    """
    Train a Random Forest on (X, y) with a stratified 80/20 split.

    Parameters
    ----------
    X            : feature matrix
    y            : label array
    test_size    : fraction held out for testing  (default 0.20)
    random_state : reproducibility seed

    Returns
    -------
    (model, X_test, y_test, y_pred)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=200,          # 200 trees — good balance of accuracy / speed
        max_depth=None,            # trees grow fully — suitable for this feature set
        min_samples_leaf=1,
        class_weight="balanced",   # compensate for any class-count imbalance
        random_state=random_state,
        n_jobs=-1,                 # use all CPU cores
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute accuracy, confusion matrix, and per-class precision/recall.

    Returns
    -------
    dict with keys: accuracy, confusion_matrix, classification_report
    """
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred, labels=CLASSES)
    rep = classification_report(
        y_test, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    return {
        "accuracy":              acc,
        "confusion_matrix":      cm,
        "classification_report": rep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, path: str) -> None:
    """Serialise the fitted model to disk using joblib compression."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path, compress=3)
    print(f"[INFO] Model saved to: {path}")


def load_model(path: str):
    """Deserialise a previously saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run:  python -m src.training.train_runner --synthetic"
        )
    return joblib.load(path)
