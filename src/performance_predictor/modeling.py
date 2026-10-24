"""Leakage-aware preprocessing, model training, evaluation, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import FEATURE_COLUMNS, TARGET


def build_pipeline(seed: int = 42) -> Pipeline:
    numeric = [c for c in FEATURE_COLUMNS if c != "department"]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("department", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), ["department"]),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )),
    ])


def train_evaluate(data: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict, pd.DataFrame]:
    X, y = data[FEATURE_COLUMNS], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=seed, stratify=y
    )
    model = build_pipeline(seed)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    labels = ["Low", "Medium", "High"]
    metrics = {
        "note": "Synthetic holdout evaluation; not evidence of real-world validity.",
        "rows_total": int(len(data)), "rows_train": int(len(X_train)), "rows_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "labels_order": labels,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "classification_report": classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0),
    }
    predictions = X_test.copy()
    predictions["actual_rating"] = y_test
    predictions["predicted_rating"] = predicted
    for i, label in enumerate(model.named_steps["classifier"].classes_):
        predictions[f"score_{label.lower()}"] = model.predict_proba(X_test)[:, i]
    return model, metrics, predictions


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python main.py` first.")
    return joblib.load(path)
