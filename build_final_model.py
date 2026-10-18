#!/usr/bin/env python3
"""Build and save a final heart-disease model artifact quickly.

This script trains a single final model (default: decision tree),
selects threshold on validation for a target recall, retrains on train+val,
and exports inference artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COL = "cardio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final model artifact")
    parser.add_argument("--data", type=Path, default=Path("cardio_train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--target-recall", type=float, default=0.85)
    return parser.parse_args()


def load_and_clean(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=";")
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    df = df.drop(columns=["age"])

    clip_map = {
        "age_years": (18, 100),
        "height": (120, 220),
        "weight": (30, 250),
        "bmi": (12, 60),
        "ap_hi": (80, 240),
        "ap_lo": (40, 160),
    }
    for col, (low, high) in clip_map.items():
        df[col] = df[col].clip(lower=low, upper=high)

    df["gender"] = df["gender"].replace({1: 0, 2: 1}).astype(int)
    for col in ["cholesterol", "gluc", "smoke", "alco", "active", TARGET_COL]:
        df[col] = df[col].astype(int)

    return df


def build_pipeline() -> Pipeline:
    nominal_features = ["gender", "smoke", "alco", "active"]
    ordinal_numeric = ["cholesterol", "gluc", "age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "nominal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                nominal_features,
            ),
            (
                "ord_num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                ordinal_numeric,
            ),
        ],
        remainder="drop",
    )

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def choose_threshold(y_true: pd.Series, y_prob: np.ndarray, target_recall: float) -> float:
    best_t = 0.5
    best_p = -1.0
    for t in np.arange(0.2, 0.81, 0.01):
        pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        if rec >= target_recall and prec > best_p:
            best_t = float(t)
            best_p = float(prec)
    return best_t


def metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(args.data)
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.17647,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    pipe = build_pipeline()
    pipe.fit(x_train, y_train)

    val_prob = pipe.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val, val_prob, args.target_recall)

    pipe.fit(x_train, y_train)

    test_prob = pipe.predict_proba(x_test)[:, 1]
    test_m = metrics(y_test, test_prob, threshold)

    model_path = args.output_dir / "final_model.joblib"
    meta_path = args.output_dir / "final_model_metadata.json"
    metrics_path = args.output_dir / "final_model_test_metrics.json"

    joblib.dump(pipe, model_path)

    meta = {
        "model_name": "decision_tree",
        "selected_threshold": threshold,
        "target_recall": args.target_recall,
        "final_training_source": "train_only",
        "feature_columns": x.columns.tolist(),
        "target_column": TARGET_COL,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_m, f, indent=2)

    print("Saved:")
    print(f"- {model_path}")
    print(f"- {meta_path}")
    print(f"- {metrics_path}")
    print("\nFinal model test metrics:")
    print(json.dumps(test_m, indent=2))


if __name__ == "__main__":
    main()
