from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib

from . import config, data_prep


def _make_onehot() -> OneHotEncoder:
    # sklearn compatibility: sparse_output introduced later; older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    # Features = all except target
    X = df.drop(columns=[config.TARGET_COL])
    # Determine types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe, num_cols, cat_cols


def train(random_state: int = 42, test_size: float = 0.25) -> Dict[str, Any]:
    df = data_prep.load_processed().copy()
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Target column missing: {config.TARGET_COL}")

    # Remove rows without target
    df = df[df[config.TARGET_COL].notna()].copy()
    y = df[config.TARGET_COL].astype(int)
    X = df.drop(columns=[config.TARGET_COL])

    pipe, num_cols, cat_cols = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)

    return {
        "model_path": str(config.MODEL_PATH),
        "metrics": metrics,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def load_model() -> Pipeline:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train first with `python -m src.cli train`.")
    return joblib.load(config.MODEL_PATH)


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
