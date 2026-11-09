"""XGBoost on age, sex, and first-day labs.

Median imputation is fit on the training split only.
scale_pos_weight is the training-split count of negatives divided by positives.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from src.data.schema import tabular_feature_names
from src.eval.metrics import safe_auroc


def kept_feature_names(train: pd.DataFrame) -> list[str]:
    kept = []
    for name in tabular_feature_names():
        if train[name].notna().any():
            kept.append(name)
        else:
            print(f"WARNING: dropping {name} because every training value is missing.")
    if "age" not in kept or "sex_male" not in kept:
        raise RuntimeError("Age and sex must be present in the training split.")
    return kept


def fit_imputer(train: pd.DataFrame, features: list[str]) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train[features])
    return imputer


def transform_features(frame: pd.DataFrame, imputer: SimpleImputer, features: list[str]) -> np.ndarray:
    return imputer.transform(frame[features])


def scale_pos_weight(y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train).astype(int)
    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise RuntimeError("The training split needs both classes.")
    weight = n_negative / n_positive
    print(f"scale_pos_weight = {weight:.3f} (train negatives / train positives)")
    return float(weight)


def train_xgboost(
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    early_stopping_rounds: int,
    seed: int,
) -> XGBClassifier:
    weight = scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        early_stopping_rounds=early_stopping_rounds,
    )
    # Early stopping watches validation AUROC. The test split is not passed in.
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    best_iteration = getattr(model, "best_iteration", None)
    print(f"XGBoost best_iteration = {best_iteration}")
    return model


def positive_probability(model: XGBClassifier, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def save_tabular_bundle(path: Path, model, imputer, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "imputer": imputer, "features": features}, path)


def load_tabular_bundle(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/02_train_tabular.py first.")
    return joblib.load(path)


def validation_auroc(model, imputer, features, frame: pd.DataFrame) -> float:
    probabilities = positive_probability(model, transform_features(frame, imputer, features))
    return safe_auroc(frame["hospital_expire_flag"], probabilities)
