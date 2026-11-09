"""Fusion: concatenate tabular features with cached image and text embeddings, then fit XGBoost.

The image and text networks are not trained again here. Their embeddings were
written to disk by scripts/05_cache_embeddings.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.tabular import positive_probability, train_xgboost, transform_features


def load_embedding_matrix(path: Path, hadm_ids) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/05_cache_embeddings.py first.")
    stored = np.load(path)
    table = pd.DataFrame(
        {
            "hadm_id": stored["hadm_id"].astype(np.int64),
            "row": np.arange(len(stored["hadm_id"])),
        }
    )
    if table["hadm_id"].duplicated().any():
        raise RuntimeError(f"Duplicate hadm_id in {path}.")
    order = pd.DataFrame({"hadm_id": np.asarray(hadm_ids, dtype=np.int64)})
    merged = order.merge(table, on="hadm_id", how="left")
    if merged["row"].isna().any() or len(merged) != len(order):
        missing = order.loc[merged["row"].isna(), "hadm_id"].head(5).tolist()
        raise RuntimeError(f"{path} is missing embeddings for hadm_id values such as {missing}.")
    index = merged["row"].to_numpy(dtype=int)
    return stored["embedding"][index]


def stack_features(tabular: np.ndarray, image: np.ndarray, text: np.ndarray) -> np.ndarray:
    return np.hstack([tabular, image, text]).astype(np.float32)


def fusion_matrix(frame: pd.DataFrame, imputer, features: list[str], embedding_dir: Path, image_model_name: str) -> np.ndarray:
    tabular = transform_features(frame, imputer, features)
    image = load_embedding_matrix(embedding_dir / f"{image_model_name}_{_split_name(frame)}.npz", frame["hadm_id"])
    text = load_embedding_matrix(embedding_dir / f"text_{_split_name(frame)}.npz", frame["hadm_id"])
    return stack_features(tabular, image, text)


def _split_name(frame: pd.DataFrame) -> str:
    splits = frame["split"].unique().tolist()
    if len(splits) != 1:
        raise RuntimeError("Build the fusion matrix on one split at a time.")
    return str(splits[0])


def train_fusion(train_frame, val_frame, imputer, features, embedding_dir, image_model_name, cfg) -> tuple:
    image_train = load_embedding_matrix(
        embedding_dir / f"{image_model_name}_train.npz", train_frame["hadm_id"]
    )
    text_train = load_embedding_matrix(embedding_dir / "text_train.npz", train_frame["hadm_id"])
    x_train = stack_features(
        transform_features(train_frame, imputer, features),
        image_train,
        text_train,
    )
    x_val = fusion_matrix(val_frame, imputer, features, embedding_dir, image_model_name)
    model = train_xgboost(
        x_train,
        train_frame["hospital_expire_flag"].astype(int).to_numpy(),
        x_val,
        val_frame["hospital_expire_flag"].astype(int).to_numpy(),
        n_estimators=int(cfg["fusion"]["n_estimators"]),
        max_depth=int(cfg["fusion"]["max_depth"]),
        learning_rate=float(cfg["fusion"]["learning_rate"]),
        early_stopping_rounds=int(cfg["fusion"]["early_stopping_rounds"]),
        seed=int(cfg["seed"]),
    )
    # Block sizes let SHAP split the concatenated vector back into modalities.
    spec = {
        "image_model": image_model_name,
        "tabular_features": features,
        "tabular_dim": int(len(features)),
        "image_dim": int(image_train.shape[1]),
        "text_dim": int(text_train.shape[1]),
    }
    return model, spec, positive_probability(model, x_val)


def save_fusion(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_fusion(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/06_train_fusion.py first.")
    return joblib.load(path)
