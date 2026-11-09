"""Stage 2. Train the tabular XGBoost baseline on the training split.

The test split is not loaded in this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import load_cohort, split_frame
from src.models.tabular import (
    fit_imputer,
    kept_feature_names,
    save_tabular_bundle,
    train_xgboost,
    transform_features,
    validation_auroc,
)
from src.utils import prepare, update_val_scores


def main() -> None:
    root, cfg = prepare()
    cohort = load_cohort(root, cfg)
    train = split_frame(cohort, "train")
    val = split_frame(cohort, "val")
    features = kept_feature_names(train)
    imputer = fit_imputer(train, features)
    model = train_xgboost(
        transform_features(train, imputer, features),
        train["hospital_expire_flag"].astype(int).to_numpy(),
        transform_features(val, imputer, features),
        val["hospital_expire_flag"].astype(int).to_numpy(),
        n_estimators=int(cfg["tabular"]["n_estimators"]),
        max_depth=int(cfg["tabular"]["max_depth"]),
        learning_rate=float(cfg["tabular"]["learning_rate"]),
        early_stopping_rounds=int(cfg["tabular"]["early_stopping_rounds"]),
        seed=int(cfg["seed"]),
    )
    checkpoint = root / cfg["outputs"]["checkpoints"] / "tabular.joblib"
    save_tabular_bundle(checkpoint, model, imputer, features)
    score = validation_auroc(model, imputer, features, val)
    print(f"Validation AUROC (not a test result): {score:.4f}")
    update_val_scores(
        root / cfg["outputs"]["val_scores"],
        "tabular",
        {"val_auroc": score, "checkpoint": checkpoint.relative_to(root).as_posix()},
    )


if __name__ == "__main__":
    main()
