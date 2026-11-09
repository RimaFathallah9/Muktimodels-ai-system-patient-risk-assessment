"""Stage 6. Train the fused XGBoost model.

Inputs are the training-split tabular features plus cached embeddings.
The test split is not loaded in this script.
The imputer is the one fit in script 02, so imputation is not fit a second time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import load_cohort, split_frame
from src.eval.metrics import safe_auroc
from src.models.fusion import save_fusion, train_fusion
from src.models.tabular import load_tabular_bundle
from src.utils import prepare, read_json, update_val_scores, write_json


def main() -> None:
    root, cfg = prepare()
    scores = read_json(root / cfg["outputs"]["val_scores"])
    if "best_image_model" not in scores:
        raise SystemExit("Run scripts/03_train_images.py before fusion. The best image model is chosen on validation.")
    image_name = scores["best_image_model"]
    print(f"Fusion image block: {image_name}")

    cohort = load_cohort(root, cfg)
    train = split_frame(cohort, "train")
    val = split_frame(cohort, "val")
    bundle = load_tabular_bundle(root / cfg["outputs"]["checkpoints"] / "tabular.joblib")
    model, spec, val_prob = train_fusion(
        train,
        val,
        bundle["imputer"],
        bundle["features"],
        root / cfg["outputs"]["embeddings"],
        image_name,
        cfg,
    )
    checkpoint = root / cfg["outputs"]["checkpoints"] / "fusion.joblib"
    save_fusion(checkpoint, model)
    spec_path = root / cfg["outputs"]["checkpoints"] / "fusion_features.json"
    write_json(spec_path, spec)
    score = safe_auroc(val["hospital_expire_flag"], val_prob)
    print(f"Validation AUROC (not a test result): {score:.4f}")
    update_val_scores(
        root / cfg["outputs"]["val_scores"],
        "fusion",
        {"val_auroc": score, "checkpoint": checkpoint.relative_to(root).as_posix(), "image_model": image_name},
    )


if __name__ == "__main__":
    main()
