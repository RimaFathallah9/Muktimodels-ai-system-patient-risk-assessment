"""Stage 3. Fine-tune DenseNet-121 and ViT on the training split.

Validation AUROC decides which checkpoint is kept.
The test split is not loaded in this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import load_cohort, split_frame
from src.models.image import train_image_model
from src.utils import get_device, prepare, update_val_scores


def main() -> None:
    root, cfg = prepare()
    device = get_device()
    cohort = load_cohort(root, cfg)
    train = split_frame(cohort, "train")
    val = split_frame(cohort, "val")
    print(f"Image training rows: train={len(train)} val={len(val)}")

    for backbone in cfg["image"]["backbones"]:
        name = backbone["name"]
        timm_name = backbone["timm_name"]
        checkpoint = root / cfg["outputs"]["checkpoints"] / f"{name}.pt"
        print(f"\nTraining {name} ({timm_name})")
        score = train_image_model(train, val, root, cfg, device, timm_name, checkpoint)
        print(f"Validation AUROC for {name} (not a test result): {score:.4f}")
        update_val_scores(
            root / cfg["outputs"]["val_scores"],
            name,
            {
                "val_auroc": score,
                "timm_name": timm_name,
                "checkpoint": checkpoint.relative_to(root).as_posix(),
            },
        )


if __name__ == "__main__":
    main()
