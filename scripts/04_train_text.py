"""Stage 4. Fine-tune Bio_ClinicalBERT. The test split is not loaded here."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import load_cohort, split_frame
from src.models.text import train_text_model
from src.utils import get_device, prepare, update_val_scores


def main() -> None:
    root, cfg = prepare()
    device = get_device()
    cohort = load_cohort(root, cfg)
    train = split_frame(cohort, "train")
    val = split_frame(cohort, "val")
    print(f"Text training rows: train={len(train)} val={len(val)}")
    print(f"Model: {cfg['text']['model_name']}")
    checkpoint = root / cfg["outputs"]["checkpoints"] / "text.pt"
    score = train_text_model(train, val, cfg, device, checkpoint)
    print(f"Validation AUROC (not a test result): {score:.4f}")
    update_val_scores(
        root / cfg["outputs"]["val_scores"],
        "text",
        {
            "val_auroc": score,
            "model_name": cfg["text"]["model_name"],
            "checkpoint": checkpoint.relative_to(root).as_posix(),
        },
    )


if __name__ == "__main__":
    main()
