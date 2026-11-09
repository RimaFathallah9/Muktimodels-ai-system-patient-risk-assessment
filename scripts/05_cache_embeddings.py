"""Stage 5. Save penultimate embeddings so fusion does not rerun the encoders.

Test embeddings are cached here for script 07. Script 06 trains only on train and validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.cohort import load_cohort, split_frame
from src.models.image import image_embeddings, load_image_model
from src.models.text import load_text_model, text_embeddings
from src.utils import ensure_dir, get_device, prepare, read_json


def _save(path: Path, hadm_ids, embedding) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, hadm_id=hadm_ids, embedding=embedding)
    print(f"Wrote {path} shape={embedding.shape}")


def main() -> None:
    root, cfg = prepare()
    device = get_device()
    scores = read_json(root / cfg["outputs"]["val_scores"])
    if "best_image_model" not in scores or "text" not in scores:
        raise SystemExit("Run scripts 03 and 04 before caching embeddings.")

    image_name = scores["best_image_model"]
    image_info = scores[image_name]
    text_info = scores["text"]
    print(f"Caching embeddings from image model '{image_name}' (chosen by validation AUROC).")

    cohort = load_cohort(root, cfg)
    image_model = load_image_model(image_info["timm_name"], root / image_info["checkpoint"], device)
    text_model = load_text_model(cfg, root / text_info["checkpoint"], device)
    embedding_dir = root / cfg["outputs"]["embeddings"]

    for split in ("train", "val", "test"):
        frame = split_frame(cohort, split)
        hadm_ids, vectors = image_embeddings(image_model, frame, root, cfg, device)
        _save(embedding_dir / f"{image_name}_{split}.npz", hadm_ids, vectors)
        hadm_ids, vectors = text_embeddings(text_model, frame, cfg, device)
        _save(embedding_dir / f"text_{split}.npz", hadm_ids, vectors)


if __name__ == "__main__":
    main()
