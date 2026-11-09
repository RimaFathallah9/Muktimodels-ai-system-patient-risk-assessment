"""Stage 8. SHAP for the tabular model, Grad-CAM for DenseNet, and modality shares for fusion.

These plots explain a model that is already trained. They do not change the metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.data.cohort import load_cohort, split_frame
from src.eval.metrics import align_values
from src.explain.gradcam import save_gradcam
from src.explain.shap_explain import explain_fusion_modalities, explain_tabular
from src.models.fusion import load_fusion
from src.models.image import image_probabilities, load_image_model
from src.models.tabular import load_tabular_bundle
from src.utils import get_device, prepare, read_json


def main() -> None:
    root, cfg = prepare()
    device = get_device()
    cohort = load_cohort(root, cfg)
    test = split_frame(cohort, "test")
    output_dir = root / cfg["outputs"]["explain"]
    scores = read_json(root / cfg["outputs"]["val_scores"])

    bundle = load_tabular_bundle(root / cfg["outputs"]["checkpoints"] / "tabular.joblib")
    explain_tabular(
        bundle,
        test,
        output_dir,
        sample_size=int(cfg["explain"]["shap_samples"]),
        n_patients=int(cfg["explain"]["n_patients"]),
        seed=int(cfg["seed"]),
    )

    densenet = scores["densenet121"]
    model = load_image_model(densenet["timm_name"], root / densenet["checkpoint"], device)
    hadm_ids, probs = image_probabilities(model, test, root, cfg, device)
    probs = align_values(test, hadm_ids, probs, "prob")
    save_gradcam(
        model,
        test,
        np.asarray(probs, dtype=float),
        root,
        int(cfg["image"]["size"]),
        device,
        output_dir,
        int(cfg["explain"]["gradcam_images"]),
    )

    spec = read_json(root / cfg["outputs"]["checkpoints"] / "fusion_features.json")
    fusion = load_fusion(root / cfg["outputs"]["checkpoints"] / "fusion.joblib")
    explain_fusion_modalities(
        fusion,
        spec,
        test,
        bundle["imputer"],
        root / cfg["outputs"]["embeddings"],
        output_dir,
        sample_size=int(cfg["explain"]["shap_samples"]),
        seed=int(cfg["seed"]),
    )


if __name__ == "__main__":
    main()
