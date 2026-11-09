"""SHAP explanations for the tabular model and for modality blocks in the fused model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.fusion import fusion_matrix
from src.models.tabular import positive_probability, transform_features


def _shap_matrix(explainer, features: np.ndarray) -> tuple[np.ndarray, float]:
    explanation = explainer(features)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # Some SHAP versions return one matrix per class. The positive class is last.
        values = values[:, :, -1]
    base = explanation.base_values
    base_array = np.asarray(base)
    if base_array.ndim > 0:
        base_value = float(base_array.reshape(-1)[-1])
    else:
        base_value = float(base_array)
    return values, base_value


def _sample_rows(count: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    take = min(count, sample_size)
    return np.sort(rng.choice(count, size=take, replace=False))


def explain_tabular(bundle: dict, test_frame: pd.DataFrame, output_dir: Path, sample_size: int, n_patients: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = bundle["model"]
    features = bundle["features"]
    matrix = transform_features(test_frame, bundle["imputer"], features)
    probabilities = positive_probability(model, matrix)
    row_index = _sample_rows(len(matrix), sample_size, seed)
    explainer = shap.TreeExplainer(model)
    values, _base_value = _shap_matrix(explainer, matrix[row_index])

    plt.figure()
    shap.summary_plot(
        values,
        matrix[row_index],
        feature_names=features,
        show=False,
        max_display=len(features),
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    labels = test_frame["hospital_expire_flag"].to_numpy()
    hadm_ids = test_frame["hadm_id"].to_numpy()
    chosen = _example_indexes(probabilities, n_patients)
    for rank, index in enumerate(chosen, start=1):
        one_values, one_base = _shap_matrix(explainer, matrix[index : index + 1])
        explanation = shap.Explanation(
            values=one_values[0],
            base_values=one_base,
            data=matrix[index],
            feature_names=features,
        )
        shap.plots.waterfall(explanation, show=False, max_display=12)
        figure = plt.gcf()
        figure.axes[0].set_title(
            f"hadm_id={int(hadm_ids[index])}  true={int(labels[index])}  predicted={probabilities[index]:.2f}"
        )
        path = output_dir / f"shap_patient_{rank}_hadm_{int(hadm_ids[index])}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        print(f"Wrote {path.name}")
    print(f"Wrote {output_dir / 'shap_summary.png'}")


def _example_indexes(probabilities: np.ndarray, n_patients: int) -> list[int]:
    """Highest risk, lowest risk, and the patient closest to 0.5."""
    order = np.argsort(probabilities)
    picks = [int(order[-1]), int(order[0])]
    closest = int(np.argmin(np.abs(probabilities - 0.5)))
    if closest not in picks:
        picks.append(closest)
    return picks[:n_patients]


def explain_fusion_modalities(
    model,
    spec: dict,
    test_frame: pd.DataFrame,
    imputer,
    embedding_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    """Sum of mean(|SHAP|) inside each block of the concatenated vector."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = fusion_matrix(test_frame, imputer, spec["tabular_features"], embedding_dir, spec["image_model"])
    row_index = _sample_rows(len(matrix), sample_size, seed)
    explainer = shap.TreeExplainer(model)
    values, _base = _shap_matrix(explainer, matrix[row_index])
    mean_abs = np.mean(np.abs(values), axis=0)

    n_tab = int(spec["tabular_dim"])
    n_image = int(spec["image_dim"])
    n_text = int(spec["text_dim"])
    if n_tab + n_image + n_text != mean_abs.shape[0]:
        raise RuntimeError("Fusion feature blocks do not add up to the SHAP vector.")
    blocks = {
        "tabular": mean_abs[:n_tab],
        "image": mean_abs[n_tab : n_tab + n_image],
        "text": mean_abs[n_tab + n_image :],
    }
    total = float(sum(float(part.sum()) for part in blocks.values()))
    rows = []
    for name, part in blocks.items():
        block_sum = float(part.sum())
        rows.append(
            {
                "modality": name,
                "n_features": int(part.shape[0]),
                "sum_mean_abs_shap": block_sum,
                "share_of_total": block_sum / total if total else float("nan"),
                "mean_abs_shap_per_feature": float(part.mean()) if len(part) else float("nan"),
                "n_test_patients_in_sample": int(len(row_index)),
            }
        )
    report = pd.DataFrame(rows)
    csv_path = output_dir / "fusion_modality_shap.csv"
    report.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(report["modality"], report["share_of_total"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share of total mean |SHAP|")
    axis.set_title("Fused model: SHAP share by modality")
    figure.tight_layout()
    figure.savefig(output_dir / "fusion_modality_shap.png", dpi=140)
    plt.close(figure)

    note = output_dir / "fusion_modality_shap_note.txt"
    note.write_text(
        "share_of_total adds the mean absolute SHAP of every feature in that block.\n"
        "The image and text blocks have many more features than the tabular block,\n"
        "so a large share can come from the number of columns. mean_abs_shap_per_feature\n"
        "is the average inside the block. This file is computed from the seeded test sample.\n",
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(f"Wrote {csv_path}")
