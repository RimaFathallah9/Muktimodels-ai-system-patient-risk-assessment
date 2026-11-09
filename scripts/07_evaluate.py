"""Stage 7. Score every model on the test split and write the results table.

F1 thresholds are chosen on validation inside this script.
Nothing is fit here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.checks import assert_subject_splits
from src.data.cohort import load_cohort, split_frame
from src.eval.metrics import (
    align_values,
    metric_row,
    save_calibration_plot,
    tune_f1_threshold,
    write_results,
)
from src.models.fusion import fusion_matrix, load_fusion
from src.models.image import image_probabilities, load_image_model
from src.models.tabular import load_tabular_bundle, positive_probability, transform_features
from src.models.text import load_text_model, text_probabilities
from src.utils import get_device, prepare, read_json


def _scores_for_split(name, y_val, p_val, y_test, p_test, source):
    threshold = tune_f1_threshold(y_val, p_val)
    return metric_row(name, y_test, p_test, threshold, source)


def main() -> None:
    root, cfg = prepare()
    device = get_device()
    cohort = load_cohort(root, cfg)
    assert_subject_splits(cohort)
    meta = read_json((root / cfg["cohort"]["output_path"]).with_name("cohort_meta.json"))
    source = str(meta["source"])
    if meta.get("synthetic"):
        print("SYNTHETIC DATA. The table written by this script is a pipeline check, not a result.")

    val = split_frame(cohort, "val")
    test = split_frame(cohort, "test")
    scores = read_json(root / cfg["outputs"]["val_scores"])
    metrics_dir = root / cfg["outputs"]["metrics"]
    rows = []
    prediction_table = test[["hadm_id", "subject_id", "hospital_expire_flag"]].copy()

    bundle = load_tabular_bundle(root / cfg["outputs"]["checkpoints"] / "tabular.joblib")
    tabular_val = positive_probability(
        bundle["model"], transform_features(val, bundle["imputer"], bundle["features"])
    )
    tabular_test = positive_probability(
        bundle["model"], transform_features(test, bundle["imputer"], bundle["features"])
    )
    rows.append(_scores_for_split("tabular", val["hospital_expire_flag"], tabular_val, test["hospital_expire_flag"], tabular_test, source))
    prediction_table["p_tabular"] = tabular_test
    save_calibration_plot(test["hospital_expire_flag"], tabular_test, "Tabular", metrics_dir / "calibration_tabular.png")

    for backbone in cfg["image"]["backbones"]:
        name = backbone["name"]
        info = scores[name]
        model = load_image_model(info["timm_name"], root / info["checkpoint"], device)
        val_ids, val_prob = image_probabilities(model, val, root, cfg, device)
        test_ids, test_prob = image_probabilities(model, test, root, cfg, device)
        val_prob = align_values(val, val_ids, val_prob, "prob")
        test_prob = align_values(test, test_ids, test_prob, "prob")
        rows.append(_scores_for_split(name, val["hospital_expire_flag"], val_prob, test["hospital_expire_flag"], test_prob, source))
        prediction_table[f"p_{name}"] = test_prob
        save_calibration_plot(test["hospital_expire_flag"], test_prob, name, metrics_dir / f"calibration_{name}.png")
        del model

    text_model = load_text_model(cfg, root / scores["text"]["checkpoint"], device)
    val_ids, val_prob = text_probabilities(text_model, val, cfg, device)
    test_ids, test_prob = text_probabilities(text_model, test, cfg, device)
    val_prob = align_values(val, val_ids, val_prob, "prob")
    test_prob = align_values(test, test_ids, test_prob, "prob")
    rows.append(_scores_for_split("text", val["hospital_expire_flag"], val_prob, test["hospital_expire_flag"], test_prob, source))
    prediction_table["p_text"] = test_prob
    save_calibration_plot(test["hospital_expire_flag"], test_prob, "Text", metrics_dir / "calibration_text.png")
    del text_model

    spec = read_json(root / cfg["outputs"]["checkpoints"] / "fusion_features.json")
    fusion = load_fusion(root / cfg["outputs"]["checkpoints"] / "fusion.joblib")
    fusion_val = positive_probability(
        fusion, fusion_matrix(val, bundle["imputer"], bundle["features"], root / cfg["outputs"]["embeddings"], spec["image_model"])
    )
    fusion_test = positive_probability(
        fusion, fusion_matrix(test, bundle["imputer"], bundle["features"], root / cfg["outputs"]["embeddings"], spec["image_model"])
    )
    rows.append(_scores_for_split("fusion", val["hospital_expire_flag"], fusion_val, test["hospital_expire_flag"], fusion_test, source))
    prediction_table["p_fusion"] = fusion_test
    save_calibration_plot(test["hospital_expire_flag"], fusion_test, "Fusion", metrics_dir / "calibration_fusion.png")

    results = pd.DataFrame(rows)
    write_results(
        results,
        metrics_dir / "results.csv",
        metrics_dir / "results.md",
        synthetic=bool(meta.get("synthetic")),
    )
    prediction_table.to_csv(metrics_dir / "predictions_test.csv", index=False)
    print(results.to_string(index=False))
    print(f"Wrote {metrics_dir / 'results.csv'}")
    print(f"Wrote {metrics_dir / 'results.md'}")


if __name__ == "__main__":
    main()
