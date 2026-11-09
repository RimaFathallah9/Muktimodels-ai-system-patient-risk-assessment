"""Test metrics. Thresholds for F1 are chosen on validation, then frozen for test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def safe_auroc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_prob, dtype=float)))


def safe_auprc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_prob, dtype=float)))


def tune_f1_threshold(y_true, y_prob) -> float:
    """Pick the threshold with the highest F1. Call this on validation only."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = (y_prob >= threshold).astype(int)
        score = float(f1_score(y_true, predictions, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def metric_row(model_name: str, y_true, y_prob, threshold: float, data_source: str) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    predictions = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        f1 = float("nan")
    else:
        f1 = float(f1_score(y_true, predictions, zero_division=0))
    return {
        "model": model_name,
        "split": "test",
        "data_source": data_source,
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "f1": f1,
        "threshold": float(threshold),
    }


def align_values(frame: pd.DataFrame, hadm_ids, values, column_name: str) -> np.ndarray:
    """Put values back into the same row order as frame."""
    table = pd.DataFrame(
        {
            "hadm_id": np.asarray(hadm_ids, dtype=np.int64),
            column_name: np.asarray(values, dtype=float) if np.ndim(values) == 1 else list(values),
        }
    )
    if table["hadm_id"].duplicated().any():
        raise RuntimeError(f"Duplicate hadm_id while aligning {column_name}.")
    merged = frame[["hadm_id"]].merge(table, on="hadm_id", how="left")
    if len(merged) != len(frame) or merged[column_name].isna().any():
        raise RuntimeError(f"Could not align {column_name} to every admission.")
    if np.ndim(values) == 1:
        return merged[column_name].to_numpy(dtype=float)
    return np.stack(merged[column_name].to_numpy())


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append("" if pd.isna(value) else f"{float(value):.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def write_results(frame: pd.DataFrame, csv_path: Path, md_path: Path, synthetic: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    lines = ["# Test results", ""]
    if synthetic:
        lines.extend(
            [
                "> **Synthetic data.** These numbers only show that the pipeline runs.",
                "> They are not results and must not be reported as model performance.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Test split only. Validation was used to pick checkpoints and the F1 threshold.",
                "",
            ]
        )
    lines.extend(
        [
            "AUROC and AUPRC do not use the threshold. F1 uses a threshold chosen on the validation split.",
            "",
            markdown_table(frame),
            "",
            "Compare each single-modality row with the fused row. Report the numbers in this file, including when fusion is not the best row.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def save_calibration_plot(y_true, y_prob, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) < 8 or len(np.unique(y_true)) < 2:
        print(f"Calibration plot skipped for {title}: need both classes and at least 8 test rows.")
        return
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    axis.plot(predicted, observed, marker="o", label=title)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
