"""
Step 5: Generate confusion matrix from full dataset diagnosis results.

Reads output/results/diagnosis_abnormal_full.json and diagnosis_normal_full.json,
then saves output/plots/confusion_matrix_test.png.

Usage:
    python 05_confusion_matrix.py
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

_BASE      = Path(__file__).resolve().parent.parent
RESULT_DIR = _BASE / "output" / "results"
PLOT_DIR   = _BASE / "output" / "plots"

CLASSES = ["Abnormal", "Normal"]


def load_results():
    y_true, y_pred = [], []

    abn_path = RESULT_DIR / "diagnosis_abnormal_full.json"
    nor_path  = RESULT_DIR / "diagnosis_normal_full.json"

    for path, true_label in [(abn_path, "Abnormal"), (nor_path, "Normal")]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data["results"]:
            y_true.append(CLASSES.index(true_label))
            y_pred.append(CLASSES.index(item["prediction"]))

    return y_true, y_pred


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, ax=ax, vmin=0,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    total   = len(y_true)
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    fig.suptitle(
        f"Full Dataset — {correct}/{total} correct  (acc={correct/total:.1%})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = load_results()

    print(classification_report(y_true, y_pred, target_names=CLASSES))
    save_confusion_matrix(y_true, y_pred, PLOT_DIR / "confusion_matrix_test.png")


if __name__ == "__main__":
    main()
