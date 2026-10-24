"""
Evaluation module for baseline and advanced models.
Calculates Accuracy, Macro/Weighted F1, Per-Class F1, Confusion Matrices,
and Expected Calibration Error (ECE), outputting Markdown and JSON benchmark summaries.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE) for multi-class predictions.
    ECE measures the difference between predicted confidence and empirical accuracy.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_details.append({
                "bin_range": f"{bin_lower:.2f}-{bin_upper:.2f}",
                "count": int(np.sum(in_bin)),
                "avg_confidence": float(avg_confidence_in_bin),
                "accuracy": float(accuracy_in_bin),
            })

    return float(ece), {"bins": bin_details}


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix using matplotlib."""
    if normalize:
        with np.errstate(all='ignore'):
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
        cm_to_plot = cm_norm
    else:
        cm_to_plot = cm

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.8), max(6, len(classes) * 0.6)))
    im = ax.imshow(cm_to_plot, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm_to_plot.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm_to_plot[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_to_plot[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {output_file}")


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without third-party dependencies."""
    headers = [str(col) for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        data_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    return "\n".join([header_line, sep_line] + data_lines)


def evaluate_baseline_predictions(
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    df_true: pd.DataFrame,
    label_encoders: Dict,
    output_dir: Union[str, Path] = "reports",
    report_prefix: str = "baseline",
    title: str = "Phase 3 Baseline Benchmark Report",
) -> Dict:
    """
    Comprehensive evaluation of all multi-head predictions against ground truth labels.
    Generates JSON and Markdown benchmark reports with confusion matrices and ECE.
    Rows whose ground-truth label is missing (NaN) are excluded per target.
    Outputs are written as `{report_prefix}_benchmark.*` and
    `confusion_matrix_{target}.png` (baseline) or `{report_prefix}_confusion_matrix_{target}.png`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summary_rows = []

    for target, y_pred in predictions.items():
        if target not in df_true.columns:
            continue

        labelled = df_true[target].notna().values
        if not labelled.any():
            continue
        y_true = df_true[target][labelled].astype(str).values
        y_pred = np.asarray(y_pred)[labelled]

        # Accuracy, Macro F1, Weighted F1
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class classification report
        cls_report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        # Union of classes across train (encoder) and test ground truth
        le = label_encoders[target]
        all_classes = sorted(list(set(le.classes_).union(set(y_true)).union(set(y_pred))))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)

        # Plot Confusion Matrix
        cm_name = f"confusion_matrix_{target}.png" if report_prefix == "baseline" else f"{report_prefix}_confusion_matrix_{target}.png"
        cm_plot_file = out_dir / cm_name
        plot_confusion_matrix(
            cm,
            classes=all_classes,
            output_path=cm_plot_file,
            title=f"{report_prefix.capitalize()} Confusion Matrix - {target.capitalize()}",
            normalize=True,
        )

        # Expected Calibration Error (computed on classes known to the classifier head)
        ece_val = None
        if target in probabilities:
            y_probs = probabilities[target][labelled]
            y_true_enc = np.array([
                np.where(le.classes_ == val)[0][0] if val in le.classes_ else -1
                for val in y_true
            ])
            valid_mask = y_true_enc >= 0
            if np.any(valid_mask):
                ece_val, _ = compute_expected_calibration_error(y_true_enc[valid_mask], y_probs[valid_mask])

        results[target] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "expected_calibration_error": ece_val,
            "classes": all_classes,
            "classification_report": cls_report_dict,
            "confusion_matrix_plot": str(cm_plot_file),
        }

        summary_rows.append({
            "Target Head": target.capitalize(),
            "Accuracy": f"{acc * 100:.2f}%",
            "Macro F1": f"{macro_f1:.4f}",
            "Weighted F1": f"{weighted_f1:.4f}",
            "ECE": f"{ece_val:.4f}" if ece_val is not None else "N/A",
            "Num Classes": len(all_classes),
            "Support": len(y_true),
        })

    # Save JSON benchmark file
    json_path = out_dir / f"{report_prefix}_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark metrics to {json_path}")

    # Generate Markdown Benchmark Report
    md_path = out_dir / f"{report_prefix}_benchmark.md"
    summary_df = pd.DataFrame(summary_rows)
    table_markdown = format_markdown_table(summary_df)

    md_content = [
        f"# {title}",
        "",
        "> **Architecture**: Frozen ViT/CLIP visual embeddings + Regularized Logistic Regression.",
        "> **Purpose**: Benchmark floor that all subsequent multi-modal and fine-tuned models must beat.",
        "",
        "## Summary Metrics",
        "",
        table_markdown,
        "",
        "## Comparison with Target Goal (85% - 90%)",
        "",
        "- **Target Performance**: 85.00% – 90.00% across critical heads (Raga, School, Period).",
        "- **Baseline Assessment**: Baseline provides the uncalibrated linear boundary on pre-trained visual representations.",
        "- **Confusion Note**: Raga classes exhibit visual overlap; confusion matrices have been generated for error diagnosis.",
        "",
        "## Confusion Matrix Artifacts",
        "",
    ]

    for row in summary_rows:
        tname = Path(results[row["Target Head"].lower()]["confusion_matrix_plot"]).name
        md_content.append(f"- **{row['Target Head']}**: [`{tname}`]({tname})")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")
    logger.info(f"Saved benchmark markdown report to {md_path}")

    return results
