"""
Phase 8 CLI Runner:
Post-Hoc Temperature Scaling on Validation Set & Uncertainty Calibration Evaluation.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.calibration import (
    MultiHeadTemperatureScaler,
    compute_calibration_metrics,
    plot_reliability_diagrams,
)
from src.dataset import load_project_splits
from src.evaluate import format_markdown_table
from src.feature_fusion import DualBranchFusionPipeline
from src.multihead_classifier import MultiHeadTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_calibration")


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Model Calibration & Uncertainty Estimation")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample", help="Dataset directory")
    parser.add_argument("--splits_dir", type=str, default="data/metadata", help="Folder with frozen train/val/test_split.jsonl (used when present)")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Cache directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Models folder")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Reports destination folder")
    parser.add_argument("--model_checkpoint", type=str, default="models/multihead_fused_classifier.pt", help="Path to trained Multi-Head model")
    parser.add_argument("--model_name", type=str, default="vit_b_16", help="Vision backbone name")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split ratio")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory '{args.data_dir}' does not exist.")
        return

    checkpoint_path = Path(args.model_checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Trained model checkpoint '{checkpoint_path}' does not exist. Please run Phase 7 first.")
        return

    out_models_dir = Path(args.models_dir)
    reports_path = Path(args.reports_dir)
    cache_path = Path(args.cache_dir)
    out_models_dir.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Manifest & Generate Identical Grouped Splits
    logger.info("=== STEP 1: Ingesting Manifest & Grouped Splits ===")
    _, val_df, test_df = load_project_splits(
        data_dir=data_dir, splits_dir=args.splits_dir, test_size=args.test_size, val_size=args.val_size
    )

    # 2. Load Trained Multi-Head Classifier
    logger.info(f"=== STEP 2: Loading Trained Multi-Head Model ({checkpoint_path}) ===")
    trainer = MultiHeadTrainer.load_checkpoint(checkpoint_path)
    model = trainer.model
    device = trainer.device

    # 3. Load Fused Features for Val and Test
    clean_model_name = args.model_name.replace("/", "_")
    cache_val = cache_path / f"fused_val_{clean_model_name}.npz"
    cache_test = cache_path / f"fused_test_{clean_model_name}.npz"

    fusion_pipeline = DualBranchFusionPipeline(backbone_model_name=args.model_name)
    X_val, _ = fusion_pipeline.extract_dataset_fused_features(df=val_df, cache_file=cache_val)
    X_test, _ = fusion_pipeline.extract_dataset_fused_features(df=test_df, cache_file=cache_test)

    # 4. Compute Raw Logits on Validation Split
    logger.info("=== STEP 3: Computing Validation Logits & Fitting Temperature Scalers ===")
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        val_logits_dict = model(X_val_t)

    # Temperatures are fitted only on val rows whose label is present and known to the head
    val_labels_dict = {}
    for target in trainer.targets:
        le = trainer.label_encoders[target]
        raw_val = val_df[target].astype(str).values
        mask = np.isin(raw_val, le.classes_) & val_df[target].notna().values
        y_enc = np.array([np.where(le.classes_ == v)[0][0] for v in raw_val[mask]], dtype=int)
        val_labels_dict[target] = torch.tensor(y_enc, dtype=torch.long).to(device)
        val_logits_dict[target] = val_logits_dict[target][torch.tensor(mask, device=device)]

    # Fit optimal temperature T per head on validation logits
    temp_scaler = MultiHeadTemperatureScaler(targets=trainer.targets)
    temperatures = temp_scaler.fit_validation_logits(
        val_logits_dict=val_logits_dict,
        val_labels_dict=val_labels_dict,
    )

    # Save temperatures
    temp_json_file = out_models_dir / "temperature_scalers.json"
    temp_scaler.save(temp_json_file)

    # 5. Evaluate Calibration on Held-Out Test Split
    logger.info("=== STEP 4: Evaluating Calibration (ECE, MCE, Brier Score) on Test Split ===")
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_logits_dict = model(X_test_t)

    # Scaled (calibrated) logits
    scaled_test_logits = temp_scaler.scale_logits(test_logits_dict)

    calib_summary_rows = []
    before_results = {}
    after_results = {}

    for target in trainer.targets:
        le = trainer.label_encoders[target]
        # Missing labels are excluded; classes unseen in training stay as -1 (always counted wrong)
        labelled = test_df[target].notna().values
        raw_test = test_df[target][labelled].astype(str).values
        y_test_enc = np.array([np.where(le.classes_ == v)[0][0] if v in le.classes_ else -1 for v in raw_test], dtype=int)

        # Probabilities Before & After
        p_uncalib = torch.softmax(test_logits_dict[target], dim=1).cpu().numpy()[labelled]
        p_calib = torch.softmax(scaled_test_logits[target], dim=1).cpu().numpy()[labelled]

        res_pre = compute_calibration_metrics(y_true=y_test_enc, y_prob=p_uncalib)
        res_post = compute_calibration_metrics(y_true=y_test_enc, y_prob=p_calib)

        before_results[target] = res_pre
        after_results[target] = res_post

        ece_pre = res_pre["ece"] * 100
        ece_post = res_post["ece"] * 100
        delta_ece = ece_post - ece_pre
        t_val = temperatures.get(target, 1.0)

        calib_summary_rows.append({
            "Target Head": target.capitalize(),
            "Optimal Temp (T)": f"{t_val:.3f}",
            "ECE Before": f"{ece_pre:.2f}%",
            "ECE After (Calibrated)": f"{ece_post:.2f}%",
            "ECE Improvement": f"{-delta_ece:.2f}%" if delta_ece <= 0 else f"+{delta_ece:.2f}%",
            "Brier Score Pre": f"{res_pre['brier_score']:.4f}",
            "Brier Score Post": f"{res_post['brier_score']:.4f}",
        })

    # 6. Render Reliability Diagrams
    rel_diagram_file = reports_path / "reliability_diagrams.png"
    plot_reliability_diagrams(
        calib_results_before=before_results,
        calib_results_after=after_results,
        output_path=rel_diagram_file,
        temperatures=temperatures,
    )

    # 7. Generate Calibration Markdown Report
    sum_df = pd.DataFrame(calib_summary_rows)
    table_md = format_markdown_table(sum_df)

    report_file = reports_path / "calibration_report.md"
    md_content = [
        "# Phase 8: Model Calibration & Uncertainty Estimation Report",
        "",
        "> **Objective**: Optimize post-hoc temperature scaling ($T > 0$) on the validation set using Negative Log-Likelihood (NLL) so that a confidence of '70%' empirically corresponds to ~70% true precision.",
        "",
        "## Calibration Performance Summary (Held-Out Test Split)",
        "",
        table_md,
        "",
        "## Domain & Methodological Insights",
        "- **Empirical Alignment**: Temperature scaling softens overconfident logits without changing top-1 class rankings (argmax is invariant).",
        "- **Validation Splitting**: Temperatures were strictly learned on the held-out validation set to eliminate data leakage.",
        f"- **Learned Scalers Saved**: [`models/temperature_scalers.json`](../models/temperature_scalers.json)",
        "",
        "## Reliability Diagrams (Pre vs. Post Scaling)",
        f"- Visual reliability curves and confidence calibration bins: [`reliability_diagrams.png`](reliability_diagrams.png)",
    ]

    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("\n".join(md_content) + "\n")

    # Save JSON benchmark
    calib_json = reports_path / "calibration_benchmark.json"
    with open(calib_json, "w", encoding="utf-8") as jf:
        json.dump({
            "temperatures": temperatures,
            "before_calibration": before_results,
            "after_calibration": after_results,
        }, jf, indent=2)

    logger.info(f"Phase 8 Calibration Report generated at: {report_file}")


if __name__ == "__main__":
    main()
