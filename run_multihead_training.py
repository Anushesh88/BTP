"""
Phase 7 CLI Runner:
Multi-Head Classifier Training & Evaluation on Dual-Branch Fused Features.
Compares performance against Phase 3 baseline toward the 85-90% accuracy benchmark.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch

from src.dataset import load_project_splits
from src.evaluate import evaluate_baseline_predictions, format_markdown_table, plot_confusion_matrix
from src.feature_fusion import DualBranchFusionPipeline
from src.multihead_classifier import MultiHeadTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_multihead_training")


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Multi-Head Classifier Training on Fused Features")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample", help="Dataset directory")
    parser.add_argument("--splits_dir", type=str, default="data/metadata", help="Folder with frozen train/val/test_split.jsonl (used when present)")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Cache directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Folder to save model weights")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Reports destination folder")
    parser.add_argument("--motifs_manifest", type=str, default="data/processed/motifs/detected_motifs_manifest.jsonl", help="Path to detected motifs manifest")
    parser.add_argument("--model_name", type=str, default="vit_b_16", help="Vision backbone tag")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    # Defaults selected on the validation split (mean macro F1 over raga/school/period)
    parser.add_argument("--hidden_dim", type=int, default=128, help="Trunk hidden width")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of fused features")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory '{args.data_dir}' does not exist.")
        return

    out_models_dir = Path(args.models_dir)
    out_models_dir.mkdir(parents=True, exist_ok=True)
    reports_path = Path(args.reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Manifest & Leak-Free Splits
    logger.info("=== STEP 1: Ingesting Manifest & Creating Leak-Free Grouped Splits ===")
    train_df, val_df, test_df = load_project_splits(
        data_dir=data_dir, splits_dir=args.splits_dir, test_size=args.test_size, val_size=args.val_size
    )

    # 2. Extract / Load Dual-Branch Fused Features for each split
    logger.info("=== STEP 2: Extracting Dual-Branch Fused Features for Train/Val/Test ===")
    fusion_pipeline = DualBranchFusionPipeline(backbone_model_name=args.model_name)

    clean_model_name = args.model_name.replace("/", "_")
    cache_train = cache_path / f"fused_train_{clean_model_name}.npz"
    cache_val = cache_path / f"fused_val_{clean_model_name}.npz"
    cache_test = cache_path / f"fused_test_{clean_model_name}.npz"

    X_train, _ = fusion_pipeline.extract_dataset_fused_features(
        df=train_df, motifs_manifest_path=args.motifs_manifest, cache_file=cache_train, force_recompute=args.force_recompute
    )
    X_val, _ = fusion_pipeline.extract_dataset_fused_features(
        df=val_df, motifs_manifest_path=args.motifs_manifest, cache_file=cache_val, force_recompute=args.force_recompute
    )
    X_test, _ = fusion_pipeline.extract_dataset_fused_features(
        df=test_df, motifs_manifest_path=args.motifs_manifest, cache_file=cache_test, force_recompute=args.force_recompute
    )

    fused_dim = X_train.shape[1]
    logger.info(f"Fused Feature Dimension: {fused_dim} (Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)})")

    # 3. Train Multi-Head Classifier with Class Weighting
    logger.info("=== STEP 3: Training Multi-Head Neural Classifier ===")
    trainer = MultiHeadTrainer(
        input_dim=fused_dim,
        targets=["raga", "school", "period", "artist"],
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = trainer.train_model(
        X_train=X_train,
        df_train=train_df,
        X_val=X_val,
        df_val=val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=12,
    )

    # Save trained checkpoint
    checkpoint_file = out_models_dir / "multihead_fused_classifier.pt"
    trainer.save_checkpoint(checkpoint_file)

    # 4. Evaluate on Test Split
    logger.info("=== STEP 4: Evaluating Multi-Head Classifier on Test Split ===")
    test_preds = trainer.predict(X_test)
    test_probs = trainer.predict_proba(X_test)

    eval_results = evaluate_baseline_predictions(
        predictions=test_preds,
        probabilities=test_probs,
        df_true=test_df,
        label_encoders=trainer.label_encoders,
        output_dir=reports_path,
        report_prefix="multihead",
        title="Phase 7 Multi-Head Fused Benchmark Report",
    )

    # 5. Load Phase 3 Baseline to Compare Improvement
    baseline_metrics = {}
    baseline_json = reports_path / "baseline_benchmark.json"
    if baseline_json.exists():
        with open(baseline_json, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)
    else:
        logger.warning(f"No Phase 3 baseline found at {baseline_json}; run run_baseline.py first for a comparison.")

    comparison_rows = []
    for t in trainer.targets:
        if t in eval_results:
            curr_acc = eval_results[t]["accuracy"] * 100
            curr_f1 = eval_results[t]["weighted_f1"]
            base_acc = baseline_metrics.get(t, {}).get("accuracy", 0.0) * 100
            base_f1 = baseline_metrics.get(t, {}).get("weighted_f1", 0.0)
            delta_acc = curr_acc - base_acc

            comparison_rows.append({
                "Target Head": t.capitalize(),
                "Phase 3 Baseline Acc": f"{base_acc:.1f}%",
                "Phase 7 Fused Acc": f"{curr_acc:.1f}%",
                "Accuracy Delta": f"{'+' if delta_acc >= 0 else ''}{delta_acc:.1f}%",
                "Phase 7 F1 (Weighted)": f"{curr_f1:.4f}",
            })

    # 6. Generate Markdown Report
    comp_df = pd.DataFrame(comparison_rows)
    table_md = format_markdown_table(comp_df)

    report_file = reports_path / "multihead_training_report.md"
    md_content = [
        "# Phase 7: Multi-Head Classifier Training & Evaluation Report",
        "",
        "> **Architecture**: Standardized Multi-Task Neural Trunk on 1632-dim Dual-Branch Fused Features (Deep Global + Region Crops + Handcrafted Descriptors) with Balanced Class Weighting.",
        "",
        "## Performance Comparison: Phase 3 Baseline vs. Phase 7 Multi-Head Fused",
        "",
        table_md,
        "",
        "## Progress Toward Target Goal (85% - 90%)",
        "- **Target Benchmark**: 85.00% – 90.00% across primary heads (*Raga*, *School*, *Period*).",
        "- **Domain Significance**: Fusing handcrafted color/texture/composition with deep region embeddings enhances fine-grained discrimination over simple frozen linear embeddings.",
        "- **Saved Model Checkpoint**: [`models/multihead_fused_classifier.pt`](../models/multihead_fused_classifier.pt)",
        "",
        "## Confusion Matrices (Fused Features)",
        "- Raga: [`multihead_confusion_matrix_raga.png`](multihead_confusion_matrix_raga.png)",
        "- School: [`multihead_confusion_matrix_school.png`](multihead_confusion_matrix_school.png)",
        "- Period: [`multihead_confusion_matrix_period.png`](multihead_confusion_matrix_period.png)",
        "- Artist: [`multihead_confusion_matrix_artist.png`](multihead_confusion_matrix_artist.png)",
    ]

    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("\n".join(md_content) + "\n")

    logger.info(f"Phase 7 Report generated at: {report_file}")


if __name__ == "__main__":
    main()
