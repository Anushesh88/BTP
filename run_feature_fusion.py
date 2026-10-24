"""
Phase 6 CLI Runner:
Dual-Branch Feature Engineering & Fusion (Handcrafted + Deep Global & Region Crops).
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import load_all_paintings
from src.feature_fusion import DualBranchFusionPipeline, HandcraftedFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_feature_fusion")


def plot_feature_profile(
    fused_matrix: np.ndarray,
    output_path: Path,
    deep_dim: int,
    handcrafted_dim: int,
) -> None:
    """Generates an inspection plot of feature activation energy across branches."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_profile = np.mean(np.abs(fused_matrix), axis=0)
    std_profile = np.std(fused_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    x_axis = np.arange(len(mean_profile))

    # Plot branches with distinct colors
    g_end = deep_dim
    r_end = deep_dim * 2

    ax.plot(x_axis[:g_end], mean_profile[:g_end], label="Branch 1: Deep Global ViT", color="#1f77b4", lw=1.5)
    ax.plot(x_axis[g_end:r_end], mean_profile[g_end:r_end], label="Branch 2: Deep Region Aggregated", color="#ff7f0e", lw=1.5)
    ax.plot(x_axis[r_end:], mean_profile[r_end:], label="Branch 3: Handcrafted (Color/Texture/Stroke/Comp)", color="#2ca02c", lw=1.5)

    ax.axvline(x=g_end, color="grey", linestyle="--", alpha=0.7)
    ax.axvline(x=r_end, color="grey", linestyle="--", alpha=0.7)

    ax.set_title("Dual-Branch Fused Feature Vector Energy Profile", fontsize=13, weight="bold")
    ax.set_xlabel("Fused Feature Dimension Index")
    ax.set_ylabel("Mean Absolute Activation")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature profile graphic to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Dual-Branch Feature Engineering & Fusion")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample", help="Directory of dataset images")
    parser.add_argument("--motifs_manifest", type=str, default="data/processed/motifs/detected_motifs_manifest.jsonl", help="Path to detected motifs manifest")
    parser.add_argument("--output_dir", type=str, default="data/cache", help="Folder to cache fused feature vectors")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Folder to save reports and figures")
    parser.add_argument("--model_name", type=str, default="vit_b_16", help="Vision backbone: vit_b_16, vit_b_32, etc.")
    parser.add_argument("--splits_dir", type=str, default="data/metadata", help="Folder with frozen split manifests (used when present)")
    parser.add_argument("--max_images", type=int, default=10, help="Number of paintings to extract")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of feature vectors")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Input directory '{args.data_dir}' does not exist.")
        return

    out_cache_dir = Path(args.output_dir)
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    reports_path = Path(args.reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Manifest
    logger.info("=== STEP 1: Ingesting Manifest ===")
    manifest_df = load_all_paintings(data_dir=data_dir, splits_dir=args.splits_dir)
    if len(manifest_df) > args.max_images:
        manifest_df = manifest_df.head(args.max_images).copy()

    logger.info(f"Extracting features for {len(manifest_df)} paintings.")

    # 2. Initialize Dual-Branch Fusion Pipeline
    pipeline = DualBranchFusionPipeline(backbone_model_name=args.model_name)

    cache_file = out_cache_dir / f"fused_profile_{args.model_name}.npz"

    # 3. Extract Fused Features
    logger.info("=== STEP 2: Extracting Handcrafted & Deep Fused Features ===")
    fused_matrix, _ = pipeline.extract_dataset_fused_features(
        df=manifest_df,
        motifs_manifest_path=args.motifs_manifest,
        cache_file=cache_file,
        force_recompute=args.force_recompute,
    )

    deep_dim = pipeline.backbone.feature_dim
    hand_dim = pipeline.handcrafted.feature_dim
    total_dim = fused_matrix.shape[1]

    # 4. Generate Visual Inspection Plot
    plot_file = reports_path / "feature_distributions.png"
    plot_feature_profile(
        fused_matrix=fused_matrix,
        output_path=plot_file,
        deep_dim=deep_dim,
        handcrafted_dim=hand_dim,
    )

    # 5. Generate Markdown Report
    report_file = reports_path / "feature_fusion_report.md"

    md_content = [
        "# Phase 6: Dual-Branch Feature Engineering & Fusion Report",
        "",
        "> **Objective**: Extract domain-tailored handcrafted descriptors (Color, Texture, Stroke, Composition) and fuse them with Deep Global and Region Crop embeddings for downstream multi-head classification.",
        "",
        "## Feature Architecture Breakdown",
        f"- **Branch 1 (Deep Global Canvas)**: {deep_dim} dimensions (from frozen {args.model_name})",
        f"- **Branch 2 (Deep Region Aggregation)**: {deep_dim} dimensions (Mean-pooled over detected motif crops)",
        f"- **Branch 3 (Handcrafted Descriptors)**: {hand_dim} dimensions:",
        "  * *Color (HSV & Lab)*: Hue/Saturation/Value histograms, Lab moments, Gold leaf / Mineral red fractions",
        "  * *Texture*: GLCM (contrast, dissimilarity, homogeneity, energy, correlation), Uniform LBP, Gabor filter bank",
        "  * *Stroke & Line*: Canny edge density, gradient orientation distribution, contour stroke lengths",
        "  * *Composition*: 3x3 Rule-of-thirds grid energy, horizontal/vertical symmetry, center-of-mass focal point",
        f"- **Total Unified Fused Dimension**: **{total_dim} dimensions**",
        "",
        "## Feature Matrix Statistics",
        f"- **Paintings Processed**: {fused_matrix.shape[0]}",
        f"- **Matrix Shape**: `{fused_matrix.shape}`",
        f"- **Cache Location**: [`{cache_file}`]({cache_file})",
        "",
        "## Feature Activation Profile",
        f"- Visual distribution profile across branches: [`feature_distributions.png`](feature_distributions.png)",
    ]

    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("\n".join(md_content) + "\n")

    logger.info(f"Phase 6 Report generated at: {report_file}")


if __name__ == "__main__":
    main()
