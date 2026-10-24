"""
Phase 4 CLI Runner:
High-Resolution Preprocessing & Sliding Window Generation for Miniature Paintings.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import load_all_paintings
from src.preprocessing import PaintingPreprocessor, SlidingWindowGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_preprocessing")


def generate_preprocessing_inspection_plot(
    orig_img: np.ndarray,
    proc_img: np.ndarray,
    windows: List,
    output_path: Path,
    image_id: str,
) -> None:
    """
    Renders an inspection graphic:
    1. Original Canvas vs Preprocessed Canvas
    2. High-magnification zoom-in demonstrating unsharp mask edge enhancement
    3. Canvas with sliding window bounding grid overlay
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = orig_img.shape[:2]

    # Select central zoom region for detail comparison
    cy, cx = h // 2, w // 2
    crop_size = min(128, h // 4, w // 4)
    y1, y2 = max(0, cy - crop_size), min(h, cy + crop_size)
    x1, x2 = max(0, cx - crop_size), min(w, cx + crop_size)

    orig_zoom = orig_img[y1:y2, x1:x2]
    proc_zoom = proc_img[y1:y2, x1:x2]

    # Create sliding window grid overlay
    grid_img = proc_img.copy()
    for _, meta in windows:
        gx1, gy1, gx2, gy2 = meta["global_box"]
        # Draw translucent window bounding box
        cv2.rectangle(grid_img, (gx1, gy1), (gx2, gy2), (255, 140, 0), 2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Full Canvas Comparison + Window Grid
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title(f"Original Canvas ({w}x{h})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(proc_img)
    axes[0, 1].set_title("Preprocessed (Denoise + CLAHE + Unsharp)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(grid_img)
    axes[0, 2].set_title(f"Sliding Window Grid ({len(windows)} windows)")
    axes[0, 2].axis("off")

    # Row 2: Micro-Detail Zoom-in Comparison + Window Sample
    axes[1, 0].imshow(orig_zoom)
    axes[1, 0].set_title(f"Original Detail Zoom ({x2-x1}x{y2-y1})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(proc_zoom)
    axes[1, 1].set_title("Enhanced Detail Zoom (Unsharp Mask)")
    axes[1, 1].axis("off")

    # Sample single window crop
    sample_win_img, sample_meta = windows[0]
    axes[1, 2].imshow(sample_win_img)
    axes[1, 2].set_title(f"Sample Window Crop (Win ID {sample_meta['window_id']})")
    axes[1, 2].axis("off")

    plt.suptitle(f"Phase 4 Preprocessing & Sliding Window Inspection: {image_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved preprocessing inspection graphic to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: High-Res Preprocessing & Sliding Window Extractor")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample", help="Directory of dataset images")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Destination folder for preprocessed data")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Destination folder for inspection reports")
    parser.add_argument("--window_size", type=int, default=512, help="Sliding window dimension (square W=H)")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride (50% overlap default)")
    parser.add_argument("--denoise", type=str, default="median", choices=["nlmeans", "median", "none"], help="Denoising method")
    parser.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE contrast clip limit")
    parser.add_argument("--gamma", type=float, default=1.05, help="Gamma lighting correction factor")
    parser.add_argument("--unsharp_strength", type=float, default=1.2, help="Unsharp masking edge enhancement strength")
    parser.add_argument("--splits_dir", type=str, default="data/metadata", help="Folder with frozen split manifests (used when present)")
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of paintings to process in this run")
    parser.add_argument("--save_window_images", action="store_true", help="Save individual window image files to disk")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Input directory '{args.data_dir}' does not exist.")
        return

    out_base = Path(args.output_dir)
    proc_img_dir = out_base / "preprocessed_images"
    windows_dir = out_base / "windows"
    proc_img_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    reports_path = Path(args.reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Manifest
    logger.info("=== STEP 1: Ingesting Painting Manifest ===")
    manifest_df = load_all_paintings(data_dir=data_dir, splits_dir=args.splits_dir)
    if len(manifest_df) > args.max_images:
        manifest_df = manifest_df.head(args.max_images).copy()

    logger.info(f"Loaded {len(manifest_df)} paintings for Phase 4 processing.")

    # 2. Initialize Preprocessor & Window Generator
    preprocessor = PaintingPreprocessor(
        denoise_method=args.denoise,
        apply_white_balance=True,
        apply_clahe=True,
        clahe_clip_limit=args.clahe_clip,
        gamma=args.gamma,
        apply_unsharp_mask=True,
        unsharp_strength=args.unsharp_strength,
    )

    window_gen = SlidingWindowGenerator(
        window_size=(args.window_size, args.window_size),
        stride=(args.stride, args.stride),
        padding_mode="reflect",
    )

    # 3. Process paintings and generate windows
    logger.info("=== STEP 2: Executing High-Res Preprocessing & Windowing ===")
    all_windows_manifest = []
    processed_summary = []

    for idx, (_, row) in enumerate(manifest_df.iterrows()):
        img_id = row["image_id"]
        raw_path = row["file_path"]
        orig_img = preprocessor.load_and_convert_rgb(raw_path)
        h, w = orig_img.shape[:2]

        # Enhance image
        proc_img = preprocessor.process(orig_img)

        # Save enhanced full canvas
        proc_path = proc_img_dir / f"{img_id}_enhanced.jpg"
        Image.fromarray(proc_img).save(proc_path, quality=95)

        # Extract sliding windows
        windows = window_gen.extract_windows(proc_img)

        # Record metadata
        for win_img, win_meta in windows:
            win_id = win_meta["window_id"]
            win_rel_path = ""
            if args.save_window_images:
                win_file = windows_dir / f"{img_id}_win_{win_id:04d}.jpg"
                Image.fromarray(win_img).save(win_file, quality=92)
                win_rel_path = str(win_file)

            record = {
                "parent_image_id": img_id,
                "window_id": win_id,
                "window_file": win_rel_path,
                "global_box": win_meta["global_box"],
                "window_x": win_meta["x"],
                "window_y": win_meta["y"],
                "window_w": win_meta["width"],
                "window_h": win_meta["height"],
                "orig_dims": win_meta["orig_dims"],
                "is_padded": win_meta["is_padded"],
                "raga": row.get("raga", "Unknown"),
                "school": row.get("school", "Unknown"),
                "period": row.get("period", "Unknown"),
            }
            all_windows_manifest.append(record)

        processed_summary.append({
            "image_id": img_id,
            "orig_resolution": f"{w}x{h}",
            "num_windows": len(windows),
            "enhanced_path": str(proc_path),
        })

        # Generate inspection visualization for first image
        if idx == 0:
            insp_plot = reports_path / f"preprocessing_inspection_{img_id}.png"
            generate_preprocessing_inspection_plot(
                orig_img=orig_img,
                proc_img=proc_img,
                windows=windows,
                output_path=insp_plot,
                image_id=img_id,
            )

        logger.info(f"[{idx+1}/{len(manifest_df)}] Processed {img_id}: {w}x{h} -> {len(windows)} windows.")

    # 4. Save Window Manifest JSONL
    manifest_file = out_base / "windows_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as jf:
        for rec in all_windows_manifest:
            jf.write(json.dumps(rec) + "\n")

    logger.info(f"Saved total of {len(all_windows_manifest)} window records to {manifest_file}")

    # 5. Generate Markdown Report
    md_report = reports_path / "preprocessing_report.md"
    sum_df = pd.DataFrame(processed_summary)

    # Format Markdown Table
    headers = ["Painting ID", "Original Canvas", "Extracted Windows", "Enhanced Image Path"]
    sep = ["---"] * len(headers)
    table_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for s in processed_summary:
        table_lines.append(f"| {s['image_id']} | {s['orig_resolution']} | {s['num_windows']} | `{s['enhanced_path']}` |")

    md_content = [
        "# Phase 4: Preprocessing & Sliding Window Generation Report",
        "",
        "> **Objective**: Preserve native resolution, apply edge-preserving denoising, color normalization, CLAHE contrast enhancement, unsharp masking, and generate overlapping sliding windows with exact coordinate tracking.",
        "",
        "## Configuration Parameters",
        f"- **Window Size**: {args.window_size} x {args.window_size} pixels",
        f"- **Stride**: {args.stride} pixels (Overlap: {((args.window_size - args.stride) / args.window_size) * 100:.1f}%)",
        f"- **Denoising Filter**: {args.denoise}",
        f"- **Lighting & Contrast**: CLAHE (clip={args.clahe_clip}, grid=8x8) + Gamma ({args.gamma})",
        f"- **Sharpening**: Unsharp Mask (strength={args.unsharp_strength})",
        f"- **Border Handling**: Reflective padding for marginalia and borders",
        "",
        "## Processed Paintings Summary",
        "",
        "\n".join(table_lines),
        "",
        f"- **Total Sliding Windows Generated**: {len(all_windows_manifest)}",
        f"- **Windows Manifest Location**: [`{manifest_file}`]({manifest_file})",
        "",
        "## Inspection Graphic",
        f"- See visual inspection comparison: [`preprocessing_inspection_{processed_summary[0]['image_id']}.png`](preprocessing_inspection_{processed_summary[0]['image_id']}.png)",
    ]

    with open(md_report, "w", encoding="utf-8") as rf:
        rf.write("\n".join(md_content) + "\n")

    logger.info(f"Phase 4 Report generated at: {md_report}")


if __name__ == "__main__":
    main()
