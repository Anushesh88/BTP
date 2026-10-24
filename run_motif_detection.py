"""
Phase 5 CLI Runner:
Open-Vocabulary Motif & Region Detection with Multi-Window Non-Maximum Suppression (NMS).
"""

import argparse
import re
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
from src.motif_detector import (
    CANONICAL_MOTIFS,
    DETECTION_PROMPTS,
    MultiWindowMotifPipeline,
    OpenVocabMotifDetector,
    evaluate_motif_recall,
)
from src.preprocessing import PaintingPreprocessor, SlidingWindowGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_motif_detection")

# Taxonomy color scheme for visual bounding box plotting
CATEGORY_COLORS = {
    "human": (255, 69, 0),         # Red-Orange
    "animal": (30, 144, 255),       # Dodger Blue
    "flora": (46, 139, 87),         # Sea Green
    "architecture": (148, 0, 211),  # Violet
    "instrument_object": (255, 215, 0), # Gold
    "text": (220, 20, 60),          # Crimson
    "object": (128, 128, 128),      # Grey
}


def draw_detection_overlay(
    img_rgb: np.ndarray,
    detections: List[dict],
    output_path: Path,
    image_id: str,
) -> None:
    """Renders visual bounding box overlay with category colors and confidence scores."""
    canvas = img_rgb.copy()
    h, w = canvas.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(canvas)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cat = det.get("category", "object")
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)

        # Plot rectangle
        color_rgb = [c / 255.0 for c in CATEGORY_COLORS.get(cat, (128, 128, 128))]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color_rgb, linewidth=2.5
        )
        ax.add_patch(rect)

        # Plot text badge
        caption_str = f"{label} ({conf*100:.0f}%)"
        ax.text(
            x1, max(12, y1 - 4), caption_str,
            fontsize=9, color="white", weight="bold",
            bbox=dict(facecolor=color_rgb, alpha=0.85, edgecolor="none", boxstyle="round,pad=0.2")
        )

    plt.title(f"Motif Detection Overlay: {image_id} ({len(detections)} motifs detected)", fontsize=13, weight="bold")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved motif detection overlay to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Open-Vocabulary Motif Detection & Region Extraction")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample", help="Directory of dataset images")
    parser.add_argument("--output_dir", type=str, default="data/processed/motifs", help="Folder to save motif crops")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Folder to save reports and overlay images")
    parser.add_argument("--model_name", type=str, default="yolov8s-worldv2.pt", help="YOLO-World weights tag")
    parser.add_argument("--use_mock", action="store_true", help="Use simulated detector for fast testing")
    parser.add_argument("--min_confidence", type=float, default=0.25, help="Minimum detection confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.40, help="Global NMS IoU threshold")
    parser.add_argument("--max_images", type=int, default=-1, help="Number of paintings to process (-1 = all)")
    parser.add_argument("--splits_dir", type=str, default="data/metadata", help="Folder with frozen split manifests (used when present)")
    parser.add_argument("--max_side", type=int, default=1280, help="Downscale canvas so its longest side is at most this many px before windowing")
    parser.add_argument("--window", type=int, default=640, help="Sliding window size (px)")
    parser.add_argument("--canonical_prompts", action="store_true", help="Use Sanskrit canonical motif names as prompts instead of plain-English prompts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Input directory '{args.data_dir}' does not exist.")
        return

    out_crops_dir = Path(args.output_dir)
    out_crops_dir.mkdir(parents=True, exist_ok=True)
    reports_path = Path(args.reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Manifest
    logger.info("=== STEP 1: Ingesting Manifest ===")
    manifest_df = load_all_paintings(data_dir=data_dir, splits_dir=args.splits_dir)
    if args.max_images > 0 and len(manifest_df) > args.max_images:
        manifest_df = manifest_df.head(args.max_images).copy()

    logger.info(f"Processing motif detection on {len(manifest_df)} paintings.")

    # 2. Initialize Preprocessor, Sliding Window Generator, and Open-Vocab Detector
    preprocessor = PaintingPreprocessor(denoise_method="median")
    window_gen = SlidingWindowGenerator(window_size=(args.window, args.window), stride=(args.window // 2, args.window // 2))
    prompts = CANONICAL_MOTIFS if args.canonical_prompts else DETECTION_PROMPTS

    detector = OpenVocabMotifDetector(
        model_name=args.model_name,
        custom_classes=prompts,
        conf_threshold=args.min_confidence,
        use_mock=args.use_mock,
    )

    pipeline = MultiWindowMotifPipeline(
        detector=detector,
        global_iou_threshold=args.iou_threshold,
        min_confidence=args.min_confidence,
    )

    all_detected_motifs = []
    summary_rows = []
    caption_hits, caption_mentions, per_prompt = 0, 0, {}

    # 3. Execute Detection & Multi-Window Aggregation
    logger.info("=== STEP 2: Running Multi-Window Detection & Global NMS ===")
    for idx, (_, row) in enumerate(manifest_df.iterrows()):
        img_id = row["image_id"]
        img_p = row["file_path"]

        # Enhance canvas & generate overlapping windows
        orig_img = preprocessor.load_and_convert_rgb(img_p)
        h0, w0 = orig_img.shape[:2]
        scale = min(1.0, args.max_side / max(h0, w0))
        if scale < 1.0:
            orig_img = cv2.resize(orig_img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        enhanced_img = preprocessor.process(orig_img)
        windows = window_gen.extract_windows(enhanced_img)

        # Detect across windows, project coordinates, apply Global NMS
        detections = pipeline.detect_painting_motifs(enhanced_img, windows)

        # Crop detected regions and save to disk
        crops_metadata = pipeline.crop_and_store_regions(
            full_image=enhanced_img,
            detections=detections,
            output_crop_dir=out_crops_dir,
            image_id=img_id,
        )

        # Record metadata
        for crop in crops_metadata:
            record = dict(crop)
            record["parent_image_id"] = img_id
            record["raga"] = row.get("raga", "Unknown")
            record["school"] = row.get("school", "Unknown")
            record["canvas_scale"] = scale
            all_detected_motifs.append(record)

        # Caption-grounded proxy recall: prompts mentioned in the caption vs prompts detected
        caption = str(row.get("caption", "")).lower()
        mentioned = {p for p in prompts if re.search(r"\b" + re.escape(p) + r"s?\b", caption)}
        found = {c["label"] for c in crops_metadata}
        caption_hits += len(mentioned & found)
        caption_mentions += len(mentioned)
        for p in mentioned:
            per_prompt.setdefault(p, [0, 0])
            per_prompt[p][1] += 1
            per_prompt[p][0] += int(p in found)

        summary_rows.append({
            "Painting ID": img_id,
            "Raga": row.get("raga", "Unknown"),
            "Motifs Count": len(crops_metadata),
            "Detected Labels": ", ".join([c["label"] for c in crops_metadata[:4]]) + ("..." if len(crops_metadata) > 4 else ""),
        })

        # Draw overlay for first painting
        if idx == 0:
            overlay_file = reports_path / f"motif_detection_{img_id}.png"
            draw_detection_overlay(
                img_rgb=enhanced_img,
                detections=crops_metadata,
                output_path=overlay_file,
                image_id=img_id,
            )

        if (idx + 1) % 25 == 0 or idx == 0 or idx + 1 == len(manifest_df):
            logger.info(f"[{idx+1}/{len(manifest_df)}] {img_id}: {len(crops_metadata)} motifs detected & cropped.")

    # 4. Save Master Detected Motifs Manifest
    manifest_jsonl_file = out_crops_dir / "detected_motifs_manifest.jsonl"
    with open(manifest_jsonl_file, "w", encoding="utf-8") as jf:
        for m in all_detected_motifs:
            jf.write(json.dumps(m) + "\n")
    logger.info(f"Saved {len(all_detected_motifs)} motif records to {manifest_jsonl_file}")

    # 5. Evaluate Recall & Fine-Tuning Decision
    # The dataset has no bounding-box annotations, so recall is measured against motifs
    # mentioned in each painting's caption (a weak, image-level proxy).
    rec = caption_hits / caption_mentions if caption_mentions else 0.0
    weak_prompts = sorted(
        [(p, h / n, n) for p, (h, n) in per_prompt.items() if n >= 5], key=lambda x: x[1]
    )
    decision = {
        "recommend_finetuning": rec < 0.6,
        "reason": (
            f"Caption-grounded recall {rec*100:.1f}% is below 60%; fine-tune on box labels once annotated."
            if rec < 0.6 else f"Caption-grounded recall {rec*100:.1f}% is adequate; zero-shot detection retained."
        ),
    }

    # 6. Generate Markdown Report
    report_file = reports_path / "motif_detection_report.md"
    sum_df = pd.DataFrame(summary_rows)

    headers = list(sum_df.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, r in sum_df.head(25).iterrows():
        table_lines.append("| " + " | ".join(str(val) for val in r.values) + " |")

    category_counts = pd.Series([m["category"] for m in all_detected_motifs]).value_counts().to_dict()

    md_content = [
        "# Phase 5: Open-Vocabulary Motif & Region Detection Report",
        "",
        "> **Objective**: Run zero-shot open-vocabulary detection across overlapping sliding windows, merge proposals with global Non-Maximum Suppression (NMS), classify into coarse taxonomies, crop motif patches, and evaluate fine-tuning necessity.",
        "",
        "## Detection Pipeline Configuration",
        f"- **Detector Architecture**: Open-Vocabulary YOLO-World ({args.model_name})",
        f"- **Vocabulary Size**: {len(prompts)} {'canonical art-historical' if args.canonical_prompts else 'plain-English'} prompts",
        f"- **Canvas**: longest side capped at {args.max_side}px; {args.window}px windows with 50% overlap",
        f"- **Paintings Processed**: {len(manifest_df)}; **Motif Crops**: {len(all_detected_motifs)}",
        f"- **Confidence Threshold**: {args.min_confidence:.2f}",
        f"- **Global NMS IoU Threshold**: {args.iou_threshold:.2f}",
        "",
        "## Category Distribution of Extracted Motifs",
        "",
    ]

    for cat, count in category_counts.items():
        md_content.append(f"- **{cat.capitalize()}**: {count} crops extracted")

    md_content.extend([
        "",
        f"## Summary by Painting (first 25 of {len(sum_df)})",
        "",
        "\n".join(table_lines),
        "",
        "## Caption-Grounded Recall & Fine-Tuning Recommendation",
        "> The dataset provides no bounding boxes, so recall is an image-level proxy: of the prompt objects named in a painting's caption, how many were detected anywhere in it.",
        f"- **Caption-grounded recall**: {rec*100:.1f}% ({caption_hits}/{caption_mentions} mentions)",
        f"- **Decision**: **{decision['reason']}**",
        "",
        "| Prompt | Recall | Captions mentioning |",
        "| --- | --- | --- |",
        *[f"| {p} | {r*100:.0f}% | {n} |" for p, r, n in weak_prompts],
        "",
        "## Inspection Graphic",
        f"- Visual bounding box overlay: [`motif_detection_{summary_rows[0]['Painting ID']}.png`](motif_detection_{summary_rows[0]['Painting ID']}.png)",
        f"- Master Motifs Manifest: [`{manifest_jsonl_file}`]({manifest_jsonl_file})",
    ])

    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("\n".join(md_content) + "\n")

    logger.info(f"Phase 5 Report generated at: {report_file}")


if __name__ == "__main__":
    main()
