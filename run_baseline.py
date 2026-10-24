"""
Phase 3 Baseline Runner:
Frozen ViT/CLIP Embeddings + Regularized Multi-Head Logistic Regression.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import create_grouped_splits, load_frozen_splits, parse_metadata_manifest
from src.feature_extractor import extract_embeddings_for_split
from src.baseline_classifier import BaselineMultiHeadClassifier
from src.evaluate import evaluate_baseline_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_baseline")


def generate_synthetic_miniature_dataset(target_dir: Path, num_samples: int = 60) -> Path:
    """
    Creates a synthetic sample dataset of miniature painting images with
    manuscript sets, captions, and metadata.jsonl for end-to-end pipeline verification.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating synthetic verification dataset at {target_dir} ({num_samples} images)...")

    series_list = ["Hamzanama_Album", "Boston_Ragamala", "Mewar_Series", "Akbarnama_Folio"]
    ragas = ["Bhairava", "Malkaus", "Hindol", "Deepak", "Megha", "Sri"]
    schools = ["Mughal", "Mewar", "Bundi", "Kangra", "Deccan"]
    periods = ["16th Century", "Early 17th Century", "Late 17th Century", "18th Century"]
    artists = ["Basawan", "Miskin", "Manohar", "Govardhan", "Unknown_Artist"]
    categories = ["Mughal Miniature", "Crop", "Ragamala"]

    jsonl_lines = []

    for i in range(num_samples):
        img_name = f"painting_{i:03d}.jpg"
        img_path = target_dir / img_name
        txt_path = target_dir / f"painting_{i:03d}.txt"

        # Create synthetic image with distinguishable synthetic visual patterns
        series = series_list[i % len(series_list)]
        raga = ragas[i % len(ragas)]
        school = schools[(i * 2) % len(schools)]
        period = periods[(i * 3) % len(periods)]
        artist = artists[i % len(artists)]
        cat = categories[i % len(categories)]

        # Unique synthetic RGB pattern based on properties
        r_val = (i * 35) % 256
        g_val = (i * 55) % 256
        b_val = (i * 75) % 256
        img = Image.new("RGB", (256, 256), color=(r_val, g_val, b_val))
        img.save(img_path)

        caption_text = (
            f"A {cat.lower()} painting in the style of the {school} school from the {period}. "
            f"Depicting {raga} raga imagery from the {series} collection, attributed to {artist}."
        )

        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(caption_text)

        metadata_dict = {
            "file_name": img_name,
            "category": cat,
            "raga": raga,
            "school": school,
            "period": period,
            "artist": artist,
            "collection_id": series,
            "caption": caption_text,
        }
        jsonl_lines.append(json.dumps(metadata_dict) if "json" in sys.modules else str(metadata_dict).replace("'", '"'))

    # Save metadata.jsonl
    jsonl_path = target_dir / "metadata.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        jf.write("\n".join(jsonl_lines) + "\n")

    # Save sources.txt
    sources_path = target_dir / "sources.txt"
    with open(sources_path, "w", encoding="utf-8") as sf:
        sf.write(
            "1. Hamzanama_Album: Victoria and Albert Museum / Metropolitan Museum of Art\n"
            "2. Boston_Ragamala: Museum of Fine Arts Boston\n"
            "3. Mewar_Series: National Museum New Delhi\n"
            "4. Akbarnama_Folio: British Library London\n"
        )

    logger.info(f"Synthetic verification dataset ready with metadata.jsonl and sources.txt.")
    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Frozen ViT/CLIP Baseline Classifier Runner")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample", help="Directory containing dataset files")
    parser.add_argument("--splits_dir", type=str, default="data/metadata", help="Folder with frozen train/val/test_split.jsonl (used when present)")
    parser.add_argument("--metadata_path", type=str, default=None, help="Explicit path to metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="vit_b_16", help="Vision backbone: vit_b_16, vit_b_32, or timm model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Path to cache extracted embeddings")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained baseline models")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--create_synthetic_test", action="store_true", help="Generate synthetic dataset for verification")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of embeddings")
    args = parser.parse_args()

    data_dir_path = Path(args.data_dir)
    use_frozen = not args.create_synthetic_test and Path(args.splits_dir, "train_split.jsonl").exists()

    if use_frozen:
        # 1-2. Load Phase 2 frozen leak-free splits (real dataset)
        logger.info(f"=== STEP 1-2: Loading Frozen Leak-Free Splits ({args.splits_dir}) ===")
        train_df, val_df, test_df = load_frozen_splits(args.splits_dir)
    else:
        # If synthetic test requested or data directory empty, generate verification sample
        if args.create_synthetic_test or not data_dir_path.exists() or not any(data_dir_path.glob("*.jpg")):
            logger.info(f"Dataset not detected in '{args.data_dir}'. Initializing synthetic verification dataset...")
            data_dir_path = generate_synthetic_miniature_dataset(Path("data/synthetic_sample"), num_samples=60)
            args.data_dir = str(data_dir_path)

        # 1. Parse Manifest
        logger.info("=== STEP 1: Ingesting Dataset Manifest ===")
        manifest_df = parse_metadata_manifest(
            data_dir=args.data_dir,
            metadata_jsonl_path=args.metadata_path,
        )

        # 2. Leak-Free Grouped Splits
        logger.info("=== STEP 2: Generating Grouped Splits (By Manuscript / Collection) ===")
        train_df, val_df, test_df = create_grouped_splits(
            df=manifest_df,
            group_col="collection_id",
            test_size=args.test_size,
            val_size=args.val_size,
        )

    # 3. Extract Frozen Vision Embeddings
    logger.info(f"=== STEP 3: Extracting Frozen Embeddings ({args.model_name}) ===")
    X_train, _ = extract_embeddings_for_split(
        df=train_df,
        split_name="train",
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )
    X_val, _ = extract_embeddings_for_split(
        df=val_df,
        split_name="val",
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )
    X_test, _ = extract_embeddings_for_split(
        df=test_df,
        split_name="test",
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )

    # 4. Train Multi-Head Classifiers
    logger.info("=== STEP 4: Training Baseline Multi-Head Classifiers ===")
    classifier = BaselineMultiHeadClassifier(
        targets=["raga", "school", "period", "artist"],
        class_weight="balanced",
    )
    classifier.fit(X_train=X_train, df_train=train_df)

    # Save trained baseline bundle
    model_save_path = Path(args.models_dir) / "baseline_classifiers.joblib"
    classifier.save(model_save_path)

    # 5. Evaluate Baseline on Test Set
    logger.info("=== STEP 5: Evaluating Baseline on Test Split ===")
    test_preds = classifier.predict(X_test)
    test_probs = classifier.predict_proba(X_test)

    eval_results = evaluate_baseline_predictions(
        predictions=test_preds,
        probabilities=test_probs,
        df_true=test_df,
        label_encoders=classifier.label_encoders,
        output_dir=args.output_dir,
    )

    logger.info("=== BASELINE EVALUATION COMPLETE ===")
    logger.info(f"Reports and confusion matrices saved to: {Path(args.output_dir).resolve()}")
    logger.info(f"Baseline benchmark report: {Path(args.output_dir).resolve() / 'baseline_benchmark.md'}")


if __name__ == "__main__":
    main()
