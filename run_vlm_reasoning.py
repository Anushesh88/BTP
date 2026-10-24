"""
Multimodal VLM Reasoning Pipeline Runner:
Grounded Art-Historical Retrieval + Qwen VLM Reasoning + Cross-Check + Expert Review.

Evaluates on the held-out test split. Classifier candidates come from the calibrated
Phase 7 multi-head model (falling back to the Phase 3 baseline), motif signals come from
the Phase 5 detector manifest, and identifying metadata (file name, collection) is withheld
from the VLM so it cannot read the answer off the file name.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from knowledge_base.retriever import ArtKnowledgeRetriever
from src.baseline_classifier import BaselineMultiHeadClassifier
from src.calibration import MultiHeadTemperatureScaler
from src.cross_check import ExpertReviewQueue
from src.dataset import load_project_splits
from src.feature_extractor import extract_embeddings_for_split
from src.feature_fusion import DualBranchFusionPipeline
from src.multihead_classifier import MultiHeadTrainer
from src.test_reasoning import evaluate_reasoning_dataset
from src.vlm_client import MockQwenVLMClient, QwenOllamaVLMClient
from src.vlm_judge import VLMJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_vlm_reasoning")

TARGETS = ["raga", "school", "period", "artist"]


def top3_from_probs(probs: np.ndarray, classes) -> list:
    idx = np.argsort(probs)[::-1][:3]
    return [(str(classes[i]), float(probs[i])) for i in idx]


def classifier_candidates(args, eval_df: pd.DataFrame):
    """Return ({image_id: {target: top3}}, label_options, source description)."""
    ckpt = Path(args.multihead_checkpoint)
    temps_file = Path(args.temperatures_path)

    if ckpt.exists():
        trainer = MultiHeadTrainer.load_checkpoint(ckpt)
        fusion = DualBranchFusionPipeline(backbone_model_name="vit_b_16")
        X, _ = fusion.extract_dataset_fused_features(
            df=eval_df,
            motifs_manifest_path=args.motifs_manifest,
            cache_file=Path(args.cache_dir) / "fused_test_vit_b_16.npz",
        )
        with torch.no_grad():
            logits = trainer.model(torch.tensor(X, dtype=torch.float32).to(trainer.device))
        if temps_file.exists():
            probs = MultiHeadTemperatureScaler.load(temps_file).scale_probabilities(logits)
            source = "Phase 7 multi-head fused classifier (temperature-calibrated)"
        else:
            probs = {t: torch.softmax(l, dim=1).cpu().numpy() for t, l in logits.items()}
            source = "Phase 7 multi-head fused classifier (uncalibrated)"
        encoders = trainer.label_encoders
    else:
        clf = BaselineMultiHeadClassifier.load(args.classifiers_path)
        X, _ = extract_embeddings_for_split(df=eval_df, split_name="test", model_name="vit_b_16")
        probs = clf.predict_proba(X)
        encoders = clf.label_encoders
        source = "Phase 3 frozen ViT baseline"

    by_image = {}
    for i, img_id in enumerate(eval_df["image_id"].values):
        by_image[img_id] = {t: top3_from_probs(probs[t][i], encoders[t].classes_) for t in TARGETS if t in probs}
    label_options = {t: [str(c) for c in encoders[t].classes_] for t in TARGETS if t in encoders}
    return by_image, label_options, source


def load_motifs(manifest_path: str) -> dict:
    """parent_image_id -> list of detections sorted by confidence."""
    motifs = {}
    p = Path(manifest_path)
    if not p.exists():
        logger.warning(f"Motif manifest not found at {p}; VLM will run without detector signals.")
        return motifs
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                motifs.setdefault(rec["parent_image_id"], []).append(rec)
    for recs in motifs.values():
        recs.sort(key=lambda r: -r.get("confidence", 0.0))
    return motifs


def main():
    parser = argparse.ArgumentParser(description="Multimodal VLM Judge & Reasoning Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample", help="Fallback dataset directory when no frozen splits exist")
    parser.add_argument("--splits_dir", type=str, default="data/metadata", help="Folder with frozen split manifests")
    parser.add_argument("--model_type", type=str, choices=["qwen_ollama", "mock"], default="qwen_ollama", help="VLM backend")
    parser.add_argument("--model_name", type=str, default="qwen2.5vl:7b", help="Qwen vision model tag on Ollama")
    parser.add_argument("--image_max_size", type=int, default=672, help="Longest image side sent to the VLM (px)")
    parser.add_argument("--num_ctx", type=int, default=8192, help="Ollama context window (tokens)")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds to wait for each VLM response")
    parser.add_argument("--hide_classifier", action="store_true",
                        help="Do not show classifier candidates to the VLM (measures its independent visual judgment); "
                             "knowledge is retrieved for every allowed label instead, and cross-check still uses the classifier")
    parser.add_argument("--image_ids_file", type=str, default=None, help="Text file of test image_ids to evaluate (one per line)")
    parser.add_argument("--num_eval_samples", type=int, default=-1, help="Number of held-out test paintings to evaluate (-1 = all)")
    parser.add_argument("--confidence_threshold", type=float, default=0.70, help="Confidence threshold before routing to review")
    parser.add_argument("--output_dir", type=str, default="reports", help="Reports destination folder")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Feature cache folder")
    parser.add_argument("--motifs_manifest", type=str, default="data/processed/motifs/detected_motifs_manifest.jsonl", help="Phase 5 motif manifest")
    parser.add_argument("--multihead_checkpoint", type=str, default="models/multihead_fused_classifier.pt", help="Phase 7 model")
    parser.add_argument("--temperatures_path", type=str, default="models/temperature_scalers.json", help="Phase 8 temperatures")
    parser.add_argument("--classifiers_path", type=str, default="models/baseline_classifiers.joblib", help="Phase 3 baseline (fallback)")
    args = parser.parse_args()

    # 1. Held-out evaluation set
    logger.info("=== STEP 1: Loading Held-Out Test Split ===")
    _, _, test_df = load_project_splits(data_dir=args.data_dir, splits_dir=args.splits_dir)
    eval_df = test_df
    if args.image_ids_file:
        wanted = [l.strip() for l in open(args.image_ids_file, encoding="utf-8") if l.strip()]
        eval_df = test_df.set_index("image_id").loc[wanted].reset_index()
    if args.num_eval_samples >= 0:
        eval_df = eval_df.head(args.num_eval_samples)
    eval_df = eval_df.reset_index(drop=True)
    logger.info(f"Evaluating {len(eval_df)} held-out paintings.")

    # 2. Classifier candidates
    logger.info("=== STEP 2: Computing Classifier Top-3 Candidates ===")
    clf_top3_by_image, label_options, clf_source = classifier_candidates(args, eval_df)
    logger.info(f"Classifier candidates from: {clf_source}")
    motifs_by_image = load_motifs(args.motifs_manifest)

    # 3. VLM client, knowledge base, review queue
    logger.info(f"=== STEP 3: Initializing VLM Client ({args.model_type}: {args.model_name}) ===")
    if args.model_type == "qwen_ollama":
        vlm_client = QwenOllamaVLMClient(
            model_name=args.model_name, timeout=args.timeout,
            image_max_size=args.image_max_size, num_ctx=args.num_ctx,
        )
    else:
        vlm_client = MockQwenVLMClient(model_name="mock-qwen")

    vlm_judge = VLMJudge(
        client=vlm_client,
        retriever=ArtKnowledgeRetriever(),
        confidence_threshold=args.confidence_threshold,
        label_options=label_options,
    )
    review_queue = ExpertReviewQueue(queue_dir=Path(args.output_dir) / "expert_review")

    # 4. Reasoning over each painting
    logger.info("=== STEP 4: Executing Multimodal VLM Reasoning & Cross-Check ===")
    eval_cases = []
    failures = 0
    for idx, row in eval_df.iterrows():
        img_id = row["image_id"]
        img_path = row["file_path"]
        gt = {t: (None if pd.isna(row.get(t)) else row.get(t)) for t in TARGETS}
        clf_top3 = clf_top3_by_image.get(img_id, {})

        detections = motifs_by_image.get(img_id, [])
        motif_labels = list(dict.fromkeys(d["label"] for d in detections))[:12]
        crop_desc = [{"label": d["label"], "bbox": [round(v) for v in d["box"]]} for d in detections[:8]]

        # Identifying metadata is withheld: real file names encode series and dates
        case_id = f"case_{idx + 1:03d}"
        blind_meta = {"image_id": case_id, "file_name": f"{case_id}.jpg", "collection_id": "Withheld"}

        logger.info(f"Analyzing [{idx + 1}/{len(eval_df)}] {img_id} (GT school={gt['school']}, period={gt['period']})")
        try:
            judgment = vlm_judge.evaluate_case(
                image_path=img_path,
                classifier_top3={} if args.hide_classifier else clf_top3,
                image_metadata=blind_meta,
                crop_descriptions=crop_desc,
                detected_motifs=motif_labels,
                ocr_text="",
                retrieval_candidates=label_options if args.hide_classifier else None,
            )
            if args.hide_classifier:
                # Cross-check against the (hidden) classifier so routing is comparable to normal runs
                judgment = vlm_judge._enforce_cross_check(judgment, clf_top3, args.confidence_threshold)
        except Exception as e:
            failures += 1
            logger.error(f"VLM call failed for {img_id}: {e}")
            judgment = vlm_judge._enforce_cross_check(
                vlm_judge._parse_json_response(""), clf_top3, args.confidence_threshold
            )

        if judgment.get("cross_check", {}).get("route_to_expert_review", False):
            review_queue.enqueue(
                image_id=img_id,
                image_path=img_path,
                classifier_top3=clf_top3,
                vlm_judgment=judgment,
                ground_truth=gt,
            )

        eval_cases.append({
            "image_id": img_id,
            "ground_truth": gt,
            "vlm_judgment": judgment,
            "classifier_top3": clf_top3,
        })

    # 5. Scoring
    logger.info("=== STEP 5: Scoring Reasoning and Generating Benchmark Report ===")
    metrics = evaluate_reasoning_dataset(eval_cases, output_dir=args.output_dir)
    review_queue.generate_queue_summary()

    with open(Path(args.output_dir) / "vlm_eval_cases.json", "w", encoding="utf-8") as f:
        json.dump({"classifier_source": clf_source,
                   "vlm_model": args.model_name + (" (classifier hidden)" if args.hide_classifier else ""), "vlm_failures": failures,
                   "cases": eval_cases}, f, indent=2, default=str)

    logger.info(f"VLM failures/timeouts: {failures}/{len(eval_df)}")
    logger.info("=== VLM REASONING PIPELINE EXECUTION COMPLETE ===")
    logger.info(f"Reasoning report: {Path(args.output_dir).resolve() / 'vlm_reasoning_benchmark.md'}")


if __name__ == "__main__":
    main()
