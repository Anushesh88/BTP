"""
Predict raga / school / period / artist for one image.

  python predict.py path/to/image.jpg
  python predict.py path/to/image.jpg --skip_gate     # force a prediction even for non-miniatures

The image first passes the out-of-distribution gate (src/ood.py). Anything that is not an
Indian miniature painting is rejected instead of being forced into a Mughal/Rajasthani/Pahari
label. Accepted images go through the same steps the model was trained with: YOLO-World motif
detection, dual-branch fused features, the Phase 7 multi-head classifier and Phase 8 calibration.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from src.feature_fusion import DualBranchFusionPipeline
from src.motif_detector import DETECTION_PROMPTS, MultiWindowMotifPipeline, OpenVocabMotifDetector
from src.ood import MiniatureGate
from src.preprocessing import PaintingPreprocessor, SlidingWindowGenerator
import torch
from src.calibration import MultiHeadTemperatureScaler
from src.multihead_classifier import MultiHeadTrainer

TARGETS = ["school", "period", "raga", "artist"]


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def motif_crops(img_rgb: np.ndarray, max_crops: int = 6):
    """Phase 5 settings: canvas capped at 1280 px, 640 px windows, confidence 0.15."""
    canvas = resize_max(img_rgb, 1280)
    enhanced = PaintingPreprocessor(denoise_method="median").process(canvas)
    windows = SlidingWindowGenerator(window_size=(640, 640), stride=(320, 320)).extract_windows(enhanced)
    detector = OpenVocabMotifDetector(custom_classes=DETECTION_PROMPTS, conf_threshold=0.15)
    dets = MultiWindowMotifPipeline(detector, global_iou_threshold=0.40, min_confidence=0.15).detect_painting_motifs(enhanced, windows)
    h, w = enhanced.shape[:2]
    crops, labels = [], []
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            crops.append(enhanced[y1:y2, x1:x2])
            labels.append(d["label"])
    return crops[:max_crops], labels[:max_crops]


def main():
    p = argparse.ArgumentParser(description="Predict a miniature painting's raga, school, period and artist")
    p.add_argument("image")
    p.add_argument("--skip_gate", action="store_true", help="Predict even if the gate rejects the image")
    p.add_argument("--models_dir", default="models")
    args = p.parse_args()
    logging.disable(logging.INFO)

    gate = MiniatureGate().check(args.image)
    print(f"Gate: similarity to training paintings {gate.knn_similarity:.2f} (needs >= 0.25), "
          f"CLIP miniature probability {gate.clip_miniature_prob:.2f} (needs >= 0.50)")
    if not gate.is_miniature:
        print(f"REJECTED: not recognised as an Indian miniature painting ({gate.reason}).")
        if not args.skip_gate:
            sys.exit(2)
        print("--skip_gate set: predicting anyway; treat the result as meaningless.\n")

    preproc = PaintingPreprocessor(denoise_method="none")
    img = resize_max(preproc.load_and_convert_rgb(args.image), 1024)
    crops, labels = motif_crops(preproc.load_and_convert_rgb(args.image))
    fused = DualBranchFusionPipeline(backbone_model_name="vit_b_16").extract_image_features(img, crop_images=crops)

    trainer = MultiHeadTrainer.load_checkpoint(Path(args.models_dir) / "multihead_fused_classifier.pt")
    with torch.no_grad():
        logits = trainer.model(torch.tensor(fused[None], dtype=torch.float32).to(trainer.device))
    temps = Path(args.models_dir) / "temperature_scalers.json"
    probs = MultiHeadTemperatureScaler.load(temps).scale_probabilities(logits) if temps.exists() else \
        {t: torch.softmax(l, 1).cpu().numpy() for t, l in logits.items()}

    print(f"Motifs used: {', '.join(labels) or 'none detected'}\n")
    for t in TARGETS:
        classes = trainer.label_encoders[t].classes_
        top = np.argsort(probs[t][0])[::-1][:3]
        cands = ", ".join(f"{classes[i]} {probs[t][0][i] * 100:.1f}%" for i in top)
        note = "  (only 4 attributed training paintings: not meaningful)" if t == "artist" else ""
        print(f"{t.capitalize():7s} {cands}{note}")


if __name__ == "__main__":
    main()
