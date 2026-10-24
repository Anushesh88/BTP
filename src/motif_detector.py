"""
Phase 5: Open-Vocabulary Motif & Region Detection with Multi-Window NMS.
Supports zero-shot open-vocabulary detection (YOLO-World), sliding window proposal
projection, Non-Maximum Suppression, coarse taxonomy grouping, and fine-tuning trigger.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.ops as ops

from src.preprocessing import SlidingWindowGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical Art-Historical Motif Taxonomy for Mughal & Ragamala Paintings
CANONICAL_MOTIFS = [
    # Figures & Personae
    "nayika", "nayaka", "shiva", "krishna", "attendant", "priest", "musician", "dancer", "ascetic",
    # Fauna
    "deer", "peacock", "elephant", "cow", "horse", "snake", "crane", "tiger",
    # Flora
    "lotus", "flowering tree", "kadamba tree", "plantain tree", "garland", "monsoon clouds",
    # Material Culture & Architecture
    "veena", "tambura", "swing", "trident", "crown", "turban", "canopy", "oil lamp", "pavilion", "sword",
    # Inscriptions
    "inscribed text", "calligraphy"
]

# Plain-English detection prompts. YOLO-World's CLIP text encoder does not know Sanskrit
# iconographic terms ("nayika", "kadamba tree"), which yield near-zero recall on real
# paintings; these everyday prompts detect far more regions on the real dataset.
DETECTION_PROMPTS = [
    "woman", "man", "person", "musician", "dancer", "child",
    "deer", "peacock", "elephant", "cow", "horse", "bird", "tiger", "snake",
    "tree", "flower", "lotus", "clouds", "water",
    "building", "pavilion", "canopy", "carpet",
    "turban", "crown", "sword", "musical instrument", "swing", "lamp",
    "text",
]

TAXONOMY_MAPPING = {
    "human": ["nayika", "nayaka", "shiva", "krishna", "attendant", "priest", "musician", "dancer", "ascetic", "person", "woman", "man", "child"],
    "animal": ["deer", "peacock", "elephant", "cow", "horse", "snake", "crane", "tiger", "bird"],
    "flora": ["lotus", "flowering tree", "kadamba tree", "plantain tree", "garland", "monsoon clouds", "plant", "tree", "flower", "clouds", "water"],
    "architecture": ["pavilion", "canopy", "arch", "terrace", "temple", "shrine", "building", "carpet"],
    "instrument_object": ["veena", "tambura", "swing", "trident", "crown", "turban", "oil lamp", "sword", "cymbals", "drum", "musical instrument", "lamp"],
    "text": ["inscribed text", "calligraphy", "text"]
}


def categorize_motif_label(label: str) -> str:
    """Assign coarse category label: human, animal, flora, architecture, instrument_object, text."""
    lbl_lower = label.lower()
    for category, members in TAXONOMY_MAPPING.items():
        if any(m in lbl_lower or lbl_lower in m for m in members):
            return category
    return "object"


class OpenVocabMotifDetector:
    """
    Open-Vocabulary Object Detector for Indian Miniature Paintings.
    Applies YOLO-World with art-historical text prompts, or falls back to
    deterministic simulation mode for offline environments.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        custom_classes: Optional[List[str]] = None,
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        use_mock: bool = False,
    ):
        self.model_name = model_name
        self.classes = custom_classes or CANONICAL_MOTIFS
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_mock = use_mock
        self.model = None

        if not self.use_mock:
            try:
                from ultralytics import YOLOWorld
                logger.info(f"Loading YOLO-World model: {model_name}")
                self.model = YOLOWorld(model_name)
                self.model.set_classes(self.classes)
                logger.info(f"YOLO-World initialized with {len(self.classes)} open-vocabulary classes.")
            except Exception as e:
                logger.warning(f"Could not load YOLOWorld ({e}). Falling back to simulated detector.")
                self.use_mock = True

    def detect_on_image(self, img_bgr_or_rgb: np.ndarray) -> List[Dict]:
        """
        Runs open-vocabulary detection on a single image array.
        Returns list of detections: [{"box": [x1, y1, x2, y2], "confidence": float, "label": str, "category": str}]
        """
        if self.use_mock:
            return self._mock_detect(img_bgr_or_rgb)

        results = self.model.predict(
            source=img_bgr_or_rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = self.classes[cls_id] if cls_id < len(self.classes) else "object"
                detections.append({
                    "box": [float(c) for c in xyxy],
                    "confidence": conf,
                    "label": label,
                    "category": categorize_motif_label(label),
                })

        return detections

    def _mock_detect(self, img: np.ndarray) -> List[Dict]:
        """Deterministic mock detector for unit testing and offline environments."""
        h, w = img.shape[:2]
        # Return synthetic motifs positioned reasonably in the canvas
        return [
            {
                "box": [float(w * 0.35), float(h * 0.25), float(w * 0.65), float(h * 0.75)],
                "confidence": 0.82,
                "label": "nayika",
                "category": "human",
            },
            {
                "box": [float(w * 0.40), float(h * 0.45), float(w * 0.60), float(h * 0.65)],
                "confidence": 0.78,
                "label": "veena",
                "category": "instrument_object",
            },
            {
                "box": [float(w * 0.10), float(h * 0.60), float(w * 0.30), float(h * 0.85)],
                "confidence": 0.74,
                "label": "deer",
                "category": "animal",
            },
        ]


class MultiWindowMotifPipeline:
    """
    Orchestrates sliding-window detection, multi-window coordinate projection,
    global Non-Maximum Suppression (NMS), and region crop extraction.
    """

    def __init__(
        self,
        detector: OpenVocabMotifDetector,
        global_iou_threshold: float = 0.40,
        min_confidence: float = 0.25,
    ):
        self.detector = detector
        self.global_iou_threshold = global_iou_threshold
        self.min_confidence = min_confidence

    def detect_painting_motifs(
        self,
        full_image: np.ndarray,
        windows: List[Tuple[np.ndarray, Dict]],
    ) -> List[Dict]:
        """
        1. Run detection on every sliding window.
        2. Project local window boxes back to global canvas coordinates.
        3. Merge overlapping window proposals via global NMS.
        4. Filter by confidence and assign coarse categories.
        """
        orig_h, orig_w = full_image.shape[:2]
        raw_global_proposals = []

        # Step 3.1: Detection on each window
        for win_img, win_meta in windows:
            local_detections = self.detector.detect_on_image(win_img)

            # Project window proposals to global canvas
            for det in local_detections:
                g_box = SlidingWindowGenerator.map_box_to_global(det["box"], win_meta)
                raw_global_proposals.append({
                    "box": g_box,
                    "confidence": det["confidence"],
                    "label": det["label"],
                    "category": det["category"],
                    "source_window_id": win_meta["window_id"],
                })

        if not raw_global_proposals:
            return []

        # Step 3.2: Merge overlapping detections using Global NMS
        merged_detections = self._apply_global_nms(raw_global_proposals)

        # Step 3.3: Refine & filter by confidence threshold
        refined_detections = [
            d for d in merged_detections if d["confidence"] >= self.min_confidence
        ]

        return refined_detections

    def _apply_global_nms(self, proposals: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression across global proposals grouped by class label.
        Uses torchvision.ops.nms.
        """
        # Group by label to avoid suppressing different co-located motifs (e.g. Nayika holding Veena)
        label_groups = {}
        for p in proposals:
            lbl = p["label"]
            if lbl not in label_groups:
                label_groups[lbl] = []
            label_groups[lbl].append(p)

        merged = []

        for lbl, group in label_groups.items():
            boxes_tensor = torch.tensor([p["box"] for p in group], dtype=torch.float32)
            scores_tensor = torch.tensor([p["confidence"] for p in group], dtype=torch.float32)

            # PyTorch torchvision NMS
            keep_indices = ops.nms(boxes_tensor, scores_tensor, self.global_iou_threshold)
            keep_indices = keep_indices.cpu().numpy().tolist()

            for idx in keep_indices:
                merged.append(group[idx])

        # Sort descending by confidence
        merged.sort(key=lambda x: x["confidence"], reverse=True)
        return merged

    def crop_and_store_regions(
        self,
        full_image: np.ndarray,
        detections: List[Dict],
        output_crop_dir: Union[str, Path],
        image_id: str,
    ) -> List[Dict]:
        """
        Step 3.5 & 3.6: Crop detected regions from original canvas and store with metadata.
        """
        crop_path_dir = Path(output_crop_dir) / image_id
        crop_path_dir.mkdir(parents=True, exist_ok=True)
        h, w = full_image.shape[:2]

        enriched_detections = []

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = [int(round(coord)) for coord in det["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            crop_arr = full_image[y1:y2, x1:x2]
            crop_filename = f"{image_id}_motif_{idx:03d}_{det['label'].replace(' ', '_')}.jpg"
            crop_file_path = crop_path_dir / crop_filename

            Image.fromarray(crop_arr).save(crop_file_path, quality=95)

            enriched = dict(det)
            enriched["motif_id"] = f"{image_id}_motif_{idx:03d}"
            enriched["box"] = [x1, y1, x2, y2]
            enriched["crop_path"] = str(crop_file_path)
            enriched_detections.append(enriched)

        return enriched_detections


def evaluate_motif_recall(
    detected_motifs: List[Dict],
    ground_truth_boxes: List[Dict],
    iou_thresh: float = 0.50,
) -> Tuple[float, float, float, Dict]:
    """
    Step 5.3: Compare open-vocabulary detections against ground truth box annotations.
    Computes precision, recall, and F1 score, and returns fine-tuning recommendation.
    """
    if not ground_truth_boxes:
        return 1.0, 1.0, 1.0, {"recommend_finetuning": False, "reason": "No ground truth boxes provided."}

    matched_gt = set()
    true_positives = 0

    for det in detected_motifs:
        d_box = det["box"]
        d_label = det["label"].lower()

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue
            gt_box = gt["box"]
            gt_label = gt.get("label", "").lower()

            # Label match or category match
            if d_label in gt_label or gt_label in d_label:
                # Compute IoU
                xA = max(d_box[0], gt_box[0])
                yA = max(d_box[1], gt_box[1])
                xB = min(d_box[2], gt_box[2])
                yB = min(d_box[3], gt_box[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                areaA = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
                areaB = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                union = areaA + areaB - inter
                iou = inter / union if union > 0 else 0.0

                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    precision = true_positives / len(detected_motifs) if detected_motifs else 0.0
    recall = len(matched_gt) / len(ground_truth_boxes) if ground_truth_boxes else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Decision rule: Fine-tune only if recall < 0.70
    recommend_finetune = recall < 0.70
    reason = (
        f"Recall ({recall*100:.1f}%) is below 70% threshold; fine-tuning on custom box labels recommended."
        if recommend_finetune
        else f"Recall ({recall*100:.1f}%) exceeds 70% threshold; zero-shot open-vocabulary detection is sufficient."
    )

    return precision, recall, f1, {
        "recommend_finetuning": recommend_finetune,
        "reason": reason,
        "matched_gt_count": len(matched_gt),
        "total_gt_count": len(ground_truth_boxes),
    }
