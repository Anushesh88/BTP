"""
Unit and integration tests for Phase 5:
Open-Vocabulary Motif Detection, Multi-Window NMS, Crop Storage, and Recall Evaluation.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.motif_detector import (
    CANONICAL_MOTIFS,
    MultiWindowMotifPipeline,
    OpenVocabMotifDetector,
    categorize_motif_label,
    evaluate_motif_recall,
)


def test_motif_taxonomy_categorization():
    assert categorize_motif_label("nayika") == "human"
    assert categorize_motif_label("krishna") == "human"
    assert categorize_motif_label("deer") == "animal"
    assert categorize_motif_label("peacock") == "animal"
    assert categorize_motif_label("lotus") == "flora"
    assert categorize_motif_label("kadamba tree") == "flora"
    assert categorize_motif_label("veena") == "instrument_object"
    assert categorize_motif_label("tambura") == "instrument_object"
    assert categorize_motif_label("pavilion") == "architecture"
    assert categorize_motif_label("inscribed text") == "text"


def test_open_vocab_detector_interface():
    detector = OpenVocabMotifDetector(use_mock=True)
    assert len(detector.classes) == len(CANONICAL_MOTIFS)

    dummy_canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    detections = detector.detect_on_image(dummy_canvas)

    assert len(detections) > 0
    for det in detections:
        assert "box" in det
        assert "confidence" in det
        assert "label" in det
        assert "category" in det
        assert len(det["box"]) == 4


def test_multi_window_nms_merging():
    detector = OpenVocabMotifDetector(use_mock=True)
    pipeline = MultiWindowMotifPipeline(detector=detector, global_iou_threshold=0.40)

    # Simulate two overlapping window proposals detecting the same nayika at [100, 100, 250, 300]
    # Proposal 1 from window A: [100, 100, 250, 300] with conf 0.85
    # Proposal 2 from overlapping window B: [104, 102, 252, 298] with conf 0.80
    duplicate_proposals = [
        {
            "box": [100.0, 100.0, 250.0, 300.0],
            "confidence": 0.85,
            "label": "nayika",
            "category": "human",
            "source_window_id": 0,
        },
        {
            "box": [104.0, 102.0, 252.0, 298.0],
            "confidence": 0.80,
            "label": "nayika",
            "category": "human",
            "source_window_id": 1,
        },
        {
            # Distinct motif: veena nearby
            "box": [120.0, 150.0, 180.0, 220.0],
            "confidence": 0.78,
            "label": "veena",
            "category": "instrument_object",
            "source_window_id": 0,
        },
    ]

    merged = pipeline._apply_global_nms(duplicate_proposals)

    # Duplicate nayika proposal must be suppressed down to the higher confidence one
    nayika_merged = [m for m in merged if m["label"] == "nayika"]
    assert len(nayika_merged) == 1, "Global NMS must merge duplicate window proposals!"
    assert nayika_merged[0]["confidence"] == 0.85

    # Distinct veena motif must be preserved
    veena_merged = [m for m in merged if m["label"] == "veena"]
    assert len(veena_merged) == 1


def test_crop_and_store_regions(tmp_path):
    detector = OpenVocabMotifDetector(use_mock=True)
    pipeline = MultiWindowMotifPipeline(detector=detector)

    canvas = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    detections = [
        {"box": [50, 50, 150, 150], "confidence": 0.85, "label": "veena", "category": "instrument_object"},
        {"box": [200, 100, 320, 250], "confidence": 0.80, "label": "nayika", "category": "human"},
    ]

    crops_meta = pipeline.crop_and_store_regions(
        full_image=canvas,
        detections=detections,
        output_crop_dir=tmp_path / "crops",
        image_id="test_canvas_01",
    )

    assert len(crops_meta) == 2
    for c in crops_meta:
        crop_file = Path(c["crop_path"])
        assert crop_file.exists()
        # Verify saved crop can be opened as an image
        with Image.open(crop_file) as im:
            assert im.size[0] > 0 and im.size[1] > 0


def test_recall_evaluation_and_finetune_trigger():
    # Ground truth: 2 motifs
    gt_boxes = [
        {"box": [50, 50, 150, 150], "label": "veena"},
        {"box": [200, 100, 300, 250], "label": "nayika"},
    ]

    # Case 1: High recall (both detected)
    detected_high = [
        {"box": [52, 48, 148, 152], "label": "veena"},
        {"box": [198, 102, 302, 248], "label": "nayika"},
    ]
    p1, r1, f1_1, decision1 = evaluate_motif_recall(detected_high, gt_boxes)
    assert r1 == 1.0
    assert decision1["recommend_finetuning"] is False

    # Case 2: Low recall (missed veena, recall = 50% < 70%)
    detected_low = [
        {"box": [198, 102, 302, 248], "label": "nayika"},
    ]
    p2, r2, f1_2, decision2 = evaluate_motif_recall(detected_low, gt_boxes)
    assert r2 == 0.50
    assert decision2["recommend_finetuning"] is True
    assert "below 70%" in decision2["reason"]


if __name__ == "__main__":
    import traceback

    print("Running Phase 5 Motif Detection Test Suite...")
    tests = [
        ("test_motif_taxonomy_categorization", test_motif_taxonomy_categorization),
        ("test_open_vocab_detector_interface", test_open_vocab_detector_interface),
        ("test_multi_window_nms_merging", test_multi_window_nms_merging),
        ("test_recall_evaluation_and_finetune_trigger", test_recall_evaluation_and_finetune_trigger),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    # Temp dir tests
    with tempfile.TemporaryDirectory() as td:
        try:
            test_crop_and_store_regions(Path(td))
            print("  [PASS] test_crop_and_store_regions")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_crop_and_store_regions: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
