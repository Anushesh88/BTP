"""
Unit and integration tests for Phase 3 Baseline pipeline.
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataset import (
    MiniaturePaintingDataset,
    create_grouped_splits,
    extract_entity_from_text,
    infer_manuscript_group,
    parse_metadata_manifest,
)
from src.feature_extractor import FrozenVisionBackbone
from src.baseline_classifier import BaselineMultiHeadClassifier
from src.evaluate import (
    compute_expected_calibration_error,
    evaluate_baseline_predictions,
)
from run_baseline import generate_synthetic_miniature_dataset


def test_manuscript_group_inference_and_entity_extraction():
    # Test entity regex matching
    caption = "A 17th Century painting depicting Bhairava raga from the Mewar school by Basawan."
    assert extract_entity_from_text(caption, ["Bhairava", "Malkaus"]) == "Bhairava"
    assert extract_entity_from_text(caption, ["Mughal", "Mewar"]) == "Mewar"
    assert extract_entity_from_text(caption, ["Basawan", "Miskin"]) == "Basawan"

    # Test manuscript series inference
    group = infer_manuscript_group("hamzanama_04.jpg", caption)
    assert group == "HAMZANAMA"


def test_leak_free_grouped_splits(tmp_path):
    # Create synthetic dataset
    data_dir = tmp_path / "data"
    generate_synthetic_miniature_dataset(data_dir, num_samples=40)

    manifest_df = parse_metadata_manifest(data_dir)
    assert len(manifest_df) == 40
    assert "collection_id" in manifest_df.columns

    train_df, val_df, test_df = create_grouped_splits(
        manifest_df,
        group_col="collection_id",
        test_size=0.25,
        val_size=0.25,
        random_state=42,
    )

    # Check that lengths sum to total
    assert len(train_df) + len(val_df) + len(test_df) == 40

    # Ensure NO leakage of manuscript collections across splits
    train_groups = set(train_df["collection_id"])
    val_groups = set(val_df["collection_id"])
    test_groups = set(test_df["collection_id"])

    assert len(train_groups.intersection(val_groups)) == 0, "Leakage between train and val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Leakage between train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Leakage between val and test!"


def test_frozen_vision_backbone():
    # Test backbone parameter freezing and feature extraction dimensions
    backbone = FrozenVisionBackbone(model_name="vit_b_16", pretrained=False)

    # Verify every single parameter is frozen
    for name, param in backbone.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"

    # Test forward pass with dummy tensor [B, C, H, W]
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone(dummy_input)

    assert out.shape == (2, backbone.feature_dim)
    assert backbone.feature_dim == 768


def test_baseline_classifier_and_evaluation(tmp_path):
    # Synthetic feature matrix
    num_samples = 30
    feat_dim = 64
    np.random.seed(42)
    X = np.random.randn(num_samples, feat_dim)

    # Synthetic labels
    df = pd.DataFrame({
        "raga": np.random.choice(["Bhairava", "Malkaus", "Hindol"], size=num_samples),
        "school": np.random.choice(["Mughal", "Mewar"], size=num_samples),
        "period": np.random.choice(["16th Century", "17th Century"], size=num_samples),
        "artist": np.random.choice(["Basawan", "Miskin"], size=num_samples),
    })

    # Train classifier
    clf = BaselineMultiHeadClassifier(targets=["raga", "school", "period", "artist"])
    clf.fit(X, df)

    # Check predictions
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    for target in ["raga", "school", "period", "artist"]:
        assert target in preds
        assert len(preds[target]) == num_samples
        assert target in probs
        assert probs[target].shape[0] == num_samples

    # Check model save and reload
    model_file = tmp_path / "model.joblib"
    clf.save(model_file)
    assert model_file.exists()

    loaded_clf = BaselineMultiHeadClassifier.load(model_file)
    preds_loaded = loaded_clf.predict(X)
    assert np.array_equal(preds["raga"], preds_loaded["raga"])

    # Test evaluation module
    reports_dir = tmp_path / "reports"
    eval_results = evaluate_baseline_predictions(
        predictions=preds,
        probabilities=probs,
        df_true=df,
        label_encoders=clf.label_encoders,
        output_dir=reports_dir,
    )

    assert (reports_dir / "baseline_benchmark.json").exists()
    assert (reports_dir / "baseline_benchmark.md").exists()
    assert (reports_dir / "confusion_matrix_raga.png").exists()

    # Test Expected Calibration Error
    y_true_mock = np.array([0, 1, 0, 1])
    prob_mock = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]])
    ece, _ = compute_expected_calibration_error(y_true_mock, prob_mock)
    assert 0.0 <= ece <= 1.0


if __name__ == "__main__":
    import tempfile
    import traceback

    tests = [
        ("test_manuscript_group_inference_and_entity_extraction", test_manuscript_group_inference_and_entity_extraction, []),
        ("test_frozen_vision_backbone", test_frozen_vision_backbone, []),
    ]

    print("Running Phase 3 Test Suite...")
    passed = 0
    failed = 0

    # Tests without tmp_path
    for name, func, args in tests:
        try:
            func(*args)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    # Tests requiring tmp_path
    tmp_tests = [
        ("test_leak_free_grouped_splits", test_leak_free_grouped_splits),
        ("test_baseline_classifier_and_evaluation", test_baseline_classifier_and_evaluation),
    ]

    for name, func in tmp_tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                func(Path(td))
                print(f"  [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
