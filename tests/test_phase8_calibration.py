"""
Unit and integration tests for Phase 8:
Model Calibration, Post-Hoc Temperature Scaling, and Uncertainty Estimation.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.calibration import (
    MultiHeadTemperatureScaler,
    SingleHeadTemperatureScaler,
    compute_calibration_metrics,
    plot_reliability_diagrams,
)


def test_single_head_temperature_scaler():
    torch.manual_seed(42)
    n_samples = 50
    n_classes = 4

    # Simulated overconfident logits
    true_labels = torch.randint(0, n_classes, (n_samples,))
    # Overconfident: high magnitudes
    raw_logits = torch.randn(n_samples, n_classes) * 5.0

    scaler = SingleHeadTemperatureScaler(initial_temp=1.0)

    # Initial NLL
    loss_fn = nn.CrossEntropyLoss()
    initial_loss = loss_fn(raw_logits, true_labels).item()

    # Fit scaler
    learned_t = scaler.fit(val_logits=raw_logits, val_labels=true_labels, lr=0.05, max_iter=50)

    # Verify temperature is positive
    assert learned_t > 0.0

    # Scaled logits
    with torch.no_grad():
        scaled_logits = scaler(raw_logits)

    calib_loss = loss_fn(scaled_logits, true_labels).item()
    # NLL should improve (decrease) or remain equivalent
    assert calib_loss <= initial_loss + 1e-4

    # Verify rank preservation: argmax must be strictly identical!
    orig_preds = torch.argmax(raw_logits, dim=1)
    scaled_preds = torch.argmax(scaled_logits, dim=1)
    assert torch.equal(orig_preds, scaled_preds), "Temperature scaling must never alter top-1 class rank!"


def test_multihead_temperature_scaler_and_serialization(tmp_path):
    targets = ["raga", "school", "period", "artist"]
    scaler = MultiHeadTemperatureScaler(targets=targets)

    val_logits = {}
    val_labels = {}
    for t in targets:
        val_logits[t] = torch.randn(30, 4) * 3.0
        val_labels[t] = torch.randint(0, 4, (30,))

    # Fit all heads
    temps = scaler.fit_validation_logits(val_logits, val_labels)

    for t in targets:
        assert t in temps
        assert temps[t] > 0.0

    # Test JSON save and load
    json_path = tmp_path / "temps.json"
    scaler.save(json_path)
    assert json_path.exists()

    loaded_scaler = MultiHeadTemperatureScaler.load(json_path)
    assert set(loaded_scaler.temperatures.keys()) == set(targets)
    for t in targets:
        assert np.isclose(loaded_scaler.temperatures[t], temps[t], atol=1e-4)


def test_calibration_metrics_computation():
    # Case 1: Perfectly calibrated probabilities
    # 10 samples with 80% confidence, exactly 8 out of 10 correct
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_prob = np.zeros((10, 2))
    y_prob[:, 0] = 0.8  # Predict class 0 with 0.8 conf
    y_prob[:, 1] = 0.2

    metrics_perfect = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    # Average conf = 0.8, Accuracy = 8/10 = 0.8 -> ECE should be near 0
    assert metrics_perfect["ece"] < 1e-4
    assert metrics_perfect["brier_score"] >= 0.0

    # Case 2: Extreme overconfidence (100% confidence, but 0% accuracy)
    y_true_bad = np.array([1, 1, 1, 1])
    y_prob_bad = np.array([[1.0, 0.0]] * 4)
    metrics_bad = compute_calibration_metrics(y_true_bad, y_prob_bad, n_bins=5)
    # ECE should be 1.0 (100% gap between conf 1.0 and acc 0.0)
    assert np.isclose(metrics_bad["ece"], 1.0, atol=1e-4)
    assert np.isclose(metrics_bad["mce"], 1.0, atol=1e-4)


def test_reliability_diagram_rendering(tmp_path):
    mock_res_pre = {
        "raga": {
            "ece": 0.25,
            "bins": [
                {"center": 0.3, "accuracy": 0.2, "confidence": 0.3},
                {"center": 0.8, "accuracy": 0.5, "confidence": 0.8},
            ],
        }
    }
    mock_res_post = {
        "raga": {
            "ece": 0.08,
            "bins": [
                {"center": 0.3, "accuracy": 0.28, "confidence": 0.3},
                {"center": 0.7, "accuracy": 0.68, "confidence": 0.7},
            ],
        }
    }

    out_plot = tmp_path / "test_diagram.png"
    plot_reliability_diagrams(
        calib_results_before=mock_res_pre,
        calib_results_after=mock_res_post,
        output_path=out_plot,
        temperatures={"raga": 1.45},
    )

    assert out_plot.exists()
    assert out_plot.stat().st_size > 0


if __name__ == "__main__":
    import traceback

    print("Running Phase 8 Calibration Test Suite...")
    tests = [
        ("test_single_head_temperature_scaler", test_single_head_temperature_scaler),
        ("test_calibration_metrics_computation", test_calibration_metrics_computation),
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
    tmp_tests = [
        ("test_multihead_temperature_scaler_and_serialization", test_multihead_temperature_scaler_and_serialization),
        ("test_reliability_diagram_rendering", test_reliability_diagram_rendering),
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
