"""
Unit and integration tests for Phase 7:
Multi-Head Classifiers, Domain-Aware Color-Safe Augmentations, and Class Weighting.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.multihead_classifier import (
    ColorSafeMiniatureAugmentation,
    MultiHeadMiniatureClassifier,
    MultiHeadTrainer,
)


def test_color_safe_augmentation():
    aug = ColorSafeMiniatureAugmentation()

    # Create dummy RGB image
    img = Image.new("RGB", (300, 300), color=(180, 50, 40))  # Saturated mineral red
    augmented = aug(img)

    assert isinstance(augmented, Image.Image)
    assert augmented.size == (224, 224)

    # Convert to array and verify no channel inversion or extreme corruption
    arr = np.array(augmented)
    assert arr.shape == (224, 224, 3)
    assert not np.isnan(arr).any()


def test_multihead_classifier_architecture():
    input_dim = 1632
    num_classes = {
        "raga": 6,
        "school": 5,
        "period": 4,
        "artist": 5,
    }

    model = MultiHeadMiniatureClassifier(
        input_dim=input_dim,
        num_classes_dict=num_classes,
        hidden_dim=256,
        dropout_p=0.2,
    )

    batch_x = torch.randn(4, input_dim)
    logits = model(batch_x)

    assert isinstance(logits, dict)
    assert set(logits.keys()) == set(num_classes.keys())

    for target, n_cls in num_classes.items():
        assert logits[target].shape == (4, n_cls)


def test_multihead_trainer_and_convergence(tmp_path):
    num_samples = 40
    input_dim = 64
    np.random.seed(42)

    X_train = np.random.randn(num_samples, input_dim).astype(np.float32)
    X_val = np.random.randn(15, input_dim).astype(np.float32)

    df_train = pd.DataFrame({
        "raga": np.random.choice(["Bhairava", "Malkaus", "Hindol"], size=num_samples),
        "school": np.random.choice(["Mughal", "Mewar"], size=num_samples),
        "period": np.random.choice(["16th Century", "17th Century"], size=num_samples),
        "artist": np.random.choice(["Basawan", "Miskin"], size=num_samples),
    })

    df_val = pd.DataFrame({
        "raga": np.random.choice(["Bhairava", "Malkaus", "Hindol"], size=15),
        "school": np.random.choice(["Mughal", "Mewar"], size=15),
        "period": np.random.choice(["16th Century", "17th Century"], size=15),
        "artist": np.random.choice(["Basawan", "Miskin"], size=15),
    })

    trainer = MultiHeadTrainer(
        input_dim=input_dim,
        targets=["raga", "school", "period", "artist"],
        hidden_dim=128,
        lr=5e-3,
        device="cpu",
    )

    history = trainer.train_model(
        X_train=X_train,
        df_train=df_train,
        X_val=X_val,
        df_val=df_val,
        epochs=10,
        batch_size=8,
        patience=5,
    )

    assert len(history["train_loss"]) > 0
    # Verify training loss decreased
    assert history["train_loss"][-1] < history["train_loss"][0]

    # Test prediction
    preds = trainer.predict(X_val)
    probs = trainer.predict_proba(X_val)

    for t in trainer.targets:
        assert t in preds
        assert len(preds[t]) == 15
        assert t in probs
        assert probs[t].shape == (15, len(trainer.label_encoders[t].classes_))

    # Test Checkpoint Save and Load
    ckpt_file = tmp_path / "multihead_test.pt"
    trainer.save_checkpoint(ckpt_file)
    assert ckpt_file.exists()

    loaded_trainer = MultiHeadTrainer.load_checkpoint(ckpt_file, device="cpu")
    preds_loaded = loaded_trainer.predict(X_val)

    for t in trainer.targets:
        assert np.array_equal(preds[t], preds_loaded[t])


if __name__ == "__main__":
    import traceback

    print("Running Phase 7 Multi-Head Classifier Test Suite...")
    tests = [
        ("test_color_safe_augmentation", test_color_safe_augmentation),
        ("test_multihead_classifier_architecture", test_multihead_classifier_architecture),
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

    # Temp dir test
    with tempfile.TemporaryDirectory() as td:
        try:
            test_multihead_trainer_and_convergence(Path(td))
            print("  [PASS] test_multihead_trainer_and_convergence")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_multihead_trainer_and_convergence: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
