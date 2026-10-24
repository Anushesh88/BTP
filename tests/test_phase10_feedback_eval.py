"""
Tests for real-data plumbing and Phase 10: honest label normalization, frozen split
loading, missing-label masking, the HITL feedback store, and the VLM response schema.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.baseline_classifier import BaselineMultiHeadClassifier
from src.dataset import load_frozen_splits, normalize_targets
from src.feedback import FeedbackStore
from src.vlm_judge import VLMJudge, build_response_schema
from src.vlm_client import MockQwenVLMClient
import torch
from src.multihead_classifier import IGNORE_INDEX, MultiHeadTrainer


def _toy_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(n)],
        "raga": ["None"] * (n - 6) + ["Shri Raga/Ragini"] * 3 + ["Kamod Raga/Ragini"] + ["Ragamala (Unspecified)"] * 2,
        "school": rng.choice(["Mughal", "Rajasthani"], n).tolist(),
        "period": rng.choice(["Akbar Period (c. 1556-1605)", "Unknown", "18th Century"], n).tolist(),
        "artist": ["Unknown"] * n,
    })


def test_normalize_targets():
    df = normalize_targets(_toy_df())
    assert "None" not in set(df["raga"]) and "Non-Ragamala" in set(df["raga"])
    assert "Ragamala (Other)" in set(df["raga"]), "Single-sample raga should be pooled"
    assert "Shri Raga/Ragini" in set(df["raga"]), "Raga with >= 3 samples should be kept"
    assert df["period"].isna().sum() == (_toy_df()["period"] == "Unknown").sum(), "'Unknown' period must become NaN"
    assert "18th Century" not in set(df["period"].dropna()), "Period alias should be merged"
    print("  [PASS] test_normalize_targets")


def test_frozen_splits_load_and_are_disjoint():
    if not Path("data/metadata/train_split.jsonl").exists():
        print("  [SKIP] test_frozen_splits_load_and_are_disjoint (no frozen splits)")
        return
    tr, va, te = load_frozen_splits(apply_feedback=False)
    assert len(tr) and len(va) and len(te)
    assert not (set(tr["group_key"]) & set(te["group_key"])), "Group leakage between train and test"
    assert Path(tr["file_path"].iloc[0]).is_absolute() and Path(tr["file_path"].iloc[0]).exists()
    assert tr["period"].isna().any(), "Unknown periods should be NaN after normalization"
    print("  [PASS] test_frozen_splits_load_and_are_disjoint")


def test_missing_labels_are_masked_in_training():
    df = normalize_targets(_toy_df())
    X = np.random.default_rng(1).normal(size=(len(df), 16)).astype(np.float32)

    base = BaselineMultiHeadClassifier(targets=["school", "period"]).fit(X, df)
    assert "nan" not in set(base.label_encoders["period"].classes_), "NaN must not become a class"

    trainer = MultiHeadTrainer(input_dim=16, targets=["school", "period"], hidden_dim=32, device="cpu")
    trainer.train_model(X, df, X, df, epochs=2, batch_size=8)
    assert "nan" not in set(trainer.label_encoders["period"].classes_)
    assert IGNORE_INDEX < 0
    print("  [PASS] test_missing_labels_are_masked_in_training")


def test_feedback_store_applies_latest_correction():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = FeedbackStore(tmp / "fb.jsonl")
        store.record("img_1", "school", "Mewar", reviewer="a")
        store.record("img_1", "school", "Bundi", reviewer="b")  # later entry wins
        store.record("img_2", "period", None, reviewer="a")     # mark unknown
        df = store.apply_to(normalize_targets(_toy_df()))
        assert df.loc[df.image_id == "img_1", "school"].item() == "Bundi"
        assert pd.isna(df.loc[df.image_id == "img_2", "period"].item())
        assert len(store.load()) == 3
        try:
            store.record("img_1", "caption", "x", reviewer="a")
            raise AssertionError("Invalid target should be rejected")
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp)
    print("  [PASS] test_feedback_store_applies_latest_correction")


def test_vlm_schema_and_label_options():
    options = {"raga": ["Non-Ragamala", "Shri Raga/Ragini"], "school": ["Mughal"], "period": ["18th"], "artist": ["Unknown"]}
    schema = build_response_schema(options)
    assert schema["properties"]["predicted_raga"]["properties"]["value"]["enum"][-1] == "uncertain"
    assert "predicted_period" in schema["required"]
    json.dumps(schema)  # must be serializable for the Ollama payload

    judge = VLMJudge(client=MockQwenVLMClient(), label_options=options)
    prompt = judge.build_user_prompt({"image_id": "case_001"}, {}, "", [], "", [])
    assert "ALLOWED ANSWER LABELS" in prompt and '"Non-Ragamala"' in prompt

    # Malformed VLM fields must not crash the cross-check
    res = judge._enforce_cross_check({"predicted_raga": None, "predicted_school": {"value": None, "confidence": "high"}}, {}, 0.7)
    assert res["cross_check"]["route_to_expert_review"] is True
    print("  [PASS] test_vlm_schema_and_label_options")


if __name__ == "__main__":
    print("Running Phase 10 / Real-Data Plumbing Test Suite...")
    tests = [
        test_normalize_targets,
        test_frozen_splits_load_and_are_disjoint,
        test_missing_labels_are_masked_in_training,
        test_feedback_store_applies_latest_correction,
        test_vlm_schema_and_label_options,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
