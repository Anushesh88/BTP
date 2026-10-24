"""
Unit and integration tests for Phase 6:
Dual-Branch Feature Engineering & Fusion.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from src.feature_fusion import DualBranchFusionPipeline, HandcraftedFeatureExtractor


def test_handcrafted_sub_extractors():
    extractor = HandcraftedFeatureExtractor()
    test_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    # 1. Color features
    c_feats = extractor.extract_color_features(test_img)
    assert len(c_feats) > 0
    assert not np.isnan(c_feats).any()
    assert not np.isinf(c_feats).any()

    # 2. Texture features (GLCM + LBP + Gabor)
    t_feats = extractor.extract_texture_features(test_img)
    assert len(t_feats) > 0
    assert not np.isnan(t_feats).any()

    # 3. Stroke features (Canny + Orientation + Lengths)
    s_feats = extractor.extract_stroke_features(test_img)
    assert len(s_feats) > 0
    assert not np.isnan(s_feats).any()

    # 4. Composition features (3x3 grid + Symmetry + Center of Mass)
    comp_feats = extractor.extract_composition_features(test_img)
    assert len(comp_feats) > 0
    assert not np.isnan(comp_feats).any()

    # Total dimension check
    total_feats = extractor.extract_all(test_img)
    assert len(total_feats) == len(c_feats) + len(t_feats) + len(s_feats) + len(comp_feats)
    assert len(total_feats) == extractor.feature_dim


def test_dual_branch_fusion_pipeline():
    pipeline = DualBranchFusionPipeline(backbone_model_name="vit_b_16")

    canvas = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Case 1: Extract with zero region crops
    fused_no_crops = pipeline.extract_image_features(canvas, crop_images=[])
    expected_dim = (pipeline.backbone.feature_dim * 2) + pipeline.handcrafted.feature_dim

    assert len(fused_no_crops) == expected_dim
    assert not np.isnan(fused_no_crops).any()

    # Case 2: Extract with multiple motif crops (mean pooling)
    crop1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    crop2 = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

    fused_with_crops = pipeline.extract_image_features(canvas, crop_images=[crop1, crop2])
    assert len(fused_with_crops) == expected_dim
    assert not np.isnan(fused_with_crops).any()

    # Ensure region branch is non-zero when crops are provided
    r_start = pipeline.backbone.feature_dim
    r_end = pipeline.backbone.feature_dim * 2
    assert np.any(fused_with_crops[r_start:r_end] != 0)


def test_fused_dataset_extraction_and_caching(tmp_path):
    pipeline = DualBranchFusionPipeline(backbone_model_name="vit_b_16")

    # Create dummy images
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ids = []

    for i in range(3):
        p = img_dir / f"test_{i}.jpg"
        Image.new("RGB", (128, 128), color=(i * 40, i * 50, i * 60)).save(p)
        paths.append(str(p))
        ids.append(f"test_{i}")

    df = pd.DataFrame({"image_id": ids, "file_path": paths})
    cache_file = tmp_path / "fused_cache.npz"

    # 1. Compute and cache
    fused_mat, _ = pipeline.extract_dataset_fused_features(
        df=df,
        cache_file=cache_file,
        force_recompute=True,
    )
    assert fused_mat.shape[0] == 3
    assert cache_file.exists()

    # 2. Verify instant reload from cache
    reloaded_mat, _ = pipeline.extract_dataset_fused_features(
        df=df,
        cache_file=cache_file,
        force_recompute=False,
    )
    assert np.array_equal(fused_mat, reloaded_mat)


if __name__ == "__main__":
    import traceback

    print("Running Phase 6 Feature Fusion Test Suite...")
    tests = [
        ("test_handcrafted_sub_extractors", test_handcrafted_sub_extractors),
        ("test_dual_branch_fusion_pipeline", test_dual_branch_fusion_pipeline),
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
            test_fused_dataset_extraction_and_caching(Path(td))
            print("  [PASS] test_fused_dataset_extraction_and_caching")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_fused_dataset_extraction_and_caching: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
