"""
Unit and integration tests for Phase 4:
High-Resolution Preprocessing & Sliding Window Generation.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.preprocessing import PaintingPreprocessor, SlidingWindowGenerator


def test_native_resolution_and_format_conversion():
    preprocessor = PaintingPreprocessor(denoise_method="none")

    # 1. Test 800x600 RGB array
    img_rgb = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    proc = preprocessor.process(img_rgb)
    assert proc.shape == (600, 800, 3), "Native resolution must be preserved!"

    # 2. Test RGBA conversion
    img_rgba = np.random.randint(0, 256, (400, 500, 4), dtype=np.uint8)
    proc_rgba = preprocessor.load_and_convert_rgb(img_rgba)
    assert proc_rgba.shape == (400, 500, 3)

    # 3. Test Grayscale conversion
    img_gray = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    proc_gray = preprocessor.load_and_convert_rgb(img_gray)
    assert proc_gray.shape == (300, 300, 3)


def test_preprocessing_filters():
    # Test on synthetic gradient with simulated scan noise
    x = np.linspace(0, 255, 256, dtype=np.uint8)
    base_grad = np.tile(x, (256, 1))
    noise = np.random.randint(-15, 15, (256, 256))
    noisy = np.clip(base_grad.astype(int) + noise, 0, 255).astype(np.uint8)
    noisy_rgb = np.stack([noisy, noisy, noisy], axis=-1)

    preprocessor = PaintingPreprocessor(
        denoise_method="median",
        apply_white_balance=True,
        apply_clahe=True,
        gamma=1.1,
        apply_unsharp_mask=True,
    )

    enhanced = preprocessor.process(noisy_rgb)

    assert enhanced.shape == noisy_rgb.shape
    assert enhanced.dtype == np.uint8
    assert not np.isnan(enhanced).any()
    assert enhanced.min() >= 0 and enhanced.max() <= 255


def test_sliding_window_generation_and_padding():
    # Test on canvas of dimension 700x900 (not exact multiple of 256/512)
    canvas = np.zeros((700, 900, 3), dtype=np.uint8)

    window_gen = SlidingWindowGenerator(
        window_size=(512, 512),
        stride=(256, 256),
        padding_mode="reflect",
    )

    windows = window_gen.extract_windows(canvas)
    assert len(windows) > 0

    for win_img, meta in windows:
        # Every window crop must strictly match requested window_size
        assert win_img.shape == (512, 512, 3), f"Window shape {win_img.shape} != 512x512"
        gx1, gy1, gx2, gy2 = meta["global_box"]

        # Global box must remain inside original canvas bounds
        assert 0 <= gx1 <= 900
        assert 0 <= gx2 <= 900
        assert 0 <= gy1 <= 700
        assert 0 <= gy2 <= 700

    # Test on smaller canvas (e.g. 256x256 smaller than 512x512 window)
    small_canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    small_windows = window_gen.extract_windows(small_canvas)
    assert len(small_windows) == 1
    assert small_windows[0][0].shape == (512, 512, 3)
    assert small_windows[0][1]["is_padded"] is True


def test_coordinate_mapping_to_global():
    window_gen = SlidingWindowGenerator(window_size=(512, 512), stride=(256, 256))

    window_meta = {
        "x": 256,
        "y": 512,
        "width": 512,
        "height": 512,
        "orig_dims": [1000, 1200],  # W=1000, H=1200
    }

    # Bounding box inside local window coordinates: [100, 50, 200, 150]
    local_box = [100.0, 50.0, 200.0, 150.0]
    global_box = window_gen.map_box_to_global(local_box, window_meta)

    # Expected: x + 256, y + 512 -> [356, 562, 456, 662]
    assert global_box == [356.0, 562.0, 456.0, 662.0]

    # Test edge boundary clipping: local box extending past original canvas width
    local_box_overflow = [700.0, 50.0, 800.0, 150.0]  # 700+256 = 956, 800+256=1056 > 1000
    clipped_global = window_gen.map_box_to_global(local_box_overflow, window_meta)
    assert clipped_global[2] <= 1000.0, "Coordinate must be clipped to original canvas width!"


if __name__ == "__main__":
    import traceback

    print("Running Phase 4 Preprocessing & Sliding Window Test Suite...")
    tests = [
        ("test_native_resolution_and_format_conversion", test_native_resolution_and_format_conversion),
        ("test_preprocessing_filters", test_preprocessing_filters),
        ("test_sliding_window_generation_and_padding", test_sliding_window_generation_and_padding),
        ("test_coordinate_mapping_to_global", test_coordinate_mapping_to_global),
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

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
