"""
High-Resolution Preprocessing & Sliding Window Generation for Miniature Paintings.
Preserves original canvas resolution, applies edge-preserving denoising,
color/lighting normalization, unsharp masking, and extracts overlapping sliding windows.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PaintingPreprocessor:
    """
    Image enhancement preprocessor tailored for Indian miniature paintings.
    Preserves fine micro-details, brushwork, and pigments while reducing scan/paper noise.
    """

    def __init__(
        self,
        denoise_method: str = "nlmeans",  # 'nlmeans', 'median', or 'none'
        denoise_h: float = 3.0,          # filter strength for NL-means luminance
        denoise_h_color: float = 3.0,    # filter strength for NL-means color
        median_kernel: int = 3,
        apply_white_balance: bool = True,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 1.05,
        apply_unsharp_mask: bool = True,
        unsharp_sigma: float = 1.0,
        unsharp_strength: float = 1.2,
    ):
        self.denoise_method = denoise_method.lower()
        self.denoise_h = denoise_h
        self.denoise_h_color = denoise_h_color
        self.median_kernel = median_kernel
        self.apply_white_balance = apply_white_balance
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.gamma = gamma
        self.apply_unsharp_mask = apply_unsharp_mask
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_strength = unsharp_strength

    def load_and_convert_rgb(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads image without downsampling and ensures 8-bit 3-channel RGB format.
        Handles RGBA, Grayscale, CMYK, Palette-indexed images.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at: {path}")
            # Use PIL first to handle various modes safely, then convert to RGB numpy array
            pil_img = Image.open(path).convert("RGB")
            img_rgb = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:  # Grayscale
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:  # RGBA
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                img_rgb = image_input.copy()
            else:
                raise ValueError(f"Unexpected image array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return img_rgb

    def denoise(self, img_rgb: np.ndarray) -> np.ndarray:
        """Apply noise reduction while preserving brushwork and delicate contours."""
        if self.denoise_method == "nlmeans":
            # Non-local means denoising for colored images
            denoised = cv2.fastNlMeansDenoisingColored(
                src=img_rgb,
                dst=None,
                h=self.denoise_h,
                hColor=self.denoise_h_color,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            return denoised
        elif self.denoise_method == "median":
            k = self.median_kernel if self.median_kernel % 2 == 1 else self.median_kernel + 1
            return cv2.medianBlur(img_rgb, k)
        return img_rgb

    def normalize_white_balance(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Color balance normalization using modified Gray World assumption.
        Carefully tuned to avoid stripping the natural warm patina of antique wasli paper.
        """
        img_float = img_rgb.astype(np.float32)
        mean_r = np.mean(img_float[:, :, 0])
        mean_g = np.mean(img_float[:, :, 1])
        mean_b = np.mean(img_float[:, :, 2])
        mean_gray = (mean_r + mean_g + mean_b) / 3.0

        if mean_r > 0 and mean_g > 0 and mean_b > 0:
            scale_r = mean_gray / mean_r
            scale_g = mean_gray / mean_g
            scale_b = mean_gray / mean_b

            # Conservative scaling: blend 70% original, 30% normalized to preserve pigment fidelity
            scale_r = 0.7 + 0.3 * scale_r
            scale_g = 0.7 + 0.3 * scale_g
            scale_b = 0.7 + 0.3 * scale_b

            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * scale_r, 0, 255)
            img_float[:, :, 1] = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * scale_b, 0, 255)

        return img_float.astype(np.uint8)

    def adjust_lighting_and_contrast(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Enhance local lighting and contrast using LAB space CLAHE and gamma correction.
        """
        # Convert to LAB color space to operate strictly on the luminance channel (L)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        if self.apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size,
            )
            l_chan = clahe.apply(l_chan)

        # Gamma correction on luminance channel
        if abs(self.gamma - 1.0) > 1e-3:
            inv_gamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l_chan = cv2.LUT(l_chan, table)

        lab_merged = cv2.merge((l_chan, a_chan, b_chan))
        return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)

    def unsharp_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking to sharpen micro-contours and miniature brush details.
        I_sharp = I + strength * (I - I_blurred)
        """
        # Gaussian blur for high-frequency extraction
        blurred = cv2.GaussianBlur(img_rgb, (0, 0), self.unsharp_sigma)
        sharpened = cv2.addWeighted(
            img_rgb, 1.0 + self.unsharp_strength,
            blurred, -self.unsharp_strength,
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def process(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Execute full end-to-end preprocessing pipeline on a painting:
        1. RGB conversion & preservation of native resolution
        2. Edge-preserving Denoising
        3. Color / White-Balance Normalization
        4. Lighting & Contrast Optimization (CLAHE + Gamma)
        5. Unsharp Masking
        """
        img = self.load_and_convert_rgb(image_input)

        if self.denoise_method != "none":
            img = self.denoise(img)

        if self.apply_white_balance:
            img = self.normalize_white_balance(img)

        if self.apply_clahe or abs(self.gamma - 1.0) > 1e-3:
            img = self.adjust_lighting_and_contrast(img)

        if self.apply_unsharp_mask:
            img = self.unsharp_mask(img)

        return img


class SlidingWindowGenerator:
    """
    Divides high-resolution painting canvases into overlapping windows
    to inspect micro-motifs without lossy global resizing.
    Maintains exact coordinate mapping back to the full canvas.
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (512, 512),
        stride: Tuple[int, int] = (256, 256),
        padding_mode: str = "reflect",  # 'reflect', 'edge', 'constant'
    ):
        self.window_w, self.window_h = window_size
        self.stride_x, self.stride_y = stride
        self.padding_mode = padding_mode

        overlap_x = (self.window_w - self.stride_x) / self.window_w
        overlap_y = (self.window_h - self.stride_y) / self.window_h
        logger.info(
            f"SlidingWindowGenerator initialized: window={self.window_w}x{self.window_h}, "
            f"stride={self.stride_x}x{self.stride_y} (Overlap: X={overlap_x*100:.1f}%, Y={overlap_y*100:.1f}%)"
        )

    def _pad_image(self, img: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Ensure canvas dimensions are covered by integer number of window strides.
        Handles canvases smaller than window size as well as non-stride-aligned canvases.
        Returns padded_img, pad_bottom, pad_right.
        """
        h, w = img.shape[:2]

        pad_bottom = 0
        pad_right = 0

        # Handle width padding
        if w < self.window_w:
            pad_right = self.window_w - w
        else:
            rem_x = (w - self.window_w) % self.stride_x
            if rem_x != 0:
                pad_right = self.stride_x - rem_x

        # Handle height padding
        if h < self.window_h:
            pad_bottom = self.window_h - h
        else:
            rem_y = (h - self.window_h) % self.stride_y
            if rem_y != 0:
                pad_bottom = self.stride_y - rem_y

        if pad_bottom == 0 and pad_right == 0:
            return img, 0, 0

        border_type = cv2.BORDER_REFLECT_101
        if self.padding_mode == "edge":
            border_type = cv2.BORDER_REPLICATE
        elif self.padding_mode == "constant":
            border_type = cv2.BORDER_CONSTANT

        padded = cv2.copyMakeBorder(
            img, 0, pad_bottom, 0, pad_right, border_type, value=[0, 0, 0]
        )
        return padded, pad_bottom, pad_right

    def extract_windows(
        self, img: np.ndarray
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extract all overlapping windows from a painting image.

        Returns list of tuples: (window_img_array, window_metadata_dict)
        where window_metadata_dict contains:
          - window_id: sequential index
          - global_x1, global_y1, global_x2, global_y2: coordinate bounding box in original image
          - orig_width, orig_height
          - is_padded: boolean indicating if window overlaps border padding
        """
        orig_h, orig_w = img.shape[:2]
        padded_img, pad_bottom, pad_right = self._pad_image(img)
        pad_h, pad_w = padded_img.shape[:2]

        windows = []
        win_id = 0

        for y in range(0, pad_h - self.window_h + 1, self.stride_y):
            for x in range(0, pad_w - self.window_w + 1, self.stride_x):
                window_crop = padded_img[y : y + self.window_h, x : x + self.window_w]

                # Bounding box in original unpadded image coordinate space
                gx1 = x
                gy1 = y
                gx2 = min(x + self.window_w, orig_w)
                gy2 = min(y + self.window_h, orig_h)
                is_padded = (x + self.window_w > orig_w) or (y + self.window_h > orig_h)

                metadata = {
                    "window_id": win_id,
                    "x": int(x),
                    "y": int(y),
                    "width": int(self.window_w),
                    "height": int(self.window_h),
                    "global_box": [int(gx1), int(gy1), int(gx2), int(gy2)],
                    "orig_dims": [int(orig_w), int(orig_h)],
                    "is_padded": bool(is_padded),
                }

                windows.append((window_crop, metadata))
                win_id += 1

        return windows

    @staticmethod
    def map_box_to_global(box_xyxy: List[float], window_metadata: Dict) -> List[float]:
        """
        Maps a bounding box detected inside a local window [lx1, ly1, lx2, ly2]
        back to the original global painting canvas coordinates.
        Clips to original image dimensions.
        """
        win_x = window_metadata["x"]
        win_y = window_metadata["y"]
        orig_w, orig_h = window_metadata["orig_dims"]

        gx1 = max(0.0, min(float(orig_w), box_xyxy[0] + win_x))
        gy1 = max(0.0, min(float(orig_h), box_xyxy[1] + win_y))
        gx2 = max(0.0, min(float(orig_w), box_xyxy[2] + win_x))
        gy2 = max(0.0, min(float(orig_h), box_xyxy[3] + win_y))

        return [gx1, gy1, gx2, gy2]
