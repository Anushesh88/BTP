"""
Phase 6: Dual-Branch Feature Engineering & Fusion Module.
Extracts handcrafted descriptors (HSV/Lab Color, GLCM/LBP/Gabor Texture, Stroke Morphology,
and Composition) and fuses them with Deep Vision Embeddings (Global + Aggregated Region Crops).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import torch
from torch.utils.data import DataLoader

from src.feature_extractor import FrozenVisionBackbone
from src.preprocessing import PaintingPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HandcraftedFeatureExtractor:
    """
    Extracts domain-tailored visual descriptors capturing color palettes,
    micro-textures, miniature brushstrokes, and compositional layouts.
    """

    def __init__(
        self,
        hsv_bins: Tuple[int, int, int] = (16, 8, 8),
        lbp_points: int = 8,
        lbp_radius: int = 1,
        gabor_frequencies: Tuple[float, ...] = (0.1, 0.25),
        gabor_angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    ):
        self.hsv_bins = hsv_bins
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.gabor_frequencies = gabor_frequencies
        self.gabor_angles = gabor_angles

        # Precompute Gabor filter kernels
        self.gabor_kernels = []
        for freq in self.gabor_frequencies:
            for theta in self.gabor_angles:
                kernel = cv2.getGaborKernel(
                    ksize=(15, 15),
                    sigma=3.0,
                    theta=theta,
                    lambd=1.0 / freq,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F,
                )
                self.gabor_kernels.append(kernel)

    def extract_color_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        4A.1 Color Extraction:
        - HSV histograms (Hue, Saturation, Value)
        - Lab color statistics (means, standard deviations)
        - Historical Gold Leaf / Mineral pigment intensity profile
        """
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # 1. Normalized HSV Histograms
        h_hist = cv2.calcHist([hsv], [0], None, [self.hsv_bins[0]], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [self.hsv_bins[1]], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [self.hsv_bins[2]], [0, 256]).flatten()

        h_hist /= (h_hist.sum() + 1e-7)
        s_hist /= (s_hist.sum() + 1e-7)
        v_hist /= (v_hist.sum() + 1e-7)

        # 2. Lab space moments
        lab_means = np.mean(lab, axis=(0, 1)) / 255.0
        lab_stds = np.std(lab, axis=(0, 1)) / 255.0

        # 3. Gold / Yellow Mineral Pigment Index (high Saturation, warm Hue ~20-40, high Value)
        # Identifies gold leaf application common in Mughal and Deccani miniatures
        hue_chan = hsv[:, :, 0]
        sat_chan = hsv[:, :, 1]
        val_chan = hsv[:, :, 2]
        gold_mask = (hue_chan >= 15) & (hue_chan <= 35) & (sat_chan >= 100) & (val_chan >= 150)
        gold_fraction = np.mean(gold_mask)

        # High-saturation mineral red (lac / hinglu in Mewar style: Hue 0-10 or 170-180, high S, V)
        red_mask = ((hue_chan <= 10) | (hue_chan >= 170)) & (sat_chan >= 120) & (val_chan >= 100)
        red_fraction = np.mean(red_mask)

        color_vec = np.concatenate([
            h_hist, s_hist, v_hist,
            lab_means, lab_stds,
            [gold_fraction, red_fraction]
        ])
        return color_vec.astype(np.float32)

    def extract_texture_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        4A.2 Texture Extraction:
        - Gray-Level Co-occurrence Matrix (GLCM: contrast, dissimilarity, homogeneity, energy, correlation)
        - Local Binary Patterns (LBP) histogram capturing wasli paper grain and paint stippling
        - Gabor filter energy responses across multi-scale orientations
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Resize small copy for fast GLCM computation while preserving texture
        gray_sub = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

        # 1. GLCM features at 4 canonical angles (0, 45, 90, 135 deg)
        glcm = graycomatrix(
            gray_sub, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256, symmetric=True, normed=True
        )

        contrast = graycoprops(glcm, "contrast").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        energy = graycoprops(glcm, "energy").mean()
        correlation = graycoprops(glcm, "correlation").mean()
        glcm_props = np.array([contrast / 1000.0, dissimilarity / 100.0, homogeneity, energy, correlation])

        # 2. Local Binary Patterns (LBP)
        lbp = local_binary_pattern(gray_sub, self.lbp_points, self.lbp_radius, method="uniform")
        n_bins = self.lbp_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        # 3. Gabor Filter responses
        gabor_feats = []
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(gray_sub, cv2.CV_32F, kernel)
            gabor_feats.append(np.mean(np.abs(filtered)) / 255.0)
            gabor_feats.append(np.std(filtered) / 255.0)

        texture_vec = np.concatenate([glcm_props, lbp_hist, gabor_feats])
        return texture_vec.astype(np.float32)

    def extract_stroke_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        4A.3 Stroke & Line Morphology:
        - Canny edge orientation distribution (horizontal, diagonal, vertical)
        - Edge density and contour curvature
        - Mean stroke/edge segment length
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = np.mean(edges > 0)

        # Sobel gradients to compute edge orientations
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Mask orientations only at edge pixels
        edge_angles = angle[edges > 0]
        if len(edge_angles) > 0:
            ang_hist, _ = np.histogram(edge_angles, bins=8, range=(0, 360), density=True)
        else:
            ang_hist = np.zeros(8, dtype=np.float32)

        # Contours to measure stroke morphology
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            lengths = [cv2.arcLength(c, False) for c in contours]
            mean_length = np.mean(lengths) / (max(gray.shape) + 1e-7)
            max_length = np.max(lengths) / (max(gray.shape) + 1e-7)
            contour_count = min(len(contours), 500) / 500.0
        else:
            mean_length, max_length, contour_count = 0.0, 0.0, 0.0

        stroke_vec = np.concatenate([
            [edge_density, mean_length, max_length, contour_count],
            ang_hist
        ])
        return stroke_vec.astype(np.float32)

    def extract_composition_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        4A.5 Composition & Spatial Layout:
        - Rule-of-thirds 3x3 energy distribution (9 cells)
        - Horizontal and Vertical symmetry profiles
        - Visual center of mass / focal point deviation from canvas center
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape

        # 1. 3x3 Grid Energy Distribution (Rule of Thirds)
        h_third = h // 3
        w_third = w // 3
        grid_energies = []
        for r in range(3):
            for c in range(3):
                cell = gray[r*h_third : (r+1)*h_third, c*w_third : (c+1)*w_third]
                grid_energies.append(np.mean(cell))
        grid_energies = np.array(grid_energies, dtype=np.float32)
        grid_energies /= (grid_energies.sum() + 1e-7)

        # 2. Symmetry Scores (difference between image and flipped counterpart)
        flipped_h = np.fliplr(gray)
        flipped_v = np.flipud(gray)
        h_symmetry = 1.0 - np.mean(np.abs(gray - flipped_h))
        v_symmetry = 1.0 - np.mean(np.abs(gray - flipped_v))

        # 3. Center of Mass / Focal Point
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        total_mass = np.sum(gray) + 1e-7
        cy = np.sum(y_coords * gray) / total_mass
        cx = np.sum(x_coords * gray) / total_mass

        # Normalize offset from canvas center [-0.5, +0.5]
        offset_x = (cx - (w / 2.0)) / w
        offset_y = (cy - (h / 2.0)) / h

        comp_vec = np.concatenate([
            grid_energies,
            [h_symmetry, v_symmetry, offset_x, offset_y]
        ])
        return comp_vec.astype(np.float32)

    def extract_all(self, img_rgb: np.ndarray) -> np.ndarray:
        """Extract full concatenated handcrafted feature vector."""
        c = self.extract_color_features(img_rgb)
        t = self.extract_texture_features(img_rgb)
        s = self.extract_stroke_features(img_rgb)
        l = self.extract_composition_features(img_rgb)
        return np.concatenate([c, t, s, l]).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Determine total handcrafted feature dimension dynamically."""
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        return len(self.extract_all(dummy))


class DualBranchFusionPipeline:
    """
    Integrates Deep Vision features (Global ViT/CLIP + Aggregated Region Crops)
    with Handcrafted features into a unified fused representation.
    """

    def __init__(
        self,
        backbone_model_name: str = "vit_b_16",
        device: Optional[str] = None,
        max_side: Optional[int] = 1024,
    ):
        self.backbone_name = backbone_model_name
        self.max_side = max_side
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = FrozenVisionBackbone(model_name=backbone_model_name).to(self.device)
        self.handcrafted = HandcraftedFeatureExtractor()
        self.preprocessor = PaintingPreprocessor(denoise_method="none")

        # Standard ViT/CLIP tensor transform
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        logger.info(
            f"DualBranchFusionPipeline ready: Deep dim={self.backbone.feature_dim}, "
            f"Handcrafted dim={self.handcrafted.feature_dim}"
        )

    def extract_image_features(
        self,
        img_rgb: np.ndarray,
        crop_images: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Extracts fused feature vector for a single painting:
        1. Deep Global Embedding (D_deep)
        2. Deep Region Aggregated Embedding via Mean/Max Pooling (D_deep)
        3. Handcrafted Descriptors (D_handcrafted)
        """
        # 1. Global Deep feature
        pil_img = Image.fromarray(img_rgb)
        tensor_global = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_global = self.backbone(tensor_global).cpu().squeeze(0).numpy()

        # 2. Region Crops Deep feature aggregation
        if crop_images and len(crop_images) > 0:
            crop_tensors = []
            for c_arr in crop_images:
                c_pil = Image.fromarray(c_arr)
                crop_tensors.append(self.transform(c_pil))
            batch_crops = torch.stack(crop_tensors).to(self.device)

            with torch.no_grad():
                crop_feats = self.backbone(batch_crops).cpu().numpy()

            # Mean-pool across region features
            feat_regions = np.mean(crop_feats, axis=0)
        else:
            # If no crops detected, zero-fill region branch
            feat_regions = np.zeros_like(feat_global)

        # 3. Handcrafted Descriptors
        feat_hand = self.handcrafted.extract_all(img_rgb)

        # L2-normalize deep and handcrafted branches before concatenation
        feat_global_norm = feat_global / (np.linalg.norm(feat_global) + 1e-7)
        feat_regions_norm = feat_regions / (np.linalg.norm(feat_regions) + 1e-7) if np.linalg.norm(feat_regions) > 0 else feat_regions
        feat_hand_norm = feat_hand / (np.linalg.norm(feat_hand) + 1e-7)

        # Fused unified feature representation
        fused = np.concatenate([feat_global_norm, feat_regions_norm, feat_hand_norm])
        return fused.astype(np.float32)

    def extract_dataset_fused_features(
        self,
        df: pd.DataFrame,
        motifs_manifest_path: Optional[Union[str, Path]] = None,
        cache_file: Optional[Union[str, Path]] = None,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extracts or loads fused features for all paintings in a dataframe.
        """
        if cache_file and Path(cache_file).exists() and not force_recompute:
            cached = np.load(cache_file, allow_pickle=True)
            cached_ids = list(cached["image_ids"]) if "image_ids" in cached else None
            if cached_ids == list(df["image_id"].values):
                logger.info(f"Loading cached fused features from: {cache_file}")
                return cached["fused_features"], df
            logger.warning(f"Fused cache {cache_file} does not match the requested paintings. Recomputing...")

        # Load motif crops map by parent_image_id
        crops_by_parent = {}
        if motifs_manifest_path and Path(motifs_manifest_path).exists():
            with open(motifs_manifest_path, "r", encoding="utf-8") as jf:
                for line in jf:
                    if line.strip():
                        rec = json.loads(line.strip())
                        pid = rec.get("parent_image_id")
                        c_path = rec.get("crop_path")
                        if pid and c_path and Path(c_path).exists():
                            if pid not in crops_by_parent:
                                crops_by_parent[pid] = []
                            crops_by_parent[pid].append(c_path)

        all_fused = []
        logger.info(f"Extracting dual-branch fused features on {len(df)} paintings...")

        for idx, (_, row) in enumerate(df.iterrows()):
            img_id = row["image_id"]
            img_p = row["file_path"]

            img_rgb = self.preprocessor.load_and_convert_rgb(img_p)
            # Normalize scale: cap the longest side so handcrafted descriptors are comparable
            # across paintings (dataset ranges from ~100px to ~10,000px) and tractable to compute
            h, w = img_rgb.shape[:2]
            if self.max_side and max(h, w) > self.max_side:
                s = self.max_side / max(h, w)
                img_rgb = cv2.resize(img_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

            # Load crops if any
            crop_arrays = []
            if img_id in crops_by_parent:
                for cp in crops_by_parent[img_id][:6]:  # Limit top 6 crops for pooling
                    try:
                        c_img = Image.open(cp).convert("RGB")
                        crop_arrays.append(np.array(c_img))
                    except Exception:
                        pass

            fused_vec = self.extract_image_features(img_rgb, crop_images=crop_arrays)
            all_fused.append(fused_vec)

            if (idx + 1) % 5 == 0 or (idx + 1) == len(df):
                logger.info(f"Extracted [{idx+1}/{len(df)}] paintings (fused dim: {len(fused_vec)}).")

        fused_matrix = np.array(all_fused, dtype=np.float32)

        if cache_file:
            cache_p = Path(cache_file)
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_p, fused_features=fused_matrix, image_ids=df["image_id"].values)
            logger.info(f"Saved fused feature matrix of shape {fused_matrix.shape} to {cache_file}")

        return fused_matrix, df
