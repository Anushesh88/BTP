"""
Frozen Vision Transformer (ViT) and CLIP feature extractor module.
Supports torchvision and timm backbones with automatic GPU acceleration,
parameter freezing, and embedding caching to disk.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset import MiniaturePaintingDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FrozenVisionBackbone(nn.Module):
    """
    Encapsulates a frozen ViT or CLIP visual backbone for feature extraction.
    Ensures all parameters have requires_grad=False and classifier heads are replaced with Identity.
    """

    def __init__(self, model_name: str = "vit_b_16", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name.lower()
        self.feature_dim = None

        logger.info(f"Initializing frozen vision backbone: {model_name} (pretrained={pretrained})")

        # 1. Try torchvision ViT models
        if "vit_b_16" in self.model_name:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            self.feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            self.model = backbone
        elif "vit_b_32" in self.model_name:
            weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
            backbone = models.vit_b_32(weights=weights)
            self.feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            self.model = backbone
        elif "vit_l_16" in self.model_name:
            weights = models.ViT_L_16_Weights.DEFAULT if pretrained else None
            backbone = models.vit_l_16(weights=weights)
            self.feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            self.model = backbone
        else:
            # 2. Try timm models (CLIP or alternative ViT)
            try:
                import timm
                backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                self.feature_dim = backbone.num_features
                self.model = backbone
            except Exception as e:
                logger.warning(f"Could not load via timm: {e}. Falling back to torchvision vit_b_16.")
                weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                backbone = models.vit_b_16(weights=weights)
                self.feature_dim = backbone.heads.head.in_features
                backbone.heads = nn.Identity()
                self.model = backbone

        # Freeze all backbone parameters completely
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"Frozen backbone initialized. Feature dimension: {self.feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x)
            # If shape is [B, tokens, D], pool tokens
            if len(features.shape) == 3:
                features = features.mean(dim=1)
        return features


def extract_embeddings_for_split(
    df: pd.DataFrame,
    split_name: str,
    model_name: str = "vit_b_16",
    cache_dir: Union[str, Path] = "data/cache",
    batch_size: int = 32,
    device: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extracts or loads cached embeddings for a given split (train/val/test).

    Returns:
        embeddings: np.ndarray of shape [N, feature_dim]
        df: pd.DataFrame with corresponding metadata
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = model_name.replace("/", "_")
    cache_file = cache_path / f"embeddings_{split_name}_{clean_model_name}.npz"

    if cache_file.exists() and not force_recompute:
        logger.info(f"Loading cached embeddings from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        embeddings = data["embeddings"]
        cached_ids = data["image_ids"]

        # Validate alignment
        if len(embeddings) == len(df) and (df["image_id"].values == cached_ids).all():
            logger.info(f"Loaded {len(embeddings)} cached embeddings of shape {embeddings.shape}.")
            return embeddings, df
        else:
            logger.warning("Cache size or IDs mismatch with current dataframe. Recomputing embeddings...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Extracting features using device: {device} on {len(df)} images...")

    backbone = FrozenVisionBackbone(model_name=model_name).to(device)
    dataset = MiniaturePaintingDataset(df=df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    all_features = []
    use_amp = (device == "cuda")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    feats = backbone(images)
            else:
                feats = backbone(images)

            all_features.append(feats.cpu().float().numpy())
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    embeddings = np.concatenate(all_features, axis=0)

    # Save to cache
    logger.info(f"Saving extracted embeddings to cache: {cache_file}")
    np.savez_compressed(
        cache_file,
        embeddings=embeddings,
        image_ids=df["image_id"].values,
        model_name=model_name,
    )

    return embeddings, df
