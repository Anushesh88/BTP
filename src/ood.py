"""
Out-of-distribution gate: "is this an Indian miniature painting at all?"

The classifier heads are closed-set: given a dog photo they still have to pick
Mughal / Rajasthani / Pahari, and the majority class (Mughal) wins with high
confidence. Calibration cannot fix that; it only calibrates in-distribution images.

The gate combines two independent checks, and both must pass:
  1. Nearest-painting similarity: mean cosine similarity of the image's frozen
     ViT-B/16 embedding to its 5 nearest training paintings.
  2. CLIP zero-shot: probability that "an Indian / Mughal or Rajput miniature
     painting" describes the image better than photo, drawing, document, etc.

Thresholds were set from data: every one of the 169 val + test paintings (unseen
albums) passes both (lowest 0.305 similarity, 0.588 CLIP), while 12 scikit-image
sample photos and documents score at most 0.189 and 0.033.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.feature_extractor import FrozenVisionBackbone

logger = logging.getLogger(__name__)

MINIATURE_PROMPTS = ["an Indian miniature painting", "a Mughal or Rajput miniature painting"]
OTHER_PROMPTS = [
    "a photograph", "a photo of an animal", "a photo of a person", "a modern painting",
    "a drawing or sketch", "a screenshot or document", "a cartoon",
]


@dataclass
class GateResult:
    is_miniature: bool
    knn_similarity: float
    clip_miniature_prob: float
    reason: str


class MiniatureGate:
    def __init__(
        self,
        train_embeddings_file: Union[str, Path] = "data/cache/embeddings_train_vit_b_16.npz",
        clip_dir: Union[str, Path] = "weights/clip",
        knn_threshold: float = 0.25,
        clip_threshold: float = 0.50,
        k: int = 5,
        device: str = None,
    ):
        import clip  # ultralytics' CLIP package (see requirements.txt)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.knn_threshold = knn_threshold
        self.clip_threshold = clip_threshold
        self.k = k

        train = np.load(train_embeddings_file)["embeddings"].astype(np.float32)
        self.train = train / np.linalg.norm(train, axis=1, keepdims=True)

        self.vit = FrozenVisionBackbone("vit_b_16").to(self.device)
        self.vit_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])

        self.clip_model, self.clip_pre = clip.load("ViT-B/32", download_root=str(clip_dir), device=self.device)
        with torch.no_grad():
            t = self.clip_model.encode_text(clip.tokenize(MINIATURE_PROMPTS + OTHER_PROMPTS).to(self.device)).float()
        self.text = t / t.norm(dim=-1, keepdim=True)

    def check(self, image: Union[str, Path, Image.Image]) -> GateResult:
        img = (Image.open(image) if not isinstance(image, Image.Image) else image).convert("RGB")
        with torch.no_grad():
            emb = self.vit(self.vit_tf(img).unsqueeze(0).to(self.device)).cpu().numpy()[0]
            c = self.clip_model.encode_image(self.clip_pre(img).unsqueeze(0).to(self.device)).float()
            c = c / c.norm(dim=-1, keepdim=True)
            probs = (100 * c @ self.text.T).softmax(-1)[0].cpu().numpy()

        emb = emb / np.linalg.norm(emb)
        knn = float(np.sort(self.train @ emb)[-self.k:].mean())
        clip_p = float(probs[: len(MINIATURE_PROMPTS)].sum())

        failed = []
        if knn < self.knn_threshold:
            failed.append(f"unlike any training painting (similarity {knn:.2f} < {self.knn_threshold})")
        if clip_p < self.clip_threshold:
            failed.append(f"CLIP does not see a miniature painting (p={clip_p:.2f} < {self.clip_threshold})")
        return GateResult(not failed, knn, clip_p, "; ".join(failed) or "looks like an Indian miniature painting")
