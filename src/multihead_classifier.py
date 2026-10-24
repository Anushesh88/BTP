"""
Phase 7: Multi-Head Classifier & Domain-Aware Training Module.
Trains separate heads for Raga, School, Period, and Artist on Dual-Branch Fused Features.
Includes domain-constrained color-safe augmentations, class weighting, and multi-task learning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ["raga", "school", "period", "artist"]

# Label index for missing targets (e.g. 'Unknown' period); ignored by the loss
IGNORE_INDEX = -100


class ColorSafeMiniatureAugmentation:
    """
    Domain-constrained data augmentation tailored for miniature paintings.
    Applies small geometric rotations and crops, with strictly conservative
    color jitter (max +/-5% brightness/contrast and +/-1% hue).
    CRITICAL: Never applies heavy hue shifts, color inversions, or extreme channel mixing.
    """

    def __init__(
        self,
        rotation_degrees: float = 10.0,
        crop_scale: Tuple[float, float] = (0.88, 1.0),
        brightness_jitter: float = 0.05,
        contrast_jitter: float = 0.05,
        saturation_jitter: float = 0.05,
        hue_jitter: float = 0.01,  # Strictly capped at 1% to protect pigment signals
    ):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ColorJitter(
                brightness=brightness_jitter,
                contrast=contrast_jitter,
                saturation=saturation_jitter,
                hue=hue_jitter,
            ),
        ])

    def __call__(self, img):
        return self.transform(img)


class MultiHeadMiniatureClassifier(nn.Module):
    """
    Multi-task Neural Classifier with a shared fused representation trunk
    and 4 dedicated classification heads:
    1. Head: Raga
    2. Head: School
    3. Head: Period
    4. Head: Artist
    """

    def __init__(
        self,
        input_dim: int,
        num_classes_dict: Dict[str, int],
        hidden_dim: int = 512,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes_dict = num_classes_dict
        self.hidden_dim = hidden_dim

        # Per-feature standardization fitted on the training split (saved with the checkpoint).
        # The fused vector concatenates separately L2-normalized blocks whose raw scales differ widely.
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        trunk_out_dim = hidden_dim // 2

        # 4 Separate Task Heads
        self.heads = nn.ModuleDict()
        for target, n_classes in num_classes_dict.items():
            self.heads[target] = nn.Sequential(
                nn.Linear(trunk_out_dim, trunk_out_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_p / 2.0),
                nn.Linear(trunk_out_dim // 2, n_classes),
            )

    def set_standardization(self, X_train: np.ndarray) -> None:
        self.feat_mean.copy_(torch.tensor(X_train.mean(axis=0), dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(X_train.std(axis=0) + 1e-6, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.trunk((x - self.feat_mean) / self.feat_std)
        logits = {}
        for target, head in self.heads.items():
            logits[target] = head(features)
        return logits


class MultiHeadTrainer:
    """
    Trainer for Multi-Head miniature classifier with class weighting,
    cosine annealing, and early stopping.
    """

    def __init__(
        self,
        input_dim: int,
        targets: Optional[List[str]] = None,
        hidden_dim: int = 512,
        dropout_p: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        task_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.targets = targets or DEFAULT_TARGETS
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.task_weights = task_weights or {t: 1.0 for t in self.targets}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.class_weights: Dict[str, torch.Tensor] = {}
        self.model: Optional[MultiHeadMiniatureClassifier] = None

    def fit_encoders_and_weights(self, df_train: pd.DataFrame) -> Dict[str, int]:
        """Fit label encoders and compute balanced class weights for all heads."""
        num_classes_dict = {}

        for target in self.targets:
            if target not in df_train.columns:
                continue

            y_raw = df_train[target].dropna().astype(str).values
            le = LabelEncoder()
            y_enc = le.fit_transform(y_raw)
            self.label_encoders[target] = le
            n_classes = len(le.classes_)
            num_classes_dict[target] = n_classes

            # Compute balanced class weights to counteract severe historical imbalance
            weights = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            self.class_weights[target] = weight_tensor

            logger.info(f"Target '{target}': {n_classes} classes. Balanced weights computed.")

        return num_classes_dict

    def train_model(
        self,
        X_train: np.ndarray,
        df_train: pd.DataFrame,
        X_val: np.ndarray,
        df_val: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the multi-head neural network using multi-task loss and early stopping.
        """
        num_classes_dict = self.fit_encoders_and_weights(df_train)

        # Initialize network
        self.model = MultiHeadMiniatureClassifier(
            input_dim=self.input_dim,
            num_classes_dict=num_classes_dict,
            hidden_dim=self.hidden_dim,
            dropout_p=self.dropout_p,
        ).to(self.device)
        self.model.set_standardization(X_train)

        # Prepare PyTorch Tensors
        # Missing train labels (NaN) are encoded as IGNORE_INDEX and skipped by the loss
        y_train_dict = {}
        for t in num_classes_dict:
            labelled = df_train[t].notna().values
            enc = np.full(len(df_train), IGNORE_INDEX, dtype=int)
            enc[labelled] = self.label_encoders[t].transform(df_train[t][labelled].astype(str))
            y_train_dict[t] = torch.tensor(enc, dtype=torch.long)

        # Handle val labels (mapping unseen or missing classes safely to 0 with mask)
        val_encoded = {}
        val_masks = {}
        for t in num_classes_dict:
            le = self.label_encoders[t]
            raw_val = df_val[t].astype(str).values
            known_mask = np.isin(raw_val, le.classes_) & df_val[t].notna().values
            enc = np.zeros(len(raw_val), dtype=int)
            enc[known_mask] = le.transform(raw_val[known_mask])
            val_encoded[t] = torch.tensor(enc, dtype=torch.long).to(self.device)
            val_masks[t] = torch.tensor(known_mask, dtype=torch.bool).to(self.device)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        # PyTorch DataLoader for training
        train_dataset = TensorDataset(X_train_t, *[y_train_dict[t] for t in num_classes_dict])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss functions per target with balanced class weights
        loss_fns = {
            t: nn.CrossEntropyLoss(weight=self.class_weights[t], ignore_index=IGNORE_INDEX)
            for t in num_classes_dict
        }

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state = None
        no_improve_epochs = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        target_keys = list(num_classes_dict.keys())

        logger.info(f"Starting Multi-Head training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_acc = 0.0

            for batch_data in train_loader:
                bx = batch_data[0].to(self.device)
                by_targets = {target_keys[i]: batch_data[i + 1].to(self.device) for i in range(len(target_keys))}

                optimizer.zero_grad()
                logits = self.model(bx)

                # Multi-task combined loss
                batch_loss = 0.0
                for t in target_keys:
                    if not (by_targets[t] != IGNORE_INDEX).any():
                        continue
                    loss_t = loss_fns[t](logits[t], by_targets[t])
                    batch_loss += self.task_weights.get(t, 1.0) * loss_t

                batch_loss.backward()
                optimizer.step()
                train_loss_acc += batch_loss.item()

            scheduler.step()
            train_loss = train_loss_acc / len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = 0.0
                val_accs = []

                for t in target_keys:
                    mask = val_masks[t]
                    if mask.sum() > 0:
                        l_t = loss_fns[t](val_logits[t][mask], val_encoded[t][mask])
                        val_loss += self.task_weights.get(t, 1.0) * l_t.item()

                        preds = torch.argmax(val_logits[t][mask], dim=1)
                        acc_t = (preds == val_encoded[t][mask]).float().mean().item()
                        val_accs.append(acc_t)

                avg_val_acc = float(np.mean(val_accs)) if val_accs else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(avg_val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {avg_val_acc*100:.2f}%"
                )

            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        logger.info("Multi-Head model training complete.")
        return history

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate class label predictions across all heads."""
        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict().")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)

        predictions = {}
        for target, logit in logits.items():
            pred_indices = torch.argmax(logit, dim=1).cpu().numpy()
            le = self.label_encoders[target]
            predictions[target] = le.inverse_transform(pred_indices)

        return predictions

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate class probabilities across all heads."""
        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict_proba().")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)

        probabilities = {}
        for target, logit in logits.items():
            probs = torch.softmax(logit, dim=1).cpu().numpy()
            probabilities[target] = probs

        return probabilities

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Serialize trained model weights and encoders."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        encoder_classes = {t: list(le.classes_) for t, le in self.label_encoders.items()}

        payload = {
            "model_state_dict": self.model.state_dict() if self.model else {},
            "input_dim": self.input_dim,
            "num_classes_dict": self.model.num_classes_dict if self.model else {},
            "hidden_dim": self.hidden_dim,
            "dropout_p": self.dropout_p,
            "targets": self.targets,
            "encoder_classes": encoder_classes,
            "task_weights": self.task_weights,
        }
        torch.save(payload, path)
        logger.info(f"Multi-Head checkpoint saved to: {path}")

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: Optional[str] = None) -> "MultiHeadTrainer":
        """Deserialize trained model weights and encoders."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            payload = torch.load(path, map_location="cpu")

        trainer = cls(
            input_dim=payload["input_dim"],
            targets=payload["targets"],
            hidden_dim=payload["hidden_dim"],
            dropout_p=payload["dropout_p"],
            task_weights=payload["task_weights"],
            device=device,
        )

        # Reconstruct label encoders
        for t, classes in payload["encoder_classes"].items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            trainer.label_encoders[t] = le

        trainer.model = MultiHeadMiniatureClassifier(
            input_dim=payload["input_dim"],
            num_classes_dict=payload["num_classes_dict"],
            hidden_dim=payload["hidden_dim"],
            dropout_p=payload["dropout_p"],
        ).to(trainer.device)

        # strict=False keeps older checkpoints (saved before input standardization existed) loadable
        missing, unexpected = trainer.model.load_state_dict(
            {k: v.to(trainer.device) for k, v in payload["model_state_dict"].items()}, strict=False
        )
        if set(missing) - {"feat_mean", "feat_std"} or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}, unexpected: {unexpected}")
        trainer.model.eval()
        logger.info(f"Multi-Head checkpoint successfully loaded from: {path}")
        return trainer
