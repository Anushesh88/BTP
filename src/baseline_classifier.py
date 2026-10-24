"""
Baseline Multi-Head Classifier module.
Trains separate regularized Logistic Regression heads for Raga, School, Period, and Artist
using frozen ViT/CLIP embeddings.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ["raga", "school", "period", "artist"]


class BaselineMultiHeadClassifier:
    """
    Manages separate Logistic Regression heads for multiple targets
    (e.g., raga, school, period, artist) with class weighting and scaling.
    """

    def __init__(
        self,
        targets: Optional[List[str]] = None,
        c_param: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.targets = targets or DEFAULT_TARGETS
        self.c_param = c_param
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.models: Dict[str, LogisticRegression] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        df_train: pd.DataFrame,
    ) -> "BaselineMultiHeadClassifier":
        """
        Fit feature scaler and individual classification heads for each target.
        """
        logger.info(f"Fitting Baseline Multi-Head Classifier on {X_train.shape[0]} samples with feature dim {X_train.shape[1]}")

        # Fit feature standardizer
        X_scaled = self.scaler.fit_transform(X_train)

        for target in self.targets:
            if target not in df_train.columns:
                logger.warning(f"Target '{target}' not present in training dataframe. Skipping.")
                continue

            # Rows with a missing label (NaN) are excluded from this head only
            labelled = df_train[target].notna().values
            y_raw = df_train[target][labelled].astype(str).values
            unique_classes = np.unique(y_raw)

            if len(unique_classes) < 2:
                logger.warning(f"Target '{target}' has only {len(unique_classes)} class ({unique_classes}). Cannot fit classifier.")
                continue

            logger.info(f"Training head for '{target}' ({len(unique_classes)} classes)...")

            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_raw)
            self.label_encoders[target] = le

            # Train Logistic Regression head with balanced class weights
            clf = LogisticRegression(
                C=self.c_param,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X_scaled[labelled], y_encoded)
            self.models[target] = clf

            train_acc = clf.score(X_scaled[labelled], y_encoded)
            logger.info(f"Head '{target}' trained successfully. Train Accuracy: {train_acc:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate class label predictions for all targets.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict() is called.")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        for target, model in self.models.items():
            le = self.label_encoders[target]
            y_pred_encoded = model.predict(X_scaled)
            predictions[target] = le.inverse_transform(y_pred_encoded)

        return predictions

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate class probabilities for all targets.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba() is called.")

        X_scaled = self.scaler.transform(X)
        probabilities = {}

        for target, model in self.models.items():
            probabilities[target] = model.predict_proba(X_scaled)

        return probabilities

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted classifier bundle to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "targets": self.targets,
            "models": self.models,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "c_param": self.c_param,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path)
        logger.info(f"Baseline classifier saved to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaselineMultiHeadClassifier":
        """Deserialize fitted classifier bundle from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        instance = cls(targets=payload["targets"], c_param=payload["c_param"])
        instance.models = payload["models"]
        instance.label_encoders = payload["label_encoders"]
        instance.scaler = payload["scaler"]
        instance.is_fitted = payload["is_fitted"]
        logger.info(f"Baseline classifier loaded from {path}")
        return instance
