"""
Phase 8: Model Calibration & Uncertainty Estimation.
Implements Post-Hoc Temperature Scaling, Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and Reliability Diagrams.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SingleHeadTemperatureScaler(nn.Module):
    """
    Learns an optimal post-hoc temperature parameter T > 0 on validation set logits
    by minimizing Negative Log-Likelihood (NLL).
    Ensures that a model confidence of e.g. 70% corresponds empirically to ~70% accuracy.
    """

    def __init__(self, initial_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by 1/T."""
        # Enforce positive temperature on matching device
        temp = torch.clamp(self.temperature, min=0.01).to(logits.device)
        return logits / temp

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature by minimizing validation NLL with a bounded 1-D search over log(T).
        (Unconstrained L-BFGS on T can diverge to the clamp on small validation sets.)
        `lr` / `max_iter` are kept for API compatibility.
        """
        from scipy.optimize import minimize_scalar

        logits = val_logits.detach().float().cpu()
        labels = val_labels.detach().long().cpu()
        nll_criterion = nn.CrossEntropyLoss()

        def nll(log_t: float) -> float:
            return float(nll_criterion(logits / np.exp(log_t), labels))

        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        with torch.no_grad():
            self.temperature.fill_(float(np.exp(res.x)))
        self.to(val_logits.device)
        self.eval()

        learned_t = float(torch.clamp(self.temperature, min=0.01).item())
        logger.info(f"Learned optimal temperature T = {learned_t:.4f}")
        return learned_t


class MultiHeadTemperatureScaler:
    """
    Manages separate post-hoc temperature scaling parameters across all 4 classification heads:
    Raga, School, Period, and Artist.
    """

    def __init__(self, targets: Optional[List[str]] = None):
        self.targets = targets or ["raga", "school", "period", "artist"]
        self.scalers: Dict[str, SingleHeadTemperatureScaler] = {
            t: SingleHeadTemperatureScaler() for t in self.targets
        }
        self.temperatures: Dict[str, float] = {t: 1.0 for t in self.targets}

    def fit_validation_logits(
        self,
        val_logits_dict: Dict[str, torch.Tensor],
        val_labels_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Fit temperature scaling on the validation set for each head independently.
        """
        for target in self.targets:
            if target not in val_logits_dict or target not in val_labels_dict:
                continue

            v_logits = val_logits_dict[target]
            v_labels = val_labels_dict[target]

            if len(v_logits) == 0:
                continue

            logger.info(f"Fitting temperature scaling for head: '{target}' ({len(v_logits)} samples)...")
            scaler = self.scalers[target]
            scaler.to(v_logits.device)
            learned_t = scaler.fit(val_logits=v_logits, val_labels=v_labels)
            self.temperatures[target] = learned_t

        return self.temperatures

    def scale_logits(
        self, logits_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Applies learned temperature scaling to logits across all heads."""
        scaled = {}
        for target, logits in logits_dict.items():
            if target in self.scalers:
                scaler = self.scalers[target]
                with torch.no_grad():
                    scaled[target] = scaler(logits)
            else:
                scaled[target] = logits
        return scaled

    def scale_probabilities(
        self, logits_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Returns calibrated probabilities for all heads."""
        scaled_logits = self.scale_logits(logits_dict)
        calibrated_probs = {}
        for target, s_logit in scaled_logits.items():
            probs = torch.softmax(s_logit, dim=1).cpu().numpy()
            calibrated_probs[target] = probs
        return calibrated_probs

    def save(self, filepath: Union[str, Path]) -> None:
        """Save learned temperatures to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.temperatures, f, indent=2)
        logger.info(f"Saved temperature scalers to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MultiHeadTemperatureScaler":
        """Load learned temperatures from JSON."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(targets=list(data.keys()))
        for t, temp_val in data.items():
            instance.temperatures[t] = float(temp_val)
            instance.scalers[t].temperature.data.fill_(float(temp_val))
        return instance


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Union[float, List]]:
    """
    Computes rigorous uncertainty metrics:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Multi-class Brier Score
    - Reliability diagram bin details
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    # Multi-class Brier Score
    n_samples, n_classes = y_prob.shape
    one_hot_true = np.zeros((n_samples, n_classes))
    valid_mask = (y_true >= 0) & (y_true < n_classes)
    for i in range(n_samples):
        if valid_mask[i]:
            one_hot_true[i, y_true[i]] = 1.0
    brier_score = float(np.mean(np.sum((y_prob - one_hot_true) ** 2, axis=1)))

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(conf_in_bin - acc_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)

            bin_details.append({
                "bin_idx": i,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "center": (bin_lower + bin_upper) / 2.0,
                "count": int(np.sum(in_bin)),
                "accuracy": acc_in_bin,
                "confidence": conf_in_bin,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "bins": bin_details,
    }


def plot_reliability_diagrams(
    calib_results_before: Dict[str, Dict],
    calib_results_after: Dict[str, Dict],
    output_path: Union[str, Path],
    temperatures: Dict[str, float],
) -> None:
    """
    Renders multi-panel Reliability Diagrams before and after Temperature Scaling.
    Visualizes empirical accuracy vs mean confidence alongside confidence distribution histograms.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    targets = [t for t in calib_results_before if t in calib_results_after]
    n_targets = len(targets)

    fig, axes = plt.subplots(n_targets, 2, figsize=(12, 4 * n_targets))
    if n_targets == 1:
        axes = np.array([axes])

    for i, target in enumerate(targets):
        res_pre = calib_results_before[target]
        res_post = calib_results_after[target]
        temp_val = temperatures.get(target, 1.0)

        # Plot 1: Uncalibrated (Pre-Scaling)
        ax_pre = axes[i, 0]
        bins_pre = res_pre["bins"]
        if bins_pre:
            centers_pre = [b["center"] for b in bins_pre]
            accs_pre = [b["accuracy"] for b in bins_pre]
            ax_pre.bar(centers_pre, accs_pre, width=0.08, alpha=0.6, color="#d9534f", edgecolor="black", label="Empirical Acc")
        ax_pre.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.5, label="Perfect Calibration")
        ax_pre.set_title(f"{target.capitalize()} - Uncalibrated (ECE: {res_pre['ece']*100:.2f}%)", fontsize=11, weight="bold")
        ax_pre.set_xlabel("Confidence")
        ax_pre.set_ylabel("Accuracy")
        ax_pre.set_xlim([0, 1])
        ax_pre.set_ylim([0, 1])
        ax_pre.grid(True, linestyle=":", alpha=0.6)
        ax_pre.legend(loc="upper left")

        # Plot 2: Calibrated (Post-Temperature Scaling)
        ax_post = axes[i, 1]
        bins_post = res_post["bins"]
        if bins_post:
            centers_post = [b["center"] for b in bins_post]
            accs_post = [b["accuracy"] for b in bins_post]
            ax_post.bar(centers_post, accs_post, width=0.08, alpha=0.6, color="#5cb85c", edgecolor="black", label="Empirical Acc")
        ax_post.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.5, label="Perfect Calibration")
        ax_post.set_title(f"{target.capitalize()} - Calibrated (T={temp_val:.2f}, ECE: {res_post['ece']*100:.2f}%)", fontsize=11, weight="bold")
        ax_post.set_xlabel("Confidence")
        ax_post.set_ylabel("Accuracy")
        ax_post.set_xlim([0, 1])
        ax_post.set_ylim([0, 1])
        ax_post.grid(True, linestyle=":", alpha=0.6)
        ax_post.legend(loc="upper left")

    plt.suptitle("Reliability Diagrams: Before vs After Validation Temperature Scaling", fontsize=14, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved reliability diagrams to {output_file}")
