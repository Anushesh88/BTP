"""
Phase 10 CLI Runner: Final Evaluation, Error Analysis & Reporting.

10.1 Quantitative assessment on the held-out test split: Top-1 accuracy, Macro F1,
     per-class F1 for all 4 heads, against the majority-class floor, the Phase 3
     baseline, and the 85-90% target.
10.2 Error analysis: most-confident misclassifications, a thumbnail sheet, and
     input-gradient attribution maps through the global ViT branch.
     Calibration check: does ~70% confidence mean ~70% precision?
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.baseline_classifier import BaselineMultiHeadClassifier
from src.calibration import MultiHeadTemperatureScaler
from src.dataset import load_frozen_splits
from src.evaluate import format_markdown_table
from src.feature_extractor import FrozenVisionBackbone, extract_embeddings_for_split
from src.feature_fusion import DualBranchFusionPipeline
from src.multihead_classifier import MultiHeadTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_evaluation")

TARGETS = ["raga", "school", "period", "artist"]
TARGET_RANGE = (0.85, 0.90)
# Heads whose labels are too sparse to train meaningfully on this dataset
INSUFFICIENT_LABEL_HEADS = {"artist"}


def head_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def gradient_saliency(backbone, trainer, img_path, fused_vec, target, class_idx, device):
    """
    Input-gradient attribution of one head's class logit w.r.t. the painting pixels,
    flowing through the global ViT branch (region + handcrafted branches held fixed).
    """
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device).requires_grad_(True)

    feat = backbone.model(x)
    feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-7)
    d = feat.shape[1]
    rest = torch.tensor(fused_vec[d:], dtype=torch.float32, device=device).unsqueeze(0)
    logits = trainer.model(torch.cat([feat, rest], dim=1))
    logits[target][0, class_idx].backward()

    sal = x.grad.abs().max(dim=1)[0].squeeze(0).cpu().numpy()
    sal = F.avg_pool2d(torch.tensor(sal)[None, None], 9, stride=1, padding=4)[0, 0].numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return np.array(img.resize((224, 224))), sal


def main():
    parser = argparse.ArgumentParser(description="Phase 10: Final Evaluation & Error Analysis")
    parser.add_argument("--splits_dir", type=str, default="data/metadata")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--motifs_manifest", type=str, default="data/processed/motifs/detected_motifs_manifest.jsonl")
    parser.add_argument("--num_error_examples", type=int, default=12)
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    err_dir = reports / "error_analysis"
    err_dir.mkdir(parents=True, exist_ok=True)

    train_df, _, test_df = load_frozen_splits(args.splits_dir)

    # Phase 3 baseline predictions
    baseline = BaselineMultiHeadClassifier.load(Path(args.models_dir) / "baseline_classifiers.joblib")
    X_emb, _ = extract_embeddings_for_split(df=test_df, split_name="test", model_name="vit_b_16", cache_dir=args.cache_dir)
    base_pred = baseline.predict(X_emb)

    # Phase 7 + 8: calibrated multi-head predictions
    trainer = MultiHeadTrainer.load_checkpoint(Path(args.models_dir) / "multihead_fused_classifier.pt")
    fusion = DualBranchFusionPipeline(backbone_model_name="vit_b_16")
    X_fused, _ = fusion.extract_dataset_fused_features(
        df=test_df, motifs_manifest_path=args.motifs_manifest,
        cache_file=Path(args.cache_dir) / "fused_test_vit_b_16.npz",
    )
    with torch.no_grad():
        logits = trainer.model(torch.tensor(X_fused, dtype=torch.float32).to(trainer.device))
    temps_file = Path(args.models_dir) / "temperature_scalers.json"
    scaler = MultiHeadTemperatureScaler.load(temps_file) if temps_file.exists() else None
    probs = scaler.scale_probabilities(logits) if scaler else {t: torch.softmax(l, 1).cpu().numpy() for t, l in logits.items()}

    # ---------------- 10.1 Quantitative assessment ----------------
    results, rows, per_class_rows = {}, [], []
    calib_pool_conf, calib_pool_correct = [], []
    error_records = []

    for t in TARGETS:
        mask = test_df[t].notna().values
        y_true = test_df[t][mask].astype(str).values
        classes = trainer.label_encoders[t].classes_
        p_t = probs[t][mask]
        y_pred = classes[p_t.argmax(1)]
        conf = p_t.max(1)

        majority = train_df[t].dropna().astype(str).value_counts().idxmax()
        maj_m = head_metrics(y_true, np.full(len(y_true), majority))
        base_m = head_metrics(y_true, np.asarray(base_pred[t])[mask]) if t in base_pred else None
        final_m = head_metrics(y_true, y_pred)
        unseen = int((~np.isin(y_true, classes)).sum())

        results[t] = {"final": final_m, "baseline": base_m, "majority_class": {"label": majority, **maj_m},
                      "support": int(mask.sum()), "test_labels_unseen_in_training": unseen,
                      "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0)}

        meets = TARGET_RANGE[0] <= final_m["accuracy"]
        note = "insufficient labels" if t in INSUFFICIENT_LABEL_HEADS else ("meets target" if meets else "below target")
        rows.append({
            "Head": t.capitalize(),
            "Test N": int(mask.sum()),
            "Majority-class Acc": f"{maj_m['accuracy']*100:.1f}%",
            "Baseline Acc": f"{base_m['accuracy']*100:.1f}%" if base_m else "N/A",
            "Final Acc": f"{final_m['accuracy']*100:.1f}%",
            "Final Macro F1": f"{final_m['macro_f1']:.3f}",
            "Baseline Macro F1": f"{base_m['macro_f1']:.3f}" if base_m else "N/A",
            "Majority Macro F1": f"{maj_m['macro_f1']:.3f}",
            "vs 85% target": note,
        })

        for cls, m in results[t]["per_class"].items():
            if isinstance(m, dict) and cls not in ("macro avg", "weighted avg"):
                per_class_rows.append({"Head": t, "Class": cls, "Precision": f"{m['precision']:.2f}",
                                       "Recall": f"{m['recall']:.2f}", "F1": f"{m['f1-score']:.2f}",
                                       "Support": int(m["support"])})

        if t not in INSUFFICIENT_LABEL_HEADS:
            calib_pool_conf.extend(conf.tolist())
            calib_pool_correct.extend((y_pred == y_true).tolist())

        sub = test_df[mask].reset_index(drop=True)
        for i in np.where(y_pred != y_true)[0]:
            error_records.append({
                "head": t, "image_id": sub.loc[i, "image_id"], "file_path": sub.loc[i, "file_path"],
                "true": y_true[i], "predicted": y_pred[i], "confidence": float(conf[i]),
                "true_in_training_classes": bool(y_true[i] in classes),
                "row_index": int(np.where(test_df["image_id"].values == sub.loc[i, "image_id"])[0][0]),
            })

    # ---------------- 10.2 Calibration check & error analysis ----------------
    calib_conf = np.array(calib_pool_conf)
    calib_correct = np.array(calib_pool_correct, dtype=float)
    band = (calib_conf >= 0.6) & (calib_conf < 0.8)
    above = calib_conf >= 0.7
    calib_check = {
        "band_0.6_0.8_n": int(band.sum()),
        "band_0.6_0.8_mean_confidence": float(calib_conf[band].mean()) if band.any() else None,
        "band_0.6_0.8_precision": float(calib_correct[band].mean()) if band.any() else None,
        "conf_ge_0.7_n": int(above.sum()),
        "conf_ge_0.7_precision": float(calib_correct[above].mean()) if above.any() else None,
    }

    errors = pd.DataFrame(error_records)
    if not errors.empty:
        errors = errors.sort_values("confidence", ascending=False)
        errors.drop(columns=["row_index"]).to_csv(err_dir / "misclassified_test_samples.csv", index=False)

    confusion_pairs = (
        errors[errors["head"].isin(["school", "period", "raga"])]
        .groupby(["head", "true", "predicted"]).size().sort_values(ascending=False).head(10)
        .reset_index(name="count") if not errors.empty else pd.DataFrame()
    )

    # Thumbnail sheet + attribution maps for the most confident errors on trainable heads
    top_err = errors[~errors["head"].isin(INSUFFICIENT_LABEL_HEADS)].head(args.num_error_examples) if not errors.empty else errors
    if len(top_err):
        backbone = FrozenVisionBackbone("vit_b_16").to(trainer.device)
        n = len(top_err)
        fig, axes = plt.subplots(n, 2, figsize=(7, 3.3 * n))
        axes = np.atleast_2d(axes)
        for k, (_, e) in enumerate(top_err.iterrows()):
            cls_idx = int(np.where(trainer.label_encoders[e["head"]].classes_ == e["predicted"])[0][0])
            img, sal = gradient_saliency(backbone, trainer, e["file_path"], X_fused[e["row_index"]],
                                         e["head"], cls_idx, trainer.device)
            axes[k, 0].imshow(img)
            axes[k, 0].set_title(f"{e['head']}: true={e['true']}\npred={e['predicted']} ({e['confidence']:.2f})", fontsize=7)
            axes[k, 1].imshow(img)
            axes[k, 1].imshow(sal, cmap="jet", alpha=0.5)
            axes[k, 1].set_title("Gradient attribution (global ViT branch)", fontsize=7)
            for a in axes[k]:
                a.axis("off")
        plt.tight_layout()
        plt.savefig(err_dir / "error_attribution_maps.png", dpi=130)
        plt.close(fig)

    # ---------------- Report ----------------
    summary = {"heads": results, "calibration_check": calib_check,
               "temperatures": scaler.temperatures if scaler else None,
               "target_range": TARGET_RANGE}
    with open(reports / "final_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    def pct(v):
        return f"{v*100:.1f}%" if v is not None else "N/A"

    trainable = [t for t in ("school", "period", "raga")]
    md = [
        "# Phase 10: Final Evaluation, Error Analysis & Feedback Loop",
        "",
        f"> Held-out test split: {len(test_df)} paintings from manuscript groups never seen in training. "
        "Final model = Phase 7 multi-head classifier on fused features with Phase 8 temperature scaling.",
        "",
        "## 10.1 Quantitative Assessment",
        "",
        format_markdown_table(pd.DataFrame(rows)),
        "",
        "**Reading this table**: *Majority-class* always predicts the most common training label. "
        "Accuracy alone is misleading on this dataset because labels are heavily imbalanced "
        "(e.g. ~90% of test paintings are 'Mughal' and ~90% 'Non-Ragamala'), so Macro F1 against the majority floor is the honest measure.",
        "",
        "### Target check (85% – 90%)",
    ]
    for t in trainable:
        r = results[t]
        md.append(
            f"- **{t.capitalize()}**: final accuracy {pct(r['final']['accuracy'])} "
            f"(majority floor {pct(r['majority_class']['accuracy'])}, baseline {pct(r['baseline']['accuracy']) if r['baseline'] else 'N/A'}); "
            f"macro F1 {r['final']['macro_f1']:.3f} vs floor {r['majority_class']['macro_f1']:.3f}."
            + (f" {r['test_labels_unseen_in_training']} test label(s) belong to classes absent from training." if r['test_labels_unseen_in_training'] else "")
        )
    md += [
        f"- **Artist**: not assessed against the target — only {int((train_df['artist'] != 'Unknown').sum())} attributed paintings in training.",
        "",
        "### Per-class F1",
        "",
        format_markdown_table(pd.DataFrame(per_class_rows)),
        "",
        "## 10.2 Error Analysis",
        "",
        "### Calibration check: does 70% confidence mean ~70% precision?",
        f"- Predictions with confidence in [0.6, 0.8): n={calib_check['band_0.6_0.8_n']}, "
        f"mean confidence {pct(calib_check['band_0.6_0.8_mean_confidence'])}, observed precision {pct(calib_check['band_0.6_0.8_precision'])}.",
        f"- Predictions with confidence ≥ 0.7: n={calib_check['conf_ge_0.7_n']}, observed precision {pct(calib_check['conf_ge_0.7_precision'])}.",
        "- Pooled over the raga, school and period heads after temperature scaling.",
        "",
        "### Most frequent confusions (test split)",
        "",
        format_markdown_table(confusion_pairs) if len(confusion_pairs) else "_No errors._",
        "",
        "### Artifacts",
        "- Most-confident misclassifications: [`error_analysis/misclassified_test_samples.csv`](error_analysis/misclassified_test_samples.csv)",
        "- Thumbnails with gradient attribution maps: [`error_analysis/error_attribution_maps.png`](error_analysis/error_attribution_maps.png)",
        "- Motif detector overlays: see [`motif_detection_report.md`](motif_detection_report.md)",
        "- Confusion matrices: `multihead_confusion_matrix_{raga,school,period,artist}.png`",
        "",
    ]
    vlm_json = reports / "vlm_reasoning_benchmark.json"
    if vlm_json.exists():
        vm = json.load(open(vlm_json, encoding="utf-8"))["metrics"]
        md += [
            "## Phase 9 VLM Judge (held-out test split)",
            "",
            "| Head | VLM accuracy | Abstention rate | Accuracy when answered |",
            "| --- | --- | --- | --- |",
            *[f"| {t.capitalize()} | {pct(vm['answer_accuracy'][t])} | {pct(vm['uncertain_rates'][t])} | {pct(vm.get('selective_accuracy', {}).get(t))} |"
              for t in ("school", "period", "raga")],
            "",
            f"- Classifier/VLM agreement: {pct(vm['classifier_vlm_agreement_rate'])}; routed to expert review: {pct(vm['expert_review_routing_rate'])}.",
            "- See [`vlm_reasoning_benchmark.md`](vlm_reasoning_benchmark.md) for the model, prompt and scoring details.",
            "",
        ]
    md += [
        "## 10.3 Human-in-the-Loop Feedback Loop",
        "- Cases where the VLM and classifiers disagree, or confidence < 0.70, are queued in [`expert_review/review_queue.jsonl`](expert_review/review_queue.jsonl) with a review card each.",
        "- Experts record corrections with `python review_feedback.py correct <image_id> --target <head> --value <label> --reviewer <name>`; they are stored in `data/feedback/expert_feedback.jsonl`.",
        "- Corrections are applied automatically whenever the frozen splits load; `python review_feedback.py retrain` re-runs Phases 3 → 7 → 8 → 10 on the corrected labels.",
    ]
    with open(reports / "final_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    logger.info(f"Phase 10 report: {reports / 'final_evaluation_report.md'}")


if __name__ == "__main__":
    main()
