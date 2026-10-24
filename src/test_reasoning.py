"""
Reasoning Evaluation and Verification Harness.
Evaluates both Answer Correctness and Evidence Citation Validity on held-out test sets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from knowledge_base.retriever import ArtKnowledgeRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def score_evidence_citation(
    vlm_judgment: Dict,
    ground_truth_target: str,
    target_type: str = "raga",
    retriever: Optional[ArtKnowledgeRetriever] = None,
) -> Tuple[float, List[str], List[str]]:
    """
    Score whether the cited visual evidence correctly aligns with canonical
    iconographic/stylistic markers defined in the knowledge base for the ground truth class.

    Returns:
        evidence_score: float in [0.0, 1.0]
        valid_citations: list of legitimate cited markers
        spurious_citations: list of ungrounded/hallucinated markers
    """
    if retriever is None:
        retriever = ArtKnowledgeRetriever()

    # Obtain valid ground truth canonical motifs from KB
    canonical_motifs = []
    if target_type == "raga":
        raga_data = retriever.ragas_db.get(ground_truth_target, {})
        canonical_motifs = [m.lower() for m in raga_data.get("key_iconographic_motifs", [])]
    elif target_type == "school":
        school_data = retriever.schools_db.get(ground_truth_target, {})
        canonical_motifs = [m.lower() for m in school_data.get("stylistic_signatures", [])]

    # Extract all cited points from VLM output
    cited_points = []
    cited_points.extend(vlm_judgment.get(f"predicted_{target_type}", {}).get("iconographic_evidence", []))
    cited_points.extend(vlm_judgment.get(f"predicted_{target_type}", {}).get("stylistic_evidence", []))
    for reg in vlm_judgment.get("region_evidence", []):
        cited_points.append(reg.get("observation", ""))

    if not cited_points:
        return 0.0, [], []

    valid_citations = []
    spurious_citations = []

    for cite in cited_points:
        cite_lower = cite.lower()
        # Check if cite overlaps with any canonical marker keywords
        is_valid = any(
            any(word in cite_lower for word in marker.split() if len(word) > 3)
            for marker in canonical_motifs
        )
        if is_valid:
            valid_citations.append(cite)
        else:
            spurious_citations.append(cite)

    # Proportion of cited evidence that is verified against ground truth iconography
    score = len(valid_citations) / len(cited_points) if cited_points else 0.0
    return float(score), valid_citations, spurious_citations


def evaluate_reasoning_dataset(
    eval_cases: List[Dict],
    output_dir: Union[str, Path] = "reports",
) -> Dict:
    """
    Batch evaluate VLM reasoning outputs across a set of test paintings:
    - Measures Answer Accuracy (top-1 match)
    - Measures Evidence Citation Correctness (canonical iconographic overlap)
    - Measures Agreement Rate with Statistical Classifiers
    - Measures Expert Review Routing Frequency
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(eval_cases)
    if total == 0:
        return {"total": 0}

    targets = ["raga", "school", "period", "artist"]
    accuracy_counts = {t: 0 for t in targets}
    uncertain_counts = {t: 0 for t in targets}
    labelled_counts = {t: 0 for t in targets}
    evidence_scores = []
    classifier_agreements = 0
    expert_routed_cases = 0

    retriever = ArtKnowledgeRetriever()

    detailed_results = []

    for case in eval_cases:
        image_id = case.get("image_id", "unknown")
        gt = case.get("ground_truth", {})
        vlm = case.get("vlm_judgment", {})
        clf = case.get("classifier_top3", {})

        case_summary = {"image_id": image_id}

        # Check answers
        for t in targets:
            entry = vlm.get(f"predicted_{t}")
            pred_val = str((entry.get("value") if isinstance(entry, dict) else None) or "uncertain")
            gt_val = gt.get(t, "")
            if gt_val is None or (isinstance(gt_val, float) and np.isnan(gt_val)) or str(gt_val).lower() in ("", "nan"):
                case_summary[f"{t}_correct"] = None  # no ground truth for this target
                continue
            labelled_counts[t] += 1

            if pred_val.lower() == "uncertain":
                uncertain_counts[t] += 1
                case_summary[f"{t}_correct"] = False
                case_summary[f"{t}_uncertain"] = True
            elif gt_val and pred_val.lower() == str(gt_val).lower():
                accuracy_counts[t] += 1
                case_summary[f"{t}_correct"] = True
                case_summary[f"{t}_uncertain"] = False
            else:
                case_summary[f"{t}_correct"] = False
                case_summary[f"{t}_uncertain"] = False

        # Check Evidence Correctness on Raga
        gt_raga = gt.get("raga", "")
        ev_score, valid_cites, spurious_cites = score_evidence_citation(
            vlm_judgment=vlm,
            ground_truth_target=gt_raga,
            target_type="raga",
            retriever=retriever,
        )
        evidence_scores.append(ev_score)
        case_summary["evidence_correctness_score"] = ev_score
        case_summary["valid_citations"] = len(valid_cites)
        case_summary["spurious_citations"] = len(spurious_cites)

        # Cross-check metrics
        cross_check = vlm.get("cross_check", {})
        agreed = cross_check.get("agrees_with_classifiers", False)
        routed = cross_check.get("route_to_expert_review", False)

        if agreed:
            classifier_agreements += 1
        if routed:
            expert_routed_cases += 1

        case_summary["classifier_agreement"] = agreed
        case_summary["routed_to_expert"] = routed

        detailed_results.append(case_summary)

    # Compute overall aggregates
    metrics = {
        "total_paintings_evaluated": total,
        "answer_accuracy": {
            t: float(accuracy_counts[t] / labelled_counts[t]) if labelled_counts[t] else None for t in targets
        },
        "uncertain_rates": {
            t: float(uncertain_counts[t] / labelled_counts[t]) if labelled_counts[t] else None for t in targets
        },
        # Accuracy over the cases where the VLM committed to an answer (did not abstain)
        "selective_accuracy": {
            t: float(accuracy_counts[t] / (labelled_counts[t] - uncertain_counts[t]))
            if labelled_counts[t] - uncertain_counts[t] > 0 else None
            for t in targets
        },
        "labelled_cases": labelled_counts,
        "mean_evidence_citation_correctness": float(np.mean(evidence_scores)) if evidence_scores else 0.0,
        "classifier_vlm_agreement_rate": float(classifier_agreements / total),
        "expert_review_routing_rate": float(expert_routed_cases / total),
    }

    # Save JSON report
    json_path = out_dir / "vlm_reasoning_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"metrics": metrics, "detailed_cases": detailed_results}, jf, indent=2)

    # Save Markdown report
    md_path = out_dir / "vlm_reasoning_benchmark.md"
    md_lines = [
        "# Multimodal VLM Reasoning Benchmark Report",
        "",
        "> **Model Architecture**: Qwen Open-Source VLM + Domain-Specific Grounded Knowledge Base.",
        "> **Evaluation Focus**: Dual scoring of **Answer Correctness** AND **Evidence Citation Validity**.",
        "",
        "## Overall Benchmark Performance",
        "",
        f"- **Paintings Evaluated in Held-Out Set**: {total}",
        f"- **Mean Evidence Citation Correctness**: **{metrics['mean_evidence_citation_correctness']*100:.2f}%** (Proportion of cited visual motifs verified against canonical Sanskrit Dhyana verses)",
        f"- **Classifier vs VLM Agreement Rate**: **{metrics['classifier_vlm_agreement_rate']*100:.2f}%**",
        f"- **Expert Review Routing Rate**: **{metrics['expert_review_routing_rate']*100:.2f}%** (Ambiguous/disputed cases safely flagged for human connoisseurs)",
        "",
        "## Target Head Accuracy & Abstention / Uncertainty",
        "",
        "| Target Attribute | Answer Top-1 Accuracy | Abstention / 'Uncertain' Rate | Accuracy When Answered | Labelled Cases |",
        "| --- | --- | --- | --- | --- |",
    ]

    def _pct(v):
        return f"{v * 100:.2f}%" if v is not None else "N/A"

    for t in targets:
        md_lines.append(
            f"| **{t.capitalize()}** | {_pct(metrics['answer_accuracy'][t])} | {_pct(metrics['uncertain_rates'][t])} "
            f"| {_pct(metrics['selective_accuracy'][t])} | {metrics['labelled_cases'][t]} |"
        )

    md_lines.extend([
        "",
        "## Human-in-the-Loop & Fine-Tuning Strategy",
        "",
        "- **Dispute Routing**: Every case where the VLM and vision classifiers disagree or confidence is low has been routed to [`reports/expert_review/review_queue.jsonl`](expert_review/review_queue.jsonl).",
        "- **Fine-Tuning Recommendation**: Only initiate parameter fine-tuning on the VLM after thousands of expert-curated corrections are logged in the review queue. Until then, prompt engineering with Dhyana retrieval is more cost-efficient and avoids catastrophic forgetting.",
    ])

    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write("\n".join(md_lines) + "\n")

    logger.info(f"Reasoning evaluation report saved to {md_path}")
    return metrics
