# Multimodal VLM Reasoning Benchmark Report

> **Model Architecture**: Qwen Open-Source VLM + Domain-Specific Grounded Knowledge Base.
> **Evaluation Focus**: Dual scoring of **Answer Correctness** AND **Evidence Citation Validity**.

## Overall Benchmark Performance

- **Paintings Evaluated in Held-Out Set**: 84
- **Mean Evidence Citation Correctness**: **0.00%** (Proportion of cited visual motifs verified against canonical Sanskrit Dhyana verses)
- **Classifier vs VLM Agreement Rate**: **45.24%**
- **Expert Review Routing Rate**: **98.81%** (Ambiguous/disputed cases safely flagged for human connoisseurs)

## Target Head Accuracy & Abstention / Uncertainty

| Target Attribute | Answer Top-1 Accuracy | Abstention / 'Uncertain' Rate | Accuracy When Answered | Labelled Cases |
| --- | --- | --- | --- | --- |
| **Raga** | 3.57% | 45.24% | 6.52% | 84 |
| **School** | 63.10% | 33.33% | 94.64% | 84 |
| **Period** | 37.29% | 38.98% | 61.11% | 59 |
| **Artist** | 64.29% | 33.33% | 96.43% | 84 |

## Human-in-the-Loop & Fine-Tuning Strategy

- **Dispute Routing**: Every case where the VLM and vision classifiers disagree or confidence is low has been routed to [`reports/expert_review/review_queue.jsonl`](expert_review/review_queue.jsonl).
- **Fine-Tuning Recommendation**: Only initiate parameter fine-tuning on the VLM after thousands of expert-curated corrections are logged in the review queue. Until then, prompt engineering with Dhyana retrieval is more cost-efficient and avoids catastrophic forgetting.
