# Multimodal VLM Reasoning Benchmark Report

> **Model Architecture**: Qwen Open-Source VLM + Domain-Specific Grounded Knowledge Base.
> **Evaluation Focus**: Dual scoring of **Answer Correctness** AND **Evidence Citation Validity**.

## Overall Benchmark Performance

- **Paintings Evaluated in Held-Out Set**: 20
- **Mean Evidence Citation Correctness**: **0.00%** (Proportion of cited visual motifs verified against canonical Sanskrit Dhyana verses)
- **Classifier vs VLM Agreement Rate**: **65.00%**
- **Expert Review Routing Rate**: **35.00%** (Ambiguous/disputed cases safely flagged for human connoisseurs)

## Target Head Accuracy & Abstention / Uncertainty

| Target Attribute | Answer Top-1 Accuracy | Abstention / 'Uncertain' Rate | Accuracy When Answered | Labelled Cases |
| --- | --- | --- | --- | --- |
| **Raga** | 75.00% | 0.00% | 75.00% | 20 |
| **School** | 90.00% | 0.00% | 90.00% | 20 |
| **Period** | 53.33% | 0.00% | 53.33% | 15 |
| **Artist** | 80.00% | 0.00% | 80.00% | 20 |

## Human-in-the-Loop & Fine-Tuning Strategy

- **Dispute Routing**: Every case where the VLM and vision classifiers disagree or confidence is low has been routed to [`reports/expert_review/review_queue.jsonl`](expert_review/review_queue.jsonl).
- **Fine-Tuning Recommendation**: Only initiate parameter fine-tuning on the VLM after thousands of expert-curated corrections are logged in the review queue. Until then, prompt engineering with Dhyana retrieval is more cost-efficient and avoids catastrophic forgetting.
