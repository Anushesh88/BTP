# Phase 7 Multi-Head Fused Benchmark Report

> **Architecture**: Frozen ViT/CLIP visual embeddings + Regularized Logistic Regression.
> **Purpose**: Benchmark floor that all subsequent multi-modal and fine-tuned models must beat.

## Summary Metrics

| Target Head | Accuracy | Macro F1 | Weighted F1 | ECE | Num Classes | Support |
| --- | --- | --- | --- | --- | --- | --- |
| Raga | 90.48% | 0.4630 | 0.9238 | 0.0627 | 4 | 84 |
| School | 90.48% | 0.4845 | 0.9207 | 0.0589 | 3 | 84 |
| Period | 66.10% | 0.4625 | 0.6438 | 0.1088 | 6 | 59 |
| Artist | 96.43% | 0.2455 | 0.9468 | 0.0355 | 4 | 84 |

## Comparison with Target Goal (85% - 90%)

- **Target Performance**: 85.00% – 90.00% across critical heads (Raga, School, Period).
- **Baseline Assessment**: Baseline provides the uncalibrated linear boundary on pre-trained visual representations.
- **Confusion Note**: Raga classes exhibit visual overlap; confusion matrices have been generated for error diagnosis.

## Confusion Matrix Artifacts

- **Raga**: [`multihead_confusion_matrix_raga.png`](multihead_confusion_matrix_raga.png)
- **School**: [`multihead_confusion_matrix_school.png`](multihead_confusion_matrix_school.png)
- **Period**: [`multihead_confusion_matrix_period.png`](multihead_confusion_matrix_period.png)
- **Artist**: [`multihead_confusion_matrix_artist.png`](multihead_confusion_matrix_artist.png)
