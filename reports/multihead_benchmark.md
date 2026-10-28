# Phase 7 Multi-Head Fused Benchmark Report

> **Architecture**: Frozen ViT/CLIP visual embeddings + Regularized Logistic Regression.
> **Purpose**: Benchmark floor that all subsequent multi-modal and fine-tuned models must beat.

## Summary Metrics

| Target Head | Accuracy | Macro F1 | Weighted F1 | ECE | Num Classes | Support |
| --- | --- | --- | --- | --- | --- | --- |
| Raga | 85.88% | 0.3363 | 0.8687 | 0.0408 | 4 | 85 |
| School | 88.24% | 0.7574 | 0.8987 | 0.0734 | 2 | 85 |
| Period | 63.04% | 0.3848 | 0.6229 | 0.2042 | 6 | 46 |
| Artist | 98.82% | 0.4970 | 0.9824 | 0.0800 | 3 | 85 |

## Comparison with Target Goal (85% - 90%)

- **Target Performance**: 85.00% – 90.00% across critical heads (Raga, School, Period).
- **Baseline Assessment**: Baseline provides the uncalibrated linear boundary on pre-trained visual representations.
- **Confusion Note**: Raga classes exhibit visual overlap; confusion matrices have been generated for error diagnosis.

## Confusion Matrix Artifacts

- **Raga**: [`multihead_confusion_matrix_raga.png`](multihead_confusion_matrix_raga.png)
- **School**: [`multihead_confusion_matrix_school.png`](multihead_confusion_matrix_school.png)
- **Period**: [`multihead_confusion_matrix_period.png`](multihead_confusion_matrix_period.png)
- **Artist**: [`multihead_confusion_matrix_artist.png`](multihead_confusion_matrix_artist.png)
