# Phase 3 Baseline Benchmark Report

> **Architecture**: Frozen ViT/CLIP visual embeddings + Regularized Logistic Regression.
> **Purpose**: Benchmark floor that all subsequent multi-modal and fine-tuned models must beat.

## Summary Metrics

| Target Head | Accuracy | Macro F1 | Weighted F1 | ECE | Num Classes | Support |
| --- | --- | --- | --- | --- | --- | --- |
| Raga | 92.94% | 0.5465 | 0.9187 | 0.0298 | 4 | 85 |
| School | 94.12% | 0.8366 | 0.9427 | 0.0144 | 2 | 85 |
| Period | 60.87% | 0.3688 | 0.5956 | 0.2052 | 6 | 46 |
| Artist | 98.82% | 0.4970 | 0.9824 | 0.0129 | 3 | 85 |

## Comparison with Target Goal (85% - 90%)

- **Target Performance**: 85.00% – 90.00% across critical heads (Raga, School, Period).
- **Baseline Assessment**: Baseline provides the uncalibrated linear boundary on pre-trained visual representations.
- **Confusion Note**: Raga classes exhibit visual overlap; confusion matrices have been generated for error diagnosis.

## Confusion Matrix Artifacts

- **Raga**: [`confusion_matrix_raga.png`](confusion_matrix_raga.png)
- **School**: [`confusion_matrix_school.png`](confusion_matrix_school.png)
- **Period**: [`confusion_matrix_period.png`](confusion_matrix_period.png)
- **Artist**: [`confusion_matrix_artist.png`](confusion_matrix_artist.png)
