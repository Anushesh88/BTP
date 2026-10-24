# Phase 3 Baseline Benchmark Report

> **Architecture**: Frozen ViT/CLIP visual embeddings + Regularized Logistic Regression.
> **Purpose**: Benchmark floor that all subsequent multi-modal and fine-tuned models must beat.

## Summary Metrics

| Target Head | Accuracy | Macro F1 | Weighted F1 | ECE | Num Classes | Support |
| --- | --- | --- | --- | --- | --- | --- |
| Raga | 94.05% | 0.3936 | 0.9449 | 0.0568 | 4 | 84 |
| School | 94.05% | 0.5298 | 0.9454 | 0.0201 | 3 | 84 |
| Period | 61.02% | 0.4414 | 0.6181 | 0.2413 | 6 | 59 |
| Artist | 96.43% | 0.2455 | 0.9468 | 0.0157 | 4 | 84 |

## Comparison with Target Goal (85% - 90%)

- **Target Performance**: 85.00% – 90.00% across critical heads (Raga, School, Period).
- **Baseline Assessment**: Baseline provides the uncalibrated linear boundary on pre-trained visual representations.
- **Confusion Note**: Raga classes exhibit visual overlap; confusion matrices have been generated for error diagnosis.

## Confusion Matrix Artifacts

- **Raga**: [`confusion_matrix_raga.png`](confusion_matrix_raga.png)
- **School**: [`confusion_matrix_school.png`](confusion_matrix_school.png)
- **Period**: [`confusion_matrix_period.png`](confusion_matrix_period.png)
- **Artist**: [`confusion_matrix_artist.png`](confusion_matrix_artist.png)
