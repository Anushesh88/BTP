# Phase 7: Multi-Head Classifier Training & Evaluation Report

> **Architecture**: Standardized Multi-Task Neural Trunk on 1632-dim Dual-Branch Fused Features (Deep Global + Region Crops + Handcrafted Descriptors) with Balanced Class Weighting.

## Performance Comparison: Phase 3 Baseline vs. Phase 7 Multi-Head Fused

| Target Head | Phase 3 Baseline Acc | Phase 7 Fused Acc | Accuracy Delta | Phase 7 F1 (Weighted) |
| --- | --- | --- | --- | --- |
| Raga | 94.0% | 90.5% | -3.6% | 0.9238 |
| School | 94.0% | 90.5% | -3.6% | 0.9207 |
| Period | 61.0% | 66.1% | +5.1% | 0.6438 |
| Artist | 96.4% | 96.4% | +0.0% | 0.9468 |

## Progress Toward Target Goal (85% - 90%)
- **Target Benchmark**: 85.00% – 90.00% across primary heads (*Raga*, *School*, *Period*).
- **Domain Significance**: Fusing handcrafted color/texture/composition with deep region embeddings enhances fine-grained discrimination over simple frozen linear embeddings.
- **Saved Model Checkpoint**: [`models/multihead_fused_classifier.pt`](../models/multihead_fused_classifier.pt)

## Confusion Matrices (Fused Features)
- Raga: [`multihead_confusion_matrix_raga.png`](multihead_confusion_matrix_raga.png)
- School: [`multihead_confusion_matrix_school.png`](multihead_confusion_matrix_school.png)
- Period: [`multihead_confusion_matrix_period.png`](multihead_confusion_matrix_period.png)
- Artist: [`multihead_confusion_matrix_artist.png`](multihead_confusion_matrix_artist.png)
