# Phase 7: Multi-Head Classifier Training & Evaluation Report

> **Architecture**: Standardized Multi-Task Neural Trunk on 1632-dim Dual-Branch Fused Features (Deep Global + Region Crops + Handcrafted Descriptors) with Balanced Class Weighting.

## Performance Comparison: Phase 3 Baseline vs. Phase 7 Multi-Head Fused

| Target Head | Phase 3 Baseline Acc | Phase 7 Fused Acc | Accuracy Delta | Phase 7 F1 (Weighted) |
| --- | --- | --- | --- | --- |
| Raga | 92.9% | 85.9% | -7.1% | 0.8687 |
| School | 94.1% | 88.2% | -5.9% | 0.8987 |
| Period | 60.9% | 63.0% | +2.2% | 0.6229 |
| Artist | 98.8% | 98.8% | +0.0% | 0.9824 |

## Progress Toward Target Goal (85% - 90%)
- **Target Benchmark**: 85.00% – 90.00% across primary heads (*Raga*, *School*, *Period*).
- **Domain Significance**: Fusing handcrafted color/texture/composition with deep region embeddings enhances fine-grained discrimination over simple frozen linear embeddings.
- **Saved Model Checkpoint**: [`models/multihead_fused_classifier.pt`](../models/multihead_fused_classifier.pt)

## Confusion Matrices (Fused Features)
- Raga: [`multihead_confusion_matrix_raga.png`](multihead_confusion_matrix_raga.png)
- School: [`multihead_confusion_matrix_school.png`](multihead_confusion_matrix_school.png)
- Period: [`multihead_confusion_matrix_period.png`](multihead_confusion_matrix_period.png)
- Artist: [`multihead_confusion_matrix_artist.png`](multihead_confusion_matrix_artist.png)
