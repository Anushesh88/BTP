# Phase 10: Final Evaluation, Error Analysis & Feedback Loop

> Held-out test split: 85 paintings from manuscript groups never seen in training. Final model = Phase 7 multi-head classifier on fused features with Phase 8 temperature scaling.

## 10.1 Quantitative Assessment

| Head | Test N | Majority-class Acc | Baseline Acc | Final Acc | Final Macro F1 | Baseline Macro F1 | Majority Macro F1 | vs 85% target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raga | 85 | 89.4% | 92.9% | 85.9% | 0.336 | 0.547 | 0.236 | meets target |
| School | 85 | 90.6% | 94.1% | 88.2% | 0.757 | 0.837 | 0.475 | meets target |
| Period | 46 | 50.0% | 60.9% | 63.0% | 0.385 | 0.369 | 0.133 | below target |
| Artist | 85 | 98.8% | 98.8% | 98.8% | 0.497 | 0.497 | 0.497 | insufficient labels |

**Reading this table**: *Majority-class* always predicts the most common training label. Accuracy alone is misleading on this dataset because labels are heavily imbalanced (e.g. ~90% of test paintings are 'Mughal' and ~90% 'Non-Ragamala'), so Macro F1 against the majority floor is the honest measure.

### Target check (85% – 90%)
- **School**: final accuracy 88.2% (majority floor 90.6%, baseline 94.1%); macro F1 0.757 vs floor 0.475.
- **Period**: final accuracy 63.0% (majority floor 50.0%, baseline 60.9%); macro F1 0.385 vs floor 0.133.
- **Raga**: final accuracy 85.9% (majority floor 89.4%, baseline 92.9%); macro F1 0.336 vs floor 0.236.
- **Artist**: not assessed against the target — only 5 attributed paintings in training.

### Per-class F1

| Head | Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- |
| raga | Non-Ragamala | 0.99 | 0.91 | 0.95 | 76 |
| raga | Ragamala (Other) | 0.00 | 0.00 | 0.00 | 2 |
| raga | Ragamala (Unspecified) | 0.27 | 0.80 | 0.40 | 5 |
| raga | Shri Raga/Ragini | 0.00 | 0.00 | 0.00 | 2 |
| school | Mughal | 0.99 | 0.88 | 0.93 | 77 |
| school | Rajasthani | 0.44 | 0.88 | 0.58 | 8 |
| period | Akbar Period (c. 1556-1605) | 0.79 | 0.83 | 0.81 | 23 |
| period | Aurangzeb Period (c. 1658-1707) | 0.00 | 0.00 | 0.00 | 4 |
| period | Classical Mughal (16th-17th Century) | 0.00 | 0.00 | 0.00 | 0 |
| period | Jahangir Period (c. 1605-1627) | 0.50 | 0.14 | 0.22 | 7 |
| period | Late Mughal / 18th Century | 0.78 | 0.78 | 0.78 | 9 |
| period | Shah Jahan Period (c. 1628-1658) | 0.40 | 0.67 | 0.50 | 3 |
| artist | Unknown | 0.99 | 1.00 | 0.99 | 84 |
| artist | Ustad Mansur | 0.00 | 0.00 | 0.00 | 1 |

## 10.2 Error Analysis

### Calibration check: does 70% confidence mean ~70% precision?
- Predictions with confidence in [0.6, 0.8): n=27, mean confidence 72.6%, observed precision 63.0%.
- Predictions with confidence ≥ 0.7: n=171, observed precision 90.6%.
- Pooled over the raga, school and period heads after temperature scaling.

### Most frequent confusions (test split)

| head | true | predicted | count |
| --- | --- | --- | --- |
| school | Mughal | Rajasthani | 9 |
| raga | Non-Ragamala | Ragamala (Unspecified) | 7 |
| period | Akbar Period (c. 1556-1605) | Classical Mughal (16th-17th Century) | 3 |
| raga | Ragamala (Other) | Ragamala (Unspecified) | 2 |
| period | Jahangir Period (c. 1605-1627) | Akbar Period (c. 1556-1605) | 2 |
| period | Aurangzeb Period (c. 1658-1707) | Shah Jahan Period (c. 1628-1658) | 2 |
| period | Jahangir Period (c. 1605-1627) | Late Mughal / 18th Century | 2 |
| raga | Shri Raga/Ragini | Ragamala (Unspecified) | 2 |
| period | Akbar Period (c. 1556-1605) | Jahangir Period (c. 1605-1627) | 1 |
| period | Aurangzeb Period (c. 1658-1707) | Classical Mughal (16th-17th Century) | 1 |

### Artifacts
- Most-confident misclassifications: [`error_analysis/misclassified_test_samples.csv`](error_analysis/misclassified_test_samples.csv)
- Thumbnails with gradient attribution maps: [`error_analysis/error_attribution_maps.png`](error_analysis/error_attribution_maps.png)
- Motif detector overlays: see [`motif_detection_report.md`](motif_detection_report.md)
- Confusion matrices: `multihead_confusion_matrix_{raga,school,period,artist}.png`

## Phase 9 VLM Judge (held-out test split)

| Head | VLM accuracy | Abstention rate | Accuracy when answered |
| --- | --- | --- | --- |
| School | 63.1% | 33.3% | 94.6% |
| Period | 37.3% | 39.0% | 61.1% |
| Raga | 3.6% | 45.2% | 6.5% |

- Classifier/VLM agreement: 45.2%; routed to expert review: 98.8%.
- See [`vlm_reasoning_benchmark.md`](vlm_reasoning_benchmark.md) for the model, prompt and scoring details.

## 10.3 Human-in-the-Loop Feedback Loop
- Cases where the VLM and classifiers disagree, or confidence < 0.70, are queued in [`expert_review/review_queue.jsonl`](expert_review/review_queue.jsonl) with a review card each.
- Experts record corrections with `python review_feedback.py correct <image_id> --target <head> --value <label> --reviewer <name>`; they are stored in `data/feedback/expert_feedback.jsonl`.
- Corrections are applied automatically whenever the frozen splits load; `python review_feedback.py retrain` re-runs Phases 3 → 7 → 8 → 10 on the corrected labels.
