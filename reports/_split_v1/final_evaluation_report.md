# Phase 10: Final Evaluation, Error Analysis & Feedback Loop

> Held-out test split: 84 paintings from manuscript groups never seen in training. Final model = Phase 7 multi-head classifier on fused features with Phase 8 temperature scaling.

## 10.1 Quantitative Assessment

| Head | Test N | Majority-class Acc | Baseline Acc | Final Acc | Final Macro F1 | Baseline Macro F1 | Majority Macro F1 | vs 85% target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raga | 84 | 94.0% | 94.0% | 90.5% | 0.463 | 0.394 | 0.323 | meets target |
| School | 84 | 94.0% | 94.0% | 90.5% | 0.485 | 0.530 | 0.323 | meets target |
| Period | 59 | 27.1% | 61.0% | 66.1% | 0.462 | 0.441 | 0.071 | below target |
| Artist | 84 | 96.4% | 96.4% | 96.4% | 0.245 | 0.245 | 0.245 | insufficient labels |

**Reading this table**: *Majority-class* always predicts the most common training label. Accuracy alone is misleading on this dataset because labels are heavily imbalanced (e.g. ~90% of test paintings are 'Mughal' and ~90% 'Non-Ragamala'), so Macro F1 against the majority floor is the honest measure.

### Target check (85% – 90%)
- **School**: final accuracy 90.5% (majority floor 94.0%, baseline 94.0%); macro F1 0.485 vs floor 0.323. 1 test label(s) belong to classes absent from training.
- **Period**: final accuracy 66.1% (majority floor 27.1%, baseline 61.0%); macro F1 0.462 vs floor 0.071.
- **Raga**: final accuracy 90.5% (majority floor 94.0%, baseline 94.0%); macro F1 0.463 vs floor 0.323.
- **Artist**: not assessed against the target — only 4 attributed paintings in training.

### Per-class F1

| Head | Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- |
| raga | Non-Ragamala | 1.00 | 0.92 | 0.96 | 79 |
| raga | Ragamala (Unspecified) | 0.30 | 0.75 | 0.43 | 4 |
| raga | Shri Raga/Ragini | 0.00 | 0.00 | 0.00 | 1 |
| school | Mughal | 1.00 | 0.91 | 0.95 | 79 |
| school | Pahari | 0.00 | 0.00 | 0.00 | 1 |
| school | Rajasthani | 0.33 | 1.00 | 0.50 | 4 |
| period | Akbar Period (c. 1556-1605) | 1.00 | 0.69 | 0.81 | 16 |
| period | Aurangzeb Period (c. 1658-1707) | 0.22 | 0.50 | 0.31 | 4 |
| period | Classical Mughal (16th-17th Century) | 0.87 | 1.00 | 0.93 | 20 |
| period | Jahangir Period (c. 1605-1627) | 0.25 | 0.20 | 0.22 | 5 |
| period | Late Mughal / 18th Century | 0.42 | 0.62 | 0.50 | 8 |
| period | Shah Jahan Period (c. 1628-1658) | 0.00 | 0.00 | 0.00 | 6 |
| artist | Bishandas | 0.00 | 0.00 | 0.00 | 1 |
| artist | Dhanu | 0.00 | 0.00 | 0.00 | 1 |
| artist | Unknown | 0.96 | 1.00 | 0.98 | 81 |
| artist | Ustad Mansur | 0.00 | 0.00 | 0.00 | 1 |

## 10.2 Error Analysis

### Calibration check: does 70% confidence mean ~70% precision?
- Predictions with confidence in [0.6, 0.8): n=19, mean confidence 73.6%, observed precision 68.4%.
- Predictions with confidence ≥ 0.7: n=180, observed precision 94.4%.
- Pooled over the raga, school and period heads after temperature scaling.

### Most frequent confusions (test split)

| head | true | predicted | count |
| --- | --- | --- | --- |
| school | Mughal | Rajasthani | 7 |
| raga | Non-Ragamala | Ragamala (Unspecified) | 6 |
| period | Shah Jahan Period (c. 1628-1658) | Aurangzeb Period (c. 1658-1707) | 4 |
| period | Aurangzeb Period (c. 1658-1707) | Late Mughal / 18th Century | 2 |
| period | Shah Jahan Period (c. 1628-1658) | Late Mughal / 18th Century | 2 |
| period | Akbar Period (c. 1556-1605) | Jahangir Period (c. 1605-1627) | 2 |
| period | Jahangir Period (c. 1605-1627) | Late Mughal / 18th Century | 2 |
| period | Akbar Period (c. 1556-1605) | Aurangzeb Period (c. 1658-1707) | 1 |
| period | Akbar Period (c. 1556-1605) | Classical Mughal (16th-17th Century) | 1 |
| period | Akbar Period (c. 1556-1605) | Late Mughal / 18th Century | 1 |

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
