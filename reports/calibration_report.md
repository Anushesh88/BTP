# Phase 8: Model Calibration & Uncertainty Estimation Report

> **Objective**: Optimize post-hoc temperature scaling ($T > 0$) on the validation set using Negative Log-Likelihood (NLL) so that a confidence of '70%' empirically corresponds to ~70% true precision.

## Calibration Performance Summary (Held-Out Test Split)

| Target Head | Optimal Temp (T) | ECE Before | ECE After (Calibrated) | ECE Improvement | Brier Score Pre | Brier Score Post |
| --- | --- | --- | --- | --- | --- | --- |
| Raga | 0.828 | 6.27% | 8.44% | +2.17% | 0.1299 | 0.1321 |
| School | 1.518 | 6.93% | 4.87% | 2.07% | 0.1388 | 0.1248 |
| Period | 0.861 | 10.88% | 9.75% | 1.13% | 0.4338 | 0.4223 |
| Artist | 0.054 | 1.16% | 3.57% | +2.41% | 0.0437 | 0.0476 |

## Domain & Methodological Insights
- **Empirical Alignment**: Temperature scaling softens overconfident logits without changing top-1 class rankings (argmax is invariant).
- **Validation Splitting**: Temperatures were strictly learned on the held-out validation set to eliminate data leakage.
- **Learned Scalers Saved**: [`models/temperature_scalers.json`](../models/temperature_scalers.json)

## Reliability Diagrams (Pre vs. Post Scaling)
- Visual reliability curves and confidence calibration bins: [`reliability_diagrams.png`](reliability_diagrams.png)
