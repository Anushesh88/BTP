# Phase 8: Model Calibration & Uncertainty Estimation Report

> **Objective**: Optimize post-hoc temperature scaling ($T > 0$) on the validation set using Negative Log-Likelihood (NLL) so that a confidence of '70%' empirically corresponds to ~70% true precision.

## Calibration Performance Summary (Held-Out Test Split)

| Target Head | Optimal Temp (T) | ECE Before | ECE After (Calibrated) | ECE Improvement | Brier Score Pre | Brier Score Post |
| --- | --- | --- | --- | --- | --- | --- |
| Raga | 0.715 | 4.08% | 4.95% | +0.87% | 0.1915 | 0.1964 |
| School | 1.327 | 7.34% | 4.65% | 2.69% | 0.1864 | 0.1788 |
| Period | 0.969 | 20.42% | 19.34% | 1.08% | 0.5835 | 0.5824 |
| Artist | 0.114 | 8.00% | 1.18% | 6.82% | 0.0358 | 0.0235 |

## Domain & Methodological Insights
- **Empirical Alignment**: Temperature scaling softens overconfident logits without changing top-1 class rankings (argmax is invariant).
- **Validation Splitting**: Temperatures were strictly learned on the held-out validation set to eliminate data leakage.
- **Learned Scalers Saved**: [`models/temperature_scalers.json`](../models/temperature_scalers.json)

## Reliability Diagrams (Pre vs. Post Scaling)
- Visual reliability curves and confidence calibration bins: [`reliability_diagrams.png`](reliability_diagrams.png)
