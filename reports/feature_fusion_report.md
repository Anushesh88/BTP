# Phase 6: Dual-Branch Feature Engineering & Fusion Report

> **Objective**: Extract domain-tailored handcrafted descriptors (Color, Texture, Stroke, Composition) and fuse them with Deep Global and Region Crop embeddings for downstream multi-head classification.

## Feature Architecture Breakdown
- **Branch 1 (Deep Global Canvas)**: 768 dimensions (from frozen vit_b_16)
- **Branch 2 (Deep Region Aggregation)**: 768 dimensions (Mean-pooled over detected motif crops)
- **Branch 3 (Handcrafted Descriptors)**: 96 dimensions:
  * *Color (HSV & Lab)*: Hue/Saturation/Value histograms, Lab moments, Gold leaf / Mineral red fractions
  * *Texture*: GLCM (contrast, dissimilarity, homogeneity, energy, correlation), Uniform LBP, Gabor filter bank
  * *Stroke & Line*: Canny edge density, gradient orientation distribution, contour stroke lengths
  * *Composition*: 3x3 Rule-of-thirds grid energy, horizontal/vertical symmetry, center-of-mass focal point
- **Total Unified Fused Dimension**: **1632 dimensions**

## Feature Matrix Statistics
- **Paintings Processed**: 60
- **Matrix Shape**: `(60, 1632)`
- **Cache Location**: [`data\cache\fused_profile_vit_b_16.npz`](data\cache\fused_profile_vit_b_16.npz)

## Feature Activation Profile
- Visual distribution profile across branches: [`feature_distributions.png`](feature_distributions.png)
