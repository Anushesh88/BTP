# Phase 4: Preprocessing & Sliding Window Generation Report

> **Objective**: Preserve native resolution, apply edge-preserving denoising, color normalization, CLAHE contrast enhancement, unsharp masking, and generate overlapping sliding windows with exact coordinate tracking.

## Configuration Parameters
- **Window Size**: 512 x 512 pixels
- **Stride**: 256 pixels (Overlap: 50.0%)
- **Denoising Filter**: median
- **Lighting & Contrast**: CLAHE (clip=2.0, grid=8x8) + Gamma (1.05)
- **Sharpening**: Unsharp Mask (strength=1.2)
- **Border Handling**: Reflective padding for marginalia and borders

## Processed Paintings Summary

| Painting ID | Original Canvas | Extracted Windows | Enhanced Image Path |
| --- | --- | --- | --- |
| painting_000 | 256x256 | 1 | `data\processed\preprocessed_images\painting_000_enhanced.jpg` |
| painting_001 | 256x256 | 1 | `data\processed\preprocessed_images\painting_001_enhanced.jpg` |
| painting_002 | 256x256 | 1 | `data\processed\preprocessed_images\painting_002_enhanced.jpg` |
| painting_003 | 256x256 | 1 | `data\processed\preprocessed_images\painting_003_enhanced.jpg` |
| painting_004 | 256x256 | 1 | `data\processed\preprocessed_images\painting_004_enhanced.jpg` |

- **Total Sliding Windows Generated**: 5
- **Windows Manifest Location**: [`data\processed\windows_manifest.jsonl`](data\processed\windows_manifest.jsonl)

## Inspection Graphic
- See visual inspection comparison: [`preprocessing_inspection_painting_000.png`](preprocessing_inspection_painting_000.png)
