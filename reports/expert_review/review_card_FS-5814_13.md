# Expert Review Card: FS-5814_13
- **Timestamp**: 2026-10-21T12:40:46.412018
- **Image Path**: `C:\Users\anush\Documents\Academic\Sem 7\BTP\data\raw\images\FS-5814_13.jpg`
- **Status**: `pending_human_review`

## Flagged Trigger Reason
> [!WARNING]
> School conflict: Classifier top-1='Rajasthani' (p=0.95) vs VLM='Mughal' (conf=1.00); Raga confidence (0.50) below threshold (0.70)

## Comparison: Classifier vs VLM Judge

| Target | Statistical Vision Classifier Top-1 | Multimodal VLM Prediction | Ground Truth (if available) | Agreement |
| --- | --- | --- | --- | --- |
| **Raga** | Ragamala (Unspecified) (53.8%) | Ragamala (Unspecified) (50.0%) | Ragamala (Unspecified) | MATCH |
| **School** | Rajasthani (94.7%) | Mughal (100.0%) | Rajasthani | DISAGREE |
| **Period** | Late Mughal / 18th Century (36.4%) | Late Mughal / 18th Century (95.0%) | None | MATCH |
| **Artist** | Unknown (100.0%) | Unknown (75.0%) | Unknown | MATCH |

## Cited Region Evidence (VLM)

- **crop_1 (nayika)**: Two women in colorful attire, one holding a mirror *(Supports: Todi Raga iconographic match)*
- **upper_sky_monsoon_clouds**: A blue sky with clouds and sun *(Supports: Shri Raga/Ragini iconographic match)*

## Synthesis & Rationale
This painting depicts a domestic scene with three women in colorful attire, one holding a mirror, which aligns more closely with the Mughal school's style of rich emotional expression. lyrical lines, and fine psychological portraiture rather than Ragamala composition. or musical mode personification.

---
## Human Curator Action
- [ ] Confirm VLM Identification
- [ ] Confirm Statistical Classifier Top-1
- [ ] Provide Correct Identification: _______________
- [ ] Notes / Feedback: ____________________________
