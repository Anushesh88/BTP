# Expert Review Card: FS-8106_16
- **Timestamp**: 2026-10-21T12:49:46.558274
- **Image Path**: `C:\Users\anush\Documents\Academic\Sem 7\BTP\data\raw\images\FS-8106_16.jpg`
- **Status**: `pending_human_review`

## Flagged Trigger Reason
> [!WARNING]
> Raga conflict: Classifier top-1='Non-Ragamala' (p=1.00) vs VLM='Ragamala (Other)' (conf=0.90)

## Comparison: Classifier vs VLM Judge

| Target | Statistical Vision Classifier Top-1 | Multimodal VLM Prediction | Ground Truth (if available) | Agreement |
| --- | --- | --- | --- | --- |
| **Raga** | Non-Ragamala (99.7%) | Ragamala (Other) (90.0%) | Non-Ragamala | DISAGREE |
| **School** | Mughal (98.6%) | Mughal (100.0%) | Mughal | MATCH |
| **Period** | Akbar Period (c. 1556-1605) (30.9%) | Akbar Period (c. 1556-1605) (90.0%) | None | MATCH |
| **Artist** | Unknown (100.0%) | Unknown (100.0%) | Unknown | MATCH |

## Cited Region Evidence (VLM)

- **crop_2 (turban)**: Bearing a turban *(Supports: Todi Raga iconographic match)*

## Synthesis & Rationale
This painting depicts the Mughal emperor Akbar seated on a throne with two attendants. The child figure in the image could represent an iconographic emblem of Todi Raga, which aligns with the Ragamala tradition.

---
## Human Curator Action
- [ ] Confirm VLM Identification
- [ ] Confirm Statistical Classifier Top-1
- [ ] Provide Correct Identification: _______________
- [ ] Notes / Feedback: ____________________________
