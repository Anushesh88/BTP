# Expert Review Card: FS-8091_09
- **Timestamp**: 2026-10-21T15:11:56.470929
- **Image Path**: `C:\Users\anush\Documents\Academic\Sem 7\BTP\data\raw\images\FS-8091_09.jpg`
- **Status**: `pending_human_review`

## Flagged Trigger Reason
> [!WARNING]
> Raga conflict: Classifier top-1='Shri Raga/Ragini' (p=0.48) vs VLM='Non-Ragamala' (conf=0.95)

## Comparison: Classifier vs VLM Judge

| Target | Statistical Vision Classifier Top-1 | Multimodal VLM Prediction | Ground Truth (if available) | Agreement |
| --- | --- | --- | --- | --- |
| **Raga** | Shri Raga/Ragini (48.2%) | Non-Ragamala (95.0%) | Ragamala (Unspecified) | DISAGREE |
| **School** | Rajasthani (88.7%) | Rajasthani (85.0%) | Rajasthani | MATCH |
| **Period** | Late Mughal / 18th Century (30.4%) | Late Mughal / 18th Century (75.0%) | None | MATCH |
| **Artist** | Unknown (100.0%) | Unknown (50.0%) | Unknown | MATCH |

## Cited Region Evidence (VLM)

- **crop_1 (building)**: A detailed architectural pavilion with red sandstone and marble jalis. *(Supports: Rajasthani architectural style)*

## Synthesis & Rationale
The painting exhibits characteristics of the late Mughal or early Rajput period, with a focus on architectural detail typical of Rajasthani art. The vibrant colors and rich detailing suggest an influence from Mughal courts during this time. However, specific attribution to a known artist is uncertain due to lack of inscriptions.

---
## Human Curator Action
- [ ] Confirm VLM Identification
- [ ] Confirm Statistical Classifier Top-1
- [ ] Provide Correct Identification: _______________
- [ ] Notes / Feedback: ____________________________
