# Expert Review Card: FS-8091_09
- **Timestamp**: 2026-10-21T13:32:15.082044
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
| **Artist** | Unknown (100.0%) | Unknown (65.0%) | Unknown | MATCH |

## Cited Region Evidence (VLM)

- **crop_1 (building)**: The architectural pavilion with chhatris is characteristic of Rajasthani art. *(Supports: Rajasthani school)*
- **crop_2 (bird)**: Birds are often used as decorative elements in Rajput paintings, especially in the Mewar and Bundi schools. *(Supports: Rajasthani school)*

## Synthesis & Rationale
This painting depicts a nayika in traditional Rajput attire, engaging with another figure within an architectural pavilion. The use of gold leaf and rich colors is characteristic of late Mughal and early Rajasthani art periods. However, the absence of inscribed attribution makes it challenging to attribute this work definitively to any specific artist.

---
## Human Curator Action
- [ ] Confirm VLM Identification
- [ ] Confirm Statistical Classifier Top-1
- [ ] Provide Correct Identification: _______________
- [ ] Notes / Feedback: ____________________________
