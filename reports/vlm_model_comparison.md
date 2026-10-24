# VLM Model Comparison (20 shared held-out paintings)

| Model | Head | N | Accuracy | Abstained | Accuracy when answered | Agrees with classifier top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| Classifier (calibrated multi-head) | raga | 20 | 85% | - | - | - |
| Classifier (calibrated multi-head) | school | 20 | 90% | - | - | - |
| Classifier (calibrated multi-head) | period | 15 | 73% | - | - | - |
| qwen2.5vl:3b | raga | 20 | 15% | 40% | 25% | 25% |
| qwen2.5vl:3b | school | 20 | 65% | 20% | 81% | 60% |
| qwen2.5vl:3b | period | 15 | 53% | 27% | 73% | 67% |
| qwen2.5vl:7b | raga | 20 | 75% | 0% | 75% | 70% |
| qwen2.5vl:7b | school | 20 | 90% | 0% | 90% | 90% |
| qwen2.5vl:7b | period | 15 | 53% | 0% | 53% | 80% |
| qwen2.5vl:7b (classifier hidden) | raga | 20 | 75% | 0% | 75% | 70% |
| qwen2.5vl:7b (classifier hidden) | school | 20 | 75% | 0% | 75% | 75% |
| qwen2.5vl:7b (classifier hidden) | period | 15 | 20% | 0% | 20% | 20% |

| Model | Routed to expert review |
| --- | --- |
| qwen2.5vl:3b | 95% |
| qwen2.5vl:7b | 35% |
| qwen2.5vl:7b (classifier hidden) | 50% |

## Raga and school answers on the non-Mughal / Ragamala paintings

| Painting | Ground truth (raga / school) | Classifier | qwen2.5vl:3b | qwen2.5vl:7b | qwen2.5vl:7b (classifier hidden) |
| --- | --- | --- | --- | --- | --- |
| ma-55333 | Shri Raga/Ragini / Rajasthani | Ragamala (Unspecified) / Rajasthani | Ragamala (Unspecified) / Rajasthani | Non-Ragamala / Rajasthani | Non-Ragamala / Rajasthani |
| FS-8091_09 | Ragamala (Unspecified) / Rajasthani | Shri Raga/Ragini / Rajasthani | uncertain / Rajasthani | Non-Ragamala / Rajasthani | Non-Ragamala / Rajasthani |
| FS-5814_13 | Ragamala (Unspecified) / Rajasthani | Ragamala (Unspecified) / Rajasthani | Ragamala (Unspecified) / Mughal | Non-Ragamala / Rajasthani | Non-Ragamala / Rajasthani |
| ma-45020 | Ragamala (Unspecified) / Rajasthani | Ragamala (Unspecified) / Rajasthani | Ragamala (Unspecified) / Mughal | Non-Ragamala / Rajasthani | Non-Ragamala / Rajasthani |
| FS-7407_27 | Ragamala (Unspecified) / Pahari | Ragamala (Unspecified) / Rajasthani | Ragamala (Unspecified) / Mughal | Non-Ragamala / Rajasthani | Non-Ragamala / Rajasthani |

Non-Ragamala paintings correctly called 'Non-Ragamala': qwen2.5vl:3b 0/15, qwen2.5vl:7b 15/15, qwen2.5vl:7b (classifier hidden) 15/15.

## Interpretation

**Blind test (classifier candidates hidden).** Retrieved knowledge is identical in both 7B runs, so the
only difference is whether the VLM sees the classifier's top-3.

- **School: some real visual skill.** Blind, the 7B correctly called all 5 non-Mughal paintings non-Mughal
  (4 Rajasthani right, the one Pahari called Rajasthani, as the classifier also does). But it also called 4 of
  15 Mughal paintings Rajasthani, giving 75% overall vs 90% when it can see the classifier.
- **Period: no independent skill.** It falls from 53% to 20% blind and splits its answers between "Late Mughal"
  and "Jahangir". With the classifier visible, it mostly copied it (80% agreement).
- **Raga: no skill.** It answers "Non-Ragamala" for all 20 paintings in both modes, missing every Ragamala.
  Its 75% equals always guessing the majority class.
- **Routing**: 50% blind vs 35% with hints. More disagreement is expected once it stops echoing the classifier.

**Overall.** Showing the classifier mostly makes the VLM agree with it rather than add evidence. The
independent signal is limited to a coarse Mughal vs non-Mughal judgment.

**Recommendation.** Keep the calibrated classifier as the decision maker. Use the 7B VLM blind, as an
independent school check for review routing, and for written rationales. Do not use it for raga or period.
It is slow on this hardware (~1.8–2.5 min per painting, ~80% on CPU).
