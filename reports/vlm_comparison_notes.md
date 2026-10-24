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
