# Split Audit and Retraining (split v2)

**Bottom line:** the first split leaked, and the earlier scores were somewhat optimistic. On a
leak-free split with grouped cross-validation, the classifier is real but modest: mean macro F1
is about 0.37 across raga, school and period. Accuracy stays high only because about 90% of
paintings are Mughal and non-Ragamala.

## What was wrong with split v1

| Problem | Count | Effect |
| --- | --- | --- |
| Tuti-nama folios whose file names were truncated ("from_a_tuti_c_1560") became one-painting groups | 7 (5 in test) | Test paintings from a manuscript with 153 folios in train |
| "From the late Shah Jahan album" and other named albums not grouped | 9 | Same album in train, val and test |
| 116 "Mughal Style Detail Studies" crops from one source, each its own group | 116 | One source spread over all three splits |
| Near-duplicate images (same painting from two sources) in different splits | 2 pairs (val/test) | The same painting seen in val and scored in test |
| Exact duplicates in one split | 2 | Painting scored twice |
| Razmnama folios under two different group keys | – | Manuscript split across sets |

Removing only the 6 leaked test paintings from split v1 lowered accuracy by 1–2 points, so the
leak was real but small. The larger issue was that split v1 put almost all non-Mughal paintings
in train, so test (4 Rajasthani paintings) could barely measure school.

## The fix (`src/group_refinement.py`, applied in Phase 2)

1. Paintings naming a series in the title or caption join one group per series (Tuti-nama, any
   "from the … album", Akbarnama, Razmnama, and so on), including truncated names.
2. Single-source collections (the detail studies) form one group.
3. Near-duplicates (frozen ViT-B/16 cosine ≥ 0.90) share a group; exact duplicates (≥ 0.98) keep one copy.
4. Groups larger than val/test go to train; the splitter balances Rajasthani, Ragamala and the
   minority periods across splits.

The audit now finds 0 shared groups, 0 series spanning splits, and no cross-split image pair above
0.84 cosine. 562 paintings: 393 train, 84 val, 85 test. Details: `data/metadata/group_refinement_report.json`.

| | Train | Val | Test |
| --- | --- | --- | --- |
| Rajasthani school | 33 | 5 | 8 |
| Any Ragamala | 33 | 8 | 9 |
| Period labelled (not Unknown) | 336 | 41 | 46 |

## Results on the held-out test split (85 paintings)

| Head | Split v1 macro F1 | Split v2 macro F1 | Split v2 baseline macro F1 | Split v2 majority-class accuracy | Split v2 accuracy |
| --- | --- | --- | --- | --- | --- |
| School | 0.485 | 0.757 | 0.837 | 90.6% | 88.2% |
| Raga | 0.463 | 0.336 | 0.547 | 89.4% | 85.9% |
| Period | 0.462 | 0.385 | 0.369 | 50.0% | 63.0% |

Split v1 and v2 numbers are not directly comparable: the test sets differ, and v2's test holds
twice as many non-Mughal paintings.

## Model choice: grouped 5-fold cross-validation on train + val (477 paintings)

The 85-painting test set is too small to rank models: on it the baseline wins school and raga, but
1–2 Rajasthani paintings move school F1 by about 0.1. The 84-painting validation set is equally
noisy. So the model was chosen by cross-validation, holding out whole albums in each fold:

| Model | Raga F1 | School F1 | Period F1 | Mean |
| --- | --- | --- | --- | --- |
| Multi-head MLP on fused features (kept) | 0.35 | 0.49 | 0.26 | **0.368** |
| Multi-head MLP, 64 hidden | 0.36 | 0.49 | 0.24 | 0.366 |
| Logistic regression on fused features | 0.35 | 0.49 | 0.25 | 0.362 |
| Logistic regression on ViT embeddings (baseline) | 0.33 | 0.46 | 0.20 | 0.332 |

Fused features help a little over the baseline; the architecture on top matters less.

## Why predictions look wrong despite high accuracy

- **Accuracy is mostly the majority class.** About 90% of test paintings are Mughal and non-Ragamala,
  so always guessing those scores about 90%. The model's real skill on the rare classes is modest.
- **Rare classes have few examples.** 33 Rajasthani and 33 Ragamala paintings in training, and
  out-of-album generalisation is weak (cross-validated school F1 about 0.49).
- **Schools absent from the data** (Mewar, Kangra, Bundi, Deccan) can only be called Mughal or Rajasthani.
- **Non-miniature images** are now rejected by `src/ood.py` before classification.

## Stale results

The Phase 9 VLM reports (`vlm_reasoning_benchmark.*`, `vlm_7b_sample/`, `vlm_7b_blind_sample/`,
`vlm_model_comparison.md`) were produced on split v1 and have not been rerun. Split v1 files are kept in
`data/metadata/_split_v1/` and `reports/_split_v1/`.
