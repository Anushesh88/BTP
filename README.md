# Mughal Miniature & Ragamala Analysis

End-to-end pipeline (checklist Phases 1–10) for classifying Indian miniature paintings by
**raga, school, period and artist**, with a calibrated classifier, a Qwen vision-language
judge grounded in an art-historical knowledge base, and an expert-review feedback loop.

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5vl:7b        # vision model for Phase 9 (~2 min/painting on a 4 GB GPU)
```

## Run

```bash
python run_all.py               # all phases (Phase 9 on all 84 test paintings: ~3 h with the 7B model on a 4 GB GPU)
python run_all.py --skip_vlm    # everything except the VLM
python run_all.py --from 7      # resume from a phase
python tests/test_phase10_feedback_eval.py   # each tests/*.py file runs standalone
```

Start with `reports/final_evaluation_report.md`.

## Predict one image

```bash
python predict.py path/to/painting.jpg
```

Images that are not Indian miniature paintings (photos, documents, other art) are **rejected**
by a gate (`src/ood.py`) instead of being forced into a Mughal/Rajasthani/Pahari label. The
classifier heads are closed-set, so without the gate a dog photo comes out "Mughal" with high
confidence. The gate needs both checks to pass: similarity to the 5 nearest training paintings
≥ 0.25 and CLIP "miniature painting" probability ≥ 0.50. All 169 val/test paintings pass; all
12 scikit-image sample photos fail. `--skip_gate` forces a (meaningless) prediction.

## Layout

| Path | Contents |
| --- | --- |
| `data/raw/` | 564 images + captions |
| `data/metadata/` | manifest, validation report, frozen leak-free `train/val/test_split.jsonl` |
| `data/processed/motifs/` | YOLO-World motif crops + `detected_motifs_manifest.jsonl` |
| `data/cache/` | ViT embeddings and fused features (validated by image ID) |
| `data/feedback/` | expert corrections (`expert_feedback.jsonl`) |
| `models/` | baseline, multi-head checkpoint, temperature scalers |
| `reports/` | per-phase reports, confusion matrices, error analysis, expert-review cards |
| `knowledge_base/` | Dhyana verses, school and period profiles, retriever |
| `docs/` | project PDFs |
| `logs/` | run logs |

## Labels ("honest heads")

The real dataset has very few raga and artist labels, so targets are normalized in
`src/dataset.py::normalize_targets`:

- **raga**: `None` → `Non-Ragamala`; ragas with < 3 paintings → `Ragamala (Other)`
- **period**: `Unknown` → missing (excluded from training and scoring for that head only)
- **artist**: kept, but reported as *insufficient labels* (only 4 attributed training paintings)

Because classes are heavily imbalanced, compare **macro F1 against the majority-class floor**,
not raw accuracy.

## Expert feedback loop (Phase 10.3)

```bash
python review_feedback.py pending
python review_feedback.py show <image_id>
python review_feedback.py correct <image_id> --target period --value "Jahangir Period (c. 1605-1627)" --reviewer "Name"
python review_feedback.py retrain   # re-runs Phases 3 → 7 → 8 → 10 with corrections applied
```

## Notes

- Always import `sklearn`/`src` modules **before** `torch` in ad-hoc scripts: the reverse order
  segfaults on this Windows setup (OpenMP clash).
- `run_pipeline.py` stages from `Dataset/`; it skips the copy if `data/raw` is already staged.
- `reports/_synthetic_archive/` and `data/_synthetic_archive/` hold the old outputs from the
  synthetic sample, kept for reference only.

## VLM comparison

```bash
python run_vlm_reasoning.py --model_name qwen2.5vl:7b --timeout 1800 --image_ids_file data/metadata/vlm_comparison_sample.txt --output_dir reports/vlm_7b_sample
python run_vlm_reasoning.py --model_name qwen2.5vl:7b --timeout 1800 --hide_classifier --image_ids_file data/metadata/vlm_comparison_sample.txt --output_dir reports/vlm_7b_blind_sample
python compare_vlm_runs.py reports/vlm_eval_cases.json reports/vlm_7b_sample/vlm_eval_cases.json reports/vlm_7b_blind_sample/vlm_eval_cases.json --notes reports/vlm_comparison_notes.md
```

The full 84-painting results in `reports/vlm_*` came from `qwen2.5vl:3b`, which has since been removed.
Results: `reports/vlm_model_comparison.md`.
