"""
End-to-end runner for Phases 1-10 on the real dataset.

  python run_all.py                 # everything (Phase 9 VLM takes ~1 hour on a 4 GB GPU)
  python run_all.py --skip_vlm      # everything except the Ollama VLM stage
  python run_all.py --from 7        # resume from a phase

Requires Ollama running locally with `ollama pull qwen2.5vl:3b` for Phase 9.
"""

import argparse
import subprocess
import sys
import time

STAGES = [
    (1, "Phases 1-2: ingestion, validation, leak-free splits", ["run_pipeline.py"]),
    (3, "Phase 3: frozen ViT baseline", ["run_baseline.py"]),
    (4, "Phase 4: preprocessing & sliding-window inspection", ["run_preprocessing.py", "--max_images", "5"]),
    (5, "Phase 5: open-vocabulary motif detection (all paintings)", ["run_motif_detection.py", "--min_confidence", "0.15"]),
    (6, "Phase 6: dual-branch feature profile", ["run_feature_fusion.py", "--max_images", "60"]),
    (7, "Phase 7: multi-head classifier on fused features", ["run_multihead_training.py", "--epochs", "100", "--force_recompute"]),
    (8, "Phase 8: temperature calibration", ["run_calibration.py"]),
    (9, "Phase 9: Qwen VLM judge on the held-out test split", ["run_vlm_reasoning.py"]),
    (10, "Phase 10: final evaluation & error analysis", ["run_evaluation.py"]),
]


def main():
    parser = argparse.ArgumentParser(description="Run the full Mughal/Ragamala pipeline")
    parser.add_argument("--from", dest="start", type=int, default=1, help="First phase to run")
    parser.add_argument("--skip_vlm", action="store_true", help="Skip the Phase 9 Ollama VLM stage")
    args = parser.parse_args()

    for phase, title, cmd in STAGES:
        if phase < args.start or (phase == 9 and args.skip_vlm):
            continue
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, *cmd])
        if result.returncode != 0:
            sys.exit(f"{cmd[0]} failed (exit {result.returncode}). Fix it and resume with: python run_all.py --from {phase}")
        print(f"[done in {time.time() - t0:.0f}s]", flush=True)

    print("\nAll phases complete. Start with reports/final_evaluation_report.md")


if __name__ == "__main__":
    main()
