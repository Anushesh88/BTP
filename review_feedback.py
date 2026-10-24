"""
Phase 10.3 CLI: Human-in-the-Loop expert review & continuous retraining.

  python review_feedback.py pending                 # cases routed to review without feedback yet
  python review_feedback.py show <image_id>         # classifier vs VLM vs current label for one case
  python review_feedback.py correct <image_id> --target school --value Rajasthani --reviewer "Dr. X" --notes "..."
  python review_feedback.py correct <image_id> --target period --unknown --reviewer "Dr. X"
  python review_feedback.py summary                 # feedback log statistics
  python review_feedback.py retrain                 # re-run Phase 3 -> 7 -> 8 on corrected labels

Values use the normalized label vocabulary (e.g. raga "Non-Ragamala", "Shri Raga/Ragini").
Corrections are appended to data/feedback/expert_feedback.jsonl; the latest one per
(image_id, target) wins and is applied automatically whenever the frozen splits are loaded.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.dataset import load_frozen_splits
from src.feedback import CORRECTABLE_TARGETS, FeedbackStore

QUEUE_FILE = Path("reports/expert_review/review_queue.jsonl")


def load_queue():
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def current_labels():
    return pd.concat(load_frozen_splits(), ignore_index=True).set_index("image_id")


def cmd_pending(args, store):
    reviewed = {img for img, _ in store.latest_corrections()}
    pending = [r for r in load_queue() if r["image_id"] not in reviewed]
    print(f"{len(pending)} case(s) awaiting expert review:\n")
    for r in pending:
        print(f"- {r['image_id']}\n    reason: {r['review_reasons'][:160]}")


def cmd_show(args, store):
    labels = current_labels()
    if args.image_id not in labels.index:
        sys.exit(f"Unknown image_id: {args.image_id}")
    row = labels.loc[args.image_id]
    case = next((r for r in load_queue() if r["image_id"] == args.image_id), None)
    print(f"Image: {row['file_path']}\n")
    print(f"{'target':<8} {'current label':<38} {'classifier top-1':<38} VLM")
    for t in CORRECTABLE_TARGETS:
        clf = case["classifier_top3"].get(t, [["-", 0]])[0] if case else ["-", 0]
        vlm = case["vlm_judgment"].get(f"predicted_{t}", {}) if case else {}
        print(f"{t:<8} {str(row[t]):<38} {f'{clf[0]} ({clf[1]:.2f})':<38} {vlm.get('value', '-')} ({vlm.get('confidence', '-')})")
    if case:
        print(f"\nReview reason: {case['review_reasons']}")


def cmd_correct(args, store):
    labels = current_labels()
    if args.image_id not in labels.index:
        sys.exit(f"Unknown image_id: {args.image_id}")
    if args.unknown == (args.value is not None):
        sys.exit("Pass exactly one of --value or --unknown.")
    previous = labels.loc[args.image_id, args.target]
    store.record(
        image_id=args.image_id,
        target=args.target,
        corrected_value=None if args.unknown else args.value,
        reviewer=args.reviewer,
        notes=args.notes,
        previous_value=None if pd.isna(previous) else str(previous),
    )
    print(f"Saved: {args.image_id}.{args.target}: {previous!r} -> {'unknown' if args.unknown else args.value!r}")


def cmd_summary(args, store):
    entries = store.load()
    print(f"{len(entries)} feedback entries, {len(store.latest_corrections())} active corrections.")
    if entries:
        df = pd.DataFrame(entries)
        print("\nBy target:\n" + df["target"].value_counts().to_string())
        print("\nBy reviewer:\n" + df["reviewer"].value_counts().to_string())


def cmd_retrain(args, store):
    n = len(store.latest_corrections())
    print(f"Retraining on frozen splits with {n} active expert correction(s)...")
    for script, extra in [("run_baseline.py", []), ("run_multihead_training.py", ["--epochs", "100"]),
                          ("run_calibration.py", []), ("run_evaluation.py", [])]:
        print(f"\n>>> {script}")
        subprocess.run([sys.executable, script, *extra], check=True)
    print("\nDone. See reports/final_evaluation_report.md")


def main():
    parser = argparse.ArgumentParser(description="Expert review & feedback loop (Phase 10.3)")
    parser.add_argument("--feedback_file", default="data/feedback/expert_feedback.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pending")
    p = sub.add_parser("show")
    p.add_argument("image_id")
    p = sub.add_parser("correct")
    p.add_argument("image_id")
    p.add_argument("--target", required=True, choices=CORRECTABLE_TARGETS)
    p.add_argument("--value", help="Corrected label")
    p.add_argument("--unknown", action="store_true", help="Mark the label as unknown")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--notes", default="")
    sub.add_parser("summary")
    sub.add_parser("retrain")

    args = parser.parse_args()
    store = FeedbackStore(args.feedback_file)
    {"pending": cmd_pending, "show": cmd_show, "correct": cmd_correct,
     "summary": cmd_summary, "retrain": cmd_retrain}[args.command](args, store)


if __name__ == "__main__":
    main()
