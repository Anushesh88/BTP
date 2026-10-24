"""
Phase 10.3: Human-in-the-Loop (HITL) Feedback Engine.

Art-historian corrections are stored as an append-only JSONL log
(data/feedback/expert_feedback.jsonl). The latest correction per (image_id, target)
wins. Corrections are applied when the frozen splits are loaded, so re-running
Phase 3 / 7 / 8 retrains and recalibrates on the corrected labels.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_FILE = Path("data/feedback/expert_feedback.jsonl")
CORRECTABLE_TARGETS = ("raga", "school", "period", "artist")


class FeedbackStore:
    """Append-only store of expert label corrections and review decisions."""

    def __init__(self, feedback_file: Union[str, Path] = DEFAULT_FEEDBACK_FILE):
        self.feedback_file = Path(feedback_file)

    def record(
        self,
        image_id: str,
        target: str,
        corrected_value: Optional[str],
        reviewer: str,
        notes: str = "",
        previous_value: Optional[str] = None,
    ) -> Dict:
        """
        Record one correction. `corrected_value=None` marks the label as unknown
        (excluded from training/evaluation for that target).
        """
        if target not in CORRECTABLE_TARGETS:
            raise ValueError(f"target must be one of {CORRECTABLE_TARGETS}, got '{target}'")
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_id": image_id,
            "target": target,
            "previous_value": previous_value,
            "corrected_value": corrected_value,
            "reviewer": reviewer,
            "notes": notes,
        }
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Recorded feedback: {image_id}.{target} -> {corrected_value!r} ({reviewer})")
        return entry

    def load(self) -> List[Dict]:
        if not self.feedback_file.exists():
            return []
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def latest_corrections(self) -> Dict[tuple, Optional[str]]:
        """{(image_id, target): corrected_value} keeping the most recent entry."""
        latest = {}
        for e in self.load():
            latest[(e["image_id"], e["target"])] = e["corrected_value"]
        return latest

    def apply_to(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` with expert corrections applied to its target columns."""
        corrections = self.latest_corrections()
        if not corrections:
            return df
        df = df.copy()
        index_by_id = {img_id: i for i, img_id in zip(df.index, df["image_id"])}
        applied = 0
        for (img_id, target), value in corrections.items():
            if img_id in index_by_id and target in df.columns:
                df.at[index_by_id[img_id], target] = np.nan if value is None else value
                applied += 1
        if applied:
            logger.info(f"Applied {applied} expert label corrections from {self.feedback_file}")
        return df
