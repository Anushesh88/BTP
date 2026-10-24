"""
Cross-Check & Expert Review Routing Module.
Detects statistical vs multimodal disagreements, flags uncertain cases,
and manages the Human-in-the-Loop (HITL) expert review queue.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ExpertReviewQueue:
    """
    Manages the queue of painting cases routed to human art-historian experts
    when statistical classifiers and the VLM judge disagree or report low confidence.
    """

    def __init__(self, queue_dir: Union[str, Path] = "reports/expert_review"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.queue_dir / "review_queue.jsonl"
        self.summary_file = self.queue_dir / "review_summary.md"

    def enqueue(
        self,
        image_id: str,
        image_path: str,
        classifier_top3: Dict,
        vlm_judgment: Dict,
        ground_truth: Optional[Dict] = None,
    ) -> Path:
        """
        Record a flagged case in the review queue and generate a formatted review card.
        """
        cross_check = vlm_judgment.get("cross_check", {})
        reasons = cross_check.get("review_reason", "Flagged for expert inspection")

        record = {
            "timestamp": datetime.now().isoformat(),
            "image_id": image_id,
            "image_path": image_path,
            "review_reasons": reasons,
            "conflict_summary": cross_check.get("conflict_summary", ""),
            "vlm_judgment": vlm_judgment,
            "classifier_top3": classifier_top3,
            "ground_truth": ground_truth or {},
            "status": "pending_human_review",
        }

        # Append to JSONL queue
        with open(self.queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Generate individual Markdown review card
        card_path = self.queue_dir / f"review_card_{image_id}.md"
        self._generate_review_card(record, card_path)

        logger.info(f"Enqueued painting {image_id} to expert review: {card_path}")
        return card_path

    def _generate_review_card(self, record: Dict, card_path: Path) -> None:
        """Create a human-readable review card for expert curators."""
        vlm = record["vlm_judgment"]
        clf = record["classifier_top3"]
        gt = record["ground_truth"]

        md = [
            f"# Expert Review Card: {record['image_id']}",
            f"- **Timestamp**: {record['timestamp']}",
            f"- **Image Path**: `{record['image_path']}`",
            f"- **Status**: `{record['status']}`",
            "",
            "## Flagged Trigger Reason",
            f"> [!WARNING]",
            f"> {record['review_reasons']}",
            "",
            "## Comparison: Classifier vs VLM Judge",
            "",
            "| Target | Statistical Vision Classifier Top-1 | Multimodal VLM Prediction | Ground Truth (if available) | Agreement |",
            "| --- | --- | --- | --- | --- |",
        ]

        targets = ["raga", "school", "period", "artist"]
        for t in targets:
            c_top = clf.get(t, [("None", 0.0)])[0]
            c_str = f"{c_top[0]} ({c_top[1]*100:.1f}%)" if isinstance(c_top, (list, tuple)) else str(c_top)
            v_val = vlm.get(f"predicted_{t}", {}).get("value", "N/A")
            v_conf = vlm.get(f"predicted_{t}", {}).get("confidence", 0.0)
            v_str = f"{v_val} ({v_conf*100:.1f}%)"
            gt_str = gt.get(t, "Unlabeled")

            agreed = "MATCH" if (v_val != "uncertain" and v_val.lower() == str(c_top[0]).lower()) else "DISAGREE"
            md.append(f"| **{t.capitalize()}** | {c_str} | {v_str} | {gt_str} | {agreed} |")

        md.extend([
            "",
            "## Cited Region Evidence (VLM)",
            "",
        ])

        for reg in vlm.get("region_evidence", []):
            md.append(f"- **{reg.get('region')}**: {reg.get('observation')} *(Supports: {reg.get('supports')})*")

        md.extend([
            "",
            "## Synthesis & Rationale",
            f"{vlm.get('synthesis_rationale', 'N/A')}",
            "",
            "---",
            "## Human Curator Action",
            "- [ ] Confirm VLM Identification",
            "- [ ] Confirm Statistical Classifier Top-1",
            "- [ ] Provide Correct Identification: _______________",
            "- [ ] Notes / Feedback: ____________________________",
        ])

        with open(card_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")

    def generate_queue_summary(self) -> Dict:
        """Generate statistical summary of all queued cases."""
        if not self.queue_file.exists():
            return {"total_queued": 0, "pending": 0}

        records = []
        with open(self.queue_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        total = len(records)
        df_queue = pd.DataFrame(records)

        md_summary = [
            "# Expert Review Queue Summary",
            "",
            f"- **Total Flagged Cases**: {total}",
            f"- **Pending Human Action**: {total}",
            f"- **Generated at**: {datetime.now().isoformat()}",
            "",
            "## Queued Cases List",
            "",
            "| Image ID | Review Reason | VLM Raga | Classifier Top-1 Raga |",
            "| --- | --- | --- | --- |",
        ]

        for r in records:
            v_raga = r["vlm_judgment"].get("predicted_raga", {}).get("value", "N/A")
            c_top = r["classifier_top3"].get("raga", [("N/A", 0)])[0]
            c_name = c_top[0] if isinstance(c_top, (list, tuple)) else str(c_top)
            md_summary.append(f"| [`{r['image_id']}`](review_card_{r['image_id']}.md) | {r['review_reasons']} | {v_raga} | {c_name} |")

        with open(self.summary_file, "w", encoding="utf-8") as sf:
            sf.write("\n".join(md_summary) + "\n")

        return {"total_queued": total, "summary_file": str(self.summary_file)}
