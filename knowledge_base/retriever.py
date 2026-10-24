"""
Art-Historical Knowledge Base Retriever.
Retrieves Dhyana verses, iconographic markers, school characteristics,
and period contexts for Ragamala and Mughal miniature painting analysis.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).resolve().parent


class ArtKnowledgeRetriever:
    """
    Retrieves domain knowledge from structured art-historical databases
    to ground Multimodal VLM reasoning without relying on ungrounded memory.
    """

    def __init__(self, kb_dir: Union[str, Path] = KB_DIR):
        self.kb_dir = Path(kb_dir)
        self.ragas_db = self._load_json(self.kb_dir / "dhyana_verses_ragas.json").get("ragas", {})
        self.schools_db = self._load_json(self.kb_dir / "schools_characteristics.json").get("schools", {})
        self.periods_db = self._load_json(self.kb_dir / "periods_characteristics.json").get("periods", {})
        logger.info(
            f"Knowledge Base loaded: {len(self.ragas_db)} ragas, "
            f"{len(self.schools_db)} schools, {len(self.periods_db)} periods."
        )

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            logger.warning(f"Knowledge file not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}

    def retrieve_raga_context(
        self,
        candidate_ragas: List[str],
        detected_motifs: Optional[List[str]] = None,
        ocr_text: str = "",
    ) -> List[Dict]:
        """
        Retrieve Dhyana verses and iconographic criteria for candidate ragas,
        calculating motif-overlap evidence scores.
        """
        results = []
        motifs_lower = [m.lower() for m in (detected_motifs or [])]
        ocr_lower = ocr_text.lower()

        # Check candidate list + search if OCR or motifs suggest other ragas
        search_ragas = list(dict.fromkeys(candidate_ragas))

        for raga_key, data in self.ragas_db.items():
            # Check if this raga matches candidates or is explicitly mentioned in OCR
            is_candidate = any(r.lower() == raga_key.lower() for r in search_ragas)
            is_in_ocr = bool(re.search(r"\b" + re.escape(raga_key.lower()) + r"\b", ocr_lower))

            if is_candidate or is_in_ocr:
                # Check motif matches against key iconographic markers
                matched_markers = []
                for marker in data.get("key_iconographic_motifs", []):
                    marker_lower = marker.lower()
                    if any(m in marker_lower for m in motifs_lower):
                        matched_markers.append(marker)

                results.append({
                    "raga": raga_key,
                    "family": data.get("melodic_family"),
                    "prahar_time": data.get("time_prahar"),
                    "season": data.get("season"),
                    "deity": data.get("presiding_deity"),
                    "rasa_mood": data.get("rasa_mood"),
                    "dhyana_verse_sanskrit": data.get("dhyana_verse_sanskrit"),
                    "dhyana_verse_translation": data.get("dhyana_verse_translation"),
                    "required_motifs": data.get("key_iconographic_motifs", []),
                    "matched_detected_motifs": matched_markers,
                    "color_rules": data.get("color_associations", []),
                })

        return results

    def retrieve_school_context(self, candidate_schools: List[str]) -> List[Dict]:
        """Retrieve stylistic signatures and characteristics for candidate schools."""
        results = []
        for school_name, data in self.schools_db.items():
            if any(s.lower() in school_name.lower() or school_name.lower() in s.lower() for s in candidate_schools):
                results.append({
                    "school": school_name,
                    "geography": data.get("geography"),
                    "stylistic_signatures": data.get("stylistic_signatures", []),
                    "color_palette": data.get("color_palette", []),
                    "facial_anatomy": data.get("facial_anatomy"),
                    "notable_masters": data.get("notable_masters", []),
                })
        return results

    def retrieve_period_context(self, candidate_periods: List[str]) -> List[Dict]:
        """Retrieve historical context and stylistic markers for candidate periods."""
        results = []
        for period_name, data in self.periods_db.items():
            if any(p.lower() in period_name.lower() or period_name.lower() in p.lower() for p in candidate_periods):
                results.append({
                    "period": period_name,
                    "historical_context": data.get("historical_context"),
                    "stylistic_markers": data.get("stylistic_markers", []),
                    "pigments_materials": data.get("pigments_materials"),
                })
        return results

    def assemble_retrieval_dossier(
        self,
        classifier_top3: Dict[str, List[Tuple[str, float]]],
        detected_motifs: Optional[List[str]] = None,
        ocr_text: str = "",
    ) -> Dict:
        """
        Assemble comprehensive grounded art-historical retrieval dossier
        based on multi-head classifier candidates, motifs, and inscriptions.
        """
        candidate_ragas = [r[0] for r in classifier_top3.get("raga", [])]
        candidate_schools = [s[0] for s in classifier_top3.get("school", [])]
        candidate_periods = [p[0] for p in classifier_top3.get("period", [])]

        raga_docs = self.retrieve_raga_context(candidate_ragas, detected_motifs, ocr_text)
        school_docs = self.retrieve_school_context(candidate_schools)
        period_docs = self.retrieve_period_context(candidate_periods)

        return {
            "raga_iconography_and_dhyana": raga_docs,
            "school_stylistic_hallmarks": school_docs,
            "period_markers": period_docs,
        }

    def format_dossier_for_prompt(self, dossier: Dict) -> str:
        """Format the retrieval dossier as readable markdown text for VLM injection."""
        sections = []

        # Ragas & Dhyana Verses
        if dossier.get("raga_iconography_and_dhyana"):
            sections.append("### Grounded Ragamala Iconography & Dhyana Verses:")
            for item in dossier["raga_iconography_and_dhyana"]:
                motifs_str = "\n    - ".join(item["required_motifs"])
                matched_str = ", ".join(item["matched_detected_motifs"]) if item["matched_detected_motifs"] else "None explicitly detected"
                sections.append(
                    f"- **{item['raga']}** ({item['family']}):\n"
                    f"  * Time/Season: {item['prahar_time']}, {item['season']}\n"
                    f"  * Presiding Deity/Mood: {item['deity']} ({item['rasa_mood']})\n"
                    f"  * Dhyana Verse: \"{item['dhyana_verse_translation']}\"\n"
                    f"  * Mandatory Visual Motifs:\n    - {motifs_str}\n"
                    f"  * Visual Motif Matches Found: {matched_str}\n"
                    f"  * Color Associations: {', '.join(item['color_rules'])}"
                )

        # Schools
        if dossier.get("school_stylistic_hallmarks"):
            sections.append("\n### Grounded School Characteristics & Styles:")
            for item in dossier["school_stylistic_hallmarks"]:
                signatures_str = "\n    - ".join(item["stylistic_signatures"])
                sections.append(
                    f"- **{item['school']} School** ({item['geography']}):\n"
                    f"  * Stylistic Signatures:\n    - {signatures_str}\n"
                    f"  * Facial Anatomy: {item['facial_anatomy']}\n"
                    f"  * Color Palette: {', '.join(item['color_palette'])}\n"
                    f"  * Notable Masters: {', '.join(item['notable_masters'])}"
                )

        # Periods
        if dossier.get("period_markers"):
            sections.append("\n### Grounded Period Characteristics:")
            for item in dossier["period_markers"]:
                markers_str = "\n    - ".join(item["stylistic_markers"])
                sections.append(
                    f"- **{item['period']}**:\n"
                    f"  * Context: {item['historical_context']}\n"
                    f"  * Stylistic Markers:\n    - {markers_str}\n"
                    f"  * Pigments & Materials: {item['pigments_materials']}"
                )

        return "\n".join(sections)
