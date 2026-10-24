"""
Multimodal LLM / VLM Judge Reasoning Engine.
Constructs grounded prompts with full image, region crops, OCR text, classifier probabilities,
and retrieved art-historical knowledge, enforcing structured JSON output and region citation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

from knowledge_base.retriever import ArtKnowledgeRetriever
from src.vlm_client import MockQwenVLMClient, QwenOllamaVLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert art historian, Sanskrit iconographer, and connoisseur specializing in Indian Miniature Paintings (Mughal, Rajput, Pahari, and Ragamala traditions).

Your responsibility is to analyze the provided painting image, region crops, extracted OCR inscriptions, statistical classifier predictions, and retrieved art-historical knowledge to provide a grounded, rigorous judgment.

STRICT CONSTRAINTS & REASONING RULES:
1. OUTPUT SCHEMA: You MUST respond ONLY with a single valid JSON object adhering strictly to the JSON schema below. Do not wrap in conversational preamble or append extra text outside the JSON.
2. GROUNDED EVIDENCE: Base your identification strictly on visible visual evidence and the retrieved Sanskrit Dhyana verses and school/period profiles. Do not invent details.
3. REGION-SPECIFIC EVIDENCE: For every identification, you MUST cite observable features in specific regions (e.g., "central_nayika", "crop_1 (instrument)", "upper_sky_monsoon_clouds", "architectural_pavilion").
4. UNCERTAINTY MANDATE: When visual evidence is thin, damaged, ambiguous, or contradictory, you MUST explicitly output "uncertain" for that field, set confidence <= 0.50, and state what vital iconographic emblem is missing. NEVER guess.
5. CLASSIFIER PROBABILITIES VISIBILITY: The statistical vision classifiers have predicted top-3 candidate classes with probabilities. You must examine these candidates against the visual evidence and explain why you agree with or refute their ranking.
6. CONFLICT DETECTION & CROSS-CHECK: If your visual judgment disagrees with the classifier's top prediction, or if your confidence is below 0.70, you MUST set "route_to_expert_review": true and articulate the precise rationale for human expert review.

JSON OUTPUT SCHEMA:
{
  "predicted_raga": {
    "value": "Raga name (e.g. 'Todi', 'Bhairava', 'Hindol') or 'uncertain'",
    "confidence": 0.0 to 1.0,
    "iconographic_evidence": ["List of confirmed motifs seen in image/crops matching Dhyana verse"]
  },
  "predicted_school": {
    "value": "School name (e.g. 'Mughal', 'Mewar', 'Kangra', 'Bundi') or 'uncertain'",
    "confidence": 0.0 to 1.0,
    "stylistic_evidence": ["Brushwork, facial anatomy, coloring, or border characteristics"]
  },
  "predicted_period": {
    "value": "Period name (e.g. '16th Century', 'Early 17th Century', '18th Century') or 'uncertain'",
    "confidence": 0.0 to 1.0,
    "evidence": ["Historical markers, spatial perspective, or pigment style"]
  },
  "predicted_artist": {
    "value": "Artist name (e.g. 'Basawan', 'Sahibdin', 'Manohar') or 'uncertain'",
    "confidence": 0.0 to 1.0,
    "evidence": ["Stylistic signature or inscribed attribution"]
  },
  "authenticity_score": 0 to 100,
  "region_evidence": [
    {
      "region": "Specific crop or coordinate description (e.g. 'crop_1 (nayika)', 'upper_sky')",
      "observation": "Exact visual elements observed",
      "supports": "Target attribute supported (e.g. 'Todi Raga iconographic match')"
    }
  ],
  "cross_check": {
    "agrees_with_classifiers": true or false,
    "conflict_summary": "Summary of agreement or dispute between vision classifier and VLM",
    "route_to_expert_review": true or false,
    "review_reason": "Specific trigger reason if routed to human review, otherwise null"
  },
  "synthesis_rationale": "Comprehensive art-historical explanation connecting Dhyana verses and visual evidence."
}
"""


EVIDENCE_KEYS = {
    "raga": "iconographic_evidence",
    "school": "stylistic_evidence",
    "period": "evidence",
    "artist": "evidence",
}


def build_response_schema(label_options: Dict[str, List[str]]) -> Dict:
    """JSON schema mirroring SYSTEM_PROMPT's output format, with label values restricted to the dataset vocabulary."""
    short_str = {"type": "string", "maxLength": 200}
    str_list = {"type": "array", "items": short_str, "maxItems": 3}
    props = {}
    for t in ("raga", "school", "period", "artist"):
        value = {"type": "string"}
        if label_options.get(t):
            value["enum"] = list(label_options[t]) + ["uncertain"]
        props[f"predicted_{t}"] = {
            "type": "object",
            "properties": {"value": value, "confidence": {"type": "number"}, EVIDENCE_KEYS[t]: str_list},
            "required": ["value", "confidence", EVIDENCE_KEYS[t]],
        }
    props["authenticity_score"] = {"type": "integer"}
    props["region_evidence"] = {
        "type": "array",
        "maxItems": 3,
        "items": {
            "type": "object",
            "properties": {"region": short_str, "observation": short_str, "supports": short_str},
            "required": ["region", "observation", "supports"],
        },
    }
    props["cross_check"] = {
        "type": "object",
        "properties": {
            "agrees_with_classifiers": {"type": "boolean"},
            "conflict_summary": short_str,
            "route_to_expert_review": {"type": "boolean"},
            "review_reason": short_str,
        },
        "required": ["agrees_with_classifiers", "conflict_summary", "route_to_expert_review", "review_reason"],
    }
    props["synthesis_rationale"] = {"type": "string", "maxLength": 600}
    return {"type": "object", "properties": props, "required": list(props)}


def _field(result: Dict, key: str) -> Tuple[str, float]:
    """Read a predicted_* field's (lower-cased value, confidence), tolerating malformed VLM output."""
    entry = result.get(key)
    if not isinstance(entry, dict):
        entry = {"value": "uncertain", "confidence": 0.0}
        result[key] = entry
    value = str(entry.get("value") or "uncertain").strip()
    try:
        conf = float(entry.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return value.lower(), conf


class VLMJudge:
    """
    Multimodal VLM Judge that orchestrates retrieval, prompt assembly,
    Qwen VLM reasoning, and structured output parsing.
    """

    def __init__(
        self,
        client: Optional[Union[QwenOllamaVLMClient, MockQwenVLMClient]] = None,
        retriever: Optional[ArtKnowledgeRetriever] = None,
        confidence_threshold: float = 0.70,
        label_options: Optional[Dict[str, List[str]]] = None,
    ):
        self.client = client or QwenOllamaVLMClient()
        self.retriever = retriever or ArtKnowledgeRetriever()
        self.confidence_threshold = confidence_threshold
        # Allowed answer vocabulary per target (the dataset's label set), so answers are scoreable
        self.label_options = label_options or {}

    def build_user_prompt(
        self,
        image_metadata: Dict,
        classifier_top3: Dict[str, List[Tuple[str, float]]],
        retrieved_dossier_text: str,
        detected_motifs: Optional[List[str]] = None,
        ocr_text: str = "",
        crop_descriptions: Optional[List[Dict]] = None,
    ) -> str:
        """Construct the comprehensive prompt integrating all multimodal signals."""
        motifs_str = ", ".join(detected_motifs) if detected_motifs else "No automated motifs pre-extracted."
        ocr_str = f'"{ocr_text.strip()}"' if ocr_text and ocr_text.strip() else "No readable inscription / signature detected."

        # Format Classifier Top-3 with probabilities
        clf_sections = []
        for target, top_list in classifier_top3.items():
            preds_str = ", ".join([f"{cls_name} ({prob*100:.1f}%)" for cls_name, prob in top_list])
            clf_sections.append(f"- **{target.capitalize()} Top Candidates**: {preds_str}")
        classifiers_text = "\n".join(clf_sections) or (
            "Withheld for this analysis. Judge from the image, motif signals and retrieved knowledge alone."
        )

        # Format Crop Descriptions
        crops_text = "Full image provided."
        if crop_descriptions:
            crops_lines = []
            for idx, c in enumerate(crop_descriptions):
                crops_lines.append(f"  * Crop {idx + 1} ({c.get('label', 'region')}): Bounding Box {c.get('bbox', 'N/A')}")
            crops_text = "\n".join(crops_lines)

        options_text = ""
        if self.label_options:
            lines = [
                f"- **{t.capitalize()}**: " + " | ".join(f'"{o}"' for o in opts) + ' | "uncertain"'
                for t, opts in self.label_options.items()
            ]
            options_text = (
                "\n### ALLOWED ANSWER LABELS (use one of these values verbatim for each field):\n"
                + "\n".join(lines)
                + '\nUse "Non-Ragamala" for the raga field when the painting is not a Ragamala illustration. '
                "Most paintings in this collection are court, narrative, portrait or manuscript scenes, not Ragamalas: "
                "only choose a raga label when the image itself shows a Ragamala composition (e.g. a musical-mode "
                "personification, often with an inscription panel naming the raga). The retrieved Dhyana verses are "
                "reference material, not evidence that the painting is a Ragamala.\n"
            )

        prompt = f"""### CASE FILE FOR ANALYSIS: {image_metadata.get('image_id', 'painting')}
File Name: {image_metadata.get('file_name', 'unknown.jpg')}
Collection Series Hint: {image_metadata.get('collection_id', 'Unknown Collection')}
{options_text}
### EXTRACTED OCR INSCRIPTION / TEXT:
{ocr_str}

### AUTOMATED MOTIF DETECTOR SIGNALS:
{motifs_str}

### INSPECTED CROPS / REGIONS:
{crops_text}

### STATISTICAL CLASSIFIER PREDICTIONS (FROZEN VIT BASELINE):
{classifiers_text}

### GROUNDED ART-HISTORICAL KNOWLEDGE RETRIEVED FOR THIS CASE:
{retrieved_dossier_text}

---
Perform a thorough iconographic and stylistic analysis following the System Instructions.
Identify the Raga, School, Period, and Artist attribution, cite visible region evidence, deliberate on the classifier probabilities, check for conflicts, and output strictly the required JSON object.
"""
        return prompt

    def evaluate_case(
        self,
        image_path: Union[str, Path],
        classifier_top3: Dict[str, List[Tuple[str, float]]],
        image_metadata: Optional[Dict] = None,
        crop_images: Optional[List[Union[str, Path, Image.Image]]] = None,
        crop_descriptions: Optional[List[Dict]] = None,
        detected_motifs: Optional[List[str]] = None,
        ocr_text: str = "",
        retrieval_candidates: Optional[Dict[str, List[str]]] = None,
    ) -> Dict:
        """
        Executes end-to-end multimodal VLM judgment on a painting.
        `retrieval_candidates` overrides the classes used for knowledge retrieval (e.g. every
        allowed label, unranked, when classifier predictions are hidden from the VLM).
        """
        img_p = Path(image_path)
        metadata = image_metadata or {"image_id": img_p.stem, "file_name": img_p.name}

        # 1. Retrieve grounded knowledge
        retrieval_keys = (
            {t: [(c, 0.0) for c in cands] for t, cands in retrieval_candidates.items()}
            if retrieval_candidates else classifier_top3
        )
        dossier = self.retriever.assemble_retrieval_dossier(
            classifier_top3=retrieval_keys,
            detected_motifs=detected_motifs,
            ocr_text=ocr_text,
        )
        dossier_text = self.retriever.format_dossier_for_prompt(dossier)

        # 2. Build User Prompt
        prompt = self.build_user_prompt(
            image_metadata=metadata,
            classifier_top3=classifier_top3,
            retrieved_dossier_text=dossier_text,
            detected_motifs=detected_motifs,
            ocr_text=ocr_text,
            crop_descriptions=crop_descriptions,
        )

        # 3. Assemble images (full image + region crops)
        images_to_send = [img_p]
        if crop_images:
            images_to_send.extend(crop_images)

        # 4. Generate reasoning via VLM
        raw_response = self.client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            images=images_to_send,
            temperature=0.1,
            format_schema=build_response_schema(self.label_options) if self.label_options else None,
        )

        # 5. Parse and post-process JSON response
        parsed_result = self._parse_json_response(raw_response)

        # 6. Apply deterministic cross-check verification
        parsed_result = self._enforce_cross_check(
            result=parsed_result,
            classifier_top3=classifier_top3,
            confidence_threshold=self.confidence_threshold,
        )

        return parsed_result

    def _parse_json_response(self, text: str) -> Dict:
        """Robustly extract and validate JSON from VLM output."""
        text = text.strip()
        # Remove code markdown fences if present
        if "```json" in text:
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Could not parse VLM response as direct JSON: {e}. Raw response snippet: {text[:200]}")
            # Fallback safe dictionary
            return {
                "predicted_raga": {"value": "uncertain", "confidence": 0.3, "iconographic_evidence": ["Malformed VLM output"]},
                "predicted_school": {"value": "uncertain", "confidence": 0.3, "stylistic_evidence": []},
                "predicted_period": {"value": "uncertain", "confidence": 0.3, "evidence": []},
                "predicted_artist": {"value": "uncertain", "confidence": 0.3, "evidence": []},
                "authenticity_score": 50,
                "region_evidence": [],
                "cross_check": {
                    "agrees_with_classifiers": False,
                    "conflict_summary": "Parsing error in VLM response; automatically routed to human expert.",
                    "route_to_expert_review": True,
                    "review_reason": "VLM output failed schema validation",
                },
                "synthesis_rationale": "Raw output was not well-formed JSON.",
            }

    def _enforce_cross_check(
        self,
        result: Dict,
        classifier_top3: Dict[str, List[Tuple[str, float]]],
        confidence_threshold: float,
    ) -> Dict:
        """
        Enforce programmatic cross-check rules:
        - If VLM and classifier top-1 disagree on primary targets (Raga or School),
          or if confidence < threshold, enforce routing to expert review.
        """
        disagreements = []
        low_confidence_reasons = []

        # Check Raga agreement
        vlm_raga, vlm_raga_conf = _field(result, "predicted_raga")
        if "raga" in classifier_top3 and classifier_top3["raga"]:
            clf_raga_top1, clf_raga_prob = classifier_top3["raga"][0]
            if vlm_raga != "uncertain" and vlm_raga != clf_raga_top1.lower():
                disagreements.append(
                    f"Raga conflict: Classifier top-1='{clf_raga_top1}' (p={clf_raga_prob:.2f}) vs VLM='{result['predicted_raga']['value']}' (conf={vlm_raga_conf:.2f})"
                )

        # Raga confidence only matters when either side thinks this is a Ragamala painting
        clf_raga_top1_l = classifier_top3.get("raga", [("", 0.0)])[0][0].lower() if classifier_top3.get("raga") else ""
        raga_relevant = not (vlm_raga == "non-ragamala" and clf_raga_top1_l in ("", "non-ragamala"))
        if raga_relevant and (vlm_raga == "uncertain" or vlm_raga_conf < confidence_threshold):
            low_confidence_reasons.append(f"Raga confidence ({vlm_raga_conf:.2f}) below threshold ({confidence_threshold:.2f})")

        # Check School agreement
        vlm_school, vlm_school_conf = _field(result, "predicted_school")
        if "school" in classifier_top3 and classifier_top3["school"]:
            clf_school_top1, clf_school_prob = classifier_top3["school"][0]
            if vlm_school != "uncertain" and vlm_school != clf_school_top1.lower():
                disagreements.append(
                    f"School conflict: Classifier top-1='{clf_school_top1}' (p={clf_school_prob:.2f}) vs VLM='{result['predicted_school']['value']}' (conf={vlm_school_conf:.2f})"
                )

        if vlm_school == "uncertain" or vlm_school_conf < confidence_threshold:
            low_confidence_reasons.append(f"School confidence ({vlm_school_conf:.2f}) below threshold ({confidence_threshold:.2f})")

        # Determine route to review
        has_conflict = len(disagreements) > 0
        has_low_conf = len(low_confidence_reasons) > 0
        must_review = has_conflict or has_low_conf

        reasons = []
        if has_conflict:
            reasons.extend(disagreements)
        if has_low_conf:
            reasons.extend(low_confidence_reasons)

        cross_check = result.get("cross_check", {})
        cross_check["agrees_with_classifiers"] = not has_conflict
        cross_check["route_to_expert_review"] = must_review
        cross_check["review_reason"] = "; ".join(reasons) if must_review else "Confirmed high confidence agreement."
        cross_check["conflict_summary"] = "; ".join(disagreements) if has_conflict else "Classifiers and VLM agree on primary targets."

        result["cross_check"] = cross_check
        return result
