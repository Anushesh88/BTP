"""
Unit and integration tests for Multimodal VLM Reasoning, Retrieval Grounding,
Cross-Check, and Expert Review Queue.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_base.retriever import ArtKnowledgeRetriever
from src.cross_check import ExpertReviewQueue
from src.test_reasoning import evaluate_reasoning_dataset, score_evidence_citation
from src.vlm_client import MockQwenVLMClient
from src.vlm_judge import VLMJudge


def test_knowledge_base_retrieval():
    retriever = ArtKnowledgeRetriever()

    # Test Raga Retrieval
    raga_res = retriever.retrieve_raga_context(
        candidate_ragas=["Todi", "Bhairava"],
        detected_motifs=["veena", "deer"],
        ocr_text="Raga Todi painting",
    )
    assert len(raga_res) >= 1
    todi_entry = next((r for r in raga_res if r["raga"] == "Todi"), None)
    assert todi_entry is not None
    assert "dhyana_verse_translation" in todi_entry
    assert len(todi_entry["required_motifs"]) > 0

    # Test School Retrieval
    school_res = retriever.retrieve_school_context(["Mughal", "Mewar"])
    assert len(school_res) >= 1
    mughal_entry = next((s for s in school_res if s["school"] == "Mughal"), None)
    assert mughal_entry is not None
    assert len(mughal_entry["stylistic_signatures"]) > 0

    # Test Dossier formatting
    clf_top3 = {
        "raga": [("Todi", 0.75), ("Bhairava", 0.15)],
        "school": [("Mughal", 0.80), ("Mewar", 0.10)],
        "period": [("17th Century", 0.70)],
    }
    dossier = retriever.assemble_retrieval_dossier(clf_top3, detected_motifs=["veena"])
    prompt_text = retriever.format_dossier_for_prompt(dossier)
    assert "Grounded Ragamala Iconography" in prompt_text
    assert "Todi" in prompt_text


def test_vlm_judge_and_prompt_assembly():
    retriever = ArtKnowledgeRetriever()
    client = MockQwenVLMClient()
    judge = VLMJudge(client=client, retriever=retriever, confidence_threshold=0.70)

    clf_top3 = {
        "raga": [("Todi", 0.85), ("Bhairava", 0.10)],
        "school": [("Mughal", 0.75), ("Mewar", 0.15)],
    }

    # Evaluate mock case
    result = judge.evaluate_case(
        image_path=Path("tests/test_vlm_reasoning.py"),  # dummy file path
        classifier_top3=clf_top3,
        detected_motifs=["veena", "deer"],
        ocr_text="Todi Ragini",
    )

    assert "predicted_raga" in result
    assert result["predicted_raga"]["value"] == "Todi"
    assert "region_evidence" in result
    assert len(result["region_evidence"]) > 0
    assert "cross_check" in result
    assert "agrees_with_classifiers" in result["cross_check"]


def test_cross_check_and_expert_review_queue(tmp_path):
    queue = ExpertReviewQueue(queue_dir=tmp_path / "review")

    clf_top3 = {
        "raga": [("Bhairava", 0.45), ("Malkaus", 0.35)],
        "school": [("Mewar", 0.50)],
    }

    vlm_judgment = {
        "predicted_raga": {
            "value": "Todi",
            "confidence": 0.85,
            "iconographic_evidence": ["Veena", "Deer"],
        },
        "predicted_school": {
            "value": "Mewar",
            "confidence": 0.70,
            "stylistic_evidence": ["Red ground"],
        },
        "region_evidence": [
            {"region": "crop_1", "observation": "Nayika playing veena", "supports": "Todi"}
        ],
        "cross_check": {
            "agrees_with_classifiers": False,
            "route_to_expert_review": True,
            "review_reason": "Raga conflict: Classifier top-1='Bhairava' vs VLM='Todi'",
            "conflict_summary": "Dispute on iconography",
        },
        "synthesis_rationale": "Clear Todi iconography observed despite classifier confusion.",
    }

    card_file = queue.enqueue(
        image_id="painting_disputed_001",
        image_path="test_painting.jpg",
        classifier_top3=clf_top3,
        vlm_judgment=vlm_judgment,
        ground_truth={"raga": "Todi", "school": "Mewar"},
    )

    assert card_file.exists()
    assert queue.queue_file.exists()

    summary = queue.generate_queue_summary()
    assert summary["total_queued"] == 1
    assert Path(summary["summary_file"]).exists()


def test_evidence_scoring_and_reasoning_benchmark(tmp_path):
    retriever = ArtKnowledgeRetriever()

    mock_judgment = {
        "predicted_raga": {
            "value": "Todi",
            "confidence": 0.90,
            "iconographic_evidence": ["Young solitary nayika", "Playing stringed veena", "Wild deer gathered peacefully"],
        },
        "region_evidence": [
            {"region": "center", "observation": "Spotted deer listening to veena music", "supports": "Todi"}
        ],
        "cross_check": {"agrees_with_classifiers": True, "route_to_expert_review": False},
    }

    score, valid, spurious = score_evidence_citation(
        vlm_judgment=mock_judgment,
        ground_truth_target="Todi",
        target_type="raga",
        retriever=retriever,
    )

    assert score > 0.5
    assert len(valid) >= 2

    # Benchmark test
    eval_cases = [
        {
            "image_id": "case_01",
            "ground_truth": {"raga": "Todi", "school": "Mughal"},
            "vlm_judgment": mock_judgment,
            "classifier_top3": {"raga": [("Todi", 0.85)]},
        }
    ]

    metrics = evaluate_reasoning_dataset(eval_cases, output_dir=tmp_path / "reports")
    assert metrics["total_paintings_evaluated"] == 1
    assert metrics["answer_accuracy"]["raga"] == 1.0
    assert (tmp_path / "reports" / "vlm_reasoning_benchmark.md").exists()


if __name__ == "__main__":
    import traceback

    print("Running Multimodal VLM Reasoning Test Suite...")
    tests = [
        ("test_knowledge_base_retrieval", test_knowledge_base_retrieval, []),
        ("test_vlm_judge_and_prompt_assembly", test_vlm_judge_and_prompt_assembly, []),
    ]

    passed = 0
    failed = 0

    for name, func, args in tests:
        try:
            func(*args)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    tmp_tests = [
        ("test_cross_check_and_expert_review_queue", test_cross_check_and_expert_review_queue),
        ("test_evidence_scoring_and_reasoning_benchmark", test_evidence_scoring_and_reasoning_benchmark),
    ]

    for name, func in tmp_tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                func(Path(td))
                print(f"  [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
