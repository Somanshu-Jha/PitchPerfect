import json
from unittest.mock import patch
from backend.services.report_generator import report_generator

SNAP = {
    "integrity_score": 90,
    "violations": [{"type": "TAB_SWITCHED", "severity": "critical", "action": "WARN", "ts_ms": 5000}],
    "answers": [
        {"turn_id": 1, "question": "Tell me about yourself.",
         "answer": "I am a backend dev with 2 years on FastAPI.",
         "score": {"score": 7.0, "dimensions": {"relevance": 8, "structure": 6, "depth": 6, "communication": 8},
                   "one_line_tip": "Lead with impact.", "source": "llm"}},
        {"turn_id": 2, "question": "Describe a hard bug.",
         "answer": "A race condition in our task queue; I bisected and fixed it.",
         "score": {"score": 8.0, "dimensions": {"relevance": 8, "structure": 8, "depth": 8, "communication": 8},
                   "one_line_tip": "Great structure.", "source": "llm"}},
    ],
    "terminated": False, "termination_reason": None,
}
HISTORY = [{"role": "assistant", "content": "Tell me about yourself."},
           {"role": "user", "content": "I am a backend dev..."}]

LLM_REPORT = json.dumps({
    "overall_score": 7.4, "verdict": "Promising candidate, needs sharper storytelling.",
    "dimension_breakdown": {"communication": 8, "content": 7, "structure": 6.5,
                            "confidence": 7, "integrity": 9},
    "per_question": [{"question": "Tell me about yourself.", "score": 7.0,
                      "good": "Clear background", "improve": "Quantify results"}],
    "strengths": ["Concrete debugging story", "Calm delivery", "Relevant stack"],
    "improvement_plan": ["Practice STAR", "Add metrics", "Mock interview weekly"],
})


def test_llm_report_parsed_and_integrity_injected():
    with patch("backend.services.report_generator.llm_router") as m:
        m.generate.return_value = LLM_REPORT
        r = report_generator.generate(HISTORY, SNAP, "Backend Engineer", "intermediate", "hr")
    assert r["source"] == "llm"
    assert r["overall_score"] == 7.4
    assert r["integrity"]["score"] == 90
    assert r["integrity"]["violations"][0]["type"] == "TAB_SWITCHED"


def test_local_fallback_when_llm_dies():
    with patch("backend.services.report_generator.llm_router") as m:
        m.generate.return_value = ""
        r = report_generator.generate(HISTORY, SNAP, "Backend Engineer", "intermediate", "hr")
    assert r["source"] == "local"
    assert r["overall_score"] == 7.5  # mean of 7.0 and 8.0
    assert len(r["per_question"]) == 2
    assert r["integrity"]["terminated"] is False
    assert isinstance(r["strengths"], list) and r["strengths"]
