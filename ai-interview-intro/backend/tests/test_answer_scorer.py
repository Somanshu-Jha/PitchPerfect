import json
from unittest.mock import patch
from backend.services.answer_scorer import answer_scorer


GOOD_JSON = json.dumps({
    "score": 7.5,
    "dimensions": {"relevance": 8, "structure": 7, "depth": 7, "communication": 8},
    "one_line_tip": "Quantify the impact of your project next time.",
})


def test_llm_path_parses_json():
    with patch("backend.services.answer_scorer.llm_router") as m:
        m.generate.return_value = GOOD_JSON
        r = answer_scorer.score("Tell me about a project.",
                                "I built a REST API with FastAPI serving 1k users.",
                                "Backend Engineer", "intermediate")
    assert r["source"] == "llm"
    assert r["score"] == 7.5
    assert r["dimensions"]["relevance"] == 8.0
    assert "Quantify" in r["one_line_tip"]


def test_llm_json_wrapped_in_prose_still_parses():
    with patch("backend.services.answer_scorer.llm_router") as m:
        m.generate.return_value = f"Sure! Here is the evaluation:\n```json\n{GOOD_JSON}\n```"
        r = answer_scorer.score("Q", "A long enough answer about deploying models to production systems.", "MLE", "advanced")
    assert r["source"] == "llm" and r["score"] == 7.5


def test_fallback_on_llm_failure():
    with patch("backend.services.answer_scorer.llm_router") as m:
        m.generate.return_value = ""
        r = answer_scorer.score("Tell me about yourself.",
                                "My name is Dev and I am studying engineering at my university with python skills.",
                                "SWE", "beginner")
    assert r["source"] == "heuristic"
    assert 1.0 <= r["score"] <= 10.0
    assert set(r["dimensions"]) == {"relevance", "structure", "depth", "communication"}
    assert isinstance(r["one_line_tip"], str) and r["one_line_tip"]


def test_score_clamped():
    bad = json.dumps({"score": 37, "dimensions": {"relevance": 99, "structure": -2,
                                                  "depth": 5, "communication": 5},
                      "one_line_tip": "x"})
    with patch("backend.services.answer_scorer.llm_router") as m:
        m.generate.return_value = bad
        r = answer_scorer.score("Q", "A", "SWE", "faang")
    assert r["score"] == 10.0
    assert r["dimensions"]["relevance"] == 10.0
    assert r["dimensions"]["structure"] == 0.0
