# =====================================================================
# ANSWER SCORER — fast per-turn LLM scoring with heuristic fallback
# =====================================================================
import json
import re
import logging

from backend.services.llm_router import llm_router
from backend.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)

_SYSTEM = """You are a senior HR interviewer scoring ONE interview answer in real time.
Score strictly for the difficulty level given. Return ONLY valid JSON, no markdown:
{"score": <0-10 float overall>,
 "dimensions": {"relevance": <0-10>, "structure": <0-10>, "depth": <0-10>, "communication": <0-10>},
 "one_line_tip": "<one specific, actionable coaching tip under 18 words, buddy tone>"}
Scoring anchors: 9-10 exceptional (specific, quantified, structured); 7-8 strong;
5-6 adequate but generic; 3-4 weak/vague; 0-2 off-topic, empty, or silent."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp(v, lo=0.0, hi=10.0):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 5.0


class AnswerScorer:
    def __init__(self):
        self._heuristic = ScoringService()

    def score(self, question: str, answer: str, job_role: str, difficulty: str,
              selected_llm: str = None) -> dict:
        try:
            raw = llm_router.generate(
                system_prompt=_SYSTEM,
                user_prompt=(f"Role: {job_role}\nDifficulty: {difficulty}\n"
                             f"Question: {question}\nAnswer: {answer}\n\nScore it."),
                max_tokens=220,
                temperature=0.2,
                selected_llm=selected_llm,
            )
            parsed = self._parse(raw)
            if parsed:
                return parsed
        except Exception as e:
            logger.error(f"[AnswerScorer] LLM scoring failed: {e}")
        return self._fallback(answer)

    def _parse(self, raw: str) -> dict:
        if not raw:
            return None
        m = _JSON_RE.search(raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        dims = data.get("dimensions") or {}
        return {
            "score": _clamp(data.get("score")),
            "dimensions": {
                "relevance": _clamp(dims.get("relevance")),
                "structure": _clamp(dims.get("structure")),
                "depth": _clamp(dims.get("depth")),
                "communication": _clamp(dims.get("communication")),
            },
            "one_line_tip": str(data.get("one_line_tip") or "Keep answers specific and structured."),
            "source": "llm",
        }

    def _fallback(self, answer: str) -> dict:
        h = self._heuristic.calculate_score(answer or "", {})
        dims = h.get("dimensions", {})
        overall = float(h.get("overall_score", 5.0))
        tip = ("Add a concrete example with numbers to lift this answer."
               if overall < 7 else "Solid answer — tighten it by leading with the result.")
        return {
            "score": overall,
            "dimensions": {
                "relevance": float(dims.get("clarity", overall)),
                "structure": float(dims.get("structure", overall)),
                "depth": float(dims.get("confidence", overall)),
                "communication": float(dims.get("fluency", overall)),
            },
            "one_line_tip": tip,
            "source": "heuristic",
        }


answer_scorer = AnswerScorer()
