# =====================================================================
# REPORT GENERATOR — post-call interview report (LLM + local fallback)
# =====================================================================
import json
import re
import logging

from backend.services.llm_router import llm_router

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = """You are a senior HR mentor writing a post-interview debrief for a candidate you
just interviewed. Be honest, warm, specific and role-focused — a buddy who wants them hired.
Return ONLY valid JSON, no markdown:
{"overall_score": <0-10 float>,
 "verdict": "<2 sentence hire-signal summary, buddy tone>",
 "dimension_breakdown": {"communication": <0-10>, "content": <0-10>, "structure": <0-10>,
                          "confidence": <0-10>, "integrity": <0-10>},
 "per_question": [{"question": "...", "score": <0-10>,
                   "good": "<what worked, 1 sentence>",
                   "improve": "<what a stronger answer looks like, 1-2 sentences>"}],
 "strengths": ["<3 specific strengths>"],
 "improvement_plan": ["<3-5 concrete practice actions for the TARGET ROLE>"]}"""


class ReportGenerator:
    def generate(self, history: list, snapshot: dict, job_role: str,
                 difficulty: str, mode: str, selected_llm: str = "kimi") -> dict:
        integrity = {
            "score": snapshot.get("integrity_score", 100),
            "violations": snapshot.get("violations", []),
            "terminated": snapshot.get("terminated", False),
            "termination_reason": snapshot.get("termination_reason"),
        }
        answers = snapshot.get("answers", [])
        transcript = "\n".join(
            f"{'Interviewer' if t.get('role') == 'assistant' else 'Candidate'}: {t.get('content', '')}"
            for t in history
        )
        per_answer = "\n".join(
            f"Q{i + 1}: {a['question']}\nAnswer: {a['answer']}\n"
            f"Live score: {a['score']['score']}/10 (tip: {a['score']['one_line_tip']})"
            for i, a in enumerate(answers)
        )
        violations_str = "; ".join(
            f"{v['type']} ({v['action']})" for v in integrity["violations"]
        ) or "none — clean session"

        try:
            raw = llm_router.generate(
                system_prompt=_SYSTEM,
                user_prompt=(f"Target role: {job_role}\nDifficulty: {difficulty}\nRound: {mode}\n"
                             f"Integrity score: {integrity['score']}/100, violations: {violations_str}\n\n"
                             f"PER-ANSWER DATA:\n{per_answer}\n\nFULL TRANSCRIPT:\n{transcript}\n\n"
                             "Write the debrief JSON now."),
                max_tokens=1400,
                temperature=0.5,
                selected_llm=selected_llm,
            )
            parsed = self._parse(raw)
            if parsed:
                parsed["integrity"] = integrity
                parsed["source"] = "llm"
                return parsed
        except Exception as e:
            logger.error(f"[ReportGenerator] LLM report failed: {e}")
        return self._local(answers, integrity)

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
        if "overall_score" not in data or "per_question" not in data:
            return None
        return data

    def _local(self, answers: list, integrity: dict) -> dict:
        scores = [a["score"]["score"] for a in answers] or [5.0]
        overall = round(sum(scores) / len(scores), 1)
        dims = ["relevance", "structure", "depth", "communication"]
        avg = {d: round(sum(a["score"]["dimensions"].get(d, 5.0) for a in answers)
                        / max(1, len(answers)), 1) for d in dims}
        return {
            "overall_score": overall,
            "verdict": ("Strong session overall — polish the weak spots below and you're interview-ready."
                        if overall >= 7 else
                        "Decent foundation — the improvement plan below is where your next points come from."),
            "dimension_breakdown": {
                "communication": avg["communication"], "content": avg["relevance"],
                "structure": avg["structure"], "confidence": avg["depth"],
                "integrity": round(integrity["score"] / 10, 1),
            },
            "per_question": [
                {"question": a["question"], "score": a["score"]["score"],
                 "good": "Answer addressed the question." if a["score"]["score"] >= 6
                         else "You attempted the question.",
                 "improve": a["score"]["one_line_tip"]}
                for a in answers
            ],
            "strengths": ["You completed the full session.",
                          f"Best answer scored {max(scores)}/10.",
                          "Consistent engagement across questions."],
            "improvement_plan": ["Re-answer your lowest-scored question using STAR.",
                                 "Add one measurable result to every project story.",
                                 "Do one timed mock round at the same difficulty this week."],
            "integrity": integrity,
            "source": "local",
        }


report_generator = ReportGenerator()
