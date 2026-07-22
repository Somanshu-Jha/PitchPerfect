# =====================================================================
# PROCTOR SESSION — per-connection violation ledger + graduated response
# =====================================================================
import logging
from backend.interview.proctor_policy import (
    VIOLATION_SEVERITY, INTEGRITY_DEDUCTIONS, resolve_policy, decide_action,
)

logger = logging.getLogger(__name__)

_COOLDOWN_MS = 8000

_WARNING_TEXTS = {
    "GAZE_OFF_SCREEN": [
        "By the way — I noticed you glanced away for a while there. All good, just try to stay with me.",
        "I'll mention it again: your eyes keep drifting off-screen. I need your attention here, please.",
        "This keeps happening. Please keep your focus on our conversation — it matters in a real interview too.",
    ],
    "READING_DETECTED": [
        "It looks like you might be reading something. Remember, this is a closed-book conversation — I want your own words.",
        "I can tell the answers are being read out. Please put the notes away — authentic answers score far better.",
    ],
    "FOCUS_LOST": [
        "I noticed the window lost focus for a moment — please keep this screen active during our chat.",
        "Again, the interview window went into the background. Let's keep distractions closed.",
    ],
    "SECOND_PERSON": [
        "I can see someone else in the room with you. This needs to be a one-on-one conversation — please make sure you're alone.",
        "There's still another person visible. I have to insist you continue this interview alone.",
    ],
    "CAMERA_ABSENT": [
        "I've lost sight of you — please adjust your camera so I can see you clearly.",
        "Your camera is still blocked or you're out of frame. Please fix it now, or we won't be able to continue.",
    ],
    "PHONE_DETECTED": [
        "I can see a phone in your hand. Please put it away — we don't allow devices during the interview.",
        "The phone is out again. Put it completely out of reach, please.",
    ],
    "SYNTHETIC_EYE_CONTACT": [
        "Something about your eye contact looks software-generated. If you're using an auto eye-contact filter, please turn it off now.",
        "I still believe an eye-contact filter is running. Turn off any camera effects — I need to see your natural gaze.",
    ],
    "TAB_SWITCHED": [
        "I noticed you switched away from this tab. Please stay on the interview screen.",
        "You've switched tabs again. In a real interview that would be a serious problem — please don't do it again.",
    ],
}


class ProctorSession:
    def __init__(self, difficulty: str):
        self.policy = resolve_policy(difficulty)
        self.terminated = False
        self.termination_reason = None
        self._last_event_ms: dict = {}
        self._warnings_by_type: dict = {}
        self._counts_by_type: dict = {}
        self._penalties_total = 0
        self._deductions = 0
        self._violations: list = []
        self._pending_notes: list = []
        self._answers: list = []

    def record_event(self, event_type: str, confidence: float, ts_ms: int,
                     candidate_speaking: bool = False, meta: dict = None):
        if self.terminated or event_type not in VIOLATION_SEVERITY:
            return None
        last = self._last_event_ms.get(event_type)
        if last is not None and ts_ms - last < _COOLDOWN_MS:
            return None
        self._last_event_ms[event_type] = ts_ms

        action = decide_action(
            self.policy, event_type,
            prior_warnings_for_type=self._warnings_by_type.get(event_type, 0),
            prior_penalties_total=self._penalties_total,
            prior_count_for_type=self._counts_by_type.get(event_type, 0),
        )
        count = self._counts_by_type.get(event_type, 0)
        self._counts_by_type[event_type] = count + 1

        severity = VIOLATION_SEVERITY[event_type]
        if action == "WARN":
            self._warnings_by_type[event_type] = self._warnings_by_type.get(event_type, 0) + 1
        elif action == "PENALIZE":
            self._penalties_total += 1
            self._deductions += INTEGRITY_DEDUCTIONS[f"PENALIZE_{severity}"]
        elif action == "TERMINATE":
            self.terminated = True
            self.termination_reason = event_type
            self._deductions += INTEGRITY_DEDUCTIONS["TERMINATE"]

        texts = _WARNING_TEXTS.get(event_type, ["Please keep to the interview rules."])
        warning_text = texts[min(count, len(texts) - 1)]

        self._violations.append({
            "type": event_type, "severity": severity, "action": action,
            "ts_ms": ts_ms, "confidence": confidence, "meta": meta or {},
        })
        logger.info(f"[Proctor] {event_type} -> {action} "
                    f"(count={count + 1}, integrity={self.integrity_score()})")

        deliver_now = (not candidate_speaking) or action == "TERMINATE"
        if not deliver_now:
            self._pending_notes.append(
                f"[PROCTOR NOTE: {event_type} occurred while the candidate was answering. "
                f"Naturally weave this warning into your next response: \"{warning_text}\"]"
            )
        return {"action": action, "event_type": event_type,
                "warning_text": warning_text, "deliver_now": deliver_now}

    def record_answer_score(self, turn_id: int, question: str, answer: str, score: dict):
        self._answers.append({"turn_id": turn_id, "question": question,
                               "answer": answer, "score": score})

    def consume_pending_prompt_notes(self) -> str:
        notes = "\n".join(self._pending_notes)
        self._pending_notes = []
        return notes

    def integrity_score(self) -> int:
        return max(0, 100 - self._deductions)

    def snapshot(self) -> dict:
        return {
            "integrity_score": self.integrity_score(),
            "violations": [
                {k: v[k] for k in ("type", "severity", "action", "ts_ms")}
                for v in self._violations
            ],
            "answers": list(self._answers),
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
        }
