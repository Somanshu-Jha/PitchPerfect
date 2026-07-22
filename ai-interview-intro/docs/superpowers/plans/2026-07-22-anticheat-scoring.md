# Anti-Cheat Proctoring, Real-Time Scoring & HR Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accurate head-pose-compensated eye tracking with calibration, multi-person/phone/synthetic-eye-contact detection, strictness-scaled graduated consequences, per-answer live scoring, natural HR persona, post-call report — plus a fix for the HR audio overlap/repeat race.

**Architecture:** All CV runs in the browser (MediaPipe via CDN; HF free tier never sees video). Browser fuses signals into structured `proctor_event` websocket messages. Backend keeps a per-connection ledger, applies a strictness policy (warn → penalize → terminate), scores each answer in parallel with question generation, and builds the final report on `end_call`. Turn sequencing (turn_id + single-in-flight lock) removes the overlap/repeat bug.

**Tech Stack:** React+TS+Vite frontend, `@mediapipe/tasks-vision@0.10.14` (CDN), FastAPI backend, existing `llm_router` (nvidia/groq/gemini/kimi), Edge TTS, pytest.

**Spec:** `ai-interview-intro/docs/superpowers/specs/2026-07-22-anticheat-scoring-design.md`

## Global Constraints

- Repo root is `C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab`; all paths below are relative to `ai-interview-intro/`.
- NEVER touch `ready_for_upload/`, `src_backup/`, `*.bak*` files, `node_modules/`, `__pycache__/`.
- No new npm dependencies. MediaPipe loads from CDN exactly like the existing `loadMediaPipeFaceLandmarker()`.
- No server-side video processing (HF free tier, 2 vCPU, no GPU).
- Difficulty values from the room UI: `beginner | intermediate | advanced | faang`. Backend `STRICTNESS_MAP` keys: `beginner | intermediate | advance | extreme`. Alias mapping `advanced→advance`, `faang→extreme` is mandatory wherever STRICTNESS_MAP is consulted.
- Frontend style: dark glassmorphism (`bg-white/5`, `border-white/10`, `rounded-2xl`, `text-white/60`, uppercase `tracking-widest` labels), icons from `lucide-react`.
- Frontend gate: `cd frontend && npx tsc --noEmit` → zero NEW errors (baseline may have pre-existing errors in unrelated files — compare against baseline captured in Task 0).
- Backend gate: `python -m pytest backend/tests -q` passes and `python -c "from backend.api import interview_routes"` succeeds (run from `ai-interview-intro/`). If pytest missing: `pip install pytest`.
- Realtime turn latency budget: scoring must NOT delay `turn_result` — always `asyncio.gather` with the dialogue call.
- All detectors: no single-frame flags. Dwell + hysteresis + confidence gating per spec.
- Backup rule (user requirement): Task 0 creates a git backup branch AND `.pre-anticheat.bak` copies of every file this plan modifies, before any edit.

---

### Task 0: Backups + baseline capture

**Files:**
- Create: `backup branch` (git), `*.pre-anticheat.bak` copies
- Create: `frontend/tsc-baseline.txt`

**Interfaces:**
- Produces: revert path — `git checkout backup/pre-anticheat-20260722 -- <path>` or copy back the `.bak` file.

- [ ] **Step 1: Create git backup branch at current HEAD**

```bash
cd "C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab"
git branch backup/pre-anticheat-20260722
git branch --list backup/*
```
Expected: `backup/pre-anticheat-20260722` listed.

- [ ] **Step 2: Copy the files this plan modifies**

```bash
cd "C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab\ai-interview-intro"
cp frontend/src/components/screens/RealtimeInterview.tsx frontend/src/components/screens/RealtimeInterview.tsx.pre-anticheat.bak
cp frontend/src/components/simulation/CandidateObserver.tsx frontend/src/components/simulation/CandidateObserver.tsx.pre-anticheat.bak
cp backend/api/interview_routes.py backend/api/interview_routes.py.pre-anticheat.bak
cp backend/agents/dialogue_manager.py backend/agents/dialogue_manager.py.pre-anticheat.bak
ls frontend/src/components/screens/*.bak frontend/src/components/simulation/*.bak backend/api/*.bak backend/agents/*.bak
```
Expected: 4 `.pre-anticheat.bak` files listed.

- [ ] **Step 3: Capture tsc baseline (pre-existing errors, if any)**

```bash
cd "C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab\ai-interview-intro\frontend"
npx tsc --noEmit > tsc-baseline.txt 2>&1; wc -l tsc-baseline.txt
```
Expected: file written (0 lines = clean baseline). Later tasks diff against this.

- [ ] **Step 4: Commit backups**

```bash
cd "C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab"
git add -f ai-interview-intro/frontend/src/components/screens/RealtimeInterview.tsx.pre-anticheat.bak ai-interview-intro/frontend/src/components/simulation/CandidateObserver.tsx.pre-anticheat.bak ai-interview-intro/backend/api/interview_routes.py.pre-anticheat.bak ai-interview-intro/backend/agents/dialogue_manager.py.pre-anticheat.bak
git commit -m "chore: pre-anticheat backup copies of files to be modified"
```

---

### Task 1: Proctor policy engine (backend, pure logic)

**Files:**
- Create: `backend/interview/proctor_policy.py`
- Test: `backend/tests/test_proctor_policy.py` (create `backend/tests/__init__.py` if absent)

**Interfaces:**
- Produces:
  - `VIOLATION_SEVERITY: dict[str, str]` — event type → `"warn" | "critical"`
  - `resolve_policy(difficulty: str) -> dict` — returns `{"key", "dwell_multiplier", "warnings_before_penalty", "terminate_after_penalties", "instant_terminate", "critical_repeat_terminate"}`
  - `decide_action(policy: dict, event_type: str, prior_warnings_for_type: int, prior_penalties_total: int, prior_count_for_type: int) -> str` — returns `"WARN" | "PENALIZE" | "TERMINATE"`
  - `INTEGRITY_DEDUCTIONS: dict[str, int]` — `{"WARN": 0, "PENALIZE_warn": 10, "PENALIZE_critical": 25, "TERMINATE": 40}`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_proctor_policy.py
from backend.interview.proctor_policy import (
    VIOLATION_SEVERITY, resolve_policy, decide_action,
)


def test_resolve_policy_aliases_and_defaults():
    assert resolve_policy("beginner")["dwell_multiplier"] == 1.5
    assert resolve_policy("advanced")["key"] == "advanced"
    assert resolve_policy("FAANG")["dwell_multiplier"] == 0.6
    assert resolve_policy("nonsense")["key"] == "intermediate"


def test_beginner_never_terminates_or_penalizes():
    p = resolve_policy("beginner")
    for i in range(10):
        assert decide_action(p, "PHONE_DETECTED", prior_warnings_for_type=i,
                             prior_penalties_total=0, prior_count_for_type=i) == "WARN"


def test_intermediate_graduation():
    p = resolve_policy("intermediate")
    # first two occurrences of a warn-severity type: warnings
    assert decide_action(p, "GAZE_OFF_SCREEN", 0, 0, 0) == "WARN"
    assert decide_action(p, "GAZE_OFF_SCREEN", 1, 0, 1) == "WARN"
    # third: penalty
    assert decide_action(p, "GAZE_OFF_SCREEN", 2, 0, 2) == "PENALIZE"
    # repeated critical terminates at threshold 3
    assert decide_action(p, "SECOND_PERSON", 2, 1, 2) == "TERMINATE"


def test_faang_instant_terminate():
    p = resolve_policy("faang")
    assert decide_action(p, "PHONE_DETECTED", 0, 0, 0) == "TERMINATE"
    assert decide_action(p, "SECOND_PERSON", 0, 0, 0) == "TERMINATE"
    # non-instant type: 1 warning then penalize, terminate after 2 penalties
    assert decide_action(p, "GAZE_OFF_SCREEN", 0, 0, 0) == "WARN"
    assert decide_action(p, "GAZE_OFF_SCREEN", 1, 0, 1) == "PENALIZE"
    assert decide_action(p, "GAZE_OFF_SCREEN", 1, 1, 2) == "TERMINATE"


def test_severity_table_complete():
    for t in ["GAZE_OFF_SCREEN", "READING_DETECTED", "FOCUS_LOST",
              "SECOND_PERSON", "CAMERA_ABSENT", "PHONE_DETECTED",
              "SYNTHETIC_EYE_CONTACT", "TAB_SWITCHED"]:
        assert VIOLATION_SEVERITY[t] in ("warn", "critical")
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `ai-interview-intro/`): `python -m pytest backend/tests/test_proctor_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: backend.interview.proctor_policy`

- [ ] **Step 3: Write the implementation**

```python
# backend/interview/proctor_policy.py
# =====================================================================
# PROCTOR POLICY — strictness-scaled graduated consequences (pure logic)
# =====================================================================

VIOLATION_SEVERITY = {
    "GAZE_OFF_SCREEN": "warn",
    "READING_DETECTED": "warn",
    "FOCUS_LOST": "warn",
    "SECOND_PERSON": "critical",
    "CAMERA_ABSENT": "critical",
    "PHONE_DETECTED": "critical",
    "SYNTHETIC_EYE_CONTACT": "critical",
    "TAB_SWITCHED": "critical",
}

INTEGRITY_DEDUCTIONS = {"WARN": 0, "PENALIZE_warn": 10, "PENALIZE_critical": 25, "TERMINATE": 40}

_ALIASES = {"advance": "advanced", "extreme": "faang"}

_POLICIES = {
    "beginner": {
        "dwell_multiplier": 1.5, "warnings_before_penalty": 999,
        "terminate_after_penalties": None, "instant_terminate": (),
        "critical_repeat_terminate": None,
    },
    "intermediate": {
        "dwell_multiplier": 1.0, "warnings_before_penalty": 2,
        "terminate_after_penalties": None, "instant_terminate": (),
        "critical_repeat_terminate": 3,
    },
    "advanced": {
        "dwell_multiplier": 0.8, "warnings_before_penalty": 1,
        "terminate_after_penalties": 3, "instant_terminate": (),
        "critical_repeat_terminate": 3,
    },
    "faang": {
        "dwell_multiplier": 0.6, "warnings_before_penalty": 1,
        "terminate_after_penalties": 2,
        "instant_terminate": ("PHONE_DETECTED", "SECOND_PERSON"),
        "critical_repeat_terminate": 2,
    },
}


def resolve_policy(difficulty: str) -> dict:
    key = (difficulty or "").lower().strip()
    key = _ALIASES.get(key, key)
    if key not in _POLICIES:
        key = "intermediate"
    return {"key": key, **_POLICIES[key]}


def decide_action(policy: dict, event_type: str,
                  prior_warnings_for_type: int,
                  prior_penalties_total: int,
                  prior_count_for_type: int) -> str:
    severity = VIOLATION_SEVERITY.get(event_type, "warn")

    if event_type in policy["instant_terminate"]:
        return "TERMINATE"

    if (severity == "critical"
            and policy["critical_repeat_terminate"] is not None
            and prior_count_for_type + 1 >= policy["critical_repeat_terminate"]):
        return "TERMINATE"

    if prior_warnings_for_type < policy["warnings_before_penalty"]:
        return "WARN"

    if (policy["terminate_after_penalties"] is not None
            and prior_penalties_total + 1 >= policy["terminate_after_penalties"]):
        return "TERMINATE"

    return "PENALIZE"
```

- [ ] **Step 4: Create `backend/tests/__init__.py` if it does not exist (empty file), run tests**

Run: `python -m pytest backend/tests/test_proctor_policy.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ai-interview-intro/backend/interview/proctor_policy.py ai-interview-intro/backend/tests/
git commit -m "feat: strictness-scaled proctor policy engine"
```

---

### Task 2: Proctor session ledger (backend)

**Files:**
- Create: `backend/interview/proctor_session.py`
- Test: `backend/tests/test_proctor_session.py`

**Interfaces:**
- Consumes: `resolve_policy`, `decide_action`, `VIOLATION_SEVERITY`, `INTEGRITY_DEDUCTIONS` from Task 1.
- Produces: class `ProctorSession`:
  - `__init__(self, difficulty: str)` — resolves policy, `self.policy: dict`
  - `record_event(self, event_type: str, confidence: float, ts_ms: int, candidate_speaking: bool = False, meta: dict | None = None) -> dict | None` — returns `{"action": "WARN"|"PENALIZE"|"TERMINATE", "event_type": str, "warning_text": str, "deliver_now": bool}` or `None` when deduped/cooldown.
  - `record_answer_score(self, turn_id: int, question: str, answer: str, score: dict) -> None`
  - `consume_pending_prompt_notes(self) -> str` — drains queued warning notes for system-prompt injection, `""` if none.
  - `integrity_score(self) -> int` — 100 minus deductions, floor 0.
  - `snapshot(self) -> dict` — `{"integrity_score", "violations": [{"type","severity","action","ts_ms"}...], "answers": [...], "terminated": bool, "termination_reason": str|None}`
  - `terminated: bool` attribute.
- Warning texts must be natural HR phrasing, escalate with count, vary by type (dict of lists indexed by min(count, len-1)).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_proctor_session.py
from backend.interview.proctor_session import ProctorSession


def test_cooldown_dedupes_repeat_events():
    s = ProctorSession("intermediate")
    a1 = s.record_event("GAZE_OFF_SCREEN", 0.9, ts_ms=1000)
    a2 = s.record_event("GAZE_OFF_SCREEN", 0.9, ts_ms=3000)  # within 8s window
    a3 = s.record_event("GAZE_OFF_SCREEN", 0.9, ts_ms=20000)
    assert a1 is not None and a2 is None and a3 is not None


def test_graduation_and_integrity():
    s = ProctorSession("intermediate")
    t = 0
    actions = []
    for _ in range(3):
        t += 10000
        r = s.record_event("GAZE_OFF_SCREEN", 0.9, ts_ms=t)
        actions.append(r["action"])
    assert actions == ["WARN", "WARN", "PENALIZE"]
    assert s.integrity_score() == 90  # one warn-severity penalty = -10


def test_terminate_sets_flag_and_snapshot():
    s = ProctorSession("faang")
    r = s.record_event("PHONE_DETECTED", 0.95, ts_ms=1000)
    assert r["action"] == "TERMINATE"
    assert s.terminated is True
    snap = s.snapshot()
    assert snap["terminated"] is True
    assert snap["integrity_score"] == 60  # -40
    assert snap["violations"][0]["type"] == "PHONE_DETECTED"


def test_deliver_now_vs_queued():
    s = ProctorSession("intermediate")
    r_idle = s.record_event("TAB_SWITCHED", 1.0, ts_ms=1000, candidate_speaking=False)
    r_busy = s.record_event("FOCUS_LOST", 1.0, ts_ms=20000, candidate_speaking=True)
    assert r_idle["deliver_now"] is True
    assert r_busy["deliver_now"] is False
    notes = s.consume_pending_prompt_notes()
    assert "FOCUS_LOST" in notes or "window" in notes.lower()
    assert s.consume_pending_prompt_notes() == ""  # drained


def test_warning_text_varies_and_escalates():
    s = ProctorSession("beginner")
    r1 = s.record_event("GAZE_OFF_SCREEN", 0.9, ts_ms=1000)
    r2 = s.record_event("GAZE_OFF_SCREEN", 0.9, ts_ms=20000)
    assert r1["warning_text"] and r2["warning_text"]
    assert r1["warning_text"] != r2["warning_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_proctor_session.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

```python
# backend/interview/proctor_session.py
# =====================================================================
# PROCTOR SESSION — per-connection violation ledger + graduated response
# =====================================================================
import logging
from backend.interview.proctor_policy import (
    VIOLATION_SEVERITY, INTEGRITY_DEDUCTIONS, resolve_policy, decide_action,
)

logger = logging.getLogger(__name__)

_COOLDOWN_MS = 8000

# Natural HR phrasing, escalating with occurrence count (index min(count, last)).
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
        self._last_event_ms = {}      # type -> ts of last ACCEPTED event
        self._warnings_by_type = {}   # type -> count of WARN actions
        self._counts_by_type = {}     # type -> accepted events count
        self._penalties_total = 0
        self._deductions = 0
        self._violations = []         # ledger for the report
        self._pending_notes = []      # queued for next system prompt
        self._answers = []            # per-answer scores

    # ── events ──────────────────────────────────────────────────────
    def record_event(self, event_type, confidence, ts_ms,
                     candidate_speaking=False, meta=None):
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
        logger.info(f"🛡️ [Proctor] {event_type} -> {action} "
                    f"(count={count + 1}, integrity={self.integrity_score()})")

        deliver_now = (not candidate_speaking) or action == "TERMINATE"
        if not deliver_now:
            self._pending_notes.append(
                f"[PROCTOR NOTE: {event_type} occurred while the candidate was answering. "
                f"Naturally weave this warning into your next response: \"{warning_text}\"]"
            )
        return {"action": action, "event_type": event_type,
                "warning_text": warning_text, "deliver_now": deliver_now}

    # ── answers ─────────────────────────────────────────────────────
    def record_answer_score(self, turn_id, question, answer, score):
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/test_proctor_session.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ai-interview-intro/backend/interview/proctor_session.py ai-interview-intro/backend/tests/test_proctor_session.py
git commit -m "feat: proctor session ledger with graduated HR responses"
```

---

### Task 3: Per-answer scorer (backend)

**Files:**
- Create: `backend/services/answer_scorer.py`
- Test: `backend/tests/test_answer_scorer.py`

**Interfaces:**
- Consumes: `llm_router.generate(system_prompt, user_prompt, max_tokens, temperature, selected_llm=...)` (returns `str`, `""` on total failure); `ScoringService.calculate_score(text, structured, precomputed_llm_score=None, ...)` from `backend/services/scoring_service.py`.
- Produces: singleton `answer_scorer` with
  - `score(question: str, answer: str, job_role: str, difficulty: str, selected_llm: str | None = None) -> dict` returning exactly `{"score": float(0-10), "dimensions": {"relevance": float, "structure": float, "depth": float, "communication": float}, "one_line_tip": str, "source": "llm"|"heuristic"}`
- Task 5 calls this inside `asyncio.to_thread` gathered with the dialogue call. It must never raise.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_answer_scorer.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_answer_scorer.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

```python
# backend/services/answer_scorer.py
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

    def score(self, question, answer, job_role, difficulty, selected_llm=None):
        try:
            raw = llm_router.generate(
                system_prompt=_SYSTEM,
                user_prompt=(f"Role: {job_role}\nDifficulty: {difficulty}\n"
                             f"Question: {question}\nAnswer: {answer}\n\nScore it."),
                max_tokens=220,
                temperature=0.2,
                selected_llm=selected_llm,  # None -> router default (nvidia fast tier)
            )
            parsed = self._parse(raw)
            if parsed:
                return parsed
        except Exception as e:
            logger.error(f"[AnswerScorer] LLM scoring failed: {e}")
        return self._fallback(answer)

    def _parse(self, raw):
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

    def _fallback(self, answer):
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/test_answer_scorer.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ai-interview-intro/backend/services/answer_scorer.py ai-interview-intro/backend/tests/test_answer_scorer.py
git commit -m "feat: realtime per-answer scorer with heuristic fallback"
```

---

### Task 4: Report generator (backend)

**Files:**
- Create: `backend/services/report_generator.py`
- Test: `backend/tests/test_report_generator.py`

**Interfaces:**
- Consumes: `llm_router.generate(...)`; `ProctorSession.snapshot()` shape from Task 2.
- Produces: singleton `report_generator` with
  - `generate(history: list[dict], snapshot: dict, job_role: str, difficulty: str, mode: str, selected_llm: str | None = "kimi") -> dict` returning
    `{"overall_score": float, "verdict": str, "dimension_breakdown": {"communication": float, "content": float, "structure": float, "confidence": float, "integrity": float}, "per_question": [{"question","score","good","improve"}...], "strengths": [str x3], "improvement_plan": [str...], "integrity": {"score": int, "violations": [...], "terminated": bool}, "source": "llm"|"local"}`
  - Must never raise; on LLM failure builds the report locally from snapshot answers.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_report_generator.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_report_generator.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

```python
# backend/services/report_generator.py
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
    def generate(self, history, snapshot, job_role, difficulty, mode,
                 selected_llm="kimi"):
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
            f"Live score: {a['score']['score']}/10 (tip given: {a['score']['one_line_tip']})"
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

    def _parse(self, raw):
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

    def _local(self, answers, integrity):
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
                          "Best answer scored " + str(max(scores)) + "/10.",
                          "Consistent engagement across questions."],
            "improvement_plan": ["Re-answer your lowest-scored question using STAR.",
                                 "Add one measurable result to every project story.",
                                 "Do one timed mock round at the same difficulty this week."],
            "integrity": integrity,
            "source": "local",
        }


report_generator = ReportGenerator()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/test_report_generator.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add ai-interview-intro/backend/services/report_generator.py ai-interview-intro/backend/tests/test_report_generator.py
git commit -m "feat: post-call interview report generator"
```

---

### Task 5: Wire websocket — sessions, sequencing, parallel scoring, end_call (backend)

**Files:**
- Modify: `backend/api/interview_routes.py` (websocket handler, lines ~315-468)
- Modify: `backend/agents/dialogue_manager.py` (difficulty alias + `proctor_context` param)
- Test: `backend/tests/test_dialogue_difficulty_alias.py`

**Interfaces:**
- Consumes: `ProctorSession` (Task 2), `answer_scorer` (Task 3), `report_generator` (Task 4).
- Produces (websocket protocol, consumed by Tasks 9-11):
  - server→client `{"type":"session_config","policy":{"key","dwell_multiplier"},"difficulty"}` — sent on kickoff.
  - client→server `{"type":"proctor_event","event_type","confidence","ts_ms","candidate_speaking","meta"}`
  - server→client `{"type":"proctor_action","action","event_type"}`
  - server→client `{"type":"hr_interject","text"}` followed by audio bytes + `{"type":"audio_end"}`
  - server→client `{"type":"session_terminated","reason","closing_text"}` (after its audio)
  - client→server `vad_pause` gains `"turn_id": int`, `"covered_topics": [...]`
  - server→client `turn_result` gains `"turn_id"`; new `{"type":"answer_score","turn_id","data":{score,dimensions,one_line_tip,source}}`
  - client→server `{"type":"end_call"}` → server `{"type":"status","message":"generating_report"}` then `{"type":"final_report","data":{...},"integrity":{...}}`

- [ ] **Step 1: Write the failing test for the dialogue alias fix**

```python
# backend/tests/test_dialogue_difficulty_alias.py
from backend.agents.dialogue_manager import DialogueManager


def test_normalize_difficulty_aliases():
    dm = DialogueManager.__new__(DialogueManager)  # no heavy init
    assert dm._normalize_difficulty("advanced") == "advance"
    assert dm._normalize_difficulty("FAANG") == "extreme"
    assert dm._normalize_difficulty("beginner") == "beginner"
    assert dm._normalize_difficulty("") == "intermediate"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest backend/tests/test_dialogue_difficulty_alias.py -q`
Expected: FAIL — `AttributeError: _normalize_difficulty`

- [ ] **Step 3: dialogue_manager.py changes**

3a. Add method to `DialogueManager` (place next to `_mode_key`, ~line 118):

```python
    @staticmethod
    def _normalize_difficulty(difficulty: str) -> str:
        """Frontend sends beginner/intermediate/advanced/faang; STRICTNESS_MAP
        uses beginner/intermediate/advance/extreme. Bridge the two."""
        key = (difficulty or "").lower().strip()
        aliases = {"advanced": "advance", "faang": "extreme"}
        key = aliases.get(key, key)
        return key if key in ("beginner", "intermediate", "advance", "extreme") else "intermediate"
```

3b. In `generate_next_turn` (~line 607), replace:

```python
        from backend.core.genai_engine import STRICTNESS_MAP
        strict_config = STRICTNESS_MAP.get(difficulty.lower(), STRICTNESS_MAP["intermediate"])
```
with:
```python
        from backend.core.genai_engine import STRICTNESS_MAP
        difficulty_key = self._normalize_difficulty(difficulty)
        strict_config = STRICTNESS_MAP.get(difficulty_key, STRICTNESS_MAP["intermediate"])
```

3c. Add `proctor_context: str = ""` parameter to `generate_next_turn` signature (after `candidate_metrics`), and pass it through to `_build_system_prompt(...)` as `proctor_context=proctor_context`.

3d. In `_build_system_prompt`: add parameter `proctor_context: str = ""`; in the `difficulty_guide` dict lookup change `.get(difficulty.lower(), ...)` to:

```python
        guide_key = {"advance": "advanced", "faang": "extreme"}.get(
            difficulty.lower().strip(), difficulty.lower().strip())
        difficulty_guide = { ... existing dict unchanged ... }.get(
            guide_key, "Behave like a professional HR recruiter.")
```

3e. Add a coach-persona block and proctor injection. After `cheating_instruction` is computed, add:

```python
        coach_instruction = ""
        if guide_key in ("beginner", "intermediate"):
            coach_instruction = """
COACH MODE (buddy HR): You are also quietly coaching this candidate for the target role.
After reacting to an answer, you MAY weave in ONE short constructive micro-tip
(max 12 words, e.g. "Pro tip — lead with the result next time."). Max one tip per turn,
never more, and never condescending. At advanced/faang difficulty this is disabled."""

        proctor_block = f"\n{proctor_context}\n" if proctor_context else ""
```

and include both in the returned f-string: place `{coach_instruction}` directly after `{difficulty_guide}` line, and `{proctor_block}` directly after `{cheating_instruction}`.

- [ ] **Step 4: Run alias test + import check**

Run: `python -m pytest backend/tests/test_dialogue_difficulty_alias.py -q && python -c "from backend.agents.dialogue_manager import dialogue_manager; print('ok')"`
Expected: `1 passed`, `ok`

- [ ] **Step 5: Rewrite the websocket handler in `interview_routes.py`**

Replace the entire `websocket_simulation_endpoint` function (from `@router.websocket("/ws/stream")` to end of file) with:

```python
@router.websocket("/ws/stream")
async def websocket_simulation_endpoint(websocket: WebSocket):
    """
    Zero-latency WebSocket endpoint with proctoring, per-answer scoring and
    turn sequencing. One ProctorSession per connection.
    """
    await websocket.accept()
    import asyncio
    import random
    from backend.interview.proctor_session import ProctorSession
    from backend.services.answer_scorer import answer_scorer
    from backend.services.report_generator import report_generator

    session: ProctorSession | None = None
    session_difficulty = "intermediate"
    session_mode = "hr"
    session_job_role = "Software Engineer"
    last_history: list = []

    async def speak(text: str, difficulty: str):
        """Stream one TTS utterance (used by turns, interjects, termination)."""
        voice = ("en-US-ChristopherNeural"
                 if difficulty.lower() in ["advanced", "extreme", "faang"]
                 else "en-US-GuyNeural")
        await websocket.send_json({"type": "status", "message": "speaking"})
        audio_buffer = bytearray()
        async for chunk in tts_service.stream_audio(text, voice=voice, rate="+0%", pitch="+0Hz"):
            if chunk:
                audio_buffer.extend(chunk)
        if audio_buffer:
            await websocket.send_bytes(bytes(audio_buffer))
        await websocket.send_json({"type": "audio_end"})

    async def send_final_report():
        await websocket.send_json({"type": "status", "message": "generating_report"})
        snapshot = session.snapshot() if session else {
            "integrity_score": 100, "violations": [], "answers": [],
            "terminated": False, "termination_reason": None}
        report = await asyncio.to_thread(
            report_generator.generate, last_history, snapshot,
            session_job_role, session_difficulty, session_mode)
        await websocket.send_json({"type": "final_report", "data": report,
                                   "integrity": report.get("integrity", {})})

    while True:
        try:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # ── keep-alive ────────────────────────────────────────────
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # ── proctor events: absorbed by ledger, NEVER a full LLM turn ──
            if msg_type == "proctor_event":
                if session is None:
                    continue
                result = session.record_event(
                    event_type=data.get("event_type", ""),
                    confidence=float(data.get("confidence", 0.0)),
                    ts_ms=int(data.get("ts_ms", 0)),
                    candidate_speaking=bool(data.get("candidate_speaking", False)),
                    meta=data.get("meta") or {},
                )
                if result is None:
                    continue
                await websocket.send_json({"type": "proctor_action",
                                           "action": result["action"],
                                           "event_type": result["event_type"]})
                if result["action"] == "TERMINATE":
                    closing = ("I'm going to stop the interview here. "
                               f"{result['warning_text']} "
                               "Integrity matters more than any single answer — "
                               "your report will explain what happened. Goodbye.")
                    await speak(closing, session_difficulty)
                    await websocket.send_json({"type": "session_terminated",
                                               "reason": result["event_type"],
                                               "closing_text": closing})
                    await send_final_report()
                elif result["deliver_now"]:
                    await speak(result["warning_text"], session_difficulty)
                continue

            # ── end call → report ─────────────────────────────────────
            if msg_type == "end_call":
                await send_final_report()
                continue

            # ── normal turn (vad_pause / kickoff) ────────────────────
            history = data.get("history", [])
            user_response = data.get("user_response", "")
            difficulty = data.get("difficulty", "intermediate")
            mode = data.get("mode", "HR")
            resume_text = data.get("resume_text", "")
            job_role = data.get("job_role", "Software Engineer")
            covered_topics = data.get("covered_topics", [])
            use_premium_mode = data.get("use_fast_mode", False)
            selected_llm = data.get("selected_llm", None)
            turn_id = int(data.get("turn_id", 0))

            session_difficulty, session_mode, session_job_role = difficulty, mode, job_role
            if session is None:
                session = ProctorSession(difficulty)
                await websocket.send_json({"type": "session_config",
                                           "policy": {"key": session.policy["key"],
                                                      "dwell_multiplier": session.policy["dwell_multiplier"]},
                                           "difficulty": difficulty})
            if session.terminated:
                continue

            await websocket.send_json({"type": "status", "message": "thinking"})

            is_kickoff = data.get("is_kickoff", False)
            if is_kickoff:
                greetings = [
                    "Hello, welcome to PitchPerfect AI! I'll be your HR recruiter today. Let's get started — could you please introduce yourself and tell me a bit about your background?",
                    "Good morning! I'm very excited to speak with you today. To kick things off, why don't you walk me through your experience?",
                    "Hi there, it's great to meet you. I'll be conducting your interview today. Could you start by giving me a brief overview of your professional journey?",
                    "Welcome! Let's dive right in. I'd love to hear more about you — could you introduce yourself and explain why you're interested in this role?",
                    "Hi, thanks for making the time today. Before we get into specifics, tell me a little about yourself and what you've been working on lately.",
                    "Welcome aboard — I've been looking forward to this conversation. Why don't you start by telling me about yourself, in your own words?",
                    "Good to see you. I like to keep these conversations fairly relaxed — so to begin, give me the short version of your story so far.",
                    "Hello! Let's make this feel like a conversation, not an interrogation. First things first — who are you, and what drives you professionally?",
                    "Thanks for joining me today. I've got your application in front of me, but I'd much rather hear it from you — walk me through your background.",
                    "Hi, welcome. Let's start simple: introduce yourself, and tell me about one piece of work you're genuinely proud of.",
                    "Great to meet you. Before I ask anything specific, set the stage for me — where are you in your career right now, and how did you get here?",
                    "Welcome — settle in. To open things up, tell me about yourself and what kind of role you're hoping this turns into."
                ]
                result = {
                    "feedback": "", "next_question": random.choice(greetings),
                    "should_end": False, "topic": "self_intro",
                    "reasoning": "Initial greeting",
                    "avatar_state": {"emotion": "FRIENDLY", "pose": "OPEN_PALMS",
                                     "gaze": "DIRECT", "micro_expression": "HEAD_TILT"},
                    "vocal_params": {"pitch_multiplier": 1.05, "speed_multiplier": 1.0,
                                     "pause_before_ms": 0, "filler_prefix": ""},
                }
                score_result = None
            else:
                is_vad_pause = msg_type == "vad_pause"
                proctor_context = session.consume_pending_prompt_notes()
                last_question = ""
                for t in reversed(history):
                    if t.get("role") == "assistant":
                        last_question = t.get("content", "")
                        break

                turn_coro = asyncio.to_thread(
                    dialogue_manager.generate_next_turn,
                    history, user_response, difficulty, mode,
                    resume_text=resume_text, pitch_text=job_role,
                    covered_topics=covered_topics,
                    is_vad_pause=is_vad_pause,
                    use_premium=use_premium_mode,
                    selected_llm=selected_llm,
                    proctor_context=proctor_context,
                )
                score_coro = asyncio.to_thread(
                    answer_scorer.score, last_question, user_response,
                    job_role, difficulty)
                result, score_result = await asyncio.gather(turn_coro, score_coro)
                session.record_answer_score(turn_id, last_question,
                                            user_response, score_result)

            last_history = [*history,
                            {"role": "user", "content": user_response}] if not is_kickoff else []

            result["integrity_score"] = session.integrity_score()
            await websocket.send_json({"type": "turn_result", "turn_id": turn_id,
                                       "data": result})
            if score_result is not None:
                await websocket.send_json({"type": "answer_score", "turn_id": turn_id,
                                           "data": score_result})

            feedback = result.get("feedback", "")
            next_q = result.get("next_question", "")
            combined_text = f"{feedback} {next_q}".strip()

            vocal_params = result.get("vocal_params", {})
            speed_mult = vocal_params.get("speed_multiplier", 1.0)
            pitch_mult = vocal_params.get("pitch_multiplier", 1.0)
            rate_pct = int((speed_mult - 1.0) * 100)
            rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
            pitch_hz = int((pitch_mult - 1.0) * 50)
            pitch_str = f"+{pitch_hz}Hz" if pitch_hz >= 0 else f"{pitch_hz}Hz"
            voice = ("en-US-ChristopherNeural"
                     if difficulty.lower() in ["advanced", "extreme", "faang"]
                     else "en-US-GuyNeural")

            if combined_text:
                await websocket.send_json({"type": "status", "message": "speaking"})
                audio_buffer = bytearray()
                async for chunk in tts_service.stream_audio(combined_text, voice=voice,
                                                            rate=rate_str, pitch=pitch_str):
                    if chunk:
                        audio_buffer.extend(chunk)
                if audio_buffer:
                    await websocket.send_bytes(bytes(audio_buffer))
                await websocket.send_json({"type": "audio_end"})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected gracefully.")
            break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
```

Note: `last_history` assignment uses `is_kickoff` — it is defined in both branches (kickoff sets it above; move the assignment INSIDE the else branch and set `last_history = []` in the kickoff branch to avoid referencing `is_kickoff` before assignment on kickoff path — concretely: in the kickoff branch add `last_history = []` after `score_result = None`, and in the else branch add `last_history = [*history, {"role": "user", "content": user_response}]` after `session.record_answer_score(...)`; delete the standalone `last_history = [*history, ...] if not is_kickoff else []` line).

- [ ] **Step 6: Import check + full backend test run**

Run: `python -c "from backend.api import interview_routes; print('ok')" && python -m pytest backend/tests -q`
Expected: `ok`, all tests pass.

- [ ] **Step 7: Commit**

```bash
git add ai-interview-intro/backend/api/interview_routes.py ai-interview-intro/backend/agents/dialogue_manager.py ai-interview-intro/backend/tests/test_dialogue_difficulty_alias.py
git commit -m "feat: proctor ledger wiring, parallel answer scoring, end-call report, difficulty alias fix"
```

---

### Task 6: ProctorEngine core (frontend, pure logic)

**Files:**
- Create: `frontend/src/components/simulation/ProctorTypes.ts`
- Create: `frontend/src/components/simulation/ProctorEngine.ts`

**Interfaces:**
- Produces (consumed by Tasks 8, 9, 10):

```ts
// ProctorTypes.ts — exact content
export type ProctorViolationType =
  | 'GAZE_OFF_SCREEN' | 'READING_DETECTED' | 'FOCUS_LOST'
  | 'SECOND_PERSON' | 'CAMERA_ABSENT' | 'PHONE_DETECTED'
  | 'SYNTHETIC_EYE_CONTACT' | 'TAB_SWITCHED';

export interface ProctorEvent {
  eventType: ProctorViolationType;
  confidence: number;      // 0-1
  tsMs: number;            // Date.now()
  meta?: Record<string, unknown>;
}

export interface ProctorMetrics {
  eyeContactPercent: number;          // 0-100 rolling
  gazeDirection: 'CENTER' | 'LEFT' | 'RIGHT' | 'UP' | 'DOWN';
  headPose: { yaw: number; pitch: number; roll: number };  // degrees
  blinkRatePerMin: number;
  facesDetected: number;
  nervousnessScore: number;           // 0-100
  confidenceLevel: 'Very Low' | 'Low' | 'Medium' | 'High' | 'Very High';
  focusLevel: 'Low' | 'Medium' | 'High';
  isReading: boolean;
  facePresent: boolean;
  isSyntheticEyeContact: boolean;
  calibrated: boolean;
  trackingConfidence: number;         // 0-1; below 0.5 detectors are gated off
}

export interface CalibrationModel {
  // screenX = ax0 + ax1*irisX + ax2*yawDeg ; screenY = ay0 + ay1*irisY + ay2*pitchDeg
  ax: [number, number, number];
  ay: [number, number, number];
  eyeOpenBaseline: number;
  quality: 'good' | 'noisy' | 'auto';
}

export interface ProctorPolicyConfig { dwellMultiplier: number; }
```

- `ProctorEngine` class API (exact):
  - `constructor(opts: { policy: ProctorPolicyConfig; onEvent: (e: ProctorEvent) => void; onMetrics: (m: ProctorMetrics) => void })`
  - `setCalibration(model: CalibrationModel | null): void` — null keeps auto-baseline mode
  - `processFrame(result: FaceLandmarkerResult, nowMs: number): void` — FaceLandmarkerResult = `{ faceLandmarks: {x,y,z}[][], faceBlendshapes?: {categories: {categoryName: string, score: number}[]}[], facialTransformationMatrixes?: {data: number[]}[] }`
  - `notePhoneDetection(score: number, nowMs: number): void` (Task 7 calls this)
  - `noteExternalViolation(type: 'TAB_SWITCHED' | 'FOCUS_LOST', nowMs: number): void`
  - static helper exported for calibration fitting: `fitCalibration(samples: Array<{irisX:number; irisY:number; yawDeg:number; pitchDeg:number; targetX:number; targetY:number}>): CalibrationModel`

- [ ] **Step 1: Write `ProctorTypes.ts`** with the exact content above.

- [ ] **Step 2: Write `ProctorEngine.ts`**

Key implementation requirements — all of this is REQUIRED content, not suggestions:

```ts
/**
 * ProctorEngine.ts — head-pose-compensated gaze + violation fusion.
 * Pure logic: no React, no MediaPipe loading (caller feeds frames).
 */
import type {
  CalibrationModel, ProctorEvent, ProctorMetrics,
  ProctorPolicyConfig, ProctorViolationType,
} from './ProctorTypes';

// MediaPipe 478-landmark indices
const LEFT_IRIS = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const LEFT_EYE_CORNERS: [number, number] = [33, 133];
const RIGHT_EYE_CORNERS: [number, number] = [362, 263];

interface P3 { x: number; y: number; z: number }
const centroid = (pts: P3[]): P3 => ({
  x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
  y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
  z: pts.reduce((s, p) => s + p.z, 0) / pts.length,
});

function irisOffset(iris: P3, inner: P3, outer: P3): { x: number; y: number } {
  const cx = (inner.x + outer.x) / 2, cy = (inner.y + outer.y) / 2;
  const w = Math.abs(outer.x - inner.x);
  if (w < 1e-3) return { x: 0, y: 0 };
  return { x: (iris.x - cx) / w, y: (iris.y - cy) / w };
}

/** Euler angles (degrees) from MediaPipe column-major 4x4 transformation matrix. */
export function headPoseFromMatrix(m: number[]): { yaw: number; pitch: number; roll: number } {
  const r = 180 / Math.PI;
  return {
    yaw: Math.atan2(m[8], m[10]) * r,
    pitch: Math.asin(Math.max(-1, Math.min(1, -m[9]))) * r,
    roll: Math.atan2(m[1], m[5]) * r,
  };
}

/** Least-squares fit screen = a0 + a1*iris + a2*headAngle from calibration samples. */
export function fitCalibration(samples: Array<{
  irisX: number; irisY: number; yawDeg: number; pitchDeg: number;
  targetX: number; targetY: number;
}>): CalibrationModel {
  // Solve A^T A x = A^T b for each axis, A rows = [1, iris, angle], via Cramer's rule.
  const solve = (rows: Array<[number, number, number]>, b: number[]): [number, number, number] => {
    let s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
    let t0 = 0, t1 = 0, t2 = 0;
    rows.forEach((r, i) => {
      s00 += r[0] * r[0]; s01 += r[0] * r[1]; s02 += r[0] * r[2];
      s11 += r[1] * r[1]; s12 += r[1] * r[2]; s22 += r[2] * r[2];
      t0 += r[0] * b[i]; t1 += r[1] * b[i]; t2 += r[2] * b[i];
    });
    const det = s00 * (s11 * s22 - s12 * s12) - s01 * (s01 * s22 - s12 * s02)
              + s02 * (s01 * s12 - s11 * s02);
    if (Math.abs(det) < 1e-9) return [0, 0, 0]; // degenerate → caller treats as noisy
    const dx = t0 * (s11 * s22 - s12 * s12) - s01 * (t1 * s22 - s12 * t2)
             + s02 * (t1 * s12 - s11 * t2);
    const dy = s00 * (t1 * s22 - t2 * s12) - t0 * (s01 * s22 - s02 * s12)
             + s02 * (s01 * t2 - t1 * s02);
    const dz = s00 * (s11 * t2 - s12 * t1) - s01 * (s01 * t2 - s02 * t1)
             + t0 * (s01 * s12 - s11 * s02);
    return [dx / det, dy / det, dz / det];
  };
  const ax = solve(samples.map(s => [1, s.irisX, s.yawDeg]), samples.map(s => s.targetX));
  const ay = solve(samples.map(s => [1, s.irisY, s.pitchDeg]), samples.map(s => s.targetY));
  const rms = Math.sqrt(samples.reduce((acc, s) => {
    const px = ax[0] + ax[1] * s.irisX + ax[2] * s.yawDeg;
    const py = ay[0] + ay[1] * s.irisY + ay[2] * s.pitchDeg;
    return acc + (px - s.targetX) ** 2 + (py - s.targetY) ** 2;
  }, 0) / Math.max(1, samples.length));
  const degenerate = ax.every(v => v === 0) || ay.every(v => v === 0);
  return {
    ax: ax as [number, number, number],
    ay: ay as [number, number, number],
    eyeOpenBaseline: 0, // caller fills
    quality: degenerate || rms > 0.35 ? 'noisy' : 'good',
  };
}

/** Fires once after `active` held for dwellMs; re-arms cooldownMs after deactivation. */
export class DwellFlag {
  private activeSince: number | null = null;
  private firedAt: number | null = null;
  private inactiveSince: number | null = null;
  constructor(private dwellMs: number, private cooldownMs: number) {}
  update(active: boolean, now: number): boolean {
    if (!active) {
      if (this.activeSince !== null || this.firedAt !== null) {
        if (this.inactiveSince === null) this.inactiveSince = now;
        if (this.firedAt !== null && now - this.inactiveSince >= this.cooldownMs) {
          this.firedAt = null; // re-arm
        }
      }
      this.activeSince = null;
      return false;
    }
    this.inactiveSince = null;
    if (this.firedAt !== null) return false; // already fired, not re-armed yet
    if (this.activeSince === null) this.activeSince = now;
    if (now - this.activeSince >= this.dwellMs) {
      this.firedAt = now;
      return true;
    }
    return false;
  }
}
```

Detector internals (implement inside the class; window sizes at ~30 fps):
- **Rolling buffers**: rawIrisX (90), screenGazeMag (30), yawDeg (90), blink events (timestamps).
- **Dwell/hysteresis helper** `DwellFlag { constructor(dwellMs, cooldownMs); update(active: boolean, now: number): boolean }` — returns true exactly once when `active` has been continuously true for `dwellMs`, then refuses to fire again until `cooldownMs` after it deactivates. All dwell times multiplied by `policy.dwellMultiplier`.
- **trackingConfidence**: 1.0 baseline; ×0.5 if face bbox height (from landmark extremes) < 0.12 of frame; ×0.6 if per-eye gaze disagreement > 0.15; below 0.5 → all detectors receive `active=false` this frame (gated).
- **Screen gaze**: with calibration → `screenX = ax0+ax1*irisX+ax2*yaw; screenY = ...`; off-screen when `|screenX|>1.15 || |screenY|>1.15`. Without calibration (auto mode) → subtract auto-baseline medians (collected over the first 240 frames), thresholds 0.28 iris units after yaw compensation `irisX - 0.006*yawDeg`.
- **Eye contact %**: rolling 90-frame window of `onScreen && |screenGaze| < 0.55` (calibrated) or gazeMag < 0.15 (auto).
- **GAZE_OFF_SCREEN**: DwellFlag(2500ms, 4000ms) on off-screen state. Grace: if gaze direction is UP and `candidateSpeaking` flag (set via `setCandidateSpeaking(b: boolean)` — add this public method) — do not count (thinking glance).
- **READING_DETECTED**: direction changes of screenX in 30-frame window > 6 AND eyeContactPercent < 50 → feed 60-frame rolling score; DwellFlag(3000ms, 8000ms) on score > 0.6.
- **SECOND_PERSON**: `result.faceLandmarks.length >= 2` → DwellFlag(1500ms, 8000ms).
- **CAMERA_ABSENT**: zero faces → DwellFlag(4000ms, 8000ms).
- **Blink**: blendshapes `eyeBlinkLeft`+`eyeBlinkRight` mean > 0.5 = closed; rising edge = blink event; rate = events in last 60s.
- **SYNTHETIC_EYE_CONTACT** — 4 cues over a 90-frame window, each 0 or 1:
  1. `varHP < 5e-7` where varHP = variance of first-differences of rawIrisX (micro-saccade absence)
  2. `std(yawDeg) > 1.5 && std(rawIrisX) < 0.004` (head moves, gaze frozen)
  3. last blink event had max |ΔrawIrisX| < 0.002 within ±5 frames (blink decorrelation)
  4. `masd/(std(rawIrisX)+1e-6) < 0.08` where masd = mean |successive difference| (jitter-spectrum proxy)
  Score = (cue1+cue2+cue3+cue4)/4; DwellFlag(5000ms, 15000ms) on score ≥ 0.5 AND eye-contact currently true. Confidence reported = score.
- **PHONE_DETECTED**: `notePhoneDetection(score, now)` — flag when 2 consecutive calls ≥ 0.55 within 3s; cooldown 15s.
- **nervousness/confidence/focus**: same formulas as current CandidateObserver (blink>25→+40, offCenter→+30, reading→+30; conf = eyeContact − 0.5·nervous; focus thresholds 40/65).
- `onMetrics` called every frame; `onEvent` only when a DwellFlag fires (event includes confidence + meta like `{direction}`).

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -50`
Expected: no NEW errors vs `tsc-baseline.txt`.

- [ ] **Step 4: Commit**

```bash
git add ai-interview-intro/frontend/src/components/simulation/ProctorTypes.ts ai-interview-intro/frontend/src/components/simulation/ProctorEngine.ts
git commit -m "feat: ProctorEngine with head-pose-compensated gaze and violation fusion"
```

---

### Task 7: Phone detector (frontend)

**Files:**
- Create: `frontend/src/components/simulation/PhoneDetector.ts`

**Interfaces:**
- Consumes: `ProctorEngine.notePhoneDetection(score, nowMs)` from Task 6.
- Produces: class `PhoneDetector`:
  - `constructor(onDetection: (score: number, nowMs: number) => void)`
  - `async start(video: HTMLVideoElement): Promise<boolean>` — loads ObjectDetector from CDN, starts a `setInterval` loop at 1000 ms; returns false (and stays inert) if CDN load fails.
  - `stop(): void`

- [ ] **Step 1: Write the implementation**

```ts
/**
 * PhoneDetector.ts — throttled MediaPipe ObjectDetector loop (~1 fps).
 * EfficientDet-Lite0 (COCO 80 classes) — we only care about "cell phone".
 * Loads from CDN like CandidateObserver's FaceLandmarker; degrades silently.
 */
export class PhoneDetector {
  private detector: any = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastVideoTime = -1;

  constructor(private onDetection: (score: number, nowMs: number) => void) {}

  async start(video: HTMLVideoElement): Promise<boolean> {
    try {
      const { ObjectDetector, FilesetResolver } = await import(
        /* @vite-ignore */
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
      ) as any;
      const fileset = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      this.detector = await ObjectDetector.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
          delegate: 'GPU',
        },
        scoreThreshold: 0.4,
        runningMode: 'VIDEO',
        maxResults: 5,
      });
    } catch (e) {
      console.warn('[PhoneDetector] ObjectDetector failed to load:', e);
      return false;
    }
    this.timer = setInterval(() => {
      if (!this.detector || video.readyState < 2) return;
      if (video.currentTime === this.lastVideoTime) return;
      this.lastVideoTime = video.currentTime;
      try {
        const res = this.detector.detectForVideo(video, performance.now());
        for (const det of res?.detections ?? []) {
          const cat = det.categories?.[0];
          if (cat?.categoryName === 'cell phone' && cat.score >= 0.4) {
            this.onDetection(cat.score, Date.now());
          }
        }
      } catch { /* single-frame failures are fine at 1 fps */ }
    }, 1000);
    return true;
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    try { this.detector?.close?.(); } catch { /* noop */ }
    this.detector = null;
  }
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -50` — no new errors.

- [ ] **Step 3: Commit**

```bash
git add ai-interview-intro/frontend/src/components/simulation/PhoneDetector.ts
git commit -m "feat: client-side phone detection at 1fps via MediaPipe ObjectDetector"
```

---

### Task 8: Calibration overlay (frontend UI)

**Files:**
- Create: `frontend/src/components/simulation/CalibrationOverlay.tsx`

**Interfaces:**
- Consumes: `fitCalibration`, `headPoseFromMatrix` (Task 6), a running FaceLandmarker instance + video element supplied via props.
- Produces: `<CalibrationOverlay faceLandmarker={any} videoRef={RefObject<HTMLVideoElement>} onComplete={(model: CalibrationModel | null) => void} onSkip={() => void} />`
  - Full-screen dark overlay; shows 5 targets sequentially: center, top-left, top-right, bottom-right, bottom-left at 8%/92% margins. Each target pulses for 1.6 s while the component runs its own rAF loop calling `faceLandmarker.detectForVideo`, collecting `{irisX, irisY, yawDeg, pitchDeg}` samples (drop the first 300 ms per target as travel time); target position in normalized coords (center = 0,0; corners = ±1,±1) recorded as `targetX/targetY`.
  - On finish: `fitCalibration(samples)`; if `quality === 'noisy'` show "Lighting looks tricky — using adaptive mode instead" for 1.5 s and call `onComplete(null)`; else `onComplete(model)`.
  - "Skip calibration" button → `onSkip()`.
  - Style: `bg-black/95` overlay, targets are `w-6 h-6 rounded-full bg-indigo-400 animate-ping` with a static center dot, progress text `text-white/60 uppercase tracking-widest text-xs`.

- [ ] **Step 1: Write the component** (per the spec above — sequential targets, sample collection, fit, complete/skip callbacks).

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -50` — no new errors.

- [ ] **Step 3: Commit**

```bash
git add ai-interview-intro/frontend/src/components/simulation/CalibrationOverlay.tsx
git commit -m "feat: 5-point gaze calibration overlay"
```

---

### Task 9: CandidateObserver rewrite as ProctorEngine shell

**Files:**
- Modify: `frontend/src/components/simulation/CandidateObserver.tsx` (full rewrite; backup exists from Task 0)

**Interfaces:**
- Consumes: `ProctorEngine`, `PhoneDetector`, types from `ProctorTypes.ts`.
- Produces (consumed by Task 10):

```ts
interface CandidateObserverProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  isActive: boolean;
  policy: ProctorPolicyConfig;                    // from session_config ws message
  calibration: CalibrationModel | null;
  candidateSpeaking: boolean;                     // true while user transcript active
  onProctorEvent?: (e: ProctorEvent) => void;
  onMetricsUpdate?: (m: ProctorMetrics) => void;
}
```
- Keeps the existing metric panel UI (MetricRow, GazeIndicator) but adds rows: `Faces` (facesDetected, red when ≥2), `Tracking` (trackingConfidence %, amber < 50%), `Calibrated` (✓/adaptive).
- Loads FaceLandmarker with `numFaces: 2, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true` (same CDN loader, updated options).
- rAF loop feeds `engine.processFrame(result, Date.now())`; `PhoneDetector` started/stopped with `isActive`; `engine.setCandidateSpeaking(candidateSpeaking)` on prop change; `engine.setCalibration(calibration)` on prop change.
- The old direct-signal logic (`emitSignal`, `CandidateSignal`) is deleted from this file; HRBehaviorEngine signal typing stays intact for `InterviewSimulation.tsx` — to keep that screen compiling, keep exporting `CandidateMetrics` as an alias: `export type CandidateMetrics = ProctorMetrics;` and keep the optional legacy prop `onSignal?: (signal: import('./HRBehaviorEngine').CandidateSignal) => void`, firing only `READING_DETECTED`/`EYE_CONTACT_LOST`-equivalent signals derived from metrics (reading → 'READING_DETECTED', facesDetected===0 → 'CAMERA_COVERED_OR_ABSENT', synthetic → 'SYNTHETIC_EYE_CONTACT_DETECTED') with the same 4 s rate limit.

- [ ] **Step 1: Rewrite the component** per above. Delete the old gaze/blink/synthetic logic (now in ProctorEngine). Keep the visual panel structure and styling.

- [ ] **Step 2: Typecheck — including InterviewSimulation.tsx must still compile**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -80`
Expected: no new errors vs baseline (pay attention to `InterviewSimulation.tsx` prop usage — it passes `videoRef`, `isActive`, `onMetricsUpdate`, `onSignal`; new required props `policy`, `calibration`, `candidateSpeaking` must have defaults: make them optional with defaults `{dwellMultiplier: 1.0}`, `null`, `false`).

- [ ] **Step 3: Commit**

```bash
git add ai-interview-intro/frontend/src/components/simulation/CandidateObserver.tsx
git commit -m "feat: CandidateObserver now a ProctorEngine shell with multi-face + tracking rows"
```

---

### Task 10: RealtimeInterview wiring — sequencing fix, proctor events, live scorecard, calibration step

**Files:**
- Modify: `frontend/src/components/screens/RealtimeInterview.tsx`

**Interfaces:**
- Consumes: ws protocol from Task 5, `CandidateObserver` props from Task 9, `CalibrationOverlay` from Task 8.
- Produces: `reportData` state handed to `InterviewReport` (Task 11); step type becomes `'setup' | 'calibration' | 'interview' | 'report'`.

Changes (all in this file):

- [ ] **Step 1: Turn sequencing — kill the overlap/repeat bug**

Add refs/state:
```ts
const turnIdRef = useRef(0);
const awaitingTurnRef = useRef(false);
const [liveScore, setLiveScore] = useState<{score: number; tip: string; avg: number; count: number} | null>(null);
const [integrity, setIntegrity] = useState(100);
const [policy, setPolicy] = useState<{dwellMultiplier: number}>({ dwellMultiplier: 1.0 });
const [calibration, setCalibration] = useState<CalibrationModel | null>(null);
const [reportData, setReportData] = useState<any>(null);
const [terminatedReason, setTerminatedReason] = useState<string | null>(null);
const scoreSumRef = useRef(0); const scoreCountRef = useRef(0);
```

In `triggerVadPause`: at the top add
```ts
if (awaitingTurnRef.current) return;   // one turn in flight, never double-send
awaitingTurnRef.current = true;
turnIdRef.current += 1;
```
and include `turn_id: turnIdRef.current` and `covered_topics: coveredTopicsRef.current` in the sent JSON (add `const coveredTopicsRef = useRef<string[]>([]);`).

In `socket.onmessage` string branch:
- `turn_result`: ignore if `msg.turn_id !== undefined && msg.turn_id !== turnIdRef.current && msg.turn_id !== 0`; on accept, `awaitingTurnRef.current = false;` push assistant turn, and `if (msg.data.topic) coveredTopicsRef.current = [...coveredTopicsRef.current, msg.data.topic];` and `if (typeof msg.data.integrity_score === 'number') setIntegrity(msg.data.integrity_score);`
- new `answer_score`: `scoreSumRef.current += msg.data.score; scoreCountRef.current += 1; setLiveScore({ score: msg.data.score, tip: msg.data.one_line_tip, avg: scoreSumRef.current / scoreCountRef.current, count: scoreCountRef.current });`
- new `session_config`: `setPolicy(msg.policy ? { dwellMultiplier: msg.policy.dwell_multiplier } : { dwellMultiplier: 1.0 });`
- new `hr_interject`: nothing special — the following audio plays through the normal binary path (status flow handles it).
- new `session_terminated`: `setTerminatedReason(msg.reason); stopListening();`
- new `final_report`: `setReportData(msg.data); setStatus('idle'); setStep('report');`
- `error`: also `awaitingTurnRef.current = false;`

Kickoff message: add `turn_id: 0`.

- [ ] **Step 2: Replace cheat_detected wiring with proctor events**

Delete the tab-switch `cheat_detected` effect and the old `onSignal` block. Add:
```ts
const sendProctorEvent = useCallback((e: ProctorEvent) => {
  const s = wsRef.current;
  if (s && s.readyState === WebSocket.OPEN) {
    s.send(JSON.stringify({
      type: 'proctor_event', event_type: e.eventType, confidence: e.confidence,
      ts_ms: e.tsMs, candidate_speaking: !!lastTranscriptRef.current, meta: e.meta ?? {},
    }));
  }
}, []);
```
Tab switching effect now sends `sendProctorEvent({ eventType: 'TAB_SWITCHED', confidence: 1, tsMs: Date.now() })`; add a `blur` listener on window sending `FOCUS_LOST` the same way.

`<CandidateObserver>` gets new props: `policy={policy} calibration={calibration} candidateSpeaking={!!transcript} onProctorEvent={sendProctorEvent}`.

- [ ] **Step 3: Calibration step**

`startInterview` sets `setStep('calibration')` (after camera acquisition) instead of `'interview'`; render `CalibrationOverlay` at step `'calibration'` with `onComplete={(m) => { setCalibration(m); setStep('interview'); initWebSocket(); }}` and `onSkip={() => { setCalibration(null); setStep('interview'); initWebSocket(); }}`. CalibrationOverlay needs a FaceLandmarker — load one with the shared loader (import the loader from CandidateObserver: export `loadMediaPipeFaceLandmarker` from it).

- [ ] **Step 4: Live scorecard panel + integrity badge**

Below the Anti-Cheat panel add:
```tsx
{liveScore && (
  <div className="absolute top-6 right-6 w-56 bg-black/60 backdrop-blur-xl border border-white/20 rounded-xl p-4 z-20 shadow-2xl">
    <div className="text-[10px] uppercase font-bold text-white/50 mb-2 flex items-center justify-between">
      <span>Live Score</span>
      <span className={integrity >= 80 ? 'text-emerald-400' : integrity >= 50 ? 'text-amber-400' : 'text-red-400'}>
        Integrity {integrity}
      </span>
    </div>
    <div className="flex items-end gap-2">
      <span className="text-3xl font-black text-white">{liveScore.score.toFixed(1)}</span>
      <span className="text-white/40 text-sm mb-1">/10 · avg {liveScore.avg.toFixed(1)}</span>
    </div>
    <p className="text-[11px] text-indigo-300 mt-2 leading-snug">💡 {liveScore.tip}</p>
  </div>
)}
```

- [ ] **Step 5: End call → report**

PhoneOff button: instead of `onBack`, send `{type: 'end_call'}` if ws open (plus `stopListening()`), and show a "Preparing your interview report…" overlay while `status === 'thinking' || msgStatus === 'generating_report'` until `final_report` arrives (add `generating_report` to the status union handling — reuse the existing `status` message plumbing). If ws is not open, fall back to `onBack()`.
At step `'report'`, render `<InterviewReport data={reportData} terminatedReason={terminatedReason} onBack={onBack} />` (component from Task 11).

- [ ] **Step 6: Typecheck**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -80` — no new errors.

- [ ] **Step 7: Commit**

```bash
git add ai-interview-intro/frontend/src/components/screens/RealtimeInterview.tsx
git commit -m "feat: turn sequencing fix, proctor wiring, live scorecard, calibration step, end-call flow"
```

---

### Task 11: Interview report screen (frontend)

**Files:**
- Create: `frontend/src/components/screens/InterviewReport.tsx`

**Interfaces:**
- Consumes: `final_report` payload shape from Task 4 (`overall_score`, `verdict`, `dimension_breakdown`, `per_question[]`, `strengths[]`, `improvement_plan[]`, `integrity{score, violations[], terminated}`).
- Produces: `<InterviewReport data={report | null} terminatedReason={string | null} onBack={() => void} />`

- [ ] **Step 1: Write the component** — full-screen `bg-black` scrollable screen, glassmorphism cards:
  - Header: overall score as a big number `/10`, verdict sentence, and if `terminatedReason` a red banner "Interview terminated: <reason>".
  - Dimension grid: 5 cards (`communication, content, structure, confidence, integrity`) each with a progress bar (`bg-white/10` track, fill `bg-indigo-500`, red `bg-red-500` below 5).
  - Per-question list: question, score chip, "What worked" (`text-emerald-300`), "Stronger answer" (`text-amber-300`).
  - Strengths + Improvement plan as two columns of bullet cards.
  - Integrity timeline: violations as rows `HH:MM:SS-style offset (ts_ms → m:ss), type, action` with severity colors; "Clean session ✓" when empty.
  - `data === null` → simple "Report unavailable — connection lost." card with Back button.
  - Back button calls `onBack`.

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -50` — no new errors.

- [ ] **Step 3: Commit**

```bash
git add ai-interview-intro/frontend/src/components/screens/InterviewReport.tsx
git commit -m "feat: post-call interview report screen"
```

---

### Task 12: Full verification + manual room checklist

**Files:** none (verification only)

- [ ] **Step 1: Full automated gates**

```bash
cd "C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab\ai-interview-intro"
python -m pytest backend/tests -q
python -c "from backend.api import interview_routes; from backend.agents.dialogue_manager import dialogue_manager; print('backend ok')"
cd frontend && npx tsc --noEmit 2>&1 | head -80
```
Expected: all tests pass, `backend ok`, no new tsc errors vs baseline.

- [ ] **Step 2: Manual room checklist** (start backend + `npm run dev` in frontend; use the preview browser where available):
  1. Setup → calibration overlay appears, 5 dots, completes → room.
  2. Normal answering with moderate head movement + brief think-glances up → NO warnings (over-activity check).
  3. Look far off-screen ≥ 3 s while idle → HR gives verbal warning once, not repeatedly.
  4. Second person enters frame → warning (beginner) / escalation (faang terminates).
  5. Hold up a phone → detection within ~3 s.
  6. Switch tabs → warning; integrity badge drops after repeated violations at intermediate+.
  7. Answer a question → live scorecard updates with score + tip; HR asks a natural follow-up; no repeated questions; no overlapping/duplicated HR audio (the sequencing fix).
  8. Press PhoneOff → "Preparing report…" → report screen renders all sections.
  9. Beginner difficulty run → warnings never terminate; coach micro-tips appear in HR speech.

- [ ] **Step 3: Final commit**

```bash
cd "C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab"
git add -A ai-interview-intro/docs ai-interview-intro/frontend/src ai-interview-intro/backend
git commit -m "feat: anti-cheat proctoring v2, realtime scoring, HR buddy feedback, post-call reports"
```

---

## Self-Review Notes

- Spec coverage: ProctorEngine (T6), calibration (T8), multi-person/phone/synthetic (T6/T7), policy (T1), ledger + graduated response (T2/T5), per-answer scoring (T3/T5/T10), HR persona + covered_topics fix (T5), report (T4/T11), overlap fix (T5 backend absorb + T10 turn lock), difficulty alias bug (T5), eyeContactLostTimer bug (dies with the T9 rewrite), backups (T0). InterviewSimulation compatibility preserved via optional legacy props (T9).
- Types consistent: `ProctorEvent.eventType` (TS camelCase) maps to `event_type` (ws JSON) — conversion happens in `sendProctorEvent` (T10) and is the only translation point.
- Known deliberate scope cut: no formal frontend unit tests (no vitest infra in repo) — pure math lives in ProctorEngine for future testability; gates are tsc + manual checklist.
