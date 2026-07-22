# Anti-Cheat, Real-Time Scoring & HR Feedback — Design Spec

**Date:** 2026-07-22
**Scope:** Realtime interview room (`RealtimeInterview.tsx` + `/interview/ws/stream` websocket flow)
**Architecture decision:** Client-side CV ProctorEngine + backend session ledger (Approach A). All computer vision runs in the candidate's browser (MediaPipe via CDN, same pattern as existing FaceLandmarker usage); the backend never processes video — mandatory for the Hugging Face free-tier deployment (2 vCPU, no GPU).

## Locked decisions

| Decision | Choice |
|---|---|
| Gaze calibration | 5-point pre-interview calibration (center + 4 corners, ~10 s), skippable → auto-baseline during HR greeting |
| Score visibility | Live scorecard in room after each answer + full post-call report |
| Violation consequences | Graduated: warn → score penalty → terminate, scaled by strictness |
| Phone detection | Included — client-side MediaPipe ObjectDetector (EfficientDet-Lite0), zero server cost |

## Current-state problems this fixes

1. `CandidateObserver.tsx` computes raw iris offset with **no head-pose compensation** — head turns register as eye movement ("only head movement detected" symptom).
2. `eyeContactLostTimer` is used (line ~348) but **never declared as a ref**; `processResult` runs inside a silent try/catch, so blink-rate, nervousness and the EYE_CONTACT_LOST signal die silently every frame.
3. `numFaces: 1` — a second person in the room is invisible.
4. No calibration; fixed thresholds guess at webcam geometry.
5. Synthetic eye-contact detection is a single variance check — weak and false-positive-prone.
6. Cheat events cause a one-off HR scolding; nothing persisted, no strictness link, no score impact.
7. No per-answer scoring in the websocket flow; PhoneOff button just leaves — **no post-call report**.
8. `covered_topics` is never sent by the realtime room, so topic-repetition protection is dead.

---

## 1. ProctorEngine (frontend)

New module `frontend/src/components/simulation/ProctorEngine.ts` (pure logic, no React) + `CandidateObserver.tsx` becomes a thin UI shell consuming its output. New `CalibrationScreen.tsx` step in the room's setup flow.

### Face pipeline
- FaceLandmarker options: `numFaces: 2`, `outputFaceBlendshapes: true`, `outputFacialTransformationMatrixes: true`, GPU delegate, VIDEO mode, CDN-loaded (`@mediapipe/tasks-vision@0.10.14`).
- **Head pose** (yaw/pitch/roll) extracted from the facial transformation matrix.
- **Gaze** = iris-offset-in-eye normalized by eye width, then **compensated by head pose**: gaze_screen = f(iris_offset, head_yaw/pitch, calibration). Head turn with eyes tracking the screen ≈ neutral gaze; head still + eyes off ≈ off-screen gaze. Both eyes averaged; per-eye disagreement lowers confidence.
- **Blink** from blendshapes `eyeBlinkLeft`/`eyeBlinkRight` (score > 0.5), not raw landmark distance.

### Calibration
- 5 targets: center, 4 corners. Candidate looks at each ~1.5 s; engine records median gaze vector per target → affine gaze→screen mapping + neutral baseline + eye-openness baseline.
- Stored in `sessionStorage` per session. Skippable; fallback auto-baseline: median gaze during the HR greeting (first ~8 s) becomes "center".
- Quality gate: if calibration samples are too noisy (variance above threshold), warn the candidate about lighting/camera and fall back to auto-baseline.

### Signal fusion — the "not over-active" rules
Every detector feeds a fusion layer; **no single frame ever flags**:
- **Dwell time**: violation state must persist (off-screen gaze > 2.5 s sustained; second face > 1.5 s; phone in ≥ 2 consecutive 1 fps detections).
- **Hysteresis**: after a flag, detector re-arms only after returning to normal for a cooldown (≥ 4 s) — no machine-gun flags.
- **Confidence gating**: face bounding box too small (< ~12 % of frame height), extreme lighting, or per-eye gaze disagreement → detector reports "insufficient confidence", never a violation.
- **Grace contexts**: thinking-glance-up allowance — brief upward gaze while speaking is not a violation (people look up to think); transcript-active check retained from current code.
- Severity levels: `info` (metrics only), `warn`, `critical`.

### Detectors
| Detector | Signal | Notes |
|---|---|---|
| Gaze off-screen | `GAZE_OFF_SCREEN` (warn) | Post-calibration zones; direction attached (L/R/U/D) |
| Reading | `READING_DETECTED` (warn) | Saccade sweep pattern + low eye contact, existing idea but on compensated gaze |
| Second person | `SECOND_PERSON` (critical) | 2nd face > 1.5 s |
| Face absent / camera covered | `CAMERA_ABSENT` (critical) | > 4 s no face |
| Phone | `PHONE_DETECTED` (critical) | ObjectDetector EfficientDet-Lite0 float16, ~1 fps throttled loop, "cell phone" score > 0.55, 2 consecutive hits |
| Synthetic eye contact | `SYNTHETIC_EYE_CONTACT` (critical) | Multi-cue, below |
| Tab switch | `TAB_SWITCHED` (critical) | existing `visibilitychange` |
| Window blur / fullscreen exit | `FOCUS_LOST` (warn) | `blur` event + fullscreen change |

### Synthetic eye-contact detection (Windows Studio Effects / NVIDIA Broadcast)
Multi-cue confidence score, all cues computed over rolling ~3 s windows:
1. **Micro-saccade absence** — natural gaze has constant tiny jitter; filtered eye-contact has an unnaturally low variance floor.
2. **Head-gaze rigidity** — head yaw changes while screen-gaze stays perfectly locked is physically implausible; natural vestibulo-ocular compensation still shows measurable iris movement.
3. **Blink decorrelation** — during real blinks iris landmarks degrade/jump; synthetic pipelines keep them locked.
4. **Jitter spectrum flatness** — FFT of gaze-x over the window; natural gaze has 1/f-like spectrum, synthetic is near-flat.
Flag only when combined confidence stays high for ≥ 5 s. Engineering honesty: probabilistic, tuned for very low false positives; multi-cue fusion is what commercial proctoring uses — literal 100 % is not achievable by any CV system.

### Output interface
```ts
interface ProctorEvent { type: ProctorViolationType; severity: 'warn'|'critical'; confidence: number; tsMs: number; meta?: Record<string, unknown>; }
interface ProctorMetrics { eyeContactPercent: number; gazeDirection: ...; headPose: {yaw,pitch,roll}; blinkRatePerMin: number; facesDetected: number; confidenceLevel: ...; nervousnessScore: number; calibrated: boolean; trackingConfidence: number; }
```
Events → websocket `{type:"proctor_event", ...}`. Metrics → local panel + per-answer aggregates attached to each `vad_pause`.

## 2. Strictness → policy mapping

Single policy table (backend `backend/interview/proctor_policy.py`), sent to frontend in a `session_config` ws message at kickoff so thresholds and consequences always agree. Keyed by existing difficulty levels:

| Level | Dwell multiplier | Warnings before penalty | Termination |
|---|---|---|---|
| beginner | 1.5× (lenient) | ∞ (warnings only) | never |
| intermediate | 1.0× | 2 | only repeated CAMERA_ABSENT / SECOND_PERSON (3+) |
| advanced | 0.8× | 1 | after 3 penalized violations |
| faang | 0.6× | 1 | after 2 penalized; PHONE or SECOND_PERSON = immediate |

## 3. Backend session ledger + graduated response

`ProctorSession` (in-memory, per websocket connection, `backend/interview/proctor_session.py`):
- Receives `proctor_event` messages; server-side cooldown/dedupe (same type ≤ 8 s apart merges).
- Policy engine actions:
  - **HR_WARN** — inject SYSTEM prompt (existing mechanism), templated per violation type × count × strictness so HR phrases warnings naturally and escalates tone ("I noticed you glanced away — all good, but try to stay with me" → "This is the second time; I need your full attention").
  - **SCORE_PENALTY** — deduction recorded in ledger (integrity −10 warn / −25 critical, capped).
  - **TERMINATE** — HR delivers a professional closing line via TTS, ws sends `session_terminated`, report generated with `integrity_flagged: true`.
- **Integrity score 0–100** from ledger; violation timeline (type, timestamp, action taken) for the report.

## 4. Real-time per-answer scoring

- On each `vad_pause` (non-kickoff): backend runs **in parallel** via `asyncio.gather`:
  1. `dialogue_manager.generate_next_turn(...)` (existing)
  2. `answer_scorer.score(question, answer, job_role, difficulty, candidate_metrics)` — fast-tier LLM, strict JSON: `{score: 0-10, dimensions: {relevance, structure, depth, communication}, one_line_tip: string}`
- Fallback: on LLM failure/JSON parse failure → existing heuristic `ScoringService`.
- New ws message `answer_score` → frontend live scorecard panel: last score /10, buddy one-line tip, running average, integrity badge. Compact, glassmorphism style, non-blocking.
- Scores + tips accumulated in `ProctorSession` for the report. Silent answers score accordingly (existing silent-candidate SYSTEM path).

## 5. HR persona + natural follow-ups

`dialogue_manager._build_system_prompt` upgrades:
- Role-aware: weave `job_role` + resume context into question selection ("You mentioned X on your resume…").
- One question at a time; acknowledge the previous answer in one natural sentence before the next question (no "Great! Moving on…" template feel — vary acknowledgment style).
- Follow-up logic explicit in prompt: vague/low-quality answer → one probing follow-up before moving on; strong answer → deeper next topic; scale probing aggression with strictness (buddy-coach at beginner: hints and encouragement; surgical at faang).
- Coach persona at beginner/intermediate: brief actionable micro-tips allowed inline; formal at advanced/faang (tips only in report).
- Wire `covered_topics` from the realtime room (fix the currently-dead repetition protection): frontend accumulates topics from `turn_result.topic` and echoes them back.

## 6. Post-call session report

- PhoneOff button → sends `{type:"end_call"}` instead of immediately leaving; UI shows "Preparing your interview report…".
- Backend aggregates: per-answer scores + tips, full transcript, proctor ledger + integrity score, CV aggregates (avg eye contact %, confidence trend, nervousness).
- One higher-tier LLM call (think/kimi tier) → structured JSON report:
  - overall score /10 + hire-signal verdict phrased as HR buddy
  - per-dimension breakdown (communication, content, structure, confidence, integrity)
  - per-question: score, what was good, what a stronger answer looks like
  - strengths (3), improvement plan for the **target role** (concrete practice actions)
  - integrity section: score + violation timeline (or clean bill)
- New `InterviewReport.tsx` screen (room's glassmorphism style); fallback to locally-assembled report if LLM fails. Persist via existing history mechanism if available; else localStorage.
- Termination path produces the same report with the termination event highlighted.

## Non-goals
- Server-side video processing (HF free tier can't).
- Literal 100 % detection guarantees (probabilistic, tuned low-false-positive).
- Screen-recording/VM/second-device detection beyond tab/focus/gaze evidence.
- Changes to InterviewSimulation.tsx flow (shares ProctorEngine benefits via CandidateObserver but its turn flow is untouched in this iteration).

## Testing
- `cd frontend && npx tsc --noEmit` — zero new errors.
- Python: import checks; unit tests for policy engine transitions (warn→penalty→terminate per strictness), scorer JSON parsing + fallback, ledger cooldown/dedupe.
- Manual: live room session per strictness level; verify no false flags during normal answering (look-up-to-think, moderate head movement), verify flags on: sustained off-screen gaze, second person entering frame, phone raised, tab switch.
