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
