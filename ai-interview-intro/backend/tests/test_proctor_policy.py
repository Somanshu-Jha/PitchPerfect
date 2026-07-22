from backend.interview.proctor_policy import (
    VIOLATION_SEVERITY, resolve_policy, decide_action,
)


def test_resolve_policy_aliases_and_defaults():
    assert resolve_policy("beginner")["dwell_multiplier"] == 1.5
    assert resolve_policy("advanced")["key"] == "advance"
    assert resolve_policy("FAANG")["dwell_multiplier"] == 0.6
    assert resolve_policy("nonsense")["key"] == "intermediate"


def test_beginner_never_terminates_or_penalizes():
    p = resolve_policy("beginner")
    for i in range(10):
        assert decide_action(p, "PHONE_DETECTED", prior_warnings_for_type=i,
                             prior_penalties_total=0, prior_count_for_type=i) == "WARN"


def test_intermediate_graduation():
    p = resolve_policy("intermediate")
    assert decide_action(p, "GAZE_OFF_SCREEN", 0, 0, 0) == "WARN"
    assert decide_action(p, "GAZE_OFF_SCREEN", 1, 0, 1) == "WARN"
    assert decide_action(p, "GAZE_OFF_SCREEN", 2, 0, 2) == "PENALIZE"
    assert decide_action(p, "SECOND_PERSON", 2, 1, 2) == "TERMINATE"


def test_faang_instant_terminate():
    p = resolve_policy("faang")
    assert decide_action(p, "PHONE_DETECTED", 0, 0, 0) == "TERMINATE"
    assert decide_action(p, "SECOND_PERSON", 0, 0, 0) == "TERMINATE"
    assert decide_action(p, "GAZE_OFF_SCREEN", 0, 0, 0) == "WARN"
    assert decide_action(p, "GAZE_OFF_SCREEN", 1, 0, 1) == "PENALIZE"
    assert decide_action(p, "GAZE_OFF_SCREEN", 1, 1, 2) == "TERMINATE"


def test_severity_table_complete():
    for t in ["GAZE_OFF_SCREEN", "READING_DETECTED", "FOCUS_LOST",
              "SECOND_PERSON", "CAMERA_ABSENT", "PHONE_DETECTED",
              "SYNTHETIC_EYE_CONTACT", "TAB_SWITCHED"]:
        assert VIOLATION_SEVERITY[t] in ("warn", "critical")
