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

_ALIASES = {"advanced": "advance", "faang": "extreme"}

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
    "advance": {
        "dwell_multiplier": 0.8, "warnings_before_penalty": 1,
        "terminate_after_penalties": 3, "instant_terminate": (),
        "critical_repeat_terminate": 3,
    },
    "extreme": {
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
