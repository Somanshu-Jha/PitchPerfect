from backend.agents.dialogue_manager import DialogueManager


def test_normalize_difficulty_aliases():
    dm = DialogueManager.__new__(DialogueManager)
    assert dm._normalize_difficulty("advanced") == "advance"
    assert dm._normalize_difficulty("FAANG") == "extreme"
    assert dm._normalize_difficulty("beginner") == "beginner"
    assert dm._normalize_difficulty("") == "intermediate"
