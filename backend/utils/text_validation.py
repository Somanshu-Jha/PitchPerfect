def is_valid_response(text: str):
    """
    #---- Check if response is meaningful
    """

    if not text:
        return False

    words = text.strip().split()

    #---- Too short → ignore
    if len(words) < 4:
        return False

    #---- Common noise phrases
    noise_phrases = [
        "hello",
        "hello everyone",
        "hi",
        "thanks",
        "thank you"
    ]

    if text.lower().strip() in noise_phrases:
        return False

    return True