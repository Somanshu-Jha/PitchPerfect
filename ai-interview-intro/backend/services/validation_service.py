# -------------------- CLASS --------------------
class ValidationService:

    def __init__(self):
        """
        #---- Minimum words threshold
        """
        self.min_words = 5

    # -------------------- VALIDATION --------------------
    def validate(self, text, audio_features):
        """
        #---- Reject weak / invalid responses BEFORE ML
        """

        if not text:
            return False, "Empty response"

        words = text.strip().split()

        #---- Too short response
        if len(words) < self.min_words:
            return False, "Response too short"

        #---- Silence / low speaking
        if audio_features.get("speech_rate", 0) < 20:
            return False, "Too slow or silence detected"

        #---- Low energy audio
        if audio_features.get("energy", 0) < 0.01:
            return False, "Low audio energy"

        return True, "Valid response"