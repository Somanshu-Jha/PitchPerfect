# -------------------- CLASS --------------------
class SemanticFilter:

    def __init__(self):
        """
        #---- Confidence threshold
        """
        self.threshold = 0.6

    # -------------------- FILTER --------------------
    def apply(self, semantic_result: dict):
        """
        #---- Reject low confidence predictions
        """

        if semantic_result["confidence"] < self.threshold:
            return {
                "intent": None,
                "confidence": semantic_result["confidence"],
                "accepted": False
            }

        return {
            "intent": semantic_result["intent"],
            "confidence": semantic_result["confidence"],
            "accepted": True
        }