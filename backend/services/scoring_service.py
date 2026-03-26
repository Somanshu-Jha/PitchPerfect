# backend/services/scoring_service.py

import logging
from backend.ml_models.scoring_model import MLScoringModel

logger = logging.getLogger(__name__)


class ScoringService:

    def __init__(self):
        self.model = MLScoringModel()
        self.model.load()

    def evaluate(self, text, semantic, audio_features, fillers):
        """
        Safe scoring
        """

        try:
            features = self.build_features(
                text,
                semantic,
                audio_features,
                fillers
            )

            result = self.model.predict(features)

            # If model returns dict → already final
            if isinstance(result, dict):
                return result

            # If model returns raw score
            return {
                "overall_score": float(result),
                "confidence": "medium"
            }

        except Exception as e:
            logger.error(f"❌ Scoring failed: {e}")

            return {
                "overall_score": 6,
                "confidence": "low",
                "note": "Fallback scoring used"
            }

    # ---------------- FEATURE BUILDER ----------------
    def build_features(self, text, semantic, audio_features, fillers):

        # semantic = { "intent": { "detected": [...], "confidence": 0.x }, "structured": {...} }
        intent = semantic.get("intent", {})
        structured = semantic.get("structured", {})

        return {
            "text_length": len(text.split()),
            "confidence": intent.get("confidence", 0),
            "speech_rate": audio_features.get("speech_rate", 0) if isinstance(audio_features, dict) else 0,
            "pitch": audio_features.get("pitch", 0) if isinstance(audio_features, dict) else 0,
            "filler_count": len(fillers) if isinstance(fillers, list) else 0,
            "structured": structured,   # ← passed to MLScoringModel.rule_based_score
        }