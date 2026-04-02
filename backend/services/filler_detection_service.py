import re
import logging

logger = logging.getLogger(__name__)

# Comprehensive filler/hesitation patterns for Indian English interviews
FILLER_WORDS = {"um", "uh", "hmm", "ah", "er", "erm"}
FILLER_PHRASES = {
    "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "like", "so", "right",
    "okay", "ok", "well", "matlab", "literally",
    "to be honest", "in a way", "i guess"
}


class FillerDetectionService:
    """
    Production-grade filler word/phrase detector.
    Returns detected fillers, count, and density (fillers per 100 words).
    """

    def __init__(self):
        logger.info("🔍 [FillerDetection] Initialized with expanded filler dictionary")

    def detect(self, text: str) -> list:
        """Detect filler words and phrases in transcript."""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        detected = []

        # Single-word fillers
        for word in words:
            clean = re.sub(r'[^a-z]', '', word)
            if clean in FILLER_WORDS:
                detected.append(clean)

        # Multi-word filler phrases
        for phrase in FILLER_PHRASES:
            count = len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower))
            for _ in range(count):
                detected.append(phrase)

        return detected

    def detect_with_stats(self, text: str) -> dict:
        """Returns fillers + density metrics for scoring/feedback."""
        fillers = self.detect(text)
        word_count = max(1, len(text.split()))
        density = (len(fillers) / word_count) * 100  # per 100 words

        return {
            "fillers": fillers,
            "count": len(fillers),
            "density": round(density, 1),
            "word_count": word_count
        }