# =====================================================================
# FILLER DETECTION SERVICE — Context-Aware Production-Grade Detector
# =====================================================================
# Detects filler words/phrases with context-aware disambiguation:
#   - Distinguishes genuine "like" (comparison/preference) from filler "like"
#   - Per-type filler counts for detailed reporting
#   - Position clustering (beginning/middle/end of pitch)
#   - Self-correction pattern detection ("I mean", restarts)
#   - Repeated starter detection ("so so so", "and and")
# =====================================================================

import re
import logging

logger = logging.getLogger(__name__)

# ── Single-word fillers (always a filler regardless of context) ──
PURE_FILLER_WORDS = {"um", "uh", "hmm", "ah", "er", "erm", "uhh", "umm",
                     "hm", "ehm", "ahh"}

# ── Phrase fillers (multi-word hesitation markers) ──
FILLER_PHRASES = {
    "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "right", "so yeah",
    "okay so", "ok so", "well basically",
    "to be honest", "in a way", "i guess",
    "literally", "at the end of the day",
}

# ── Context-dependent fillers ──
# These words are ONLY fillers when used in specific contexts.
# "like" → filler when: "I was like", "like um", standalone filler
# "like" → genuine when: "I like Python", "technologies like React"
# "so" → filler when: sentence-starter without continuation purpose
# "so" → genuine when: "so that", "so I built"
# "well" → filler when: sentence-starter hesitation
# "well" → genuine when: "as well", "very well", "well-known"

# Patterns where "like" is GENUINE (not a filler)
_GENUINE_LIKE_PATTERNS = [
    r'\bi\s+like\s+(?:python|java|javascript|react|node|sql|coding|programming|working|building|learning|reading|designing|developing|creating|using|exploring|studying|teaching|writing|to\b)',
    r'\b(?:would|i\'d|i\'ll|we)\s+like\s+to\b',
    r'\b(?:technologies?|tools?|frameworks?|languages?|skills?|things?|subjects?|platforms?|areas?|fields?|topics?)\s+like\b',
    r'\bsomething\s+like\b',
    r'\bjust\s+like\b',
    r'\blooks?\s+like\b',
    r'\bfeel\s+like\b',
    r'\bsounds?\s+like\b',
    r'\bseems?\s+like\b',
    r'\bmore\s+like\b',
    r'\bmuch\s+like\b',
    r'\bnothing\s+like\b',
    r'\banything\s+like\b',
    r'\b(?:don\'t|didn\'t|doesn\'t|do|does|did|would|could|can|will)\s+(?:not\s+)?like\b',
]

# Patterns where "so" is GENUINE (not a filler)
_GENUINE_SO_PATTERNS = [
    r'\bso\s+that\b',
    r'\bso\s+i\s+(?:built|developed|created|decided|started|began|learned|chose|went|applied|completed)\b',
    r'\bso\s+(?:far|much|many|long|well|quickly|deeply|effectively|successfully)\b',
    r'\band\s+so\b',
    r'\bdo\s+so\b',
    r'\beven\s+so\b',
    r'\balso\s+so\b',
    r'\bnot\s+so\b',
]

# Patterns where "well" is GENUINE (not a filler)
_GENUINE_WELL_PATTERNS = [
    r'\bas\s+well\b',
    r'\bvery\s+well\b',
    r'\bquite\s+well\b',
    r'\bpretty\s+well\b',
    r'\breally\s+well\b',
    r'\bwell[- ](?:known|versed|equipped|prepared|rounded|structured|organized|defined|established|documented)\b',
    r'\bdoing\s+well\b',
    r'\bgoing\s+well\b',
    r'\bworked\s+well\b',
    r'\bworks?\s+well\b',
    r'\bperformed?\s+well\b',
]


class FillerDetectionService:
    """
    Context-aware filler word/phrase detector.

    Key feature: Disambiguates genuine usage from filler usage for
    context-dependent words (like, so, well). Uses regex patterns to
    check surrounding context before flagging.

    Returns:
    - Detected filler instances
    - Per-type counts (e.g., {"um": 3, "basically": 2})
    - Position clustering (beginning/middle/end of pitch)
    - Most frequent filler identification
    - Self-correction patterns
    """

    def __init__(self):
        # Pre-compile genuine patterns for speed
        self._genuine_like = [re.compile(p, re.IGNORECASE) for p in _GENUINE_LIKE_PATTERNS]
        self._genuine_so = [re.compile(p, re.IGNORECASE) for p in _GENUINE_SO_PATTERNS]
        self._genuine_well = [re.compile(p, re.IGNORECASE) for p in _GENUINE_WELL_PATTERNS]
        logger.info("🔍 [FillerDetection] Initialized with context-aware disambiguation")

    def _is_genuine_like(self, text: str, position: int) -> bool:
        """Check if 'like' at given position is genuine, not a filler."""
        # Get surrounding context (50 chars before and after)
        start = max(0, position - 50)
        end = min(len(text), position + 60)
        context = text[start:end].lower()

        for pattern in self._genuine_like:
            if pattern.search(context):
                return True
        return False

    def _is_genuine_so(self, text: str, position: int) -> bool:
        """Check if 'so' at given position is genuine."""
        start = max(0, position - 30)
        end = min(len(text), position + 60)
        context = text[start:end].lower()

        for pattern in self._genuine_so:
            if pattern.search(context):
                return True
        return False

    def _is_genuine_well(self, text: str, position: int) -> bool:
        """Check if 'well' at given position is genuine."""
        start = max(0, position - 30)
        end = min(len(text), position + 60)
        context = text[start:end].lower()

        for pattern in self._genuine_well:
            if pattern.search(context):
                return True
        return False

    def detect(self, text: str) -> list:
        """Detect filler words and phrases with context awareness. Returns list of filler strings."""
        if not text or not text.strip():
            return []

        text_lower = text.lower()
        words = text_lower.split()
        detected = []

        # ── Pure single-word fillers (always a filler) ──
        for word in words:
            clean = re.sub(r'[^a-z]', '', word)
            if clean in PURE_FILLER_WORDS:
                detected.append(clean)

        # ── Multi-word filler phrases ──
        for phrase in FILLER_PHRASES:
            count = len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower))
            for _ in range(count):
                detected.append(phrase)

        # ── Context-aware "like" detection ──
        for match in re.finditer(r'\blike\b', text_lower):
            if not self._is_genuine_like(text_lower, match.start()):
                detected.append("like")

        # ── Context-aware "so" detection (only sentence starters) ──
        # "so" is a filler when it starts a sentence/clause without purpose
        for match in re.finditer(r'(?:^|[.!?]\s+)so\b', text_lower):
            if not self._is_genuine_so(text_lower, match.start()):
                detected.append("so")

        # ── Context-aware "well" detection (only sentence starters) ──
        for match in re.finditer(r'(?:^|[.!?]\s+)well\b', text_lower):
            if not self._is_genuine_well(text_lower, match.start()):
                detected.append("well")

        # ── Repeated starters ("and and", "but but", "so so") ──
        repeated = re.findall(r'\b(\w+)\s+\1\b', text_lower)
        for r in repeated:
            if r in {"and", "but", "so", "the", "i", "is", "a", "um", "uh"}:
                detected.append(f"{r} {r}")

        # ── Self-correction patterns ──
        correction_patterns = [
            r'\bi\s+mean\s+(?:like|no|actually|sorry)',
            r'\bwait\s+(?:no|sorry|actually)',
            r'\bsorry\s+(?:i\s+mean|let\s+me)',
            r'\bno\s+(?:wait|sorry|actually)\b',
        ]
        for pattern in correction_patterns:
            if re.search(pattern, text_lower):
                detected.append("self-correction")

        return detected

    def detect_with_stats(self, text: str) -> dict:
        """
        Full filler analysis with per-type counts, position clustering,
        and most frequent filler identification.
        """
        if not text or not text.strip():
            return {
                "fillers": [], "count": 0, "density": 0.0,
                "word_count": 0, "per_type": {},
                "most_frequent": None, "position_cluster": "none",
                "self_corrections": 0
            }

        fillers = self.detect(text)
        word_count = max(1, len(text.split()))
        density = (len(fillers) / word_count) * 100

        # ── Per-type counts ──
        per_type = {}
        self_corrections = 0
        for f in fillers:
            if f == "self-correction":
                self_corrections += 1
            else:
                per_type[f] = per_type.get(f, 0) + 1

        # ── Most frequent filler ──
        most_frequent = None
        if per_type:
            most_frequent = max(per_type, key=per_type.get)

        # ── Position clustering ──
        # Divide text into thirds, detect where fillers concentrate
        text_lower = text.lower()
        words = text_lower.split()
        total = len(words)
        if total > 6:
            third = total // 3
            first_third = " ".join(words[:third])
            mid_third = " ".join(words[third:2*third])
            last_third = " ".join(words[2*third:])

            counts = [0, 0, 0]
            for f in fillers:
                if f == "self-correction":
                    continue
                f_lower = f.lower()
                counts[0] += first_third.count(f_lower)
                counts[1] += mid_third.count(f_lower)
                counts[2] += last_third.count(f_lower)

            total_filler_positions = sum(counts)
            if total_filler_positions > 0:
                ratios = [c / total_filler_positions for c in counts]
                max_ratio = max(ratios)
                if max_ratio > 0.5:
                    idx = ratios.index(max_ratio)
                    position_cluster = ["beginning", "middle", "end"][idx]
                else:
                    position_cluster = "distributed"
            else:
                position_cluster = "none"
        else:
            position_cluster = "none"

        result = {
            "fillers": fillers,
            "count": len(fillers),
            "density": round(density, 1),
            "word_count": word_count,
            "per_type": per_type,
            "most_frequent": most_frequent,
            "position_cluster": position_cluster,
            "self_corrections": self_corrections,
        }

        if fillers:
            logger.info(
                f"🔍 [FillerDetection] {len(fillers)} fillers | "
                f"Top: {most_frequent}({per_type.get(most_frequent, 0)}x) | "
                f"Cluster: {position_cluster} | "
                f"SelfCorr: {self_corrections}"
            )

        return result