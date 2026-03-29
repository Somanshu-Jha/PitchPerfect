import logging
from backend.core.genai_engine import genai_engine
from backend.nlp.phonetic import PhoneticService

logger = logging.getLogger(__name__)

# Fields where GenAI returns a list of dicts [{value, confidence, evidence}, ...]
_ARRAY_FIELDS = {"skills", "strengths", "areas_of_interest", "qualities", "experience"}

# Fields where GenAI returns a single dict {value, confidence, evidence}
_SCALAR_FIELDS = {"greetings", "name", "education", "career_goals"}

# All 9 tracked fields
_ALL_FIELDS = _SCALAR_FIELDS | _ARRAY_FIELDS


class SemanticService:
    """
    Production-Grade Semantic Extraction System.
    
    Relies strictly on Local LLM GenAI for structured extraction across 9 semantic fields:
      - Scalar: greetings, name, education, career_goals
      - Array:  skills, strengths, areas_of_interest, qualities, experience
    
    Each extracted field carries: value, confidence, evidence.
    
    Pipeline:
      1. GenAI JSON extraction (9-field schema)
      2. Normalization  — flatten nested {value, confidence, evidence} → clean_structured
      3. Name correction — phonetic (Stage 1+2) → transcript validation (Stage 3) → final
      4. Intent mapping  — build pseudo-intents for legacy API compatibility
    
    Maintains backward-compatible API:
      {"intent": {...}, "structured": {...}}
    """

    def __init__(self):
        logger.info("🧠 [SemanticService] Initializing GenAI router.")
        self.phonetic_service = PhoneticService()

    def analyze(self, text: str, raw_text: str = "", precomputed_genai: dict = None) -> dict:
        """
        Execute semantic extraction on refined transcript text.

        Args:
            text:              Refined/corrected transcript (input to LLM extraction).
            raw_text:          Raw ASR transcript before correction (used for Stage 3 name validation).
            precomputed_genai: If provided, skip LLM extraction and normalize this dict directly.
                               Used by the combined single-LLM call path to avoid redundant inference.

        Returns:
            {
                "intent": {"detected": [...], "confidence": float},
                "structured": {field: value_or_list, ...},
                "confidence_map": {field: float, ...},
                "evidence_map": {field: str_or_list, ...}
            }
        """
        if not text.strip():
            logger.warning("⚡ [SemanticService] Empty transcript received — returning empty result.")
            return {
                "intent": {"detected": [], "confidence": 0.0},
                "structured": {},
                "confidence_map": {},
                "evidence_map": {}
            }

        logger.info(f"🔍 [SemanticService] Analyzing transcript ({len(text)} chars)...")

        # ── Step 1: GenAI Extraction (or use precomputed) ────────────────────────
        if precomputed_genai is not None:
            logger.info("⚡ [SemanticService] Using precomputed GenAI output — LLM call skipped.")
            genai_structured = precomputed_genai
        else:
            genai_structured = genai_engine.extract_semantic(text)

        # ── Step 2: Normalize structured output ──────────────────────────────────
        clean_structured = {}
        confidence_map = {}
        evidence_map = {}
        all_confidences = []

        for field in _ALL_FIELDS:
            raw_field = genai_structured.get(field)

            if field in _ARRAY_FIELDS:
                # Expect: [{value, confidence, evidence}, ...]
                if isinstance(raw_field, list):
                    values = []
                    field_evidences = []
                    for item in raw_field:
                        if isinstance(item, dict):
                            v = item.get("value", "")
                            c = item.get("confidence", 0.5)
                            e = item.get("evidence", "")
                            if v:
                                values.append(v)
                                field_evidences.append(e)
                                all_confidences.append(c)
                    clean_structured[field] = values
                    confidence_map[field] = (
                        sum(c.get("confidence", 0.5) for c in raw_field if isinstance(c, dict)) / max(len(raw_field), 1)
                        if raw_field else 0.0
                    )
                    evidence_map[field] = field_evidences
                else:
                    clean_structured[field] = []
                    confidence_map[field] = 0.0
                    evidence_map[field] = []

            else:
                # Expect: {value, confidence, evidence}
                if isinstance(raw_field, dict):
                    v = raw_field.get("value", "")
                    c = raw_field.get("confidence", 0.5)
                    e = raw_field.get("evidence", "")
                    clean_structured[field] = v
                    confidence_map[field] = c
                    evidence_map[field] = e
                    if v:
                        all_confidences.append(c)
                else:
                    clean_structured[field] = ""
                    confidence_map[field] = 0.0
                    evidence_map[field] = ""

        # ── Step 3: Name Correction (4-Stage Pipeline) ───────────────────────────
        raw_name = clean_structured.get("name", "")
        if raw_name:
            corrected_name = self.phonetic_service.correct(
                extracted_name=raw_name,
                transcript_context=text,
                raw_transcript=raw_text        # Stage 3 validation uses original ASR
            )
            clean_structured["name"] = corrected_name
            logger.info(f"🏷️  [SemanticService] Name pipeline: '{raw_name}' → '{corrected_name}'")

        # ── Step 4: Intent Mapping (legacy API compatibility) ────────────────────
        avg_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        detected_intents = [
            field for field in _ALL_FIELDS
            if clean_structured.get(field)  # non-empty string or non-empty list
        ]

        logger.info(
            f"✅ [SemanticService] Extracted {len(detected_intents)} intents. "
            f"Avg confidence: {avg_confidence:.2f}. "
            f"Fields: {detected_intents}"
        )

        return {
            "intent": {
                "detected": detected_intents,
                "confidence": round(avg_confidence, 3)
            },
            "structured": clean_structured,
            "confidence_map": confidence_map,
            "evidence_map": evidence_map
        }