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

    def _keyword_fallback(self, text: str) -> dict:
        """
        Lightweight regex + keyword based extraction when LLM fails.
        Returns structured dict in the same format as GenAI output.
        """
        import re
        result = {}
        text_lower = text.lower()
        
        # Name extraction: "my name is X" / "I am X" / "I'm X"
        name_match = re.search(r"(?:my name is|i am|i'm|this is|myself)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
        if name_match:
            result["name"] = {"value": name_match.group(1), "confidence": 0.8, "evidence": name_match.group(0)}
        
        # Education extraction
        edu_match = re.search(r"(?:studying|student|enrolled|pursuing|degree|btech|b\.tech|university|college|institute)\s+(.{10,80}?)(?:\.|,|and\s)", text, re.IGNORECASE)
        if edu_match:
            result["education"] = {"value": edu_match.group(0).strip("., "), "confidence": 0.7, "evidence": edu_match.group(0)}
        
        # Skills extraction — find tech keywords
        tech_kw = ["python", "java", "javascript", "react", "machine learning", "deep learning",
                   "ai", "sql", "html", "css", "node", "flutter", "docker", "aws", "git",
                   "c++", "data science", "tensorflow", "pytorch", "web development"]
        found_skills = [kw for kw in tech_kw if kw in text_lower]
        if found_skills:
            result["skills"] = [{"value": s, "confidence": 0.7, "evidence": f"'{s}' found in transcript"} for s in found_skills[:8]]
        
        # Career goals
        goal_match = re.search(r"(?:goal|aspire|want to|aim|dream|become|future|career)\s+(.{10,80}?)(?:\.|$)", text, re.IGNORECASE)
        if goal_match:
            result["career_goals"] = {"value": goal_match.group(0).strip("., "), "confidence": 0.6, "evidence": goal_match.group(0)}
        
        # Experience
        exp_match = re.search(r"(?:worked|working|intern|project|built|developed|team)\s+(.{10,80}?)(?:\.|,|$)", text, re.IGNORECASE)
        if exp_match:
            result["experience"] = [{"value": exp_match.group(0).strip("., "), "confidence": 0.6, "evidence": exp_match.group(0)}]
        
        return result

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

        # ── Step 1b: Keyword-based fallback if LLM extraction failed ─────────────
        # When the LLM returns empty (0 intents), use regex patterns on the raw text
        if not genai_structured or len(genai_structured) == 0:
            logger.warning("⚠️ [SemanticService] LLM extraction empty — using keyword fallback")
            genai_structured = self._keyword_fallback(text)

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