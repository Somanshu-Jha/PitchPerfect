import json
import os
import logging
import jellyfish
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class PhoneticService:
    """
    Ultimate Name Correction System (Phase 4) — 4-Stage Pipeline:
    Stage 1: Direct fuzzy match (>85 score) against known Indian names dataset.
    Stage 2: Soundex phonetic match against acoustic pool from dataset.
    Stage 3: Transcript validation — confirm corrected name sounds like something in the raw ASR.
    Stage 4: Fallback — trust LLM extraction capitalized.
    """

    def __init__(self):
        self.dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "indian_names.json")
        self.known_names = self._load_dataset()

    def _load_dataset(self) -> list:
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                names = data.get("names", [])
                logger.info(f"✅ [PhoneticService] Loaded {len(names)} known names from dataset.")
                return names
        except Exception as e:
            logger.error(f"❌ [PhoneticService] Failed to load {self.dataset_path}: {e}")
            return []

    def correct(self, extracted_name: str, transcript_context: str, raw_transcript: str = "") -> str:
        """
        Full 4-stage name correction pipeline.
        
        Args:
            extracted_name:     Name as extracted by the LLM from the refined transcript.
            transcript_context: The refined transcript (used as a string context anchor).
            raw_transcript:     The raw ASR output before correction (used for Stage 3 validation).
        
        Returns:
            The most phonetically accurate name string.
        """
        if not extracted_name:
            return extracted_name

        if not self.known_names:
            logger.warning("⚠️ [PhoneticService] No dataset available, returning capitalized LLM name.")
            return extracted_name.capitalize()

        # ── Stage 1: Direct High-Confidence Fuzzy Match ──────────────────────────
        # Handles slight spelling errors (e.g. "Rahull" → "Rahul")
        best_match, score, _ = process.extractOne(
            extracted_name, self.known_names, scorer=fuzz.ratio
        )

        if score > 85:
            logger.info(
                f"✅ [NamePipeline | Stage1] Precise fuzzy match: "
                f"'{extracted_name}' → '{best_match}' ({score}%)"
            )
            corrected = best_match
        else:
            # ── Stage 2: Soundex Phonetic Match ──────────────────────────────────
            # Handles deep ASR acoustic misinterpretations (e.g. "Super man" → "Subramanian")
            extracted_soundex = jellyfish.soundex(extracted_name)

            acoustic_pool = [
                name for name in self.known_names
                if jellyfish.soundex(name) == extracted_soundex
            ]

            if acoustic_pool:
                subset_best, sub_score, _ = process.extractOne(
                    extracted_name, acoustic_pool, scorer=fuzz.ratio
                )
                logger.info(
                    f"🔊 [NamePipeline | Stage2] Acoustic override: "
                    f"'{extracted_name}' → '{subset_best}' (Soundex: {extracted_soundex})"
                )
                corrected = subset_best
            else:
                # No phonetic match — trust LLM baseline
                logger.info(
                    f"⚠️ [NamePipeline | Stage2] No acoustic match. "
                    f"Trusting LLM baseline: '{extracted_name}'"
                )
                corrected = extracted_name.capitalize()

        # ── Stage 3: Transcript Validation ───────────────────────────────────────
        # Verify the corrected name actually reflects something said in the raw ASR output.
        # This prevents overcorrection when the dataset match is phonetically close but wrong.
        validated_name = self._validate_against_transcript(corrected, extracted_name, raw_transcript)

        # ── Stage 4: LLM Validation ───────────────────────────────────────
        from backend.core.genai_engine import genai_engine
        final_name = genai_engine.validate_name(extracted_name, validated_name, transcript_context)
        
        logger.info(f"🏁 [NamePipeline | Final LLM Chosen] '{extracted_name}' → '{final_name}'")
        return final_name

    def _validate_against_transcript(
        self, corrected_name: str, original_name: str, raw_transcript: str
    ) -> str:
        """
        Stage 3 — Transcript Validation.
        
        Cross-references the phonetically-corrected name against the raw ASR transcript.
        If the corrected name (or a phonetically close word) exists in the raw ASR output,
        we trust the correction. Otherwise, we revert to the LLM-extracted name to avoid
        overwriting a name with a dataset artifact.
        
        Logic:
        - Split raw transcript into individual tokens.
        - For each token, compute fuzzy similarity to the corrected name.
        - If any token scores > 60 with corrected_name → validation PASSES.
        - If no token matches → revert to original_name (LLM extraction).
        """
        if not raw_transcript or not corrected_name:
            return corrected_name

        raw_words = raw_transcript.lower().split()

        # Check if corrected name sounds like anything in the raw ASR
        for word in raw_words:
            # Skip very short words (articles, prepositions)
            if len(word) < 3:
                continue
            similarity = fuzz.ratio(corrected_name.lower(), word)
            if similarity > 60:
                logger.info(
                    f"✅ [NamePipeline | Stage3] Transcript validation PASSED: "
                    f"'{corrected_name}' matched word '{word}' ({similarity}%) in raw ASR."
                )
                return corrected_name

        # No match found — the phonetic correction may have overshot
        # Revert to the LLM-extracted name which is grounded in the refined transcript
        logger.warning(
            f"⚠️ [NamePipeline | Stage3] Transcript validation FAILED for '{corrected_name}'. "
            f"Reverting to LLM extracted name: '{original_name}'."
        )
        return original_name.capitalize()