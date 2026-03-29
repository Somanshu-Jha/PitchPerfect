# =====================================================================
# FEEDBACK SERVICE — Non-Repetitive GenAI Coaching Engine
# =====================================================================
# Generates personalized interview coaching using LLM + RAG context.
#
# Two-Pass Architecture:
#   Pass 1 (deterministic, temp=0.0):
#     - Extract structured coaching POINTS from transcript
#     - {"strengths": [...], "weaknesses": [...]} as topic phrases
#     - Same input → same points ALWAYS
#
#   Pass 2 (creative, temp=0.7):
#     - Convert points → full coaching sentences
#     - Persona injection → linguistic style variation
#     - Different wording each time, same logical content
#
# Additional mechanisms:
#   - Deterministic Contradiction Guard (blocks hallucinated "missing" fields)
#   - Sentence-embedding similarity blocker (threshold: 0.45)
#   - Count enforcement: 4-8 strengths, 4-8 improvements
#   - RAG historical context injection
#   - English level tone adaptation
#
# All parameters documented with impact notes.
# =====================================================================

import logging
import textdistance
from collections import deque
from backend.core.genai_engine import genai_engine
from backend.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# ── VARIATION MEMORY ────────────────────────────────────────────────────
# Stores flattened feedback text per user to detect repetition.
# maxlen=3 → only compares against last 3 attempts
VARIATION_MEMORY = {}

# ── SIMILARITY THRESHOLD ───────────────────────────────────────────────
# 0.40 → regenerate WORDING if feedback is >40% similar to any recent attempt
# Note: POINTS (logic) can be similar — only wording must vary
SIMILARITY_THRESHOLD = 0.40

# ── WORDING TEMPERATURE ESCALATION ────────────────────────────────────
# Each wording retry uses a higher temperature to force more creative sentences
# 0.7 → controlled creativity (first attempt)
# 0.85 → moderate randomness (second attempt)
# 1.0 → maximum diversity (final attempt)
WORDING_TEMP_SEQUENCE = [0.7, 0.85, 1.0]


class FeedbackService:
    """
    Two-pass Non-Repetitive Feedback System.
    
    Pass 1: Deterministic structured point extraction (same logic every time)
    Pass 2: Varied wording generation (sentence-level diversity per session)
    
    Integrates FAISS RAG for historical context.
    """

    def __init__(self):
        logger.info("🧠 [FeedbackService] Routing to GenAI Two-Pass Engine + RAG Database...")
        self.rag_service = RAGService()
        # SBERT for embedding-based similarity blocking
        self._sbert = None
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info("✅ [FeedbackService] SBERT loaded for embedding similarity.")
        except Exception as e:
            logger.warning(f"⚠️ [FeedbackService] SBERT unavailable, using jaccard-only: {e}")

    def _deterministic_guard(self, improvements: list, semantic_data: dict) -> list:
        """
        CONTRADICTION GUARD:
        Removes feedback items claiming a field is "missing" when the semantic
        extraction actually found data for it.
        Prevents the most common LLM hallucination: saying something is missing
        when it was successfully extracted.
        """
        guarded = []
        missing_keywords = ["missing", "didn't mention", "forgot", "lack", "did not mention", "no mention"]

        feature_map = {
            "role": ["experience", "career_goals"],
            "experience": ["experience"],
            "education": ["education"],
            "degree": ["education"],
            "name": ["name"],
            "skills": ["skills"],
            "strength": ["strengths"],
            "interest": ["areas_of_interest"],
            "goal": ["career_goals"],
            "quality": ["qualities"]
        }

        for imp in improvements:
            lower_imp = imp.lower()
            is_contradiction = False

            if any(k in lower_imp for k in missing_keywords):
                for keyword, semantic_keys in feature_map.items():
                    if keyword in lower_imp:
                        for sk in semantic_keys:
                            data = semantic_data.get(sk, {})
                            if isinstance(data, list) and len(data) > 0 and data[0].get("value"):
                                is_contradiction = True
                                break
                            elif isinstance(data, dict) and data.get("value"):
                                is_contradiction = True
                                break
                    if is_contradiction:
                        break

            if is_contradiction:
                logger.warning(f"🛡️ [Guard] Blocked contradicting feedback: '{imp}'")
            else:
                guarded.append(imp)

        if not guarded:
            guarded.append("Continue practicing to increase the overall depth of your responses.")

        return guarded

    def _compute_similarity(self, new_text: str, old_text: str) -> float:
        """
        Wording similarity check using SBERT embeddings (cosine similarity).
        Only used to vary WORDING — not to block repeating the same logical POINTS.
        """
        if self._sbert is not None:
            try:
                import numpy as np
                embeddings = self._sbert.encode([new_text, old_text])
                cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
                ))
                return max(0.0, cos_sim)
            except Exception:
                pass
        # Jaccard fallback
        set1 = set(new_text.lower().split())
        set2 = set(old_text.lower().split())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def generate(
        self,
        user_id: str,
        transcript: str,
        semantic: dict,
        scores: dict,
        fillers: list,
        english_level: str = "Intermediate"
    ) -> dict:
        """
        Main entry point: produces dynamic, non-repetitive coaching feedback.

        Architecture:
        1. Extract structured POINTS (deterministic, temp=0.0)
           - Same input → same points
        2. Generate WORDING from points (varied, temp=0.7-1.0)
           - Different sentences each call
        3. Apply similarity blocking if wording too similar to history
           - Only blocks wording repetition, not point repetition

        Args:
            user_id: for tracking variation memory and RAG context
            transcript: the refined transcript text
            semantic: full semantic extraction result
            scores: current scoring data
            fillers: list of detected filler words
            english_level: Beginner/Intermediate/Advanced — controls tone
        """
        if not transcript.strip():
            return {
                "positives": ["System initialized."],
                "improvements": ["No transcript found. Please speak louder."],
                "coaching_summary": "Audio was unintelligible or silent."
            }

        # ── RAG CONTEXT RETRIEVAL ───────────────────────────────────────────
        logger.info("⚡ [FeedbackService] Querying Local RAG VectorDB...")
        historical_records = self.rag_service.retrieve_context(user_id, transcript, top_k=2)
        historical_context = []
        if historical_records:
            historical_context = [rec.get("improvements", []) for rec in historical_records if rec.get("improvements")]

        semantic_structured = semantic.get("structured", {})

        # ── ENGLISH LEVEL TONE NOTE ─────────────────────────────────────────
        tone_instruction = f"User English level: {english_level}. "
        if english_level == "Advanced":
            tone_instruction += "Use deep industry terminology and critique advanced narrative structure."
        elif english_level == "Beginner":
            tone_instruction += "Use highly simplified, encouraging language. Avoid jargon."
        else:
            tone_instruction += "Use standard professional coaching language."

        filler_note = ""
        if fillers and len(fillers) > 3:
            filler_note = f" Candidate used {len(fillers)} filler words — include advice to reduce fillers."

        logger.info(f"⏳ [FeedbackService] PASS 1: Extracting structured points (temp=0.0)...")

        # ── PASS 1: STRUCTURED POINT EXTRACTION (DETERMINISTIC) ─────────────
        # Same input → same points. This defines WHAT to coach.
        inject_data = {
            "entity_extractions": semantic_structured,
            "dl_score": scores.get("overall_score", 0),
            "filler_count": len(fillers) if fillers else 0,
            "system_override_note": tone_instruction + filler_note
        }

        structure = genai_engine.extract_feedback_structure(
            transcript, inject_data, historical_context
        )
        logger.info(f"✅ [FeedbackService] Structure extracted: {len(structure.get('strengths', []))} strengths, {len(structure.get('weaknesses', []))} weaknesses")

        # ── PASS 2: WORDING GENERATION WITH SIMILARITY BLOCKING ─────────────
        # Generate wording from structure. Retry with higher temperature if
        # the generated wording is too similar to recent history.
        import concurrent.futures

        def _gen_wording(temp_val):
            result = genai_engine.generate_feedback_wording(structure, english_level, temperature_override=temp_val)
            # Apply contradiction guard to improvements
            result["improvements"] = self._deterministic_guard(
                result.get("improvements", []),
                semantic_structured
            )
            return temp_val, result

        logger.info("🎨 [FeedbackService] PASS 2: Generating wording (multi-sample, similarity-blocked)...")

        final_feedback = None
        best_sim = 1.0

        for attempt in range(2):
            samples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(_gen_wording, t): t for t in WORDING_TEMP_SEQUENCE}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        samples.append(future.result())
                    except Exception as e:
                        logger.error(f"Wording sample failed: {e}")

            if not samples:
                break

            # Select sample with lowest similarity to history
            for temp_val, sample in samples:
                if user_id not in VARIATION_MEMORY or not list(VARIATION_MEMORY[user_id]):
                    final_feedback = sample
                    best_sim = 0.0
                    logger.info(f"✅ Selected temp={temp_val} (No wording history).")
                    break

                new_text = " ".join(sample.get("improvements", [])).lower()
                sample_max_sim = 0.0
                for old_text in list(VARIATION_MEMORY[user_id]):
                    sim = self._compute_similarity(new_text, old_text)
                    if sim > sample_max_sim:
                        sample_max_sim = sim

                if sample_max_sim < best_sim:
                    best_sim = sample_max_sim
                    final_feedback = sample

            if final_feedback is not None and best_sim < SIMILARITY_THRESHOLD:
                logger.info(f"✅ Found varied wording on attempt {attempt+1} (sim={best_sim:.3f} < {SIMILARITY_THRESHOLD}).")
                break

            logger.warning(f"⚠️ Attempt {attempt+1}: All wordings repetitive (best sim={best_sim:.3f}). Regenerating...")

        if final_feedback is None:
            logger.warning("⚠️ All wording generators failed. Using point text directly as fallback.")
            final_feedback = {
                "positives": structure.get("strengths", ["Good effort."]),
                "improvements": structure.get("weaknesses", ["Keep practicing."]),
                "coaching_summary": "Your performance showed dedication. Continue developing your interview skills."
            }

        # ── ENFORCE COUNT CONSTRAINTS ───────────────────────────────────────
        positives = final_feedback.get("positives", [])
        improvements = final_feedback.get("improvements", [])

        # Ensure 4–8 items
        if len(positives) < 4:
            positives += ["Good effort demonstrated."] * (4 - len(positives))
        if len(improvements) < 4:
            improvements += ["Continue practicing to improve depth."] * (4 - len(improvements))
        positives = positives[:8]
        improvements = improvements[:8]

        logger.info(f"✅ [FeedbackService] Final: {len(positives)} strengths, {len(improvements)} improvements")

        # ── STORE IN VARIATION MEMORY ───────────────────────────────────────
        flattened_text = " ".join(improvements).lower()
        if user_id not in VARIATION_MEMORY:
            VARIATION_MEMORY[user_id] = deque(maxlen=3)
        VARIATION_MEMORY[user_id].append(flattened_text)

        # Ingest into RAG for long-term memory
        self.rag_service.ingest(user_id, transcript, semantic_structured, final_feedback, scores)

        return {
            "positives": positives,
            "improvements": improvements,
            "coaching_summary": final_feedback.get("coaching_summary", "Excellent effort overall.")
        }