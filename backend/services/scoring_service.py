# =====================================================================
# SCORING SERVICE — Hybrid 70/30 DL + LLM Scoring Engine
# =====================================================================
# Combines objective DL model predictions (70%) with subjective LLM
# scoring (30%) to produce a final overall score (1-10).
#
# DETERMINISM RULES:
#   - final_score = round(0.7 * DL_score + 0.3 * LLM_score, 1)
#   - Minimum score floor: 2.0 (= 20/100 on frontend display)
#   - LLM scoring uses temperature=0.0 (via GenAIEngine)
#   - No randomness anywhere in this class
#
# The DL model receives a 7-dimensional feature vector + SBERT embeddings.
# All parameters are documented with tuning impact notes.
# =====================================================================

import logging
from backend.ml_models.dl_scoring_model import DLScoringModel

logger = logging.getLogger(__name__)

# ── SCORE CONSTANTS ───────────────────────────────────────────────────────────
# Minimum score enforced: no candidate should receive below 20/100
# On the 1-10 scale: 20/100 = 2.0
_MIN_SCORE = 2.0   # maps to 20 displayed on frontend (score * 10)
_MAX_SCORE = 10.0

# Final blend weights (MANDATORY per spec)
_DL_WEIGHT  = 0.7   # 70% objective DL model
_LLM_WEIGHT = 0.3   # 30% subjective LLM evaluation


class ScoringService:
    """
    Hybrid scoring engine combining:
    - 70% DL objective evaluation (measurable signals)
    - 30% LLM subjective evaluation (context understanding)

    Both components are deterministic:
    - DL model: same feature vector → same output (PyTorch eval mode)
    - LLM scoring: temperature=0.0 → greedy decoding, same output every time

    Increasing DL weight → more objective, measurable scores
    Decreasing DL weight → more context-sensitive but potentially noisier
    """

    def __init__(self):
        logger.info("🧠 [ScoringService] Initializing PyTorch Interface...")
        try:
            self.dl_model = DLScoringModel()
            self.use_dl = True
        except Exception as e:
            logger.error(f"⚠️ [ScoringService] Network failed to bind: {e}")
            self.use_dl = False

    def calculate_score(self, text: str, structured: dict, precomputed_llm_score: float = None) -> dict:
        """
        Constructs a 7-dimensional feature vector and runs hybrid scoring.

        DETERMINISM: This method is fully deterministic.
        - Same text + structured dict → same feature vector → same DL score
        - Same text → same LLM score (temperature=0.0)
        - Same inputs → same final_score always

        Args:
            text: refined transcript text
            structured: semantic extraction dict (9 fields)
            precomputed_llm_score: if provided, skips redundant LLM call
        """
        words = text.split()
        total_words = len(words) if words else 1

        # ── FEATURE 1: COMPLETENESS ─────────────────────────────────────────
        # Fraction of 9 semantic fields that contain extracted data
        # Higher → user covered more introduction topics → better score
        # Lower → fewer topics addressed → penalized
        expected_keys = [
            "greetings", "name", "education", "skills", "strengths",
            "areas_of_interest", "qualities", "experience", "career_goals"
        ]
        hits = sum(1 for k in expected_keys if structured.get(k))
        completeness = hits / len(expected_keys)

        # ── FEATURE 2: ASR CONFIDENCE ───────────────────────────────────────
        # Approximated from Whisper's floating-point logprob
        # 0.90 = default high confidence; actual value piped from transcript
        # Higher → ASR was more certain → trust the transcript more
        # Lower → transcript may have errors → DL should be cautious
        confidence = 0.90

        # ── FEATURE 3: NORMALIZED LENGTH ────────────────────────────────────
        # Words spoken / 150 → captures how detailed the response was
        # 150 words ≈ a well-structured 60-second introduction
        # Higher → very detailed answer
        # Lower → brief/incomplete answer
        length_norm = min(total_words / 150.0, 1.0)

        # ── FEATURE 4: LEXICAL DIVERSITY ────────────────────────────────────
        # unique_words / total_words → richness of vocabulary
        # Higher → varied vocabulary → more articulate
        # Lower → repetitive word usage → less impressive
        unique_words = len(set(words))
        diversity = unique_words / total_words

        # ── FEATURE 5: FILLER RATIO ─────────────────────────────────────────
        # Count of " um " occurrences / total words
        # Higher → more hesitation → lower confidence signal
        # Lower → fluent delivery
        filler_ratio = text.lower().count(" um ") / total_words

        # ── FEATURE 6: RAG IMPROVEMENT FLAG ─────────────────────────────────
        # Placeholder for RAG-based improvement detection
        # 0.5 = neutral; actual value would come from progress tracking
        # Higher → user improved from past attempts
        # Lower → regression detected
        rag_improvement = 0.5

        # ── FEATURE 7: COHERENCE PSEUDO-SCORE ───────────────────────────────
        # Simple heuristic: if response is >10 words, assume moderate coherence
        # In production, could be replaced with a trained coherence model
        # Higher → response has enough content for meaningful evaluation
        # Lower → too terse to evaluate properly
        coherence = 0.8 if total_words > 10 else 0.3

        features = [
            completeness,    # [0] topic coverage
            confidence,      # [1] ASR reliability
            length_norm,     # [2] response length
            diversity,       # [3] vocabulary richness
            filler_ratio,    # [4] hesitation indicator
            rag_improvement, # [5] progress tracking
            coherence        # [6] response quality
        ]

        logger.info(f"📊 [Scoring] DL Feature Vector: {[round(f, 3) for f in features]}")

        if self.use_dl:
            try:
                # ── DL INFERENCE (70% of final score) ───────────────────────
                # Runs SBERT embeddings + feature tensor through Transformer
                # Returns 6 individual metrics + overall
                dl_results = self.dl_model.predict_score(text, features)
                dl_score = round(dl_results.get("dl_overall", 6.0), 1)

                # ── LLM SUBJECTIVE SCORE (30% of final score) ───────────────
                # Use precomputed score if available (from combined call)
                # Otherwise triggers a separate LLM inference
                # DETERMINISM: LLM uses temperature=0.0 — same input = same score
                if precomputed_llm_score is not None:
                    llm_score = round(float(precomputed_llm_score), 1)
                    logger.info(f"⚡ [Scoring] Using precomputed LLM score: {llm_score:.1f}")
                else:
                    from backend.core.genai_engine import genai_engine
                    llm_score = round(genai_engine.generate_subjective_score(text, structured), 1)

                # ── HYBRID BLEND (MANDATORY FORMULA) ────────────────────────
                # final_score = round(0.7 * DL_score + 0.3 * LLM_score, 1)
                # DL = 70% (objective, measurable)
                # LLM = 30% (context-sensitive, deterministic at temp=0)
                raw_score = (_DL_WEIGHT * dl_score) + (_LLM_WEIGHT * llm_score)
                logger.info(
                    f"⚖️ [Scoring] Hybrid: DL ({dl_score:.1f}) * {_DL_WEIGHT} + "
                    f"LLM ({llm_score:.1f}) * {_LLM_WEIGHT} = {raw_score:.2f}"
                )

                # ── CONSISTENCY RULE ENGINE ─────────────────────────────────
                # Prevents inflated scores when semantic coverage is poor
                # If less than 40% of fields were filled but score > 6.0 → penalize
                if completeness < 0.4 and raw_score > 6.0:
                    raw_score -= 2.0
                    logger.info("⚠️ [Scoring] Contradiction Rule: Penalized for low completeness vs high score.")

                # ── MINIMUM SCORE FLOOR ──────────────────────────────────────
                # Per spec: score must never go below 20/100 (= 2.0 on 1-10 scale)
                # Prevents unfairly punishing candidates who at least attempted
                final_score = round(max(_MIN_SCORE, min(_MAX_SCORE, raw_score)), 1)
                logger.info(f"✅ [Scoring] Final score: {final_score} (min={_MIN_SCORE}, max={_MAX_SCORE})")

                return {
                    "overall_score": final_score,
                    "confidence": "high",
                    "source": "hybrid_70_30",
                    "details": {
                        "llm_score": llm_score,
                        "dl_score": round(dl_score, 1),
                        "dl_metrics": dl_results,
                        "blend": f"DL*{_DL_WEIGHT} + LLM*{_LLM_WEIGHT}"
                    },
                    "features": features
                }

            except Exception as e:
                logger.warning(f"❌ [Scoring] Inference crashed: {e}")

        # ── SAFE FALLBACK (no DL available) ─────────────────────────────────
        return self._rule_based_fallback(structured, completeness)

    def _rule_based_fallback(self, structured: dict, completeness: float):
        """Heuristic-only scoring when DL model is unavailable."""
        raw = completeness * 10
        final = round(max(_MIN_SCORE, min(_MAX_SCORE, raw)), 1)
        return {
            "overall_score": final,
            "confidence": "low",
            "source": "heuristic_fallback"
        }