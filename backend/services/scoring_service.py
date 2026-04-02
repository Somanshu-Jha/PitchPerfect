# =====================================================================
# SCORING SERVICE — Strictly Calibrated 10-Feature DL Scoring Engine
# =====================================================================
# Uses a PyTorch Multi-Head DL model targeting 6 scores:
# [clarity, confidence, structure, tone, fluency, overall_score]
#
# Feature Vector (10 dimensions):
#   [0] completeness      - section coverage (0-1)
#   [1] asr_confidence     - transcript reliability (0-1)
#   [2] length_norm        - response length / 150 words (0-1)
#   [3] diversity          - unique/total words (0-1)
#   [4] filler_ratio       - filler count / total words (0+)
#   [5] rag_improvement    - historical progress flag (0-1)
#   [6] coherence          - speech stability (0-1)
#   [7] tone_expressiveness - pitch range metric (0-1)
#   [8] fluency_score      - pause+rhythm composite (0-1)
#   [9] pronunciation      - spectral clarity (0-1)
#
# CALIBRATION PHILOSOPHY:
#   - A good interview intro covers name, education, skills, experience, goals
#   - Content is KING: short/irrelevant content = low score, period
#   - Audio features enhance but never compensate for missing content
#   - Scores reflect ACTUAL quality, not participation trophies
# =====================================================================

import re
import logging
from backend.ml_models.dl_scoring_model import DLScoringModel

logger = logging.getLogger(__name__)

_MIN_SCORE = 1.0
_MAX_SCORE = 10.0

# Filler words for ratio computation
_FILLER_WORDS = {"um", "uh", "like", "basically", "actually", "you know",
                 "so", "well", "i mean", "right", "okay", "ok", "matlab"}

# Interview-relevant vocabulary for content relevance scoring
_INTERVIEW_VOCAB = {
    "name_intro": {"my name", "i am", "i'm", "myself", "introduce", "hello", "hi", "good morning", "good afternoon"},
    "education": {"university", "college", "degree", "student", "b.tech", "btech", "engineering", "institute",
                  "school", "studying", "studied", "academic", "bsc", "msc", "mba", "diploma", "graduate",
                  "pursuing", "completed", "bca", "mca", "phd"},
    "skills": {"skill", "python", "java", "javascript", "programming", "coding", "html", "css", "react",
               "sql", "machine learning", "data", "software", "web", "development", "c++", "cloud",
               "communication", "teamwork", "leadership", "problem solving", "proficient", "expertise"},
    "experience": {"experience", "worked", "working", "project", "intern", "internship", "company",
                   "role", "position", "freelance", "developed", "built", "created", "managed", "led"},
    "goals": {"goal", "aspire", "aim", "future", "dream", "want to", "plan to", "hope to", "career",
              "become", "ambition", "objective", "vision", "passionate about", "interested in"},
}


def _compute_content_relevance(text: str) -> tuple:
    """
    Compute how relevant the transcript is to an interview introduction.
    Returns (relevance_score: 0-1, category_hits: int, matching_categories: list)
    """
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) if words else 1

    matching_categories = []
    total_keyword_matches = 0

    for category, keywords in _INTERVIEW_VOCAB.items():
        category_matches = 0
        for keyword in keywords:
            if keyword in text_lower:
                category_matches += 1
        if category_matches > 0:
            matching_categories.append(category)
            total_keyword_matches += category_matches

    category_hits = len(matching_categories)

    # Relevance = combination of category coverage and keyword density
    category_coverage = category_hits / len(_INTERVIEW_VOCAB)  # 0-1
    keyword_density = min(1.0, total_keyword_matches / max(10, total_words * 0.3))  # 0-1

    relevance_score = category_coverage * 0.7 + keyword_density * 0.3

    return relevance_score, category_hits, matching_categories


class ScoringService:
    """
    Strictly calibrated 10-feature, 6-head DL scoring engine.
    Evaluates clarity, confidence, structure, tone, fluency, and overall.
    
    CALIBRATION RULES:
    - Content quality is weighted highest
    - Short/irrelevant responses get appropriately low scores
    - Audio features enhance but cannot compensate for missing content
    """

    def __init__(self):
        logger.info("🧠 [ScoringService] Initializing Calibrated DL Interface...")
        try:
            self.dl_model = DLScoringModel()
            self.use_dl = True
        except Exception as e:
            logger.error(f"⚠️ [ScoringService] Model failed to load: {e}")
            self.use_dl = False

    def calculate_score(self, text: str, structured: dict, precomputed_llm_score: float = None,
                        transcript_confidence: float = 0.90, audio_features: dict = None) -> dict:
        """Constructs 10-dim feature vector and runs 6-head DL scoring with strict calibration."""
        words = text.split()
        total_words = len(words) if words else 1

        # ── FEATURE 0: COMPLETENESS ──
        expected_keys = [
            "greetings", "name", "education", "skills", "strengths",
            "areas_of_interest", "qualities", "experience", "career_goals"
        ]
        hits = sum(1 for k in expected_keys if structured.get(k))
        completeness = hits / len(expected_keys)

        # ── FEATURE 1: ASR CONFIDENCE ──
        confidence = max(0.0, min(1.0, transcript_confidence))

        # ── FEATURE 2: NORMALIZED LENGTH ──
        length_norm = min(total_words / 150.0, 1.0)

        # ── FEATURE 3: LEXICAL DIVERSITY ──
        unique_words = len(set(w.lower() for w in words))
        diversity = unique_words / total_words

        # ── FEATURE 4: FILLER RATIO ──
        text_lower = text.lower()
        filler_count = 0
        for filler in _FILLER_WORDS:
            filler_count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
        filler_ratio = filler_count / total_words

        # ── FEATURE 5: RAG IMPROVEMENT ──
        rag_improvement = 0.5

        # ── FEATURE 6: COHERENCE (from audio) ──
        if audio_features and "speech_rate_stability" in audio_features:
            coherence = max(0.0, min(1.0, audio_features["speech_rate_stability"]))
        else:
            coherence = 0.3 if total_words > 10 else 0.15

        # ── FEATURE 7: TONE EXPRESSIVENESS (from audio) ──
        if audio_features and "tone_expressiveness" in audio_features:
            tone_expressiveness = max(0.0, min(1.0, audio_features["tone_expressiveness"]))
        else:
            tone_expressiveness = 0.3

        # ── FEATURE 8: FLUENCY SCORE (from audio) ──
        if audio_features and "fluency_score" in audio_features:
            fluency_score = max(0.0, min(1.0, audio_features["fluency_score"]))
        else:
            fluency_score = 0.3

        # ── FEATURE 9: PRONUNCIATION CLARITY (from audio) ──
        if audio_features and "pronunciation_score" in audio_features:
            pronunciation = max(0.0, min(1.0, audio_features["pronunciation_score"]))
        else:
            pronunciation = 0.3

        features = [
            completeness,         # [0]
            confidence,           # [1]
            length_norm,          # [2]
            diversity,            # [3]
            filler_ratio,         # [4]
            rag_improvement,      # [5]
            coherence,            # [6]
            tone_expressiveness,  # [7]
            fluency_score,        # [8]
            pronunciation,        # [9]
        ]

        logger.info(f"📊 [Scoring] 10-dim Feature Vector: {[round(f, 3) for f in features]}")

        # ── CONTENT RELEVANCE ANALYSIS ──
        relevance_score, category_hits, matching_categories = _compute_content_relevance(text)
        logger.info(f"📝 [Scoring] Content Relevance: {relevance_score:.2f} | "
                    f"Categories: {category_hits}/5 {matching_categories}")

        if self.use_dl:
            try:
                dl_results = self.dl_model.predict_score(text, features)
                raw_score = dl_results.get("dl_overall", 5.0)

                # ══════════════════════════════════════════════════════════════
                # STRICT CALIBRATION PENALTIES
                # These penalties ensure scores reflect ACTUAL quality
                # ══════════════════════════════════════════════════════════════
                penalty_notes = []
                score_cap = 10.0  # Maximum possible score after penalties

                # ── PENALTY 1: LENGTH-BASED (with content-quality adjustment) ──
                # A proper intro should be 60-180 words (60-90 seconds)
                # BUT if content coverage is excellent, a concise intro
                # shouldn't be penalized as harshly
                content_bonus = 0.0
                if category_hits >= 4:
                    content_bonus = 1.5  # Covers 4-5 categories = concise but thorough
                elif category_hits >= 3:
                    content_bonus = 0.8  # Covers 3 categories = decent coverage

                if total_words < 10:
                    score_cap = min(score_cap, 2.5 + content_bonus * 0.3)
                    penalty_notes.append(f"Extremely short ({total_words} words) → cap {score_cap:.1f}")
                elif total_words < 20:
                    score_cap = min(score_cap, 3.5 + content_bonus * 0.5)
                    penalty_notes.append(f"Very short ({total_words} words) → cap {score_cap:.1f}")
                elif total_words < 35:
                    score_cap = min(score_cap, 4.5 + content_bonus * 0.7)
                    penalty_notes.append(f"Short response ({total_words} words) → cap {score_cap:.1f}")
                elif total_words < 50:
                    score_cap = min(score_cap, 5.5 + content_bonus)
                    penalty_notes.append(f"Below average length ({total_words} words) → cap {score_cap:.1f}")
                elif total_words < 80:
                    score_cap = min(score_cap, 7.0 + content_bonus)
                    penalty_notes.append(f"Adequate length ({total_words} words) → cap {score_cap:.1f}")

                # ── PENALTY 2: CONTENT COMPLETENESS ──
                # Must cover at least some interview-relevant topics
                if completeness < 0.1 and category_hits < 2:
                    score_cap = min(score_cap, 3.0)
                    penalty_notes.append(f"Near-zero content coverage → cap 3.0")
                elif completeness < 0.2 and category_hits < 2:
                    score_cap = min(score_cap, 4.0)
                    penalty_notes.append(f"Very low content coverage → cap 4.0")
                elif completeness < 0.3 and category_hits < 3:
                    score_cap = min(score_cap, 5.5)
                    penalty_notes.append(f"Low content coverage → cap 5.5")

                # ── PENALTY 3: CONTENT RELEVANCE ──
                # If transcript doesn't contain interview-relevant content
                if relevance_score < 0.10:
                    score_cap = min(score_cap, 2.5)
                    penalty_notes.append(f"Content not relevant to interview → cap 2.5")
                elif relevance_score < 0.20:
                    score_cap = min(score_cap, 4.0)
                    penalty_notes.append(f"Low interview relevance ({relevance_score:.2f}) → cap 4.0")
                elif relevance_score < 0.35:
                    score_cap = min(score_cap, 5.5)
                    penalty_notes.append(f"Limited interview relevance ({relevance_score:.2f}) → cap 5.5")

                # ── PENALTY 4: SPEAKING PACE ──
                wpm = 0
                if audio_features:
                    wpm = audio_features.get("wpm_estimate", 0)
                    if wpm > 0 and wpm < 80:
                        raw_score -= 0.5
                        penalty_notes.append(f"Very slow pace ({int(wpm)} WPM) → -0.5")
                    elif wpm > 200:
                        raw_score -= 0.5
                        penalty_notes.append(f"Very fast pace ({int(wpm)} WPM) → -0.5")

                # ── PENALTY 5: HIGH FILLER RATIO ──
                if filler_ratio > 0.3:
                    raw_score -= 1.0
                    penalty_notes.append(f"Excessive fillers ({filler_ratio:.2f}) → -1.0")
                elif filler_ratio > 0.15:
                    raw_score -= 0.5
                    penalty_notes.append(f"High fillers ({filler_ratio:.2f}) → -0.5")

                # ── APPLY SCORE CAP ──
                raw_score = min(raw_score, score_cap)

                # ── ALSO CAP SUB-SCORES ──
                # Sub-scores should never be dramatically higher than overall
                sub_score_cap = min(10.0, score_cap + 1.0)  # Allow sub-scores to be slightly above cap
                for dim_key in ["clarity", "confidence", "structure", "tone", "fluency"]:
                    if dl_results.get(dim_key, 5.0) > sub_score_cap:
                        original = dl_results[dim_key]
                        dl_results[dim_key] = round(sub_score_cap, 1)
                        penalty_notes.append(f"{dim_key} capped {original} → {dl_results[dim_key]}")

                final_score = round(max(_MIN_SCORE, min(_MAX_SCORE, raw_score)), 1)
                final_pct = round(final_score * 10)  # 1-10 → 10-100%

                if penalty_notes:
                    logger.info(f"⚠️ [Scoring] Calibration penalties applied: {penalty_notes}")
                logger.info(f"✅ [Scoring] 6-head DL Final: {final_score} | Details: {dl_results}")

                # ══════════════════════════════════════════════════════════════
                # SCORE BREAKDOWN — shows exact contribution of each dimension
                # ══════════════════════════════════════════════════════════════
                _WEIGHTS = {
                    "clarity":    0.20,
                    "confidence": 0.20,
                    "structure":  0.25,
                    "tone":       0.15,
                    "fluency":    0.20,
                }
                breakdown = []
                weighted_sum = 0
                for dim, weight in _WEIGHTS.items():
                    dim_score = dl_results.get(dim, 5.0)
                    dim_pct = round(dim_score * 10)
                    contribution = round(dim_score * weight * 10, 1)
                    max_possible = round(10 * weight * 10, 1)
                    weighted_sum += dim_score * weight
                    breakdown.append({
                        "dimension": dim.capitalize(),
                        "raw_score": dim_score,
                        "raw_pct": dim_pct,
                        "weight": round(weight * 100),
                        "contribution": contribution,
                        "max_possible": max_possible,
                    })

                computed_overall = round(max(1.0, min(10.0, weighted_sum)), 1)

                return {
                    "overall_score": final_score,
                    "overall_pct": final_pct,
                    "confidence": "high",
                    "source": "dl_6head_v2",
                    "details": {
                        "dl_metrics": dl_results,
                    },
                    "score_breakdown": breakdown,
                    "penalty_notes": penalty_notes,
                    "content_relevance": {
                        "score": round(relevance_score, 3),
                        "category_hits": category_hits,
                        "categories": matching_categories,
                    },
                    "features": features
                }

            except Exception as e:
                logger.warning(f"❌ [Scoring] Inference crashed: {e}")

        return self._rule_based_fallback(text, structured, completeness, features)

    def _rule_based_fallback(self, text: str, structured: dict, completeness: float, features: list = None):
        """Rule-based fallback with strict calibration."""
        words = text.split()
        total_words = len(words) if words else 1
        
        # Content relevance check
        relevance_score, category_hits, _ = _compute_content_relevance(text)
        
        # Base score from completeness (max 5.0 from content)
        raw = completeness * 5.0
        
        # Length contribution (max 2.0)
        length_factor = min(1.0, total_words / 100.0)
        raw += length_factor * 2.0
        
        # Relevance contribution (max 2.0)
        raw += relevance_score * 2.0
        
        # Audio features (max 1.0)
        if features:
            tone = features[7] if len(features) > 7 else 0.3
            fluency = features[8] if len(features) > 8 else 0.3
            raw += (tone + fluency) * 0.5
        
        final = round(max(_MIN_SCORE, min(_MAX_SCORE, raw)), 1)
        return {
            "overall_score": final,
            "confidence": "low",
            "source": "heuristic_fallback",
            "features": features or []
        }