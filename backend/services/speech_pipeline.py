import os
import torch
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ── Hardware Thread Enforcement ───────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

from backend.services.transcription_service import TranscriptionService
from backend.services.correction_service import CorrectionService
from backend.services.semantic_service import SemanticService
from backend.services.audio_analysis_service import AudioAnalysisService
from backend.services.scoring_service import ScoringService
from backend.services.audio_preprocessing_service import AudioPreprocessingService
from backend.services.filler_detection_service import FillerDetectionService
from backend.services.feedback_service import FeedbackService
from backend.ml_models.english_level_model import EnglishLevelClassifier
from backend.core.model_manager import model_manager
from backend.core.database import db
from backend.core.rlhf_filter import rlhf_filter
from backend.core.genai_engine import genai_engine
from backend.core.result_cache import result_cache

logger = logging.getLogger(__name__)

# ── Confidence Gate Constants ─────────────────────────────────────────────────
_SKIP_LLM_WORD_THRESHOLD = 12          # Skip LLM if < 12 words
_SKIP_LLM_CONFIDENCE_THRESHOLD = 0.85 # Skip LLM if ASR confidence > 0.85
_SKIP_LLM_SCORE_CAP = 7.5             # Max score on gated path

# ── Top-level module function for ProcessPoolExecutor (must be picklable) ─────
def _run_audio_analysis_subprocess(audio_path: str) -> dict:
    """
    Standalone function (module-level) used by ProcessPoolExecutor.
    Cannot be a class method — multiprocessing requires picklable callables.
    Runs AudioAnalysisService.extract() in a dedicated CPU process,
    completely bypassing the GIL for true CPU parallelism.
    """
    try:
        # Import inside subprocess to avoid pickling service objects
        from backend.services.audio_analysis_service import AudioAnalysisService
        svc = AudioAnalysisService()
        return svc.extract(audio_path)
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).error(f"❌ [AudioAnalysis Subprocess] Failed: {e}")
        return {
            "speech_rate": 1.5, "pause_ratio": 0.2, "pitch": 150.0,
            "pitch_variance": 0.3, "energy_consistency": 0.6,
            "pronunciation_score": 0.5, "speech_rate_stability": 0.5,
            "dynamic_confidence": 50.0, "confidence_label": "MEDIUM"
        }


class SpeechPipeline:
    def __init__(self):
        logger.info("\n========== [Pipeline] Booting Engine ==========")
        self.preprocessor = AudioPreprocessingService()
        self.transcriber = TranscriptionService()
        self.corrector = CorrectionService()
        self.semantic = SemanticService()
        self.audio_analyzer = AudioAnalysisService()
        self.scorer = ScoringService()
        self.filler_service = FillerDetectionService()
        self.feedback_service = FeedbackService()
        self.english_level_model = EnglishLevelClassifier()
        # CPU process pool — persisted across calls to avoid spawn overhead
        # max_workers = cpu_count - 1 to leave one core for the main process
        _cpu_workers = max(1, (os.cpu_count() or 4) - 1)
        self._cpu_pool = ProcessPoolExecutor(max_workers=_cpu_workers)
        logger.info(f"✅ Pipeline Initialized. CPU Pool: {_cpu_workers} workers\n")

    async def process(self, audio_path: str, user_id: str = "local_demo", audio_bytes: bytes = None):  # noqa: C901
        logger.info(f"\n========== PIPELINE START (User: {user_id}) ==========")

        # Safe sentinel — overwritten by whichever scoring path runs
        score_packet: dict = {"overall_score": 6.0, "features": []}

        # ── CACHE CHECK (ABSOLUTE FIRST STEP) ────────────────────────────────
        # SHA256(audio_bytes + user_id) → same audio + same user = identical key
        if audio_bytes is not None:
            cached = result_cache.get(audio_bytes, user_id)
            if cached is not None:
                logger.info("⚡ [Pipeline] Returning cached result — skipping full pipeline.")
                return cached

        # ── STEP 1: PREPROCESS ───────────────────────────────────────────────
        clean_audio, audio_flags = self.preprocessor.process(audio_path, "clean.wav")
        logger.info(f"[AudioFlags]: {audio_flags}")

        # ── STEP 2: TRUE PARALLEL EXECUTION ──────────────────────────────────
        #
        #  GPU path  → ASR (Whisper) via ThreadPoolExecutor (GPU model not picklable)
        #  CPU path  → AudioAnalysis via ProcessPoolExecutor (bypasses GIL)
        #
        #  Both are dispatched as futures, then awaited together with asyncio.gather
        #  so they run concurrently without blocking each other.
        #
        loop = asyncio.get_running_loop()

        # Dispatch GPU task: ASR → single thread (needs shared VRAM)
        _gpu_pool = ThreadPoolExecutor(max_workers=1)
        asr_task = loop.run_in_executor(_gpu_pool, self.transcriber.transcribe, clean_audio)

        # Dispatch CPU task: audio feature extraction → process pool (true parallelism)
        cpu_task = loop.run_in_executor(
            self._cpu_pool, _run_audio_analysis_subprocess, clean_audio
        )

        # Await both in parallel — neither blocks the other
        asr_result, audio_features = await asyncio.gather(asr_task, cpu_task)
        _gpu_pool.shutdown(wait=False)

        (raw_text, transcript_confidence) = asr_result
        dynamic_confidence = audio_features.get("dynamic_confidence", transcript_confidence * 100)
        confidence_label = audio_features.get("confidence_label", "MEDIUM")  # LOW/MEDIUM/HIGH from audio analysis
        logger.info(f"[RAW TRANSCRIPT] ({dynamic_confidence:.1f}% Dynamic Conf [{confidence_label}]): {raw_text}")

        # ── STEP 3: CORRECTION ───────────────────────────────────────────────
        refined_text = self.corrector.refine(raw_text)

        # ── CONFIDENCE GATING ─────────────────────────────────────────────────
        # For very short / very confident transcripts, skip LLM entirely
        word_count = len(refined_text.split())
        skip_llm = (
            word_count < _SKIP_LLM_WORD_THRESHOLD
            and transcript_confidence > _SKIP_LLM_CONFIDENCE_THRESHOLD
        )

        if skip_llm:
            logger.info(
                f"🚀 [ConfidenceGate] Skipping LLM — {word_count} words, "
                f"confidence={transcript_confidence:.2f}. Fast-path activated."
            )
            semantic_result = {
                "intent": {"detected": [], "confidence": transcript_confidence},
                "structured": {},
                "confidence_map": {},
                "evidence_map": {}
            }
            scores = {
                "overall_score": max(2.0, min(_SKIP_LLM_SCORE_CAP, word_count * 0.3)),
                "note": "Confidence-gated fast path — LLM skipped",
                "source": "heuristic_fast"
            }
            feedback = {
                "positives": ["Your response was received."],
                "improvements": ["Please provide a more detailed answer for full AI evaluation."],
                "coaching_summary": "Your answer was too short for full AI analysis. Please elaborate on your background."
            }
            english_lvl = "Beginner"
            fillers = []
            completeness_issues = []
            llm_used = False

        else:
            # ── STEP 4: COMBINED LLM CALL (semantic + scores + feedback) ─────────
            llm_used = True
            combined_result = None
            historical_context = []
            
            english_lvl = self.english_level_model.classify(refined_text, audio_features)
            logger.info(f"🧠 [EnglishLevelClassifier] Determined User Level: {english_lvl}")

            try:
                # Pull RAG context for combined call injection
                rag_records = self.feedback_service.rag_service.retrieve_context(user_id, refined_text, top_k=2)
                if rag_records:
                    historical_context = [r.get("improvements", []) for r in rag_records if r.get("improvements")]

                combined_result = genai_engine.comprehensive_analyze(refined_text, historical_context)
            except Exception as e:
                logger.warning(f"⚠️ Combined LLM call failed: {e}. Falling back to multi-call.")

            if combined_result is not None:
                # ── COMBINED PATH: unpack semantic + score from single JSON ──────
                logger.info("✅ [Pipeline] Using combined single-LLM output.")
                raw_semantic = combined_result.get("semantic", {})
                llm_score = round(float(combined_result.get("scores", {}).get("llm_score", 6.0)), 1)

                # Normalize semantic through SemanticService's normalization logic
                try:
                    semantic_result = self.semantic.analyze(refined_text, raw_text=raw_text,
                                                            precomputed_genai=raw_semantic)
                except Exception as e:
                    logger.warning(f"⚠️ Semantic normalization failed: {e}")
                    semantic_result = {
                        "intent": {"detected": [], "confidence": 0.0},
                        "structured": raw_semantic,
                        "confidence_map": {},
                        "evidence_map": {}
                    }

                # ── TWO-PASS FEEDBACK WITH RAG CONTEXT ───────────────────────
                # Instead of using the raw combined feedback directly, run the
                # proper two-pass system which injects RAG historical context
                # into the structure extraction (not just filtering).
                try:
                    feedback = self.feedback_service.generate(
                        user_id=user_id,
                        transcript=refined_text,
                        semantic=semantic_result,
                        scores={"overall_score": llm_score},
                        fillers=[],
                        english_level=english_lvl
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Two-pass feedback failed, using combined fallback: {e}")
                    raw_feedback = combined_result.get("feedback", {})
                    raw_positives = raw_feedback.get("positives", [])
                    raw_improvements = raw_feedback.get("improvements", [])
                    if len(raw_positives) < 4:
                        raw_positives += ["Good effort overall."] * (4 - len(raw_positives))
                    if len(raw_improvements) < 4:
                        raw_improvements += ["Continue practicing to deepen your responses."] * (4 - len(raw_improvements))
                    feedback = {
                        "positives": raw_positives[:8],
                        "improvements": raw_improvements[:8],
                        "coaching_summary": raw_feedback.get("coaching_summary", "Good effort.")
                    }

            else:
                # ── FALLBACK: original multi-call pipeline ────────────────────────
                logger.info("⚠️ [Pipeline] Falling back to multi-call pipeline.")
                try:
                    semantic_result = self.semantic.analyze(refined_text, raw_text=raw_text)
                except Exception as e:
                    logger.error(f"❌ Semantic GenAI failure: {e}")
                    semantic_result = {
                        "intent": {"detected": [], "confidence": 0.0},
                        "structured": {},
                        "confidence_map": {},
                        "evidence_map": {}
                    }
                llm_score = None  # Will be calculated inside scorer

                try:
                    feedback = self.feedback_service.generate(
                        user_id=user_id,
                        transcript=refined_text,
                        semantic=semantic_result,
                        scores={"overall_score": 6.0},
                        fillers=[],
                        english_level=english_lvl
                    )
                except Exception as e:
                    logger.error(f"❌ GenAI Feedback failure: {e}")
                    feedback = {
                        "positives": ["Feedback Generation failed."],
                        "improvements": [],
                        "coaching_summary": "Error"
                    }

            # ── STEP 5 (removed — no rule-based completeness) ────────────────────
            completeness_issues = []

            # ── STEP 6: FILLERS ──────────────────────────────────────────────────
            try:
                fillers = self.filler_service.detect(refined_text)
            except Exception as e:
                logger.error(f"❌ Filler detection failure: {e}")
                fillers = []

            # ── STEP 7: HYBRID SCORING ───────────────────────────────────────────
            try:
                score_packet = self.scorer.calculate_score(
                    refined_text,
                    semantic_result.get("structured", {}),
                    precomputed_llm_score=llm_score  # Pass through if already computed
                )
                scores = {
                    "overall_score": score_packet["overall_score"],
                    "source": score_packet.get("source", "hybrid_70_30"),
                    "details": score_packet.get("details", {})
                }
            except Exception as e:
                logger.warning(f"⚠️ Scoring fallback: {e}")
                score_packet = {"overall_score": 6.0, "features": [0.5] * 7}
                scores = {"overall_score": 6.0, "note": "Fallback scoring used"}

        # Guarantee score_packet always has a 'features' key (may be empty on combined path)
        if "features" not in score_packet:
            score_packet["features"] = []

        # ── STEP 8: SQLITE TRACKING + ML RETRAINING ──────────────────────────
        try:
            user_name = semantic_result.get("structured", {}).get("name", "Unknown Evaluator")
            db.upsert_user(user_id, user_name)

            db.store_attempt(
                user_id=user_id,
                transcript=refined_text,
                semantic=semantic_result.get("structured", {}),
                num_fillers=len(fillers) if fillers else 0,
                score=scores["overall_score"],
                feedback=feedback,
                confidence=dynamic_confidence
            )

            historical_progress = db.get_user_progress(user_id)

            dl_features = score_packet.get("features", []) if (not skip_llm and score_packet.get("features")) else []
            if dl_features:
                if dynamic_confidence >= 75.0 and scores["overall_score"] >= 7.0:
                    logger.info("📈 [ML Ingest] Sample meets quality threshold (conf>=75, score>=7.0). Storing...")
                    rlhf_filter.validate_and_ingest(
                        transcript=refined_text,
                        score=scores["overall_score"],
                        DL_features=dl_features
                    )
                else:
                    logger.info("🚫 [ML Ingest] Sample quality too low for training. Discarded from ML dataset.")

        except Exception as e:
            logger.error(f"❌ [Database/ML Ingestion Error]: {e}")
            historical_progress = {"error": "Tracking failed"}

        # ── VRAM FLUSH ────────────────────────────────────────────────────────
        model_manager.clear()

        logger.info(f"[SEMANTIC]: {semantic_result.get('structured', {})}")
        logger.info(f"[SCORES]: {scores}")
        logger.info(f"[LLM_USED]: {llm_used} | [CONF]: {transcript_confidence:.2f}")
        logger.info("========== PIPELINE END ==========")

        result = {
            "user_id": user_id,
            "raw_transcript": raw_text,         # Full — all segments joined
            "refined_transcript": refined_text,  # Full — no truncation
            "semantic": semantic_result,
            "audio_features": audio_features,
            "audio_flags": audio_flags,
            "fillers": fillers,
            "scores": scores,
            "feedback": feedback,
            "completeness_issues": completeness_issues,
            "historical_progress": historical_progress,
            "confidence": {
                "transcript_confidence": round(transcript_confidence, 3),
                "dynamic_confidence": dynamic_confidence,       # Numeric 0-100
                "confidence_label": confidence_label,           # LOW / MEDIUM / HIGH
                "llm_used": llm_used
            },
            "english_level": english_lvl
        }

        # Cache result for repeated uploads
        if audio_bytes is not None:
            result_cache.set(audio_bytes, user_id, result)

        return result