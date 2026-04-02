import os
import time
import uuid
import hashlib
import random
import torch
import numpy as np
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ── Hardware Thread Enforcement — Use ALL CPU Cores ──────────────────────────
_NUM_CORES = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(_NUM_CORES)
os.environ["MKL_NUM_THREADS"] = str(_NUM_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(_NUM_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(_NUM_CORES)

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

logger = logging.getLogger(__name__)

# ── Confidence Gate Constants ─────────────────────────────────────────────────
_SKIP_LLM_WORD_THRESHOLD = 12
_SKIP_LLM_CONFIDENCE_THRESHOLD = 0.85
_SKIP_LLM_SCORE_CAP = 7.5

# ── Top-level subprocess functions (must be picklable) ────────────────────────
def _run_audio_analysis_subprocess(audio_path: str) -> dict:
    """CPU process: deep audio feature extraction (bypasses GIL)."""
    try:
        from backend.services.audio_analysis_service import AudioAnalysisService
        svc = AudioAnalysisService()
        return svc.extract(audio_path)
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).error(f"❌ [AudioAnalysis] Failed: {e}")
        return {
            "speech_rate": 1.5, "wpm_estimate": 140, "pace_label": "ideal",
            "pause_ratio": 0.2, "long_pauses": 0, "avg_pause_duration": 0,
            "pitch": 150.0, "pitch_range": 50, "pitch_variance": 0.3,
            "energy_consistency": 0.6, "energy_trajectory": "unknown",
            "speech_rate_stability": 0.5,
            "tone_expressiveness": 0.5, "tone_label": "moderate", "tone_richness": 0.5,
            "fluency_score": 0.5, "pronunciation_score": 0.5,
            "spectral_flatness": 0.1, "hnr_score": 0.5,
            "dynamic_confidence": 50.0, "confidence_label": "MEDIUM",
            "reasoning": {
                "tone": "Analysis unavailable.", "fluency": "Analysis unavailable.",
                "pronunciation": "Analysis unavailable.", "pace": "Analysis unavailable.",
                "energy": "Analysis unavailable."
            }
        }

def _run_preprocessing_subprocess(audio_path: str, output_path: str) -> tuple:
    """CPU process: audio preprocessing (ffmpeg conversion)."""
    try:
        from backend.services.audio_preprocessing_service import AudioPreprocessingService
        svc = AudioPreprocessingService()
        return svc.process(audio_path, output_path)
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).error(f"❌ [Preprocessing] Failed: {e}")
        return output_path, {"clipping": False, "distorted": False, "low_energy": False}


class SpeechPipeline:
    """
    High-performance parallel pipeline with full CPU/GPU utilization:
    
    Stage 1 (CPU):       Preprocessing (ffmpeg) — unique temp files
    Stage 2 (Parallel):  ASR (GPU) ‖ AudioAnalysis (CPU, all cores)
    Stage 3 (Parallel):  Semantic + Feedback (GPU/CPU) ‖ Filler Detection (CPU)
    Stage 4 (CPU):       Scoring (DL inference)
    Stage 5 (CPU):       DB tracking + caching
    
    DETERMINISM: Uses audio hash as seed for all random operations.
    Same audio file → same seed → same results every time.
    """
    
    def __init__(self):
        logger.info(f"\n========== [Pipeline] Booting Engine ({_NUM_CORES} CPU cores) ==========")
        self.preprocessor = AudioPreprocessingService()
        self.transcriber = TranscriptionService()
        self.corrector = CorrectionService()
        self.semantic = SemanticService()
        self.audio_analyzer = AudioAnalysisService()
        self.scorer = ScoringService()
        self.filler_service = FillerDetectionService()
        self.feedback_service = FeedbackService()
        self.english_level_model = EnglishLevelClassifier()

        # Persistent thread/process pools
        self._cpu_pool = ProcessPoolExecutor(max_workers=_NUM_CORES)
        self._gpu_pool = ThreadPoolExecutor(max_workers=1)
        logger.info(f"✅ Pipeline ready. CPU Pool: {_NUM_CORES} workers, GPU Pool: 1 thread\n")

    def _seed_from_audio(self, audio_bytes: bytes) -> int:
        """
        Generate a deterministic seed from audio bytes.
        Same audio file → same seed → same random choices → same results.
        """
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        return int(audio_hash[:8], 16)  # First 8 hex chars → int seed

    async def process(self, audio_path: str, user_id: str = "local_demo", audio_bytes: bytes = None):
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 PIPELINE START (User: {user_id}) — {_NUM_CORES} CPU cores + GPU")
        logger.info(f"{'='*60}")
        t_pipeline_start = time.perf_counter()
        timings = {}

        score_packet: dict = {"overall_score": 6.0, "features": []}
        
        # Track temp files for cleanup
        temp_files = []

        # ── DETERMINISTIC SEEDING ──
        # Same audio bytes → same seed → same random operations → consistent results
        if audio_bytes is not None:
            seed = self._seed_from_audio(audio_bytes)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            logger.info(f"🎲 [Pipeline] Deterministic seed set: {seed}")

        loop = asyncio.get_running_loop()

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 1: PREPROCESSING (CPU subprocess)
        # ══════════════════════════════════════════════════════════════════════
        t_step = time.perf_counter()
        # Use unique output path to prevent race conditions
        unique_id = uuid.uuid4().hex[:12]
        preprocess_output = f"clean_{unique_id}.wav"
        
        clean_audio, audio_flags = await loop.run_in_executor(
            self._cpu_pool, _run_preprocessing_subprocess, audio_path, preprocess_output
        )
        temp_files.append(clean_audio)
        timings["preprocessing"] = time.perf_counter() - t_step
        logger.info(f"✅ [STAGE 1] Preprocessing: {timings['preprocessing']:.2f}s | Flags: {audio_flags}")

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 2: TRUE PARALLEL — ASR (GPU) ‖ AudioAnalysis (CPU, all cores)
        # ══════════════════════════════════════════════════════════════════════
        t_step = time.perf_counter()

        # GPU: ASR transcription (ThreadPool — CUDA models not picklable)
        asr_task = loop.run_in_executor(self._gpu_pool, self.transcriber.transcribe, clean_audio)
        # CPU: Deep audio feature extraction (ProcessPool — true parallelism)
        cpu_task = loop.run_in_executor(self._cpu_pool, _run_audio_analysis_subprocess, clean_audio)

        # Await BOTH in parallel — GPU and CPU working simultaneously
        asr_result, audio_features = await asyncio.gather(asr_task, cpu_task)
        timings["parallel_asr_audio"] = time.perf_counter() - t_step
        logger.info(f"✅ [STAGE 2] Parallel ASR(GPU)+AudioAnalysis(CPU): {timings['parallel_asr_audio']:.2f}s")

        (raw_text, transcript_confidence) = asr_result

        # ── ASR RETRY if empty ───────────────────────────────────────────────
        if not raw_text or not raw_text.strip() or raw_text.strip() == "Audio processing failed or transcript empty.":
            logger.warning("⚠️ [ASR RETRY] Empty transcript — retrying...")
            try:
                t_retry = time.perf_counter()
                asr_result = await loop.run_in_executor(self._gpu_pool, self.transcriber.transcribe, clean_audio)
                (raw_text, transcript_confidence) = asr_result
                logger.info(f"✅ [ASR RETRY] Recovered {len(raw_text.split())} words in {time.perf_counter()-t_retry:.2f}s")
            except Exception as e:
                logger.error(f"❌ [ASR RETRY] Failed: {e}")
                raw_text = "Audio transcription failed after retry."
                transcript_confidence = 0.0

        dynamic_confidence = audio_features.get("dynamic_confidence", transcript_confidence * 100)
        confidence_label = audio_features.get("confidence_label", "MEDIUM")
        logger.info(f"📝 Transcript ({dynamic_confidence:.1f}% conf [{confidence_label}]): {raw_text[:100]}...")

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 3: CORRECTION + SEMANTIC + FEEDBACK (Parallel with Fillers)
        # ══════════════════════════════════════════════════════════════════════
        t_step = time.perf_counter()
        refined_text = self.corrector.refine(raw_text)
        word_count = len(refined_text.split())
        timings["correction"] = time.perf_counter() - t_step

        # Short transcript still gets processed — no early return
        if word_count < 10:
            logger.info(f"⚠️ [ShortInput] {word_count} words — processing with reduced expectations.")

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 4: LLM PIPELINE (Semantic) + RULE-BASED FEEDBACK
        # ══════════════════════════════════════════════════════════════════════
        t_llm = time.perf_counter()
        llm_used = True

        english_lvl = self.english_level_model.classify(refined_text, audio_features)
        logger.info(f"🧠 [Level] English Level: {english_lvl}")

        try:
            semantic_result = self.semantic.analyze(refined_text, raw_text=raw_text)
        except Exception as e:
            logger.error(f"❌ Semantic failure: {e}")
            semantic_result = {
                "intent": {"detected": [], "confidence": 0.0},
                "structured": {}, "confidence_map": {}, "evidence_map": {}
            }

        llm_score = None

        # ── Filler detection (lightweight, CPU) ──
        try:
            fillers = self.filler_service.detect(refined_text)
        except Exception:
            fillers = []

        # ── Feedback Generation (Rule-based with audio reasoning) ──
        try:
            feedback = self.feedback_service.generate(
                user_id=user_id, transcript=refined_text,
                semantic=semantic_result, scores={"overall_score": 6.0},
                fillers=fillers, english_level=english_lvl,
                audio_features=audio_features
            )
        except Exception as e:
            logger.error(f"❌ Feedback failure: {e}")
            feedback = {
                "positives": [],
                "improvements": [], 
                "coaching_summary": "Feedback generation temporarily unavailable. Please try again.",
                "audio_reasoning": {}
            }

        completeness_issues = []
        timings["llm_semantic_feedback"] = time.perf_counter() - t_llm
        logger.info(f"✅ [STAGE 4] LLM+Semantic+Feedback: {timings['llm_semantic_feedback']:.2f}s")

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 5: HYBRID SCORING (DL 6-head with audio features)
        # ══════════════════════════════════════════════════════════════════════
        t_score = time.perf_counter()
        try:
            score_packet = self.scorer.calculate_score(
                refined_text,
                semantic_result.get("structured", {}),
                llm_score,
                transcript_confidence,
                audio_features
            )
            scores = {
                "overall_score": score_packet["overall_score"],
                "source": score_packet.get("source", "dl_6head_v2"),
                "details": score_packet.get("details", {}),
            }
        except Exception as e:
            logger.warning(f"⚠️ Scoring fallback: {e}")
            score_packet = {"overall_score": 6.0, "features": [0.5] * 10}
            scores = {"overall_score": 6.0, "note": "Fallback scoring"}

        if "features" not in score_packet:
            score_packet["features"] = []

        scores["confidence"] = confidence_label.lower()
        timings["scoring"] = time.perf_counter() - t_score

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 6: DATABASE TRACKING (async-safe)
        # ══════════════════════════════════════════════════════════════════════
        t_db = time.perf_counter()
        try:
            user_name = semantic_result.get("structured", {}).get("name", "Unknown")
            db.upsert_user(user_id, user_name)

            db.store_attempt(
                user_id=user_id, transcript=refined_text,
                semantic=semantic_result.get("structured", {}),
                num_fillers=len(fillers) if fillers else 0,
                score=scores["overall_score"],
                feedback=feedback,
                confidence=dynamic_confidence
            )

            historical_progress = db.get_user_progress(user_id)

            dl_features = score_packet.get("features", []) if score_packet.get("features") else []
            if dl_features:
                word_count_check = len(refined_text.split())
                noise_ratio = len(fillers) / max(1, word_count_check) if fillers else 0.0
                if word_count_check >= 10 and dynamic_confidence >= 60.0 and noise_ratio <= 0.3:
                    logger.info("📈 [ML] High-quality sample → storing for retraining")
                    rlhf_filter.validate_and_ingest(
                        transcript=refined_text, score=scores["overall_score"], DL_features=dl_features
                    )
                else:
                    logger.info("🚫 [ML] Rejected from retraining due to safety limits.")

        except Exception as e:
            logger.error(f"❌ [DB] Tracking error: {e}")
            historical_progress = {"error": "Tracking failed"}

        timings["db_tracking"] = time.perf_counter() - t_db

        # ── Light VRAM cleanup ───────────────────────
        model_manager.clear()

        # ── Cleanup temp files ───────────────────────
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"🧹 Cleaned up temp file: {temp_file}")
            except OSError:
                pass

        # ══════════════════════════════════════════════════════════════════════
        # PERFORMANCE SUMMARY
        # ══════════════════════════════════════════════════════════════════════
        total_time = time.perf_counter() - t_pipeline_start
        timings["total"] = total_time

        logger.info(f"\n{'─'*50}")
        logger.info(f"⏱️  PERFORMANCE REPORT ({_NUM_CORES} CPU + GPU)")
        logger.info(f"{'─'*50}")
        for stage, t in timings.items():
            logger.info(f"   {stage:.<35} {t:.2f}s")
        logger.info(f"{'─'*50}")
        logger.info(f"   {'TOTAL':.<35} {total_time:.2f}s")
        logger.info(f"{'─'*50}")
        logger.info(f"   Score: {scores.get('overall_score')} | Source: {scores.get('source')} | LLM: {llm_used}")
        logger.info(f"{'='*60}\n")

        result = {
            "user_id": user_id,
            "raw_transcript": raw_text,
            "refined_transcript": refined_text,
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
                "dynamic_confidence": dynamic_confidence,
                "confidence_label": confidence_label,
                "llm_used": llm_used
            },
            "english_level": english_lvl,
            "timings": timings
        }

        return result
