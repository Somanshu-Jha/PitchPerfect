# =====================================================================
# SPEECH PIPELINE — CPU+GPU Parallel Processing Engine
# =====================================================================
# Architecture: Maximized CPU+GPU parallelism at 90% cap.
#
# Performance Strategy:
#   - CPU capped at 90% cores for audio analysis + preprocessing
#   - GPU dedicated to ASR (Whisper) + LLM inference
#   - CPU and GPU work simultaneously at every possible stage
#   - Single-pass LLM call (2→1 merge) halves GPU time
#
# Flow:
#   Stage 1: Preprocessing [CPU]
#   Stage 2: ASR [GPU] ‖ Audio Analysis + Resume Extract [CPU]  — FULL PARALLEL
#   Stage 3: Correction + Fillers + Level [CPU] — parallel sub-tasks
#   Stage 4: Deep HR Analysis [GPU] — SINGLE unified LLM call
#   Stage 5: Semantic + Scoring + Feedback [CPU] — parallel
#   Stage 6: DB Persist
# =====================================================================

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

# ── Hardware Thread Enforcement — 90% CPU Cap ──────────────────────────
_TOTAL_CORES = os.cpu_count() or 8
_NUM_CORES = max(2, int(_TOTAL_CORES * 0.9))  # 90% of CPU cores
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
from backend.services.document_service import document_service
from backend.ml_models.english_level_model import EnglishLevelClassifier
from backend.core.model_manager import model_manager
from backend.core.database import db
from backend.core.genai_engine import genai_engine

logger = logging.getLogger(__name__)

# ── Top-level subprocess functions (must be picklable) ────────────────────
def _run_audio_analysis_subprocess(audio_path: str) -> dict:
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
            "energy_consistency": 0.6, "energy_trajectory": "stable",
            "speech_rate_stability": 0.5,
            "tone_expressiveness": 0.5, "tone_label": "moderate", "tone_richness": 0.5,
            "fluency_score": 0.5, "pronunciation_score": 0.5,
            "spectral_flatness": 0.1, "hnr_score": 0.5,
            "dynamic_confidence": 50.0, "confidence_label": "MEDIUM",
            "reasoning": {}
        }

def _run_preprocessing_subprocess(audio_path: str, output_path: str) -> tuple:
    try:
        from backend.services.audio_preprocessing_service import AudioPreprocessingService
        svc = AudioPreprocessingService()
        return svc.process(audio_path, output_path)
    except Exception as e:
        return output_path, {"clipping": False, "distorted": False, "low_energy": False}

def _run_resume_extraction(resume_path: str) -> str:
    """Extract resume text in subprocess — CPU bound, runs parallel with GPU."""
    try:
        from backend.services.document_service import document_service
        return document_service.extract_text(resume_path)
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).error(f"❌ [ResumeExtract] Failed: {e}")
        return ""


class SpeechPipeline:
    """
    High-performance parallel pipeline with CPU+GPU at 90% utilization.

    CPU (90% cores): Preprocessing, Audio Analysis, Resume Extract,
                     Filler Detection, Correction, Scoring, Feedback Assembly
    GPU: Whisper ASR, LLM Inference (single unified call)

    Both CPU and GPU work simultaneously at every possible stage.
    """

    def __init__(self):
        logger.info(f"\n========== [Pipeline] Booting Engine ({_NUM_CORES}/{_TOTAL_CORES} CPU cores @ 90%) ==========")
        self.preprocessor = AudioPreprocessingService()
        self.transcriber = TranscriptionService()
        self.corrector = CorrectionService()
        self.semantic = SemanticService()
        self.audio_analyzer = AudioAnalysisService()
        self.scorer = ScoringService()
        self.filler_service = FillerDetectionService()
        self.feedback_service = FeedbackService()
        self.english_level_model = EnglishLevelClassifier()

        # CPU pool at 90% cap — for heavy CPU tasks (audio analysis, preprocessing)
        self._cpu_pool = ProcessPoolExecutor(max_workers=_NUM_CORES)
        # Thread pool for CPU-bound async tasks (correction, fillers, scoring)
        self._cpu_thread_pool = ThreadPoolExecutor(max_workers=_NUM_CORES)
        # GPU pool — single thread for GPU-bound tasks (ASR, LLM)
        self._gpu_pool = ThreadPoolExecutor(max_workers=1)

        if torch.cuda.is_available():
            # Set GPU memory fraction to 90%
            try:
                torch.cuda.set_per_process_memory_fraction(0.90, 0)
                logger.info(f"🖥️  [Pipeline] GPU memory fraction set to 90%")
            except Exception:
                pass

        logger.info(f"✅ Pipeline ready — CPU: {_NUM_CORES} workers, GPU: 1 worker\n")

    def _seed_from_audio(self, audio_bytes: bytes) -> int:
        return int(hashlib.sha256(audio_bytes).hexdigest()[:8], 16)

    async def process(self, audio_path: str, user_id: str = "local_demo",
                      audio_bytes: bytes = None, resume_path: str = None,
                      strictness: str = "intermediate"):
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 PIPELINE START (User: {user_id}, Strictness: {strictness})")
        logger.info(f"{'='*60}")
        t_pipeline = time.perf_counter()
        timings = {}
        temp_files = []
        resume_text = None

        # ── Deterministic Seeding ────────────────────────────────────────
        if audio_bytes is not None:
            seed = self._seed_from_audio(audio_bytes)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            logger.info(f"🎲 Seed: {seed}")

        loop = asyncio.get_running_loop()

        # ══════════════════════════════════════════════════════════════════
        # STAGE 1: PREPROCESSING [CPU] + RESUME EXTRACT [CPU]  — PARALLEL
        # Start resume extraction immediately alongside preprocessing
        # ══════════════════════════════════════════════════════════════════
        t = time.perf_counter()
        uid = uuid.uuid4().hex[:12]
        preprocess_out = f"clean_{uid}.wav"

        # Launch both in parallel on CPU
        preprocess_task = loop.run_in_executor(
            self._cpu_pool, _run_preprocessing_subprocess, audio_path, preprocess_out
        )
        resume_task = None
        if resume_path:
            logger.info(f"📄 [Pipeline] Resume extraction starting in parallel...")
            resume_task = loop.run_in_executor(
                self._cpu_pool, _run_resume_extraction, resume_path
            )

        if resume_task:
            (clean_audio, audio_flags), resume_text = await asyncio.gather(
                preprocess_task, resume_task
            )
            logger.info(f"✅ [Pipeline] Resume: {len(resume_text or '')} chars extracted")
        else:
            clean_audio, audio_flags = await preprocess_task

        temp_files.append(clean_audio)
        timings["s1_preprocess"] = round(time.perf_counter() - t, 2)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2: ASR [GPU] ‖ Audio Analysis [CPU]  — TRUE PARALLEL
        # GPU does Whisper while CPU does spectral/pitch/fluency analysis
        # ══════════════════════════════════════════════════════════════════
        t = time.perf_counter()
        asr_task = loop.run_in_executor(self._gpu_pool, self.transcriber.transcribe, clean_audio)
        cpu_task = loop.run_in_executor(self._cpu_pool, _run_audio_analysis_subprocess, clean_audio)
        (raw_text, transcript_confidence), audio_features = await asyncio.gather(asr_task, cpu_task)
        timings["s2_asr_audio"] = round(time.perf_counter() - t, 2)
        logger.info(f"✅ [S2] ASR+Audio: {timings['s2_asr_audio']}s | {len(raw_text.split())} words")

        # ══════════════════════════════════════════════════════════════════
        # STAGE 3: CORRECTION + FILLERS + LEVEL  — PARALLEL [CPU]
        # All three run simultaneously on CPU thread pool
        # ══════════════════════════════════════════════════════════════════
        t = time.perf_counter()
        refined_text = self.corrector.refine(raw_text)

        def _fillers_stats():
            return self.filler_service.detect_with_stats(refined_text)

        def _level():
            return self.english_level_model.classify(refined_text, audio_features)

        filler_stats, english_lvl = await asyncio.gather(
            loop.run_in_executor(self._cpu_thread_pool, _fillers_stats),
            loop.run_in_executor(self._cpu_thread_pool, _level),
        )

        # Inject filler data into audio_features for downstream services
        audio_features["filler_count"] = filler_stats.get("count", 0)
        audio_features["filler_density"] = filler_stats.get("density", 0.0)
        audio_features["filler_per_type"] = filler_stats.get("per_type", {})
        audio_features["filler_most_frequent"] = filler_stats.get("most_frequent", None)
        audio_features["filler_position_cluster"] = filler_stats.get("position_cluster", "none")

        fillers = filler_stats.get("fillers", [])
        timings["s3_correction"] = round(time.perf_counter() - t, 2)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 4: DEEP HR ANALYSIS — SINGLE UNIFIED LLM CALL [GPU]
        # This is the OPTIMIZED stage: ONE call instead of TWO
        # CPU is free during this stage — can precompute semantic data
        # ══════════════════════════════════════════════════════════════════
        t = time.perf_counter()
        logger.info("🧠 [S4] Starting HR Analysis (single-pass unified inference)...")

        # Run GPU LLM AND CPU semantic extraction in parallel
        def _run_llm():
            return genai_engine.comprehensive_analyze(
                refined_text, resume_text, audio_features,
                filler_stats, strictness
            )

        def _run_semantic():
            return self.semantic.analyze(refined_text, raw_text=raw_text)

        llm_task = loop.run_in_executor(self._gpu_pool, _run_llm)
        semantic_task = loop.run_in_executor(self._cpu_thread_pool, _run_semantic)

        llm_analysis, semantic_result = await asyncio.gather(llm_task, semantic_task)

        timings["s4_hr_analysis"] = round(time.perf_counter() - t, 2)
        logger.info(f"✅ [S4] HR Analysis: {timings['s4_hr_analysis']}s")

        # ══════════════════════════════════════════════════════════════════
        # STAGE 5: FEEDBACK ASSEMBLY + DL SCORING [CPU] — PARALLEL
        # Both run on CPU simultaneously
        # ══════════════════════════════════════════════════════════════════
        t = time.perf_counter()

        def _feedback():
            return self.feedback_service.generate(
                user_id=user_id, transcript=refined_text,
                semantic=semantic_result, scores={},
                fillers=fillers, english_level=english_lvl,
                audio_features=audio_features,
                precomputed_llm=llm_analysis,
                strictness=strictness
            )

        def _dl_scoring():
            llm_score = None
            if llm_analysis:
                llm_score = llm_analysis.get("overall_score")
            return self.scorer.calculate_score(
                refined_text,
                semantic_result.get("structured", {}),
                llm_score, transcript_confidence, audio_features
            )

        feedback, score_packet = await asyncio.gather(
            loop.run_in_executor(self._cpu_thread_pool, _feedback),
            loop.run_in_executor(self._cpu_thread_pool, _dl_scoring),
        )

        timings["s5_scoring_feedback"] = round(time.perf_counter() - t, 2)

        # ══════════════════════════════════════════════════════════════════
        # FINAL: Merge Scores — Rubric + DL Hybrid
        # ══════════════════════════════════════════════════════════════════
        rubric_score = feedback.get("overall_rubric_score", 5.0)
        dl_score = score_packet.get("overall_score", 5.0)

        # Hybrid: 60% rubric (HR reasoning) + 40% DL (neural network)
        hybrid_score = round(rubric_score * 0.6 + dl_score * 0.4, 1)
        hybrid_score = max(1.0, min(10.0, hybrid_score))

        score_packet["overall_score"] = hybrid_score
        score_packet["rubric_score"] = rubric_score
        score_packet["dl_raw_score"] = dl_score
        score_packet["score_source"] = "hybrid_hr_dl"

        # Cleanup temp files
        for f in temp_files:
            try: os.remove(f)
            except: pass

        pipeline_time = round(time.perf_counter() - t_pipeline, 2)
        timings["total"] = pipeline_time

        # ══════════════════════════════════════════════════════════════════
        # BUILD FINAL RESPONSE
        # ══════════════════════════════════════════════════════════════════
        confidence_label = audio_features.get("confidence_label", "MEDIUM")

        fb_positives = feedback.get("positives", [])
        fb_improvements = feedback.get("improvements", [])
        fb_coaching = feedback.get("coaching_summary", "")
        fb_suggestions = []

        # Suggestions from LLM
        if llm_analysis:
            llm_fb = llm_analysis.get("feedback", {})
            if isinstance(llm_fb, dict):
                fb_suggestions = llm_fb.get("suggestions", [])

        # Resume alignment — prefer LLM's analysis
        resume_align = {}
        if llm_analysis and llm_analysis.get("resume_alignment"):
            resume_align = llm_analysis["resume_alignment"]
        elif feedback.get("resume_alignment"):
            resume_align = feedback["resume_alignment"]

        logger.info(
            f"📋 [Pipeline] Final: "
            f"{len(fb_positives)}P, {len(fb_improvements)}I, "
            f"{len(fb_suggestions)}S, "
            f"coaching={'yes' if fb_coaching else 'no'}, "
            f"resume_m={len(resume_align.get('matched', []))}, "
            f"resume_x={len(resume_align.get('missed', []))}"
        )

        final_result = {
            "user_id": user_id,
            "raw_transcript": raw_text,
            "refined_transcript": refined_text,
            "semantic": semantic_result,
            "audio_features": audio_features,
            "audio_flags": audio_flags,
            "fillers": fillers,
            "filler_stats": filler_stats,
            "scores": score_packet,
            "feedback": {
                "positives": fb_positives,
                "improvements": fb_improvements,
                "suggestions": fb_suggestions,
                "coaching_summary": fb_coaching,
            },
            "rubric_breakdown": [],
            "resume_alignment": resume_align,
            "processing_time": pipeline_time,
            "timings": timings,
            "english_level": english_lvl,
            "confidence": {
                "transcript_confidence": transcript_confidence,
                "dynamic_confidence": audio_features.get("dynamic_confidence", 50.0),
                "confidence_label": confidence_label,
            },
        }

        # ── DB PERSIST ───────────────────────────────────────────────────
        try:
            user_name = semantic_result.get("structured", {}).get("name", "Unknown")
            db.upsert_user(user_id, user_name)
            db.store_attempt(
                user_id, refined_text, semantic_result, len(fillers),
                hybrid_score, final_result["feedback"],
                confidence=final_result["confidence"]["dynamic_confidence"],
                processing_time=pipeline_time,
                grammar_score=score_packet.get("details", {}).get("dl_metrics", {}).get("clarity", 5.0),
                speaking_score=score_packet.get("details", {}).get("dl_metrics", {}).get("fluency", 5.0),
                content_score=score_packet.get("details", {}).get("dl_metrics", {}).get("structure", 5.0),
            )
        except Exception as e:
            logger.error(f"❌ DB Store failed: {e}")

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ PIPELINE COMPLETE — {pipeline_time}s")
        logger.info(f"   Score: {hybrid_score}/10 (Rubric: {rubric_score}, DL: {dl_score})")
        logger.info(f"   Feedback: {len(fb_positives)} positives, {len(fb_improvements)} improvements")
        logger.info(f"   Resume: {len(resume_align.get('matched', []))} matched, {len(resume_align.get('missed', []))} missed")
        logger.info(f"   Timings: {timings}")
        logger.info(f"{'='*60}\n")
        return final_result
