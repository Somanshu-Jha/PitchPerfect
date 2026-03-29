# =====================================================================
# TRANSCRIPTION SERVICE — Dual-ASR Hybrid Pipeline
# =====================================================================
# Implements Whisper (GPU) + Wav2Vec2 (CPU) voting with:
# - Force-complete segment merge (NO truncation)
# - Duration-based length validation (2.2 words/sec expected)
# - Multi-pass recovery with escalating beam/temperature
# - Final fallback: full decode WITHOUT VAD filter
#
# Every parameter is documented with increase/decrease impact.
# =====================================================================

import torch
import logging
from backend.core.model_manager import model_manager
import textdistance

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Dual-ASR service: Faster-Whisper-Large-v3 (GPU) + Wav2Vec2 (CPU).
    Implements 3-tier transcript recovery to guarantee FULL output.
    """

    def __init__(self):
        logger.info("🎙️ [Hybrid_ASR_Service] Initializing...")

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        """
        Returns: (transcript_text, transcript_confidence)
        transcript_confidence: avg_logprob from faster-whisper mapped to 0–1
        """
        logger.info(f"🎧 Processing {audio_path}...")

        # 1. Load models sequentially to stay within 12GB VRAM budget
        whisper_model = model_manager.load_faster_whisper()
        wav2vec_handler = model_manager.load_wav2vec()

        # ── PRIMARY ASR: Faster-Whisper-Large-v3 (GPU) ──────────────────────
        logger.info("[ASR_Primary] Starting Faster-Whisper-Large-v3 Inference...")
        transcript_confidence = 0.5  # safe default if computation fails
        try:
            segments, info = whisper_model.transcribe(
                audio_path,
                # beam_size=5: balances accuracy vs latency
                # higher → better accuracy, slower inference
                # lower → faster but may miss words
                beam_size=5,
                # vad_filter=True: removes non-speech segments
                # Prevents hallucinated text during silence
                vad_filter=True,
                vad_parameters=dict(
                    # min_silence_duration_ms=500: pauses shorter than 500ms
                    # are NOT treated as segment breaks
                    # higher → fewer segments, may merge sentences
                    # lower → more segments, finer granularity
                    min_silence_duration_ms=500
                ),
                # initial_prompt: biases the decoder toward expected vocabulary
                # Critical for Indian English accent handling
                initial_prompt="The following is a transcript of an Indian English speaker answering interview questions.",
                # condition_on_previous_text=False: each segment decoded independently
                # Prevents error propagation from one segment to the next
                # Set True only in recovery passes where context helps
                condition_on_previous_text=False
            )

            # ── FORCE COMPLETE MERGE (MANDATORY) ────────────────────────────
            # Materialize ALL segments and concatenate without dropping any
            seg_list = list(segments)
            # Build transcript by iterating every segment — no filtering
            text = ""
            for seg in seg_list:
                text += seg.text + " "
            transcript_1 = text.strip()

            # Compute average log-probability as confidence metric
            if seg_list:
                avg_lp = sum(s.avg_logprob for s in seg_list) / len(seg_list)
                # avg_logprob range: typically -1.0 (bad) to 0.0 (perfect)
                # Map to 0–1 by adding 1.0 and clamping
                transcript_confidence = float(max(0.0, min(1.0, 1.0 + avg_lp)))
        except Exception as e:
            logger.error(f"❌ Faster-Whisper execution failed! Exception: {e}")
            transcript_1 = ""
            transcript_confidence = 0.0

        # ── LENGTH VALIDATION (MANDATORY) ───────────────────────────────────
        # expected_words = audio_duration * 2.2 (average conversational WPS)
        # If actual < 50% of expected → truncation detected → reprocess
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_path)
            actual_words = len(transcript_1.split())
            expected_words = audio_duration * 2.2  # 2.2 words/sec = conversational pace

            if audio_duration > 5.0 and actual_words < (expected_words * 0.5):
                logger.warning(
                    f"⚠️ Truncation detected: {actual_words} words in {audio_duration:.1f}s "
                    f"(expected ~{int(expected_words)}). Running multi-pass recovery..."
                )

                # ── PASS 2: Higher beam + context stitching ─────────────────
                segments_re, _ = whisper_model.transcribe(
                    audio_path,
                    # beam_size=7: heavier search → finds more paths through audio
                    beam_size=7,
                    vad_filter=True,
                    vad_parameters=dict(
                        # min_silence_duration_ms=200: much more aggressive
                        # Catches words in short pauses that Pass 1 may skip
                        min_silence_duration_ms=200
                    ),
                    # condition_on_previous_text=True: uses prior context as decoder seed
                    # Helps stitch cut-off sentences back together
                    condition_on_previous_text=True,
                    # temperature=0.2: slight randomness to explore alternate decodings
                    # without high hallucination risk
                    temperature=0.2
                )

                # Force-merge all recovered segments
                text_re = ""
                for seg in segments_re:
                    text_re += seg.text + " "
                transcript_re = text_re.strip()

                if len(transcript_re.split()) > actual_words:
                    transcript_1 = transcript_re
                    actual_words = len(transcript_1.split())
                    logger.info(f"✅ Pass 2 recovered {actual_words} words!")

                # ── PASS 3 (FINAL SAFETY): No VAD at all ────────────────────
                # If still short after Pass 2, disable VAD entirely
                # This catches edge cases where VAD incorrectly clips speech
                if actual_words < (expected_words * 0.5):
                    logger.warning("⚠️ Still truncated after Pass 2. Running full decode WITHOUT VAD...")
                    segments_novad, _ = whisper_model.transcribe(
                        audio_path,
                        beam_size=7,
                        vad_filter=False,  # VAD completely disabled
                        condition_on_previous_text=True,
                        temperature=0.2
                    )
                    text_novad = ""
                    for seg in segments_novad:
                        text_novad += seg.text + " "
                    transcript_novad = text_novad.strip()

                    if len(transcript_novad.split()) > actual_words:
                        transcript_1 = transcript_novad
                        logger.info(f"✅ Pass 3 (no-VAD) recovered {len(transcript_1.split())} words!")

        except Exception as ve:
            logger.warning(f"Validation step failed: {ve}")

        # ── SECONDARY ASR: Wav2Vec2 (CPU fallback) ──────────────────────────
        logger.info("[ASR_Secondary] Starting Wav2Vec2 Inference (CPU fallback)...")
        try:
            import soundfile as sf
            import librosa
            import numpy as np

            waveform_np, sample_rate = sf.read(audio_path)
            # Force mono: Wav2Vec expects single-channel input
            if getattr(waveform_np, "ndim", 1) > 1:
                waveform_np = waveform_np.mean(axis=1)

            # Resample to 16kHz if needed — Wav2Vec2 trained on 16kHz audio
            if sample_rate != 16000:
                waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=16000)

            processor = wav2vec_handler["processor"]
            model = wav2vec_handler["model"]

            inputs = processor(waveform_np, return_tensors="pt", sampling_rate=16000)

            # CPU-only inference to avoid GPU contention with Whisper
            with torch.no_grad():
                logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcript_2 = processor.batch_decode(predicted_ids)[0]
        except Exception as e:
            logger.error(f"❌ Wav2Vec execution failed! Exception: {e}")
            transcript_2 = ""

        # ── HYBRID RESOLVER ─────────────────────────────────────────────────
        final_transcript = self.select_best(transcript_1, transcript_2)
        if not final_transcript or not final_transcript.strip():
            logger.warning("Blank output generated by ASR, substituting fallback")
            final_transcript = "Audio processing failed or transcript empty."

        # ── VRAM CLEANUP ────────────────────────────────────────────────────
        # Unload ASR models before LLM loads to stay within 12GB budget
        logger.info("🧹 [Hybrid_ASR] Invoking ModelManager GPU Sweep...")
        model_manager.unload("faster_whisper")
        model_manager.unload("wav2vec")
        model_manager.clear()  # torch.cuda.empty_cache()

        logger.info(f"✅ Final Resolved Transcript ({transcript_confidence:.2f} conf): {final_transcript[:50]}...")
        return final_transcript, transcript_confidence

    def select_best(self, transcript_1: str, transcript_2: str) -> str:
        """
        Dual-ASR Voting Matrix.
        t1 = whisper-large-v3 (GPU), t2 = wav2vec2 (CPU).
        Whisper is favored unless it catastrophically failed and Wav2Vec
        retrieved a significantly more coherent output.
        """
        if not transcript_1 and not transcript_2:
            return ""
        if not transcript_1:
            return transcript_2
        if not transcript_2:
            return transcript_1

        # Lexical richness: ratio of unique words to total words
        # Wav2Vec often hallucinates repetitive characters (THHHHEEEE)
        # so its richness score drops dramatically when it fails
        unique_t1 = len(set(transcript_1.split()))
        unique_t2 = len(set(transcript_2.split()))

        score_1 = unique_t1 / max(len(transcript_1.split()), 1)
        score_2 = unique_t2 / max(len(transcript_2.split()), 1)

        # Structural similarity between transcripts
        similarity = textdistance.jaccard(transcript_1.split(), transcript_2.lower().split())

        logger.info(f"[Resolver] T1 Coherence: {score_1:.2f} | T2 Coherence: {score_2:.2f} | Similarity: {similarity:.2f}")

        # Favor Whisper if coherent (>0.4 richness) OR transcripts agree (>0.5 sim)
        if score_1 > 0.4 or similarity > 0.5:
            return transcript_1
        elif score_2 > score_1 * 1.5:
            # Only prefer Wav2Vec if it is drastically more coherent
            return transcript_2

        return transcript_1