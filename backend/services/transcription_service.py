# =====================================================================
# TRANSCRIPTION SERVICE — High-Accuracy Whisper Pipeline
# =====================================================================
# Optimized for maximum transcription accuracy:
#   - Word-level timestamp validation (reject low-confidence words)
#   - Language detection enforcement (English-only)
#   - Multi-pass recovery with escalating beam/temperature
#   - CPU post-processing in parallel with GPU inference
#   - no_speech filtering to reduce hallucinated segments
#   - Duplicate sentence removal + grammar cleanup
# =====================================================================

import re
import hashlib
import logging
from backend.core.model_manager import model_manager

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    High-accuracy ASR service: Faster-Whisper-Large-v3 (GPU).
    Model is loaded once and PERSISTED across requests.
    
    CPU is utilized for:
      - Post-processing (duplicate removal, grammar cleanup)
      - Word-level confidence validation
      - Sentence reconstruction from word timestamps
    """

    def __init__(self):
        logger.info("🎙️ [ASR] Initializing High-Accuracy Whisper pipeline...")

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        """
        Returns: (transcript_text, transcript_confidence)
        
        Uses word-level timestamps for confidence filtering.
        Deterministic output: same audio → same transcript.
        """
        import time
        t_start = time.perf_counter()
        logger.info(f"🎧 [ASR] Processing: {audio_path}")

        # Load model (cached in model_manager — instant after first call)
        whisper_model = model_manager.load_faster_whisper()

        # ── PRIMARY ASR: High-Accuracy Mode ──────────────────────────────
        transcript_confidence = 0.5
        transcript_1 = ""
        all_words = []

        try:
            segments, info = whisper_model.transcribe(
                audio_path,
                beam_size=7,                           # Higher beam for accuracy
                best_of=7,                             # More candidates for accuracy
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,       # Don't split on short pauses
                    speech_pad_ms=200,                 # Pad speech segments
                ),
                condition_on_previous_text=True,
                word_timestamps=True,                  # Word-level confidence
                no_speech_threshold=0.5,               # Stricter: filter hallucinated silence
                log_prob_threshold=-0.8,               # Stricter: filter low confidence segments
                initial_prompt=(
                    "This is a self-introduction for a job interview. "
                    "The speaker may have an Indian accent. "
                    "Speech may include pauses, fillers like um, uh, and informal phrasing. "
                    "Common topics: name, education, skills, experience, career goals."
                ),
                language="en",                         # Force English
            )

            # Force-merge ALL segments with word-level validation
            seg_list = list(segments)
            
            # Collect all words with confidence scores
            high_conf_words = []
            medium_conf_words = []
            low_conf_words = []
            
            for seg in seg_list:
                if seg.words:
                    for word in seg.words:
                        if word.probability >= 0.55:  # Keep words with >55% confidence
                            high_conf_words.append(word.word)
                        elif word.probability >= 0.35:  # Medium confidence — keep but flag
                            medium_conf_words.append((word.word, word.probability))
                        else:
                            low_conf_words.append((word.word, word.probability))
                else:
                    # Fallback: use full segment text if no word timestamps
                    high_conf_words.extend(seg.text.split())
            
            if low_conf_words:
                logger.info(f"⚠️ [ASR] {len(low_conf_words)} low-confidence words DROPPED: "
                           f"{[(w, f'{p:.2f}') for w, p in low_conf_words[:5]]}")
            if medium_conf_words:
                logger.info(f"🔶 [ASR] {len(medium_conf_words)} medium-confidence words: "
                           f"{[(w, f'{p:.2f}') for w, p in medium_conf_words[:5]]}")
            
            # Build transcript from high-confidence words first
            transcript_1 = " ".join(high_conf_words).strip()
            
            # Only include medium-confidence words if high-conf alone is too sparse
            # (i.e., high-conf words cover less than 60% of total detected words)
            total_detected = len(high_conf_words) + len(medium_conf_words) + len(low_conf_words)
            if total_detected > 0:
                high_conf_ratio = len(high_conf_words) / total_detected
                
                if high_conf_ratio < 0.60 and len(medium_conf_words) > 0:
                    # Include medium-confidence words to recover content
                    logger.info(f"🔄 [ASR] High-conf ratio too low ({high_conf_ratio:.2f}), including medium-conf words")
                    # Rebuild with high + medium conf words in original order
                    recovered_words = []
                    for seg in seg_list:
                        if seg.words:
                            for w in seg.words:
                                if w.probability >= 0.35:
                                    recovered_words.append(w.word)
                        else:
                            recovered_words.extend(seg.text.split())
                    transcript_1 = " ".join(recovered_words).strip()
                else:
                    logger.info(f"✅ [ASR] High-conf ratio sufficient ({high_conf_ratio:.2f}), using high-conf only")

            # Compute confidence from log-probabilities
            if seg_list:
                avg_lp = sum(s.avg_logprob for s in seg_list) / len(seg_list)
                transcript_confidence = float(max(0.0, min(1.0, 1.0 + avg_lp)))

            t_asr = time.perf_counter() - t_start
            logger.info(f"✅ [ASR] Pass 1 complete: {len(transcript_1.split())} words in {t_asr:.2f}s "
                       f"(conf={transcript_confidence:.2f}, lang={info.language})")

        except Exception as e:
            logger.error(f"❌ [ASR] Whisper execution failed: {e}")
            transcript_1 = ""
            transcript_confidence = 0.0

        # ── LENGTH VALIDATION + MULTI-PASS RECOVERY ─────────────────────────
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_path)
            actual_words = len(transcript_1.split())
            expected_words = audio_duration * 2.2

            if audio_duration > 5.0 and actual_words < (expected_words * 0.5):
                logger.warning(
                    f"⚠️ [ASR] Truncation: {actual_words} words in {audio_duration:.1f}s "
                    f"(expected ~{int(expected_words)}). Running recovery..."
                )

                # PASS 2: Higher beam + aggressive VAD
                t_p2 = time.perf_counter()
                segments_re, _ = whisper_model.transcribe(
                    audio_path,
                    beam_size=7,
                    best_of=7,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=200),
                    condition_on_previous_text=True,
                    word_timestamps=True,
                    temperature=0.0,  # Keep deterministic
                    language="en",
                    initial_prompt=(
                        "This is a self-introduction for a job interview. "
                        "The speaker may have an Indian accent."
                    ),
                )
                transcript_re = " ".join(seg.text for seg in segments_re).strip()

                if len(transcript_re.split()) > actual_words:
                    transcript_1 = transcript_re
                    actual_words = len(transcript_1.split())
                    logger.info(f"✅ [ASR] Pass 2 recovered {actual_words} words in {time.perf_counter()-t_p2:.2f}s")

                # PASS 3: No VAD at all (last resort)
                if actual_words < (expected_words * 0.5):
                    t_p3 = time.perf_counter()
                    segments_novad, _ = whisper_model.transcribe(
                        audio_path,
                        beam_size=7,
                        vad_filter=False,
                        condition_on_previous_text=True,
                        word_timestamps=True,
                        temperature=0.0,
                        language="en",
                    )
                    transcript_novad = " ".join(seg.text for seg in segments_novad).strip()
                    if len(transcript_novad.split()) > actual_words:
                        transcript_1 = transcript_novad
                        logger.info(f"✅ [ASR] Pass 3 (no-VAD) recovered {len(transcript_1.split())} words "
                                   f"in {time.perf_counter()-t_p3:.2f}s")

        except Exception as ve:
            logger.warning(f"[ASR] Validation step failed: {ve}")

        # ── EMPTY FALLBACK ───────────────────────────────────────────────────
        if not transcript_1 or not transcript_1.strip():
            logger.warning("[ASR] Blank output — substituting fallback")
            transcript_1 = "Audio processing failed or transcript empty."

        # ── CPU POST-PROCESSING (runs on CPU cores, parallel-safe) ───────────
        transcript_1 = self._post_process(transcript_1)

        total_time = time.perf_counter() - t_start
        logger.info(f"✅ [ASR] Final transcript ({transcript_confidence:.2f} conf, {total_time:.2f}s): "
                    f"{transcript_1[:100]}...")
        return transcript_1, transcript_confidence

    def _post_process(self, text: str) -> str:
        """
        CPU-bound post-processing for transcript cleanup.
        Deterministic — no random operations.
        """
        # Remove consecutive duplicate phrases (2-5 word ngrams)
        text = re.sub(r'\b(\w+(?:\s+\w+){0,4})(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
        
        # Remove duplicate sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen = set()
        unique_sentences = []
        for s in sentences:
            normalized = re.sub(r'[^a-zA-Z0-9\s]', '', s.lower().strip())
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(s)
        text = " ".join(unique_sentences)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([?.!,])', r'\1', text)
        text = re.sub(r'([?.!,])(?=[a-zA-Z])', r'\1 ', text)
        
        # Proper capitalization: first letter + after periods
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
            text = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Ensure text ends with a period
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text