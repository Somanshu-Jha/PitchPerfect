# =====================================================================
# AUDIO ANALYSIS SERVICE — Deep Audio Intelligence Engine
# =====================================================================
# Extracts production-grade speech features from raw audio:
#   - Tone analysis (MFCC, pitch contour expressiveness)
#   - Fluency score (pause patterns, rhythm consistency, speech flow)
#   - Pronunciation clarity (spectral flatness, HNR, formant stability)
#   - Speaking pace classification (WPM estimate)
#   - Voice energy distribution (sustain/fade detection)
#   - Dynamic confidence (multi-factor weighted)
#
# Returns raw metrics + human-readable 'reasoning' dict for feedback.
# Runs on CPU — safe to parallelize alongside GPU-bound ASR.
# =====================================================================

import librosa
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Use all available CPU cores for librosa/numpy operations
_NUM_CORES = os.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_NUM_CORES))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_NUM_CORES))


class AudioAnalysisService:
    """
    Deep audio feature extraction engine.
    Produces 15+ metrics covering tone, fluency, pronunciation,
    pace, energy, and confidence — with reasoning explanations.
    """

    def extract(self, audio_path: str) -> dict:
        """Main entry point. Returns dict of features + reasoning."""

        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration <= 0.5:
            return self._empty_result("Audio too short for analysis")

        # ═══════════════════════════════════════════════════════════════
        # 1. SPEECH RATE & PACE
        # ═══════════════════════════════════════════════════════════════
        words_estimate = len(y) / sr / 0.4  # ~0.4s per word average
        speech_rate = words_estimate / max(0.1, duration)
        wpm_estimate = speech_rate * 60

        if wpm_estimate < 80:
            pace_label = "too_slow"
            pace_reasoning = f"Your speaking pace (~{int(wpm_estimate)} WPM) is quite slow. Ideal interview pace is 120-160 WPM. Try increasing your energy and pace slightly."
        elif wpm_estimate < 120:
            pace_label = "slightly_slow"
            pace_reasoning = f"Your pace (~{int(wpm_estimate)} WPM) is slightly below the ideal 120-160 WPM range. A bit more energy would make your delivery more engaging."
        elif wpm_estimate <= 170:
            pace_label = "ideal"
            pace_reasoning = f"Excellent speaking pace (~{int(wpm_estimate)} WPM). This is right in the ideal 120-160 WPM range for interviews."
        elif wpm_estimate <= 200:
            pace_label = "slightly_fast"
            pace_reasoning = f"Your pace (~{int(wpm_estimate)} WPM) is slightly fast. Slowing down a touch will improve clarity and show composure."
        else:
            pace_label = "too_fast"
            pace_reasoning = f"Your pace (~{int(wpm_estimate)} WPM) is too fast. Slow down significantly — interviewers need time to process what you say."

        # ═══════════════════════════════════════════════════════════════
        # 2. SPEECH RATE STABILITY (rhythm consistency)
        # ═══════════════════════════════════════════════════════════════
        window_samples = sr  # 1-second windows
        window_energies = []
        for i in range(0, len(y), window_samples):
            chunk = y[i:i + window_samples]
            if len(chunk) < window_samples // 2:
                break
            chunk_energy = float(np.mean(np.abs(chunk)))
            window_energies.append(chunk_energy)

        if len(window_energies) > 1:
            rate_std = float(np.std(window_energies))
            rate_mean = float(np.mean(window_energies))
            cv = rate_std / max(rate_mean, 1e-6)  # coefficient of variation
            speech_rate_stability = max(0.0, min(1.0, 1.0 - cv * 2))
        else:
            speech_rate_stability = 0.5

        # ═══════════════════════════════════════════════════════════════
        # 3. PAUSE ANALYSIS (fluency component)
        # ═══════════════════════════════════════════════════════════════
        silence_threshold = 0.01
        silence_frames = np.sum(np.abs(y) < silence_threshold)
        pause_ratio = float(silence_frames / len(y))
        pause_control = max(0.0, min(1.0, 1.0 - (pause_ratio * 1.5)))

        # Detect individual pause segments
        is_silent = np.abs(y) < silence_threshold
        pause_starts = np.where(np.diff(is_silent.astype(int)) == 1)[0]
        pause_ends = np.where(np.diff(is_silent.astype(int)) == -1)[0]
        if len(pause_starts) > 0 and len(pause_ends) > 0:
            if pause_ends[0] < pause_starts[0]:
                pause_ends = pause_ends[1:]
            min_len = min(len(pause_starts), len(pause_ends))
            pause_durations = (pause_ends[:min_len] - pause_starts[:min_len]) / sr
            long_pauses = np.sum(pause_durations > 0.8)  # pauses > 0.8s
            avg_pause_dur = float(np.mean(pause_durations)) if len(pause_durations) > 0 else 0
        else:
            long_pauses = 0
            avg_pause_dur = 0

        # ═══════════════════════════════════════════════════════════════
        # 4. PITCH / TONE ANALYSIS 
        # ═══════════════════════════════════════════════════════════════
        pitch_track = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        # Filter out unreliable frames (0 or extreme values)
        valid_pitch = pitch_track[(pitch_track > 50) & (pitch_track < 400)]

        if len(valid_pitch) > 5:
            pitch_mean = float(np.mean(valid_pitch))
            pitch_std = float(np.std(valid_pitch))
            pitch_range = float(np.max(valid_pitch) - np.min(valid_pitch))
            pitch_var = float(np.var(valid_pitch))
        else:
            pitch_mean, pitch_std, pitch_range, pitch_var = 150.0, 20.0, 50.0, 400.0

        # Pitch variation normalized (0-1): higher = more expressive
        pitch_variation = min(1.0, pitch_var / 2000.0)

        # Tone expressiveness: based on pitch range and standard deviation
        # Monotone: pitch_range < 30Hz, Expressive: pitch_range > 80Hz
        tone_expressiveness = min(1.0, pitch_range / 150.0)

        if tone_expressiveness < 0.25:
            tone_label = "monotone"
            tone_reasoning = f"Your voice has very little pitch variation (range: {pitch_range:.0f}Hz). This sounds monotone. Try varying your intonation — go higher when making key points and lower when wrapping up."
        elif tone_expressiveness < 0.45:
            tone_label = "flat"
            tone_reasoning = f"Your pitch variation is limited (range: {pitch_range:.0f}Hz). Adding more vocal variety would make you sound more energetic and engaging."
        elif tone_expressiveness < 0.7:
            tone_label = "moderate"
            tone_reasoning = f"Good tonal variety (range: {pitch_range:.0f}Hz). Your voice carries natural expression which helps maintain listener interest."
        else:
            tone_label = "expressive"
            tone_reasoning = f"Excellent vocal expressiveness (range: {pitch_range:.0f}Hz). Your voice modulation is natural and engaging — a strong interviewing trait."

        # ═══════════════════════════════════════════════════════════════
        # 5. MFCC-BASED TONE QUALITY
        # ═══════════════════════════════════════════════════════════════
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_var = np.var(mfccs, axis=1)
        # Higher variance in MFCCs 2-5 indicates richer vocal tone quality
        tone_richness = float(np.mean(mfcc_var[1:5])) / 100.0  # normalize
        tone_richness = max(0.0, min(1.0, tone_richness))

        # ═══════════════════════════════════════════════════════════════
        # 6. FLUENCY SCORE (composite)
        # ═══════════════════════════════════════════════════════════════
        # Fluency = f(pause_control, rhythm_stability, speech_continuity)
        speech_continuity = 1.0 - min(1.0, long_pauses / max(1, duration / 10))
        fluency_score = (
            0.35 * pause_control +
            0.30 * speech_rate_stability +
            0.35 * speech_continuity
        )
        fluency_score = max(0.0, min(1.0, fluency_score))

        if fluency_score < 0.3:
            fluency_reasoning = f"Your speech has significant fluency issues — {long_pauses} long pauses detected, average pause {avg_pause_dur:.1f}s. Practice speaking continuously without extended breaks."
        elif fluency_score < 0.5:
            fluency_reasoning = f"Moderate fluency — some hesitations and pauses detected ({long_pauses} long pauses). Try rehearsing your introduction to reduce pauses and build flow."
        elif fluency_score < 0.75:
            fluency_reasoning = f"Good fluency with mostly smooth delivery. Minor pauses don't significantly impact your presentation. Keep practicing for even smoother flow."
        else:
            fluency_reasoning = "Excellent fluency — your speech flows naturally with well-managed pauses. This indicates strong preparation and confidence."

        # ═══════════════════════════════════════════════════════════════
        # 7. ENERGY CONSISTENCY & DISTRIBUTION
        # ═══════════════════════════════════════════════════════════════
        rms = librosa.feature.rms(y=y)[0]
        rms_var = float(np.var(rms))
        rms_mean = float(np.mean(rms))
        energy_consistency = min(1.0, 1.0 / (1.0 + rms_var * 50))

        # Energy trajectory: how does energy change from start to end?
        n_segments = 4
        seg_len = len(rms) // n_segments if n_segments > 0 else len(rms)
        segment_energies = []
        for i in range(n_segments):
            seg = rms[i * seg_len:(i + 1) * seg_len]
            if len(seg) > 0:
                segment_energies.append(float(np.mean(seg)))
        
        if len(segment_energies) >= 2:
            energy_trend = segment_energies[-1] - segment_energies[0]
            if energy_trend > 0.005:
                energy_trajectory = "building"
                energy_reasoning = "Your voice energy builds as you speak — this is excellent! It shows growing confidence and engagement."
            elif energy_trend < -0.01:
                energy_trajectory = "fading"
                energy_reasoning = "Your voice energy drops towards the end. Try maintaining a strong, consistent volume throughout — especially in your closing statement."
            else:
                energy_trajectory = "steady"
                energy_reasoning = "Your voice energy stays consistent throughout — good volume control and steady delivery."
        else:
            energy_trajectory = "unknown"
            energy_reasoning = "Audio too short to determine energy trajectory."

        # ═══════════════════════════════════════════════════════════════
        # 8. PRONUNCIATION CLARITY (spectral + HNR)
        # ═══════════════════════════════════════════════════════════════
        # Spectral centroid — brightness/clarity
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        centroid_norm = min(1.0, spectral_centroid / 4000.0)

        # Zero-crossing rate — consonant articulation
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        zcr_norm = min(1.0, zcr / 0.15)

        # Spectral flatness — noisy vs tonal (lower = more tonal/clearer)
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        clarity_from_flatness = max(0.0, min(1.0, 1.0 - spectral_flatness * 5))

        # Harmonic-to-Noise Ratio approximation
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.sum(harmonic ** 2))
        percussive_energy = float(np.sum(percussive ** 2))
        hnr = harmonic_energy / max(percussive_energy, 1e-8)
        hnr_score = min(1.0, hnr / 20.0)  # typical speech HNR is 10-25

        # Combined pronunciation score
        pronunciation_score = (
            centroid_norm * 0.25 +
            zcr_norm * 0.15 +
            clarity_from_flatness * 0.30 +
            hnr_score * 0.30
        )
        pronunciation_score = max(0.0, min(1.0, pronunciation_score))

        if pronunciation_score < 0.3:
            pronunciation_reasoning = "Your pronunciation clarity is low — words may be hard to understand. Practice speaking slowly and enunciating each syllable clearly."
        elif pronunciation_score < 0.5:
            pronunciation_reasoning = "Pronunciation is understandable but could be clearer. Focus on opening your mouth wider and articulating consonants more distinctly."
        elif pronunciation_score < 0.7:
            pronunciation_reasoning = "Good pronunciation clarity. Most words are well-articulated and easy to understand."
        else:
            pronunciation_reasoning = "Excellent pronunciation — your words are clear, well-articulated, and easy to follow. This significantly enhances your professional impression."

        # Duration-based dampening: short audio gives less reliable analysis
        # For recordings < 30s, dampen all composite scores proportionally
        if duration < 30.0:
            duration_factor = max(0.3, duration / 30.0)  # 0.3-1.0
            tone_expressiveness *= duration_factor
            fluency_score *= duration_factor
            pronunciation_score *= duration_factor
            speech_rate_stability *= duration_factor
            logger.info(f"⚠️ [AudioAnalysis] Duration dampening applied: {duration:.1f}s → factor={duration_factor:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # 9. DYNAMIC CONFIDENCE (multi-factor)
        # ═══════════════════════════════════════════════════════════════
        base_confidence = (
            0.15 * speech_rate_stability +
            0.15 * pause_control +
            0.15 * pitch_variation +
            0.15 * energy_consistency +
            0.15 * pronunciation_score +
            0.15 * fluency_score +
            0.10 * tone_expressiveness
        ) * 100

        # NO artificial scaling — use raw computed confidence
        dynamic_confidence = max(0.0, min(100.0, base_confidence))

        if dynamic_confidence < 40:
            confidence_label = "LOW"
        elif dynamic_confidence < 75:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "HIGH"

        logger.info(
            f"🎯 [AudioAnalysis] Confidence={dynamic_confidence:.1f}% ({confidence_label}) | "
            f"Tone={tone_label}({tone_expressiveness:.2f}) | Fluency={fluency_score:.2f} | "
            f"Pronun={pronunciation_score:.3f} | Pace={pace_label}({wpm_estimate:.0f}WPM) | "
            f"Energy={energy_trajectory}"
        )

        return {
            # Core metrics
            "speech_rate": round(speech_rate, 2),
            "wpm_estimate": round(wpm_estimate, 0),
            "pace_label": pace_label,
            "pause_ratio": round(pause_ratio, 3),
            "long_pauses": int(long_pauses),
            "avg_pause_duration": round(avg_pause_dur, 2),
            "pitch": round(pitch_mean, 2),
            "pitch_range": round(pitch_range, 2),
            "pitch_variance": round(pitch_variation, 4),
            "energy_consistency": round(energy_consistency, 4),
            "energy_trajectory": energy_trajectory,
            "speech_rate_stability": round(speech_rate_stability, 4),
            
            # Deep analysis metrics
            "tone_expressiveness": round(tone_expressiveness, 4),
            "tone_label": tone_label,
            "tone_richness": round(tone_richness, 4),
            "fluency_score": round(fluency_score, 4),
            "pronunciation_score": round(pronunciation_score, 4),
            "spectral_flatness": round(spectral_flatness, 4),
            "hnr_score": round(hnr_score, 4),
            
            # Confidence
            "dynamic_confidence": round(dynamic_confidence, 1),
            "confidence_label": confidence_label,

            # Human-readable reasoning (for feedback service)
            "reasoning": {
                "tone": tone_reasoning,
                "fluency": fluency_reasoning,
                "pronunciation": pronunciation_reasoning,
                "pace": pace_reasoning,
                "energy": energy_reasoning,
            }
        }

    def _empty_result(self, reason: str) -> dict:
        return {
            "speech_rate": 0, "wpm_estimate": 0, "pace_label": "unknown",
            "pause_ratio": 0, "long_pauses": 0, "avg_pause_duration": 0,
            "pitch": 0, "pitch_range": 0, "pitch_variance": 0,
            "energy_consistency": 0, "energy_trajectory": "unknown",
            "speech_rate_stability": 0,
            "tone_expressiveness": 0, "tone_label": "unknown", "tone_richness": 0,
            "fluency_score": 0, "pronunciation_score": 0,
            "spectral_flatness": 0, "hnr_score": 0,
            "dynamic_confidence": 0.0, "confidence_label": "LOW",
            "reasoning": {
                "tone": reason, "fluency": reason,
                "pronunciation": reason, "pace": reason, "energy": reason
            }
        }