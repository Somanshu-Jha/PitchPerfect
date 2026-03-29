# =====================================================================
# AUDIO ANALYSIS SERVICE — Dynamic Confidence Engine
# =====================================================================
# Extracts speech-level features from raw audio and computes a
# multi-factor dynamic confidence score (0-100).
#
# ALL parameters are commented with their impact and tuning notes.
# =====================================================================

import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AudioAnalysisService:
    """
    Extracts acoustic features and computes a weighted dynamic confidence
    score using 5 independent signal components.
    Runs on CPU — safe to parallelize alongside GPU-bound ASR.
    """

    def extract(self, audio_path: str) -> dict:
        """
        Main entry point.  Returns a dict of raw features + dynamic_confidence + label.
        """

        # sr=16000 → standard telephony sample rate, matches Whisper/Wav2Vec input
        # Higher sr (e.g. 22050) would waste memory without ASR benefit
        y, sr = librosa.load(audio_path, sr=16000)

        duration = librosa.get_duration(y=y, sr=sr)

        if duration <= 0:
            return {
                "speech_rate": 0, "pause_ratio": 0, "pitch": 0,
                "pitch_variance": 0, "energy_consistency": 0,
                "pronunciation_score": 0,
                "dynamic_confidence": 0.0, "confidence_label": "LOW"
            }

        # ── SPEECH RATE (words per second estimate) ─────────────────────────
        # 0.4s per word is an empirical average for conversational English
        # Increasing → fewer estimated words → lower speech_rate
        # Decreasing → more estimated words → higher speech_rate (overcount)
        words_estimate = len(y) / sr / 0.4
        speech_rate = words_estimate / duration  # ideal range ~1.5 – 2.5 wps

        # ── SPEECH RATE STABILITY ───────────────────────────────────────────
        # Split audio into 1-second windows and measure rate variance
        # Low variance = stable speaking pace = higher confidence
        window_samples = sr  # 1 second worth of samples
        window_rates = []
        for i in range(0, len(y), window_samples):
            chunk = y[i:i + window_samples]
            if len(chunk) < window_samples // 2:
                break  # ignore trailing sub-half-second fragments
            chunk_energy = np.sum(np.abs(chunk) > 0.01) / len(chunk)
            window_rates.append(chunk_energy)

        # Stability = inverse of standard deviation (capped 0-1)
        # Higher std → unstable pace → lower stability score
        if len(window_rates) > 1:
            rate_std = float(np.std(window_rates))
            speech_rate_stability = max(0.0, min(1.0, 1.0 - rate_std))
        else:
            speech_rate_stability = 0.5  # default for very short clips

        # ── PAUSE CONTROL ───────────────────────────────────────────────────
        # Threshold 0.01 → amplitude below this is considered silence
        # Increasing threshold → more frames classified as pause → higher ratio
        # Decreasing → less pause detected → lower ratio
        silence_frames = np.sum(np.abs(y) < 0.01)
        pause_ratio = float(silence_frames / len(y))

        # pause_control: 1.0 = no pauses, 0.0 = entirely silent
        # Users with moderate pauses (~20%) should still score well
        pause_control = max(0.0, min(1.0, 1.0 - (pause_ratio * 1.5)))

        # ── PITCH VARIATION ─────────────────────────────────────────────────
        # fmin=50 / fmax=300 → human speech fundamental range
        # Increasing fmax → captures higher pitched speakers but adds noise
        # Decreasing fmin → captures very deep voices but may pick up rumble
        pitch_track = librosa.yin(y, fmin=50, fmax=300)
        pitch_mean = float(np.mean(pitch_track))
        pitch_var = float(np.var(pitch_track))

        # Normalize pitch variance to 0-1 range
        # Dividing by 2000 → empirical scaling so typical speech maps ~0.3-0.7
        # Higher → more expressive / animated speech
        # Lower → monotone delivery
        pitch_variation = min(1.0, pitch_var / 2000.0)

        # ── ENERGY CONSISTENCY ──────────────────────────────────────────────
        # RMS (Root Mean Square) energy per frame
        # Low variance in RMS → consistent volume → high energy_consistency
        rms = librosa.feature.rms(y=y)[0]
        rms_var = float(np.var(rms))
        # Inverse variance scaled: smaller variance → score closer to 1.0
        # Multiplier 50 chosen so typical speech variance maps to ~0.5-0.9
        energy_consistency = min(1.0, 1.0 / (1.0 + rms_var * 50))

        # ── PRONUNCIATION SCORE (heuristic) ─────────────────────────────────
        # Approximated via spectral clarity:
        # Spectral centroid = "brightness" of the signal
        # Higher centroid → clearer articulation → better pronunciation
        # Zero-crossing rate → correlates with fricatives/consonant clarity
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Normalize centroid (typical human speech centroid ~1000-4000 Hz)
        centroid_norm = min(1.0, spectral_centroid / 4000.0)
        # ZCR ~0.05-0.15 for clean speech
        zcr_norm = min(1.0, zcr / 0.15)
        # Weighted blend: centroid contributes more to perceived clarity
        pronunciation_score = (centroid_norm * 0.6) + (zcr_norm * 0.4)

        # ── DYNAMIC CONFIDENCE FORMULA (MANDATORY 5-FACTOR) ─────────────────
        # Uses exact uniform 0.2 weight across all variables.
        # Note: using 'pause_control' as the 'pause_ratio' metric to ensure
        # that lower actual silence properly yields higher confidence.
        base_confidence = (
            0.2 * speech_rate_stability +
            0.2 * pause_control +
            0.2 * pitch_variation +
            0.2 * energy_consistency +
            0.2 * pronunciation_score
        ) * 100 

        # Scale confidence distribution (as requested: confidence = base_score * 1.3)
        scaled_confidence = base_confidence * 1.3

        # Clamp to valid range
        dynamic_confidence = max(0.0, min(100.0, scaled_confidence))

        # ── CONFIDENCE LABEL (MANDATORY RANGE ENFORCEMENT) ──────────────────
        if dynamic_confidence < 40:
            confidence_label = "LOW"
        elif dynamic_confidence < 75:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "HIGH"

        logger.info(
            f"🎯 [AudioAnalysis] Confidence={dynamic_confidence:.1f}% ({confidence_label}) | "
            f"Rate={speech_rate:.2f} | Pause={pause_ratio:.2f} | PitchVar={pitch_variation:.3f} | "
            f"Energy={energy_consistency:.3f} | Pronun={pronunciation_score:.3f}"
        )

        return {
            "speech_rate": round(speech_rate, 2),
            "pause_ratio": round(pause_ratio, 2),
            "pitch": round(pitch_mean, 2),
            "pitch_variance": round(pitch_variation, 4),
            "energy_consistency": round(energy_consistency, 4),
            "pronunciation_score": round(pronunciation_score, 4),
            "speech_rate_stability": round(speech_rate_stability, 4),
            "dynamic_confidence": round(dynamic_confidence, 1),
            "confidence_label": confidence_label
        }