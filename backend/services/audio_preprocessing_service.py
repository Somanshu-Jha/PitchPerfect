# =====================================================================
# AUDIO PREPROCESSING SERVICE — Production Audio Normalization
# =====================================================================
# Converts any audio format → clean 16kHz mono WAV for Whisper.
# 
# Key improvements:
#   - UUID-based temp files (no race conditions)
#   - Gentler frequency filter (60-8000 Hz, wider speech range)
#   - Moderate volume boost (1.15x vs 1.3x) for consistency
#   - Deterministic output for same input audio
# =====================================================================

import librosa
import numpy as np
import soundfile as sf
import subprocess
import os
import uuid

# Dynamically add extracted FFmpeg to PATH so running uvicorn picks it up without restart
FFMPEG_BIN_DIR = r"C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab\ffmpeg_extracted\ffmpeg-master-latest-win64-gpl-shared\bin"
if FFMPEG_BIN_DIR not in os.environ["PATH"]:
    os.environ["PATH"] = os.environ["PATH"] + os.pathsep + FFMPEG_BIN_DIR


class AudioPreprocessingService:

    def process(self, input_path: str, output_path: str) -> tuple[str, dict]:
        """
        Returns: (output_path, audio_flags)
        audio_flags: {'clipping': bool, 'distorted': bool, 'low_energy': bool}
        
        Uses UUID-based temp files to prevent race conditions when
        multiple requests run simultaneously.
        """
        audio_flags = {"clipping": False, "distorted": False, "low_energy": False}

        # Generate unique temp file to prevent race conditions
        unique_id = uuid.uuid4().hex[:12]
        temp_wav = f"{input_path}_{unique_id}_converted.wav"
        
        try:
            # Gentler filter chain: wider frequency range preserves more speech detail
            # highpass=60 (vs 80): captures lower male voice fundamentals
            # lowpass=8000 (vs 7500): preserves sibilant consonants for clarity
            # afftdn=nf=-30 (vs -25): lighter noise reduction to avoid speech distortion
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-af", "highpass=f=60,lowpass=f=8000,afftdn=nf=-30",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_wav
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Use stable soundfile-based loading instead of librosa's audioread fallback
            y, sr = sf.read(temp_wav)

        except Exception as e:
            # Fallback if FFmpeg fails
            print(f"⚠️ FFmpeg conversion failed: {e}. Falling back to soundfile direct load.")
            y, sr = sf.read(input_path)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        finally:
            # Always clean up temp file
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

        # ── AUDIO EDGE DETECTION ──────────────────────────────────────────────────
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak >= 0.98:
            audio_flags["clipping"] = True
            print(f"⚠️ [AudioFlags] Clipping detected (peak={peak:.3f}).")

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        if zcr > 0.35:
            audio_flags["distorted"] = True
            print(f"⚠️ [AudioFlags] High ZCR detected ({zcr:.3f}) — possible distortion.")

        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
        if rms < 0.01:
            audio_flags["low_energy"] = True
            print(f"⚠️ [AudioFlags] Low energy audio detected (rms={rms:.4f}).")

        # ── NORMALIZATION ─────────────────────────────────────────────────────────
        y = librosa.util.normalize(y)
        
        # DC offset removal
        noise = np.mean(y[:1000]) if len(y) >= 1000 else 0.0
        y = y - noise
        
        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y, top_db=25)
        
        # Moderate volume boost (1.15x vs 1.3x for consistency)
        y = y * 1.15
        y = np.clip(y, -1.0, 1.0)

        # Write to the output path provided by caller
        sf.write(output_path, y, 16000)

        return output_path, audio_flags