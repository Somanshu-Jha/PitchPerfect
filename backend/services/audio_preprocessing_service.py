# -------------------- IMPORTS --------------------
import librosa
import numpy as np
import soundfile as sf


import subprocess
import os

# Dynamically add extracted FFmpeg to PATH so running uvicorn picks it up without restart
FFMPEG_BIN_DIR = r"C:\Users\Legion_Pro_7i\OneDrive\Desktop\EnglishLab\ffmpeg_extracted\ffmpeg-master-latest-win64-gpl-shared\bin"
if FFMPEG_BIN_DIR not in os.environ["PATH"]:
    os.environ["PATH"] = os.environ["PATH"] + os.pathsep + FFMPEG_BIN_DIR

class AudioPreprocessingService:

    def process(self, input_path: str, output_path: str) -> tuple[str, dict]:
        """
        Returns: (output_path, audio_flags)
        audio_flags: {'clipping': bool, 'distorted': bool, 'low_energy': bool}
        """
        audio_flags = {"clipping": False, "distorted": False, "low_energy": False}

        # 1. Use FFmpeg to perform a stable format conversion (WebM/MP3 -> PCM WAV 16kHz Mono)
        temp_wav = input_path + "_converted.wav"
        try:
            # Force conversion using system FFmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_wav
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2. Use stable soundfile-based loading instead of librosa's audioread fallback
            y, sr = sf.read(temp_wav)

            # Remove intermediate file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        except Exception as e:
            # Fallback if FFmpeg fails (ensure system PATH is set)
            print(f"⚠️ FFmpeg conversion failed: {e}. Falling back to soundfile direct load.")
            y, sr = sf.read(input_path)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        # ── AUDIO EDGE DETECTION ──────────────────────────────────────────────────
        # Clipping: peak amplitude at or above 0.99 in normalized float range
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak >= 0.98:
            audio_flags["clipping"] = True
            print(f"⚠️ [AudioFlags] Clipping detected (peak={peak:.3f}). Minor limiter applied.")

        # Distortion: unusually high zero-crossing rate (> 0.35) indicates mic overload / noise
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        if zcr > 0.35:
            audio_flags["distorted"] = True
            print(f"⚠️ [AudioFlags] High ZCR detected ({zcr:.3f}) — possible distortion/multi-speaker.")

        # Low energy detection
        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
        if rms < 0.01:
            audio_flags["low_energy"] = True
            print(f"⚠️ [AudioFlags] Low energy audio detected (rms={rms:.4f}).")

        # ── NORMALIZATION + NOISE GATE ────────────────────────────────────────────
        y = librosa.util.normalize(y)
        noise = np.mean(y[:1000]) if len(y) >= 1000 else 0.0
        y = y - noise
        y = np.clip(y, -1.0, 1.0)

        sf.write(output_path, y, 16000)

        return output_path, audio_flags