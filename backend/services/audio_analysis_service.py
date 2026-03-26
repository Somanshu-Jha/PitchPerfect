# -------------------- IMPORTS --------------------
import librosa
import numpy as np


# -------------------- CLASS --------------------
class AudioAnalysisService:

    # -------------------- EXTRACT FEATURES --------------------
    def extract(self, audio_path: str):

        y, sr = librosa.load(audio_path, sr=16000)

        duration = librosa.get_duration(y=y, sr=sr)

        # ---------- SPEECH RATE ----------
        words_estimate = len(y) / sr / 0.4
        speech_rate = words_estimate / duration if duration > 0 else 0

        # ---------- PAUSE RATIO ----------
        silence = np.sum(np.abs(y) < 0.01)
        pause_ratio = silence / len(y)

        # ---------- PITCH ----------
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))

        return {
            "speech_rate": round(speech_rate, 2),
            "pause_ratio": round(float(pause_ratio), 2),
            "pitch": round(float(pitch), 2)
        }