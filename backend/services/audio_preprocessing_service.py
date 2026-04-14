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
        Role: Audio Cleaner.
        Logic: Jab bacha bolta hai to uski awaz me pankhay ka shor (noise) ya volume kam-zyada ho sakti hai. 
        Ye file Whisper (ASR) ke pas jaane se pehle har awaz ko "Standard Studio" quality me set krti hai.
        
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
            # SIKHO (Frequencies & Deep Learning Info):
            # 'highpass=60': 60Hz se neechy ki awazein (jaise dewar ke pechay truck guzrana) nikal deta hai. Male awaz 85Hz se shuru hoti hai.
            # 'lowpass=8000': 8000Hz k upper awaz kat deta hai kyu k sibilants (s aur sh sounds) cover hojati hn idhr.
            # 'afftdn=nf=-30': Ye ek intelligent noise filter hai jo background hissaying sound kam karta hai. (-30 mtlb moderate noise cancel).
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
        # SIKHO (Clipping): Audio Volume -1 se 1 ke darmian hoti hai. Agar awaz 1.0 boundary ko hit karey iska 
        # matlab mic "phat" gaya hai. (Peak > 0.98 signifies this). Ai phir isko minus marks deti hy confidence may.
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak >= 0.98:
            audio_flags["clipping"] = True
            print(f"⚠️ [AudioFlags] Clipping detected (peak={peak:.3f}).")

        # ZCR (Zero Crossing Rate) = Awaz ki frequency kitni tezi se positive-to-negative me jaa rahi hai.
        # High ZCR ka matlab mechanical noise ya background shor boht tez hai.
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        if zcr > 0.35:
            audio_flags["distorted"] = True
            print(f"⚠️ [AudioFlags] High ZCR detected ({zcr:.3f}) — possible distortion.")

        # SIKHO (RMS Energy): Root Mean Square audio ki 'Energy' ya loudness ko measure karta hai.
        # Agar RMS boht kam (< 0.01) hai matlab bacha khamosh betha hai ya bht slowly fussfussa raha hai.
        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
        if rms < 0.01:
            audio_flags["low_energy"] = True
            print(f"⚠️ [AudioFlags] Low energy audio detected (rms={rms:.4f}).")

        # ── NORMALIZATION ─────────────────────────────────────────────────────────
        # librosa.util.normalize(y) math operations karke highest volume point ko 1.0 (Maximum) pe set kardeta hai,
        # baqi sab bhi ussi proportion se badh jatay hn. AI ko clean sunehne kelye.
        y = librosa.util.normalize(y)
        
        # DC offset removal: Shruuat ka initial background shor (pehle 1000 pieces of noise) sub me se Minus krdo!
        noise = np.mean(y[:1000]) if len(y) >= 1000 else 0.0
        y = y - noise
        
        # SIKHO (Decibels - dB limits): 
        # Yeh shru aur akhri ka completely khamosh (Silence) hissa trim kardeta hai. 
        # top_db=25 matlab reference volume level k -25dB se neeche wala sab trim. 
        # Agar 10 kardo, tou normal words bhi trim hjynge (Cut jayenge).
        y, _ = librosa.effects.trim(y, top_db=25)
        
        # SIKHO (Volume Boost): Output awaz ko 15% (1.15x) boost krdo ta k clear ho. 
        # Agar 2.0x (200%) krdenge tou clip hoky kaan faad awaz auegi.
        y = y * 1.15
        
        # np.clip sure kerta hai k multiplier se agar awaz 1.0 (Max Limit) exceed kargayi tou usey wapis boundary me lock kry.
        y = np.clip(y, -1.0, 1.0)

        # Write to the output path provided by caller
        sf.write(output_path, y, 16000)

        return output_path, audio_flags