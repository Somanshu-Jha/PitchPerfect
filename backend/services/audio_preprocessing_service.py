# -------------------- IMPORTS --------------------
import librosa
import numpy as np
import soundfile as sf


class AudioPreprocessingService:

    def process(self, input_path: str, output_path: str):

        y, sr = librosa.load(input_path, sr=16000)

        y = librosa.util.normalize(y)

        noise = np.mean(y[:1000])
        y = y - noise

        y = np.clip(y, -1.0, 1.0)

        sf.write(output_path, y, 16000)

        return output_path