# -------------------- IMPORTS --------------------
from faster_whisper import WhisperModel


# -------------------- CLASS --------------------
class TranscriptionService:

    def __init__(self):
        # Lazy-load: don't load Whisper on startup (it's 1.5GB, blocks the server)
        self._model = None
        print("✅ TranscriptionService ready (model will load on first request)")

    def _get_model(self):
        if self._model is None:
            print("[TranscriptionService] Loading Whisper tiny model...")
            try:
                self._model = WhisperModel(
                    "tiny",
                    device="cpu",
                    compute_type="int8"
                )
                print("✅ Whisper tiny model loaded successfully")
            except Exception as e:
                print(f"❌ Whisper model load failed: {e}")
                raise
        return self._model

    # -------------------- TRANSCRIBE --------------------
    def transcribe(self, audio_path: str) -> str:
        """
        #---- Convert audio → text
        # Uses beam search for better accuracy
        # Uses context prompt for Indian accent handling
        """

        model = self._get_model()

        segments, _ = model.transcribe(
            audio_path,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt="""
            This is a professional interview.
            Speaker may have Indian accent.
            Names like  Somanshu,Indian universities, and technical terms may appear.
            """
        )

        #---- Combine all segments into final text
        text = " ".join([seg.text for seg in segments])

        return text.strip()