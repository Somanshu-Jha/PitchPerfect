# -------------------- IMPORTS --------------------
from transformers import pipeline


# -------------------- CLASS --------------------
class LLMService:

    def __init__(self):
        self._model = None

    # -------------------- LAZY LOAD --------------------
    def _load_model(self):
        if self._model is None:
            self._model = pipeline(
                "text2text-generation",   # ✅ CORRECT
                model="google/flan-t5-large",  # large model
                device=0
            )

    # -------------------- CORRECT TEXT --------------------
    def correct(self, text: str):

        self._load_model()

        prompt = f"""
        Correct the following interview response:
        - Fix grammar
        - Fix spelling
        - Correct names if obvious
        Keep meaning same.

        Text: {text}
        """

        try:
            result = self._model(prompt, max_length=128)
            return result[0]["generated_text"]

        except Exception as e:
            print(f"LLM Error: {e}")
            return text