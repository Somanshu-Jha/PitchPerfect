from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

from backend.nlp.entity_extractor import EntityExtractor


class SemanticService:

    def __init__(self):
        # ✅ Lazy load — don't block startup, model loads on first request
        self._embedder = None
        self.model = None
        model_path = "data/models/semantic_model.pkl"

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

        # ✅ structured extractor (lightweight, loads immediately)
        self.extractor = EntityExtractor()

    @property
    def embedder(self):
        if self._embedder is None:
            print("[SemanticService] Loading SentenceTransformer model...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # ---------------- SAFE INTENT PREDICT ----------------
    def predict_intent(self, text):

        if self.model is None:
            return {"detected": [], "confidence": 0.0}

        try:
            emb = self.embedder.encode([text])
            pred = self.model.predict(emb)[0]

            labels = [
                "introduction",
                "education",
                "skills",
                "experience",
                "career_goals"
            ]

            detected = [labels[i] for i, val in enumerate(pred) if val == 1]

            return {
                "detected": detected,
                "confidence": float(np.mean(pred))
            }

        except Exception as e:
            print(f"❌ Semantic predict error: {e}")
            return {"detected": [], "confidence": 0.0}

    # ---------------- MAIN ANALYZE ----------------
    def analyze(self, text: str):

        intent = self.predict_intent(text)

        structured = self.extractor.extract(text)

        return {
            "intent": intent,
            "structured": structured
        }