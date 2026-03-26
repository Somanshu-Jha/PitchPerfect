# backend/ml_models/semantic_model.py

import os
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "semantic_model.pkl"
)


class SemanticModel:

    def __init__(self):
        self.model_path = MODEL_PATH
        self.model = None

        # ✅ ensure model directory exists
        Path(os.path.dirname(self.model_path)).mkdir(parents=True, exist_ok=True)

    # ---------------- LOAD ----------------
    def load(self):

        print("📦 Loading semantic model from:", self.model_path)

        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info("✅ Semantic model loaded")

            except Exception as e:
                logger.error(f"❌ Model load failed: {e}")
                self.model = None

        else:
            logger.warning("⚠️ Semantic model not found")

    # ---------------- PREDICT ----------------
    def predict(self, text):

        if self.model is None:
            return None, 0.0

        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

            X = embedder.encode([text])

            model = self.model["model"]
            labels = self.model["labels"]

            preds = model.predict(X)[0]

            # ✅ FIX: convert numpy → python int
            detected = {
                label: int(pred)
                for label, pred in zip(labels, preds)
            }

            confidence = float(sum(detected.values()) / len(detected))

            return detected, confidence

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None, 0.0