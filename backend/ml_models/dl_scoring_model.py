# backend/ml_models/dl_scoring_model.py
# Production DL Scoring Model — 10 features, 6 output heads

import os
import torch
import torch.nn as nn
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class HybridDLScoringModel(nn.Module):
    """
    Multi-head PyTorch Neural Network.
    Inputs: DistilBERT embeddings (768-dim) + 10 numeric features.
    Outputs: [clarity, confidence, structure, tone, fluency, overall]
    """
    def __init__(self, model_name="distilbert-base-uncased", num_numeric_features=10):
        super().__init__()
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # Deeper MLP for better multi-head separation
        self.shared_mlp = nn.Sequential(
            nn.Linear(768 + num_numeric_features, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # 6 output heads: clarity, confidence, structure, tone, fluency, overall
        self.head = nn.Linear(128, 6)

    def forward(self, input_ids, attention_mask, numeric_features, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, numeric_features], dim=1)
        shared = self.shared_mlp(combined)
        scores = self.head(shared)

        loss = None
        if labels is not None:
            loss_fct = nn.SmoothL1Loss()  # More robust than MSE for score regression
            loss = loss_fct(scores, labels)

        return {"loss": loss, "logits": scores}


class DLScoringModel:
    """
    Inference Wrapper for the PyTorch Hybrid Scoring Model.
    Loads checkpoint, runs inference, returns clamped 1-10 scores.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = HybridDLScoringModel().to(self.device)
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "dl_scoring", "pytorch_model.bin")

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models", "dl_scoring")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
            logger.info("✅ [DLScoringModel] Tokenizer loaded from cache.")
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.is_loaded = False
        self._load_weights()

    def _load_weights(self):
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_loaded = True
                logger.info("✅ [DLScoringModel] Checkpoint loaded successfully (6-head model).")
            except Exception as e:
                logger.warning(f"⚠️ [DLScoringModel] Checkpoint incompatible (likely architecture change): {e}")
                logger.warning("⚠️ [DLScoringModel] Using uncalibrated weights — please retrain.")
        else:
            logger.warning("⚠️ [DLScoringModel] No .bin checkpoint found. Will use heuristic fallback.")

    def predict_score(self, transcript: str, feature_vector: list) -> dict:
        """
        Runs inference. Expects 10-dim feature vector.
        Returns dict with clarity, confidence, structure, tone, fluency, dl_overall.
        """
        if len(feature_vector) != 10:
            logger.error(f"❌ [DLScoringModel] Invalid dim: {len(feature_vector)}! Expected 10.")
            return self._heuristic_fallback(feature_vector)

        try:
            self.model.eval()
            with torch.no_grad():
                encoding = self.tokenizer(
                    transcript,
                    truncation=True,
                    max_length=128,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)

                features_tensor = torch.tensor([feature_vector], dtype=torch.float32).to(self.device)
                outputs = self.model(encoding["input_ids"], encoding["attention_mask"], features_tensor)
                scores_array = outputs["logits"].numpy()[0]

            def clamp(val):
                return round(max(1.0, min(10.0, float(val))), 1)

            if not self.is_loaded:
                return self._heuristic_fallback(feature_vector)

            return {
                "clarity": clamp(scores_array[0]),
                "confidence": clamp(scores_array[1]),
                "structure": clamp(scores_array[2]),
                "tone": clamp(scores_array[3]),
                "fluency": clamp(scores_array[4]),
                "dl_overall": clamp(scores_array[5])
            }

        except Exception as e:
            logger.error(f"❌ [DLScoringModel] Inference crashed: {e}")
            return self._heuristic_fallback(feature_vector)

    def _heuristic_fallback(self, features: list) -> dict:
        """Rule-based fallback when model isn't loaded."""
        comp = features[0] if len(features) > 0 else 0.5
        conf = features[1] if len(features) > 1 else 0.5
        tone = features[7] if len(features) > 7 else 0.5
        fluency = features[8] if len(features) > 8 else 0.5
        pronun = features[9] if len(features) > 9 else 0.5

        def clamp(val):
            return round(max(1.0, min(10.0, float(val))), 1)

        base = comp * 10
        return {
            "clarity": clamp(base * 0.8 + pronun * 2),
            "confidence": clamp(conf * 10),
            "structure": clamp(base * 0.9),
            "tone": clamp(tone * 10),
            "fluency": clamp(fluency * 10),
            "dl_overall": clamp((base + conf * 10 + tone * 10 + fluency * 10) / 4)
        }