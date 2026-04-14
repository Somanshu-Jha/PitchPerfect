# backend/scoring_inference.py
import os
import torch
import torch.nn as nn
import logging
from .ml_models.dl_scoring_model import HybridDLScoringModel
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

class ScoringInference:
    """
    THE LEFT BRAIN: Fast, Mathematical Scoring.
    Loads the 26M parameter FFNN and runs instant inference.
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default path to our newly trained 'Advanced' weights
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "data", "models", "advanced_scoring_ffnn.pth")
        
        self.model = HybridDLScoringModel().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        self._load_weights(model_path)

    def _load_weights(self, path):
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info(f"✅ [ScoringInference] Loaded 'Left Brain' from {path}")
            except Exception as e:
                logger.error(f"❌ [ScoringInference] Failed to load weights: {e}")
        else:
            logger.warning(f"⚠️ [ScoringInference] Weights not found at {path}. Model is uncalibrated!")

    def get_scores(self, transcript: str, numeric_features: list) -> dict:
        """
        Runs the FFNN to generate the 6 categories of scores.
        Inputs: 
          - transcript: The raw text of the interview answer.
          - numeric_features: 10-dim list [WPM, Tone, Fluency, etc.]
        """
        if len(numeric_features) != 10:
            logger.error(f"❌ [ScoringInference] Feature vector must have 10 dimensions. Got {len(numeric_features)}")
            return self._get_default_scores()

        try:
            with torch.no_grad():
                encoded = self.tokenizer(
                    transcript,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                features_tensor = torch.tensor([numeric_features], dtype=torch.float32).to(self.device)
                
                outputs = self.model(encoded["input_ids"], encoded["attention_mask"], features_tensor)
                logits = outputs["logits"].cpu().numpy()[0]
                
            def clamp(val):
                return round(max(0.0, min(10.0, float(val))), 1)

            results = {
                "clarity": clamp(logits[0]),
                "completeness": clamp(logits[1]),
                "structure": clamp(logits[2]),
                "confidence": clamp(logits[3]),
                "tech_depth": clamp(logits[4]),
                "overall": clamp(logits[5])
            }
            logger.info(f"🌀 [ScoringInference] FFNN Output: {results['overall']}/10")
            return results
            
        except Exception as e:
            logger.error(f"❌ [ScoringInference] Scoring failed: {e}")
            return self._get_default_scores()

    def _get_default_scores(self):
        return {k: 5.0 for k in ["clarity", "completeness", "structure", "confidence", "tech_depth", "overall"]}

if __name__ == "__main__":
    # Internal test run
    engine = ScoringInference()
    dummy_text = "I have a keen interest in Python and I have built several web scrapers."
    dummy_features = [120, 0.8, 0.9, 50, 0.7, 0.8, 0.6, 0.8, 0.9, 0.8] # 10 features
    print(engine.get_scores(dummy_text, dummy_features))
