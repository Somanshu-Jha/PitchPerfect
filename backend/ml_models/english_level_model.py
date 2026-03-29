# =====================================================================
# ENGLISH LEVEL CLASSIFIER — Adaptive DL-based Language Assessment
# =====================================================================
# Classifies users into [Beginner, Intermediate, Advanced] based on:
# - Vocabulary richness (unique word ratio)
# - Sentence complexity (avg words per sentence)
# - Audio fluency (dynamic_confidence from audio analysis)
# - Speech rate (normalized words per second)
#
# The classification directly influences LLM feedback tone:
# - Advanced → technical, critical coaching
# - Beginner → simplified, encouraging guidance
# - Intermediate → standard professional coaching
#
# Runs on CPU to avoid GPU contention. Has heuristic fallback.
# =====================================================================

import torch
import torch.nn as nn
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class LevelClassifierNN(nn.Module):
    def __init__(self, input_dim=4):
        super(LevelClassifierNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3) # 3 classes: Beginner, Intermediate, Advanced
        )
        
    def forward(self, x):
        return self.net(x)

class EnglishLevelClassifier:
    """
    Adaptive English Level Classifier.
    Uses DL to classify [Beginner, Intermediate, Advanced] based on 4 features.
    
    The output MODIFIES feedback behavior:
    - 'Advanced' → LLM highlights strengths + technical depth
    - 'Beginner' → LLM simplifies language + provides basic guidance
    - 'Intermediate' → standard professional coaching
    """
    
    def __init__(self):
        self.device = torch.device("cpu") # Force CPU to avoid GPU locking with Whisper
        self.model = LevelClassifierNN().to(self.device)
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "english_level_nn.pth")
        self.is_loaded = False
        self.levels = ["Beginner", "Intermediate", "Advanced"]
        self._load_weights()

    def _load_weights(self):
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                self.is_loaded = True
                logger.info("✅ [EnglishLevelClassifier] Neural checkpoint loaded.")
            except Exception as e:
                logger.error(f"❌ [EnglishLevelClassifier] Corrupted checkpoint: {e}")

    def classify(self, text: str, audio_features: dict) -> str:
        words = text.split()
        if len(words) < 5:
            return "Beginner"
            
        # Extract Linguistic Features
        unique_ratio = len(set(words)) / len(words)
        
        sentences = [s for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
        avg_sentence_len = len(words) / max(1, len(sentences))
        complexity_norm = min(1.0, avg_sentence_len / 20.0)
        
        fluency_norm = audio_features.get("dynamic_confidence", 50.0) / 100.0
        speech_rate_norm = min(1.0, audio_features.get("speech_rate", 2.0) / 3.0)
        
        features = [unique_ratio, complexity_norm, fluency_norm, speech_rate_norm]
        
        if self.is_loaded:
            try:
                self.model.eval()
                with torch.no_grad():
                    t_feat = torch.tensor([features], dtype=torch.float32).to(self.device)
                    logits = self.model(t_feat)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    return self.levels[pred_idx]
            except Exception as e:
                logger.warning(f"DL Classification failed, falling back to heuristics: {e}")
                
        # Statistical Fallback
        score = unique_ratio * 0.3 + complexity_norm * 0.3 + fluency_norm * 0.4
        if score > 0.65:
            return "Advanced"
        elif score > 0.4:
            return "Intermediate"
        else:
            return "Beginner"
