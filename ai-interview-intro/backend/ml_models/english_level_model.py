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
        # Maine ye Neural Network (net) banaya hai. Ye 4 inputs lega:
        # 1. Unique words (Vocabulary)
        # 2. Avg sentence length (Complexity)
        # 3. Audio fluency
        # 4. Speech rate
        # 
        # nn.Linear(input_dim, 16): 4 inputs ko 16 alag angles se dekhega (brain ki capacity badhayega). Agar 16 ko 32 kar diya, toh model detail mein sochega par thoda slow ho sakta hai.
        # nn.ReLU(): Ye negative results ko 0 kar deta hai (Brain ko clear decision lene mein help karta hai).
        # nn.Dropout(0.2): Ye 20% neurons ko random band kar deta hai taki model "ratta" na mare (overfitting roke). Agar 0.5 kar dia, toh model theek se seekhega nahi.
        # nn.Linear(16, 3): Aakhir mein ye 16 outputs ko 3 options mein badal dega (Beginner, Intermediate, Advanced).
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3) 
        )
        
    def forward(self, x):
        # Ye pyTorch ka basic rule hai. Jo data 'x' aayega, wo seedha iss network mein pass ho jayega.
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
        # Maine device 'cpu' set kiya hai. Kyonki Whisper (Audio to Text) aur LLM GPU use kar rahe hain, 
        # toh yahan heavy load na pade, isliye ye chota net CPU par aram se chal jayega.
        self.device = torch.device("cpu") 
        self.model = LevelClassifierNN().to(self.device)
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "english_level_nn.pth")
        self.is_loaded = False
        self.levels = ["Beginner", "Intermediate", "Advanced"]
        self._load_weights()

    def _load_weights(self):
        # Ye function purane "seekhe hue" weights (.pth file) load karta hai.
        # Agar ye na karu, toh neural net bilkul naya aur dumb hoga.
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                self.model.eval() # model.eval() karna zaroori hai inference (live prediction) time par taki Dropout band ho jaye!
                self.is_loaded = True
                logger.info("✅ [EnglishLevelClassifier] Neural checkpoint loaded.")
            except Exception as e:
                logger.error(f"❌ [EnglishLevelClassifier] Corrupted checkpoint: {e}")

    def classify(self, text: str, audio_features: dict) -> str:
        # Ye mera main function hai. Audio features aur text ko combine karke final Level nikalta hai.
        words = text.split()
        if len(words) < 5:
            return "Beginner"
            
        # Step 1: Features nikalo
        # Unique ratio: Bande ke words kitne alag thay? Agar baar baar ek hi word bolega toh score kam aayega.
        unique_ratio = len(set(words)) / len(words)
        
        # Complexity: Ek sentence mai kitne words the? Lamba sentence = Achhi English.
        sentences = [s for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
        avg_sentence_len = len(words) / max(1, len(sentences))
        complexity_norm = min(1.0, avg_sentence_len / 20.0) # Assume 20 words per sentence is max (1.0)
        
        # Fluency aur Speech rate audio_features se lenge
        fluency_norm = audio_features.get("dynamic_confidence", 50.0) / 100.0
        speech_rate_norm = min(1.0, audio_features.get("speech_rate", 2.0) / 3.0)
        
        features = [unique_ratio, complexity_norm, fluency_norm, speech_rate_norm]
        
        # Step 2: DL Model se prediction karao
        if self.is_loaded:
            try:
                self.model.eval()
                with torch.no_grad(): # torch.no_grad() isliye kyuki hum naya seekh nahi rahe, bs test kar rahe hain. Isse memory save hoti hai.
                    t_feat = torch.tensor([features], dtype=torch.float32).to(self.device)
                    logits = self.model(t_feat) # Logits = Raw scores 3 categories ke liye
                    pred_idx = torch.argmax(logits, dim=1).item() # argmax = Jo sabse bada score hai uski array index return karega (0, 1, or 2)
                    return self.levels[pred_idx]
            except Exception as e:
                logger.warning(f"DL Classification failed, falling back to heuristics: {e}")
                
        # Statistical Fallback: Agar upar wala AI model load nahi hua (file gayab hai), toh math based logic use karo!
        score = unique_ratio * 0.3 + complexity_norm * 0.3 + fluency_norm * 0.4
        if score > 0.65:
            return "Advanced"
        elif score > 0.4:
            return "Intermediate"
        else:
            return "Beginner"
