# =====================================================================
# DL SCORING MODEL — Advanced Transformer-based Interview Scorer
# =====================================================================
# Architecture: SBERT embeddings (384d) + audio/semantic features (7d)
# fed through a text encoder + fusion network → 6 output scores.
#
# Scores: [clarity, completeness, structure, confidence, technical_depth, overall]
# All clamped to 1.0-10.0 range.
#
# Runs on CPU to avoid GPU contention with Whisper/LLM pipeline.
# Falls back to heuristic scoring if no trained weights (.pth) available.
# =====================================================================

import os
import torch
import torch.nn as nn
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class AdvancedScoringTransformer(nn.Module):
    """
    Advanced PyTorch Network for interview scoring.
    Predicts: [clarity, completeness, structure, confidence, technical_depth, overall_score]
    
    Architecture:
    - Text encoder: Linear(384→128) + LayerNorm + ReLU + Dropout(0.2)
    - Fusion: Concatenate encoded text (128d) + raw features (7d) = 135d
    - Dense: 135 → 64 → 32 → 6 output nodes
    
    Input:
    - embeddings: SBERT text embeddings (384 dimensions)
    - features: audio/semantic feature tensor (7 dimensions)
      [completeness, confidence, length_norm, diversity, filler_ratio, rag_flag, coherence]
    """
    def __init__(self, embed_dim=384, feature_dim=7):
        super(AdvancedScoringTransformer, self).__init__()
        
        # Dense layer over embeddings
        self.text_encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion of text features and numeric features
        self.fusion = nn.Sequential(
            nn.Linear(128 + feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6) # 6 outputs
        )
        
    def forward(self, embeddings, features):
        text_encoded = self.text_encoder(embeddings)
        combined = torch.cat((text_encoded, features), dim=1)
        return self.fusion(combined)


class DLScoringModel:
    """
    Inference Wrapper for the PyTorch Advanced Scoring Model.
    Designed for strict sub-10ms processing avoiding GPU bottlenecks.
    """
    def __init__(self):
        # Bind to CPU explicitly to prevent blocking Whisper or wasting VRAM context transfers
        self.device = torch.device("cpu")
        self.model = AdvancedScoringTransformer().to(self.device)
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "advanced_scoring_ffnn.pth")
        
        try:
            self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info("✅ [DLScoringModel] SentenceTransformer loaded.")
        except Exception as e:
            logger.error(f"❌ [DLScoringModel] Failed to load SentenceTransformer: {e}")
            self.sbert = None
            
        self.is_loaded = False
        self._load_weights()

    def _load_weights(self):
        """Loads the .pth checkpoint if it exists. Reverts to safe fallback zero-weights otherwise."""
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device, weights_only=True)
                )
                self.model.eval()
                self.is_loaded = True
                logger.info("✅ [DLScoringModel] Checkpoint loaded successfully.")
            except Exception as e:
                logger.error(f"❌ [DLScoringModel] Corrupt checkpoint: {e}")
        else:
            logger.warning("⚠️ [DLScoringModel] No .pth checkpoint found. Inference will use uncalibrated weights.")

    def predict_score(self, transcript: str, feature_vector: list) -> dict:
        """
        Runs sub-1ms inference.
        Returns explicit dictionary of 6 attributes.
        """
        if len(feature_vector) != 7:
            logger.error(f"❌ [DLScoringModel] Invalid dim: {len(feature_vector)}! Expected 7.")
            # Fallback 
            return {"clarity": 6.0, "completeness": 6.0, "structure": 6.0, "confidence": 6.0, "technical_depth": 6.0, "dl_overall": 6.0}
            
        if self.sbert is None:
            return {"clarity": 6.0, "completeness": 6.0, "structure": 6.0, "confidence": 6.0, "technical_depth": 6.0, "dl_overall": 6.0}

        try:
            self.model.eval()
            with torch.no_grad():
                # 1. Text embedding
                embedding = self.sbert.encode([transcript], convert_to_tensor=True, device=self.device)
                
                # 2. Structural/Audio features
                features_tensor = torch.tensor([feature_vector], dtype=torch.float32).to(self.device)
                
                # 3. Forward Pass
                scores_tensor = self.model(embedding, features_tensor)
                scores_array = scores_tensor.numpy()[0]
                
            # Clamp outputs rigorously to 1-10 scale
            def clamp(val):
                return round(max(1.0, min(10.0, float(val))), 1)
                
            # If model is entirely uncalibrated (garbage outputs near 0), force to safe middle ground heuristics
            if not self.is_loaded:
                # Mock weights simulation until trained
                mock_base = float(feature_vector[0] * 10) # Completeness maps to base
                return {
                    "clarity": clamp(mock_base * 0.9),
                    "completeness": clamp(mock_base),
                    "structure": clamp(mock_base * 0.95),
                    "confidence": clamp(feature_vector[1] * 10),
                    "technical_depth": clamp(mock_base * 0.8),
                    "dl_overall": clamp((mock_base + feature_vector[1]*10) / 2)
                }

            return {
                "clarity": clamp(scores_array[0]),
                "completeness": clamp(scores_array[1]),
                "structure": clamp(scores_array[2]),
                "confidence": clamp(scores_array[3]),
                "technical_depth": clamp(scores_array[4]),
                "dl_overall": clamp(scores_array[5])
            }
            
        except Exception as e:
            logger.error(f"❌ [DLScoringModel] Inference crashed: {e}")
            return {"clarity": 6.0, "completeness": 6.0, "structure": 6.0, "confidence": 6.0, "technical_depth": 6.0, "dl_overall": 6.0}