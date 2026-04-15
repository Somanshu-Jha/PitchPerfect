# backend/ml_models/dl_scoring_model.py
# Production DL Scoring Model — 10 features, 6 output heads

import os
import torch
import torch.nn as nn
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Ek specialized 'Thinking Unit' jo data ko process karta hai aur final result me
    original input wapas jod deta hai (ADDS back). Ye AI ko previous memory bhoolne nahi deta.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            # nn.LayerNorm(dim)
            # KYA HAI: Ye numbers ko normalize karta hai (healthy range me rakhta hai).
            # KYU: Taki calculations achanak se bohot badi(explode) ya choti na ho jayein.
            # IMPACT: Isko hataane se "Exploding Gradients" ho jayega aur AI crash ho sakta hai.
            nn.LayerNorm(dim),
            
            # nn.Linear(dim, dim * 2)
            # KYA HAI: Ye layer neural net ko expand karti hai. 'dim' neurons ko double (dim * 2) me convert karti hai.
            # KYU: Dimag ka daira (capacity) badhane ke liye, taki model difficult patterns ko phail kar samajh sake.
            # IMPACT: Agar (dim * 4) karden to AI zyada smart hoga par RAM aur lag time dugna badh jayega.
            nn.Linear(dim, dim * 2),
            
            # nn.GELU()
            # KYA HAI: Ye ek "Activation Function" hai. Ye filter karta hai ki neuron ka konsa signal aage jayega.
            # KYU: Iske bina AI network bilkul basic aur sedha (linear) ho jata. GELU aadi-tedhi / complex baatein aur emotion samajhne ki takat deta hai.
            nn.GELU(),
            
            # nn.Dropout(dropout)
            # KYA HAI: Training ke waqt randomly kuch percentage neurons ko "off" kar deta hai (Jaise dropout=0.1 matlab 10% band).
            # KYU: Isse AI "Ratta" (Overfitting) nai marta. Har bar naye bache k neurons se answer dhundna padta hai to AI flexible banta hai.
            # IMPACT: Agar 0.5 (50%) kardi, to AI boht late seekhega. Agar value 0 kardi, to AI sirf training set learn karlega(book pass) aur real interview me fail ho jayega.
            nn.Dropout(dropout),
            
            # nn.Linear(dim * 2, dim)
            # KYA HAI: Upar jo badi calculations hui, unko wapas chote, original shape (dim) me compress karta hai.
            nn.Linear(dim * 2, dim),
            
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # ── THE SKIP CONNECTION ──
        # Formula: Naya_Result = "Purana Input (x)" + "Naya seekhi hui baat block(x)"
        # IMPACT: Ye highway ki tarah information ko flow karne deta hai. Deep networks iske bina theek se training nahi le pate.
        return x + self.block(x)


class AdvancedScoringTransformer(nn.Module):
    """
    26 MILLION PARAMETER FFNN (Full Fusion Neural Network).
    Ye Text Embeddings (Words ka deep meaning) aur Numeric Features (Bolne ki speed, Pause, etc.) ko mila ke smart score nikalta hai.
    """
    def __init__(self, embed_dim=384, feature_dim=10):
        super().__init__()
        # input_dim: 384 (SBERT Sentence Embedding dimension) + 10 (Custom Audio & Text features) = 394 total initial datpoints.
        input_dim = embed_dim + feature_dim
        
        # hidden_dim (1024): AI ke brain ki "Width" (chaudai).
        # IMPACT:
        #   - 512: Fast hoga, low RAM use karega, but minor details (bareek ghaltiyaan) miss kar dega.
        #   - 1024: Humne ye rakha hai. Ye best 'Sweet Spot' hai 26 Million parameters bananai kelye.
        #   - 2048: Boht extra smart ho jayega pr inference slow hoga aur server RAM overload ho sakti.
        hidden_dim = 1024

        self.input_proj = nn.Sequential(
            # Stage 1: Initial 394 inputs ko 1024 k baday vector space(array) me phila deta hai.
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.backbone = nn.Sequential(
            # Stage 2: 6 "Thinking Units" layer by layer. 
            # Shruuat me Dropout 12% rakha taki basic layers overfit(ratta) na hon.
            ResidualBlock(hidden_dim, dropout=0.12),
            ResidualBlock(hidden_dim, dropout=0.12),
            
            # Beech ki thinking thori secure chahie to drop rate 10% krdiya.
            ResidualBlock(hidden_dim, dropout=0.10),
            ResidualBlock(hidden_dim, dropout=0.10),
            
            # Deep down me complex ideas judte hain, islie siraf 8% dropout hai.
            ResidualBlock(hidden_dim, dropout=0.08),
            ResidualBlock(hidden_dim, dropout=0.08),
        )

        self.head = nn.Sequential(
            # Stage 3: Abhi tak ki saari knowledge 1024 array k form me hai hmen 6 marks nikaalne hain.
            nn.LayerNorm(hidden_dim),
            
            # Thora Thora shrink karte hain. Aik dum se 1024 -> 6 krenge tou info loss hogi.
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.08),
            
            # Thora or shrink.
            nn.Linear(512, 256),
            nn.GELU(),
            
            # Final output: Clarity, Completeness, Structure, Confidence, Tech_Depth, Overall
            nn.Linear(256, 6),
        )

    def forward(self, embeddings, features):
        # 1. Text or Audio features ko aps me jod dia
        x = torch.cat([embeddings, features], dim=1)
        # 2. Bade network ki size tk laog isay
        x = self.input_proj(x)
        # 3. Main processing
        x = self.backbone(x)
        # 4. Filter krk just 6 final marks de do
        return self.head(x)

class DLScoringModel:
    """
    Inference Wrapper for the PyTorch Hybrid Scoring Model.
    Loads checkpoint, runs inference, returns clamped 1-10 scores.
    """
    def __init__(self):
        # SIKHO (Device Selection): AI code me device bohat zaroori hota hai.
        # Agar "cuda" (Nvidia GPU) available hai, to model GPU me load hoga aur calculations 100x fast hongi.
        # Agar nahi hai to "cpu" use hoga (laptop ka normal processor). 
        # Hum RTX 5070 Ti use kar rahy hain tou yeh 'cuda' lega.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AdvancedScoringTransformer(embed_dim=384, feature_dim=10).to(self.device)
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "advanced_scoring_ffnn.pth")

        try:
            # SBERT: Deep Neural Net ko words direct samajh nahi aate, usey numbers chahiye hote.
            # SBERT ek "Translator" hai jo English ko "Feature Vector" (Matrix) mein badal deta hai.
            from backend.core.model_manager import model_manager
            self.sbert = model_manager.load_embedder()
            logger.info("✅ [DLScoringModel] SBERT Embedder linked.")
        except Exception as e:
            logger.error(f"❌ [DLScoringModel] SBERT Embedder failed: {e}")

        self.is_loaded = False
        self._load_weights()

    def _load_weights(self):
        # SIKHO (Weights Loading): Bina weights ke model ek khaali dimaag (Brain with no memories) hai.
        # PyTorch ka `torch.load` weights file (.pth) ko RAM/VRAM me daalta hai.
        logger.info(f"🔍 [DLScoringModel] Searching for weights at: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                # KYA HAI: model.eval() evaluation mode on karta hai.
                # KYU: Isse Dropout aur BatchNorm turn OFF ho jate hain, taaki live test ke time results change na hon.
                self.model.eval()
                self.is_loaded = True
                logger.info("✅ [DLScoringModel] 26M Advanced weights loaded successfully.")
            except Exception as e:
                logger.error(f"❌ [DLScoringModel] Weight loading failed: {e}")
        else:
            logger.warning(f"⚠️ [DLScoringModel] No file found at {self.model_path}")

    def predict_score(self, transcript: str, feature_vector: list) -> dict:
        """
        Runs inference on 26M parameter FFNN.
        Returns mapped categories: Clarity, Completeness, Structure, Confidence, Tech_Depth, Overall.
        """
        # SIKHO: Input constraint. Agar features 10 nahi hue, to Tensor shapes mismatch hoke phatt jaenge.
        if len(feature_vector) != 10:
            logger.error(f"❌ [DLScoringModel] Feature mismatch (need 10, got {len(feature_vector)})")
            return self._heuristic_fallback(feature_vector)

        try:
            self.model.eval()
            # KYA HAI: torch.no_grad() Model ko kehta hai "Seekhna band kardo (No gradients)".
            # KYU: Live prediction (Inference) ke time gradients record nahi hote, isse GPU memory bohot bachti hai aur speed fast aati hai.
            with torch.no_grad():
                # 1. Transcript ko numbers (embeddings) mein convert karo
                embeddings_np = self.sbert.encode([transcript])
                emb_tensor = torch.tensor(embeddings_np, dtype=torch.float32).to(self.device)

                # 2. Audio features ko tensor mein daalo
                features_tensor = torch.tensor([feature_vector], dtype=torch.float32).to(self.device)
                
                # 3. Model se poocho
                outputs = self.model(emb_tensor, features_tensor)
                scores_array = outputs.cpu().numpy()[0]

            # Output hamesha 1 se 10 ke andar rakhna hai warna graph phat jaega UI me.
            def clamp(val):
                return round(max(1.0, min(10.0, float(val))), 1)

            return {
                "clarity": clamp(scores_array[0]),
                "completeness": clamp(scores_array[1]),
                "structure": clamp(scores_array[2]),
                "confidence": clamp(scores_array[3]),
                "tech_depth": clamp(scores_array[4]),
                "dl_overall": clamp(scores_array[5])
            }

        except Exception as e:
            logger.error(f"❌ [DLScoringModel] Inference crashed: {e}")
            return self._heuristic_fallback(feature_vector)

    def _heuristic_fallback(self, features: list) -> dict:
        """
        SIKHO (Fallback System): Agar Neural Net fat jaye ya file nikal jae, to AI error thanda 
        karke chalao math rules ke through (If-Else & equations se score banao). Is se app down nahi hogi.
        """
        comp = features[0] if len(features) > 0 else 0.5
        conf = features[1] if len(features) > 1 else 0.5
        tone = features[7] if len(features) > 7 else 0.5
        fluency = features[8] if len(features) > 8 else 0.5
        pronun = features[9] if len(features) > 9 else 0.5

        def clamp(val):
            return round(max(1.0, min(10.0, float(val))), 1)

        base = comp * 10
        # Equations banayi gayi hen features parameters k mutabiq.
        return {
            "clarity": clamp(base * 0.8 + pronun * 2),
            "confidence": clamp(conf * 10),
            "structure": clamp(base * 0.9),
            "tone": clamp(tone * 10),
            "fluency": clamp(fluency * 10),
            "dl_overall": clamp((base + conf * 10 + tone * 10 + fluency * 10) / 4)
        }