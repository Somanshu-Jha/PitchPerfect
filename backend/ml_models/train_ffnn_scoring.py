# =====================================================================
# TRAIN FFNN SCORING MODEL — Generate advanced_scoring_ffnn.pth
# =====================================================================
# Trains the AdvancedScoringTransformer (SBERT 384d + 7 features → 6 scores)
# using existing dataset CSVs with synthetic score targets.
#
# Usage:
#   cd ai-interview-intro
#   python -m backend.ml_models.train_ffnn_scoring
#
# Output: backend/data/models/advanced_scoring_ffnn.pth
# =====================================================================

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import multiprocessing as mp
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── THE RESIDUAL BLOCK (The "Brain's Highway") ──────────────────────────
# Deep networks often get "stuck" or "confused" as information travels deep.
# This block uses a "Skip Connection" (Residual) to keep the signal strong.

class ResidualBlock(nn.Module):
    """
    Ek specialized 'Thinking Unit' jo data ko process karta hai aur final result me
    original input wapas jod deta hai (ADDS back). Ye AI ko previous memory bhoolne nahi deta.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # nn.Sequential: A helper that stacks different layers in order.
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


def compute_features_from_text(text: str) -> list:
    """
    Sikho: "Feature Extraction" (AI ko clues dena).
    Ye function sirf text padh kar 7 clues nikalta hai (Jaise kitne lambe sentence hain, words kitne alag hain).
    Neural Network inhi 7 clues (features) ko use karke apna score banayega.
    """
    words = text.split()
    total_words = max(len(words), 1)

    # Feature 1: Completeness proxy (longer text → more topics covered)
    # We'll use a heuristic based on keyword presence
    keywords = {
        "greetings": ["hello", "hi", "good morning", "hey", "greetings", "namaste"],
        "name": ["name", "i am", "i'm", "my name", "myself", "call me"],
        "education": ["degree", "university", "college", "studied", "graduate", "school", "education", "btech", "mtech", "bsc", "msc", "engineering"],
        "skills": ["skills", "proficient", "experience in", "python", "java", "machine learning", "data", "programming", "coding"],
        "strengths": ["strength", "strong", "good at", "excel", "ability", "capable"],
        "interests": ["interest", "passionate", "love", "enjoy", "curious", "fascinated"],
        "qualities": ["team", "leadership", "communicate", "creative", "innovative", "dedicated", "hardworking"],
        "experience": ["work", "project", "intern", "company", "role", "position", "years", "experience"],
        "career_goals": ["goal", "future", "aim", "aspire", "plan", "dream", "career", "grow", "contribute"]
    }
    text_lower = text.lower()
    hits = sum(1 for category_words in keywords.values() if any(kw in text_lower for kw in category_words))
    completeness = hits / len(keywords)

    # Feature 2: Confidence proxy (longer, more fluent → higher)
    confidence = min(1.0, total_words / 80.0) * 0.7 + 0.3

    # Feature 3: Normalized length
    length_norm = min(total_words / 150.0, 1.0)

    # Feature 4: Lexical diversity
    unique_words = len(set(w.lower() for w in words))
    diversity = unique_words / total_words

    # Feature 5: Filler ratio
    filler_ratio = text_lower.count(" um ") / total_words

    # Feature 6: RAG improvement (neutral)
    rag_improvement = 0.5

    # Feature 7: Coherence proxy (variety of sentence lengths)
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sentences) > 1:
        sent_lens = [len(s.split()) for s in sentences]
        coherence = max(0.0, min(1.0, 1.0 - np.std(sent_lens) / (np.mean(sent_lens) + 1e-8)))
    else:
        coherence = 0.5

    return [completeness, confidence, length_norm, diversity, filler_ratio, rag_improvement, coherence]


def compute_target_scores(text: str, features: list) -> list:
    """
    Sikho: "Target / Label Generation".
    Neural Network ko training ke time "Sahi Jawab" (Target Score) chahiye hota hai taaki wo seekh sake.
    Ye function mathematical rules laga kar ek nakli (synthetic) 'Sahi Jawab' banata hai.
    Example: Agar bachhe ne jyada Filler ("Um, Ah") use kiye hain, toh usko Clarity mein kam mark milenge.
    """
    completeness, confidence, length_norm, diversity, filler_ratio, rag_improvement, coherence = features
    words = text.split()
    total_words = max(len(words), 1)

    # Clarity: diversity + low fillers
    clarity = (diversity * 0.6 + (1.0 - filler_ratio) * 0.4) * 10.0

    # Completeness: direct from feature
    comp_score = completeness * 10.0

    # Structure: coherence + length
    structure = (coherence * 0.6 + length_norm * 0.4) * 10.0

    # Confidence: from feature
    conf_score = confidence * 10.0

    # Technical depth: length + completeness + diversity
    tech_depth = (length_norm * 0.3 + completeness * 0.4 + diversity * 0.3) * 10.0

    # Overall: weighted blend
    overall = (clarity * 0.15 + comp_score * 0.25 + structure * 0.15 + conf_score * 0.15 + tech_depth * 0.15 + length_norm * 10.0 * 0.15)

    # Clamp all to 1.0-10.0
    def clamp(v): return max(1.0, min(10.0, v))

    return [clamp(clarity), clamp(comp_score), clamp(structure), clamp(conf_score), clamp(tech_depth), clamp(overall)]


def main():
    # Ye humari training ki factory hai. Yahan se data uthayenge aur AI ko padhayenge.
    logger.info("=" * 60)
    logger.info("🚀 TRAINING AdvancedScoringTransformer (FFNN)")
    logger.info("=" * 60)

    # ── Load SBERT ───────────────────────────────────────────────────
    # SBERT: Ye Text ko array of numbers (Embeddings) mein badalta hai (384 dimensions).
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    logger.info("✅ SBERT loaded (all-MiniLM-L6-v2)")

    # ── Load training data ───────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")

    texts = []

    # Source 1: semantic_dataset_multi.csv (has 'text' column)
    csv_path = os.path.join(data_dir, "semantic_dataset_multi.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            if "text" in df.columns:
                texts.extend(df["text"].dropna().tolist())
                logger.info(f"✅ Loaded {len(df)} rows from semantic_dataset_multi.csv")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load semantic_dataset_multi.csv: {e}")

    # Source 2: training_data.csv
    csv_path2 = os.path.join(data_dir, "training_data.csv")
    if os.path.exists(csv_path2):
        try:
            df2 = pd.read_csv(csv_path2, on_bad_lines='skip')
            if "text" in df2.columns:
                texts.extend(df2["text"].dropna().tolist())
                logger.info(f"✅ Loaded {len(df2)} rows from training_data.csv")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load training_data.csv: {e}")

    # Source 3: scoring_dataset.csv
    csv_path3 = os.path.join(data_dir, "scoring_dataset.csv")
    if os.path.exists(csv_path3):
        try:
            df3 = pd.read_csv(csv_path3, on_bad_lines='skip')
            if "text" in df3.columns:
                for t in df3["text"].dropna().tolist():
                    texts.append(t)
                logger.info(f"✅ Loaded {len(df3)} rows from scoring_dataset.csv")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load scoring_dataset.csv: {e}")

    # Source (NEW Golden Truth): deepseek_10k_distilled_scores.json
    # Ultra-smart targets via RAG
    distilled_path = os.path.join(data_dir, "deepseek_10k_distilled_scores.json")
    distilled_labels = {}
    if os.path.exists(distilled_path):
        import json
        try:
            with open(distilled_path, "r", encoding="utf-8") as f:
                distilled_data = json.load(f)
            for item in distilled_data:
                # Store the exact audio features generated by the augmentor
                feat = item["features"]
                features_array = [
                    feat.get("completeness", 0.5),
                    feat.get("asr_confidence", 0.9),
                    min(len(item["text"].split())/150.0, 1.0), # length_norm
                    feat.get("diversity", 0.5),
                    feat.get("filler_ratio", 0.0),
                    feat.get("rag_improvement", 0.5),
                    feat.get("coherence", 0.5),
                    feat.get("tone_expressiveness", 0.5),
                    feat.get("fluency_score", 0.5),
                    feat.get("pronunciation_score", 0.5)
                ]
                
                distilled_labels[item["text"]] = {
                    "scores": [
                        item["scores"]["clarity"],
                        item["scores"]["completeness"],
                        item["scores"]["structure"],
                        item["scores"]["confidence"],
                        item["scores"]["technical_depth"],
                        item["scores"]["overall"]
                    ],
                    "features": features_array
                }
                texts.append(item["text"])
            logger.info(f"🧠 Loaded {len(distilled_labels)} DISTILLED TRUE RAG LABELS from deepseek_10k_distilled_scores.json")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load deepseek_10k_distilled_scores.json: {e}")

    if not texts:
        logger.error("❌ No training data found! Cannot train model.")
        sys.exit(1)

    # Deduplicate and filter
    texts = list(set(t for t in texts if isinstance(t, str) and len(t.strip()) > 10))
    logger.info(f"📊 Total unique training samples: {len(texts)}")

    # ── Compute features and targets ─────────────────────────────────
    all_features = []
    all_targets = []
    all_embeddings = []

    logger.info("🔄 Computing features and SBERT embeddings...")

    # Batch encode for speed
    embeddings_np = sbert.encode(texts, batch_size=64, show_progress_bar=True)

    for i, text in enumerate(texts):
        # KEY UPGRADE: The Intelligence System
        if text in distilled_labels:
            targets = distilled_labels[text]["scores"]
            features = distilled_labels[text]["features"]
        else:
            # Fallback legacy 7-dim logic zero-padded to 10
            f7 = compute_features_from_text(text)
            targets = compute_target_scores(text, f7)
            features = f7[:2] + [f7[2], f7[3], f7[4], f7[5], f7[6], 0.5, 0.5, 0.5] # Pad out the audio metrics
            
        all_features.append(features)
        all_targets.append(targets)
        all_embeddings.append(embeddings_np[i])

    # Convert to tensors
    X_embed = torch.tensor(np.array(all_embeddings), dtype=torch.float32)
    X_feat = torch.tensor(np.array(all_features), dtype=torch.float32)
    Y = torch.tensor(np.array(all_targets), dtype=torch.float32)

    logger.info(f"✅ Tensors prepared: embeddings={X_embed.shape}, features={X_feat.shape}, targets={Y.shape}")

    # ── Initialize model ─────────────────────────────────────────────
    model = AdvancedScoringTransformer(embed_dim=384, feature_dim=10)
    
    # torch.optim.AdamW
    # WHAT: The 'Coach' or 'Optimizer'. It looks at mistakes and fixes weights.
    # WHY: AdamW is the industry standard—it includes 'Weight Decay' which
    # keeps the weights from growing too large (and becoming too 'stubborn').
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    # nn.SmoothL1Loss
    # WHAT: The 'Scoring Rubric'. Measures how far predicted score is from target.
    # WHY: Much better than MSE (Mean Squared Error) because it doesn't overreact 
    # if one data point is slightly weird (Outliers).
    criterion = nn.SmoothL1Loss()

    # ── One-Click Auto-Visualizer (Internal Queue Mode) ───────────
    from backend.ml_models.training_visualizer import ui_entry_point
    viz_queue = mp.Queue()
    viz_proc = mp.Process(target=ui_entry_point, args=(viz_queue,))
    viz_proc.start()
    
    logger.info("📡 Auto-launched Neon Glow Dashboard Process")

    # ── Training loop (Padhai Shuru) ────────────────────────────────────────────────
    EPOCHS = 200 # EPOCHS: AI ko poora syllabus kitni baar padhana hai. Agar 5 kardia toh AI bahut dumb rahega. Agar 2000 kardia toh 'Ratta' maar lega.
    BATCH_SIZE = 32 # BATCH_SIZE: Ek baar mein kitne examples dikhane hain. Kam memory hai toh 16 kar do, fast chahiye toh 64.
    N = len(texts)
    total_steps = EPOCHS * ((N + BATCH_SIZE - 1) // BATCH_SIZE)
    # OneCycleLR
    # WHAT: A dynamic 'Speed Limit' for the optimizer. 
    # It starts slow → speeds up in the middle → slows down at the end for precision.
    # IMPACT: Crucial for hitting R² > 99%. Without this, the AI might 'overshoot' the best weights.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos"
    )

    logger.info(f"🏋️ Training for {EPOCHS} epochs, batch_size={BATCH_SIZE}, samples={N}")
    logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        # Shuffle
        perm = torch.randperm(N)
        X_embed_s = X_embed[perm]
        X_feat_s = X_feat[perm]
        Y_s = Y[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            batch_embed = X_embed_s[start:end]
            batch_feat = X_feat_s[start:end]
            batch_y = Y_s[start:end]

            optimizer.zero_grad()
            pred = model(batch_embed, batch_feat)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR steps per batch

            epoch_loss += loss.item()
            n_batches += 1

            # Pulse (Internal queue for auto-launch dashboard)
            if n_batches % 10 == 0:
                viz_queue.put({
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "lr": scheduler.get_last_lr()[0]
                })

        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # ── Live Visualization Update ────────────────────────────────

        # ── Live Visualization Update (Full Pulse) ───────────────────
        topology = {"names": [], "mags": []}
        with torch.no_grad():
            for name, p in list(model.named_parameters())[:12]:
                if p.grad is not None and "weight" in name:
                    topology["names"].append(name.split(".")[-1])
                    topology["mags"].append(float(p.grad.abs().mean().item()))

        # Detailed Samples for Ultimate Dashboard (150 samples for High-Res 3D Analysis)
        p_sample = pred[:150].detach().cpu().numpy().tolist()
        t_sample = batch_y[:150].detach().cpu().numpy().tolist()
        w_sample = list(model.parameters())[0].detach().cpu().numpy().flatten()[:1000].tolist()
        e_sample = batch_embed[:150].detach().cpu().numpy().tolist() if hasattr(batch_embed, 'cpu') else []

        viz_queue.put({
            "epoch": epoch,
            "loss": float(avg_loss),
            "lr": float(scheduler.get_last_lr()[0]),
            "topology": topology,
            "weights": {"w": w_sample},
            "embeddings": e_sample,
            "preds": p_sample,
            "targets": t_sample,
            "maes": {h: float(np.mean(np.abs(pred[:50, i].detach().cpu().numpy() - batch_y[:50, i].detach().cpu().numpy()))) 
                     for i, h in enumerate(["Clarity", "Completeness", "Structure", "Confidence", "Tech_Depth", "Overall"])}
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"   Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # ── Save model ──────────────────────────────────────────────────
    model_dir = os.path.join(data_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "advanced_scoring_ffnn.pth")

    model.eval()
    torch.save(model.state_dict(), output_path)
    logger.info(f"✅ Model saved to: {output_path}")
    logger.info(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # ── Quick validation ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        sample_pred = model(X_embed[:5], X_feat[:5])
        logger.info(f"✅ Validation predictions (first 5):")
        for i in range(5):
            scores = [round(float(v), 1) for v in sample_pred[i]]
            logger.info(f"   Sample {i+1}: {scores}")

    # ── Final Summary Report ──
    report_content = f"""# Final Training Summary Report
**Model:** AdvancedScoringTransformer (FFNN)
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Architecture Stats
- **Total Parameters:** {sum(p.numel() for p in model.parameters()):,}
- **Total Biases:** {sum(p.numel() for n, p in model.named_parameters() if 'bias' in n):,}
- **Trainable Parameters:** {sum(p.numel() for p in model.parameters() if p.requires_grad):,}

## Training Results
- **Best Loss:** {best_loss:.5f}
- **Epochs:** {EPOCHS}

## Per-Head Final MAE
"""
    model.eval()
    with torch.no_grad():
        final_preds = model(X_embed, X_feat).numpy()
    final_targets = Y.numpy()
    
    heads = ["Clarity", "Completeness", "Structure", "Confidence", "Tech_Depth", "Overall"]
    for i, head in enumerate(heads):
        mae = float(np.mean(np.abs(final_preds[:, i] - final_targets[:, i])))
        report_content += f"- **{head}:** {mae:.4f}\n"

    # Save to disk
    report_path = os.path.join(data_dir, "training_summary_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    
    logger.info("=" * 60)
    logger.info(f"✅ Report saved to: {report_path}")
    logger.info("🎉 TRAINING COMPLETE — Model ready for production")
    logger.info("=" * 60)

    # ── Signal Training Completion ───────────────────────────────────
    if viz_proc.is_alive():
        logger.info("🔔 Training complete. Closing dashboard in 5s...")
        time.sleep(5)
        viz_proc.terminate()


if __name__ == "__main__":
    main()
