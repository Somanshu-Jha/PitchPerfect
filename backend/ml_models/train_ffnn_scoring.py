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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Import the EXACT architecture from dl_scoring_model ──────────────
# This ensures weights are compatible
from backend.ml_models.dl_scoring_model import dScoringTransformeAdvancer


def compute_features_from_text(text: str) -> list:
    """
    Compute the same 7 features that ScoringService uses,
    given only text (no audio). For training purposes.
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
    Generate 6 training target scores [clarity, completeness, structure, confidence, technical_depth, overall]
    from text features. All scores on 1-10 scale.
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
    logger.info("=" * 60)
    logger.info("🚀 TRAINING AdvancedScoringTransformer (FFNN)")
    logger.info("=" * 60)

    # ── Load SBERT ───────────────────────────────────────────────────
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
                texts.extend(df3["text"].dropna().tolist())
                logger.info(f"✅ Loaded {len(df3)} rows from scoring_dataset.csv")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load scoring_dataset.csv: {e}")

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
        features = compute_features_from_text(text)
        targets = compute_target_scores(text, features)
        all_features.append(features)
        all_targets.append(targets)
        all_embeddings.append(embeddings_np[i])

    # Convert to tensors
    X_embed = torch.tensor(np.array(all_embeddings), dtype=torch.float32)
    X_feat = torch.tensor(np.array(all_features), dtype=torch.float32)
    Y = torch.tensor(np.array(all_targets), dtype=torch.float32)

    logger.info(f"✅ Tensors prepared: embeddings={X_embed.shape}, features={X_feat.shape}, targets={Y.shape}")

    # ── Initialize model ─────────────────────────────────────────────
    model = AdvancedScoringTransformer(embed_dim=384, feature_dim=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────────
    EPOCHS = 50
    BATCH_SIZE = 32
    N = len(texts)

    logger.info(f"🏋️ Training for {EPOCHS} epochs, batch_size={BATCH_SIZE}, samples={N}")

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

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

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

    logger.info("=" * 60)
    logger.info("🎉 TRAINING COMPLETE — Model ready for production")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
