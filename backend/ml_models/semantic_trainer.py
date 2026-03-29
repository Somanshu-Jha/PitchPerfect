# backend/ml_models/semantic_trainer.py

import os
import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import torch 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "data", "semantic_dataset_multi.csv")
model_path = os.path.join(BASE_DIR, "data", "models", "semantic_model.pkl")

os.makedirs(os.path.dirname(model_path), exist_ok=True)

print("📂 Loading dataset:", dataset_path)

df = pd.read_csv(dataset_path)

texts = df["text"].tolist()

labels = ["introduction", "education", "skills", "experience", "career_goals"]
y = df[labels]

# ---------------- GPU / CPU AUTO ----------------
import torch
device = "cpu"

print("🚀 Using device:", device)

# ------------- 🔥 CPU optimization -------------
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ------------- 🔥 Torch optimization -------------
torch.set_num_threads(8)

# ---------------- EMBEDDINGS ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

X = embedder.encode(
    texts,
    batch_size=128,
    show_progress_bar=True
)

# ---------------- TRAIN ----------------
model = MultiOutputClassifier(LogisticRegression(max_iter=2000))
model.fit(X, y)

# ---------------- SAVE ----------------
joblib.dump({
    "model": model,
    "labels": labels
}, model_path)

print("✅ Model saved:", model_path)