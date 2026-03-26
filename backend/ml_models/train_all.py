import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.ml_models.scoring_model import MLScoringModel

#---- Load dataset
df = pd.read_csv("backend/data/training_data.csv")

#---- Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#---- Generate embeddings
embeddings = embedder.encode(df["text"].tolist())

#---- Build features
X = []
for i, emb in enumerate(embeddings):
    row = df.iloc[i]

    features = list(emb) + [
        row["speech_rate"],
        row["pause_ratio"],
        row["pitch"],
        row["filler_count"]
    ]

    X.append(features)

y = df["score"]

#---- Train model
model = MLScoringModel()
model.train(X, y)