import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# load dataset
df = pd.read_csv("backend/data/semantic_structured.csv")

texts = df["text"].tolist()

labels = df[
    [
        "name_present",
        "degree_present",
        "college_present",
        "skills_present",
        "experience_present",
        "career_goal_present",
    ]
].values

# embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2", device = "cpu")
torch.set_num_threads(8)
X = embedder.encode(texts, batch_size=128, show_progress_bar=True)

# model
model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model.fit(X, labels)

# save
with open("backend/data/models/structured_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Structured model trained and saved")