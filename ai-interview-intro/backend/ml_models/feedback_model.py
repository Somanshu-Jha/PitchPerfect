import json
import joblib
import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from backend.ml_models.feature_builder import FeatureBuilder

with open("backend/data/feedback_dataset.json") as f:
    data = json.load(f)

builder = FeatureBuilder()

X = []
y = []

labels = [
    "missing_name",
    "missing_degree",
    "missing_skills",
    "missing_experience",
    "missing_goals"
]

for item in data:
    text = item["text"]

    semantic = {"detected": []}
    audio = {"speech_rate": 120, "pitch": 150, "pause_ratio": 0.3}
    fillers = []
    completeness = {}

    features = builder.build(text, semantic, audio, fillers, completeness)

    target = [1 if l in item["feedback"] else 0 for l in labels]

    X.append(features)
    y.append(target)

# SIKHO (Random Forest & Multi-Labeling): 
# 'RandomForestClassifier' bahut saaray chotay-chotay Decison Trees ka majmoa(group) hai.
# Yahan MultiOutput lga hai jis ka mtlb feedback me siraf 1 bemari nahi, balky multiple bemariyan 
# pakri ja sakti hyn ek sth (For example: Missing Name AND Missing Degree dono ek sath True asakte hyn uski English dekh kr).
model = MultiOutputClassifier(RandomForestClassifier())
model.fit(X, y)

joblib.dump(model, "backend/ml_models/feedback.pkl")

print("✅ Feedback model trained")