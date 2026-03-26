import pandas as pd
import json
import random

# Load semantic dataset
df = pd.read_csv("backend/data/semantic_dataset_multi.csv")

dataset = []

for _, row in df.iterrows():
    text = row["text"]

    detected = []

    if row["introduction"] == 1:
        detected.append("introduction")
    if row["education"] == 1:
        detected.append("education")
    if row["skills"] == 1:
        detected.append("skills")
    if row["experience"] == 1:
        detected.append("experience")
    if row["career_goals"] == 1:
        detected.append("career_goals")

    # SCORING LOGIC (DATA GENERATION — NOT RUNTIME LOGIC)
    score = 3

    if "introduction" in detected:
        score += 1
    if "education" in detected:
        score += 1
    if "skills" in detected:
        score += 1
    if "experience" in detected:
        score += 1
    if "career_goals" in detected:
        score += 1

    # Add randomness (important)
    score += random.uniform(-0.5, 0.5)

    score = min(10, max(1, score))

    dataset.append({
        "text": text,
        "score": round(score, 2)
    })

# Save dataset
with open("backend/data/training_data.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ Generated {len(dataset)} samples")