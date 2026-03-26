# -------------------- IMPORTS --------------------
import pandas as pd

# -------------------- LOAD --------------------
df = pd.read_csv("data/semantic_dataset.csv")

# -------------------- CREATE MULTI-LABEL COLUMNS --------------------
labels = ["introduction", "education", "skills", "experience", "career_goals"]

for label in labels:
    df[label] = 0

# -------------------- ASSIGN VALUES --------------------
for i, row in df.iterrows():
    df.loc[i, row["label"]] = 1

# -------------------- DROP OLD COLUMN --------------------
df = df.drop(columns=["label"])

# -------------------- SAVE --------------------
df.to_csv("data/semantic_dataset_multi.csv", index=False)

print("✅ Converted to multi-label dataset")