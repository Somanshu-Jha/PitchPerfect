import numpy as np
from sentence_transformers import SentenceTransformer


class FeatureBuilder:

    def __init__(self):
        # SIKHO (Embedder Architecture): 'SentenceTransformer' text ko asool(math) se 384 numbers(vectors) me tod deta hai.
        # "all-MiniLM-L6-v2": Ye 6 layer ka small model hai, fast CPU pr easily 50ms me kam karta hai. 
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    def build(
        self,
        text,
        semantic,
        audio_features,
        fillers,
        completeness
    ):
        # ---------------- EMBEDDING ----------------
        # SIKHO: Text ko numbers ki ek line(384 dimensions ki array) me convert karna 'Encoding' kehlata hai.
        # Agar text hoga "I am good", array bnegi [0.12, -0.45, 0.99...]
        embedding = self.embedder.encode(text)

        # ---------------- SEMANTIC VECTOR ----------------
        labels = ["introduction", "education", "skills", "experience", "career_goals"]

        semantic_vector = np.array([
            1 if l in semantic.get("detected", []) else 0
            for l in labels
        ])

        # ---------------- AUDIO ----------------
        audio_vector = np.array([
            audio_features.get("speech_rate", 0),
            audio_features.get("pitch", 0),
            audio_features.get("pause_ratio", 0)
        ])

        # ---------------- FILLERS ----------------
        filler_vector = np.array([
            len(fillers),
            len(fillers) / max(len(text.split()), 1)
        ])

        # ---------------- COMPLETENESS ----------------
        completeness_vector = np.array([
            0 if completeness.get("missing_name") else 1,
            0 if completeness.get("missing_degree") else 1,
            0 if completeness.get("missing_skills") else 1,
            0 if completeness.get("missing_experience") else 1,
            0 if completeness.get("missing_goals") else 1
        ])

        # SIKHO (Neural Network Inputs): 
        # Deep learning model texts/strings nahi samjhte, wo bs eik lambi see list of floats (1D Array) dekhte hain.
        # Yahan hum Embedding(384) + Semantic(5) + Audio(3) + Filler(2) + Completeness(5) ko jod kar
        # Total Shape = 399 numbers ka ek single vector / array bana k forward karte hain.
        return np.concatenate([
            embedding,
            semantic_vector,
            audio_vector,
            filler_vector,
            completeness_vector
        ])