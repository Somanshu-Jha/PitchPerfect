import numpy as np
from sentence_transformers import SentenceTransformer


class FeatureBuilder:

    def __init__(self):
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

        return np.concatenate([
            embedding,
            semantic_vector,
            audio_vector,
            filler_vector,
            completeness_vector
        ])