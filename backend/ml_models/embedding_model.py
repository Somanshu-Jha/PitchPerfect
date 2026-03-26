# -------------------- IMPORTS --------------------
from sentence_transformers import SentenceTransformer


# -------------------- CLASS --------------------
class EmbeddingModel:

    def __init__(self):
        """
        #---- Load model once (IMPORTANT for performance)
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def encode(self, text: str):
        """
        #---- Convert text → embedding vector
        """
        return self.model.encode(text)

    def similarity(self, vec1, vec2):
        """
        #---- Cosine similarity
        """
        return float((vec1 @ vec2) / ((vec1**2).sum()**0.5 * (vec2**2).sum()**0.5))