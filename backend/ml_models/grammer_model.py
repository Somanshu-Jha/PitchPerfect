from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

class LanguageQualityModel:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.good_patterns = [
            "My name is",
            "I am",
            "Hello my name is"
        ]

        self.bad_patterns = [
            "Myself",
            "Myself XYZ",
            "Myself is"
        ]

    def evaluate(self, sentence):

        sentence_embedding = self.model.encode([sentence])

        good_score = max(
            cosine_similarity(
                sentence_embedding,
                self.model.encode([g])
            )[0][0]
            for g in self.good_patterns
        )

        bad_score = max(
            cosine_similarity(
                sentence_embedding,
                self.model.encode([b])
            )[0][0]
            for b in self.bad_patterns
        )

        return good_score - bad_score