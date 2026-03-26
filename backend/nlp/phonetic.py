# -------------------- IMPORTS --------------------
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


# -------------------- CLASS --------------------
class PhoneticService:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------- DYNAMIC CORRECTION --------------------
    def correct(self, word: str, context: str):

        # Generate embedding for word and context
        word_emb = self.model.encode(word, convert_to_tensor=True)
        context_emb = self.model.encode(context, convert_to_tensor=True)

        similarity = util.cos_sim(word_emb, context_emb).item()

        # Phonetic similarity check (self-consistency)
        phonetic_score = fuzz.ratio(word.lower(), context.lower())

        # Decision logic (no dataset required)
        if similarity > 0.6 and phonetic_score > 70:
            return word.capitalize()

        return word