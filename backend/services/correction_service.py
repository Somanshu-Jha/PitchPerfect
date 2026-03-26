# -------------------- CLASS --------------------
class CorrectionService:

    # -------------------- REFINE (NO GRAMMAR CHANGE) --------------------
    def refine(self, text: str):

        text = text.strip()
        text = " ".join(text.split())

        if len(text) > 0:
            text = text[0].upper() + text[1:]

        return text