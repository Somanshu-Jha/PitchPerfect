import spacy


class NERService:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")

    def extract_name(self, text):
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text

        return None