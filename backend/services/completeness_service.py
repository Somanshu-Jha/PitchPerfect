from backend.nlp.ner_service import NERService


class CompletenessService:

    def check(self, text, semantic):

        detected = semantic.get("detected", {})

        issues = []

        if not detected.get("introduction"):
            issues.append("Introduction missing")

        if not detected.get("education"):
            issues.append("Education not mentioned")

        if not detected.get("skills"):
            issues.append("Skills missing")

        if not detected.get("career_goals"):
            issues.append("Career goals missing")

        return issues