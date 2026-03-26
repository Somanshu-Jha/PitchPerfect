class MLScoringModel:

    def __init__(self):
        pass

    def load(self):
        pass

    def predict(self, features):
        return self.rule_based_score(features)

    def rule_based_score(self, features):

        structured = features.get("structured", {})

        # Each criterion contributes 1 point
        criteria = {
            "name_present":        "Introduction is incomplete — mention your name.",
            "degree_present":      "Mention your degree or academic background.",
            "college_present":     "Mention your college or institution.",
            "skills_present":      "Talk about your technical skills (Python, ML, Web Dev, etc.).",
            "experience_present":  "Mention internships, projects, or work experience.",
            "career_goal_present": "Share your career goals and what role you're targeting.",
        }

        score = 4  # base score out of 10
        feedback = []
        hits = 0

        for key, suggestion in criteria.items():
            if structured.get(key):
                score += 1
                hits += 1
            else:
                feedback.append(suggestion)

        # Dynamic confidence based on how many sections were covered
        if hits >= 5:
            confidence = "high"
        elif hits >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "overall_score": min(score, 10),
            "confidence": confidence,
            "hits": hits,
            "feedback": feedback
        }