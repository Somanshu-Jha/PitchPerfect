# backend/services/feedback_service.py

import logging

logger = logging.getLogger(__name__)

class FeedbackService:

    def __init__(self):
        pass

    def generate(self, text, semantic, scores, fillers):

        positives = []
        improvements = []

        # semantic = { "intent": {"detected": [...], "confidence": 0.x}, "structured": {...} }
        intent = semantic.get("intent", {})
        structured = semantic.get("structured", {})
        detected_list = intent.get("detected", [])

        # Use structured (entity extractor) as reliable detection source
        has_intro = structured.get("name_present", False) or ("introduction" in detected_list)
        has_education = structured.get("degree_present", False) or structured.get("college_present", False) or ("education" in detected_list)
        has_skills = structured.get("skills_present", False) or ("skills" in detected_list)
        has_experience = structured.get("experience_present", False) or ("experience" in detected_list)
        has_goals = structured.get("career_goal_present", False) or ("career_goals" in detected_list)

        # -------- POSITIVES --------
        if has_intro:
            positives.append("You introduced yourself clearly.")
        if has_education:
            positives.append("You mentioned your education background.")
        if has_skills:
            positives.append("You included your technical skills.")
        if has_experience:
            positives.append("You discussed your experience or projects.")
        if has_goals:
            positives.append("You shared your career goals.")
        if fillers and len(fillers) == 0:
            positives.append("You spoke without filler words. Great fluency!")

        # -------- IMPROVEMENTS --------
        if not has_intro:
            improvements.append("Start with a clear introduction — mention your name.")
        if not has_skills:
            improvements.append("Talk about your technical skills (e.g. Python, ML, Web Dev).")
        if not has_experience:
            improvements.append("Mention relevant internships, projects, or work experience.")
        if not has_goals:
            improvements.append("Explain your career goals and what you're looking for.")
        if fillers and len(fillers) > 3:
            improvements.append(f"Reduce filler words — detected {len(fillers)} filler(s) like 'um', 'uh'.")

        if not positives:
            positives.append("You completed the interview session.")

        return {
            "positives": positives,
            "improvements": improvements
        }