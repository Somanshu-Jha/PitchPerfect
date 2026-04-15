# backend/nlp/entity_extractor.py

import re

class EntityExtractor:

    def __init__(self):
        # patterns
        self.name_patterns = [
            r"\bmy name is ([a-z]+)\b",
            r"\bi am ([a-z]+)\b",
            r"\bthis is ([a-z]+)\b"
        ]

        self.degree_keywords = [
            "btech", "b.tech", "bca", "mca",
            "engineering", "computer science", "cse"
        ]

        self.college_keywords = [
            "university", "college", "institute", "school"
        ]

        self.skill_keywords = [
            "python", "machine learning", "deep learning",
            "web development", "data structures", "ai"
        ]

        self.experience_keywords = [
            "internship", "worked", "experience",
            "company", "project"
        ]

        self.goal_keywords = [
            "goal", "aim", "aspire", "want to",
            "dream", "become"
        ]

    def detect_name(self, text):
        for pattern in self.name_patterns:
            match = re.search(pattern, text)
            if match:
                return True
        return False

    def detect_degree(self, text):
        return any(k in text for k in self.degree_keywords)

    def detect_college(self, text):
        return any(k in text for k in self.college_keywords)

    def detect_skills(self, text):
        return any(k in text for k in self.skill_keywords)

    def detect_experience(self, text):
        return any(k in text for k in self.experience_keywords)

    def detect_goal(self, text):
        return any(k in text for k in self.goal_keywords)

    def extract(self, text: str):
        text = text.lower()

        return {
            "name_present": self.detect_name(text),
            "degree_present": self.detect_degree(text),
            "college_present": self.detect_college(text),
            "skills_present": self.detect_skills(text),
            "experience_present": self.detect_experience(text),
            "career_goal_present": self.detect_goal(text),
        }