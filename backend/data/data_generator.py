import pandas as pd
import random
import json
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_json(filename):
    path = os.path.join(BASE_DIR, "data", filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)

            # ✅ FIX: handle dict format
            if isinstance(data, dict):
                return data.get("names", [])

            return data
    return []


names = load_json("indian_names.json") or ["Somanshu", "Akash", "Shivam"]
cities = ["Delhi", "Mumbai", "Lucknow", "Bangalore"]

degrees = ["BTech", "BCA", "MCA", "Mechanical Engineering"]
colleges = ["Chandigarh University", "Delhi University", "IIT Delhi"]

skills = ["Python", "Machine Learning", "Web Development"]
companies = ["Infosys", "TCS", "Startup"]
goals = ["become AI engineer", "work in tech", "be developer"]

# ---------------- HARD CASES ----------------

hard_cases = [
    # missing name
    {"text": "hello everyone", "name_present": 0},

    # missing degree
    {"text": "I study in Chandigarh University", "degree_present": 0, "college_present": 1},

    # missing college
    {"text": "I am doing BTech in CSE", "degree_present": 1, "college_present": 0},

    # short answers
    {"text": "want AI role", "career_goal_present": 1},

    # broken sentences
    {"text": "myself studying computer science", "degree_present": 1},

    # noisy
    {"text": "uh i want to become ai engineer", "career_goal_present": 1},
]


def generate_row():

    text_parts = []

    labels = {
        "name_present": 0,
        "degree_present": 0,
        "college_present": 0,
        "skills_present": 0,
        "experience_present": 0,
        "career_goal_present": 0,
    }

    # intro
    if random.random() > 0.3:
        name = random.choice(names)
        text_parts.append(f"My name is {name}")
        labels["name_present"] = 1

    # education
    if random.random() > 0.3:
        degree = random.choice(degrees)
        text_parts.append(f"I am studying {degree}")
        labels["degree_present"] = 1

    # college
    if random.random() > 0.5:
        college = random.choice(colleges)
        text_parts.append(f"in {college}")
        labels["college_present"] = 1

    # skills
    if random.random() > 0.4:
        skill = random.choice(skills)
        text_parts.append(f"I know {skill}")
        labels["skills_present"] = 1

    # experience
    if random.random() > 0.6:
        company = random.choice(companies)
        text_parts.append(f"I worked at {company}")
        labels["experience_present"] = 1

    # goal
    if random.random() > 0.4:
        goal = random.choice(goals)
        text_parts.append(f"My goal is to {goal}")
        labels["career_goal_present"] = 1

    text = " ".join(text_parts)

    # inject hard case
    if random.random() > 0.75:
        case = random.choice(hard_cases)
        text = case["text"]

        # override labels safely
        for k in labels:
            labels[k] = case.get(k, 0)

    return {
        "text": text,
        **labels
    }


def generate_dataset(n=5000):

    data = [generate_row() for _ in range(n)]
    df = pd.DataFrame(data).drop_duplicates()

    output_path = os.path.join(BASE_DIR, "data", "semantic_structured.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} samples")

if __name__ == "__main__":
    generate_dataset(5000)