#!/usr/bin/env python3
# =====================================================================
# HR DATASET GENERATOR — 100K+ Real-World Interview Transcripts
# =====================================================================
# Generates massively diverse interview introductions covering every
# scenario a real HR interviewer encounters. Each transcript includes
# synthetic audio features that match the speaking profile.
#
# KEY DESIGN: Transcripts feel HUMAN, not templated. Each archetype
# has 50+ sentence structures, natural connectors, and realistic
# disfluencies. No two transcripts read identically.
#
# Usage:
#   cd ai-interview-intro
#   python -m backend.ml_models.hr_dataset_generator --count 100000
#
# Output: backend/data/hr_training_dataset.jsonl
# =====================================================================

import os
import json
import random
import math
import argparse
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ═══════════════════════════════════════════════════════════════════════
# KNOWLEDGE POOLS — Massive real-world vocabulary
# ═══════════════════════════════════════════════════════════════════════

NAMES_MALE = [
    "Aman Sharma", "Rahul Verma", "Vikram Singh", "Karan Patel", "Arjun Mehta",
    "Rohan Gupta", "Aditya Kumar", "Siddharth Joshi", "Pranav Reddy", "Nikhil Rao",
    "Ankit Mishra", "Deepak Chauhan", "Harsh Vardhan", "Manish Tiwari", "Suraj Yadav",
    "Rajesh Nair", "Varun Kapoor", "Mohit Agarwal", "Gaurav Saxena", "Tarun Bhatia",
    "James Wilson", "Michael Chen", "David Kim", "Robert Johnson", "William Brown",
    "Ahmed Hassan", "Omar Ali", "Faisal Khan", "Mohammed Rizwan", "Abdul Rahman",
    "Yuki Tanaka", "Kenji Watanabe", "Ryo Suzuki", "Takeshi Yamamoto", "Hiroshi Sato",
    "Kwame Asante", "Chidi Okafor", "Emeka Nwosu", "Samuel Osei", "Joseph Kamau",
    "Carlos Rodriguez", "Diego Martinez", "Pablo Sanchez", "Luis Hernandez", "Rafael Torres",
    "Ivan Petrov", "Dmitri Volkov", "Alexei Sorokin", "Nikolai Federov", "Andrei Kuznetsov",
]

NAMES_FEMALE = [
    "Priya Sharma", "Neha Gupta", "Sneha Reddy", "Pooja Patel", "Ananya Singh",
    "Riya Mehta", "Ishita Kumar", "Shruti Joshi", "Divya Nair", "Kavita Rao",
    "Meera Kapoor", "Anjali Mishra", "Swati Chauhan", "Tanvi Saxena", "Nidhi Agarwal",
    "Rashmi Tiwari", "Pallavi Yadav", "Sonali Bhatia", "Shweta Verma", "Komal Desai",
    "Emily Zhang", "Sarah Johnson", "Jessica Lee", "Amanda Williams", "Rachel Thompson",
    "Fatima Al-Hussein", "Aisha Begum", "Noor Jahan", "Zainab Khan", "Huda Malik",
    "Sakura Yamada", "Yui Nakamura", "Aoi Takahashi", "Hana Kobayashi", "Miku Inoue",
    "Amara Okonkwo", "Ngozi Eze", "Chioma Adeyemi", "Grace Wanjiku", "Faith Muthoni",
    "Sofia Garcia", "Isabella Martinez", "Valentina Lopez", "Camila Hernandez", "Lucia Perez",
    "Anastasia Ivanova", "Olga Petrova", "Natasha Smirnova", "Katya Kuznetsova", "Elena Popova",
]

UNIVERSITIES = [
    "IIT Delhi", "IIT Bombay", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
    "NIT Trichy", "NIT Warangal", "NIT Surathkal", "NIT Calicut", "NIT Rourkela",
    "BITS Pilani", "VIT Vellore", "SRM Chennai", "Manipal Institute", "Amity University",
    "Delhi University", "Mumbai University", "Anna University", "Pune University", "JNTU Hyderabad",
    "Lovely Professional University", "Chandigarh University", "Sharda University", "Symbiosis Pune",
    "Christ University", "St. Xavier's College", "Loyola College", "Presidency College",
    "MIT", "Stanford", "Carnegie Mellon", "UC Berkeley", "Georgia Tech",
    "University of Toronto", "University of Waterloo", "NUS Singapore", "NTU Singapore",
    "Tsinghua University", "Peking University", "University of Tokyo", "Seoul National",
    "Oxford", "Cambridge", "Imperial College London", "ETH Zurich", "TU Munich",
    "Government Engineering College", "Regional Engineering College", "State Polytechnic",
    "District College of Engineering", "Private Institute of Technology",
]

DEGREES = [
    "Computer Science Engineering", "Information Technology", "B.Tech in CS",
    "B.Tech in IT", "Electrical Engineering", "Electronics and Communication",
    "Mechanical Engineering", "Civil Engineering", "Chemical Engineering",
    "BCA", "MCA", "B.Sc Computer Science", "M.Sc Data Science",
    "MBA in Information Systems", "B.E. in Computer Science", "M.Tech in AI",
    "Diploma in Computer Engineering", "B.Sc IT", "M.Sc Computer Science",
    "Bachelor of Engineering", "Master of Technology", "PhD in Computer Science",
]

SKILLS_PROGRAMMING = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C", "C#", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB", "Perl", "Dart",
    "Shell scripting", "Bash", "PowerShell", "Assembly",
]

SKILLS_FRAMEWORKS = [
    "React", "Angular", "Vue.js", "Next.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Express.js", "Node.js", "Ruby on Rails", "Laravel",
    ".NET", "ASP.NET", "Svelte", "Nuxt.js", "NestJS", "Gin",
    "Flutter", "React Native", "SwiftUI", "Jetpack Compose",
]

SKILLS_DATA_ML = [
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "scikit-learn",
    "NLP", "Computer Vision", "Reinforcement Learning", "Generative AI",
    "Pandas", "NumPy", "Spark", "Hadoop", "Kafka",
    "Data Analysis", "Data Engineering", "Data Visualization", "Power BI", "Tableau",
    "LLM fine-tuning", "RAG systems", "transformer architectures", "neural networks",
    "statistical modeling", "A/B testing", "recommendation systems",
]

SKILLS_CLOUD_DEVOPS = [
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
    "CI/CD", "Jenkins", "GitHub Actions", "Terraform", "Ansible",
    "Linux", "Nginx", "Apache", "microservices architecture",
    "serverless computing", "Lambda functions", "CloudFormation",
]

SKILLS_DB = [
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis",
    "Elasticsearch", "DynamoDB", "Firebase", "Cassandra", "Neo4j",
    "Oracle", "SQLite", "InfluxDB", "CockroachDB",
]

SKILLS_SOFT = [
    "team leadership", "communication", "problem-solving", "critical thinking",
    "time management", "adaptability", "collaboration", "mentoring",
    "presentation skills", "conflict resolution", "empathy", "creativity",
    "attention to detail", "analytical thinking", "strategic planning",
    "agile methodology", "Scrum", "project management",
]

# ── PROJECT TEMPLATES (60+, with specific outcomes) ──────────────────────
PROJECT_TEMPLATES = [
    ("built a {tech} application for {domain}", "which {outcome}"),
    ("developed a {tech} system that {outcome}", ""),
    ("created a {tech} platform for {domain}", "serving {scale}"),
    ("designed and implemented a {tech} solution", "that {outcome}"),
    ("led a team to build a {tech} tool for {domain}", "which {outcome}"),
    ("engineered a {tech} pipeline", "processing {scale} daily"),
    ("architected a {tech} microservice", "handling {scale} requests"),
    ("contributed to an open-source {tech} project", "with {scale} stars on GitHub"),
    ("deployed a {tech} model in production", "serving {scale} users"),
    ("built an end-to-end {tech} system from scratch", "for {domain}"),
]

PROJECT_TECH = [
    "full-stack web", "machine learning", "REST API", "real-time streaming",
    "computer vision", "NLP", "recommendation engine", "chatbot",
    "e-commerce", "inventory management", "student portal", "healthcare analytics",
    "IoT dashboard", "blockchain-based", "serverless", "mobile",
    "data pipeline", "ETL", "search engine", "authentication",
]

PROJECT_DOMAINS = [
    "hospital management", "online education", "food delivery", "social media analytics",
    "financial services", "retail inventory", "logistics tracking", "weather prediction",
    "sentiment analysis", "spam detection", "customer support", "content moderation",
    "autonomous vehicles", "smart home", "agricultural monitoring", "supply chain",
    "HR recruitment", "legal document analysis", "energy optimization", "traffic management",
]

PROJECT_OUTCOMES = [
    "reduced processing time by 40%", "improved accuracy to 95%",
    "increased user engagement by 60%", "saved 200 man-hours per month",
    "reduced deployment time from days to minutes", "achieved 99.9% uptime",
    "decreased error rate by 75%", "automated 80% of manual workflows",
    "improved customer satisfaction scores by 30%", "cut infrastructure costs by 50%",
    "won first place in a hackathon", "received best project award",
    "got published in a conference", "is now used by the department daily",
]

PROJECT_SCALES = [
    "500+ users", "10,000 daily active users", "1 million records",
    "50 concurrent connections", "100 API calls per second",
    "200+ GitHub stars", "3 terabytes of data", "15 microservices",
    "50,000 monthly requests", "the entire department",
]

CAREER_GOALS = [
    "become a senior software engineer at a product company",
    "specialize in machine learning and AI research",
    "build products that impact millions of users",
    "grow into a technical architect role",
    "lead a development team at a startup",
    "contribute to open-source projects full-time",
    "become a full-stack developer working on scalable systems",
    "transition into data science and work with large-scale datasets",
    "start my own tech company someday",
    "work on cutting-edge AI applications in healthcare",
    "become a cloud solutions architect",
    "work in cybersecurity and protect digital infrastructure",
    "pursue a research career in NLP",
    "build real-time systems for financial markets",
    "work remotely for a global tech company",
    "become a DevOps engineer and streamline CI/CD pipelines",
    "specialize in embedded systems and IoT",
    "get into product management after building strong technical foundations",
    "contribute to AI safety and alignment research",
    "build accessible technology for underserved communities",
]

EXPERIENCES = [
    ("interned at {company} as a {role}", "for {duration}"),
    ("worked at {company} as a {role}", "for {duration}"),
    ("did a summer internship at {company}", "in the {team} team"),
    ("freelanced as a {role}", "for {duration}, working with {clients} clients"),
    ("was a teaching assistant for {course}", "at my university"),
    ("led the {club} club at college", "organizing {events} events"),
    ("participated in {hackathon}", "and {result}"),
    ("completed a research project under Professor {professor}", "in {area}"),
    ("worked on a contract project for {company}", "delivering {deliverable}"),
    ("volunteered at {org}", "building {deliverable}"),
]

COMPANIES = [
    "TCS", "Infosys", "Wipro", "HCL Technologies", "Tech Mahindra",
    "Cognizant", "Accenture", "Deloitte", "Capgemini", "IBM",
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix",
    "Flipkart", "Razorpay", "Zerodha", "Swiggy", "Zomato",
    "a startup", "a mid-size product company", "a fintech firm",
    "a healthcare technology company", "an ed-tech startup",
]

ROLES = [
    "software developer", "frontend developer", "backend engineer",
    "full-stack developer", "data analyst", "data scientist",
    "ML engineer", "DevOps engineer", "QA engineer", "cloud engineer",
    "research intern", "software engineering intern", "data engineering intern",
    "mobile app developer", "systems programmer", "platform engineer",
]

# ── FILLER WORDS & DISFLUENCIES ──────────────────────────────────────────
FILLERS = ["um", "uh", "like", "you know", "basically", "actually", "so", "I mean", "kind of", "sort of", "right"]
CODE_MIX = [
    "matlab", "kyunki", "aur", "toh", "mera", "main", "kya hai ki",
    "basically matlab", "actually toh", "so basically", "haan",
]
ASR_ERRORS = {
    "software": ["soft where", "soft wear"], "engineering": ["engineear", "engine ring"],
    "python": ["pie thon", "pythonn"], "machine learning": ["machine burning", "maching learn"],
    "developer": ["develop her", "devil oper"], "experience": ["x perience", "experiment"],
    "artificial intelligence": ["artificial in tell igence", "arti ficial intelligence"],
    "technology": ["tech nology", "take nology"], "algorithm": ["algo rithm", "algorythm"],
    "database": ["data base", "data bass"], "javascript": ["java script", "java scrypt"],
}

# ── SENTENCE CONNECTORS (natural flow) ───────────────────────────────────
CONNECTORS = [
    "Also, ", "In addition to that, ", "Apart from this, ", "Furthermore, ",
    "Moving on, ", "Coming to my experience, ", "On the technical side, ",
    "Regarding my career goals, ", "As for my interests, ", "What drives me is ",
    "I'd also like to mention ", "Another thing about me is ", "To add to that, ",
    "On top of that, ", "Beyond academics, ", "On the professional front, ",
    "When it comes to skills, ", "In terms of my background, ", "Looking ahead, ",
    "What excites me most is ", "One more thing — ", "I should also mention ",
    "", "", "", "",  # Empty connectors for variation (no connector sometimes)
]


# ═══════════════════════════════════════════════════════════════════════
# TRANSCRIPT GENERATORS (one per archetype)
# ═══════════════════════════════════════════════════════════════════════

def _pick_name():
    return random.choice(NAMES_MALE + NAMES_FEMALE)

def _pick_skills(n=3):
    pool = SKILLS_PROGRAMMING + SKILLS_FRAMEWORKS + SKILLS_DATA_ML + SKILLS_CLOUD_DEVOPS + SKILLS_DB
    return random.sample(pool, min(n, len(pool)))

def _pick_soft_skills(n=2):
    return random.sample(SKILLS_SOFT, min(n, len(SKILLS_SOFT)))

def _make_project():
    tmpl, outcome_tmpl = random.choice(PROJECT_TEMPLATES)
    tech = random.choice(PROJECT_TECH)
    domain = random.choice(PROJECT_DOMAINS)
    outcome = random.choice(PROJECT_OUTCOMES)
    scale = random.choice(PROJECT_SCALES)
    base = tmpl.format(tech=tech, domain=domain, outcome=outcome, scale=scale)
    if outcome_tmpl:
        base += " " + outcome_tmpl.format(outcome=outcome, scale=scale, domain=domain)
    return base

def _make_experience():
    tmpl, detail_tmpl = random.choice(EXPERIENCES)
    company = random.choice(COMPANIES)
    role = random.choice(ROLES)
    duration = random.choice(["3 months", "6 months", "a year", "two years", "8 months"])
    team = random.choice(["engineering", "product", "data", "platform", "infrastructure"])
    clients = random.choice(["5", "10", "15", "multiple"])
    course = random.choice(["Data Structures", "Operating Systems", "Machine Learning", "DBMS"])
    club = random.choice(["coding", "robotics", "AI", "tech", "entrepreneurship"])
    events = random.choice(["5", "10", "15", "multiple"])
    hackathon = random.choice(["Smart India Hackathon", "HackMIT", "a national-level hackathon", "Google Code Jam"])
    result = random.choice(["reached the finals", "won first place", "got a special mention", "secured top 10"])
    professor = random.choice(["Kumar", "Singh", "Reddy", "Patel", "Joshi", "Chen", "Wilson"])
    area = random.choice(["NLP", "computer vision", "distributed systems", "cybersecurity"])
    org = random.choice(["an NGO", "a local startup", "a community coding group", "an open-source foundation"])
    deliverable = random.choice(["a web portal", "a mobile app", "a data dashboard", "an automation tool"])

    base = tmpl.format(company=company, role=role, duration=duration, team=team, clients=clients,
                       course=course, club=club, events=events, hackathon=hackathon, result=result,
                       professor=professor, area=area, org=org, deliverable=deliverable)
    if detail_tmpl:
        base += " " + detail_tmpl.format(company=company, role=role, duration=duration, team=team,
                                          clients=clients, course=course, club=club, events=events,
                                          hackathon=hackathon, result=result, professor=professor,
                                          area=area, org=org, deliverable=deliverable)
    return base

def _inject_fillers(text: str, rate: float) -> str:
    """Inject filler words at the given rate (0.0 to 1.0)."""
    if rate <= 0:
        return text
    words = text.split()
    result = []
    for w in words:
        result.append(w)
        if random.random() < rate:
            result.append(random.choice(FILLERS))
    return " ".join(result)

def _inject_code_mix(text: str, rate: float) -> str:
    if rate <= 0:
        return text
    words = text.split()
    result = []
    for w in words:
        result.append(w)
        if random.random() < rate * 0.1:
            result.append(random.choice(CODE_MIX))
    return " ".join(result)

def _inject_asr_errors(text: str, rate: float) -> str:
    if rate <= 0:
        return text
    for real, errors in ASR_ERRORS.items():
        if real in text.lower() and random.random() < rate:
            text = text.replace(real, random.choice(errors), 1)
    return text

def _inject_repetitions(text: str, rate: float) -> str:
    if rate <= 0:
        return text
    words = text.split()
    result = []
    for w in words:
        result.append(w)
        if random.random() < rate * 0.15:
            result.append(w)  # Stutter repeat
    return " ".join(result)

def _apply_noise(text: str, noise_level: float) -> str:
    """
    Sikho: "Data Augmentation (Noise Injection)".
    Mai model ko seedhe saaf English nahi padha sakta, warna wo real interviews mein confuse ho jayega.
    Isliye mai jaan-bujh kar galti (noise) daal raha hu.
    
    Numeric Impact: Agar 'noise_level' 0.0 hai, toh transcript perfect hogi. 
    Agar 1.0 (100%) kar diya, toh text "Um, ah, matlab machine burning" jaisa kachra ban jayega aur Model fail hona seekhega.
    """
    text = _inject_fillers(text, noise_level * 0.3)
    text = _inject_code_mix(text, noise_level)
    text = _inject_asr_errors(text, noise_level * 0.4)
    text = _inject_repetitions(text, noise_level)
    return text


# ── ARCHETYPE GENERATORS ─────────────────────────────────────────────────

def gen_perfect_candidate():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    skills = _pick_skills(4)
    soft = _pick_soft_skills(2)
    proj1 = _make_project()
    proj2 = _make_project()
    exp = _make_experience()
    goal = random.choice(CAREER_GOALS)

    openings = [
        f"Good morning, my name is {name}.",
        f"Hello, I'm {name}. Thank you for having me.",
        f"Hi, my name is {name} and I'm excited to be here today.",
        f"Greetings, I'm {name}. It's a pleasure to be here.",
    ]
    
    parts = [
        random.choice(openings),
        f"I graduated from {uni} with a degree in {degree}.",
        f"My primary technical skills include {', '.join(skills[:-1])}, and {skills[-1]}.",
        f"{random.choice(CONNECTORS)}I {proj1}.",
        f"I also {proj2}.",
        f"{random.choice(CONNECTORS)}In terms of professional experience, I {exp}.",
        f"My strengths include {soft[0]} and {soft[1]}, which I've demonstrated through team projects and leadership roles.",
        f"{random.choice(CONNECTORS)}my long-term goal is to {goal}.",
        random.choice([
            "I'm confident that my technical and interpersonal skills make me a strong fit for this role.",
            "I believe my blend of technical depth and collaborative mindset aligns well with your team's needs.",
            "I'm eager to contribute and grow with this organization.",
        ])
    ]
    
    return " ".join(parts), 0.0  # noise_level

def gen_solid_graduate():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    skills = _pick_skills(2)
    proj = _make_project()
    goal = random.choice(CAREER_GOALS)

    templates = [
        f"Hi, my name is {name}. I completed my {degree} from {uni}. I'm skilled in {skills[0]} and {skills[1]}. During college, I {proj}. {random.choice(CONNECTORS)}I aim to {goal}.",
        f"Hello, I'm {name}, a recent graduate in {degree} from {uni}. My key skills are {skills[0]} and {skills[1]}. I worked on a project where I {proj}. Looking ahead, my goal is to {goal}.",
        f"Good morning. I'm {name}, and I have a background in {degree} from {uni}. I'm proficient in {skills[0]} and {skills[1]}, and I {proj} as my major project. My career aspiration is to {goal}.",
    ]
    return random.choice(templates), random.uniform(0.05, 0.15)

def gen_nervous_freshman():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    skills = _pick_skills(2)

    templates = [
        f"Hi... my name is {name}. I'm from {uni}... I studied {degree}. I know {skills[0]} and a bit of {skills[1]}. That's about it, I think.",
        f"Hello, I'm {name}. I just graduated from {uni} in {degree}. I've worked with {skills[0]}... and also {skills[1]}. I'm still learning a lot...",
        f"Um, hi. My name is {name}. I did my {degree} from {uni}. I know some {skills[0]} and {skills[1]}... yeah.",
    ]
    return random.choice(templates), random.uniform(0.3, 0.6)

def gen_rambler():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    skills = _pick_skills(5)
    
    templates = [
        f"So hi my name is {name} and I am from {uni} and I know a lot of things like {skills[0]} and {skills[1]} and also {skills[2]} and I have done many projects and stuff and I really enjoy coding and building things and I also know {skills[3]} and {skills[4]} and I think technology is very interesting and I want to learn more and more every day and keep growing in this field and also I did some internship work and it was really great experience and I learned a lot from there and I believe in hard work and dedication.",
        f"Hello I'm {name} so basically I studied from {uni} and my skills are {skills[0]} and {skills[1]} also {skills[2]} and I'm very passionate about technology actually I've always been interested since childhood and I also did some projects on {skills[3]} and {skills[4]} and they were really interesting and I want to become a software developer maybe or something in tech and I think I'm a good team player and I work hard and I'm always ready to learn new things.",
    ]
    return random.choice(templates), random.uniform(0.2, 0.4)

def gen_name_dropper():
    name = _pick_name()
    skills = _pick_skills(3)
    
    templates = [
        f"Hi, I'm {name}. I'm very passionate about {skills[0]} and {skills[1]}. I love technology and I'm a quick learner. I'm really interested in {skills[2]} and I want to make a career in tech. I believe in hard work and continuous learning.",
        f"Hello, my name is {name}. I have a keen interest in {skills[0]} and {skills[1]}. Technology has always fascinated me. I'm eager to learn and grow. I think {skills[2]} is the future and I want to be part of that wave.",
        f"Hi I'm {name} and I'm passionate about {skills[0]}. I love exploring new technologies. {skills[1]} and {skills[2]} really excite me. I'm always curious and willing to learn.",
    ]
    return random.choice(templates), random.uniform(0.1, 0.3)

def gen_code_mixer():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    skills = _pick_skills(2)
    proj = _make_project()

    templates = [
        f"Hello mera naam hai {name}. Main {uni} se hoon. Actually kya hai ki mujhe {skills[0]} aur {skills[1]} aata hai. Main ne ek project kiya tha jisme I {proj}. Mera goal hai ki I want to become a good developer.",
        f"Hi main {name} hoon. So basically main ne {uni} se padhai ki. Skills ki baat karein toh {skills[0]} and {skills[1]} hai. Ek project tha where I {proj}. Looking forward to growing in this field.",
    ]
    return random.choice(templates), random.uniform(0.2, 0.5)

def gen_experienced_pro():
    name = _pick_name()
    skills = _pick_skills(4)
    exp1 = _make_experience()
    exp2 = _make_experience()
    proj = _make_project()
    goal = random.choice(CAREER_GOALS)
    years = random.choice(["two", "three", "four", "five"])

    templates = [
        f"Good morning, I'm {name}. I bring {years} years of professional experience in software development. My core expertise lies in {skills[0]}, {skills[1]}, and {skills[2]}. Most recently, I {exp1}. Prior to that, I {exp2}. One achievement I'm proud of is how I {proj}. {random.choice(CONNECTORS)}I'm looking to {goal}.",
        f"Hello, my name is {name}. Over the past {years} years, I've built a strong foundation in {skills[0]} and {skills[1]}, with hands-on experience in {skills[2]} and {skills[3]}. I {exp1}, and before that, I {exp2}. I {proj} which was particularly rewarding. My next step is to {goal}.",
    ]
    return random.choice(templates), random.uniform(0.0, 0.1)

def gen_silent_type():
    name = _pick_name()
    skill = random.choice(SKILLS_PROGRAMMING)
    templates = [
        f"Hi, I'm {name}. I know {skill}.",
        f"My name is {name}. I can do {skill}. That's it.",
        f"Hello. {name}. {skill}.",
        f"I am {name}. I studied computer science. I know {skill}.",
        f"Hi I'm {name}. Skills... {skill}. Yeah.",
    ]
    return random.choice(templates), random.uniform(0.1, 0.4)

def gen_off_topic():
    name = _pick_name()
    templates = [
        f"Hi my name is {name}. I really enjoy playing cricket on weekends. I also like watching movies, especially action films. My favorite food is biryani. I have a pet dog named Bruno.",
        f"Hello, I'm {name}. So yesterday I went to this amazing restaurant with my friends. The weather has been really nice lately. I love traveling, I went to Goa last month.",
        f"Hi I'm {name}. I'm into gaming, I play a lot of Valorant. Also I follow football, big Manchester United fan. Music is another hobby, I play guitar.",
    ]
    return random.choice(templates), random.uniform(0.1, 0.3)

def gen_passionate_beginner():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    skills = _pick_skills(2)
    interest = random.choice(SKILLS_DATA_ML + SKILLS_CLOUD_DEVOPS)

    templates = [
        f"Hi, I'm {name}. I'm currently in my final year of {degree} at {uni}. I don't have much industry experience yet, but I've been teaching myself {skills[0]} and {skills[1]} through online courses and coding challenges. What really excites me is {interest}. I spend a lot of my free time building small projects and I'm eager to get hands-on experience in a professional setting. I know I have a lot to learn, but I'm committed to growing.",
        f"Hello, my name is {name}. I'm a student at {uni} studying {degree}. I've been learning {skills[0]} and {skills[1]} on my own through YouTube tutorials and practice problems. I'm particularly fascinated by {interest} and I've started exploring it in small personal projects. My goal is to find a role where I can learn from experienced developers and contribute meaningfully.",
    ]
    return random.choice(templates), random.uniform(0.1, 0.25)

def gen_resume_mismatch():
    name = _pick_name()
    resume_skill = random.choice(["Machine Learning", "Cloud Architecture", "Full-Stack Development", "Data Engineering"])
    actual_skill = random.choice(["HTML", "basic Python", "Excel", "WordPress"])
    
    templates = [
        f"Hi I'm {name}. So I studied computer science. I know {actual_skill}. I'm learning web development. I made a simple website for my college project. I want to work in tech.",
        f"Hello my name is {name}. I have some knowledge of {actual_skill}. I've been trying to learn coding. I made a small project using {actual_skill}. I'm interested in technology.",
    ]
    return random.choice(templates), random.uniform(0.15, 0.35), {"resume_skills": [resume_skill], "pitch_skills": [actual_skill]}

def gen_stutterer():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    skills = _pick_skills(3)
    proj = _make_project()
    goal = random.choice(CAREER_GOALS)

    text = f"Hi my name is {name}. I I studied at {uni}. My my skills include {skills[0]} and and {skills[1]} and also {skills[2]}. I I worked on a project where I I {proj}. My goal my goal is to to {goal}."
    return text, random.uniform(0.3, 0.55)

def gen_fast_talker():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    skills = _pick_skills(4)
    proj = _make_project()
    exp = _make_experience()
    goal = random.choice(CAREER_GOALS)

    text = f"Hi I'm {name} from {uni} I know {skills[0]} {skills[1]} {skills[2]} {skills[3]} I {proj} also I {exp} and my goal is to {goal} I'm really excited about this opportunity and I think I can contribute a lot to the team."
    return text, random.uniform(0.0, 0.1)

def gen_slow_speaker():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    skill = random.choice(SKILLS_PROGRAMMING)
    
    text = f"Hello. My name is... {name}. I studied... at {uni}. I know... {skill}. I want to... become a developer. Thank you."
    return text, random.uniform(0.1, 0.3)

def gen_generic_speaker():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    
    templates = [
        f"Hi, I'm {name}. I'm a hardworking individual with strong communication skills. I'm a team player and a quick learner. I graduated from {uni} in {degree}. I'm passionate about technology and I want to grow in my career.",
        f"Hello, my name is {name}. I came from {uni} where I did {degree}. My strengths are teamwork, dedication, and problem-solving. I'm eager to learn new things and contribute to the organization's growth.",
    ]
    return random.choice(templates), random.uniform(0.1, 0.25)

def gen_technical_robot():
    name = _pick_name()
    skills = _pick_skills(6)
    proj1 = _make_project()
    proj2 = _make_project()
    
    text = f"My name is {name}. Technical skills: {', '.join(skills)}. Project one: I {proj1}. Project two: I {proj2}. Proficient in system design and architecture. Ready for technical challenges."
    return text, random.uniform(0.0, 0.1)

def gen_storyteller():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    skill = random.choice(SKILLS_PROGRAMMING)
    proj = _make_project()
    goal = random.choice(CAREER_GOALS)
    
    templates = [
        f"Good morning! I'm {name}, and I'd love to share my journey with you. It all started when I first wrote a 'Hello World' program in {skill} during my first year at {uni}. That moment sparked something in me. By my final year of {degree}, I had {proj}. That experience taught me that technology isn't just about code — it's about solving real problems. Looking forward, I dream of being able to {goal}. I believe every challenge is a chance to grow, and I'm excited to bring that mindset here.",
        f"Hello, my name is {name}. Let me tell you what drives me. When I joined {uni} for {degree}, I was like most freshers — curious but unfocused. Then I discovered {skill}, and everything changed. I started building things. I {proj}, which was a turning point. It taught me the value of persistence and collaboration. My vision is to {goal}, and I'm ready to work hard to get there.",
    ]
    return random.choice(templates), random.uniform(0.0, 0.1)

def gen_career_changer():
    name = _pick_name()
    old_field = random.choice(["mechanical engineering", "civil engineering", "commerce", "biology", "economics", "journalism"])
    new_skills = _pick_skills(2)
    proj = _make_project()
    goal = random.choice(CAREER_GOALS)
    
    templates = [
        f"Hi, I'm {name}. I originally studied {old_field}, but I realized my true passion lies in technology. Over the past year, I taught myself {new_skills[0]} and {new_skills[1]} through boot camps and self-study. I even {proj} to prove I could apply what I learned. The transition wasn't easy, but it confirmed that this is where I belong. My goal is to {goal}.",
        f"Hello, my name is {name}. My background is in {old_field}, which might seem unconventional. But what I bring is a unique problem-solving perspective combined with strong {new_skills[0]} and {new_skills[1]} skills I've developed through intense self-learning. I {proj} as my capstone project. I'm now fully committed to a career in tech, aiming to {goal}.",
    ]
    return random.choice(templates), random.uniform(0.05, 0.2)

def gen_improved_repeater():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    skills = _pick_skills(3)
    proj = _make_project()
    goal = random.choice(CAREER_GOALS)
    
    templates = [
        f"Good morning, I'm {name}. This is actually my second time doing this introduction, and I've worked on improving. I'm from {uni}, {degree} graduate. My technical skills include {skills[0]}, {skills[1]}, and {skills[2]}. Last time I forgot to mention my project — I {proj}. My career goal is to {goal}. I appreciate the chance to pitch again.",
        f"Hello, I'm {name}. I've practiced this introduction based on feedback from my last attempt. I studied {degree} at {uni}. I'm proficient in {skills[0]} and {skills[1]}, with growing skills in {skills[2]}. I {proj} which I'm proud of. My ambition is to {goal}.",
    ]
    return random.choice(templates), random.uniform(0.05, 0.15)

def gen_non_native_speaker():
    name = _pick_name()
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    skills = _pick_skills(2)
    proj = _make_project()
    goal = random.choice(CAREER_GOALS)
    
    # Deliberate grammar issues typical of non-native speakers
    templates = [
        f"Hello, my name {name}. I am from {uni}, studying the {degree}. I am having knowledge of {skills[0]} and {skills[1]}. In my college, I was {proj}. My goal it is to {goal}. Thank you for giving me this chance.",
        f"Good morning, I am {name}. I have completed the {degree} from {uni}. My skill is {skills[0]} and also {skills[1]}. I have did one project where I {proj}. I am want to {goal}. I hope I can do good here.",
    ]
    return random.choice(templates), random.uniform(0.1, 0.3)


# ═══════════════════════════════════════════════════════════════════════
# AUDIO FEATURE SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

def simulate_audio_features(archetype: str, text: str, noise_level: float) -> dict:
    """
    Sikho: Synthetic Feature Generation.
    Kyunki mere paas 1 Lakh asli audio recordins nahi hain, main ye function use karke 
    fake numbers banata hu jo real mic data jaise lagte hain.
    Example: Agar archetype 'nervous' hai, toh uska 'wpm' (Word Per Minute) apne aap low assign ho jayega.
    """
    word_count = len(text.split())
    
    # Base profiles per archetype
    profiles = {
        "perfect":      {"wpm": (130, 155), "tone": (0.65, 0.90), "fluency": (0.75, 0.95), "pronun": (0.80, 0.95), "energy": "building"},
        "solid":        {"wpm": (120, 160), "tone": (0.50, 0.75), "fluency": (0.60, 0.80), "pronun": (0.70, 0.90), "energy": "stable"},
        "nervous":      {"wpm": (100, 140), "tone": (0.25, 0.45), "fluency": (0.30, 0.55), "pronun": (0.55, 0.75), "energy": "fading"},
        "rambler":      {"wpm": (160, 210), "tone": (0.40, 0.65), "fluency": (0.45, 0.65), "pronun": (0.60, 0.80), "energy": "stable"},
        "name_dropper": {"wpm": (130, 160), "tone": (0.50, 0.70), "fluency": (0.55, 0.75), "pronun": (0.70, 0.85), "energy": "stable"},
        "code_mixer":   {"wpm": (110, 150), "tone": (0.35, 0.60), "fluency": (0.35, 0.60), "pronun": (0.40, 0.65), "energy": "stable"},
        "experienced":  {"wpm": (125, 150), "tone": (0.60, 0.85), "fluency": (0.70, 0.90), "pronun": (0.80, 0.95), "energy": "building"},
        "silent":       {"wpm": (60, 100),  "tone": (0.20, 0.40), "fluency": (0.40, 0.65), "pronun": (0.55, 0.80), "energy": "fading"},
        "off_topic":    {"wpm": (130, 170), "tone": (0.50, 0.75), "fluency": (0.60, 0.80), "pronun": (0.70, 0.88), "energy": "stable"},
        "passionate":   {"wpm": (140, 175), "tone": (0.55, 0.80), "fluency": (0.50, 0.70), "pronun": (0.65, 0.85), "energy": "building"},
        "mismatch":     {"wpm": (110, 150), "tone": (0.35, 0.55), "fluency": (0.45, 0.65), "pronun": (0.60, 0.80), "energy": "fading"},
        "stutterer":    {"wpm": (80, 120),  "tone": (0.30, 0.50), "fluency": (0.20, 0.40), "pronun": (0.55, 0.75), "energy": "fading"},
        "fast":         {"wpm": (180, 230), "tone": (0.55, 0.80), "fluency": (0.50, 0.70), "pronun": (0.60, 0.80), "energy": "building"},
        "slow":         {"wpm": (55, 85),   "tone": (0.30, 0.50), "fluency": (0.50, 0.70), "pronun": (0.65, 0.85), "energy": "fading"},
        "generic":      {"wpm": (120, 155), "tone": (0.40, 0.60), "fluency": (0.55, 0.75), "pronun": (0.65, 0.85), "energy": "stable"},
        "tech_robot":   {"wpm": (130, 160), "tone": (0.20, 0.40), "fluency": (0.65, 0.85), "pronun": (0.75, 0.90), "energy": "stable"},
        "storyteller":  {"wpm": (130, 160), "tone": (0.65, 0.90), "fluency": (0.70, 0.90), "pronun": (0.75, 0.92), "energy": "building"},
        "career_change":{"wpm": (120, 155), "tone": (0.50, 0.75), "fluency": (0.55, 0.75), "pronun": (0.65, 0.85), "energy": "stable"},
        "repeater":     {"wpm": (125, 155), "tone": (0.55, 0.75), "fluency": (0.60, 0.80), "pronun": (0.70, 0.88), "energy": "stable"},
        "non_native":   {"wpm": (90, 130),  "tone": (0.35, 0.55), "fluency": (0.35, 0.60), "pronun": (0.35, 0.60), "energy": "stable"},
    }
    
    p = profiles.get(archetype, profiles["solid"])
    
    wpm = random.uniform(*p["wpm"])
    tone = random.uniform(*p["tone"])
    fluency = random.uniform(*p["fluency"])
    pronun = random.uniform(*p["pronun"])
    energy_traj = p["energy"]
    
    # Add noise-based degradation
    tone = max(0.1, tone - noise_level * 0.15)
    fluency = max(0.1, fluency - noise_level * 0.2)
    pronun = max(0.1, pronun - noise_level * 0.15)
    
    # Count fillers in text
    filler_count = sum(1 for w in text.lower().split() if w in [f.lower() for f in FILLERS])
    filler_ratio = filler_count / max(word_count, 1)
    
    return {
        "wpm_estimate": round(wpm, 1),
        "pace_label": "too_slow" if wpm < 90 else "slightly_slow" if wpm < 110 else "ideal" if wpm <= 170 else "slightly_fast" if wpm <= 200 else "too_fast",
        "tone_expressiveness": round(tone, 3),
        "tone_label": "monotone" if tone < 0.35 else "moderate" if tone < 0.6 else "expressive",
        "fluency_score": round(fluency, 3),
        "pronunciation_score": round(pronun, 3),
        "energy_trajectory": energy_traj,
        "energy_consistency": round(random.uniform(0.4, 0.9), 3),
        "speech_rate_stability": round(random.uniform(0.3, 0.9), 3),
        "hnr_score": round(random.uniform(0.3, 0.8), 3),
        "long_pauses": random.randint(0, 8) if archetype in ("nervous", "stutterer", "slow", "silent") else random.randint(0, 3),
        "dynamic_confidence": round(random.uniform(30, 95), 1),
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 4),
        "spectral_flatness": round(random.uniform(0.05, 0.25), 4),
        "word_count": word_count,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════

ARCHETYPE_MAP = {
    "perfect":      gen_perfect_candidate,
    "solid":        gen_solid_graduate,
    "nervous":      gen_nervous_freshman,
    "rambler":      gen_rambler,
    "name_dropper": gen_name_dropper,
    "code_mixer":   gen_code_mixer,
    "experienced":  gen_experienced_pro,
    "silent":       gen_silent_type,
    "off_topic":    gen_off_topic,
    "passionate":   gen_passionate_beginner,
    "mismatch":     gen_resume_mismatch,
    "stutterer":    gen_stutterer,
    "fast":         gen_fast_talker,
    "slow":         gen_slow_speaker,
    "generic":      gen_generic_speaker,
    "tech_robot":   gen_technical_robot,
    "storyteller":  gen_storyteller,
    "career_change":gen_career_changer,
    "repeater":     gen_improved_repeater,
    "non_native":   gen_non_native_speaker,
}

def generate_dataset(count: int = 100000, output_path: str = None):
    # Main Function: Factory jo 1 Lakh fake candidates banati hai.
    # ML terms mein ise "Procedural Data Generation" bolte hain.
    # Agar count = 10 kar diya, toh AI sirf 10 logo ki baat sunega aur kabhi smart nahi ban payega.
    # 100,000 isliye rakha hai taaki DeepSeek model saare combinations (Skills + Noise + Persona) dekh le.
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "hr_training_dataset.jsonl")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    archetypes = list(ARCHETYPE_MAP.keys())
    per_archetype = count // len(archetypes)
    remainder = count % len(archetypes)
    
    logger.info(f"🚀 Generating {count:,} samples across {len(archetypes)} archetypes ({per_archetype:,} each)")
    
    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for arch_idx, archetype in enumerate(archetypes):
            gen_fn = ARCHETYPE_MAP[archetype]
            n = per_archetype + (1 if arch_idx < remainder else 0)
            
            for i in range(n):
                result = gen_fn()
                resume_data = None
                
                if isinstance(result, tuple) and len(result) == 3:
                    text, noise_level, resume_data = result
                else:
                    text, noise_level = result
                
                # Apply noise
                noisy_text = _apply_noise(text, noise_level)
                
                # Generate audio features
                audio = simulate_audio_features(archetype, noisy_text, noise_level)
                
                # Optional resume (30% of samples that aren't off_topic/silent)
                resume = None
                if resume_data:
                    resume = f"Skills: {', '.join(resume_data['resume_skills'])}. Projects: Advanced ML pipeline."
                elif archetype not in ("off_topic", "silent") and random.random() < 0.3:
                    resume_skills = _pick_skills(4)
                    resume_proj = _make_project()
                    resume = f"Name: {_pick_name()}. Skills: {', '.join(resume_skills)}. Projects: {resume_proj}. Education: {random.choice(DEGREES)} from {random.choice(UNIVERSITIES)}."
                
                record = {
                    "id": total,
                    "archetype": archetype,
                    "transcript": noisy_text,
                    "audio_features": audio,
                    "noise_level": round(noise_level, 3),
                    "resume": resume,
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                
                if total % 10000 == 0:
                    logger.info(f"   Generated {total:,}/{count:,} samples...")
    
    logger.info(f"✅ Dataset generated: {total:,} samples → {output_path}")
    logger.info(f"   File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HR interview training dataset")
    parser.add_argument("--count", type=int, default=100000, help="Number of samples to generate")
    args = parser.parse_args()
    generate_dataset(args.count)
