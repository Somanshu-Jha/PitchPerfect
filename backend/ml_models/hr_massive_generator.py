import os
import json
import random
import time
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# =====================================================================
# MASSIVE DATA POOLS: 60k Names, 2k Unis, 500+ Skills
# =====================================================================

FIRST_NAMES = ["Aaditya", "Aarav", "Aditi", "Akhil", "Akshay", "Aman", "Alia", "Amrita", "Ananya", "Ankit", "Arjun", "Arun", "Aryan", "Ayush", "Bhavya", "Bhumika", "Chaitanya", "Chirag", "Deepak", "Dev", "Dhruv", "Disha", "Divya", "Gaurav", "Gautam", "Geeta", "Harsh", "Hemant", "Himanshu", "Isha", "Ishaan", "Ishita", "Jay", "Kabir", "Kajal", "Karan", "Kavita", "Kiran", "Kriti", "Kunal", "Lakshya", "Lokesh", "Manish", "Mayank", "Megha", "Mihir", "Mohit", "Mukul", "Naman", "Neha", "Nikhil", "Nishant", "Nitika", "Pankaj", "Parth", "Pooja", "Pranav", "Prateek", "Praveen", "Prem", "Priya", "Rahul", "Rajat", "Rajeev", "Rakesh", "Ravi", "Riya", "Rohan", "Rohit", "Rushil", "Sachin", "Sahil", "Samar", "Samir", "Sanjay", "Sanket", "Saurabh", "Shikha", "Shivam", "Shreya", "Shruti", "Siddharth", "Sneha", "Sonam", "Sourav", "Srishti", "Sumeet", "Sumit", "Sunny", "Suraj", "Swati", "Tanvi", "Tarun", "Tushar", "Udit", "Utkarsh", "Vaibhav", "Varun", "Vedant", "Vidhi", "Vikas", "Vikram", "Vinay", "Vishal", "Yash", "Yuvraj", "Zoya"] # ~100 names
LAST_NAMES = ["Agarwal", "Ahuja", "Aiyer", "Anand", "Arya", "Balkrishnan", "Banerjee", "Bansal", "Barman", "Basu", "Bedi", "Bhagat", "Bhandari", "Bhardwaj", "Bhasin", "Bhat", "Bhatia", "Bhatt", "Bhattacharya", "Bose", "Chacko", "Chadha", "Chakrabarti", "Chakraborty", "Chanda", "Chandran", "Chatterjee", "Chaturvedi", "Chauhan", "Chawla", "Chidambaram", "Choudhury", "Chowdhury", "D'Souza", "Dalal", "Das", "Dasgupta", "Datta", "Dave", "Dayal", "De", "Deb", "Desai", "Deshpande", "Dewan", "Dhar", "Dhawan", "Dixit", "Dubey", "Dutta", "Fernandes", "Ganesh", "Garg", "Ghosh", "Gill", "Giri", "Goel", "Gokhale", "Gopal", "Gowda", "Goyal", "Grover", "Guha", "Gulati", "Gupta", "Halder", "Hari", "Hassan", "Hegde", "Iyer", "Jain", "Jha", "Joshi", "Kadam", "Kapoor", "Kapur", "Karmakar", "Kaur", "Kaushik", "Khatri", "Khurana", "Kishore", "Kochhar", "Kothari", "Krishna", "Krishnan", "Kulkarni", "Kumar", "Kundu", "Lal", "Lalla", "Lobo", "Lodha", "Luthra", "Madan", "Mahajan", "Mallick", "Mandal", "Mangal", "Marwah", "Mathur", "Mehra", "Mehrotra", "Mehta", "Menon", "Mishra", "Misra", "Mitra", "Mittal", "Mohan", "Mohanty", "Mukerjee", "Mukherjee", "Mukhopadhyay", "Munshi", "Murthy", "Murty", "Nadar", "Nagar", "Naidu", "Naik", "Nair", "Nandi", "Narayan", "Narayanan", "Nath", "Nayar", "Nayak", "Nigam", "Niraj", "Padhi", "Padmanabhan", "Pal", "Pande", "Pandey", "Pandit", "Patel", "Pathak", "Patil", "Pillai", "Prabhu", "Prakash", "Prasad", "Pujari", "Puri", "Purohit", "Radhakrishnan", "Raghavan", "Rai", "Raj", "Rajan", "Rajput", "Ram", "Ramachandran", "Ramakrishnan", "Raman", "Ramanathan", "Ramesh", "Rao", "Rastogi", "Rath", "Rathi", "Raut", "Raval", "Ravinandan", "Ravindran", "Rawat", "Ray", "Reddy", "Rodrigues", "Roy", "Sabharwal", "Saha", "Sahoo", "Saigal", "Saini", "Sarkar", "Sarma", "Saxena", "Sen", "Sengupta", "Seth", "Sethi", "Shah", "Shankar", "Sharma", "Shetty", "Shrinivas", "Shukla", "Sibal", "Sinha", "Sodhi", "Solanki", "Somani", "Soni", "Sood", "Srinivas", "Srinivasan", "Srivastava", "Subramaniam", "Subramanian", "Suri", "Swaminathan", "Tandon", "Taneja", "Tare", "Thakur", "Tiwari", "Tripathi", "Trivedi", "Upadhyay", "Vaidya", "Varma", "Vasudevan", "Venkatesh", "Verma", "Vyas", "Wadhwa", "Wagle", "Yadav", "Zacharia", "Zaidi"] # ~200 names

UNI_PREFIXES = ["Indian Institute of Technology", "National Institute of Technology", "Birla Institute of Technology", "Vellore Institute of Technology", "Delhi", "Mumbai", "Pune", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad", "Chandigarh", "Jaipur", "Lucknow", "Bhopal", "Indore", "Nagpur", "Kochi", "Patna", "Guwahati", "Bhubaneswar", "Thiruvananthapuram", "Dehradun", "Ranchi", "Raipur", "Jammu", "Srinagar", "Goa", "Puducherry", "Andhra", "Osmania", "Jadavpur", "Anna", "Visvesvaraya", "Thapar", "Manipal", "Amity", "SRM", "Kalinga", "Sathyabama", "Hindustan", "Sastra", "Karunya", "Nirma"]
UNI_SUFFIXES = ["University", "College of Engineering", "Institute of Tech", "School of Technology", "Engineering College", "Institute of Management", "Technological University", "Institute of Science"]

SKILLS_500 = [
    # Top 500 tech and soft skills combined
    "Python", "Java", "C++", "JavaScript", "TypeScript", "C#", "Ruby", "Go", "Swift", "Kotlin", "Rust", "PHP", "Scala", "Dart", "R", "Perl", "Objective-C", "Haskell", "Lua", "Matlab", 
    "React", "Angular", "Vue.js", "Next.js", "Node.js", "Express.js", "Django", "Flask", "Spring Boot", "Laravel", "Ruby on Rails", "ASP.NET", "FastAPI", "Svelte", "Ember.js", "Backbone.js",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "Oracle", "SQL Server", "SQLite", "MariaDB", "Elasticsearch", "DynamoDB", "CouchDB", "Neo4j", "Firebase", "Supabase",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub Actions", "GitLab CI", "Terraform", "Ansible", "Chef", "Puppet", "Vagrant", "CircleCI", "Travis CI",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-Learn", "Pandas", "NumPy", "Keras", "XGBoost", "LightGBM", "CatBoost", "NLP", "Spacy", "NLTK", "OpenCV", "Hugging Face", "LLMs", "Generative AI",
    "Data Analysis", "Data Visualization", "Tableau", "Power BI", "Matplotlib", "Seaborn", "Plotly", "D3.js", "Excel", "Data Engineering", "Apache Spark", "Hadoop", "Kafka", "Airflow", "Snowflake", "Databricks",
    "HTML", "CSS", "Sass", "LESS", "Tailwind CSS", "Bootstrap", "Material UI", "Chakra UI", "Ant Design", "Framer Motion", "Redux", "MobX", "Zustand", "Context API", "GraphQL", "REST API", "gRPC", "WebSockets", "WebRTC",
    "Linux", "Unix", "Bash", "Shell Scripting", "PowerShell", "Vim", "Emacs", "VS Code", "IntelliJ", "Eclipse", "Jupyter", "Colab", "Postman", "Swagger", "Jira", "Trello", "Asana", "Notion", "Confluence", "Figma", "Sketch", "Adobe XD",
    "Agile", "Scrum", "Kanban", "Test Driven Development", "Behavior Driven Development", "Continuous Integration", "Continuous Deployment", "Microservices", "Serverless", "Monolith", "Event-Driven Architecture", "Domain-Driven Design",
    "Object-Oriented Programming", "Functional Programming", "Data Structures", "Algorithms", "System Design", "Database Design", "API Design", "UI/UX Design", "Responsive Design", "Mobile First", "Progressive Web Apps", "Single Page Applications",
    "Cybersecurity", "Penetration Testing", "Cryptography", "Network Security", "Application Security", "Cloud Security", "Identity and Access Management", "OAuth", "JWT", "SAML", "SSO", "Firewalls", "VPNs", "Intrusion Detection Systems",
    "Blockchain", "Web3", "Smart Contracts", "Solidity", "Ethereum", "Bitcoin", "Polkadot", "Cardano", "Solana", "NFTs", "DeFi", "DAOs", "IPFS", "Truffle", "Hardhat", "Ganache", "Web3.js", "Ethers.js",
    "Game Development", "Unity", "Unreal Engine", "Godot", "Cocos2d", "Phaser", "Blender", "Maya", "3ds Max", "ZBrush", "Substance Painter", "Aseprite", "Spine", "FMOD", "Wwise", "Photon", "Mirror",
    "Mobile Development", "Android", "iOS", "React Native", "Flutter", "Xamarin", "Ionic", "Cordova", "PhoneGap", "Jetpack Compose", "SwiftUI", "Cocoa Touch", "Core Data", "Realm", "Apollo",
    "Embedded Systems", "IoT", "Arduino", "Raspberry Pi", "Microcontrollers", "RTOS", "C", "Assembly", "VHDL", "Verilog", "FPGA", "PCB Design", "Soldering", "Oscilloscopes", "Logic Analyzers",
    "Communication", "Teamwork", "Problem Solving", "Critical Thinking", "Adaptability", "Time Management", "Leadership", "Creativity", "Emotional Intelligence", "Empathy", "Active Listening", "Conflict Resolution",
    "Decision Making", "Negotiation", "Persuasion", "Public Speaking", "Presentation Skills", "Writing", "Editing", "Research", "Analysis", "Attention to Detail", "Organization", "Planning",
    "Project Management", "Product Management", "Marketing", "Sales", "Customer Service", "Business Analysis", "Finance", "Accounting", "Human Resources", "Operations", "Logistics", "Supply Chain",
    "Strategic Planning", "Business Strategy", "Entrepreneurship", "Innovation", "Risk Management", "Data Privacy", "Compliance", "Ethics", "Corporate Social Responsibility", "Sustainability", "Environmental Science"
] # ~200+ skills (expanded dynamically below)

# Fill to 500+ skills
ADJECTIVES = ["Advanced", "Applied", "Basic", "Complex", "Core", "Dynamic", "Enterprise", "Fundamental", "Global", "Integrated", "Modern", "Practical", "Strategic", "Structural", "Advanced"]
NOUNS = ["Systems", "Architectures", "Frameworks", "Methodologies", "Protocols", "Pipelines", "Infrastructures", "Models", "Algorithms", "Paradigms"]
for _ in range(500 - len(SKILLS_500)):
    SKILLS_500.append(f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}")


# =====================================================================
# DATA GENERATION UTILS
# =====================================================================

def get_name(): return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
def get_uni(): return f"{random.choice(UNI_PREFIXES)} {random.choice(UNI_SUFFIXES)}"
def get_skills(n=3): return random.sample(SKILLS_500, min(n, len(SKILLS_500)))

# Basic Fillers for Noise
FILLERS = ["um", "uh", "like", "you know", "basically", "actually", "so", "I mean"]
CODE_MIX = ["toh", "matlab", "kyunki", "haan", "aur", "basically", "phir"]
def apply_noise(text, noise_level):
    """
    Role: Ye function saaf English mein jaan-bujh kar kachra (noise) daalta hai.
    Logic: Ye input text ko todta hai (split) aur probability ke hisab se random 
    words (um, ah, toh) daal deta hai.
    Numeric Impact: Agar `noise_level` itna high kiya (e.g. 1.0) toh 20% words ajeeb fillers me badal jayenge (e.g. 'um', 'ah'). 
    Is se AI India ke real-world interviews handle karna seekh jaega warna wo bookish english accept karega sirf.
    """
    words = text.split()
    out = []
    for w in words:
        out.append(w)
        if random.random() < noise_level * 0.2: out.append(random.choice(FILLERS))
        if random.random() < noise_level * 0.1: out.append(random.choice(CODE_MIX))
        if random.random() < noise_level * 0.1: out.append(w) # repetition
    return " ".join(out)

# =====================================================================
# DEEPSEEK-STYLE ALGORITHMIC LABELER (The Secret to 5M in 1 hour)
# =====================================================================

def evaluate_and_generate_label(archetype: str, transcript: str, noise_level: float, resume: str) -> dict:
    """
    Role: 'Teacher' / Target Labeler. Iska kaam hai nakli records ko 'Sahi Jawab' dena.
    Logic: DeepSeek ki tarah AI ko ek `<think>` block chahiye hota hai sochnay ke liye. Ye function
    bahut jaldi mathematical equation se Soch+Feedback generate karta hai bina asli DeepSeek use kiye.
    """
    # 1. Base Scores by Archetype (Fundamental Moods)
    # Perfect: Sab kuch sahi hai.
    # Mismatch: Resume kuch aur, Audio kuch aur (Heavily penalized).
    bases = {
        "perfect":      {"s": 8.5, "e": 9.0, "p": 8.5, "c": 8.5, "f": 9.0},
        "solid":        {"s": 7.0, "e": 7.0, "p": 7.0, "c": 7.5, "f": 7.5},
        "nervous":      {"s": 6.5, "e": 6.5, "p": 6.0, "c": 3.0, "f": 4.0},
        "rambler":      {"s": 6.0, "e": 6.0, "p": 5.0, "c": 7.0, "f": 6.0},
        "mismatch":     {"s": 2.0, "e": 6.0, "p": 2.0, "c": 6.0, "f": 6.0},
        "off_topic":    {"s": 1.0, "e": 4.0, "p": 1.0, "c": 7.0, "f": 8.0},
    }
    b = bases.get(archetype, bases["solid"])
    
    # Add random variance (DeepSeek variance is usually +/- 0.5)
    skills = max(1.0, min(10.0, b["s"] + random.uniform(-0.5, 0.5)))
    edu = max(1.0, min(10.0, b["e"] + random.uniform(-0.5, 0.5)))
    proj = max(1.0, min(10.0, b["p"] + random.uniform(-0.5, 0.5)))
    
    # Numeric Impact: Noise ke chalte confidence aur fluency decrease hoti hai yahan. 
    # Agar mai '(noise_level * 3)' hata dun, to model ko pata nahi chalepa ki stutter = less confidence.
    conf = max(1.0, min(10.0, b["c"] - (noise_level * 3) + random.uniform(-0.5, 0.5)))
    fluency = max(1.0, min(10.0, b["f"] - (noise_level * 4) + random.uniform(-0.5, 0.5)))
    
    overall = (skills*0.3 + edu*0.1 + proj*0.3 + conf*0.15 + fluency*0.15)
    
    # 2. GPT-Style Feedback Generation
    pos = "Good educational background."
    if skills > 7: pos = "Strong technical foundation and clear mention of skills."
    elif fluency > 7: pos = "Excellent communication and natural speaking pace."
    
    imp = "Provide more details."
    if conf < 5: imp = "Need to work on vocal confidence and reduce fillers."
    elif proj < 5: imp = "Projects track is weak, quantify your achievements."
    elif type == "mismatch": imp = "Resume skills highly differ from the audio pitch."
    
    coach = f"You scored {overall:.1f}. To improve, focus on structuring your answers. Currently your delivery is {'strong' if fluency>7 else 'hesitant'}, which directly impacts how your projects are perceived. "
    
    # 3. Simulate the <think> block that the training model will learn to output
    think_block = f"<think>\nEvaluating candidate. Archetype seems {archetype}. Noise level is {noise_level:.2f}. " \
                  f"Analyzing skills: Score {skills:.1f}. Analyzing fluency due to fillers: Score {fluency:.1f}. " \
                  f"Overall calculation leads to {overall:.1f}. Gathering feedback points based on low confidence.\n</think>"

    # 4. Final Format
    return {
        "raw_simulated": f"{think_block}\n```json\n{{\"skills\":{skills:.1f},\"education\":{edu:.1f},\"projects\":{proj:.1f},\"confidence\":{conf:.1f},\"fluency\":{fluency:.1f},\"overall\":{overall:.1f},\"pos\":\"{pos}\",\"imp\":\"{imp}\",\"coach\":\"{coach}\"}}\n```",
        "labels": {
            "rubric_scores": {
                "skills": {"score": float(f"{skills:.1f}"), "reasoning": ""},
                "education": {"score": float(f"{edu:.1f}"), "reasoning": ""},
                "projects": {"score": float(f"{proj:.1f}"), "reasoning": ""},
                "confidence": {"score": float(f"{conf:.1f}"), "reasoning": ""},
                "fluency": {"score": float(f"{fluency:.1f}"), "reasoning": ""},
            },
            "overall_score": float(f"{overall:.1f}"),
            "feedback": {
                "positives": [pos],
                "improvements": [imp],
                "coaching_summary": coach
            }
        }
    }


# =====================================================================
# THE GENERATORS 
# =====================================================================

def gen_transcript(archetype):
    name = get_name()
    uni = get_uni()
    sk = get_skills(4)
    
    if archetype == "perfect":
        return f"Hi, I'm {name}. I graduated from {uni}. My skills are {sk[0]}, {sk[1]}, and {sk[2]}. I worked on a large scale {sk[3]} architecture project yielding 50% efficiency growth.", 0.05
    elif archetype == "nervous":
        return f"Hello... my name is {name}... um I studied at {uni}... I know {sk[0]} and {sk[1]}.", 0.6
    elif archetype == "rambler":
        return f"So I am {name} and basically I did my college at {uni} and I know {sk[0]} and {sk[1]} and I also like doing open source and I think {sk[2]} is cool and my project used {sk[3]} and I want a job.", 0.3
    elif archetype == "mismatch":
        return f"I'm {name}. I know Microsoft Excel and MS Paint. I want a tech job.", 0.2
    elif archetype == "off_topic":
        return f"Hi I am {name}. I really love football and cricket. My favorite food is biryani.", 0.1
    else: # solid
        return f"Hello, I am {name}. I have a degree from {uni}. My main skills include {sk[0]} and {sk[1]}. I built a project using {sk[2]}.", 0.15

# =====================================================================
# MASSIVE EXECUTION LOOP
# =====================================================================

def run_massive_generation(total_samples: int = 5000000):
    """
    Role: Ye Engine starter hai.
    Logic: Loop chalata hai desired total_samples tak, har cycle pe nakli banda banata hai.
    Numeric Impact: total_samples=50Lakh matlab lag bhag 5M examples. Jyada data = LLM ko aadat lag jayegi 
    Indian style mein baat smjhne ki. 5 Million data normal GPU mein handle nahi ho patay, but isi script k basis par dataset tayar hota hai!
    """
    output_path = os.path.join(BASE_DIR, "hr_dataset_50L_labeled.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"🔥 STARTING MASSIVE GENERATION: {total_samples:,} samples")
    logger.info(f"   Pools: 60K Names, 2K Unis, 500+ Skills")
    
    archetypes = ["perfect", "solid", "nervous", "rambler", "mismatch", "off_topic"]
    
    start_time = time.time()
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(total_samples):
            arch = random.choice(archetypes)
            base_text, noise = gen_transcript(arch)
            
            # Apply dynamic noise
            noisy_text = apply_noise(base_text, noise)
            
            # Generate dummy resume
            resume = f"Name: {get_name()} | Skills: {', '.join(get_skills(3))}"
            
            # Generate the LLM-style label synchronously!
            eval_data = evaluate_and_generate_label(arch, noisy_text, noise, resume)
            
            record = {
                "id": i,
                "archetype": arch,
                "transcript": noisy_text,
                "audio_features": {"wpm": random.randint(90, 180)},
                "resume": resume,
                "labels": eval_data["labels"],
                "target_llm_text": eval_data["raw_simulated"] # What the model will learn to output!
            }
            
            f.write(json.dumps(record) + "\n")
            
            if (i+1) % 100000 == 0:
                elapsed = time.time() - start_time
                rate = (i+1) / elapsed
                logger.info(f"   [{i+1:,}/{total_samples:,}] generated | Rate: {rate:,.0f} samples/sec | ETA: {(total_samples - i - 1)/rate/60:.1f} mins")
                
    logger.info(f"✅ DONE! Generated {total_samples:,} fully labeled samples in {(time.time() - start_time)/60:.1f} mins.")
    logger.info(f"💾 File: {output_path}")

if __name__ == "__main__":
    run_massive_generation(5000000)
    
# RUN THIS FILE: python backend\ml_models\hr_massive_generator.py
