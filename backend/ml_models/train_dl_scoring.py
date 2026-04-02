# backend/ml_models/train_dl_scoring.py
# Production 6-head DL scoring model trainer with STRICTLY CALIBRATED targets
#
# CALIBRATION PHILOSOPHY:
#   - Content quality is the PRIMARY driver of scores
#   - Short/irrelevant responses MUST score low (1-3)
#   - Only comprehensive, well-structured intros should score 7+
#   - Audio features enhance text quality, never compensate for missing content
#   - Training data must include robust examples of BAD introductions

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import multiprocessing
import random
import re
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# ── CONFIG ──
MODEL_NAME = "distilbert-base-uncased"
CACHE_DIR = "backend/data/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

from backend.ml_models.dl_scoring_model import HybridDLScoringModel

FILLER_WORDS = {"um", "uh", "like", "basically", "actually", "you know",
                "so", "well", "i mean", "right", "okay", "ok", "matlab"}

# Interview-relevant vocabulary (same as scoring_service)
_INTERVIEW_KEYWORDS = {
    "name_intro": ["my name", "i am", "i'm", "myself", "introduce"],
    "education": ["university", "college", "degree", "student", "b.tech", "engineering",
                  "institute", "studying", "studied", "bsc", "msc", "pursuing", "bca"],
    "skills": ["skill", "python", "java", "javascript", "programming", "coding",
               "react", "sql", "machine learning", "data", "software", "web",
               "communication", "teamwork", "leadership", "proficient"],
    "experience": ["experience", "worked", "working", "project", "intern", "internship",
                   "company", "role", "developed", "built", "created"],
    "goals": ["goal", "aspire", "aim", "future", "career", "become", "dream",
              "want to", "plan to", "hope to", "passionate"],
}


def _content_relevance(text: str) -> float:
    """Compute how relevant text is to an interview introduction (0-1)."""
    text_lower = text.lower()
    category_hits = 0
    total_matches = 0
    for keywords in _INTERVIEW_KEYWORDS.values():
        cat_matches = sum(1 for kw in keywords if kw in text_lower)
        if cat_matches > 0:
            category_hits += 1
        total_matches += cat_matches
    coverage = category_hits / len(_INTERVIEW_KEYWORDS)
    density = min(1.0, total_matches / 10.0)
    return coverage * 0.7 + density * 0.3


# ── STRICTLY CALIBRATED TARGET GENERATION (10 features, 6 heads) ──
def count_fillers(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for filler in FILLER_WORDS:
        count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
    return count


def generate_calibrated_targets(row):
    """
    Generate STRICTLY calibrated score targets.
    
    Score Distribution Guide:
      1-2: Gibberish, single words, pure fillers
      2-3: Very short (< 10 words), no real content
      3-4: Short (10-25 words), minimal interview content
      4-5: Below average, some content but major gaps
      5-6: Average, covers 2-3 areas briefly
      6-7: Good, covers most areas with some detail
      7-8: Very good, comprehensive with good delivery
      8-9: Excellent, thorough coverage + professional language
      9-10: Outstanding, covers everything with sophisticated expression
    """
    text = str(row["text"])
    words = text.split()
    word_count = len(words)
    
    # Section completeness (from labeled data)
    intro = float(row.get("introduction", 0))
    edu = float(row.get("education", 0))
    skills = float(row.get("skills", 0))
    exp = float(row.get("experience", 0))
    goals = float(row.get("career_goals", 0))
    completeness = (intro + edu + skills + exp + goals) / 5.0
    
    # Content relevance from actual text
    relevance = _content_relevance(text)
    
    # Length factor — STRICT scaling
    # Very short responses are harshly penalized
    if word_count < 3: length_factor = 0.05
    elif word_count < 5: length_factor = 0.10
    elif word_count < 10: length_factor = 0.20
    elif word_count < 20: length_factor = 0.35
    elif word_count < 35: length_factor = 0.50
    elif word_count < 50: length_factor = 0.65
    elif word_count < 70: length_factor = 0.80
    elif word_count < 100: length_factor = 0.90
    else: length_factor = 1.0
    
    length_norm = min(word_count / 150.0, 1.0)
    
    # Filler analysis
    filler_count = count_fillers(text)
    filler_ratio = filler_count / max(1, word_count)
    filler_penalty = min(1.0, filler_ratio * 5.0)
    
    # Diversity
    unique_words = len(set(w.lower() for w in words))
    diversity = unique_words / max(1, word_count)
    
    # Confidence signal — penalized by fillers AND short length
    base_conf = 0.7 + np.random.uniform(-0.1, 0.1)
    confidence_signal = base_conf * (1.0 - filler_penalty * 0.6) * length_factor
    
    # Coherence — strongly tied to content quality
    coherence = 0.15 + (completeness * 0.45) + (length_factor * 0.25) + (relevance * 0.15)
    rag = 0.5

    # Synthetic audio features (tied to quality indicators)
    tone_expr = min(1.0, max(0.05, (diversity * 0.3 + length_factor * 0.4 + (1.0 - filler_penalty) * 0.2 + relevance * 0.1) + np.random.uniform(-0.08, 0.08)))
    fluency_feat = min(1.0, max(0.05, (1.0 - filler_penalty) * 0.4 + length_factor * 0.35 + relevance * 0.15 + 0.1 + np.random.uniform(-0.08, 0.08)))
    pronunciation = min(1.0, max(0.05, 0.3 + length_factor * 0.3 + (1.0 - filler_penalty) * 0.2 + relevance * 0.2 + np.random.uniform(-0.05, 0.05)))

    # ═══════════ STRICTLY CALIBRATED SCORE TARGETS (1-10) ═══════════
    # Key principle: content quality > audio quality > length
    
    # STRUCTURE — primarily driven by completeness + relevance
    # A short irrelevant message should score 1-2 for structure
    raw_structure = (
        completeness * 5.0 +      # How many sections covered (max 5)
        relevance * 2.0 +          # How interview-relevant (max 2)
        length_factor * 2.0 +      # Length contribution (max 2)
        1.0                        # Base (everyone gets 1 point for trying)
    )
    # Cap structure at length_factor * 10 — short responses can't score high
    raw_structure = min(raw_structure, length_factor * 10.0)
    true_structure = np.clip(raw_structure + np.random.uniform(-0.3, 0.3), 1.0, 10.0)
    
    # CLARITY — driven by diversity, pronunciation, length, relevance
    raw_clarity = (
        diversity * 2.0 +          # Vocabulary variety (max 2)
        length_factor * 3.0 +      # Need enough words to be clear (max 3)
        (1.0 - filler_penalty) * 2.0 +  # Fillers hurt clarity (max 2)
        pronunciation * 1.5 +      # Clear speech (max 1.5)
        relevance * 1.5            # Relevant content is clearer (max 1.5)
    )
    raw_clarity = min(raw_clarity, length_factor * 10.0)
    true_clarity = np.clip(raw_clarity + np.random.uniform(-0.3, 0.3), 1.0, 10.0)
    
    # CONFIDENCE — driven by confidence signal, length, low fillers
    raw_confidence = (
        confidence_signal * 4.0 +  # Core confidence signal (max ~3.5)
        length_factor * 2.5 +      # Longer = more confident (max 2.5)
        (1.0 - filler_penalty) * 2.0 +  # Fillers = nervous (max 2)
        relevance * 1.5            # Relevant content shows preparation (max 1.5)
    )
    raw_confidence = min(raw_confidence, length_factor * 10.0)
    true_confidence = np.clip(raw_confidence + np.random.uniform(-0.3, 0.3), 1.0, 10.0)
    
    # TONE — driven by expressiveness + length
    raw_tone = (
        tone_expr * 4.0 +         # Synthetic tone signal (max 4)
        length_factor * 3.0 +      # Need sufficient audio to judge (max 3)
        (1.0 - filler_penalty) * 1.5 +  # Clean speech (max 1.5)
        relevance * 1.5            # Relevant = engaged = better tone (max 1.5)
    )
    raw_tone = min(raw_tone, length_factor * 10.0)
    true_tone = np.clip(raw_tone + np.random.uniform(-0.3, 0.3), 1.0, 10.0)
    
    # FLUENCY — driven by fluency feature, low pauses, clean delivery
    raw_fluency = (
        fluency_feat * 4.0 +       # Core fluency signal (max 4)
        length_factor * 2.5 +      # Need sufficient speech to judge (max 2.5)
        (1.0 - filler_penalty) * 2.0 +  # Fillers break flow (max 2)
        relevance * 1.5            # Relevant content = smoother delivery (max 1.5)
    )
    raw_fluency = min(raw_fluency, length_factor * 10.0)
    true_fluency = np.clip(raw_fluency + np.random.uniform(-0.3, 0.3), 1.0, 10.0)
    
    # OVERALL — weighted blend with CONTENT given highest weight
    true_overall = (
        true_structure * 0.30 +    # Content coverage is most important
        true_clarity * 0.20 +
        true_confidence * 0.15 +
        true_tone * 0.10 +
        true_fluency * 0.15 +
        relevance * 1.0            # Direct relevance bonus (max 1.0)
    )
    # Overall also capped by length
    true_overall = min(true_overall, length_factor * 10.0)
    true_overall = np.clip(true_overall, 1.0, 10.0)
    
    return pd.Series([
        completeness, confidence_signal, length_norm, diversity,
        filler_ratio, rag, coherence, tone_expr, fluency_feat, pronunciation,
        round(true_clarity, 2), round(true_confidence, 2), round(true_structure, 2),
        round(true_tone, 2), round(true_fluency, 2), round(true_overall, 2)
    ])


# ── DATASET ──
class InterviewDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.numeric_features = df[["completeness", "confidence", "length_norm", "diversity",
                                     "filler_ratio", "rag_imp", "coherence",
                                     "tone_expr", "fluency_feat", "pronunciation"]].values.astype(np.float32)
        self.labels = df[["true_clarity", "true_confidence", "true_structure",
                          "true_tone", "true_fluency", "true_overall"]].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx], truncation=True, max_length=128, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "numeric_features": torch.tensor(self.numeric_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ── AUGMENTATION ──
def augment_text_noisy(text):
    words = str(text).split()
    if not words: return text
    hinglish_inserts = ["matlab", "actually", "like", "you know", "basically", "so"]
    augmented = []
    for w in words:
        r = random.random()
        if r < 0.12:
            augmented.append(random.choice(["um", "uh"]))
            augmented.append(w)
        elif r < 0.17:
            augmented.append(random.choice(hinglish_inserts))
            augmented.append(w)
        else:
            augmented.append(w)
    augmented = [w for w in augmented if w.lower() not in ["is", "the", "a", "an"] or random.random() > 0.3]
    return " ".join(augmented)

def augment_text_minimal(text):
    words = str(text).split()
    if not words: return text
    augmented = list(words)
    for i in range(len(augmented) - 1):
        if random.random() < 0.05:
            augmented[i], augmented[i + 1] = augmented[i + 1], augmented[i]
    return " ".join(augmented)


def create_weak_samples(n=500):
    """
    Create MANY weak/bad samples so the model learns to score them LOW.
    These cover: gibberish, pure fillers, irrelevant content, very short,
    off-topic conversations, etc.
    """
    weak_texts = [
        # Pure fillers / gibberish (should score 1-2)
        "um", "uh", "hello", "hi", "yes", "no", "okay",
        "um uh basically", "uh hello", "yes basically", "okay um", "uh i am",
        "my name", "i am", "hello sir", "good morning",
        "um um um", "uh uh uh basically", "so like um",
        "i want", "i like", "coding", "python",
        "um actually basically like you know so",
        "uh i dont know", "basically um", "okay so like",
        
        # Irrelevant content (should score 1-3) — the key missing category
        "today is a nice day", "i had breakfast this morning",
        "the weather is very hot today", "i went to market yesterday",
        "my friend told me something funny", "i watched a movie last night",
        "the food in canteen is good", "i came here by bus today",
        "sir ka mood swing ho raha hai", "rajat sir mood swings today",
        "i want to share one thing about today",
        "today something happened in class", "we had fun yesterday",
        "the cricket match was good", "india won the match",
        "my phone battery is low", "i need to charge my phone",
        "lets go to the cafeteria", "class was boring today",
        "homework is pending", "exam is coming next week",
        
        # Short with minimal relevance (should score 2-3)
        "hello my name is", "i am a student", "good morning sir",
        "i am from mumbai", "i study computer", "my hobby is reading",
        "i like to code", "hello everyone", "thank you sir",
        
        # Slightly better but still inadequate (should score 3-4)
        "hello my name is rahul and i am student from delhi",
        "hi i am priya i like coding",
        "good morning my name is arun i study engineering",
        "hello i am meera from bangalore",
        "my name is rani gupta today i want to share something",
    ]
    
    rows = []
    for _ in range(n):
        text = random.choice(weak_texts)
        if random.random() < 0.3:
            text = text + " " + random.choice(weak_texts)
        
        # Most weak samples have NO interview sections
        has_intro = 1 if any(kw in text.lower() for kw in ["my name", "i am", "hello"]) else 0
        rows.append({
            "text": text, 
            "introduction": has_intro,
            "education": 0, 
            "skills": 0, 
            "experience": 0, 
            "career_goals": 0
        })
    return pd.DataFrame(rows)


def create_medium_samples(n=200):
    """Create medium-quality samples (should score 4-6)."""
    templates = [
        "Hello my name is {name}. I am from {city}. I am studying {degree}.",
        "Good morning I am {name} from {city}. I know {skill1}.",
        "Hi my name is {name}. I am a student at {city} university. I like {skill1} and {skill2}.",
        "My name is {name}. I studied {degree}. I want to {goal}.",
        "Hello everyone my name is {name}. I am pursuing {degree} from {city}. I know {skill1}.",
    ]
    names = ["Akash", "Neha", "Rohan", "Priya", "Vikas", "Sneha", "Arun", "Meera", "Rani", "Rahul"]
    cities = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune"]
    degrees = ["BTech in Computer Science", "BSc IT", "BCA", "BE in ECE"]
    skills = ["Python", "Java", "Web Development", "Data Structures", "SQL", "React"]
    goals = ["become an engineer", "get a good job", "learn more technology"]

    rows = []
    for _ in range(n):
        template = random.choice(templates)
        s1, s2 = random.sample(skills, 2)
        text = template.format(
            name=random.choice(names), city=random.choice(cities),
            degree=random.choice(degrees), skill1=s1, skill2=s2,
            goal=random.choice(goals)
        )
        # Medium samples have partial coverage
        rows.append({
            "text": text,
            "introduction": 1,
            "education": random.choice([0, 1]),
            "skills": random.choice([0, 1]),
            "experience": 0,
            "career_goals": random.choice([0, 1])
        })
    return pd.DataFrame(rows)


def create_strong_samples(n=200):
    """Create strong samples (should score 7-9+)."""
    templates = [
        "Good morning my name is {name} from {city}. I am currently pursuing {degree}. "
        "I have strong skills in {skill1} and {skill2}. I completed my internship at {company} "
        "where I worked on {project}. My career goal is to {goal}.",
        "Hello everyone I am {name} from {city}. I completed my {degree}. "
        "I am proficient in {skill1} and {skill2}. I have professional experience at {company} "
        "working on {project}. I aspire to {goal} in the future.",
        "Hi my name is {name} and I am from {city}. I am studying {degree}. "
        "My technical skills include {skill1} and {skill2}. "
        "I did an internship at {company} on {project}. My goal is to {goal}.",
        "Good morning everyone. My name is {name} and I hail from {city}. "
        "I am a final year student pursuing {degree}. Over the past few years I have developed "
        "expertise in {skill1} and {skill2}. I had the opportunity to intern at {company} "
        "where I contributed to {project}. Looking ahead I aim to {goal}. "
        "I believe my technical foundation and passion for learning make me a strong candidate.",
    ]
    names = ["Akash", "Neha", "Rohan", "Priya", "Vikas", "Sneha", "Arun", "Meera"]
    cities = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"]
    degrees = ["BTech in Computer Science", "BSc IT", "BCA", "MTech in AI", "BE in ECE"]
    skills = ["Python", "Machine Learning", "Deep Learning", "Web Development", "Data Structures",
              "AI", "Cloud Computing", "Java", "React", "SQL"]
    companies = ["Google India", "TCS", "Infosys", "Wipro", "Microsoft", "Amazon"]
    projects = ["building recommendation systems", "natural language processing pipelines",
                "web application development", "data analytics dashboards", "computer vision models"]
    goals = ["become an AI engineer", "contribute to AI research", "build scalable systems",
             "work in a top technology company", "lead innovative projects"]
    rows = []
    for _ in range(n):
        template = random.choice(templates)
        s1, s2 = random.sample(skills, 2)
        text = template.format(name=random.choice(names), city=random.choice(cities),
            degree=random.choice(degrees), skill1=s1, skill2=s2,
            company=random.choice(companies), project=random.choice(projects),
            goal=random.choice(goals))
        rows.append({"text": text, "introduction": 1, "education": 1,
                      "skills": 1, "experience": 1, "career_goals": 1})
    return pd.DataFrame(rows)


# ── METRICS ──
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    mae = torch.mean(torch.abs(predictions - labels)).item()
    mse = torch.mean((predictions - labels) ** 2).item()
    head_names = ["clarity", "confidence", "structure", "tone", "fluency", "overall"]
    per_head = {}
    for i, name in enumerate(head_names):
        per_head[f"mae_{name}"] = torch.mean(torch.abs(predictions[:, i] - labels[:, i])).item()
    return {"mae": mae, "mse": mse, **per_head}


# ── MAIN ──
def main():
    num_cores = multiprocessing.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    torch.set_num_threads(num_cores)
    print(f"🚀 Using {num_cores} CPU cores for training")

    df = pd.read_csv("backend/data/semantic_dataset_multi.csv")
    df.fillna(0, inplace=True)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"📊 Base dataset: {len(df)} samples")

    np.random.seed(42)
    random.seed(42)
    
    new_cols = ["completeness", "confidence", "length_norm", "diversity",
                "filler_ratio", "rag_imp", "coherence",
                "tone_expr", "fluency_feat", "pronunciation",
                "true_clarity", "true_confidence", "true_structure",
                "true_tone", "true_fluency", "true_overall"]
    df[new_cols] = df.apply(generate_calibrated_targets, axis=1)
    
    # Augmented copies
    noisy_df = df.copy()
    noisy_df["text"] = noisy_df["text"].apply(augment_text_noisy)
    noisy_df[new_cols] = noisy_df.apply(generate_calibrated_targets, axis=1)
    
    minimal_df = df.copy()
    minimal_df["text"] = minimal_df["text"].apply(augment_text_minimal)
    minimal_df[new_cols] = minimal_df.apply(generate_calibrated_targets, axis=1)
    
    # MORE weak samples — critical for calibration
    weak_df = create_weak_samples(500)
    weak_df[new_cols] = weak_df.apply(generate_calibrated_targets, axis=1)
    
    # Medium samples for mid-range calibration
    medium_df = create_medium_samples(200)
    medium_df[new_cols] = medium_df.apply(generate_calibrated_targets, axis=1)
    
    strong_df = create_strong_samples(200)
    strong_df[new_cols] = strong_df.apply(generate_calibrated_targets, axis=1)
    
    df = pd.concat([df, noisy_df, minimal_df, weak_df, medium_df, strong_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"📈 Full augmented dataset: {len(df)} samples")
    
    # Print score distribution to verify calibration
    for col in ["true_clarity", "true_confidence", "true_structure", "true_tone", "true_fluency", "true_overall"]:
        vals = df[col]
        print(f"   {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"min={vals.min():.2f}, max={vals.max():.2f}, "
              f"<3.0={(vals < 3.0).sum()}, 3-5={((vals >= 3.0) & (vals < 5.0)).sum()}, "
              f"5-7={((vals >= 5.0) & (vals < 7.0)).sum()}, 7+={((vals >= 7.0)).sum()}")
    
    train_df, eval_df = train_test_split(df, test_size=0.15, random_state=42)
    train_dataset = InterviewDataset(train_df)
    eval_dataset = InterviewDataset(eval_df)
    
    model = HybridDLScoringModel()

    training_args = TrainingArguments(
        output_dir="backend/data/models/dl_scoring",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=8,              # More epochs for better convergence
        logging_steps=20,
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        learning_rate=1.5e-5,            # Slightly lower LR for more careful training
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
        eval_strategy="epoch",
        save_strategy="no",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    MAX_LOOPS = 3
    current_loop = 1
    best_loss = float('inf')

    while current_loop <= MAX_LOOPS:
        print(f"\n{'='*60}")
        print(f"🔄 [Retrain Loop] Generation {current_loop}/{MAX_LOOPS}")
        print(f"{'='*60}")
        
        trainer.train()
        eval_metrics = trainer.evaluate()
        current_loss = eval_metrics.get("eval_loss", float('inf'))
        mae = eval_metrics.get("eval_mae", -1)
        
        print(f"📊 Gen {current_loop} | Eval Loss: {current_loss:.4f} | MAE: {mae:.4f}")
        for key in ["eval_mae_clarity", "eval_mae_confidence", "eval_mae_structure",
                     "eval_mae_tone", "eval_mae_fluency", "eval_mae_overall"]:
            if key in eval_metrics:
                print(f"   {key}: {eval_metrics[key]:.4f}")
        
        if current_loss < best_loss:
            print("✅ Improvement! Saving model checkpoint.")
            best_loss = current_loss
            os.makedirs("backend/data/models/dl_scoring", exist_ok=True)
            torch.save(model.state_dict(), "backend/data/models/dl_scoring/pytorch_model.bin")
            tokenizer.save_pretrained("backend/data/models/dl_scoring")
        else:
            print("⚠️ No improvement. Skipping save.")
            
        current_loop += 1

    if best_loss == float('inf'):
        print("⚠️ Failsafe: saving final weights.")
        os.makedirs("backend/data/models/dl_scoring", exist_ok=True)
        torch.save(model.state_dict(), "backend/data/models/dl_scoring/pytorch_model.bin")
        tokenizer.save_pretrained("backend/data/models/dl_scoring")

    print(f"\n✅ Training complete. Best eval loss: {best_loss:.4f}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()