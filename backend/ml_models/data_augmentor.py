# backend/ml_models/data_augmentor.py
import os
import json
import random
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ── 1. REAL-WORLD NOISE DICTIONARIES ──
FILLERS = ["um", "uh", "like", "basic... basically", "you know", "i mean", "sort of", "kind of"]
CODE_MIX_FRAGMENTS = [
    "mera naam", "main from", "actually kya hai ki", "so basically matlab", "i am doing btech from",
    "aur main", "mujhe lagta hai", "i am very interested in this field kyunki"
]
ASR_FAILURES = {
    "software": ["soft where", "soft wear"],
    "engineering": ["engine ring", "engineear"],
    "python": ["pie thon", "pattern", "pythonn"],
    "machine learning": ["machine burning", "machine leaving", "maching learn"],
    "experience": ["x perience", "experiment"],
    "developer": ["develop her", "devil oper"]
}

# ── 2. BASE KNOWLEDGE POOL (The "Clean" Data) ──
NAMES = ["Aman", "Neha", "Rahul", "Priya", "Vikram", "Sneha", "Karan", "Pooja"]
DEGREES = ["Computer Science", "Information Technology", "Mechanical Engineering", "BCA", "Electrical Engg"]
SKILLS = ["Python", "Java", "Machine Learning", "React", "Cloud Computing", "Data Analysis", "SQL", "C++"]
EXPERIENCES = [
    "interned at TCS for 6 months", "worked on a college project building an AI interface",
    "freelanced as a web developer", "did a summer training in machine learning",
    "led the technical team in our college fest"
]
GOALS = [
    "become a successful software engineer", "contribute to cutting edge AI development",
    "grow into a leadership role", "keep learning and build scalable products",
    "work in a dynamic tech environment"
]

def generate_base_transcript():
    """Generates a perfect, clean interview introduction."""
    name = random.choice(NAMES)
    degree = random.choice(DEGREES)
    skill_1, skill_2 = random.sample(SKILLS, 2)
    exp = random.choice(EXPERIENCES)
    goal = random.choice(GOALS)
    
    parts = [
        f"Hi, my name is {name}.",
        f"I recently graduated with a degree in {degree}.",
        f"My core strengths are {skill_1} and {skill_2}.",
        f"In terms of experience, I {exp}.",
        f"My ultimate career goal is to {goal}."
    ]
    
    # Randomly shuffle or drop parts to simulate incomplete answers
    random.shuffle(parts)
    if random.random() < 0.3:  # 30% chance to drop a piece of info
        parts.pop()
        
    return " ".join(parts), len(parts)

def apply_noise(text: str, noise_level: float = 0.5) -> tuple:
    """
    Role: Ye funtion saf suthri English me randomly stutter ("um", "ah") and ASR ki mistakes dalta hai.
    Logic: text ko array me break kark loop lagata hai aur ek randomly chance (random.random()) se ghaltiya push krdeta hai.
    Numeric Impact: Agar 'noise_level = 0.5' hy, tou 50% chance barh jaega stutter ka jis se LLM normal/nervous banday 
    aur perfect confident speaker ke darmian farq karna seekhega.
    """
    words = text.split()
    noisy_words = []
    
    filler_count = 0
    asr_fail_count = 0
    code_mix_count = 0
    
    for word in words:
        # ASR Failure Injection (High impact on audio confidence)
        clean_word = word.lower().strip(".,!?")
        if clean_word in ASR_FAILURES and random.random() < noise_level * 0.4:
            noisy_words.append(random.choice(ASR_FAILURES[clean_word]))
            asr_fail_count += 1
            continue
            
        noisy_words.append(word)
        
        # Filler Injection (High impact on fluency score)
        if random.random() < noise_level * 0.5:
            filler = random.choice(FILLERS)
            noisy_words.append(filler)
            filler_count += 1
            
        # Code-Mixing Injection (Medium impact on pronunciation/fluency)
        if random.random() < noise_level * 0.05:
            mix = random.choice(CODE_MIX_FRAGMENTS)
            noisy_words.append(mix)
            code_mix_count += 1
            
        # Repetition Simulation (Stuttering)
        if random.random() < noise_level * 0.2:
            noisy_words.append(word)
            
    final_text = " ".join(noisy_words).replace(" .", ".").replace(" ,", ",")
    total_words = max(len(noisy_words), 1)
    
    return (
        final_text, 
        filler_count / total_words, 
        asr_fail_count / total_words, 
        code_mix_count / total_words
    )

def simulate_audio_features(content_completeness: int, filler_ratio: float, asr_fail_ratio: float, code_mix_ratio: float) -> dict:
    """
    Role: Audio Mathematics Simulator.
    Logic: Jab model asool me run hota hai, AI ko audio k numbers milte hain (fluency, pitch). Lekin training data hamare pas lack karta hai. Ye function real audio ka mathematics (equations) mimic karta hai.
    Numeric Impact: Agar 'asr_confidence' ya 'fluency' minus formula se low hui, tou FFNN (Deep Leaning model) ko ye pata lgega "Haan fluency low=Low Score den ahin".
    """
    # 1. ASR Confidence: drops hard if ASR fails or code mixing is extremely high
    base_confidence = random.uniform(0.85, 0.98)
    asr_confidence = max(0.1, base_confidence - (asr_fail_ratio * 2.0) - (code_mix_ratio * 1.5))
    
    # 2. Coherence (Trembling voice / Speech stability): Drops heavily with fillers and stuttering
    base_coherence = random.uniform(0.7, 1.0)
    coherence = max(0.1, base_coherence - (filler_ratio * 1.5))
    
    # 3. Fluency Score: Speed and rhythm. Drops with fillers.
    base_fluency = random.uniform(0.75, 0.95)
    fluency = max(0.1, base_fluency - (filler_ratio * 2.0))
    
    # 4. Tone Expressiveness: Voice modulation.
    # Anxious speakers (high fillers) tend to have flat/monotone voices.
    base_tone = random.uniform(0.4, 0.9)
    if filler_ratio > 0.2:
        base_tone *= 0.6  # Monotone penalty for anxious speakers
    tone = max(0.1, base_tone)
    
    # 5. Pronunciation
    base_pronun = random.uniform(0.7, 0.95)
    pronun = max(0.1, base_pronun - (code_mix_ratio * 0.5) - (asr_fail_ratio * 1.0))
    
    return {
        "completeness": content_completeness / 5.0, # Proxied out of 5 parts
        "asr_confidence": asr_confidence,
        "coherence": coherence,
        "fluency_score": fluency,
        "tone_expressiveness": tone,
        "pronunciation_score": pronun,
        "filler_ratio": filler_ratio,
        "diversity": random.uniform(0.4, 0.8), # Synthesized random diversity
        "rag_improvement": random.uniform(0.0, 1.0) # Historical context
    }

def generate_dataset(num_samples: int = 10000):
    logger.info(f"🚀 Starting Mass Generation of {num_samples} Real-World Noisy Audio/Text Samples...")
    
    dataset = []
    
    for _ in tqdm(range(num_samples)):
        # 1. Generate core knowledge
        base_text, content_parts_count = generate_base_transcript()
        
        # 2. Determine Speaker Profile (Confident, Anxious, Under-prepared, Broken)
        # SIKHO: Yahan 4 qisam ke students simulate kiye jate hyn probabilities k sath (Jaise game k characters)
        profile_roll = random.random()
        if profile_roll < 0.2:
            noise_level = 0.05 # Top tier speaker (Siraf 5% ghaltiya)
        elif profile_roll < 0.6:
            noise_level = 0.3 # Average speaker (30% ghaltiya - kuch um/uhs)
        elif profile_roll < 0.9:
            noise_level = 0.7 # Anxious speaker (70% ghaltiya - heavy stuttering, trembling)
        else:
            noise_level = 1.0 # Completely broken audio / mic failure (100% kachra)
            
        # 3. Apply Noise to Transcript
        noisy_text, f_ratio, a_ratio, c_ratio = apply_noise(base_text, noise_level)
        
        # 4. Generate corresponding Audio features (Trembling, modulation, etc)
        audio_features = simulate_audio_features(content_parts_count, f_ratio, a_ratio, c_ratio)
        
        # Append to dataset
        row = {
            "text": noisy_text,
            "noise_profile_level": noise_level,
            **audio_features
        }
        dataset.append(row)
        
    df = pd.DataFrame(dataset)
    
    # Save to disk
    os.makedirs(BASE_DIR, exist_ok=True)
    out_path = os.path.join(BASE_DIR, "advanced_synthetic_noisy_train.csv")
    df.to_csv(out_path, index=False)
    
    logger.info(f"✅ Successfully generated {len(df)} extreme noisy samples.")
    logger.info(f"💾 Saved to: {out_path}")
    
    return df

if __name__ == "__main__":
    generate_dataset(10000)
