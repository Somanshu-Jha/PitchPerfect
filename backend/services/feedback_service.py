# =====================================================================
# FEEDBACK SERVICE — Production-Grade HR Evaluation Engine
# =====================================================================
# Ye service humari "Final Assembly Line" hai! 🏭
# 
# Iska kaam hai teen alag sources se data lena aur ek unified feedback
# response banake frontend ko dena:
#
#   Source 1: LLM (GenAI Engine) — Content dimensions ka score + reasoning + coaching
#             (opening_greeting, education, technical_skills, etc.)
#   Source 2: Audio Analysis — Delivery dimensions ka MEASURED score
#             (vocal_confidence, speech_fluency, pronunciation, pace, energy)
#   Source 3: Text Analysis — Structure dimensions ka heuristic score
#             (logical_flow, response_length, lexical_diversity)
#
# Architecture:
#   1. LLM provides per-dimension scores + reasoning + coaching text
#   2. This service validates, adjusts, and enriches the LLM output
#   3. Audio metrics OVERRIDE LLM's delivery scores (measured > guessed)
#   4. Final rubric_breakdown is computed with weighted scores
#   5. Resume alignment modifiers applied last
#   6. If LLM feedback is empty, fallback data-driven feedback is used
#
# Candidate ko finally ye dikhta hai:
#   - Overall score (1-10)
#   - Positives list (8-10 specific strengths)
#   - Improvements list (8-10 specific areas to work on)
#   - Coaching summary paragraph
#   - Resume alignment (matched/missed skills)
#   - Audio reasoning (voice analysis)
# =====================================================================

import logging
import random
import re
from backend.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# ── Dimension Weights (Ye decide karta hai ki kaunsi cheez kitni important hai) ──
# 
# Rule of Thumb: Inka sum 1.0 (yaani 100%) hona chahiye.
# - Content (60%): Aapne kya kaha (Skills, Projects). Ye sabse important hai.
# - Delivery (25%): Kaise kaha (Voice, Confidence).
# - Structure (15%): Flow kaisa tha.
#
# Kaise modify karein: Agar Technical Skills ki importance badhani hai,
# toh 0.12 ko 0.20 kar do — lekin total 1.0 rehna chahiye!
RUBRIC_WEIGHTS = {
    # Content (60%) — "Kya kaha" (What was said)
    "opening_greeting":     0.04,  # Greeting aur name introduction
    "education":            0.08,  # Education ka zikr (college, degree, year)
    "technical_skills":     0.12,  # Technical skills mentioned (Python, React, etc.)
    "project_evidence":     0.14,  # Projects ke baare mein baat ki ya nahi
    "work_experience":      0.08,  # Kaam ka experience (internship, job, freelance)
    "career_goals":         0.06,  # Future plans aur career aspirations
    "strengths_qualities":  0.04,  # Personal strengths (leadership, teamwork, etc.)
    "areas_of_interest":    0.04,  # Interests aur passions
    # Delivery (25%) — "Kaise kaha" (How it was said)
    "vocal_confidence":     0.05,  # Awaaz mein confidence
    "speech_fluency":       0.05,  # Bina ruke fluently bola ya nahi
    "pronunciation_clarity":0.05,  # Shabd kitne saaf the
    "speaking_pace":        0.05,  # Speed — too fast/slow/ideal
    "energy_trajectory":    0.05,  # Energy badhti gayi ya girti gayi
    # Structure (15%) — "Sequence kaisa tha" (How it was structured)
    "logical_flow":         0.05,  # Naam → Education → Skills → Goals ka flow
    "response_length":      0.05,  # Response kitna lamba/chota tha
    "lexical_diversity":    0.05,  # Vocabulary variety (unique words ratio)
}

# ── Human-readable names for frontend display ──
# Dimension keys short/technical hain, ye unke user-friendly labels hain.
DIMENSION_LABELS = {
    "opening_greeting": "Opening & Greeting",
    "education": "Educational Background",
    "technical_skills": "Technical Skills",
    "project_evidence": "Project Evidence",
    "work_experience": "Work Experience",
    "career_goals": "Career Goals",
    "strengths_qualities": "Strengths & Qualities",
    "areas_of_interest": "Areas of Interest",
    "vocal_confidence": "Vocal Confidence",
    "speech_fluency": "Speech Fluency",
    "pronunciation_clarity": "Pronunciation Clarity",
    "speaking_pace": "Speaking Pace",
    "energy_trajectory": "Energy & Engagement",
    "logical_flow": "Logical Flow",
    "response_length": "Response Length",
    "lexical_diversity": "Vocabulary Diversity",
}

# ── Category mapping — har dimension ek category mein aata hai ──
# Ye grouping frontend pe category-wise display ke liye use hota hai.
DIMENSION_CATEGORIES = {
    "opening_greeting": "Content", "education": "Content",
    "technical_skills": "Content", "project_evidence": "Content",
    "work_experience": "Content", "career_goals": "Content",
    "strengths_qualities": "Content", "areas_of_interest": "Content",
    "vocal_confidence": "Delivery", "speech_fluency": "Delivery",
    "pronunciation_clarity": "Delivery", "speaking_pace": "Delivery",
    "energy_trajectory": "Delivery",
    "logical_flow": "Structure", "response_length": "Structure",
    "lexical_diversity": "Structure",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """
    Value ko ek safe range mein rakhta hai.
    
    Problem: Kabhi kabhi calculation se score 11.5 ya -2.0 aa sakta hai.
    Ye function ensure karta hai ki score hamesha 0.0-10.0 ke beech rahe.
    
    Args:
        value: Input score (koi bhi float).
        lo: Minimum allowed value (default 0.0).
        hi: Maximum allowed value (default 10.0).
    
    Returns:
        Clamped value, rounded to 1 decimal place.
    """
    return round(max(lo, min(hi, value)), 1)


def _compute_audio_delivery_scores(audio_features: dict, fillers: list, word_count: int) -> dict:
    """
    Audio metrics se "Sacha" (Factual/Measured) delivery score nikalta hai.
    
    IMPORTANT: Ye scores LLM ke guessed delivery scores ko OVERRIDE karte hain.
    Kyunki audio metrics actually MEASURED hain (librosa se), jabki LLM sirf
    transcript padh kar guess karta hai ki confidence kaisi thi.
    
    Ye function 5 delivery dimensions ke liye scores calculate karta hai:
    1. Vocal Confidence — tone expressiveness + energy consistency se
    2. Speech Fluency — fluency_score + filler_count se
    3. Pronunciation Clarity — pronunciation_score + HNR se
    4. Speaking Pace — WPM range check (120-160 ideal)
    5. Energy Trajectory — building/stable/fading trend
    
    Har dimension ka score 0-10 mein aata hai, saath mein deductions list.
    Deductions mein exact metric values hoti hain (e.g., "fluency: 0.35").
    
    Args:
        audio_features: AudioAnalysisService.extract() se aaya dict.
        fillers: List of detected filler words.
        word_count: Total words in transcript.
    
    Returns:
        Tuple of (scores_dict, deductions_dict).
    """
    af = audio_features or {}
    scores = {}
    deductions = {}

    # ── Vocal Confidence ─────────────────────────────
    # Formula: tone_expressiveness × 5 + energy_consistency × 5 = max 10
    # Penalties: monotone voice (-2), fading energy (-1.5)
    tone_expr = af.get("tone_expressiveness", 0.5)
    energy_con = af.get("energy_consistency", 0.5)
    conf_score = (tone_expr * 5 + energy_con * 5)
    conf_deductions = []
    if tone_expr < 0.3:
        conf_score -= 2.0
        conf_deductions.append(f"Monotone voice (expressiveness: {tone_expr:.2f})")
    if af.get("energy_trajectory") == "fading":
        conf_score -= 1.5
        conf_deductions.append("Voice energy fades toward the end")
    scores["vocal_confidence"] = _clamp(conf_score)
    deductions["vocal_confidence"] = conf_deductions

    # ── Speech Fluency ───────────────────────────────
    # Base score = fluency_score × 10 (0-10 range)
    # Filler scoring now uses DENSITY (per 100 words) + per-type analysis
    # from the enhanced filler detection service
    fluency_raw = af.get("fluency_score", 0.5)
    flu_score = fluency_raw * 10
    flu_deductions = []
    filler_count = len(fillers) if fillers else 0
    filler_density = af.get("filler_density", 0.0)
    filler_per_type = af.get("filler_per_type", {})
    filler_cluster = af.get("filler_position_cluster", "none")
    filler_most_freq = af.get("filler_most_frequent", None)

    # Density-based penalties (more accurate than raw count)
    if filler_density > 8.0:
        flu_score -= 3.5
        flu_deductions.append(f"Critical filler density: {filler_density:.1f} per 100 words ({filler_count} total)")
    elif filler_density > 5.0:
        flu_score -= 2.5
        flu_deductions.append(f"High filler density: {filler_density:.1f} per 100 words ({filler_count} total)")
    elif filler_density > 3.0:
        flu_score -= 1.5
        flu_deductions.append(f"Moderate fillers: {filler_density:.1f} per 100 words ({filler_count} total)")
    elif filler_count > 2:
        flu_score -= 0.5
        flu_deductions.append(f"Minor fillers: {filler_count} detected")

    # Per-type penalty: repetitive use of same filler is worse
    if filler_most_freq and filler_per_type.get(filler_most_freq, 0) >= 4:
        flu_score -= 1.0
        flu_deductions.append(
            f"Repetitive filler: '{filler_most_freq}' used {filler_per_type[filler_most_freq]}x"
        )

    # Cluster penalty: fillers concentrated in one area suggests nervousness
    if filler_cluster in ("beginning", "end") and filler_count > 3:
        flu_score -= 0.5
        flu_deductions.append(f"Fillers concentrated at {filler_cluster} of pitch")

    if fluency_raw < 0.35:
        flu_score -= 1.5
        flu_deductions.append(f"Low fluency metric ({fluency_raw:.2f})")
    scores["speech_fluency"] = _clamp(flu_score)
    deductions["speech_fluency"] = flu_deductions

    # ── Pronunciation Clarity ────────────────────────
    # Formula: pronunciation_score × 6 + hnr_score × 4 = max 10
    # HNR (Harmonic-to-Noise Ratio) — zyada harmonic = clearer voice
    pronun = af.get("pronunciation_score", 0.5)
    hnr = af.get("hnr_score", 0.5)
    pron_score = (pronun * 6 + hnr * 4)
    pron_deductions = []
    if pronun < 0.4:
        pron_score -= 2.0
        pron_deductions.append(f"Low pronunciation clarity ({pronun:.2f})")
    scores["pronunciation_clarity"] = _clamp(pron_score)
    deductions["pronunciation_clarity"] = pron_deductions

    # ── Speaking Pace ────────────────────────────────
    # WPM ranges: 120-160 = ideal (8.5), 100-180 = acceptable (7.0),
    # <90 = too slow (4.0), >200 = too fast (4.0)
    wpm = af.get("wpm_estimate", 140)
    pace_label = af.get("pace_label", "ideal")
    pace_deductions = []
    if 120 <= wpm <= 160:
        pace_score = 8.5
    elif 100 <= wpm <= 180:
        pace_score = 7.0
    elif wpm < 90:
        pace_score = 4.0
        pace_deductions.append(f"Too slow ({int(wpm)} WPM) — ideal is 120-160")
    elif wpm > 200:
        pace_score = 4.0
        pace_deductions.append(f"Too fast ({int(wpm)} WPM) — ideal is 120-160")
    else:
        pace_score = 6.0
        pace_deductions.append(f"Pace slightly off ({int(wpm)} WPM)")
    scores["speaking_pace"] = _clamp(pace_score)
    deductions["speaking_pace"] = pace_deductions

    # ── Energy Trajectory ────────────────────────────
    # building → great (9.0), stable → ok (7.0), fading → weak (4.0)
    energy_traj = af.get("energy_trajectory", "stable")
    energy_deductions = []
    if energy_traj == "building":
        energy_score = 9.0
    elif energy_traj == "stable":
        energy_score = 7.0
    elif energy_traj == "fading":
        energy_score = 4.0
        energy_deductions.append("Voice energy drops through the pitch — closing is weak")
    else:
        energy_score = 6.0
    scores["energy_trajectory"] = _clamp(energy_score)
    deductions["energy_trajectory"] = energy_deductions

    return scores, deductions


def _compute_structure_scores(transcript: str, semantic: dict) -> dict:
    """
    Transcript analysis se structural dimension scores nikalta hai.
    
    Ye function teen cheezein check karta hai:
    1. Response Length — kitne words bole (too short / ideal / too long)
    2. Lexical Diversity — unique words ka ratio (higher = better vocabulary)
    3. Logical Flow — kya naam pehle aaya, skills baad mein, goals end mein?
    
    Ye HEURISTIC scores hain — LLM nahi, NLP maths hai.
    Agar LLM ne inka score diya hai toh ye override ho jayenge
    (priority: audio > struct > LLM > content_heuristics).
    
    Args:
        transcript: Candidate ka refined transcript text.
        semantic: SemanticService se aaya structured extraction dict.
    
    Returns:
        Tuple of (scores_dict, deductions_dict).
    """
    words = transcript.split()
    word_count = len(words)
    scores = {}
    deductions = {}

    # ── Response Length ───────────────────────────────
    # Ideal: 80-200 words. Too short (<30) = very weak first impression.
    # Too long (>300) = rambling, loses interviewer's attention.
    len_deductions = []
    if word_count < 15:
        len_score = 1.5
        len_deductions.append(f"Extremely short ({word_count} words)")
    elif word_count < 30:
        len_score = 3.0
        len_deductions.append(f"Very short ({word_count} words)")
    elif word_count < 50:
        len_score = 4.5
        len_deductions.append(f"Short response ({word_count} words)")
    elif word_count < 80:
        len_score = 6.0
        len_deductions.append(f"Below ideal length ({word_count} words)")
    elif word_count <= 200:
        len_score = 8.5
    elif word_count <= 300:
        len_score = 7.0
        len_deductions.append(f"Slightly long ({word_count} words)")
    else:
        len_score = 5.0
        len_deductions.append(f"Excessively long/rambling ({word_count} words)")
    scores["response_length"] = _clamp(len_score)
    deductions["response_length"] = len_deductions

    # ── Lexical Diversity ────────────────────────────
    # Unique words / Total words ratio — zyada = better vocabulary
    # 0.7+ = excellent, 0.55+ = good, 0.4+ = average, <0.4 = weak
    unique_words = len(set(w.lower() for w in words)) if words else 0
    ratio = unique_words / max(word_count, 1)
    lex_deductions = []
    if ratio >= 0.7:
        lex_score = 9.0
    elif ratio >= 0.55:
        lex_score = 7.5
    elif ratio >= 0.4:
        lex_score = 6.0
    else:
        lex_score = 4.0
        lex_deductions.append(f"Low vocabulary diversity (ratio: {ratio:.2f})")
    scores["lexical_diversity"] = _clamp(lex_score)
    deductions["lexical_diversity"] = lex_deductions

    # ── Logical Flow ─────────────────────────────────
    # Check: Naam pehle aaya? Skills ke baad goals?
    # Agar naam transcript mein nahi mila aur response 20+ words hai → penalty.
    flow_deductions = []
    flow_score = 7.0  # Default: acceptable

    structured = semantic.get("structured", {})
    evidence_map = semantic.get("evidence_map", {})

    text_lower = transcript.lower()
    name_pos = text_lower.find(str(structured.get("name", "ZZZZZ")).lower())
    skill_found = False
    goal_found = False
    for s in (structured.get("skills", []) or []):
        if isinstance(s, str) and s.lower() in text_lower:
            skill_found = True
            break
    goal_val = structured.get("career_goals", "")
    if isinstance(goal_val, str) and goal_val and goal_val.lower()[:10] in text_lower:
        goal_found = True

    if name_pos < 0 and word_count > 20:
        flow_score -= 1.5
        flow_deductions.append("Name not clearly stated early in the introduction")

    # Agar goals mention hue lekin skills nahi → structural issue
    if goal_found and not skill_found and word_count > 30:
        flow_score -= 1.0
        flow_deductions.append("Career goals mentioned without establishing skills first")

    scores["logical_flow"] = _clamp(flow_score)
    deductions["logical_flow"] = flow_deductions

    return scores, deductions


def _compute_content_scores(transcript: str, semantic: dict) -> dict:
    """
    Content dimensions ke liye FALLBACK heuristic scores nikalta hai.
    
    IMPORTANT: Ye scores sirf tab use hote hain jab LLM ne kisi dimension
    ka score nahi diya. LLM priority zyada hai — ye backup plan hai.
    
    Har content dimension ke liye keyword matching karta hai:
    - Opening: greeting words ("hello", "hi") + name patterns ("my name is")
    - Education: college/degree keywords
    - Technical Skills: Python, React, etc. keywords
    - Projects: "built", "developed", "hackathon" etc.
    - Work Experience: "intern", "company", "role" etc.
    - Career Goals: "goal", "aspire", "future" etc.
    - Strengths: "leadership", "problem solving" etc.
    - Areas of Interest: "passionate about", "curious" etc.
    
    Kaise kaam karta hai: keyword count check karo, semantic extraction
    data check karo, dono combine karke score do.
    
    Args:
        transcript: Candidate ka refined transcript.
        semantic: SemanticService se aaya extraction dict.
    
    Returns:
        Tuple of (scores_dict, deductions_dict).
    """
    text_lower = transcript.lower()
    words = transcript.split()
    word_count = len(words)
    structured = semantic.get("structured", {})
    scores = {}
    deductions = {}

    # ── Opening & Greeting ───────────────────────────
    greeting_deductions = []
    greeting_keywords = ["hello", "hi ", "good morning", "good afternoon", "good evening",
                         "hey", "greetings", "pleased to meet", "thank you for"]
    name_patterns = ["my name is", "i am ", "i'm ", "this is ", "myself "]
    has_greeting = any(kw in text_lower[:100] for kw in greeting_keywords)
    has_name_intro = any(pat in text_lower[:150] for pat in name_patterns)
    
    if has_greeting and has_name_intro:
        greeting_score = 9.0
    elif has_name_intro:
        greeting_score = 7.0
        greeting_deductions.append("Name stated but no formal greeting (Hello/Good morning)")
    elif has_greeting:
        greeting_score = 6.0
        greeting_deductions.append("Greeting present but didn't introduce name clearly")
    else:
        greeting_score = 3.5
        greeting_deductions.append("No greeting and no clear self-introduction at the start")
    scores["opening_greeting"] = _clamp(greeting_score)
    deductions["opening_greeting"] = greeting_deductions

    # ── Work Experience ──────────────────────────────
    exp_deductions = []
    exp_keywords = ["work", "worked", "working", "intern", "internship", "company",
                    "organization", "role", "position", "job", "employed", "freelance",
                    "project", "team", "managed", "developed", "built", "led"]
    exp_matches = sum(1 for kw in exp_keywords if kw in text_lower)
    exp_text = structured.get("experience", "")
    
    if exp_matches >= 5 or (exp_text and len(str(exp_text)) > 30):
        exp_score = 8.5
    elif exp_matches >= 3 or (exp_text and len(str(exp_text)) > 10):
        exp_score = 7.0
    elif exp_matches >= 1:
        exp_score = 5.0
        exp_deductions.append("Work experience mentioned briefly — add specific roles, companies, and achievements")
    else:
        exp_score = 3.0
        exp_deductions.append("No work experience discussed — mention internships, projects, or freelance work")
    scores["work_experience"] = _clamp(exp_score)
    deductions["work_experience"] = exp_deductions

    # ── Career Goals ─────────────────────────────────
    goal_deductions = []
    goal_keywords = ["goal", "aspire", "want to", "plan to", "aim", "dream", "vision",
                     "become", "future", "career", "ambition", "hoping to", "looking forward",
                     "passionate about", "interested in becoming", "long term", "short term"]
    goal_matches = sum(1 for kw in goal_keywords if kw in text_lower)
    goal_text = structured.get("career_goals", "")
    
    if goal_matches >= 3 or (goal_text and len(str(goal_text)) > 30):
        goal_score = 8.5
    elif goal_matches >= 1 or (goal_text and len(str(goal_text)) > 5):
        goal_score = 6.5
        goal_deductions.append("Career goals mentioned briefly — be more specific about short/long-term objectives")
    else:
        goal_score = 3.5
        goal_deductions.append("No career goals mentioned — always end with where you see yourself heading")
    scores["career_goals"] = _clamp(goal_score)
    deductions["career_goals"] = goal_deductions

    # ── Strengths & Qualities ────────────────────────
    str_deductions = []
    str_keywords = ["strength", "strong", "good at", "excel", "quick learner", "team player",
                    "leadership", "problem solving", "creative", "analytical", "dedicated",
                    "motivated", "hardworking", "adaptable", "detail-oriented", "communication"]
    str_matches = sum(1 for kw in str_keywords if kw in text_lower)
    
    if str_matches >= 3:
        str_score = 8.5
    elif str_matches >= 1:
        str_score = 6.5
        str_deductions.append("Mentioned strengths briefly — back them up with examples or evidence")
    else:
        str_score = 4.0
        str_deductions.append("No personal strengths highlighted — mention 2-3 key qualities that make you stand out")
    scores["strengths_qualities"] = _clamp(str_score)
    deductions["strengths_qualities"] = str_deductions

    # ── Areas of Interest ────────────────────────────
    int_deductions = []
    int_keywords = ["interested in", "passionate about", "love", "enjoy", "fascinated",
                    "curious about", "hobby", "hobbies", "exploring", "enthusiastic",
                    "volunteer", "community", "research", "extracurricular"]
    int_matches = sum(1 for kw in int_keywords if kw in text_lower)
    
    if int_matches >= 2:
        int_score = 8.0
    elif int_matches >= 1:
        int_score = 6.5
        int_deductions.append("Areas of interest touched briefly — elaborate on what drives your passion")
    else:
        int_score = 4.5
        int_deductions.append("No areas of interest shared — mention what excites you beyond academics/work")
    scores["areas_of_interest"] = _clamp(int_score)
    deductions["areas_of_interest"] = int_deductions

    # ── Educational Background ───────────────────────
    edu_deductions = []
    edu_keywords = ["university", "college", "school", "degree", "bachelor", "master",
                    "btech", "b.tech", "mtech", "m.tech", "bsc", "msc", "mba", "phd",
                    "student", "studying", "semester", "year", "class", "cgpa", "gpa",
                    "computer science", "engineering", "artificial intelligence",
                    "graduated", "diploma", "certification", "institute", "iit", "nit"]
    edu_matches = sum(1 for kw in edu_keywords if kw in text_lower)
    edu_text = structured.get("education", "")
    
    if edu_matches >= 4 or (edu_text and len(str(edu_text)) > 40):
        edu_score = 8.5
    elif edu_matches >= 2 or (edu_text and len(str(edu_text)) > 10):
        edu_score = 7.0
    elif edu_matches >= 1:
        edu_score = 5.5
        edu_deductions.append("Education mentioned briefly — include degree, institution, and year")
    else:
        edu_score = 3.5
        edu_deductions.append("No educational background discussed — always mention your degree and institution")
    scores["education"] = _clamp(edu_score)
    deductions["education"] = edu_deductions

    # ── Technical Skills ─────────────────────────────
    tech_deductions = []
    tech_keywords = ["python", "java", "javascript", "react", "node", "sql", "html", "css",
                     "machine learning", "deep learning", "ai", "data science", "cloud",
                     "aws", "docker", "kubernetes", "git", "api", "database", "algorithm",
                     "programming", "coding", "software", "development", "framework",
                     "tensorflow", "pytorch", "flutter", "android", "ios", "web",
                     "c++", "c#", "rust", "go", "typescript", "django", "flask",
                     "linux", "devops", "ci/cd", "agile", "scrum", "microservices"]
    tech_matches = sum(1 for kw in tech_keywords if kw in text_lower)
    skills_list = structured.get("skills", [])
    skill_count = len(skills_list) if isinstance(skills_list, list) else 0
    
    if tech_matches >= 5 or skill_count >= 4:
        tech_score = 9.0
    elif tech_matches >= 3 or skill_count >= 2:
        tech_score = 7.5
    elif tech_matches >= 1 or skill_count >= 1:
        tech_score = 5.5
        tech_deductions.append("Few technical skills mentioned — list 3-5 specific technologies with proficiency levels")
    else:
        tech_score = 3.0
        tech_deductions.append("No technical skills discussed — always highlight your core technical stack")
    scores["technical_skills"] = _clamp(tech_score)
    deductions["technical_skills"] = tech_deductions

    # ── Project Evidence ─────────────────────────────
    proj_deductions = []
    proj_keywords = ["project", "built", "developed", "created", "designed", "implemented",
                     "hackathon", "competition", "research paper", "published", "deployed",
                     "application", "app", "system", "platform", "website", "tool",
                     "contributed", "open source", "github", "portfolio", "demo"]
    proj_matches = sum(1 for kw in proj_keywords if kw in text_lower)
    proj_text = structured.get("projects", "") or structured.get("experience", "")
    
    if proj_matches >= 4 or (proj_text and len(str(proj_text)) > 40):
        proj_score = 8.5
    elif proj_matches >= 2 or (proj_text and len(str(proj_text)) > 15):
        proj_score = 7.0
    elif proj_matches >= 1:
        proj_score = 5.0
        proj_deductions.append("Projects mentioned briefly — describe the problem solved, tech used, and impact")
    else:
        proj_score = 3.0
        proj_deductions.append("No project evidence provided — describe 1-2 key projects with outcomes")
    scores["project_evidence"] = _clamp(proj_score)
    deductions["project_evidence"] = proj_deductions

    return scores, deductions


class FeedbackService:
    """
    Production HR Evaluation Engine — Final Assembly Line.

    Ye class teen alag sources ka data merge karke ek unified feedback response
    banati hai jo frontend ko jaata hai:
    
    1. LLM deep reasoning (content dimensions 1-8, overall coaching)
       → GenAI Engine se aata hai via precomputed_llm parameter
    2. Audio signal processing (delivery dimensions 9-13)
       → AudioAnalysisService se measured metrics — LLM ke guesses ko OVERRIDE karta hai
    3. Deterministic text analysis (structure dimensions 14-16)
       → Keyword matching + NLP heuristics
    
    generate() method:
    - 16-dimension rubric_breakdown with per-dimension scores (internal only)
    - Weighted overall score with strictness multiplier
    - Specific deduction reasons for each dimension
    - HR-quality coaching summary
    - Positives and improvements lists (8-10 each)
    - Resume alignment data (matched/missed skills)
    """

    def __init__(self):
        """
        FeedbackService initialize karta hai.
        RAGService load karta hai (Retrieval-Augmented Generation ke liye).
        """
        logger.info("⚡ [FeedbackService] Initializing Production HR Engine (16 dimensions)...")
        self.rag_service = RAGService()

    def _deduplicate_feedback(self, items: list) -> list:
        """
        Duplicate feedback items remove karta hai.
        
        Multi-point matching: checks prefix (first 60 chars), midpoint, and
        key phrase overlap to catch rephrased duplicates.
        
        Args:
            items: List of feedback strings or dicts.
        
        Returns:
            Deduplicated list of strings.
        """
        if not items: return items
        seen_prefixes = set()
        seen_phrases = set()
        unique = []

        for item in items:
            if isinstance(item, dict):
                txt = item.get("text", "")
            else:
                txt = str(item)
            txt = txt.strip()
            if not txt:
                continue

            txt_lower = txt.lower()
            prefix_key = txt_lower[:60]

            # Extract key phrases (3-word sequences) for semantic dedup
            words = txt_lower.split()
            key_phrases = set()
            for i in range(len(words) - 2):
                key_phrases.add(f"{words[i]} {words[i+1]} {words[i+2]}")

            # Check prefix match
            if prefix_key in seen_prefixes:
                continue

            # Check key phrase overlap (>50% overlap = duplicate)
            if seen_phrases and key_phrases:
                overlap = len(key_phrases & seen_phrases)
                if overlap > len(key_phrases) * 0.5:
                    continue

            seen_prefixes.add(prefix_key)
            seen_phrases.update(key_phrases)
            unique.append(txt)  # Always return as string

        return unique

    def generate(self, user_id: str, transcript: str, semantic: dict, scores: dict,
                 fillers: list, english_level: str = "Intermediate",
                 audio_features: dict = None, precomputed_llm: dict = None,
                 strictness: str = "intermediate") -> dict:
        """
        Main entry point — 16-dimension evaluation build karta hai.

        Ye function sabse important hai — pipeline ka final assembly step.
        
        Flow:
        1. LLM rubric scores collect karo (precomputed_llm se).
        2. Audio delivery scores compute karo (measured metrics se).
        3. Structural + content heuristic scores compute karo.
        4. Final rubric breakdown banao — priority: audio > struct > LLM > heuristics.
        5. Weighted overall score calculate karo.
        6. Resume alignment modifier apply karo.
        7. Strictness penalty apply karo (beginner=+10%, extreme=-22%).
        8. LLM se aaye generative feedback (positives/improvements) passthrough karo.
        9. Agar LLM feedback empty hai toh NEVER return empty — skeleton response do.
        
        Args:
            user_id: Database user identifier.
            transcript: Candidate ka refined transcript.
            semantic: SemanticService se aaya extraction dict.
            scores: Previous scoring data (usually empty {}).
            fillers: List of detected filler words.
            english_level: English proficiency classification.
            audio_features: AudioAnalysisService se aaye metrics.
            precomputed_llm: GenAI Engine ka comprehensive_analyze() ka output.
                Contains: rubric_scores, feedback, semantic, resume_alignment.
            strictness: Difficulty level (beginner/intermediate/advance/extreme).

        Returns:
            Complete feedback dict with positives, improvements, coaching_summary,
            rubric_breakdown (internal), overall_rubric_score, resume_alignment,
            audio_reasoning.
        """
        if not transcript or not transcript.strip():
            return {
                "positives": [], "improvements": [],
                "coaching_summary": "No transcript detected.",
                "rubric_breakdown": [], "audio_reasoning": {},
                "overall_score": 1.0
            }

        words = transcript.split()
        word_count = len(words)

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Collect LLM rubric scores (content dimensions)
        # precomputed_llm mein se rubric_scores, feedback, resume_alignment
        # extract karte hain. KEY_MAP se short keys ko 16-dim keys mein map karte hain.
        # ═══════════════════════════════════════════════════════════════
        llm_rubric = {}
        llm_deductions = {}
        llm_feedback = {"positives": [], "improvements": [], "coaching_summary": ""}
        resume_alignment = {"matched": [], "missed": [], "score_modifier": 0.0}

        if precomputed_llm:
            raw_rubric = precomputed_llm.get("rubric_scores", {})
            llm_feedback = precomputed_llm.get("feedback", llm_feedback)
            resume_alignment = precomputed_llm.get("resume_alignment", resume_alignment)
            
            # CRITICAL: Map HR model's short keys → feedback_service's 16-dim keys
            # HR model returns: skills, education, projects, confidence, fluency, structure
            # Feedback service expects: technical_skills, education, project_evidence, etc.
            KEY_MAP = {
                "skills": "technical_skills",
                "projects": "project_evidence",
                "confidence": "vocal_confidence",
                "fluency": "speech_fluency",
                "structure": "logical_flow",
                # These already match — identity mapping:
                "education": "education",
                "opening_greeting": "opening_greeting",
                "technical_skills": "technical_skills",
                "project_evidence": "project_evidence",
                "work_experience": "work_experience",
                "career_goals": "career_goals",
                "strengths_qualities": "strengths_qualities",
                "areas_of_interest": "areas_of_interest",
                "vocal_confidence": "vocal_confidence",
                "speech_fluency": "speech_fluency",
                "pronunciation_clarity": "pronunciation_clarity",
                "speaking_pace": "speaking_pace",
                "energy_trajectory": "energy_trajectory",
                "logical_flow": "logical_flow",
                "response_length": "response_length",
                "lexical_diversity": "lexical_diversity",
            }
            for key, data in raw_rubric.items():
                mapped_key = KEY_MAP.get(key, key)
                llm_rubric[mapped_key] = data
            
            # score_deduction_reason propagate karo — coaching context ke liye useful
            if precomputed_llm.get("score_deduction_reason"):
                llm_feedback.setdefault("score_deduction_reason", precomputed_llm["score_deduction_reason"])

            # Extract LLM deductions from rubric data
            for dim, data in llm_rubric.items():
                if isinstance(data, dict):
                    llm_deductions[dim] = data.get("deductions", [])

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Compute audio delivery scores (override LLM guesses)
        # Audio metrics MEASURED hain (librosa) — LLM sirf guess karta hai.
        # Isliye audio scores > LLM scores priority mein.
        # ═══════════════════════════════════════════════════════════════
        audio_scores, audio_deductions = _compute_audio_delivery_scores(
            audio_features or {}, fillers, word_count
        )

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Compute structural + content heuristic scores
        # Ye backup scores hain — LLM ka score available nahi hoga toh ye use honge.
        # ═══════════════════════════════════════════════════════════════
        struct_scores, struct_deductions = _compute_structure_scores(transcript, semantic)
        content_scores, content_deductions = _compute_content_scores(transcript, semantic)

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Build final rubric breakdown
        # Priority chain: audio > structure > LLM > content_heuristics > default
        # Audio scores sabse zyada reliable hain (measured), isliye top priority.
        # ═══════════════════════════════════════════════════════════════
        rubric_breakdown = []
        weighted_total = 0.0

        for dim, weight in RUBRIC_WEIGHTS.items():
            if dim in audio_scores:
                raw_score = audio_scores[dim]
                dim_deductions = audio_deductions.get(dim, [])
                source = "audio_analysis"
            elif dim in struct_scores:
                raw_score = struct_scores[dim]
                dim_deductions = struct_deductions.get(dim, [])
                source = "text_analysis"
            elif dim in llm_rubric:
                data = llm_rubric[dim]
                raw_score = _clamp(float(data.get("score", 5.0)) if isinstance(data, dict) else float(data))
                dim_deductions = llm_deductions.get(dim, [])
                source = "llm_reasoning"
            elif dim in content_scores:
                raw_score = content_scores[dim]
                dim_deductions = content_deductions.get(dim, [])
                source = "content_analysis"
            else:
                raw_score = 5.0
                dim_deductions = ["Insufficient data to evaluate"]
                source = "default"

            weighted_contribution = round(raw_score * weight, 3)
            weighted_total += weighted_contribution

            reasoning = ""
            if dim in llm_rubric and isinstance(llm_rubric[dim], dict):
                reasoning = llm_rubric[dim].get("reasoning", "")

            rubric_breakdown.append({
                "dimension": dim,
                "label": DIMENSION_LABELS.get(dim, dim),
                "category": DIMENSION_CATEGORIES.get(dim, "Other"),
                "raw_score": raw_score,
                "weight_pct": round(weight * 100),
                "weighted_contribution": weighted_contribution,
                "max_possible": round(10.0 * weight, 3),
                "deductions": dim_deductions if isinstance(dim_deductions, list) else [str(dim_deductions)],
                "reasoning": reasoning,
                "source": source,
            })

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Compute weighted overall score
        # weighted_total already 0-10 scale hai (kyunki weights sum = 1.0).
        # ═══════════════════════════════════════════════════════════════
        overall = _clamp(weighted_total)

        # Resume alignment modifier apply karo (-1.5 to +1.5 range)
        resume_mod = float(resume_alignment.get("score_modifier", 0.0))
        resume_mod = max(-1.5, min(1.5, resume_mod))
        overall = _clamp(overall + resume_mod)

        # ═══════════════════════════════════════════════════════════════
        # STEP 5b: Apply strictness penalty
        # Higher strictness = harsher scoring = lower final score
        # Beginner: +10% lenient boost (encouraging)
        # Intermediate: baseline (no change)
        # Advance: -12% penalty (rigorous)
        # Extreme: -22% penalty (FAANG-level brutal)
        # ═══════════════════════════════════════════════════════════════
        STRICTNESS_MULTIPLIERS = {
            "beginner":      1.10,   # +10% lenient boost
            "intermediate":  1.00,   # baseline
            "advance":       0.88,   # -12% penalty
            "extreme":       0.78,   # -22% penalty  
        }
        strictness_key = strictness.lower().strip() if strictness else "intermediate"
        multiplier = STRICTNESS_MULTIPLIERS.get(strictness_key, 1.0)
        overall = _clamp(overall * multiplier)
        
        logger.info(f"📊 [Feedback] Rubric: {weighted_total:.2f} → After strictness ({strictness_key}): {overall:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: PURE GENERATIVE FEEDBACK — Pass-Through from GenAI Engine
        # ALL feedback text comes from the local AI model.
        # This service only handles scoring, NOT text generation.
        # Format normalize karo — ensure each item is a dict with "text" key.
        # ═══════════════════════════════════════════════════════════════
        positives = []
        improvements = []

        # GenAI Engine se aaye generative positives — normalize to strings
        if llm_feedback.get("positives"):
            for p in llm_feedback["positives"]:
                if isinstance(p, dict):
                    text = p.get("text", str(p))
                elif isinstance(p, str):
                    text = p
                else:
                    text = str(p)
                text = text.strip()
                if text:
                    positives.append(text)

        # GenAI Engine se aaye generative improvements — normalize to strings
        if llm_feedback.get("improvements"):
            for imp in llm_feedback["improvements"]:
                if isinstance(imp, dict):
                    text = imp.get("text", str(imp))
                elif isinstance(imp, str):
                    text = imp
                else:
                    text = str(imp)
                text = text.strip()
                if text:
                    improvements.append(text)

        # Deduplicate (model sometimes repeats in different words)
        positives = self._deduplicate_feedback(positives)
        improvements = self._deduplicate_feedback(improvements)

        # Coaching summary — also fully generative from the LLM
        coaching_summary = llm_feedback.get("coaching_summary", "")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Log feedback counts for debugging
        # Agar positives/improvements empty hain toh ye log se pata chalega
        # ki problem genai_engine mein hai (generation/parsing failed).
        # ═══════════════════════════════════════════════════════════════
        logger.info(
            f"📋 [Feedback] Final counts: {len(positives)} positives, "
            f"{len(improvements)} improvements, "
            f"coaching: {'yes' if coaching_summary else 'no'}, "
            f"resume_matched: {len(resume_alignment.get('matched', []))}, "
            f"resume_missed: {len(resume_alignment.get('missed', []))}"
        )

        return {
            "positives": positives[:12],
            "improvements": improvements[:12],
            "coaching_summary": coaching_summary,
            "rubric_breakdown": rubric_breakdown,
            "overall_rubric_score": overall,
            "resume_alignment": resume_alignment,
            "audio_reasoning": {
                dim: audio_deductions.get(dim, [])
                for dim in ["vocal_confidence", "speech_fluency", "pronunciation_clarity", "speaking_pace", "energy_trajectory"]
            }
        }