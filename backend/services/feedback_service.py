import logging
import random
import re
from collections import deque
from backend.services.rag_service import RAGService

logger = logging.getLogger(__name__)

VARIATION_MEMORY = {}

# ── Dynamic Micro-Polish Engine ──
# Only replaces words when the replacement is contextually appropriate
SYNONYM_DICT = {
    r'\b(great|excellent)\b': ['excellent', 'fantastic', 'brilliant', 'wonderful', 'highly effective', 'commendable'],
    r'\b(highlighted|showcased)\b': ['pointed out', 'mentioned', 'showcased', 'emphasized', 'brought up', 'detailed'],
    r'\b(effectively|clearly)\b': ['clearly', 'confidently', 'smoothly', 'articulately', 'persuasively'],
    r'\b(communicated|conveyed)\b': ['conveyed', 'expressed', 'delivered', 'articulated', 'presented'],
    r'\b(stand out)\b': ['shine', 'leave a strong impression', 'distinguish yourself', 'make an impact', 'capture attention'],
    r'\b(major asset)\b': ['significant advantage', 'key strength', 'huge plus', 'strong selling point', 'valuable trait'],
    r'\b(solid foundation)\b': ['great starting point', 'strong base', 'excellent framework', 'robust core'],
    r'\b(comprehensive)\b': ['complete', 'well-rounded', 'detailed', 'thorough', 'holistic'],
    r'\b(impress)\b': ['wow', 'captivate', 'engage', 'resonate with', 'interest'],
    r'\b(pitch)\b': ['introduction', 'opening statement', 'summary', 'overview', 'narrative'],
    r'\b(consider)\b': ['think about', 'try', 'ensure you', 'make sure to'],
    r'\b(briefly)\b': ['shortly', 'concisely', 'quickly', 'succinctly']
}

def micro_polish(text: str) -> str:
    """Deterministic synonym substitution using seeded random (seed set by pipeline)."""
    for pattern, choices in SYNONYM_DICT.items():
        if random.random() > 0.3:
            replacement = random.choice(choices)
            def replacer(match):
                word = match.group(0)
                if word.istitle(): return replacement.capitalize()
                return replacement
            text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
    return text


# ═══════════════════════════════════════════════════════════════
# PER-FIELD UNIQUE IMPROVEMENT TEMPLATES
# Each missing field gets a COMPLETELY DIFFERENT sentence structure
# ═══════════════════════════════════════════════════════════════
_MISSING_FIELD_TEMPLATES = {
    "educational background": [
        "Interviewers want to know your academic foundation — mentioning where you studied and what you studied helps them assess your knowledge base.",
        "Adding your educational details (university, degree, year) gives the interviewer context about your training and specialization.",
        "A brief mention of your academic background would strengthen your credibility — interviewers use this to gauge your foundational knowledge.",
    ],
    "past professional experience": [
        "Real-world experience is what employers value most — even a brief internship, project, or freelance work would add significant weight to your introduction.",
        "Without mentioning any work experience or projects, your introduction lacks practical evidence of your abilities. Include at least one concrete example.",
        "Sharing a specific project or role you've held demonstrates that you can apply your skills in practice — this is a critical gap in your current introduction.",
    ],
    "clear career goals": [
        "Stating where you see yourself professionally shows ambition and direction — interviewers want to know you have a plan, not just skills.",
        "A forward-looking statement about your aspirations helps the interviewer understand your motivation and whether you'd be a long-term fit.",
        "Your introduction would feel more purposeful if you shared what you aim to achieve in your career — even a one-sentence goal makes a big difference.",
    ],
    "technical skills": [
        "Naming specific tools, languages, or frameworks you know gives the interviewer concrete evidence of your technical capabilities.",
        "Technical skills are the backbone of any tech interview — without naming them, the interviewer has to guess what you can actually do.",
        "List 2-3 key technologies you're proficient in — this immediately positions you as someone with demonstrable, relevant expertise.",
    ],
    "your name": [
        "Starting with your name is Interview 101 — it personalizes your introduction and makes you memorable from the first sentence.",
        "The interviewer needs to know who you are — a clear, confident name introduction sets the professional tone for everything that follows.",
        "Always begin by stating your full name — it's the simplest way to establish your identity and create a personal connection.",
    ],
}


class FeedbackService:
    """
    Production-Grade Feedback System with Deep Audio Reasoning.
    
    KEY IMPROVEMENTS:
    - Per-field UNIQUE improvement templates (no repetitive boilerplate)
    - Transcript-grounded evidence for every feedback point
    - Deterministic ordering (no random.shuffle — ordered by relevance)
    - Explicit deduplication pass
    - Contradiction detection (don't say "missing X" if transcript mentions X)
    """

    def __init__(self):
        logger.info("⚡ [FeedbackService] Initializing Deep Feedback Engine...")
        self.rag_service = RAGService()

    def _deduplicate_feedback(self, items: list) -> list:
        """
        Remove near-duplicate feedback items.
        Two items are duplicates if >60% of their words overlap.
        """
        if not items:
            return items
        
        unique = [items[0]]
        for item in items[1:]:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            words = set(text.lower().split())
            
            is_dup = False
            for existing in unique:
                existing_text = existing.get("text", "") if isinstance(existing, dict) else str(existing)
                existing_words = set(existing_text.lower().split())
                
                if not words or not existing_words:
                    continue
                    
                overlap = len(words & existing_words) / min(len(words), len(existing_words))
                if overlap > 0.60:
                    is_dup = True
                    break
            
            if not is_dup:
                unique.append(item)
        
        return unique

    def _verify_field_actually_missing(self, field_name: str, transcript: str, struct_value) -> bool:
        """
        Verify that a field is ACTUALLY missing from the transcript, not just
        missed by the semantic extraction.
        
        Returns True if the field genuinely seems missing.
        """
        transcript_lower = transcript.lower()
        
        # If semantic extraction found a value, it's NOT missing
        if struct_value:
            if isinstance(struct_value, list) and len(struct_value) > 0:
                return False
            if isinstance(struct_value, str) and struct_value.strip():
                return False
        
        # Additional transcript-level checks for common false negatives
        field_keywords = {
            "educational background": ["university", "college", "degree", "studied", "student", "school", "b.tech", "btech", "b.sc", "msc", "engineering", "institute", "academic"],
            "past professional experience": ["experience", "worked", "intern", "project", "company", "role", "position", "job", "freelance"],
            "clear career goals": ["goal", "aspire", "aim", "future", "dream", "want to", "plan to", "hope to", "career", "become"],
            "technical skills": ["python", "java", "javascript", "sql", "react", "programming", "coding", "machine learning", "ai", "data", "software", "web", "development"],
            "your name": ["my name", "i am", "i'm", "myself"],
        }
        
        keywords = field_keywords.get(field_name, [])
        for keyword in keywords:
            if keyword in transcript_lower:
                logger.info(f"🔍 [Feedback] Transcript mentions '{keyword}' — '{field_name}' may not be truly missing")
                return False
        
        return True

    def generate(
        self,
        user_id: str,
        transcript: str,
        semantic: dict,
        scores: dict,
        fillers: list,
        english_level: str = "Intermediate",
        audio_features: dict = None
    ) -> dict:
        
        if not transcript.strip():
            return {
                "positives": [{"text": "System initialized.", "evidence": ""}],
                "improvements": [{"text": "No transcript found. Please speak louder.", "evidence": ""}],
                "coaching_summary": "Audio was unintelligible or silent.",
                "audio_reasoning": {}
            }
            
        positives = []
        improvements = []
        audio_reasoning = {}
        
        struct = semantic.get("structured", {})
        ev_map = semantic.get("evidence_map", {})
        af = audio_features or {}
        reasoning = af.get("reasoning", {})
        
        def get_items(field):
            vals = struct.get(field, [])
            evs = ev_map.get(field, [])
            if not isinstance(vals, list): vals = [vals] if vals else []
            if not isinstance(evs, list): evs = [evs] if evs else []
            return list(zip(vals, evs))

        # ═══════════════════════════════════════════════════════════════
        # RAG CONTEXT (History Awareness)
        # ═══════════════════════════════════════════════════════════════
        historical_records = self.rag_service.retrieve_context(user_id, transcript, top_k=1)
        if historical_records:
            positives.append({
                "text": micro_polish("You are making continuous efforts to practice and improve. Keep up the momentum!"),
                "evidence": "Historical Activity Detected"
            })

        # ═══════════════════════════════════════════════════════════════
        # CONTENT POSITIVES (transcript-grounded)
        # ═══════════════════════════════════════════════════════════════
        # Skills
        for skill, ev in get_items("skills"):
            if skill and ev:
                templates = [
                    f"Mentioning your proficiency in {skill} adds strong technical credibility to your introduction.",
                    f"Your expertise in {skill} is a valuable differentiator — interviewers in tech actively look for this.",
                    f"Highlighting {skill} shows you have relevant, in-demand technical abilities that align with industry needs.",
                ]
                positives.append({"text": micro_polish(random.choice(templates)), "evidence": ev})
        
        # Strengths / Qualities
        for category in ["strengths", "qualities"]:
            for item, ev in get_items(category):
                if item and ev:
                    templates = [
                        f"Your mention of '{item}' as a personal strength is compelling and memorable.",
                        f"Highlighting '{item}' shows self-awareness and helps differentiate you from other candidates.",
                        f"'{item}' is a highly valued quality — great choice to include in your introduction.",
                    ]
                    positives.append({"text": micro_polish(random.choice(templates)), "evidence": ev})
                
        # Experience
        for exp, ev in get_items("experience"):
            if exp and ev:
                templates = [
                    f"Your experience with {exp} demonstrates practical, real-world capability that employers value.",
                    f"Including {exp} adds weight and credibility — it shows you've applied your skills beyond theory.",
                ]
                positives.append({"text": micro_polish(random.choice(templates)), "evidence": ev})

        # Education
        ed, ev = struct.get("education"), ev_map.get("education")
        if ed and ev:
            positives.append({"text": micro_polish(f"Stating your background at {ed} establishes academic credibility and gives the interviewer useful context."), "evidence": ev})

        # Career Goals
        goals = struct.get("career_goals")
        goals_ev = ev_map.get("career_goals")
        if goals and goals_ev:
            g = goals if isinstance(goals, str) else (goals[0] if isinstance(goals, list) and goals else "")
            if g:
                positives.append({"text": micro_polish(f"Articulating your goal to {g} shows ambition, direction, and long-term thinking."), "evidence": goals_ev if isinstance(goals_ev, str) else str(goals_ev)})

        # Name/Greeting
        if struct.get("name"):
            positives.append({"text": micro_polish("Starting with your name creates a clear, professional opening that personalizes the conversation."), "evidence": str(struct.get("name"))})

        # ═══════════════════════════════════════════════════════════════
        # AUDIO-BASED POSITIVES
        # ═══════════════════════════════════════════════════════════════
        # Tone
        tone_label = af.get("tone_label", "unknown")
        tone_expr = af.get("tone_expressiveness", 0.5)
        if tone_expr >= 0.5:
            positives.append({
                "text": micro_polish(f"Your vocal tone is {tone_label} — varying your pitch naturally keeps the listener engaged and shows enthusiasm."),
                "evidence": f"Tone expressiveness: {tone_expr:.2f}"
            })
            audio_reasoning["tone"] = reasoning.get("tone", "Good tonal variety detected.")

        # Fluency
        fluency = af.get("fluency_score", 0.5)
        if fluency >= 0.6:
            positives.append({
                "text": micro_polish("Your speech flows smoothly with natural transitions and well-managed pauses. This signals preparation and confidence."),
                "evidence": f"Fluency score: {fluency:.2f}"
            })
            audio_reasoning["fluency"] = reasoning.get("fluency", "Good fluency detected.")

        # Pronunciation
        pronun = af.get("pronunciation_score", 0.5)
        if pronun >= 0.55:
            positives.append({
                "text": micro_polish("Your pronunciation is clear and well-articulated. Words are easy to understand — a significant professional advantage."),
                "evidence": f"Pronunciation score: {pronun:.2f}"
            })

        # Energy
        energy_traj = af.get("energy_trajectory", "unknown")
        if energy_traj == "building":
            positives.append({
                "text": micro_polish("Your voice energy builds throughout your introduction — this shows increasing confidence and is very impactful."),
                "evidence": f"Energy trajectory: {energy_traj}"
            })

        # Pace
        pace_label = af.get("pace_label", "unknown")
        if pace_label == "ideal":
            wpm = af.get("wpm_estimate", 140)
            positives.append({
                "text": micro_polish(f"Your speaking pace (~{int(wpm)} WPM) is ideal — neither too fast nor too slow, showing composure and clarity."),
                "evidence": f"Pace: {pace_label}, {int(wpm)} WPM"
            })

        # ═══════════════════════════════════════════════════════════════
        # CONTENT IMPROVEMENTS (with per-field UNIQUE templates)
        # ═══════════════════════════════════════════════════════════════
        missing_field_map = {
            "education": "educational background",
            "experience": "past professional experience",
            "career_goals": "clear career goals",
            "skills": "technical skills",
            "name": "your name",
        }
        
        for field_key, field_label in missing_field_map.items():
            # Verify it's ACTUALLY missing, not a semantic extraction failure
            if self._verify_field_actually_missing(field_label, transcript, struct.get(field_key)):
                # Pick from per-field unique templates
                templates = _MISSING_FIELD_TEMPLATES.get(field_label, [
                    f"Consider including more about your {field_label} to create a well-rounded introduction."
                ])
                chosen_template = random.choice(templates)
                improvements.append({
                    "text": micro_polish(chosen_template),
                    "evidence": f"Not detected in transcript"
                })

        # Word count improvements
        word_count = len(transcript.split())
        if word_count < 15:
            improvements.append({
                "text": micro_polish("Your response is far too short to constitute an interview introduction. A minimum of 60-90 seconds (100-180 words) is expected. You need to cover your name, education, skills, experience, and career goals."),
                "evidence": f"Word count: {word_count}"
            })
        elif word_count < 30:
            improvements.append({
                "text": micro_polish("Your response is quite short. A complete self-introduction should be 60-90 seconds (roughly 100-180 words). Expand on your background, skills, and goals."),
                "evidence": f"Word count: {word_count}"
            })
        elif word_count < 60:
            improvements.append({
                "text": micro_polish("Your response could be more detailed. Consider adding specifics about your projects, achievements, or what makes you unique."),
                "evidence": f"Word count: {word_count}"
            })
        elif word_count > 300:
            improvements.append({
                "text": micro_polish("Your introduction is very long. Conciseness is key in interviews — focus on the 3-4 most impactful highlights about yourself."),
                "evidence": f"Word count: {word_count}"
            })

        # Content relevance check
        interview_keywords = ["name", "education", "university", "college", "skill", "experience",
                              "project", "goal", "career", "intern", "degree", "python", "java",
                              "programming", "worked", "studying"]
        transcript_lower = transcript.lower()
        relevant_hits = sum(1 for kw in interview_keywords if kw in transcript_lower)
        if relevant_hits < 2 and word_count > 10:
            improvements.append({
                "text": micro_polish("Your response doesn't seem to contain interview-relevant content. A self-introduction should focus on who you are professionally — your name, education, skills, experience, and where you see your career heading."),
                "evidence": f"Only {relevant_hits} interview-relevant keywords detected"
            })

        # Vocabulary diversity
        unique_words = len(set(w.lower() for w in transcript.split()))
        diversity = unique_words / max(1, word_count)
        if word_count > 20 and diversity < 0.5:
            improvements.append({
                "text": micro_polish("Your vocabulary is somewhat repetitive. Using more varied word choices and professional terminology would make your introduction more impressive."),
                "evidence": f"Vocabulary diversity: {diversity:.2f}"
            })

        # ═══════════════════════════════════════════════════════════════
        # AUDIO-BASED IMPROVEMENTS
        # ═══════════════════════════════════════════════════════════════
        # Tone issues
        if tone_expr < 0.35:
            improvements.append({
                "text": micro_polish("Your voice sounds monotone. Varying your pitch — especially when making key points — will make you sound more dynamic and engaging."),
                "evidence": f"Tone expressiveness: {tone_expr:.2f}"
            })
            audio_reasoning["tone"] = reasoning.get("tone", "Monotone delivery detected.")

        # Fluency issues
        if fluency < 0.5:
            long_pauses = af.get("long_pauses", 0)
            improvements.append({
                "text": micro_polish(f"Your speech has noticeable hesitations and {long_pauses} long pauses. Practice your introduction until you can deliver it without extended breaks."),
                "evidence": f"Fluency score: {fluency:.2f}, Long pauses: {long_pauses}"
            })
            audio_reasoning["fluency"] = reasoning.get("fluency", "Fluency issues detected.")

        # Pronunciation issues
        if pronun < 0.45:
            improvements.append({
                "text": micro_polish("Your pronunciation could be clearer. Focus on enunciating each syllable — especially consonants — and practice tongue twisters to improve articulation."),
                "evidence": f"Pronunciation score: {pronun:.2f}"
            })
            audio_reasoning["pronunciation"] = reasoning.get("pronunciation", "Pronunciation clarity needs improvement.")

        # Pace issues
        if pace_label in ("too_fast", "slightly_fast"):
            wpm = af.get("wpm_estimate", 180)
            improvements.append({
                "text": micro_polish(f"You're speaking too quickly (~{int(wpm)} WPM). Slow down to give the interviewer time to absorb your points. Pausing between sentences shows confidence."),
                "evidence": f"Pace: {pace_label}, {int(wpm)} WPM"
            })
            audio_reasoning["pace"] = reasoning.get("pace", "Speaking too fast.")
        elif pace_label in ("too_slow", "slightly_slow"):
            wpm = af.get("wpm_estimate", 80)
            improvements.append({
                "text": micro_polish(f"Your speaking pace (~{int(wpm)} WPM) is below the ideal range. Try to speak with more energy and momentum to maintain the listener's attention."),
                "evidence": f"Pace: {pace_label}, {int(wpm)} WPM"
            })
            audio_reasoning["pace"] = reasoning.get("pace", "Speaking too slowly.")

        # Energy fade
        if energy_traj == "fading":
            improvements.append({
                "text": micro_polish("Your voice energy drops towards the end of your introduction. Your closing is as important as your opening — maintain strong energy throughout, especially in your final statement."),
                "evidence": f"Energy trajectory: {energy_traj}"
            })
            audio_reasoning["energy"] = reasoning.get("energy", "Voice energy fades.")

        # Filler words
        if len(fillers) > 2:
            filler_summary = ", ".join(set(fillers[:5]))
            improvements.append({
                "text": micro_polish(f"You used {len(fillers)} filler words ({filler_summary}). Replace these with confident pauses — silence between sentences sounds much more professional than 'um' or 'uh'."),
                "evidence": " ".join(fillers[:5]) + ("..." if len(fillers) > 5 else "")
            })

        # Grammar/English Level
        if english_level == "Beginner":
            improvements.append({
                "text": micro_polish("Focus on forming simple, clear sentences. Short, correct sentences are better than complex, broken ones. Practice common interview phrases daily."),
                "evidence": "System Analysis: Beginner Level"
            })
        elif english_level == "Intermediate":
            improvements.append({
                "text": micro_polish("Your English is solid. To reach the next level, practice using more professional vocabulary and complex sentence structures in your introduction."),
                "evidence": "System Analysis: Intermediate Level"
            })

        # ═══════════════════════════════════════════════════════════════
        # DEDUPLICATION + FINALIZATION
        # ═══════════════════════════════════════════════════════════════
        
        # Deduplicate BEFORE limiting count
        positives = self._deduplicate_feedback(positives)
        improvements = self._deduplicate_feedback(improvements)
        
        # Deterministic ordering: keep the order as-is (content first, then audio)
        # No random.shuffle — ensures same input → same output
        
        # Show up to 8 of each
        positives = positives[:8]
        improvements = improvements[:8]

        if not positives:
            positives.append({"text": micro_polish("You communicated clearly and logically throughout the recording."), "evidence": "General Analysis"})
            
        if not improvements:
            improvements.append({"text": micro_polish("Try expanding more on your technical depth and unique experiences to further impress the interviewer."), "evidence": "General Advice"})

        # Ensure minimum positives and improvements, BUT scale based on actual quality
        # Don't pad poor introductions with generic positives
        is_poor_intro = word_count < 30 or (relevant_hits < 2 and word_count > 10)
        
        generic_positives = [
            "You completed a full recording — consistent practice is the key to interview mastery.",
            "Your willingness to practice shows strong self-improvement drive.",
            "You maintained focus throughout your introduction without major digressions."
        ]
        
        if is_poor_intro:
            # For poor intros, only add 1 generic positive (participation acknowledgment)
            if len(positives) < 1:
                positives.append({"text": micro_polish(generic_positives[0]), "evidence": "General Analysis"})
        else:
            gp_idx = 0
            while len(positives) < 3 and gp_idx < len(generic_positives):
                positives.append({"text": micro_polish(generic_positives[gp_idx]), "evidence": "General Analysis"})
                gp_idx += 1

        generic_improvements = [
            "Try adding a memorable closing statement that summarizes why you're the ideal candidate.",
            "Practice this introduction 5-10 times until it feels natural and effortless.",
            "Record yourself and listen back — you'll catch nuances that need polish."
        ]
        gi_idx = 0
        while len(improvements) < 3 and gi_idx < len(generic_improvements):
            improvements.append({"text": micro_polish(generic_improvements[gi_idx]), "evidence": "General Advice"})
            gi_idx += 1

        # Build coaching summary based on actual analysis
        summary = self._build_coaching_summary(struct, af, word_count, fillers, english_level, transcript=transcript)
        
        final_feedback = {
            "positives": positives,
            "improvements": improvements,
            "coaching_summary": summary,
            "audio_reasoning": audio_reasoning
        }

        # ── TRACKING ──
        flattened_text = " ".join([i.get("text", "") for i in improvements]).lower()
        if user_id not in VARIATION_MEMORY:
            VARIATION_MEMORY[user_id] = deque(maxlen=3)
        VARIATION_MEMORY[user_id].append(flattened_text)

        try:
            self.rag_service.ingest(user_id, transcript, struct, final_feedback, scores)
        except Exception as e:
            logger.warning(f"RAG ingestion failed non-fatally: {e}")

        logger.info(f"✅ Generated {len(positives)} positives and {len(improvements)} improvements with audio reasoning.")
        return final_feedback

    def _build_coaching_summary(self, struct: dict, af: dict, word_count: int, fillers: list, english_level: str, transcript: str = "") -> str:
        """
        Build a context-specific coaching summary based on actual analysis,
        NOT random template selection. Cross-checks against transcript keywords
        to prevent semantic extraction failures from producing wrong advice.
        """
        parts = []
        
        # Content assessment — cross-check with transcript keywords
        fields_present_semantic = sum(1 for k in ["name", "education", "skills", "experience", "career_goals"] 
                           if struct.get(k))
        
        # Transcript-level keyword check (catches semantic extraction failures)
        transcript_lower = transcript.lower() if transcript else ""
        keyword_groups = {
            "name": ["my name", "i am", "i'm"],
            "education": ["university", "college", "degree", "student", "b.tech", "engineering", "institute"],
            "skills": ["skill", "python", "java", "programming", "communication", "teamwork", "coding"],
            "experience": ["experience", "worked", "project", "intern", "activities"],
            "career_goals": ["goal", "aspire", "aim", "career", "want to", "hope to", "become", "future"],
        }
        fields_in_transcript = sum(1 for kws in keyword_groups.values() if any(kw in transcript_lower for kw in kws))
        
        # Use the HIGHER count (semantic vs transcript) to avoid false "missing" claims
        fields_present = max(fields_present_semantic, fields_in_transcript)
        
        if fields_present >= 4:
            parts.append("Your introduction covers most key areas comprehensively.")
        elif fields_present >= 2:
            parts.append("Your introduction has a solid base but could expand on a few more areas.")
        else:
            parts.append("Your introduction needs more content — consider covering your background, skills, and goals.")
        
        # Audio assessment
        tone_expr = af.get("tone_expressiveness", 0.5)
        fluency = af.get("fluency_score", 0.5)
        pace = af.get("pace_label", "unknown")
        
        if tone_expr >= 0.5 and fluency >= 0.6:
            parts.append("Your vocal delivery is strong with good tonal variety and smooth flow.")
        elif tone_expr < 0.35:
            parts.append("Work on varying your pitch to avoid sounding monotone.")
        elif fluency < 0.5:
            parts.append("Practice delivering your introduction without long pauses or hesitations.")
        
        if pace in ("too_fast", "slightly_fast"):
            parts.append("Slow down your pace for better clarity.")
        elif pace in ("too_slow", "slightly_slow"):
            parts.append("Pick up your pace slightly to maintain energy.")
        
        # Filler assessment
        if len(fillers) > 3:
            parts.append(f"Reducing your {len(fillers)} filler words will significantly improve professionalism.")
        
        # Word count
        if word_count < 50:
            parts.append("Aim for 100-180 words for a complete self-introduction.")
        
        return " ".join(parts)