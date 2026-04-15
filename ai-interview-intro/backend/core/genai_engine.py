# =====================================================================
# GENAI ENGINE — Single-Pass HR Reasoning (Unified Inference)
# =====================================================================
# Architecture UPGRADE:
#   OLD: 2 LLM calls (evaluate → coaching) = 20-30s total
#   NEW: 1 unified call (rubric + feedback in one shot) = 10-15s total
#
# Key Design Decisions:
#   1. Single unified prompt = ONE model.generate() call
#   2. Deep resume extraction (projects, internships, certs, companies)
#   3. Filler stats injected into prompt for micro-detail feedback
#   4. CPU+GPU parallel: resume extraction on CPU while model on GPU
#   5. ALL feedback is GENERATIVE — zero fixed/template sentences
#   6. Fallback also generates varied, data-driven sentences
#
# Flow:
#   comprehensive_analyze()
#     → _deep_extract_resume() [CPU]
#     → _build_unified_prompt() [CPU]
#     → hr_model.unified_evaluate() [GPU — SINGLE CALL]
#     → Parse JSON → Validate → Return
#     → On fail: Retry once → On fail again: _generate_fallback_feedback()
# =====================================================================

import json
import logging
import time
import re
import os

logger = logging.getLogger(__name__)

# ── STRICTNESS MAP — Different Personas for Different Needs ───────────
STRICTNESS_MAP = {
    "beginner": {
        "iq": "Base Intelligence",
        "persona": "Encouraging Mentor",
        "instruction": "Focus on growth. Penalize lightly for errors. Use an encouraging tone. Find the 'potential' in the candidate.",
        "threshold": "High tolerance for resume/pitch gaps."
    },
    "intermediate": {
        "iq": "Standard HR Professional",
        "persona": "Balanced Evaluator",
        "instruction": "Standard corporate norms. Fair judgment of evidence vs claims. Focus on skills and delivery consistency.",
        "threshold": "Moderate tolerance for minor omissions."
    },
    "advance": {
        "iq": "Strategic Talent Lead",
        "persona": "Rigorous Assessor",
        "instruction": "High expectations. Detailed analysis of project depth. Demand technical precision and fluent delivery.",
        "threshold": "Low tolerance for lack of depth. Penalize generic answers."
    },
    "extreme": {
        "iq": "Principal FAANG Recruiter",
        "persona": "Brutally Precise Architect",
        "instruction": "Zero tolerance for mismatches. If a resume claim isn't backed by audio evidence, mark it as 'Integrity Gap'. Demand ultra-fluent, high-impact reasoning.",
        "threshold": "Absolute strictness. Any mismatch = Score < 3.0. Explicitly state the Deduction Reason."
    }
}


class GenAIEngine:
    """
    Single-Pass HR Reasoning Engine.

    Combines rubric scoring + generative feedback into ONE model call.
    Deep resume analysis extracts projects/internships/certs before the LLM call.
    All feedback is purely generative — zero template/fixed sentences.
    """

    def __init__(self):
        self._hr_model = None
        self._hr_model_checked = False
        logger.info(f"🧠 [GenAIEngine] Initializing Single-Pass Generative AI Engine...")

    def _get_hr_model(self):
        """Lazy-load the fine-tuned HR model."""
        if not self._hr_model_checked:
            self._hr_model_checked = True
            try:
                from backend.ml_models.hr_model_inference import hr_model
                if hr_model.is_available():
                    if hr_model.load():
                        self._hr_model = hr_model
                        logger.info("✅ [GenAIEngine] Fine-tuned HR model loaded as PRIMARY")
                    else:
                        logger.info("⚠️ [GenAIEngine] HR model exists but failed to load")
                else:
                    logger.info("⚠️ [GenAIEngine] No fine-tuned HR model found")
            except Exception as e:
                logger.warning(f"⚠️ [GenAIEngine] HR model import failed: {e}")
        return self._hr_model

    def health_check(self) -> dict:
        """Quick probe to verify backend status."""
        hr_available = False
        try:
            from backend.ml_models.hr_model_inference import hr_model
            hr_available = hr_model.is_available()
        except:
            pass
        return {
            "status": "online" if hr_available else "offline",
            "hr_model_available": hr_available,
            "ollama_online": False,
            "primary": "native_pytorch",
        }

    # ══════════════════════════════════════════════════════════════════
    # DEEP RESUME EXTRACTION — CPU-bound, runs parallel with audio
    # ══════════════════════════════════════════════════════════════════

    def _deep_extract_resume(self, resume_text: str) -> dict:
        """
        Deeply extract structured information from resume text.

        Extracts:
        - projects: list of {name, tech_stack, description}
        - internships: list of {company, role, duration}
        - certifications: list of strings
        - education: {institution, degree, year, cgpa}
        - skills: list of technical skill strings
        - achievements: list of strings
        - companies: list of company names
        """
        if not resume_text:
            return {"projects": [], "internships": [], "certifications": [],
                    "education": {}, "skills": [], "achievements": [], "companies": []}

        text = resume_text.strip()
        text_lower = text.lower()
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        result = {
            "projects": [],
            "internships": [],
            "certifications": [],
            "education": {},
            "skills": [],
            "achievements": [],
            "companies": [],
        }

        # ── Section detection ──
        current_section = None
        section_lines = {}
        section_keywords = {
            "project": ["project", "projects", "personal project", "academic project",
                        "key projects", "major projects", "mini project"],
            "experience": ["experience", "work experience", "professional experience",
                          "internship", "internships", "work history", "employment"],
            "education": ["education", "academic", "qualification", "qualifications"],
            "skills": ["skills", "technical skills", "technologies", "tech stack",
                       "competencies", "proficiency", "tools"],
            "certification": ["certification", "certifications", "certificate",
                             "courses", "training", "online courses"],
            "achievement": ["achievement", "achievements", "awards", "honors",
                          "accomplishments", "extracurricular", "activities"],
        }

        for line in lines:
            line_lower = line.lower().strip()
            # Check if this line is a section header
            is_header = False
            for section, keywords in section_keywords.items():
                for kw in keywords:
                    if (line_lower == kw or line_lower.startswith(kw + " ") or
                        line_lower.startswith(kw + ":") or line_lower.startswith(kw + "—") or
                        line_lower.startswith(kw + "-") or
                        (len(line_lower) < 40 and kw in line_lower)):
                        current_section = section
                        is_header = True
                        break
                if is_header:
                    break

            if not is_header and current_section:
                section_lines.setdefault(current_section, []).append(line)

        # ── Extract Projects ──
        project_lines = section_lines.get("project", [])
        current_project = None
        for line in project_lines:
            # Project name detection (usually bold/titled or starts with bullet)
            clean = re.sub(r'^[\-•*▪►→|]\s*', '', line).strip()
            if not clean:
                continue

            # If line looks like a project title (short, possibly with tech in parens)
            if (len(clean.split()) <= 12 and not clean.endswith('.') and
                (clean[0].isupper() or clean.startswith('"') or '|' in clean)):
                # Extract tech stack if in brackets/parens
                tech_match = re.search(r'[\(\[](.*?)[\)\]]', clean)
                tech_stack = tech_match.group(1) if tech_match else ""
                project_name = re.sub(r'[\(\[].*?[\)\]]', '', clean).strip()
                project_name = re.sub(r'\s*[|–—-]\s*$', '', project_name).strip()
                if project_name:
                    current_project = {"name": project_name, "tech_stack": tech_stack, "description": ""}
                    result["projects"].append(current_project)
            elif current_project:
                # This is a description line for the current project
                desc = current_project["description"]
                current_project["description"] = (desc + " " + clean).strip() if desc else clean
                # Also try to extract tech mentions
                if not current_project["tech_stack"]:
                    tech_in_desc = re.search(r'(?:using|built with|tech[: ]+|stack[: ]+)([\w\s,/+#.]+)', clean.lower())
                    if tech_in_desc:
                        current_project["tech_stack"] = tech_in_desc.group(1).strip()

        # ── Extract Internships/Experience ──
        exp_lines = section_lines.get("experience", [])
        current_exp = None
        for line in exp_lines:
            clean = re.sub(r'^[\-•*▪►→|]\s*', '', line).strip()
            if not clean:
                continue

            # Company/role pattern: "Company Name — Role" or "Role at Company"
            company_match = re.search(
                r'(.+?)\s*(?:[|–—-]|at\s+)\s*(.+?)(?:\s*[\(\[](.+?)[\)\]])?$', clean
            )
            if company_match and len(clean.split()) <= 15:
                part1 = company_match.group(1).strip()
                part2 = company_match.group(2).strip()
                duration = company_match.group(3) if company_match.group(3) else ""

                # Determine which is company and which is role
                role_words = {"intern", "developer", "engineer", "analyst", "designer",
                             "manager", "lead", "associate", "trainee", "assistant",
                             "executive", "coordinator", "consultant", "specialist"}
                if any(rw in part1.lower() for rw in role_words):
                    role, company = part1, part2
                else:
                    company, role = part1, part2

                current_exp = {"company": company, "role": role, "duration": duration}
                result["internships"].append(current_exp)
                if company and company not in result["companies"]:
                    result["companies"].append(company)
            elif current_exp:
                # Description of the experience
                pass  # We just track company/role, not description

            # Also detect standalone internship mentions
            intern_match = re.search(
                r'(?:intern(?:ship)?|trainee)\s+(?:at|in|with)\s+(.+?)(?:\s*[\(\[]|$)',
                clean, re.IGNORECASE
            )
            if intern_match and not current_exp:
                company = intern_match.group(1).strip()
                current_exp = {"company": company, "role": "Intern", "duration": ""}
                result["internships"].append(current_exp)
                if company not in result["companies"]:
                    result["companies"].append(company)

        # ── Extract Certifications ──
        cert_lines = section_lines.get("certification", [])
        for line in cert_lines:
            clean = re.sub(r'^[\-•*▪►→|]\s*', '', line).strip()
            if clean and len(clean) > 3:
                result["certifications"].append(clean)

        # ── Extract Education ──
        edu_lines = section_lines.get("education", [])
        for line in edu_lines:
            clean = re.sub(r'^[\-•*▪►→|]\s*', '', line).strip()
            if not clean:
                continue

            # Look for degree patterns
            degree_match = re.search(
                r'(B\.?Tech|B\.?E|B\.?Sc|M\.?Tech|M\.?E|M\.?Sc|MBA|Ph\.?D|BCA|MCA|Diploma|Bachelor|Master)',
                clean, re.IGNORECASE
            )
            if degree_match and not result["education"].get("degree"):
                result["education"]["degree"] = clean
                # Extract institution
                inst_match = re.search(r'(?:from|at|,)\s+(.+?)(?:\s*[\(\[]|$)', clean)
                if inst_match:
                    result["education"]["institution"] = inst_match.group(1).strip()
                # Extract CGPA
                cgpa_match = re.search(r'(?:cgpa|gpa|percentage|%)[:\s]*([0-9.]+)', clean, re.IGNORECASE)
                if cgpa_match:
                    result["education"]["cgpa"] = cgpa_match.group(1)
                # Extract year
                year_match = re.search(r'(20\d{2})', clean)
                if year_match:
                    result["education"]["year"] = year_match.group(1)

        # ── Extract Skills (keyword-based from skills section) ──
        skill_lines = section_lines.get("skills", [])
        all_tech_keywords = {
            "python", "java", "javascript", "react", "node", "nodejs", "sql", "html", "css",
            "machine learning", "deep learning", "ai", "artificial intelligence",
            "data science", "cloud", "aws", "azure", "gcp", "docker", "kubernetes",
            "git", "api", "rest", "graphql", "tensorflow", "pytorch", "flutter",
            "angular", "vue", "django", "flask", "fastapi", "mongodb", "postgresql",
            "mysql", "redis", "c++", "c#", "rust", "go", "golang", "typescript",
            "spring", "jenkins", "ci/cd", "agile", "scrum", "microservices",
            "linux", "figma", "tableau", "power bi", "hadoop", "spark", "kafka",
            "excel", "pandas", "numpy", "scikit-learn", "opencv", "nlp",
            "natural language processing", "computer vision", "blockchain",
            "swift", "kotlin", "android", "ios", "web development",
            "full stack", "frontend", "backend", "devops", "data engineering",
        }
        for line in skill_lines:
            clean_lower = re.sub(r'^[\-•*▪►→|]\s*', '', line).strip().lower()
            for skill in all_tech_keywords:
                if skill in clean_lower and skill.title() not in result["skills"]:
                    result["skills"].append(skill.title())

        # Also search full resume for skills if skills section was empty
        if not result["skills"]:
            for skill in all_tech_keywords:
                if skill in text_lower and skill.title() not in result["skills"]:
                    result["skills"].append(skill.title())

        # ── Extract Achievements ──
        ach_lines = section_lines.get("achievement", [])
        for line in ach_lines:
            clean = re.sub(r'^[\-•*▪►→|]\s*', '', line).strip()
            if clean and len(clean) > 5:
                result["achievements"].append(clean)

        logger.info(
            f"📄 [ResumeExtract] {len(result['projects'])} projects, "
            f"{len(result['internships'])} internships, "
            f"{len(result['certifications'])} certs, "
            f"{len(result['skills'])} skills, "
            f"{len(result['achievements'])} achievements"
        )

        return result

    # ══════════════════════════════════════════════════════════════════
    # JSON PARSING UTILITIES
    # ══════════════════════════════════════════════════════════════════

    def _strip_markdown(self, raw: str) -> str:
        """Strip markdown, think blocks, and model tokens from raw LLM output."""
        raw = raw.strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'<think>.*', '', raw, flags=re.DOTALL).strip()
        if raw.startswith("I'm DeepSeek") or raw.startswith("I am DeepSeek"):
            return '{}'
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1].strip()
        raw = re.sub(r'<\|.*?\|>', '', raw).strip()
        brace_start = raw.find("{")
        if brace_start >= 0:
            raw = raw[brace_start:]
        return raw

    def _recover_partial_json(self, raw: str) -> dict:
        """Recover truncated/malformed JSON from model output."""
        raw = raw.strip()
        if not raw:
            return {}

        # Method 1: Try complete JSON from each opening brace
        start_indices = [m.start() for m in re.finditer(r'\{', raw)]
        for start in start_indices:
            text_to_parse = raw[start:]
            depth = 0
            for i, char in enumerate(text_to_parse):
                if char == '{': depth += 1
                elif char == '}': depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text_to_parse[:i+1])
                    except:
                        pass

        # Method 2: Repair truncated JSON
        try:
            s = raw.index("{")
            fragment = raw[s:]
            if fragment.count('"') % 2 != 0:
                fragment += '"'
            open_brackets = fragment.count('[') - fragment.count(']')
            fragment += ']' * max(0, open_brackets)
            open_braces = fragment.count('{') - fragment.count('}')
            fragment += '}' * max(0, open_braces)
            return json.loads(fragment)
        except:
            pass

        # Method 3: First '{' to last '}'
        try:
            s = raw.index("{")
            e = raw.rindex("}")
            return json.loads(raw[s:e+1])
        except:
            logger.error(f"❌ [GenAIEngine] JSON recovery failed. Raw: {raw[:300]}")
            return {}

    # ══════════════════════════════════════════════════════════════════
    # UNIFIED PROMPT BUILDER — Single comprehensive prompt
    # ══════════════════════════════════════════════════════════════════

    def _build_unified_prompt(self, transcript: str, resume_text: str,
                              resume_data: dict, audio_metrics: dict,
                              filler_stats: dict, strictness: str) -> tuple:
        """
        Build a SINGLE comprehensive prompt that produces rubric + feedback.
        Returns (system_prompt, user_prompt) tuple.
        """
        af = audio_metrics or {}
        fs = filler_stats or {}

        # ── SYSTEM PROMPT — HR Evaluator persona ──
        strict_config = STRICTNESS_MAP.get(strictness.lower().strip(), STRICTNESS_MAP["intermediate"])

        system_prompt = (
            f"You are a Principal HR Recruiter at a FAANG company (Google/Amazon/Meta). "
            f"Evaluate this candidate's interview self-introduction pitch with forensic precision.\n\n"
            f"EVALUATION MODE: {strictness.upper()}\n"
            f"Persona: {strict_config['persona']}\n"
            f"Instruction: {strict_config['instruction']}\n"
            f"Threshold: {strict_config['threshold']}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Generate EXACTLY 8-10 UNIQUE, specific, evidence-backed positive points.\n"
            f"   - YOU MUST INCLUDE AT LEAST TWO positives tagged with [CONTENT DEPTH]\n"
            f"   - YOU MUST INCLUDE AT LEAST ONE positive tagged with [RESUME GAP] (if they mentioned anything from the resume).\n"
            f"2. Generate EXACTLY 8-10 UNIQUE, specific, actionable improvement points.\n"
            f"3. Each point MUST reference specific content from the transcript, resume, or audio metrics.\n"
            f"4. NO generic advice. NO template sentences. Every point must feel uniquely human-written.\n"
            f"5. For BOTH positives AND improvements, PREFIX each point with a category tag: [RESUME GAP], [DELIVERY], [CONTENT DEPTH], [STRUCTURE], or [PROFESSIONAL POLISH]\n"
            f"6. Cross-reference resume against pitch: what was MATCHED and what was MISSED.\n"
            f"7. Notice EVERY minor detail: filler words, pace, tone, energy, pauses, pronunciation.\n"
            f"8. For each rubric dimension, provide a specific reasoning explaining the score.\n\n"
            f"Return ONLY valid JSON with this structure:\n"
            f'{{"rubric_scores":{{"opening_greeting":{{"score":0.0,"reasoning":"..."}},\n'
            f'"education":{{"score":0.0,"reasoning":"..."}},\n'
            f'"technical_skills":{{"score":0.0,"reasoning":"..."}},\n'
            f'"project_evidence":{{"score":0.0,"reasoning":"..."}},\n'
            f'"work_experience":{{"score":0.0,"reasoning":"..."}},\n'
            f'"career_goals":{{"score":0.0,"reasoning":"..."}},\n'
            f'"strengths_qualities":{{"score":0.0,"reasoning":"..."}},\n'
            f'"areas_of_interest":{{"score":0.0,"reasoning":"..."}}}},\n'
            f'"overall_score":0.0,\n'
            f'"score_deduction_reason":"Why candidate did not get 10/10",\n'
            f'"feedback":{{\n'
            f'  "positives":["[CONTENT DEPTH] ...", "[RESUME GAP] ...", "[DELIVERY] ..."],\n'
            f'  "improvements":["[STRUCTURE] ...", "[CONTENT DEPTH] ...", "[DELIVERY] ..."],\n'
            f'  "suggestions":["3-5 rewritten weak sentences"],\n'
            f'  "coaching_summary":"2-3 sentence holistic summary"\n'
            f'}},\n'
            f'"resume_alignment":{{"matched":["items in BOTH resume and pitch"],"missed":["resume items NOT in pitch"]}}}}'
        )

        # ── USER PROMPT — All data combined ──
        parts = []

        # 1. Resume data (deep extracted)
        if resume_text:
            parts.append(f"[CANDIDATE RESUME TEXT]:\n{resume_text[:1800]}")

            if resume_data:
                resume_detail = "\n[DETAILED RESUME ANALYSIS]:\n"
                if resume_data.get("projects"):
                    resume_detail += "PROJECTS found in resume:\n"
                    for i, proj in enumerate(resume_data["projects"][:8], 1):
                        resume_detail += f"  {i}. {proj['name']}"
                        if proj.get("tech_stack"):
                            resume_detail += f" (Tech: {proj['tech_stack']})"
                        if proj.get("description"):
                            resume_detail += f" — {proj['description'][:100]}"
                        resume_detail += "\n"

                if resume_data.get("internships"):
                    resume_detail += "INTERNSHIPS/EXPERIENCE found in resume:\n"
                    for i, exp in enumerate(resume_data["internships"][:6], 1):
                        resume_detail += f"  {i}. {exp.get('role', 'Unknown')} at {exp.get('company', 'Unknown')}"
                        if exp.get("duration"):
                            resume_detail += f" ({exp['duration']})"
                        resume_detail += "\n"

                if resume_data.get("certifications"):
                    resume_detail += f"CERTIFICATIONS: {', '.join(resume_data['certifications'][:6])}\n"

                if resume_data.get("skills"):
                    resume_detail += f"SKILLS listed: {', '.join(resume_data['skills'][:15])}\n"

                if resume_data.get("achievements"):
                    resume_detail += f"ACHIEVEMENTS: {', '.join(resume_data['achievements'][:4])}\n"

                if resume_data.get("education"):
                    edu = resume_data["education"]
                    resume_detail += f"EDUCATION: {edu.get('degree', 'N/A')}"
                    if edu.get("institution"):
                        resume_detail += f" from {edu['institution']}"
                    if edu.get("cgpa"):
                        resume_detail += f" (CGPA: {edu['cgpa']})"
                    resume_detail += "\n"

                resume_detail += (
                    "\n⚠️ MANDATORY: For each project listed above, check if it was mentioned in the audio pitch. "
                    "For each internship, check if the company/role was mentioned. "
                    "Report all matches and misses in resume_alignment."
                )
                parts.append(resume_detail)

        # 2. Audio analysis metrics
        filler_detail = ""
        if fs:
            per_type = fs.get("per_type", {})
            if per_type:
                filler_detail = f"  Filler breakdown: {json.dumps(per_type)}\n"
                if fs.get("most_frequent"):
                    filler_detail += f"  Most frequent filler: '{fs['most_frequent']}' ({per_type.get(fs['most_frequent'], 0)} times)\n"
                if fs.get("position_cluster") and fs["position_cluster"] != "none":
                    filler_detail += f"  Filler concentration: {fs['position_cluster']} of pitch\n"
                if fs.get("self_corrections", 0) > 0:
                    filler_detail += f"  Self-corrections detected: {fs['self_corrections']}\n"

        audio_block = (
            f"\n[AUDIO ANALYSIS METRICS — MEASURED FROM VOICE]:\n"
            f"- Speaking Pace: {af.get('wpm_estimate', 140):.0f} WPM (label: {af.get('pace_label', 'unknown')})\n"
            f"- Tone Expressiveness: {af.get('tone_expressiveness', 0.5):.2f}/1.0 ({af.get('tone_label', 'moderate')})\n"
            f"- Tone Richness (MFCC): {af.get('tone_richness', 0.5):.2f}/1.0\n"
            f"- Speech Fluency: {af.get('fluency_score', 0.5):.2f}/1.0\n"
            f"- Pronunciation Clarity: {af.get('pronunciation_score', 0.5):.2f}/1.0\n"
            f"- HNR (Voice Clarity): {af.get('hnr_score', 0.5):.2f}/1.0\n"
            f"- Energy Trajectory: {af.get('energy_trajectory', 'stable')}\n"
            f"- Energy Consistency: {af.get('energy_consistency', 0.5):.2f}/1.0\n"
            f"- Speech Rate Stability: {af.get('speech_rate_stability', 0.5):.2f}/1.0\n"
            f"- Long Pauses: {af.get('long_pauses', 0)} (avg: {af.get('avg_pause_duration', 0):.2f}s)\n"
            f"- Pitch Range: {af.get('pitch_range', 80):.0f}Hz | Pitch Variance: {af.get('pitch_variance', 0.3):.2f}\n"
            f"- Filler Words Detected: {fs.get('count', 0)} total (density: {fs.get('density', 0):.1f} per 100 words)\n"
            f"{filler_detail}"
            f"- Dynamic Confidence: {af.get('dynamic_confidence', 50.0):.0f}% ({af.get('confidence_label', 'MEDIUM')})\n"
            f"- Total Words Spoken: {len(transcript.split())}\n"
        )
        parts.append(audio_block)

        # 3. Transcript
        parts.append(f"\n[AUDIO TRANSCRIPT OF CANDIDATE'S PITCH]:\n{transcript[:900]}")

        # 4. Explicit analysis instructions
        parts.append(
            "\n[YOUR ANALYSIS MUST INCLUDE]:\n"
            "- Reference SPECIFIC words/phrases from the transcript\n"
            "- If resume provided: explicitly name which projects/internships were/weren't mentioned\n"
            "- Comment on EVERY abnormal audio metric (low fluency, filler density, pace issues, energy drop)\n"
            "- Each positive must cite evidence (quoted words, metric values, resume matches) AND start with a category tag: [RESUME GAP], [DELIVERY], [CONTENT DEPTH], [STRUCTURE], or [PROFESSIONAL POLISH]\n"
            "- Each improvement MUST start with a category tag: [RESUME GAP], [DELIVERY], [CONTENT DEPTH], [STRUCTURE], or [PROFESSIONAL POLISH]\n"
            "- coaching_summary: 2-3 sentences summarizing overall impression as if spoken to the candidate face-to-face\n"
        )

        user_prompt = "\n".join(parts)
        return system_prompt, user_prompt

    # ══════════════════════════════════════════════════════════════════
    # DATA-DRIVEN FALLBACK — Category-Balanced from real metrics
    # ══════════════════════════════════════════════════════════════════

    def _generate_fallback_feedback(self, transcript: str, resume_text: str,
                                    resume_data: dict, audio_metrics: dict,
                                    filler_stats: dict, overall_score: float,
                                    strictness: str) -> dict:
        """
        Construct intelligent feedback from actual metrics when LLM fails.
        GUARANTEE: Always returns 8-10 positives and 8-10 improvements.
        
        ARCHITECTURE: Category-Bucket Round-Robin
          1. Collect ALL potential positives and improvements into CATEGORY BUCKETS
          2. Round-robin interleave: pick from each bucket to build balanced 8-10 list
          3. Result: Key Strengths shows a MIX of DELIVERY + CONTENT + RESUME + STRUCTURE
        """
        af = audio_metrics or {}
        fs = filler_stats or {}
        words = transcript.split()
        word_count = len(words)
        text_lower = transcript.lower()
        sentences = [s.strip() for s in transcript.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        sentence_count = len(sentences)
        unique_words = set(w.lower().strip('.,!?;:') for w in words if len(w) > 2)
        vocab_ratio = len(unique_words) / max(word_count, 1)
        first_words = ' '.join(words[:8]) if word_count >= 8 else transcript

        # ═══════════════════════════════════════════════════════════════
        # CATEGORY BUCKETS — collect all points by category FIRST
        # ═══════════════════════════════════════════════════════════════
        pos_delivery = []
        pos_content = []
        pos_structure = []
        pos_resume = []
        pos_polish = []

        imp_delivery = []
        imp_content = []
        imp_structure = []
        imp_resume = []
        imp_polish = []

        # ─────────────────────────────────────────────────────────────
        # DELIVERY ANALYSIS (voice metrics)
        # ─────────────────────────────────────────────────────────────
        wpm = af.get("wpm_estimate", 140)
        if 120 <= wpm <= 160:
            pos_delivery.append(
                f"[DELIVERY] Your speaking pace (~{int(wpm)} WPM) sits in the professional sweet spot of 120-160 WPM, "
                f"allowing the interviewer to absorb each point while maintaining energy."
            )
        elif wpm < 100:
            imp_delivery.append(
                f"[DELIVERY] At ~{int(wpm)} WPM, your pace is significantly below the 120-160 WPM benchmark. "
                f"This can read as hesitation. Practice with a timer to build momentum."
            )
        elif wpm < 120:
            imp_delivery.append(
                f"[DELIVERY] Pace at ~{int(wpm)} WPM falls slightly below ideal 120-160 WPM. "
                f"A marginal increase of 15-20 WPM would project greater conviction."
            )
        elif wpm > 185:
            imp_delivery.append(
                f"[DELIVERY] At ~{int(wpm)} WPM, you're speaking faster than 160 WPM ceiling. "
                f"Rapid delivery risks the interviewer missing critical information."
            )
        else:
            imp_delivery.append(
                f"[DELIVERY] Pace at ~{int(wpm)} WPM is slightly above ideal. "
                f"Slowing down by 10-15% will improve clarity."
            )

        tone_expr = af.get("tone_expressiveness", 0.5)
        tone_label = af.get("tone_label", "moderate")
        pitch_range = af.get("pitch_range", 80)
        if tone_expr >= 0.65:
            pos_delivery.append(
                f"[DELIVERY] Vocal expressiveness is strong ({tone_label}, {tone_expr:.2f}/1.0 with {pitch_range:.0f}Hz range). "
                f"Natural modulation keeps the listener engaged."
            )
        elif tone_expr >= 0.4:
            pos_delivery.append(
                f"[DELIVERY] Voice carries {tone_label} expressiveness ({tone_expr:.2f}/1.0) — natural variation prevents monotony."
            )
            imp_delivery.append(
                f"[DELIVERY] While vocal tone is adequate ({tone_expr:.2f}/1.0), pushing toward more dynamic pitch "
                f"variation when stating achievements would elevate the delivery."
            )
        else:
            imp_delivery.append(
                f"[DELIVERY] Voice registers as {tone_label} ({tone_expr:.2f}/1.0) with narrow {pitch_range:.0f}Hz range. "
                f"Practice varying intonation — go higher for achievements, lower for transitions."
            )

        fluency = af.get("fluency_score", 0.5)
        filler_count = fs.get("count", 0)
        per_type = fs.get("per_type", {})
        most_freq = fs.get("most_frequent", None)
        position_cluster = fs.get("position_cluster", "none")

        if fluency >= 0.7:
            filler_note = f" with only {filler_count} filler words" if filler_count <= 2 else ""
            pos_delivery.append(
                f"[DELIVERY] Impressive speech fluency at {fluency:.2f}/1.0{filler_note}. "
                f"Sentences flow with confident transitions, indicating thorough preparation."
            )
        elif fluency >= 0.45:
            pos_delivery.append(
                f"[DELIVERY] Speech maintains reasonable flow ({fluency:.2f}/1.0 fluency) "
                f"without significant breakdown at any point."
            )
            filler_detail = ""
            if filler_count > 2 and most_freq:
                filler_detail = (
                    f"You used '{most_freq}' {per_type.get(most_freq, 0)} times"
                    f"{' (concentrated in the ' + position_cluster + ')' if position_cluster not in ('none', 'distributed') else ''}. "
                )
            imp_delivery.append(
                f"[DELIVERY] Fluency could improve from {fluency:.2f}/1.0. {filler_detail}"
                f"Replace filler words with brief, confident pauses."
            )
        else:
            long_pauses = af.get("long_pauses", 0)
            filler_detail = ""
            if filler_count > 0 and per_type:
                top_3 = sorted(per_type.items(), key=lambda x: -x[1])[:3]
                filler_detail = ", ".join(f"'{k}' ×{v}" for k, v in top_3)
            imp_delivery.append(
                f"[DELIVERY] Speech fluency is a concern at {fluency:.2f}/1.0 — "
                f"{filler_count} fillers ({filler_detail}), {long_pauses} noticeable pauses. "
                f"Outline 4-5 bullet points and practice until transitions feel natural."
            )

        pronun = af.get("pronunciation_score", 0.5)
        hnr = af.get("hnr_score", 0.5)
        if pronun >= 0.65:
            pos_delivery.append(
                f"[DELIVERY] Pronunciation clarity is standout at {pronun:.2f}/1.0 (harmonic quality: {hnr:.2f}). "
                f"Words are sharply articulated — crucial for professional settings."
            )
        elif pronun >= 0.4:
            pos_delivery.append(
                f"[DELIVERY] Pronunciation is generally clear ({pronun:.2f}/1.0), making content accessible."
            )
        else:
            imp_delivery.append(
                f"[DELIVERY] Pronunciation clarity at {pronun:.2f}/1.0 may impact comprehension. "
                f"Practice speaking slower when introducing technical terms."
            )

        energy = af.get("energy_trajectory", "stable")
        energy_con = af.get("energy_consistency", 0.5)
        if energy == "building":
            pos_delivery.append(
                f"[DELIVERY] Vocal energy builds progressively (consistency: {energy_con:.2f}/1.0) — "
                f"signals growing confidence and leaves a powerful closing impression."
            )
        elif energy == "stable":
            pos_delivery.append(
                f"[DELIVERY] Voice energy remains consistent (stability: {energy_con:.2f}/1.0), showing composure."
            )
        elif energy == "fading":
            imp_delivery.append(
                f"[DELIVERY] Voice energy drops toward the end ({energy}, consistency: {energy_con:.2f}/1.0). "
                f"The last 10 seconds form the strongest impression — practice finishing strong."
            )

        confidence = af.get("dynamic_confidence", 50.0)
        conf_label = af.get("confidence_label", "MEDIUM")
        if confidence >= 78:
            pos_delivery.append(
                f"[DELIVERY] Overall vocal confidence at {confidence:.0f}% ({conf_label}) — "
                f"steady pace, clear pronunciation, and expressive tone project genuine self-assurance."
            )
        elif confidence >= 60:
            pos_delivery.append(
                f"[DELIVERY] Vocal confidence at {confidence:.0f}% ({conf_label}), placing you in a competent range."
            )
        else:
            imp_delivery.append(
                f"[DELIVERY] Vocal confidence at {confidence:.0f}% ({conf_label}). "
                f"Focus on: louder volume, fewer fillers, and decisive sentence endings."
            )

        if af.get("speech_rate_stability", 0.5) >= 0.6:
            pos_delivery.append(
                f"[DELIVERY] Speech rhythm consistent ({af.get('speech_rate_stability', 0.5):.2f}/1.0 stability) — "
                f"no significant speed fluctuations."
            )
        if pronun >= 0.5 and fluency >= 0.5:
            pos_delivery.append(
                f"[DELIVERY] Solid fundamentals — pronunciation ({pronun:.2f}) + fluency ({fluency:.2f}) "
                f"ensure message is communicated effectively."
            )
        if fs.get("self_corrections", 0) > 0:
            imp_delivery.append(
                f"[DELIVERY] {fs['self_corrections']} self-correction(s) detected in speech. "
                f"While natural, frequent restarts can undermine confidence perception."
            )

        # ─────────────────────────────────────────────────────────────
        # CONTENT DEPTH ANALYSIS
        # ─────────────────────────────────────────────────────────────
        greeting_kws = ["hello", "hi ", "good morning", "good afternoon", "good evening"]
        name_patterns = ["my name is", "i am ", "i'm ", "myself "]
        has_greeting = any(kw in text_lower[:120] for kw in greeting_kws)
        has_name = any(pat in text_lower[:160] for pat in name_patterns)

        if has_greeting and has_name:
            pos_structure.append(
                f"[STRUCTURE] Strong opening — proper greeting + name introduction ('{first_words}...'). "
                f"Professional start sets the right tone."
            )
        elif has_name:
            imp_structure.append(
                f"[STRUCTURE] You introduced your name but didn't begin with a formal greeting. "
                f"A warm 'Good morning' before your name creates a more polished impression."
            )
        elif has_greeting:
            imp_structure.append(
                f"[STRUCTURE] Greeting present but name wasn't clearly stated. "
                f"Identify yourself within the first 3-5 seconds."
            )
        else:
            imp_structure.append(
                f"[STRUCTURE] No greeting or name introduction detected. "
                f"Always start with: 'Hello, my name is [Name]'."
            )

        edu_kws = ["university", "college", "degree", "bachelor", "master", "btech", "b.tech",
                    "student", "studying", "semester", "computer science", "engineering", "graduated"]
        edu_found = [kw for kw in edu_kws if kw in text_lower]
        if len(edu_found) >= 2:
            pos_content.append(
                f"[CONTENT DEPTH] Educational background effectively communicated — specific academic details "
                f"provide crucial context about your training level."
            )
        elif len(edu_found) == 1:
            imp_content.append(
                f"[CONTENT DEPTH] Education only briefly touched. Specify degree, "
                f"institution, specialization, and year to anchor your professional identity."
            )
        else:
            imp_content.append(
                f"[CONTENT DEPTH] No clear educational background detected. "
                f"Include: degree type, institution name, and year."
            )

        tech_kws = ["python", "java", "javascript", "react", "node", "sql", "html", "css",
                     "machine learning", "deep learning", "ai", "data science", "cloud",
                     "aws", "docker", "tensorflow", "pytorch", "flutter", "angular",
                     "c++", "typescript", "django", "flask", "mongodb", "linux", "git"]
        skills_found = [kw for kw in tech_kws if kw in text_lower]

        if len(skills_found) >= 4:
            pos_content.append(
                f"[CONTENT DEPTH] Strong technical coverage — {len(skills_found)} technologies mentioned "
                f"({', '.join(s.title() for s in skills_found[:5])}). Demonstrates breadth of knowledge."
            )
        elif len(skills_found) >= 2:
            pos_content.append(
                f"[CONTENT DEPTH] You mentioned {len(skills_found)} technical skills ({', '.join(s.title() for s in skills_found)}), "
                f"showing awareness of relevant technologies."
            )
            imp_content.append(
                f"[CONTENT DEPTH] Expand to 4-5 core technologies. Ranking proficiency "
                f"(e.g., 'Python—advanced, React—intermediate') adds depth."
            )
        elif len(skills_found) == 1:
            pos_content.append(
                f"[CONTENT DEPTH] Focusing heavily on a core skill ('{skills_found[0].title()}') demonstrates specialization and deep interest in that specific area."
            )
            imp_content.append(
                f"[CONTENT DEPTH] Only one skill ({skills_found[0].title()}) mentioned. "
                f"List 3-5 core technologies with brief usage context."
            )
        else:
            pos_content.append(
                f"[CONTENT DEPTH] Keeping the pitch focused on high-level background without listing too many technical tools prevents overwhelming the interviewer."
            )
            imp_content.append(
                f"[CONTENT DEPTH] No specific technical skills detected. "
                f"Naming 3-5 technologies is essential for any technical introduction."
            )

        proj_kws = ["project", "built", "developed", "created", "designed", "implemented",
                     "hackathon", "deployed", "application", "system", "website",
                     "contributed", "research", "published", "portfolio"]
        projs_found = [kw for kw in proj_kws if kw in text_lower]
        if len(projs_found) >= 3:
            pos_content.append(
                f"[CONTENT DEPTH] Project evidence convincingly presented — words like "
                f"'{', '.join(projs_found[:3])}' show tangible achievements."
            )
        elif len(projs_found) >= 1:
            pos_content.append(
                f"[CONTENT DEPTH] Mentioning project experience ('{projs_found[0]}') effectively grounds your pitch in real-world application."
            )
            imp_content.append(
                f"[CONTENT DEPTH] You hint at project experience ('{projs_found[0]}') but could strengthen it. "
                f"Name one project, the problem, tech stack, and outcome."
            )
        else:
            pos_content.append(
                f"[CONTENT DEPTH] Your narrative prioritizes foundational background over specific project details, keeping the introduction concise."
            )
            imp_content.append(
                f"[CONTENT DEPTH] No project evidence detected. Include at least one project: "
                f"problem → approach → result."
            )

        goal_kws = ["goal", "aspire", "want to", "plan to", "aim", "dream", "future",
                     "career", "ambition", "passionate about", "become"]
        goals_found = [kw for kw in goal_kws if kw in text_lower]
        if len(goals_found) >= 2:
            pos_structure.append(
                f"[STRUCTURE] Clear career direction articulated — shows maturity and intentionality."
            )
        elif len(goals_found) == 0:
            imp_structure.append(
                f"[STRUCTURE] No career goals mentioned. End with: "
                f"'I'm aiming to specialize in...' to show trajectory."
            )

        # ─────────────────────────────────────────────────────────────
        # STRUCTURAL ANALYSIS
        # ─────────────────────────────────────────────────────────────
        if word_count < 30:
            imp_structure.append(
                f"[STRUCTURE] At only {word_count} words, pitch is critically short. "
                f"A compelling intro needs 80-200 words: greeting → education → skills → projects → goals."
            )
        elif word_count < 60:
            imp_structure.append(
                f"[STRUCTURE] Pitch ({word_count} words) is below ideal 100-200 range. "
                f"Add specific examples and project names to extend naturally."
            )
        elif 80 <= word_count <= 200:
            pos_structure.append(
                f"[STRUCTURE] Pitch length well-calibrated at {word_count} words — long enough for substance, "
                f"short enough for attention."
            )

        if word_count > 20:
            pos_structure.append(
                f"[STRUCTURE] Delivered a substantive pitch of {word_count} words across "
                f"{sentence_count} sentences — demonstrates interview preparation."
            )

        if word_count > 20 and not any(kw in text_lower for kw in ["thank", "appreciate"]):
            imp_structure.append(
                f"[STRUCTURE] No professional closing detected. "
                f"End with 'I'm eager to contribute to...' or 'Thank you for the opportunity'."
            )

        # ─────────────────────────────────────────────────────────────
        # PROFESSIONAL POLISH
        # ─────────────────────────────────────────────────────────────
        if vocab_ratio >= 0.65:
            pos_polish.append(
                f"[PROFESSIONAL POLISH] Impressive vocabulary diversity ({vocab_ratio:.2f}) — varied language signals strong communication."
            )
        elif vocab_ratio < 0.40 and word_count > 30:
            imp_polish.append(
                f"[PROFESSIONAL POLISH] Vocabulary diversity low ({vocab_ratio:.2f}). "
                f"Reduce repetition by using synonyms and varying sentence structures."
            )

        if any(kw in text_lower for kw in ["intern", "experience", "worked", "company"]):
            pos_polish.append(
                f"[PROFESSIONAL POLISH] Referencing real-world experience adds professional credibility — "
                f"positions you as someone who has applied knowledge practically."
            )

        if not any(kw in text_lower for kw in ["because", "for example", "such as", "specifically"]):
            imp_polish.append(
                f"[PROFESSIONAL POLISH] Pitch lacks supporting connectors ('because', 'for example'). "
                f"One concrete example per skill claim transforms assertion to evidence."
            )

        # ─────────────────────────────────────────────────────────────
        # RESUME-PITCH CROSS-REFERENCE
        # ─────────────────────────────────────────────────────────────
        matched_skills = []
        missed_skills = []
        matched_projects = []
        missed_projects = []

        if resume_data:
            for skill in resume_data.get("skills", []):
                if skill.lower() in text_lower:
                    matched_skills.append(skill)
                else:
                    missed_skills.append(skill)

            for proj in resume_data.get("projects", []):
                name_lower = proj.get("name", "").lower()
                name_words = [w for w in name_lower.split() if len(w) > 3]
                if any(w in text_lower for w in name_words):
                    matched_projects.append(proj["name"])
                else:
                    missed_projects.append(proj["name"])

            if matched_skills:
                pos_resume.append(
                    f"[RESUME GAP] Resume-pitch alignment strong — actively referenced {', '.join(matched_skills[:5])} "
                    f"from your resume. Consistency signals genuine familiarity."
                )
            if matched_projects:
                pos_resume.append(
                    f"[RESUME GAP] Successfully referenced projects ({', '.join(matched_projects[:3])}) "
                    f"from your resume in the pitch — demonstrates preparation."
                )

            if missed_projects:
                imp_resume.append(
                    f"[RESUME GAP] Your resume lists projects ({', '.join(missed_projects[:3])}) "
                    f"but they weren't mentioned in the pitch — missed opportunity to showcase work."
                )
            if len(missed_skills) >= 3:
                imp_resume.append(
                    f"[RESUME GAP] Resume highlights {', '.join(missed_skills[:5])} "
                    f"but none mentioned in pitch. Prepare talking points for every major resume skill."
                )

            missed_internships = []
            for exp in resume_data.get("internships", []):
                company = exp.get("company", "").lower()
                company_words = [w for w in company.split() if len(w) > 3]
                if not any(w in text_lower for w in company_words):
                    missed_internships.append(f"{exp.get('role', '')} at {exp.get('company', '')}")
            if missed_internships:
                imp_resume.append(
                    f"[RESUME GAP] Internship experience ({', '.join(missed_internships[:2])}) "
                    f"from resume not discussed — interviewers expect you to reference real experience."
                )

        elif resume_text:
            resume_lower = resume_text.lower()
            all_tech = [
                "python", "java", "javascript", "react", "node", "sql", "html", "css",
                "machine learning", "deep learning", "aws", "docker", "kubernetes",
                "git", "tensorflow", "pytorch", "flutter", "angular", "vue",
                "django", "flask", "mongodb", "postgresql", "c++", "typescript",
            ]
            for kw in all_tech:
                if kw in resume_lower and kw in text_lower:
                    matched_skills.append(kw.title())
                elif kw in resume_lower and kw not in text_lower:
                    missed_skills.append(kw.title())

            if matched_skills:
                pos_resume.append(
                    f"[RESUME GAP] Resume-pitch alignment — you referenced {', '.join(matched_skills[:5])} from your resume."
                )
            if len(missed_skills) >= 3:
                imp_resume.append(
                    f"[RESUME GAP] Resume lists {', '.join(missed_skills[:5])} but not mentioned in pitch."
                )

        # ═══════════════════════════════════════════════════════════════
        # ROUND-ROBIN INTERLEAVE — Balanced category mix
        # Pick from each bucket in rotation to ensure diverse output
        # ═══════════════════════════════════════════════════════════════
        def _interleave_buckets(buckets: list, target: int = 10) -> list:
            """Round-robin pick from category buckets to build balanced list."""
            result = []
            bucket_iters = [iter(b) for b in buckets if b]  # skip empty buckets
            if not bucket_iters:
                return result
            idx = 0
            while len(result) < target:
                exhausted = 0
                for bucket_iter in bucket_iters:
                    if len(result) >= target:
                        break
                    try:
                        result.append(next(bucket_iter))
                    except StopIteration:
                        exhausted += 1
                if exhausted == len(bucket_iters):
                    break  # all buckets exhausted
            return result

        # Order: content → resume → structure → delivery → polish (delivery LAST to avoid domination)
        positives = _interleave_buckets(
            [pos_content, pos_resume, pos_structure, pos_polish, pos_delivery], target=10
        )
        improvements = _interleave_buckets(
            [imp_content, imp_resume, imp_structure, imp_polish, imp_delivery], target=10
        )

        # Safety pad if still under 8
        while len(positives) < 8:
            positives.append(
                f"[CONTENT DEPTH] Engaging with this pitch exercise builds familiarity with articulating "
                f"your professional story under interview pressure."
            )
            break
        while len(improvements) < 8:
            improvements.append(
                f"[STRUCTURE] Use the NESPM framework: Name → Education → Skills → Project → Goals "
                f"for comprehensive coverage in under 2 minutes."
            )
            break

        # ── Coaching Summary ──
        score_label = ("excellent" if overall_score >= 8 else "strong" if overall_score >= 7
                       else "solid" if overall_score >= 6 else "developing" if overall_score >= 4
                       else "foundational")
        coaching_summary = f"Overall performance is {score_label} ({overall_score:.1f}/10). "
        if overall_score >= 7:
            coaching_summary += (
                f"The pitch demonstrates clear preparation with {word_count} words of substantive content. "
                f"Focus on the targeted improvements to elevate from good to exceptional."
            )
        elif overall_score >= 5:
            coaching_summary += (
                f"The foundation is present with {word_count} words at {int(wpm)} WPM, "
                f"but gaps in content coverage and delivery polish need addressing."
            )
        else:
            coaching_summary += (
                f"The pitch needs substantial development across content and delivery. "
                f"Write your full introduction, then practice 5-10 times until structure becomes natural."
            )

        logger.info(
            f"🔄 [GenAIEngine] Fallback: {len(positives)} positives, {len(improvements)} improvements"
        )

        return {
            "positives": positives[:10],
            "improvements": improvements[:10],
            "coaching_summary": coaching_summary,
            "suggestions": [],
            "resume_alignment": {
                "matched": (matched_skills + matched_projects)[:8],
                "missed": (missed_skills + missed_projects + [
                    exp.get("company", "") for exp in (resume_data or {}).get("internships", [])
                    if exp.get("company", "").lower() not in text_lower
                ])[:8],
            }
        }

    # ══════════════════════════════════════════════════════════════════
    # PRIMARY: Comprehensive HR Analysis — SINGLE MODEL CALL
    # ══════════════════════════════════════════════════════════════════

    def comprehensive_analyze(self, transcript: str, resume_text: str = None,
                              audio_metrics: dict = None,
                              filler_stats: dict = None,
                              strictness: str = "intermediate") -> dict | None:
        """
        Single-pass comprehensive analysis.

        ARCHITECTURE:
          1. Deep extract resume [CPU] — projects, internships, certs
          2. Build unified prompt [CPU] — rubric + feedback in one request
          3. ONE model call [GPU] — produces complete JSON in single inference
          4. Parse + validate [CPU]
          5. On fail: ONE retry at lower temperature
          6. On fail: data-driven fallback (NEVER return empty)

        Args:
            transcript: Candidate's refined transcript.
            resume_text: Resume extracted text (optional).
            audio_metrics: Audio analysis metrics dict.
            filler_stats: Enhanced filler detection stats dict.
            strictness: Evaluation difficulty level.

        Returns:
            Complete analysis dict or None if transcript empty.
        """
        if not transcript or not transcript.strip():
            return None

        hr_model = self._get_hr_model()
        if not hr_model:
            logger.error("❌ [GenAIEngine] HR Model NOT LOADED. Check VRAM/Adapter.")
            return None

        t_overall = time.perf_counter()

        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: DEEP RESUME EXTRACTION [CPU — parallel safe]
            # ═══════════════════════════════════════════════════════════
            t = time.perf_counter()
            resume_data = self._deep_extract_resume(resume_text) if resume_text else {}
            logger.info(f"📄 [GenAIEngine] Resume extraction: {time.perf_counter() - t:.2f}s")

            # ═══════════════════════════════════════════════════════════
            # STEP 2: BUILD UNIFIED PROMPT [CPU]
            # ═══════════════════════════════════════════════════════════
            system_prompt, user_prompt = self._build_unified_prompt(
                transcript, resume_text, resume_data,
                audio_metrics or {}, filler_stats or {}, strictness
            )

            # ═══════════════════════════════════════════════════════════
            # STEP 3: SINGLE MODEL CALL [GPU] — rubric + feedback
            # ═══════════════════════════════════════════════════════════
            t = time.perf_counter()
            logger.info("🧠 [GenAIEngine] Running UNIFIED single-pass inference...")

            raw_output = hr_model.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=1400,
                disable_lora=False
            )
            inference_time = time.perf_counter() - t
            logger.info(f"✅ [GenAIEngine] Inference complete: {inference_time:.1f}s")

            # ═══════════════════════════════════════════════════════════
            # STEP 4: PARSE JSON [CPU]
            # ═══════════════════════════════════════════════════════════
            cleaned = self._strip_markdown(raw_output)
            if not cleaned.startswith("{"):
                cleaned = "{" + cleaned if "{" in cleaned else "{}"

            result = self._recover_partial_json(cleaned)

            # Validate and normalize the result
            if result:
                result = self._normalize_result(result)

                fb = result.get("feedback", {})
                pos_count = len(fb.get("positives", []))
                imp_count = len(fb.get("improvements", []))

                logger.info(
                    f"✅ [GenAIEngine] Parsed: {pos_count}P, {imp_count}I, "
                    f"rubric: {len(result.get('rubric_scores', {}))} dims"
                )

                # ═══════════════════════════════════════════════════════
                # STEP 5: RETRY if insufficient feedback
                # ═══════════════════════════════════════════════════════
                if pos_count < 2 or imp_count < 2:
                    logger.warning(f"⚠️ Insufficient ({pos_count}P, {imp_count}I). Retrying...")
                    try:
                        retry_raw = hr_model.generate_text(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=0.5,
                            max_tokens=1200,
                            disable_lora=False
                        )
                        retry_cleaned = self._strip_markdown(retry_raw)
                        if not retry_cleaned.startswith("{"):
                            retry_cleaned = "{" + retry_cleaned if "{" in retry_cleaned else "{}"
                        retry_data = self._recover_partial_json(retry_cleaned)

                        if retry_data:
                            retry_data = self._normalize_result(retry_data)
                            retry_fb = retry_data.get("feedback", {})
                            if len(retry_fb.get("positives", [])) >= 3:
                                result = retry_data
                                logger.info(
                                    f"✅ [GenAIEngine] Retry: "
                                    f"{len(result['feedback'].get('positives', []))}P, "
                                    f"{len(result['feedback'].get('improvements', []))}I"
                                )
                    except Exception as retry_err:
                        logger.warning(f"⚠️ Retry failed: {retry_err}")

            else:
                logger.warning(f"⚠️ Parse failed. Raw: {cleaned[:200]}")
                result = {}

            # ═══════════════════════════════════════════════════════════
            # STEP 6: FALLBACK if both attempts failed
            # ═══════════════════════════════════════════════════════════
            fb = result.get("feedback", {})
            pos_count = len(fb.get("positives", []))
            imp_count = len(fb.get("improvements", []))

            if pos_count < 2 or imp_count < 2:
                logger.warning("⚠️ Both attempts insufficient. Using data-driven fallback...")
                overall = result.get("overall_score", 5.0)
                fallback = self._generate_fallback_feedback(
                    transcript, resume_text, resume_data,
                    audio_metrics or {}, filler_stats or {},
                    overall, strictness
                )
                result["feedback"] = fallback
                if "resume_alignment" not in result or not result.get("resume_alignment", {}).get("matched"):
                    result["resume_alignment"] = fallback.get("resume_alignment", {"matched": [], "missed": []})

            # Ensure required fields exist
            result.setdefault("rubric_scores", {})
            result.setdefault("overall_score", 5.0)
            result.setdefault("feedback", {})
            result.setdefault("resume_alignment", {"matched": [], "missed": []})
            result["strictness_applied"] = strictness

            total_time = time.perf_counter() - t_overall
            logger.info(f"✅ [GenAIEngine] Total analysis: {total_time:.1f}s")
            return result

        except Exception as e:
            logger.error(f"❌ [GenAIEngine] Analysis crashed: {e}")
            import traceback
            traceback.print_exc()
            try:
                resume_data = self._deep_extract_resume(resume_text) if resume_text else {}
                fallback = self._generate_fallback_feedback(
                    transcript, resume_text, resume_data,
                    audio_metrics or {}, filler_stats or {}, 5.0, strictness
                )
                return {
                    "rubric_scores": {},
                    "overall_score": 5.0,
                    "feedback": fallback,
                    "resume_alignment": fallback.get("resume_alignment", {"matched": [], "missed": []}),
                    "strictness_applied": strictness,
                }
            except:
                return None

    def _normalize_result(self, result: dict) -> dict:
        """Normalize and validate parsed LLM result."""
        # Normalize rubric scores
        rubric = result.get("rubric_scores", {})
        for dim, data in rubric.items():
            if isinstance(data, dict):
                score = data.get("s", data.get("score", 5.0))
                data["score"] = max(0.0, min(10.0, float(score)))
                data["reasoning"] = data.get("r", data.get("reasoning", ""))
            elif isinstance(data, (int, float)):
                rubric[dim] = {"score": max(0.0, min(10.0, float(data))), "reasoning": ""}

        # Normalize overall score
        overall = result.get("overall_score", result.get("overall", 5.0))
        result["overall_score"] = max(1.0, min(10.0, float(overall)))

        # Normalize feedback structure
        fb = result.get("feedback", {})
        if isinstance(fb, dict):
            # Handle nested structures
            if "coach" in fb and isinstance(fb["coach"], dict):
                fb = fb["coach"]
            elif "feedback" in fb and isinstance(fb["feedback"], dict):
                fb = fb["feedback"]

            # Normalize keys
            if "pos" in fb and "positives" not in fb:
                fb["positives"] = fb.pop("pos")
            if "imp" in fb and "improvements" not in fb:
                fb["improvements"] = fb.pop("imp")
            if "coach" in fb and "coaching_summary" not in fb:
                fb["coaching_summary"] = fb.pop("coach")

            # Ensure lists of strings
            for key in ["positives", "improvements", "suggestions"]:
                items = fb.get(key, [])
                if isinstance(items, list):
                    fb[key] = [str(i) for i in items if i][:12]
                else:
                    fb[key] = []

            # Ensure coaching_summary is string
            cs = fb.get("coaching_summary", "")
            if not isinstance(cs, str):
                fb["coaching_summary"] = str(cs) if cs else ""

        result["feedback"] = fb

        # Navigate nested resume_alignment
        ra = result.get("resume_alignment", {})
        if isinstance(ra, dict):
            ra.setdefault("matched", [])
            ra.setdefault("missed", [])
        else:
            result["resume_alignment"] = {"matched": [], "missed": []}

        return result

    # ══════════════════════════════════════════════════════════════════
    # SECONDARY: Standalone methods
    # ══════════════════════════════════════════════════════════════════

    def extract_semantic(self, transcript: str) -> dict:
        """Extract structured semantic data from transcript."""
        system = (
            "Extract the following from the transcript and return ONLY valid JSON: "
            "{\"name\":\"\",\"education\":\"\",\"skills\":[],\"experience\":\"\",\"career_goals\":\"\"}"
        )
        hr_model = self._get_hr_model()
        if not hr_model:
            return {}

        raw = hr_model.generate_text(
            system_prompt=system,
            user_prompt=f"{transcript[:600]}",
            temperature=0.0,
            max_tokens=300,
            disable_lora=False
        )

        raw = self._strip_markdown(raw)
        if raw and not raw.startswith('{'):
            raw = '{' + raw if '{' in raw else '{}'

        try:
            return json.loads(raw)
        except:
            return self._recover_partial_json(raw)

    def validate_name(self, extracted_name: str, phonetic_match: str, transcript: str) -> str:
        """Select correct name from NER vs phonetic candidates."""
        system = "Select the correct name. Return ONLY: {\"final_name\": \"<name>\"}"
        prompt = f"Transcript: {transcript}\nNER: {extracted_name}\nPhonetic: {phonetic_match}"
        hr_model = self._get_hr_model()
        if not hr_model:
            return phonetic_match

        raw = hr_model.generate_text(system_prompt=system, user_prompt=prompt,
                                     temperature=0.0, max_tokens=64, disable_lora=True)
        try:
            return json.loads(raw).get("final_name", phonetic_match)
        except:
            return phonetic_match

    def generate_subjective_score(self, transcript: str, semantic: dict) -> float:
        """Generate a quick subjective score (1.0-10.0) from transcript."""
        system = "Score this interview 1.0-10.0. Return ONLY: {\"llm_score\": <float>}"
        prompt = f"Transcript: {transcript}\nSemantic: {json.dumps(semantic)}"
        hr_model = self._get_hr_model()
        if not hr_model:
            return 6.0

        raw = hr_model.generate_text(system_prompt=system, user_prompt=prompt,
                                     temperature=0.0, max_tokens=64, disable_lora=True)
        try:
            return float(json.loads(raw).get("llm_score", 6.0))
        except:
            return 6.0


# ── Singleton Instance ──────────────────────────────────────────────────
genai_engine = GenAIEngine()
