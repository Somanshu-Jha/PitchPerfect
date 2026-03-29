# =====================================================================
# GENAI ENGINE — Centralized LLM Inference Router
# =====================================================================
# Handles all LLM calls: semantic extraction, feedback, scoring, name
# validation, and combined comprehensive analysis.
#
# Key architecture:
# - Scoring calls: temperature=0.0 (fully deterministic)
# - Feedback calls: structured-points extraction (temp=0) + wording
#   generation (temp=0.7) for variation at sentence level only
# - Anti-repetition persona injection for linguistic diversity
# =====================================================================

import json
import random
import logging
from backend.core.model_manager import model_manager

logger = logging.getLogger(__name__)

# ── COACHING STYLE POOL ─────────────────────────────────────────────────
# Randomly selected per feedback WORDING call to force linguistic diversity
# Only used for the wording-generation step (not point-extraction step)
FEEDBACK_STYLES = [
    "mentor",           # warm, experienced, guiding
    "coach",            # direct, motivational, action-oriented
    "strict evaluator", # formal, critical, standards-focused
    "friendly advisor"  # casual, encouraging, empathetic
]

class GenAIEngine:
    """
    Centralized inference engine utilizing a 4-bit Local LLM (Qwen/Llama) via ModelManager.
    Built heavily for strict JSON generation (semantic extraction) and zero-hallucination string feedback.
    Upgraded: 9-field semantic schema with value/confidence/evidence per field.

    Scoring: temperature=0.0 (deterministic)
    Feedback: two-pass (structure=temp 0.0, wording=temp 0.7) for consistent logic + varied language
    """

    # Fields that are arrays of dicts vs single dict
    ARRAY_FIELDS = {"skills", "strengths", "areas_of_interest", "qualities", "experience"}
    SCALAR_FIELDS = {"greetings", "name", "education", "career_goals"}

    def __init__(self):
        logger.info("🧠 [GenAIEngine] Initializing Local Deep Learning Router...")
        self.system_instructions = {
            "semantic": (
                "You are an elite NLP data extraction system. Analyze the given transcript and return ONLY valid JSON. "
                "The JSON must strictly follow this schema:\n"
                "{\n"
                '  "greetings": {"value": "<greeting phrase>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"},\n'
                '  "name": {"value": "<full name>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"},\n'
                '  "education": {"value": "<degree and institution>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"},\n'
                '  "skills": [{"value": "<skill>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"}],\n'
                '  "strengths": [{"value": "<strength>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"}],\n'
                '  "areas_of_interest": [{"value": "<area>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"}],\n'
                '  "qualities": [{"value": "<quality>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"}],\n'
                '  "experience": [{"value": "<experience>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"}],\n'
                '  "career_goals": {"value": "<goal>", "confidence": <0.0-1.0>, "evidence": "<exact quote from transcript>"}\n'
                "}\n\n"
                "CRITICAL RULES:\n"
                "- Scalar fields (greetings, name, education, career_goals): single dict with value/confidence/evidence.\n"
                "- Array fields (skills, strengths, areas_of_interest, qualities, experience): list of dicts.\n"
                "- 'evidence' must be the verbatim phrase or sentence from the transcript that justified the extraction.\n"
                "- If a field is absent from the transcript: value = \"\", confidence = 0.0, evidence = \"\".\n"
                "- Do NOT invent or hallucinate data. Do NOT include conversational text outside the JSON block."
            ),
            # ── TWO-PASS FEEDBACK: STEP 1 — Structured points extraction ──────────
            # temperature=0.0 → fully deterministic (same input = same points)
            # This defines WHAT feedback to give (logic level, not wording level)
            "feedback_structure": (
                "You are an elite interview evaluator. Analyze the transcript and semantic data. "
                "Extract EXACTLY the key coaching points — what the candidate did well, and what they must improve.\n\n"
                "RULES:\n"
                "- Provide 4 to 8 strengths (positives). Each as a short topic phrase, not full sentences.\n"
                "- Provide 4 to 8 improvement areas (weaknesses). Each as a short topic phrase, not full sentences.\n"
                "- Do NOT contradict semantic data. If a field is extracted, do NOT say it is missing.\n"
                "- Be specific and factual — reference exact content from the transcript.\n"
                "- Do NOT hallucinate. Only include points directly supported by the transcript.\n\n"
                "Return ONLY valid JSON:\n"
                "{\n"
                '  "strengths": ["<short topic 1>", "<short topic 2>", ..., "<short topic N>"],\n'
                '  "weaknesses": ["<short topic 1>", "<short topic 2>", ..., "<short topic N>"]\n'
                "}\n"
                "No markdown. No text outside the JSON block."
            ),
            # ── TWO-PASS FEEDBACK: STEP 2 — Wording generation ───────────────────
            # temperature=0.7 → sentence-level variation only (not logic level)
            # Persona injected here to vary linguistic style across sessions
            "feedback_wording": (
                "You are an interview coach. You are given a structured list of coaching points. "
                "Your task is to write them as complete, natural coaching sentences.\n\n"
                "CRITICAL RULES:\n"
                "- NEVER reuse sentence structure. Paraphrase completely each time.\n"
                "- Use varied vocabulary: synonyms, different openings, different patterns.\n"
                "- Each bullet must feel unique — as if written by a different person.\n"
                "- NEVER add new points not in the given structure list.\n"
                "- NEVER remove any point from the given structure list.\n"
                "- Adapt tone to the user's English level.\n\n"
                "Return ONLY valid JSON:\n"
                "{\n"
                '  "positives": ["full sentence 1", "full sentence 2", ...],\n'
                '  "improvements": ["full sentence 1", "full sentence 2", ...],\n'
                '  "coaching_summary": "One coherent paragraph summarizing their overall performance"\n'
                "}\n"
                "No markdown. No text outside the JSON block."
            ),
            "name_validation": (
                "You are an AI assistant correcting Indian names extracted from speech transcripts. "
                "You are given an original transcript context, a raw extracted name (NER), and a phonetically-matched dataset name (RapidFuzz). "
                "Select the final corrected name by determining which name fits the context best. "
                "If the phonetic match makes sense in context, use it. If not, stick to the extracted name. "
                "Return ONLY a JSON object matching this schema: {\"final_name\": \"<Chosen Name>\"}"
            ),
            # ── SCORING: temperature=0.0 (MANDATORY for determinism) ───────────────
            "subjective_scoring": (
                "You are an expert interviewer scoring a candidate out of 10 based on their transcript and semantic extraction. "
                "Evaluate ONLY clarity, completeness, structure, and technical depth. "
                "Be consistent — the same transcript must always produce the same score. "
                "Return ONLY a JSON object: {\"llm_score\": <float 1.0 to 10.0>}"
            ),
            # ── COMPREHENSIVE: scoring part is deterministic (temp=0) ───────────────
            # The comprehensive call splits: scoring always uses temp=0.
            # Feedback wording uses the two-pass system above.
            "comprehensive_analysis": (
                "You are an elite AI interview evaluation engine. Analyze the given transcript and return ONLY one valid JSON object with EXACTLY these three top-level keys.\n"
                "1. 'semantic': Extract all 9 fields from the transcript using this exact schema:\n"
                "{\"greetings\": {\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"},\n"
                " \"name\": {\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"},\n"
                " \"education\": {\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"},\n"
                " \"skills\": [{\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"}],\n"
                " \"strengths\": [{\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"}],\n"
                " \"areas_of_interest\": [{\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"}],\n"
                " \"qualities\": [{\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"}],\n"
                " \"experience\": [{\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"}],\n"
                " \"career_goals\": {\"value\": \"\", \"confidence\": 0.0, \"evidence\": \"\"}}\n"
                "2. 'scores': {\"llm_score\": <float 1.0-10.0>} — evaluate clarity, completeness, structure, technical depth. BE CONSISTENT: same input = same score.\n"
                "3. 'feedback': {\"positives\": [4-8 items], \"improvements\": [4-8 items], \"coaching_summary\": \"string\"}\n"
                "FEEDBACK RULES: NEVER contradict extracted semantic fields. NEVER say a field is missing if it appears in semantic. "
                "Provide 4-8 positives and 4-8 improvement points. "
                "CRITICAL: Return ONLY the top-level JSON object. No markdown, no extra text. Missing transcript fields: value=\"\", confidence=0.0."
            )
        }

    def _strip_markdown(self, raw: str) -> str:
        """Strips markdown code fences from LLM output."""
        raw = raw.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return raw

    def _recover_partial_json(self, raw: str) -> dict:
        """Attempts to recover a truncated/malformed JSON by finding the outermost brace pair."""
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.error("❌ [GenAIEngine] JSON recovery also failed. Returning empty dict.")
            return {}

    def _infer(self, prompt: str, task: str, temperature_override: float = None) -> str:
        """
        Core LLM inference method.

        Temperature strategy:
          - Deterministic tasks (scoring, semantic, name_validation, feedback_structure):
              temperature=0.0 → do_sample=False
          - Creative tasks (feedback_wording, comprehensive_analysis):
              temperature=0.7 → do_sample=True

        temperature_override: if explicitly set, overrides the above rules.
        """
        llm_handler = model_manager.load_llm()
        tokenizer = llm_handler["tokenizer"]
        model = llm_handler["model"]

        system_prompt = self.system_instructions.get(task, "You are a helpful assistant.")

        # ── PERSONA INJECTION (wording-only tasks) ──────────────────────────
        # Only inject random style for tasks that produce varied wording.
        # NEVER inject for deterministic scoring/extraction tasks.
        if task in ("feedback_wording",):
            style = random.choice(FEEDBACK_STYLES)
            system_prompt = f"[PERSONA: Act as a {style}. Use a distinct, varied style.]\n" + system_prompt
            logger.info(f"🎭 [GenAI] Style injection for wording: {style}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # ── TEMPERATURE STRATEGY ─────────────────────────────────────────────
        # DETERMINISTIC tasks (scoring must be the same every time):
        #   semantic, subjective_scoring, name_validation, feedback_structure → temp=0.0
        # CREATIVE tasks (feedback wording only varies sentence-level):
        #   feedback_wording → temp=0.7
        # COMPREHENSIVE: the comprehensive_analysis single-call uses temp=0.0 for
        #   the scoring section. Since it's one call, we use temp=0.0 and rely on
        #   the structured prompt to get consistent scoring. Wording variation
        #   is achieved via the two-pass system when using fallback feedback.
        DETERMINISTIC_TASKS = {
            "semantic", "subjective_scoring", "name_validation",
            "feedback_structure", "comprehensive_analysis"
        }

        if temperature_override is not None:
            temp = temperature_override
        elif task in DETERMINISTIC_TASKS:
            temp = 0.0   # Fully deterministic → same input = same output
        else:
            temp = 0.7   # feedback_wording → sentence-level variation

        # do_sample MUST be False when temp=0.0 for greedy decoding
        do_sample = temp > 0.0

        logger.info(f"⏳ [GenAIEngine] Generating inference for task: {task} (temp={temp:.2f}, do_sample={do_sample})...")
        try:
            generated_ids = model.generate(
                **model_inputs,
                # comprehensive_analysis needs more tokens for 3-key JSON
                max_new_tokens=1024 if task in ("comprehensive_analysis", "feedback_wording") else 768,
                temperature=temp if do_sample else None,
                do_sample=do_sample,
                top_p=0.9 if do_sample else None,
                repetition_penalty=1.2 if do_sample else 1.0
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            logger.error(f"❌ [GenAIEngine] Inference failed: {e}")
            return "{}" if task in ("semantic", "feedback_structure", "feedback_wording", "comprehensive_analysis") else ""

    def comprehensive_analyze(self, transcript: str, historical_context: list = None) -> dict | None:
        """
        COMBINED single-call analysis: semantic + scores + feedback in one LLM inference.
        Returns dict with keys {'semantic', 'scores', 'feedback'} or None on failure.

        Scoring is deterministic (temp=0.0 for comprehensive_analysis task).
        Feedback from this call has 4-8 items enforced. For additional wording
        variation in multi-attempt scenarios, FeedbackService's two-pass system is used
        in the fallback path.
        """
        if not transcript:
            return None

        system_injection = ""
        if historical_context:
            system_injection = (
                f"[RAG CONTEXT - HISTORICAL ENRICHMENT: The user previously received this coaching: {json.dumps(historical_context)}. "
                "Do NOT just avoid repeating this. ENRICH your current feedback by explicitly comparing their current performance against this past advice. "
                "Acknowledge their growth if they improved, or provide deeper, escalated strategies if they are stuck on the same issue.]\n"
            )

        prompt = f"{system_injection}Transcript: {transcript}"
        raw = self._infer(prompt, "comprehensive_analysis")
        raw = self._strip_markdown(raw)

        try:
            parsed = json.loads(raw)
            # Validate all 3 top-level keys exist
            if all(k in parsed for k in ("semantic", "scores", "feedback")):
                logger.info("✅ [GenAIEngine] Combined analysis parsed successfully.")
                return parsed
            else:
                logger.warning(f"⚠️ [GenAIEngine] Combined JSON missing keys: {list(parsed.keys())}")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ [GenAIEngine] Combined JSON parse failed: {e}. Attempting partial recovery...")
            recovered = self._recover_partial_json(raw)
            if recovered and all(k in recovered for k in ("semantic", "scores", "feedback")):
                logger.info("✅ [GenAIEngine] Partial JSON recovery succeeded for combined call.")
                return recovered
            logger.warning("⚠️ [GenAIEngine] Partial recovery failed. Triggering multi-call fallback.")
            return None

    def extract_semantic(self, transcript: str) -> dict:
        """Forces the LLM to produce strict JSON with 9 semantic fields (temp=0.0 = deterministic)."""
        if not transcript:
            return {}

        raw_json_string = self._infer(f"Transcript: {transcript}", "semantic")
        raw_json_string = self._strip_markdown(raw_json_string)

        try:
            parsed = json.loads(raw_json_string)
            logger.info(f"✅ [GenAIEngine] Semantic JSON parsed. Fields extracted: {list(parsed.keys())}")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"❌ [GenAIEngine] Failed to parse semantic JSON: {e}\nRaw: {raw_json_string[:300]}")
            return self._recover_partial_json(raw_json_string)

    def extract_feedback_structure(self, transcript: str, semantic_data: dict, historical_context: list = None) -> dict:
        """
        STEP 1 of two-pass feedback: Extract structured coaching points (deterministic, temp=0.0).
        Returns {"strengths": [...], "weaknesses": [...]} as topic phrases.
        Same input → same points. Wording varies in step 2.
        """
        system_injection = ""
        if historical_context:
            system_injection = (
                f"[HISTORY: User previously received coaching on: {json.dumps(historical_context)}. "
                "Do NOT repeat the same advice. Choose distinct, complementary coaching points.]\n"
            )

        prompt = (
            f"{system_injection}"
            f"Transcript: {transcript}\n\n"
            f"Extracted Semantic Data: {json.dumps(semantic_data)}\n\n"
            "Extract the structured coaching points now."
        )

        raw = self._infer(prompt, "feedback_structure")
        raw = self._strip_markdown(raw)

        try:
            parsed = json.loads(raw)
            strengths = parsed.get("strengths", [])
            weaknesses = parsed.get("weaknesses", [])
            # Enforce 4–8 range
            if len(strengths) < 4:
                strengths += ["Positive effort demonstrated"] * (4 - len(strengths))
            if len(weaknesses) < 4:
                weaknesses += ["Continue practicing to improve depth"] * (4 - len(weaknesses))
            return {
                "strengths": strengths[:8],
                "weaknesses": weaknesses[:8]
            }
        except json.JSONDecodeError as e:
            logger.error(f"❌ [GenAIEngine] Feedback structure parse failed: {e}")
            return {
                "strengths": ["Good effort", "Clear communication", "Attempted the interview", "Showed confidence"],
                "weaknesses": ["Expand your response", "Add more detail", "Improve structure", "Practice more"]
            }

    def generate_feedback_wording(
        self,
        structure: dict,
        english_level: str = "Intermediate",
        temperature_override: float = None
    ) -> dict:
        """
        STEP 2 of two-pass feedback: Generate wording from structured points (temp=0.7).
        Input is the deterministic structure from step 1.
        Wording varies per call (sentence-level only), but logic/points stay the same.
        """
        level_note = f"User's English level: {english_level}. "
        if english_level == "Advanced":
            level_note += "Use technical, industry-specific language with deep nuance."
        elif english_level == "Beginner":
            level_note += "Use simple, encouraging, jargon-free language."
        else:
            level_note += "Use standard professional coaching language."

        prompt = (
            f"{level_note}\n\n"
            f"Strengths to expand into coaching sentences:\n{json.dumps(structure.get('strengths', []))}\n\n"
            f"Improvement areas to expand into coaching sentences:\n{json.dumps(structure.get('weaknesses', []))}\n\n"
            "Write each item as a complete, varied coaching sentence. "
            "Do NOT add, remove, or reorder points. Just convert them to professional coaching language."
        )

        raw = self._infer(prompt, "feedback_wording", temperature_override=temperature_override)
        raw = self._strip_markdown(raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"❌ [GenAIEngine] Feedback wording parse failed: {e}")
            return {
                "positives": structure.get("strengths", ["Good effort."]),
                "improvements": structure.get("weaknesses", ["Keep practicing."]),
                "coaching_summary": "Your performance showed promise. Continue developing your interview skills."
            }

    def generate_feedback(self, transcript: str, semantic_data: dict, historical_context: list = None, temperature_override: float = None) -> dict:
        """
        Legacy single-pass feedback (used by FeedbackService fallback path).
        Now internally uses the two-pass system for consistency.

        Args:
            temperature_override: if provided, used for wording step only.
        """
        # Step 1: Extract structure (deterministic)
        structure = self.extract_feedback_structure(transcript, semantic_data, historical_context)

        # Determine english level from semantic_data if injected via inject_data
        english_level = "Intermediate"
        if isinstance(semantic_data, dict):
            override_note = semantic_data.get("system_override_note", "")
            if "Advanced" in override_note:
                english_level = "Advanced"
            elif "Beginner" in override_note:
                english_level = "Beginner"

        # Step 2: Generate wording (varied, temp=0.7)
        result = self.generate_feedback_wording(structure, english_level, temperature_override)

        # Ensure required keys exist
        if not result.get("positives"):
            result["positives"] = structure.get("strengths", ["You completed the interview."])
        if not result.get("improvements"):
            result["improvements"] = structure.get("weaknesses", ["Continue practicing."])
        if not result.get("coaching_summary"):
            result["coaching_summary"] = "Your performance showed effort and dedication."

        return result

    def validate_name(self, extracted_name: str, phonetic_match: str, transcript: str) -> str:
        prompt = f"Transcript Context: {transcript}\nExtracted Name (NER): {extracted_name}\nPhonetic Match (RapidFuzz): {phonetic_match}\n\nReturn the final corrected name in JSON."
        raw_json_string = self._infer(prompt, "name_validation")

        try:
            start = raw_json_string.index("{")
            end = raw_json_string.rindex("}") + 1
            parsed = json.loads(raw_json_string[start:end])
            final_name = parsed.get("final_name")
            return final_name if final_name else phonetic_match
        except Exception as e:
            logger.error(f"❌ [GenAIEngine] LLM Name Validation failed: {e}")
            return phonetic_match

    def generate_subjective_score(self, transcript: str, semantic: dict) -> float:
        """
        Deterministic LLM scoring (temperature=0.0).
        Same transcript + semantic → same score every time.
        """
        prompt = f"Transcript: {transcript}\nSemantic Data: {json.dumps(semantic)}\n\nProvide the LLM-evaluated subjective score now."
        raw_json = self._infer(prompt, "subjective_scoring")  # temp=0.0 via DETERMINISTIC_TASKS
        try:
            start = raw_json.index("{")
            end = raw_json.rindex("}") + 1
            parsed = json.loads(raw_json[start:end])
            return float(parsed.get("llm_score", 6.0))
        except Exception as e:
            logger.error(f"❌ [GenAIEngine] LLM subjective scoring failed: {e}")
            return 6.0

genai_engine = GenAIEngine()
