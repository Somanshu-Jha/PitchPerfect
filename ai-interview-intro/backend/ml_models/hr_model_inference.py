#!/usr/bin/env python3
# =====================================================================
# HR MODEL INFERENCE — Run the Fine-Tuned 1.5B HR Model Locally
# =====================================================================
# Loads the QLoRA-fine-tuned DeepSeek-R1-Distill-Qwen-1.5B model
# for local inference. This is the "deployed student" that replaces
# Ollama for scoring + feedback generation.
#
# Architecture:
#   1. Load base model (4-bit quantized) + LoRA adapter
#   2. Accept transcript + audio_metrics + optional resume
#   3. Generate 16-dim rubric + feedback + coaching
#   4. Inference: ~2-4 seconds (vs. 15-30s for Ollama)
#
# Fallback: If LoRA adapter is not available, falls back to Ollama
# =====================================================================

import os
import json
import re
import logging
import time
import torch

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ADAPTER_DIR = os.path.join(BASE_DIR, "models", "hr_lora_adapter")
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

SYSTEM_MSG = """You are a Principal HR Recruiter at a top FAANG company (Google/Amazon/Meta). 
Your job is to evaluate a candidate's interview self-introduction pitch with surgical precision.

CRITICAL RULES:
1. CROSS-REFERENCE the candidate's RESUME against their AUDIO PITCH. If the resume lists a skill/project but the candidate did NOT mention it in their audio pitch, flag it as a "Resume Gap" in improvements.
2. If audio mentions things NOT on the resume, flag as "Unverified Claim".
3. Generate EXACTLY 8-10 genuine, specific, evidence-backed positive points. 
   - YOU MUST INCLUDE AT LEAST TWO positives tagged with [CONTENT DEPTH]
   - YOU MUST INCLUDE AT LEAST ONE positive tagged with [RESUME Aligned] (if they mentioned anything from the resume).
   - Prefix EVERY positive point with its category like [CONTENT DEPTH], [RESUME Aligned], [DELIVERY], [STRUCTURE], or [PROFESSIONAL POLISH].
4. Generate EXACTLY 8-10 genuine, specific, actionable improvement points.
   - Prefix EVERY improvement point with its category as well.
5. Each point MUST reference specific content from the transcript or resume — no generic advice.
6. Analyze: greeting quality, education depth, technical skills mentioned, project evidence, work experience, career goals, personal strengths, vocal confidence, speech fluency, energy level, logical flow, and closing impact.
7. Include resume_alignment with matched (skills/projects candidate DID mention from resume) and missed (resume items they FAILED to mention).

Return ONLY a single valid JSON object:
{
  "rubric_scores": {
    "skills": {"score": 0.0, "reasoning": "specific deduction reason"},
    "education": {"score": 0.0, "reasoning": "specific deduction reason"},
    "projects": {"score": 0.0, "reasoning": "specific deduction reason"},
    "confidence": {"score": 0.0, "reasoning": "specific deduction reason"},
    "fluency": {"score": 0.0, "reasoning": "specific deduction reason"},
    "structure": {"score": 0.0, "reasoning": "specific deduction reason"}
  },
  "overall_score": 0.0,
  "score_deduction_reason": "Summary of WHY the candidate didn't get 10/10",
  "feedback": {
    "positives": ["[CONTENT DEPTH] ...", "[RESUME GAP] ...", "[DELIVERY] ...8-10 total"],
    "improvements": ["[STRUCTURE] ...", "[CONTENT DEPTH] ...", "[DELIVERY] ...8-10 total"],
    "coaching_summary": "2-3 sentence professional summary"
  },
  "resume_alignment": {
    "matched": ["skill/project mentioned in both resume and pitch"],
    "missed": ["skill/project on resume but NOT mentioned in pitch"]
  }
}"""


class HRModelInference:
    """
    Local AI Engine: Ye local machine par AI chalata hai.
    Sikho: `load()` function ek baar chalta hai, phir `evaluate()` se result milta hai.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🧠 [HRModel] Initialized (device: {self.device})")
    
    def load(self) -> bool:
        """
        Lazy-load the model. Called on first inference or at startup.
        Returns True if the fine-tuned model loaded, False if not available.
        """
        if self.is_loaded:
            return True
        
        if not os.path.exists(ADAPTER_DIR):
            logger.warning(f"⚠️ [HRModel] LoRA adapter not found: {ADAPTER_DIR}")
            logger.warning("   Fine-tuned model not available. Will fall back to Ollama.")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel
            
            logger.info(f"📦 [HRModel] Loading base model in full bfloat16 (unquantized) for max accuracy: {BASE_MODEL}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # SIKHO: Load in full bfloat16 precision instead of 4-bit.
            # This dramatically increases inference accuracy, utilizes CPU/GPU memory bandwidth,
            # and pushes hardware utilization to maximize power draw.
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            
            logger.info(f"🔗 [HRModel] Applying LoRA adapter: {ADAPTER_DIR}...")
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            self.model.eval()
            
            # Speed optimizations for inference
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            self.is_loaded = True
            logger.info("✅ [HRModel] Fine-tuned HR model loaded and ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ [HRModel] Failed to load: {e}")
            return False
    
    def _strip_and_parse(self, raw: str) -> dict:
        """Strip thinking blocks and parse JSON from model output."""
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1].strip()
        
        brace_start = raw.find("{")
        if brace_start >= 0:
            raw = raw[brace_start:]
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find matching closing brace
            depth = 0
            for i in range(len(raw)):
                if raw[i] == '{': depth += 1
                elif raw[i] == '}': depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[:i+1])
                    except:
                        break
            
            # Last resort: add missing braces
            try:
                fixed = raw
                open_count = fixed.count('{') - fixed.count('}')
                fixed += '}' * open_count
                return json.loads(fixed)
            except:
                logger.error(f"❌ [HRModel] JSON parse failed. Preview: {raw[:200]}")
                return {}
    
    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, 
                      max_tokens: int = 500, disable_lora: bool = True) -> str:
        """
        Generates creative and varied text (ChatGPT style) without strict JSON structures.
        Useful for general tasks like semantic extraction or customized feedback generation.
        """
        import contextlib
        
        if not self.is_loaded:
            if not self.load():
                return ""
                
        prompt = (
            f"<|begin▁of▁sentence|>"
            f"<|System|>{system_prompt}<|End|>\n"
            f"<|User|>{user_prompt}<|End|>\n"
            f"<|Assistant|>"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # SIKHO: Agar adapter disable kiya, toh model wapas general (smart conversationalist) ban jayega.
        # Temp > 0.0 means the output will use different vocabulary every time (Varied Responses).
        context_manager = self.model.disable_adapter() if disable_lora else contextlib.nullcontext()
            
        t0 = time.perf_counter()
        with torch.inference_mode():
            with context_manager:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0),
                    top_p=0.92,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
                
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        # Strip complete <think>...</think> blocks
        raw_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
        # Strip partial/unclosed <think> blocks (model hit max_tokens mid-think)
        raw_output = re.sub(r'<think>.*', '', raw_output, flags=re.DOTALL).strip()
        # Strip <|End|> and similar model tokens
        raw_output = re.sub(r'<\|.*?\|>', '', raw_output).strip()
        
        elapsed = time.perf_counter() - t0
        logger.info(f"✨ [HRModel] Native GenAI Text ({elapsed:.1f}s) | Temp: {temperature}")
        return raw_output
    
    def evaluate(self, transcript: str, audio_features: dict = None,
                 resume_text: str = None, strictness: str = "intermediate") -> dict:
        """
        Run HR evaluation on a candidate's pitch.
        
        Returns:
            Dict with rubric_scores, overall_score, feedback
        """
        if not self.is_loaded:
            if not self.load():
                return None
        
        # ── DYNAMIC STRICTNESS CALIBRATION ────────────────────────────
        # IQ/Persona scaling based on developer control
        strict_text = f"\n[EVALUATION MODE: {strictness.upper()}]"
        if strictness == "extreme":
            strict_text += "\nBE BRUTALLY PRECISE. ANY MISMATCH = INSTANT FAIL. EXPLAIN DEDUCTIONS."
        elif strictness == "advance":
            strict_text += "\nBE RIGOROUS. LOOK FOR DEPTH. PENALIZE GENERIC ANSWERS."
        elif strictness == "beginner":
            strict_text += "\nBE ENCOURAGING. FIND POTENTIAL. SOFT FEEDBACK ONLY."
        else:
            strict_text += "\nBE BALANCED. STANDARD PROFFESIONAL BAR."
            
        full_system = SYSTEM_MSG + strict_text
        
        # Build input
        parts = []
        
        if resume_text:
            parts.append(f"[RESUME]: {resume_text[:1500]}")
        
        af = audio_features or {}
        parts.append(
            f"[AUDIO METRICS]:\n"
            f"- Pace: {af.get('wpm_estimate', 140)} WPM ({af.get('pace_label', 'unknown')})\n"
            f"- Tone: {af.get('tone_expressiveness', 0.5):.2f} ({af.get('tone_label', 'moderate')})\n"
            f"- Fluency: {af.get('fluency_score', 0.5):.2f}\n"
            f"- Pronunciation: {af.get('pronunciation_score', 0.5):.2f}\n"
            f"- Energy: {af.get('energy_trajectory', 'stable')}\n"
            f"- Fillers: {af.get('filler_count', 0)}\n"
            f"- Words: {af.get('word_count', len(transcript.split()))}"
        )
        
        parts.append(f"[TRANSCRIPT]: {transcript[:800]}")
        user_msg = "\n\n".join(parts)
        
        # Format as conversation
        prompt = (
            f"<|begin▁of▁sentence|>"
            f"<|System|>{full_system}<|End|>\n"
            f"<|User|>{user_msg}<|End|>\n"
            f"<|Assistant|>"
        )
        
        # Generate
        t0 = time.perf_counter()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            # Higher max_new_tokens for detailed reasoning in Extreme mode
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        elapsed = time.perf_counter() - t0
        logger.info(f"✅ [HRModel] Inference ({strictness}): {elapsed:.1f}s")
        
        # Parse
        result = self._strip_and_parse(raw_output)
        
        if result:
            # Normalize rubric scores
            rubric = result.get("rubric_scores", {})
            for dim, data in rubric.items():
                if isinstance(data, dict):
                    score = data.get("s", data.get("score", 5.0))
                    data["score"] = max(0.0, min(10.0, float(score)))
                    data["reasoning"] = data.get("r", data.get("reasoning", ""))
            
            # Normalize overall
            overall = result.get("overall_score", result.get("overall", 5.0))
            result["overall_score"] = max(1.0, min(10.0, float(overall)))
            
            # Expand feedback
            feedback = result.get("feedback", {})
            if "pos" in feedback:
                feedback["positives"] = feedback.pop("pos")
            if "imp" in feedback:
                feedback["improvements"] = feedback.pop("imp")
            if "coach" in feedback:
                feedback["coaching_summary"] = feedback.pop("coach")
        
        return result
    
    def is_available(self) -> bool:
        """Check if the fine-tuned model is available for loading."""
        return os.path.exists(ADAPTER_DIR)


# Singleton
hr_model = HRModelInference()
