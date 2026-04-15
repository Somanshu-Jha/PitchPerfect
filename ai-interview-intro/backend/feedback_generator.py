# backend/feedback_generator.py
import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

class GenerativeCoach:
    """
    THE RIGHT BRAIN: Contextual Reasoning & GPT-Level Feedback.
    Takes the FFNN's math and turns it into DeepSeek's wisdom.
    """
    def __init__(self, ollama_url="http://127.0.0.1:11434/api/generate", model="deepseek-r1:14b"):
        self.url = ollama_url
        self.model = model

    def generate_feedback(self, transcript: str, scores: dict) -> str:
        """
        Creates the 'Smartest Possible Reasoning' feedback.
        It doesn't just list scores; it analyzes why the candidate got them.
        """
        logger.info("🧠 [GenerativeCoach] Analyzing transcript with DeepSeek Reasoning...")
        
        # 1. Identify the 'Weakest Link'
        # Sort scores to find the lowest category
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        weakest_category, weakest_score = sorted_scores[0]
        
        # 2. Extract Contextual Gaps (The 'Perfection' Logic)
        # We look for vague claims in the transcript that aren't backed by evidence.
        context_clues = []
        vague_claims = ["interest in", "passionate about", "keen on", "learning", "enthusiast"]
        
        has_vague_claim = any(claim in transcript.lower() for claim in vague_claims)
        has_evidence = any(word in transcript.lower() for word in ["built", "developed", "project", "github", "deployed", "implemented"])

        if has_vague_claim and not has_evidence:
            context_clues.append("The candidate mentions 'interest' but provides zero evidence of projects or implementation.")
        if weakest_score < 6.0:
            context_clues.append(f"The {weakest_category} score is critically low ({weakest_score}). Focus heavily on this.")

        # 3. Construct the 'Rebel Engine' Prompt (Fact-Based Reasoning)
        prompt = f"""
        Role: Senior AI Interview Evaluator (GPT-Level Reasoning)
        Task: Provide 3 paragraphs of high-impact, professional feedback.
        
        INPUT DATA:
        - Transcript: "{transcript}"
        - Metrics (from FFNN Core): 
          - Overall: {scores['overall']}/10
          - Confidence: {scores['confidence']}/10
          - Tech Depth: {scores['tech_depth']}/10
          - Clarity: {scores['clarity']}/10
          
        OBSERVED GAPS:
        {" ".join(context_clues)}
        
        INSTRUCTIONS:
        1. Do not repeat the transcript back to the user.
        2. Identify exactly why the scores are what they are. 
        3. If they said they have 'interest' in something but didn't mention projects, CALL IT OUT (e.g., 'You mentioned an interest in Python, but failed to ground this in a specific project or library you have used').
        4. Provide 2 'Actionable Items' to reach a Perfect 10.
        
        TONE: Professional, direct, and elite. No fluff.
        """

        try:
            # Note: We use stream=False for a single response
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3, # Low temp for factual consistency
                    "num_predict": 400   # Limit response length
                }
            }
            
            response = requests.post(self.url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                feedback_text = result.get("response", "⚠️ DeepSeek reasoning failed to generate.")
                logger.info("✅ [GenerativeCoach] Feedback generated successfully.")
                return feedback_text
            else:
                logger.error(f"❌ [GenerativeCoach] Ollama Error: {response.status_code}")
                return self._get_fallback_feedback(scores)

        except Exception as e:
            logger.error(f"❌ [GenerativeCoach] Could not connect to DeepSeek: {e}")
            return self._get_fallback_feedback(scores)

    def _get_fallback_feedback(self, scores: dict):
        return f"""
        Scoring Engine Analysis Complete.
        Overall Performance: {scores['overall']}/10.
        Your primary focus should be improving your {list(scores.keys())[0]} through structured practice and clearer project evidence.
        (Note: Connect to DeepSeek 14B to unlock GPT-Level detailed reasoning).
        """

if __name__ == "__main__":
    # Test Run
    coach = GenerativeCoach()
    test_scores = {"overall": 5.8, "tech_depth": 4.1, "confidence": 7.5, "clarity": 8.0, "completeness": 6.0, "structure": 5.5}
    test_transcript = "I have a keen interest in Python and web development, really passionate about it."
    print(coach.generate_feedback(test_transcript, test_scores))
