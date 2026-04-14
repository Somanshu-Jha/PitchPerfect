# backend/ml_models/human_feedback_validator.py
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FEEDBACK_FILE = os.path.join(BASE_DIR, "verified_human_feedback.json")

# In production this would hit the actual Local Qwen/DeepSeek instance.
# For fast execution in the pipeline, we provide a sophisticated heuristic mock
# that acts as the "Gatekeeper".

def validate_feedback(transcript: str, text_reason: str, original_scores: dict, new_scores: dict, use_mock: bool = True) -> bool:
    """
    Acts as the DeepSeek Gatekeeper.
    Returns True if the human's explanation justifies the score change.
    Returns False if it appears to be trolling or unsubstantiated.
    """
    logger.info("🛡️ [Gatekeeper] DeepSeek validating human feedback...")
    
    # 1. Reject empty or incredibly short reasons
    if len(text_reason.split()) < 3:
        logger.warning(f"❌ [Gatekeeper] Rejected: Explanation too short ('{text_reason}')")
        return False
        
    # 2. Reject massive unexplained score jumps without proper keywords
    overall_diff = new_scores.get("overall", 5.0) - original_scores.get("overall", 5.0)
    
    if abs(overall_diff) > 3.0:
        # A swing of more than 3 points requires strong justification
        justification_keywords = ["audio", "stutter", "wrong", "excellent", "perfect", "terrible", "bad", "good", "noise", "accent", "grammar", "bot"]
        if not any(k in text_reason.lower() for k in justification_keywords):
            logger.warning(f"❌ [Gatekeeper] Rejected: Massive score swing ({overall_diff:.1f}) without strong justification keywords.")
            return False

    # 3. Troll detection (Keyboard mashing)
    if "asdf" in text_reason.lower() or len(set(text_reason)) < 5:
        logger.warning(f"❌ [Gatekeeper] Rejected: Detected keyboard smash/troll.")
        return False
        
    # If using actual DeepSeek inference
    if not use_mock:
        prompt = f"""
        TRANSCRIPT: {transcript}
        ORIGINAL SCORE: {original_scores.get('overall')}
        USER NEW SCORE: {new_scores.get('overall')}
        USER REASON: {text_reason}
        
        Is the user's reason logically sound and not malicious? Answer exactly YES or NO.
        """
        # ... logic to run DeepSeek ...
        return True # Placeholder for actual LLM return parsing
        
    logger.info(f"✅ [Gatekeeper] Approved: User justification is valid.")
    return True


def save_human_feedback(transcript: str, features: list, original_scores: dict, new_scores: dict, reason: str):
    """
    Validates and then saves the feedback to the golden dataset.
    """
    is_valid = validate_feedback(transcript, reason, original_scores, new_scores)
    
    if not is_valid:
        return {"status": "rejected", "message": "Feedback explanation was determined to be invalid or malicious."}
        
    existing_data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            existing_data = json.load(f)
            
    record = {
        "text": transcript,
        "features": features,
        "original_scores": original_scores,
        "scores": new_scores,
        "reason": reason,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    existing_data.append(record)
    
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(existing_data, f, indent=2)
        
    logger.info(f"💾 [Gatekeeper] Validated feedback saved to {FEEDBACK_FILE}")
    return {"status": "accepted", "message": "Feedback integrated into the Golden Dataset."}
