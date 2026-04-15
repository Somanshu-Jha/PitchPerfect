import os
import sys
import json
import time
import logging
import pandas as pd
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Make sure backend is in path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from backend.core.model_manager import model_manager

def load_existing_json(filepath: str) -> list:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_json(filepath: str, data: list):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def generate_distilled_score(tokenizer, model, transcript: str) -> Dict[str, float]:
    prompt = f"""<|im_start|>system
You are an ultra-smart AI judge. Your task is to deeply analyze an interview transcript and provide proxy scoring metrics to train a lightweight neural network.
Evaluate the spoken transcript on the following parameters from 1.0 to 10.0:
1. clarity
2. completeness
3. structure
4. confidence
5. technical_depth
6. overall
Produce ONLY valid JSON containing EXACTLY those 6 keys with float values. No other text.
<|im_end|>
<|im_start|>user
TRANSCRIPT: "{transcript}"
<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=400, 
        temperature=0.3, # Low temp for analytical stability
        do_sample=True
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_reply = response_text.split("assistant")[-1].strip()
    
    # Strip DeepSeek R1 <think> block
    if "</think>" in assistant_reply:
        assistant_reply = assistant_reply.split("</think>")[-1].strip()
        
    start = assistant_reply.find('{')
    end = assistant_reply.rfind('}') + 1
    
    if start != -1 and end != -1:
        return json.loads(assistant_reply[start:end])
    raise ValueError("JSON not found in model output")

def main():
    logger.info("="*60)
    logger.info("🧠 KNOWLEDGE DISTILLATION - DEEPSEEK 32B SCORING ENGINE")
    logger.info("="*60)

    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    csv_path = os.path.join(data_dir, "training_data.csv")
    json_path = os.path.join(data_dir, "deepseek_distilled_scores.json")
    
    raw_texts = []
    try:
        # Check if file is actually structured as JSON
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("[") or content.startswith("{"):
                data = json.loads(content)
                for item in data:
                    if "text" in item:
                        raw_texts.append(item["text"])
            else:
                # Fallback to pandas
                df = pd.read_csv(csv_path, on_bad_lines='skip')
                if "text" in df.columns:
                    raw_texts = df["text"].dropna().unique().tolist()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Filter out empty or extremely short ones
    raw_texts = [str(t).strip() for t in raw_texts if len(str(t).strip()) > 10]
    
    # 2. Check completed files so we don't duplicate work
    completed_data = load_existing_json(json_path)
    completed_texts = {item["text"] for item in completed_data}
    
    pending_texts = [t for t in raw_texts if t not in completed_texts]
    logger.info(f"📊 Found {len(raw_texts)} total transcripts.")
    logger.info(f"📊 Previously scored: {len(completed_data)}")
    logger.info(f"⏳ Remaining to score: {len(pending_texts)}")
    
    if not pending_texts:
        logger.info("✅ All done!")
        return
        
    # 3. Load Model
    logger.info("Loading DeepSeek 32B... (This may take a moment)")
    llm_data = model_manager.load_llm()
    tokenizer = llm_data["tokenizer"]
    model = llm_data["model"]
    
    # 4. Generate
    consecutive_errors = 0
    start_time = time.time()
    
    for idx, text in enumerate(pending_texts):
        logger.info(f"--- Scoring {idx+1}/{len(pending_texts)} ---")
        try:
            scores = generate_distilled_score(tokenizer, model, text)
            
            new_item = {
                "text": text,
                "scores": {
                    "clarity": float(scores.get("clarity", 5.0)),
                    "completeness": float(scores.get("completeness", 5.0)),
                    "structure": float(scores.get("structure", 5.0)),
                    "confidence": float(scores.get("confidence", 5.0)),
                    "technical_depth": float(scores.get("technical_depth", 5.0)),
                    "overall": float(scores.get("overall", 5.0))
                }
            }
            completed_data.append(new_item)
            
            # Save every 5 records to disk to prevent data loss safely
            if idx % 5 == 0:
                save_json(json_path, completed_data)
                
            logger.info(f"   Overall Score: {new_item['scores']['overall']}")
            consecutive_errors = 0 # reset
            
        except Exception as e:
            logger.error(f"   Failed to process: {e}")
            consecutive_errors += 1
            if consecutive_errors > 5:
                logger.error("🛑 Too many consecutive errors. Halting distillation.")
                break
                
    # Final save
    save_json(json_path, completed_data)
    logger.info(f"✅ Distillation Session complete! Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
