import os
import json
import time

FILE = "backend/data/hr_teacher_labels.jsonl"

def check():
    """
    Role: Dashboard/Tracker.
    Logic: Jab 'Teacher' (DeepSeek) bacho ko marks de raha hota hai training dataset kelye, ye script 
    jsonl file line-by-line parh kar progress dikhata hai.
    Numeric Impact: 5000 Target. 5k samples ko train karne par 1.5B model proper HR ban jata hai. 
    Agar target 100 rakhenge, to wo 'ratta' (overfit) marega, aur ager 50k karden tou 1 week training me lagjyga.
    """
    if not os.path.exists(FILE):
        print("Waiting for first labels...")
        return
    
    with open(FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        count = len(lines)
        
    print(f"\n🚀 [HR ENGINE STATUS — TRUE DEEPSEEK TRAINING]")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Total True Labels: {count} / 5,000 Target")
    print(f"Progress: {(count/5000)*100:.1f}%")
    
    if count > 0:
        try:
            last = json.loads(lines[-1])
            print(f"\nLatest Reasoning (from {last.get('archetype', 'unknown')}):")
            labels = last.get('labels', {})
            rubric = labels.get('rubric_scores', {})
            
            # Print a few rubric items
            for dim, data in list(rubric.items())[:3]:
                 print(f"  - {dim:12s}: {data.get('score'):3.1f} | {data.get('reasoning')}")
            
            feedback = labels.get('feedback', {})
            if feedback.get('positives'):
                print(f"  Positives: {feedback['positives'][0][:100]}...")
            
            # Estimate time till 5k
            # (Note: This is rough based on 15s avg)
            remaining = 5000 - count
            if remaining > 0:
                hours = (remaining * 15) / 3600
                print(f"\nETA to 5k Gold Labels: {hours:.1f} hours")
        except Exception as e:
            print(f"Error parsing last line: {e}")

if __name__ == "__main__":
    check()
