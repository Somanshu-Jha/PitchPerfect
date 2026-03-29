import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.services.scoring_service import ScoringService

def run_dl_scoring_test():
    print("\n--- INITIALIZING DL SCORING NETWORK ---")
    
    scoring_service = ScoringService()
    
    test_cases = [
        {
            "desc": "Perfect Semantic Interview",
            "text": "Hello, my name is John. I have a degree from MIT. My skills are python and java. I have 3 years of experience building apps. My goal is to become a Senior AI Engineer.",
            "structured": {
                "name": "John",
                "education": "MIT",
                "skills": ["python", "java"],
                "experience": ["3 years"],
                "career_goals": "Senior AI Engineer"
            }
        },
        {
            "desc": "Poorly Executed Interview (Incomplete + Fillers)",
            "text": "Um hi um my name is um... yeah that's it.",
            "structured": {
                "name": "Unknown"
            }
        }
    ]
    
    print("\n[Action] Extracting 7-param Vectors & Inferencing...")
    import time
    
    for idx, case in enumerate(test_cases):
        print(f"\nEvaluating Context {idx+1}: {case['desc']}")
        
        start_t = time.time()
        score_packet = scoring_service.calculate_score(case["text"], case["structured"])
        end_t = time.time()
        
        print(f"Output Score: {score_packet['overall_score']} / 10.0")
        print(f"Inference Latency: {(end_t - start_t)*1000:.2f} ms")
        print(f"Source Engaged: {score_packet['source']}")
        print(f"DL Feature Vector: {score_packet.get('features')}")
        
    print("\n--- DL SCORING TEST COMPLETE ---")

if __name__ == "__main__":
    run_dl_scoring_test()
