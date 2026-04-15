import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.core.database import db
from backend.core.rlhf_filter import rlhf_filter

def run_db_test():
    print("\n--- INITIALIZING SQLITE TRACKING SEQUENCE ---")
    
    test_user = "user_akash_001"
    db.upsert_user(test_user, "Akash")
    print(f"✅ User [{test_user}] registered securely in SQLite.")
    
    # 1. Simulate Attempt 1 (Poor)
    print("\n[Action] Simulating Initial Interview Attempt...")
    sem_1 = {"name": "Akash"}
    fb_1 = {"improvements": ["You forgot Python.", "Speak louder."]}
    dl_features_1 = [0.2, 0.9, 0.1, 0.5, 0.2, 0.5, 0.4] # Mock vector
    score_1 = 3.5
    
    db.store_attempt(test_user, "Hi my name is Akash.", sem_1, 2, score_1, fb_1)
    rlhf_filter.validate_and_ingest("Hi my name is Akash.", score_1, dl_features_1)
    
    # Simulate time passing
    time.sleep(1)
    
    # 2. Simulate Attempt 2 (Improved)
    print("\n[Action] Simulating RAG Guided Second Attempt...")
    sem_2 = {"name": "Akash", "skills": ["Python", "PyTorch"], "career_goals": "ML Engineer"}
    fb_2 = {"positives": ["Great integration of PyTorch!"]}
    dl_features_2 = [0.8, 0.9, 0.5, 0.8, 0.05, 1.0, 0.9] # Mock vector
    score_2 = 8.5
    
    db.store_attempt(test_user, "Hi I am Akash. I have skills in Python and PyTorch. I want to be an ML Engineer.", sem_2, 0, score_2, fb_2)
    rlhf_filter.validate_and_ingest("Hi I am Akash. I have skills in Python and PyTorch. I want to be an ML Engineer.", score_2, dl_features_2)
    
    # 3. Query Progress Analytics API
    print("\n[Action] Fetching Progression Trajectory Analytics...")
    start_t = time.time()
    progress = db.get_user_progress(test_user)
    end_t = time.time()
    
    print("\n---------------- USER TRAJECTORY ----------------")
    print(f"User ID:           {progress['user_id']}")
    print(f"Total Attempts:    {progress['total_attempts']}")
    print(f"Net Score Delta:   +{progress['score_delta']} Points")
    print(f"Improvements Count:{progress['improvements_made']}")
    print(f"Query Latency:     {(end_t - start_t) * 1000:.2f} ms")
    print("-------------------------------------------------")
    
    print("\n[Action] Verification of RLHF Dataset appending...")
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "scoring_dataset.csv")
    with open(dataset_path, "r", encoding="utf-8") as f:
        row_count = sum(1 for line in f)
    print(f"Total Model Training Samples: {row_count - 1} (+2 added live)")

if __name__ == "__main__":
    run_db_test()
