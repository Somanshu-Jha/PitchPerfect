import os
import sys

# Ensure backend modules can be imported directly
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services.rag_service import RAGService
from backend.core.model_manager import model_manager

def run_rag_test():
    print("\n--- INITIALIZING RAG TEST ---")
    
    # Pre-warm the CPU embedder
    print("Loading all-MiniLM-L6-v2 on CPU...")
    model_manager.load_embedder()
    
    # Initialize RAG
    rag = RAGService()
    
    # 1. Simulate an old attempt
    user_id = "test_user_001"
    old_transcript = "Hi my name is John. I have skills in python. I want to build AI."
    old_semantic = {"skills": ["Python"], "career_goals": "Build AI"}
    old_feedback = {
        "improvements": ["You forgot to mention your education.", "Try speaking louder."]
    }
    old_score = {"overall_score": 6}
    
    print("\n[Action] Ingesting Past Attempt...")
    rag.ingest(user_id, old_transcript, old_semantic, old_feedback, old_score)
    print(f"Total FAISS Records: {rag.index.ntotal}")
    
    # 2. Simulate a new attempt from the same user
    new_transcript = "Hello, I am John. I graduated from MIT and my skills include Python and C++."
    print("\n[Action] Querying RAG Database for user history...")
    
    # Retrieve
    import time
    start_t = time.time()
    context = rag.retrieve_context(user_id, new_transcript)
    end_t = time.time()
    
    print(f"\n[RAG RESULTS] (Latency: {(end_t - start_t)*1000:.2f} ms)")
    if context:
        past_improvements = [rec.get("improvements") for rec in context]
        print(f"Past areas to improve: {past_improvements}")
    else:
        print("No historical context found.")
        
    print("\n--- RAG TEST COMPLETE ---")

if __name__ == "__main__":
    run_rag_test()
