import os
import json
import faiss
import numpy as np
import logging
from backend.core.model_manager import model_manager

logger = logging.getLogger(__name__)

class RAGService:
    """
    Retrieval-Augmented Generation Service using Local FAISS.
    Forces embedding layer generation onto CPU to maintain the strict 12GB VRAM pipeline.
    Ensures persistent storage across attempts.
    """
    def __init__(self):
        self.db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "vector_db")
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.db_dir, "historical_attempts.index")
        self.meta_path = os.path.join(self.db_dir, "historical_meta.json")
        
        # We use a 384-dimensional space (all-MiniLM-L6-v2)
        self.dimension = 384 
        
        self.index = self._load_index()
        self.metadata = self._load_metadata()
        logger.info(f"📚 [RAG Service] Loaded Vector DB with {self.index.ntotal} historical records.")

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"❌ [RAG] Corrupt FAISS index: {e}. Rebuilding...")
        return faiss.IndexFlatL2(self.dimension)

    def _load_metadata(self) -> dict:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"records": []}

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

    def ingest(self, user_id: str, transcript: str, semantic: dict, feedback: dict, score: dict):
        """Creates embeddings explicitly on CPU and commits to local FAISS."""
        try:
            embedder = model_manager.load_embedder()
            
            # Formulate the semantic context block for embedding
            context_block = f"{transcript} | Skills: {semantic.get('skills', [])} | Goal: {semantic.get('career_goals', '')}"
            
            # Encode to numpy array
            vector = embedder.encode([context_block])[0]
            vector_np = np.array([vector]).astype("float32")
            
            self.index.add(vector_np)
            
            record = {
                "id": self.index.ntotal - 1,
                "user_id": user_id,
                "transcript": transcript,
                "improvements": feedback.get("improvements", []),
                "overall_score": score.get("overall_score", 0)
            }
            self.metadata["records"].append(record)
            
            self._save()
            logger.info("✅ [RAG] Attempt successfully ingested into Vector DB.")
        except Exception as e:
            logger.error(f"❌ [RAG Ingestion] Failed to ingest record: {e}. Non-fatal, skipping...")

    def retrieve_context(self, user_id: str, current_transcript: str, top_k: int = 2) -> list:
        """
        Sub-100ms vector lookup. Retrieves the user's past mistakes to formulate personalized feedback.
        """
        if self.index.ntotal == 0:
            return []
            
        try:
            embedder = model_manager.load_embedder()
            vector = embedder.encode([current_transcript])[0]
            vector_np = np.array([vector]).astype("float32")
            
            # Search FAISS (Extremely fast, <10ms)
            distances, indices = self.index.search(vector_np, top_k * 3)
            
            retrieved = []
            for idx in indices[0]:
                if idx == -1: 
                    continue
                record = self.metadata["records"][idx]
                
                # Filter strictly by user (or fetch "perfect global examples" if we had them)
                if record["user_id"] == user_id:
                    retrieved.append(record)
                    if len(retrieved) >= top_k:
                        break
                        
            return retrieved
        except Exception as e:
            logger.error(f"❌ [RAG Retrieval] Failed to query Vector DB: {e}")
            return [] # Fail safely without crashing pipeline
