import sqlite3
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseSystem:
    """
    Central localized SQLite Database bridging historical metrics, 
    ML retraining datasets, and frontend user evaluation continuity tracking.
    """
    
    def __init__(self):
        self.db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "databases")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "introlytics.sqlite")
        
        self._initialize_schema()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self):
        """Builds necessary schema natively to track continuous learning goals."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Users System
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        password_hash TEXT DEFAULT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                try:
                    cursor.execute("ALTER TABLE Users ADD COLUMN password_hash TEXT DEFAULT NULL")
                except Exception:
                    pass
                
                # Attempt Tracking (The heart of continuous analysis)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Attempts (
                        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        transcript TEXT NOT NULL,
                        semantic_json TEXT NOT NULL,
                        num_fillers INTEGER DEFAULT 0,
                        overall_score REAL NOT NULL,
                        feedback_json TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Tracking progression flags (Calculated by Data Quality Filter)
                        flagged_as_improvement BOOLEAN DEFAULT FALSE,
                        flushed_to_ml_dataset BOOLEAN DEFAULT FALSE,
                        confidence REAL DEFAULT 0.0,
                        
                        FOREIGN KEY (user_id) REFERENCES Users(user_id)
                    )
                ''')
                
                conn.commit()
                logger.info("✅ [DatabaseSystem] SQLite schema completely initialized.")
                
                # Migrate confidence column if it doesn't exist
                try:
                    cursor.execute("ALTER TABLE Attempts ADD COLUMN confidence REAL DEFAULT 0.0")
                    conn.commit()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"❌ [DatabaseSystem] SQLite Corrupted: {e}")

    def upsert_user(self, user_id: str, name: str, password_hash: str = None):
        """Registers a user non-destructively."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if password_hash:
                cursor.execute('''
                    INSERT INTO Users (user_id, name, password_hash)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, password_hash=excluded.password_hash
                ''', (user_id, name, password_hash))
            else:
                cursor.execute('''
                    INSERT INTO Users (user_id, name)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET name=excluded.name
                ''', (user_id, name))
            conn.commit()

    def get_user(self, user_id: str):
        """Fetches user details for authentication verification."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, name, password_hash FROM Users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                return {"user_id": row[0], "name": row[1], "password_hash": row[2]}
            return None

    def store_attempt(self, user_id: str, transcript: str, semantic: dict, num_fillers: int, score: float, feedback: dict, confidence: float = 0.0) -> int:
        """Stores the completely evaluated payload and compares it to the last payload for improvement validation."""
        
        # 1. Automatic Progress Engine Logic
        improvement_flag = False
        last_score = self.get_latest_score(user_id)
        
        if last_score is not None:
            if score > last_score:
                improvement_flag = True
                logger.info(f"📈 [DatabaseSystem] Improvement detected for {user_id}: {last_score} -> {score}")

        # 2. Schema Persistence
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Attempts (user_id, transcript, semantic_json, num_fillers, overall_score, feedback_json, flagged_as_improvement, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, 
                transcript, 
                json.dumps(semantic), 
                num_fillers, 
                score, 
                json.dumps(feedback), 
                improvement_flag,
                confidence
            ))
            
            attempt_id = cursor.lastrowid
            conn.commit()
            return attempt_id

    def get_latest_score(self, user_id: str):
        """Fetches the previous historical score for trajectory modeling."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT overall_score FROM Attempts
                WHERE user_id = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_user_progress(self, user_id: str) -> dict:
        """Fully integrated API analytics wrapper computing user growth trajectories across history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Retrieve last 10 attempts
            cursor.execute('''
                SELECT timestamp, overall_score, num_fillers, flagged_as_improvement, confidence 
                FROM Attempts 
                WHERE user_id = ? 
                ORDER BY timestamp ASC LIMIT 10
            ''', (user_id,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {"user_id": user_id, "has_data": False}
                
            history = []
            improvement_count = 0
            
            for r in rows:
                history.append({
                    "timestamp": r[0],
                    "score": r[1],
                    "fillers": r[2],
                    "confidence": r[4]
                })
                if r[3]: # flagged_as_improvement array proxy
                    improvement_count += 1
            
            # Simple math delta
            baseline = history[0]["score"]
            current = history[-1]["score"]
            score_delta = round(current - baseline, 1)
            
            return {
                "user_id": user_id,
                "has_data": True,
                "total_attempts": len(rows),
                "improvements_made": improvement_count,
                "score_delta": score_delta,
                "history": history
            }

    def delete_attempt(self, attempt_id: int):
        """Allows users to delete specific attempts from their history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Attempts WHERE attempt_id = ?", (attempt_id,))
            conn.commit()
            
db = DatabaseSystem()
