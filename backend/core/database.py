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

    SIKHO: SQLite vs TinyDB (NoSQL)
    Pehle humein lagta hai ki NoSQL/TinyDB simple json files me data save karte hain to easy hai.
    Lekin limits kab aati hain? Job portal par 10k users apna data dalian to json file phat jati hai 
    aur ram full ho jati hai. Isliye SQLite theek hai kyu k yeh structured Relational Database hai 
    jis mein foreign keys hain, ye file size 140 TB tak handle kar leta hai smoothly!
    """
    
    def __init__(self):
        self.db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "databases")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "introlytics.sqlite")
        
        self._initialize_schema()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self):
        """
        Builds necessary schema natively to track continuous learning goals.
        Role: Architecture creator.
        Logic: Is function mein SQL commands (DDL) likhe hain jo check karte hain ki tables already hain ya nai, agar nai to banate hain (CREATE TABLE IF NOT EXISTS).
        """
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

                # Safe column migrations — each wrapped individually
                migrations = [
                    ("Users", "password_hash", "TEXT DEFAULT NULL"),
                    ("Users", "email", "TEXT DEFAULT NULL"),
                ]
                for table, col, coltype in migrations:
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                    except Exception:
                        pass  # Column already exists
                
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
                
                # New Structured Table (Clean for UI & Export)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS interview_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        transcript TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        grammar_score REAL NOT NULL,
                        speaking_score REAL NOT NULL,
                        content_score REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        processing_time REAL DEFAULT 0.0,
                        strengths TEXT NOT NULL,
                        weaknesses TEXT NOT NULL,
                        suggestions TEXT NOT NULL,
                        feedback_text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
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
            print(f"❌ [DatabaseSystem] SQLite schema error: {e}")

    def upsert_user(self, user_id: str, name: str, password_hash: str = None):
        """Registers a user non-destructively. Stores email = user_id."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if password_hash:
                    cursor.execute('''
                        INSERT INTO Users (user_id, name, email, password_hash)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET 
                            name=excluded.name, 
                            email=excluded.email,
                            password_hash=excluded.password_hash
                    ''', (user_id, name, user_id, password_hash))
                else:
                    cursor.execute('''
                        INSERT INTO Users (user_id, name, email)
                        VALUES (?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET 
                            name=excluded.name,
                            email=excluded.email
                    ''', (user_id, name, user_id))
                conn.commit()
                logger.info(f"✅ [DB] User upserted: {user_id}")
        except Exception as e:
            logger.error(f"❌ [DB] upsert_user failed: {e}")
            print(f"❌ [DB] upsert_user CRASHED: {e}")
            raise

    def get_user(self, user_id: str):
        """Fetches user details for authentication verification."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT user_id, name, password_hash FROM Users WHERE user_id = ?', 
                    (user_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "user_id": row[0], 
                        "name": row[1], 
                        "password_hash": row[2],
                        "email": row[0],  # user_id IS the email
                    }
                return None
        except Exception as e:
            logger.error(f"❌ [DB] get_user failed for {user_id}: {e}")
            print(f"❌ [DB] get_user CRASHED: {e}")
            return None

    def store_attempt(self, user_id: str, transcript: str, semantic: dict, num_fillers: int, 
                      score: float, feedback: dict, confidence: float = 0.0, processing_time: float = 0.0, 
                      grammar_score: float = 0.0, speaking_score: float = 0.0, content_score: float = 0.0) -> int:
        """
        Stores the completely evaluated payload and compares it to the last payload for improvement validation.
        Role: Engine jab candidate ko marks deta hai to is function ke zariye database me insert hote hain (Attempt Table me).
        Logic: 'INSERT INTO' query chalata hai tables me record ko permanent karne kelye. JSON ko string me convert kark (json.dumps) SQL me bhejta hai.
        """
        
        # 1. Automatic Progress Engine Logic
        improvement_flag = False
        last_score = self.get_latest_score(user_id)
        
        # Normalization as requested
        score = round(min(max(score, 0), 10), 2)
        grammar_score = round(min(max(grammar_score, 0), 10), 2)
        speaking_score = round(min(max(speaking_score, 0), 10), 2)
        content_score = round(min(max(content_score, 0), 10), 2)
        confidence_score = round(min(max(confidence, 0), 100), 2)
        processing_time = round(processing_time, 2)
        
        if last_score is not None:
            if score > last_score:
                improvement_flag = True
                logger.info(f"📈 [DatabaseSystem] Improvement detected for {user_id}: {last_score} -> {score}")

        # Extract strict JSON fields for feedback formatting
        strengths = json.dumps(feedback.get("positives", []))
        weaknesses = json.dumps(feedback.get("improvements", []))
        suggestions = json.dumps(feedback.get("suggestions", []))
        feedback_text = feedback.get("coaching_summary", "")

        # 2. Schema Persistence
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Old schema insertion (keep ML pipeline unbroken)
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
            
            # New schema insertion
            cursor.execute('''
                INSERT INTO interview_attempts (
                    user_id, transcript, overall_score, grammar_score, speaking_score, 
                    content_score, confidence_score, processing_time, strengths, 
                    weaknesses, suggestions, feedback_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, transcript, score, grammar_score, speaking_score, 
                content_score, confidence_score, processing_time, strengths, 
                weaknesses, suggestions, feedback_text
            ))
            
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
        """
        Fully integrated API analytics wrapper computing user growth trajectories across history.
        Role: Ye tracker function UI graph k lye backend maths solve karta hai (ki bacha improve karraha hai ya nahi).
        Logic: Pichlay 10 attempts nikal kar chronlogical order me karta hai (older -> newer), phr pehle aur akhri k darmiyan difference nikalta hai delta karke.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Retrieve last 10 attempts (newest first, then reverse for chronological chart order)
            cursor.execute('''
                SELECT timestamp, overall_score, num_fillers, flagged_as_improvement, confidence 
                FROM Attempts 
                WHERE user_id = ? 
                ORDER BY timestamp DESC LIMIT 10
            ''', (user_id,))
            
            rows = list(reversed(cursor.fetchall()))  # Reverse to chronological (oldest→newest)
            
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
