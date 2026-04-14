import json
from datetime import datetime, timedelta
from backend.core.database import db

class HistoryService:
    """Service to fetch and format cleanly structured interview attempts."""
    
    def get_user_history(self, user_id: str, days: int = 30):
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT transcript, overall_score, grammar_score, speaking_score, 
                       content_score, confidence_score, strengths, weaknesses, 
                       suggestions, feedback_text, created_at
                FROM interview_attempts 
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC LIMIT 50
            ''', (user_id, cutoff_date))
            rows = cursor.fetchall()

        attempts = []
        for r in rows:
            try:
                strengths = json.loads(r[6]) if r[6] else []
            except: strengths = []
            
            try:
                weaknesses = json.loads(r[7]) if r[7] else []
            except: weaknesses = []
            
            try:
                suggestions = json.loads(r[8]) if r[8] else []
            except: 
                suggestions = []
            
            attempts.append({
                "date": r[10],
                "transcript": r[0],
                "scores": {
                    "overall": r[1],
                    "grammar": r[2],
                    "speaking": r[3],
                    "content": r[4],
                    "confidence": r[5]
                },
                "feedback": r[9],
                "strengths": strengths,
                "weaknesses": weaknesses,
                "suggestions": suggestions
            })
            
        return {
            "status": "success",
            "data": attempts,
            "period_days": days,
            "total": len(attempts)
        }
