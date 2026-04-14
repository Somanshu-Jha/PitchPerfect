import pandas as pd
import io
import json
import re
from backend.services.history_service import HistoryService

class ExportService:
    def __init__(self):
        self.history_service = HistoryService()
        
    def _clean_transcript(self, text: str) -> str:
        if not text:
            return ""
        # Remove repeated sequential words roughly (case insensitive)
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def generate_excel(self, user_id: str) -> bytes:
        from backend.core.database import db
        user_info = db.get_user(user_id)
        user_name = user_info.get("name", "Unknown") if user_info else "Unknown"
        user_email = user_info.get("email", user_id) if user_info else user_id

        response = self.history_service.get_user_history(user_id, days=90)
        attempts = response.get("data", [])
        
        # ── 1. Summary Sheet ──
        if not attempts:
            df_summary = pd.DataFrame([{
                "Email": user_email,
                "Name": user_name,
                "User ID": user_id,
                "Total Attempts": 0,
                "Average Score": 0.0,
                "Best Score": 0.0,
                "Lowest Score": 0.0,
                "Improvement Trend": "N/A"
            }])
            df_detailed = pd.DataFrame()
        else:
            total_attempts = len(attempts)
            scores = [a["scores"]["overall"] for a in attempts]
            avg_score = round(sum(scores) / len(scores), 2)
            best_score = max(scores)
            lowest_score = min(scores)
            
            trend = "Stable"
            if len(scores) >= 2:
                oldest = scores[-1]
                latest = scores[0]
                if latest > oldest + 0.5: trend = "Increasing"
                elif latest < oldest - 0.5: trend = "Decreasing"
                
            df_summary = pd.DataFrame([{
                "Email": user_email,
                "Name": user_name,
                "User ID": user_id,
                "Total Attempts": total_attempts,
                "Average Score": avg_score,
                "Best Score": best_score,
                "Lowest Score": lowest_score,
                "Improvement Trend": trend
            }])
            
            # ── 2. Detailed Attempts Sheet ──
            detailed_data = []
            for a in attempts:
                detailed_data.append({
                    "Date": a.get("date", ""),
                    "Transcript": self._clean_transcript(a.get("transcript", "")),
                    "Overall Score": a["scores"].get("overall", 0.0),
                    "Clarity Score": a["scores"].get("grammar", 0.0),
                    "Speaking Score": a["scores"].get("speaking", 0.0),
                    "Content Score": a["scores"].get("content", 0.0),
                    "Confidence Score": a["scores"].get("confidence", 0.0),
                    "Feedback": a.get("feedback", ""),
                    "Strengths": ", ".join([str(s) for s in a.get("strengths", [])]),
                    "Weaknesses": ", ".join([str(w) for w in a.get("weaknesses", [])]),
                    "Suggestions": ", ".join([str(s) for s in a.get("suggestions", [])])
                })
            df_detailed = pd.DataFrame(detailed_data)
            
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            if not df_detailed.empty:
                df_detailed.to_excel(writer, sheet_name='Detailed Attempts', index=False)
                
            # Optional auto-styling could go here using openpyxl directly
            # For now, pandas handles basic clean formatting
                
        output.seek(0)
        return output.getvalue()
        
    def generate_all_excel(self) -> bytes:
        """Admin function to export all users and all attempts from the entire database."""
        from backend.core.database import db
        import json
        
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ia.user_id, u.email, u.name, ia.transcript, ia.overall_score, 
                       ia.grammar_score, ia.speaking_score, ia.content_score, 
                       ia.confidence_score, ia.processing_time, ia.strengths, ia.weaknesses, 
                       ia.suggestions, ia.feedback_text, ia.created_at
                FROM interview_attempts ia
                LEFT JOIN Users u ON ia.user_id = u.user_id
                ORDER BY u.email ASC, ia.created_at DESC
            ''')
            rows = cursor.fetchall()
            
        if not rows:
            df_detailed = pd.DataFrame([{"Message": "No attempts found in the database."}])
            df_summary = pd.DataFrame()
        else:
            detailed_data = []
            user_stats = {}
            
            for r in rows:
                user_id = r[0]
                user_email = r[1] or user_id
                user_name = r[2] or "Unknown"
                strengths = ", ".join(json.loads(r[10])) if r[10] else ""
                weaknesses = ", ".join(json.loads(r[11])) if r[11] else ""
                suggestions = ", ".join(json.loads(r[12])) if r[12] else ""
                
                detailed_data.append({
                    "Email": user_email,
                    "Name": user_name,
                    "User ID": user_id,
                    "Date": r[14],
                    "Transcript": self._clean_transcript(r[3]),
                    "Overall Score": r[4],
                    "Clarity Score": r[5],
                    "Speaking Score": r[6],
                    "Content Score": r[7],
                    "Confidence Score": r[8],
                    "Processing Time (s)": r[9],
                    "Feedback": r[13],
                    "Strengths": strengths,
                    "Weaknesses": weaknesses,
                    "Suggestions": suggestions
                })
                
                if user_id not in user_stats:
                    user_stats[user_id] = {"email": user_email, "name": user_name, "attempts": 0, "scores": []}
                user_stats[user_id]["attempts"] += 1
                user_stats[user_id]["scores"].append(r[4])
                
            df_detailed = pd.DataFrame(detailed_data)
            
            # Build Summary
            summary_data = []
            for uid, stats in user_stats.items():
                scores = stats["scores"]
                summary_data.append({
                    "Email": stats["email"],
                    "Name": stats["name"],
                    "User ID": uid,
                    "Total Attempts": stats["attempts"],
                    "Average Score": round(sum(scores)/len(scores), 2),
                    "Best Score": max(scores),
                    "Lowest Score": min(scores)
                })
            df_summary = pd.DataFrame(summary_data)
            
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='Global Summary', index=False)
            df_detailed.to_excel(writer, sheet_name='All Attempts Master', index=False)
            
        output.seek(0)
        return output.getvalue()
