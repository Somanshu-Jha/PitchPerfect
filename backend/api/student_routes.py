# -------------------- IMPORTS --------------------
from fastapi import APIRouter, UploadFile, File, Form, Request
import shutil
import os
import asyncio

from backend.services.speech_pipeline import SpeechPipeline
from backend.core.result_cache import result_cache


# -------------------- INIT --------------------
router = APIRouter()
pipeline = SpeechPipeline()


# -------------------- ENDPOINT --------------------

@router.post("/evaluate")
async def evaluate(request: Request, file: UploadFile = File(...), user_id: str = Form("local_demo")):

    print("\n========== REQUEST RECEIVED ==========")
    print(f"🔥 API HIT - Received file: {getattr(file, 'filename', 'Unknown')} | User: {user_id}")

    file_path = f"temp_{file.filename}"

    # Read bytes for cache key computation (before writing to disk)
    audio_bytes = await file.read()
    
    # ── CACHE CHECK BEFORE PIPELINE & DISK WRITE ─────────────────────────────────
    cached_result = result_cache.get(audio_bytes, user_id)
    if cached_result is not None:
        print("⚡ [API] Returning cached result — extremely fast response.")
        return {"status": "success", "data": cached_result}
        
    with open(file_path, "wb") as buffer:
        buffer.write(audio_bytes)

    print(f"[FILE SAVED]: {file_path} ({len(audio_bytes)} bytes)")

    # ── ASYNC EXECUTION: pipeline.process is now natively async and handles its own threadpools
    result = await pipeline.process(file_path, user_id, audio_bytes)

    # Cleanup temp file
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

    print("========== REQUEST COMPLETED ==========")

    return {
        "status": "success",
        "data": result
    }


# -------------------- NEW: PROGRESS TRACKING (SAFE ADD) --------------------
from backend.core.database import db
import json

@router.get("/progress/{user_id}")
async def get_progress(user_id: str):
    """Returns user score trajectory and improvement analytics from SQLite."""
    progress = db.get_user_progress(user_id)
    return {"status": "success", "data": progress}


@router.get("/history/{user_id}")
async def get_history(user_id: str):
    """Returns full attempt details (transcript, feedback, scores) for history view."""
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT attempt_id, transcript, semantic_json, overall_score, 
                       feedback_json, num_fillers, timestamp, flagged_as_improvement, confidence
                FROM Attempts WHERE user_id = ?
                ORDER BY timestamp DESC LIMIT 20
            ''', (user_id,))
            rows = cursor.fetchall()

        attempts = []
        for r in rows:
            attempts.append({
                "attempt_id": r[0],
                "transcript": r[1],
                "semantic": json.loads(r[2]) if r[2] else {},
                "score": r[3],
                "feedback": json.loads(r[4]) if r[4] else {},
                "fillers": r[5],
                "timestamp": r[6],
                "improved": bool(r[7]),
                "confidence": r[8]
            })

        return {"status": "success", "data": attempts}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}

@router.delete("/history/{user_id}/{attempt_id}")
async def delete_history_attempt(user_id: str, attempt_id: int):
    """Deletes a specific attempt and visually updates the user trajectory."""
    try:
        db.delete_attempt(attempt_id)
        return {"status": "success", "message": f"Attempt {attempt_id} deleted."}
    except Exception as e:
        return {"status": "error", "message": str(e)}