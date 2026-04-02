# -------------------- IMPORTS --------------------
from fastapi import APIRouter, UploadFile, File, Form, Request
import shutil
import os
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

from backend.services.streaming_asr import StreamingASRService

from backend.services.speech_pipeline import SpeechPipeline


# -------------------- INIT --------------------
router = APIRouter()
pipeline = SpeechPipeline()
streaming_asr_service = StreamingASRService()


# -------------------- ENDPOINT --------------------

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for live ASR transcription.
    Only does transcription for live feedback, doesn't do full semantic scoring.
    """
    await websocket.accept()
    try:
        await streaming_asr_service.process_stream(websocket)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


@router.post("/evaluate")
async def evaluate(request: Request, file: UploadFile = File(...), user_id: str = Form("local_demo")):

    import time as _time
    import uuid
    t_start = _time.perf_counter()

    print("\n" + "=" * 60)
    print("🔥 API HIT — /student/evaluate")
    print(f"   ✅ Audio received: {getattr(file, 'filename', 'Unknown')}")
    print(f"   ✅ Content-Type: {getattr(file, 'content_type', 'Unknown')}")
    print(f"   ✅ User: {user_id}")

    # Unique temp file path to prevent race conditions
    unique_id = uuid.uuid4().hex[:12]
    file_path = f"temp_{unique_id}_{file.filename}"

    # Read bytes for deterministic seeding
    audio_bytes = await file.read()
    print(f"   ✅ File size: {len(audio_bytes)} bytes")

    if len(audio_bytes) == 0:
        print("   ❌ EMPTY FILE RECEIVED — aborting")
        return {"status": "error", "message": "Empty audio file received"}

    with open(file_path, "wb") as buffer:
        buffer.write(audio_bytes)

    print(f"   ✅ File saved to disk: {file_path}")

    # ── ASYNC EXECUTION with GLOBAL CRASH PROTECTION ────────────────────────
    # No cache — pipeline is deterministic via audio-hash seeding
    try:
        result = await pipeline.process(file_path, user_id, audio_bytes)
        print(f"   ✅ Pipeline completed successfully")
    except Exception as e:
        print(f"   💥 CRITICAL PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
        result = {
            "user_id": user_id,
            "raw_transcript": "",
            "refined_transcript": "",
            "semantic": {"intent": {"detected": [], "confidence": 0.0}, "structured": {}, "confidence_map": {}, "evidence_map": {}},
            "audio_features": {},
            "audio_flags": {},
            "fillers": [],
            "scores": {"overall_score": 2.0, "confidence": "low", "note": "Pipeline error — safe fallback returned"},
            "feedback": {
                "positives": ["Your audio was received by the system."],
                "improvements": ["An internal error occurred. Please try recording again.",
                                 "If the issue persists, check that your microphone is working properly."],
                "coaching_summary": "We encountered a processing error. Please try again."
            },
            "completeness_issues": [],
            "historical_progress": {},
            "confidence": {"transcript_confidence": 0.0, "dynamic_confidence": 0.0, "confidence_label": "LOW", "llm_used": False},
            "english_level": "Beginner"
        }

    # Cleanup temp file
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

    elapsed = _time.perf_counter() - t_start
    print(f"   ⏱️ Total request time: {elapsed:.2f}s")
    print(f"   ✅ Score: {result.get('scores', {}).get('overall_score', 'N/A')}")
    print("=" * 60)

    return {
        "status": "success",
        "payload": result
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
async def get_history(user_id: str, days: int = 30):
    """Returns full attempt details for the last N days (default 30). Supports charts."""
    try:
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT attempt_id, transcript, semantic_json, overall_score, 
                       feedback_json, num_fillers, timestamp, flagged_as_improvement, confidence
                FROM Attempts WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 50
            ''', (user_id, cutoff_date))
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

        return {"status": "success", "data": attempts, "period_days": days, "total": len(attempts)}
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