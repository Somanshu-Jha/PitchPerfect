# -------------------- IMPORTS --------------------
from fastapi import APIRouter, UploadFile, File, Form, Request
import shutil
import os
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

from backend.services.streaming_asr import StreamingASRService
from backend.services.speech_pipeline import SpeechPipeline

import logging
logger = logging.getLogger(__name__)


# -------------------- INIT (App start hote hi ye cheezein banti hain) --------------------
router = APIRouter() # Ye rasta batata hai (URL endpoints).
pipeline = SpeechPipeline() # Ye humari main logic machine hai.
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
async def evaluate(
    request: Request, 
    # file: Candidate ki voice recording.
    file: UploadFile = File(...), 
    # resume: Candidate ka PDF/Text data (Optional).
    resume: UploadFile = File(None),
    # user_id: Database mein save karne ke liye.
    user_id: str = Form("local_demo"),
    # strictness: Kitni sakhti se check karna hai (Set by Developer Panel).
    strictness: str = Form("intermediate")
):
    # SIKHO: Yaha hum Universal strictness enforce kar rahe hain!
    from backend.core.global_config import load_global_strictness
    universal_strictness = load_global_strictness()
    # Frontend jo marzi bheje, admin ki universal configuration override karegi
    final_strictness = universal_strictness if universal_strictness else strictness

    import time as _time
    import uuid
    t_start = _time.perf_counter()

    logger.info("=" * 60)
    logger.info("API HIT -- /student/evaluate")
    logger.info(f"   Audio received: {getattr(file, 'filename', 'Unknown')}")
    if resume:
        logger.info(f"   Resume received: {resume.filename}")
    logger.info(f"   User: {user_id}")

    # Unique temp file path to prevent race conditions
    unique_id = uuid.uuid4().hex[:12]
    audio_path = f"temp_{unique_id}_{file.filename}"
    resume_path = None
    
    if resume:
        resume_path = f"temp_res_{unique_id}_{resume.filename}"
        with open(resume_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        logger.info(f"   Resume saved to disk: {resume_path}")

    # Read bytes for deterministic seeding
    audio_bytes = await file.read()
    logger.info(f"   File size: {len(audio_bytes)} bytes")

    if len(audio_bytes) == 0:
        logger.error("   EMPTY FILE RECEIVED -- aborting")
        return {"status": "error", "message": "Empty audio file received"}

    with open(audio_path, "wb") as buffer:
        buffer.write(audio_bytes)

    logger.info(f"   File saved to disk: {audio_path}")

    # ── ASYNC EXECUTION ──
    # await pipeline.process: Matlab jab tak pipeline result na de, API wait karegi.
    try:
        result = await pipeline.process(audio_path, user_id, audio_bytes, resume_path=resume_path, strictness=final_strictness)
        logger.info(f"   Pipeline completed under Strictness: {final_strictness.upper()}")
    except Exception as e:
        logger.error(f"   CRITICAL PIPELINE ERROR: {e}", exc_info=True)
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

    # Cleanup temp files
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if resume_path and os.path.exists(resume_path):
            os.remove(resume_path)
    except Exception:
        pass

    processing_time = _time.perf_counter() - t_start
    result["processing_time"] = round(processing_time, 2)
    
    logger.info(f"   Final Result ready in {processing_time:.2f}s")
    logger.info("=" * 60)
    
    return result


@router.get("/progress/{user_id}")
async def get_progress(user_id: str):
    """
    Returns historical progress for a user (Alias).
    """
    from backend.core.database import db
    return db.get_user_progress(user_id)


@router.get("/history/{user_id}")
async def get_history(user_id: str, days: int = 30):
    """
    Returns full historical attempts with feedback for a user.
    Uses HistoryService for structured output with positives/improvements/suggestions.
    """
    from backend.services.history_service import HistoryService
    history_svc = HistoryService()
    return history_svc.get_user_history(user_id, days=days)


@router.get("/export/{user_id}")
async def export_history(user_id: str):
    """
    Dummy CSV export endpoint to prevent 404 frontend crash.
    """
    return {"status": "ok", "message": "Export functionality coming soon!"}


@router.get("/health")
async def health_check():
    """Simple check to see if API is alive."""
    return {"status": "ok", "service": "Introlytics API"}