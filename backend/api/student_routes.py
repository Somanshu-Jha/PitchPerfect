# -------------------- IMPORTS --------------------
from fastapi import APIRouter, UploadFile, File
import shutil
import os

from backend.services.speech_pipeline import SpeechPipeline


# -------------------- INIT --------------------
router = APIRouter()
pipeline = SpeechPipeline()


# -------------------- ENDPOINT --------------------
@router.post("/evaluate")
async def evaluate(audio: UploadFile = File(...)):

    print("\n========== REQUEST RECEIVED ==========")
    print("🔥 API HIT")

    file_path = f"temp_{audio.filename}"

    #---- Save uploaded audio
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    print(f"[FILE SAVED]: {file_path}")

    #---- Run pipeline
    result = pipeline.process(file_path)

    #---- Cleanup temp file
    #if os.path.exists(file_path):
    #    os.remove(file_path)

    print("========== REQUEST COMPLETED ==========")

    return {
        "status": "success",
        "data": result
    }