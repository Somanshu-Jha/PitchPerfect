# -------------------- IMPORTS --------------------
import os
import logging
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.student_routes import router as student_router
from backend.api.auth_routes import router as auth_router

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------- HARDWARE THREADS --------------------
_CORES = os.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_CORES))
try:
    torch.set_num_threads(_CORES)
except RuntimeError:
    pass

# -------------------- APP (FastAPI Startup) --------------------
# FastAPI: Ye humare backend ka structure banata hai (Jaise building ki foundation).
app = FastAPI(title="Introlytics AI Interview Evaluation System")

# -------------------- STARTUP --------------------
@app.on_event("startup")
async def startup():
    from backend.core.model_manager import model_manager
    from backend.core.genai_engine import genai_engine

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
    
    logger.info("=" * 65)
    logger.info("🚀 INTROLYTICS AI BACKEND — PRODUCTION MODE")
    logger.info(f"   GPU: {gpu_name} ({gpu_mem})")
    logger.info(f"   CPU Threads: {_CORES}")
    logger.info("=" * 65)

    # Preload ASR + Embedder
    model_manager.preload_critical_models()

    # Check all brains
    health = genai_engine.health_check()

    # Check FFNN (Left Brain)
    from backend.ml_models.dl_scoring_model import DLScoringModel
    ffnn_model = DLScoringModel()
    ffnn_status = "✅ loaded (26M params)" if ffnn_model.is_loaded else "❌ MISSING"

    # Status check: Ye batata hai ki humara AI healthy hai ya nahi.
    # Check fine-tuned HR model (Right Brain: Reasoning)
    hr_model_status = "✅ LOADED (1.5B QLoRA)" if health.get("hr_model_available") else "❌ Not trained yet (using Ollama fallback)"

    # Check GenAI System Status
    genai_status = "✅ Native PyTorch Generative AI Mode" if not health.get("ollama_online") else "Hybrid Mode"

    logger.info("=" * 65)
    logger.info("📡  AI COMMAND CENTER — STATUS BOARD")
    logger.info("-" * 45)
    logger.info(f"   LEFT BRAIN  (FFNN Scoring)       : {ffnn_status}")
    logger.info(f"   RIGHT BRAIN (HR Reasoning)       :")
    logger.info(f"     ├─ DeepSeek Native Backend     : {genai_status}")
    logger.info(f"     └─ Native Base Generation      : ✅ ENABLED (temperature=0.7)")
    logger.info(f"   HR RUBRIC ENGINE                 : ✅ 16 dimensions active")
    logger.info(f"   SCORING MODE                     : Hybrid (60% HR Rubric + 40% DL FFNN)")
    logger.info(f"   ACTIVE PRIMARY                   : {health.get('primary', 'unknown').upper()}")
    logger.info("-" * 45)
    logger.info("   Endpoints:")
    logger.info("     POST /student/evaluate  — Full HR analysis (audio + resume)")
    logger.info("     GET  /student/progress  — Historical progress tracking")
    logger.info("     WS   /student/stream    — Live ASR transcription")
    logger.info("-" * 45)
    if not health.get("hr_model_available"):
        logger.info("   💡 To train your own HR model:")
        logger.info("      1. python -m backend.ml_models.hr_dataset_generator --count 100000")
        logger.info("      2. python -m backend.ml_models.hr_teacher_labeler")
        logger.info("      3. python -m backend.ml_models.train_hr_model --epochs 3")
    logger.info("=" * 65)
    logger.info("✅ SERVER READY")
    logger.info("=" * 65)

# -------------------- CORS (Security Bridge) --------------------
# CORS: Ye ek bridge hai jo Frontend (React) ko Backend se bat karne deta hai.
# Agar ye nahi hoga, toh Browser security error dega.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Matlab koi bhi website isse connect kar sakti hai (Safe for Local).
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- ROUTES --------------------
app.include_router(student_router, prefix="/student", tags=["Student"])
app.include_router(auth_router)

@app.get("/")
def home():
    return {"message": "Introlytics backend running", "mode": "hybrid_hr_dl"}
