# -------------------- IMPORTS --------------------
import os
import logging
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.student_routes import router as student_router
from backend.api.auth_routes import router as auth_router

# -------------------- LOGGING (MANDATORY) --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------- HARDWARE THREADS --------------------
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
try:
    torch.set_num_threads(8)
except RuntimeError:
    pass

# -------------------- APP INIT --------------------
app = FastAPI(
    title="Introlytics AI Interview Evaluation System"
)

# -------------------- STARTUP: PRELOAD MODELS --------------------
@app.on_event("startup")
async def startup():
    """
    Preload heavy models at server startup so the first request
    is fast (~2-3s) instead of paying a 60-120s cold-start penalty.
    """
    from backend.core.model_manager import model_manager

    gpu_status = f"✅ {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "❌ CPU only"
    logger.info("=" * 60)
    logger.info("🚀 INTROLYTICS AI BACKEND — PRODUCTION MODE")
    logger.info(f"   GPU: {gpu_status}")
    logger.info(f"   Threads: {torch.get_num_threads()}")
    logger.info(f"   Endpoints: /student/evaluate, /student/progress, /auth/*")
    logger.info("=" * 60)

    # Preload Whisper + SBERT at startup (async-safe: runs in event loop startup)
    model_manager.preload_critical_models()

    logger.info("=" * 60)
    logger.info("✅ SERVER READY — All models preloaded, accepting requests")
    logger.info("=" * 60)

# -------------------- CORS SETUP --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- ROUTES --------------------
app.include_router(student_router, prefix="/student", tags=["Student"])
app.include_router(auth_router)


# -------------------- ROOT --------------------
@app.get("/")
def home():
    return {"message": "Introlytics backend running"}
