# -------------------- IMPORTS --------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.student_routes import router as student_router
from backend.api.auth_routes import router as auth_router


# -------------------- APP INIT --------------------
app = FastAPI(
    title="Introlytics AI Interview Evaluation System"
)

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
