"""
Production-grade Authentication API.
Endpoints: /auth/signup, /auth/login, /auth/forgot-password, /auth/reset-password, /auth/verify-token
All responses follow: { "success": bool, "token"?: str, "user"?: { "email": str }, "message"?: str }
"""
import logging
import traceback
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import re

logger = logging.getLogger(__name__)

from backend.core.database import db

router = APIRouter(prefix="/auth", tags=["auth"])

# ─── Security Constants ──────────────────────────────────────────────────────
# SIKHO (JWT and Secret Keys): JWT (JSON Web Token) ek pass(ticket) hai jo user ko login ke baad milta hai.
# 'SECRET_KEY' wo master chaabi(key) hai jisse server is ticket pe stamp lagata hai taki koi usko change na kar sake.
# Agar ye key leak ho jaye to koi bhi nakli ticket(login token) bana k admin ban sakta hai.
SECRET_KEY = "introlytics_super_secret_jwt_key_production_grade"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
RESET_TOKEN_EXPIRE_MINUTES = 15

# ─── bcrypt Hashing ──────────────────────────────────────────────────────────
# SIKHO (Bcrypt & Numeric Impact): Hashing ka matlab password ko ajeeb code me badal dena jo wapas english me na aye.
# `bcrypt__rounds=12`: Ye backend math(encryption) ko 12 dafa ghumata hai(loop).
# Agar main yahan `rounds=4` kardun, to fast login hoga par hackers 1 second me crack kar lenge.
# Agar `rounds=24` kardun, to itna secure/heavy hojayega ki 1 login hone me 1 minute lagega. 12 ek perfect "Sweet Spot" hai.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

security = HTTPBearer(auto_error=False)


# ─── Input Validation Models ─────────────────────────────────────────────────
class AuthRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=4, description="Min 4 characters")
    name: Optional[str] = Field(None, description="User display name (used during signup)")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not v or not re.match(r"^\S+@\S+\.\S+$", v):
            raise ValueError("Invalid email format")
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v or len(v.strip()) < 4:
            raise ValueError("Password must be at least 4 characters")
        return v.strip()


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., description="Email to send reset link to")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not v or not re.match(r"^\S+@\S+\.\S+$", v):
            raise ValueError("Invalid email format")
        return v


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., description="Reset token from forgot-password")
    new_password: str = Field(..., min_length=4, description="New password, min 4 chars")

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v or len(v.strip()) < 4:
            raise ValueError("Password must be at least 4 characters")
        return v.strip()


# ─── Helper Functions ─────────────────────────────────────────────────────────
def _truncate_password(password: str) -> str:
    """bcrypt silently truncates at 72 bytes. Do it explicitly."""
    return password.encode("utf-8")[:72].decode("utf-8", errors="ignore")


def hash_password(password: str) -> str:
    return pwd_context.hash(_truncate_password(password))


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(_truncate_password(plain), hashed)
    except Exception as e:
        logger.error(f"PASSWORD VERIFY ERROR: {e}")
        return False


def create_token(payload: dict, expires_minutes: int) -> str:
    """
    Role: Ticket Creator.
    Logic: User ka email data leta hai, usme Expire Time (exp) dalta hai, aur SECRET_KEY k sath lock(encode) kardeta hai.
    """
    data = payload.copy()
    data["exp"] = datetime.utcnow() + timedelta(minutes=expires_minutes)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """
    Returns decoded payload or None if invalid/expired.
    Role: Ticket Checker (Guard).
    Logic: Token ko khol kar parhta hai, agar Expire ho gya (e.g. 24 hours se zyada) ya kisi hacker ne change kiya hou 
    to error de dega.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "message": message})


def _success(token: str, email: str, name: str = "User") -> dict:
    return {"success": True, "token": token, "user": {"email": email, "name": name}}


# ─── POST /auth/signup ────────────────────────────────────────────────────────
@router.post("/signup")
async def signup(request: Request):
    logger.info("========== /auth/signup HIT ==========")
    
    # Parse & validate
    try:
        body = await request.json()
        logger.info(f"[SIGNUP] Body received: {body}")
        req = AuthRequest(**body)
    except Exception as e:
        logger.error(f"[SIGNUP] PARSE ERROR: {e}")
        return _error(400, "Invalid input. Provide a valid email and password (min 4 chars).")

    try:
        # 1. Check if user already exists
        existing = db.get_user(req.email)
        logger.info(f"[SIGNUP] Existing user check: {existing is not None}")
        
        if existing and existing.get("password_hash"):
            logger.info(f"[SIGNUP] User already exists: {req.email}")
            return _error(409, "An account with this email already exists.")

        # 2. Hash password
        display_name = (req.name or "").strip() or "User"
        hashed = hash_password(req.password)
        logger.info(f"[SIGNUP] Password hashed for {req.email}")

        # 3. Store in database
        db.upsert_user(user_id=req.email, name=display_name, password_hash=hashed)
        logger.info(f"[SIGNUP] User stored in DB: {req.email}")

        # 4. Create JWT token
        token = create_token(
            {"email": req.email, "type": "access"},
            expires_minutes=ACCESS_TOKEN_EXPIRE_HOURS * 60
        )
        logger.info(f"[SIGNUP] SUCCESS: {req.email} (name={display_name})")
        
        return {"success": True, "token": token, "user": {"email": req.email, "name": display_name}}

    except Exception as e:
        logger.error(f"[SIGNUP] CRASH: {e}", exc_info=True)
        logger.error(f"Signup error: {e}", exc_info=True)
        return _error(500, "Server error during signup.")


# ─── POST /auth/login ─────────────────────────────────────────────────────────
@router.post("/login")
async def login(request: Request):
    logger.info("========== /auth/login HIT ==========")
    
    try:
        body = await request.json()
        logger.info(f"[LOGIN] Body received: {body.get('email', 'N/A')}")
        req = AuthRequest(**body)
    except Exception as e:
        logger.error(f"[LOGIN] PARSE ERROR: {e}")
        return _error(400, "Invalid input. Provide a valid email and password (min 4 chars).")

    try:
        existing = db.get_user(req.email)
        logger.info(f"[LOGIN] User lookup: {existing is not None}")

        # Auto-create for demo convenience (first-time login = auto-signup)
        if not existing or not existing.get("password_hash"):
            logger.info(f"[LOGIN] Auto-creating account for: {req.email}")
            hashed = hash_password(req.password)
            db.upsert_user(user_id=req.email, name="User", password_hash=hashed)
            existing = db.get_user(req.email)

        # Verify password
        if not verify_password(req.password, existing["password_hash"]):
            logger.warning(f"[LOGIN] Wrong password for: {req.email}")
            return _error(401, "Invalid email or password.")

        token = create_token(
            {"email": req.email, "type": "access"},
            expires_minutes=ACCESS_TOKEN_EXPIRE_HOURS * 60
        )
        user_name = existing.get("name", "User")
        logger.info(f"[LOGIN] SUCCESS: {req.email}")
        return _success(token, req.email, user_name)

    except Exception as e:
        logger.error(f"[LOGIN] CRASH: {e}", exc_info=True)
        logger.error(f"Login error: {e}", exc_info=True)
        return _error(500, "Server error during login.")


# ─── POST /auth/forgot-password ───────────────────────────────────────────────
@router.post("/forgot-password")
async def forgot_password(request: Request):
    logger.info("========== /auth/forgot-password HIT ==========")
    
    try:
        body = await request.json()
        req = ForgotPasswordRequest(**body)
    except Exception:
        return _error(400, "Provide a valid email address.")

    try:
        existing = db.get_user(req.email)
        if not existing:
            # Don't reveal whether the account exists (security best practice)
            return {"success": True, "message": "If this email is registered, a reset link has been sent."}

        reset_token = create_token(
            {"email": req.email, "type": "reset"},
            expires_minutes=RESET_TOKEN_EXPIRE_MINUTES
        )
        # In production: send email with reset_token link
        # For now: log the token so it can be used in testing
        logger.info(f"[FORGOT-PASSWORD] Reset token generated for {req.email}")

        return {"success": True, "message": "If this email is registered, a reset link has been sent.", "reset_token": reset_token}

    except Exception as e:
        logger.error(f"[FORGOT-PASSWORD] CRASH: {e}", exc_info=True)
        logger.error(f"Forgot-password error: {e}", exc_info=True)
        return _error(500, "Server error. Please try again later.")


# ─── POST /auth/reset-password ────────────────────────────────────────────────
@router.post("/reset-password")
async def reset_password(request: Request):
    logger.info("========== /auth/reset-password HIT ==========")
    
    try:
        body = await request.json()
        req = ResetPasswordRequest(**body)
    except Exception:
        return _error(400, "Invalid input. Provide a valid reset token and new password (min 4 chars).")

    try:
        decoded = decode_token(req.token)
        if not decoded or decoded.get("type") != "reset":
            return _error(401, "Invalid or expired reset token.")

        email = decoded.get("email")
        if not email:
            return _error(401, "Malformed reset token.")

        existing = db.get_user(email)
        if not existing:
            return _error(404, "Account not found.")

        new_hash = hash_password(req.new_password)
        db.upsert_user(user_id=email, name=existing.get("name", "User"), password_hash=new_hash)

        logger.info(f"[RESET-PASSWORD] Password reset successful for: {email}")
        return {"success": True, "message": "Password has been reset successfully. You can now log in."}

    except Exception as e:
        logger.error(f"[RESET-PASSWORD] CRASH: {e}", exc_info=True)
        logger.error(f"Reset-password error: {e}", exc_info=True)
        return _error(500, "Server error during password reset.")


# ─── POST /auth/verify-token ──────────────────────────────────────────────────
@router.post("/verify-token")
async def verify_token(request: Request):
    """
    Frontend calls this on app load to check if a stored JWT is still valid.
    Request: { "token": "JWT_STRING" }
    Response: { "success": true, "user": { "email": "..." } } or { "success": false }
    """
    try:
        body = await request.json()
        token = body.get("token", "")
    except Exception:
        return _error(400, "Provide a token.")

    if not token:
        return _error(400, "Token is required.")

    decoded = decode_token(token)
    if not decoded or decoded.get("type") != "access":
        return _error(401, "Invalid or expired token.")

    email = decoded.get("email")
    if not email:
        return _error(401, "Malformed token.")

    # Verify user still exists in DB
    existing = db.get_user(email)
    if not existing:
        return _error(401, "User no longer exists.")

    return {"success": True, "user": {"email": email, "name": existing.get("name", "User")}}


# ─── JWT Middleware Helper (for protected routes) ─────────────────────────────
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Use as a FastAPI dependency on any protected route:
      @router.get("/protected")
      async def protected(user: dict = Depends(get_current_user)):
    """
    if not credentials:
        raise _error(401, "Authentication required.")

    decoded = decode_token(credentials.credentials)
    if not decoded or decoded.get("type") != "access":
        raise _error(401, "Invalid or expired token.")

    email = decoded.get("email")
    existing = db.get_user(email)
    if not existing:
        raise _error(401, "User not found.")

    return {"email": email, "name": existing.get("name", "User")}

# ─── ADMIN CONFIG Endpoints ───────────────────────────────────────────────
@router.get("/admin/config")
async def get_admin_config():
    """
    Returns the currently active universal strictness.
    """
    from backend.core.global_config import load_global_strictness
    current = load_global_strictness()
    return {"success": True, "universal_strictness": current}

class AdminConfigRequest(BaseModel):
    strictness: str = Field(..., description="The universal strictness level")

@router.post("/admin/config")
async def update_admin_config(req: AdminConfigRequest):
    """
    Updates the universal strictness permanently across all sessions.
    """
    from backend.core.global_config import save_global_strictness
    valid_levels = ["beginner", "intermediate", "advance", "extreme"]
    
    if req.strictness not in valid_levels:
         return _error(400, "Invalid strictness level.")
         
    success = save_global_strictness(req.strictness)
    if success:
         return {"success": True, "message": "Universal configuration updated."}
    return _error(500, "Failed to update configuration.")
