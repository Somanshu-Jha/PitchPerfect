import logging
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

from backend.core.database import db

router = APIRouter(prefix="/auth", tags=["auth"])

# Using a hardcoded demo key. In production this would be in .env
SECRET_KEY = "introlytics_super_secret_jwt_key_production_grade"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days validity

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

class UserSignup(BaseModel):
    user_id: str
    name: str
    password: str

class UserLogin(BaseModel):
    user_id: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    name: str

def _truncate_password(password: str) -> str:
    """bcrypt silently fails on passwords > 72 bytes. Truncate to be safe."""
    return password.encode("utf-8")[:72].decode("utf-8", errors="ignore")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(_truncate_password(password))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(_truncate_password(plain_password), hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/signup", response_model=Token)
async def signup(user: UserSignup):
    try:
        existing = db.get_user(user.user_id)
        if existing and existing.get("password_hash"):
            raise HTTPException(status_code=400, detail="User email already registered.")

        hashed_pwd = get_password_hash(user.password)
        db.upsert_user(user.user_id, user.name, hashed_pwd)

        access_token = create_access_token(
            data={"sub": user.user_id},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {"access_token": access_token, "token_type": "bearer", "user_id": user.user_id, "name": user.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        raise HTTPException(status_code=400, detail=f"Signup failed: {str(e)}")

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    try:
        existing = db.get_user(user.user_id)
        if not existing or not existing.get("password_hash"):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not verify_password(user.password, existing.get("password_hash")):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token = create_access_token(
            data={"sub": user.user_id},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {"access_token": access_token, "token_type": "bearer", "user_id": existing["user_id"], "name": existing["name"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")
