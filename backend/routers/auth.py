"""Authentication API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from schemas.auth import UserRegister, UserLogin, AuthResponse, UserResponse
from services.auth_service import register_user, login_user, get_user_profile
from utils.security import get_current_user_id

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user account."""
    return register_user(db, data)


@router.post("/login", response_model=AuthResponse)
def login(data: UserLogin, db: Session = Depends(get_db)):
    """Login with email and password."""
    return login_user(db, data)


@router.get("/me", response_model=UserResponse)
def get_me(
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Get current user profile."""
    return get_user_profile(db, user_id)
