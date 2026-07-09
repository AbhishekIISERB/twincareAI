"""Authentication service — registration, login, JWT token management."""

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from models.user import User
from schemas.auth import UserRegister, UserLogin, AuthResponse, UserResponse
from utils.security import hash_password, verify_password, create_access_token

logger = logging.getLogger(__name__)


def register_user(db: Session, data: UserRegister) -> AuthResponse:
    """Register a new user and return auth token."""
    # Check if email already exists
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create user
    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        full_name=data.full_name,
        date_of_birth=data.date_of_birth,
        gender=data.gender,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate token
    token = create_access_token(user.id)

    logger.info(f"User registered: {user.email}")
    return AuthResponse(
        token=token,
        user=UserResponse.model_validate(user),
    )


def login_user(db: Session, data: UserLogin) -> AuthResponse:
    """Authenticate user and return auth token."""
    user = db.query(User).filter(User.email == data.email).first()

    if not user or not verify_password(data.password, user.password_hash):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(user.id)

    logger.info(f"User logged in: {user.email}")
    return AuthResponse(
        token=token,
        user=UserResponse.model_validate(user),
    )


def get_user_profile(db: Session, user_id: UUID) -> UserResponse:
    """Get user profile by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse.model_validate(user)
