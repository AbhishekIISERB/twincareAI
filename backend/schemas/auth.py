"""Authentication schemas."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    """Registration request body."""
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(..., min_length=1, max_length=255)
    date_of_birth: date | None = None
    gender: str | None = Field(None, pattern=r"^(male|female|other|prefer_not_to_say)$")


class UserLogin(BaseModel):
    """Login request body."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User profile response."""
    id: UUID
    email: str
    full_name: str
    date_of_birth: date | None = None
    gender: str | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    """Auth response with JWT token."""
    token: str
    user: UserResponse


class TokenData(BaseModel):
    """Decoded JWT token data."""
    user_id: UUID
