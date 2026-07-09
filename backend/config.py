"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings from environment variables."""

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://twincare:twincare_dev@localhost:5432/twincare"
    )

    # Auth
    JWT_SECRET: str = os.getenv("JWT_SECRET", "twincare-hackathon-secret-change-in-prod")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"

    # CORS
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    ]

    # Fireworks AI
    FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
    FIREWORKS_BASE_URL: str = "https://api.fireworks.ai/inference/v1"
    FIREWORKS_MODEL: str = os.getenv(
        "FIREWORKS_MODEL",
        "accounts/fireworks/models/gemma2-9b-it"
    )

    # File upload
    MAX_UPLOAD_SIZE_MB: int = 10
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")

    # ML Model paths
    HEART_MODEL_PATH: str = os.getenv("HEART_MODEL_PATH", "ml_models/heart_disease_model.pt")

    # Disclaimer
    DISCLAIMER: str = (
        "⚠️ This is not a medical diagnosis. "
        "Always consult a qualified healthcare professional for medical advice."
    )


settings = Settings()
