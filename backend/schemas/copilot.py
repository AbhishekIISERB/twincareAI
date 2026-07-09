"""Copilot chat schemas."""

from datetime import datetime

from pydantic import BaseModel, Field

from config import settings


class ChatRequest(BaseModel):
    """Chat message request."""
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """Chat response from the Copilot."""
    response: str
    context_used: list[str] = []
    disclaimer: str = settings.DISCLAIMER


class ChatMessageItem(BaseModel):
    """Single chat message in history."""
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """Chat history response."""
    messages: list[ChatMessageItem]
