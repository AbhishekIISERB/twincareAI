"""AI Copilot chat API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from schemas.copilot import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ChatMessageItem,
)
from services.copilot_service import handle_chat, get_chat_history
from utils.security import get_current_user_id

router = APIRouter(prefix="/copilot", tags=["AI Copilot"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    data: ChatRequest,
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Send a message to the AI Health Copilot.
    
    The Copilot answers questions grounded in the user's own health data:
    biomarkers, risk predictions, and report history.
    """
    result = await handle_chat(db, user_id, data.message)
    return ChatResponse(**result)


@router.get("/history", response_model=ChatHistoryResponse)
def history(
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Get the chat conversation history."""
    messages = get_chat_history(db, user_id)
    return ChatHistoryResponse(
        messages=[ChatMessageItem.model_validate(m) for m in messages]
    )
