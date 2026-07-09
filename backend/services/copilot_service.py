"""AI Copilot service — retrieval-augmented chat grounded in user's own health data."""

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from config import settings
from models.chat_message import ChatMessage
from models.biomarker import Biomarker
from models.risk_prediction import RiskPrediction
from models.report import Report
from services.digital_twin_service import get_or_create_twin
from ai.llm_client import call_fireworks_api

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TwinCare AI Health Copilot — a friendly, knowledgeable health assistant. 
You help users understand their health data in simple, clear language.

CRITICAL RULES:
1. ONLY answer based on the user's health data provided in the context below. 
2. If the user asks about data you don't have, say "I don't have that information in your records."
3. NEVER diagnose conditions. Use phrases like "your levels suggest" or "this may indicate."
4. ALWAYS be encouraging and suggest consulting a healthcare professional for medical advice.
5. Keep responses concise (2-3 paragraphs max).
6. When citing specific values, include the unit and whether it's within the normal range.
7. End every response with a brief disclaimer.

USER'S HEALTH DATA CONTEXT:
{context}

DISCLAIMER TO APPEND: ⚠️ This is not a medical diagnosis. Always consult a qualified healthcare professional for medical advice."""


async def handle_chat(
    db: Session,
    user_id: UUID,
    message: str,
) -> dict:
    """
    Handle a Copilot chat message.
    
    1. Retrieve relevant health context for the user
    2. Build grounded prompt
    3. Call LLM
    4. Save messages and return response
    """
    # Step 1: Retrieve relevant context
    context, context_refs = _retrieve_context(db, user_id, message)

    # Step 2: Build messages
    system_msg = SYSTEM_PROMPT.format(context=context)

    # Get recent chat history (last 6 messages for context)
    recent_history = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(6)
        .all()
    )
    recent_history.reverse()

    messages = [{"role": "system", "content": system_msg}]
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": message})

    # Step 3: Call LLM
    logger.info(f"Copilot chat for user {user_id}: '{message[:50]}...'")
    response = await call_fireworks_api(
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )

    # Step 4: Save messages to database
    user_msg = ChatMessage(
        user_id=user_id,
        role="user",
        content=message,
    )
    assistant_msg = ChatMessage(
        user_id=user_id,
        role="assistant",
        content=response,
        context_used={"references": context_refs},
    )
    db.add(user_msg)
    db.add(assistant_msg)
    db.commit()

    return {
        "response": response,
        "context_used": context_refs,
    }


def get_chat_history(db: Session, user_id: UUID, limit: int = 50) -> list[ChatMessage]:
    """Get chat history for a user."""
    return (
        db.query(ChatMessage)
        .filter(
            ChatMessage.user_id == user_id,
            ChatMessage.role.in_(["user", "assistant"]),
        )
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )


def _retrieve_context(db: Session, user_id: UUID, query: str) -> tuple[str, list[str]]:
    """
    Retrieve relevant health context for the user's query.
    
    Simple keyword-based retrieval:
    1. Always include current biomarkers from Digital Twin
    2. Always include risk predictions
    3. Match query terms against biomarker names for specifics
    
    Returns (context_text, list_of_references)
    """
    context_parts = []
    references = []

    # Get Digital Twin state
    twin = get_or_create_twin(db, user_id)
    biomarkers = twin.current_biomarkers or {}

    if biomarkers:
        context_parts.append("=== Current Biomarker Values ===")
        for name, data in biomarkers.items():
            display = data.get("display_name", name)
            value = data.get("value", "N/A")
            unit = data.get("unit", "")
            status = data.get("status", "unknown")
            context_parts.append(f"- {display}: {value} {unit} (Status: {status})")
            references.append(f"biomarker:{name}")

    # Get risk predictions
    predictions = (
        db.query(RiskPrediction)
        .filter(RiskPrediction.user_id == user_id)
        .all()
    )

    if predictions:
        context_parts.append("\n=== Risk Predictions ===")
        for pred in predictions:
            context_parts.append(
                f"- {pred.disease_type.replace('_', ' ').title()}: "
                f"{pred.probability:.1%} probability, {pred.risk_level} risk"
            )
            if pred.feature_importance:
                top_factors = pred.feature_importance[:3]
                factors_str = ", ".join(
                    f"{f['feature']} ({f['direction'].replace('_', ' ')})"
                    for f in top_factors
                )
                context_parts.append(f"  Top risk factors: {factors_str}")
            references.append(f"prediction:{pred.id}")

    # Get health score
    context_parts.append(f"\n=== Overall Health ===")
    context_parts.append(f"- Health Score: {twin.health_score:.1f}/100")
    if twin.organ_scores:
        for organ, score in twin.organ_scores.items():
            context_parts.append(f"- {organ.title()} Health: {score:.0%}")

    # Get relevant report text (search for query terms in raw text)
    query_lower = query.lower()
    reports = (
        db.query(Report)
        .filter(Report.user_id == user_id, Report.status == "extracted")
        .order_by(Report.uploaded_at.desc())
        .limit(3)
        .all()
    )

    for report in reports:
        if report.raw_text and "pages" in report.raw_text:
            raw = report.raw_text["pages"]
            # Simple keyword matching
            if any(term in raw.lower() for term in query_lower.split() if len(term) > 3):
                # Include a snippet of relevant text
                snippet = raw[:500]
                context_parts.append(f"\n=== Relevant Report Text (uploaded {report.uploaded_at.strftime('%Y-%m-%d')}) ===")
                context_parts.append(snippet)
                references.append(f"report:{report.id}")

    context = "\n".join(context_parts) if context_parts else "No health data available for this user yet."

    return context, references
