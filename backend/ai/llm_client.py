"""Fireworks AI API client for LLM inference.

Used for:
- Biomarker extraction fallback (structured output)
- Copilot chat (conversational, grounded in user data)
- AI insight generation (one-shot summaries)

Prefers Gemma 2 for the AMD hackathon Gemma bonus eligibility.
"""

import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


async def call_fireworks_api(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
) -> str:
    """
    Call the Fireworks AI API for chat completions.
    
    Args:
        messages: List of chat messages [{"role": "user/system/assistant", "content": "..."}]
        model: Model ID (defaults to settings.FIREWORKS_MODEL)
        temperature: Sampling temperature
        max_tokens: Max response tokens
        stream: Whether to stream (not implemented — returns full response)
    
    Returns:
        The assistant's response text.
    """
    if not settings.FIREWORKS_API_KEY:
        logger.warning("Fireworks API key not configured — returning fallback response")
        return _fallback_response(messages)

    model = model or settings.FIREWORKS_MODEL

    headers = {
        "Authorization": f"Bearer {settings.FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.FIREWORKS_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        logger.error(f"Fireworks API HTTP error {e.response.status_code}: {e.response.text}")
        return _fallback_response(messages)
    except Exception as e:
        logger.error(f"Fireworks API error: {e}")
        return _fallback_response(messages)


def _fallback_response(messages: list[dict]) -> str:
    """
    Generate a basic fallback when Fireworks API is unavailable.
    This ensures the demo can still function without API connectivity.
    """
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if "extract" in last_user_msg.lower() or "biomarker" in last_user_msg.lower():
        return "[]"  # Empty biomarker array for extraction fallback

    return (
        "I'm currently unable to process this request as the AI service is temporarily "
        "unavailable. Please try again in a moment. In the meantime, you can review your "
        "biomarker data and risk predictions on the dashboard.\n\n"
        "⚠️ This is not a medical diagnosis. Always consult a healthcare professional."
    )


async def generate_health_insight(biomarkers: dict, risk_predictions: list[dict]) -> str:
    """
    Generate a one-sentence AI health insight for the dashboard.
    
    Uses Fireworks API (Gemma) for natural language generation.
    """
    # Build context
    abnormal = []
    for name, data in biomarkers.items():
        if data.get("status") in ("high", "low", "critical"):
            abnormal.append(f"{data.get('display_name', name)}: {data.get('value')} {data.get('unit')} ({data.get('status')})")

    risk_info = []
    for pred in risk_predictions:
        risk_info.append(f"{pred.get('disease_type', 'unknown')}: {pred.get('risk_level', 'unknown')} risk ({pred.get('probability', 0):.0%})")

    if not abnormal and not risk_info:
        return "Your recent lab results look good! Keep maintaining a healthy lifestyle. 🌟"

    prompt = f"""You are a health AI assistant. Based on the following health data, generate ONE concise, 
actionable insight sentence (max 30 words). Be encouraging but honest. Do NOT diagnose.

Abnormal biomarkers: {', '.join(abnormal) if abnormal else 'None'}
Risk assessments: {', '.join(risk_info) if risk_info else 'None'}

Generate only the insight sentence, nothing else."""

    response = await call_fireworks_api(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
    )

    return response.strip().strip('"')
