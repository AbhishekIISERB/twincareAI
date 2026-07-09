"""Dashboard aggregation API endpoint."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from schemas.dashboard import DashboardResponse, RiskSummaryItem, TimelineEvent
from services.dashboard_service import get_dashboard_data
from utils.security import get_current_user_id

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("", response_model=DashboardResponse)
async def get_dashboard(
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Get aggregated dashboard data for the current user.
    
    Combines health score, risk summary, recent biomarkers, 
    AI-generated insight, and timeline into one response.
    """
    data = await get_dashboard_data(db, user_id)

    # Convert risk_summary dicts to schema objects
    risk_summary = {
        k: RiskSummaryItem(**v) for k, v in data.get("risk_summary", {}).items()
    }

    # Convert timeline dicts to schema objects
    timeline = [TimelineEvent(**t) for t in data.get("timeline", [])]

    return DashboardResponse(
        health_score=data["health_score"],
        risk_summary=risk_summary,
        recent_biomarkers=data.get("recent_biomarkers", []),
        ai_insight=data.get("ai_insight", ""),
        timeline=timeline,
        disclaimer=data.get("disclaimer", ""),
    )
