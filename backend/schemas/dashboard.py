"""Dashboard aggregation schemas."""

from datetime import datetime

from pydantic import BaseModel

from config import settings


class RiskSummaryItem(BaseModel):
    """Summary of a single disease risk."""
    probability: float
    risk_level: str


class TimelineEvent(BaseModel):
    """Single event in the health timeline."""
    date: datetime
    event: str
    type: str  # report, prediction, insight


class DashboardResponse(BaseModel):
    """Aggregated dashboard data."""
    health_score: float
    risk_summary: dict[str, RiskSummaryItem]
    recent_biomarkers: list[dict]
    ai_insight: str
    timeline: list[TimelineEvent]
    disclaimer: str = settings.DISCLAIMER
