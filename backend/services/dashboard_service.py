"""Dashboard aggregation service — combines all health data into one response."""

import logging
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session

from config import settings
from models.report import Report
from models.biomarker import Biomarker
from models.risk_prediction import RiskPrediction
from services.digital_twin_service import get_or_create_twin
from ai.llm_client import generate_health_insight

logger = logging.getLogger(__name__)


async def get_dashboard_data(db: Session, user_id: UUID) -> dict:
    """
    Aggregate all health data for the dashboard view.
    
    Returns: health_score, risk_summary, recent_biomarkers, ai_insight, timeline
    """
    # Get Digital Twin state
    twin = get_or_create_twin(db, user_id)

    # Get risk predictions
    predictions = (
        db.query(RiskPrediction)
        .filter(RiskPrediction.user_id == user_id)
        .all()
    )

    risk_summary = {}
    predictions_data = []
    for pred in predictions:
        risk_summary[pred.disease_type] = {
            "probability": pred.probability,
            "risk_level": pred.risk_level,
        }
        predictions_data.append({
            "disease_type": pred.disease_type,
            "risk_level": pred.risk_level,
            "probability": pred.probability,
        })

    # Get recent biomarkers (from the latest report)
    recent_biomarkers = (
        db.query(Biomarker)
        .filter(Biomarker.user_id == user_id)
        .order_by(Biomarker.recorded_at.desc())
        .limit(10)
        .all()
    )

    recent_bio_list = [
        {
            "name": bio.name,
            "value": bio.value,
            "unit": bio.unit,
            "status": bio.status,
            "reference_range": bio.reference_range,
            "recorded_at": bio.recorded_at.isoformat() if bio.recorded_at else None,
        }
        for bio in recent_biomarkers
    ]

    # Build timeline
    timeline = _build_timeline(db, user_id)

    # Generate AI insight
    try:
        ai_insight = await generate_health_insight(
            twin.current_biomarkers or {},
            predictions_data,
        )
    except Exception as e:
        logger.error(f"AI insight generation failed: {e}")
        ai_insight = "Upload a health report to get personalized insights powered by AI. 🧬"

    return {
        "health_score": twin.health_score,
        "risk_summary": risk_summary,
        "recent_biomarkers": recent_bio_list,
        "ai_insight": ai_insight,
        "timeline": timeline,
        "disclaimer": settings.DISCLAIMER,
    }


def _build_timeline(db: Session, user_id: UUID) -> list[dict]:
    """Build a chronological timeline of health events."""
    events = []

    # Report uploads
    reports = (
        db.query(Report)
        .filter(Report.user_id == user_id)
        .order_by(Report.uploaded_at.desc())
        .limit(10)
        .all()
    )

    for report in reports:
        bio_count = db.query(Biomarker).filter(Biomarker.report_id == report.id).count()
        events.append({
            "date": report.uploaded_at.isoformat() if report.uploaded_at else None,
            "event": f"Blood report uploaded — {bio_count} biomarkers extracted" if report.status == "extracted"
                     else f"Report uploaded ({report.status})",
            "type": "report",
        })

    # Risk predictions
    predictions = (
        db.query(RiskPrediction)
        .filter(RiskPrediction.user_id == user_id)
        .order_by(RiskPrediction.predicted_at.desc())
        .limit(5)
        .all()
    )

    for pred in predictions:
        events.append({
            "date": pred.predicted_at.isoformat() if pred.predicted_at else None,
            "event": f"{pred.disease_type.replace('_', ' ').title()} risk assessed: {pred.risk_level}",
            "type": "prediction",
        })

    # Sort by date (most recent first)
    events.sort(key=lambda x: x["date"] or "", reverse=True)

    return events[:15]  # Limit to 15 most recent events
