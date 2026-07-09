"""Digital Twin service — manages the living health state snapshot."""

import logging
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from models.digital_twin import DigitalTwinState
from models.biomarker import Biomarker
from models.user import User
from utils.biomarker_ranges import (
    normalize_biomarker_name,
    classify_biomarker,
    ORGAN_SYSTEMS,
    BIOMARKER_RANGES,
    get_display_name,
)

logger = logging.getLogger(__name__)


def get_or_create_twin(db: Session, user_id: UUID) -> DigitalTwinState:
    """Get or create a Digital Twin state for a user."""
    twin = db.query(DigitalTwinState).filter(DigitalTwinState.user_id == user_id).first()
    if not twin:
        twin = DigitalTwinState(
            user_id=user_id,
            current_biomarkers={},
            health_score=0.0,
            organ_scores={},
        )
        db.add(twin)
        db.commit()
        db.refresh(twin)
    return twin


def update_twin_from_report(db: Session, user_id: UUID, report_id: UUID) -> DigitalTwinState:
    """
    Update the Digital Twin state from newly extracted biomarkers.
    
    Merges new biomarker values into the existing state, 
    recalculates health score and organ scores.
    """
    twin = get_or_create_twin(db, user_id)

    # Get biomarkers from the new report
    biomarkers = (
        db.query(Biomarker)
        .filter(Biomarker.report_id == report_id)
        .all()
    )

    if not biomarkers:
        return twin

    # Update current biomarkers snapshot (newer values overwrite older)
    current = dict(twin.current_biomarkers) if twin.current_biomarkers else {}
    for bio in biomarkers:
        canonical = normalize_biomarker_name(bio.name)
        current[canonical] = {
            "value": bio.value,
            "unit": bio.unit,
            "status": bio.status,
            "display_name": get_display_name(canonical),
            "recorded_at": bio.recorded_at.isoformat() if bio.recorded_at else datetime.now(timezone.utc).isoformat(),
        }

    twin.current_biomarkers = current

    # Recalculate organ scores
    organ_scores = _calculate_organ_scores(current)
    twin.organ_scores = organ_scores

    # Recalculate health score
    twin.health_score = _calculate_health_score(current, organ_scores)

    twin.last_updated = datetime.now(timezone.utc)

    db.commit()
    db.refresh(twin)

    logger.info(f"Digital Twin updated for user {user_id}: health_score={twin.health_score:.1f}")
    return twin


def get_twin_state(db: Session, user_id: UUID) -> dict:
    """Get the current Digital Twin state for a user."""
    twin = get_or_create_twin(db, user_id)
    return {
        "health_score": twin.health_score,
        "organ_scores": twin.organ_scores or {},
        "current_biomarkers": twin.current_biomarkers or {},
        "last_updated": twin.last_updated,
    }


def _calculate_organ_scores(biomarkers: dict) -> dict:
    """
    Calculate a health score (0-1) for each organ system based on its biomarkers.
    
    Score of 1.0 = all biomarkers normal
    Score decreases for abnormal values
    """
    organ_scores = {}

    for organ, bio_names in ORGAN_SYSTEMS.items():
        organ_bios = []
        for bio_name in bio_names:
            if bio_name in biomarkers:
                organ_bios.append(biomarkers[bio_name])

        if not organ_bios:
            continue  # No data for this organ system

        # Score each biomarker
        scores = []
        for bio in organ_bios:
            status = bio.get("status", "normal")
            if status == "normal":
                scores.append(1.0)
            elif status == "low" or status == "high":
                scores.append(0.6)
            elif status == "critical":
                scores.append(0.2)
            else:
                scores.append(0.8)

        organ_scores[organ] = round(sum(scores) / len(scores), 2)

    return organ_scores


def _calculate_health_score(biomarkers: dict, organ_scores: dict) -> float:
    """
    Calculate overall health score (0-100) from organ scores and biomarker statuses.
    """
    if not organ_scores:
        return 0.0

    # Weighted average of organ scores
    # Heart and metabolic are weighted higher (more clinical importance)
    weights = {
        "heart": 1.5,
        "liver": 1.2,
        "kidney": 1.3,
        "blood": 1.0,
        "metabolic": 1.4,
        "thyroid": 0.8,
    }

    total_weight = 0
    weighted_sum = 0

    for organ, score in organ_scores.items():
        weight = weights.get(organ, 1.0)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # Convert 0-1 scale to 0-100
    return round((weighted_sum / total_weight) * 100, 1)
