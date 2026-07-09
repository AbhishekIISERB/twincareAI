"""Risk prediction service — runs the heart disease model and stores results."""

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from models.risk_prediction import RiskPrediction
from models.user import User
from services.digital_twin_service import get_or_create_twin
from ai.heart_model import predict_risk
from ai.shap_explainer import compute_shap_values

logger = logging.getLogger(__name__)


async def run_prediction_for_user(
    db: Session,
    user_id: UUID,
    report_id: UUID | None = None,
) -> RiskPrediction:
    """
    Run heart disease risk prediction for a user using their current Digital Twin state.
    
    Creates a new RiskPrediction record with probability, confidence, SHAP values,
    and feature importance.
    """
    # Get user profile
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Get current Digital Twin state
    twin = get_or_create_twin(db, user_id)
    biomarkers = twin.current_biomarkers or {}

    if not biomarkers:
        logger.warning(f"No biomarker data for user {user_id} — skipping prediction")
        return None

    # Build user profile dict
    user_profile = {
        "date_of_birth": str(user.date_of_birth) if user.date_of_birth else None,
        "gender": user.gender,
    }

    # Run prediction
    logger.info(f"Running heart disease prediction for user {user_id}")
    result = predict_risk(biomarkers, user_profile)

    # Compute SHAP explainability
    explanation = compute_shap_values(result["features_used"], result)

    # Create or update prediction record
    # Delete old heart_disease prediction for this user (keep only latest)
    db.query(RiskPrediction).filter(
        RiskPrediction.user_id == user_id,
        RiskPrediction.disease_type == "heart_disease",
    ).delete()

    prediction = RiskPrediction(
        user_id=user_id,
        report_id=report_id,
        disease_type="heart_disease",
        probability=result["probability"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        shap_values=explanation["shap_values"],
        feature_importance=explanation["feature_importance"],
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    logger.info(
        f"Prediction saved: {result['risk_level']} risk "
        f"({result['probability']:.1%} probability, {result['confidence']:.1%} confidence)"
    )
    return prediction


def get_user_predictions(db: Session, user_id: UUID) -> list[RiskPrediction]:
    """Get all risk predictions for a user."""
    return (
        db.query(RiskPrediction)
        .filter(RiskPrediction.user_id == user_id)
        .order_by(RiskPrediction.predicted_at.desc())
        .all()
    )


def get_prediction_by_id(
    db: Session, prediction_id: UUID, user_id: UUID
) -> RiskPrediction:
    """Get a specific prediction by ID."""
    prediction = db.query(RiskPrediction).filter(
        RiskPrediction.id == prediction_id,
        RiskPrediction.user_id == user_id,
    ).first()

    if not prediction:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found"
        )

    return prediction
