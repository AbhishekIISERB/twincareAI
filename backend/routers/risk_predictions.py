"""Risk prediction API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from schemas.risk_prediction import (
    RiskPredictionResponse,
    RiskPredictionListResponse,
    ShapValues,
    FeatureImportance,
)
from config import settings
from services.prediction_service import get_user_predictions, get_prediction_by_id
from utils.security import get_current_user_id

router = APIRouter(prefix="/risk-predictions", tags=["Risk Predictions"])


@router.get("", response_model=RiskPredictionListResponse)
def list_predictions(
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Get all risk predictions for the current user."""
    predictions = get_user_predictions(db, user_id)
    return RiskPredictionListResponse(
        predictions=[_format_prediction(p) for p in predictions]
    )


@router.get("/{prediction_id}", response_model=RiskPredictionResponse)
def get_prediction(
    prediction_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Get a specific risk prediction with SHAP explainability data."""
    prediction = get_prediction_by_id(db, prediction_id, user_id)
    return _format_prediction(prediction)


def _format_prediction(prediction) -> RiskPredictionResponse:
    """Format a prediction model into the response schema."""
    shap = None
    if prediction.shap_values:
        shap = ShapValues(
            features=prediction.shap_values.get("features", []),
            values=prediction.shap_values.get("values", []),
            base_value=prediction.shap_values.get("base_value", 0),
        )

    feature_imp = None
    if prediction.feature_importance:
        feature_imp = [
            FeatureImportance(
                feature=fi.get("feature", ""),
                importance=fi.get("importance", 0),
                direction=fi.get("direction", "unknown"),
            )
            for fi in prediction.feature_importance
        ]

    return RiskPredictionResponse(
        id=prediction.id,
        disease_type=prediction.disease_type,
        probability=prediction.probability,
        confidence=prediction.confidence,
        risk_level=prediction.risk_level,
        shap_values=shap,
        feature_importance=feature_imp,
        predicted_at=prediction.predicted_at,
        disclaimer=settings.DISCLAIMER,
    )
