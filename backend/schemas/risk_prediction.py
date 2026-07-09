"""Risk prediction schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from config import settings


class ShapValues(BaseModel):
    """SHAP explainability data for waterfall chart."""
    features: list[str]
    values: list[float]
    base_value: float


class FeatureImportance(BaseModel):
    """Single feature's importance in the prediction."""
    feature: str
    importance: float
    direction: str  # increases_risk / decreases_risk


class RiskPredictionResponse(BaseModel):
    """Single risk prediction response."""
    id: UUID
    disease_type: str
    probability: float
    confidence: float
    risk_level: str
    shap_values: ShapValues | None = None
    feature_importance: list[FeatureImportance] | None = None
    predicted_at: datetime
    disclaimer: str = settings.DISCLAIMER

    model_config = {"from_attributes": True}


class RiskPredictionListResponse(BaseModel):
    """List of risk predictions."""
    predictions: list[RiskPredictionResponse]
