"""Digital Twin schemas."""

from datetime import datetime

from pydantic import BaseModel

from config import settings


class BiomarkerSnapshot(BaseModel):
    """A single biomarker in the Digital Twin state."""
    value: float
    unit: str
    status: str
    recorded_at: str


class DigitalTwinResponse(BaseModel):
    """Digital Twin state response."""
    health_score: float
    organ_scores: dict[str, float]
    current_biomarkers: dict[str, BiomarkerSnapshot]
    last_updated: datetime | None = None
    disclaimer: str = settings.DISCLAIMER

    model_config = {"from_attributes": True}
