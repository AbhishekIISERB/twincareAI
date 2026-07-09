"""Digital Twin state model — the living snapshot of a user's health."""

import uuid
from datetime import datetime

from sqlalchemy import Float, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class DigitalTwinState(Base):
    __tablename__ = "digital_twin_state"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # Latest biomarker snapshot: {name: {value, unit, status, recorded_at}}
    current_biomarkers: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    health_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Organ system scores: {heart: 0.82, liver: 0.91, kidney: 0.95, ...}
    organ_scores: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user = relationship("User", back_populates="digital_twin")
