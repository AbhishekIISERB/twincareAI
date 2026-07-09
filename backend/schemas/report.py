"""Report and biomarker schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ReportUploadResponse(BaseModel):
    """Response after uploading a report."""
    report_id: UUID
    status: str

    model_config = {"from_attributes": True}


class ReportStatusResponse(BaseModel):
    """Report processing status."""
    report_id: UUID
    status: str
    biomarkers_count: int = 0
    error_message: str | None = None

    model_config = {"from_attributes": True}


class ReportListItem(BaseModel):
    """Single report in the list."""
    id: UUID
    file_type: str
    original_filename: str
    status: str
    uploaded_at: datetime
    biomarkers_count: int = 0

    model_config = {"from_attributes": True}


class ReportListResponse(BaseModel):
    """List of user reports."""
    reports: list[ReportListItem]


class BiomarkerResponse(BaseModel):
    """Single biomarker data."""
    id: UUID
    name: str
    value: float
    unit: str
    reference_range: str | None = None
    status: str
    recorded_at: datetime

    model_config = {"from_attributes": True}


class BiomarkerListResponse(BaseModel):
    """List of biomarkers from a report."""
    report_id: UUID
    biomarkers: list[BiomarkerResponse]
