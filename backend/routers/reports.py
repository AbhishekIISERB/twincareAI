"""Report upload and extraction API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from database import get_db
from schemas.report import (
    ReportUploadResponse,
    ReportStatusResponse,
    ReportListResponse,
    ReportListItem,
    BiomarkerListResponse,
    BiomarkerResponse,
)
from services.extraction_service import (
    handle_upload,
    process_report,
    get_report_status,
    get_user_reports,
    get_report_biomarkers,
)
from utils.security import get_current_user_id

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("/upload", response_model=ReportUploadResponse, status_code=202)
async def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Upload a health report (PDF or image) for processing.
    
    The report will be processed in the background:
    1. OCR text extraction
    2. Biomarker identification
    3. Digital Twin state update
    4. Risk prediction
    
    Check processing status via GET /reports/{report_id}/status
    """
    file_content = await file.read()
    report = await handle_upload(db, user_id, file_content, file.filename)

    # Process in background
    background_tasks.add_task(_process_report_task, report.id, user_id)

    return ReportUploadResponse(report_id=report.id, status=report.status)


async def _process_report_task(report_id: UUID, user_id: UUID):
    """Background task to process a report."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        await process_report(db, report_id, user_id)
    finally:
        db.close()


@router.get("/{report_id}/status", response_model=ReportStatusResponse)
def check_status(
    report_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Check the processing status of an uploaded report."""
    status_data = get_report_status(db, report_id, user_id)
    return ReportStatusResponse(**status_data)


@router.get("", response_model=ReportListResponse)
def list_reports(
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """List all reports for the current user."""
    reports = get_user_reports(db, user_id)
    return ReportListResponse(
        reports=[ReportListItem(**r) for r in reports]
    )


@router.get("/{report_id}/biomarkers", response_model=BiomarkerListResponse)
def list_biomarkers(
    report_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Get all biomarkers extracted from a specific report."""
    biomarkers = get_report_biomarkers(db, report_id, user_id)
    return BiomarkerListResponse(
        report_id=report_id,
        biomarkers=[BiomarkerResponse.model_validate(b) for b in biomarkers],
    )
