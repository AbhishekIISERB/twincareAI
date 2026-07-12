"""Patient intake API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.patient_intake import PatientIntake
from schemas.patient_intake import PatientIntakeCreate, PatientIntakeResponse
from utils.security import get_current_user_id

router = APIRouter(prefix="/patient-intake", tags=["Patient Intake"])


@router.get("", response_model=PatientIntakeResponse)
def get_intake(
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Retrieve the current user's clinical intake questionnaire."""
    intake = db.query(PatientIntake).filter(PatientIntake.user_id == user_id).first()
    if not intake:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient intake questionnaire has not been completed yet."
        )
    return intake


@router.post("", response_model=PatientIntakeResponse)
async def create_or_update_intake(
    data: PatientIntakeCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Create or update the current user's clinical intake questionnaire and optional manual biomarkers."""
    intake = db.query(PatientIntake).filter(PatientIntake.user_id == user_id).first()
    
    # Separate intake fields and manual biomarker fields
    biomarker_keys = {"fasting_glucose", "total_cholesterol", "systolic_bp", "diastolic_bp", "bmi", "heart_rate"}
    intake_data = data.model_dump(exclude_unset=True, exclude=biomarker_keys)
    
    if intake:
        for field, val in intake_data.items():
            setattr(intake, field, val)
    else:
        intake = PatientIntake(user_id=user_id, **intake_data)
        db.add(intake)
        
    db.commit()
    db.refresh(intake)
    
    # Process Manual Biomarkers if provided
    biomarkers_provided = {
        "fasting_glucose": data.fasting_glucose,
        "total_cholesterol": data.total_cholesterol,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "bmi": data.bmi,
        "heart_rate": data.heart_rate
    }
    
    if any(v is not None for v in biomarkers_provided.values()):
        from models.report import Report
        from models.biomarker import Biomarker
        from ai.biomarker_extractor import get_biomarker_unit, get_reference_range_str, classify_biomarker
        from datetime import datetime, timezone
        import logging
        
        # Create a "Manual Entry" report to anchor the biomarkers
        report = Report(
            user_id=user_id,
            file_path="manual_entry",
            file_type="manual",
            original_filename="Manual Entry",
            status="extracted",
            processed_at=datetime.now(timezone.utc)
        )
        db.add(report)
        db.flush()
        
        for name, value in biomarkers_provided.items():
            if value is not None:
                b = Biomarker(
                    report_id=report.id,
                    user_id=user_id,
                    name=name,
                    value=value,
                    unit=get_biomarker_unit(name),
                    reference_range=get_reference_range_str(name),
                    status=classify_biomarker(name, value)
                )
                db.add(b)
                
        db.commit()
        
        # Update Digital Twin and Run Prediction
        try:
            from services.digital_twin_service import update_twin_from_report
            update_twin_from_report(db, user_id, report.id)
            
            from services.prediction_service import run_prediction_for_user
            await run_prediction_for_user(db, user_id, report.id)
        except Exception as e:
            logging.error(f"Prediction failed after manual biomarker entry: {e}")
            
    return intake
