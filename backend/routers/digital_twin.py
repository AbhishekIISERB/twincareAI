"""Digital Twin API endpoint."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from schemas.digital_twin import DigitalTwinResponse
from services.digital_twin_service import get_twin_state
from utils.security import get_current_user_id

router = APIRouter(prefix="/digital-twin", tags=["Digital Twin"])


@router.get("", response_model=DigitalTwinResponse)
def get_digital_twin(
    user_id: UUID = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Get the current Digital Twin state for the authenticated user.
    
    Returns health score, organ scores, and current biomarker snapshot.
    """
    state = get_twin_state(db, user_id)
    return DigitalTwinResponse(**state)
