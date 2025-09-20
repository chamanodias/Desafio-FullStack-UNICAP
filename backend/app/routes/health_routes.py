from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter()

@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok")
