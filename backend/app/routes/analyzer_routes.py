from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from app.models.schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from app.services.simple_ai_service import SentimentAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text using AI models
    
    Supports:
    - Sentiment analysis (local and external)
    - Text input via JSON
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Analysis request {analysis_id}: task={request.task}, use_external={request.use_external}")
        
        # Validate input
        if not request.input_text or request.input_text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="input_text is required and cannot be empty"
            )
        
        # Currently only supporting sentiment analysis
        if request.task != "sentiment":
            raise HTTPException(
                status_code=400,
                detail=f"Task '{request.task}' is not yet supported. Currently supported: sentiment"
            )
        
        # Perform analysis
        result, engine = await sentiment_analyzer.analyze_sentiment(
            text=request.input_text,
            use_external=request.use_external,
            options=request.options
        )
        
        # Calculate processing time
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Create response
        response = AnalyzeResponse(
            id=analysis_id,
            task=request.task,
            engine=engine,
            result=result,
            elapsed_ms=elapsed_ms,
            received_at=datetime.now()
        )
        
        logger.info(f"Analysis {analysis_id} completed successfully in {elapsed_ms}ms")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error in analysis {analysis_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/analyze/file")
async def analyze_file(
    task: str = Form(...),
    use_external: bool = Form(False),
    file: UploadFile = File(...)
):
    """
    Analyze uploaded file (for future implementation)
    Currently returns not implemented
    """
    raise HTTPException(
        status_code=501,
        detail="File upload analysis is not yet implemented"
    )
