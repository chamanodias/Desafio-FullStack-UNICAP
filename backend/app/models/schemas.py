from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    SENTIMENT = "sentiment"
    NER = "ner"
    OCR = "ocr"
    CAPTION = "caption"
    CUSTOM = "custom"

class AnalyzeRequest(BaseModel):
    task: TaskType
    input_text: Optional[str] = None
    use_external: bool = False
    options: Optional[Dict[str, Any]] = None

class SentimentResult(BaseModel):
    label: str
    score: float

class AnalyzeResponse(BaseModel):
    id: str
    task: str
    engine: str
    result: Dict[str, Any]
    elapsed_ms: int
    received_at: datetime

class HealthResponse(BaseModel):
    status: str = "ok"

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
