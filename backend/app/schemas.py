"""
Modelos Pydantic para requests e responses da API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
import uuid


class TaskType(str, Enum):
    """Tipos de tarefas suportadas pelo sistema"""
    SENTIMENT = "sentiment"
    IMAGE_SENTIMENT = "image_sentiment"
    AUDIO_SENTIMENT = "audio_sentiment"
    TEXT_TO_IMAGE = "text_to_image"


class SentimentLabel(str, Enum):
    """Labels de sentimentos possíveis"""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class AnalyzeRequest(BaseModel):
    """Request para análise de texto"""
    task: TaskType = TaskType.SENTIMENT
    input_text: str = Field(..., min_length=1, max_length=2000, description="Texto para análise")
    use_external: bool = Field(False, description="Usar API externa (Hugging Face)")
    options: Optional[Dict[str, Any]] = Field(None, description="Opções adicionais")
    
    @validator('input_text')
    def validate_input_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Texto não pode estar vazio')
        return v.strip()


class ImageAnalyzeRequest(BaseModel):
    """Request para análise de imagem"""
    image_data: str = Field(..., description="Imagem codificada em Base64")
    filename: str = Field(..., description="Nome do arquivo")
    extract_text: bool = Field(True, description="Extrair texto da imagem")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v:
            raise ValueError('Dados da imagem são obrigatórios')
        # Remover prefixo data:image/* se presente
        if v.startswith('data:image'):
            v = v.split(',')[1]
        return v
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v:
            raise ValueError('Nome do arquivo é obrigatório')
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f'Extensão não suportada. Use: {", ".join(allowed_extensions)}')
        return v


class AudioAnalyzeRequest(BaseModel):
    """Request para análise de áudio"""
    audio_data: str = Field(..., description="Áudio codificado em Base64")
    filename: str = Field(..., description="Nome do arquivo")
    language: str = Field("pt-BR", description="Idioma do áudio")
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        if not v:
            raise ValueError('Dados do áudio são obrigatórios')
        return v
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v:
            raise ValueError('Nome do arquivo é obrigatório')
        allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f'Extensão não suportada. Use: {", ".join(allowed_extensions)}')
        return v


class SentimentResult(BaseModel):
    """Resultado da análise de sentimento"""
    label: SentimentLabel = Field(..., description="Classificação do sentimento")
    score: float = Field(..., ge=0.0, le=1.0, description="Pontuação de confiança")
    debug: Optional[Dict[str, Any]] = Field(None, description="Informações de debug")


class MediaAnalysis(BaseModel):
    """Resultado da análise de mídia"""
    extracted_text: Optional[str] = Field(None, description="Texto extraído da mídia")
    media_type: str = Field(..., description="Tipo de mídia processada")
    file_size: Optional[int] = Field(None, description="Tamanho do arquivo em bytes")
    dimensions: Optional[Dict[str, int]] = Field(None, description="Dimensões (para imagens)")
    duration: Optional[float] = Field(None, description="Duração (para áudios)")
    processing_method: Optional[str] = Field(None, description="Método usado para processamento")


class AnalyzeResponse(BaseModel):
    """Response padrão para análises"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID único da análise")
    task: TaskType = Field(..., description="Tipo de tarefa executada")
    engine: str = Field(..., description="Engine utilizada para processamento")
    result: SentimentResult = Field(..., description="Resultado da análise")
    media_analysis: Optional[MediaAnalysis] = Field(None, description="Análise de mídia (se aplicável)")
    elapsed_ms: int = Field(..., ge=0, description="Tempo de processamento em millisegundos")
    received_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp do recebimento")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Response do health check"""
    status: str = Field("ok", description="Status do serviço")
    version: str = Field(..., description="Versão da API")
    uptime: Optional[float] = Field(None, description="Tempo ativo em segundos")
    features: Optional[Dict[str, bool]] = Field(None, description="Funcionalidades disponíveis")


class CapabilitiesResponse(BaseModel):
    """Response das capacidades do sistema"""
    text_analysis: bool = Field(True, description="Análise de texto disponível")
    media_support: bool = Field(..., description="Suporte a mídia disponível")
    image_processing: bool = Field(..., description="Processamento de imagens disponível")
    audio_processing: bool = Field(..., description="Processamento de áudios disponível")
    ocr: bool = Field(..., description="OCR disponível")
    speech_recognition: bool = Field(..., description="Reconhecimento de fala disponível")
    external_apis: bool = Field(..., description="APIs externas configuradas")
    supported_formats: Dict[str, List[str]] = Field(..., description="Formatos suportados")


class ErrorResponse(BaseModel):
    """Response de erro padronizado"""
    error: str = Field(..., description="Tipo do erro")
    message: str = Field(..., description="Mensagem detalhada")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalhes adicionais")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp do erro")
    request_id: Optional[str] = Field(None, description="ID da requisição que gerou o erro")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
