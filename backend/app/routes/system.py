"""
Rotas do sistema - health check, capabilities e informações gerais
"""
from fastapi import APIRouter
import time
import psutil
import platform
from typing import Dict, Any

from ..schemas import HealthResponse, CapabilitiesResponse
from ..config import config

# Tempo de inicialização para calcular uptime
START_TIME = time.time()

# Criar roteador
router = APIRouter(prefix="/api/v1", tags=["System"])


@router.get("/healthz", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """
    Verifica o status de saúde do sistema
    
    Retorna informações sobre:
    - Status do serviço
    - Versão da API
    - Tempo de atividade
    - Funcionalidades disponíveis
    """
    uptime_seconds = time.time() - START_TIME
    
    # Verificar funcionalidades disponíveis
    features = {
        "text_analysis": True,
        "media_support": _check_media_support(),
        "external_apis": _check_external_apis(),
        "database": _check_database_connection(),
        "cache": config.ENABLE_CACHE
    }
    
    return HealthResponse(
        status="ok",
        version=config.VERSAO,
        uptime=round(uptime_seconds, 2),
        features=features
    )


@router.get("/capabilities", response_model=CapabilitiesResponse, summary="Capacidades do Sistema")
async def get_capabilities() -> CapabilitiesResponse:
    """
    Retorna as capacidades e funcionalidades disponíveis no sistema
    
    Inclui informações sobre:
    - Tipos de análise suportados
    - Formatos de arquivo aceitos
    - APIs externas configuradas
    - Recursos de processamento
    """
    # Verificar se serviços de mídia estão disponíveis
    media_support = _check_media_support()
    
    return CapabilitiesResponse(
        text_analysis=True,
        media_support=media_support,
        image_processing=media_support,
        audio_processing=media_support,  # Mesmo status que mídia geral
        ocr=media_support,
        speech_recognition=media_support,
        external_apis=_check_external_apis(),
        supported_formats={
            "image": config.ALLOWED_IMAGE_EXTENSIONS,
            "audio": config.ALLOWED_AUDIO_EXTENSIONS,
            "text": [".txt", ".json"]
        }
    )


@router.get("/info", summary="Informações do Sistema")
async def get_system_info() -> Dict[str, Any]:
    """
    Retorna informações detalhadas sobre o sistema e ambiente
    
    Útil para debugging e monitoramento
    """
    try:
        # Informações do sistema
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor() or "Unknown",
            "hostname": platform.node()
        }
        
        # Informações de recursos
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        resource_info = {
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent
        }
        
        # Informações da aplicação
        app_info = {
            "name": config.NOME_PROJETO,
            "version": config.VERSAO,
            "description": config.DESCRICAO,
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "debug_mode": config.DEBUG
        }
        
        return {
            "system": system_info,
            "resources": resource_info,
            "application": app_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        # Fallback caso alguma informação não esteja disponível
        return {
            "system": {"error": str(e)},
            "application": {
                "name": config.NOME_PROJETO,
                "version": config.VERSAO,
                "uptime_seconds": round(time.time() - START_TIME, 2)
            },
            "timestamp": time.time()
        }


def _check_media_support() -> bool:
    """Verifica se o suporte a mídia está disponível"""
    try:
        from ..services.media_service import MediaService
        return True
    except ImportError:
        return False


def _check_external_apis() -> bool:
    """Verifica se APIs externas estão configuradas"""
    return bool(config.HUGGINGFACE_API_KEY or config.OPENAI_API_KEY)


def _check_database_connection() -> bool:
    """Verifica conexão com banco de dados (para implementações futuras)"""
    # Por enquanto sempre retorna False, mas pode ser implementado
    return False
