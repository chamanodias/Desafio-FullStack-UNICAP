"""
Rotas para análise de sentimentos em texto, imagem e áudio
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import time
import logging
from typing import Dict, Any

from ..schemas import (
    AnalyzeRequest, ImageAnalyzeRequest, AudioAnalyzeRequest,
    AnalyzeResponse, SentimentResult, MediaAnalysis, TaskType
)
from ..services.sentiment_service import SentimentService
from ..services.media_service import MediaService
from ..config import config

# Configurar logger
logger = logging.getLogger(__name__)

# Criar roteador
router = APIRouter(prefix="/api/v1", tags=["Analysis"])

# Inicializar serviços
sentiment_service = SentimentService()
media_service = MediaService()


@router.post("/analyze", response_model=AnalyzeResponse, summary="Análise de Sentimento em Texto")
async def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analisa o sentimento de um texto fornecido
    
    - **input_text**: Texto para análise (até 2000 caracteres)
    - **use_external**: Se deve usar API externa (Hugging Face)
    - **options**: Opções adicionais para análise
    
    Retorna a classificação do sentimento (POSITIVE/NEGATIVE/NEUTRAL) com pontuação de confiança
    """
    start_time = time.time()
    
    try:
        logger.info(f"📝 Iniciando análise de texto: {len(request.input_text)} chars, external={request.use_external}")
        
        # Processar análise de sentimento
        result = await sentiment_service.analyze_sentiment(
            text=request.input_text,
            use_external=request.use_external,
            options=request.options
        )
        
        # Calcular tempo decorrido
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Determinar engine utilizada
        engine = "external:huggingface" if request.use_external else "local:simple-analyzer"
        
        logger.info(f"✅ Análise concluída em {elapsed_ms}ms: {result['label']} ({result['score']:.2f})")
        
        return AnalyzeResponse(
            task=TaskType.SENTIMENT,
            engine=engine,
            result=SentimentResult(**result),
            elapsed_ms=elapsed_ms
        )
        
    except ValueError as e:
        logger.error(f"❌ Erro de validação: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro de validação: {str(e)}"
        )
    except Exception as e:
        logger.error(f"💥 Erro interno na análise: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno do servidor: {str(e)}"
        )


@router.post("/analyze/image", response_model=AnalyzeResponse, summary="Análise de Sentimento em Imagem")
async def analyze_image(request: ImageAnalyzeRequest) -> AnalyzeResponse:
    """
    Analisa o sentimento do texto extraído de uma imagem via OCR
    
    - **image_data**: Imagem codificada em Base64
    - **filename**: Nome do arquivo da imagem
    - **extract_text**: Se deve extrair texto da imagem (padrão: true)
    
    Retorna análise de sentimento do texto encontrado na imagem
    """
    start_time = time.time()
    
    try:
        logger.info(f"🖼️ Iniciando análise de imagem: {request.filename}")
        
        # Processar imagem e extrair texto
        media_result = await media_service.process_image(
            image_data=request.image_data,
            filename=request.filename,
            extract_text=request.extract_text
        )
        
        # Se texto foi extraído, analisar sentimento
        if media_result.get('extracted_text'):
            sentiment_result = await sentiment_service.analyze_sentiment(
                text=media_result['extracted_text'],
                use_external=False,  # Usar método local para imagens
                options={'source': 'image_ocr'}
            )
        else:
            # Fallback: análise neutra se não conseguiu extrair texto
            logger.warning("⚠️ Nenhum texto extraído da imagem")
            sentiment_result = {
                'label': 'NEUTRAL',
                'score': 0.5,
                'debug': {'reason': 'no_text_extracted', 'fallback': True}
            }
        
        # Calcular tempo decorrido
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"✅ Análise de imagem concluída em {elapsed_ms}ms")
        
        return AnalyzeResponse(
            task=TaskType.IMAGE_SENTIMENT,
            engine="local:ocr+sentiment",
            result=SentimentResult(**sentiment_result),
            media_analysis=MediaAnalysis(**media_result),
            elapsed_ms=elapsed_ms
        )
        
    except ValueError as e:
        logger.error(f"❌ Erro de validação na imagem: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro de validação: {str(e)}"
        )
    except Exception as e:
        logger.error(f"💥 Erro interno na análise de imagem: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro processando imagem: {str(e)}"
        )


@router.post("/analyze/audio", response_model=AnalyzeResponse, summary="Análise de Sentimento em Áudio")
async def analyze_audio(request: AudioAnalyzeRequest) -> AnalyzeResponse:
    """
    Analisa o sentimento do texto extraído de um áudio via Speech-to-Text
    
    - **audio_data**: Áudio codificado em Base64
    - **filename**: Nome do arquivo de áudio
    - **language**: Idioma do áudio (padrão: pt-BR)
    
    Retorna análise de sentimento do texto transcrito do áudio
    """
    start_time = time.time()
    
    try:
        logger.info(f"🎧 Iniciando análise de áudio: {request.filename} ({request.language})")
        
        # Processar áudio e extrair texto
        media_result = await media_service.process_audio(
            audio_data=request.audio_data,
            filename=request.filename,
            language=request.language
        )
        
        # Se texto foi transcrito, analisar sentimento
        if media_result.get('extracted_text'):
            sentiment_result = await sentiment_service.analyze_sentiment(
                text=media_result['extracted_text'],
                use_external=False,  # Usar método local para áudios
                options={'source': 'audio_transcription', 'language': request.language}
            )
        else:
            # Fallback: análise neutra se não conseguiu transcrever
            logger.warning("⚠️ Nenhum texto transcrito do áudio")
            sentiment_result = {
                'label': 'NEUTRAL',
                'score': 0.5,
                'debug': {'reason': 'no_text_transcribed', 'fallback': True}
            }
        
        # Calcular tempo decorrido
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"✅ Análise de áudio concluída em {elapsed_ms}ms")
        
        return AnalyzeResponse(
            task=TaskType.AUDIO_SENTIMENT,
            engine="local:speech-to-text+sentiment",
            result=SentimentResult(**sentiment_result),
            media_analysis=MediaAnalysis(**media_result),
            elapsed_ms=elapsed_ms
        )
        
    except ValueError as e:
        logger.error(f"❌ Erro de validação no áudio: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro de validação: {str(e)}"
        )
    except Exception as e:
        logger.error(f"💥 Erro interno na análise de áudio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro processando áudio: {str(e)}"
        )
