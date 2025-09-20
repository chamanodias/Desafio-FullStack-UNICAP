"""
Sistema de Análise de Sentimentos com IA - Servidor Principal

Aplicação FastAPI moderna para análise de sentimentos em texto, imagens e áudios
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import time
import logging

# Imports da aplicação
from app.config import config
from app.routes import main_router
from app.utils.logger import setup_logging
from app.utils.helpers import format_elapsed_time

# Configurar logging
logger = setup_logging()

# Criar aplicação FastAPI
app = FastAPI(
    title=config.NOME_PROJETO,
    description=config.DESCRICAO,
    version=config.VERSAO,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ===== MIDDLEWARES =====

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted Host Middleware (segurança)
if not config.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.local"]
    )


# ===== MIDDLEWARE PERSONALIZADO =====

@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Middleware para medir tempo de resposta e log de requisições"""
    start_time = time.time()
    
    # Log da requisição
    logger.info(f"📨 {request.method} {request.url.path} - Cliente: {request.client.host if request.client else 'unknown'}")
    
    # Processar requisição
    response = await call_next(request)
    
    # Calcular tempo
    elapsed_ms = format_elapsed_time(start_time)
    
    # Adicionar header com tempo de resposta
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    
    # Log da resposta
    status_emoji = "✅" if response.status_code < 400 else "❌"
    logger.info(f"{status_emoji} {request.method} {request.url.path} - {response.status_code} - {elapsed_ms}ms")
    
    return response


# ===== EXCEPTION HANDLERS =====

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handler para 404 Not Found"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Endpoint {request.url.path} não encontrado",
            "available_endpoints": [
                "/docs - Documentação da API",
                "/api/v1/healthz - Health check",
                "/api/v1/capabilities - Capacidades do sistema",
                "/api/v1/analyze - Análise de texto",
                "/api/v1/analyze/image - Análise de imagem",
                "/api/v1/analyze/audio - Análise de áudio"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handler para erros internos"""
    logger.error(f"💥 Erro interno no endpoint {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Erro interno do servidor. Verifique os logs para mais detalhes.",
            "debug": str(exc) if config.DEBUG else "Detalhes ocultados em produção"
        }
    )


# ===== EVENTOS =====

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização do servidor"""
    logger.info("🚀 Iniciando Sistema de Análise de Sentimentos")
    logger.info(f"📊 Versão: {config.VERSAO}")
    logger.info(f"🔧 Debug Mode: {config.DEBUG}")
    logger.info(f"🌐 CORS Origins: {len(config.cors_origins)} origens configuradas")
    
    # Log das capacidades disponíveis
    try:
        from app.services.media_service import MediaService
        media_service = MediaService()
        capabilities = media_service.get_capabilities()
        
        logger.info("🎞️ Capacidades de mídia:")
        logger.info(f"   - Processamento de imagens: {capabilities['image_processing']}")
        logger.info(f"   - Processamento de áudios: {capabilities['audio_processing']}")
        logger.info(f"   - OCR disponível: {capabilities['ocr_available']}")
        logger.info(f"   - Speech recognition: {capabilities['speech_recognition']}")
        
    except ImportError:
        logger.warning("⚠️ Serviços de mídia não disponíveis")
    
    logger.info(f"🎯 Servidor pronto em http://{config.HOST}:{config.PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento executado no encerramento do servidor"""
    logger.info("🛑 Encerrando Sistema de Análise de Sentimentos")
    logger.info("👋 Servidor encerrado com sucesso")


# ===== ROTAS =====

# Incluir todas as rotas
app.include_router(main_router)


# Rota raiz informativa
@app.get("/", tags=["Info"])
async def root():
    """Endpoint raiz com informações básicas do sistema"""
    return {
        "project": config.NOME_PROJETO,
        "description": config.DESCRICAO,
        "version": config.VERSAO,
        "status": "online",
        "docs": "/docs",
        "health_check": "/api/v1/healthz",
        "capabilities": "/api/v1/capabilities",
        "endpoints": {
            "analyze_text": "POST /api/v1/analyze",
            "analyze_image": "POST /api/v1/analyze/image", 
            "analyze_audio": "POST /api/v1/analyze/audio"
        },
        "features": [
            "📝 Análise de sentimentos em texto",
            "🖼️ Extração de texto de imagens (OCR)",
            "🎧 Conversão de fala para texto",
            "🤖 Análise local e via APIs externas",
            "⚡ Análise em tempo real",
            "📊 Interface moderna e responsiva"
        ]
    }


# ===== OPENAPI CUSTOMIZATION =====

def custom_openapi():
    """Customiza a documentação OpenAPI"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=config.NOME_PROJETO,
        version=config.VERSAO,
        description=f"""
        {config.DESCRICAO}
        
        ## 🚀 Funcionalidades
        
        - **Análise de Texto**: Determina sentimento (positivo/negativo/neutro) de textos em português
        - **Processamento de Imagens**: Extrai texto de imagens via OCR e analisa sentimento
        - **Processamento de Áudio**: Converte fala em texto e analisa sentimento
        - **Análise Local**: Algoritmo próprio otimizado para português
        - **APIs Externas**: Integração com Hugging Face para análise avançada
        
        ## 📊 Formato das Respostas
        
        Todas as análises retornam:
        - `label`: Classificação do sentimento (POSITIVE/NEGATIVE/NEUTRAL)  
        - `score`: Pontuação de confiança (0.0 a 1.0)
        - `debug`: Informações técnicas da análise
        
        ## 🔗 Links Úteis
        
        - [Documentação Interativa](/docs)
        - [Health Check](/api/v1/healthz)
        - [Capacidades do Sistema](/api/v1/capabilities)
        """,
        routes=app.routes,
    )
    
    # Adicionar informações customizadas
    openapi_schema["info"]["contact"] = {
        "name": "Sistema de Análise de Sentimentos",
        "url": "https://github.com/lucas/sentiment-analyzer"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Tags para organização
    openapi_schema["tags"] = [
        {
            "name": "Analysis",
            "description": "Endpoints para análise de sentimentos"
        },
        {
            "name": "System", 
            "description": "Endpoints de sistema e monitoramento"
        },
        {
            "name": "Info",
            "description": "Informações gerais da API"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ===== EXECUÇÃO =====

if __name__ == "__main__":
    import uvicorn
    
    # Configuração do servidor
    server_config = {
        "host": config.HOST,
        "port": config.PORT,
        "reload": config.DEBUG,
        "log_level": config.LOG_LEVEL.lower(),
        "access_log": config.DEBUG
    }
    
    logger.info(f"🌟 Iniciando servidor com configuração: {server_config}")
    
    # Executar servidor
    uvicorn.run(
        "new_main:app" if config.DEBUG else app,
        **server_config
    )
