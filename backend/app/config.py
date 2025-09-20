import os
from dotenv import load_dotenv
from typing import List

# Carregar variáveis de ambiente
load_dotenv()

class Config:
    """Configurações centralizadas do sistema"""
    
    # Informações do projeto
    NOME_PROJETO: str = "Sistema de Análise de Sentimentos com IA"
    VERSAO: str = "2.0.0"
    DESCRICAO: str = "Sistema completo para análise de sentimentos em texto, imagens e áudios"
    API_V1_STR: str = "/api/v1"
    
    # Configurações do servidor
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API Keys externas
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # URLs de APIs externas
    HUGGINGFACE_API_URL: str = "https://api-inference.huggingface.com/models"
    
    # Configurações de modelos IA
    MODELO_LOCAL_SENTIMENTOS: str = "pysentimento/roberta-base-pt"
    MODELO_EXTERNO_SENTIMENTOS: str = "pysentimento/roberta-base-pt"
    MODELO_OCR: str = "microsoft/trocr-base-printed"
    MODELO_AUDIO: str = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"
    
    # CORS - Origens permitidas
    ORIGINS_ALLOWED: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    # Configurações de upload
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".aac"]
    
    # Timeouts e limites
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 30))
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    
    # Cache e performance
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))  # 1 hora
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "sentiment_analyzer.log")
    
    # Configurações de segurança
    SECRET_KEY: str = os.getenv("SECRET_KEY", "sua-chave-secreta-aqui-mude-em-producao")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # Database (para futuras implementações)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sentiment_analyzer.db")
    
    @property
    def cors_origins(self) -> List[str]:
        """Retorna lista de origens permitidas para CORS"""
        custom_origins = os.getenv("CORS_ORIGINS", "").split(",")
        if custom_origins and custom_origins[0]:  # Se há origens customizadas
            return [origin.strip() for origin in custom_origins]
        return self.ORIGINS_ALLOWED
    
    def __repr__(self):
        return f"<Config {self.NOME_PROJETO} v{self.VERSAO}>"

# Instância global da configuração
config = Config()

# Validações básicas
if not config.SECRET_KEY or config.SECRET_KEY == "sua-chave-secreta-aqui-mude-em-producao":
    print("⚠️  AVISO: Configure a SECRET_KEY para produção!")

if config.DEBUG:
    print(f"🔧 Debug Mode: {config.DEBUG}")
    print(f"🌐 CORS Origins: {config.cors_origins}")
