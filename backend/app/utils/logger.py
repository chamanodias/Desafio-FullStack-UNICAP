"""
Sistema de Logging Estruturado

Configura logging profissional para toda a aplicação
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..config import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_file_logging: bool = True
) -> logging.Logger:
    """
    Configura o sistema de logging da aplicação
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Arquivo de log (opcional)
        enable_file_logging: Se deve salvar logs em arquivo
        
    Returns:
        Logger configurado
    """
    # Usar configurações padrão se não especificadas
    log_level = log_level or config.LOG_LEVEL
    log_file = log_file or config.LOG_FILE
    
    # Converter string para nível
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Formato detalhado para logs
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formato simples para console em produção
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Configurar logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Limpar handlers existentes
    root_logger.handlers.clear()
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Usar formato simples em produção, detalhado em debug
    if config.DEBUG:
        console_handler.setFormatter(detailed_formatter)
    else:
        console_handler.setFormatter(simple_formatter)
    
    root_logger.addHandler(console_handler)
    
    # Handler para arquivo (se habilitado)
    if enable_file_logging and log_file:
        try:
            # Criar diretório se não existir
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handler rotativo (máximo 10MB, manter 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            # Se falhar, continuar apenas com console
            console_handler.formatter.format(
                logging.LogRecord(
                    name='logger_setup',
                    level=logging.WARNING,
                    pathname='',
                    lineno=0,
                    msg=f"Não foi possível configurar log em arquivo: {e}",
                    args=(),
                    exc_info=None
                )
            )
    
    # Configurar loggers de bibliotecas externas
    _configure_external_loggers()
    
    # Obter logger principal da aplicação
    app_logger = logging.getLogger('sentiment_analyzer')
    
    # Log de inicialização
    app_logger.info(f"🚀 Logging configurado - Nível: {log_level}, Arquivo: {log_file if enable_file_logging else 'Desabilitado'}")
    
    return app_logger


def _configure_external_loggers():
    """Configura níveis de log para bibliotecas externas"""
    external_loggers = {
        'uvicorn': logging.INFO,
        'uvicorn.access': logging.WARNING,
        'httpx': logging.WARNING,
        'PIL': logging.WARNING,
        'matplotlib': logging.WARNING,
        'asyncio': logging.WARNING
    }
    
    for logger_name, level in external_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger nomeado
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(f'sentiment_analyzer.{name}')


class LoggerMixin:
    """Mixin para adicionar logger a classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Retorna logger específico da classe"""
        return get_logger(self.__class__.__name__)


# Configuração automática quando o módulo é importado
if not logging.getLogger().handlers:
    setup_logging()
