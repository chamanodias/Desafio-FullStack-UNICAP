"""
Funções utilitárias e helpers

Contém funções reutilizáveis usadas em toda a aplicação
"""
import time
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import json
import base64


def generate_request_id() -> str:
    """Gera um ID único para rastreamento de requisições"""
    return str(uuid.uuid4())


def generate_hash(text: str) -> str:
    """Gera hash MD5 de um texto (para cache ou identificação)"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Formata timestamp para string ISO"""
    if timestamp is None:
        timestamp = datetime.utcnow()
    return timestamp.isoformat()


def format_elapsed_time(start_time: float) -> int:
    """Calcula tempo decorrido em millisegundos"""
    return int((time.time() - start_time) * 1000)


def sanitize_text(text: str, max_length: int = 2000) -> str:
    """
    Sanitiza e limpa texto de entrada
    
    Args:
        text: Texto para sanitizar
        max_length: Tamanho máximo permitido
        
    Returns:
        Texto limpo e truncado se necessário
    """
    if not text:
        return ""
    
    # Remover caracteres de controle
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Normalizar espaços em branco
    text = ' '.join(text.split())
    
    # Truncar se muito longo
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return text.strip()


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Valida se extensão do arquivo é permitida
    
    Args:
        filename: Nome do arquivo
        allowed_extensions: Lista de extensões permitidas
        
    Returns:
        True se extensão é válida
    """
    if not filename:
        return False
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext.lower()) for ext in allowed_extensions)


def get_file_size_mb(data: Union[str, bytes]) -> float:
    """
    Calcula tamanho do arquivo em MB
    
    Args:
        data: Dados do arquivo (base64 ou bytes)
        
    Returns:
        Tamanho em MB
    """
    if isinstance(data, str):
        # Se for base64, decodificar para obter tamanho real
        try:
            if data.startswith('data:'):
                data = data.split(',')[1]
            decoded = base64.b64decode(data)
            size_bytes = len(decoded)
        except:
            size_bytes = len(data.encode('utf-8'))
    else:
        size_bytes = len(data)
    
    return size_bytes / (1024 * 1024)


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Trunca string mantendo legibilidade
    
    Args:
        text: Texto para truncar
        max_length: Tamanho máximo
        suffix: Sufixo para indicar truncamento
        
    Returns:
        Texto truncado
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length-len(suffix)] + suffix


def format_confidence_percentage(score: float) -> str:
    """Formata pontuação de confiança como porcentagem"""
    return f"{score * 100:.1f}%"


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Faz merge profundo de dois dicionários
    
    Args:
        dict1: Dicionário base
        dict2: Dicionário a ser mesclado
        
    Returns:
        Dicionário mesclado
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Parse JSON seguro que não levanta exceção
    
    Args:
        json_string: String JSON
        default: Valor padrão se parse falhar
        
    Returns:
        Objeto parseado ou valor padrão
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Serialização JSON segura
    
    Args:
        obj: Objeto para serializar
        default: String padrão se serialização falhar
        
    Returns:
        String JSON ou valor padrão
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return default


def classify_sentiment_by_score(score: float, threshold_positive: float = 0.6, threshold_negative: float = 0.4) -> str:
    """
    Classifica sentimento baseado na pontuação
    
    Args:
        score: Pontuação normalizada (-1 a 1 ou 0 a 1)
        threshold_positive: Limiar para positivo
        threshold_negative: Limiar para negativo
        
    Returns:
        Label do sentimento (POSITIVE, NEGATIVE, NEUTRAL)
    """
    if score >= threshold_positive:
        return "POSITIVE"
    elif score <= threshold_negative:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def normalize_score(raw_score: float, min_val: float = -4.0, max_val: float = 4.0) -> float:
    """
    Normaliza pontuação para escala 0-1
    
    Args:
        raw_score: Pontuação bruta
        min_val: Valor mínimo esperado
        max_val: Valor máximo esperado
        
    Returns:
        Pontuação normalizada (0-1)
    """
    # Clampar valor dentro dos limites
    clamped = max(min_val, min(max_val, raw_score))
    
    # Normalizar para 0-1
    normalized = (clamped - min_val) / (max_val - min_val)
    
    return round(normalized, 3)


def get_system_info() -> Dict[str, Any]:
    """Retorna informações básicas do sistema"""
    import platform
    import sys
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }


def format_bytes(bytes_count: int) -> str:
    """Formata bytes em formato legível"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"


class Timer:
    """Context manager para medir tempo de execução"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed_ms(self) -> int:
        """Tempo decorrido em millisegundos"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0
    
    @property
    def elapsed_seconds(self) -> float:
        """Tempo decorrido em segundos"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
