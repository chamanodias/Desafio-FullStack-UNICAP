# AI Module - Sistema Avançado de Análise de Sentimentos

__version__ = "2.0.0"

# Imports condicionais para evitar erros se dependências não estiverem instaladas
try:
    from .advanced_sentiment import AdvancedSentimentAnalyzer
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

try:
    from .transformer_models import (
        ModelEnsemble, 
        TransformerSentimentAnalyzer, 
        BERTimbauAnalyzer,
        AdvancedMLAnalyzer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from .dataset_manager import BrazilianSentimentDatasets, DatasetAnalyzer
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from .metrics_system import ModelPerformanceTracker, BenchmarkSuite, RealTimeMonitor
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Fallback imports
try:
    from .simple_ai_service import SimpleSentimentService
    BASIC_AVAILABLE = True
except ImportError:
    BASIC_AVAILABLE = False

# Export principais
__all__ = [
    'AdvancedSentimentAnalyzer',
    'ModelEnsemble', 
    'TransformerSentimentAnalyzer',
    'BERTimbauAnalyzer',
    'BrazilianSentimentDatasets',
    'ModelPerformanceTracker',
    'BenchmarkSuite',
    'SimpleSentimentService'
]

# Status das funcionalidades
FEATURES = {
    'advanced_sentiment': ADVANCED_AVAILABLE,
    'transformer_models': TRANSFORMERS_AVAILABLE,
    'brazilian_datasets': DATASETS_AVAILABLE,
    'performance_metrics': METRICS_AVAILABLE,
    'basic_fallback': BASIC_AVAILABLE
}
