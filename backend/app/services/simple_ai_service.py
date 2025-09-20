import os
import httpx
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

class BaseAIService(ABC):
    """Base class for AI services"""
    
    @abstractmethod
    async def analyze(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        pass

class SimpleSentimentService(BaseAIService):
    """Análise de sentimento simples baseada em palavras-chave"""
    
    def __init__(self):
        # Palavras-chave para análise
        self.positive_words = [
            'amo', 'love', 'excelente', 'excellent', 'ótimo', 'great', 'bom', 'good', 
            'feliz', 'happy', 'maravilhoso', 'wonderful', 'incrível', 'incredible',
            'perfeito', 'perfect', 'fantástico', 'fantastic', 'adorei', 'loved',
            'recomendo', 'recommend', 'melhor', 'best', 'incrível', 'amazing'
        ]
        
        self.negative_words = [
            'odeio', 'hate', 'terrível', 'terrible', 'péssimo', 'awful', 'ruim', 'bad',
            'triste', 'sad', 'horrível', 'horrible', 'odioso', 'hateful', 'pior', 'worst',
            'detesto', 'detest', 'não gosto', "don't like", 'não recomendo', "don't recommend"
        ]
    
    async def analyze(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze sentiment using keyword-based approach"""
        try:
            if not text:
                raise ValueError("Text input is required")
            
            # Converter para minúsculas para análise
            text_lower = text.lower()
            
            # Contar palavras positivas e negativas
            positive_count = 0
            negative_count = 0
            
            for word in self.positive_words:
                if word in text_lower:
                    positive_count += text_lower.count(word)
            
            for word in self.negative_words:
                if word in text_lower:
                    negative_count += text_lower.count(word)
            
            # Determinar sentimento baseado na contagem
            if positive_count > negative_count:
                label = "POSITIVE"
                confidence = min(0.95, 0.6 + (positive_count * 0.1))
            elif negative_count > positive_count:
                label = "NEGATIVE"
                confidence = min(0.95, 0.6 + (negative_count * 0.1))
            else:
                # Análise adicional para textos neutros
                if any(word in text_lower for word in ['ok', 'normal', 'regular', 'comum']):
                    label = "NEUTRAL"
                    confidence = 0.75
                elif len(text.strip()) < 10:  # Textos muito curtos
                    label = "NEUTRAL"
                    confidence = 0.65
                else:
                    label = "NEUTRAL"
                    confidence = 0.70
            
            return {
                "label": label,
                "score": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in simple sentiment analysis: {e}")
            raise

class ExternalSentimentService(BaseAIService):
    """External sentiment analysis using Hugging Face API"""
    
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        if not self.api_key:
            logger.warning("HUGGINGFACE_API_KEY not found in environment variables")
    
    async def analyze(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze sentiment using Hugging Face API"""
        try:
            if not text:
                raise ValueError("Text input is required")
            
            if not self.api_key:
                raise ValueError("HUGGINGFACE_API_KEY is required for external API")
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"inputs": text}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Hugging Face returns list of predictions
                    if isinstance(result, list) and len(result) > 0:
                        best_result = max(result[0], key=lambda x: x["score"])
                        
                        # Map HF labels to our format
                        label_map = {
                            "LABEL_0": "NEGATIVE",
                            "LABEL_1": "NEUTRAL", 
                            "LABEL_2": "POSITIVE"
                        }
                        
                        label = label_map.get(best_result["label"], best_result["label"])
                        score = float(best_result["score"])
                        
                        return {
                            "label": label,
                            "score": score
                        }
                    else:
                        raise ValueError("Unexpected API response format")
                
                elif response.status_code == 503:
                    raise ValueError("Model is currently loading, please try again later")
                else:
                    raise ValueError(f"API request failed: {response.status_code} - {response.text}")
                    
        except httpx.TimeoutException:
            logger.error("Timeout when calling Hugging Face API")
            raise ValueError("Request timeout - API took too long to respond")
        except Exception as e:
            logger.error(f"Error in external sentiment analysis: {e}")
            raise

class SentimentAnalyzer:
    """Main sentiment analyzer that can use local or external services"""
    
    def __init__(self):
        self.local_service = SimpleSentimentService()
        self.external_service = ExternalSentimentService()
    
    async def analyze_sentiment(self, text: str, use_external: bool = False, options: Dict[str, Any] = None) -> tuple[Dict[str, Any], str]:
        """
        Analyze sentiment using specified service
        Returns: (result_dict, engine_name)
        """
        if use_external:
            try:
                result = await self.external_service.analyze(text, options)
                engine = "external:huggingface-roberta"
                return result, engine
            except Exception as e:
                logger.warning(f"External service failed, falling back to local: {e}")
                # Fallback to local
                result = await self.local_service.analyze(text, options)
                engine = "local:keyword-based (fallback)"
                return result, engine
        else:
            result = await self.local_service.analyze(text, options)
            engine = "local:keyword-based"
            return result, engine
