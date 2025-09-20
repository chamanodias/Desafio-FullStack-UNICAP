import os
import httpx
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

logger = logging.getLogger(__name__)

class BaseAIService(ABC):
    """Base class for AI services"""
    
    @abstractmethod
    async def analyze(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        pass

class LocalSentimentService(BaseAIService):
    """Local sentiment analysis using scikit-learn"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load pre-trained model"""
        try:
            # In a real scenario, you would load a pre-trained model
            # For this demo, we'll create a simple model
            logger.info("Initializing local sentiment model...")
            
            # Simple training data for demo
            sample_texts = [
                "I love this product", "This is amazing", "Great job", "Excellent work",
                "I hate this", "This is terrible", "Bad quality", "Worst experience",
                "It's okay", "Not bad", "Average product", "Could be better"
            ]
            sample_labels = [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]  # 1=positive, 0=negative, 2=neutral
            
            self.vectorizer = TfidfVectorizer(max_features=1000)
            X = self.vectorizer.fit_transform(sample_texts)
            
            self.model = LogisticRegression()
            self.model.fit(X, sample_labels)
            
            logger.info("Local sentiment model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing local model: {e}")
            raise
    
    async def analyze(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze sentiment using local model"""
        try:
            if not text:
                raise ValueError("Text input is required")
            
            # Vectorize the input text
            X = self.vectorizer.transform([text])
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Map prediction to label
            label_map = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
            label = label_map[prediction]
            score = float(max(probabilities))
            
            return {
                "label": label,
                "score": score
            }
        except Exception as e:
            logger.error(f"Error in local sentiment analysis: {e}")
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
        self.local_service = LocalSentimentService()
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
                engine = "local:sklearn-logistic (fallback)"
                return result, engine
        else:
            result = await self.local_service.analyze(text, options)
            engine = "local:sklearn-logistic"
            return result, engine
