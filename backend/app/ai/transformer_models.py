import os
import logging
import pickle
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TransformerSentimentAnalyzer:
    """Análise de sentimentos usando modelos Transformer (BERT, RoBERTa)"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo Transformer"""
        try:
            logger.info(f"Inicializando modelo Transformer: {self.model_name}")
            logger.info(f"Usando device: {self.device}")
            
            # Tentar carregar via pipeline (mais simples)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info("Modelo Transformer carregado com sucesso via pipeline")
            
        except Exception as e:
            logger.warning(f"Falha ao carregar {self.model_name} via pipeline: {e}")
            
            # Fallback: tentar carregar manualmente
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("Modelo Transformer carregado manualmente com sucesso")
                
            except Exception as e2:
                logger.error(f"Falha ao carregar modelo Transformer: {e2}")
                self.model = None
                self.tokenizer = None
                self.pipeline = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Análise usando modelo Transformer"""
        if not text or not text.strip():
            raise ValueError("Text input is required")
        
        if not self.pipeline and not self.model:
            raise RuntimeError("Modelo Transformer não foi carregado")
        
        try:
            # Usar pipeline se disponível
            if self.pipeline:
                results = self.pipeline(text)
                
                # Processar resultados do pipeline
                if isinstance(results, list) and len(results) > 0:
                    # Para modelos que retornam múltiplos scores
                    if isinstance(results[0], list):
                        scores = results[0]
                        best_result = max(scores, key=lambda x: x['score'])
                    else:
                        best_result = results[0]
                    
                    # Mapear labels para formato padrão
                    label = self._map_label(best_result['label'])
                    confidence = float(best_result['score'])
                    
                    return {
                        "label": label,
                        "score": round(confidence, 3),
                        "model_used": self.model_name,
                        "raw_output": results
                    }
            
            # Fallback manual se pipeline não disponível
            elif self.model and self.tokenizer:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = float(predictions[0][predicted_class])
                
                # Mapear classe para label
                class_to_label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
                label = class_to_label.get(predicted_class, "NEUTRAL")
                
                return {
                    "label": label,
                    "score": round(confidence, 3),
                    "model_used": self.model_name,
                    "predicted_class": predicted_class
                }
            
            else:
                raise RuntimeError("Nenhum método de inferência disponível")
                
        except Exception as e:
            logger.error(f"Erro na análise Transformer: {e}")
            raise
    
    def _map_label(self, raw_label: str) -> str:
        """Mapeia labels dos modelos para formato padrão"""
        label_mapping = {
            # RoBERTa Twitter
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL", 
            "LABEL_2": "POSITIVE",
            
            # Outros modelos
            "NEGATIVE": "NEGATIVE",
            "POSITIVE": "POSITIVE", 
            "NEUTRAL": "NEUTRAL",
            
            # Variações
            "neg": "NEGATIVE",
            "pos": "POSITIVE",
            "neu": "NEUTRAL"
        }
        
        return label_mapping.get(raw_label.upper(), raw_label.upper())


class BERTimbauAnalyzer:
    """Analisador específico para BERTimbau (BERT português brasileiro)"""
    
    def __init__(self):
        self.model_name = "neuralmind/bert-base-portuguese-cased"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o BERTimbau"""
        try:
            logger.info("Inicializando BERTimbau...")
            
            # Tentar carregar modelo pré-treinado de sentiment
            try:
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("BERTimbau carregado via pipeline")
                
            except Exception as e:
                logger.warning(f"Pipeline BERTimbau não disponível: {e}")
                # Carregar modelo base para fine-tuning posterior
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3,  # NEGATIVE, NEUTRAL, POSITIVE
                    output_attentions=False,
                    output_hidden_states=False,
                )
                self.model.to(self.device)
                self.model.eval()
                logger.info("BERTimbau base carregado para fine-tuning")
                
        except Exception as e:
            logger.error(f"Erro ao carregar BERTimbau: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Análise usando BERTimbau"""
        if not text or not text.strip():
            raise ValueError("Text input is required")
        
        if hasattr(self, 'pipeline') and self.pipeline:
            try:
                results = self.pipeline(text)
                best_result = results[0] if isinstance(results, list) else results
                
                return {
                    "label": best_result['label'].upper(),
                    "score": round(float(best_result['score']), 3),
                    "model_used": "BERTimbau",
                    "specialized_for": "Portuguese Brazilian"
                }
            except Exception as e:
                logger.error(f"Erro no pipeline BERTimbau: {e}")
        
        if self.model and self.tokenizer:
            try:
                # Encoding
                encoded = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = float(predictions[0][predicted_class])
                
                labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
                
                return {
                    "label": labels[predicted_class],
                    "score": round(confidence, 3),
                    "model_used": "BERTimbau (base)",
                    "note": "Modelo base - requer fine-tuning para melhor performance"
                }
                
            except Exception as e:
                logger.error(f"Erro na inferência BERTimbau: {e}")
                raise
        
        raise RuntimeError("BERTimbau não disponível")


class AdvancedMLAnalyzer:
    """Analisador usando algoritmos clássicos de ML mais avançados"""
    
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.is_trained = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa modelos ML clássicos"""
        try:
            # Múltiplos modelos para ensemble
            self.models = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            }
            
            # Vetorizador TF-IDF avançado
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
                stop_words=None,  # Manter stop words para português
                min_df=2,
                max_df=0.8,
                lowercase=True,
                analyzer='word'
            )
            
            logger.info("Modelos ML clássicos inicializados")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelos ML: {e}")
    
    def train_with_sample_data(self):
        """Treina com dados de exemplo expandidos"""
        try:
            # Dataset expandido para treinamento
            sample_texts = [
                # Muito positivos
                "Amo muito este produto, é fantástico!", "Experiência incrível, recomendo demais!",
                "Perfeito em todos os aspectos, superou expectativas", "Maravilhoso, melhor compra da vida",
                "Excelente qualidade, vale cada centavo", "Sensacional, não tenho palavras",
                
                # Positivos
                "Muito bom, gostei bastante", "Produto de boa qualidade", "Vale a pena comprar",
                "Satisfeito com a compra", "Recomendo, é legal", "Bacana, funcionou bem",
                "Ótimo custo benefício", "Chegou rápido, gostei", "Produto confiável",
                
                # Neutros
                "É um produto normal", "Nem bom nem ruim", "Razoável, dentro do esperado",
                "OK, funciona", "Regular, nada demais", "Comum, sem surpresas",
                "Padrão do mercado", "Adequado para o preço", "Aceitável",
                
                # Negativos
                "Não gostei, produto ruim", "Qualidade deixa a desejar", "Decepcionante",
                "Não recomendo, problemático", "Inferior ao esperado", "Não vale o preço",
                "Produto fraco, insatisfeito", "Muitos defeitos", "Não funcionou bem",
                
                # Muito negativos
                "Péssimo, odiei completamente!", "Terrível experiência, dinheiro jogado fora",
                "Horrível, pior compra da vida", "Detesto este produto, não funciona",
                "Desastre total, não comprem", "Lixo completo, muito ruim"
            ]
            
            # Labels correspondentes
            sample_labels = [
                # Muito positivos (6)
                2, 2, 2, 2, 2, 2,
                # Positivos (9)  
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                # Neutros (9)
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                # Negativos (9)
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                # Muito negativos (6)
                0, 0, 0, 0, 0, 0
            ]
            
            # Vectorização
            X = self.vectorizer.fit_transform(sample_texts)
            y = np.array(sample_labels)
            
            # Treinar todos os modelos
            for name, model in self.models.items():
                model.fit(X, y)
                logger.info(f"Modelo {name} treinado")
            
            self.is_trained = True
            logger.info("Treinamento concluído com dados de exemplo")
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            self.is_trained = False
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Análise usando ensemble de modelos ML"""
        if not text or not text.strip():
            raise ValueError("Text input is required")
        
        if not self.is_trained:
            self.train_with_sample_data()
        
        if not self.is_trained:
            raise RuntimeError("Modelos ML não foram treinados")
        
        try:
            # Vetorizar texto
            X = self.vectorizer.transform([text])
            
            # Predições de todos os modelos
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred = model.predict(X)[0]
                pred_proba = model.predict_proba(X)[0]
                
                predictions[name] = pred
                probabilities[name] = pred_proba
            
            # Ensemble por voto majoritário
            votes = list(predictions.values())
            ensemble_prediction = max(set(votes), key=votes.count)
            
            # Confidence médio
            avg_probabilities = np.mean(list(probabilities.values()), axis=0)
            confidence = float(avg_probabilities[ensemble_prediction])
            
            # Mapear para labels
            labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            
            return {
                "label": labels[ensemble_prediction],
                "score": round(confidence, 3),
                "ensemble_details": {
                    "individual_predictions": {name: labels[pred] for name, pred in predictions.items()},
                    "confidence_scores": {name: float(max(prob)) for name, prob in probabilities.items()}
                },
                "model_used": "ML Ensemble (Logistic + RandomForest)",
                "features_used": self.vectorizer.get_feature_names_out().shape[0]
            }
            
        except Exception as e:
            logger.error(f"Erro na análise ML: {e}")
            raise


class ModelEnsemble:
    """Ensemble de múltiplos modelos para análise mais robusta"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa todos os modelos disponíveis"""
        logger.info("Inicializando ensemble de modelos...")
        
        # Modelo Transformer padrão
        try:
            self.models['roberta'] = TransformerSentimentAnalyzer()
            self.weights['roberta'] = 0.4
            logger.info("RoBERTa carregado no ensemble")
        except Exception as e:
            logger.warning(f"RoBERTa não disponível: {e}")
        
        # BERTimbau para português
        try:
            self.models['bertimbau'] = BERTimbauAnalyzer()
            self.weights['bertimbau'] = 0.4  # Peso alto por ser específico para PT-BR
            logger.info("BERTimbau carregado no ensemble")
        except Exception as e:
            logger.warning(f"BERTimbau não disponível: {e}")
        
        # Modelos ML clássicos
        try:
            self.models['ml_ensemble'] = AdvancedMLAnalyzer()
            self.weights['ml_ensemble'] = 0.2
            logger.info("ML Ensemble carregado")
        except Exception as e:
            logger.warning(f"ML Ensemble não disponível: {e}")
        
        # Normalizar pesos
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            logger.info(f"Ensemble inicializado com {len(self.models)} modelos")
        else:
            logger.error("Nenhum modelo foi carregado no ensemble")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Análise usando ensemble de modelos"""
        if not text or not text.strip():
            raise ValueError("Text input is required")
        
        if not self.models:
            raise RuntimeError("Nenhum modelo disponível no ensemble")
        
        results = {}
        scores = {}
        
        # Executar todos os modelos
        for name, model in self.models.items():
            try:
                result = model.analyze(text)
                results[name] = result
                
                # Converter label para score numérico
                label_to_score = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
                numeric_score = label_to_score[result['label']]
                
                # Ponderar pelo confidence e peso do modelo
                weighted_score = numeric_score * result['score'] * self.weights[name]
                scores[name] = weighted_score
                
                logger.debug(f"Modelo {name}: {result['label']} (confidence: {result['score']})")
                
            except Exception as e:
                logger.error(f"Erro no modelo {name}: {e}")
                continue
        
        if not results:
            raise RuntimeError("Todos os modelos falharam")
        
        # Calcular resultado final do ensemble
        final_score = sum(scores.values())
        
        # Determinar label final
        if final_score >= 0.15:
            final_label = "POSITIVE"
        elif final_score <= -0.15:
            final_label = "NEGATIVE"
        else:
            final_label = "NEUTRAL"
        
        # Calcular confidence final (média ponderada)
        weighted_confidences = [
            results[name]['score'] * self.weights[name] 
            for name in results.keys()
        ]
        final_confidence = sum(weighted_confidences)
        
        return {
            "label": final_label,
            "score": round(final_confidence, 3),
            "ensemble_score": round(final_score, 3),
            "individual_results": results,
            "model_weights": self.weights,
            "models_used": list(results.keys()),
            "engine": "Advanced AI Ensemble"
        }
