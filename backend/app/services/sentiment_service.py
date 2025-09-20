"""
Serviço de Análise de Sentimentos

Contém toda a lógica de análise de sentimentos, incluindo análise local
e integração com APIs externas (Hugging Face)
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
import requests
import json

from ..config import config

logger = logging.getLogger(__name__)


class SentimentService:
    """Serviço principal para análise de sentimentos"""
    
    def __init__(self):
        self.local_analyzer = LocalSentimentAnalyzer()
        self._external_api_available = bool(config.HUGGINGFACE_API_KEY)
        
        if self._external_api_available:
            logger.info("🤗 API externa Hugging Face configurada")
        else:
            logger.info("🏠 Usando apenas análise local")
    
    async def analyze_sentiment(
        self, 
        text: str, 
        use_external: bool = False, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analisa o sentimento de um texto
        
        Args:
            text: Texto para análise
            use_external: Se deve tentar usar API externa
            options: Opções adicionais (source, language, etc.)
            
        Returns:
            Dict com label, score e informações de debug
        """
        if not text or not text.strip():
            raise ValueError("Texto não pode estar vazio")
        
        text = text.strip()
        options = options or {}
        
        # Decidir qual método usar
        if use_external and self._external_api_available:
            try:
                result = await self._analyze_external(text, options)
                logger.info(f"✅ Análise externa concluída: {result['label']} ({result['score']:.3f})")
                return result
            except Exception as e:
                logger.warning(f"⚠️ Falha na API externa, usando análise local: {e}")
                # Fallback para análise local
        
        # Usar análise local
        result = self.local_analyzer.analyze(text, options)
        logger.info(f"✅ Análise local concluída: {result['label']} ({result['score']:.3f})")
        return result
    
    async def _analyze_external(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa usando API externa (Hugging Face)"""
        if not config.HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY não configurada")
        
        # Modelo otimizado para português
        model = "pysentimiento/robertuito-sentiment-analysis"
        url = f"{config.HUGGINGFACE_API_URL}/{model}"
        
        headers = {
            "Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": text}
        
        # Fazer requisição (bloqueante, mas em thread separada)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.post(url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT)
        )
        
        if response.status_code != 200:
            raise Exception(f"API externa falhou: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Processar resultado da API
        if isinstance(result, list) and len(result) > 0:
            predictions = result[0]
            
            # Encontrar a melhor predição
            best_pred = max(predictions, key=lambda x: x['score'])
            
            # Mapear labels para nosso padrão
            label_mapping = {
                'POSITIVE': 'POSITIVE',
                'POS': 'POSITIVE',
                'NEGATIVE': 'NEGATIVE', 
                'NEG': 'NEGATIVE',
                'NEUTRAL': 'NEUTRAL',
                'NEU': 'NEUTRAL'
            }
            
            mapped_label = label_mapping.get(best_pred['label'].upper(), 'NEUTRAL')
            
            return {
                'label': mapped_label,
                'score': float(best_pred['score']),
                'debug': {
                    'method': 'external_api',
                    'model': model,
                    'raw_predictions': predictions,
                    'source': options.get('source', 'text')
                }
            }
        
        raise ValueError("Formato de resposta inválido da API externa")


class LocalSentimentAnalyzer:
    """Analisador de sentimento local baseado em regras e palavras-chave"""
    
    def __init__(self):
        self._load_sentiment_data()
    
    def _load_sentiment_data(self):
        """Carrega dados de palavras e frases para análise"""
        
        # Palavras MUITO positivas - Alegria intensa (peso 4)
        self.very_positive = [
            'amo', 'adoro', 'love', 'excelente', 'excellent', 'fantástico', 'fantastic',
            'maravilhoso', 'wonderful', 'incrível', 'incredible', 'perfeito', 'perfect',
            'sensacional', 'espetacular', 'extraordinário', 'magnífico', 'divino',
            'alegria', 'felicidade', 'euforia', 'êxtase', 'radiante', 'encantado',
            'deslumbrado', 'fascinado', 'gratidão', 'ternura', 'sorrir', 'sorrindo',
            'nunca imaginei', 'tomou conta', 'mais do que tudo'
        ]
        
        # Palavras positivas - Sentimentos bons (peso 2.5)
        self.positive_words = [
            'ótimo', 'great', 'bom', 'good', 'feliz', 'happy', 'adorei', 'loved',
            'recomendo', 'recommend', 'melhor', 'best', 'amazing', 'legal', 'bacana',
            'gostei', 'like', 'aprovado', 'sucesso', 'show', 'top', 'massa', 'demais',
            'esperança', 'otimismo', 'confiança', 'satisfeito', 'contente', 'orgulhoso',
            'animado', 'entusiasmado', 'inspirado', 'motivado', 'tudo vai dar certo',
            'amor da minha vida', 'especial', 'presente', 'surpresa', 'expectativa'
        ]
        
        # Palavras levemente positivas (peso 1)
        self.light_positive = [
            'ok', 'okay', 'aceitável', 'razoável', 'decent', 'not bad', 'pode ser',
            'interessante', 'útil', 'válido', 'funciona', 'serve', 'normal', 'regular',
            'ansioso', 'curioso', 'esperando', 'aguardando'
        ]
        
        # Palavras MUITO negativas - Sentimentos intensos ruins (peso -4)
        self.very_negative = [
            'odeio', 'detesto', 'hate', 'terrível', 'terrible', 'péssimo', 'awful',
            'horrível', 'horrible', 'odioso', 'hateful', 'nojento', 'lixo', 'merda',
            'raiva', 'ódio', 'fúria', 'irritado', 'extremamente irritado', 'pavor',
            'pânico', 'terror', 'desespero', 'devastado', 'destruído', 'arrasado',
            'não conseguiu esconder', 'profunda tristeza', 'coração pesado'
        ]
        
        # Palavras negativas - Tristeza e problemas (peso -2.5)
        self.negative_words = [
            'ruim', 'bad', 'triste', 'sad', 'pior', 'worst', 'não gosto', "don't like",
            'não recomendo', "don't recommend", 'chato', 'boring', 'desapontado', 'disappointed',
            'tristeza', 'melancolia', 'saudade', 'lágrimas', 'choro', 'chorando',
            'medo', 'medo do escuro', 'desconhecido', 'sozinhas', 'assustado',
            'preocupado', 'angustiado', 'estressado', 'tenso', 'nervoso',
            'decepcionado', 'frustrado', 'magoado', 'machucado', 'ferido'
        ]
        
        # Palavras levemente negativas (peso -1)
        self.light_negative = [
            'estranho', 'weird', 'confuso', 'confusing', 'difícil', 'hard', 'complicado',
            'problema', 'issue', 'erro', 'error', 'falha', 'bug', 'cansado', 'fatiga',
            'ansiedade', 'preocupação', 'dúvida', 'incerteza', 'inseguro', 'receoso'
        ]
        
        # Intensificadores - aumentam o peso das palavras
        self.intensifiers = [
            'muito', 'super', 'extremely', 'really', 'totally', 'absolutely',
            'completamente', 'demais', 'bastante', 'quite', 'bem', 'mega',
            'extremamente', 'profundamente', 'incrivelmente', 'imensamente',
            'profunda', 'intensa', 'grande', 'enorme', 'tanto', 'tanta'
        ]
        
        # Negadores - invertem o sentimento
        self.negators = [
            'não', 'not', 'nunca', 'never', 'jamais', 'nenhum', 'nenhuma',
            'sem', 'without', 'impossível', 'impossible', 'nem', 'nada de',
            'deixou de', 'parou de', 'falhou em'
        ]
        
        # Frases completas positivas (peso 3)
        self.positive_phrases = [
            'estou me sentindo muito feliz', 'alegria tomou conta', 'nunca imaginei que pudesse sorrir',
            'tenho esperança de que tudo', 'amor da minha vida', 'amo minha família',
            'sinto ternura em tudo', 'profunda gratidão', 'tudo vai dar certo',
            'mal posso esperar', 'não esperava receber', 'presente tão especial'
        ]
        
        # Frases completas negativas (peso -3)
        self.negative_phrases = [
            'senti uma tristeza profunda', 'coração está pesado', 'não consigo segurar as lágrimas',
            'com muita raiva', 'extremamente irritado', 'não conseguiu esconder a raiva',
            'tenho medo do escuro', 'senti um pavor', 'estão com medo', 'ansiedade constante',
            'decepcionado com', 'tomado pela melancolia'
        ]
    
    def analyze(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analisa o sentimento usando método local
        
        Args:
            text: Texto para análise
            options: Opções adicionais
            
        Returns:
            Dict com label, score e debug info
        """
        options = options or {}
        
        if not text:
            raise ValueError("Texto é obrigatório")
        
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Para textos muito curtos, ser mais conservador
        is_short_text = len(words) <= 2
        
        score = 0
        word_count = 0
        phrase_matches = []
        
        # ETAPA 1: Verificar frases completas primeiro (prioridade máxima)
        for phrase in self.positive_phrases:
            if phrase in text_lower:
                score += 3.5
                word_count += 2
                phrase_matches.append(f"+{phrase}")
        
        for phrase in self.negative_phrases:
            if phrase in text_lower:
                score -= 3.5
                word_count += 2
                phrase_matches.append(f"-{phrase}")
        
        # ETAPA 2: Análise palavra por palavra (se não achou frases completas)
        if not phrase_matches:
            for i, word in enumerate(words):
                word_score = self._get_word_score(word)
                
                # Se encontrou uma palavra de sentimento
                if word_score != 0:
                    word_count += 1
                    
                    # Verificar intensificadores nas 2 palavras anteriores
                    for j in range(max(0, i-2), i):
                        if words[j] in self.intensifiers:
                            word_score *= 1.6  # Aumenta o peso em 60%
                            break
                    
                    # Verificar negadores nas 3 palavras anteriores
                    for j in range(max(0, i-3), i):
                        if words[j] in self.negators:
                            word_score *= -1  # Inverte o sentimento
                            break
                    
                    score += word_score
        
        # ETAPA 3: Fallback - se não encontrou nada
        if word_count == 0:
            # Detectar padrões de pontuação e emojis
            if any(pattern in text_lower for pattern in ['!!!', '!', ':)', '😊', '❤', 'nossa', 'uau']):
                score = 1
                word_count = 1
            elif any(pattern in text_lower for pattern in [':(', '😞', '💔', '???']):
                score = -1
                word_count = 1
            else:
                score = 0
                word_count = 1
        
        # Normalizar pontuação
        if word_count > 0:
            normalized_score = score / word_count
            
            # Aplicar ajustes para textos curtos
            if is_short_text:
                normalized_score *= 0.7
            
            # Converter para escala 0-1
            confidence = min(abs(normalized_score) / 4.0, 1.0)  # Dividir por peso máximo
            
            # Determinar label
            if normalized_score > 0.3:
                label = 'POSITIVE'
            elif normalized_score < -0.3:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
        else:
            normalized_score = 0
            confidence = 0.5
            label = 'NEUTRAL'
        
        return {
            'label': label,
            'score': round(confidence, 3),
            'debug': {
                'method': 'local_analysis',
                'raw_score': round(normalized_score, 3),
                'words_analyzed': word_count,
                'phrase_matches': phrase_matches,
                'is_short_text': is_short_text,
                'source': options.get('source', 'text')
            }
        }
    
    def _get_word_score(self, word: str) -> float:
        """Retorna a pontuação de uma palavra específica"""
        if word in self.very_positive:
            return 4.0
        elif word in self.positive_words:
            return 2.5
        elif word in self.light_positive:
            return 1.0
        elif word in self.very_negative:
            return -4.0
        elif word in self.negative_words:
            return -2.5
        elif word in self.light_negative:
            return -1.0
        else:
            return 0.0
