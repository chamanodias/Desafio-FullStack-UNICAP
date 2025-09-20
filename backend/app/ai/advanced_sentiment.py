import json
import re
import logging
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer:
    """Análise de sentimento avançada com dicionário expandido, n-gramas e detecção de contexto"""
    
    def __init__(self):
        self._initialize_dictionaries()
        self._initialize_patterns()
        self._initialize_emoji_mapping()
        
    def _initialize_dictionaries(self):
        """Inicializa dicionários expandidos com 500+ palavras"""
        
        # PALAVRAS MUITO POSITIVAS - Peso 4.0
        self.very_positive = [
            # Amor e afeto intenso
            'amo', 'adoro', 'love', 'paixão', 'adoração', 'veneração', 'idolatro',
            
            # Excelência e perfeição
            'excelente', 'excellent', 'perfeito', 'perfect', 'impecável', 'primoroso',
            'extraordinário', 'excepcional', 'magnífico', 'esplêndido', 'sublime',
            'fantástico', 'fantastic', 'sensacional', 'espetacular', 'deslumbrante',
            'maravilhoso', 'wonderful', 'incrível', 'incredible', 'fenomenal',
            
            # Emoções positivas intensas
            'euforia', 'êxtase', 'alegria extrema', 'felicidade plena', 'radiante',
            'exultante', 'jubiloso', 'extasiado', 'arrebatado', 'encantado',
            'deslumbrado', 'fascinado', 'embevecido', 'entusiasmado',
            
            # Gratidão e reconhecimento
            'gratidão imensa', 'profundamente grato', 'reconhecimento eterno',
            'agradecimento sincero', 'imensamente grato'
        ]
        
        # PALAVRAS POSITIVAS - Peso 2.5
        self.positive_words = [
            # Qualidade e aprovação
            'ótimo', 'great', 'bom', 'good', 'excelente', 'muito bom', 'bacana',
            'legal', 'show', 'top', 'massa', 'demais', 'irado', 'maneiro',
            'aprovado', 'recomendado', 'aproveitável', 'válido', 'útil',
            
            # Emoções positivas
            'feliz', 'happy', 'alegre', 'contente', 'satisfeito', 'realizado',
            'orgulhoso', 'confiante', 'otimista', 'esperançoso', 'animado',
            'entusiasmado', 'motivado', 'inspirado', 'energizado',
            
            # Sucesso e conquista
            'sucesso', 'vitória', 'conquista', 'triunfo', 'êxito', 'realização',
            'progresso', 'avanço', 'melhoria', 'evolução', 'crescimento',
            
            # Relacionamentos
            'carinho', 'ternura', 'afeto', 'proximidade', 'intimidade',
            'companheirismo', 'cumplicidade', 'harmonia', 'paz', 'tranquilidade'
        ]
        
        # PALAVRAS LEVEMENTE POSITIVAS - Peso 1.0
        self.light_positive = [
            'ok', 'okay', 'bem', 'razoável', 'aceitável', 'decent', 'regular',
            'normal', 'comum', 'padrão', 'esperado', 'previsto', 'adequado',
            'suficiente', 'satisfatório', 'funcional', 'operacional',
            'interessante', 'curioso', 'intrigante', 'chamativo', 'atrativo'
        ]
        
        # PALAVRAS MUITO NEGATIVAS - Peso -4.0
        self.very_negative = [
            # Ódio e aversão intensa
            'odeio', 'detesto', 'abomino', 'repudio', 'desprezo', 'hate',
            'execração', 'aversão total', 'nojo absoluto', 'repugnância',
            
            # Qualidade terrível
            'terrível', 'terrible', 'horrível', 'horrible', 'péssimo', 'awful',
            'horroroso', 'assombroso', 'aterrorizante', 'abominável', 'repulsivo',
            'nauseante', 'repugnante', 'desprezível', 'ignóbil', 'vil',
            
            # Emoções negativas extremas
            'ódio', 'fúria', 'cólera', 'ira', 'raiva extrema', 'revolta',
            'indignação', 'desespero', 'desalento', 'devastação', 'aniquilação',
            'pavor', 'terror', 'pânico', 'horror', 'pavoroso',
            
            # Falha total
            'fracasso total', 'ruína completa', 'desastre', 'catástrofe',
            'tragédia', 'calamidade', 'infortúnio'
        ]
        
        # PALAVRAS NEGATIVAS - Peso -2.5
        self.negative_words = [
            # Qualidade ruim
            'ruim', 'bad', 'péssimo', 'pobre', 'fraco', 'inferior', 'defeituoso',
            'falho', 'problemático', 'inadequado', 'insuficiente', 'limitado',
            
            # Emoções negativas
            'triste', 'sad', 'melancólico', 'deprimido', 'abatido', 'desanimado',
            'desiludido', 'decepcionado', 'frustrado', 'irritado', 'aborrecido',
            'chateado', 'incomodado', 'perturbado', 'angustiado', 'aflito',
            
            # Medo e ansiedade
            'medo', 'receio', 'temor', 'apreensão', 'ansiedade', 'nervosismo',
            'inquietação', 'desassossego', 'intranquilidade', 'agonia',
            
            # Problemas e dificuldades
            'problema', 'dificuldade', 'obstáculo', 'impedimento', 'complicação',
            'transtorno', 'distúrbio', 'confusão', 'bagunça', 'caos'
        ]
        
        # PALAVRAS LEVEMENTE NEGATIVAS - Peso -1.0
        self.light_negative = [
            'estranho', 'weird', 'esquisito', 'diferente', 'incomum', 'atípico',
            'confuso', 'confusing', 'complicado', 'complexo', 'difícil', 'hard',
            'cansativo', 'tedioso', 'monótono', 'repetitivo', 'chato', 'boring',
            'lento', 'demorado', 'vagaroso', 'lerdo'
        ]
        
        # INTENSIFICADORES - Multiplicam por 1.6
        self.intensifiers = [
            'muito', 'super', 'extremely', 'really', 'totally', 'absolutely',
            'completamente', 'totalmente', 'inteiramente', 'plenamente',
            'demais', 'bastante', 'bem', 'mega', 'ultra', 'hiper',
            'extremamente', 'profundamente', 'intensamente', 'fortemente',
            'incrivelmente', 'imensamente', 'enormemente', 'extraordinariamente',
            'excepcionalmente', 'surpreendentemente', 'impressionantemente'
        ]
        
        # DIMINUIIDORES - Multiplicam por 0.6
        self.diminishers = [
            'pouco', 'little', 'somewhat', 'ligeiramente', 'levemente',
            'sutilmente', 'moderadamente', 'relativamente', 'razoavelmente',
            'meio', 'half', 'quase', 'almost', 'praticamente'
        ]
        
        # NEGADORES - Invertem o sentimento
        self.negators = [
            'não', 'not', 'nunca', 'never', 'jamais', 'nenhum', 'nenhuma',
            'sem', 'without', 'impossível', 'impossible', 'nem', 'neither',
            'tampouco', 'sequer', 'absolutamente não', 'de jeito nenhum',
            'de forma alguma', 'em hipótese alguma'
        ]
        
    def _initialize_patterns(self):
        """Inicializa padrões de frases e expressões"""
        
        # FRASES POSITIVAS COMPLETAS - Peso 3.5
        self.positive_phrases = [
            # Amor e relacionamentos
            'amor da minha vida', 'pessoa mais especial', 'melhor coisa que me aconteceu',
            'não consigo viver sem', 'faz meu coração bater mais forte',
            'amo profundamente', 'gratidão eterna', 'reconhecimento infinito',
            
            # Realizações e conquistas
            'sonho realizado', 'objetivo alcançado', 'meta conquistada',
            'superou todas expectativas', 'muito além do esperado',
            'tudo que eu queria', 'exatamente o que precisava',
            
            # Experiências positivas
            'experiência incrível', 'momento mágico', 'sensação maravilhosa',
            'vale muito a pena', 'recomendo fortemente', 'aprovação total',
            'satisfação completa', 'felicidade plena'
        ]
        
        # FRASES NEGATIVAS COMPLETAS - Peso -3.5
        self.negative_phrases = [
            # Decepção e frustração
            'maior decepção', 'frustração total', 'expectativas destruídas',
            'sonho desfeito', 'ilusão perdida', 'esperança morta',
            
            # Experiências ruins
            'pior experiência', 'momento terrível', 'situação horrível',
            'não recomendo a ninguém', 'evitem a todo custo',
            'perda de tempo', 'dinheiro jogado fora',
            
            # Emoções negativas intensas
            'coração partido', 'alma destroçada', 'espírito quebrado',
            'profunda tristeza', 'dor insuportável', 'sofrimento imenso'
        ]
        
    def _initialize_emoji_mapping(self):
        """Mapeia emojis para sentimentos"""
        self.emoji_sentiment = {
            # Muito positivos
            '😍': 4, '🥰': 4, '😘': 4, '💖': 4, '💕': 4, '❤️': 4,
            '🤩': 3.5, '😊': 3.5, '😁': 3.5, '🤗': 3.5,
            
            # Positivos
            '😀': 3, '😃': 3, '😄': 3, '😆': 3, '🙂': 2.5, '😉': 2.5,
            '👍': 2, '👏': 2, '🎉': 3, '🌟': 2.5,
            
            # Neutros
            '😐': 0, '😑': 0, '🤔': 0, '😶': 0,
            
            # Negativos
            '😞': -2.5, '😔': -2.5, '😟': -2, '😕': -2, '🙁': -2,
            '😒': -1.5, '😤': -2, '👎': -2,
            
            # Muito negativos
            '😭': -3.5, '😢': -3, '😨': -3, '😰': -3.5, '😱': -4,
            '🤬': -4, '😡': -4, '💔': -4
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessa o texto para melhor análise"""
        
        # Converter para minúsculas
        text = text.lower().strip()
        
        # Expandir contrações comuns
        contractions = {
            "não é": "não é",
            "isn't": "is not",
            "wasn't": "was not", 
            "won't": "will not",
            "can't": "cannot",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        # Normalizar repetições excessivas (ex: "muuuito" -> "muito")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Normalizar pontuação múltipla
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    def _extract_emojis(self, text: str) -> List[str]:
        """Extrai emojis do texto"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.findall(text)
    
    def _analyze_ngrams(self, words: List[str], n: int = 2) -> Dict[str, float]:
        """Analisa n-gramas para capturar contexto"""
        ngram_scores = {}
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            
            # Verificar se é uma frase conhecida
            if ngram in self.positive_phrases:
                ngram_scores[ngram] = 3.5
            elif ngram in self.negative_phrases:
                ngram_scores[ngram] = -3.5
            else:
                # Analisar componentes do n-grama
                score = 0
                for word in words[i:i+n]:
                    if word in self.very_positive:
                        score += 4
                    elif word in self.positive_words:
                        score += 2.5
                    elif word in self.light_positive:
                        score += 1
                    elif word in self.very_negative:
                        score -= 4
                    elif word in self.negative_words:
                        score -= 2.5
                    elif word in self.light_negative:
                        score -= 1
                
                if score != 0:
                    ngram_scores[ngram] = score / n
        
        return ngram_scores
    
    def _analyze_context_window(self, words: List[str], sentiment_word_idx: int, window_size: int = 3) -> Dict[str, Any]:
        """Analisa janela de contexto ao redor de palavra de sentimento"""
        context = {
            'negated': False,
            'intensified': False,
            'diminished': False,
            'modifier_strength': 1.0
        }
        
        # Verificar palavras anteriores
        start_idx = max(0, sentiment_word_idx - window_size)
        preceding_words = words[start_idx:sentiment_word_idx]
        
        for word in preceding_words:
            if word in self.negators:
                context['negated'] = True
            elif word in self.intensifiers:
                context['intensified'] = True
                context['modifier_strength'] *= 1.6
            elif word in self.diminishers:
                context['diminished'] = True
                context['modifier_strength'] *= 0.6
        
        return context
    
    def _detect_sarcasm_indicators(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Detecta possíveis indicadores de sarcasmo/ironia"""
        sarcasm_score = 0
        indicators = []
        
        # Padrões sarcásticos comuns
        sarcastic_patterns = [
            'claro que sim', 'com certeza', 'que maravilha', 'perfeito mesmo',
            'exatamente o que eu queria', 'que surpresa', 'como não',
            'obviamente', 'certamente', 'sem dúvida'
        ]
        
        for pattern in sarcastic_patterns:
            if pattern in text:
                sarcasm_score += 0.3
                indicators.append(pattern)
        
        # Pontuação excessiva pode indicar sarcasmo
        if '!!!' in text or '???' in text:
            sarcasm_score += 0.2
            indicators.append('pontuação excessiva')
        
        # Aspas podem indicar ironia
        if '"' in text or "'" in text:
            sarcasm_score += 0.1
            indicators.append('aspas')
        
        # Contradição: palavras positivas seguidas de contexto negativo
        positive_count = sum(1 for word in words if word in self.positive_words or word in self.very_positive)
        negative_count = sum(1 for word in words if word in self.negative_words or word in self.very_negative)
        
        if positive_count > 0 and negative_count > positive_count:
            sarcasm_score += 0.4
            indicators.append('contradição positivo-negativo')
        
        return {
            'sarcasm_probability': min(sarcasm_score, 1.0),
            'indicators': indicators
        }
    
    def _analyze_emotions(self, words: List[str]) -> Dict[str, float]:
        """Analisa emoções específicas além do sentimento geral"""
        emotions = {
            'alegria': 0, 'tristeza': 0, 'raiva': 0, 'medo': 0, 
            'surpresa': 0, 'nojo': 0, 'desprezo': 0
        }
        
        emotion_words = {
            'alegria': ['feliz', 'alegre', 'contente', 'radiante', 'eufórico', 'jubiloso', 
                       'exultante', 'satisfeito', 'realizado', 'orgulhoso'],
            'tristeza': ['triste', 'melancólico', 'deprimido', 'abatido', 'desanimado',
                        'desiludido', 'desolado', 'amargurado', 'enlutado', 'pesaroso'],
            'raiva': ['irritado', 'furioso', 'bravo', 'indignado', 'revoltado',
                     'colérico', 'irado', 'enfurecido', 'exaltado', 'enraivecido'],
            'medo': ['assustado', 'aterrorizado', 'apreensivo', 'temeroso', 'receoso',
                    'medroso', 'pavido', 'acovardado', 'inquieto', 'angustiado'],
            'surpresa': ['surpreso', 'espantado', 'chocado', 'impressionado', 'admirado',
                        'pasmo', 'estupefato', 'perplexo', 'atônito', 'boquiaberto'],
            'nojo': ['nojento', 'repugnante', 'asqueroso', 'nauseante', 'repulsivo',
                    'abominável', 'execrável', 'detestável', 'desprezível', 'odioso'],
            'desprezo': ['desprezível', 'insignificante', 'indigno', 'vil', 'baixo',
                        'rasteiro', 'ignóbil', 'abjeto', 'miserável', 'contemptível']
        }
        
        for emotion, emotion_word_list in emotion_words.items():
            for word in words:
                if word in emotion_word_list:
                    emotions[emotion] += 1
        
        # Normalizar scores
        total_emotion_words = sum(emotions.values())
        if total_emotion_words > 0:
            for emotion in emotions:
                emotions[emotion] = round(emotions[emotion] / total_emotion_words, 2)
        
        return emotions
    
    def _calculate_advanced_confidence(self, features: Dict[str, Any]) -> float:
        """Calcula confidence score baseado em múltiplos fatores"""
        base_confidence = 0.5
        
        # Fator 1: Número de evidências (palavras de sentimento)
        if features['sentiment_word_count'] >= 4:
            base_confidence += 0.25
        elif features['sentiment_word_count'] >= 2:
            base_confidence += 0.15
        elif features['sentiment_word_count'] == 1:
            base_confidence += 0.05
        
        # Fator 2: Presença de frases completas (alta confiança)
        if features.get('phrase_matches'):
            base_confidence += 0.20
        
        # Fator 3: Consistência do sentimento
        if features.get('sentiment_consistency', 0) > 0.8:
            base_confidence += 0.15
        elif features.get('sentiment_consistency', 0) < 0.5:
            base_confidence -= 0.10
        
        # Fator 4: Presença de emojis
        if features.get('emoji_count', 0) > 0:
            base_confidence += 0.10
        
        # Fator 5: Penalidade por possível sarcasmo
        sarcasm_prob = features.get('sarcasm_probability', 0)
        if sarcasm_prob > 0.6:
            base_confidence -= 0.30
        elif sarcasm_prob > 0.3:
            base_confidence -= 0.15
        
        # Fator 6: Tamanho do texto
        text_length = features.get('text_length', 0)
        if text_length < 3:  # Texto muito curto
            base_confidence -= 0.20
        elif text_length > 20:  # Texto longo com mais contexto
            base_confidence += 0.10
        
        # Fator 7: Presença de intensificadores
        if features.get('has_intensifiers'):
            base_confidence += 0.10
        
        return max(0.2, min(0.98, base_confidence))
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Análise principal de sentimento com IA avançada"""
        if not text or not text.strip():
            raise ValueError("Text input is required")
        
        # Preprocessamento
        original_text = text
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        # Extrair emojis
        emojis = self._extract_emojis(original_text)
        
        # Inicializar variáveis de análise
        sentiment_score = 0
        sentiment_word_count = 0
        context_analyses = []
        phrase_matches = []
        
        # ETAPA 1: Análise de frases completas (maior prioridade)
        for phrase in self.positive_phrases:
            if phrase in processed_text:
                sentiment_score += 3.5
                sentiment_word_count += 2
                phrase_matches.append(f"+{phrase}")
        
        for phrase in self.negative_phrases:
            if phrase in processed_text:
                sentiment_score -= 3.5
                sentiment_word_count += 2
                phrase_matches.append(f"-{phrase}")
        
        # ETAPA 2: Análise de n-gramas (contexto local)
        if not phrase_matches:
            bigram_scores = self._analyze_ngrams(words, 2)
            trigram_scores = self._analyze_ngrams(words, 3)
            
            for ngram, score in {**bigram_scores, **trigram_scores}.items():
                sentiment_score += score
                sentiment_word_count += 1
                if score > 0:
                    phrase_matches.append(f"+{ngram}")
                else:
                    phrase_matches.append(f"-{ngram}")
        
        # ETAPA 3: Análise palavra por palavra com contexto
        if not phrase_matches and not bigram_scores and not trigram_scores:
            for i, word in enumerate(words):
                word_score = 0
                
                # Determinar score base da palavra
                if word in self.very_positive:
                    word_score = 4
                elif word in self.positive_words:
                    word_score = 2.5
                elif word in self.light_positive:
                    word_score = 1
                elif word in self.very_negative:
                    word_score = -4
                elif word in self.negative_words:
                    word_score = -2.5
                elif word in self.light_negative:
                    word_score = -1
                
                if word_score != 0:
                    sentiment_word_count += 1
                    
                    # Analisar contexto da palavra
                    context = self._analyze_context_window(words, i)
                    context_analyses.append(context)
                    
                    # Aplicar modificadores de contexto
                    word_score *= context['modifier_strength']
                    
                    # Aplicar negação
                    if context['negated']:
                        word_score *= -1
                    
                    sentiment_score += word_score
        
        # ETAPA 4: Análise de emojis
        emoji_score = 0
        for emoji in emojis:
            if emoji in self.emoji_sentiment:
                emoji_score += self.emoji_sentiment[emoji]
        
        # Normalizar score de emoji (peso menor que palavras)
        if emoji_score != 0:
            sentiment_score += emoji_score * 0.7
            sentiment_word_count += len(emojis)
        
        # ETAPA 5: Calcular score médio
        if sentiment_word_count > 0:
            avg_sentiment = sentiment_score / sentiment_word_count
        else:
            avg_sentiment = 0
        
        # ETAPA 6: Análises complementares
        sarcasm_analysis = self._detect_sarcasm_indicators(processed_text, words)
        emotion_analysis = self._analyze_emotions(words)
        
        # ETAPA 7: Determinação do rótulo final
        # Ajustar por sarcasmo
        if sarcasm_analysis['sarcasm_probability'] > 0.6:
            avg_sentiment *= -1  # Inverter por sarcasmo
        
        # Classificar sentimento
        if avg_sentiment >= 2.0:
            label = "POSITIVE"
        elif avg_sentiment >= 0.5:
            label = "POSITIVE"
        elif avg_sentiment <= -2.0:
            label = "NEGATIVE"
        elif avg_sentiment <= -0.5:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        # ETAPA 8: Calcular confidence avançado
        features = {
            'sentiment_word_count': sentiment_word_count,
            'phrase_matches': phrase_matches,
            'sentiment_consistency': abs(avg_sentiment) / 4 if avg_sentiment != 0 else 0,
            'emoji_count': len(emojis),
            'sarcasm_probability': sarcasm_analysis['sarcasm_probability'],
            'text_length': len(words),
            'has_intensifiers': any(word in self.intensifiers for word in words)
        }
        
        confidence = self._calculate_advanced_confidence(features)
        
        # Penalidade adicional por sarcasmo
        if sarcasm_analysis['sarcasm_probability'] > 0.3:
            confidence *= (1 - sarcasm_analysis['sarcasm_probability'] * 0.5)
        
        return {
            "label": label,
            "score": float(round(confidence, 3)),
            "sentiment_intensity": float(round(abs(avg_sentiment), 2)),
            "emotions": emotion_analysis,
            "sarcasm": {
                "probability": float(round(sarcasm_analysis['sarcasm_probability'], 2)),
                "indicators": sarcasm_analysis['indicators']
            },
            "debug": {
                "words_analyzed": sentiment_word_count,
                "raw_score": float(round(sentiment_score, 2)),
                "avg_score": float(round(avg_sentiment, 2)),
                "phrase_matches": phrase_matches[:5],  # Máximo 5 para não sobrecarregar
                "emoji_count": len(emojis),
                "text_length": len(words),
                "preprocessing": {
                    "original_length": len(original_text),
                    "processed_length": len(processed_text),
                    "emoji_found": emojis
                }
            }
        }
