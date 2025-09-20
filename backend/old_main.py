import json
import uuid
import asyncio
import logging
import base64
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional
import threading
import time
import sys
import os
from pathlib import Path

# Adicionar o diretório app ao path
sys.path.append(str(Path(__file__).parent / 'app'))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar processador de mídia
try:
    from services.media_processor import MediaProcessor
    HAS_MEDIA_SUPPORT = True
    logger.info("🎞️ Suporte a mídia habilitado")
except ImportError as e:
    HAS_MEDIA_SUPPORT = False
    logger.warning(f"⚠️ Suporte a mídia não disponível: {e}")

class SimpleSentimentAnalyzer:
    """Análise de sentimento aprimorada baseada em palavras-chave e regras"""
    
    def __init__(self):
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
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Análise de sentimento avançada com frases completas e contexto"""
        if not text:
            raise ValueError("Text input is required")
        
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Para textos muito curtos (menos de 3 palavras), ser mais conservador
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
                word_score = 0
                
                # Verificar tipo de palavra e atribuir peso
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
                
                # Se encontrou uma palavra de sentimento
                if word_score != 0:
                    word_count += 1
                    original_score = word_score
                    
                    # Verificar intensificadores nas 2 palavras anteriores
                    intensifier_found = False
                    for j in range(max(0, i-2), i):
                        if words[j] in self.intensifiers:
                            word_score *= 1.6  # Aumenta o peso em 60%
                            intensifier_found = True
                            break
                    
                    # Verificar negadores nas 3 palavras anteriores
                    negated = False
                    for j in range(max(0, i-3), i):
                        if words[j] in self.negators:
                            word_score *= -1  # Inverte o sentimento
                            negated = True
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
            elif any(pattern in text_lower for pattern in ['?', 'hmm', 'sei lá', 'talvez']):
                score = 0
                word_count = 1
            else:
                # Neutro padrão
                score = 0
                word_count = 1
        
        # Calcular score médio
        avg_score = score / max(word_count, 1)
        
        # NOVA LÓGICA DE CLASSIFICAÇÃO - mais precisa
        
        # Para frases identificadas (alta confiança)
        if phrase_matches:
            if avg_score >= 1.0:
                label = "POSITIVE"
                confidence = min(0.95, 0.85 + abs(avg_score) * 0.02)
            elif avg_score <= -1.0:
                label = "NEGATIVE"
                confidence = min(0.95, 0.85 + abs(avg_score) * 0.02)
            else:
                label = "NEUTRAL"
                confidence = 0.75
        
        # Para textos curtos (cuidado especial)
        elif is_short_text:
            if avg_score >= 3.0:
                label = "POSITIVE"
                confidence = min(0.85, 0.70 + abs(avg_score) * 0.03)
            elif avg_score >= 1.5:
                label = "POSITIVE"
                confidence = min(0.75, 0.60 + abs(avg_score) * 0.05)
            elif avg_score <= -3.0:
                label = "NEGATIVE"
                confidence = min(0.85, 0.70 + abs(avg_score) * 0.03)
            elif avg_score <= -1.5:
                label = "NEGATIVE"
                confidence = min(0.75, 0.60 + abs(avg_score) * 0.05)
            else:
                label = "NEUTRAL"
                confidence = 0.55 + (0.10 * (1 - abs(avg_score) / 2))
        
        # Para textos normais (melhor precisão)
        else:
            if avg_score >= 2.5:
                label = "POSITIVE"
                confidence = min(0.95, 0.80 + abs(avg_score) * 0.02)
            elif avg_score >= 1.2:
                label = "POSITIVE"
                confidence = min(0.85, 0.70 + abs(avg_score) * 0.03)
            elif avg_score >= 0.3:
                label = "POSITIVE"
                confidence = min(0.75, 0.60 + abs(avg_score) * 0.05)
            elif avg_score <= -2.5:
                label = "NEGATIVE"
                confidence = min(0.95, 0.80 + abs(avg_score) * 0.02)
            elif avg_score <= -1.2:
                label = "NEGATIVE"
                confidence = min(0.85, 0.70 + abs(avg_score) * 0.03)
            elif avg_score <= -0.3:
                label = "NEGATIVE"
                confidence = min(0.75, 0.60 + abs(avg_score) * 0.05)
            else:
                label = "NEUTRAL"
                confidence = 0.65 + (0.15 * (1 - abs(avg_score)))
        
        # Ajustes finais de confiança
        if word_count >= 4:  # Muitas palavras de sentimento = mais confiança
            confidence = min(0.98, confidence + 0.05)
        elif word_count == 1 and is_short_text:
            confidence = max(0.40, confidence - 0.20)  # Palavra isolada = baixa confiança
        
        # Palavras muito comuns = neutro forçado
        if len(words) == 1 and words[0] in ['o', 'a', 'um', 'uma', 'de', 'da', 'do', 'para', 'com', 'em']:
            label = "NEUTRAL"
            confidence = 0.35
        
        return {
            "label": label,
            "score": float(round(confidence, 3)),
            "debug": {
                "words_analyzed": word_count,
                "raw_score": float(round(score, 2)),
                "avg_score": float(round(avg_score, 2)),
                "is_short_text": is_short_text,
                "phrase_matches": phrase_matches[:3] if phrase_matches else None
            }
        }

class APIHandler(BaseHTTPRequestHandler):
    analyzer = SimpleSentimentAnalyzer()
    media_processor = MediaProcessor() if HAS_MEDIA_SUPPORT else None
    
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        # CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/v1/healthz':
            self._set_headers()
            response = {"status": "ok"}
            self.wfile.write(json.dumps(response).encode())
        elif parsed_path.path == '/api/v1/capabilities':
            self._handle_capabilities()
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/v1/analyze':
            self._handle_text_analysis()
        elif parsed_path.path == '/api/v1/analyze/image':
            self._handle_image_analysis()
        elif parsed_path.path == '/api/v1/analyze/audio':
            self._handle_audio_analysis()
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())
    
    def _handle_text_analysis(self):
        """Handle text sentiment analysis"""
        try:
            start_time = time.time()
                
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Validate request
            if 'task' not in data or data['task'] != 'sentiment':
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Only 'sentiment' task is supported"}).encode())
                return
            
            if 'input_text' not in data or not data['input_text']:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "input_text is required"}).encode())
                return
            
            # Analyze sentiment
            result = self.analyzer.analyze(data['input_text'])
            
            # Create response
            elapsed_ms = int((time.time() - start_time) * 1000)
            response = {
                "id": str(uuid.uuid4()),
                "task": "sentiment",
                "engine": "local:keyword-based",
                "result": result,
                "elapsed_ms": elapsed_ms,
                "received_at": datetime.now().isoformat()
            }
            
            self._set_headers()
            self.wfile.write(json.dumps(response).encode())
            
            logger.info(f"Processed sentiment analysis in {elapsed_ms}ms")
            
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
    
    def _handle_capabilities(self):
        """Return system capabilities"""
        capabilities = {
            "text_analysis": True,
            "media_support": HAS_MEDIA_SUPPORT,
            "supported_formats": {
                "text": ["plain/text", "application/json"],
                "image": [],
                "audio": []
            }
        }
        
        if HAS_MEDIA_SUPPORT and self.media_processor:
            media_caps = self.media_processor.get_capabilities()
            capabilities["supported_formats"]["image"] = media_caps.get("supported_image_formats", [])
            capabilities["supported_formats"]["audio"] = media_caps.get("supported_audio_formats", [])
            capabilities["image_processing"] = media_caps.get("image_processing", False)
            capabilities["audio_processing"] = media_caps.get("audio_processing", False)
            capabilities["ocr"] = media_caps.get("ocr", False)
            capabilities["speech_recognition"] = media_caps.get("speech_recognition", False)
        
        self._set_headers()
        self.wfile.write(json.dumps(capabilities, indent=2).encode())
    
    def _handle_image_analysis(self):
        """Handle image sentiment analysis"""
        if not HAS_MEDIA_SUPPORT or not self.media_processor:
            self._set_headers(501)
            self.wfile.write(json.dumps({
                "error": "Suporte a imagem não disponível",
                "detail": "Instale as dependências: pip install Pillow opencv-python pytesseract"
            }).encode())
            return
        
        try:
            start_time = time.time()
            
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Validate request
            if 'image_data' not in data:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "image_data is required"}).encode())
                return
            
            # Process image
            filename = data.get('filename', 'uploaded_image')
            media_result = self.media_processor.process_image(data['image_data'], filename)
            
            # Analyze sentiment if text was extracted
            if media_result['has_text'] and media_result['extracted_text'].strip():
                sentiment_result = self.analyzer.analyze(media_result['extracted_text'])
            else:
                sentiment_result = {
                    "label": "NEUTRAL",
                    "score": 0.5,
                    "debug": {"reason": "No text found in image"}
                }
            
            # Create response
            elapsed_ms = int((time.time() - start_time) * 1000)
            response = {
                "id": str(uuid.uuid4()),
                "task": "image_sentiment",
                "engine": "ocr+keyword-based",
                "result": sentiment_result,
                "media_analysis": media_result,
                "elapsed_ms": elapsed_ms,
                "received_at": datetime.now().isoformat()
            }
            
            self._set_headers()
            self.wfile.write(json.dumps(response).encode())
            
            logger.info(f"Processed image analysis in {elapsed_ms}ms")
            
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _handle_audio_analysis(self):
        """Handle audio sentiment analysis"""
        if not HAS_MEDIA_SUPPORT or not self.media_processor:
            self._set_headers(501)
            self.wfile.write(json.dumps({
                "error": "Suporte a áudio não disponível",
                "detail": "Instale as dependências: pip install SpeechRecognition pydub librosa"
            }).encode())
            return
        
        try:
            start_time = time.time()
            
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Validate request
            if 'audio_data' not in data:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "audio_data is required"}).encode())
                return
            
            # Process audio
            filename = data.get('filename', 'uploaded_audio')
            media_result = self.media_processor.process_audio(data['audio_data'], filename)
            
            # Analyze sentiment if text was extracted
            if media_result['has_text'] and media_result['extracted_text'].strip():
                sentiment_result = self.analyzer.analyze(media_result['extracted_text'])
            else:
                sentiment_result = {
                    "label": "NEUTRAL",
                    "score": 0.5,
                    "debug": {"reason": "No speech recognized in audio"}
                }
            
            # Create response
            elapsed_ms = int((time.time() - start_time) * 1000)
            response = {
                "id": str(uuid.uuid4()),
                "task": "audio_sentiment",
                "engine": "speech-recognition+keyword-based",
                "result": sentiment_result,
                "media_analysis": media_result,
                "elapsed_ms": elapsed_ms,
                "received_at": datetime.now().isoformat()
            }
            
            self._set_headers()
            self.wfile.write(json.dumps(response).encode())
            
            logger.info(f"Processed audio analysis in {elapsed_ms}ms")
            
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")

def run_server(port=8000):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, APIHandler)
    
    print(f"🚀 Servidor iniciado em http://localhost:{port}")
    print(f"📖 API Docs: http://localhost:{port}/api/v1/healthz")
    print(f"💚 Health: http://localhost:{port}/api/v1/healthz")
    print("🔥 Backend pronto para receber requisições!")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido")
        httpd.shutdown()

if __name__ == "__main__":
    run_server()
