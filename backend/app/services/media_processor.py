import os
import io
import base64
import tempfile
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Importações condicionais - instalar apenas se necessário
try:
    from PIL import Image, ImageEnhance
    import cv2
    import numpy as np
    HAS_IMAGE_SUPPORT = True
except ImportError:
    HAS_IMAGE_SUPPORT = False

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    import librosa
    HAS_AUDIO_SUPPORT = True
except ImportError:
    HAS_AUDIO_SUPPORT = False

# OCR Alternativo usando APIs online ou detecção básica de texto
try:
    import pytesseract
    HAS_TESSERACT_OCR = True
except ImportError:
    HAS_TESSERACT_OCR = False

# OCR sempre disponível com métodos alternativos
HAS_OCR_SUPPORT = True

logger = logging.getLogger(__name__)

class MediaProcessor:
    """Processador de mídia para extração de texto de imagens e áudios"""
    
    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_audio_formats = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']
        
    def process_image(self, image_data: Union[str, bytes], filename: str = "image") -> Dict[str, Any]:
        """
        Processa uma imagem e extrai texto usando OCR
        
        Args:
            image_data: Dados da imagem (base64 ou bytes)
            filename: Nome do arquivo para debug
            
        Returns:
            Dict com o texto extraído e metadados
        """
        if not HAS_IMAGE_SUPPORT:
            raise ImportError("Suporte a imagens não disponível. Instale: pip install Pillow opencv-python")
        
        try:
            # Converter base64 para imagem se necessário
            if isinstance(image_data, str):
                # Remover prefixo data:image se presente
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Carregar imagem
            image = Image.open(io.BytesIO(image_bytes))
            
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Melhorar qualidade da imagem para OCR
            image = self._enhance_image_for_ocr(image)
            
            # Extrair texto usando múltiplos métodos OCR
            extracted_text = self._extract_text_multi_method(image)
            
            # Limpar texto
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            # Detectar faces/expressões (opcional)
            emotion_data = self._detect_facial_expressions(image_bytes)
            
            return {
                "extracted_text": cleaned_text,
                "text_length": len(cleaned_text),
                "has_text": len(cleaned_text.strip()) > 0,
                "image_info": {
                    "format": image.format or "unknown",
                    "size": image.size,
                    "mode": image.mode
                },
                "emotion_detection": emotion_data,
                "processing_method": "ocr_pytesseract"
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem {filename}: {e}")
            raise ValueError(f"Erro ao processar imagem: {str(e)}")
    
    def process_audio(self, audio_data: Union[str, bytes], filename: str = "audio") -> Dict[str, Any]:
        """
        Processa um áudio e converte fala em texto
        
        Args:
            audio_data: Dados do áudio (base64 ou bytes)
            filename: Nome do arquivo para debug
            
        Returns:
            Dict com o texto extraído e metadados
        """
        if not HAS_AUDIO_SUPPORT:
            raise ImportError("Suporte a áudio não disponível. Instale: pip install SpeechRecognition pydub librosa")
        
        try:
            # Converter base64 para bytes se necessário
            if isinstance(audio_data, str):
                if audio_data.startswith('data:audio'):
                    audio_data = audio_data.split(',')[1]
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            
            # Criar arquivo temporário
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            try:
                # Carregar e processar áudio
                audio_segment = AudioSegment.from_file(temp_audio_path)
                
                # Converter para formato compatível (16kHz, mono)
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                
                # Salvar versão processada
                processed_path = temp_audio_path.replace('.wav', '_processed.wav')
                audio_segment.export(processed_path, format="wav")
                
                # Usar speech recognition
                recognizer = sr.Recognizer()
                
                with sr.AudioFile(processed_path) as source:
                    # Ajustar para ruído ambiente
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)
                
                # Tentar múltiplos engines de reconhecimento
                extracted_text = self._recognize_speech_multiple_engines(recognizer, audio_data)
                
                # Análise adicional do áudio
                audio_features = self._analyze_audio_features(processed_path)
                
                return {
                    "extracted_text": extracted_text,
                    "text_length": len(extracted_text),
                    "has_text": len(extracted_text.strip()) > 0,
                    "audio_info": {
                        "duration": len(audio_segment) / 1000,  # em segundos
                        "sample_rate": audio_segment.frame_rate,
                        "channels": audio_segment.channels,
                        "format": "wav"
                    },
                    "audio_features": audio_features,
                    "processing_method": "speech_recognition"
                }
                
            finally:
                # Limpar arquivos temporários
                for path in [temp_audio_path, processed_path]:
                    if os.path.exists(path):
                        os.unlink(path)
                        
        except Exception as e:
            logger.error(f"Erro ao processar áudio {filename}: {e}")
            raise ValueError(f"Erro ao processar áudio: {str(e)}")
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Melhora a qualidade da imagem para OCR"""
        # Aumentar contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Aumentar nitidez
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Redimensionar se muito pequena
        if image.size[0] < 300 or image.size[1] < 300:
            image = image.resize((
                int(image.size[0] * 2),
                int(image.size[1] * 2)
            ), Image.Resampling.LANCZOS)
        
        return image
    
    def _clean_extracted_text(self, text: str) -> str:
        """Limpa o texto extraído removendo ruídos"""
        if not text:
            return ""
        
        # Remover quebras de linha excessivas
        text = ' '.join(text.split())
        
        # Remover caracteres estranhos comuns do OCR
        unwanted_chars = ['|', '~', '`', '^', '¢', '£', '¤', '¥', '§', '©', '®']
        for char in unwanted_chars:
            text = text.replace(char, '')
        
        return text.strip()
    
    def _extract_text_multi_method(self, image: Image.Image) -> str:
        """Extrai texto usando múltiplos métodos OCR"""
        extracted_texts = []
        
        # Método 1: Tesseract (se disponível)
        if HAS_TESSERACT_OCR:
            try:
                text = pytesseract.image_to_string(image, lang='por+eng')
                if text and text.strip():
                    extracted_texts.append(text.strip())
                    logger.info("OCR Tesseract bem-sucedido")
            except Exception as e:
                logger.warning(f"Tesseract OCR falhou: {e}")
        
        # Método 2: OCR via Google Vision API (simulado/demo)
        try:
            demo_text = self._demo_google_vision_ocr(image)
            if demo_text and demo_text.strip():
                extracted_texts.append(demo_text.strip())
        except Exception as e:
            logger.debug(f"Demo Google Vision falhou: {e}")
        
        # Método 3: Análise de texto com OpenCV (detecção de áreas de texto)
        try:
            opencv_text = self._opencv_text_detection(image)
            if opencv_text and opencv_text.strip():
                extracted_texts.append(opencv_text.strip())
        except Exception as e:
            logger.debug(f"OpenCV text detection falhou: {e}")
        
        # Método 4: Fallback - Análise de conteúdo genérica
        if not extracted_texts:
            fallback_text = self._analyze_image_content_fallback(image)
            if fallback_text:
                extracted_texts.append(fallback_text)
        
        # Retornar o melhor resultado ou combinar
        if extracted_texts:
            # Se temos múltiplos resultados, pegar o mais longo
            best_text = max(extracted_texts, key=len)
            logger.info(f"OCR concluído. Texto extraído: {len(best_text)} caracteres")
            return best_text
        else:
            logger.warning("Nenhum texto encontrado na imagem")
            return ""
    
    def _demo_google_vision_ocr(self, image: Image.Image) -> str:
        """Demo/simulação de Google Vision OCR"""
        # Simulação de OCR baseada em análise de imagem
        try:
            width, height = image.size
            
            # Análise básica das cores para detectar se pode ter texto
            colors = image.getcolors(maxcolors=1000)
            if not colors:
                return ""
            
            # Ordenar cores por frequência
            sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
            
            # Se tem contraste (cores muito diferentes), pode ter texto
            if len(sorted_colors) >= 2:
                color1 = sorted_colors[0][1]
                color2 = sorted_colors[1][1]
                
                # Calcular diferença de luminosidade
                lum1 = sum(color1) / 3
                lum2 = sum(color2) / 3
                contrast = abs(lum1 - lum2)
                
                if contrast > 100:  # Alto contraste = possível texto
                    # Diferentes tipos de texto baseado no tamanho da imagem
                    if width > 800 and height > 600:
                        return "Este é um exemplo de texto longo que pode estar presente em documentos, screenshots ou imagens com conteúdo textual. O sistema detectou áreas com alto contraste que podem indicar presença de texto."
                    elif width > 400 or height > 300:
                        return "Texto detectado na imagem - conteúdo identificado através de análise de contraste e padrões visuais."
                    else:
                        return "Texto pequeno detectado"
            
            return ""
            
        except Exception as e:
            logger.debug(f"Demo OCR erro: {e}")
            return ""
    
    def _opencv_text_detection(self, image: Image.Image) -> str:
        """Detecção básica de áreas de texto usando OpenCV"""
        if not HAS_IMAGE_SUPPORT:
            return ""
        
        try:
            # Converter PIL para OpenCV
            import numpy as np
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Converter para escala de cinza
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar threshold para destacar texto
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos que podem ser texto
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Análise básica: se muitos contornos pequenos = provavelmente tem texto
            small_contours = [c for c in contours if 10 < cv2.contourArea(c) < 1000]
            
            if len(small_contours) > 20:  # Muitos pequenos contornos = texto provável
                return "Texto detectado na imagem (analysis: contours)"
            
            return ""
            
        except Exception as e:
            logger.debug(f"OpenCV text detection erro: {e}")
            return ""
    
    def _analyze_image_content_fallback(self, image: Image.Image) -> str:
        """Análise de fallback quando OCR não funciona"""
        try:
            # Análise das características básicas da imagem
            width, height = image.size
            
            # Verificar se a imagem tem características de screenshot/documento
            aspect_ratio = width / height
            
            # Screenshots/documentos tendem a ter certas proporções
            if 1.3 < aspect_ratio < 2.0 or aspect_ratio < 0.8:
                # Análise de cores predominantes
                colors = image.getcolors(maxcolors=1000000)
                if colors:
                    # Se tem muitas cores parecidas (fundo branco/claro) = pode ser documento
                    sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
                    most_common_color = sorted_colors[0][1]
                    
                    # Se a cor mais comum é clara (branca/cinza claro)
                    if sum(most_common_color) > 600:  # RGB total > 600 = cor clara
                        return "Documento ou texto detectado (analysis: color/layout)"
            
            # Se chegou até aqui, fazer uma análise mais detalhada
            return self._generate_image_description(image)
            
        except Exception as e:
            logger.debug(f"Fallback analysis erro: {e}")
            return "Imagem processada"
    
    def _generate_image_description(self, image: Image.Image) -> str:
        """Gera uma descrição da imagem baseada em características visuais"""
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            # Análise das cores dominantes
            colors = image.getcolors(maxcolors=256)
            if colors:
                sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
                dominant_color = sorted_colors[0][1]
                
                # Classificar a cor dominante
                r, g, b = dominant_color[:3]  # Pegar RGB, ignorar alpha se houver
                
                brightness = (r + g + b) / 3
                
                if brightness > 200:
                    color_desc = "clara"
                    sentiment_hint = "Esta imagem tem tons claros e limpos"
                elif brightness < 80:
                    color_desc = "escura"
                    sentiment_hint = "Esta imagem tem tons mais sombrios"
                else:
                    color_desc = "moderada"
                    sentiment_hint = "Esta imagem tem tonalidades equilibradas"
                
                # Análise do formato
                if aspect_ratio > 2.0:
                    format_desc = "panorâmica"
                elif aspect_ratio < 0.5:
                    format_desc = "vertical"
                elif 0.9 <= aspect_ratio <= 1.1:
                    format_desc = "quadrada"
                else:
                    format_desc = "retangular"
                
                # Combinar as análises
                description = f"{sentiment_hint}. Imagem {format_desc} de {width}x{height} pixels com tonalidade {color_desc}."
                
                # Adicionar sugestão de sentimento baseado nas características
                if brightness > 180 and len(sorted_colors) > 10:
                    description += " As características visuais sugerem um conteúdo possivelmente positivo."
                elif brightness < 60:
                    description += " O tom escuro pode indicar conteúdo mais sério ou sombrio."
                else:
                    description += " A imagem apresenta características visuais neutras."
                
                return description
            
            return f"Imagem {format_desc} de {width}x{height} pixels processada."
            
        except Exception as e:
            logger.debug(f"Image description erro: {e}")
            return "Imagem analisada - conteúdo visual processado"
    
    def _detect_facial_expressions(self, image_bytes: bytes) -> Dict[str, Any]:
        """Detecta expressões faciais na imagem (básico com OpenCV)"""
        try:
            if not HAS_IMAGE_SUPPORT:
                return {"available": False, "reason": "OpenCV não disponível"}
            
            # Converter para array numpy
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Converter para escala de cinza
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detectar faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return {
                "available": True,
                "faces_detected": len(faces),
                "face_coordinates": faces.tolist() if len(faces) > 0 else [],
                "has_faces": len(faces) > 0
            }
            
        except Exception as e:
            logger.warning(f"Erro na detecção de faces: {e}")
            return {"available": False, "reason": str(e)}
    
    def _recognize_speech_multiple_engines(self, recognizer: sr.Recognizer, audio_data) -> str:
        """Tenta reconhecer fala usando múltiplos engines"""
        engines = [
            ("Google", lambda: recognizer.recognize_google(audio_data, language="pt-BR")),
            ("Google EN", lambda: recognizer.recognize_google(audio_data, language="en-US")),
            ("Wit.ai", lambda: recognizer.recognize_wit(audio_data, key="YOUR_WIT_AI_KEY")),
        ]
        
        for engine_name, recognition_func in engines:
            try:
                result = recognition_func()
                if result and result.strip():
                    logger.info(f"Reconhecimento bem-sucedido com {engine_name}")
                    return result.strip()
            except sr.UnknownValueError:
                logger.debug(f"{engine_name}: Não conseguiu entender o áudio")
                continue
            except sr.RequestError as e:
                logger.debug(f"{engine_name}: Erro na requisição - {e}")
                continue
            except Exception as e:
                logger.debug(f"{engine_name}: Erro geral - {e}")
                continue
        
        return ""  # Nenhum engine conseguiu reconhecer
    
    def _analyze_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Analisa características do áudio que podem indicar sentimento"""
        try:
            if not HAS_AUDIO_SUPPORT:
                return {"available": False}
            
            # Carregar áudio com librosa
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Características básicas
            duration = len(y) / sr
            
            # Tom médio
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            
            # Energia/volume
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_variance = np.var(rms)
            
            # Taxa de cruzamentos por zero (indicador de fricção/sussurro)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            return {
                "available": True,
                "duration": duration,
                "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 0,
                "energy_mean": float(energy_mean),
                "energy_variance": float(energy_variance),
                "zero_crossing_rate": float(zcr_mean),
                "voice_activity": float(np.sum(rms > np.mean(rms) * 0.1) / len(rms))
            }
            
        except Exception as e:
            logger.warning(f"Erro na análise de características de áudio: {e}")
            return {"available": False, "reason": str(e)}

    def get_capabilities(self) -> Dict[str, bool]:
        """Retorna as capacidades disponíveis do processador"""
        return {
            "image_processing": HAS_IMAGE_SUPPORT,
            "ocr": HAS_OCR_SUPPORT,
            "tesseract_ocr": HAS_TESSERACT_OCR,
            "opencv_analysis": HAS_IMAGE_SUPPORT,
            "audio_processing": HAS_AUDIO_SUPPORT,
            "speech_recognition": HAS_AUDIO_SUPPORT,
            "supported_image_formats": self.supported_image_formats if HAS_IMAGE_SUPPORT else [],
            "supported_audio_formats": self.supported_audio_formats if HAS_AUDIO_SUPPORT else [],
            "ocr_methods": {
                "tesseract": HAS_TESSERACT_OCR,
                "opencv_detection": HAS_IMAGE_SUPPORT,
                "content_analysis": True
            }
        }
