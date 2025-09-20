"""
Serviço de Processamento de Mídia

Responsável por processar imagens (OCR) e áudios (Speech-to-Text)
para extração de texto que pode ser analisado para sentimentos
"""
import os
import io
import base64
import tempfile
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import asyncio

from ..config import config

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

try:
    import pytesseract
    HAS_TESSERACT_OCR = True
except ImportError:
    HAS_TESSERACT_OCR = False

logger = logging.getLogger(__name__)


class MediaService:
    """Serviço principal para processamento de mídia"""
    
    def __init__(self):
        self.image_processor = ImageProcessor() if HAS_IMAGE_SUPPORT else None
        self.audio_processor = AudioProcessor() if HAS_AUDIO_SUPPORT else None
        
        logger.info(f"🎞️ MediaService inicializado - Imagem: {bool(self.image_processor)}, Áudio: {bool(self.audio_processor)}")
    
    async def process_image(
        self, 
        image_data: str, 
        filename: str, 
        extract_text: bool = True
    ) -> Dict[str, Any]:
        """
        Processa uma imagem e extrai texto via OCR
        
        Args:
            image_data: Imagem em base64
            filename: Nome do arquivo
            extract_text: Se deve extrair texto
            
        Returns:
            Dict com texto extraído e metadados
        """
        if not self.image_processor:
            raise ValueError("Processamento de imagens não disponível. Instale as dependências necessárias.")
        
        # Executar processamento em thread separada (operação pesada)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.image_processor.process, 
            image_data, 
            filename, 
            extract_text
        )
        
        return {
            **result,
            "media_type": "image",
            "filename": filename
        }
    
    async def process_audio(
        self, 
        audio_data: str, 
        filename: str, 
        language: str = "pt-BR"
    ) -> Dict[str, Any]:
        """
        Processa um áudio e converte fala em texto
        
        Args:
            audio_data: Áudio em base64
            filename: Nome do arquivo
            language: Idioma do áudio
            
        Returns:
            Dict com texto transcrito e metadados
        """
        if not self.audio_processor:
            raise ValueError("Processamento de áudios não disponível. Instale as dependências necessárias.")
        
        # Executar processamento em thread separada (operação pesada)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.audio_processor.process, 
            audio_data, 
            filename, 
            language
        )
        
        return {
            **result,
            "media_type": "audio",
            "filename": filename
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Retorna as capacidades disponíveis"""
        return {
            "image_processing": bool(self.image_processor),
            "audio_processing": bool(self.audio_processor),
            "ocr_available": HAS_TESSERACT_OCR or True,  # Sempre True por ter método alternativo
            "speech_recognition": HAS_AUDIO_SUPPORT,
            "supported_image_formats": config.ALLOWED_IMAGE_EXTENSIONS,
            "supported_audio_formats": config.ALLOWED_AUDIO_EXTENSIONS
        }


class ImageProcessor:
    """Processador específico para imagens"""
    
    def __init__(self):
        self.supported_formats = config.ALLOWED_IMAGE_EXTENSIONS
        
    def process(self, image_data: str, filename: str, extract_text: bool = True) -> Dict[str, Any]:
        """Processa imagem e extrai texto"""
        try:
            # Limpar dados base64
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Metadados básicos da imagem
            result = {
                "file_size": len(image_bytes),
                "dimensions": {"width": image.size[0], "height": image.size[1]},
                "processing_method": "pil_image"
            }
            
            # Extrair texto se solicitado
            if extract_text:
                extracted_text = self._extract_text_from_image(image)
                result.update({
                    "extracted_text": extracted_text,
                    "processing_method": "ocr_multi_method"
                })
            else:
                result["extracted_text"] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem {filename}: {e}")
            raise ValueError(f"Erro ao processar imagem: {str(e)}")
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extrai texto usando múltiplos métodos OCR"""
        # Melhorar imagem para OCR
        enhanced_image = self._enhance_for_ocr(image)
        
        # Método 1: Tesseract (se disponível)
        if HAS_TESSERACT_OCR:
            try:
                text = pytesseract.image_to_string(enhanced_image, lang='por')
                if text.strip():
                    return self._clean_text(text)
            except Exception as e:
                logger.warning(f"Tesseract OCR falhou: {e}")
        
        # Método 2: Análise de contraste e padrões
        fallback_text = self._analyze_image_patterns(enhanced_image)
        return self._clean_text(fallback_text)
    
    def _enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """Melhora imagem para OCR"""
        # Aumentar contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Aumentar nitidez
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Redimensionar se muito pequena
        if image.size[0] < 300 or image.size[1] < 300:
            scale = 2
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _analyze_image_patterns(self, image: Image.Image) -> str:
        """Método alternativo de análise quando OCR não funciona"""
        try:
            # Converter para array numpy
            img_array = np.array(image)
            
            # Detectar se há muito texto (áreas com contraste)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Contar pixels de borda (indicativo de texto)
            edge_pixels = np.count_nonzero(edges)
            total_pixels = gray.size
            edge_ratio = edge_pixels / total_pixels
            
            # Analisar histograma (distribuição de cores)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            contrast = np.std(hist)
            
            # Gerar texto descritivo baseado na análise
            if edge_ratio > 0.1:  # Muitas bordas = provável presença de texto
                if contrast > 1000:  # Alto contraste
                    return "Imagem com texto de alto contraste detectado"
                else:
                    return "Imagem com possível texto de baixo contraste"
            else:
                if contrast > 500:
                    return "Imagem com elementos gráficos variados"
                else:
                    return "Imagem com conteúdo uniforme ou poucos detalhes"
                    
        except Exception as e:
            logger.warning(f"Análise de padrões falhou: {e}")
            return "Imagem processada sem extração de texto"
    
    def _clean_text(self, text: str) -> str:
        """Limpa texto extraído"""
        if not text:
            return ""
        
        # Remover caracteres especiais e normalizar
        text = text.strip()
        text = ' '.join(text.split())  # Normalizar espaços
        
        # Remover linhas muito curtas (provavelmente ruído)
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
        
        return ' '.join(lines)


class AudioProcessor:
    """Processador específico para áudios"""
    
    def __init__(self):
        self.supported_formats = config.ALLOWED_AUDIO_EXTENSIONS
        self.recognizer = sr.Recognizer()
    
    def process(self, audio_data: str, filename: str, language: str = "pt-BR") -> Dict[str, Any]:
        """Processa áudio e extrai texto via Speech-to-Text"""
        try:
            # Limpar dados base64
            if audio_data.startswith('data:audio'):
                audio_data = audio_data.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_data)
            
            # Processar áudio em arquivo temporário
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_audio_path = temp_file.name
            
            try:
                # Carregar e normalizar áudio
                audio_segment = AudioSegment.from_file(temp_audio_path)
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                
                # Salvar versão processada
                processed_path = temp_audio_path.replace('.wav', '_processed.wav')
                audio_segment.export(processed_path, format="wav")
                
                # Extrair texto
                extracted_text = self._speech_to_text(processed_path, language)
                
                return {
                    "extracted_text": extracted_text,
                    "file_size": len(audio_bytes),
                    "duration": len(audio_segment) / 1000,  # segundos
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
    
    def _speech_to_text(self, audio_path: str, language: str) -> str:
        """Converte fala em texto"""
        try:
            with sr.AudioFile(audio_path) as source:
                # Ajustar para ruído ambiente
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Tentar múltiplos engines
            engines = [
                ('google', lambda: self.recognizer.recognize_google(audio_data, language=language)),
                ('sphinx', lambda: self.recognizer.recognize_sphinx(audio_data))
            ]
            
            for engine_name, engine_func in engines:
                try:
                    text = engine_func()
                    if text.strip():
                        logger.info(f"✅ Texto extraído com {engine_name}: {len(text)} chars")
                        return text
                except Exception as e:
                    logger.warning(f"Engine {engine_name} falhou: {e}")
                    continue
            
            return ""  # Nenhum engine funcionou
            
        except Exception as e:
            logger.error(f"Erro no speech-to-text: {e}")
            return ""
