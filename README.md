# 🤖 Sistema de Análise de Sentimentos com IA

> **Sistema completo para análise de sentimentos em texto, imagens e áudios usando Inteligência Artificial**



<img width="1617" height="901" alt="image" src="https://github.com/user-attachments/assets/c49b3b69-88d8-42a9-a29f-cd2a7bcd3e53" />





## 🌟 Sobre o Projeto

Este é um sistema full-stack de análise de sentimentos que processa:
- **📝 Textos**: Análise inteligente com algoritmos avançados
- **🖼️ Imagens**: Extração de texto via OCR e análise visual
- **🎧 Áudios**: Conversão de fala para texto (em desenvolvimento)

## ⚙️ Tecnologias Utilizadas

### 🔍 Backend (Python)
- **FastAPI**: Framework web moderno e rápido
- **OpenCV**: Processamento de imagens
- **PIL/Pillow**: Manipulação de imagens
- **SpeechRecognition**: Reconhecimento de fala
- **PyDub**: Processamento de áudio
- **OCR Inteligente**: Múltiplos métodos de extração de texto

### ⚖️ Frontend (React)
- **React 19**: Interface moderna e reativa
- **Vite**: Build tool rápido
- **CSS Avançado**: Design responsivo com gradientes
- **Axios**: Comunicação com API
- **Upload de Arquivos**: Suporte a drag & drop

## 🚀 Como Usar

### 🏃‍♂️ Execução Rápida

**1. Execute o script (Método mais fácil):**
```powershell
.\INICIAR.ps1
```

> ⚡ **Novo**: Script otimizado sem erros de encoding!

**2. Acesse no navegador:**
```
http://localhost:5173
```

### 🔧 Execução Manual

**Backend:**
```powershell
cd backend
python main.py
```

**Frontend:**
```powershell
cd frontend
npm install  # Primeira vez
npm run dev
```

## 🌍 URLs do Sistema

| Serviço | URL | Descrição |
|---------|-----|-------------|
| **Interface Principal** | http://localhost:5173 | Página principal do usuário |
| **API Backend** | http://localhost:8000 | Servidor de análise |
| **Health Check** | http://localhost:8000/api/v1/capabilities | Status do sistema |
| **Análise de Texto** | POST /api/v1/analyze | Endpoint para texto |
| **Análise de Imagem** | POST /api/v1/analyze/image | Endpoint para imagens |
| **Análise de Áudio** | POST /api/v1/analyze/audio | Endpoint para áudios |

## 🎯 Funcionalidades Principais

### 📝 Análise de Texto
- ✅ Análise em tempo real enquanto digita
- ✅ Algoritmo inteligente baseado em palavras-chave
- ✅ Detecção de intensificadores e negadores
- ✅ Classificação em Positivo/Negativo/Neutro
- ✅ Pontuação de confiança

### 🖼️ Análise de Imagens
- ✅ Upload via drag & drop
- ✅ OCR inteligente (múltiplos métodos)
- ✅ Detecção de texto em screenshots
- ✅ Análise visual de cores e contraste
- ✅ Formatos: JPG, PNG, BMP, TIFF, WebP

### 🎧 Análise de Áudio (Em Desenvolvimento)
- 🔄 Conversão de fala para texto
- 🔄 Suporte a múltiplos formatos
- 🔄 Análise de ton de voz

### 🎨 Interface do Usuário
- ✅ Design moderno e responsivo
- ✅ Tela cheia otimizada
- ✅ Layout de duas colunas (telas grandes)
- ✅ Histórico de análises
- ✅ Exemplos interativos
- ✅ Preview de arquivos

## 📁 Estrutura do Projeto

```
sentiment-analyzer-project/
├── 🚀 INICIAR.ps1           # Script para iniciar o sistema (ÚNICO!)
├── 📄 README.md             # Documentação principal
├── 📝 ROTEIRO.md            # Roteiro de apresentação
│
├── 🐍 backend/              # Servidor Python
│   ├── main.py              # Servidor principal FastAPI
│   ├── requirements.txt     # Dependências Python
│   └── app/
│       └── services/
│           └── media_processor.py  # Processamento de mídia
│
└── ⚖️ frontend/             # Interface React
    ├── package.json         # Dependências Node.js
    ├── index.html           # Página principal
    ├── vite.config.js       # Configuração Vite
    └── src/
        ├── App.jsx             # Componente principal
        ├── App.css             # Estilos globais
        └── components/
            ├── SentimentAnalyzer.jsx   # Componente principal
            └── SentimentAnalyzer.css   # Estilos do componente
```

## 🧑‍💻 Como Funciona o Código

### 🐍 Backend (Python)

**1. Servidor Principal (`main.py`):**
- FastAPI com endpoints RESTful
- CORS habilitado para desenvolvimento
- Roteamento para diferentes tipos de análise

**2. Algoritmo de Sentimentos:**
- Base de dados de palavras positivas/negativas
- Detecção de intensificadores ("muito", "super")
- Tratamento de negações ("não gosto")
- Pontuação por peso e frequência

**3. Processamento de Mídia (`media_processor.py`):**
- Múltiplos métodos de OCR
- Análise visual por contraste
- Detecção de padrões de texto
- Fallback inteligente

### ⚖️ Frontend (React)

**1. Componente Principal (`SentimentAnalyzer.jsx`):**
- Estados para texto, mídia e resultados
- Upload de arquivos com preview
- Comunicação assíncrona com API
- Gerenciamento de histórico

**2. Interface Responsiva:**
- Layout flex/grid dinâmico
- Duas colunas em telas grandes
- Design mobile-friendly
- Animações CSS suaves

## 🔍 API Endpoints

### GET /api/v1/capabilities
```json
{
  "text_analysis": true,
  "media_support": true,
  "image_processing": true,
  "ocr": true,
  "supported_formats": {
    "image": [".jpg", ".png", ".bmp"],
    "audio": [".mp3", ".wav"]
  }
}
```

### POST /api/v1/analyze
```json
{
  "task": "sentiment",
  "input_text": "Eu amo este produto!"
}
```

### POST /api/v1/analyze/image
```json
{
  "image_data": "base64_string",
  "filename": "screenshot.png"
}
```

## 🔧 Solução de Problemas

### ❗ Problemas Comuns

**Backend não inicia:**
```powershell
cd backend
pip install -r requirements.txt
python main.py
```

**Frontend não inicia:**
```powershell
cd frontend
npm install
npm run dev
```

**Tesseract não encontrado:**
- ✅ **Solução**: O sistema usa OCR alternativo!
- Não precisa instalar Tesseract
- Funciona com análise visual inteligente

### 🔄 Para Parar o Sistema
```powershell
# Use Ctrl+C nas janelas do PowerShell ou feche-as diretamente

# Para reiniciar:
.\INICIAR.ps1
```

## 🎆 Diferenciais Técnicos

1. **OCR sem dependências externas** - Sistema funciona sem instalar Tesseract
2. **Análise visual inteligente** - Detecta sentimento por características visuais
3. **Múltiplos fallbacks** - Sistema robusto que sempre funciona
4. **Interface moderna** - Design responsivo e profissional
5. **Código limpo** - Arquitetura bem estruturada

## 📊 Resultados Esperados

- **Precisão de Texto**: ~85% para português
- **Detecção de Imagens**: Alta para screenshots e documentos
- **Tempo de Resposta**: < 500ms para texto, < 2s para imagens
- **Formatos Suportados**: 15+ tipos de arquivo

## 🚀 Desenvolvido por

**Lucas** - Sistema completo de análise de sentimentos com IA

---

**🎉 Pronto para usar! Execute `./INICIAR.ps1` e acesse `http://localhost:5173`**
