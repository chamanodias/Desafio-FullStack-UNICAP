import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './SentimentAnalyzer.css';

const SentimentAnalyzer = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [liveResult, setLiveResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useExternal, setUseExternal] = useState(false);
  const [history, setHistory] = useState([]);
  const [enableLiveAnalysis, setEnableLiveAnalysis] = useState(true);
  const [mediaFile, setMediaFile] = useState(null);
  const [mediaPreview, setMediaPreview] = useState(null);
  const [mediaType, setMediaType] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  const API_URL = 'http://localhost:8000/api/v1';

  // Monitorar mudanças de viewport
  useEffect(() => {
    const handleResize = () => {
      setViewport({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
    };
  }, []);

  // Carregar capacidades do sistema
  useEffect(() => {
    const loadCapabilities = async () => {
      try {
        const response = await axios.get(`${API_URL}/capabilities`);
        setCapabilities(response.data);
        console.log('Capacidades do sistema:', response.data);
      } catch (err) {
        console.warn('Erro ao carregar capacidades:', err);
        setCapabilities({ media_support: false });
      }
    };
    loadCapabilities();
  }, []);

  // Análise em tempo real (debounced)
  const performLiveAnalysis = useCallback(async (text) => {
    if (!text.trim() || text.length < 2 || !enableLiveAnalysis) {
      setLiveResult(null);
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/analyze`, {
        task: 'sentiment',
        input_text: text,
        use_external: false // Sempre usar local para análise em tempo real
      });
      setLiveResult(response.data.result);
    } catch (err) {
      console.warn('Live analysis failed:', err);
      setLiveResult(null);
    }
  }, [enableLiveAnalysis]);

  // Debounce para análise em tempo real
  useEffect(() => {
    const timer = setTimeout(() => {
      performLiveAnalysis(inputText);
    }, 800); // Aguarda 800ms após parar de digitar

    return () => clearTimeout(timer);
  }, [inputText, performLiveAnalysis]);

  const analyzeText = async () => {
    if (!inputText.trim()) {
      setError('Por favor, digite um texto para analisar.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/analyze`, {
        task: 'sentiment',
        input_text: inputText,
        use_external: useExternal
      });

      const newResult = response.data;
      setResult(newResult);
      setLiveResult(null); // Limpar resultado ao vivo quando analisar oficialmente
      
      // Adicionar ao histórico
      const historyItem = {
        id: newResult.id,
        text: inputText,
        result: newResult.result,
        engine: newResult.engine,
        timestamp: new Date().toLocaleString()
      };
      setHistory(prev => [historyItem, ...prev.slice(0, 4)]); // Manter apenas 5 itens
      
    } catch (err) {
      console.error('Erro na análise:', err);
      setError(
        err.response?.data?.detail || 
        'Erro ao conectar com o servidor. Verifique se o backend está rodando.'
      );
    } finally {
      setLoading(false);
    }
  };
  
  const clearHistory = () => {
    setHistory([]);
  };
  
  // Funções para upload de mídia
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const fileType = file.type;
    let mediaCategory = null;
    
    if (fileType.startsWith('image/')) {
      mediaCategory = 'image';
    } else if (fileType.startsWith('audio/')) {
      mediaCategory = 'audio';
    } else {
      setError('Tipo de arquivo não suportado. Use imagens ou áudios.');
      return;
    }
    
    setMediaFile(file);
    setMediaType(mediaCategory);
    setInputText(''); // Limpar texto quando selecionar arquivo
    setError(null);
    setResult(null);
    setLiveResult(null);
    
    // Criar preview
    if (mediaCategory === 'image') {
      const reader = new FileReader();
      reader.onload = (e) => {
        setMediaPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else if (mediaCategory === 'audio') {
      setMediaPreview(URL.createObjectURL(file));
    }
  };
  
  const clearMedia = () => {
    setMediaFile(null);
    setMediaPreview(null);
    setMediaType(null);
    if (mediaPreview && mediaType === 'audio') {
      URL.revokeObjectURL(mediaPreview);
    }
  };
  
  const analyzeMedia = async () => {
    if (!mediaFile) {
      setError('Por favor, selecione uma imagem ou áudio.');
      return;
    }
    
    if (!capabilities?.media_support) {
      setError('Suporte a mídia não disponível no servidor.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      // Converter arquivo para base64
      const base64Data = await fileToBase64(mediaFile);
      
      const endpoint = mediaType === 'image' ? '/analyze/image' : '/analyze/audio';
      const dataField = mediaType === 'image' ? 'image_data' : 'audio_data';
      
      const response = await axios.post(`${API_URL}${endpoint}`, {
        [dataField]: base64Data,
        filename: mediaFile.name
      });
      
      const newResult = response.data;
      setResult(newResult);
      setLiveResult(null);
      
      // Adicionar ao histórico
      const historyItem = {
        id: newResult.id,
        text: `[${mediaType.toUpperCase()}] ${mediaFile.name}`,
        result: newResult.result,
        engine: newResult.engine,
        timestamp: new Date().toLocaleString(),
        mediaType: mediaType,
        extractedText: newResult.media_analysis?.extracted_text
      };
      setHistory(prev => [historyItem, ...prev.slice(0, 4)]);
      
    } catch (err) {
      console.error('Erro na análise de mídia:', err);
      setError(
        err.response?.data?.error || 
        'Erro ao processar arquivo. Verifique o formato e tente novamente.'
      );
    } finally {
      setLoading(false);
    }
  };
  
  // Conversão de arquivo para base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        // Remover o prefixo data:mime-type;base64,
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
    });
  };
  
  const loadExample = (example) => {
    setInputText(example);
    setError(null);
    setResult(null);
    setLiveResult(null);
  };

  const getSentimentColor = (label) => {
    switch (label) {
      case 'POSITIVE': return '#28a745';
      case 'NEGATIVE': return '#dc3545';
      case 'NEUTRAL': return '#ffc107';
      default: return '#6c757d';
    }
  };

  const getSentimentEmoji = (label) => {
    switch (label) {
      case 'POSITIVE': return '😊';
      case 'NEGATIVE': return '😞';
      case 'NEUTRAL': return '😐';
      default: return '🤔';
    }
  };

  const getSentimentText = (label) => {
    switch (label) {
      case 'POSITIVE': return 'Positivo';
      case 'NEGATIVE': return 'Negativo';
      case 'NEUTRAL': return 'Neutro';
      default: return label;
    }
  };
  
  const examples = [
    { text: "Eu amo este produto, é fantástico!", emoji: "😍", expected: "POSITIVE" },
    { text: "Este produto é terrível, não recomendo", emoji: "😠", expected: "NEGATIVE" },
    { text: "O produto é ok, nada demais", emoji: "😐", expected: "NEUTRAL" },
    { text: "Estou muito feliz com essa compra!", emoji: "🎉", expected: "POSITIVE" },
    { text: "Não gosto nada, péssima qualidade", emoji: "😡", expected: "NEGATIVE" },
    { text: "Funciona bem, mas poderia ser melhor", emoji: "🤔", expected: "NEUTRAL" }
  ];
  
  const getResultMessage = (result) => {
    const messages = {
      POSITIVE: [
        "Que sentimento positivo! 🎆",
        "Detectei muita positividade! ✨",
        "Sentimento otimista identificado! 🌈"
      ],
      NEGATIVE: [
        "Hmm, sentimento negativo detectado 😔",
        "Parece que há frustração aqui 😒",
        "Sentimento crítico identificado 💢"
      ],
      NEUTRAL: [
        "Sentimento neutro, balanced! ⚖️",
        "Opinião equilibrada detectada 📊",
        "Perspectiva neutra identificada 📝"
      ]
    };
    const messageList = messages[result.label] || ["Resultado analisado! 🤖"];
    return messageList[Math.floor(Math.random() * messageList.length)];
  };

  // Gerar classes CSS dinâmicas baseadas no viewport
  const getViewportClasses = () => {
    const { width, height } = viewport;
    const classes = ['sentiment-analyzer'];
    
    // Classes baseadas na largura
    if (width < 480) classes.push('viewport-xs');
    else if (width < 768) classes.push('viewport-sm');
    else if (width < 1024) classes.push('viewport-md');
    else if (width < 1400) classes.push('viewport-lg');
    else if (width < 1920) classes.push('viewport-xl');
    else classes.push('viewport-xxl');
    
    // Classes baseadas na altura
    if (height < 600) classes.push('viewport-short');
    else if (height < 800) classes.push('viewport-medium');
    else classes.push('viewport-tall');
    
    // Classes para orientação
    if (width > height) classes.push('viewport-landscape');
    else classes.push('viewport-portrait');
    
    // Classes específicas para layouts - muito conservador
    if (width >= 1600 && height >= 900) classes.push('enable-two-column');
    if (width >= 1920 && height >= 1080) classes.push('enable-4k-layout');
    
    return classes.join(' ');
  };

  return (
    <div className={getViewportClasses()}>
      <div className="analyzer-card">
        <div className="header-section">
          <h2>🤖 Análise de Sentimento com IA</h2>
          <p className="subtitle">Descubra o sentimento por trás das suas palavras</p>
        </div>
        
        {/* Layout principal com duas colunas em telas grandes */}
        <div className="main-content-wrapper">
          <div className="left-column">
            {/* Seção de Exemplos */}
            <div className="examples-section">
              <h4>📝 Exemplos rápidos:</h4>
              <div className="example-buttons">
                {examples.slice(0, 3).map((example, index) => (
                  <button 
                    key={index}
                    onClick={() => loadExample(example.text)}
                    className="example-button"
                    title={`${example.text} (Esperado: ${example.expected})`}
                  >
                    <span className="example-emoji">{example.emoji}</span>
                    <span className="example-text">
                      {example.text.length > 25 ? example.text.substring(0, 25) + '...' : example.text}
                    </span>
                  </button>
                ))}
              </div>
              <p className="examples-hint">
                💡 <strong>Dica:</strong> Clique nos exemplos para testar rapidamente!
              </p>
            </div>
            

            <div className="input-section">
              <div className="textarea-wrapper">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Digite seu texto aqui para análise... Exemplo: 'Estou muito feliz hoje!'"
                  rows={4}
                  className="text-input"
                  maxLength={500}
                />
                <div className="char-counter">
                  {inputText.length}/500 caracteres
                </div>
                
                {/* Análise ao vivo */}
                {liveResult && enableLiveAnalysis && (
                  <div className="live-analysis">
                    <div className="live-indicator">
                      ⚡ Análise rápida:
                    </div>
                    <div className="live-result">
                      <span className="live-emoji">{getSentimentEmoji(liveResult.label)}</span>
                      <span 
                        className="live-label"
                        style={{ color: getSentimentColor(liveResult.label) }}
                      >
                        {getSentimentText(liveResult.label)}
                      </span>
                      <span className="live-confidence">
                        ({(liveResult.score * 100).toFixed(0)}%)
                      </span>
                      {liveResult.debug?.is_short_text && (
                        <span className="live-note">
                          📝 texto curto
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
          
              <div className="controls">
                <div className="options">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={useExternal}
                      onChange={(e) => setUseExternal(e.target.checked)}
                    />
                    <span className="checkmark"></span>
                    🌐 Usar API Externa (Hugging Face)
                  </label>
                  <small className="option-hint">
                    {useExternal ? 'Usando modelo avançado online' : 'Usando modelo local rápido'}
                  </small>
                  
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={enableLiveAnalysis}
                      onChange={(e) => setEnableLiveAnalysis(e.target.checked)}
                    />
                    <span className="checkmark"></span>
                    ⚡ Análise em tempo real
                  </label>
                  <small className="option-hint">
                    {enableLiveAnalysis ? 'Analisando enquanto você digita' : 'Análise apenas ao clicar'}
                  </small>
                </div>

                <div className="button-group">
                  <button
                    onClick={() => setInputText('')}
                    className="clear-button"
                    disabled={!inputText.trim()}
                  >
                    🗑️ Limpar
                  </button>
                  
                  <button
                    onClick={analyzeText}
                    disabled={loading || !inputText.trim()}
                    className="analyze-button"
                  >
                    {loading ? (
                      <>
                        <div className="spinner"></div>
                        Analisando...
                      </>
                    ) : (
                      <>🔍 Analisar Sentimento</>
                    )}
                  </button>
                </div>
              </div>
            </div>
            
            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                <div>
                  <strong>Ops! Algo deu errado:</strong>
                  <p>{error}</p>
                </div>
              </div>
            )}
          </div>
          
          <div className="right-column">
            {/* Upload de mídia sempre visível */}
            <div className="media-upload-section">
              <h3>🖼️ Upload de Mídia</h3>
              
              <div className="upload-area">
                <input
                  type="file"
                  onChange={handleFileUpload}
                  accept="image/*,audio/*"
                  className="file-input"
                  id="media-file"
                />
                <label htmlFor="media-file" className="upload-button">
                  📎 Selecionar Arquivo
                </label>
                <div className="upload-hint">
                  🖼️ Imagens: JPG, PNG, GIF | 🎧 Áudios: MP3, WAV, M4A
                </div>
              </div>
              
              {mediaFile && (
                <div className="media-preview">
                  <div className="preview-header">
                    <span className="file-info">
                      {mediaType === 'image' ? '🖼️' : '🎧'} {mediaFile.name}
                    </span>
                    <button 
                      onClick={clearMedia} 
                      className="clear-media-btn"
                      title="Remover arquivo"
                    >
                      ❌
                    </button>
                  </div>
                  
                  {mediaPreview && mediaType === 'image' && (
                    <img 
                      src={mediaPreview} 
                      alt="Preview" 
                      className="image-preview"
                    />
                  )}
                  
                  {mediaPreview && mediaType === 'audio' && (
                    <>
                      <audio controls className="audio-preview">
                        <source src={mediaPreview} type={mediaFile.type} />
                        Seu navegador não suporta o elemento de áudio.
                      </audio>
                    </>
                  )}
                  
                  <div className="media-actions">
                    <button
                      onClick={analyzeMedia}
                      disabled={loading || !mediaFile}
                      className="analyze-media-button"
                    >
                      {loading ? (
                        <>
                          <div className="spinner"></div>
                          Analisando...
                        </>
                      ) : (
                        <>🔍 Analisar {mediaType === 'image' ? 'Imagem' : 'Áudio'}</>
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Resultados ou placeholder */}
            {!result ? (
              <div className="results-placeholder">
                <div className="placeholder-content">
                  <div className="placeholder-icon">🤖</div>
                  <h4>Aguardando Análise</h4>
                  <p>Digite um texto ou faça upload de uma mídia para começar!</p>
                  
                  <div className="tips-section">
                    <h5>💡 Dicas rápidas:</h5>
                    <ul>
                      <li>✨ Use a análise em tempo real</li>
                      <li>🌐 Experimente a API externa para mais precisão</li>
                      <li>🖼️ Faça upload de imagens com texto</li>
                      <li>🎧 Analise áudios convertendo para texto</li>
                    </ul>
                  </div>
                </div>
              </div>
            ) : (
              <div className="result-section">
                <h3>🎯 Resultado da Análise</h3>
                
                <div className="result-message">
                  {getResultMessage(result.result)}
                </div>
                
                <div className="result-card">
                  <div className="sentiment-result">
                    <div className="sentiment-main">
                      <div 
                        className="sentiment-label"
                        style={{ color: getSentimentColor(result.result.label) }}
                      >
                        <span className="emoji">{getSentimentEmoji(result.result.label)}</span>
                        <span className="label-text">{getSentimentText(result.result.label)}</span>
                      </div>
                      
                      <div className="confidence-bar-container">
                        <div className="confidence-label">
                          <strong>Confiança: {(result.result.score * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="confidence-bar">
                          <div 
                            className="confidence-fill"
                            style={{ 
                              width: `${result.result.score * 100}%`,
                              backgroundColor: getSentimentColor(result.result.label)
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="analysis-details">
                    <div className="detail-grid">
                      <div className="detail-item">
                        <span className="detail-icon">🤖</span>
                        <div>
                          <strong>Engine:</strong>
                          <span>{result.engine}</span>
                        </div>
                      </div>
                      <div className="detail-item">
                        <span className="detail-icon">⏱️</span>
                        <div>
                          <strong>Tempo:</strong>
                          <span>{result.elapsed_ms}ms</span>
                        </div>
                      </div>
                      {result.result.debug && (
                        <div className="detail-item">
                          <span className="detail-icon">🔍</span>
                          <div>
                            <strong>Análise:</strong>
                            <span>{result.result.debug.words_analyzed} palavras</span>
                          </div>
                        </div>
                      )}
                      
                      {/* Informações de mídia */}
                      {result.media_analysis && (
                        <>
                          <div className="detail-item">
                            <span className="detail-icon">
                              {result.task === 'image_sentiment' ? '🖼️' : '🎧'}
                            </span>
                            <div>
                              <strong>Tipo:</strong>
                              <span>{result.task === 'image_sentiment' ? 'Imagem' : 'Áudio'}</span>
                            </div>
                          </div>
                          
                          {result.media_analysis.extracted_text && (
                            <div className="detail-item">
                              <span className="detail-icon">📝</span>
                              <div>
                                <strong>Texto Extraído:</strong>
                                <span className="extracted-text">
                                  "{result.media_analysis.extracted_text.substring(0, 100)}
                                  {result.media_analysis.extracted_text.length > 100 ? '...' : ''}"
                                </span>
                              </div>
                            </div>
                          )}
                          
                          {result.media_analysis.image_info && (
                            <div className="detail-item">
                              <span className="detail-icon">📈</span>
                              <div>
                                <strong>Resolução:</strong>
                                <span>{result.media_analysis.image_info.size[0]} x {result.media_analysis.image_info.size[1]}</span>
                              </div>
                            </div>
                          )}
                          
                          {result.media_analysis.audio_info && (
                            <div className="detail-item">
                              <span className="detail-icon">⏱️</span>
                              <div>
                                <strong>Duração:</strong>
                                <span>{result.media_analysis.audio_info.duration.toFixed(1)}s</span>
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Histórico */}
            {history.length > 0 && (
              <div className="history-section">
                <div className="history-header">
                  <h4>📜 Histórico Recente</h4>
                  <button onClick={clearHistory} className="clear-history-btn">
                    Limpar
                  </button>
                </div>
                
                <div className="history-list">
                  {history.map((item, index) => (
                    <div key={item.id} className="history-item">
                      <div className="history-text">
                        <span className="history-emoji">
                          {getSentimentEmoji(item.result.label)}
                        </span>
                        <span className="history-content">
                          {item.text.length > 50 ? 
                            item.text.substring(0, 50) + '...' : 
                            item.text
                          }
                        </span>
                      </div>
                      <div className="history-meta">
                        <span className="history-sentiment">
                          {getSentimentText(item.result.label)}
                        </span>
                        <span className="history-confidence">
                          {(item.result.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SentimentAnalyzer;
