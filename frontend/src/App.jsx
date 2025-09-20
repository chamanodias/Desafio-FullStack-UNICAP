import { useState, useEffect } from 'react'
import SentimentAnalyzer from './components/SentimentAnalyzer'
import './App.css'

function App() {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  })

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth
      const height = window.innerHeight
      
      setWindowSize({ width, height })
      
      // Detectar se está em tela cheia ou janela maximizada
      const isLikelyFullscreen = 
        width >= screen.availWidth * 0.95 && 
        height >= screen.availHeight * 0.90
      
      setIsFullscreen(isLikelyFullscreen)
      
      // Adicionar/remover classe no body para CSS específico
      if (isLikelyFullscreen) {
        document.body.classList.add('fullscreen')
      } else {
        document.body.classList.remove('fullscreen')
      }
    }

    // Executar na inicialização
    handleResize()
    
    // Adicionar listener para mudanças de tamanho
    window.addEventListener('resize', handleResize)
    window.addEventListener('orientationchange', handleResize)
    
    return () => {
      window.removeEventListener('resize', handleResize)
      window.removeEventListener('orientationchange', handleResize)
      document.body.classList.remove('fullscreen')
    }
  }, [])

  return (
    <div className={`App ${isFullscreen ? 'fullscreen-mode' : ''}`}>
      <header className="App-header">
        <h1>Análise de Sentimento - IA</h1>
        <p>Sistema de análise de sentimento com modelos locais e externos</p>
      </header>
      <main>
        <SentimentAnalyzer />
      </main>
      {!isFullscreen && (
        <footer className="App-footer">
          <p>Desenvolvido para o desafio FASA/UNICAP - 2025</p>
        </footer>
      )}
    </div>
  )
}

export default App
