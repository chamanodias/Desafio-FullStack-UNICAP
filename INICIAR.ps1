Write-Host "🤖 INICIANDO SISTEMA DE ANÁLISE DE SENTIMENTOS..." -ForegroundColor Green -BackgroundColor DarkBlue
Write-Host ""

# Parar processos antigos
Write-Host "🛑 Parando processos antigos..." -ForegroundColor Yellow
Get-Process -Name "python" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2

# Verificar se backend existe
if (!(Test-Path "backend\main.py")) {
    Write-Host "❌ Erro: backend\main.py não encontrado!" -ForegroundColor Red
    exit 1
}

# Verificar se frontend existe
if (!(Test-Path "frontend\package.json")) {
    Write-Host "❌ Erro: frontend\package.json não encontrado!" -ForegroundColor Red
    exit 1
}

# Iniciar Backend
Write-Host "🐍 Iniciando Backend Python..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; Write-Host 'BACKEND RODANDO' -ForegroundColor Green; python main.py"

# Aguardar backend
Write-Host "⏳ Aguardando backend..." -ForegroundColor Yellow
Start-Sleep 8

# Iniciar Frontend
Write-Host "⚛️ Iniciando Frontend React..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; Write-Host 'FRONTEND RODANDO' -ForegroundColor Magenta; npm run dev"

# Aguardar frontend
Write-Host "⏳ Aguardando frontend..." -ForegroundColor Yellow
Start-Sleep 10

# Abrir navegador
Write-Host "🌐 Abrindo navegador..." -ForegroundColor White
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "✅ SISTEMA INICIADO!" -ForegroundColor Green -BackgroundColor DarkGreen
Write-Host ""
Write-Host "📍 URLs:" -ForegroundColor Cyan
Write-Host "   🌐 Interface: http://localhost:5173" -ForegroundColor Yellow
Write-Host "   🔧 Backend: http://localhost:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "🚀 Pronto para usar!" -ForegroundColor Green
