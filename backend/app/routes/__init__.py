"""
Módulo de rotas da API

Centraliza todas as rotas da aplicação para fácil importação e organização
"""
from fastapi import APIRouter
from .analysis import router as analysis_router
from .system import router as system_router

# Criar roteador principal
main_router = APIRouter()

# Incluir todos os roteadores
main_router.include_router(analysis_router)
main_router.include_router(system_router)

# Exportar para fácil importação
__all__ = ["main_router", "analysis_router", "system_router"]
