"""Configuração compartilhada dos testes."""

import sys
from pathlib import Path

# Garante que o diretório raiz está no path
sys.path.insert(0, str(Path(__file__).parent.parent))
