FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências
COPY pyproject.toml .

# Instalar dependências Python (sem extras de dev)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copiar código-fonte e modelos
COPY src/ src/
COPY models/ models/

# GCP Cloud Run injeta a variável PORT automaticamente; fallback para 8080
ENV PORT=8080

# Expor porta
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:${PORT}/health'); assert r.status_code == 200"

# Comando de execução — Cloud Run exige escutar em $PORT
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
