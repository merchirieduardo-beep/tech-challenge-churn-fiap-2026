"""API de inferência FastAPI para previsão de churn."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, PredictionInput, PredictionResponse
from src.models.mlp import ChurnMLP
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

MODELS_DIR = Path("models")


class ModelRegistry:
    """Registro de modelos carregados em memória."""

    model: ChurnMLP | None = None
    preprocessor: object | None = None
    metadata: dict | None = None
    is_ready: bool = False


registry = ModelRegistry()


def load_model() -> None:
    """Carrega modelo e preprocessor do disco."""
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    model_path = MODELS_DIR / "mlp_churn.pth"
    metadata_path = MODELS_DIR / "model_metadata.joblib"

    if not all(p.exists() for p in [preprocessor_path, model_path, metadata_path]):
        logger.warning("modelos_nao_encontrados", path=str(MODELS_DIR))
        return

    registry.preprocessor = joblib.load(preprocessor_path)
    registry.metadata = joblib.load(metadata_path)

    input_dim = registry.metadata["input_dim"]
    registry.model = ChurnMLP(input_dim=input_dim)
    registry.model.load_state_dict(torch.load(model_path, weights_only=True))
    registry.model.eval()

    # Remover sigmoid se usando BCEWithLogitsLoss
    registry.model.network[-1] = nn.Identity()

    registry.is_ready = True
    logger.info("modelo_carregado", input_dim=input_dim)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifecycle: carrega modelo na inicialização."""
    setup_logging()
    logger.info("api_iniciando")
    load_model()
    yield
    logger.info("api_encerrando")


app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência para previsão de churn em telecomunicações usando MLP (PyTorch)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_latency(request, call_next):
    """Middleware para logging de latência das requisições."""
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000

    logger.info(
        "request_processada",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=f"{latency_ms:.2f}",
    )

    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Endpoint de health check."""
    return HealthResponse(
        status="healthy" if registry.is_ready else "degraded",
        model_loaded=registry.is_ready,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput) -> PredictionResponse:
    """Endpoint de predição de churn.

    Recebe dados de um cliente e retorna a probabilidade de churn.
    """
    if not registry.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro.",
        )

    try:
        import pandas as pd

        input_df = pd.DataFrame([input_data.model_dump()])
        features = registry.preprocessor.transform(input_df)

        with torch.no_grad():
            tensor = torch.FloatTensor(features)
            probability = torch.sigmoid(registry.model(tensor)).item()

        threshold = registry.metadata.get("threshold", 0.5)
        prediction = int(probability >= threshold)

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=_classify_risk(probability),
        )

    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=422, detail=f"Erro na predição: {e}") from e


def _classify_risk(probability: float) -> str:
    """Classifica o nível de risco baseado na probabilidade."""
    if probability >= 0.7:
        return "alto"
    if probability >= 0.4:
        return "medio"
    return "baixo"
