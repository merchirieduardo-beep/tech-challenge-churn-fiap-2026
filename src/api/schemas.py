"""Schemas Pydantic para validação de entrada/saída da API."""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Schema de entrada para predição de churn."""

    gender: str = Field(..., description="Gênero do cliente", examples=["Male", "Female"])
    SeniorCitizen: int = Field(..., description="É idoso (0 ou 1)", ge=0, le=1)
    Partner: str = Field(..., description="Tem parceiro", examples=["Yes", "No"])
    Dependents: str = Field(..., description="Tem dependentes", examples=["Yes", "No"])
    tenure: int = Field(..., description="Meses como cliente", ge=0)
    PhoneService: str = Field(..., description="Serviço de telefone", examples=["Yes", "No"])
    MultipleLines: str = Field(
        ..., description="Múltiplas linhas", examples=["Yes", "No", "No phone service"]
    )
    InternetService: str = Field(
        ..., description="Tipo de internet", examples=["DSL", "Fiber optic", "No"]
    )
    OnlineSecurity: str = Field(
        ..., description="Segurança online", examples=["Yes", "No", "No internet service"]
    )
    OnlineBackup: str = Field(
        ..., description="Backup online", examples=["Yes", "No", "No internet service"]
    )
    DeviceProtection: str = Field(
        ..., description="Proteção de dispositivo", examples=["Yes", "No", "No internet service"]
    )
    TechSupport: str = Field(
        ..., description="Suporte técnico", examples=["Yes", "No", "No internet service"]
    )
    StreamingTV: str = Field(
        ..., description="Streaming TV", examples=["Yes", "No", "No internet service"]
    )
    StreamingMovies: str = Field(
        ..., description="Streaming Filmes", examples=["Yes", "No", "No internet service"]
    )
    Contract: str = Field(
        ..., description="Tipo de contrato", examples=["Month-to-month", "One year", "Two year"]
    )
    PaperlessBilling: str = Field(..., description="Fatura digital", examples=["Yes", "No"])
    PaymentMethod: str = Field(
        ...,
        description="Método de pagamento",
        examples=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    MonthlyCharges: float = Field(..., description="Valor mensal", ge=0)
    TotalCharges: float = Field(..., description="Valor total acumulado", ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 79.85,
                    "TotalCharges": 958.20,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema de resposta da predição."""

    churn_prediction: int = Field(..., description="Predição binária (0=No Churn, 1=Churn)")
    churn_probability: float = Field(..., description="Probabilidade de churn (0 a 1)")
    risk_level: str = Field(..., description="Nível de risco (baixo, medio, alto)")


class HealthResponse(BaseModel):
    """Schema de resposta do health check."""

    status: str = Field(..., description="Status da API")
    model_loaded: bool = Field(..., description="Se o modelo está carregado")
    version: str = Field(..., description="Versão da API")
