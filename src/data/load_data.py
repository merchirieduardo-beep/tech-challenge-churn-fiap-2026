"""Módulo para carregamento e validação inicial dos dados."""

from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from src.utils.logging import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed")

# Schema de validação dos dados brutos
raw_schema = DataFrameSchema(
    {
        "customerID": Column(str, nullable=False),
        "gender": Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1])),
        "Partner": Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": Column(int, pa.Check.ge(0)),
        "PhoneService": Column(str, pa.Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str, nullable=False),
        "InternetService": Column(str, nullable=False),
        "OnlineSecurity": Column(str, nullable=False),
        "OnlineBackup": Column(str, nullable=False),
        "DeviceProtection": Column(str, nullable=False),
        "TechSupport": Column(str, nullable=False),
        "StreamingTV": Column(str, nullable=False),
        "StreamingMovies": Column(str, nullable=False),
        "Contract": Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(str, nullable=False),
        "MonthlyCharges": Column(float, pa.Check.ge(0)),
        "TotalCharges": Column(object, nullable=True),
        "Churn": Column(str, pa.Check.isin(["Yes", "No"])),
    },
    strict=False,
)


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Carrega dados brutos do CSV.

    Args:
        path: Caminho para o arquivo CSV. Se None, usa o caminho padrão.

    Returns:
        DataFrame com os dados brutos validados.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        pandera.errors.SchemaError: Se a validação falhar.
    """
    data_path = path or RAW_DATA_PATH

    if not data_path.exists():
        msg = f"Arquivo não encontrado: {data_path}"
        raise FileNotFoundError(msg)

    logger.info("carregando_dados", path=str(data_path))
    df = pd.read_csv(data_path)
    logger.info("dados_carregados", linhas=len(df), colunas=len(df.columns))

    # Validação do schema
    validated_df = raw_schema.validate(df, lazy=True)
    logger.info("schema_validado", status="ok")

    return validated_df


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza inicial dos dados brutos.

    - Converte TotalCharges para numérico
    - Remove customerID (não é feature)
    - Converte target para binário (1=Churn, 0=No Churn)
    """
    logger.info("preprocessamento_iniciado")

    df = df.copy()

    # TotalCharges tem espaços em branco → converter para numérico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Preencher NaN em TotalCharges com 0 (clientes novos sem cobrança)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Remover customerID
    df = df.drop(columns=["customerID"])

    # Converter target para binário
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    logger.info(
        "preprocessamento_concluido",
        linhas=len(df),
        colunas=len(df.columns),
        churn_rate=f"{df['Churn'].mean():.2%}",
    )

    return df
