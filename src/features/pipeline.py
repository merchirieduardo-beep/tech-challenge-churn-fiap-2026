"""Pipeline de pré-processamento com transformadores customizados."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Definição das features por tipo
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET = "Churn"


class TenureGroupTransformer(BaseEstimator, TransformerMixin):
    """Transformador customizado que cria grupos de tenure."""

    def fit(self, X: pd.DataFrame, y: None = None) -> "TenureGroupTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 72, np.inf],
            labels=["0-12", "13-24", "25-48", "49-72", "73+"],
        )
        return X


class ChargesRatioTransformer(BaseEstimator, TransformerMixin):
    """Transformador customizado que cria razão entre charges."""

    def fit(self, X: pd.DataFrame, y: None = None) -> "ChargesRatioTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["charges_ratio"] = np.where(
            X["TotalCharges"] > 0,
            X["MonthlyCharges"] / X["TotalCharges"],
            0.0,
        )
        return X


def build_preprocessor() -> ColumnTransformer:
    """Constrói o preprocessador sklearn para features numéricas e categóricas."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    logger.info(
        "preprocessor_construido",
        num_features=len(NUMERIC_FEATURES),
        cat_features=len(CATEGORICAL_FEATURES),
    )

    return preprocessor


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e target do DataFrame.

    Returns:
        Tupla (X, y) com features e target.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    logger.info("features_preparadas", X_shape=X.shape, y_shape=y.shape)
    return X, y
