"""Modelos baseline para comparação com a MLP."""

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_preprocessor
from src.utils.logging import get_logger
from src.utils.reproducibility import GLOBAL_SEED

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula métricas de avaliação do modelo.

    Returns:
        Dicionário com accuracy, precision, recall, f1, roc_auc.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_baselines(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict[str, dict]:
    """Treina e avalia modelos baseline com validação cruzada estratificada.

    Args:
        X: Features do dataset.
        y: Target binário.
        n_splits: Número de folds para cross-validation.

    Returns:
        Dicionário com métricas de cada modelo.
    """
    preprocessor = build_preprocessor()

    models = {
        "DummyClassifier": DummyClassifier(strategy="stratified", random_state=GLOBAL_SEED),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=GLOBAL_SEED, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=GLOBAL_SEED, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=GLOBAL_SEED),
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    results = {}

    for name, model in models.items():
        logger.info("treinando_baseline", modelo=name)

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        # Cross-validation com predições out-of-fold
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")

        # Para ROC-AUC, precisamos de probabilidades
        if hasattr(model, "predict_proba"):
            y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        else:
            y_prob = y_pred.astype(float)

        metrics = compute_metrics(y, y_pred, y_prob)
        results[name] = metrics

        # Log no MLflow
        with mlflow.start_run(run_name=f"baseline_{name}", nested=True):
            mlflow.log_params({"model": name, "cv_folds": n_splits, "seed": GLOBAL_SEED})
            mlflow.log_metrics(metrics)

        logger.info("baseline_avaliado", modelo=name, **metrics)

    return results
