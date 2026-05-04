"""Script principal de treinamento: baselines + MLP com tracking MLflow."""

from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data, preprocess_raw_data
from src.features.pipeline import build_preprocessor, prepare_features
from src.models.baseline import compute_metrics, train_baselines
from src.models.mlp import predict_mlp, train_mlp
from src.utils.logging import get_logger, setup_logging
from src.utils.reproducibility import GLOBAL_SEED, set_seeds

logger = get_logger(__name__)

MODELS_DIR = Path("models")
EXPERIMENT_NAME = "churn-prediction"


def run_training() -> None:
    """Executa pipeline completo de treinamento."""
    setup_logging()
    set_seeds(GLOBAL_SEED)

    logger.info("pipeline_iniciado")

    # 1. Carregar e preprocessar dados
    df = load_raw_data()
    df = preprocess_raw_data(df)
    X, y = prepare_features(df)

    # 2. Split estratificado treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=y
    )

    # 3. Split treino/validação para MLP
    X_train_mlp, X_val, y_train_mlp, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=GLOBAL_SEED, stratify=y_train
    )

    logger.info(
        "splits_criados",
        train=len(X_train),
        val=len(X_val),
        test=len(X_test),
    )

    # 4. Configurar MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full_pipeline"):
        mlflow.log_params(
            {
                "seed": GLOBAL_SEED,
                "test_size": 0.2,
                "val_size": 0.15,
                "dataset_size": len(df),
                "churn_rate": f"{y.mean():.4f}",
                "n_features": X.shape[1],
            }
        )

        # 5. Treinar baselines
        logger.info("treinando_baselines")
        baseline_results = train_baselines(X_train, y_train.values)

        # 6. Preprocessar dados para MLP
        preprocessor = build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train_mlp)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        # Salvar preprocessor
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
        mlflow.log_artifact(str(MODELS_DIR / "preprocessor.joblib"))

        # 7. Calcular peso da classe positiva para lidar com desbalanceamento
        n_pos = y_train_mlp.sum()
        n_neg = len(y_train_mlp) - n_pos
        class_weight = n_neg / n_pos

        # 8. Treinar MLP
        logger.info("treinando_mlp")
        with mlflow.start_run(run_name="mlp_churn", nested=True):
            mlp_params = {
                "model": "MLP",
                "architecture": "128-64-32-1",
                "epochs_max": 200,
                "batch_size": 64,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "patience": 15,
                "class_weight": f"{class_weight:.2f}",
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
            }
            mlflow.log_params(mlp_params)

            model, history = train_mlp(
                X_train_processed,
                y_train_mlp.values.astype(np.float32),
                X_val_processed,
                y_val.values.astype(np.float32),
                epochs=200,
                batch_size=64,
                learning_rate=1e-3,
                weight_decay=1e-4,
                patience=15,
                class_weight=class_weight,
            )

            # Avaliar no conjunto de teste
            y_pred, y_prob = predict_mlp(model, X_test_processed)
            mlp_metrics = compute_metrics(y_test.values, y_pred, y_prob)

            mlflow.log_metrics(mlp_metrics)
            mlflow.log_metric("epochs_trained", history["epochs_trained"])

            # Salvar modelo
            torch.save(model.state_dict(), MODELS_DIR / "mlp_churn.pth")
            mlflow.log_artifact(str(MODELS_DIR / "mlp_churn.pth"))

            # Salvar metadados do modelo
            model_metadata = {
                "input_dim": X_train_processed.shape[1],
                "architecture": "128-64-32-1",
                "threshold": 0.5,
            }
            joblib.dump(model_metadata, MODELS_DIR / "model_metadata.joblib")
            mlflow.log_artifact(str(MODELS_DIR / "model_metadata.joblib"))

            logger.info("mlp_avaliado", **mlp_metrics)

        # 9. Tabela comparativa
        logger.info("=== RESULTADOS COMPARATIVOS ===")
        all_results = {**baseline_results, "MLP_PyTorch": mlp_metrics}
        for name, metrics in all_results.items():
            logger.info(
                "resultado_modelo",
                modelo=name,
                accuracy=f"{metrics['accuracy']:.4f}",
                f1=f"{metrics['f1']:.4f}",
                roc_auc=f"{metrics['roc_auc']:.4f}",
            )

    logger.info("pipeline_concluido")


if __name__ == "__main__":
    run_training()
