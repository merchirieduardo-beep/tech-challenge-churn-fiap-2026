"""Rede Neural MLP com PyTorch para classificação de churn."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logging import get_logger
from src.utils.reproducibility import GLOBAL_SEED, set_seeds

logger = get_logger(__name__)


class ChurnMLP(nn.Module):
    """Multi-Layer Perceptron para previsão de churn.

    Arquitetura:
        - Input → Hidden1 (128) → BatchNorm → ReLU → Dropout(0.3)
        - Hidden1 → Hidden2 (64) → BatchNorm → ReLU → Dropout(0.2)
        - Hidden2 → Hidden3 (32) → BatchNorm → ReLU → Dropout(0.1)
        - Hidden3 → Output (1) → Sigmoid
    """

    def __init__(self, input_dim: int, dropout_rates: tuple = (0.3, 0.2, 0.1)):
        super().__init__()

        self.network = nn.Sequential(
            # Camada 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            # Camada 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            # Camada 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            # Output
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Inicialização Xavier
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inicializa pesos com Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EarlyStopping:
    """Implementa early stopping para evitar overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Cria DataLoaders para treino e validação."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1),
    )

    generator = torch.Generator().manual_seed(GLOBAL_SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    class_weight: float | None = None,
) -> tuple[ChurnMLP, dict]:
    """Treina a MLP com early stopping.

    Args:
        X_train: Features de treino (já processadas).
        y_train: Target de treino.
        X_val: Features de validação.
        y_val: Target de validação.
        epochs: Número máximo de épocas.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado.
        weight_decay: Regularização L2.
        patience: Paciência para early stopping.
        class_weight: Peso da classe positiva (para desbalanceamento).

    Returns:
        Tupla (modelo treinado, histórico de treinamento).
    """
    set_seeds(GLOBAL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("dispositivo_selecionado", device=str(device))

    input_dim = X_train.shape[1]
    model = ChurnMLP(input_dim=input_dim).to(device)

    # Loss com peso para lidar com desbalanceamento
    if class_weight is not None:
        pos_weight = torch.tensor([class_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Remover sigmoid da última camada quando usando BCEWithLogitsLoss
        model.network[-1] = nn.Identity()
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience)

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    history = {"train_loss": [], "val_loss": [], "epochs_trained": 0}

    for epoch in range(epochs):
        # Treino
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validação
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(
                "epoch_concluida",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
            )

        if early_stopping(val_loss):
            logger.info("early_stopping_ativado", epoch=epoch + 1)
            break

    history["epochs_trained"] = epoch + 1
    logger.info("treinamento_concluido", epochs_treinadas=history["epochs_trained"])

    return model, history


def predict_mlp(
    model: ChurnMLP,
    X: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera predições com o modelo MLP.

    Returns:
        Tupla (classes preditas, probabilidades).
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        probabilities = model(X_tensor).cpu().numpy().flatten()

    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
