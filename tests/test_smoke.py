"""Smoke tests: verificam que os componentes principais funcionam."""

import numpy as np
import pytest
import torch
from src.models.mlp import ChurnMLP, EarlyStopping, predict_mlp


@pytest.mark.smoke
class TestModelSmoke:
    """Testes de fumaça para o modelo MLP."""

    def test_mlp_forward_pass(self):
        """Verifica que o modelo faz forward pass sem erro."""
        input_dim = 46
        model = ChurnMLP(input_dim=input_dim)
        X = torch.randn(10, input_dim)

        output = model(X)

        assert output.shape == (10, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_mlp_predict(self):
        """Verifica que predict_mlp retorna arrays válidos."""
        input_dim = 46
        model = ChurnMLP(input_dim=input_dim)
        X = np.random.randn(5, input_dim).astype(np.float32)

        predictions, probabilities = predict_mlp(model, X)

        assert predictions.shape == (5,)
        assert probabilities.shape == (5,)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_early_stopping_triggers(self):
        """Verifica que early stopping é acionado após paciência excedida."""
        es = EarlyStopping(patience=3)

        # Loss melhorando
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)

        # Loss piorando
        assert not es(0.9)
        assert not es(0.95)
        assert es(1.0)  # Deve parar

    def test_early_stopping_resets_on_improvement(self):
        """Verifica que o counter reseta quando há melhora."""
        es = EarlyStopping(patience=3)

        es(1.0)
        es(1.1)  # Pior - counter = 1
        es(1.2)  # Pior - counter = 2
        es(0.5)  # Melhor - counter reseta

        assert not es.should_stop
        assert es.counter == 0
