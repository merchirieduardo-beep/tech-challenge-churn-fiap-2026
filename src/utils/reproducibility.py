"""Utilidades para garantir reprodutibilidade dos experimentos."""

import random

import numpy as np
import torch

GLOBAL_SEED = 42


def set_seeds(seed: int = GLOBAL_SEED) -> None:
    """Fixa seeds para reprodutibilidade completa."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
