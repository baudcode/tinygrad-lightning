"""Seed all RNGs we touch.

Mirrors ``lightning.pytorch.seed_everything`` semantics but only seeds the
sources tinygrad-lightning actually uses: Python ``random``, NumPy,
``PYTHONHASHSEED``, and tinygrad's ``Tensor.manual_seed``.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int, workers: bool = False) -> int:
    """Seed Python, NumPy and tinygrad. Returns the seed used.

    ``workers`` is accepted for PL signature compatibility; tinygrad-lightning's
    DataLoader does not currently spawn workers that need separate seeds.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        from tinygrad import Tensor
        Tensor.manual_seed(seed)
    except ImportError:
        pass
    return seed
