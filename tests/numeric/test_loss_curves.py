"""Numeric regression: seed-pinned loss thresholds for tinygrad-lightning only.

These are guard-rails against silent backend drift. They are NOT compared to
PyTorch Lightning — they pin tinygrad-lightning to its own past behavior.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_mlp_regression_loss_below_threshold():
    """Seeded 1-epoch MLP must reach a loss substantially below the initial loss."""
    from tests.parity._helpers import train_with_tinygrad

    losses, _ = train_with_tinygrad(seed=42, epochs=4, batch_size=8)
    final = float(np.mean(losses[-4:]))
    initial = float(np.mean(losses[:4]))
    # Pinned threshold: 4-epoch run must reach at least an 80% loss reduction.
    assert final < 0.2 * initial, f"final loss {final:.3f} not below 0.2 * initial {initial:.3f}"


def test_loss_monotonic_on_average():
    """The mean of the second half of an epoch must be lower than the mean of the first half."""
    from tests.parity._helpers import train_with_tinygrad

    losses, _ = train_with_tinygrad(seed=42, epochs=4, batch_size=8)
    half = len(losses) // 2
    assert half >= 4, f"too few steps: {len(losses)}"
    first = float(np.mean(losses[:half]))
    second = float(np.mean(losses[half:]))
    assert second < first, f"second-half mean {second:.3f} not below first-half mean {first:.3f}"
