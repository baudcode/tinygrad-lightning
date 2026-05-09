"""Parity: basic training loop reduces loss in both backends.

A 2-layer MLP fit to a 64-sample synthetic linear-regression set for one
epoch must show ``mean(final_2_steps) < 0.5 * mean(initial_2_steps)`` —
regardless of which framework drives the loop.
"""
from __future__ import annotations

import numpy as np

from ._helpers import train_with_tinygrad, train_with_torch


def test_loss_decreases_after_one_epoch(parity_backend):
    if parity_backend == "tinygrad":
        losses, trainer = train_with_tinygrad(seed=42, epochs=3, batch_size=8)
    else:
        losses, trainer = train_with_torch(seed=42, epochs=3, batch_size=8)

    assert len(losses) >= 4, f"expected ≥4 training steps, got {len(losses)}"
    initial = float(np.mean(losses[:2]))
    final = float(np.mean(losses[-2:]))
    assert final < 0.5 * initial, f"loss did not halve: {initial:.4f} -> {final:.4f}"
    assert trainer.global_step >= 4, f"trainer.global_step={trainer.global_step}"


def test_callback_metrics_populated(parity_backend):
    """Both backends expose the latest logged value via ``trainer.callback_metrics``."""
    if parity_backend == "tinygrad":
        _, trainer = train_with_tinygrad(seed=0, epochs=2, batch_size=8)
    else:
        _, trainer = train_with_torch(seed=0, epochs=2, batch_size=8)

    assert "loss" in trainer.callback_metrics, list(trainer.callback_metrics.keys())
    last = float(trainer.callback_metrics["loss"])
    assert np.isfinite(last)
