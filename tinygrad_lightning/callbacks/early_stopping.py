"""EarlyStopping — halt training when a monitored metric stagnates.

Mirrors the configuration of ``lightning.pytorch.callbacks.EarlyStopping``:

    EarlyStopping(monitor="val_loss", patience=3, mode="min", min_delta=0.0)

Each ``on_validation_epoch_end`` (or ``on_train_epoch_end`` if ``check_on_train_epoch_end``):
    - if metric improved by > ``min_delta``, reset the patience counter
    - else increment; once it exceeds ``patience``, set ``trainer._should_stop = True``
"""
from __future__ import annotations

import math

from .base import Callback


_MODE_OP = {
    "min": (lambda new, best, delta: new < best - delta, math.inf, -1),
    "max": (lambda new, best, delta: new > best + delta, -math.inf, 1),
}


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str,
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        check_on_train_epoch_end: bool = False,
        verbose: bool = False,
    ) -> None:
        if mode not in _MODE_OP:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.check_on_train_epoch_end = bool(check_on_train_epoch_end)
        self.verbose = bool(verbose)
        cmp, init, _ = _MODE_OP[mode]
        self._cmp = cmp
        self.best_score: float = init
        self.wait_count: int = 0
        self.stopped_epoch: int | None = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.check_on_train_epoch_end:
            self._check(trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.check_on_train_epoch_end:
            self._check(trainer)

    def _check(self, trainer) -> None:
        if self.monitor not in trainer.callback_metrics:
            return
        current = float(trainer.callback_metrics[self.monitor])
        if self._cmp(current, self.best_score, self.min_delta):
            self.best_score = current
            self.wait_count = 0
            return
        self.wait_count += 1
        if self.wait_count > self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer._should_stop = True
            if self.verbose:
                from ..utilities.rank_zero import rank_zero_info
                rank_zero_info(
                    f"EarlyStopping: monitor={self.monitor!r} did not improve in "
                    f"{self.patience + 1} checks; best={self.best_score:.4f}"
                )
