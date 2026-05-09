"""LearningRateMonitor — record per-param-group LRs into ``trainer.callback_metrics``.

Mirrors ``lightning.pytorch.callbacks.LearningRateMonitor`` (the subset we need).
LR keys land as ``lr-0``, ``lr-1``, ... reflecting the ordering of param groups
on the optimizer.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .base import Callback


class LearningRateMonitor(Callback):
    def __init__(self, logging_interval: str | None = "step") -> None:
        if logging_interval not in (None, "step", "epoch"):
            raise ValueError(f"logging_interval must be None|'step'|'epoch', got {logging_interval!r}")
        self.logging_interval = logging_interval
        self.lrs: dict[str, list[float]] = defaultdict(list)

    def _record(self, trainer) -> None:
        for opt_idx, opt in enumerate(trainer.optimizers):
            lrs = opt.get_lrs()
            for grp_idx, lr in enumerate(lrs):
                key = f"lr-{grp_idx}" if len(trainer.optimizers) == 1 else f"lr-{opt_idx}/pg{grp_idx}"
                trainer.callback_metrics[key] = lr
                trainer.logged_metrics[key] = lr
                self.lrs[key].append(lr)

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.logging_interval == "step":
            self._record(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.logging_interval in ("epoch", None):
            self._record(trainer)
