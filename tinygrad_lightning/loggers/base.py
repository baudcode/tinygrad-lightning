"""Logger ABC — minimum surface to mirror ``lightning.pytorch.loggers.Logger``.

A logger is responsible for persisting metrics and hyperparameters to some
external store (CSV file, TensorBoard event log, MLflow run, ...). The
``Trainer`` calls these hooks in this order:

    1. ``log_hyperparams(model.hparams)`` once at fit start
    2. ``log_metrics(metrics, step)`` whenever ``trainer.callback_metrics`` advances
    3. ``save()`` opportunistically (e.g. after each epoch)
    4. ``finalize(status)`` once at fit end

Subclasses must implement ``log_metrics`` and ``log_hyperparams``; ``save`` and
``finalize`` default to no-ops.
"""
from __future__ import annotations

from collections.abc import Mapping


class Logger:
    @property
    def name(self) -> str:  # pragma: no cover - subclasses override
        return self.__class__.__name__

    @property
    def version(self) -> int | str:  # pragma: no cover
        return 0

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        raise NotImplementedError

    def log_hyperparams(self, params: Mapping[str, object]) -> None:
        raise NotImplementedError

    def save(self) -> None:
        return None

    def finalize(self, status: str) -> None:
        return None
