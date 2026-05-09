"""Unit: EarlyStopping triggers ``trainer._should_stop`` after patience exceeded."""
from __future__ import annotations

from tinygrad_lightning.callbacks.early_stopping import EarlyStopping


class _StubTrainer:
    def __init__(self):
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch: int = 0
        self._should_stop: bool = False


def _emit(es: EarlyStopping, trainer: _StubTrainer, value: float) -> None:
    trainer.callback_metrics["val_loss"] = value
    es.on_validation_epoch_end(trainer, None)
    trainer.current_epoch += 1


def test_min_mode_stops_after_patience_without_improvement():
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    trainer = _StubTrainer()
    _emit(es, trainer, 1.0)   # initial best
    _emit(es, trainer, 1.1)   # wait=1
    _emit(es, trainer, 1.2)   # wait=2 (still <= patience)
    assert not trainer._should_stop, "stopped too early"
    _emit(es, trainer, 1.3)   # wait=3 > patience=2 → stop
    assert trainer._should_stop


def test_min_mode_resets_on_improvement():
    es = EarlyStopping(monitor="val_loss", patience=1, mode="min")
    trainer = _StubTrainer()
    _emit(es, trainer, 1.0)
    _emit(es, trainer, 1.1)   # wait=1
    _emit(es, trainer, 0.9)   # improvement → wait=0
    _emit(es, trainer, 1.0)   # wait=1
    assert not trainer._should_stop
    _emit(es, trainer, 1.0)   # wait=2 > patience=1 → stop
    assert trainer._should_stop


def test_max_mode_stops_when_metric_does_not_grow():
    es = EarlyStopping(monitor="acc", patience=1, mode="max")
    trainer = _StubTrainer()
    trainer.callback_metrics["acc"] = 0.5
    es.on_validation_epoch_end(trainer, None)
    trainer.callback_metrics["acc"] = 0.4
    es.on_validation_epoch_end(trainer, None)  # wait=1
    trainer.callback_metrics["acc"] = 0.4
    es.on_validation_epoch_end(trainer, None)  # wait=2 > 1
    assert trainer._should_stop


def test_min_delta_requires_meaningful_improvement():
    es = EarlyStopping(monitor="val_loss", patience=0, mode="min", min_delta=0.05)
    trainer = _StubTrainer()
    _emit(es, trainer, 1.00)
    _emit(es, trainer, 0.99)   # improvement < min_delta → wait=1
    assert trainer._should_stop


def test_missing_monitor_does_not_stop():
    es = EarlyStopping(monitor="not_logged", patience=0, mode="min")
    trainer = _StubTrainer()
    es.on_validation_epoch_end(trainer, None)
    assert not trainer._should_stop
