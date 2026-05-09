"""LightningModule — tinygrad-backed mirror of ``lightning.pytorch.LightningModule``.

Hook subset implemented in Phase 1:
- ``forward``, ``training_step``, ``validation_step``, ``test_step``
- ``configure_optimizers`` (optimizer | (opts, scheds) | dict — schedulers are
  consumed by the trainer in Phase 2; here only the optimizer side is exercised)
- ``self.log`` / ``self.log_dict`` with on_step/on_epoch reduction
- ``save_hyperparameters``
- ``parameters``, ``state_dict``, ``load_state_dict``

A ``LightningModule`` is wired to a ``Trainer`` via ``_attach_trainer``; until
that happens, ``self.log`` is a no-op (matching PL's "log called outside of
training" behavior).
"""
from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np


class AttributeDict(dict):
    """Dict that also exposes keys as attributes (PL parity)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _is_tensor(value: Any) -> bool:
    try:
        from tinygrad import Tensor
        return isinstance(value, Tensor)
    except ImportError:
        return False


def _realize_scalar(value: Any) -> float:
    """Coerce a logged value into a Python float — REALIZES tinygrad Tensors.

    Calling this on a Tensor that's part of an active autograd graph BEFORE
    ``backward()`` will break gradient flow under tinygrad 0.12. Use only
    after backward (or for already-realized values).
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.floating) or isinstance(value, np.integer):
        return float(value)
    if isinstance(value, np.ndarray):
        return float(value.mean())
    if _is_tensor(value):
        arr = value.detach().numpy()
        return float(arr.mean()) if arr.ndim else float(arr)
    raise TypeError(f"unsupported log value type: {type(value).__name__}")


class _MetricBuffer:
    """Per-name accumulator for a single stage (train/val/test).

    Values may be stored as raw tinygrad Tensors (still attached to the
    autograd graph) and only realized into floats lazily — see
    ``epoch_mean()`` and ``drain_step_value()``. Realizing a Tensor that is
    part of an active autograd graph before ``backward()`` breaks gradient
    flow under tinygrad 0.12, so the trainer always realizes values **after**
    backward has been called.

    ``on_step`` values are surfaced after each step; ``on_epoch`` values are
    accumulated and reduced (mean) at epoch end.
    """

    __slots__ = ("on_step", "on_epoch", "prog_bar", "logger", "step_value", "epoch_values")

    def __init__(self, *, on_step: bool, on_epoch: bool, prog_bar: bool, logger: bool):
        self.on_step = on_step
        self.on_epoch = on_epoch
        self.prog_bar = prog_bar
        self.logger = logger
        # ``step_value`` and entries of ``epoch_values`` may be either floats
        # OR un-realized tinygrad Tensors. They get realized at drain time.
        self.step_value: Any = None
        self.epoch_values: list = []

    def push(self, value: Any) -> None:
        self.step_value = value
        if self.on_epoch:
            self.epoch_values.append(value)

    def drain_step_value(self) -> float | None:
        if self.step_value is None:
            return None
        out = _realize_scalar(self.step_value)
        self.step_value = None
        return out

    def epoch_mean(self) -> float | None:
        if not self.epoch_values:
            return None
        scalars = [_realize_scalar(v) for v in self.epoch_values]
        return float(np.mean(scalars))

    def reset_epoch(self) -> None:
        self.epoch_values = []
        self.step_value = None


class LightningModule:
    """Base class for trainable models.

    Subclasses override ``training_step``/``validation_step``/``test_step``,
    ``configure_optimizers``, and ``forward``.
    """

    def __init__(self) -> None:
        self._trainer = None
        self._stage: str = "train"
        # stage -> name -> buffer
        self._metric_buffers: dict[str, dict[str, _MetricBuffer]] = defaultdict(dict)
        self.hparams = AttributeDict()

    # ---- training hooks -------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError("override forward()")

    def training_step(self, batch, batch_idx):  # pragma: no cover - abstract
        raise NotImplementedError("override training_step()")

    def validation_step(self, batch, batch_idx):  # noqa: D401
        return None

    def test_step(self, batch, batch_idx):  # noqa: D401
        return None

    def configure_optimizers(self):  # pragma: no cover - abstract
        raise NotImplementedError("override configure_optimizers()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ---- logging --------------------------------------------------------

    def log(
        self,
        name: str,
        value: Any,
        *,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        prog_bar: bool = False,
        logger: bool = True,
    ) -> None:
        """Record a metric. Defaults follow PL: training defaults to on_step=True/on_epoch=False; validation/test to on_step=False/on_epoch=True."""
        if self._trainer is None:
            return  # logging outside of trainer is a no-op
        if on_step is None:
            on_step = self._stage == "train"
        if on_epoch is None:
            on_epoch = self._stage in ("val", "test")
        # NOTE: do NOT realize ``value`` here — if it's a tinygrad Tensor in
        # an active autograd graph, realizing it before ``backward()`` will
        # break gradient flow. The buffer realizes lazily, after backward.
        buf = self._metric_buffers[self._stage].get(name)
        if buf is None:
            buf = _MetricBuffer(on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)
            self._metric_buffers[self._stage][name] = buf
        buf.push(value)

    def log_dict(self, dictionary: Mapping[str, Any], **kwargs) -> None:
        for k, v in dictionary.items():
            self.log(k, v, **kwargs)

    # ---- hparams --------------------------------------------------------

    def save_hyperparameters(self, *args, ignore: Iterable[str] | str | None = None) -> None:
        """Capture hyperparameters into ``self.hparams``.

        Forms supported:
        - ``save_hyperparameters()``: capture caller's locals (frame introspection)
        - ``save_hyperparameters(dict)``: capture given mapping
        - ``save_hyperparameters("lr", "batch_size")``: capture named locals
        """
        ignore_set: set[str] = set()
        if ignore is not None:
            if isinstance(ignore, str):
                ignore_set = {ignore}
            else:
                ignore_set = set(ignore)

        if args and len(args) == 1 and isinstance(args[0], Mapping):
            hp = dict(args[0])
        else:
            frame = inspect.currentframe().f_back
            local_vars = frame.f_locals if frame is not None else {}
            if args:
                hp = {k: local_vars[k] for k in args if k in local_vars}
            else:
                hp = {
                    k: v
                    for k, v in local_vars.items()
                    if k not in ("self", "__class__") and not k.startswith("_")
                }
        for k in ignore_set:
            hp.pop(k, None)
        for k, v in hp.items():
            self.hparams[k] = v

    # ---- parameters / state dict ---------------------------------------

    def parameters(self):
        from tinygrad.nn.state import get_parameters
        return get_parameters(self)

    def state_dict(self):
        from tinygrad.nn.state import get_state_dict
        return get_state_dict(self)

    def load_state_dict(self, state_dict, strict: bool = True):
        from tinygrad.nn.state import load_state_dict
        return load_state_dict(self, state_dict, strict=strict, verbose=False)

    # ---- trainer wiring (internal) -------------------------------------

    def _attach_trainer(self, trainer) -> None:
        self._trainer = trainer

    def _set_stage(self, stage: str) -> None:
        assert stage in ("train", "val", "test"), stage
        self._stage = stage

    def _drain_step_metrics(self, stage: str) -> dict[str, float]:
        """Return on_step metrics recorded during the most recent step.

        Tensors stored in the buffer are realized to floats here. The trainer
        calls this AFTER ``backward()`` so realization doesn't truncate the
        autograd graph.
        """
        out = {}
        for name, buf in self._metric_buffers[stage].items():
            if not buf.on_step:
                continue
            value = buf.drain_step_value()
            if value is not None:
                out[name] = value
        return out

    def _drain_epoch_metrics(self, stage: str) -> dict[str, float]:
        """Reduce and return on_epoch metrics; resets the buffer."""
        out = {}
        buffers = self._metric_buffers.get(stage, {})
        for name, buf in buffers.items():
            if buf.on_epoch:
                mean = buf.epoch_mean()
                if mean is not None:
                    out[name] = mean
            buf.reset_epoch()
        return out

    def _prog_bar_names(self, stage: str) -> set[str]:
        return {n for n, b in self._metric_buffers[stage].items() if b.prog_bar}

    @property
    def trainer(self):
        return self._trainer
