"""LightningOptimizer — wrap a tinygrad optimizer with a PL-shaped param-group view.

Tinygrad's ``Optimizer`` takes a flat ``list[Tensor]`` and exposes a single
``lr`` (a mutable ``Tensor`` scalar). PL expresses per-module learning rates
through *param groups*:

    optim = SGD([
        {"params": encoder_params, "lr": 1e-5},
        {"params": head_params,    "lr": 1e-2},
    ])

This module provides:

- ``LightningOptimizer``: a wrapper that exposes ``param_groups`` over either
  a single tinygrad optimizer or a tinygrad ``OptimizerGroup``. Schedulers and
  ``LearningRateMonitor`` read/write LRs through this surface.

- ``param_group_optimizer``: a factory that takes a list of param-group dicts
  and produces an ``OptimizerGroup`` (one tinygrad optimizer per group, each
  with its own ``lr`` Tensor).
"""
from __future__ import annotations

from typing import Any, Callable


def _to_float(value: Any) -> float:
    """Coerce a tinygrad LR Tensor (or plain number) to a Python float."""
    if hasattr(value, "numpy"):
        arr = value.numpy()
        return float(arr.flat[0]) if hasattr(arr, "flat") else float(arr)
    return float(value)


def param_group_optimizer(optimizer_cls: Callable, param_groups: list[dict], **defaults):
    """Build a tinygrad ``OptimizerGroup`` from a list of ``{"params": ..., "lr": ...}`` dicts.

    Each dict produces one underlying tinygrad optimizer. Per-group keys (e.g.
    ``lr``, ``weight_decay``, ``momentum``) override the ``defaults``.
    """
    from tinygrad.nn.optim import OptimizerGroup

    if not param_groups:
        raise ValueError("param_groups must be non-empty")

    sub_opts = []
    for i, group in enumerate(param_groups):
        if "params" not in group:
            raise ValueError(f"param group {i} missing 'params' key")
        kwargs = dict(defaults)
        for k, v in group.items():
            if k != "params":
                kwargs[k] = v
        sub_opts.append(optimizer_cls(group["params"], **kwargs))

    if len(sub_opts) == 1:
        return sub_opts[0]
    return OptimizerGroup(*sub_opts)


class LightningOptimizer:
    """Wrap a tinygrad optimizer (or ``OptimizerGroup``) and expose ``param_groups``.

    The wrapper does not own state — calls to ``step``/``zero_grad`` delegate
    to the wrapped optimizer. The ``param_groups`` attribute is a list of
    ``{"params": list[Tensor], "lr": Tensor (the underlying mutable scalar),
    "_optim": <sub-optimizer>}`` dicts that schedulers and the LR monitor read
    and mutate.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        sub_opts = self._sub_optimizers(optimizer)
        self.param_groups: list[dict] = [
            {"params": list(o.params), "lr": o.lr, "_optim": o}
            for o in sub_opts
        ]

    @staticmethod
    def _sub_optimizers(optimizer) -> list:
        """Return the list of underlying single-LR optimizers."""
        if isinstance(optimizer, LightningOptimizer):
            return LightningOptimizer._sub_optimizers(optimizer.optimizer)
        if hasattr(optimizer, "optimizers"):  # tinygrad OptimizerGroup
            return list(optimizer.optimizers)
        return [optimizer]

    @property
    def params(self):
        return [p for g in self.param_groups for p in g["params"]]

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def get_lrs(self) -> list[float]:
        return [_to_float(g["lr"]) for g in self.param_groups]

    def set_lrs(self, lrs: list[float]) -> None:
        from tinygrad import Tensor

        if len(lrs) != len(self.param_groups):
            raise ValueError(f"expected {len(self.param_groups)} lrs, got {len(lrs)}")
        for group, new_lr in zip(self.param_groups, lrs):
            opt = group["_optim"]
            opt.lr.assign(Tensor([new_lr], device=opt.lr.device, dtype=opt.lr.dtype))
