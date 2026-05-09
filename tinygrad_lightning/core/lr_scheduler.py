"""LR schedulers — mirror torch.optim.lr_scheduler signatures and curves.

A scheduler holds a reference to an optimizer (or LightningOptimizer) and
computes a per-param-group LR from a step counter. ``step()`` advances the
counter and writes the new LRs into the optimizer's ``lr`` tensors.

Schedulers don't care whether the underlying optimizer has 1 or N param groups;
the base class introspects via ``LightningOptimizer.param_groups`` (which itself
flattens an ``OptimizerGroup``).
"""
from __future__ import annotations

import math
from typing import Callable

from .optimizer import LightningOptimizer, _to_float


def _wrap(optimizer) -> LightningOptimizer:
    return optimizer if isinstance(optimizer, LightningOptimizer) else LightningOptimizer(optimizer)


class LRScheduler:
    """Base class. Subclasses override ``get_lr()`` to return a list of floats
    (one per param group). ``step()`` advances the counter and applies them."""

    def __init__(self, optimizer):
        self.optimizer = _wrap(optimizer)
        self.base_lrs: list[float] = [_to_float(g["lr"]) for g in self.optimizer.param_groups]
        self._step_count: int = 0
        self.last_lr: list[float] = list(self.base_lrs)

    def get_lr(self) -> list[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self) -> None:
        self._step_count += 1
        new_lrs = self.get_lr()
        self.optimizer.set_lrs(new_lrs)
        self.last_lr = new_lrs

    def state_dict(self) -> dict:
        return {"step_count": self._step_count, "base_lrs": list(self.base_lrs), "last_lr": list(self.last_lr)}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = int(state["step_count"])
        self.base_lrs = list(state["base_lrs"])
        self.last_lr = list(state.get("last_lr", self.base_lrs))


class StepLR(LRScheduler):
    """Decay LR by ``gamma`` every ``step_size`` scheduler steps."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        n_decays = self._step_count // self.step_size
        factor = self.gamma ** n_decays
        return [base * factor for base in self.base_lrs]


class MultiStepLR(LRScheduler):
    """Decay LR by ``gamma`` once per milestone reached."""

    def __init__(self, optimizer, milestones: list[int], gamma: float = 0.1):
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        n_decays = sum(1 for m in self.milestones if self._step_count >= m)
        factor = self.gamma ** n_decays
        return [base * factor for base in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing from base_lr to ``eta_min`` over ``T_max`` steps."""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        t = min(self._step_count, self.T_max)
        cos_factor = (1 + math.cos(math.pi * t / self.T_max)) / 2
        return [self.eta_min + (base - self.eta_min) * cos_factor for base in self.base_lrs]


class LambdaLR(LRScheduler):
    """Multiply each base_lr by ``lr_lambda(step)`` (callable or list-per-group)."""

    def __init__(self, optimizer, lr_lambda: Callable[[int], float] | list[Callable[[int], float]]):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer)
        if isinstance(self.lr_lambda, (list, tuple)):
            if len(self.lr_lambda) != len(self.base_lrs):
                raise ValueError(
                    f"lr_lambda list length {len(self.lr_lambda)} != param groups {len(self.base_lrs)}"
                )

    def get_lr(self) -> list[float]:
        if callable(self.lr_lambda):
            f = self.lr_lambda(self._step_count)
            return [base * f for base in self.base_lrs]
        return [base * fn(self._step_count) for base, fn in zip(self.base_lrs, self.lr_lambda)]


class OneCycleLR(LRScheduler):
    """One-cycle policy: linear warmup to ``max_lr``, then cosine anneal to ``max_lr / final_div_factor``.

    Mirrors the ``cos`` annealing strategy used by torch's ``OneCycleLR`` (default).
    """

    def __init__(
        self,
        optimizer,
        max_lr: float | list[float],
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 < pct_start < 1.0:
            raise ValueError("pct_start must be in (0, 1)")
        wrapped = _wrap(optimizer)
        n_groups = len(wrapped.param_groups)
        if isinstance(max_lr, (int, float)):
            self.max_lrs = [float(max_lr)] * n_groups
        else:
            self.max_lrs = [float(x) for x in max_lr]
            if len(self.max_lrs) != n_groups:
                raise ValueError(f"max_lr list length {len(self.max_lrs)} != param groups {n_groups}")

        self.total_steps = int(total_steps)
        self.pct_start = float(pct_start)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self._warmup_end = max(int(round(self.total_steps * self.pct_start)), 1)

        # Override base_lrs so the policy starts from max_lr / div_factor.
        initial_lrs = [m / self.div_factor for m in self.max_lrs]
        wrapped.set_lrs(initial_lrs)
        super().__init__(wrapped)
        self.base_lrs = list(initial_lrs)

    def get_lr(self) -> list[float]:
        step = self._step_count
        out = []
        for max_lr, initial in zip(self.max_lrs, self.base_lrs):
            final = max_lr / self.final_div_factor
            if step <= self._warmup_end:
                pct = step / self._warmup_end
                lr = initial + (max_lr - initial) * pct
            else:
                anneal_total = max(self.total_steps - self._warmup_end, 1)
                pct = min((step - self._warmup_end) / anneal_total, 1.0)
                lr = final + (max_lr - final) * (1 + math.cos(math.pi * pct)) / 2
            out.append(lr)
        return out
