"""PrecisionPlugin — drives loss-scaling, grad-finite checks, and autocast for mixed precision.

What this plugin owns:

- ``autocast(module)``       — context manager that swaps every Tensor
                               reachable from ``module.__dict__`` for a
                               cast-to-compute-dtype copy. Tinygrad has no
                               native autocast; we approximate it by mutating
                               module attributes for the duration of the
                               forward pass and relying on ``Tensor.cast``
                               being differentiable so backward routes grads
                               back to the original fp32 leaves.
- ``scale_loss(loss)``       — promote loss to fp32 then multiply by the
                               dynamic scale (fp16 only)
- ``unscale_grads(opts)``    — divide each parameter's ``.grad`` by the scale
- ``check_grads_finite(opts)`` — return ``False`` if any grad is NaN/Inf
- ``update_scaler(found_inf)`` — DynamicLossScaler growth/backoff bookkeeping

For ``"32-true"`` (the default) every method is a no-op. For ``"bf16-mixed"``
the dynamic loss scaler is disabled (bf16 has the same dynamic range as fp32).

The dynamic loss scaler matches torch's ``torch.amp.GradScaler`` defaults:

    init_scale=2**16, growth_factor=2, backoff_factor=0.5, growth_interval=2000
"""
from __future__ import annotations

import contextlib
from typing import Iterable, Sequence

import numpy as np


class DynamicLossScaler:
    """Mirror of ``torch.amp.GradScaler``'s growth/backoff scheme."""

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        if init_scale <= 0:
            raise ValueError("init_scale must be positive")
        if not 0 < backoff_factor < 1:
            raise ValueError("backoff_factor must be in (0, 1)")
        if growth_factor <= 1:
            raise ValueError("growth_factor must be > 1")
        if growth_interval < 1:
            raise ValueError("growth_interval must be >= 1")
        self.scale: float = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._growth_counter: int = 0

    def update(self, found_inf: bool) -> None:
        if found_inf:
            self.scale *= self.backoff_factor
            self._growth_counter = 0
            return
        self._growth_counter += 1
        if self._growth_counter >= self.growth_interval:
            self.scale *= self.growth_factor
            self._growth_counter = 0

    def state_dict(self) -> dict:
        return {"scale": self.scale, "growth_counter": self._growth_counter}

    def load_state_dict(self, state: dict) -> None:
        self.scale = float(state["scale"])
        self._growth_counter = int(state.get("growth_counter", 0))


_VALID_PRECISIONS = ("32-true", "16-mixed", "bf16-mixed")


class PrecisionPlugin:
    def __init__(self, precision: str = "32-true") -> None:
        if precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"unsupported precision: {precision!r}; expected one of {_VALID_PRECISIONS}"
            )
        self.precision = precision
        self.use_loss_scaling = precision == "16-mixed"
        self.scaler: DynamicLossScaler | None = (
            DynamicLossScaler() if self.use_loss_scaling else None
        )

    @property
    def compute_dtype(self):
        from tinygrad import dtypes

        if self.precision == "16-mixed":
            return dtypes.float16
        if self.precision == "bf16-mixed":
            return dtypes.bfloat16
        return dtypes.float32

    @property
    def loss_scale(self) -> float:
        return self.scaler.scale if self.scaler else 1.0

    def autocast(self, module):
        """Return a context manager that casts module Tensors to ``compute_dtype``.

        For ``"32-true"`` returns a no-op context. For ``"16-mixed"`` /
        ``"bf16-mixed"`` swaps every ``Tensor`` reachable from
        ``module.__dict__`` (recursively, including lists/dicts and child
        objects) for a ``.cast(compute_dtype)`` copy. Originals are restored on
        exit. The autograd graph still references the cast tensors after exit,
        and ``Tensor.cast`` is differentiable, so backward correctly routes
        gradients to the original fp32 leaves.

        Notes:
        - Attributes whose names start with ``_`` are skipped (LightningModule
          internals like ``_trainer``, ``_metric_buffers``, etc.).
        - Items inside *tuples* are not swapped (tuples are immutable). Lists
          and dicts are.
        """
        if self.precision == "32-true":
            return contextlib.nullcontext()
        return _AutocastContext(module, self.compute_dtype)

    def scale_loss(self, loss):
        """Return ``loss * scale`` (or ``loss`` unchanged when no scaling).

        The loss is first cast to fp32 to avoid overflow when the dynamic
        scale (default 2¹⁶) multiplies an fp16 tensor whose max is 65504.
        """
        if self.scaler is None:
            return loss
        from tinygrad import dtypes

        loss_fp32 = loss.cast(dtypes.float32) if loss.dtype != dtypes.float32 else loss
        return loss_fp32 * self.scaler.scale

    def unscale_grads(self, optimizers: Sequence) -> None:
        """Divide each parameter's ``.grad`` by the current scale (in place)."""
        if self.scaler is None or self.scaler.scale == 1.0:
            return
        scale = self.scaler.scale
        seen_ids: set[int] = set()
        for opt in optimizers:
            for p in opt.params:
                if id(p) in seen_ids:
                    continue
                seen_ids.add(id(p))
                if getattr(p, "grad", None) is None:
                    continue
                p.grad = p.grad * (1.0 / scale)

    def check_grads_finite(self, optimizers: Sequence) -> bool:
        """Return ``False`` if any grad in ``optimizers`` is NaN or Inf."""
        if self.precision == "32-true" or self.precision == "bf16-mixed":
            return True
        seen_ids: set[int] = set()
        for opt in optimizers:
            for p in opt.params:
                if id(p) in seen_ids:
                    continue
                seen_ids.add(id(p))
                if getattr(p, "grad", None) is None:
                    continue
                arr = p.grad.detach().numpy()
                if not np.all(np.isfinite(arr)):
                    return False
        return True

    def update_scaler(self, *, found_inf: bool) -> None:
        if self.scaler is not None:
            self.scaler.update(found_inf)

    def state_dict(self) -> dict:
        return {"precision": self.precision, "scaler": self.scaler.state_dict() if self.scaler else None}

    def load_state_dict(self, state: dict) -> None:
        if state.get("precision") != self.precision:
            raise RuntimeError(
                f"precision mismatch on resume: ckpt has {state.get('precision')!r}, "
                f"current trainer has {self.precision!r}"
            )
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])


# ---- autocast context implementation ----------------------------------------


class _AutocastContext:
    """Swap every Tensor reachable from ``module.__dict__`` for a ``.cast(dtype)`` copy.

    Used by ``PrecisionPlugin.autocast``; not part of the public API.
    """

    def __init__(self, module, dtype) -> None:
        self.module = module
        self.dtype = dtype
        self._saved: list[tuple[object, object, object]] = []  # (parent, key, original)

    def __enter__(self):
        for parent, key, original in _walk_module_tensors(self.module):
            # Skip buffers (requires_grad=False) — e.g. BatchNorm running_mean /
            # running_var / num_batches_tracked. Casting these to fp16 breaks
            # the in-place ``running_mean.assign(...)`` on the next forward
            # because tinygrad's BN computes batch_mean in fp32 internally.
            # Also skip integer tensors (counters, indices).
            from tinygrad import dtypes
            if original.requires_grad is False:
                continue
            if original.dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64):
                continue
            cast_t = original.cast(self.dtype)
            self._saved.append((parent, key, original))
            _setitem(parent, key, cast_t)
        return self

    def __exit__(self, exc_type, exc, tb):
        for parent, key, original in self._saved:
            _setitem(parent, key, original)
        self._saved = []
        return False


def _walk_module_tensors(obj, visited: set | None = None) -> Iterable[tuple[object, object, object]]:
    """Yield ``(parent, key, tensor)`` for every Tensor reachable from ``obj``.

    ``parent`` is the container the tensor is attached to; ``key`` is the
    attribute name (for objects), index (for lists), or key (for dicts).
    Tuples are skipped because they're immutable.
    """
    from tinygrad import Tensor

    if visited is None:
        visited = set()
    oid = id(obj)
    if oid in visited:
        return
    visited.add(oid)

    if isinstance(obj, list):
        for i, val in enumerate(obj):
            if isinstance(val, Tensor):
                yield obj, i, val
            elif isinstance(val, (list, dict)) or hasattr(val, "__dict__"):
                yield from _walk_module_tensors(val, visited)
        return
    if isinstance(obj, dict):
        for k, val in list(obj.items()):
            if isinstance(val, Tensor):
                yield obj, k, val
            elif isinstance(val, (list, dict)) or hasattr(val, "__dict__"):
                yield from _walk_module_tensors(val, visited)
        return
    if not hasattr(obj, "__dict__"):
        return
    for name, val in list(obj.__dict__.items()):
        if name.startswith("_"):
            continue
        if isinstance(val, Tensor):
            yield obj, name, val
        elif isinstance(val, (list, dict)) or hasattr(val, "__dict__"):
            yield from _walk_module_tensors(val, visited)


def _setitem(parent, key, value) -> None:
    """Set ``value`` on ``parent`` at ``key`` regardless of container type."""
    if isinstance(parent, (list, dict)):
        parent[key] = value
    else:
        setattr(parent, key, value)
