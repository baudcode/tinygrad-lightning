"""Unit: PrecisionPlugin.autocast swaps module Tensors for the duration of a forward."""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

from tinygrad_lightning.core.precision import PrecisionPlugin


class _Inner:
    def __init__(self):
        self.weight = Tensor.uniform(2, 2, low=-0.1, high=0.1).contiguous()


class _Module:
    def __init__(self):
        self.w = Tensor.uniform(4, 2, low=-0.1, high=0.1).contiguous()
        self.inner = _Inner()
        self.layers = [_Inner(), _Inner()]
        self.named: dict[str, Tensor] = {"k": Tensor.ones(2)}
        self._private = Tensor.zeros(2)  # should NOT be swapped (leading underscore)


def _captured_dtypes(module):
    """Snapshot dtypes of all swappable Tensors on the module."""
    return {
        "w": module.w.dtype,
        "inner.weight": module.inner.weight.dtype,
        "layers.0.weight": module.layers[0].weight.dtype,
        "layers.1.weight": module.layers[1].weight.dtype,
        "named.k": module.named["k"].dtype,
        "_private": module._private.dtype,
    }


def test_autocast_no_op_for_fp32():
    pp = PrecisionPlugin("32-true")
    module = _Module()
    before = _captured_dtypes(module)
    with pp.autocast(module):
        during = _captured_dtypes(module)
    after = _captured_dtypes(module)
    assert before == during == after


def test_autocast_swaps_to_fp16():
    pp = PrecisionPlugin("16-mixed")
    module = _Module()
    original_w = module.w
    with pp.autocast(module):
        # Direct attr swapped
        assert module.w.dtype == dtypes.float16
        # Nested object's attr swapped
        assert module.inner.weight.dtype == dtypes.float16
        # List items swapped
        assert module.layers[0].weight.dtype == dtypes.float16
        assert module.layers[1].weight.dtype == dtypes.float16
        # Dict values swapped
        assert module.named["k"].dtype == dtypes.float16
        # Private (underscore-prefixed) NOT swapped
        assert module._private.dtype == dtypes.float32
    # Restored on exit
    assert module.w.dtype == dtypes.float32
    assert module.w is original_w  # exact same Tensor object


def test_autocast_swaps_to_bf16():
    pp = PrecisionPlugin("bf16-mixed")
    module = _Module()
    with pp.autocast(module):
        assert module.w.dtype == dtypes.bfloat16


def test_autocast_restores_originals_on_exception():
    pp = PrecisionPlugin("16-mixed")
    module = _Module()
    original_w = module.w
    try:
        with pp.autocast(module):
            assert module.w.dtype == dtypes.float16
            raise ValueError("boom")
    except ValueError:
        pass
    assert module.w is original_w
    assert module.w.dtype == dtypes.float32


def test_autocast_grads_route_back_to_fp32_leaves():
    """The autograd graph references the cast tensors after the with block exits;
    backward must still produce fp32 grads on the original leaves."""
    pp = PrecisionPlugin("16-mixed")
    module = _Module()
    original_w = module.w
    original_w.requires_grad_(True)

    x = Tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtypes.float32)
    with pp.autocast(module):
        # x is fp32, module.w is fp16 → out is fp32 (tinygrad upcasts)
        out = (x @ module.w).sum()
    # After exit, module.w is back to fp32 but the loss graph still references fp16 cast.
    out.backward()
    assert original_w.grad is not None
    assert original_w.grad.dtype == dtypes.float32, original_w.grad.dtype
    assert np.all(np.isfinite(original_w.grad.numpy()))
