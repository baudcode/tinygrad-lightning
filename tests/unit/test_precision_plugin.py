"""Unit: PrecisionPlugin + DynamicLossScaler."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tinygrad_lightning.core.precision import DynamicLossScaler, PrecisionPlugin


def test_invalid_precision_string_raises():
    with pytest.raises(ValueError, match="unsupported precision"):
        PrecisionPlugin("fp32")


def test_default_precision_is_no_op():
    pp = PrecisionPlugin("32-true")
    assert pp.scaler is None
    assert pp.loss_scale == 1.0
    # scale_loss must return the input unchanged (no multiplication wrapper)
    sentinel = object()
    assert pp.scale_loss(sentinel) is sentinel


def test_dynamic_loss_scaler_halves_on_nonfinite():
    s = DynamicLossScaler(init_scale=1024.0, backoff_factor=0.5)
    s.update(found_inf=True)
    assert s.scale == 512.0
    s.update(found_inf=True)
    assert s.scale == 256.0


def test_dynamic_loss_scaler_grows_after_growth_interval():
    s = DynamicLossScaler(init_scale=2.0, growth_factor=2.0, growth_interval=3)
    for _ in range(2):
        s.update(found_inf=False)
    assert s.scale == 2.0  # not yet at growth interval
    s.update(found_inf=False)  # 3rd successful → grow
    assert s.scale == 4.0
    # Counter resets; need another full interval to grow again.
    s.update(found_inf=False)
    s.update(found_inf=False)
    assert s.scale == 4.0
    s.update(found_inf=False)
    assert s.scale == 8.0


def test_dynamic_loss_scaler_growth_counter_resets_on_inf():
    s = DynamicLossScaler(init_scale=2.0, growth_factor=2.0, growth_interval=3)
    s.update(False); s.update(False)  # counter=2
    s.update(True)                     # found inf → counter=0, scale halves
    assert s.scale == 1.0
    s.update(False); s.update(False)
    assert s.scale == 1.0  # not yet 3 successful since reset


def test_bf16_skips_loss_scaling():
    pp = PrecisionPlugin("bf16-mixed")
    assert pp.scaler is None
    assert pp.use_loss_scaling is False
    # check_grads_finite should always return True (no scaling → no NaN-skip path)
    assert pp.check_grads_finite([]) is True


def test_state_dict_roundtrip():
    pp = PrecisionPlugin("16-mixed")
    pp.scaler.update(found_inf=True)  # scale = 32768.0
    state = pp.state_dict()

    pp2 = PrecisionPlugin("16-mixed")
    pp2.load_state_dict(state)
    assert pp2.scaler.scale == pp.scaler.scale


def test_state_dict_precision_mismatch_raises():
    pp = PrecisionPlugin("16-mixed")
    state = pp.state_dict()
    pp32 = PrecisionPlugin("32-true")
    with pytest.raises(RuntimeError, match="precision mismatch"):
        pp32.load_state_dict(state)


class _FakeOpt:
    """Just enough of an optimizer to drive unscale_grads / check_grads_finite."""

    def __init__(self, params):
        self.params = params


class _FakeTensor:
    """Mimics tinygrad's Tensor enough for grad scaling tests."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)
        self.grad = None

    def detach(self):
        return self

    def numpy(self):
        return self.arr


def test_unscale_grads_divides_by_scale():
    p = _FakeTensor([1.0])
    p.grad = _FakeTensor([100.0])
    # Substitute mul with __mul__-compatible behavior; the real Tensor uses *.
    # We monkeypatch __mul__ on the fake so `p.grad * (1/scale)` works.
    _FakeTensor.__mul__ = lambda self, other: _FakeTensor(self.arr * other)
    pp = PrecisionPlugin("16-mixed")
    pp.scaler.scale = 100.0  # easier number
    pp.unscale_grads([_FakeOpt([p])])
    assert np.allclose(p.grad.arr, 1.0)


def test_check_grads_finite_returns_false_on_nan():
    p = _FakeTensor([1.0])
    p.grad = _FakeTensor([float("nan")])
    pp = PrecisionPlugin("16-mixed")
    assert pp.check_grads_finite([_FakeOpt([p])]) is False
