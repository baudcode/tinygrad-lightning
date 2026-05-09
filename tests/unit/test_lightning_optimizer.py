"""Unit: LightningOptimizer wrapper around tinygrad optims."""
from __future__ import annotations

import numpy as np

from tinygrad import Tensor
from tinygrad.nn.optim import SGD

from tinygrad_lightning.core.optimizer import (
    LightningOptimizer,
    _to_float,
    param_group_optimizer,
)


def _make_params(n: int) -> list[Tensor]:
    return [Tensor.ones(4).contiguous() for _ in range(n)]


def test_single_optimizer_exposes_one_param_group():
    params = _make_params(2)
    opt = SGD(params, lr=1e-3)
    wrapped = LightningOptimizer(opt)
    assert len(wrapped.param_groups) == 1
    assert np.isclose(_to_float(wrapped.param_groups[0]["lr"]), 1e-3)
    assert np.allclose(wrapped.get_lrs(), [1e-3])


def test_param_group_optimizer_builds_optimizer_group_for_two_groups():
    params_a = _make_params(2)
    params_b = _make_params(2)
    opt = param_group_optimizer(SGD, [
        {"params": params_a, "lr": 1e-4},
        {"params": params_b, "lr": 1e-2},
    ])
    wrapped = LightningOptimizer(opt)
    assert len(wrapped.param_groups) == 2
    assert np.allclose(wrapped.get_lrs(), [1e-4, 1e-2])


def test_param_group_dict_constructor_passes_extra_kwargs_per_group():
    """`momentum` set per group should land on the underlying tinygrad optimizer."""
    p_a, p_b = _make_params(1), _make_params(1)
    opt = param_group_optimizer(SGD, [
        {"params": p_a, "lr": 1e-3, "momentum": 0.0},
        {"params": p_b, "lr": 1e-3, "momentum": 0.9},
    ])
    wrapped = LightningOptimizer(opt)
    sub_a = wrapped.param_groups[0]["_optim"]
    sub_b = wrapped.param_groups[1]["_optim"]
    # tinygrad's SGD stores momentum as float on the optimizer
    assert sub_a.momentum == 0.0
    assert sub_b.momentum == 0.9


def test_set_lrs_mutates_underlying_optimizer_lr_tensor():
    """A scheduler must be able to mutate per-group LR; verify the Tensor scalar updates."""
    p_a, p_b = _make_params(1), _make_params(1)
    opt = param_group_optimizer(SGD, [
        {"params": p_a, "lr": 1e-3},
        {"params": p_b, "lr": 1e-2},
    ])
    wrapped = LightningOptimizer(opt)
    wrapped.set_lrs([5e-5, 5e-3])
    assert np.isclose(wrapped.get_lrs()[0], 5e-5)
    assert np.isclose(wrapped.get_lrs()[1], 5e-3)
    # the underlying tinygrad optimizer's lr Tensor should reflect the new value
    assert np.isclose(_to_float(wrapped.param_groups[0]["_optim"].lr), 5e-5)


def test_per_group_lr_mutable_by_scheduler():
    """Smoke: a fake scheduler that mutates set_lrs() is reflected in get_lrs()."""
    from tinygrad_lightning.core.lr_scheduler import StepLR

    p_a, p_b = _make_params(1), _make_params(1)
    opt = param_group_optimizer(SGD, [
        {"params": p_a, "lr": 1.0},
        {"params": p_b, "lr": 2.0},
    ])
    wrapped = LightningOptimizer(opt)
    sched = StepLR(wrapped, step_size=1, gamma=0.1)
    sched.step()
    lrs = wrapped.get_lrs()
    assert np.isclose(lrs[0], 0.1, atol=1e-6), lrs
    assert np.isclose(lrs[1], 0.2, atol=1e-6), lrs
    sched.step()
    lrs = wrapped.get_lrs()
    assert np.isclose(lrs[0], 0.01, atol=1e-6), lrs
    assert np.isclose(lrs[1], 0.02, atol=1e-6), lrs
