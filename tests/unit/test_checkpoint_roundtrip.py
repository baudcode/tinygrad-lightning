"""Unit: safetensors checkpoint roundtrip + meta sidecar."""
from __future__ import annotations

import json

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import SGD

import tinygrad_lightning as L
from tinygrad_lightning.callbacks.checkpoint import (
    META_SUFFIX,
    load_checkpoint,
    save_checkpoint,
)


class _StubTrainer:
    """Minimal trainer-shaped object for save/load helpers."""

    def __init__(self, current_epoch: int = 0, global_step: int = 0):
        self.current_epoch = current_epoch
        self.global_step = global_step
        self.lr_scheduler_configs: list = []
        self.callback_metrics: dict[str, float] = {}
        self.default_root_dir = "/tmp/_unused"


class _SmallModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w1 = Tensor.uniform(4, 4, low=-0.1, high=0.1).contiguous()
        self.w2 = Tensor.uniform(4, 1, low=-0.1, high=0.1).contiguous()


def test_safetensors_roundtrip(tmp_path):
    model = _SmallModel()
    snap = {k: v.numpy().copy() for k, v in model.state_dict().items()}

    trainer = _StubTrainer(current_epoch=2, global_step=42)
    ckpt = save_checkpoint(trainer, model, tmp_path / "ckpt.safetensors")
    assert ckpt.exists()
    assert (ckpt.with_suffix(META_SUFFIX)).exists()

    # mutate weights, then load and verify they're restored
    model.w1.assign(Tensor.zeros_like(model.w1).contiguous())
    model.w2.assign(Tensor.zeros_like(model.w2).contiguous())
    assert np.linalg.norm(model.w1.numpy()) == 0.0

    fresh_trainer = _StubTrainer()
    meta = load_checkpoint(fresh_trainer, model, ckpt)
    assert np.allclose(model.w1.numpy(), snap["w1"])
    assert np.allclose(model.w2.numpy(), snap["w2"])
    assert meta["epoch"] == 2
    assert meta["global_step"] == 42


def test_meta_sidecar_roundtrip(tmp_path):
    """Meta JSON contains epoch, global_step, scheduler states; resume restores them."""
    model = _SmallModel()
    opt = SGD([model.w1, model.w2], lr=1e-2)
    sched = L.StepLR(opt, step_size=2, gamma=0.1)
    sched.step()
    sched.step()
    sched.step()  # _step_count = 3

    from tinygrad_lightning.trainer.trainer import LRSchedulerConfig

    trainer = _StubTrainer(current_epoch=5, global_step=100)
    trainer.lr_scheduler_configs = [LRSchedulerConfig(scheduler=sched)]

    ckpt = save_checkpoint(trainer, model, tmp_path / "ckpt.safetensors")
    with (ckpt.with_suffix(META_SUFFIX)).open() as f:
        raw = json.load(f)
    assert raw["epoch"] == 5
    assert raw["global_step"] == 100
    assert raw["scheduler_states"][0]["step_count"] == 3
    assert "version" in raw

    # Build a fresh scheduler at step 0; resume should bring it to step 3.
    fresh_opt = SGD([model.w1, model.w2], lr=1e-2)
    fresh_sched = L.StepLR(fresh_opt, step_size=2, gamma=0.1)
    fresh_trainer = _StubTrainer()
    fresh_trainer.lr_scheduler_configs = [LRSchedulerConfig(scheduler=fresh_sched)]
    load_checkpoint(fresh_trainer, model, ckpt)
    assert fresh_sched._step_count == 3
    assert fresh_trainer.current_epoch == 5
    assert fresh_trainer.global_step == 100


def test_load_checkpoint_rejects_corrupt_file(tmp_path):
    """A safetensors file with no model.* keys is rejected."""
    from tinygrad.nn.state import safe_save

    bogus = tmp_path / "bogus.safetensors"
    safe_save({"unrelated.key": Tensor.zeros(2)}, str(bogus))
    model = _SmallModel()
    fresh = _StubTrainer()
    import pytest
    with pytest.raises(RuntimeError, match="model"):
        load_checkpoint(fresh, model, bogus)
