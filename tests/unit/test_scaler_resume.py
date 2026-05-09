"""Unit: ModelCheckpoint persists DynamicLossScaler state in meta.json."""
from __future__ import annotations

import json

import numpy as np
from tinygrad import Tensor

import tinygrad_lightning as L
from tinygrad_lightning.callbacks.checkpoint import (
    META_SUFFIX,
    load_checkpoint,
    save_checkpoint,
)
from tinygrad_lightning.core.precision import PrecisionPlugin


class _StubTrainer:
    def __init__(self, precision: str = "16-mixed"):
        self.current_epoch = 0
        self.global_step = 0
        self.lr_scheduler_configs: list = []
        self.optimizers: list = []
        self.callback_metrics: dict[str, float] = {}
        self.default_root_dir = "/tmp/_unused"
        self.precision_plugin = PrecisionPlugin(precision)


class _SmallModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w = Tensor.uniform(4, 4, low=-0.1, high=0.1).contiguous()


def test_scaler_state_persisted_in_meta(tmp_path):
    trainer = _StubTrainer()
    # Force the scaler to back off twice → scale = init/4
    trainer.precision_plugin.scaler.update(found_inf=True)
    trainer.precision_plugin.scaler.update(found_inf=True)
    expected_scale = trainer.precision_plugin.scaler.scale

    model = _SmallModel()
    ckpt = save_checkpoint(trainer, model, tmp_path / "ckpt.safetensors")

    with ckpt.with_suffix(META_SUFFIX).open() as f:
        meta = json.load(f)
    assert "precision" in meta, list(meta.keys())
    assert meta["precision"]["scaler"]["scale"] == expected_scale


def test_scaler_state_restored_on_load(tmp_path):
    trainer_a = _StubTrainer()
    trainer_a.precision_plugin.scaler.update(found_inf=True)  # scale halved once
    expected = trainer_a.precision_plugin.scaler.scale
    expected_growth = trainer_a.precision_plugin.scaler._growth_counter

    model = _SmallModel()
    ckpt = save_checkpoint(trainer_a, model, tmp_path / "ckpt.safetensors")

    trainer_b = _StubTrainer()  # fresh, scale = 2**16
    assert trainer_b.precision_plugin.scaler.scale != expected
    load_checkpoint(trainer_b, model, ckpt)
    assert trainer_b.precision_plugin.scaler.scale == expected
    assert trainer_b.precision_plugin.scaler._growth_counter == expected_growth


def test_load_skips_scaler_when_precision_changed(tmp_path):
    """Loading a 16-mixed ckpt into a 32-true trainer doesn't blow up."""
    src = _StubTrainer("16-mixed")
    src.precision_plugin.scaler.update(found_inf=True)
    model = _SmallModel()
    ckpt = save_checkpoint(src, model, tmp_path / "ckpt.safetensors")

    dst = _StubTrainer("32-true")  # different precision
    # Should NOT raise; scaler load is silenced for precision-mismatch.
    load_checkpoint(dst, model, ckpt)
    assert dst.precision_plugin.scaler is None
