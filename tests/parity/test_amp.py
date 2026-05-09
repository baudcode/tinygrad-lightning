"""Parity: mixed-precision training advances without NaN losses.

Smoke test for the precision plugin pipeline. The model itself stays in fp32;
``precision="16-mixed"`` exercises the dynamic loss-scaling + finite-check
path. ``"bf16-mixed"`` skips loss scaling.

Numeric tolerance is loose. We assert:
- 5+ training steps complete (``trainer.global_step`` advanced)
- No NaN/Inf in recorded per-step losses
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ._helpers import ACCEL, ArrayDataset, make_regression_data

EPOCHS = 1
BATCH_SIZE = 8
N_STEPS_EXPECTED = 64 // BATCH_SIZE  # 8


def _run_tinygrad(precision: str):
    from tinygrad import Tensor
    from tinygrad.nn.optim import SGD

    import tinygrad_lightning as L

    L.seed_everything(0)

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = Tensor.kaiming_uniform(8, 1)
            self.losses: list[float] = []

        def forward(self, x):
            return x @ self.w

        def training_step(self, batch, batch_idx):
            X, y = batch
            x_t = X if isinstance(X, Tensor) else Tensor(X)
            y_t = y if isinstance(y, Tensor) else Tensor(y)
            pred = self(x_t).squeeze(-1)
            loss = ((pred - y_t) ** 2).mean()
            self.losses.append(float(loss.numpy()))
            return loss

        def configure_optimizers(self):
            return SGD([self.w], lr=1e-2)

    X, y = make_regression_data(seed=0)
    dl = L.DataLoader(ArrayDataset(X, y), batch_size=BATCH_SIZE)
    # Construct the trainer FIRST so its set_default_device() runs before any model
    # tensors are allocated.
    trainer = L.Trainer(
        accelerator=ACCEL,
        max_epochs=EPOCHS,
        precision=precision,
        enable_progress_bar=False,
        logger=False,
    )
    model = M()
    trainer.fit(model, train_dataloaders=dl)
    return model.losses, trainer


def _run_torch(precision: str):
    import lightning.pytorch as PL
    import torch
    from torch.utils.data import DataLoader as TDL
    from torch.utils.data import Dataset as TDS

    PL.seed_everything(0)

    class M(PL.LightningModule):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 1)
            self.losses: list[float] = []

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self.fc(X.float()).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, y.float())
            self.losses.append(float(loss.detach()))
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    X, y = make_regression_data(seed=0)

    class DS(TDS):
        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            return torch.from_numpy(X[i]), torch.tensor(y[i])

    model = M()
    trainer = PL.Trainer(
        accelerator="cpu", devices=1, max_epochs=EPOCHS, enable_progress_bar=False,
        logger=False, enable_checkpointing=False, precision=precision,
    )
    trainer.fit(model, train_dataloaders=TDL(DS(), batch_size=BATCH_SIZE))
    return model.losses, trainer


def _check_smoke(losses, trainer, expected_steps: int) -> None:
    assert trainer.global_step >= expected_steps, (
        f"global_step {trainer.global_step} < expected {expected_steps}"
    )
    assert losses, "no losses recorded"
    assert all(np.isfinite(losses)), f"non-finite loss observed: {losses}"


def test_16_mixed_completes(parity_backend):
    if parity_backend == "tinygrad":
        losses, trainer = _run_tinygrad("16-mixed")
    else:
        losses, trainer = _run_torch("16-mixed")
    _check_smoke(losses, trainer, expected_steps=N_STEPS_EXPECTED)


def test_bf16_mixed_completes(parity_backend):
    if parity_backend == "tinygrad":
        losses, trainer = _run_tinygrad("bf16-mixed")
    else:
        losses, trainer = _run_torch("bf16-mixed")
    _check_smoke(losses, trainer, expected_steps=N_STEPS_EXPECTED)


def test_16_mixed_forward_output_is_fp16_end_to_end():
    """With precision='16-mixed', Trainer auto-sets dataloader dtype AND autocasts
    module params, so a forward pass with no manual casts produces fp16 output.

    This is the end-to-end fp16 path: DataLoader yields fp16 → autocast swaps
    weights to fp16 → fp16 @ fp16 = fp16 forward → backward routes fp32 grads
    to original leaves.
    """
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.optim import SGD

    import tinygrad_lightning as L

    L.seed_everything(0)
    captured: dict = {}

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = Tensor.kaiming_uniform(8, 1)

        def forward(self, x):
            return x @ self.w

        def training_step(self, batch, batch_idx):
            X, y = batch
            assert isinstance(X, Tensor), "DataLoader did not auto-cast batch to Tensor"
            captured["X.dtype"] = X.dtype
            captured["w.dtype"] = self.w.dtype
            pred = self(X)
            captured["pred.dtype"] = pred.dtype
            loss = ((pred.squeeze(-1) - y.cast(dtypes.float32)) ** 2).mean()
            return loss

        def configure_optimizers(self):
            return SGD([self.w], lr=1e-2)

    X, y = make_regression_data(seed=0)
    dl = L.DataLoader(ArrayDataset(X, y), batch_size=BATCH_SIZE)
    trainer = L.Trainer(
        accelerator=ACCEL, max_epochs=1, precision="16-mixed",
        enable_progress_bar=False, logger=False, limit_train_batches=1,
    )
    trainer.fit(M(), train_dataloaders=dl)

    assert captured["X.dtype"] == dtypes.float16, captured["X.dtype"]
    assert captured["w.dtype"] == dtypes.float16, captured["w.dtype"]
    assert captured["pred.dtype"] == dtypes.float16, captured["pred.dtype"]


def test_16_mixed_recovers_from_nan_grad():
    """Inject a NaN grad mid-training; trainer must not advance the optimizer
    on that step but must continue + halve the loss scale."""
    from tinygrad import Tensor
    from tinygrad.nn.optim import SGD

    import tinygrad_lightning as L

    L.seed_everything(0)

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = Tensor.kaiming_uniform(8, 1)
            self.steps_with_nan = 0
            self.last_w_before_nan_step = None

        def forward(self, x):
            return x @ self.w

        def training_step(self, batch, batch_idx):
            X, y = batch
            x_t = X if isinstance(X, Tensor) else Tensor(X)
            y_t = y if isinstance(y, Tensor) else Tensor(y)
            pred = self(x_t).squeeze(-1)
            loss = ((pred - y_t) ** 2).mean()
            # Inject a NaN multiplier on step 1 only — multiplying by NaN makes
            # d_loss/d_param = NaN through the chain rule, so the optimizer must
            # detect it and skip the step.
            if batch_idx == 1:
                self.last_w_before_nan_step = self.w.numpy().copy()
                loss = loss * Tensor([float("nan")]).reshape(())
            return loss

        def configure_optimizers(self):
            return SGD([self.w], lr=1e-2)

    X, y = make_regression_data(seed=0)
    dl = L.DataLoader(ArrayDataset(X, y), batch_size=BATCH_SIZE)
    initial_scale = 2.0 ** 16
    trainer = L.Trainer(
        accelerator=ACCEL, max_epochs=1, precision="16-mixed",
        enable_progress_bar=False, logger=False,
    )
    model = M()
    trainer.fit(model, train_dataloaders=dl)

    # Loss scaler must have backed off at least once.
    assert trainer.precision_plugin.scaler.scale < initial_scale, (
        f"loss scale did not back off: {trainer.precision_plugin.scaler.scale}"
    )
    # Training continued past the NaN step.
    assert trainer.global_step >= N_STEPS_EXPECTED
