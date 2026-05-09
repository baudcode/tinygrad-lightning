"""Unit: DataLoader dtype param + Trainer auto-set behavior."""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

import tinygrad_lightning as L


class _DS:
    def __init__(self, n: int = 16):
        self.X = np.random.default_rng(0).standard_normal((n, 4)).astype(np.float32)
        self.y = np.arange(n, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def test_dataloader_default_yields_numpy():
    dl = L.DataLoader(_DS(), batch_size=4)
    batch = next(iter(dl))
    X, y = batch
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_dataloader_with_fp16_dtype_yields_fp16_tensor_for_floats():
    dl = L.DataLoader(_DS(), batch_size=4, dtype=dtypes.float16)
    X, y = next(iter(dl))
    assert isinstance(X, Tensor)
    assert X.dtype == dtypes.float16
    # Integer labels must NOT be cast to fp16.
    assert isinstance(y, Tensor)
    assert y.dtype != dtypes.float16
    assert y.dtype.itemsize >= 4 or y.dtype in (dtypes.int32, dtypes.int64), y.dtype


def test_dataloader_with_bf16_dtype():
    dl = L.DataLoader(_DS(), batch_size=4, dtype=dtypes.bfloat16)
    X, y = next(iter(dl))
    assert isinstance(X, Tensor)
    assert X.dtype == dtypes.bfloat16


def test_trainer_auto_sets_dataloader_dtype_for_16mixed():
    """Trainer with precision='16-mixed' should mutate ``dl.dtype`` to fp16."""
    dl = L.DataLoader(_DS(), batch_size=4)
    assert dl.dtype is None

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = Tensor.kaiming_uniform(4, 1)

        def training_step(self, batch, batch_idx):
            X, y = batch
            x_t = X if isinstance(X, Tensor) else Tensor(X)
            return ((x_t @ self.w).squeeze(-1).cast(dtypes.float32) - y.cast(dtypes.float32)).pow(2).mean()

        def configure_optimizers(self):
            from tinygrad.nn.optim import SGD
            return SGD([self.w], lr=1e-2)

    trainer = L.Trainer(
        accelerator="cpu", max_epochs=1, precision="16-mixed",
        enable_progress_bar=False, logger=False, limit_train_batches=1,
    )
    trainer.fit(M(), train_dataloaders=dl)
    assert dl.dtype == dtypes.float16


def test_trainer_does_not_override_user_dtype():
    """If the user pinned ``dtype=fp32`` explicitly, the trainer must not flip it."""
    dl = L.DataLoader(_DS(), batch_size=4, dtype=dtypes.float32)

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = Tensor.kaiming_uniform(4, 1)

        def training_step(self, batch, batch_idx):
            X, y = batch
            return ((X @ self.w).squeeze(-1) - y.cast(dtypes.float32)).pow(2).mean()

        def configure_optimizers(self):
            from tinygrad.nn.optim import SGD
            return SGD([self.w], lr=1e-2)

    trainer = L.Trainer(
        accelerator="cpu", max_epochs=1, precision="16-mixed",
        enable_progress_bar=False, logger=False, limit_train_batches=1,
    )
    trainer.fit(M(), train_dataloaders=dl)
    assert dl.dtype == dtypes.float32  # unchanged


def test_trainer_does_not_set_dtype_for_fp32_precision():
    dl = L.DataLoader(_DS(), batch_size=4)

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = Tensor.kaiming_uniform(4, 1)

        def training_step(self, batch, batch_idx):
            X, y = batch
            return ((Tensor(X) @ self.w).squeeze(-1) - Tensor(y).cast(dtypes.float32)).pow(2).mean()

        def configure_optimizers(self):
            from tinygrad.nn.optim import SGD
            return SGD([self.w], lr=1e-2)

    trainer = L.Trainer(
        accelerator="cpu", max_epochs=1,
        enable_progress_bar=False, logger=False, limit_train_batches=1,
    )
    trainer.fit(M(), train_dataloaders=dl)
    assert dl.dtype is None
