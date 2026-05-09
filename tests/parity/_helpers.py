"""Shared helpers for parity tests.

The accelerator passed to ``Trainer`` defaults to ``cpu``. Override via
``TG_TEST_ACCEL=<auto|gpu|cpu|...>`` for environments where ``cpu`` is not
viable (e.g. workstations without ``clang`` available, where tinygrad's CPU
backend cannot compile kernels).
"""
from __future__ import annotations

import os

import numpy as np

ACCEL = os.environ.get("TG_TEST_ACCEL", "cpu")


def make_regression_data(seed: int = 0, n: int = 64, in_dim: int = 8):
    """Synthetic linear-regression dataset; deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, in_dim)).astype(np.float32)
    w = rng.standard_normal((in_dim,)).astype(np.float32)
    y = (X @ w).astype(np.float32)
    return X, y


class ArrayDataset:
    """Map-style dataset over (X, y) pairs of NumPy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def train_with_tinygrad(seed: int = 42, epochs: int = 3, batch_size: int = 8, lr: float = 5e-2):
    """Train a 2-layer MLP under tinygrad-lightning; return per-step training losses."""
    from tinygrad import Tensor
    from tinygrad.nn.optim import SGD

    import tinygrad_lightning as L

    L.seed_everything(seed)

    class MLP(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w1 = Tensor.kaiming_uniform(8, 16)
            self.b1 = Tensor.zeros(16)
            self.w2 = Tensor.kaiming_uniform(16, 1)
            self.b2 = Tensor.zeros(1)
            self.losses: list[float] = []

        def forward(self, x):
            return (x @ self.w1 + self.b1).relu() @ self.w2 + self.b2

        def training_step(self, batch, batch_idx):
            X_np, y_np = batch
            x = Tensor(X_np)
            y = Tensor(y_np).reshape(-1, 1)
            pred = self(x)
            loss = ((pred - y) ** 2).mean()
            self.losses.append(float(loss.numpy()))
            self.log("loss", loss, on_step=True, on_epoch=False)
            return loss

        def configure_optimizers(self):
            return SGD([self.w1, self.b1, self.w2, self.b2], lr=lr)

    X, y = make_regression_data(seed=seed)
    train_dl = L.DataLoader(ArrayDataset(X, y), batch_size=batch_size)

    trainer = L.Trainer(accelerator=ACCEL, max_epochs=epochs, enable_progress_bar=False)
    model = MLP()
    trainer.fit(model, train_dataloaders=train_dl)
    return model.losses, trainer


def train_with_torch(seed: int = 42, epochs: int = 3, batch_size: int = 8, lr: float = 5e-2):
    """Train an equivalent 2-layer MLP under lightning.pytorch."""
    import lightning.pytorch as PL
    import torch
    from torch.utils.data import DataLoader as TDataLoader
    from torch.utils.data import Dataset as TDataset

    PL.seed_everything(seed)

    class MLP(PL.LightningModule):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(8, 16)
            self.fc2 = torch.nn.Linear(16, 1)
            self.losses: list[float] = []

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self(X.float()).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, y.float())
            self.losses.append(float(loss.detach()))
            self.log("loss", loss, on_step=True, on_epoch=False)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=lr)

    X, y = make_regression_data(seed=seed)

    class TorchArrayDS(TDataset):
        def __len__(self):
            return len(X)

        def __getitem__(self, idx):
            return torch.from_numpy(X[idx]), torch.tensor(y[idx])

    train_dl = TDataLoader(TorchArrayDS(), batch_size=batch_size)

    model = MLP()
    trainer = PL.Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_dl)
    return model.losses, trainer
