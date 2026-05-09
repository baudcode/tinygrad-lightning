"""Parity: accelerator/device resolution.

The CPU test runs against tinygrad's CPU device, which requires ``clang`` on
``$PATH`` (the CPU backend compiles kernels via clang). On hosts without
clang, the test skips. Override the accelerator the rest of the parity
harness uses via ``TG_TEST_ACCEL``.
"""
from __future__ import annotations

import shutil

import pytest

from ._helpers import ACCEL, train_with_tinygrad, train_with_torch


def _tg_cpu_runnable() -> bool:
    """tinygrad's CPU backend invokes ``clang`` to compile kernels."""
    return shutil.which("clang") is not None


def test_cpu_accelerator_runs(parity_backend):
    if parity_backend == "tinygrad":
        if not _tg_cpu_runnable():
            pytest.skip("clang not on PATH; tinygrad CPU backend cannot compile kernels here")
        # Force CPU regardless of TG_TEST_ACCEL — this test's contract is the CPU path.
        import tinygrad_lightning as L
        from ._helpers import ArrayDataset, make_regression_data
        from tinygrad import Tensor
        from tinygrad.nn.optim import SGD

        L.seed_everything(1)

        class M(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.w = Tensor.kaiming_uniform(8, 1)

            def forward(self, x):
                return x @ self.w

            def training_step(self, batch, batch_idx):
                X, y = batch
                pred = self(Tensor(X)).squeeze(-1)
                return ((pred - Tensor(y)) ** 2).mean()

            def configure_optimizers(self):
                return SGD([self.w], lr=1e-2)

        X, y = make_regression_data(seed=1)
        dl = L.DataLoader(ArrayDataset(X, y), batch_size=8)
        L.Trainer(accelerator="cpu", max_epochs=1, enable_progress_bar=False).fit(M(), train_dataloaders=dl)
    else:
        losses, _ = train_with_torch(seed=1, epochs=1, batch_size=8)
        assert losses, "expected at least one training step"


def _gpu_available_for(backend: str) -> bool:
    if backend == "tinygrad":
        from tinygrad_lightning.core.accelerator import GPU_DEVICES, _available_devices
        return any(d in _available_devices() for d in GPU_DEVICES)
    if backend == "torch":
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            return False
    return False


def test_gpu_accelerator_runs(parity_backend):
    if not _gpu_available_for(parity_backend):
        pytest.skip(f"no GPU device available for backend={parity_backend}")
    # Smoke test only — re-run training with accelerator="gpu".
    if parity_backend == "tinygrad":
        from tests.parity._helpers import ArrayDataset, make_regression_data
        from tinygrad import Tensor
        from tinygrad.nn.optim import SGD

        import tinygrad_lightning as L

        L.seed_everything(0)

        class M(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.w = Tensor.kaiming_uniform(8, 1)

            def forward(self, x):
                return x @ self.w

            def training_step(self, batch, batch_idx):
                X, y = batch
                pred = self(Tensor(X))
                loss = ((pred.squeeze(-1) - Tensor(y)) ** 2).mean()
                self.log("loss", loss, on_step=True)
                return loss

            def configure_optimizers(self):
                return SGD([self.w], lr=1e-2)

        X, y = make_regression_data(seed=0)
        dl = L.DataLoader(ArrayDataset(X, y), batch_size=8)
        L.Trainer(accelerator="gpu", max_epochs=1, enable_progress_bar=False).fit(M(), train_dataloaders=dl)
    else:
        import lightning.pytorch as PL
        import torch
        from torch.utils.data import DataLoader as TDL
        from torch.utils.data import Dataset as TDS

        from ._helpers import make_regression_data

        PL.seed_everything(0)

        class M(PL.LightningModule):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)

            def training_step(self, batch, batch_idx):
                X, y = batch
                pred = self.fc(X.float()).squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, y.float())
                self.log("loss", loss)
                return loss

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=1e-2)

        X, y = make_regression_data(seed=0)

        class DS(TDS):
            def __len__(self):
                return len(X)

            def __getitem__(self, i):
                return torch.from_numpy(X[i]), torch.tensor(y[i])

        PL.Trainer(
            accelerator="gpu", devices=1, max_epochs=1, enable_progress_bar=False,
            logger=False, enable_checkpointing=False,
        ).fit(M(), train_dataloaders=TDL(DS(), batch_size=8))
