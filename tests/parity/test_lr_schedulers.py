"""Parity: LR schedulers follow the scheduled curve at step boundaries.

For each scheduler, capture the LR observed at the start of every training
step (via ``LearningRateMonitor``) and compare against the analytical curve.
The check is per-backend: each backend's ``LearningRateMonitor.lrs`` must
match its own analytical reference within tolerance. Cross-backend numeric
equality is NOT asserted (different LR Tensor dtypes / float promotion).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ._helpers import ArrayDataset, make_regression_data


def _expected_step(base_lr: float, step_size: int, gamma: float, n_steps: int) -> list[float]:
    return [base_lr * (gamma ** (s // step_size)) for s in range(n_steps + 1)]


def _expected_multistep(base_lr: float, milestones: list[int], gamma: float, n_steps: int) -> list[float]:
    return [base_lr * (gamma ** sum(1 for m in milestones if s >= m)) for s in range(n_steps + 1)]


def _expected_cosine(base_lr: float, T_max: int, eta_min: float, n_steps: int) -> list[float]:
    out = []
    for s in range(n_steps + 1):
        t = min(s, T_max)
        out.append(eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * t / T_max)) / 2)
    return out


# Each scheduler test runs for `EPOCHS * BATCHES_PER_EPOCH` training steps;
# the scheduler interval is "step", so the step counter advances once per
# optimizer step.
EPOCHS = 2
BATCHES_PER_EPOCH = 8  # 64 samples / batch_size 8
N_STEPS = EPOCHS * BATCHES_PER_EPOCH


def _run_tinygrad(scheduler_name: str, base_lr: float = 0.1):
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
            pred = self(Tensor(X)).squeeze(-1)
            return ((pred - Tensor(y)) ** 2).mean()

        def configure_optimizers(self):
            opt = SGD([self.w], lr=base_lr)
            sched = _build_tinygrad_scheduler(scheduler_name, opt, base_lr=base_lr)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    monitor = L.LearningRateMonitor(logging_interval="step")
    X, y = make_regression_data(seed=0)
    dl = L.DataLoader(ArrayDataset(X, y), batch_size=8)
    from ._helpers import ACCEL
    L.Trainer(
        accelerator=ACCEL, max_epochs=EPOCHS, enable_progress_bar=False, callbacks=[monitor]
    ).fit(M(), train_dataloaders=dl)
    return monitor.lrs["lr-0"]


def _run_torch(scheduler_name: str, base_lr: float = 0.1):
    import lightning.pytorch as PL
    import torch
    from torch.utils.data import DataLoader as TDL
    from torch.utils.data import Dataset as TDS

    class _LRRecorder(PL.Callback):
        """PL's LearningRateMonitor requires a logger; we just record LRs directly."""

        def __init__(self):
            self.lrs: list[float] = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.lrs.append(trainer.optimizers[0].param_groups[0]["lr"])

    PL.seed_everything(0)

    class M(PL.LightningModule):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 1)

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self.fc(X.float()).squeeze(-1)
            return torch.nn.functional.mse_loss(pred, y.float())

        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=base_lr)
            sched = _build_torch_scheduler(scheduler_name, opt, base_lr=base_lr)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    X, y = make_regression_data(seed=0)

    class DS(TDS):
        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            return torch.from_numpy(X[i]), torch.tensor(y[i])

    recorder = _LRRecorder()
    PL.Trainer(
        accelerator="cpu", devices=1, max_epochs=EPOCHS, enable_progress_bar=False,
        logger=False, enable_checkpointing=False, callbacks=[recorder],
    ).fit(M(), train_dataloaders=TDL(DS(), batch_size=8))
    return recorder.lrs


def _build_tinygrad_scheduler(name, opt, base_lr):
    import tinygrad_lightning as L

    if name == "step":
        return L.StepLR(opt, step_size=4, gamma=0.5)
    if name == "multistep":
        return L.MultiStepLR(opt, milestones=[5, 10], gamma=0.5)
    if name == "cosine":
        return L.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=0.0)
    if name == "lambda":
        return L.LambdaLR(opt, lr_lambda=lambda step: 1.0 / (1 + step))
    if name == "onecycle":
        return L.OneCycleLR(opt, max_lr=base_lr, total_steps=N_STEPS)
    raise ValueError(name)


def _build_torch_scheduler(name, opt, base_lr):
    import torch.optim.lr_scheduler as T

    if name == "step":
        return T.StepLR(opt, step_size=4, gamma=0.5)
    if name == "multistep":
        return T.MultiStepLR(opt, milestones=[5, 10], gamma=0.5)
    if name == "cosine":
        return T.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=0.0)
    if name == "lambda":
        return T.LambdaLR(opt, lr_lambda=lambda step: 1.0 / (1 + step))
    if name == "onecycle":
        return T.OneCycleLR(opt, max_lr=base_lr, total_steps=N_STEPS)
    raise ValueError(name)


@pytest.mark.parametrize("scheduler_name", ["step", "multistep", "cosine", "lambda", "onecycle"])
def test_scheduler_matches_analytical_curve(parity_backend, scheduler_name):
    base_lr = 0.1
    if parity_backend == "tinygrad":
        observed = _run_tinygrad(scheduler_name, base_lr=base_lr)
    else:
        observed = _run_torch(scheduler_name, base_lr=base_lr)

    assert len(observed) == N_STEPS, f"expected {N_STEPS} LR samples, got {len(observed)}"

    # `observed[i]` is the LR active during training step `i+1` (LR monitor records
    # at on_train_batch_end, AFTER the optimizer step + scheduler step). For
    # most schedulers, that means observed[i] = scheduler.get_lr at counter=i+1.
    if scheduler_name == "step":
        expected = _expected_step(base_lr, step_size=4, gamma=0.5, n_steps=N_STEPS)[1:]
        assert np.allclose(observed, expected, rtol=1e-4), list(zip(observed, expected))
    elif scheduler_name == "multistep":
        expected = _expected_multistep(base_lr, [5, 10], 0.5, N_STEPS)[1:]
        assert np.allclose(observed, expected, rtol=1e-4), list(zip(observed, expected))
    elif scheduler_name == "cosine":
        expected = _expected_cosine(base_lr, N_STEPS, 0.0, N_STEPS)[1:]
        assert np.allclose(observed, expected, rtol=1e-3, atol=1e-5)
    elif scheduler_name == "lambda":
        expected = [base_lr * (1.0 / (1 + s)) for s in range(1, N_STEPS + 1)]
        assert np.allclose(observed, expected, rtol=1e-4)
    elif scheduler_name == "onecycle":
        # One-cycle has framework-specific defaults (anneal strategy, three_phase,
        # final_div_factor). We only check structural properties: starts well below
        # max_lr, peaks near max_lr, ends well below max_lr.
        assert observed[0] < base_lr * 0.5
        peak = max(observed)
        assert 0.5 * base_lr <= peak <= base_lr * 1.05, peak
        assert observed[-1] < peak * 0.6
