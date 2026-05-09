"""Parity: per-module learning rates via param groups.

Build a model with ``encoder`` and ``head`` modules, train both backends with
``encoder`` LR ≪ ``head`` LR, and assert the head's weight delta is at least
10× the encoder's weight delta — in both backends.
"""
from __future__ import annotations

import numpy as np

from ._helpers import ArrayDataset, make_regression_data

EPOCHS = 2
ENCODER_LR = 1e-5
HEAD_LR = 1e-1


def _run_tinygrad():
    from tinygrad import Tensor
    from tinygrad.nn.optim import SGD

    import tinygrad_lightning as L

    L.seed_everything(0)

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.enc_w = Tensor.kaiming_uniform(8, 16)
            self.enc_b = Tensor.zeros(16)
            self.head_w = Tensor.kaiming_uniform(16, 1)
            self.head_b = Tensor.zeros(1)

        def encoder_params(self):
            return [self.enc_w, self.enc_b]

        def head_params(self):
            return [self.head_w, self.head_b]

        def forward(self, x):
            return (x @ self.enc_w + self.enc_b).relu() @ self.head_w + self.head_b

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self(Tensor(X))
            return ((pred - Tensor(y).reshape(-1, 1)) ** 2).mean()

        def configure_optimizers(self):
            return L.param_group_optimizer(SGD, [
                {"params": self.encoder_params(), "lr": ENCODER_LR},
                {"params": self.head_params(),    "lr": HEAD_LR},
            ])

    model = M()
    enc_before = [p.numpy().copy() for p in model.encoder_params()]
    head_before = [p.numpy().copy() for p in model.head_params()]

    X, y = make_regression_data(seed=0)
    dl = L.DataLoader(ArrayDataset(X, y), batch_size=8)
    from ._helpers import ACCEL
    L.Trainer(accelerator=ACCEL, max_epochs=EPOCHS, enable_progress_bar=False).fit(model, train_dataloaders=dl)

    enc_after = [p.numpy() for p in model.encoder_params()]
    head_after = [p.numpy() for p in model.head_params()]
    return _delta(enc_before, enc_after), _delta(head_before, head_after)


def _run_torch():
    import lightning.pytorch as PL
    import torch
    from torch.utils.data import DataLoader as TDL
    from torch.utils.data import Dataset as TDS

    PL.seed_everything(0)

    class M(PL.LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(8, 16)
            self.head = torch.nn.Linear(16, 1)

        def forward(self, x):
            return self.head(torch.relu(self.encoder(x)))

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self(X.float()).squeeze(-1)
            return torch.nn.functional.mse_loss(pred, y.float())

        def configure_optimizers(self):
            return torch.optim.SGD([
                {"params": list(self.encoder.parameters()), "lr": ENCODER_LR},
                {"params": list(self.head.parameters()),    "lr": HEAD_LR},
            ])

    model = M()
    enc_before = [p.detach().clone() for p in model.encoder.parameters()]
    head_before = [p.detach().clone() for p in model.head.parameters()]

    X, y = make_regression_data(seed=0)

    class DS(TDS):
        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            return torch.from_numpy(X[i]), torch.tensor(y[i])

    PL.Trainer(
        accelerator="cpu", devices=1, max_epochs=EPOCHS, enable_progress_bar=False,
        logger=False, enable_checkpointing=False,
    ).fit(model, train_dataloaders=TDL(DS(), batch_size=8))

    enc_after = [p.detach() for p in model.encoder.parameters()]
    head_after = [p.detach() for p in model.head.parameters()]
    return (
        _delta([b.numpy() for b in enc_before], [a.numpy() for a in enc_after]),
        _delta([b.numpy() for b in head_before], [a.numpy() for a in head_after]),
    )


def _delta(before: list[np.ndarray], after: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(a - b) for a, b in zip(after, before)))


def test_per_module_lrs_diverge(parity_backend):
    if parity_backend == "tinygrad":
        enc_delta, head_delta = _run_tinygrad()
    else:
        enc_delta, head_delta = _run_torch()

    assert head_delta > 0
    # head LR is 10000x larger; a 10x weight-delta ratio is a very loose lower bound.
    assert head_delta > 10 * enc_delta, (
        f"head_delta {head_delta:.6f} not >> enc_delta {enc_delta:.6f}"
    )
