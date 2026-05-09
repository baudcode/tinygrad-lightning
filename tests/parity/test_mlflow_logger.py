"""Parity: MLFlowLogger writes the same set of metric keys + hparams in both backends.

Uses an isolated file-store ``tracking_uri`` per test so runs don't leak across
parametrized invocations.
"""
from __future__ import annotations

import numpy as np

from ._helpers import ArrayDataset, make_regression_data

EPOCHS = 2
BATCH_SIZE = 8


def _run_tinygrad(tracking_uri: str, exp_name: str):
    from tinygrad import Tensor
    from tinygrad.nn.optim import SGD

    import tinygrad_lightning as L

    L.seed_everything(0)

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = Tensor.kaiming_uniform(8, 1)
            self.save_hyperparameters({"lr": 1e-2, "model": "mlp"})

        def forward(self, x):
            return x @ self.w

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self(Tensor(X)).squeeze(-1)
            loss = ((pred - Tensor(y)) ** 2).mean()
            self.log("loss", loss, on_step=True, on_epoch=False)
            return loss

        def configure_optimizers(self):
            return SGD([self.w], lr=1e-2)

    logger = L.MLFlowLogger(experiment_name=exp_name, tracking_uri=tracking_uri)
    X, y = make_regression_data(seed=0)
    dl = L.DataLoader(ArrayDataset(X, y), batch_size=BATCH_SIZE)
    from ._helpers import ACCEL
    trainer = L.Trainer(
        accelerator=ACCEL,
        max_epochs=EPOCHS,
        enable_progress_bar=False,
        logger=logger,
    )
    trainer.fit(M(), train_dataloaders=dl)
    return logger.run_id


def _run_torch(tracking_uri: str, exp_name: str):
    import lightning.pytorch as PL
    import torch
    from lightning.pytorch.loggers import MLFlowLogger as PLMlf
    from torch.utils.data import DataLoader as TDL
    from torch.utils.data import Dataset as TDS

    PL.seed_everything(0)

    class M(PL.LightningModule):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 1)
            self.save_hyperparameters({"lr": 1e-2, "model": "mlp"})

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self.fc(X.float()).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, y.float())
            self.log("loss", loss, on_step=True, on_epoch=False)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    X, y = make_regression_data(seed=0)

    class DS(TDS):
        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            return torch.from_numpy(X[i]), torch.tensor(y[i])

    logger = PLMlf(experiment_name=exp_name, tracking_uri=tracking_uri)
    PL.Trainer(
        accelerator="cpu", devices=1, max_epochs=EPOCHS, enable_progress_bar=False,
        logger=logger, enable_checkpointing=False, log_every_n_steps=1,
    ).fit(M(), train_dataloaders=TDL(DS(), batch_size=BATCH_SIZE))
    return logger.run_id


def _read_run(tracking_uri: str, run_id: str):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    metrics = client.get_metric_history(run_id, "loss")
    return run, metrics


def test_metrics_logged_with_step(parity_backend, tmp_path, needs_mlflow):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    exp_name = f"parity-{parity_backend}"

    if parity_backend == "tinygrad":
        run_id = _run_tinygrad(tracking_uri, exp_name)
    else:
        run_id = _run_torch(tracking_uri, exp_name)

    run, history = _read_run(tracking_uri, run_id)
    assert "loss" in run.data.metrics, list(run.data.metrics.keys())
    assert len(history) >= EPOCHS * (64 // BATCH_SIZE), (
        f"expected ≥{EPOCHS * (64 // BATCH_SIZE)} loss samples, got {len(history)}"
    )
    # Step values are monotonic non-decreasing in both backends.
    steps = [h.step for h in history]
    assert steps == sorted(steps), steps


def test_hyperparams_logged_once(parity_backend, tmp_path, needs_mlflow):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    exp_name = f"parity-{parity_backend}"

    if parity_backend == "tinygrad":
        run_id = _run_tinygrad(tracking_uri, exp_name)
    else:
        run_id = _run_torch(tracking_uri, exp_name)

    run, _ = _read_run(tracking_uri, run_id)
    # PL stores hparams as strings; we coerce non-primitives to strings too.
    assert run.data.params.get("model") == "mlp"
    assert "lr" in run.data.params
    # Float-string roundtrip: 1e-2 may render as "0.01" in mlflow; both fine.
    assert float(run.data.params["lr"]) == 1e-2
