"""Tiny MLP regression example exercising most of the tinygrad-lightning surface.

Run with:

    python examples/mlp_regression.py
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import AdamW

import tinygrad_lightning as L


class _SyntheticDataset:
    def __init__(self, n: int = 256, in_dim: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n, in_dim)).astype(np.float32)
        w_true = rng.standard_normal(in_dim).astype(np.float32)
        self.y = (self.X @ w_true + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TinyMLP(L.LightningModule):
    def __init__(self, in_dim: int = 8, hidden: int = 32, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters({"in_dim": in_dim, "hidden": hidden, "lr": lr})
        self.w1 = Tensor.kaiming_uniform(in_dim, hidden)
        self.b1 = Tensor.zeros(hidden)
        self.w2 = Tensor.kaiming_uniform(hidden, 1)
        self.b2 = Tensor.zeros(1)

    def forward(self, x):
        return (x @ self.w1 + self.b1).relu() @ self.w2 + self.b2

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = ((pred - y) ** 2).mean()
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = ((pred - y) ** 2).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = AdamW([self.w1, self.b1, self.w2, self.b2], lr=self.hparams.lr)
        sched = L.CosineAnnealingLR(opt, T_max=64)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


def main() -> None:
    L.seed_everything(42)

    train_ds = _SyntheticDataset(n=256, seed=0)
    val_ds = _SyntheticDataset(n=64, seed=1)
    train_dl = L.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = L.DataLoader(val_ds, batch_size=32)

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=4,
        precision="16-mixed",
        callbacks=[
            L.ModelCheckpoint(dirpath="lightning_logs/checkpoints", save_top_k=1),
            L.EarlyStopping(monitor="val_loss", patience=3),
            L.LearningRateMonitor(logging_interval="step"),
        ],
        logger=L.CSVLogger(save_dir="lightning_logs"),
    )
    trainer.fit(TinyMLP(), train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
