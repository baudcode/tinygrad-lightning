# tinygrad-lightning

A high-level training loop for [tinygrad](https://github.com/tinygrad/tinygrad), with an API that mirrors [`lightning.pytorch`](https://lightning.ai/docs/pytorch/) where it makes sense.

Targets tinygrad ≥ 0.12 (Python 3.11+).

## Install

```bash
pip install tinygrad-lightning            # core
pip install tinygrad-lightning[mlflow]    # + MLFlowLogger
pip install tinygrad-lightning[tensorboard]  # + TensorBoardLogger
pip install tinygrad-lightning[test]      # all of the above + pytest + lightning + torch (for parity tests)
```

## What's in the box

| Layer | Mirrors |
| --- | --- |
| `LightningModule`, `Trainer.fit/validate/test` | PL `LightningModule`, `Trainer.fit/validate/test` |
| `Callback`, `ModelCheckpoint`, `EarlyStopping`, `LearningRateMonitor`, `TQDMProgressBar` | PL callbacks |
| `Logger`, `CSVLogger`, `TensorBoardLogger`, `MLFlowLogger` | PL loggers |
| `LightningDataModule`, `DataLoader` (`dtype=` aware) | PL data |
| `LightningOptimizer` + `param_group_optimizer` (per-module LRs) | PL optimizer wiring |
| `StepLR`, `MultiStepLR`, `CosineAnnealingLR`, `LambdaLR`, `OneCycleLR` | torch LR schedulers |
| `PrecisionPlugin` (`32-true` / `16-mixed` / `bf16-mixed`) with autocast and dynamic loss scaling | PL precision |
| `seed_everything` | PL utilities |

## Example

```python
import numpy as np
import tinygrad_lightning as L
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import AdamW


class TinyMLP(L.LightningModule):
    def __init__(self, in_dim=8, hidden=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters({"in_dim": in_dim, "hidden": hidden, "lr": lr})
        self.w1 = Tensor.kaiming_uniform(in_dim, hidden)
        self.b1 = Tensor.zeros(hidden)
        self.w2 = Tensor.kaiming_uniform(hidden, 1)
        self.b2 = Tensor.zeros(1)

    def forward(self, x):
        return (x @ self.w1 + self.b1).relu() @ self.w2 + self.b2

    def training_step(self, batch, batch_idx):
        x, y = batch  # x and y are tinygrad Tensors (in compute_dtype if AMP is on)
        pred = self(x).squeeze(-1)
        loss = ((pred - y) ** 2).mean()
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = AdamW([self.w1, self.b1, self.w2, self.b2], lr=self.hparams.lr)
        sched = L.CosineAnnealingLR(opt, T_max=64)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class _DS:
    def __init__(self, n=128):
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((n, 8)).astype(np.float32)
        w = rng.standard_normal(8).astype(np.float32)
        self.y = (self.X @ w).astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


L.seed_everything(42)

trainer = L.Trainer(
    accelerator="auto",          # METAL / NV / CUDA / CLANG, in that order
    max_epochs=4,
    precision="16-mixed",        # autocast + dynamic loss scaling
    callbacks=[
        L.ModelCheckpoint(dirpath="ckpts", monitor=None, save_top_k=1),
        L.EarlyStopping(monitor="loss", patience=3),
        L.LearningRateMonitor(logging_interval="step"),
    ],
    logger=L.CSVLogger(save_dir="logs"),
)

# DataLoader's dtype is auto-set by the trainer when precision != 32-true.
trainer.fit(TinyMLP(), train_dataloaders=L.DataLoader(_DS(), batch_size=8))
```

## Design notes

The full design + per-phase plan lives at
[`docs/plans/2026-05-06-tinygrad-lightning-pl-parity-design.md`](docs/plans/2026-05-06-tinygrad-lightning-pl-parity-design.md).
Highlights worth knowing:

- **Single device, no DDP.** `devices > 1` raises `NotImplementedError`. Resolution: `accelerator="auto"` picks the first available GPU-class device (METAL, CUDA, NV, AMD, ...) else CPU; `"cpu"`/`"gpu"` work as PL-style aliases.
- **Checkpoint format.** Safetensors (model + optimizer state) plus a JSON sidecar (epoch, global step, scheduler/precision state, hparams). PL `.ckpt` files are intentionally not loadable.
- **Resume.** Adam, SGD-with-momentum, and SGD all reproduce the contiguous training trajectory within `rtol=1e-3` after a checkpoint→resume round-trip. The dynamic loss scaler state is also persisted.
- **AMP.** `precision="16-mixed"`/`"bf16-mixed"` auto-casts module Tensors to compute dtype during `training_step` and (when batches arrive via tinygrad-lightning's `DataLoader`) auto-casts float arrays to that dtype too. Backward routes fp32 grads back to the original master weights via tinygrad's differentiable `cast`.
- **MLflow is optional.** `MLFlowLogger` lazy-imports `mlflow`; raises a clean `ImportError` if you didn't install the extras.

## Tests

The project ships a parity test suite that exercises each feature against
both `tinygrad_lightning` and `lightning.pytorch`. Run locally:

```bash
pip install -r requirements-test.txt
pytest tests/                            # full suite
pytest tests/parity/                     # cross-backend parity
pytest tests/numeric/                    # seed-pinned regression
TG_TEST_ACCEL=auto pytest tests/         # use whichever device tinygrad finds
```

## Status

WIP. The current pass rate is 94 (macOS METAL) / 93 (Linux NV/CUDA) of 97 tests; the rest skip on platforms without an applicable feature (no GPU, no clang, no mlflow). See the design doc for what's *not* in scope: DDP, the full PL hook surface, fault tolerance, loading PL `.ckpt` files.
