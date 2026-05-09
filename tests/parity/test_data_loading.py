"""Parity: LightningDataModule lifecycle (prepare_data → setup → dataloader)."""
from __future__ import annotations

import numpy as np

from ._helpers import ArrayDataset, make_regression_data


def _build_tg_dm():
    import tinygrad_lightning as L

    class SpyDM(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.calls: list[str] = []

        def prepare_data(self):
            self.calls.append("prepare_data")

        def setup(self, stage):
            self.calls.append(f"setup({stage})")
            X, y = make_regression_data(seed=0)
            self._train_dl = L.DataLoader(ArrayDataset(X, y), batch_size=8)

        def train_dataloader(self):
            self.calls.append("train_dataloader")
            return self._train_dl

        def teardown(self, stage):
            self.calls.append(f"teardown({stage})")

    return SpyDM


def _build_torch_dm():
    import lightning.pytorch as PL
    import torch
    from torch.utils.data import DataLoader as TDL
    from torch.utils.data import Dataset as TDS

    class TArrayDS(TDS):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

    class SpyDM(PL.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.calls: list[str] = []

        def prepare_data(self):
            self.calls.append("prepare_data")

        def setup(self, stage):
            self.calls.append(f"setup({stage})")
            X, y = make_regression_data(seed=0)
            self._train_dl = TDL(TArrayDS(X, y), batch_size=8)

        def train_dataloader(self):
            self.calls.append("train_dataloader")
            return self._train_dl

        def teardown(self, stage):
            self.calls.append(f"teardown({stage})")

    return SpyDM


def _tg_module():
    from tinygrad import Tensor
    from tinygrad.nn.optim import SGD

    import tinygrad_lightning as L

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

    return M


def _torch_module():
    import lightning.pytorch as PL
    import torch

    class M(PL.LightningModule):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 1)

        def training_step(self, batch, batch_idx):
            X, y = batch
            pred = self.fc(X.float()).squeeze(-1)
            return torch.nn.functional.mse_loss(pred, y.float())

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    return M


def test_datamodule_lifecycle_invoked(parity_backend):
    if parity_backend == "tinygrad":
        import tinygrad_lightning as L

        L.seed_everything(0)
        dm = _build_tg_dm()()
        from ._helpers import ACCEL
        L.Trainer(accelerator=ACCEL, max_epochs=1, enable_progress_bar=False, logger=False).fit(
            _tg_module()(), datamodule=dm
        )
    else:
        import lightning.pytorch as PL

        PL.seed_everything(0)
        dm = _build_torch_dm()()
        PL.Trainer(
            accelerator="cpu", devices=1, max_epochs=1, enable_progress_bar=False,
            logger=False, enable_checkpointing=False,
        ).fit(_torch_module()(), datamodule=dm)

    # Normalize PL's TrainerFn.FITTING vs tinygrad-lightning's "fit" to a common token.
    def _norm(c: str) -> str:
        return (c.replace("TrainerFn.FITTING", "fit")
                 .replace("TrainerFn.VALIDATING", "validate")
                 .replace("TrainerFn.TESTING", "test"))

    calls = [_norm(c) for c in dm.calls]
    assert calls.count("prepare_data") == 1, calls
    assert calls.count("setup(fit)") == 1, calls
    assert calls.count("train_dataloader") >= 1, calls
    # The first three lifecycle hooks must arrive in this order.
    ordered = [c for c in calls if c in ("prepare_data", "setup(fit)", "train_dataloader")]
    assert ordered[:3] == ["prepare_data", "setup(fit)", "train_dataloader"], ordered


def test_dataloader_workers_and_shuffle(parity_backend):
    """``shuffle=True`` re-orders batches across epochs (smoke check, not parity)."""
    if parity_backend == "tinygrad":
        import tinygrad_lightning as L

        L.seed_everything(0)
        X, y = make_regression_data(seed=0)
        dl = L.DataLoader(ArrayDataset(X, y), batch_size=8, shuffle=True)
        e1 = [tuple(b[1].tolist()) for b in dl]
        e2 = [tuple(b[1].tolist()) for b in dl]
    else:
        import torch
        from torch.utils.data import DataLoader as TDL
        from torch.utils.data import Dataset as TDS

        torch.manual_seed(0)
        X, y = make_regression_data(seed=0)

        class DS(TDS):
            def __len__(self):
                return len(X)

            def __getitem__(self, i):
                return torch.from_numpy(X[i]), torch.tensor(y[i])

        dl = TDL(DS(), batch_size=8, shuffle=True)
        e1 = [tuple(b[1].tolist()) for b in dl]
        e2 = [tuple(b[1].tolist()) for b in dl]

    # With true shuffling, two epochs should produce different orderings (prob.
    # of equal orderings for 8 batches of 8 is negligible).
    assert e1 != e2, "shuffle did not re-order batches between epochs"
