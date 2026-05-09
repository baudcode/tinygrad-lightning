"""LightningDataModule — encapsulates dataset/dataloader lifecycle.

Mirrors ``lightning.pytorch.LightningDataModule`` for the hooks ``Trainer``
calls today:

- ``prepare_data()`` — invoked once before any setup (downloads, etc.)
- ``setup(stage)`` — invoked once per stage in ``{"fit", "validate", "test"}``
  before dataloaders are requested
- ``train_dataloader()`` / ``val_dataloader()`` / ``test_dataloader()``
- ``teardown(stage)`` — invoked at the end of the stage

The trainer guarantees ``prepare_data`` runs at most once per ``DataModule``
instance per process; ``setup`` runs at most once per stage.
"""
from __future__ import annotations


class LightningDataModule:
    def __init__(self) -> None:
        self._prepare_data_called: bool = False
        self._setup_stages_called: set[str] = set()

    # ---- override these -----------------------------------------------

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str) -> None:
        return None

    def train_dataloader(self):  # pragma: no cover - abstract
        raise NotImplementedError("override train_dataloader()")

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def teardown(self, stage: str) -> None:
        return None

    # ---- internal: trainer-driven invocation guards --------------------

    def _maybe_prepare(self) -> None:
        if self._prepare_data_called:
            return
        self.prepare_data()
        self._prepare_data_called = True

    def _maybe_setup(self, stage: str) -> None:
        if stage in self._setup_stages_called:
            return
        self.setup(stage)
        self._setup_stages_called.add(stage)
