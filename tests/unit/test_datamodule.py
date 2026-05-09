"""Unit: LightningDataModule lifecycle gates."""
from __future__ import annotations

from tinygrad_lightning.data.datamodule import LightningDataModule


class _SpyDM(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def prepare_data(self):
        self.calls.append("prepare_data")

    def setup(self, stage):
        self.calls.append(f"setup({stage})")

    def train_dataloader(self):
        self.calls.append("train_dataloader")
        return []

    def val_dataloader(self):
        self.calls.append("val_dataloader")
        return None

    def teardown(self, stage):
        self.calls.append(f"teardown({stage})")


def test_prepare_data_runs_at_most_once():
    dm = _SpyDM()
    dm._maybe_prepare()
    dm._maybe_prepare()
    dm._maybe_prepare()
    assert dm.calls.count("prepare_data") == 1


def test_setup_runs_once_per_stage():
    dm = _SpyDM()
    dm._maybe_setup("fit")
    dm._maybe_setup("fit")
    dm._maybe_setup("validate")
    assert dm.calls.count("setup(fit)") == 1
    assert dm.calls.count("setup(validate)") == 1
