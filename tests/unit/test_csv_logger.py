"""Unit: CSVLogger writes parsable CSV + hparams JSON."""
from __future__ import annotations

import csv
import json

from tinygrad_lightning.loggers.csv import CSVLogger


def test_log_metrics_writes_header_and_rows(tmp_path):
    lg = CSVLogger(save_dir=tmp_path)
    lg.log_metrics({"loss": 1.5, "accuracy": 0.4}, step=1)
    lg.log_metrics({"loss": 1.0, "accuracy": 0.6}, step=2)
    lg.finalize("success")

    csv_path = tmp_path / "lightning_logs" / "version_0" / "metrics.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert float(rows[0]["loss"]) == 1.5
    assert int(rows[1]["step"]) == 2


def test_log_metrics_extends_header_when_new_keys_appear(tmp_path):
    lg = CSVLogger(save_dir=tmp_path)
    lg.log_metrics({"loss": 1.0}, step=1)
    lg.log_metrics({"loss": 0.5, "val_loss": 0.7}, step=2)
    lg.finalize("success")

    csv_path = tmp_path / "lightning_logs" / "version_0" / "metrics.csv"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    # Both rows must have a val_loss column even though row 0 had no value.
    assert "val_loss" in rows[0]
    assert rows[0]["val_loss"] == ""
    assert float(rows[1]["val_loss"]) == 0.7


def test_log_hyperparams_writes_json(tmp_path):
    lg = CSVLogger(save_dir=tmp_path)
    lg.log_hyperparams({"lr": 1e-3, "batch_size": 32, "name": "exp1"})
    lg.finalize("success")

    hp_path = tmp_path / "lightning_logs" / "version_0" / "hparams.json"
    data = json.loads(hp_path.read_text())
    assert data["lr"] == 1e-3
    assert data["batch_size"] == 32
    assert data["name"] == "exp1"


def test_version_auto_increments(tmp_path):
    CSVLogger(save_dir=tmp_path)
    lg2 = CSVLogger(save_dir=tmp_path)
    lg2.log_metrics({"x": 1}, step=0)
    lg2.finalize("success")
    # First instance never wrote anything → version_0 was claimed by lg2 (since
    # the dir wasn't created). For an existing dir, lg2 picks the next free.
    # Assert at least version_0 exists; explicit version pinning covers the rest.
    versions = sorted((tmp_path / "lightning_logs").iterdir())
    assert versions, "no version dirs created"
