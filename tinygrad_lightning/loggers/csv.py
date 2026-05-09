"""CSVLogger — append metric rows to a CSV file.

On-disk layout (matching PL's CSVLogger format):

    <save_dir>/<name>/version_<N>/
        metrics.csv          # one row per `log_metrics` call; columns = metric names + step + epoch
        hparams.json         # JSON dump of the hparams dict (PL uses YAML; JSON keeps deps light)

A new ``version_<N>`` directory is created per ``CSVLogger`` instance unless
``version`` is supplied. Stable across resumes when the user passes the same
``version``.
"""
from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path

from .base import Logger


class CSVLogger(Logger):
    METRICS_FILE = "metrics.csv"
    HPARAMS_FILE = "hparams.json"

    def __init__(
        self,
        save_dir: str | Path,
        name: str = "lightning_logs",
        version: int | str | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self._name = name
        self._version = self._resolve_version(version)
        self._fieldnames: list[str] | None = None
        self._csv_handle = None
        self._csv_writer = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int | str:
        return self._version

    @property
    def log_dir(self) -> Path:
        return self.save_dir / self._name / f"version_{self._version}"

    def _resolve_version(self, version) -> int | str:
        if version is not None:
            return version
        root = self.save_dir / self._name
        if not root.exists():
            return 0
        existing = [
            int(d.name.removeprefix("version_"))
            for d in root.iterdir()
            if d.is_dir() and d.name.startswith("version_") and d.name.removeprefix("version_").isdigit()
        ]
        return (max(existing) + 1) if existing else 0

    def _ensure_dir(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._ensure_dir()
        row = {k: _scalar(v) for k, v in metrics.items()}
        if step is not None:
            row["step"] = int(step)

        # On first call, decide the header from observed keys; later calls extend
        # the header on the fly (re-writing the file is too costly for an append-
        # only logger; instead we accept that older rows lack new columns).
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            self._open_writer(write_header=True)
        else:
            new_keys = [k for k in row.keys() if k not in self._fieldnames]
            if new_keys:
                self._fieldnames = list(self._fieldnames) + new_keys
                self._reopen_with_new_header()

        complete = {k: row.get(k, "") for k in self._fieldnames}
        self._csv_writer.writerow(complete)
        self._csv_handle.flush()

    def log_hyperparams(self, params: Mapping[str, object]) -> None:
        self._ensure_dir()
        path = self.log_dir / self.HPARAMS_FILE
        with path.open("w") as f:
            json.dump({k: _to_json(v) for k, v in params.items()}, f, indent=2, default=str)

    def save(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.flush()

    def finalize(self, status: str) -> None:
        if self._csv_handle is not None:
            self._csv_handle.flush()
            self._csv_handle.close()
            self._csv_handle = None
            self._csv_writer = None

    # ---- internals -----------------------------------------------------

    def _open_writer(self, write_header: bool) -> None:
        path = self.log_dir / self.METRICS_FILE
        self._csv_handle = path.open("a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self._fieldnames)
        if write_header and self._csv_handle.tell() == 0:
            self._csv_writer.writeheader()

    def _reopen_with_new_header(self) -> None:
        # Re-read existing rows, rewrite with the extended header.
        path = self.log_dir / self.METRICS_FILE
        if self._csv_handle is not None:
            self._csv_handle.close()

        existing_rows: list[dict] = []
        if path.exists():
            with path.open() as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})

        self._open_writer(write_header=False)


def _scalar(v):
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return float(v) if isinstance(v, (int, float)) else v


def _to_json(v):
    """Best-effort JSON-friendly conversion for hparams values."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            pass
    return str(v)
