"""TensorBoardLogger — wraps ``tensorboardX`` (lazy-imported)."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .base import Logger


class TensorBoardLogger(Logger):
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "lightning_logs",
        version: int | str | None = None,
    ) -> None:
        try:
            import tensorboardX  # noqa: F401
        except ImportError as e:
            raise ImportError("TensorBoardLogger requires `tensorboardX`. Install with `pip install tensorboardX`.") from e

        self.save_dir = Path(save_dir)
        self._name = name
        self._version = version if version is not None else self._next_version()
        self._writer = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int | str:
        return self._version

    @property
    def log_dir(self) -> Path:
        return self.save_dir / self._name / f"version_{self._version}"

    def _next_version(self) -> int:
        root = self.save_dir / self._name
        if not root.exists():
            return 0
        existing = [
            int(d.name.removeprefix("version_"))
            for d in root.iterdir()
            if d.is_dir() and d.name.startswith("version_") and d.name.removeprefix("version_").isdigit()
        ]
        return (max(existing) + 1) if existing else 0

    @property
    def writer(self):
        if self._writer is None:
            import tensorboardX
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = tensorboardX.SummaryWriter(str(self.log_dir))
        return self._writer

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            try:
                value = float(v)
            except (TypeError, ValueError):
                continue
            self.writer.add_scalar(k, value, step or 0)

    def log_hyperparams(self, params: Mapping[str, object]) -> None:
        # tensorboardX's add_hparams requires a metric_dict; we pass an empty
        # one to register the hparams pane.
        scalar_params = {k: v for k, v in params.items() if isinstance(v, (int, float, bool, str))}
        if scalar_params:
            self.writer.add_hparams(scalar_params, {})

    def save(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def finalize(self, status: str) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
