"""MLFlowLogger — lazy-imports ``mlflow``.

Designed to mirror ``lightning.pytorch.loggers.MLFlowLogger`` for the subset
relevant to parity tests:

- ``experiment_name``: experiment to nest the run under
- ``tracking_uri``: pass ``file://...`` for local file-store testing
- ``run_name``: optional, otherwise mlflow autogenerates
- ``tags``: dict of mlflow tags applied to the run

Construction is cheap; the run is started on the first log call. We DON'T
install ``mlflow`` as a hard dependency — it's an extras_require under
``tinygrad-lightning[mlflow]``.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import Logger


class MLFlowLogger(Logger):
    def __init__(
        self,
        experiment_name: str = "tinygrad_lightning_default",
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        try:
            import mlflow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MLFlowLogger requires `mlflow`. Install with `pip install tinygrad-lightning[mlflow]`."
            ) from e

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.tags = dict(tags) if tags else {}
        self._run_id: str | None = None
        self._client = None

    @property
    def name(self) -> str:
        return self.experiment_name

    @property
    def version(self) -> str:
        return self._run_id or "<not-yet-started>"

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def _ensure_run(self) -> None:
        if self._run_id is not None:
            return
        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        # We use mlflow.start_run with a fresh run id; we own it for the
        # lifetime of this logger and call mlflow.end_run() in finalize.
        run = mlflow.start_run(run_name=self.run_name, tags=self.tags or None)
        self._run_id = run.info.run_id

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        import mlflow

        self._ensure_run()
        # mlflow rejects non-finite values; filter them.
        for k, v in metrics.items():
            try:
                value = float(v)
            except (TypeError, ValueError):
                continue
            if value != value or value in (float("inf"), float("-inf")):
                continue
            mlflow.log_metric(k, value, step=int(step or 0))

    def log_hyperparams(self, params: Mapping[str, object]) -> None:
        import mlflow

        self._ensure_run()
        # mlflow.log_param expects strings/numerics; coerce others to str.
        casted = {k: _coerce(v) for k, v in params.items()}
        mlflow.log_params(casted)

    def save(self) -> None:
        return None

    def finalize(self, status: str) -> None:
        import mlflow

        if self._run_id is None:
            return
        ml_status = {"success": "FINISHED", "failed": "FAILED"}.get(status, "FINISHED")
        mlflow.end_run(status=ml_status)
        # Keep _run_id around so callers can introspect the run after fit
        # (matches PL's MLFlowLogger semantics).


def _coerce(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return str(v)
