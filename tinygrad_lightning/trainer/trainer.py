"""Trainer — drives fit/validate/test loops over a LightningModule.

Phase 1 features:
- ``fit(model, train_dataloaders=, val_dataloaders=, datamodule=, ckpt_path=)``
  (``datamodule`` accepted only as a stand-in for now; full lifecycle in Phase 4;
   ``ckpt_path`` accepted but raises until Phase 3.)
- ``max_epochs`` / ``max_steps`` early termination
- ``limit_train_batches`` / ``limit_val_batches`` (int or fraction)
- ``gradient_clip_val`` (norm-based)
- ``accumulate_grad_batches``
- ``accelerator`` / ``devices`` resolution

Not yet supported (raise on use):
- ``precision != "32-true"`` (Phase 5)
- ``logger`` argument (Phase 4)
- ``ckpt_path`` resume (Phase 3)
- LR schedulers in ``configure_optimizers`` return value (Phase 2)
- ``EarlyStopping`` / ``ModelCheckpoint`` callbacks (Phases 3+4)
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from dataclasses import dataclass

from ..callbacks.base import Callback
from ..core.accelerator import resolve_accelerator, set_default_device
from ..core.optimizer import LightningOptimizer
from ..core.precision import PrecisionPlugin
from ..module import LightningModule


@dataclass
class LRSchedulerConfig:
    """Per-scheduler config matching PL's lr_scheduler dict shape."""

    scheduler: object
    interval: str = "epoch"   # "step" | "epoch"
    frequency: int = 1
    monitor: str | None = None    # used by ReduceLROnPlateau (deferred)
    name: str | None = None


def _resolve_limit(limit: int | float, total: int) -> int:
    if isinstance(limit, float):
        if not 0.0 <= limit <= 1.0:
            raise ValueError(f"limit_*_batches as float must be in [0, 1], got {limit}")
        return int(math.ceil(total * limit))
    if isinstance(limit, int):
        if limit < 0:
            return total
        return min(limit, total)
    raise TypeError(f"limit_*_batches must be int or float, got {type(limit).__name__}")


def _clip_grad_norm(parameters: Sequence[Any], max_norm: float) -> None:
    """In-place gradient clipping by global L2 norm."""
    import numpy as np

    grads = [p.grad for p in parameters if getattr(p, "grad", None) is not None]
    if not grads:
        return
    total_sq = 0.0
    for g in grads:
        arr = g.detach().numpy()
        total_sq += float(np.sum(arr * arr))
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm or total_norm == 0.0:
        return
    scale = max_norm / (total_norm + 1e-6)
    for p in parameters:
        if getattr(p, "grad", None) is not None:
            p.grad = p.grad * scale


def _coerce_scheduler_entry(entry: Any) -> LRSchedulerConfig:
    """Accept either a scheduler instance or a PL-shaped dict."""
    if isinstance(entry, LRSchedulerConfig):
        return entry
    if isinstance(entry, dict):
        if "scheduler" not in entry:
            raise RuntimeError("lr_scheduler dict missing 'scheduler' key")
        return LRSchedulerConfig(
            scheduler=entry["scheduler"],
            interval=entry.get("interval", "epoch"),
            frequency=int(entry.get("frequency", 1)),
            monitor=entry.get("monitor"),
            name=entry.get("name"),
        )
    return LRSchedulerConfig(scheduler=entry)


def _normalize_optimizers(configured: Any):
    """Polymorphic ``configure_optimizers`` return -> (optimizers, scheduler_configs).

    Returns a tuple ``(list[LightningOptimizer], list[LRSchedulerConfig])``.
    """
    if configured is None:
        raise RuntimeError("configure_optimizers returned None")

    optimizers: list = []
    scheduler_configs: list[LRSchedulerConfig] = []

    if isinstance(configured, tuple) and len(configured) == 2 and isinstance(configured[0], (list, tuple)):
        # ([opts], [scheds]) form
        optimizers = list(configured[0])
        scheduler_configs = [_coerce_scheduler_entry(s) for s in configured[1]]
    elif isinstance(configured, dict):
        opt = configured.get("optimizer")
        if opt is None:
            raise RuntimeError("configure_optimizers dict missing 'optimizer' key")
        optimizers = [opt]
        sched = configured.get("lr_scheduler")
        if sched is not None:
            scheduler_configs = [_coerce_scheduler_entry(sched)]
    elif isinstance(configured, list):
        optimizers = list(configured)
    else:
        optimizers = [configured]

    wrapped = [o if isinstance(o, LightningOptimizer) else LightningOptimizer(o) for o in optimizers]
    return wrapped, scheduler_configs


class Trainer:
    def __init__(
        self,
        accelerator: str = "auto",
        devices: int | str | list[int] = 1,
        max_epochs: int | None = None,
        max_steps: int = -1,
        limit_train_batches: int | float = 1.0,
        limit_val_batches: int | float = 1.0,
        val_check_interval: int | float = 1.0,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
        accumulate_grad_batches: int = 1,
        precision: str = "32-true",
        callbacks: list[Callback] | None = None,
        logger: Any = True,
        default_root_dir: str | None = None,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        deterministic: bool = False,
    ) -> None:
        self.precision_plugin = PrecisionPlugin(precision)
        if gradient_clip_algorithm not in ("norm",):
            raise NotImplementedError(f"gradient_clip_algorithm={gradient_clip_algorithm!r} not supported")
        if accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")

        self.accelerator_str = accelerator
        self.devices = devices
        self.device = resolve_accelerator(accelerator, devices)
        set_default_device(self.device)

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.val_check_interval = val_check_interval
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.default_root_dir = default_root_dir or "lightning_logs"
        self.enable_checkpointing = enable_checkpointing
        self.enable_progress_bar = enable_progress_bar
        self.deterministic = deterministic

        self.loggers = self._resolve_loggers(logger)

        self.callbacks: list[Callback] = list(callbacks or [])
        if enable_progress_bar and not any(_is_progress_bar(c) for c in self.callbacks):
            from ..callbacks.progress import TQDMProgressBar
            self.callbacks.append(TQDMProgressBar())

        # runtime state
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.callback_metrics: dict[str, float] = {}
        self.logged_metrics: dict[str, float] = {}
        self.estimated_train_batches_per_epoch: int = 0
        self.estimated_val_batches_per_epoch: int = 0
        self.optimizers: list[LightningOptimizer] = []
        self.lr_scheduler_configs: list[LRSchedulerConfig] = []
        self._should_stop: bool = False
        self._hparams_logged: bool = False

    @property
    def logger(self):
        """First (or only) logger; convenience for PL parity."""
        return self.loggers[0] if self.loggers else None

    def _resolve_loggers(self, logger):
        if logger is False or logger is None:
            return []
        if logger is True:
            from ..loggers.csv import CSVLogger
            return [CSVLogger(save_dir=self.default_root_dir)]
        if isinstance(logger, (list, tuple)):
            return list(logger)
        return [logger]

    def _log_to_loggers(self, metrics: dict, step: int) -> None:
        if not metrics:
            return
        for lg in self.loggers:
            lg.log_metrics(metrics, step=step)

    def _log_hparams_once(self, model) -> None:
        if self._hparams_logged or not self.loggers:
            return
        if hasattr(model, "hparams") and model.hparams:
            for lg in self.loggers:
                try:
                    lg.log_hyperparams(model.hparams)
                except Exception:  # pragma: no cover - logger-specific
                    pass
        self._hparams_logged = True

    # ---- public entry points ------------------------------------------

    def fit(
        self,
        model: LightningModule,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=None,
        ckpt_path: str | None = None,
    ) -> None:
        if datamodule is not None and (train_dataloaders is not None or val_dataloaders is not None):
            raise ValueError("pass either dataloaders or datamodule, not both")
        if datamodule is not None:
            self._invoke_datamodule_lifecycle(datamodule, "fit")
            train_dataloaders, val_dataloaders = self._unpack_datamodule(datamodule)

        if train_dataloaders is None:
            raise ValueError("fit() requires train_dataloaders or a datamodule with train_dataloader()")

        self._apply_precision_to_dataloader(train_dataloaders)
        self._apply_precision_to_dataloader(val_dataloaders)

        model._attach_trainer(self)
        optimizers, scheduler_configs = _normalize_optimizers(model.configure_optimizers())
        self.optimizers = optimizers
        self.lr_scheduler_configs = scheduler_configs

        start_epoch = 0
        if ckpt_path is not None:
            from ..callbacks.checkpoint import load_checkpoint

            meta = load_checkpoint(self, model, ckpt_path)
            # `current_epoch` in the checkpoint is the epoch that just finished;
            # resume on the next one.
            start_epoch = self.current_epoch + 1 if meta else 0

        from tinygrad import Tensor
        Tensor.training = True

        for cb in self.callbacks:
            cb.setup(self, model, "fit")
        self._log_hparams_once(model)
        for cb in self.callbacks:
            cb.on_train_start(self, model)

        max_epochs = self.max_epochs if self.max_epochs is not None else 1
        status = "success"
        try:
            for epoch in range(start_epoch, max_epochs):
                self.current_epoch = epoch
                self._train_one_epoch(model, train_dataloaders)
                self._step_schedulers(interval="epoch")
                if val_dataloaders is not None:
                    self._validate_one_epoch(model, val_dataloaders)
                for lg in self.loggers:
                    lg.save()
                if self._should_stop:
                    break
        except Exception:
            status = "failed"
            raise
        finally:
            for cb in self.callbacks:
                cb.on_train_end(self, model)
            for cb in self.callbacks:
                cb.teardown(self, model, "fit")
            for lg in self.loggers:
                lg.finalize(status)
            if datamodule is not None:
                datamodule.teardown("fit")
            Tensor.training = False

    def validate(self, model: LightningModule, dataloaders=None, datamodule=None, ckpt_path: str | None = None) -> None:
        if datamodule is not None:
            self._invoke_datamodule_lifecycle(datamodule, "validate")
            dataloaders = datamodule.val_dataloader()
        if dataloaders is None:
            raise ValueError("validate() requires dataloaders or a datamodule")
        model._attach_trainer(self)
        if ckpt_path is not None:
            from ..callbacks.checkpoint import load_checkpoint
            optimizers, scheduler_configs = _normalize_optimizers(model.configure_optimizers())
            self.optimizers = optimizers
            self.lr_scheduler_configs = scheduler_configs
            load_checkpoint(self, model, ckpt_path)
        from tinygrad import Tensor
        Tensor.training = False
        try:
            self._validate_one_epoch(model, dataloaders)
        finally:
            if datamodule is not None:
                datamodule.teardown("validate")

    def test(self, model: LightningModule, dataloaders=None, datamodule=None, ckpt_path: str | None = None) -> None:
        if datamodule is not None:
            self._invoke_datamodule_lifecycle(datamodule, "test")
            dataloaders = datamodule.test_dataloader()
        if dataloaders is None:
            raise ValueError("test() requires dataloaders or a datamodule")
        model._attach_trainer(self)
        if ckpt_path is not None:
            from ..callbacks.checkpoint import load_checkpoint
            optimizers, scheduler_configs = _normalize_optimizers(model.configure_optimizers())
            self.optimizers = optimizers
            self.lr_scheduler_configs = scheduler_configs
            load_checkpoint(self, model, ckpt_path)
        from tinygrad import Tensor
        Tensor.training = False
        try:
            self._test_one_epoch(model, dataloaders)
        finally:
            if datamodule is not None:
                datamodule.teardown("test")

    # ---- inner loops --------------------------------------------------

    def _train_one_epoch(self, model: LightningModule, train_dl) -> None:
        optimizers = self.optimizers
        from tinygrad import Tensor
        Tensor.training = True

        total = len(train_dl) if hasattr(train_dl, "__len__") else 0
        limit = _resolve_limit(self.limit_train_batches, total) if total else -1
        self.estimated_train_batches_per_epoch = limit if limit >= 0 else 0

        model._set_stage("train")
        for cb in self.callbacks:
            cb.on_train_epoch_start(self, model)

        accum = self.accumulate_grad_batches
        accum_count = 0
        for batch_idx, batch in enumerate(train_dl):
            if limit >= 0 and batch_idx >= limit:
                break
            for cb in self.callbacks:
                cb.on_train_batch_start(self, model, batch, batch_idx)

            with self.precision_plugin.autocast(model):
                output = model.training_step(batch, batch_idx)
            loss = _extract_loss(output)
            self.precision_plugin.scale_loss(loss / accum).backward()

            accum_count += 1
            do_step = (accum_count >= accum) or (limit >= 0 and (batch_idx + 1) >= limit)
            if do_step:
                self.precision_plugin.unscale_grads(optimizers)
                if self.gradient_clip_val is not None:
                    for opt in optimizers:
                        _clip_grad_norm(opt.params, self.gradient_clip_val)
                grads_finite = self.precision_plugin.check_grads_finite(optimizers)
                if grads_finite:
                    for opt in optimizers:
                        opt.step()
                self.precision_plugin.update_scaler(found_inf=not grads_finite)
                for opt in optimizers:
                    opt.zero_grad()
                accum_count = 0
                self.global_step += 1
                if grads_finite:
                    self._step_schedulers(interval="step")
                if 0 < self.max_steps <= self.global_step:
                    self._should_stop = True

            self._flush_step_metrics(model, "train")
            for cb in self.callbacks:
                cb.on_train_batch_end(self, model, output, batch, batch_idx)

            if self._should_stop:
                break

        self._flush_epoch_metrics(model, "train")
        for cb in self.callbacks:
            cb.on_train_epoch_end(self, model)

    def _validate_one_epoch(self, model: LightningModule, val_dl) -> None:
        from tinygrad import Tensor
        was_training = Tensor.training
        Tensor.training = False

        total = len(val_dl) if hasattr(val_dl, "__len__") else 0
        limit = _resolve_limit(self.limit_val_batches, total) if total else -1
        self.estimated_val_batches_per_epoch = limit if limit >= 0 else 0

        model._set_stage("val")
        for cb in self.callbacks:
            cb.on_validation_epoch_start(self, model)

        for batch_idx, batch in enumerate(val_dl):
            if limit >= 0 and batch_idx >= limit:
                break
            for cb in self.callbacks:
                cb.on_validation_batch_start(self, model, batch, batch_idx)
            output = model.validation_step(batch, batch_idx)
            self._flush_step_metrics(model, "val")
            for cb in self.callbacks:
                cb.on_validation_batch_end(self, model, output, batch, batch_idx)

        self._flush_epoch_metrics(model, "val")
        for cb in self.callbacks:
            cb.on_validation_epoch_end(self, model)
        Tensor.training = was_training

    def _test_one_epoch(self, model: LightningModule, test_dl) -> None:
        model._set_stage("test")
        for batch_idx, batch in enumerate(test_dl):
            output = model.test_step(batch, batch_idx)
            self._flush_step_metrics(model, "test")
        self._flush_epoch_metrics(model, "test")

    # ---- schedulers ----------------------------------------------------

    def _step_schedulers(self, *, interval: str) -> None:
        for cfg in self.lr_scheduler_configs:
            if cfg.interval != interval:
                continue
            counter = self.global_step if interval == "step" else (self.current_epoch + 1)
            if cfg.frequency > 1 and counter % cfg.frequency != 0:
                continue
            cfg.scheduler.step()

    # ---- metrics plumbing ---------------------------------------------

    def _flush_step_metrics(self, model: LightningModule, stage: str) -> None:
        new = model._drain_step_metrics(stage)
        if not new:
            return
        for k, v in new.items():
            self.callback_metrics[k] = v
            self.logged_metrics[k] = v
        self._log_to_loggers(new, step=self.global_step)

    def _flush_epoch_metrics(self, model: LightningModule, stage: str) -> None:
        new = model._drain_epoch_metrics(stage)
        if not new:
            return
        for k, v in new.items():
            self.callback_metrics[k] = v
            self.logged_metrics[k] = v
        # Epoch metrics are logged at the current global_step (the last train step
        # in the epoch). PL uses the same convention.
        self._log_to_loggers(new, step=self.global_step)

    # ---- helpers ------------------------------------------------------

    def _apply_precision_to_dataloader(self, dataloader) -> None:
        """If ``dataloader`` is a tinygrad-lightning DataLoader and the user
        didn't pin a dtype, set it to the precision plugin's compute dtype so
        float batches arrive on the device in the right precision.
        """
        if dataloader is None:
            return
        if self.precision_plugin.precision == "32-true":
            return
        if hasattr(dataloader, "dtype") and dataloader.dtype is None:
            dataloader.dtype = self.precision_plugin.compute_dtype

    def _invoke_datamodule_lifecycle(self, datamodule, stage: str) -> None:
        if hasattr(datamodule, "_maybe_prepare"):
            datamodule._maybe_prepare()
        elif hasattr(datamodule, "prepare_data"):
            datamodule.prepare_data()
        if hasattr(datamodule, "_maybe_setup"):
            datamodule._maybe_setup(stage)
        elif hasattr(datamodule, "setup"):
            datamodule.setup(stage)

    def _unpack_datamodule(self, datamodule):
        train = datamodule.train_dataloader() if hasattr(datamodule, "train_dataloader") else None
        val = datamodule.val_dataloader() if hasattr(datamodule, "val_dataloader") else None
        return train, val


def _extract_loss(output):
    """training_step may return a Tensor (loss) or a dict with key 'loss'."""
    if isinstance(output, dict):
        if "loss" not in output:
            raise RuntimeError("training_step dict must contain 'loss' key")
        return output["loss"]
    return output


def _is_progress_bar(callback) -> bool:
    from ..callbacks.progress import TQDMProgressBar
    return isinstance(callback, TQDMProgressBar)
