"""ModelCheckpoint — save model weights + optimizer state at training milestones.

On-disk format:

    <filename>.safetensors   # model.* keys + optimizer_<i>_<j>.* keys
    <filename>.meta.json     # epoch, global_step, scheduler states, hparams

This is intentionally not compatible with ``lightning.pytorch``'s ``.ckpt``
files (those are pickle); the JSON sidecar is human-readable, and safetensors
prevents pickle-deserialization risks.

Optimizer state (Adam's ``m``/``v``/``b1_t``/``b2_t``, SGD-with-momentum's
``b``) is preserved by walking the underlying tinygrad optimizer's tensor
attributes, excluding the model-owned param/buffer tensors. Resume restores
these via ``load_state_dict(strict=False)`` so optimizers that were created
without state (e.g. SGD with momentum=0) don't reject the missing keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..version import __version__
from .base import Callback


CHECKPOINT_SUFFIX = ".safetensors"
META_SUFFIX = ".meta.json"


def _sub_optimizers(optimizer) -> list:
    """Resolve a LightningOptimizer / OptimizerGroup / single optimizer into a flat list."""
    if hasattr(optimizer, "optimizer"):
        return _sub_optimizers(optimizer.optimizer)
    if hasattr(optimizer, "optimizers"):
        return list(optimizer.optimizers)
    return [optimizer]


def _collect_optimizer_state(sub_opt) -> dict:
    """Return optimizer-owned tensors (momentum/Adam moments/lr) excluding params."""
    from tinygrad.nn.state import get_state_dict

    state = get_state_dict(sub_opt)
    excluded_ids = {id(p) for p in sub_opt.params}
    excluded_ids |= {id(p) for p in getattr(sub_opt, "buffers", [])}
    return {k: v for k, v in state.items() if id(v) not in excluded_ids}


def save_checkpoint(trainer, pl_module, ckpt_path: str | Path, meta_path: str | Path | None = None) -> Path:
    """Persist model weights + optimizer state (safetensors) + metadata (JSON)."""
    from tinygrad.nn.state import get_state_dict, safe_save

    ckpt_path = Path(ckpt_path)
    if meta_path is None:
        meta_path = ckpt_path.with_suffix(META_SUFFIX)
    else:
        meta_path = Path(meta_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    sd: dict = {f"model.{k}": v for k, v in get_state_dict(pl_module).items()}

    optimizer_meta: list[dict] = []
    for opt_idx, lopt in enumerate(getattr(trainer, "optimizers", []) or []):
        sub_opts = _sub_optimizers(lopt)
        sub_meta: list[dict] = []
        for sub_idx, sub in enumerate(sub_opts):
            sub_state = _collect_optimizer_state(sub)
            for k, v in sub_state.items():
                sd[f"optimizer_{opt_idx}_{sub_idx}.{k}"] = v
            sub_meta.append({"keys": list(sub_state.keys())})
        optimizer_meta.append(sub_meta)

    safe_save(sd, str(ckpt_path))

    meta: dict[str, Any] = {
        "version": __version__,
        "epoch": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "scheduler_states": [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs],
        "optimizer_state_keys": optimizer_meta,
        "hparams": dict(pl_module.hparams) if hasattr(pl_module, "hparams") else {},
    }
    if hasattr(trainer, "precision_plugin") and trainer.precision_plugin is not None:
        meta["precision"] = trainer.precision_plugin.state_dict()
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    return ckpt_path


def load_checkpoint(trainer, pl_module, ckpt_path: str | Path) -> dict[str, Any]:
    """Restore model weights + optimizer state + scheduler state + step counters."""
    from tinygrad.nn.state import load_state_dict, safe_load

    ckpt_path = Path(ckpt_path)
    meta_path = ckpt_path.with_suffix(META_SUFFIX)

    sd = safe_load(str(ckpt_path))
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if not model_sd:
        raise RuntimeError(f"checkpoint {ckpt_path} has no 'model.*' keys")
    load_state_dict(pl_module, model_sd, strict=True, verbose=False)

    # Restore optimizer state (best-effort: missing keys do not fail).
    for opt_idx, lopt in enumerate(getattr(trainer, "optimizers", []) or []):
        sub_opts = _sub_optimizers(lopt)
        for sub_idx, sub in enumerate(sub_opts):
            prefix = f"optimizer_{opt_idx}_{sub_idx}."
            sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not sub_sd:
                continue
            load_state_dict(sub, sub_sd, strict=False, verbose=False)

    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        meta = json.load(f)
    trainer.current_epoch = int(meta.get("epoch", 0))
    trainer.global_step = int(meta.get("global_step", 0))
    for cfg, state in zip(trainer.lr_scheduler_configs, meta.get("scheduler_states", [])):
        cfg.scheduler.load_state_dict(state)
    precision_state = meta.get("precision")
    if precision_state is not None and hasattr(trainer, "precision_plugin"):
        try:
            trainer.precision_plugin.load_state_dict(precision_state)
        except RuntimeError:
            # precision changed across runs; skip silently rather than block the resume
            pass
    return meta


class ModelCheckpoint(Callback):
    """Save model weights at training milestones.

    Args:
        dirpath: Directory to write into. Defaults to ``trainer.default_root_dir / 'checkpoints'``.
        filename: Format string. Replaceable fields: ``{epoch}``, ``{step}``,
            plus any key in ``trainer.callback_metrics``.
        monitor: Metric name to track for top-K selection. ``None`` = save always.
        mode: ``"min"`` (default) or ``"max"``.
        save_top_k: Keep best K checkpoints. ``-1`` keeps all; ``0`` saves none.
        save_last: Always also write a ``last.safetensors`` pointer.
        every_n_epochs: Save once per N epochs. Mutually exclusive with ``every_n_train_steps``.
        every_n_train_steps: Save once per N optimizer steps.
    """

    CHECKPOINT_SUFFIX = CHECKPOINT_SUFFIX
    META_SUFFIX = META_SUFFIX

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str = "epoch={epoch}-step={step}",
        *,
        monitor: str | None = None,
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = False,
        every_n_epochs: int = 1,
        every_n_train_steps: int | None = None,
    ) -> None:
        if every_n_train_steps and every_n_epochs not in (None, 0, 1):
            raise ValueError("every_n_train_steps and every_n_epochs are mutually exclusive")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.dirpath = Path(dirpath) if dirpath else None
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = int(save_top_k)
        self.save_last = bool(save_last)
        self.every_n_epochs = int(every_n_epochs) if every_n_epochs else 0
        self.every_n_train_steps = int(every_n_train_steps) if every_n_train_steps else 0

        self._best_k_models: list[tuple[float, Path]] = []  # (score, ckpt path)
        self.best_model_path: Path | None = None
        self.last_model_path: Path | None = None

    # ---- hooks ---------------------------------------------------------

    def setup(self, trainer, pl_module, stage: str) -> None:
        if self.dirpath is None:
            self.dirpath = Path(trainer.default_root_dir) / "checkpoints"
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_train_steps:
            return
        if self.monitor is not None:
            return  # wait for validation_epoch_end so the monitored value is fresh
        if self.every_n_epochs and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_save(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.monitor is None or self.every_n_train_steps:
            return
        if self.every_n_epochs and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_save(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self.every_n_train_steps:
            return
        if trainer.global_step > 0 and trainer.global_step % self.every_n_train_steps == 0:
            self._maybe_save(trainer, pl_module)

    # ---- internals -----------------------------------------------------

    def _maybe_save(self, trainer, pl_module) -> None:
        score = self._score(trainer)
        if self.save_top_k == 0:
            return
        if self.save_top_k > 0 and not self._should_save(score):
            return

        ckpt_path = self._build_path(trainer)
        save_checkpoint(trainer, pl_module, ckpt_path)

        if self.save_top_k > 0:
            # When no monitor is set, "top K" means the K most recent saves.
            # Otherwise rank by score; keep the best K.
            sort_score = float(score) if score is not None else float(trainer.global_step)
            self._best_k_models.append((sort_score, ckpt_path))
            keep_best = score is not None
            self._best_k_models.sort(
                key=lambda x: x[0],
                reverse=(self.mode == "max") if keep_best else True,  # newest first when no monitor
            )
            while len(self._best_k_models) > self.save_top_k:
                _, victim = self._best_k_models.pop()
                _safe_unlink(victim)
                _safe_unlink(victim.with_suffix(META_SUFFIX))
            self.best_model_path = self._best_k_models[0][1] if self._best_k_models else None

        self.last_model_path = ckpt_path
        if self.save_last:
            last_ckpt = self.dirpath / ("last" + CHECKPOINT_SUFFIX)
            save_checkpoint(trainer, pl_module, last_ckpt)

    def _build_path(self, trainer) -> Path:
        format_args: dict[str, Any] = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        }
        format_args.update(trainer.callback_metrics)
        try:
            stem = self.filename.format(**format_args)
        except KeyError as e:
            raise RuntimeError(
                f"ModelCheckpoint.filename uses {{{e.args[0]}}} but no such metric in "
                f"trainer.callback_metrics={list(trainer.callback_metrics)}"
            ) from e
        return self.dirpath / (stem + CHECKPOINT_SUFFIX)

    def _score(self, trainer) -> float | None:
        if self.monitor is None:
            return None
        if self.monitor not in trainer.callback_metrics:
            return None
        return float(trainer.callback_metrics[self.monitor])

    def _should_save(self, score: float | None) -> bool:
        if score is None:
            return True
        if len(self._best_k_models) < self.save_top_k:
            return True
        worst = self._best_k_models[-1][0]
        return score < worst if self.mode == "min" else score > worst


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass
