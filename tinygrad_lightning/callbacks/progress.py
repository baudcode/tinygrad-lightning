"""TQDM-based progress bar.

Reads metrics from ``trainer.callback_metrics`` (the merged step+epoch metric
dict) and surfaces those marked ``prog_bar=True`` in the postfix.
"""
from __future__ import annotations

from typing import Any

from .base import Callback


class TQDMProgressBar(Callback):
    def __init__(self, refresh_rate: int = 1) -> None:
        self.refresh_rate = max(1, int(refresh_rate))
        self._train_bar = None
        self._val_bar = None
        self._train_step = 0
        self._val_step = 0

    # ---- train ---------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        import tqdm

        total = trainer.estimated_train_batches_per_epoch
        desc = f"Epoch {trainer.current_epoch} train"
        self._train_bar = tqdm.tqdm(total=total, desc=desc, leave=False)
        self._train_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._train_bar is None:
            return
        self._train_step += 1
        if self._train_step % self.refresh_rate == 0:
            self._train_bar.set_postfix(**self._postfix(trainer, pl_module, "train"))
        self._train_bar.update(1)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._train_bar is not None:
            self._train_bar.set_postfix(**self._postfix(trainer, pl_module, "train"))
            self._train_bar.close()
            self._train_bar = None

    # ---- validation ----------------------------------------------------

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        import tqdm

        total = trainer.estimated_val_batches_per_epoch
        desc = f"Epoch {trainer.current_epoch} val"
        self._val_bar = tqdm.tqdm(total=total, desc=desc, leave=False)
        self._val_step = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._val_bar is None:
            return
        self._val_step += 1
        if self._val_step % self.refresh_rate == 0:
            self._val_bar.set_postfix(**self._postfix(trainer, pl_module, "val"))
        self._val_bar.update(1)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._val_bar is not None:
            self._val_bar.set_postfix(**self._postfix(trainer, pl_module, "val"))
            self._val_bar.close()
            self._val_bar = None

    # ---- helpers -------------------------------------------------------

    def _postfix(self, trainer, pl_module, stage: str) -> dict[str, str]:
        names = pl_module._prog_bar_names(stage)
        return {k: f"{v:.4f}" for k, v in trainer.callback_metrics.items() if k in names}
