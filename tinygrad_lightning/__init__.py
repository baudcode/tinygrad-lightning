"""Public API for tinygrad-lightning.

Mirrors ``lightning.pytorch``'s top-level exports where the underlying feature
is implemented; see ``docs/plans/2026-05-06-tinygrad-lightning-pl-parity-design.md``
for the per-phase rollout.
"""
from .callbacks.base import Callback
from .callbacks.checkpoint import ModelCheckpoint
from .callbacks.early_stopping import EarlyStopping
from .callbacks.lr_monitor import LearningRateMonitor
from .callbacks.progress import TQDMProgressBar
from .core.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    StepLR,
)
from .core.optimizer import LightningOptimizer, param_group_optimizer
from .data.dataloader import DataLoader, Dataset
from .data.datamodule import LightningDataModule
from .loggers.base import Logger
from .loggers.csv import CSVLogger
from .module import LightningModule
from .trainer.trainer import Trainer
from .utilities.seed import seed_everything
from .version import __version__


def __getattr__(name):
    """Lazy access for optional loggers (avoid importing tensorboardX/mlflow)."""
    if name == "TensorBoardLogger":
        from .loggers.tensorboard import TensorBoardLogger
        return TensorBoardLogger
    if name == "MLFlowLogger":
        from .loggers.mlflow import MLFlowLogger
        return MLFlowLogger
    raise AttributeError(f"module 'tinygrad_lightning' has no attribute {name!r}")

__all__ = [
    "LightningModule",
    "Trainer",
    "DataLoader",
    "Dataset",
    "Callback",
    "TQDMProgressBar",
    "LearningRateMonitor",
    "ModelCheckpoint",
    "EarlyStopping",
    "LightningDataModule",
    "Logger",
    "CSVLogger",
    "TensorBoardLogger",
    "MLFlowLogger",
    "LightningOptimizer",
    "param_group_optimizer",
    "LRScheduler",
    "StepLR",
    "MultiStepLR",
    "CosineAnnealingLR",
    "LambdaLR",
    "OneCycleLR",
    "seed_everything",
    "__version__",
]
