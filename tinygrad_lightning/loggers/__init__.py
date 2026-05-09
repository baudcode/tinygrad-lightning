from .base import Logger
from .csv import CSVLogger

__all__ = ["Logger", "CSVLogger"]


# Lazy attribute access for optional loggers — avoid importing tensorboardX or
# mlflow at module-import time.
def __getattr__(name):
    if name == "TensorBoardLogger":
        from .tensorboard import TensorBoardLogger
        return TensorBoardLogger
    if name == "MLFlowLogger":
        from .mlflow import MLFlowLogger
        return MLFlowLogger
    raise AttributeError(name)
