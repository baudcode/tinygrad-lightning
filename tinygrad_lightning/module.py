import numpy as np
from PIL import Image
from typing import Union
from pathlib import Path

from tinygrad.nn import optim

class LightningModule:
    def __init__(self):
        self._logs = []

    def log(self, name: str, value: any):
        for callback in (self.callbacks if hasattr(self, "callbacks") else []):
            callback.log(name, value) 
    
    def log_image(self, name: str, image: Union[np.ndarray, Image.Image]):
        for callback in (self.callbacks if hasattr(self, "callbacks") else []):
            callback.log_image(name, image)

    def _set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def forward(self):
        raise NotImplementedError("forward pass not implemented")

    def __call__(self):
        return self.forward()

    def _get_optimizers(self):
        optim = self.configure_optimizers()

        if isinstance(optim, list):
            return optim
        else:
            return [optim]

    def configure_optimizers(self):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass

    def backward(self, loss, optim, optimizer_idx):
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    def save(self, filename: str):
        path = Path(filename)
        assert(path.name.endswith(".npy"))
        with path.open("wb") as f:
            for par in optim.get_parameters(self):
                np.save(f, par.cpu().numpy())

    def load(self, filename: Union[str, Path], gpu=False):
        path = Path(filename)
        with path.open("rb") as f:
            for par in optim.get_parameters(self):
                #if par.requires_grad:
                try:
                    par.cpu().numpy()[:] = np.load(f)
                    if gpu:
                        par.gpu()
                except:
                    print('Could not load parameter')