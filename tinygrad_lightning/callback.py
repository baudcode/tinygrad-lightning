from typing import Any, Union
from tinygrad.tensor import Tensor
import tqdm
from collections import defaultdict
import numpy as np
from PIL import Image
import numpy as np
from tinygrad.nn import optim
from pathlib import Path


class LightningCallback(object):

    def _set_refs(self, model, train_loader, val_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def log(self, name: str, value: any) -> Any:
        pass

    def log_image(self, name: str, image: Union[np.ndarray, Image.Image]):
        pass

    def configure(self, steps: int, mode='train'):
        pass

    def update(self, step: int):
        """ call the visualization update after each step """
        pass

    def on_epoch_end(self, logs):
        pass

    def on_batch_end(self, logs):
        pass


class CheckpointCallback(LightningCallback):
    
    def __init__(self, path: str, every: int = 1, mode = "val"):
        self.current_mode = "train"
        self.target_mode = mode
        self.path = path
        self.every = every
        self.scalars = defaultdict(lambda: [])
        self.current_epoch = 1

    def configure(self, steps, mode='train'):
        self.current_mode = mode
    
    def log(self, name: str, value: any) -> Any:
        self.scalars[f"{self.current_mode}_{name}"].append(value)

    def on_epoch_end(self):
        if self.target_mode == self.current_mode and self.current_epoch % self.every == 0:
            logs = {k: np.asarray(v).mean() for k, v in self.scalars.items()}
            logs.update(epoch=self.current_epoch)
            print("Path: ", self.path)
            save_path = self.path.format(**logs)
            self.model.save(save_path)
        
        # reset scalars
        self.scalars = {}
        self.current_epoch += 1

class TQDMProgressBar(LightningCallback):
    def __init__(self, refresh_rate=10):
        self.refresh_rate = refresh_rate
        self.tqdm = tqdm.tqdm()
        self.buffer = defaultdict(lambda: [])
        self.total_step = 0
    
    def configure(self, steps: int, mode='train'):
        self.tqdm = tqdm.tqdm(total=steps, desc=mode)

    def log(self, name: str, value: any) -> Any:
        if isinstance(value, float) or isinstance(value, int):
            self.buffer[name].append(value)
        elif isinstance(value, np.floating):
            self.buffer[name] = float(value)
        # elif isinstance(value, Tensor):
        # self.buffer[name].append(value.realize().numpy())
        else:
            print(f"[TQDMProgressBar] unsupported log type {type(value)}: {value}")

    def update(self, step: int):
        if step % self.refresh_rate == 0:
            self.tqdm.set_postfix(**dict((k, np.array(v).mean()) for k, v in self.buffer.items()))
            self.tqdm.update(1)

    def on_epoch_end(self, logs):
        self.buffer = defaultdict(lambda: [])
        self.tqdm.close()

    
class TensorboardLogger(LightningCallback):

    def __init__(self, log_dir: str, **kwargs):
        import tensorboardX
        self.writer = tensorboardX.SummaryWriter(log_dir)
        self.scalars = dict()
        self.images = dict()
        self.mode = "train"
    
    def on_epoch_end(self):
        self.buffer = {}
    
    def on_batch_end(self):
        self.buffer = {}
    
    def configure(self, steps: int, mode='train'):
        self.mode = mode
    
    def log(self, name: str, value: any):
        # log any kind of data
        if isinstance(value, float) or isinstance(value, int):
            self.scalars[name] = value
        elif isinstance(value, np.floating):
            self.scalars[name] = value
        elif isinstance(value, Tensor):
            self.scalars[name] = value.realize().numpy()
        else:
            raise Exception(f"unsupported logging type for log_scalar {type(value)}")
    
    def log_image(self, name: str, image):
        if isinstance(image, Image.Image):
            image = np.asarray(image)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)

            self.images[name] = image
        elif isinstance(image, np.ndarray):
            self.images[name] = image
        elif isinstance(image, Tensor):
            self.images[name] = image.realize().numpy()
        else:
            raise Exception(f"unsupported logging type for log_image {type(image)}")

    def update(self, step: int):
        for k, v in self.scalars.items():
            self.writer.add_scalar(f"{self.mode}/{k}", v, step)

        for k, v in self.images.items():
            assert(len(v.shape) == 3), "image has to have len(shape) == 3"
            data_format = "HWC" if v.shape[-1] <= 4 else "CHW"
            self.writer.add_image(f"{self.mode}/{k}", v, step, dataformats=data_format)