from tinygrad_lightning.callback import CheckpointCallback
from tinygrad_lightning import DataLoader, LightningModule

from pathlib import Path
from tinygrad.tensor import Tensor

class TinyConvNet(LightningModule):
    def __init__(self, classes=10):
        conv = 3
        inter_chan, out_chan = 8, 16   # for speed
        self.c1 = Tensor.uniform(inter_chan,3,conv,conv)
        self.c2 = Tensor.uniform(out_chan,inter_chan,conv,conv)
        self.l1 = Tensor.uniform(out_chan*6*6, classes)

    def forward(self, x):
        x = x.conv2d(self.c1).relu().max_pool2d()
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return x.dot(self.l1)

class SimpleDataset:
   
    def __len__(self) -> None:
        return 100
    def __iter__(self):
        for i in range(len(self)):
            yield i, i

def test_model_checkpointing():
    callback = CheckpointCallback("/tmp/checkpoint_{train_loss:.2f} {epoch:d}.npy", 1, mode='train')
    
    ds = SimpleDataset()
    train_loader = DataLoader(
        ds, 1, workers=1
    )
    model = TinyConvNet(10)
    callback._set_refs(
        model, train_loader, val_loader=None
    )
    callback.configure(1, "train")
    callback.log("loss", 1)
    callback.log("loss", 2)
    callback.log("loss", 3)
    callback.on_epoch_end()

    target_path = Path("/tmp/checkpoint_2.00 1.npy")
    assert(target_path.exists())

    model.load(target_path)
    
