from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import tinygrad_lightning as pl
from tinygrad_lightning.losses import sparse_categorical_crossentropy
import numpy as np
from PIL import Image
from .resnet import ResNet18
import numpy as np
from tinygrad_lightning import Trainer


class TinyBobNet(pl.LightningModule):
    def __init__(self, filters=64):
        self.model = ResNet18(num_classes=10)

    def forward(self, input: Tensor):
        return self.model(input)

    def configure_optimizers(self):
        return optim.SGD(optim.get_parameters(self), lr=5e-3, momentum=0.9)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        for image in x:
            self.log_image("inputs", image)

        out = self.forward(x)

        cat = np.argmax(out.cpu().numpy(), axis=-1)
        accuracy = (cat == y).mean()

        loss = sparse_categorical_crossentropy(out, y)
        loss_value = loss.detach().cpu().numpy()

        self.log("loss", loss_value.mean())
        self.log("accuracy", accuracy)

        return loss

import os, gzip

def fetch_mnist():
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    X_train = parse(os.path.dirname(__file__)+"/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_train = parse(os.path.dirname(__file__)+"/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = parse(os.path.dirname(__file__)+"/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_test = parse(os.path.dirname(__file__)+"/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test


class MnistDataset(pl.Dataset):
    def __init__(self, variant:str = "train"):
        X_train, Y_train, X_test, Y_test = fetch_mnist()
        
        if variant == 'train':
            self.x = X_train.reshape(-1, 28, 28).astype(np.uint8)
            self.y = Y_train
        else:
            self.x = X_test.reshape(-1, 28, 28).astype(np.uint8)
            self.y = Y_test

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]

        transform = pl.ComposeTransforms([
            lambda x: Image.fromarray(x, mode='L').convert("RGB").resize((64, 64)),
            lambda x: np.asarray(x, "float32").transpose((2, 0, 1)),
            # lambda x: np.stack([np.asarray(xx) for xx in x], 0),
            lambda x: x / 255.0,
            # lambda x: np.tile(np.expand_dims(x, 1), (1, 3, 1, 1)).astype(np.float32),
        ])

        x = transform(x)
        return x, y


def main():
    batch_size = 4

    train_ds = MnistDataset(variant='train')
    val_ds = MnistDataset(variant='val')
    
    train_loader = pl.DataLoader(train_ds, batch_size, workers=1, shuffle=True)
    val_loader = pl.DataLoader(val_ds, batch_size, workers=1)
    
    model = TinyBobNet()
    callbacks=[pl.TQDMProgressBar(refresh_rate=10), pl.TensorboardLogger("./logdir")]
    trainer = Trainer(model, train_loader=train_loader, val_loader=val_loader, callbacks=callbacks)
    trainer.fit(epochs=1, train_batches=2, val_batches=4)

    # TODO: freeze network | evaluate only
    #trainer.evaluate(test_dataset)

if __name__ == "__main__":
    main()

# ... and complete like pytorch, with (x,y) data

