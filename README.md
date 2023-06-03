# Tinygrad Lightning - WIP

Pytorch Lightning clone for tinygrad. Easy data loading, training, logging and checkpointing.

### Example

```
import tinygrad_lightning as pl

### model ###

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

        # automatically logs to train/loss, ...
        self.log("loss", loss_value.mean())
        self.log("accuracy", accuracy)

        return loss
    
    def validation_step(self, val_batch, val_idx):
        x, y = val_batch
        out = self.forward(x)

        cat = np.argmax(out.cpu().numpy(), axis=-1)
        accuracy = (cat == y).mean()

        loss = sparse_categorical_crossentropy(out, y)
        loss_value = loss.detach().cpu().numpy()

        # automatically logs to val/loss, ...
        self.log("loss", loss_value.mean())
        self.log("accuracy", accuracy)

        return loss

batch_size = 4

test_ds = MnistDataset(variant='test') # same as torch dataset
train_loader = pl.DataLoader(train_ds, batch_size, workers=1, shuffle=True)

# define your model
model = TinyBobNet()
callbacks=[pl.TQDMProgressBar(refresh_rate=10), pl.TensorboardLogger("./logdir")]

trainer = pl.Trainer(model, train_loader=train_loader, callbacks=callbacks)
trainer.fit(epochs=1) # train_batches=2, val_batches=4
```