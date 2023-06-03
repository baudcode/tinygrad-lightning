from typing import List
from .callback import LightningCallback
from .module import LightningModule
from .data import DataLoader

class Trainer:

    def __init__(self, model: LightningModule, train_loader: DataLoader = None, 
                 val_loader: DataLoader = None, 
                 callbacks: List[LightningCallback] = []):
        self.model = model
        self.train_loader = train_loader or model.train_dataloader()
        self.val_loader = val_loader or model.val_dataloader()
        self.callbacks: List[LightningCallback] = callbacks
        self.model._set_callbacks(self.callbacks)


    def fit(self, epochs=1, gpu=False, train_batches=-1, val_batches=-1):
        optimizers = self.model._get_optimizers()
        
        if gpu:
            # move weights to gpu
            for optim in optimizers:
                for param in optim.parameters(self.model):
                    param.gpu_()


        for epoch in range(epochs):
            num_train_steps = train_batches if train_batches > 0 else len(self.train_loader)
            
            for callback in self.callbacks:
                callback.configure(num_train_steps, mode='train')

            for step, (x, y) in enumerate(self.train_loader):

                total_step = epoch * num_train_steps + step
                loss = self.model.training_step((x, y), step)

                for optimizer_index, optim in enumerate(optimizers):
                    self.model.backward(loss, optim, optimizer_index)

                for callback in self.callbacks:
                    callback.update(total_step)
                    callback.on_batch_end()
                
                # TODO: support clipping gradients
                if train_batches > 0 and step == train_batches:
                    break
            
            # validation
            num_val_steps = val_batches if val_batches > 0 else len(self.val_loader)
            
            for callback in self.callbacks:
                callback.configure(num_val_steps, mode='val')

            for step, (x, y) in enumerate(self.val_loader):

                total_step = epoch * num_train_steps + step
                loss = self.model.validation_step((x, y), step)

                # for optimizer_index, optim in enumerate(optimizers):
                # self.model.backward(loss, optim, optimizer_index)

                for callback in self.callbacks:
                    callback.update(total_step)
                    callback.on_batch_end()
                
                # TODO: support clipping gradients
                if val_batches > 0 and step == val_batches:
                    break
                
            # reset callbacks
            for callback in self.callbacks:
                callback.on_epoch_end()
            
            # reset data loaders
            self.train_loader.on_epoch_end()

            if self.val_loader:
                self.val_loader.on_epoch_end()

    def evaluate(self):
        raise NotImplementedError("evaluate() not implemented yet")
