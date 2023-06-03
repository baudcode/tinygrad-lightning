import numpy as np
from tinygrad.tensor import Tensor

def sparse_categorical_crossentropy(out: Tensor, Y: np.ndarray):
    num_classes = out.shape[-1]
    YY = Y.flatten().astype(np.int32)
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),YY] = -1.0*num_classes
    y = y.reshape(list(Y.shape)+[num_classes])
    y = Tensor(y)
    return out.mul(y).mean()