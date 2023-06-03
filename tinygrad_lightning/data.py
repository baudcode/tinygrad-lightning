import numpy as np
import random
from typing import Protocol
import multiprocessing
from typing import Optional
from tinygrad.tensor import Tensor

"""

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
"""


class ComposeTransforms:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        for t in self.trans:
            x = t(x)
        return x


class Dataset(Protocol):
    def __init__(self):
        pass
    def __len__(self):
        raise NotImplementedError("need to implement __len__")

    def __getitem__(self):
        raise NotImplementedError("need to implement __get_item__")

class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int, workers: Optional[int] = None, shuffle=False):
        self.workers = workers or multiprocessing.cpu_count() // 2 + 1
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.idxs = np.arange(len(self))

        if shuffle:
            np.random.shuffle(self.idxs)

    def __len__(self):
        # TODO: add support for skipping end batch if not len(end_batch) == batch_size
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        dataset_size = len(self.dataset)

        # TODO: more advanced method with prefetching
        with multiprocessing.Pool(self.workers) as pool:
            for n in range(len(self)):
                batch_idx = list(range(n * self.batch_size, min((n + 1) * self.batch_size, dataset_size)))
                translated_idx = self.idxs[np.asarray(batch_idx)]

                results = pool.map(self.dataset.__getitem__, translated_idx)
                # results are x, y pairs
                batch_x = list(map(lambda x: x[0], results))
                batch_y = list(map(lambda x: x[1], results))

                x, y = Tensor(batch_x, requires_grad=False), np.asarray(batch_y)
                # print(x.shape, y.shape)
                yield x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)




def random_split(l: list, p: float):
    copy = l.copy()
    random.shuffle(copy)

    s = int(len(l) * p)
    train, test = l[:-s], l[s:]
    return train, test
