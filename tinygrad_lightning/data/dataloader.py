"""DataLoader — minimal batching iterator for tinygrad-lightning.

Compatible with map-style datasets that implement ``__len__`` and ``__getitem__``.
Yields batches via ``collate_fn`` (default: stack into NumPy arrays).

When ``dtype`` is set (or auto-set by the ``Trainer`` to match the precision
plugin's ``compute_dtype``), float-valued NumPy arrays in each batch are
converted to ``tinygrad.Tensor`` with the given dtype. Integer arrays are
left as int Tensors so labels don't get accidentally cast to fp16.

The ``Trainer`` mutates ``dataloader.dtype`` on each tinygrad-lightning
DataLoader passed to ``fit``/``validate``/``test`` whenever the precision
plugin demands a non-fp32 compute dtype, *unless* the user already set one.
This is the auto-cast for inputs.
"""
from __future__ import annotations

import multiprocessing
from typing import Any, Callable, Iterator, Protocol

import numpy as np


class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


def _default_collate(samples: list[Any]) -> Any:
    """Stack (x, y) tuples into (X, Y) NumPy batches."""
    if isinstance(samples[0], tuple):
        cols = list(zip(*samples))
        return tuple(np.asarray(col) for col in cols)
    return np.asarray(samples)


def _np_to_compute_dtype(np_dtype, target):
    """Map a tinygrad dtype to a numpy dtype suitable for ``np.ndarray.astype``."""
    from tinygrad import dtypes

    if target == dtypes.float16:
        return np.float16
    if target == dtypes.bfloat16:
        # NumPy has no native bfloat16; cast to fp32 and let tinygrad handle the
        # actual bf16 conversion when the Tensor is built.
        return np.float32
    if target == dtypes.float32:
        return np.float32
    return np_dtype


def _coerce_to_tensors(obj, target_dtype):
    """Convert NumPy arrays in ``obj`` to ``tinygrad.Tensor``. Float arrays are
    cast to ``target_dtype``; integer arrays become int Tensors unchanged."""
    from tinygrad import Tensor

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "f":
            np_dt = _np_to_compute_dtype(obj.dtype, target_dtype)
            arr = obj.astype(np_dt, copy=False)
            t = Tensor(arr)
            # bf16 numpy doesn't exist; we asked for fp32 numpy and now cast in tinygrad.
            from tinygrad import dtypes
            if target_dtype == dtypes.bfloat16 and t.dtype != dtypes.bfloat16:
                t = t.cast(dtypes.bfloat16)
            return t
        return Tensor(obj)
    if isinstance(obj, tuple):
        return tuple(_coerce_to_tensors(x, target_dtype) for x in obj)
    if isinstance(obj, list):
        return [_coerce_to_tensors(x, target_dtype) for x in obj]
    if isinstance(obj, dict):
        return {k: _coerce_to_tensors(v, target_dtype) for k, v in obj.items()}
    return obj


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        dtype=None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = int(num_workers)
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn or _default_collate
        # When set (either by the user or by Trainer), batches are returned as
        # tinygrad Tensors with float arrays cast to ``dtype``.
        self.dtype = dtype

        n = len(dataset)
        self._idxs = np.arange(n)
        if self.shuffle:
            np.random.shuffle(self._idxs)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        if self.shuffle:
            np.random.shuffle(self._idxs)

        batches = list(self._batched_indices(n))

        if self.num_workers > 0:
            with multiprocessing.Pool(self.num_workers) as pool:
                for indices in batches:
                    samples = pool.map(self.dataset.__getitem__, indices)
                    yield self._post_collate(self.collate_fn(samples))
        else:
            for indices in batches:
                samples = [self.dataset[i] for i in indices]
                yield self._post_collate(self.collate_fn(samples))

    def _post_collate(self, batch):
        if self.dtype is None:
            return batch
        return _coerce_to_tensors(batch, self.dtype)

    def _batched_indices(self, n: int):
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.drop_last and (end - start) < self.batch_size:
                return
            yield self._idxs[start:end].tolist()
