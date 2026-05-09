"""Unit: accelerator string -> tinygrad device resolution."""
from __future__ import annotations

import pytest

from tinygrad_lightning.core.accelerator import (
    CPU_DEVICES,
    GPU_DEVICES,
    _available_devices,
    resolve_accelerator,
)


def test_auto_returns_an_available_device():
    chosen = resolve_accelerator("auto", devices=1)
    assert chosen in _available_devices()


def test_auto_prefers_gpu_when_available():
    available = _available_devices()
    chosen = resolve_accelerator("auto", devices=1)
    gpu_present = any(g in available for g in GPU_DEVICES)
    if gpu_present:
        assert chosen in GPU_DEVICES, f"auto picked {chosen!r} despite GPU available in {available}"


def test_cpu_forces_clang_or_llvm():
    chosen = resolve_accelerator("cpu", devices=1)
    assert chosen in CPU_DEVICES


def test_gpu_raises_when_unavailable():
    available = _available_devices()
    if any(g in available for g in GPU_DEVICES):
        pytest.skip("a GPU is actually available; cannot test the missing-GPU path")
    with pytest.raises(RuntimeError, match="no GPU-class device"):
        resolve_accelerator("gpu", devices=1)


def test_devices_gt_one_raises_ddp_not_supported():
    with pytest.raises(NotImplementedError, match="DDP"):
        resolve_accelerator("cpu", devices=2)
    with pytest.raises(NotImplementedError, match="DDP"):
        resolve_accelerator("cpu", devices=[0, 1])


def test_explicit_unknown_device_raises():
    with pytest.raises(RuntimeError, match="not available"):
        resolve_accelerator("nonexistent_device", devices=1)


def test_devices_zero_raises():
    with pytest.raises(ValueError):
        resolve_accelerator("cpu", devices=0)
