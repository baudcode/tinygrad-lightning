"""Resolve PL-style accelerator/devices into a concrete tinygrad device name.

The mapping (per design doc):

    "auto"  -> first available GPU-class device, else first CPU-class
    "cpu"   -> CLANG (or LLVM/PYTHON fallback)
    "gpu"   -> first GPU-class device (METAL/CUDA/AMD/NV/HIP)
    "<dev>" -> passthrough to tinygrad device name

``devices > 1`` raises ``NotImplementedError("DDP not supported")``.
"""
from __future__ import annotations

import os
from typing import Iterable

GPU_DEVICES: tuple[str, ...] = ("METAL", "CUDA", "AMD", "NV", "HIP", "QCOM", "WEBGPU", "CL")
CPU_DEVICES: tuple[str, ...] = ("CPU", "CLANG", "LLVM", "DSP", "PYTHON")


def _available_devices() -> list[str]:
    from tinygrad.device import Device

    return [d.split(":", 1)[0].upper() for d in Device.get_available_devices()]


def _pick_first(candidates: Iterable[str], available: list[str]) -> str | None:
    for c in candidates:
        if c in available:
            return c
    return None


def resolve_accelerator(accelerator: str = "auto", devices: int | str | list[int] = 1) -> str:
    """Return the canonical tinygrad device name to use."""
    if isinstance(devices, list):
        if len(devices) > 1:
            raise NotImplementedError("DDP not supported: tinygrad-lightning is single-device only")
        if len(devices) == 0:
            raise ValueError("devices must not be an empty list")
    elif isinstance(devices, int):
        if devices > 1:
            raise NotImplementedError("DDP not supported: tinygrad-lightning is single-device only")
        if devices < 1:
            raise ValueError(f"devices must be >= 1, got {devices}")
    elif isinstance(devices, str):
        if devices not in ("auto", "1"):
            raise ValueError(f"unsupported devices={devices!r}; only int 1 or 'auto' supported")

    available = _available_devices()
    if not available:
        raise RuntimeError("no tinygrad devices available")

    name = accelerator.lower()
    if name == "auto":
        return _pick_first(GPU_DEVICES, available) or _pick_first(CPU_DEVICES, available) or available[0]
    if name == "cpu":
        chosen = _pick_first(CPU_DEVICES, available)
        if chosen is None:
            raise RuntimeError(f"no CPU-class device available; have {available}")
        return chosen
    if name == "gpu":
        chosen = _pick_first(GPU_DEVICES, available)
        if chosen is None:
            raise RuntimeError(f"no GPU-class device available; have {available}")
        return chosen

    upper = name.upper()
    if upper not in available:
        raise RuntimeError(f"device {upper!r} not available; available: {available}")
    return upper


def set_default_device(device: str) -> None:
    """Force tinygrad's default device. Must be called before tensors are created
    (otherwise existing tensors stay on the previous default)."""
    from tinygrad.device import Device

    os.environ[device] = "1"
    Device.__dict__.pop("DEFAULT", None)
    Device.DEFAULT = device
