"""Shared pytest fixtures for tinygrad-lightning parity tests.

The `parity_backend` fixture parametrizes tests over the two backends:
- "tinygrad": run the test against tinygrad_lightning
- "torch":    run the test against lightning.pytorch (skipped if unavailable)

A test that uses this fixture should branch on `parity_backend` to pick which
framework's classes to instantiate, then assert the *same* observable behavior
holds in both branches.
"""
from __future__ import annotations

import importlib.util

import pytest


def _is_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


HAS_LIGHTNING = _is_importable("lightning")
HAS_MLFLOW = _is_importable("mlflow")
HAS_TENSORBOARDX = _is_importable("tensorboardX")


@pytest.fixture(
    params=[
        pytest.param("tinygrad", id="tinygrad"),
        pytest.param(
            "torch",
            id="torch",
            marks=pytest.mark.skipif(
                not HAS_LIGHTNING,
                reason="lightning.pytorch not installed (install requirements-test.txt)",
            ),
        ),
    ]
)
def parity_backend(request) -> str:
    """Parametrized backend name; tests should branch on the returned string."""
    return request.param


@pytest.fixture
def needs_mlflow():
    if not HAS_MLFLOW:
        pytest.skip("mlflow not installed")
    return True


@pytest.fixture
def needs_tensorboardx():
    if not HAS_TENSORBOARDX:
        pytest.skip("tensorboardX not installed")
    return True
