"""Phase 0 smoke test: the package imports and exposes its public API."""
from __future__ import annotations


def test_package_imports():
    import tinygrad_lightning  # noqa: F401


def test_version_exposed():
    from tinygrad_lightning.version import __version__

    assert isinstance(__version__, str)
    assert __version__


def test_public_exports_present():
    """Names the design commits to. Most are stubs in Phase 0; only existence is checked."""
    import tinygrad_lightning as L

    expected = [
        "LightningModule",
        "Trainer",
        "DataLoader",
    ]
    missing = [name for name in expected if not hasattr(L, name)]
    assert not missing, f"missing public exports: {missing}"
